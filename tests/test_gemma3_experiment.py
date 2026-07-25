import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from fisher_graph.gemma3_experiment import (
    DEFAULT_SMOKE_PROMPTS,
    default_gemma3_output,
    load_gemma3,
    load_gemma3_fisher_artifact,
    main,
    make_causal_lm_calibration_batches,
    read_prompts,
    run_gemma3_fisher,
)
from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.model import ToyTransformer


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        if len(prompts) != 2:
            raise AssertionError("test tokenizer expects two prompts")
        return {
            "input_ids": torch.tensor(
                [
                    [2, 10, 11, 1],
                    [2, 12, 1, 0],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                ]
            ),
        }


class LoadedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.config = SimpleNamespace(use_cache=True)
        self.checkpointing_disabled = False

    def gradient_checkpointing_disable(self) -> None:
        self.checkpointing_disabled = True


class RecordingTokenizerClass:
    calls: list[tuple[str, dict[str, object]]] = []
    result = object()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
        cls.calls.append((model_id, dict(kwargs)))
        return cls.result


class RecordingModelClass:
    calls: list[tuple[str, dict[str, object]]] = []
    result = LoadedModel()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> nn.Module:
        cls.calls.append((model_id, dict(kwargs)))
        return cls.result


class Gemma3ExperimentTests(unittest.TestCase):
    def test_explicit_next_token_targets_respect_padding(self) -> None:
        tokenizer = FakeTokenizer()

        batches = tuple(
            make_causal_lm_calibration_batches(
                tokenizer,
                ("first prompt", "second prompt"),
                max_length=8,
                tokenization_batch_size=2,
                device=torch.device("cpu"),
            )
        )

        self.assertEqual(len(batches), 1)
        batch = batches[0]
        torch.testing.assert_close(
            batch.targets,
            torch.tensor(
                [
                    [10, 11, 1, -100],
                    [12, 1, -100, -100],
                ]
            ),
        )
        torch.testing.assert_close(
            batch.valid_positions,
            torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, False],
                ]
            ),
        )
        self.assertEqual(
            batch.example_ids,
            ("prompt.000000", "prompt.000001"),
        )
        self.assertEqual(tokenizer.padding_side, "right")

    def test_prompt_file_and_inline_prompts_are_combined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text(" alpha \n\n beta\n", encoding="utf-8")

            prompts = read_prompts(
                prompt_file=path,
                inline_prompts=(" gamma ",),
            )

        self.assertEqual(prompts, ("alpha", "beta", "gamma"))
        self.assertEqual(
            read_prompts(prompt_file=None, inline_prompts=()),
            DEFAULT_SMOKE_PROMPTS,
        )

    def test_explicit_empty_prompts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("\n \n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no nonempty text"):
                read_prompts(prompt_file=path, inline_prompts=())
        with self.assertRaisesRegex(ValueError, "no nonempty text"):
            read_prompts(prompt_file=None, inline_prompts=("",))
        with self.assertRaisesRegex(TypeError, "must be strings"):
            read_prompts(
                prompt_file=None,
                inline_prompts=(object(),),  # type: ignore[arg-type]
            )

    def test_default_output_identifies_model_and_layer(self) -> None:
        self.assertEqual(
            default_gemma3_output("google/gemma-3-270m", 6),
            Path(
                ".local-runs/google--gemma-3-270m/"
                "layer-6-streaming-fisher.pt"
            ),
        )

    def test_path_preflight_exits_without_loading_transformers(self) -> None:
        paths = {
            "hub_cache": Path("/external/hf/hub"),
            "assets_cache": Path("/external/hf/assets"),
            "xet_cache": Path("/external/hf/xet"),
            "token_path": Path("/external/hf/token"),
        }
        with patch(
            "fisher_graph.gemma3_experiment."
            "resolve_gemma3_huggingface_paths",
            return_value=paths,
        ) as resolver, patch(
            "fisher_graph.gemma3_experiment.run_gemma3_fisher"
        ) as runner, patch("builtins.print") as printer:
            main(["--check-paths-only", "--cache-dir", "/external/hf/hub"])

        resolver.assert_called_once_with(Path("/external/hf/hub"))
        runner.assert_not_called()
        rendered = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in printer.call_args_list
        )
        self.assertIn("no model loaded", rendered)
        self.assertIn("/external/hf/token", rendered)

    def test_loader_uses_one_external_cache_and_never_saves_model(self) -> None:
        RecordingTokenizerClass.calls.clear()
        RecordingModelClass.calls.clear()
        model = LoadedModel()
        RecordingModelClass.result = model
        cache = Path("/external/huggingface/hub")

        with patch(
            "fisher_graph.gemma3_experiment._transformers_classes",
            return_value=(RecordingTokenizerClass, RecordingModelClass),
        ):
            tokenizer, loaded = load_gemma3(
                model_id="google/gemma-3-270m",
                revision="fixed-revision",
                cache_dir=cache,
                device=torch.device("cpu"),
                dtype="float32",
                local_files_only=True,
            )

        self.assertIs(tokenizer, RecordingTokenizerClass.result)
        self.assertIs(loaded, model)
        tokenizer_kwargs = RecordingTokenizerClass.calls[0][1]
        model_kwargs = RecordingModelClass.calls[0][1]
        self.assertEqual(tokenizer_kwargs["cache_dir"], str(cache))
        self.assertEqual(model_kwargs["cache_dir"], str(cache))
        self.assertEqual(model_kwargs["revision"], "fixed-revision")
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertFalse(model_kwargs["trust_remote_code"])
        self.assertTrue(model_kwargs["use_safetensors"])
        self.assertEqual(model_kwargs["attn_implementation"], "eager")
        self.assertEqual(model_kwargs["torch_dtype"], torch.float32)
        self.assertFalse(model.training)
        self.assertFalse(model.weight.requires_grad)
        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.checkpointing_disabled)
        self.assertFalse(hasattr(model, "save_pretrained_called"))

    def test_invalid_options_fail_before_model_loading(self) -> None:
        invalid = (
            ({"rank": 0}, "rank must be positive"),
            (
                {"rank": 4, "sketch_rows": 4},
                "greater than rank",
            ),
            ({"max_length": 1}, "at least 2"),
            ({"layer_index": -1}, "nonnegative"),
            ({"output": Path("analysis.json")}, ".pt suffix"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs), patch(
                "fisher_graph.gemma3_experiment.load_gemma3"
            ) as loader:
                with self.assertRaisesRegex(ValueError, message):
                    run_gemma3_fisher(**kwargs)
                loader.assert_not_called()

    def test_analysis_artifact_contains_no_source_model_state(self) -> None:
        torch.manual_seed(907)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=19,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=2,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        model.requires_grad_(False)
        tokenizer = FakeTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run" / "fisher.pt"
            with patch(
                "fisher_graph.gemma3_experiment.load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_experiment.Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_fisher(
                    cache_dir=root / "cache",
                    inline_prompts=("first", "second"),
                    max_length=8,
                    tokenization_batch_size=2,
                    rank=4,
                    sketch_rows=8,
                    device_name="cpu",
                    output=output,
                )
            state = torch.load(output, weights_only=True)
            collection, metadata = load_gemma3_fisher_artifact(output)

            self.assertFalse(state["contains_model_weights"])
            self.assertNotIn("state_dict", state)
            self.assertNotIn("model_state_dict", state)
            self.assertIn("collection", state)
            self.assertFalse(report["artifact"]["contains_model_state_dict"])
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertEqual(collection.sequences, 2)
            self.assertFalse(metadata["contains_model_weights"])


if __name__ == "__main__":
    unittest.main()
