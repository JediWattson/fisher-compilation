import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_gated_executor_experiment import (
    _scientific_payload_sha256,
    build_parser,
    load_gemma3_gated_executor_artifact,
    run_gemma3_gated_executor,
)
from fisher_graph.gemma3_weighted_jacobian_experiment import (
    run_gemma3_weighted_jacobian,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"
    name_or_path = "gated-executor-test-tokenizer"
    vocab_size = 31
    model_max_length = 8
    bos_token_id = 2
    eos_token_id = 1
    init_kwargs: dict[str, object] = {}

    def __init__(self) -> None:
        self.prompt_calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        self.prompt_calls.append(tuple(prompts))
        rows = [
            [2, 4 + index, 9 + index, 1]
            for index in range(len(prompts))
        ]
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(
                input_ids,
                dtype=torch.bool,
            ),
        }


def prompt_payload(prefix: str) -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": "synthetic_test_only",
        "calibration_a": [f"{prefix} a one", f"{prefix} a two"],
        "calibration_b": [f"{prefix} b one", f"{prefix} b two"],
        "validation": [f"{prefix} v one", f"{prefix} v two"],
        "test": [f"{prefix} t one", f"{prefix} t two"],
    }


def make_model() -> ToyTransformer:
    model = ToyTransformer(
        TransformerConfig(
            vocab_size=31,
            max_sequence_length=8,
            d_model=8,
            n_heads=2,
            n_layers=3,
            d_ff=12,
            dropout=0.0,
        )
    ).eval()
    model.requires_grad_(False)
    return model


class Gemma3GatedExecutorExperimentTests(unittest.TestCase):
    def test_split_safe_fit_lock_validation_and_artifact(self) -> None:
        torch.manual_seed(4401)
        model = make_model()
        parameter_versions = {
            name: int(parameter._version)
            for name, parameter in model.named_parameters()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_prompts = root / "source.json"
            source_prompts.write_text(
                json.dumps(prompt_payload("source")),
                encoding="utf-8",
            )
            source_artifact = root / "source.pt"
            source_tokenizer = RecordingTokenizer()
            with patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "load_gemma3",
                return_value=(source_tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                run_gemma3_weighted_jacobian(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=source_prompts,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 8),
                    generalized_regularization_pairs=((1e-4, 1e-4),),
                    sketch_rows=9,
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    full_rank_nll_atol=1e-5,
                    jacobian_max_sequences=0,
                    device_name="cpu",
                    dtype="float32",
                    output=source_artifact,
                )

            fresh_prompts = root / "fresh.json"
            fresh = prompt_payload("fresh")
            fresh_prompts.write_text(
                json.dumps(fresh),
                encoding="utf-8",
            )
            output = root / "gated.pt"
            tokenizer = RecordingTokenizer()
            with patch(
                "fisher_graph.gemma3_gated_executor_experiment.load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_gated_executor_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_gated_executor(
                    weighted_artifact_path=source_artifact,
                    cache_dir=root / "external-cache",
                    prompt_splits_path=fresh_prompts,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 6),
                    expert_counts=(1, 2),
                    expert_ranks=(2,),
                    router_widths=(2,),
                    max_positive_lags=(2,),
                    fit_steps=2,
                    learning_rate=1e-3,
                    max_retained_fraction=1.0,
                    max_stored_coefficient_ratio=100.0,
                    max_analytic_mac_ratio=100.0,
                    max_block_delta_nrmse=1e6,
                    min_block_delta_cosine=-1.0,
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    identity_nll_atol=1e-4,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertEqual(
                tokenizer.prompt_calls,
                [
                    ("fresh a one", "fresh a two"),
                    ("fresh b one", "fresh b two"),
                    ("fresh v one", "fresh v two"),
                ],
            )
            self.assertNotIn(
                "fresh t one",
                [
                    prompt
                    for call in tokenizer.prompt_calls
                    for prompt in call
                ],
            )
            status = report["scientific_status"]
            self.assertFalse(status["test_split_evaluated"])
            self.assertFalse(status["model_weights_changed"])
            self.assertEqual(
                status[
                    "locked_direct_boundary_evaluations_per_batch"
                ],
                1,
            )
            self.assertEqual(
                status["locked_behavioral_interventions_per_batch"],
                1,
            )
            self.assertEqual(
                report["source_analysis"]["prompt_disjointness"][
                    "overlap_count"
                ],
                0,
            )
            selection = report["analysis"]["selection"]
            locked_id = selection["lock"]["locked_candidate_id"]
            self.assertIn(locked_id, selection["candidate_direct"])
            validation = report["analysis"]["validation"]
            self.assertEqual(
                validation["locked_candidate"]["candidate_id"],
                locked_id,
            )
            self.assertIn(
                "rank_projection_reference_behavior",
                validation,
            )
            self.assertTrue(
                validation["full_width_codec_delta_roundtrip"]["passed"]
            )
            accounting = validation["accounting"]
            self.assertGreater(
                accounting["fixed_runtime_codec_coefficient_count"],
                0,
            )
            self.assertGreater(accounting["graph_mac_count"], 0)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            self.assertFalse(raw["contains_model_weights"])
            self.assertFalse(raw["contains_prompt_text"])
            self.assertFalse(raw["contains_tokenizer_state"])
            self.assertNotIn("test", raw["protocol"]["tokenized_splits"])
            self.assertEqual(
                set(raw["fit"]["executors"]),
                set(selection["candidate_direct"]),
            )
            for state in raw["fit"]["executors"].values():
                self.assertEqual(
                    state["artifact_kind"],
                    "fisher_graph.residual_gated_causal_modal_executor",
                )
            loaded = load_gemma3_gated_executor_artifact(output)
            self.assertEqual(
                loaded["locked_candidate"]["candidate_id"],
                locked_id,
            )
            self.assertEqual(
                loaded["locked_executor"].config.input_modes,
                validation["locked_candidate"]["retained_rank"],
            )

            tampered = output.with_name("tampered.pt")
            changed = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            changed["selection"]["lock"]["locked_candidate_id"] = (
                "forged"
            )
            payload = {
                key: value
                for key, value in changed.items()
                if key
                not in {
                    "scientific_payload_sha256",
                    "report_sha256",
                }
            }
            changed["scientific_payload_sha256"] = (
                _scientific_payload_sha256(payload)
            )
            torch.save(changed, tampered)
            shutil.copyfile(
                output.with_suffix(".json"),
                tampered.with_suffix(".json"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "selection lock",
            ):
                load_gemma3_gated_executor_artifact(tampered)

            ledger_tampered = output.with_name("ledger-tampered.pt")
            changed = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            changed["selection"]["lock"]["ledger"][0]["direct"][
                "block_delta_nrmse"
            ] += 0.25
            payload = {
                key: value
                for key, value in changed.items()
                if key
                not in {
                    "scientific_payload_sha256",
                    "report_sha256",
                }
            }
            changed["scientific_payload_sha256"] = (
                _scientific_payload_sha256(payload)
            )
            torch.save(changed, ledger_tampered)
            shutil.copyfile(
                output.with_suffix(".json"),
                ledger_tampered.with_suffix(".json"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "ledger does not match candidate metrics",
            ):
                load_gemma3_gated_executor_artifact(ledger_tampered)
            serialized = repr(raw)
            for split in fresh.values():
                if isinstance(split, list):
                    for prompt in split:
                        self.assertNotIn(prompt, serialized)
            for name, parameter in model.named_parameters():
                self.assertEqual(
                    int(parameter._version),
                    parameter_versions[name],
                )
                self.assertFalse(parameter.requires_grad)

    def test_cli_includes_decisive_widths_and_lag(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--weighted-artifact",
                "source.pt",
                "--retained-ranks",
                "320",
                "480",
                "--max-positive-lags",
                "8",
                "none",
            ]
        )
        self.assertEqual(parsed.ranks, [320, 480])
        self.assertEqual(parsed.max_positive_lags, [8, None])


if __name__ == "__main__":
    unittest.main()
