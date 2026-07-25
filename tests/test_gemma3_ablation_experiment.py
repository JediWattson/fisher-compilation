import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _scientific_payload_sha256,
    build_parser,
    default_gemma3_ablation_output,
    load_gemma3_ablation_artifact,
    run_gemma3_ablation,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"
    name_or_path = "synthetic-recording-tokenizer"
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


def prompt_payload() -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": "synthetic_test_only",
        "calibration_a": ["a one", "a two"],
        "calibration_b": ["b one", "b two"],
        "validation": ["v one", "v two"],
        "test": ["t one", "t two"],
    }


def save_resigned_artifact(
    raw: dict[str, object],
    path: Path,
) -> None:
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    raw["scientific_payload_sha256"] = _scientific_payload_sha256(
        payload
    )
    torch.save(raw, path)


class Gemma3AblationExperimentTests(unittest.TestCase):
    def test_synthetic_joint_full_width_ablation_artifact(self) -> None:
        torch.manual_seed(1771)
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
        parameter_objects = {
            name: parameter
            for name, parameter in model.named_parameters()
        }
        parameter_versions = {
            name: int(parameter._version)
            for name, parameter in parameter_objects.items()
        }
        tokenizer = RecordingTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "splits.json"
            prompt_path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            output = root / "analysis" / "ablation.pt"
            with patch(
                "fisher_graph.gemma3_ablation_experiment.load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_ablation_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_ablation(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=prompt_path,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(8, 4, 0),
                    sketch_rows=9,
                    include_singletons=True,
                    full_rank_nll_atol=1e-5,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertFalse(model.training)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in model.parameters()
                )
            )
            for name, parameter in model.named_parameters():
                self.assertIs(parameter, parameter_objects[name])
                self.assertEqual(
                    int(parameter._version),
                    parameter_versions[name],
                )

            tokenized_prompts = [
                prompt
                for call in tokenizer.prompt_calls
                for prompt in call
            ]
            self.assertEqual(
                tokenized_prompts,
                [
                    "a one",
                    "a two",
                    "b one",
                    "b two",
                    "v one",
                    "v two",
                ],
            )
            self.assertTrue(
                report["scientific_status"]["full_rank_identity_passed"]
            )
            self.assertFalse(
                report["scientific_status"]["test_split_evaluated"]
            )
            self.assertFalse(
                report["scientific_status"]["compression_claim"]
            )
            self.assertEqual(
                report["protocol"]["canonical_boundaries"],
                (
                    "layer.0.input",
                    "layer.0.output",
                    "layer.1.output",
                ),
            )
            self.assertEqual(
                report["protocol"]["gated_output_sites"],
                ("layer.0.output", "layer.1.output"),
            )
            self.assertEqual(report["protocol"]["extraction_rank"], 8)
            self.assertEqual(report["protocol"]["ranks"], (8, 4, 0))
            self.assertEqual(
                len(report["analysis"]["validation"]["conditions"]),
                9,
            )
            self.assertIn(
                "joint.rank_0",
                {
                    condition["condition"]["name"]
                    for condition in report["analysis"]["validation"][
                        "conditions"
                    ]
                },
            )
            identity = report["analysis"]["full_rank_identity"]
            self.assertTrue(identity["passed"])
            self.assertTrue(identity["top1_identity"])
            self.assertLessEqual(
                identity[
                    "maximum_absolute_example_delta_nll_per_token"
                ],
                identity["tolerance"],
            )
            for audit in report["analysis"]["calibration"][
                "full_width_basis_audit"
            ].values():
                self.assertEqual(audit["width"], 8)
                self.assertEqual(audit["modes"], 8)
                self.assertTrue(audit["trace_passed"])

            raw = torch.load(output, map_location="cpu", weights_only=True)
            self.assertFalse(raw["contains_model_weights"])
            self.assertFalse(raw["contains_prompt_text"])
            self.assertFalse(raw["contains_tokenizer_state"])
            self.assertNotIn("model_state_dict", raw)
            self.assertNotIn("test", raw["protocol"]["tokenized_splits"])
            serialized = repr(raw)
            for prompt in (
                "a one",
                "a two",
                "b one",
                "b two",
                "v one",
                "v two",
                "t one",
                "t two",
            ):
                self.assertNotIn(prompt, serialized)
            loaded = load_gemma3_ablation_artifact(output)
            self.assertEqual(
                tuple(loaded["calibration"].bases),
                report["protocol"]["canonical_boundaries"],
            )
            self.assertEqual(
                loaded["metadata"]["scientific_payload_sha256"],
                raw["scientific_payload_sha256"],
            )

            tampered = root / "analysis" / "tampered.pt"
            raw = torch.load(output, map_location="cpu", weights_only=True)
            first_basis = next(
                iter(raw["calibration"]["collection"]["bases"].values())
            )
            first_basis["fisher"]["vectors"][0, 0] += 0.25
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "scientific payload digest",
            ):
                load_gemma3_ablation_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            first_site = next(
                iter(raw["calibration"]["full_width_basis_audit"])
            )
            raw["calibration"]["full_width_basis_audit"][first_site][
                "absolute_trace_error"
            ] += 1.0
            save_resigned_artifact(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "basis audit mismatch",
            ):
                load_gemma3_ablation_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            first_example = raw["validation"]["modal_ablation"][
                "conditions"
            ][0]["examples"][0]
            first_example["ablated_summed_nll"] += 1.0
            save_resigned_artifact(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "ablated NLL per token",
            ):
                load_gemma3_ablation_artifact(tampered)

            for field, replacement, message in (
                ("rows_seen", None, "row accounting"),
                ("score_reduction", "mean", "estimator semantics"),
                (
                    "normalizer",
                    "supervised_positions",
                    "estimator semantics",
                ),
                ("scope", "positionwise", "estimator semantics"),
                ("sketch_rows", None, "estimator semantics"),
            ):
                with self.subTest(fisher_field=field):
                    raw = torch.load(
                        output,
                        map_location="cpu",
                        weights_only=True,
                    )
                    first_basis = next(
                        iter(
                            raw["calibration"]["collection"][
                                "bases"
                            ].values()
                        )
                    )
                    fisher = first_basis["fisher"]
                    fisher[field] = (
                        fisher[field] + 1
                        if replacement is None
                        else replacement
                    )
                    save_resigned_artifact(raw, tampered)
                    with self.assertRaisesRegex(ValueError, message):
                        load_gemma3_ablation_artifact(tampered)

            raw = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            first_basis = next(
                iter(
                    raw["calibration"]["collection"]["bases"].values()
                )
            )
            fisher = first_basis["fisher"]
            fisher["observations"] += 1
            fisher["rows_seen"] += 1
            fisher["fisher_trace"] = (
                fisher["squared_gradient_norm_sum"]
                / fisher["observations"]
            )
            save_resigned_artifact(raw, tampered)
            with self.assertRaisesRegex(ValueError, "row accounting"):
                load_gemma3_ablation_artifact(tampered)

            raw = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            for condition in raw["validation"]["modal_ablation"][
                "conditions"
            ]:
                first, second = condition["examples"]
                first["supervised_tokens"] += 1
                second["supervised_tokens"] -= 1
                total_matches = condition["aggregate"]["top1_matches"]
                first["top1_matches"] = max(
                    0,
                    total_matches - second["supervised_tokens"],
                )
                second["top1_matches"] = (
                    total_matches - first["top1_matches"]
                )
                for example in (first, second):
                    tokens = example["supervised_tokens"]
                    baseline_nll = example["baseline_summed_nll"]
                    ablated_nll = example["ablated_summed_nll"]
                    example["baseline_nll_per_token"] = (
                        baseline_nll / tokens
                    )
                    example["ablated_nll_per_token"] = (
                        ablated_nll / tokens
                    )
                    example["delta_summed_nll"] = (
                        ablated_nll - baseline_nll
                    )
                    example["delta_nll_per_token"] = (
                        ablated_nll - baseline_nll
                    ) / tokens
                    example["top1_agreement_to_baseline"] = (
                        example["top1_matches"] / tokens
                    )
            save_resigned_artifact(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "per-example supervised-token counts",
            ):
                load_gemma3_ablation_artifact(tampered)

            for field, replacement, message in (
                ("hidden_size", 9, "hidden size"),
                ("num_hidden_layers", 1, "layer range"),
            ):
                with self.subTest(model_field=field):
                    raw = torch.load(
                        output,
                        map_location="cpu",
                        weights_only=True,
                    )
                    raw["model"][field] = replacement
                    save_resigned_artifact(raw, tampered)
                    with self.assertRaisesRegex(ValueError, message):
                        load_gemma3_ablation_artifact(tampered)

            report_path = output.with_suffix(".json")
            original_report = report_path.read_text(encoding="utf-8")
            report_payload = json.loads(original_report)
            report_payload["scientific_status"]["compression_claim"] = True
            report_path.write_text(
                json.dumps(report_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "JSON report digest",
            ):
                load_gemma3_ablation_artifact(output)
            report_path.write_text(original_report, encoding="utf-8")
            load_gemma3_ablation_artifact(output)

    def test_default_output_cli_aliases_and_guard(self) -> None:
        self.assertEqual(
            default_gemma3_ablation_output(
                "google/gemma-3-270m",
                4,
                6,
            ),
            Path(
                ".local-runs/google--gemma-3-270m/"
                "layers-4-6-modal-ablation.pt"
            ),
        )
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "--retained-ranks",
                "640",
                "0",
                "--include-single-sites",
            ]
        )
        self.assertEqual(parsed.ranks, [640, 0])
        self.assertTrue(parsed.include_singletons)
        legacy = parser.parse_args(
            ["--ranks", "640", "--include-singletons"]
        )
        self.assertEqual(legacy.ranks, [640])
        self.assertTrue(legacy.include_singletons)

        model = ToyTransformer(
            TransformerConfig(
                vocab_size=17,
                max_sequence_length=4,
                d_model=4,
                n_heads=1,
                n_layers=1,
                d_ff=8,
                dropout=0.0,
            )
        ).eval()
        model.requires_grad_(False)
        guard = _FrozenModelTensorGuard(model)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "mutated in place"):
            guard.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
