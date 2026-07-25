import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_trajectory_experiment import (
    _report_sha256,
    _scientific_payload_sha256,
    default_gemma3_trajectory_output,
    load_gemma3_trajectory_artifact,
    run_gemma3_trajectory,
)
from fisher_graph.model import ToyTransformer


class TrajectoryTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        rows = [
            [2, 4 + index, 7 + index, 1]
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


class Gemma3TrajectoryExperimentTests(unittest.TestCase):
    def test_synthetic_multi_boundary_artifact_and_loader(self) -> None:
        torch.manual_seed(1441)
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
        tokenizer = TrajectoryTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "splits.json"
            prompt_path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            output = root / "analysis" / "trajectory.pt"
            with patch(
                "fisher_graph.gemma3_trajectory_experiment.load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_trajectory_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_trajectory(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=prompt_path,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(2, 4),
                    sketch_rows=6,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            loaded = load_gemma3_trajectory_artifact(output)
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertEqual(
                tuple(
                    loaded["calibration"]["calibration_full"].bases
                ),
                (
                    "layer.0.input",
                    "layer.0.output",
                    "layer.1.output",
                ),
            )
            self.assertFalse(report["scientific_status"]["test_split_evaluated"])
            self.assertFalse(report["scientific_status"]["executor_fit"])
            self.assertTrue(
                report["scientific_status"][
                    "cross_position_reverse_gradient_predictor_evaluated"
                ]
            )
            self.assertFalse(
                report["scientific_status"][
                    "per_position_jacobian_blocks_measured"
                ]
            )
            self.assertEqual(
                set(report["protocol"]["tokenized_splits"]),
                {
                    "calibration_a",
                    "calibration_b",
                    "calibration_full",
                    "transport_fit",
                    "validation",
                },
            )
            self.assertNotIn(
                "test",
                report["protocol"]["tokenized_splits"],
            )
            self.assertEqual(
                report["protocol"]["canonical_boundaries"],
                (
                    "layer.0.input",
                    "layer.0.output",
                    "layer.1.output",
                ),
            )
            self.assertEqual(len(report["analysis"]["edge_curves"]), 2)
            self.assertEqual(
                len(report["analysis"]["causal_edge_curves"]),
                3,
            )
            self.assertEqual(
                report["protocol"]["causal_comparison_pairs"][-1],
                ("layer.0.input", "layer.1.output"),
            )
            for curve in report["analysis"][
                "causal_edge_curves"
            ].values():
                for point in curve:
                    self.assertEqual(
                        [
                            window["max_lag"]
                            for window in point["lag_windows"]
                        ],
                        [0, 1, 4],
                    )
                    self.assertEqual(
                        point["row_local_ridge_baseline"][
                            "baseline_kind"
                        ],
                        "zero",
                    )
            self.assertIn("causal_transport_moments", loaded)
            self.assertIn("frozen_causal_transports", loaded)
            for curve in report["analysis"]["edge_curves"].values():
                self.assertEqual(
                    [point["rank"] for point in curve],
                    [2, 4],
                )
                for point in curve:
                    self.assertIn(
                        point["diagnostic_classification"],
                        {
                            "rank_budget_insufficient",
                            "inconclusive_basis_not_identifiable",
                            "persistent_subspace",
                            "predictable_rotation",
                            "unstructured_or_context_dependent_drift",
                            "mixed_or_inconclusive",
                        },
                    )

            legacy_raw = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            legacy_raw["format_version"] = 1
            for field in (
                "causal_comparison_pairs",
                "causal_visibility_windows",
                "causal_transport",
                "causal_transport_lags",
                "causal_transport_max_lag",
                "causal_transport_relative_ridge",
            ):
                legacy_raw["protocol"].pop(field)
            legacy_raw["transport_fit"].pop(
                "causal_score_gradient_moments"
            )
            legacy_raw["transport_fit"].pop(
                "causal_score_gradient_frozen"
            )
            legacy_raw["validation"].pop(
                "causal_score_gradient_moments"
            )
            legacy_raw["validation"].pop(
                "causal_score_gradient_evaluations"
            )
            legacy_report = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )
            legacy_report["format_version"] = 1
            for field in (
                "causal_comparison_pairs",
                "causal_visibility_windows",
                "causal_transport",
                "causal_transport_lags",
                "causal_transport_max_lag",
                "causal_transport_relative_ridge",
            ):
                legacy_report["protocol"].pop(field)
            legacy_report["analysis"].pop("causal_transport_fit")
            legacy_report["analysis"].pop("causal_edge_curves")
            legacy_status = legacy_report["scientific_status"]
            legacy_status["gradient_transport_scope"] = (
                "same_position_reverse_gradient_map"
            )
            for field in (
                "cross_position_reverse_gradient_predictor_evaluated",
                "per_position_jacobian_blocks_measured",
                "context_conditioning",
                "fit_validation_prompt_splits_verified_disjoint",
            ):
                legacy_status.pop(field)
            legacy_payload = {
                key: value
                for key, value in legacy_raw.items()
                if key != "report_sha256"
            }
            legacy_report["artifact"][
                "scientific_payload_sha256"
            ] = _scientific_payload_sha256(legacy_payload)
            legacy_report["artifact"]["tensor_output"] = "legacy.pt"
            legacy_raw["report_sha256"] = _report_sha256(legacy_report)
            legacy_output = root / "analysis" / "legacy.pt"
            torch.save(legacy_raw, legacy_output)
            legacy_output.with_suffix(".json").write_text(
                json.dumps(
                    legacy_report,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            legacy_loaded = load_gemma3_trajectory_artifact(
                legacy_output
            )
            self.assertEqual(
                legacy_loaded["metadata"]["format_version"],
                1,
            )
            self.assertNotIn(
                "causal_transport_moments",
                legacy_loaded,
            )

            report_path = output.with_suffix(".json")
            original_report_text = report_path.read_text(encoding="utf-8")
            tampered_report = json.loads(original_report_text)
            tampered_report["analysis"]["block_classification"] = (
                "tampered_sidecar_classification"
            )
            report_path.write_text(
                json.dumps(
                    tampered_report,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "JSON report digest",
            ):
                load_gemma3_trajectory_artifact(output)
            report_path.write_text(
                original_report_text,
                encoding="utf-8",
            )
            load_gemma3_trajectory_artifact(output)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["protocol"]["canonical_boundaries"] = (
                "invented.input",
                "invented.output",
            )
            tampered = root / "analysis" / "tampered.pt"
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "boundaries",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(iter(raw["transport_fit"]["frozen"]))
            raw["transport_fit"]["frozen"][edge]["activation"][2][
                "source_basis_sha256"
            ] = "0" * 64
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "basis binding",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(iter(raw["transport_fit"]["frozen"]))
            raw["transport_fit"]["frozen"][edge]["activation"][2][
                "matrix"
            ][0, 0] += 1.0
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "does not match calibration moments",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            geometry_point = raw["geometry"]["calibration_full_depth"][
                "depth_alignment"
            ][0]["points"][0]
            geometry_point["principal_cosines"] = torch.zeros_like(
                geometry_point["principal_cosines"]
            )
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "geometry.*recomputed",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            boundary = next(iter(raw["validation"]["own_rayleigh"]))
            raw["validation"]["own_rayleigh"][boundary][
                "basis_sha256"
            ] = "0" * 64
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "Rayleigh result",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(
                iter(raw["validation"]["transport_evaluations"])
            )
            raw["validation"]["transport_evaluations"][edge]["activation"][2][
                "transport_squared_error"
            ] += 1.0
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "held-out transport evaluation.*recomputed",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(
                iter(raw["transport_fit"]["causal_score_gradient_frozen"])
            )
            raw["transport_fit"]["causal_score_gradient_frozen"][edge][2][1][
                "matrix"
            ][0, 0] += 1.0
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "lag matrix norms|frozen causal transport",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(
                iter(raw["validation"]["causal_score_gradient_evaluations"])
            )
            raw["validation"]["causal_score_gradient_evaluations"][edge][2][1][
                "transport_squared_error"
            ] += 1.0
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "causal modal evaluation|held-out causal transport",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["protocol"]["test_evaluation_logits"] = torch.ones(2)
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "protocol fields",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["validation"]["mean_loss"] = float("nan")
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "validation mean loss",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["validation"]["per_prompt_influence"][0] = None
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "per-prompt influence fields",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            first_influence = raw["validation"]["per_prompt_influence"][0]
            boundary = next(iter(first_influence["boundaries"]))
            first_influence["boundaries"][boundary][
                "fisher_trace_sum"
            ] = -1.0
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "boundary influence values",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            edge = next(iter(raw["validation"]["cross_rayleigh_sums"]))
            cross_modes = raw["validation"]["cross_rayleigh_sums"][edge][
                "target_gradients_in_source_basis"
            ]
            original_cross_modes = cross_modes.clone()
            sorted_cross_modes = torch.sort(cross_modes).values
            if torch.equal(sorted_cross_modes, cross_modes):
                sorted_cross_modes = torch.flip(sorted_cross_modes, dims=(0,))
            self.assertFalse(torch.equal(sorted_cross_modes, cross_modes))
            self.assertAlmostEqual(
                sorted_cross_modes.sum().item(),
                original_cross_modes.sum().item(),
            )
            raw["validation"]["cross_rayleigh_sums"][edge][
                "target_gradients_in_source_basis"
            ] = sorted_cross_modes
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "target cross-Rayleigh mode accounting",
            ):
                load_gemma3_trajectory_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            influence = raw["validation"]["per_prompt_influence"]
            boundary = next(iter(influence[0]["boundaries"]))
            first_values = influence[0]["boundaries"][boundary]
            second_values = influence[1]["boundaries"][boundary]
            first_trace = first_values["fisher_trace_sum"]
            second_trace = second_values["fisher_trace_sum"]
            if first_trace > 0:
                transfer = first_trace / 4
                first_values["fisher_trace_sum"] -= transfer
                second_values["fisher_trace_sum"] += transfer
            else:
                self.assertGreater(second_trace, 0)
                transfer = second_trace / 4
                first_values["fisher_trace_sum"] += transfer
                second_values["fisher_trace_sum"] -= transfer
            self.assertAlmostEqual(
                first_values["fisher_trace_sum"]
                + second_values["fisher_trace_sum"],
                first_trace + second_trace,
            )
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "per-prompt influence digest",
            ):
                load_gemma3_trajectory_artifact(tampered)

    def test_default_output_and_invalid_options(self) -> None:
        self.assertEqual(
            default_gemma3_trajectory_output(
                "google/gemma-3-270m",
                4,
                6,
            ),
            Path(
                ".local-runs/google--gemma-3-270m/"
                "layers-4-6-modal-trajectory.pt"
            ),
        )
        invalid = (
            (
                {"start_layer": 2, "end_layer": 1},
                "nonnegative and ascending",
            ),
            ({"ranks": ()}, "ranks cannot be empty"),
            (
                {"ranks": (4,), "sketch_rows": 4},
                "greater than the maximum rank",
            ),
            ({"output": Path("report.json")}, ".pt suffix"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs), patch(
                "fisher_graph.gemma3_trajectory_experiment.load_gemma3"
            ) as loader:
                with self.assertRaisesRegex(ValueError, message):
                    run_gemma3_trajectory(**kwargs)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
