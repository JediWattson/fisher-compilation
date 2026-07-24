import json
import unittest
from pathlib import Path

import torch

from fisher_graph.associative import (
    AssociativeRecallTaskConfig,
    build_associative_recall_splits,
)
from fisher_graph.fused_executor_experiment import (
    _EXPECTED_LAZY_STORAGE,
    _VALIDATION_GATE,
    _compare_fused_to_unfused,
    _lazy_benchmark_comparison,
    _passes_fusion_gate,
    _require_no_sidecar_activity,
)


class FusedExecutorExperimentTests(unittest.TestCase):
    def test_equivalence_gate_accepts_small_numerical_delta(self) -> None:
        split = build_associative_recall_splits(
            AssociativeRecallTaskConfig(
                n_keys=2,
                n_values=2,
                train_fraction=0.5,
            )
        ).validation
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(
            split.samples,
            7,
            generator=generator,
        )
        fused = logits + 1e-7

        comparison = _compare_fused_to_unfused(
            split=split,
            unfused_logits=logits,
            fused_logits=fused,
        )

        self.assertTrue(
            comparison["argmax_predictions_exactly_equal"]
        )
        self.assertTrue(
            _passes_fusion_gate(comparison, _VALIDATION_GATE)
        )

    def test_equivalence_gate_rejects_changed_argmax(self) -> None:
        split = build_associative_recall_splits(
            AssociativeRecallTaskConfig(
                n_keys=2,
                n_values=2,
                train_fraction=0.5,
            )
        ).validation
        logits = torch.zeros(split.samples, 7)
        logits[:, 0] = 1
        fused = logits.clone()
        fused[0, 0] = 0
        fused[0, 1] = 2

        comparison = _compare_fused_to_unfused(
            split=split,
            unfused_logits=logits,
            fused_logits=fused,
        )

        self.assertFalse(
            comparison["argmax_predictions_exactly_equal"]
        )
        self.assertFalse(
            _passes_fusion_gate(comparison, _VALIDATION_GATE)
        )

    def test_lazy_benchmark_comparison_reports_relative_regression(self) -> None:
        benchmark = [
            {
                "batch_size": 1,
                "timings": {
                    "monolithic": {"median_microseconds": 100.0},
                    "lazy": {"median_microseconds": 105.0},
                },
                "speedup_ratios": {
                    "lazy_vs_monolithic": 100.0 / 105.0,
                },
            },
            {
                "batch_size": 8,
                "timings": {
                    "monolithic": {"median_microseconds": 120.0},
                    "lazy": {"median_microseconds": 114.0},
                },
                "speedup_ratios": {
                    "lazy_vs_monolithic": 120.0 / 114.0,
                },
            },
        ]

        comparison = _lazy_benchmark_comparison(benchmark)

        self.assertFalse(comparison["hard_latency_gate_applied"])
        per_batch = comparison["per_batch"]
        self.assertEqual(len(per_batch), 2)
        self.assertAlmostEqual(
            per_batch[0]["lazy_latency_regression_fraction"],
            0.05,
        )
        self.assertAlmostEqual(
            per_batch[1]["lazy_latency_regression_fraction"],
            -0.05,
        )
        self.assertAlmostEqual(
            comparison[
                "geometric_mean_lazy_to_monolithic_latency_ratio"
            ],
            (1.05 * 0.95) ** 0.5,
        )

    def test_fast_status_contract_rejects_sidecar_activity(self) -> None:
        status = {
            "residency": "unloaded",
            "loaded": False,
            "load_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "instrumented_path_calls": 0,
            "resident_sidecar_tensor_bytes": 0,
            "sidecar_file_bytes_read": 0,
        }
        _require_no_sidecar_activity(status, phase="unit test")

        changed = dict(status)
        changed["load_attempts"] = 1
        with self.assertRaisesRegex(
            RuntimeError,
            "touched instrumentation",
        ):
            _require_no_sidecar_activity(
                changed,
                phase="unit test",
            )

    def test_locked_lazy_storage_numbers_are_explicit(self) -> None:
        self.assertEqual(
            _EXPECTED_LAZY_STORAGE,
            {
                "fast_stack_resident_tensor_bytes": 199_808,
                "sidecar_resident_tensor_bytes": 203_648,
                "model_shell_tensor_bytes": 6_144,
                "default_full_runtime_resident_tensor_bytes": 205_952,
                "loaded_full_runtime_resident_tensor_bytes": 409_600,
            },
        )

    def test_checked_in_report_records_lazy_runtime_contract(self) -> None:
        artifact_dir = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "associative_recall"
        )
        report = json.loads(
            (artifact_dir / "fused_executor_report.json").read_text()
        )

        self.assertEqual(report["format_version"], 2)
        self.assertTrue(
            (artifact_dir / "fused_modal_runtime.pt").is_file()
        )
        self.assertEqual(
            report["storage"]["lazy_storage_contract"],
            _EXPECTED_LAZY_STORAGE,
        )
        self.assertTrue(
            report["lazy_fast_runtime_status"][
                "zero_sidecar_loads_throughout"
            ]
        )
        fast_status = report["lazy_fast_runtime_status"][
            "after_benchmark"
        ]
        self.assertEqual(fast_status["successful_loads"], 0)
        self.assertEqual(fast_status["sidecar_file_bytes_read"], 0)
        trace = report["dispatch_and_trace_contract"]
        self.assertEqual(
            trace["status_after_first_capture"]["successful_loads"],
            1,
        )
        self.assertEqual(
            trace["status_after_reused_intervention"][
                "successful_loads"
            ],
            1,
        )
        self.assertEqual(
            trace["status_after_explicit_eviction"][
                "resident_sidecar_tensor_bytes"
            ],
            0,
        )
        for batch in report["benchmark"]:
            self.assertEqual(
                set(batch["timings"]),
                {"teacher", "unfused", "monolithic", "lazy"},
            )


if __name__ == "__main__":
    unittest.main()
