import unittest

import torch

from fisher_graph.fused_benchmark import (
    benchmark_batch_sizes,
    one_torch_thread,
)


class FusedBenchmarkTests(unittest.TestCase):
    def test_rotates_systems_and_reports_batches_and_speedups(self) -> None:
        observed_inference_modes: list[bool] = []

        def operation() -> torch.Tensor:
            observed_inference_modes.append(
                torch.is_inference_mode_enabled()
            )
            return torch.ones(2).square()

        reports = benchmark_batch_sizes(
            {
                8: {
                    "teacher": operation,
                    "unfused": operation,
                    "fused": operation,
                },
                1: {
                    "teacher": operation,
                    "unfused": operation,
                    "fused": operation,
                },
            },
            repeats=4,
            minimum_block_seconds=1e-5,
            warmup_iterations=1,
            minimum_warmup_seconds=0,
        )

        self.assertEqual([report.batch_size for report in reports], [1, 8])
        expected_orders = (
            ("teacher", "unfused", "fused"),
            ("unfused", "fused", "teacher"),
            ("fused", "teacher", "unfused"),
            ("teacher", "unfused", "fused"),
        )
        for report in reports:
            self.assertEqual(report.round_orders, expected_orders)
            self.assertEqual(
                set(report.speedup_ratios),
                {"fused_vs_unfused", "fused_vs_teacher"},
            )
            for timing in report.timings.values():
                self.assertEqual(timing.repeats, 4)
                self.assertEqual(len(timing.raw_microseconds), 4)
                self.assertGreater(timing.median_microseconds, 0)
                self.assertGreaterEqual(timing.calibration_seconds, 1e-5)
            for throughput in report.examples_per_second.values():
                self.assertGreater(throughput, 0)
            for speedup in report.speedup_ratios.values():
                self.assertGreater(speedup, 0)
        self.assertTrue(observed_inference_modes)
        self.assertTrue(all(observed_inference_modes))

    def test_one_thread_context_restores_previous_setting(self) -> None:
        original_threads = torch.get_num_threads()
        alternate_threads = 2 if original_threads != 2 else 3
        try:
            torch.set_num_threads(alternate_threads)
            with one_torch_thread():
                self.assertEqual(torch.get_num_threads(), 1)
            self.assertEqual(torch.get_num_threads(), alternate_threads)

            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with one_torch_thread():
                    self.assertEqual(torch.get_num_threads(), 1)
                    raise RuntimeError("deliberate")
            self.assertEqual(torch.get_num_threads(), alternate_threads)
        finally:
            torch.set_num_threads(original_threads)

    def test_rejects_inconsistent_systems_and_invalid_speedup(self) -> None:
        operation = lambda: None
        with self.assertRaisesRegex(ValueError, "same benchmark systems"):
            benchmark_batch_sizes(
                {
                    1: {"a": operation},
                    2: {"b": operation},
                },
                repeats=1,
                minimum_block_seconds=1e-5,
                warmup_iterations=0,
                minimum_warmup_seconds=0,
            )
        with self.assertRaisesRegex(ValueError, "system names"):
            benchmark_batch_sizes(
                {1: {"a": operation, "b": operation}},
                repeats=1,
                minimum_block_seconds=1e-5,
                warmup_iterations=0,
                minimum_warmup_seconds=0,
                speedup_pairs={"unknown": ("a", "missing")},
            )


if __name__ == "__main__":
    unittest.main()
