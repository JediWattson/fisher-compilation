import math
import unittest
from collections.abc import Iterable

from fisher_graph.model_runtime_benchmark import benchmark_model_runtimes


class ScriptedCounter:
    def __init__(self, durations: Iterable[float]) -> None:
        self._durations = iter(durations)
        self._now = 0.0
        self._started = False

    def __call__(self) -> float:
        if not self._started:
            self._started = True
            return self._now
        self._now += next(self._durations)
        self._started = False
        return self._now


class ModelRuntimeBenchmarkTests(unittest.TestCase):
    def test_reports_cold_quantiles_raw_rounds_and_throughput(self) -> None:
        counter = ScriptedCounter((0.5, 1.0, 2.0, 3.0, 4.0))
        operation_calls = 0

        def operation() -> None:
            nonlocal operation_calls
            operation_calls += 1

        report = benchmark_model_runtimes(
            {"candidate": operation},
            processed_token_count=10,
            rounds=4,
            warmup_calls=2,
            perf_counter=counter,
        )

        self.assertEqual(report.system_names, ("candidate",))
        self.assertEqual(report.rounds, 4)
        self.assertEqual(report.warmup_calls, 2)
        self.assertEqual(operation_calls, 7)
        self.assertEqual(
            report.round_orders,
            (("candidate",),) * 4,
        )
        timing = report.timings["candidate"]
        self.assertEqual(timing.cold_seconds, 0.5)
        self.assertEqual(timing.raw_round_seconds, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(timing.median_seconds, 2.5)
        self.assertAlmostEqual(timing.p10_seconds, 1.3)
        self.assertAlmostEqual(timing.p90_seconds, 3.7)
        self.assertAlmostEqual(timing.p95_seconds, 3.85)
        self.assertEqual(timing.processed_token_count, 10)
        self.assertEqual(timing.tokens_per_second, 4.0)

    def test_rotates_and_keeps_prepare_outside_timed_region(self) -> None:
        events: list[str] = []

        class EventCounter:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                events.append("counter")
                self.now += 0.25
                return self.now

        def operation(name: str):
            def run() -> None:
                events.append(f"operation:{name}")

            return run

        report = benchmark_model_runtimes(
            {
                "candidate": operation("candidate"),
                "native": operation("native"),
                "source": operation("source"),
            },
            expected_systems=("native", "source", "candidate"),
            processed_token_count=32,
            rounds=4,
            warmup_calls=1,
            prepare_system=lambda name: events.append(f"prepare:{name}"),
            synchronize=lambda: events.append("synchronize"),
            perf_counter=EventCounter(),
        )

        self.assertEqual(
            report.round_orders,
            (
                ("native", "source", "candidate"),
                ("source", "candidate", "native"),
                ("candidate", "native", "source"),
                ("native", "source", "candidate"),
            ),
        )
        timed_pattern = (
            "prepare:native",
            "synchronize",
            "counter",
            "operation:native",
            "synchronize",
            "counter",
        )
        self.assertEqual(tuple(events[:6]), timed_pattern)
        self.assertEqual(events.count("counter"), 30)
        self.assertEqual(events.count("synchronize"), 36)
        for name in report.system_names:
            self.assertEqual(events.count(f"prepare:{name}"), 6)
            self.assertEqual(events.count(f"operation:{name}"), 6)

        first_warmup = events[18:22]
        self.assertEqual(
            first_warmup,
            [
                "prepare:native",
                "synchronize",
                "operation:native",
                "synchronize",
            ],
        )

    def test_rejects_inexact_system_sets_and_invalid_operations(self) -> None:
        operation = lambda: None
        with self.assertRaisesRegex(ValueError, "at least one"):
            benchmark_model_runtimes(
                {},
                processed_token_count=1,
            )
        with self.assertRaisesRegex(ValueError, "exact expected system set"):
            benchmark_model_runtimes(
                {"native": operation},
                expected_systems=("native", "candidate"),
                processed_token_count=1,
            )
        with self.assertRaisesRegex(ValueError, "unique nonempty"):
            benchmark_model_runtimes(
                {"native": operation},
                expected_systems=("native", "native"),
                processed_token_count=1,
            )
        with self.assertRaisesRegex(TypeError, "must be callable"):
            benchmark_model_runtimes(
                {"native": None},  # type: ignore[dict-item]
                processed_token_count=1,
            )

    def test_rejects_nonpositive_and_nonfinite_measurements(self) -> None:
        for duration in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(duration=duration):
                counter = ScriptedCounter((duration,))
                with self.assertRaisesRegex(
                    ValueError,
                    "time must be finite and positive",
                ):
                    benchmark_model_runtimes(
                        {"native": lambda: None},
                        processed_token_count=1,
                        rounds=1,
                        warmup_calls=0,
                        perf_counter=counter,
                    )

    def test_rejects_invalid_counts_and_callbacks(self) -> None:
        operation = {"native": lambda: None}
        for keyword, value in (
            ("rounds", 0),
            ("warmup_calls", -1),
            ("processed_token_count", 0),
        ):
            with self.subTest(keyword=keyword):
                arguments = {
                    "operations": operation,
                    "processed_token_count": 1,
                    keyword: value,
                }
                with self.assertRaisesRegex(ValueError, keyword):
                    benchmark_model_runtimes(**arguments)

        with self.assertRaisesRegex(TypeError, "prepare_system"):
            benchmark_model_runtimes(
                operation,
                processed_token_count=1,
                prepare_system=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "synchronize"):
            benchmark_model_runtimes(
                operation,
                processed_token_count=1,
                synchronize=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
