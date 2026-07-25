from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fisher_graph.mlx_benchmark import (
    MLX_BENCHMARK_SYSTEMS,
    benchmark_synchronized_operations,
    make_synchronized_mlx_operation,
    render_mlx_benchmark_markdown,
    unpack_packed_triangular_kernel,
)
from fisher_graph.mlx_executor import mlx_metal_kernel_source_sha256


ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "associative_recall"
)


def test_unpack_packed_triangular_kernel_preserves_pair_order() -> None:
    packed = np.arange(6 * 2, dtype=np.float32).reshape(6, 2, 1)

    dense = unpack_packed_triangular_kernel(
        packed,
        sequence_length=3,
    )

    assert dense.shape == (3, 3, 2, 1)
    assert dense.dtype == np.float32
    assert np.array_equal(dense[0, 0], packed[0])
    assert np.array_equal(dense[1, 0], packed[1])
    assert np.array_equal(dense[1, 1], packed[2])
    assert np.array_equal(dense[2, 0], packed[3])
    assert np.array_equal(dense[2, 1], packed[4])
    assert np.array_equal(dense[2, 2], packed[5])
    assert np.count_nonzero(dense[0, 1:]) == 0
    assert np.count_nonzero(dense[1, 2:]) == 0


def test_unpack_packed_triangular_kernel_rejects_invalid_shapes() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        unpack_packed_triangular_kernel(
            [[[1.0]]],  # type: ignore[arg-type]
            sequence_length=1,
        )
    with pytest.raises(ValueError, match="shape"):
        unpack_packed_triangular_kernel(
            np.ones((3, 2), dtype=np.float32),
            sequence_length=2,
        )
    with pytest.raises(ValueError, match="expected 6"):
        unpack_packed_triangular_kernel(
            np.ones((5, 2, 1), dtype=np.float32),
            sequence_length=3,
        )


def test_synchronized_operation_creates_and_finishes_each_output() -> None:
    events: list[tuple[str, object]] = []

    class FakeCore:
        @staticmethod
        def eval(output: object) -> None:
            events.append(("eval", output))

        @staticmethod
        def synchronize() -> None:
            events.append(("synchronize", None))

    next_output = 0

    def forward(prefix: str) -> tuple[str, int]:
        nonlocal next_output
        next_output += 1
        output = (prefix, next_output)
        events.append(("forward", output))
        return output

    operation = make_synchronized_mlx_operation(
        FakeCore,
        forward,
        "lazy",
    )
    first = operation()
    second = operation()

    assert first == ("lazy", 1)
    assert second == ("lazy", 2)
    assert events == [
        ("forward", first),
        ("eval", first),
        ("synchronize", None),
        ("forward", second),
        ("eval", second),
        ("synchronize", None),
    ]


def test_benchmark_reports_cold_warmup_and_rotating_raw_rounds() -> None:
    call_counts: dict[tuple[int, str], int] = {}

    def operation(batch_size: int, name: str):
        def run() -> int:
            key = (batch_size, name)
            call_counts[key] = call_counts.get(key, 0) + 1
            return sum(range(20))

        return run

    operations = {
        batch_size: {
            name: operation(batch_size, name)
            for name in MLX_BENCHMARK_SYSTEMS
        }
        for batch_size in (8, 1)
    }
    reports = benchmark_synchronized_operations(
        operations,
        rounds=4,
        warmup_calls=2,
        minimum_warmup_seconds=0.0,
        iterations_per_round=3,
    )

    assert [report.batch_size for report in reports] == [1, 8]
    expected_orders = (
        MLX_BENCHMARK_SYSTEMS,
        MLX_BENCHMARK_SYSTEMS[1:] + MLX_BENCHMARK_SYSTEMS[:1],
        MLX_BENCHMARK_SYSTEMS[2:] + MLX_BENCHMARK_SYSTEMS[:2],
        MLX_BENCHMARK_SYSTEMS,
    )
    expected_calls = 1 + 2 + 4 * 3
    for report in reports:
        assert report.round_orders == expected_orders
        assert set(report.timings) == set(MLX_BENCHMARK_SYSTEMS)
        assert set(report.speedup_ratios) == {
            "packed_compiled_vs_dense_compiled",
            "packed_metal_vs_dense_compiled",
            "packed_metal_vs_packed_compiled",
        }
        for name, timing in report.timings.items():
            assert timing.first_observed_call_microseconds > 0
            assert timing.warmup_calls == 2
            assert timing.iterations_per_round == 3
            assert timing.rounds == 4
            assert len(timing.raw_microseconds) == 4
            assert timing.median_microseconds > 0
            assert call_counts[(report.batch_size, name)] == expected_calls
        assert all(
            value > 0
            for value in report.examples_per_second.values()
        )
        assert all(value > 0 for value in report.speedup_ratios.values())

    serialized = json.dumps(
        [
            {
                "batch_size": report.batch_size,
                "raw": {
                    name: timing.raw_microseconds
                    for name, timing in report.timings.items()
                },
            }
            for report in reports
        ]
    )
    assert '"batch_size": 1' in serialized


def test_benchmark_rejects_invalid_contracts() -> None:
    operation = lambda: None
    with pytest.raises(ValueError, match="same order"):
        benchmark_synchronized_operations(
            {
                1: {"a": operation, "b": operation},
                2: {"b": operation, "a": operation},
            },
            rounds=1,
            warmup_calls=0,
            minimum_warmup_seconds=0.0,
            iterations_per_round=1,
        )
    with pytest.raises(ValueError, match="speedup pairs"):
        benchmark_synchronized_operations(
            {1: {"a": operation, "b": operation}},
            rounds=1,
            warmup_calls=0,
            minimum_warmup_seconds=0.0,
            iterations_per_round=1,
            speedup_pairs={"bad": ("a", "missing")},
        )
    with pytest.raises(ValueError, match="rounds"):
        benchmark_synchronized_operations(
            {1: {"a": operation}},
            rounds=0,
            minimum_warmup_seconds=0.0,
        )


def test_markdown_renderer_labels_scope_and_speedups() -> None:
    timing = {
        "median_microseconds": 10.0,
    }
    report = {
        "environment": {
            "device_info": {"device_name": "Test GPU"},
            "mlx_version": "1.2.3",
        },
        "output_validation": {"gate_passed": True},
        "measurement_contract": {"rounds": 7},
        "runtime": {
            "causal_pair_count": 36,
            "state_bytes": 124_544,
        },
        "batches": [
            {
                "batch_size": 8,
                "timings": {
                    name: dict(timing)
                    for name in MLX_BENCHMARK_SYSTEMS
                },
                "speedup_ratios": {
                    "packed_metal_vs_dense_compiled": 1.25,
                    "packed_metal_vs_packed_compiled": 2.5,
                },
            }
        ],
    }

    markdown = render_mlx_benchmark_markdown(report)

    assert "Test GPU" in markdown
    assert "| 8 | 10.000 us" in markdown
    assert "1.250x" in markdown
    assert "7 measurement rounds" in markdown
    assert "not a task-validation report" in markdown


def test_committed_mlx_benchmark_report_is_self_consistent() -> None:
    report_path = ARTIFACTS / "mlx_metal_benchmark.json"
    report = json.loads(report_path.read_text())
    assert (
        (ARTIFACTS / "mlx_metal_benchmark.md").read_text()
        == render_mlx_benchmark_markdown(report)
    )

    assert report["schema"] == "fisher_graph.mlx_metal_benchmark"
    assert report["format_version"] == 1
    assert report["claim_scope"]["test_split_used"] is False
    assert report["claim_scope"]["weights_updated"] is False
    assert report["claim_scope"]["default_runtime_changed"] is False
    assert report["output_validation"]["gate_passed"] is True
    assert report["source_lazy_instrumentation"]["unchanged"] is True
    assert (
        report["source_lazy_instrumentation"]["after"][
            "sidecar_file_bytes_read"
        ]
        == 0
    )
    assert (
        report["runtime"]["metal_kernel_source_sha256"]
        == mlx_metal_kernel_source_sha256()
    )

    runtime_path = ARTIFACTS / "fused_modal_runtime.pt"
    assert report["source_artifact"]["sha256"] == hashlib.sha256(
        runtime_path.read_bytes()
    ).hexdigest()
    assert [batch["batch_size"] for batch in report["batches"]] == [
        1,
        8,
        64,
        256,
    ]
    for batch in report["batches"]:
        timings = batch["timings"]
        for timing in timings.values():
            assert timing["median_microseconds"] > 0
            assert len(timing["raw_microseconds"]) == 9
        expected = (
            timings["mlx_dense_compiled"]["median_microseconds"]
            / timings["mlx_packed_metal"]["median_microseconds"]
        )
        assert (
            batch["speedup_ratios"][
                "packed_metal_vs_dense_compiled"
            ]
            == pytest.approx(expected)
        )
