from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisher_graph.hierarchy_speed_benchmark import (
    HIERARCHY_SPEED_SYSTEMS,
    benchmark_prepared_torch_decomposition,
    build_synthetic_connectivity_decomposition,
    hierarchy_speed_accounting,
    make_synchronized_mlx_chain_operation,
    render_hierarchy_speed_benchmark_markdown,
    run_hierarchy_speed_benchmark,
)


ARTIFACTS = (
    Path(__file__).resolve().parents[1] / "artifacts" / "hierarchy_speed"
)


def test_mlx_chain_feeds_each_output_forward_and_synchronizes_once() -> None:
    events: list[object] = []

    class FakeCore:
        @staticmethod
        def eval(value: object) -> None:
            events.append(("eval", value))

        @staticmethod
        def synchronize() -> None:
            events.append("synchronize")

    def forward(inputs: tuple[int, ...]) -> tuple[int, ...]:
        events.append(("forward", inputs))
        return tuple(value + 1 for value in inputs)

    operation = make_synchronized_mlx_chain_operation(
        FakeCore,
        forward,
        (1, 10),
        stages_per_call=3,
    )

    assert operation() == (4, 13)
    assert events == [
        ("forward", (1, 10)),
        ("forward", (2, 11)),
        ("forward", (3, 12)),
        ("eval", (4, 13)),
        "synchronize",
    ]


def test_synthetic_shape_has_known_rank_energy_and_break_even() -> None:
    full = build_synthetic_connectivity_decomposition(
        input_width=8,
        output_width=8,
    )
    rank_two = full.truncate(2)
    accounting = hierarchy_speed_accounting(rank_two)

    assert full.factors[0].retained_rank == 8
    assert rank_two.factors[0].retained_rank == 2
    assert accounting["source_dense_macs_per_row"] == 64
    assert accounting["candidate_factorized_macs_per_row"] == 32
    assert accounting["ideal_arithmetic_speedup"] == 2.0
    assert accounting["candidate_has_fewer_macs"] is True
    assert accounting["source_dense_stored_scalars"] == 72
    assert accounting["direct_candidate_stored_scalars"] == 48
    assert accounting["prepared_candidate_stored_scalars"] == 40
    assert accounting["prepared_candidate_has_fewer_scalars"] is True
    assert (
        0.0
        < accounting["retained_weighted_energy_fraction"]
        < 1.0
    )


def test_square_rank_half_is_arithmetic_tie_but_not_storage_tie() -> None:
    decomposition = build_synthetic_connectivity_decomposition(
        input_width=8,
        output_width=8,
        retained_rank=4,
    )
    accounting = hierarchy_speed_accounting(decomposition)

    assert accounting["source_dense_macs_per_row"] == 64
    assert accounting["candidate_factorized_macs_per_row"] == 64
    assert accounting["ideal_arithmetic_speedup"] == 1.0
    assert accounting["candidate_has_fewer_macs"] is False
    # The folded candidate stores R, P, and an output bias: 64 + 8 scalars.
    # The source dense control also stores a 64-scalar matrix and 8-scalar bias.
    assert accounting["prepared_candidate_stored_scalars"] == 72
    assert accounting["prepared_candidate_has_fewer_scalars"] is False


def test_torch_probe_compares_all_three_prepared_systems() -> None:
    decomposition = build_synthetic_connectivity_decomposition(
        input_width=4,
        output_width=4,
        retained_rank=1,
    )
    report = benchmark_prepared_torch_decomposition(
        decomposition,
        row_counts=(1, 3),
        repeats=2,
        minimum_block_seconds=1e-5,
        warmup_iterations=1,
        minimum_warmup_seconds=0.0,
    )

    assert report["backend"] == "torch_cpu"
    assert report["output_validation"]["gate_passed"] is True
    assert report["runtime"]["accounting"]["analysis_only"] is True
    assert [value["row_count"] for value in report["rows"]] == [1, 3]
    for row in report["rows"]:
        assert set(row["timings"]) == set(HIERARCHY_SPEED_SYSTEMS)
        assert set(row["speedup_ratios"]) == {
            "candidate_dense_vs_source_dense",
            "candidate_factorized_vs_source_dense",
            "candidate_factorized_vs_candidate_dense",
        }
        assert all(
            value > 0 for value in row["speedup_ratios"].values()
        )


def test_portable_report_and_markdown_deny_model_claims() -> None:
    report = run_hierarchy_speed_benchmark(
        input_width=4,
        output_width=4,
        retained_ranks=(1,),
        row_counts=(1,),
        backend="torch",
        torch_repeats=1,
        torch_minimum_block_seconds=1e-5,
        torch_warmup_iterations=0,
        torch_minimum_warmup_seconds=0.0,
    )
    serialized = json.dumps(report, allow_nan=False)
    markdown = render_hierarchy_speed_benchmark_markdown(report)

    assert report["schema"] == "fisher_graph.hierarchy_speed_benchmark"
    assert report["claim_scope"]["shape_only"] is True
    assert report["claim_scope"]["task_validation_included"] is False
    assert report["claim_scope"]["replacement_authorized"] is False
    assert '"retained_rank": 1' in serialized
    assert "not a Gemma quality" in markdown
    assert "Factorized vs source" in markdown


def test_committed_width640_report_is_self_consistent() -> None:
    report_path = ARTIFACTS / "width640_prepared_benchmark.json"
    report = json.loads(report_path.read_text())

    assert (
        (ARTIFACTS / "width640_prepared_benchmark.md").read_text()
        == render_hierarchy_speed_benchmark_markdown(report)
    )
    assert report["schema"] == "fisher_graph.hierarchy_speed_benchmark"
    assert report["format_version"] == 1
    assert report["shape"] == {
        "input_width": 640,
        "output_width": 640,
        "retained_ranks": [80, 160, 256, 320],
        "row_counts": [1, 8, 128, 512, 2048],
        "spectrum": "geometric_1_to_0.01",
    }
    assert report["claim_scope"]["shape_only"] is True
    assert report["claim_scope"]["task_validation_included"] is False
    assert report["claim_scope"]["model_level_latency_measured"] is False
    assert report["claim_scope"]["replacement_authorized"] is False
    assert report["claim_scope"]["deployed_storage_reduction_claimed"] is False
    assert [case["retained_rank"] for case in report["cases"]] == [
        80,
        160,
        256,
        320,
    ]
    for case in report["cases"]:
        assert set(case["backends"]) == {
            "torch",
            "mlx_sync_1",
            "mlx_sync_18",
        }
        for backend in case["backends"].values():
            assert backend["output_validation"]["gate_passed"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"input_width": 0, "output_width": 2},
            "input_width",
        ),
        (
            {
                "input_width": 2,
                "output_width": 2,
                "retained_rank": 3,
            },
            "retained rank",
        ),
        (
            {
                "input_width": 2,
                "output_width": 2,
                "spectrum_floor": 1.0,
            },
            "spectrum_floor",
        ),
    ],
)
def test_synthetic_builder_rejects_invalid_shapes(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_synthetic_connectivity_decomposition(**kwargs)
