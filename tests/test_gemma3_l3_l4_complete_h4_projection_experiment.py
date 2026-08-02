from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_projection_experiment as runner,
)


def _geometry(*, passed: bool = True) -> dict[str, object]:
    good = {"normalized_rmse": 0.04, "cosine": 0.996}
    bad = {"normalized_rmse": 0.051, "cosine": 0.996}
    pooled = {
        "full": dict(good),
        "graph_core": dict(good),
        "causal_tail": dict(good),
    }
    if not passed:
        pooled["causal_tail"] = bad
    result: dict[str, object] = {
        "pooled": pooled,
        "families": (
            {
                "family_id": "family-a",
                "strata": {
                    "full": {"normalized_rmse": 0.09, "cosine": 0.991},
                    "graph_core": {
                        "normalized_rmse": 0.09,
                        "cosine": 0.991,
                    },
                },
            },
            {
                "family_id": "family-b",
                "strata": {
                    "full": {"normalized_rmse": 0.08, "cosine": 0.992},
                    "graph_core": {
                        "normalized_rmse": 0.08,
                        "cosine": 0.992,
                    },
                    "causal_tail": {
                        "normalized_rmse": 0.10,
                        "cosine": 0.99,
                    },
                },
            },
        ),
    }
    result["gates"] = runner._boundary_geometry_gates(result)
    return result


def _behavior(*, passed: bool) -> dict[str, object]:
    return {"gates": {"passed": passed}}


def test_projection_moments_report_normalized_error_and_cosine() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    candidate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    moments = runner._ProjectionMoments()

    moments.add(source, candidate)
    summary = moments.summary()

    assert summary["rows"] == 2
    assert summary["width"] == 2
    assert summary["normalized_rmse"] == pytest.approx(1.0 / 5.0**0.5)
    assert summary["cosine"] == pytest.approx(3.0 / (5.0 * 2.0) ** 0.5)


def test_boundary_geometry_gates_apply_tight_pooled_and_relaxed_family_limits() -> None:
    geometry = _geometry()

    gates = geometry["gates"]

    assert gates["passed"] is True
    assert gates["pooled"]["causal_tail"]["passed"] is True
    assert gates["families"][1]["strata"]["causal_tail"]["passed"] is True

    family_failure = deepcopy(geometry)
    family_failure.pop("gates")
    family_failure["families"][1]["strata"]["causal_tail"][
        "normalized_rmse"
    ] = 0.10001
    assert runner._boundary_geometry_gates(family_failure)["passed"] is False


@pytest.mark.parametrize(
    ("failed_axis", "pattern"),
    (
        (None, "11111"),
        ("identity", "01111"),
        ("support", "10111"),
        ("geometry", "11011"),
        ("ordinary", "11101"),
        ("support_behavior", "11110"),
    ),
)
def test_capacity_classifier_requires_every_preregistered_axis(
    failed_axis: str | None,
    pattern: str,
) -> None:
    geometry = _geometry(passed=failed_axis != "geometry")
    comparison = runner.classify_complete_h4_projection_capacity(
        identity_validated=failed_axis != "identity",
        support_integrity={"passed": failed_axis != "support"},
        boundary_geometry=geometry,
        ordinary_behavioral=_behavior(passed=failed_axis != "ordinary"),
        support_behavioral=_behavior(
            passed=failed_axis != "support_behavior"
        ),
    )

    assert comparison["pass_pattern"] == pattern
    assert comparison["classification"] == (
        "rank64_h4_projection_capacity_validated"
        if failed_axis is None
        else "rank64_h4_projection_insufficient"
    )
    assert comparison["serving_authorized"] is False
    assert comparison["compression_claim"] is False
    assert comparison["speed_or_latency_claim"] is False
    assert (comparison["success_authorizes"] is not None) is (
        failed_axis is None
    )


def test_supervised_grid_indices_add_the_single_batch_coordinate() -> None:
    positions = torch.tensor([1, 3], dtype=torch.int64)

    observed = runner._supervised_grid_indices(positions)

    assert torch.equal(
        observed,
        torch.tensor([[0, 1], [0, 3]], dtype=torch.int64),
    )
    assert observed.is_contiguous()
    with pytest.raises(ValueError, match="nonempty int64"):
        runner._supervised_grid_indices(torch.tensor([1.0]))


def test_output_is_confined_to_ignored_local_json() -> None:
    assert runner._validate_output(
        ".local-runs/model/projection.json"
    ) == runner.Path(".local-runs/model/projection.json")
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output("projection.json")
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(".local-runs/model/projection.pt")


def test_cli_defaults_to_locked_rank64_capacity_screen() -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.max_length == runner.DEFAULT_MAX_LENGTH
    assert arguments.rank64_x4_baseline == runner.DEFAULT_RANK64_X4_BASELINE
    assert arguments.complete_h4_identity == runner.DEFAULT_COMPLETE_H4_IDENTITY
    assert arguments.output == runner.DEFAULT_OUTPUT
