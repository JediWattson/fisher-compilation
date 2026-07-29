from __future__ import annotations

import copy
import math

import pytest
import torch

from fisher_graph.contrast_objective_balancing import (
    UnitRmsFisherGauge,
    audit_objective_contributions,
)
from fisher_graph.state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastTrainingMetrics,
)


def _objective() -> ContrastAwareObjective:
    return ContrastAwareObjective(
        pointwise_weight=1.0,
        sensitivity_relative_delta_weight=2.0,
        sensitivity_direction_weight=0.5,
        midpoint_jvp_weight=1.0,
        intended_null_weight=1.0,
    )


def _metrics(*, weighted_total: float = 7.75) -> ContrastTrainingMetrics:
    return ContrastTrainingMetrics(
        pointwise_mse=4.0,
        sensitivity_relative_delta_mse=0.25,
        sensitivity_direction_loss=0.5,
        midpoint_jvp_relative_mse=1.0,
        intended_null_absolute_mse=2.0,
        weighted_total=weighted_total,
        endpoint_count=4,
        sensitivity_pair_count=2,
        jvp_pair_count=1,
        intended_null_pair_count=1,
    )


def test_unit_rms_gauge_preserves_relative_weights_and_removes_global_scale(
) -> None:
    raw = torch.tensor([2.0, 3.0, 7.0, 11.0], dtype=torch.float64)
    gauge = UnitRmsFisherGauge.from_metric_weight(raw)
    scaled = UnitRmsFisherGauge.from_metric_weight(19.0 * raw)

    assert torch.sqrt(gauge.metric_weight.square().mean()).item() == (
        pytest.approx(1.0, rel=1e-14, abs=1e-14)
    )
    assert torch.allclose(
        gauge.metric_weight,
        scaled.metric_weight,
        rtol=1e-15,
        atol=1e-15,
    )
    assert gauge.raw_rms == pytest.approx(
        float(torch.sqrt(raw.square().mean()))
    )
    assert scaled.raw_rms == pytest.approx(19.0 * gauge.raw_rms)
    assert gauge.raw_mean_square == pytest.approx(gauge.raw_rms**2)
    assert torch.equal(
        gauge.metric_weight / gauge.metric_weight[0],
        raw / raw[0],
    )
    assert gauge.artifact_sha256 != scaled.artifact_sha256
    gauge.validate_source(raw)
    scaled.validate_source(19.0 * raw)


def test_unit_rms_gauge_state_round_trips_and_rejects_tampering() -> None:
    raw = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    gauge = UnitRmsFisherGauge.from_metric_weight(raw)
    state = gauge.state_dict()
    restored = UnitRmsFisherGauge.from_state_dict(state)

    assert restored.artifact_sha256 == gauge.artifact_sha256
    assert restored.raw_metric_weight_sha256 == (
        gauge.raw_metric_weight_sha256
    )
    assert torch.equal(restored.metric_weight, gauge.metric_weight)
    restored.validate_source(raw)

    tensor_tamper = copy.deepcopy(state)
    tensor_tamper["metric_weight"][0] += 0.1
    with pytest.raises(ValueError):
        UnitRmsFisherGauge.from_state_dict(tensor_tamper)

    metadata_tamper = copy.deepcopy(state)
    metadata_tamper["raw_rms"] = float(metadata_tamper["raw_rms"]) + 0.1
    with pytest.raises(ValueError):
        UnitRmsFisherGauge.from_state_dict(metadata_tamper)

    with pytest.raises(ValueError, match="binding differs"):
        gauge.validate_source(raw + 0.01)


@pytest.mark.parametrize(
    "value,exception,match",
    (
        (
            torch.tensor([1, 2]),
            TypeError,
            "floating dtype",
        ),
        (
            torch.ones((1, 2), dtype=torch.float64),
            ValueError,
            "rank-one",
        ),
        (
            torch.tensor([], dtype=torch.float64),
            ValueError,
            "nonempty",
        ),
        (
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            ValueError,
            "strictly positive",
        ),
        (
            torch.tensor([1.0, math.inf], dtype=torch.float64),
            ValueError,
            "finite",
        ),
    ),
)
def test_unit_rms_gauge_rejects_invalid_metrics(
    value: torch.Tensor,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        UnitRmsFisherGauge.from_metric_weight(value)


def test_objective_contribution_audit_reports_exact_budget() -> None:
    objective = _objective()
    metrics = _metrics()
    audit = audit_objective_contributions(metrics, objective)

    assert audit.pointwise == pytest.approx(4.0)
    assert audit.sensitivity_relative_delta == pytest.approx(0.5)
    assert audit.sensitivity_direction == pytest.approx(0.25)
    assert audit.midpoint_jvp == pytest.approx(1.0)
    assert audit.intended_null == pytest.approx(2.0)
    assert audit.total == pytest.approx(7.75)
    assert audit.pointwise_fraction == pytest.approx(4.0 / 7.75)
    assert audit.contrast_fraction == pytest.approx(1.75 / 7.75)
    assert audit.intended_null_fraction == pytest.approx(2.0 / 7.75)
    assert audit.reported_total_matches is True
    assert audit.objective_sha256 == objective.artifact_sha256
    assert audit.metrics_sha256 == metrics.artifact_sha256
    assert audit.state_dict()["artifact_kind"] == (
        "fisher_graph.objective_contribution_audit"
    )


def test_objective_contribution_audit_flags_a_mismatched_reported_total(
) -> None:
    audit = audit_objective_contributions(
        _metrics(weighted_total=7.0),
        _objective(),
    )

    assert audit.total == pytest.approx(7.75)
    assert audit.reported_total == pytest.approx(7.0)
    assert audit.reported_total_matches is False
