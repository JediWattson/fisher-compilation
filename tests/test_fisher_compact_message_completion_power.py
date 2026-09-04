from __future__ import annotations

import hashlib
import inspect
import math

import pytest

from fisher_graph.fisher_compact_message_completion_artifacts import (
    artifact_dependencies,
    build_development_artifact,
)
from fisher_graph.fisher_compact_message_completion_power import (
    GUARD_ANALYSIS_UNIT_COUNT,
    POWER_PROJECTION_ALTERNATIVE,
    POWER_PROJECTION_CONFIDENCE,
    POWER_PROJECTION_METHOD,
    POWER_PROJECTION_REPLICATES,
    POWER_PROJECTION_SEED,
    POWER_RELATIVE_DENOMINATOR_FLOOR,
    PowerProjectionError,
    SupportedChartMembership,
    _hamilton_guard_allocations,
    project_development_power,
)
from fisher_graph.fisher_compact_message_completion_splits import stratum_order_key


def _unit_id(index: int) -> bytes:
    return hashlib.sha256(f"development-unit-{index}".encode()).digest()


def _fixture():
    baseline = (1.00, 1.10, 0.90, 1.20, 0.80, 1.30, 1.05, 0.95)
    effects = (0.10, 0.04, -0.03, 0.08, 0.02, -0.06, 0.07, 0.01)
    j4 = tuple(base - effect for base, effect in zip(baseline, effects, strict=True))
    strata = ("family-a",) * 4 + ("family-b",) * 4
    memberships = tuple(
        (
            SupportedChartMembership(
                chart_id=0 if index < 4 else 1,
                sequence_count=1 + index % 2,
                row_count=2 + index % 3,
            ),
        )
        for index in range(8)
    )
    unit_ids = tuple(_unit_id(index) for index in range(8))
    return baseline, j4, strata, memberships, unit_ids


def test_projection_is_deterministic_finite_and_matches_current_schema() -> None:
    baseline, j4, strata, memberships, unit_ids = _fixture()
    first = project_development_power(
        baseline,
        j4,
        strata,
        memberships,
        unit_ids=unit_ids,
        active_chart_ids=(0, 1, 2),
    )
    second = project_development_power(
        baseline,
        j4,
        strata,
        memberships,
        unit_ids=unit_ids,
        active_chart_ids=(0, 1, 2),
    )
    assert first == second
    assert first.guard_analysis_unit_count_assumed == 12
    assert first.reporting_only
    assert not first.eligibility_affecting
    assert 0.0 <= first.probability_confirming_five_percent_effect <= 1.0
    assert all(
        math.isfinite(value)
        for value in (
            first.expected_paired_interval_width,
            first.probability_confirming_five_percent_effect,
            first.one_adverse_unit_sensitivity,
            first.two_adverse_unit_sensitivity,
        )
    )
    assert [record.chart_id for record in first.expected_chart_support] == [0, 1, 2]
    assert first.expected_chart_support[0].analysis_unit_count == 6
    assert first.expected_chart_support[1].analysis_unit_count == 6
    assert first.expected_chart_support[2].analysis_unit_count == 0
    assert not first.expected_chart_support[2].supported

    name = "development_power_assessment.json"
    dependencies = {
        dependency: "00" * 32 for dependency in artifact_dependencies(name)
    }
    payload = build_development_artifact(
        name,
        campaign_id="synthetic-power-test",
        dependency_hashes=dependencies,
        body=first.as_artifact_body(),
    )
    assert payload["body"] == first.as_artifact_body()
    assert set(first.as_artifact_body()) == {
        "guard_analysis_unit_count_assumed",
        "expected_paired_interval_width",
        "probability_confirming_five_percent_effect",
        "one_adverse_unit_sensitivity",
        "two_adverse_unit_sensitivity",
        "expected_chart_support",
        "reporting_only",
        "eligibility_affecting",
    }


def test_projection_is_input_order_invariant_with_unit_commitments() -> None:
    baseline, j4, strata, memberships, unit_ids = _fixture()
    original = project_development_power(
        baseline,
        j4,
        strata,
        memberships,
        unit_ids=unit_ids,
        active_chart_ids=(0, 1, 2),
    )
    order = (6, 2, 7, 0, 4, 1, 5, 3)
    reordered = project_development_power(
        tuple(baseline[index] for index in order),
        tuple(j4[index] for index in order),
        tuple(strata[index] for index in order),
        tuple(memberships[index] for index in order),
        unit_ids=tuple(unit_ids[index] for index in order),
        active_chart_ids=(2, 0, 1),
    )
    assert reordered == original


def test_constant_within_stratum_effect_has_zero_width_and_full_projection_power() -> None:
    baseline = (2.0,) * 8
    j4 = (1.9,) * 8
    strata = ("a",) * 4 + ("b",) * 4
    memberships = tuple(
        (SupportedChartMembership(0, 1, 2),) for _ in range(8)
    )
    result = project_development_power(
        baseline,
        j4,
        strata,
        memberships,
        unit_ids=tuple(_unit_id(index) for index in range(8)),
    )
    assert result.expected_paired_interval_width == pytest.approx(0.0, abs=1e-15)
    assert result.probability_confirming_five_percent_effect == 1.0
    assert result.two_adverse_unit_sensitivity < result.one_adverse_unit_sensitivity
    assert result.expected_chart_support[0].analysis_unit_count == 12


def test_method_is_frozen_and_api_has_no_eligibility_input() -> None:
    assert GUARD_ANALYSIS_UNIT_COUNT == 12
    assert POWER_PROJECTION_REPLICATES == 10_000
    assert POWER_PROJECTION_SEED == 424_242
    assert POWER_PROJECTION_CONFIDENCE == 0.95
    assert POWER_PROJECTION_ALTERNATIVE == 0.05
    assert POWER_RELATIVE_DENOMINATOR_FLOOR == 1e-12
    assert POWER_PROJECTION_METHOD == "deterministic_stratified_empirical_basic_v1"
    assert "eligibility" not in inspect.signature(project_development_power).parameters


def test_power_hamilton_equal_remainders_use_structured_stratum_digest_order() -> None:
    labels = ("alpha", "beta", "gamma", "delta", "epsilon")
    ordered_allocations = _hamilton_guard_allocations(labels * 2)
    allocations = dict(ordered_allocations)
    digest_order = sorted(labels, key=stratum_order_key)
    assert digest_order[:2] != sorted(labels)[:2]
    assert {label for label in labels if allocations[label] == 3} == set(
        digest_order[:2]
    )
    assert [label for label, _ in ordered_allocations] == digest_order
    assert sum(allocations.values()) == GUARD_ANALYSIS_UNIT_COUNT


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("misaligned", "aligned"),
        ("negative_loss", "nonnegative"),
        ("singleton_stratum", "at least two"),
        ("duplicate_chart", "one record per chart"),
        ("inactive_chart", "inactive chart"),
    ],
)
def test_projection_rejects_invalid_development_inputs(
    mutation: str, message: str
) -> None:
    baseline, j4, strata, memberships, unit_ids = _fixture()
    active = (0, 1)
    if mutation == "misaligned":
        j4 = j4[:-1]
    elif mutation == "negative_loss":
        baseline = (-1.0,) + baseline[1:]
    elif mutation == "singleton_stratum":
        strata = ("singleton",) + strata[1:]
    elif mutation == "duplicate_chart":
        memberships = (
            (
                SupportedChartMembership(0, 1, 1),
                SupportedChartMembership(0, 1, 1),
            ),
        ) + memberships[1:]
    elif mutation == "inactive_chart":
        active = (0,)
    with pytest.raises(PowerProjectionError, match=message):
        project_development_power(
            baseline,
            j4,
            strata,
            memberships,
            unit_ids=unit_ids,
            active_chart_ids=active,
        )


def test_zero_baseline_projection_remains_finite_and_cannot_confirm() -> None:
    result = project_development_power(
        (0.0,) * 4,
        (0.0,) * 4,
        ("one",) * 4,
        tuple((SupportedChartMembership(0, 1, 1),) for _ in range(4)),
        unit_ids=tuple(_unit_id(index) for index in range(4)),
    )
    assert result.expected_paired_interval_width == 0.0
    assert result.probability_confirming_five_percent_effect == 0.0
    assert result.one_adverse_unit_sensitivity == 0.0
    assert result.two_adverse_unit_sensitivity == 0.0
