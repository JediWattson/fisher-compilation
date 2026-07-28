from __future__ import annotations

import json

import pytest
import torch

from fisher_graph.state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
    ContrastDefinition,
    ContrastObservation,
    assess_state_conditioned_contrasts,
    score_state_conditioned_contrast,
)


def _gates(**overrides: object) -> ContrastAssessmentGates:
    values: dict[str, object] = {
        "minimum_family_eligible_count": 1,
        "minimum_family_eligible_fraction": 1.0,
    }
    values.update(overrides)
    return ContrastAssessmentGates(**values)  # type: ignore[arg-type]


def _observation(
    *,
    contrast_id: str = "contrast-0",
    family: str = "family-a",
    role: str = "expected_sensitivity",
    teacher_left: tuple[float, ...] = (1.0, 0.0),
    teacher_right: tuple[float, ...] = (0.0, 0.0),
    candidate_left: tuple[float, ...] | None = None,
    candidate_right: tuple[float, ...] | None = None,
    repeated_teacher_left: tuple[float, ...] | None = None,
    repeated_teacher_right: tuple[float, ...] | None = None,
    repeated_candidate_left: tuple[float, ...] | None = None,
    repeated_candidate_right: tuple[float, ...] | None = None,
) -> ContrastObservation:
    candidate_left = candidate_left or teacher_left
    candidate_right = candidate_right or teacher_right
    repeated_teacher_left = repeated_teacher_left or teacher_left
    repeated_teacher_right = repeated_teacher_right or teacher_right
    repeated_candidate_left = repeated_candidate_left or candidate_left
    repeated_candidate_right = repeated_candidate_right or candidate_right

    def tensor(value: tuple[float, ...]) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)

    return ContrastObservation(
        definition=ContrastDefinition(
            contrast_id=contrast_id,
            family=family,
            role=role,  # type: ignore[arg-type]
            coefficients=(1.0, -1.0),
        ),
        teacher_endpoints=(tensor(teacher_left), tensor(teacher_right)),
        repeated_teacher_endpoints=(
            tensor(repeated_teacher_left),
            tensor(repeated_teacher_right),
        ),
        candidate_endpoints=(tensor(candidate_left), tensor(candidate_right)),
        repeated_candidate_endpoints=(
            tensor(repeated_candidate_left),
            tensor(repeated_candidate_right),
        ),
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        (1.0, 0.0),
        (0.5, -0.5),
        (1.0,),
    ],
)
def test_contrast_definition_requires_zero_sum_l1_two(
    coefficients: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        ContrastDefinition(
            contrast_id="bad",
            family="family",
            role="expected_sensitivity",
            coefficients=coefficients,
        )


def test_gate_and_definition_state_round_trip_with_hashes() -> None:
    gates = ContrastAssessmentGates()
    definition = ContrastDefinition(
        contrast_id="radial-0",
        family="radial",
        role="expected_sensitivity",
        coefficients=(1.0, -1.0),
    )

    assert ContrastAssessmentGates.from_state_dict(gates.state_dict()) == gates
    assert ContrastDefinition.from_state_dict(definition.state_dict()) == definition
    assert len(gates.artifact_sha256) == 64
    assert len(definition.artifact_sha256) == 64


def test_exact_sensitivity_recovery_passes_all_vector_metrics() -> None:
    score = score_state_conditioned_contrast(
        _observation(),
        gates=_gates(),
    )

    assert score.teacher_status == "eligible_sensitivity"
    assert score.candidate_scored
    assert score.decision_status == "pass"
    assert score.candidate_contrast_relative_error == pytest.approx(0.0)
    assert score.candidate_direction_cosine == pytest.approx(1.0)
    assert score.candidate_projection_gain == pytest.approx(1.0)
    assert score.candidate_orthogonal_leakage == pytest.approx(0.0)
    assert score.candidate_magnitude_ratio == pytest.approx(1.0)
    assert all(passed for _, passed in score.candidate_gate_flags)


def test_weak_sensitivity_is_inconclusive_before_candidate_scoring() -> None:
    score = score_state_conditioned_contrast(
        _observation(
            teacher_left=(1.0, 0.0),
            teacher_right=(0.9999, 0.0),
            candidate_left=(100.0, -100.0),
            candidate_right=(-100.0, 100.0),
        ),
        gates=_gates(),
    )

    assert score.teacher_status == "underpowered_sensitivity"
    assert score.decision_status == "panel_inconclusive"
    assert not score.candidate_scored
    assert score.candidate_contrast_relative_error is None
    assert score.candidate_direction_cosine is None
    assert score.candidate_projection_gain is None
    assert score.candidate_gate_flags == ()


@pytest.mark.parametrize(
    ("candidate_left", "candidate_right", "failed_gate"),
    [
        ((0.0, 0.0), (0.0, 0.0), "direction_cosine"),
        ((-1.0, 0.0), (0.0, 0.0), "direction_cosine"),
        ((0.5, 0.0), (0.0, 0.0), "projection_gain"),
        ((0.0, 1.0), (0.0, 0.0), "orthogonal_leakage"),
    ],
)
def test_sensitivity_rejects_zero_reversed_wrong_gain_and_orthogonal_contrasts(
    candidate_left: tuple[float, ...],
    candidate_right: tuple[float, ...],
    failed_gate: str,
) -> None:
    score = score_state_conditioned_contrast(
        _observation(
            candidate_left=candidate_left,
            candidate_right=candidate_right,
        ),
        gates=_gates(),
    )

    assert score.decision_status == "candidate_fail"
    flags = dict(score.candidate_gate_flags)
    assert not flags[failed_gate]


def test_intended_null_uses_absolute_metrics_without_direction_metrics() -> None:
    score = score_state_conditioned_contrast(
        _observation(
            role="intended_null",
            teacher_left=(1.0, 0.0),
            teacher_right=(1.0, 0.0),
            candidate_left=(1.0, 0.0),
            candidate_right=(1.0, 0.0),
        ),
        gates=_gates(),
    )

    assert score.teacher_status == "valid_intended_null"
    assert score.decision_status == "pass"
    assert score.candidate_null_relative_effect_upper is not None
    assert score.candidate_null_relative_error_upper is not None
    assert score.candidate_direction_cosine is None
    assert score.candidate_projection_gain is None
    assert score.candidate_orthogonal_leakage is None
    assert set(dict(score.candidate_gate_flags)) == {
        "null_relative_effect",
        "null_relative_error",
    }


def test_intended_null_rejects_candidate_hallucinated_contrast() -> None:
    score = score_state_conditioned_contrast(
        _observation(
            role="intended_null",
            teacher_left=(1.0, 0.0),
            teacher_right=(1.0, 0.0),
            candidate_left=(1.2, 0.0),
            candidate_right=(1.0, 0.0),
        ),
        gates=_gates(),
    )

    assert score.teacher_status == "valid_intended_null"
    assert score.decision_status == "candidate_fail"
    assert not all(dict(score.candidate_gate_flags).values())


def test_teacher_null_failure_precedes_panel_inconclusive() -> None:
    null_failure = _observation(
        contrast_id="null-failure",
        family="null-family",
        role="intended_null",
        teacher_left=(1.0, 0.0),
        teacher_right=(0.9, 0.0),
    )
    weak_sensitivity = _observation(
        contrast_id="weak",
        family="sensitivity-family",
        teacher_left=(1.0, 0.0),
        teacher_right=(0.9999, 0.0),
    )

    result = assess_state_conditioned_contrasts(
        (weak_sensitivity, null_failure),
        gates=_gates(),
    )

    assert result.teacher_null_failure_family_count == 1
    assert result.panel_inconclusive_family_count == 1
    assert result.overall_status == "teacher_null_failure"


def test_zero_teacher_baseline_is_invalid_and_has_highest_priority() -> None:
    invalid = _observation(
        contrast_id="invalid",
        family="invalid-family",
        teacher_left=(0.0, 0.0),
        teacher_right=(0.0, 0.0),
        candidate_left=(0.0, 0.0),
        candidate_right=(0.0, 0.0),
    )
    candidate_failure = _observation(
        contrast_id="candidate-failure",
        family="candidate-family",
        candidate_left=(0.0, 0.0),
        candidate_right=(0.0, 0.0),
    )

    result = assess_state_conditioned_contrasts(
        (candidate_failure, invalid),
        gates=_gates(),
    )

    assert result.invalid_family_count == 1
    assert result.candidate_failed_family_count == 1
    assert result.overall_status == "invalid"


def test_family_requires_preregistered_eligible_count() -> None:
    observations = tuple(
        _observation(
            contrast_id=f"eligible-{index}",
            family="sensitivity-family",
        )
        for index in range(3)
    )
    result = assess_state_conditioned_contrasts(
        observations,
        gates=ContrastAssessmentGates(
            minimum_family_eligible_count=4,
            minimum_family_eligible_fraction=0.75,
        ),
    )

    family = result.family_scores[0]
    assert family.eligible_contrast_count == 3
    assert family.required_eligible_count == 4
    assert family.decision_status == "panel_inconclusive"
    assert result.overall_status == "panel_inconclusive"


def test_family_macro_error_fails_even_when_each_contrast_gate_passes() -> None:
    observations = tuple(
        _observation(
            contrast_id=f"gain-070-{index}",
            family="sensitivity-family",
            candidate_left=(0.7, 0.0),
            candidate_right=(0.0, 0.0),
        )
        for index in range(4)
    )
    result = assess_state_conditioned_contrasts(
        observations,
        gates=ContrastAssessmentGates(),
    )

    assert all(
        score.decision_status == "pass" for score in result.contrast_scores
    )
    family = result.family_scores[0]
    assert family.macro_rms_contrast_relative_error == pytest.approx(0.3)
    assert family.decision_status == "candidate_fail"
    assert result.overall_status == "candidate_fail"


def test_source_safe_state_is_json_serializable_and_contains_no_tensors() -> None:
    result = assess_state_conditioned_contrasts(
        (_observation(),),
        gates=_gates(),
    )
    state = result.state_dict()

    json.dumps(state, allow_nan=False)

    def assert_no_tensors(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for nested in value.values():
                assert_no_tensors(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                assert_no_tensors(nested)

    assert_no_tensors(state)
    assert not result.weak_teacher_contrasts_entered_candidate_relative_metrics
    assert not result.intended_null_contrasts_entered_direction_metrics
    assert len(result.artifact_sha256) == 64
