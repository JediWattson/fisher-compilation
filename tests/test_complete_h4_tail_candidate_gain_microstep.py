from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_microstep import (
    SYMMETRIC_GAIN_MICROSTEP_EPSILON,
    CandidateConditionedK64SymmetricMicrostepExample,
    select_candidate_conditioned_k64_symmetric_microstep,
    symmetric_microstep_gains,
)
from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    CandidateConditionedK64GainGradientExampleV4,
    fit_candidate_conditioned_k64_mean_kl_gains,
)


def _refit(*, no_op: bool = False):
    examples = tuple(
        CandidateConditionedK64GainGradientExampleV4(
            example_id=f"fit-{index}",
            family_id=f"family-{index}",
            token_gain_gradients=(
                torch.zeros(2, 64, dtype=torch.float64)
                if no_op
                else torch.eye(64, dtype=torch.float64)[index : index + 2]
            ),
            token_teacher_kl=torch.ones(2, dtype=torch.float64),
        )
        for index in range(7)
    )
    return fit_candidate_conditioned_k64_mean_kl_gains(
        examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.ones(64, dtype=torch.float64),
    )


def _microstep_examples(
    refit,
    *,
    unit_by_family: tuple[float, ...] | None = None,
    plus_by_family: tuple[float, ...] | None = None,
    minus_by_family: tuple[float, ...] | None = None,
    token_counts: tuple[int, ...] | None = None,
    prefix: str = "tune",
):
    units = unit_by_family or (1.0,) * 7
    plus = plus_by_family or (0.99,) * 7
    minus = minus_by_family or (1.01,) * 7
    counts = token_counts or (3,) * 7
    return tuple(
        CandidateConditionedK64SymmetricMicrostepExample(
            held_family_id=refit.held_family_id,
            example_id=f"{prefix}-{index}",
            family_id=f"family-{index}",
            v4_refit_artifact_sha256=refit.artifact_sha256,
            pinned_v4_tune_example_artifact_sha256=str(index + 1) * 64,
            pinned_v4_unit_mean_teacher_kl=units[index],
            pinned_v4_unit_token_teacher_kl_sha256=str(index + 2) * 64,
            pinned_v4_unit_receipt_sha256=str(index + 3) * 64,
            structural_no_op_replayed_pinned_v4_unit_exactly=(
                True if refit.mean_no_op else None
            ),
            plus_token_teacher_kl=torch.full(
                (counts[index],), plus[index], dtype=torch.float64
            ),
            minus_token_teacher_kl=torch.full(
                (counts[index],), minus[index], dtype=torch.float64
            ),
        )
        for index in range(7)
    )


def test_epsilon_and_symmetric_gains_are_exact_and_defensive() -> None:
    refit = _refit()
    assert SYMMETRIC_GAIN_MICROSTEP_EPSILON == 1.0 / 64.0
    assert SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex() == "0x1.0000000000000p-6"
    proposal = refit.mean_proposed_gains_tensor()
    plus = symmetric_microstep_gains(refit, 1)
    minus = symmetric_microstep_gains(refit, -1)
    assert torch.equal(
        plus,
        1.0 + (proposal - 1.0) / 64.0,
    )
    assert torch.equal(
        minus,
        1.0 - (proposal - 1.0) / 64.0,
    )
    plus[0] = 999.0
    assert symmetric_microstep_gains(refit, 1)[0] != 999.0
    with pytest.raises(ValueError, match=r"exactly -1 or \+1"):
        symmetric_microstep_gains(refit, 0)


def test_example_binds_v4_unit_and_copies_new_observations() -> None:
    refit = _refit()
    plus = torch.tensor([0.8, 1.0], dtype=torch.float32)
    minus = torch.tensor([1.2, 1.0], dtype=torch.float32)
    example = CandidateConditionedK64SymmetricMicrostepExample(
        held_family_id="held",
        example_id="tune-0",
        family_id="family-0",
        v4_refit_artifact_sha256=refit.artifact_sha256,
        pinned_v4_tune_example_artifact_sha256="c" * 64,
        pinned_v4_unit_mean_teacher_kl=1.0,
        pinned_v4_unit_token_teacher_kl_sha256="d" * 64,
        pinned_v4_unit_receipt_sha256="e" * 64,
        structural_no_op_replayed_pinned_v4_unit_exactly=None,
        plus_token_teacher_kl=plus,
        minus_token_teacher_kl=minus,
    )
    plus[0] = 9.0
    minus[0] = 9.0
    assert example.plus_token_kl_tensor().tolist() == pytest.approx([0.8, 1.0])
    assert example.minus_token_kl_tensor().tolist() == pytest.approx([1.2, 1.0])
    metadata = example.metadata()
    assert metadata["minus_arm_role"] == "diagnostic_only"
    assert metadata["minus_arm_participates_in_central_slope_sign_estimate"] is True
    assert metadata["minus_arm_is_selectable"] is False
    assert metadata["minus_arm_independently_authorizes_selection"] is False
    assert metadata["pinned_v4_unit_token_teacher_kl_sha256"] == "d" * 64
    assert metadata["pinned_v4_unit_receipt_sha256"] == "e" * 64
    assert metadata["microstep_epsilon_hex"] == "0x1.0000000000000p-6"
    assert metadata["raw_tensors_serialized"] is False
    assert "plus_token_teacher_kl" not in metadata
    assert "minus_token_teacher_kl" not in metadata


def test_selects_only_plus_epsilon_when_every_guard_passes() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit, _microstep_examples(refit)
    )
    assert selection.central_slope < 0.0
    assert selection.plus_improvement >= selection.minimum_required_plus_improvement
    assert selection.plus_family_nonworse_count == 7
    assert selection.plus_family_cap_passed is True
    assert selection.metadata()[
        "central_slope_sign_agrees_with_v4_prediction"
    ] is True
    assert selection.selected_step == SYMMETRIC_GAIN_MICROSTEP_EPSILON
    assert selection.selected_arm == "plus_epsilon"
    assert selection.selection_reason == (
        "plus_epsilon_cleared_symmetric_microstep_guards"
    )
    assert torch.equal(selection.selected_gains_tensor(), selection.plus_gains_tensor())
    metadata = selection.metadata()
    assert metadata["minus_arm_participates_in_central_slope_sign_guard"] is True
    assert metadata["minus_arm_is_selectable"] is False
    assert metadata["minus_arm_independently_authorizes_selection"] is False


def test_nonnegative_central_slope_forces_unit_despite_plus_improvement() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(0.99,) * 7,
            minus_by_family=(0.98,) * 7,
        ),
    )
    assert selection.plus_improvement > 0.0
    assert selection.central_slope > 0.0
    assert selection.selected_arm == "unit"
    assert selection.selection_reason == (
        "unit_nonnegative_family_equal_central_slope"
    )


def test_macro_improvement_threshold_is_exact_and_not_a_grid() -> None:
    refit = _refit()
    below = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(0.99995,) * 7,
            minus_by_family=(1.001,) * 7,
        ),
    )
    assert below.minimum_required_plus_improvement == pytest.approx(1.0e-4)
    assert below.plus_improvement == pytest.approx(5.0e-5)
    assert below.selected_arm == "unit"
    assert below.selection_reason == "unit_plus_macro_improvement_below_threshold"
    metadata = below.metadata()
    assert metadata["v5_executed_tune_steps_hex"] == (
        "-0x1.0000000000000p-6",
        "0x1.0000000000000p-6",
    )
    assert metadata["selection_candidate_steps_hex"] == (
        "0x0.0p+0",
        "0x1.0000000000000p-6",
    )
    assert metadata["posthoc_step_grid_searched"] is False
    assert metadata["zero_step_source"] == "pinned_v4_not_reexecuted"


def test_family_equal_aggregation_does_not_token_weight_long_family() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            minus_by_family=(1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            token_counts=(1000, 1, 1, 1, 1, 1, 1),
        ),
    )
    assert selection.aggregate_plus_mean_teacher_kl == pytest.approx(6.5 / 7.0)
    assert selection.aggregate_minus_mean_teacher_kl == pytest.approx(7.5 / 7.0)
    assert selection.central_slope == pytest.approx(-32.0 / 7.0)


def test_fewer_than_four_nonworse_families_forces_unit() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(0.6, 0.6, 0.6, 1.01, 1.01, 1.01, 1.01),
            minus_by_family=(1.1,) * 7,
        ),
    )
    assert selection.aggregate_plus_mean_teacher_kl < 1.0
    assert selection.central_slope < 0.0
    assert selection.plus_family_nonworse_count == 3
    assert selection.plus_family_cap_passed is True
    assert selection.selected_arm == "unit"
    assert selection.selection_reason == (
        "unit_fewer_than_four_of_seven_plus_families_nonworse"
    )


def test_any_family_above_five_percent_cap_forces_unit() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.06),
            minus_by_family=(1.1,) * 7,
        ),
    )
    assert selection.plus_family_nonworse_count == 6
    assert selection.plus_family_cap_passed is False
    assert selection.selected_arm == "unit"
    assert selection.selection_reason == "unit_plus_family_cap_failed"


def test_no_op_refit_has_distinct_unit_reason_and_cannot_fake_safety() -> None:
    refit = _refit(no_op=True)
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit,
        _microstep_examples(
            refit,
            plus_by_family=(1.0,) * 7,
            minus_by_family=(1.0,) * 7,
        ),
    )
    assert selection.refit_no_op_or_zero_delta is True
    assert selection.central_slope == 0.0
    assert selection.forward_slope == 0.0
    assert selection.backward_slope == 0.0
    assert selection.central_curvature == 0.0
    assert selection.metadata()["curve_diagnostics_defined"] is False
    assert selection.plus_is_eligible is False
    assert selection.selected_arm == "unit"
    assert selection.selection_reason == "unit_mean_refit_no_op_or_zero_delta"
    assert torch.equal(selection.plus_gains_tensor(), torch.ones(64))
    assert torch.equal(selection.minus_gains_tensor(), torch.ones(64))
    assert torch.equal(selection.selected_gains_tensor(), torch.ones(64))

    unauthenticated = tuple(
        replace(
            value,
            structural_no_op_replayed_pinned_v4_unit_exactly=None,
        )
        for value in _microstep_examples(
            refit,
            plus_by_family=(1.0,) * 7,
            minus_by_family=(1.0,) * 7,
        )
    )
    with pytest.raises(ValueError, match="authenticate exact v4 unit replay"):
        select_candidate_conditioned_k64_symmetric_microstep(
            refit, unauthenticated
        )


def test_tune_examples_must_be_disjoint_and_match_exact_v4_fold() -> None:
    refit = _refit()
    values = list(_microstep_examples(refit))
    values[0] = replace(values[0], example_id="fit-0")
    with pytest.raises(ValueError, match="disjoint and match"):
        select_candidate_conditioned_k64_symmetric_microstep(refit, values)

    with pytest.raises(TypeError, match="typed records"):
        select_candidate_conditioned_k64_symmetric_microstep(
            refit, [object()]  # type: ignore[list-item]
        )

    values = list(_microstep_examples(refit))
    values[0] = replace(values[0], held_family_id="other-held")
    with pytest.raises(ValueError, match="disjoint and match"):
        select_candidate_conditioned_k64_symmetric_microstep(refit, values)

    values = list(_microstep_examples(refit))
    values[0] = replace(values[0], v4_refit_artifact_sha256="e" * 64)
    with pytest.raises(ValueError, match="disjoint and match"):
        select_candidate_conditioned_k64_symmetric_microstep(refit, values)

    values = list(_microstep_examples(refit))
    values[0] = replace(
        values[0],
        pinned_v4_unit_receipt_sha256=values[1].pinned_v4_unit_receipt_sha256,
    )
    with pytest.raises(ValueError, match="disjoint and match"):
        select_candidate_conditioned_k64_symmetric_microstep(refit, values)


def test_evidence_rejects_bad_hashes_negative_losses_and_shape_drift() -> None:
    refit = _refit()
    kwargs = dict(
        held_family_id="held",
        example_id="tune-0",
        family_id="family-0",
        v4_refit_artifact_sha256=refit.artifact_sha256,
        pinned_v4_tune_example_artifact_sha256="c" * 64,
        pinned_v4_unit_mean_teacher_kl=1.0,
        pinned_v4_unit_token_teacher_kl_sha256="d" * 64,
        pinned_v4_unit_receipt_sha256="e" * 64,
        structural_no_op_replayed_pinned_v4_unit_exactly=None,
        plus_token_teacher_kl=torch.ones(2),
        minus_token_teacher_kl=torch.ones(2),
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CandidateConditionedK64SymmetricMicrostepExample(
            **{**kwargs, "pinned_v4_unit_token_teacher_kl_sha256": "bad"}
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CandidateConditionedK64SymmetricMicrostepExample(
            **{**kwargs, "pinned_v4_unit_receipt_sha256": "bad"}
        )
    with pytest.raises(ValueError, match="geometry differs"):
        CandidateConditionedK64SymmetricMicrostepExample(
            **{
                **kwargs,
                "plus_token_teacher_kl": torch.tensor([-1.0, 1.0]),
            }
        )
    with pytest.raises(ValueError, match="geometry differs"):
        CandidateConditionedK64SymmetricMicrostepExample(
            **{**kwargs, "minus_token_teacher_kl": torch.ones(3)}
        )


def test_tensor_mutation_is_detected_and_accessors_return_copies() -> None:
    refit = _refit()
    examples = _microstep_examples(refit)
    example = examples[0]
    example.plus_token_teacher_kl[0] = 123.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        example.validate_integrity()

    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit, _microstep_examples(refit)
    )
    selected = selection.selected_gains_tensor()
    selected[0] = 123.0
    assert selection.selected_gains_tensor()[0] != 123.0
    selection.plus_family_mean_teacher_kl[0] = 123.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        selection.validate_integrity()


def test_selection_metadata_is_hash_and_scalar_only_for_numeric_ledgers() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_symmetric_microstep(
        refit, _microstep_examples(refit)
    )
    metadata = selection.metadata()
    assert not any(isinstance(value, torch.Tensor) for value in metadata.values())
    assert "unit_family_mean_teacher_kl" not in metadata
    assert "plus_family_mean_teacher_kl" not in metadata
    assert "minus_family_mean_teacher_kl" not in metadata
    assert metadata["minus_arm_diagnostic_only"] is True
    assert metadata["raw_tensors_serialized"] is False
