from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_refit import (
    CANDIDATE_GAIN_ALPHAS,
    CandidateConditionedK64GainGradientExample,
    CandidateConditionedK64GainTuneExample,
    contract_candidate_teacher_kl_gain_scores,
    fit_candidate_conditioned_k64_gains,
    select_candidate_conditioned_k64_gain_alpha,
)


def _gradient_examples(
    *, token_kl: torch.Tensor | None = None
) -> tuple[CandidateConditionedK64GainGradientExample, ...]:
    q = torch.eye(64, dtype=torch.float64)
    losses = (
        torch.ones(64, dtype=torch.float64)
        if token_kl is None
        else token_kl
    )
    return tuple(
        CandidateConditionedK64GainGradientExample(
            example_id=f"e{index}",
            family_id=f"f{index}",
            token_gain_gradients=q,
            token_teacher_kl=losses,
        )
        for index in range(7)
    )


def _refit(*, token_kl: torch.Tensor | None = None):
    return fit_candidate_conditioned_k64_gains(
        _gradient_examples(token_kl=token_kl),
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.ones(64, dtype=torch.float64),
    )


def test_gain_score_contraction_matches_autograd_pullback() -> None:
    torch.manual_seed(4)
    rows = 3
    tokens = 2
    width = 91
    directions = torch.linalg.qr(
        torch.randn(width, 64, dtype=torch.float64), mode="reduced"
    ).Q.T.contiguous()
    tail = torch.randn(rows, width, dtype=torch.float64)
    token_gradients = torch.randn(tokens, rows, width, dtype=torch.float64)
    gains = torch.ones(64, dtype=torch.float64, requires_grad=True)
    amplitudes = tail @ directions.T
    candidate_rows = (amplitudes * gains) @ directions
    losses = torch.einsum("rw,trw->t", candidate_rows, token_gradients)
    expected = torch.stack(
        [
            torch.autograd.grad(losses[index], gains, retain_graph=True)[0]
            for index in range(tokens)
        ]
    )
    actual = contract_candidate_teacher_kl_gain_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=token_gradients,
    )
    assert torch.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_fit_is_family_equal_not_pooled_token_weighted() -> None:
    examples = []
    for index in range(7):
        token_count = index + 1
        gradients = torch.zeros(token_count, 64, dtype=torch.float64)
        losses = torch.ones(token_count, dtype=torch.float64)
        if index == 0:
            gradients[:, 0] = 1.0
            losses[:] = 2.0
        elif index == 1:
            gradients[:, 1] = 1.0
            losses[:] = 4.0
        examples.append(
            CandidateConditionedK64GainGradientExample(
                example_id=f"e{index}",
                family_id=f"f{index}",
                token_gain_gradients=gradients,
                token_teacher_kl=losses,
            )
        )
    refit = fit_candidate_conditioned_k64_gains(
        examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.ones(64),
    )
    expected_c = torch.zeros(64, dtype=torch.float64)
    expected_c[0] = 2.0 / 7.0
    expected_c[1] = 4.0 / 7.0
    expected_gram = torch.zeros(64, 64, dtype=torch.float64)
    expected_gram[0, 0] = 1.0 / 7.0
    expected_gram[1, 1] = 1.0 / 7.0
    assert torch.equal(refit.residual_gradient_c, expected_c)
    assert torch.equal(refit.gradient_gram, expected_gram)


def test_residual_gauss_newton_uses_c_equal_mean_z_times_loss_and_gram() -> None:
    refit = _refit()
    expected_c = torch.full((64,), 1.0 / 64.0, dtype=torch.float64)
    expected_gram = torch.eye(64, dtype=torch.float64) / 64.0
    assert torch.equal(refit.residual_gradient_c, expected_c)
    assert torch.equal(refit.gradient_gram, expected_gram)
    assert refit.damping == pytest.approx(0.1 / 64.0)
    assert refit.predicted_derivative < 0.0
    assert refit.no_op is False
    metadata = refit.metadata()
    assert metadata["method"] == "one_step_damped_residual_Gauss_Newton"
    assert metadata["not_claimed_as"] == "mean_KL_natural_gradient_or_exact_GGN"


def test_rms_trust_clip_limits_applied_gain_step_to_point_25() -> None:
    refit = _refit()
    applied = refit.proposed_gains_tensor() - 1.0
    assert float(applied.square().mean().sqrt()) == pytest.approx(0.25)
    assert refit.trust_scale < 1.0
    assert torch.allclose(
        refit.proposed_gains_tensor(),
        torch.full((64,), 0.75, dtype=torch.float64),
    )


def test_zero_residual_gradient_becomes_explicit_no_op() -> None:
    refit = _refit(token_kl=torch.zeros(64, dtype=torch.float64))
    assert refit.no_op is True
    assert refit.no_op_reason == (
        "nonnegative_predicted_derivative_after_trust_and_box"
    )
    assert refit.predicted_derivative == 0.0
    assert torch.equal(
        refit.proposed_gains_tensor(), torch.ones(64, dtype=torch.float64)
    )


def test_all_zero_relevance_and_singular_gram_use_explicit_floors() -> None:
    examples = tuple(
        CandidateConditionedK64GainGradientExample(
            example_id=f"e{index}",
            family_id=f"f{index}",
            token_gain_gradients=torch.zeros(index + 1, 64),
            token_teacher_kl=torch.ones(index + 1),
        )
        for index in range(7)
    )
    refit = fit_candidate_conditioned_k64_gains(
        examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.zeros(64),
    )
    tiny = torch.finfo(torch.float64).tiny
    assert refit.damping == 1.0e-12
    assert torch.equal(
        refit.relevance_regularizer.diagonal(),
        torch.full((64,), tiny, dtype=torch.float64),
    )
    assert bool(torch.isfinite(refit.raw_delta).all())
    assert refit.no_op is True


def _tune_examples(
    values_by_alpha: tuple[float, float, float, float],
    *,
    family_zero_override: tuple[float, float, float, float] | None = None,
) -> tuple[CandidateConditionedK64GainTuneExample, ...]:
    rows = []
    for index in range(7):
        values = (
            family_zero_override
            if index == 0 and family_zero_override is not None
            else values_by_alpha
        )
        rows.append(
            CandidateConditionedK64GainTuneExample(
                example_id=f"t{index}",
                family_id=f"f{index}",
                token_teacher_kl_by_alpha=tuple(
                    torch.full((3,), value, dtype=torch.float64)
                    for value in values
                ),
            )
        )
    return tuple(rows)


def _tune_vector_examples(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    candidates_by_family: dict[int, torch.Tensor] | None = None,
    example_prefix: str = "t",
) -> tuple[CandidateConditionedK64GainTuneExample, ...]:
    return tuple(
        CandidateConditionedK64GainTuneExample(
            example_id=f"{example_prefix}{index}",
            family_id=f"f{index}",
            token_teacher_kl_by_alpha=(
                baseline,
                (candidates_by_family or {}).get(index, candidate),
                (candidates_by_family or {}).get(index, candidate),
                (candidates_by_family or {}).get(index, candidate),
            ),
        )
        for index in range(7)
    )


def test_tune_selects_largest_safe_positive_alpha() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_examples((1.0, 0.9, 0.8, 0.7)),
    )
    assert CANDIDATE_GAIN_ALPHAS == (0.0, 0.25, 0.5, 1.0)
    assert selection.selected_alpha == 1.0
    assert torch.equal(
        selection.selected_gains_tensor(), refit.proposed_gains_tensor()
    )


def test_tune_family_guard_can_reject_alpha_one_and_choose_point_five() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_examples(
            (1.0, 0.9, 0.8, 0.7),
            family_zero_override=(1.0, 0.9, 0.8, 1.2),
        ),
    )
    assert selection.selected_alpha == 0.5


def test_tune_falls_back_to_zero_without_squared_KL_improvement() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_examples((1.0, 1.0, 1.0, 1.0)),
    )
    assert selection.selected_alpha == 0.0
    assert selection.selection_reason == "alpha_zero_fallback"
    assert torch.equal(
        selection.selected_gains_tensor(), torch.ones(64, dtype=torch.float64)
    )


def test_tune_requires_mean_KL_improvement_separately_from_squared_objective() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_vector_examples(
            torch.tensor([0.0, 1.5, 1.5]),
            torch.ones(3),
        ),
    )
    assert selection.aggregate_half_mean_squared_teacher_kl_by_alpha[1] < (
        selection.aggregate_half_mean_squared_teacher_kl_by_alpha[0]
    )
    assert selection.aggregate_mean_teacher_kl_by_alpha[1] == pytest.approx(
        selection.aggregate_mean_teacher_kl_by_alpha[0]
    )
    assert selection.selected_alpha == 0.0


def test_tune_requires_squared_objective_improvement_separately_from_mean_KL() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_vector_examples(
            torch.ones(3),
            torch.tensor([0.0, 0.0, 2.9]),
        ),
    )
    assert selection.aggregate_mean_teacher_kl_by_alpha[1] < (
        selection.aggregate_mean_teacher_kl_by_alpha[0]
    )
    assert selection.aggregate_half_mean_squared_teacher_kl_by_alpha[1] > (
        selection.aggregate_half_mean_squared_teacher_kl_by_alpha[0]
    )
    assert selection.selected_alpha == 0.0


def test_tune_requires_every_family_to_clear_five_percent_cap() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_vector_examples(
            torch.ones(3),
            torch.full((3,), 0.5),
            candidates_by_family={0: torch.full((3,), 1.051)},
        ),
    )
    assert selection.family_improved_or_equal_count_by_alpha[1] == 6
    assert selection.worst_family_ratio_by_alpha[1] > 1.05
    assert selection.selected_alpha == 0.0


def test_tune_zero_baseline_large_candidate_abstains_without_ratio_overflow() -> None:
    selection = select_candidate_conditioned_k64_gain_alpha(
        _refit(),
        _tune_vector_examples(torch.zeros(3), torch.full((3,), 10.0)),
    )
    assert selection.selected_alpha == 0.0
    assert all(
        torch.isfinite(torch.tensor(value, dtype=torch.float64))
        for value in selection.worst_family_ratio_by_alpha
    )


def test_tune_requires_four_of_seven_families_improve_or_equal() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit,
        _tune_vector_examples(
            torch.ones(3),
            torch.full((3,), 1.04),
            candidates_by_family={
                0: torch.full((3,), 0.5),
                1: torch.full((3,), 0.5),
                2: torch.full((3,), 0.5),
            },
        ),
    )
    assert selection.aggregate_mean_teacher_kl_by_alpha[1] < (
        selection.aggregate_mean_teacher_kl_by_alpha[0]
    )
    assert selection.worst_family_ratio_by_alpha[1] <= 1.05
    assert selection.family_improved_or_equal_count_by_alpha[1] == 3
    assert selection.selected_alpha == 0.0


def test_tune_prompt_ids_must_be_disjoint_from_fit_prompt_ids() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        select_candidate_conditioned_k64_gain_alpha(
            _refit(),
            _tune_vector_examples(
                torch.ones(3), torch.full((3,), 0.5), example_prefix="e"
            ),
        )


def test_refit_excludes_held_and_requires_exactly_seven_families() -> None:
    with pytest.raises(ValueError, match="seven disjoint families"):
        fit_candidate_conditioned_k64_gains(
            _gradient_examples()[:-1],
            held_family_id="held",
            parent_fold_artifact_sha256="a" * 64,
            ordered_directions_sha256="b" * 64,
            ordered_token_fisher_relevance=torch.ones(64),
        )
    with pytest.raises(ValueError, match="seven disjoint families"):
        fit_candidate_conditioned_k64_gains(
            _gradient_examples(),
            held_family_id="f0",
            parent_fold_artifact_sha256="a" * 64,
            ordered_directions_sha256="b" * 64,
            ordered_token_fisher_relevance=torch.ones(64),
        )


def test_mutating_hidden_refit_tensor_breaks_integrity() -> None:
    refit = _refit()
    refit.proposed_gains[0] = 0.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        refit.validate_integrity()


def test_direct_contradictory_refit_derived_fields_are_rejected() -> None:
    refit = _refit()
    contradictions = (
        {"relevance_regularizer": refit.relevance_regularizer + torch.eye(64)},
        {"damping": refit.damping * 2.0},
        {"damped_system": refit.damped_system + torch.eye(64)},
        {"raw_delta": refit.raw_delta + 0.01},
        {"raw_delta_rms": refit.raw_delta_rms + 0.01},
        {"trust_scale": 1.0},
        {"proposed_gains": torch.ones(64)},
        {"predicted_derivative": refit.predicted_derivative * 0.5},
    )
    for contradiction in contradictions:
        with pytest.raises(ValueError, match="refit payload is invalid"):
            replace(refit, **contradiction)


def test_direct_contradictory_tune_selection_is_rejected() -> None:
    selection = select_candidate_conditioned_k64_gain_alpha(
        _refit(), _tune_examples((1.0, 1.0, 1.0, 1.0))
    )
    with pytest.raises(ValueError, match="tune selection payload is invalid"):
        replace(
            selection,
            selected_alpha=1.0,
            selected_gains=torch.ones(64),
            selection_reason="fabricated_pass",
        )


def test_metadata_is_json_scalar_hash_only_without_tensor_objects() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_gain_alpha(
        refit, _tune_examples((1.0, 0.9, 0.8, 0.7))
    )
    metadata = {
        "gradient": _gradient_examples()[0].metadata(),
        "refit": refit.metadata(),
        "tune": _tune_examples((1.0, 0.9, 0.8, 0.7))[0].metadata(),
        "selection": selection.metadata(),
    }

    def assert_no_tensor(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for nested in value.values():
                assert_no_tensor(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                assert_no_tensor(nested)

    assert_no_tensor(metadata)
    json.dumps(metadata, sort_keys=True)
