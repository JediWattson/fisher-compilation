from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    MEAN_KL_GAIN_ALPHAS,
    REVERSE_RESIDUAL_GAIN_BETAS,
    CandidateConditionedK64DualTuneExample,
    CandidateConditionedK64GainGradientExampleV4,
    contract_candidate_teacher_kl_gain_scores,
    fit_candidate_conditioned_k64_mean_kl_gains,
    select_candidate_conditioned_k64_dual_tune_steps,
)


def _gradient_examples(
    *, token_kl: torch.Tensor | None = None
) -> tuple[CandidateConditionedK64GainGradientExampleV4, ...]:
    q = torch.eye(64, dtype=torch.float64)
    losses = torch.ones(64, dtype=torch.float64) if token_kl is None else token_kl
    return tuple(
        CandidateConditionedK64GainGradientExampleV4(
            example_id=f"e{index}",
            family_id=f"f{index}",
            token_gain_gradients=q,
            token_teacher_kl=losses,
        )
        for index in range(7)
    )


def _refit(*, token_kl: torch.Tensor | None = None):
    return fit_candidate_conditioned_k64_mean_kl_gains(
        _gradient_examples(token_kl=token_kl),
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.ones(64, dtype=torch.float64),
    )


def _dual_tune_examples(
    *,
    unit: torch.Tensor | None = None,
    mean: tuple[torch.Tensor, ...] | None = None,
    reverse: tuple[torch.Tensor, ...] | None = None,
    mean_by_family: dict[int, tuple[torch.Tensor, ...]] | None = None,
    reverse_by_family: dict[int, tuple[torch.Tensor, ...]] | None = None,
    example_prefix: str = "t",
) -> tuple[CandidateConditionedK64DualTuneExample, ...]:
    unit_value = torch.ones(3, dtype=torch.float64) if unit is None else unit
    mean_values = mean or tuple(
        torch.full((3,), value, dtype=torch.float64)
        for value in (0.9, 0.8, 0.7, 0.6)
    )
    reverse_values = reverse or tuple(
        torch.full((3,), value, dtype=torch.float64)
        for value in (0.95, 0.85, 0.75)
    )
    return tuple(
        CandidateConditionedK64DualTuneExample(
            example_id=f"{example_prefix}{index}",
            family_id=f"f{index}",
            unit_token_teacher_kl=unit_value,
            mean_token_teacher_kl_by_positive_alpha=(mean_by_family or {}).get(
                index, mean_values
            ),
            reverse_token_teacher_kl_by_positive_beta=(
                reverse_by_family or {}
            ).get(index, reverse_values),
        )
        for index in range(7)
    )


def test_v4_gain_score_contraction_matches_autograd_pullback() -> None:
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
    candidate_rows = ((tail @ directions.T) * gains) @ directions
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


def test_v4_fit_builds_family_equal_b_c_and_shared_F() -> None:
    examples = []
    for index in range(7):
        token_count = index + 1
        q = torch.zeros(token_count, 64, dtype=torch.float64)
        kl = torch.ones(token_count, dtype=torch.float64)
        if index == 0:
            q[:, 0] = 1.0
            kl[:] = 2.0
        elif index == 1:
            q[:, 1] = 1.0
            kl[:] = 4.0
        examples.append(
            CandidateConditionedK64GainGradientExampleV4(
                example_id=f"e{index}",
                family_id=f"f{index}",
                token_gain_gradients=q,
                token_teacher_kl=kl,
            )
        )
    refit = fit_candidate_conditioned_k64_mean_kl_gains(
        examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.ones(64),
    )
    expected_b = torch.zeros(64, dtype=torch.float64)
    expected_b[:2] = 1.0 / 7.0
    expected_c = torch.zeros(64, dtype=torch.float64)
    expected_c[0] = 2.0 / 7.0
    expected_c[1] = 4.0 / 7.0
    expected_F = torch.zeros(64, 64, dtype=torch.float64)
    expected_F[0, 0] = 1.0 / 7.0
    expected_F[1, 1] = 1.0 / 7.0
    assert torch.equal(refit.mean_gradient_b, expected_b)
    assert torch.equal(refit.residual_gradient_c, expected_c)
    assert torch.equal(refit.gradient_gram, expected_F)


def test_mean_and_residual_directions_share_H_but_can_differ() -> None:
    losses = torch.linspace(0.1, 3.0, 64, dtype=torch.float64)
    refit = _refit(token_kl=losses)
    assert not torch.equal(refit.mean_gradient_b, refit.residual_gradient_c)
    assert not torch.equal(
        refit.mean_proposed_gains_tensor(),
        refit.residual_proposed_gains_tensor(),
    )
    assert refit.mean_predicted_derivative < 0.0
    assert refit.residual_predicted_derivative < 0.0
    assert refit.metadata()["mean_fit_objective"] == "expected_token_teacher_KL"
    assert refit.metadata()["reverse_residual_is_optimizer"] is False


def test_each_direction_has_independent_point_25_rms_trust() -> None:
    refit = _refit()
    mean_delta = refit.mean_proposed_gains_tensor() - 1.0
    residual_delta = refit.residual_proposed_gains_tensor() - 1.0
    assert float(mean_delta.square().mean().sqrt()) == pytest.approx(0.25)
    assert float(residual_delta.square().mean().sqrt()) == pytest.approx(0.25)
    assert torch.allclose(
        refit.mean_proposed_gains_tensor(),
        torch.full((64,), 0.75, dtype=torch.float64),
    )


def test_cross_derivatives_and_cosines_reconstruct_from_applied_deltas() -> None:
    refit = _refit(token_kl=torch.linspace(0.1, 3.0, 64))
    mean_delta = refit.mean_proposed_gains_tensor() - 1.0
    residual_delta = refit.residual_proposed_gains_tensor() - 1.0
    assert refit.mean_gradient_on_residual_delta == pytest.approx(
        float(torch.dot(refit.mean_gradient_b, residual_delta))
    )
    assert refit.residual_gradient_on_mean_delta == pytest.approx(
        float(torch.dot(refit.residual_gradient_c, mean_delta))
    )
    expected_gradient_cosine = float(
        torch.dot(refit.mean_gradient_b, refit.residual_gradient_c)
        / (
            torch.linalg.vector_norm(refit.mean_gradient_b)
            * torch.linalg.vector_norm(refit.residual_gradient_c)
        )
    )
    assert refit.gradient_cosine == pytest.approx(expected_gradient_cosine)
    assert -1.0 <= refit.applied_delta_cosine <= 1.0


def test_zero_bank_makes_both_directions_explicit_no_ops() -> None:
    examples = tuple(
        CandidateConditionedK64GainGradientExampleV4(
            example_id=f"e{index}",
            family_id=f"f{index}",
            token_gain_gradients=torch.zeros(index + 1, 64),
            token_teacher_kl=torch.ones(index + 1),
        )
        for index in range(7)
    )
    refit = fit_candidate_conditioned_k64_mean_kl_gains(
        examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256="b" * 64,
        ordered_token_fisher_relevance=torch.zeros(64),
    )
    assert refit.mean_no_op is True
    assert refit.residual_no_op is True
    assert refit.mean_predicted_derivative == 0.0
    assert refit.residual_predicted_derivative == 0.0
    assert torch.equal(refit.mean_proposed_gains, torch.ones(64))
    assert refit.damping == 1.0e-12


def test_reverse_residual_grid_is_bounded_and_is_an_ascent_control() -> None:
    refit = _refit(token_kl=torch.linspace(0.1, 3.0, 64))
    for beta in REVERSE_RESIDUAL_GAIN_BETAS:
        gains = refit.reverse_residual_gains_tensor(beta)
        assert bool((gains >= 0.0).all())
        assert bool((gains <= 1.5).all())
        derivative = float(torch.dot(refit.residual_gradient_c, gains - 1.0))
        if beta > 0.0:
            assert derivative > 0.0


def test_dual_tune_evidence_structurally_shares_unit_execution() -> None:
    example = _dual_tune_examples()[0]
    assert MEAN_KL_GAIN_ALPHAS == (0.0, 0.125, 0.25, 0.5, 1.0)
    assert REVERSE_RESIDUAL_GAIN_BETAS == (0.0, 0.125, 0.25, 0.5)
    assert torch.equal(example.mean_token_kl(0.0), example.reverse_token_kl(0.0))
    metadata = example.metadata()
    assert metadata["alpha_zero_and_beta_zero_share_one_unit_execution"] is True
    assert "unit_token_teacher_kl_sha256" in metadata
    assert "mean_token_teacher_kl_sha256_by_alpha" not in metadata


def test_dual_tune_selects_largest_safe_mean_and_reverse_steps() -> None:
    refit = _refit()
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        refit, _dual_tune_examples()
    )
    assert selection.selected_mean_alpha == 1.0
    assert selection.selected_reverse_beta == 0.5
    assert torch.equal(
        selection.selected_mean_gains_tensor(), refit.mean_proposed_gains
    )
    assert torch.equal(
        selection.selected_reverse_gains_tensor(),
        refit.reverse_residual_gains_tensor(0.5),
    )
    assert selection.reverse_diagnostic_only is True
    assert selection.metadata()["reverse_arm_can_authorize_primary_refit"] is False


def test_mean_selection_does_not_gate_on_squared_KL() -> None:
    baseline = torch.ones(3, dtype=torch.float64)
    lower_mean_higher_squared = torch.tensor([0.0, 0.0, 2.9])
    mean_values = (lower_mean_higher_squared,) * 4
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        _refit(),
        _dual_tune_examples(unit=baseline, mean=mean_values),
    )
    assert selection.mean_aggregate_mean_teacher_kl_by_alpha[1] < (
        selection.mean_aggregate_mean_teacher_kl_by_alpha[0]
    )
    assert selection.mean_aggregate_half_mean_squared_teacher_kl_by_alpha[1] > (
        selection.mean_aggregate_half_mean_squared_teacher_kl_by_alpha[0]
    )
    assert selection.selected_mean_alpha == 1.0
    assert selection.metadata()["squared_KL_is_selection_gate"] is False


def test_mean_tune_requires_every_family_to_clear_five_percent_cap() -> None:
    safe = tuple(torch.full((3,), 0.5) for _ in range(4))
    unsafe = tuple(torch.full((3,), 1.051) for _ in range(4))
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        _refit(), _dual_tune_examples(mean=safe, mean_by_family={0: unsafe})
    )
    assert selection.mean_family_improved_or_equal_count_by_alpha[1] == 6
    assert selection.mean_worst_family_ratio_by_alpha[1] > 1.05
    assert selection.selected_mean_alpha == 0.0


def test_mean_tune_requires_four_of_seven_families_nonworse() -> None:
    worse = tuple(torch.full((3,), 1.04) for _ in range(4))
    better = tuple(torch.full((3,), 0.5) for _ in range(4))
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        _refit(),
        _dual_tune_examples(
            mean=worse,
            mean_by_family={0: better, 1: better, 2: better},
        ),
    )
    assert selection.mean_aggregate_mean_teacher_kl_by_alpha[1] < 1.0
    assert selection.mean_worst_family_ratio_by_alpha[1] <= 1.05
    assert selection.mean_family_improved_or_equal_count_by_alpha[1] == 3
    assert selection.selected_mean_alpha == 0.0


def test_reverse_selection_uses_exact_mean_KL_but_remains_diagnostic() -> None:
    baseline = torch.ones(3)
    candidate = torch.tensor([0.0, 0.0, 2.9])
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        _refit(),
        _dual_tune_examples(
            unit=baseline,
            reverse=(candidate, candidate, candidate),
        ),
    )
    assert selection.reverse_aggregate_mean_teacher_kl_by_beta[1] < 1.0
    assert selection.reverse_aggregate_half_mean_squared_teacher_kl_by_beta[1] > 0.5
    assert selection.selected_reverse_beta == 0.5
    assert "diagnostic" in selection.metadata()["reverse_selection_reason"] or (
        selection.metadata()["reverse_diagnostic_only"] is True
    )


def test_tune_prompt_ids_must_be_disjoint_from_fit_prompt_ids() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        select_candidate_conditioned_k64_dual_tune_steps(
            _refit(), _dual_tune_examples(example_prefix="e")
        )


def test_refit_excludes_held_and_requires_exactly_seven_families() -> None:
    with pytest.raises(ValueError, match="seven disjoint families"):
        fit_candidate_conditioned_k64_mean_kl_gains(
            _gradient_examples()[:-1],
            held_family_id="held",
            parent_fold_artifact_sha256="a" * 64,
            ordered_directions_sha256="b" * 64,
            ordered_token_fisher_relevance=torch.ones(64),
        )
    with pytest.raises(ValueError, match="seven disjoint families"):
        fit_candidate_conditioned_k64_mean_kl_gains(
            _gradient_examples(),
            held_family_id="f0",
            parent_fold_artifact_sha256="a" * 64,
            ordered_directions_sha256="b" * 64,
            ordered_token_fisher_relevance=torch.ones(64),
        )


def test_mutating_hidden_v4_tensors_breaks_integrity() -> None:
    refit = _refit()
    refit.mean_gradient_b[0] = 100.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        refit.validate_integrity()
    tune = _dual_tune_examples()[0]
    tune.unit_token_teacher_kl[0] = 100.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        tune.validate_integrity()


def test_direct_contradictory_v4_derived_fields_are_rejected() -> None:
    refit = _refit()
    contradictions = (
        {"mean_raw_delta": refit.mean_raw_delta + 0.01},
        {"residual_raw_delta": refit.residual_raw_delta + 0.01},
        {"mean_proposed_gains": torch.ones(64)},
        {"mean_gradient_on_residual_delta": 123.0},
        {"gradient_cosine": -1.0},
    )
    for contradiction in contradictions:
        with pytest.raises(ValueError, match="refit payload is invalid"):
            replace(refit, **contradiction)


def test_direct_contradictory_dual_selection_is_rejected() -> None:
    selection = select_candidate_conditioned_k64_dual_tune_steps(
        _refit(), _dual_tune_examples()
    )
    with pytest.raises(ValueError, match="selection payload is invalid"):
        replace(
            selection,
            selected_mean_alpha=0.0,
            selected_mean_gains=torch.ones(64),
            mean_selection_reason="fabricated",
        )
    with pytest.raises(ValueError, match="selection payload is invalid"):
        replace(selection, reverse_diagnostic_only=False)


def test_v4_metadata_is_json_scalar_hash_only_without_tensor_objects() -> None:
    refit = _refit(token_kl=torch.linspace(0.1, 3.0, 64))
    tune = _dual_tune_examples()
    selection = select_candidate_conditioned_k64_dual_tune_steps(refit, tune)
    metadata = {
        "gradient": _gradient_examples()[0].metadata(),
        "refit": refit.metadata(),
        "tune": tune[0].metadata(),
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
    serialized = json.dumps(metadata, sort_keys=True)
    assert "raw_tensors_serialized\": true" not in serialized
