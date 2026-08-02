from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_state_router import (
    GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE,
    ROUTE_OPERATOR_NORM_BOUND,
    GemmaCausalTop2BalanceH4Provider,
    GemmaIterativeStateRouterFitRecord,
    GemmaIterativeStateRouterFoldFit,
    _project_route_coefficients,
    build_gemma_iterative_state_router_fit_record,
    fit_gemma_iterative_state_router_fold,
    top2_lag_b_output_modes,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _parent() -> GemmaCausalResidualHead:
    return GemmaCausalResidualHead(
        site="layer.4.output",
        parent_runtime_binding_sha256=_sha(1),
        residual_map_sha256=_sha(2),
        analysis_artifact_sha256=_sha(3),
        fit_manifest_sha256=_sha(4),
        bridge_binding_sha256=_sha(5),
        decoder=torch.eye(3, dtype=torch.float64),
        lag_kernel=torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.1],
                ]
            ],
            dtype=torch.float64,
        ),
        state_kernel=torch.empty((0, 0), dtype=torch.float64),
        conditioning="l3_source_modes",
        ridge=1.0e-6,
        fit_row_count=8,
        family_ids=("fit-a", "fit-b"),
        fit_sequence_sha256s=(_sha(6), _sha(7)),
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
        weighted_residual_rmse=1.0,
        normalized_nll_direction_rmse=1.0,
        linearized_nll_residual_rmse=1.0,
    )


def _parent_with_history() -> GemmaCausalResidualHead:
    parent = _parent()
    return GemmaCausalResidualHead(
        site=parent.site,
        parent_runtime_binding_sha256=parent.parent_runtime_binding_sha256,
        residual_map_sha256=parent.residual_map_sha256,
        analysis_artifact_sha256=parent.analysis_artifact_sha256,
        fit_manifest_sha256=parent.fit_manifest_sha256,
        bridge_binding_sha256=parent.bridge_binding_sha256,
        decoder=parent.decoder.clone(),
        lag_kernel=torch.cat(
            (
                parent.lag_kernel.clone(),
                torch.tensor(
                    [[[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]]],
                    dtype=torch.float64,
                ),
            ),
            dim=0,
        ),
        state_kernel=parent.state_kernel.clone(),
        conditioning=parent.conditioning,
        ridge=parent.ridge,
        fit_row_count=parent.fit_row_count,
        family_ids=parent.family_ids,
        fit_sequence_sha256s=parent.fit_sequence_sha256s,
        fit_objective=parent.fit_objective,
        weighted_residual_rmse=parent.weighted_residual_rmse,
        normalized_nll_direction_rmse=parent.normalized_nll_direction_rmse,
        linearized_nll_residual_rmse=parent.linearized_nll_residual_rmse,
    )


def _prefix(
    source_modes: torch.Tensor | None = None,
    *,
    positions: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> Gemma3L3L4OnePassPrefix:
    if source_modes is None:
        source_modes = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 0.5],
                    [0.5, -1.0],
                ]
            ],
            dtype=torch.float64,
        )
    batch, length, _rank = source_modes.shape
    if positions is None:
        positions = torch.arange(length, dtype=torch.int64).expand(
            batch,
            -1,
        )
    valid = torch.ones((batch, length), dtype=torch.bool)
    if active is None:
        active = valid.clone()
    return Gemma3L3L4OnePassPrefix(
        source_modes=source_modes,
        clamped_y3=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        predicted_target_modal_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        decoded_base_x4_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        logical_positions=positions,
        valid_target_mask=valid,
        source_eligible_mask=valid.clone(),
        target_affected_mask=active,
        bridge_binding_sha256=_sha(5),
    )


def _slice_prefix(
    prefix: Gemma3L3L4OnePassPrefix,
    start: int,
    stop: int,
) -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=prefix.source_modes[:, start:stop].clone(),
        clamped_y3=prefix.clamped_y3[:, start:stop].clone(),
        predicted_target_modal_delta=(
            prefix.predicted_target_modal_delta[:, start:stop].clone()
        ),
        decoded_base_x4_delta=(
            prefix.decoded_base_x4_delta[:, start:stop].clone()
        ),
        logical_positions=prefix.logical_positions[:, start:stop].clone(),
        valid_target_mask=prefix.valid_target_mask[:, start:stop].clone(),
        source_eligible_mask=(
            prefix.source_eligible_mask[:, start:stop].clone()
        ),
        target_affected_mask=(
            prefix.target_affected_mask[:, start:stop].clone()
        ),
        bridge_binding_sha256=prefix.bridge_binding_sha256,
    )


def _fold(
    coefficients: tuple[float, float, float, float],
) -> GemmaIterativeStateRouterFoldFit:
    operator_norm = float(
        torch.linalg.svdvals(
            torch.tensor(coefficients, dtype=torch.float64).reshape(2, 2)
        ).max()
    )
    return GemmaIterativeStateRouterFoldFit(
        held_family_id="held",
        train_example_ids=("train",),
        train_family_ids=("family",),
        train_fit_record_sha256s=(_sha(8),),
        coefficients_by_route_edge=coefficients,
        unsupported_route_edge_indices=(),
        active_row_count=5,
        weighted_column_norm_by_route_edge=(1.0, 1.0, 1.0, 1.0),
        weighted_design_rank=4,
        normal_condition_number=1.0,
        pre_projection_operator_norm=operator_norm,
        post_projection_operator_norm=operator_norm,
        linearized_rmse_before=1.0,
        linearized_rmse_after=0.5,
        trust_projection_applied=False,
    )


def _provider(
    coefficients: tuple[float, float, float, float],
    *,
    parent: GemmaCausalResidualHead | None = None,
) -> GemmaCausalTop2BalanceH4Provider:
    return GemmaCausalTop2BalanceH4Provider(
        parent_h4=_parent() if parent is None else parent,
        parent_artifact_sha256=_sha(9),
        fold_fit=_fold(coefficients),
    )


def _record(
    example: str,
    family: str,
    *,
    delta: float,
    jacobian: tuple[float, float, float, float],
) -> GemmaIterativeStateRouterFitRecord:
    return GemmaIterativeStateRouterFitRecord(
        example_id=example,
        family_id=family,
        model_inputs_sha256=_sha(10),
        parent_execution_sha256=_sha(11),
        parent_observation_sha256=_sha(12),
        parent_h4_artifact_sha256=_parent().artifact_sha256,
        prefix_sha256=_sha(13),
        gradient_sha256=_sha(14),
        parent_modal_sha256=_sha(15),
        balance_feature_sha256=_sha(16),
        gated_modal_sha256=_sha(17),
        supervised_tokens=2,
        parent_signed_delta_nll_per_token=delta,
        jacobian_by_route_edge=jacobian,
        active_row_count=5,
        top_mode_indices=(0, 1),
        top_mode_norms=(2.0, 1.0),
        balance_feature_std=0.25,
        top2_modal_energy_fraction=0.99,
    )


def test_modal_helpers_preserve_parent_bytes_and_theta_zero_equivalence() -> None:
    parent = _parent()
    prefix = _prefix(
        active=torch.tensor([[True, True, False, True, True]])
    )
    realized = torch.randn(1, 5, 3, dtype=torch.float64)
    modal = parent.modal_correction(prefix, realized)
    decoded = parent.decode_modal(prefix, modal, like=prefix.clamped_y3)
    assert torch.equal(decoded, parent.correction(prefix, realized))
    assert torch.equal(modal[0, 2], torch.zeros(3, dtype=torch.float64))

    provider = _provider((0.0, 0.0, 0.0, 0.0))
    actual = provider.correction(prefix, realized)
    expected = provider.parent_h4.correction(prefix, realized)
    assert torch.equal(actual, expected)
    assert provider.marginal_learned_float_scalar_count == 4
    assert provider.marginal_prepared_float_scalar_count == 6
    assert provider.marginal_logical_macs_per_token_upper_bound == 6
    assert provider.nonlinear_scalar_ops_per_token_upper_bound == 5
    assert provider.runtime_state_float_scalars_per_sequence == 2


def test_top_modes_are_deterministic_and_off_diagonal_route_fans_out() -> None:
    parent = _parent()
    assert top2_lag_b_output_modes(parent) == ((0, 1), (2.0, 1.0))
    provider = _provider((0.0, 0.2, 0.0, 0.0))
    prefix = _prefix(
        source_modes=torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]]],
            dtype=torch.float64,
        )
    )
    parent_modal = parent.modal_correction(
        prefix,
        torch.zeros(1, 2, 3, dtype=torch.float64),
    )
    state = provider.initial_state(
        1,
        device=parent_modal.device,
        dtype=parent_modal.dtype,
    )
    routed, _next_state = provider.route_modal_with_state(
        prefix,
        parent_modal,
        state,
    )
    assert torch.equal(parent_modal[..., 1], torch.zeros(1, 2))
    assert torch.equal(
        routed[..., 1],
        torch.full((1, 2), 0.4, dtype=torch.float64),
    )
    assert torch.equal(routed[..., 2], parent_modal[..., 2])


def test_zero_balance_denominator_is_exactly_zero_and_finite() -> None:
    provider = _provider((0.1, 0.1, 0.1, 0.1))
    prefix = _prefix(
        source_modes=torch.zeros(1, 3, 2, dtype=torch.float64)
    )
    state = provider.initial_state(1, device="cpu")
    parent_modal = provider.parent_h4.modal_correction(
        prefix,
        torch.zeros(1, 3, 3, dtype=torch.float64),
    )
    routed, next_state = provider.route_modal_with_state(
        prefix,
        parent_modal,
        state,
    )
    assert torch.equal(routed, parent_modal)
    assert torch.equal(next_state.numerator, torch.zeros(1))
    assert torch.equal(next_state.denominator, torch.zeros(1))
    assert bool(torch.isfinite(routed).all())


def test_router_is_causal_and_chunk_carry_matches_full_execution() -> None:
    provider = _provider(
        (0.1, 0.05, -0.025, 0.1),
        parent=_parent_with_history(),
    )
    prefix = _prefix()
    realized = torch.zeros(1, 5, 3, dtype=torch.float64)
    parent_modal = provider.parent_h4.modal_correction(prefix, realized)
    full, full_state = provider.correction_from_parent_modal_with_state(
        prefix,
        parent_modal,
        provider.initial_state(1, device="cpu"),
    )

    first_prefix = _slice_prefix(prefix, 0, 2)
    second_prefix = _slice_prefix(prefix, 2, 5)
    first, carry = provider.correction_from_parent_modal_with_state(
        first_prefix,
        parent_modal[:, :2],
        provider.initial_state(1, device="cpu"),
    )
    second, chunk_state = provider.correction_from_parent_modal_with_state(
        second_prefix,
        parent_modal[:, 2:],
        carry,
    )
    assert torch.equal(torch.cat((first, second), dim=1), full)
    assert torch.equal(chunk_state.numerator, full_state.numerator)
    assert torch.equal(chunk_state.denominator, full_state.denominator)

    changed_source = prefix.source_modes.clone()
    changed_source[:, 3:] = 1000.0
    changed = provider.correction(
        _prefix(source_modes=changed_source),
        realized,
    )
    assert torch.equal(changed[:, :3], full[:, :3])


def test_projection_preserves_support_and_exact_bound_is_not_projected() -> None:
    exact_bound = torch.tensor(
        [ROUTE_OPERATOR_NORM_BOUND, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    projected, pre_norm, post_norm, applied = _project_route_coefficients(
        exact_bound,
        supported=(0, 1, 2, 3),
    )
    assert torch.equal(projected, exact_bound)
    assert pre_norm == ROUTE_OPERATOR_NORM_BOUND
    assert post_norm == ROUTE_OPERATOR_NORM_BOUND
    assert applied is False

    support_limited = torch.tensor(
        [10.0, 10.0, 10.0, 0.0],
        dtype=torch.float64,
    )
    projected, pre_norm, post_norm, applied = _project_route_coefficients(
        support_limited,
        supported=(0, 1, 2),
    )
    assert applied is True
    assert pre_norm > ROUTE_OPERATOR_NORM_BOUND
    assert post_norm <= ROUTE_OPERATOR_NORM_BOUND
    assert projected[3].item() == 0.0


@dataclass
class _Example:
    example_id: str = "example"
    family_id: str = "family"
    model_inputs_sha256: str = _sha(20)


@dataclass
class _Execution:
    prefix: Gemma3L3L4OnePassPrefix
    candidate_h4: torch.Tensor
    h4_head_sha256: str
    artifact_sha256: str = _sha(21)
    model_inputs_sha256: str = _sha(20)

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()


def _observation() -> GemmaH4DampingFiniteNLLObservation:
    return GemmaH4DampingFiniteNLLObservation(
        example_id="example",
        family_id="family",
        supervised_tokens=5,
        source_summed_nll=4.0,
        candidate_summed_nll=4.5,
        source_to_candidate_summed_kl=0.2,
        top1_matches=3,
        source_logits_sha256=_sha(22),
        candidate_logits_sha256=_sha(23),
        targets_sha256=_sha(24),
    )


def test_fit_record_jacobian_matches_finite_displacement() -> None:
    parent = _parent()
    prefix = _prefix()
    candidate_h4 = torch.zeros(1, 5, 3, dtype=torch.float64)
    execution = _Execution(
        prefix=prefix,
        candidate_h4=candidate_h4,
        h4_head_sha256=parent.artifact_sha256,
    )
    gradient = torch.tensor(
        [
            [
                [0.25, -0.5, 0.1],
                [1.0, 0.25, -0.2],
                [-0.5, 0.75, 0.3],
                [0.2, -0.1, 0.4],
                [-0.7, 0.5, -0.6],
            ]
        ],
        dtype=torch.float64,
    )
    record = build_gemma_iterative_state_router_fit_record(
        example=_Example(),
        parent_execution=execution,
        gradient=gradient,
        parent_h4=parent,
        parent_observation=_observation(),
    )
    assert record.active_row_count == 5
    assert record.top_mode_indices == (0, 1)
    assert record.balance_feature_std > 0
    assert record.top2_modal_energy_fraction > 0.99
    assert all(
        len(value) == 64
        for value in (
            record.prefix_sha256,
            record.gradient_sha256,
            record.parent_modal_sha256,
            record.balance_feature_sha256,
            record.gated_modal_sha256,
        )
    )

    epsilon = 1.0e-6
    for edge in range(4):
        plus = [0.0] * 4
        minus = [0.0] * 4
        plus[edge] = epsilon
        minus[edge] = -epsilon
        plus_correction = _provider(tuple(plus)).correction(
            prefix,
            candidate_h4,
        )
        minus_correction = _provider(tuple(minus)).correction(
            prefix,
            candidate_h4,
        )
        finite_difference = float(
            (
                gradient
                * (plus_correction - minus_correction)
            ).sum()
            / (2.0 * epsilon * _observation().supervised_tokens)
        )
        assert finite_difference == pytest.approx(
            record.jacobian_by_route_edge[edge],
            rel=1.0e-9,
            abs=1.0e-9,
        )


def test_fold_is_family_balanced_replayable_and_svd_bounded() -> None:
    records = []
    basis = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    for family_index in range(7):
        family = f"family-{family_index}"
        for example_index in range(2):
            records.append(
                _record(
                    f"{family}-{example_index}",
                    family,
                    delta=-10.0,
                    jacobian=basis[
                        (2 * family_index + example_index) % len(basis)
                    ],
                )
            )
    fit = fit_gemma_iterative_state_router_fold(
        records,
        held_family_id="held",
    )
    replay = fit_gemma_iterative_state_router_fold(
        [record.to_dict() for record in reversed(records)],
        held_family_id="held",
    )
    assert replay.to_dict() == fit.to_dict()
    assert fit.weighted_design_rank == 4
    assert fit.trust_projection_applied is True
    assert fit.post_projection_operator_norm <= ROUTE_OPERATOR_NORM_BOUND
    assert fit.pre_projection_operator_norm > ROUTE_OPERATOR_NORM_BOUND

    with pytest.raises(ValueError, match="held family leaked"):
        fit_gemma_iterative_state_router_fold(
            [
                _record(
                    "leaked",
                    "held",
                    delta=1.0,
                    jacobian=(1.0, 1.0, 1.0, 1.0),
                )
            ],
            held_family_id="held",
        )

    tampered = records[0].to_dict()
    tampered["jacobian_by_route_edge"] = (9.0, 9.0, 9.0, 9.0)
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_gemma_iterative_state_router_fold(
            [tampered, *records[1:]],
            held_family_id="held",
        )


def test_provider_integrity_and_campaign_recipe_bind_resources() -> None:
    provider = _provider((0.1, 0.0, 0.0, 0.1))
    receipt = GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE.provider_resource_receipt(
        provider
    )
    assert receipt == {
        "learned_parameter_count": 4,
        "logical_macs_per_token_upper_bound": 6,
        "derived_constant_float_count": 2,
        "runtime_state_float_count_per_sequence": 2,
        "nonlinear_scalar_ops_per_token_upper_bound": 5,
    }
    audit = GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE.provider_audit_receipt(
        provider
    )
    assert audit["routed_parent_decoder_mode_indices"] == (0, 1)
    GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE.validate_resource_envelope(
        resources=receipt,
        residual_width=provider.width,
    )

    provider.parent_h4.lag_kernel[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        provider.validate_integrity()
