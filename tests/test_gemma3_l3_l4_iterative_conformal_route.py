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
from fisher_graph.gemma3_l3_l4_iterative_conformal_route import (
    CONFORMAL_OPERATOR_NORM_BOUND,
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE,
    GemmaCausalTop2ConformalRouteH4Provider,
    GemmaIterativeConformalRouteFitRecord,
    GemmaIterativeConformalRouteFoldFit,
    _endpoint_operator_norms,
    _project_conformal_coefficients,
    build_gemma_iterative_conformal_route_fit_record,
    fit_gemma_iterative_conformal_route_fold,
)
from fisher_graph.gemma3_l3_l4_iterative_state_router import (
    _balance_feature,
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
) -> GemmaIterativeConformalRouteFoldFit:
    endpoints = _endpoint_operator_norms(coefficients)
    return GemmaIterativeConformalRouteFoldFit(
        held_family_id="held",
        train_example_ids=("train",),
        train_family_ids=("family",),
        train_fit_record_sha256s=(_sha(8),),
        coefficients_by_conformal_coefficient=coefficients,
        unsupported_conformal_coefficient_indices=(),
        active_row_count=5,
        weighted_column_norm_by_conformal_coefficient=(
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        weighted_design_rank=4,
        normal_condition_number=1.0,
        pre_projection_endpoint_operator_norms=endpoints,
        post_projection_endpoint_operator_norms=endpoints,
        trust_projection_scale=1.0,
        linearized_rmse_before=1.0,
        linearized_rmse_after=0.5,
        trust_projection_applied=False,
    )


def _provider(
    coefficients: tuple[float, float, float, float],
    *,
    parent: GemmaCausalResidualHead | None = None,
) -> GemmaCausalTop2ConformalRouteH4Provider:
    return GemmaCausalTop2ConformalRouteH4Provider(
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
) -> GemmaIterativeConformalRouteFitRecord:
    return GemmaIterativeConformalRouteFitRecord(
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
        shared_gated_feature_sha256=_sha(17),
        contrast_gated_feature_sha256=_sha(18),
        supervised_tokens=2,
        parent_signed_delta_nll_per_token=delta,
        jacobian_by_conformal_coefficient=jacobian,
        active_row_count=5,
        top_mode_indices=(0, 1),
        top_mode_norms=(2.0, 1.0),
        balance_feature_std=0.25,
        top2_modal_energy_fraction=0.99,
    )


def test_runtime_matches_dense_dynamic_conformal_matrix_exactly() -> None:
    coefficients = (0.08, -0.04, 0.05, 0.03)
    provider = _provider(coefficients)
    prefix = _prefix(
        active=torch.tensor([[True, True, False, True, True]])
    )
    realized = torch.zeros(1, 5, 3, dtype=torch.float64)
    parent_modal = provider.parent_h4.modal_correction(prefix, realized)
    state = provider.initial_state(1, device="cpu")
    routed, actual_state = provider.route_parent_modal_with_state(
        prefix,
        parent_modal,
        state,
    )

    balance, numerator, denominator = _balance_feature(
        prefix=prefix,
        parent_modal=parent_modal,
        top_mode_indices=provider.top_mode_indices,
        top_mode_norms=provider.top_mode_norms,
        initial_numerator=state.numerator,
        initial_denominator=state.denominator,
    )
    selected_indices = torch.tensor(provider.top_mode_indices)
    selected = parent_modal.index_select(2, selected_indices)
    a = coefficients[0] + balance * coefficients[2]
    b = coefficients[1] + balance * coefficients[3]
    dense = torch.zeros(1, 5, 2, 2, dtype=torch.float64)
    dense[..., 0, 0] = a
    dense[..., 0, 1] = -b
    dense[..., 1, 0] = b
    dense[..., 1, 1] = a
    delta = torch.einsum(
        "bti,btij->btj",
        balance.unsqueeze(-1) * selected,
        dense,
    )
    expected = parent_modal.clone()
    expected.index_copy_(2, selected_indices, selected + delta)
    assert torch.allclose(routed, expected, rtol=0.0, atol=1.0e-15)
    assert torch.equal(actual_state.numerator, numerator)
    assert torch.equal(actual_state.denominator, denominator)

    zero = _provider((0.0, 0.0, 0.0, 0.0))
    assert torch.equal(
        zero.correction(prefix, realized),
        zero.parent_h4.correction(prefix, realized),
    )


def test_parent_modal_chunk_api_preserves_balance_and_upstream_lag() -> None:
    provider = _provider(
        (0.07, 0.03, -0.02, 0.04),
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

    reset_parent_modal = provider.parent_h4.modal_correction(
        second_prefix,
        realized[:, 2:],
    )
    assert not torch.equal(reset_parent_modal, parent_modal[:, 2:])


def test_projection_is_global_radial_endpoint_bounded_and_support_safe() -> None:
    exact = torch.tensor(
        [CONFORMAL_OPERATOR_NORM_BOUND, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    projected, pre, post, scale, applied = (
        _project_conformal_coefficients(
            exact,
            supported=(0, 1, 2, 3),
        )
    )
    assert torch.equal(projected, exact)
    assert pre == (CONFORMAL_OPERATOR_NORM_BOUND,) * 2
    assert post == pre
    assert scale == 1.0
    assert applied is False

    raw = torch.tensor([3.0, -4.0, 2.0, 0.0], dtype=torch.float64)
    projected, pre, post, scale, applied = (
        _project_conformal_coefficients(
            raw,
            supported=(0, 1, 2),
        )
    )
    assert applied is True
    assert scale < 1.0
    assert max(pre) > CONFORMAL_OPERATOR_NORM_BOUND
    assert max(post) <= CONFORMAL_OPERATOR_NORM_BOUND
    assert projected[3].item() == 0.0
    assert torch.equal(projected[:3], raw[:3] * scale)
    for g in torch.linspace(-1.0, 1.0, 101):
        a = float(projected[0] + g * projected[2])
        b = float(projected[1] + g * projected[3])
        assert math_hypot(a, b) <= CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12


def math_hypot(left: float, right: float) -> float:
    return float(torch.tensor([left, right], dtype=torch.float64).norm())


def test_fold_recovers_zero_contrast_even_when_contrast_is_supported() -> None:
    basis = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    deltas = (-0.1, 0.05, 0.0, 0.0)
    records = tuple(
        _record(
            f"{family}-{index}",
            family,
            delta=deltas[index],
            jacobian=basis[index],
        )
        for family in ("a", "b")
        for index in range(4)
    )
    fit = fit_gemma_iterative_conformal_route_fold(
        records,
        held_family_id="held",
    )
    assert fit.unsupported_conformal_coefficient_indices == ()
    assert fit.weighted_design_rank == 4
    assert fit.coefficients_by_conformal_coefficient[2:] == (0.0, 0.0)
    assert fit.coefficients_by_conformal_coefficient[0] == pytest.approx(
        0.1,
        rel=1.0e-5,
    )
    assert fit.coefficients_by_conformal_coefficient[1] == pytest.approx(
        -0.05,
        rel=1.0e-5,
    )
    assert fit.trust_projection_applied is False


def test_family_balanced_fold_is_deterministic_and_endpoint_projected() -> None:
    basis = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    records = [
        _record(
            f"family-{family}-{example}",
            f"family-{family}",
            delta=-10.0,
            jacobian=basis[(family + example) % len(basis)],
        )
        for family in range(6)
        for example in range(4)
    ]
    fit = fit_gemma_iterative_conformal_route_fold(
        records,
        held_family_id="held",
    )
    replay = fit_gemma_iterative_conformal_route_fold(
        [row.to_dict() for row in reversed(records)],
        held_family_id="held",
    )
    assert replay.to_dict() == fit.to_dict()
    assert fit.weighted_design_rank == 4
    assert fit.trust_projection_applied is True
    assert fit.trust_projection_scale < 1.0
    assert (
        max(fit.post_projection_endpoint_operator_norms)
        <= CONFORMAL_OPERATOR_NORM_BOUND
    )

    with pytest.raises(ValueError, match="held family leaked"):
        fit_gemma_iterative_conformal_route_fold(
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


def test_full_rank_tiny_columns_report_huge_normal_condition() -> None:
    tiny = 1.0e-7
    basis = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, tiny, 0.0),
        (0.0, 0.0, 0.0, tiny),
    )
    records = tuple(
        _record(
            f"{family}-{index}",
            family,
            delta=0.0,
            jacobian=basis[index],
        )
        for family in ("a", "b")
        for index in range(4)
    )
    fit = fit_gemma_iterative_conformal_route_fold(
        records,
        held_family_id="held",
    )
    assert fit.weighted_design_rank == 4
    assert fit.unsupported_conformal_coefficient_indices == ()
    assert fit.normal_condition_number == pytest.approx(
        1.0e14,
        rel=1.0e-12,
    )
    assert fit.normal_condition_number > 100.0


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


def test_analytic_coefficient_order_matches_finite_displacements() -> None:
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
    record = build_gemma_iterative_conformal_route_fit_record(
        example=_Example(),
        parent_execution=execution,
        gradient=gradient,
        parent_h4=parent,
        parent_observation=_observation(),
    )
    assert record.shared_gated_feature_sha256 != (
        record.contrast_gated_feature_sha256
    )
    assert record.balance_feature_std > 0.0

    epsilon = 1.0e-6
    for coefficient in range(4):
        plus = [0.0] * 4
        minus = [0.0] * 4
        plus[coefficient] = epsilon
        minus[coefficient] = -epsilon
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
            record.jacobian_by_conformal_coefficient[coefficient],
            rel=1.0e-9,
            abs=1.0e-9,
        )


def test_strict_receipts_resources_and_integrity_fail_closed() -> None:
    provider = _provider((0.1, 0.02, 0.03, -0.01))
    receipt = (
        GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE
        .provider_resource_receipt(provider)
    )
    assert receipt == {
        "learned_parameter_count": 4,
        "logical_macs_per_token_upper_bound": 8,
        "prepared_float_scalar_count": 6,
        "derived_constant_float_count": 2,
        "runtime_state_float_count_per_sequence": 2,
        "nonlinear_scalar_ops_per_token_upper_bound": 5,
        "linear_accumulator_scalar_ops_per_token_upper_bound": 4,
        "zero_denominator_comparisons_per_token_upper_bound": 1,
        "parent_decoder_invocations_per_token": 1,
    }
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE.validate_resource_envelope(
        resources=receipt,
        residual_width=provider.width,
    )
    assert provider.conformal_coefficient_order == (
        "shared_real",
        "shared_imag",
        "contrast_real",
        "contrast_imag",
    )

    record = _record(
        "one",
        "family",
        delta=0.1,
        jacobian=(1.0, 2.0, 3.0, 4.0),
    )
    tampered = record.to_dict()
    tampered["jacobian_by_conformal_coefficient"] = (
        9.0,
        9.0,
        9.0,
        9.0,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_gemma_iterative_conformal_route_fold(
            [tampered],
            held_family_id="held",
        )

    provider.parent_h4.lag_kernel[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        provider.validate_integrity()
