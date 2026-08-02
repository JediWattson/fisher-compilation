from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_state_experts import (
    GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE,
    GemmaCausalTop2StateExpertsH4Provider,
    GemmaIterativeStateExpertsFitRecord,
    GemmaIterativeStateExpertsFoldFit,
    build_gemma_iterative_state_experts_fit_record,
    fit_gemma_iterative_state_experts_fold,
)
from fisher_graph.gemma3_l3_l4_iterative_state_router import (
    ROUTE_OPERATOR_NORM_BOUND,
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
                ],
                [
                    [0.5, 0.0, 0.0],
                    [0.0, -0.25, 0.0],
                ],
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


def _source_modes() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.5],
                [1.0, -0.5],
                [0.5, 1.0],
            ]
        ],
        dtype=torch.float64,
    )


def _prefix(
    source_modes: torch.Tensor | None = None,
    *,
    positions: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> Gemma3L3L4OnePassPrefix:
    source = _source_modes() if source_modes is None else source_modes
    batch, length, _rank = source.shape
    if positions is None:
        positions = torch.arange(length, dtype=torch.int64).expand(
            batch,
            -1,
        )
    valid = torch.ones((batch, length), dtype=torch.bool)
    if active is None:
        active = valid.clone()
    return Gemma3L3L4OnePassPrefix(
        source_modes=source,
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
    coefficients: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> GemmaIterativeStateExpertsFoldFit:
    tensor = torch.tensor(coefficients, dtype=torch.float64).reshape(2, 2, 2)
    norms = tuple(
        float(torch.linalg.svdvals(tensor[index]).max())
        for index in range(2)
    )
    return GemmaIterativeStateExpertsFoldFit(
        held_family_id="held",
        train_example_ids=("train",),
        train_family_ids=("family",),
        train_fit_record_sha256s=(_sha(8),),
        coefficients_by_expert_route_edge=coefficients,
        unsupported_expert_route_edge_indices=(),
        active_row_count=6,
        active_row_count_by_expert=(2, 4),
        supported_route_edge_count_by_expert=(4, 4),
        weighted_column_norm_by_expert_route_edge=(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        weighted_design_rank=8,
        weighted_design_rank_by_expert=(4, 4),
        normal_condition_number=1.0,
        pre_projection_operator_norm_by_expert=norms,
        post_projection_operator_norm_by_expert=norms,
        trust_projection_applied_by_expert=(False, False),
        trust_projection_applied=False,
        linearized_rmse_before=1.0,
        linearized_rmse_after=0.5,
    )


def _provider(
    coefficients: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> GemmaCausalTop2StateExpertsH4Provider:
    return GemmaCausalTop2StateExpertsH4Provider(
        parent_h4=_parent(),
        parent_artifact_sha256=_sha(9),
        fold_fit=_fold(coefficients),
    )


def _record(
    example: str,
    family: str,
    *,
    delta: float,
    jacobian: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> GemmaIterativeStateExpertsFitRecord:
    support = (
        any(abs(value) > 1.0e-12 for value in jacobian[:4]),
        any(abs(value) > 1.0e-12 for value in jacobian[4:]),
    )
    indices, norms = top2_lag_b_output_modes(_parent())
    return GemmaIterativeStateExpertsFitRecord(
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
        negative_gated_modal_sha256=_sha(17),
        nonnegative_gated_modal_sha256=_sha(18),
        supervised_tokens=2,
        parent_signed_delta_nll_per_token=delta,
        jacobian_by_expert_route_edge=jacobian,
        active_row_count=2,
        active_row_count_by_expert=(1, 1),
        active_expert_mask=(True, True),
        jacobian_support_by_expert=support,
        top_mode_indices=indices,
        top_mode_norms=norms,
        balance_feature_std=0.5,
        top2_modal_energy_fraction=0.99,
    )


def _balance(
    prefix: Gemma3L3L4OnePassPrefix,
    parent_modal: torch.Tensor,
) -> torch.Tensor:
    indices, norms = top2_lag_b_output_modes(_parent())
    zeros = torch.zeros(parent_modal.shape[0], dtype=parent_modal.dtype)
    balance, _numerator, _denominator = _balance_feature(
        prefix=prefix,
        parent_modal=parent_modal,
        top_mode_indices=indices,
        top_mode_norms=norms,
        initial_numerator=zeros,
        initial_denominator=zeros,
    )
    return balance


def test_zero_experts_exactly_reproduce_parent_and_resources_are_honest() -> None:
    provider = _provider((0.0,) * 8)
    prefix = _prefix()
    realized = torch.randn(1, 6, 3, dtype=torch.float64)
    assert torch.equal(
        provider.correction(prefix, realized),
        provider.parent_h4.correction(prefix, realized),
    )
    assert provider.marginal_learned_float_scalar_count == 8
    assert provider.marginal_derived_prepared_float_scalar_count == 2
    assert provider.marginal_prepared_float_scalar_count == 10
    assert provider.marginal_logical_macs_per_token_upper_bound == 6
    assert provider.nonlinear_scalar_ops_per_token_upper_bound == 6
    assert provider.runtime_state_float_scalars_per_sequence == 2
    assert provider.resource_receipt["experts_evaluated_per_active_row"] == 1
    assert provider.resource_receipt["parent_decoder_invocations_per_token"] == 1


def test_both_regimes_execute_and_only_selected_expert_affects_each_row() -> None:
    both = _provider((0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0))
    negative_only = _provider((0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0))
    nonnegative_only = _provider((0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0))
    prefix = _prefix()
    realized = torch.zeros(1, 6, 3, dtype=torch.float64)
    parent_modal = both.parent_h4.modal_correction(prefix, realized)
    balance = _balance(prefix, parent_modal)
    negative = balance < 0
    nonnegative = ~negative
    assert bool(negative.any())
    assert bool(nonnegative.any())

    both_result = both.correction(prefix, realized)
    negative_result = negative_only.correction(prefix, realized)
    nonnegative_result = nonnegative_only.correction(prefix, realized)
    assert torch.equal(both_result[negative], negative_result[negative])
    assert torch.equal(both_result[nonnegative], nonnegative_result[nonnegative])
    assert bool((both_result[negative] != nonnegative_result[negative]).any())
    assert bool((both_result[nonnegative] != negative_result[nonnegative]).any())


def test_future_rows_do_not_change_earlier_expert_dispatch_or_output() -> None:
    provider = _provider((0.1, 0.05, 0.0, 0.1, -0.1, 0.0, 0.05, 0.1))
    prefix = _prefix()
    realized = torch.zeros(1, 6, 3, dtype=torch.float64)
    expected = provider.correction(prefix, realized)
    changed_source = prefix.source_modes.clone()
    changed_source[:, 4:] = 1000.0
    actual = provider.correction(
        _prefix(source_modes=changed_source),
        realized,
    )
    assert torch.equal(actual[:, :4], expected[:, :4])


def test_upstream_multilag_parent_modal_chunks_preserve_exact_carry() -> None:
    provider = _provider((0.1, 0.05, 0.0, 0.1, -0.1, 0.0, 0.05, 0.1))
    prefix = _prefix()
    realized = torch.zeros(1, 6, 3, dtype=torch.float64)
    parent_modal = provider.parent_h4.modal_correction(prefix, realized)
    full, full_state = provider.correction_from_parent_modal_with_state(
        prefix,
        parent_modal,
        provider.initial_state(1, device="cpu"),
    )

    first_prefix = _slice_prefix(prefix, 0, 2)
    second_prefix = _slice_prefix(prefix, 2, 6)
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

    # Recomputing the lag-2 parent from the second chunk alone is not the API:
    # it drops the preceding source row and produces a different parent modal.
    wrong_second_modal = provider.parent_h4.modal_correction(
        second_prefix,
        realized[:, 2:],
    )
    assert not torch.equal(wrong_second_modal, parent_modal[:, 2:])


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
        supervised_tokens=6,
        source_summed_nll=4.0,
        candidate_summed_nll=4.5,
        source_to_candidate_summed_kl=0.2,
        top1_matches=3,
        source_logits_sha256=_sha(22),
        candidate_logits_sha256=_sha(23),
        targets_sha256=_sha(24),
    )


def test_fit_record_jacobian_matches_all_eight_finite_differences() -> None:
    parent = _parent()
    prefix = _prefix()
    candidate_h4 = torch.zeros(1, 6, 3, dtype=torch.float64)
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
                [0.1, 0.8, -0.3],
            ]
        ],
        dtype=torch.float64,
    )
    record = build_gemma_iterative_state_experts_fit_record(
        example=_Example(),
        parent_execution=execution,
        gradient=gradient,
        parent_h4=parent,
        parent_observation=_observation(),
    )
    assert record.active_expert_mask == (True, True)
    assert sum(record.active_row_count_by_expert) == 6
    assert record.negative_gated_modal_sha256 != (
        record.nonnegative_gated_modal_sha256
    )

    epsilon = 1.0e-6
    for edge in range(8):
        plus = [0.0] * 8
        minus = [0.0] * 8
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
                gradient * (plus_correction - minus_correction)
            ).sum()
            / (2.0 * epsilon * _observation().supervised_tokens)
        )
        assert finite_difference == pytest.approx(
            record.jacobian_by_expert_route_edge[edge],
            rel=1.0e-9,
            abs=1.0e-9,
        )


def test_joint_ridge_projects_experts_independently_and_replays() -> None:
    records = []
    for family_index in range(7):
        family = f"family-{family_index}"
        for example_index in range(2):
            edge = (2 * family_index + example_index) % 8
            scale = 1.0 if edge < 4 else 100.0
            jacobian = [0.0] * 8
            jacobian[edge] = scale
            records.append(
                _record(
                    f"{family}-{example_index}",
                    family,
                    delta=-10.0,
                    jacobian=tuple(jacobian),
                )
            )
    fit = fit_gemma_iterative_state_experts_fold(
        records,
        held_family_id="held",
    )
    replay = fit_gemma_iterative_state_experts_fold(
        [record.to_dict() for record in reversed(records)],
        held_family_id="held",
    )
    assert replay.to_dict() == fit.to_dict()
    assert fit.trust_projection_applied_by_expert == (True, False)
    assert fit.trust_projection_applied is True
    assert fit.weighted_design_rank == 8
    assert fit.weighted_design_rank_by_expert == (4, 4)
    assert all(
        value <= ROUTE_OPERATOR_NORM_BOUND
        for value in fit.post_projection_operator_norm_by_expert
    )

    with pytest.raises(ValueError, match="held family leaked"):
        fit_gemma_iterative_state_experts_fold(
            [
                _record(
                    "leaked",
                    "held",
                    delta=1.0,
                    jacobian=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            ],
            held_family_id="held",
        )


def test_projection_keeps_unsupported_edges_exactly_zero() -> None:
    records = []
    for family_index in range(7):
        family = f"family-{family_index}"
        for example_index in range(2):
            jacobian = [0.0] * 8
            jacobian[0 if example_index == 0 else 4] = 1.0
            records.append(
                _record(
                    f"{family}-{example_index}",
                    family,
                    delta=-100.0,
                    jacobian=tuple(jacobian),
                )
            )
    fit = fit_gemma_iterative_state_experts_fold(
        records,
        held_family_id="held",
    )
    assert fit.supported_route_edge_count_by_expert == (1, 1)
    assert fit.unsupported_expert_route_edge_indices == (1, 2, 3, 5, 6, 7)
    assert all(
        fit.coefficients_by_expert_route_edge[index] == 0.0
        for index in fit.unsupported_expert_route_edge_indices
    )
    assert fit.trust_projection_applied_by_expert == (True, True)


def test_exact_operator_bound_receipt_and_serialized_tamper_fail_closed() -> None:
    boundary_fit = _fold(
        (ROUTE_OPERATOR_NORM_BOUND, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert boundary_fit.pre_projection_operator_norm_by_expert[0] == (
        ROUTE_OPERATOR_NORM_BOUND
    )
    assert boundary_fit.post_projection_operator_norm_by_expert[0] == (
        ROUTE_OPERATOR_NORM_BOUND
    )
    assert boundary_fit.trust_projection_applied_by_expert == (False, False)
    assert boundary_fit.trust_projection_applied is False

    records = []
    for family_index in range(7):
        for example_index in range(2):
            edge = (2 * family_index + example_index) % 8
            jacobian = [0.0] * 8
            jacobian[edge] = 1.0
            records.append(
                _record(
                    f"e-{family_index}-{example_index}",
                    f"f-{family_index}",
                    delta=-10.0,
                    jacobian=tuple(jacobian),
                )
            )
    fit = fit_gemma_iterative_state_experts_fold(
        records,
        held_family_id="held",
    )
    with pytest.raises(ValueError, match="norm receipt differs"):
        replace(
            fit,
            post_projection_operator_norm_by_expert=(0.0, 0.0),
        )

    tampered = records[0].to_dict()
    tampered["jacobian_by_expert_route_edge"] = (9.0,) * 8
    tampered["jacobian_support_by_expert"] = (True, True)
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_gemma_iterative_state_experts_fold(
            [tampered, *records[1:]],
            held_family_id="held",
        )


def test_campaign_recipe_resources_and_parent_tensor_tamper_are_bound() -> None:
    provider = _provider((0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0))
    receipt = (
        GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE.provider_resource_receipt(
            provider
        )
    )
    assert receipt == {
        "learned_parameter_count": 8,
        "logical_macs_per_token_upper_bound": 6,
        "derived_constant_float_count": 2,
        "runtime_state_float_count_per_sequence": 2,
        "nonlinear_scalar_ops_per_token_upper_bound": 6,
    }
    GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE.validate_resource_envelope(
        resources=receipt,
        residual_width=provider.width,
    )
    provider.parent_h4.lag_kernel[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        provider.validate_integrity()
