from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
    ChartConditionedAttentionTransportRouter,
    ChartConditionedRouterConfig,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    attention_transport_resource_accounting,
    evaluate_chart_conditioned_attention_transport_router,
)
from fisher_graph.gemma3_attention_transport_sparse_v2_global_router import (
    GLOBAL_SUPPORT_ROUTER_SEMANTICS,
    GlobalSupportAttentionTransportRouter,
    evaluate_global_support_attention_transport_router,
    fit_global_support_attention_transport_router,
    global_support_attention_transport_router_from_state_dict,
)


def _fixture() -> tuple[
    FisherAttentionTransportGeneratorBank,
    AttentionTransportRowBatch,
    FrozenAttentionChartAtlas,
]:
    rows = 12
    hidden = 4
    heads = 2
    head_contexts = torch.zeros(rows, heads, hidden, dtype=torch.float64)
    head_contexts[0::2, 0, 0] = 1.0
    head_contexts[1::2, 1, 1] = 1.0
    head_operators = torch.zeros(heads, hidden, hidden, dtype=torch.float64)
    head_operators[0, 0, 0] = 1.0
    head_operators[1, 1, 1] = 1.0
    outputs = torch.einsum("aoh,nah->no", head_operators, head_contexts)
    chart_inputs = outputs.clone()
    capture = AttentionTransportRowBatch(
        invocation_id="mutation.global-router",
        head_contexts=head_contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=chart_inputs,
        head_operators=head_operators,
        native_v_o_parameter_count=64,
    )
    left = torch.zeros(2, hidden, 1, dtype=torch.float64)
    right = torch.zeros_like(left)
    left[0, 0, 0] = right[0, 0, 0] = 1.0
    left[1, 1, 0] = right[1, 1, 0] = 1.0
    bank = FisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.ones(2, 1, dtype=torch.float64),
        right_factors=right,
        head_coefficients=torch.eye(2, dtype=torch.float64),
        head_ids=capture.head_ids,
    )
    centers = torch.zeros(2, hidden, dtype=torch.float64)
    centers[0, 0] = 1.0
    centers[1, 1] = 1.0
    bases = torch.zeros(2, hidden, 1, dtype=torch.float64)
    bases[:, 2, 0] = 1.0
    atlas = FrozenAttentionChartAtlas(
        centers=centers,
        bases=bases,
        temperature=0.01,
    )
    return bank, capture, atlas


def _opposed_chart_router() -> tuple[
    FisherAttentionTransportGeneratorBank,
    AttentionTransportRowBatch,
    GlobalSupportAttentionTransportRouter,
]:
    bank, capture, atlas = _fixture()
    feature_count = atlas.chart_rank + bank.generator_count
    router = GlobalSupportAttentionTransportRouter(
        invocation_id=capture.invocation_id,
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas,
        selection_weight=torch.zeros(
            atlas.chart_count,
            bank.generator_count,
            feature_count,
            dtype=torch.float64,
        ),
        selection_bias=torch.tensor(
            [[20.0, -20.0], [-18.0, 18.0]],
            dtype=torch.float64,
        ),
        amplitude_weight=torch.zeros(
            atlas.chart_count,
            bank.generator_count,
            feature_count,
            dtype=torch.float64,
        ),
        amplitude_bias=torch.tensor(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=torch.float64,
        ),
        active_count=1,
    )
    return bank, capture, router


def test_membership_mix_precedes_one_global_topk_and_avoids_chart_union() -> None:
    bank, _, router = _opposed_chart_router()
    midpoint = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
    context = torch.zeros(1, bank.generator_count, dtype=torch.float64)

    execution = router.execute(midpoint, context)
    per_chart_indices = torch.topk(router.selection_bias, k=1, dim=-1).indices

    assert per_chart_indices.unique().numel() == 2
    assert torch.allclose(
        execution.chart_memberships,
        torch.full((1, 2), 0.5, dtype=torch.float64),
    )
    assert torch.equal(execution.active_mask[:, 0], execution.active_mask[:, 1])
    assert int(execution.active_mask.any(dim=1).sum()) == 1
    assert int((execution.generator_coefficients != 0.0).sum()) <= 1
    assert torch.allclose(
        execution.sparse_selection_probabilities.sum(dim=-1),
        torch.ones(1, 2, dtype=torch.float64),
    )
    assert torch.allclose(
        execution.generator_coefficients,
        torch.tensor([[4.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        torch.einsum(
            "nc,nck->nk",
            execution.chart_memberships,
            execution.sparse_selection_probabilities
            * execution.signed_amplitudes,
        ),
        execution.generator_coefficients,
    )


def test_existing_resource_accounting_accepts_the_global_router_subclass() -> None:
    bank, capture, router = _opposed_chart_router()
    accounting = attention_transport_resource_accounting(
        bank,
        (capture,),
        (router,),
    )
    exact_router_parameters = sum(
        value.numel()
        for value in (
            router.selection_weight,
            router.selection_bias,
            router.amplitude_weight,
            router.amplitude_bias,
        )
    )

    assert router.stored_parameter_count == exact_router_parameters
    assert accounting.router_parameters == exact_router_parameters


def test_one_step_fit_and_ridge_free_core_evaluation_are_finite() -> None:
    bank, capture, atlas = _fixture()
    fit = fit_global_support_attention_transport_router(
        bank,
        (capture,),
        atlas,
        config=ChartConditionedRouterConfig(
            active_count=1,
            steps=1,
            learning_rate=1.0e-3,
            oracle_ridge=1.0e-10,
        ),
    )

    assert isinstance(fit.router, ChartConditionedAttentionTransportRouter)
    assert math.isfinite(fit.router_output_metrics.fisher_loss)
    assert math.isfinite(fit.oracle_output_metrics.fisher_loss)
    evaluation = evaluate_global_support_attention_transport_router(
        bank,
        fit.router,
        (capture,),
        euclidean_floor=1.0e-3,
    )
    core_evaluation = evaluate_chart_conditioned_attention_transport_router(
        bank,
        fit.router,
        (capture,),
        euclidean_floor=1.0e-3,
    )

    assert evaluation.relative_fisher_loss == core_evaluation.relative_fisher_loss
    assert (
        evaluation.oracle_router_fisher_gap
        == core_evaluation.oracle_router_fisher_gap
    )
    assert torch.equal(
        evaluation.generator_usage,
        core_evaluation.generator_usage,
    )
    assert evaluation.oracle_coefficient_ridge == 0.0
    assert math.isfinite(evaluation.relative_fisher_loss)
    assert math.isfinite(evaluation.oracle_relative_fisher_loss)
    assert math.isfinite(evaluation.oracle_router_fisher_gap)
    assert evaluation.active_count_minimum == 1
    assert evaluation.active_count_maximum == 1
    assert evaluation.effective_support_count_minimum == 1
    assert evaluation.effective_support_count_maximum == 1
    assert evaluation.sparse_probability_renormalization_max_error < 1.0e-12
    assert torch.allclose(
        evaluation.generator_usage.sum(),
        torch.ones((), dtype=torch.float64),
    )


def test_state_is_frozen_hashable_and_bound_to_bank_and_atlas() -> None:
    bank, _, router = _opposed_chart_router()
    legacy_equation_router = ChartConditionedAttentionTransportRouter(
        invocation_id=router.invocation_id,
        generator_bank_sha256=router.generator_bank_sha256,
        atlas=router.atlas,
        selection_weight=router.selection_weight,
        selection_bias=router.selection_bias,
        amplitude_weight=router.amplitude_weight,
        amplitude_bias=router.amplitude_bias,
        active_count=router.active_count,
    )
    state = router.state_dict()
    reloaded = global_support_attention_transport_router_from_state_dict(
        state,
        atlas=router.atlas,
        bank=bank,
    )

    assert state["support_semantics"] == GLOBAL_SUPPORT_ROUTER_SEMANTICS
    assert state["contains_oracle_outputs_or_coefficients"] is False
    assert state["contains_teacher_outputs_or_gradients"] is False
    assert router.artifact_sha256 != legacy_equation_router.artifact_sha256
    assert reloaded == router
    assert hash(reloaded) == hash(router)
    assert {router: "bound"}[reloaded] == "bound"
    assert reloaded.artifact_sha256 == router.artifact_sha256
    with pytest.raises(FrozenInstanceError):
        reloaded.active_count = 2  # type: ignore[misc]

    drifted = dict(state)
    drifted["generator_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        global_support_attention_transport_router_from_state_dict(
            drifted,
            atlas=router.atlas,
            bank=bank,
        )

    wrong_width_atlas = FrozenAttentionChartAtlas(
        centers=torch.zeros(2, 3, dtype=torch.float64),
        bases=torch.tensor(
            [[[0.0], [0.0], [1.0]], [[0.0], [0.0], [1.0]]],
            dtype=torch.float64,
        ),
        temperature=1.0,
    )
    wrong_width_state = dict(state)
    wrong_width_state["atlas_sha256"] = wrong_width_atlas.artifact_sha256
    with pytest.raises(ValueError, match="parent dimensions"):
        global_support_attention_transport_router_from_state_dict(
            wrong_width_state,
            atlas=wrong_width_atlas,
            bank=bank,
        )

    feature_count = router.atlas.chart_rank + 3
    wrong_inventory_router = GlobalSupportAttentionTransportRouter(
        invocation_id=router.invocation_id,
        generator_bank_sha256=bank.artifact_sha256,
        atlas=router.atlas,
        selection_weight=torch.zeros(2, 3, feature_count, dtype=torch.float64),
        selection_bias=torch.zeros(2, 3, dtype=torch.float64),
        amplitude_weight=torch.zeros(2, 3, feature_count, dtype=torch.float64),
        amplitude_bias=torch.ones(2, 3, dtype=torch.float64),
        active_count=1,
    )
    with pytest.raises(ValueError, match="generator inventory"):
        global_support_attention_transport_router_from_state_dict(
            wrong_inventory_router.state_dict(),
            atlas=router.atlas,
            bank=bank,
        )


def test_in_place_parameter_or_atlas_mutation_is_rejected() -> None:
    bank, _, router = _opposed_chart_router()
    router.selection_bias[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        router.execute(
            torch.zeros(1, bank.hidden_width, dtype=torch.float64),
            torch.zeros(1, bank.generator_count, dtype=torch.float64),
        )

    _, _, fresh = _opposed_chart_router()
    fresh.atlas.centers[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="atlas changed"):
        fresh.execute(
            torch.zeros(1, bank.hidden_width, dtype=torch.float64),
            torch.zeros(1, bank.generator_count, dtype=torch.float64),
        )
