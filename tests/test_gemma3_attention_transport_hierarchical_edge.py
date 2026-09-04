from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
    ChartConditionedAttentionTransportRouter,
    FrozenAttentionChartAtlas,
    FisherAttentionTransportGeneratorBank,
    evaluate_chart_conditioned_attention_transport_router,
)
from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
    GLOBAL_SUPPORT_SEMANTICS,
    HIERARCHY_SEMANTICS,
    SIGNED_ROUTER_FEATURE_SEMANTICS,
    AttentionTransportHierarchyLayout,
    HierarchicalEdgeGlobalSupportRouter,
    HierarchicalFisherAttentionTransportGeneratorBank,
    HierarchicalSparseAttentionTransportFitConfig,
    attention_transport_hierarchy_layout_from_state_dict,
    fit_hierarchical_sparse_attention_transport_generator_bank,
    hierarchical_edge_global_support_router_from_state_dict,
    hierarchical_fisher_attention_transport_generator_bank_from_state_dict,
)
from fisher_graph.gemma3_attention_transport_sparse_v2 import (
    SparseAwareAttentionTransportFitConfig,
)


def _layout() -> AttentionTransportHierarchyLayout:
    # Intentionally lexical: the constructor must use numeric mutation suffixes.
    lexical = sorted(f"mutation.{2 * index + 1}" for index in range(18))
    return AttentionTransportHierarchyLayout.gemma3_270m_attention(lexical)


def _shared_bank(
    *,
    generator_count: int = 2,
) -> FisherAttentionTransportGeneratorBank:
    layout = _layout()
    hidden = generator_count
    left = torch.zeros(generator_count, hidden, 1, dtype=torch.float64)
    right = torch.zeros_like(left)
    for generator in range(generator_count):
        left[generator, generator, 0] = 1.0
        right[generator, generator, 0] = 1.0
    coefficients = torch.ones(
        len(layout.invocation_ids), generator_count, dtype=torch.float64
    )
    head_ids = tuple(
        f"{invocation_id}/query-head.0"
        for invocation_id in layout.invocation_ids
    )
    return FisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.arange(
            1, generator_count + 1, dtype=torch.float64
        )[:, None],
        right_factors=right,
        head_coefficients=coefficients,
        head_ids=head_ids,
    )


def _hierarchical_bank(
    *,
    generator_count: int = 2,
    block: torch.Tensor | None = None,
    site: torch.Tensor | None = None,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    shared = _shared_bank(generator_count=generator_count)
    layout = _layout()
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=shared.left_factors,
        singular_scales=shared.singular_scales,
        right_factors=shared.right_factors,
        head_coefficients=shared.head_coefficients,
        head_ids=shared.head_ids,
        hierarchy=layout,
        block_scale_contrasts=(
            torch.zeros(2, generator_count, 1, dtype=torch.float64)
            if block is None
            else block
        ),
        site_scale_contrasts=(
            torch.zeros(5, generator_count, 1, dtype=torch.float64)
            if site is None
            else site
        ),
    )


def _atlas(*, hidden: int, charts: int = 1) -> FrozenAttentionChartAtlas:
    centers = torch.zeros(charts, hidden, dtype=torch.float64)
    if charts == 2:
        centers[0, 0] = 1.0
        centers[1, min(1, hidden - 1)] = 1.0
    bases = torch.zeros(charts, hidden, 1, dtype=torch.float64)
    bases[:, -1, 0] = 1.0
    return FrozenAttentionChartAtlas(
        centers=centers,
        bases=bases,
        temperature=0.01,
    )


def _router(
    bank: HierarchicalFisherAttentionTransportGeneratorBank,
    *,
    atlas: FrozenAttentionChartAtlas | None = None,
    active_count: int = 1,
    selection_bias: torch.Tensor | None = None,
    amplitude_bias: torch.Tensor | None = None,
    selection_weight: torch.Tensor | None = None,
) -> HierarchicalEdgeGlobalSupportRouter:
    atlas_value = atlas or _atlas(hidden=bank.hidden_width)
    charts = atlas_value.chart_count
    generators = bank.generator_count
    feature_count = atlas_value.chart_rank + 2 * generators
    return HierarchicalEdgeGlobalSupportRouter(
        invocation_id="mutation.1",
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas_value,
        selection_weight=(
            torch.zeros(
                charts, generators, feature_count, dtype=torch.float64
            )
            if selection_weight is None
            else selection_weight
        ),
        selection_bias=(
            torch.zeros(charts, generators, dtype=torch.float64)
            if selection_bias is None
            else selection_bias
        ),
        amplitude_weight=torch.zeros(
            charts, generators, feature_count, dtype=torch.float64
        ),
        amplitude_bias=(
            torch.ones(charts, generators, dtype=torch.float64)
            if amplitude_bias is None
            else amplitude_bias
        ),
        active_count=active_count,
    )


def test_numeric_suffix_mapping_uses_three_macroblocks_and_six_global_slots() -> None:
    layout = _layout()

    assert layout.invocation_ids[:7] == (
        "mutation.1",
        "mutation.3",
        "mutation.5",
        "mutation.7",
        "mutation.9",
        "mutation.11",
        "mutation.13",
    )
    assert layout.coordinates("mutation.1") == (0, 0)
    assert layout.coordinates("mutation.11") == (0, 5)
    assert layout.coordinates("mutation.13") == (1, 0)
    assert layout.coordinates("mutation.23") == (1, 5)
    assert layout.coordinates("mutation.25") == (2, 0)
    assert layout.coordinates("mutation.35") == (2, 5)
    with pytest.raises(ValueError, match="positive and odd"):
        layout.coordinates("mutation.2")
    with pytest.raises(ValueError, match="absent"):
        layout.coordinates("mutation.37")


def test_zero_contrasts_exactly_reproduce_the_shared_generator_bank() -> None:
    shared = _shared_bank()
    hierarchical = _hierarchical_bank()
    contexts = torch.tensor(
        [[[1.5, -2.0]], [[-0.25, 3.0]]], dtype=torch.float64
    )

    for invocation_id in ("mutation.1", "mutation.17", "mutation.35"):
        assert torch.equal(
            hierarchical.effective_signed_diagonal(invocation_id),
            shared.singular_scales,
        )
        assert torch.equal(
            hierarchical.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
            shared.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
        )


def test_helmert_contrasts_are_identifiable_and_zero_sum() -> None:
    layout = _layout()
    generator = torch.Generator().manual_seed(17)
    block = torch.randn(2, 2, 1, generator=generator, dtype=torch.float64)
    site = torch.randn(5, 2, 1, generator=generator, dtype=torch.float64)
    bank = _hierarchical_bank(block=block, site=site)

    assert torch.allclose(
        layout.block_contrast.sum(dim=0),
        torch.zeros(2, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert torch.allclose(
        layout.site_contrast.sum(dim=0),
        torch.zeros(5, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert torch.allclose(
        layout.block_contrast.T @ layout.block_contrast,
        torch.eye(2, dtype=torch.float64),
    )
    assert torch.allclose(
        layout.site_contrast.T @ layout.site_contrast,
        torch.eye(5, dtype=torch.float64),
    )
    all_effective = torch.stack(
        tuple(
            bank.effective_signed_diagonal(invocation_id)
            for invocation_id in layout.invocation_ids
        )
    )
    assert torch.allclose(
        all_effective.mean(dim=0),
        bank.singular_scales,
        atol=1.0e-15,
        rtol=0.0,
    )


def test_effective_diagonals_are_signed_and_rho_pi_features_are_sign_sensitive() -> None:
    block = torch.zeros(2, 2, 1, dtype=torch.float64)
    block[0, 0, 0] = -4.0
    bank = _hierarchical_bank(block=block)
    diagonal = bank.effective_signed_diagonal("mutation.1")
    expected_first = 1.0 - 4.0 / math.sqrt(2.0)

    assert diagonal[0, 0] == pytest.approx(expected_first)
    assert diagonal[0, 0] < 0.0
    positive = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    negative = -positive
    positive_features = bank.router_context_features_from_head_contexts(
        invocation_id="mutation.1", head_contexts=positive
    )
    negative_features = bank.router_context_features_from_head_contexts(
        invocation_id="mutation.1", head_contexts=negative
    )

    assert positive_features.shape == (1, 2 * bank.generator_count)
    assert torch.equal(
        positive_features[:, : bank.generator_count],
        negative_features[:, : bank.generator_count],
    )
    assert positive_features[0, bank.generator_count] < 0.0
    assert negative_features[0, bank.generator_count] > 0.0
    assert torch.allclose(
        positive_features[:, bank.generator_count :],
        -negative_features[:, bank.generator_count :],
    )


def test_parameter_counts_close_exactly_for_fixture_and_production_shapes() -> None:
    bank = _hierarchical_bank()
    router = _router(bank)

    shared_fixture = 2 * (2 * 2 * 1 + 1) + 18 * 2
    hierarchy_fixture = (2 + 5) * 2 * 1
    assert bank.hierarchy_parameter_count == hierarchy_fixture
    assert bank.stored_parameter_count == shared_fixture + hierarchy_fixture
    assert router.stored_parameter_count == 2 * 1 * 2 * (1 + 2 * 2 + 1)

    shared_production = 8 * (2 * 640 * 8 + 8) + (18 * 4) * 8
    hierarchy_production = (2 + 5) * 8 * 8
    per_router = 2 * 4 * 8 * (8 + 2 * 8 + 1)
    assert shared_production == 82_560
    assert hierarchy_production == 448
    assert shared_production + hierarchy_production == 83_008
    assert per_router == 1_600
    assert 18 * per_router == 28_800


def test_planted_crossed_hierarchy_fit_strictly_improves_and_never_regresses() -> None:
    layout = _layout()
    planted_block = torch.tensor(
        [[0.35, -0.25], [0.10, -0.15]], dtype=torch.float64
    )
    planted_site = torch.tensor(
        [
            [0.12, -0.08],
            [0.08, -0.06],
            [0.04, -0.03],
            [0.02, -0.015],
            [0.01, -0.008],
        ],
        dtype=torch.float64,
    )
    contexts = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[1.0, -1.0]]],
        dtype=torch.float64,
    )
    captures = []
    for invocation_id in layout.invocation_ids:
        block_index, site_index = layout.coordinates(invocation_id)
        signed_diagonal = (
            torch.ones(2, dtype=torch.float64)
            + layout.block_contrast[block_index] @ planted_block
            + layout.site_contrast[site_index] @ planted_site
        )
        operator = torch.diag(signed_diagonal)[None]
        outputs = torch.einsum("aoh,nah->no", operator, contexts)
        captures.append(
            AttentionTransportRowBatch(
                invocation_id=invocation_id,
                head_contexts=contexts,
                operator_outputs=outputs,
                operator_output_score_gradients=torch.ones_like(outputs),
                chart_inputs=contexts[:, 0, :].clone(),
                head_operators=operator,
                native_v_o_parameter_count=8,
            )
        )
    shared = SparseAwareAttentionTransportFitConfig(
        generator_count=1,
        generator_rank=2,
        active_count=1,
        dense_warm_start_steps=0,
        alternating_rounds=1,
        steps_per_round=1,
        learning_rate=1.0e-2,
        euclidean_floor=0.1,
        oracle_coefficient_ridge=1.0e-8,
        dense_head_coefficient_ridge=1.0e-8,
        gradient_clip_norm=10.0,
        gauge_floor=1.0e-12,
    )
    fit = fit_hierarchical_sparse_attention_transport_generator_bank(
        tuple(captures),
        config=HierarchicalSparseAttentionTransportFitConfig(shared=shared),
    )

    history = fit.fit_round_relative_fisher_losses
    assert len(history) == 5
    assert history[-1] < history[0]
    assert all(right <= left + 1.0e-12 for left, right in zip(history, history[1:]))
    assert fit.final_fit_oracle.relative_fisher_loss <= (
        fit.initial_fit_oracle.relative_fisher_loss
    )
    assert bool((fit.bank.block_scale_contrasts.abs() > 1.0e-6).any())
    assert bool((fit.bank.site_scale_contrasts.abs() > 1.0e-6).any())


def test_membership_mixing_precedes_one_global_top2() -> None:
    bank = _hierarchical_bank(generator_count=3)
    atlas = _atlas(hidden=3, charts=2)
    feature_count = atlas.chart_rank + 2 * bank.generator_count
    selection_bias = torch.tensor(
        [[12.0, 10.0, -12.0], [-10.0, 9.0, 11.0]],
        dtype=torch.float64,
    )
    router = _router(
        bank,
        atlas=atlas,
        active_count=2,
        selection_bias=selection_bias,
        amplitude_bias=torch.ones(2, 3, dtype=torch.float64),
        selection_weight=torch.zeros(
            2, 3, feature_count, dtype=torch.float64
        ),
    )
    midpoint = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
    context = torch.zeros(1, 2 * bank.generator_count, dtype=torch.float64)
    execution = router.execute(midpoint, context)
    per_chart = torch.topk(selection_bias, k=2, dim=-1).indices.unique()

    assert per_chart.numel() == 3
    assert torch.allclose(
        execution.chart_memberships,
        torch.full((1, 2), 0.5, dtype=torch.float64),
    )
    assert torch.equal(execution.active_mask[:, 0], execution.active_mask[:, 1])
    assert int(execution.active_mask.any(dim=1).sum()) == 2
    assert int((execution.generator_coefficients != 0.0).sum()) == 2
    assert torch.allclose(
        execution.sparse_selection_probabilities.sum(dim=-1),
        torch.ones(1, 2, dtype=torch.float64),
    )


def test_signed_pi_feature_can_reverse_global_router_selection() -> None:
    bank = _hierarchical_bank()
    atlas = _atlas(hidden=2)
    feature_count = atlas.chart_rank + 2 * bank.generator_count
    weights = torch.zeros(1, 2, feature_count, dtype=torch.float64)
    pi0 = atlas.chart_rank + bank.generator_count
    weights[0, 0, pi0] = 8.0
    weights[0, 1, pi0] = -8.0
    router = _router(
        bank,
        atlas=atlas,
        selection_weight=weights,
        active_count=1,
    )
    contexts = torch.tensor(
        [[[1.0, 0.0]], [[-1.0, 0.0]]], dtype=torch.float64
    )
    features = bank.router_context_features_from_head_contexts(
        invocation_id="mutation.1", head_contexts=contexts
    )
    execution = router.execute(
        torch.zeros(2, 2, dtype=torch.float64), features
    )

    assert torch.equal(
        execution.active_mask[:, 0],
        torch.tensor([[True, False], [False, True]]),
    )


def test_existing_core_evaluator_accepts_hierarchical_bank_and_router() -> None:
    bank = _hierarchical_bank()
    router = _router(
        bank,
        selection_bias=torch.tensor([[20.0, -20.0]], dtype=torch.float64),
        amplitude_bias=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )
    contexts = torch.tensor(
        [[[1.0, 0.0]], [[-1.0, 0.0]], [[0.5, 0.0]], [[-0.5, 0.0]]],
        dtype=torch.float64,
    )
    head_operator = torch.zeros(1, 2, 2, dtype=torch.float64)
    head_operator[0, 0, 0] = 1.0
    outputs = torch.einsum("aoh,nah->no", head_operator, contexts)
    capture = AttentionTransportRowBatch(
        invocation_id="mutation.1",
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=torch.zeros_like(outputs),
        head_operators=head_operator,
        native_v_o_parameter_count=8,
    )

    assert isinstance(router, ChartConditionedAttentionTransportRouter)
    evaluation = evaluate_chart_conditioned_attention_transport_router(
        bank,
        router,
        (capture,),
        euclidean_floor=1.0e-3,
    )

    assert evaluation.relative_fisher_loss < 1.0e-12
    assert evaluation.oracle_coefficient_ridge == 0.0
    assert evaluation.active_count_minimum == 1
    assert evaluation.active_count_maximum == 1
    assert evaluation.effective_support_count_minimum == 1
    assert evaluation.effective_support_count_maximum == 1
    assert evaluation.sparse_probability_renormalization_max_error < 1.0e-12


def test_frozen_state_is_content_addressed_and_rejects_lineage_tampering() -> None:
    bank = _hierarchical_bank()
    router = _router(bank)
    layout_state = bank.hierarchy.state_dict()
    bank_state = bank.state_dict()
    router_state = router.state_dict()
    layout_reload = attention_transport_hierarchy_layout_from_state_dict(
        layout_state
    )
    bank_reload = hierarchical_fisher_attention_transport_generator_bank_from_state_dict(
        bank_state
    )
    router_reload = hierarchical_edge_global_support_router_from_state_dict(
        router_state,
        atlas=router.atlas,
        bank=bank_reload,
    )

    assert layout_reload == bank.hierarchy
    assert bank_reload == bank
    assert hash(bank_reload) == hash(bank)
    assert router_reload == router
    assert hash(router_reload) == hash(router)
    assert bank_state["hierarchy_semantics"] == HIERARCHY_SEMANTICS
    assert (
        bank_state["signed_router_feature_semantics"]
        == SIGNED_ROUTER_FEATURE_SEMANTICS
    )
    assert router_state["support_semantics"] == GLOBAL_SUPPORT_SEMANTICS
    assert bank_state["contains_oracle_outputs_or_coefficients"] is False
    assert router_state["contains_teacher_outputs_or_gradients"] is False
    with pytest.raises(FrozenInstanceError):
        bank.signed_feature_floor = 1.0  # type: ignore[misc]

    drifted_bank = dict(bank_state)
    drifted_bank["shared_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        hierarchical_fisher_attention_transport_generator_bank_from_state_dict(
            drifted_bank
        )

    drifted_router = dict(router_state)
    drifted_router["generator_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        hierarchical_edge_global_support_router_from_state_dict(
            drifted_router,
            atlas=router.atlas,
            bank=bank,
        )

    bank.block_scale_contrasts[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = bank.artifact_sha256
