from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
    FrozenAttentionChartAtlas,
    evaluate_chart_conditioned_attention_transport_router,
)
from fisher_graph.gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
    GlobalRankOneAttentionTransportFitConfig,
    GlobalRankOneFisherAttentionTransportGeneratorBank,
    _fixed_support_prediction,
    _global_rank_one_bank_with_factors,
    _untied_sum_difference_atoms,
    evaluate_global_rank_one_coupling_diagnostic,
    fit_global_rank_one_attention_transport_generator_bank,
    global_rank_one_fisher_attention_transport_generator_bank_from_state_dict,
)
from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
    AttentionTransportHierarchyLayout,
    HierarchicalEdgeGlobalSupportRouter,
    HierarchicalFisherAttentionTransportGeneratorBank,
    HierarchicalSparseAttentionTransportFitConfig,
)
from fisher_graph.gemma3_attention_transport_sparse_v2 import (
    SparseAwareAttentionTransportFitConfig,
)


def _layout() -> AttentionTransportHierarchyLayout:
    return AttentionTransportHierarchyLayout.gemma3_270m_attention(
        sorted(f"mutation.{2 * index + 1}" for index in range(18))
    )


def _hierarchical_bank(
    *, generator_count: int = 2, hidden_width: int | None = None
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    hidden = generator_count if hidden_width is None else hidden_width
    rank = 1
    left = torch.zeros(generator_count, hidden, rank, dtype=torch.float64)
    right = torch.zeros_like(left)
    for generator in range(generator_count):
        coordinate = min(generator, hidden - 1)
        left[generator, coordinate, 0] = 1.0
        right[generator, coordinate, 0] = 1.0
    layout = _layout()
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.ones(
            generator_count, rank, dtype=torch.float64
        ),
        right_factors=right,
        head_coefficients=torch.ones(
            len(layout.invocation_ids), generator_count, dtype=torch.float64
        ),
        head_ids=tuple(
            f"{invocation_id}/query-head.0"
            for invocation_id in layout.invocation_ids
        ),
        hierarchy=layout,
        block_scale_contrasts=torch.zeros(
            2, generator_count, rank, dtype=torch.float64
        ),
        site_scale_contrasts=torch.zeros(
            5, generator_count, rank, dtype=torch.float64
        ),
    )


def _rank_one_bank(
    *,
    generator_count: int = 2,
    hidden_width: int | None = None,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
) -> GlobalRankOneFisherAttentionTransportGeneratorBank:
    parent = _hierarchical_bank(
        generator_count=generator_count, hidden_width=hidden_width
    )
    hidden = parent.hidden_width
    return GlobalRankOneFisherAttentionTransportGeneratorBank(
        left_factors=parent.left_factors,
        singular_scales=parent.singular_scales,
        right_factors=parent.right_factors,
        head_coefficients=parent.head_coefficients,
        head_ids=parent.head_ids,
        hierarchy=parent.hierarchy,
        block_scale_contrasts=parent.block_scale_contrasts,
        site_scale_contrasts=parent.site_scale_contrasts,
        signed_feature_floor=parent.signed_feature_floor,
        global_residual_left_vectors=(
            torch.zeros(generator_count, hidden, dtype=torch.float64)
            if left is None
            else left
        ),
        global_residual_right_vectors=(
            torch.zeros(generator_count, hidden, dtype=torch.float64)
            if right is None
            else right
        ),
    )


def _atlas(*, hidden: int) -> FrozenAttentionChartAtlas:
    centers = torch.zeros(1, hidden, dtype=torch.float64)
    bases = torch.zeros(1, hidden, 1, dtype=torch.float64)
    bases[0, -1, 0] = 1.0
    return FrozenAttentionChartAtlas(
        centers=centers,
        bases=bases,
        temperature=0.01,
    )


def _router(
    bank: GlobalRankOneFisherAttentionTransportGeneratorBank,
    *,
    active_count: int,
    invocation_id: str = "mutation.1",
) -> HierarchicalEdgeGlobalSupportRouter:
    atlas = _atlas(hidden=bank.hidden_width)
    feature_count = atlas.chart_rank + 2 * bank.generator_count
    return HierarchicalEdgeGlobalSupportRouter(
        invocation_id=invocation_id,
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas,
        selection_weight=torch.zeros(
            1,
            bank.generator_count,
            feature_count,
            dtype=torch.float64,
        ),
        selection_bias=torch.arange(
            bank.generator_count, dtype=torch.float64
        )[None],
        amplitude_weight=torch.zeros(
            1,
            bank.generator_count,
            feature_count,
            dtype=torch.float64,
        ),
        amplitude_bias=torch.ones(
            1, bank.generator_count, dtype=torch.float64
        ),
        active_count=active_count,
    )


def test_rank_one_residual_is_active_only_at_the_three_global_attention_sites() -> None:
    left = torch.tensor([[1.0, 1.0], [0.5, -0.25]], dtype=torch.float64)
    right = torch.tensor([[1.0, -2.0], [0.75, 0.5]], dtype=torch.float64)
    bank = _rank_one_bank(left=left, right=right)
    parent = _hierarchical_bank()
    contexts = torch.tensor(
        [[[1.0, 2.0]], [[-0.5, 3.0]]], dtype=torch.float64
    )

    assert GLOBAL_ATTENTION_INVOCATION_IDS == (
        "mutation.11",
        "mutation.23",
        "mutation.35",
    )
    for invocation_id in bank.hierarchy.invocation_ids:
        actual = bank.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        baseline = parent.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        if invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
            assert not torch.equal(actual, baseline)
        else:
            # The local path must remain bit-exact, not merely close.
            assert torch.equal(actual, baseline)


def test_zero_rank_one_residual_exactly_reproduces_hierarchical_bank() -> None:
    bank = _rank_one_bank()
    parent = _hierarchical_bank()
    generator = torch.Generator().manual_seed(31)
    contexts = torch.randn(7, 1, 2, generator=generator, dtype=torch.float64)

    for invocation_id in bank.hierarchy.invocation_ids:
        assert torch.equal(
            bank.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
            parent.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
        )
        assert torch.equal(
            bank.router_context_features_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
            parent.router_context_features_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
        )


def test_rank_one_factor_gauge_is_balanced_and_state_is_canonical() -> None:
    left = torch.tensor([[-8.0, 0.0], [0.0, 0.5]], dtype=torch.float64)
    right = torch.tensor([[0.5, 0.0], [0.0, -2.0]], dtype=torch.float64)
    bank = _rank_one_bank(left=left, right=right)

    # Gauge fixing preserves each outer product, puts all scale in the left
    # vector, and gives the unit right vector a deterministic sign.
    assert torch.allclose(
        torch.einsum("ko,kh->koh", left, right),
        torch.einsum(
            "ko,kh->koh",
            bank.global_residual_left_vectors,
            bank.global_residual_right_vectors,
        ),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(bank.global_residual_right_vectors, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    right_pivots = torch.argmax(
        bank.global_residual_right_vectors.abs(), dim=-1
    )
    assert torch.all(
        bank.global_residual_right_vectors[
            torch.arange(2), right_pivots
        ]
        > 0.0
    )

    state = bank.state_dict()
    reloaded = global_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
        state
    )
    assert reloaded == bank
    assert hash(reloaded) == hash(bank)
    assert reloaded.artifact_sha256 == bank.artifact_sha256
    reloaded_state = reloaded.state_dict()
    assert set(reloaded_state) == set(state)
    for name, expected in state.items():
        actual = reloaded_state[name]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        elif isinstance(expected, dict):
            # The nested hierarchy has its own authenticated artifact hash.
            assert actual["artifact_sha256"] == expected["artifact_sha256"]
        else:
            assert actual == expected

    rescaled = _rank_one_bank(left=left * 4.0, right=right / 4.0)
    assert rescaled.artifact_sha256 == bank.artifact_sha256
    assert torch.equal(
        rescaled.global_residual_left_vectors,
        bank.global_residual_left_vectors,
    )
    assert torch.equal(
        rescaled.global_residual_right_vectors,
        bank.global_residual_right_vectors,
    )


def test_parameter_counts_close_for_fixture_and_production() -> None:
    bank = _rank_one_bank()
    parent = _hierarchical_bank()

    assert bank.global_rank_one_parameter_count == 2 * 2 * 2
    assert bank.stored_parameter_count == parent.stored_parameter_count + 8

    hierarchy_bank = 83_008
    rank_one_increment = 2 * 8 * 640
    chart = 23_041
    routers = 28_800
    assert rank_one_increment == 10_240
    assert hierarchy_bank + rank_one_increment == 93_248
    assert hierarchy_bank + rank_one_increment + chart + routers == 145_089
    assert 1.0 - 145_089 / 14_745_600 > 0.99


def test_nonunit_optimizer_right_matches_canonical_checkpoint_prediction() -> None:
    template = _rank_one_bank()
    contexts = torch.tensor(
        [[[1.0, 2.0]], [[-0.5, 3.0]]], dtype=torch.float64
    )
    capture = AttentionTransportRowBatch(
        invocation_id="mutation.11",
        head_contexts=contexts,
        operator_outputs=torch.zeros(2, 2, dtype=torch.float64),
        operator_output_score_gradients=torch.ones(2, 2, dtype=torch.float64),
        chart_inputs=contexts[:, 0].clone(),
        head_operators=torch.zeros(1, 2, 2, dtype=torch.float64),
        native_v_o_parameter_count=8,
    )
    coefficients = torch.tensor(
        [[0.4, -0.7], [1.2, 0.25]], dtype=torch.float64
    )
    raw_left = torch.tensor([[0.8, -0.3], [0.1, 1.5]], dtype=torch.float64)
    raw_right = torch.tensor([[3.0, 4.0], [-2.0, 5.0]], dtype=torch.float64)
    assert not torch.allclose(
        torch.linalg.vector_norm(raw_right, dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    unit_right = raw_right / torch.linalg.vector_norm(
        raw_right, dim=-1, keepdim=True
    )

    optimization_prediction = _fixed_support_prediction(
        template,
        capture,
        coefficients,
        raw_left,
        unit_right,
    )
    checkpoint = _global_rank_one_bank_with_factors(
        template, raw_left, unit_right
    )
    checkpoint_candidates = checkpoint.candidate_outputs_from_head_contexts(
        invocation_id=capture.invocation_id,
        head_contexts=capture.head_contexts,
    )
    checkpoint_prediction = torch.einsum(
        "nk,nkh->nh", coefficients, checkpoint_candidates
    )

    assert torch.allclose(
        torch.linalg.vector_norm(
            checkpoint.global_residual_right_vectors, dim=-1
        ),
        torch.ones(2, dtype=torch.float64),
    )
    assert torch.allclose(
        optimization_prediction,
        checkpoint_prediction,
        atol=2.0e-15,
        rtol=2.0e-15,
    )


def test_signed_router_features_remain_2k_and_one_global_top2() -> None:
    bank = _rank_one_bank(generator_count=3, hidden_width=3)
    parent = _hierarchical_bank(generator_count=3, hidden_width=3)
    contexts = torch.tensor(
        [[[1.0, -2.0, 0.5]], [[-1.0, 0.25, 2.0]]],
        dtype=torch.float64,
    )
    features = bank.router_context_features_from_head_contexts(
        invocation_id="mutation.11", head_contexts=contexts
    )
    parent_features = parent.router_context_features_from_head_contexts(
        invocation_id="mutation.11", head_contexts=contexts
    )
    router = _router(bank, active_count=2, invocation_id="mutation.11")
    execution = router.execute(
        torch.zeros(2, 3, dtype=torch.float64), features
    )

    assert features.shape == (2, 2 * bank.generator_count)
    assert torch.equal(features, parent_features)
    assert torch.equal(
        execution.active_mask.any(dim=1).sum(dim=-1),
        torch.full((2,), 2, dtype=torch.int64),
    )
    assert int((execution.generator_coefficients != 0.0).sum(dim=-1).max()) == 2
    assert torch.allclose(
        execution.sparse_selection_probabilities.sum(dim=-1),
        torch.ones(2, 1, dtype=torch.float64),
    )


def test_existing_core_evaluator_accepts_rank_one_bank_with_exact_top2() -> None:
    bank = _rank_one_bank(generator_count=3, hidden_width=3)
    router = _router(bank, active_count=2, invocation_id="mutation.11")
    contexts = torch.tensor(
        [[[1.0, 2.0, 0.0]], [[-1.0, 0.5, 0.0]]], dtype=torch.float64
    )
    candidates = bank.candidate_outputs_from_head_contexts(
        invocation_id="mutation.11", head_contexts=contexts
    )
    features = bank.router_context_features_from_head_contexts(
        invocation_id="mutation.11", head_contexts=contexts
    )
    chart_inputs = torch.zeros_like(contexts[:, 0])
    execution = router.execute(chart_inputs, features)
    coefficients = execution.generator_coefficients
    assert torch.equal(coefficients[0], coefficients[1])
    outputs = torch.einsum("nk,nkh->nh", coefficients, candidates)
    head_operator = torch.diag(coefficients[0])[None]
    assert torch.allclose(
        outputs,
        torch.einsum("aoh,nah->no", head_operator, contexts),
        atol=1.0e-15,
        rtol=0.0,
    )
    capture = AttentionTransportRowBatch(
        invocation_id="mutation.11",
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=chart_inputs,
        head_operators=head_operator,
        native_v_o_parameter_count=18,
    )
    evaluation = evaluate_chart_conditioned_attention_transport_router(
        bank,
        router,
        (capture,),
        euclidean_floor=1.0e-3,
    )

    assert evaluation.relative_fisher_loss < 1.0e-12
    assert evaluation.oracle_coefficient_ridge == 0.0
    assert evaluation.active_count_minimum == 2
    assert evaluation.active_count_maximum == 2
    assert evaluation.effective_support_count_minimum == 2
    assert evaluation.effective_support_count_maximum == 2
    assert evaluation.sparse_probability_renormalization_max_error < 1.0e-12


def _operator_capture(
    invocation_id: str,
    contexts: torch.Tensor,
    operator: torch.Tensor,
    *,
    score_gradients: torch.Tensor | None = None,
) -> AttentionTransportRowBatch:
    head_contexts = contexts[:, None, :]
    head_operators = operator[None]
    outputs = torch.einsum("aoh,nah->no", head_operators, head_contexts)
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=head_contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=(
            torch.ones_like(outputs)
            if score_gradients is None
            else score_gradients
        ),
        chart_inputs=contexts.clone(),
        head_operators=head_operators,
        native_v_o_parameter_count=2 * contexts.shape[1] ** 2,
    )


def _planted_amplitude_coupling_fixture() -> tuple[
    GlobalRankOneFisherAttentionTransportGeneratorBank,
    tuple[AttentionTransportRowBatch, ...],
]:
    # Generator 0 has base [x0, 0] and residual [0, x0]. Generator 1 is
    # inactive because every planted context has x1=0. The three operators
    # separately prefer base-only, tied base+residual, and untied amplitudes.
    bank = _rank_one_bank(
        left=torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64),
        right=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
    )
    contexts = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    operators = {
        "mutation.11": torch.tensor(
            [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        ),
        "mutation.23": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64
        ),
        "mutation.35": torch.tensor(
            [[2.0, 0.0], [-0.5, 0.0]], dtype=torch.float64
        ),
    }
    # Deliberately scramble input order; the diagnostic contract is numeric
    # global-site order, independent of capture materialization order.
    captures = tuple(
        _operator_capture(invocation_id, contexts, operators[invocation_id])
        for invocation_id in ("mutation.35", "mutation.11", "mutation.23")
    )
    return bank, captures


def test_coupling_diagnostic_untied_oracles_are_nested_and_expose_tying() -> None:
    bank, captures = _planted_amplitude_coupling_fixture()
    diagnostic = evaluate_global_rank_one_coupling_diagnostic(
        bank,
        captures,
        active_count=2,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    scopes = (diagnostic, *diagnostic.per_site)
    for scope in scopes:
        tolerance = 1.0e-11 * max(1.0, scope.tied.fisher_loss)
        assert scope.fixed_support_untied.fisher_loss <= (
            scope.tied.fisher_loss + tolerance
        )
        assert scope.reselected_support_untied.fisher_loss <= (
            scope.fixed_support_untied.fisher_loss + tolerance
        )

    assert diagnostic.tied.relative_fisher_loss > 1.0e-4
    assert diagnostic.fixed_support_untied.relative_fisher_loss < 1.0e-20
    assert diagnostic.reselected_support_untied.relative_fisher_loss < 1.0e-20


def test_coupling_diagnostic_orders_sites_and_closes_leave_one_ablations() -> None:
    bank, captures = _planted_amplitude_coupling_fixture()
    diagnostic = evaluate_global_rank_one_coupling_diagnostic(
        bank,
        captures,
        active_count=2,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    assert diagnostic.global_attention_invocation_ids == (
        "mutation.11",
        "mutation.23",
        "mutation.35",
    )
    assert tuple(site.invocation_id for site in diagnostic.per_site) == (
        "mutation.11",
        "mutation.23",
        "mutation.35",
    )
    leave_one = dict(
        diagnostic.leave_one_site_residual_off_relative_fisher_losses
    )
    assert tuple(leave_one) == diagnostic.global_attention_invocation_ids

    total_rows = sum(site.baseline.row_count for site in diagnostic.per_site)
    for disabled_id, observed in leave_one.items():
        expected_fisher = sum(
            site.baseline.row_count
            * (
                site.baseline.fisher_loss
                if site.invocation_id == disabled_id
                else site.tied.fisher_loss
            )
            for site in diagnostic.per_site
        ) / total_rows
        assert observed == pytest.approx(
            expected_fisher / diagnostic.tied.zero_predictor_fisher_loss,
            abs=1.0e-12,
            rel=1.0e-11,
        )

    assert leave_one["mutation.11"] < diagnostic.tied.relative_fisher_loss
    assert leave_one["mutation.23"] > diagnostic.tied.relative_fisher_loss
    assert leave_one["mutation.35"] < diagnostic.tied.relative_fisher_loss


def test_grouped_top2_counts_generator_identities_not_four_amplitudes() -> None:
    # Two selected generator identities contribute four independent scalar
    # atoms: base/residual for generator 0 and base/residual for generator 1.
    # A channel-level Top2 could not span this four-coordinate target.
    bank = _rank_one_bank(
        generator_count=3,
        hidden_width=4,
        left=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        right=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    contexts = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
            [2.0, -0.5, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    operator = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    captures = tuple(
        _operator_capture(invocation_id, contexts, operator)
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )

    diagnostic = evaluate_global_rank_one_coupling_diagnostic(
        bank,
        captures,
        active_count=2,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    assert diagnostic.active_generator_count_limit == 2
    assert diagnostic.tied.relative_fisher_loss > 1.0e-4
    assert diagnostic.fixed_support_untied.relative_fisher_loss < 1.0e-20
    assert diagnostic.reselected_support_untied.relative_fisher_loss < 1.0e-20
    assert all(
        site.reselected_support_untied.active_generator_count_limit == 2
        and site.reselected_support_untied.relative_fisher_loss < 1.0e-20
        for site in diagnostic.per_site
    )


def test_positive_ridge_tied_embedding_is_nested_with_equal_penalty() -> None:
    generator = torch.Generator().manual_seed(73)
    base = torch.randn(3, 3, 4, generator=generator, dtype=torch.float64)
    residual = torch.randn(3, 3, 4, generator=generator, dtype=torch.float64)
    # Each row uses at most two generator identities, matching grouped Top2.
    tied_beta = torch.tensor(
        [[1.0, -0.5, 0.0], [0.0, 2.0, -1.0], [0.25, 0.0, 0.75]],
        dtype=torch.float64,
    )
    assert int((tied_beta != 0.0).sum(dim=-1).max()) == 2

    atoms = _untied_sum_difference_atoms(base, residual)
    # z=B+R and d=B-R. Embedding tied beta as gamma=beta, delta=0
    # reproduces both its prediction and its positive ridge cost exactly.
    untied_coefficients = torch.zeros(3, 6, dtype=torch.float64)
    untied_coefficients[:, 0::2] = tied_beta
    tied_prediction = torch.einsum(
        "nk,nkh->nh", tied_beta, base + residual
    )
    untied_prediction = torch.einsum(
        "np,nph->nh", untied_coefficients, atoms
    )
    ridge = 0.37

    assert torch.equal(untied_prediction, tied_prediction)
    assert torch.equal(
        ridge * untied_coefficients.square().sum(dim=-1),
        ridge * tied_beta.square().sum(dim=-1),
    )


def test_reselected_grouped_top2_strictly_beats_fixed_tied_support() -> None:
    # Tied generator 2 collapses B2+R2 to e3, which is orthogonal to the
    # Euclidean-only target, so tied Top2 selects generators 0 and 1. Untying
    # generator 2 recovers its hidden e2 direction; the best grouped Top2 then
    # reselects generator 2 plus one of {0,1}, strictly beating fixed support.
    bank = _rank_one_bank(
        generator_count=3,
        hidden_width=4,
        left=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        right=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    contexts = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [-1.0, 2.0, 0.5, 0.0],
            [2.0, -1.0, 1.5, 0.0],
        ],
        dtype=torch.float64,
    )
    operator = torch.diag(
        torch.tensor([1.0, 1.0, 8.0, 0.0], dtype=torch.float64)
    )
    captures = tuple(
        _operator_capture(
            invocation_id,
            contexts,
            operator,
            score_gradients=torch.zeros_like(contexts),
        )
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )

    diagnostic = evaluate_global_rank_one_coupling_diagnostic(
        bank,
        captures,
        active_count=2,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    assert diagnostic.active_generator_count_limit == 2
    assert diagnostic.fixed_support_untied.relative_fisher_loss == pytest.approx(
        diagnostic.tied.relative_fisher_loss,
        abs=1.0e-12,
        rel=1.0e-11,
    )
    assert diagnostic.reselected_support_untied.relative_fisher_loss < (
        diagnostic.fixed_support_untied.relative_fisher_loss - 1.0e-3
    )
    assert all(
        site.reselected_support_untied.active_generator_count_limit == 2
        and site.reselected_support_untied.relative_fisher_loss
        < site.fixed_support_untied.relative_fisher_loss - 1.0e-3
        for site in diagnostic.per_site
    )


def test_zero_residual_makes_all_coupling_oracles_equivalent() -> None:
    _planted_bank, captures = _planted_amplitude_coupling_fixture()
    diagnostic = evaluate_global_rank_one_coupling_diagnostic(
        _rank_one_bank(),
        captures,
        active_count=2,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    for scope in (diagnostic, *diagnostic.per_site):
        losses = (
            scope.baseline.fisher_loss,
            scope.tied.fisher_loss,
            scope.fixed_support_untied.fisher_loss,
            scope.reselected_support_untied.fisher_loss,
        )
        assert losses == pytest.approx(
            (losses[0],) * 4,
            abs=1.0e-12,
            rel=1.0e-11,
        )
    assert all(
        relative == pytest.approx(
            diagnostic.tied.relative_fisher_loss,
            abs=1.0e-12,
            rel=1.0e-11,
        )
        for _invocation_id, relative in (
            diagnostic.leave_one_site_residual_off_relative_fisher_losses
        )
    )


def test_coupling_diagnostic_fails_closed_and_retains_metrics_only() -> None:
    bank, captures = _planted_amplitude_coupling_fixture()
    invalid_calls = (
        ({"active_count": False}, "active_count"),
        ({"active_count": 0}, "active_count"),
        ({"active_count": bank.generator_count + 1}, "active_count"),
        ({"euclidean_floor": False}, "euclidean_floor"),
        ({"euclidean_floor": 0.0}, "euclidean_floor"),
        ({"euclidean_floor": float("nan")}, "euclidean_floor"),
        ({"coefficient_ridge": False}, "coefficient_ridge"),
        ({"coefficient_ridge": -1.0}, "coefficient_ridge"),
        ({"coefficient_ridge": float("inf")}, "coefficient_ridge"),
    )
    for kwargs, message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            evaluate_global_rank_one_coupling_diagnostic(
                bank,
                captures,
                **kwargs,
            )

    incomplete = tuple(
        capture
        for capture in captures
        if capture.invocation_id != "mutation.35"
    )
    with pytest.raises(ValueError, match="omit.*global-attention"):
        evaluate_global_rank_one_coupling_diagnostic(bank, incomplete)

    diagnostic = evaluate_global_rank_one_coupling_diagnostic(bank, captures)
    retained_field_names: set[str] = set()

    def assert_metrics_only(value: object) -> None:
        if is_dataclass(value):
            for item in fields(value):
                retained_field_names.add(item.name)
                assert_metrics_only(getattr(value, item.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                assert_metrics_only(item)
        else:
            assert not isinstance(value, torch.Tensor)

    assert_metrics_only(diagnostic)
    assert retained_field_names.isdisjoint(
        {
            "coefficients",
            "generator_support",
            "support_mask",
            "targets",
            "score_gradients",
            "head_contexts",
            "operator_outputs",
            "raw_activations",
        }
    )


def _tiny_captures() -> tuple[AttentionTransportRowBatch, ...]:
    layout = _layout()
    contexts = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    captures = []
    for invocation_id in layout.invocation_ids:
        operator = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.15]]], dtype=torch.float64
        )
        if invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
            operator = operator + torch.tensor(
                [[[0.0, 0.7], [-0.4, 0.0]]], dtype=torch.float64
            )
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
    return tuple(captures)


def _tiny_fit_config() -> GlobalRankOneAttentionTransportFitConfig:
    shared = SparseAwareAttentionTransportFitConfig(
        generator_count=1,
        generator_rank=1,
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
    return GlobalRankOneAttentionTransportFitConfig(
        hierarchical=HierarchicalSparseAttentionTransportFitConfig(
            shared=shared,
            alternating_rounds=4,
        ),
        alternating_rounds=2,
    )


def test_a_only_rank_one_fit_is_deterministic_and_never_regresses() -> None:
    captures = _tiny_captures()
    config = _tiny_fit_config()
    first = fit_global_rank_one_attention_transport_generator_bank(
        captures, config=config
    )
    second = fit_global_rank_one_attention_transport_generator_bank(
        captures, config=config
    )

    assert first.bank.artifact_sha256 == second.bank.artifact_sha256
    assert first.fit_round_relative_fisher_losses == pytest.approx(
        second.fit_round_relative_fisher_losses, abs=0.0, rel=0.0
    )
    assert first.fit_round_global_relative_fisher_losses == pytest.approx(
        second.fit_round_global_relative_fisher_losses, abs=0.0, rel=0.0
    )
    assert len(first.fit_round_relative_fisher_losses) == (
        config.alternating_rounds + 1
    )
    assert len(first.fit_round_global_relative_fisher_losses) == (
        config.alternating_rounds + 1
    )
    assert first.fit_round_relative_fisher_losses[0] == pytest.approx(
        first.initial_fit_oracle.relative_fisher_loss,
        abs=1.0e-12,
        rel=1.0e-11,
    )
    assert first.fit_round_relative_fisher_losses[-1] == pytest.approx(
        first.final_fit_oracle.relative_fisher_loss,
        abs=1.0e-12,
        rel=1.0e-11,
    )
    assert first.fit_round_global_relative_fisher_losses[0] == pytest.approx(
        first.initial_global_fit_oracle.relative_fisher_loss,
        abs=1.0e-12,
        rel=1.0e-11,
    )
    assert first.fit_round_global_relative_fisher_losses[-1] == pytest.approx(
        first.final_global_fit_oracle.relative_fisher_loss,
        abs=1.0e-12,
        rel=1.0e-11,
    )
    assert first.final_fit_oracle.relative_fisher_loss <= (
        first.initial_fit_oracle.relative_fisher_loss + 1.0e-12
    )
    assert first.final_global_fit_oracle.relative_fisher_loss <= (
        first.initial_global_fit_oracle.relative_fisher_loss + 1.0e-12
    )
    assert all(
        right <= left + 1.0e-12
        for left, right in zip(
            first.fit_round_relative_fisher_losses,
            first.fit_round_relative_fisher_losses[1:],
        )
    )
    assert all(
        right <= left + 1.0e-12
        for left, right in zip(
            first.fit_round_global_relative_fisher_losses,
            first.fit_round_global_relative_fisher_losses[1:],
        )
    )


def test_state_hash_and_lineage_tampering_fail_closed() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    right = torch.tensor([[0.5, -1.0], [-0.25, 0.75]], dtype=torch.float64)
    bank = _rank_one_bank(left=left, right=right)
    state = bank.state_dict()

    assert state["contains_native_value_or_output_weights"] is False
    assert state["contains_oracle_outputs_or_coefficients"] is False
    assert state["contains_raw_teacher_outputs_or_gradients"] is False
    with pytest.raises(FrozenInstanceError):
        bank.signed_feature_floor = 1.0  # type: ignore[misc]

    drifted = dict(state)
    drifted["hierarchical_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        global_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            drifted
        )

    changed = dict(state)
    changed["global_residual_left_vectors"] = (
        torch.as_tensor(state["global_residual_left_vectors"]) + 1.0
    )
    with pytest.raises(ValueError, match="artifact"):
        global_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            changed
        )

    bank.global_residual_left_vectors[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = bank.artifact_sha256
