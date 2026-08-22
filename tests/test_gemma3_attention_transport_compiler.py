from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
)
from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportCapture,
    AttentionTransportFitConfig,
    AttentionTransportHeadOperatorCatalog,
    ChartConditionedRouterConfig,
    EXACT_SPARSE_ORACLE_SOLVER,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    HELDOUT_ORACLE_COEFFICIENT_RIDGE,
    _capture_catalog,
    attention_transport_capture_from_exact_attention,
    attention_transport_row_batch_from_capture,
    attention_transport_resource_accounting,
    chart_conditioned_attention_transport_router_from_state_dict,
    evaluate_chart_conditioned_attention_transport_router,
    fisher_attention_transport_generator_bank_from_state_dict,
    fit_chart_conditioned_attention_transport_router,
    fit_frozen_attention_chart_atlas,
    fit_fisher_attention_transport_generator_bank,
    frozen_attention_chart_atlas_from_state_dict,
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)


def _exact_attention_generator():
    hidden = 4
    head_dimension = 2
    norm = NormalizationSpec(
        kind="rms_norm",
        width=hidden,
        epsilon=1.0e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    config = Gemma3AttentionGeneratorConfig(
        invocation_id="mutation.native",
        residual_width=hidden,
        attention=AttentionSpec(
            kind="global_causal",
            query_heads=2,
            key_value_heads=1,
            head_dimension=head_dimension,
            query_scale=head_dimension**-0.5,
            qk_norm=True,
            window_size=None,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=norm,
        output_norm=norm,
        qk_norm=replace(norm, width=head_dimension),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    random = torch.Generator().manual_seed(90_001)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns,
            dtype=torch.float64,
            generator=random,
        )

    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.zeros(hidden, dtype=torch.float64),
            q_weight=matrix(4, hidden),
            k_weight=matrix(2, hidden),
            v_weight=matrix(2, hidden),
            o_weight=matrix(hidden, 4),
            output_norm_weight=torch.zeros(hidden, dtype=torch.float64),
            q_norm_weight=torch.zeros(head_dimension, dtype=torch.float64),
            k_norm_weight=torch.zeros(head_dimension, dtype=torch.float64),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _sequence(length: int) -> SequenceContext:
    valid = torch.ones(1, length, dtype=torch.bool)
    positions = torch.arange(length, dtype=torch.int64)[None]
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )


def _structured_weights(invocation_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = 6
    value_weight = torch.zeros(2, hidden, dtype=torch.float64)
    value_weight[0, 0] = 1.0
    value_weight[1, 1] = 1.0
    left_zero = torch.zeros(hidden, 2, dtype=torch.float64)
    left_zero[2, 0] = 1.0
    left_zero[3, 1] = 0.75
    left_one = torch.zeros(hidden, 2, dtype=torch.float64)
    left_one[4, 0] = 1.25
    left_one[5, 1] = 0.5
    coefficients = (
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.75, 0.5], [1.0, -0.75]],
            dtype=torch.float64,
        )
        if invocation_index == 0
        else torch.tensor(
            [[0.5, 1.0], [1.25, 0.25], [-0.5, 1.0], [0.75, -1.25]],
            dtype=torch.float64,
        )
    )
    output_weight = torch.cat(
        tuple(
            coefficients[head, 0] * left_zero
            + coefficients[head, 1] * left_one
            for head in range(4)
        ),
        dim=1,
    )
    return value_weight, output_weight


def _structured_capture(
    invocation_index: int,
    *,
    activation_seed: int,
) -> AttentionTransportCapture:
    random = torch.Generator().manual_seed(activation_seed)
    batch = 3
    length = 4
    hidden = 6
    heads = 4
    normalized = torch.randn(
        batch,
        length,
        hidden,
        dtype=torch.float64,
        generator=random,
    )
    logits = torch.randn(
        batch,
        heads,
        length,
        length,
        dtype=torch.float64,
        generator=random,
    )
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    probabilities = torch.softmax(
        logits.masked_fill(~causal[None, None], -torch.inf),
        dim=-1,
    )
    value_weight, output_weight = _structured_weights(invocation_index)
    head_contexts = torch.einsum("baqs,bsh->bqah", probabilities, normalized)
    head_operators = torch.stack(
        tuple(
            output_weight[:, head * 2 : (head + 1) * 2] @ value_weight
            for head in range(heads)
        )
    )
    outputs = torch.einsum("aoh,bqah->bqo", head_operators, head_contexts)
    gradients = torch.randn(
        outputs.shape,
        dtype=torch.float64,
        generator=random,
    )
    valid = torch.ones(batch, length, dtype=torch.bool)
    return AttentionTransportCapture(
        invocation_id=f"mutation.{invocation_index}",
        normalized_inputs=normalized,
        attention_probabilities=probabilities,
        operator_outputs=outputs,
        operator_output_score_gradients=gradients,
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        value_projection_weight=value_weight,
        output_projection_weight=output_weight,
        query_head_count=heads,
        key_value_head_count=1,
        head_dimension=2,
    )


def _routing_fixture() -> tuple[
    FisherAttentionTransportGeneratorBank,
    AttentionTransportCapture,
    FrozenAttentionChartAtlas,
]:
    hidden = 4
    length = 12
    normalized = torch.zeros(1, length, hidden, dtype=torch.float64)
    normalized[0, 0::2, 0] = 1.0
    normalized[0, 1::2, 1] = 1.0
    probabilities = torch.eye(length, dtype=torch.float64)[None, None].repeat(
        1,
        2,
        1,
        1,
    )
    value_weight = torch.zeros(2, hidden, dtype=torch.float64)
    value_weight[0, 0] = 1.0
    value_weight[1, 1] = 1.0
    output_weight = torch.zeros(hidden, 2, dtype=torch.float64)
    output_weight[0, 0] = 1.0
    output_weight[1, 1] = 1.0
    outputs = normalized.clone()
    gradients = torch.ones_like(outputs)
    valid = torch.ones(1, length, dtype=torch.bool)
    capture = AttentionTransportCapture(
        invocation_id="mutation.router",
        normalized_inputs=normalized,
        attention_probabilities=probabilities,
        operator_outputs=outputs,
        operator_output_score_gradients=gradients,
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        value_projection_weight=value_weight,
        output_projection_weight=output_weight,
        query_head_count=2,
        key_value_head_count=2,
        head_dimension=1,
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


def test_samplewise_fisher_loss_is_joint_after_head_summation() -> None:
    head_errors = torch.tensor(
        [[[1.0, -1.0], [-1.0, 1.0]]],
        dtype=torch.float64,
    )
    gradient = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    joint_prediction = head_errors.sum(dim=1)
    joint = samplewise_joint_fisher_loss(
        joint_prediction,
        torch.zeros_like(joint_prediction),
        gradient,
        euclidean_floor=0.1,
    )
    separate = (
        (head_errors * gradient[:, None]).sum(dim=-1).square()
        + 0.1 * head_errors.square().sum(dim=-1)
    ).mean()
    assert float(joint) == 0.0
    assert float(separate) > 0.0


def test_low_rank_candidate_application_matches_dense_generator_expansion() -> None:
    random = torch.Generator().manual_seed(90_000)
    generator_count = 3
    hidden = 7
    rank = 2
    head_count = 4
    left, _ = torch.linalg.qr(
        torch.randn(
            generator_count,
            hidden,
            rank,
            dtype=torch.float64,
            generator=random,
        ),
        mode="reduced",
    )
    right, _ = torch.linalg.qr(
        torch.randn(
            generator_count,
            hidden,
            rank,
            dtype=torch.float64,
            generator=random,
        ),
        mode="reduced",
    )
    bank = FisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.rand(
            generator_count,
            rank,
            dtype=torch.float64,
            generator=random,
        ),
        right_factors=right,
        head_coefficients=torch.randn(
            head_count,
            generator_count,
            dtype=torch.float64,
            generator=random,
        ),
        head_ids=tuple(
            f"mutation.low-rank/query-head.{index}"
            for index in range(head_count)
        ),
    )
    head_contexts = torch.randn(
        5,
        head_count,
        hidden,
        dtype=torch.float64,
        generator=random,
    )
    contextual = bank.contextual_inputs_from_head_contexts(
        invocation_id="mutation.low-rank",
        head_contexts=head_contexts,
    )
    dense = torch.einsum(
        "koi,nki->nko",
        bank.generator_matrices,
        contextual,
    )

    low_rank = bank.candidate_outputs_from_head_contexts(
        invocation_id="mutation.low-rank",
        head_contexts=head_contexts,
    )

    assert low_rank.is_contiguous()
    assert torch.allclose(low_rank, dense, atol=1.0e-12, rtol=1.0e-12)


def test_native_attention_bridge_authenticates_v_o_boundary_capture() -> None:
    generator = _exact_attention_generator()
    hidden = torch.randn(
        1,
        5,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(90_002),
    )
    execution = generator(
        hidden,
        _sequence(hidden.shape[1]),
        capture_route_factors=True,
    )
    assert execution.route_factor_snapshot is not None
    capture = attention_transport_capture_from_exact_attention(
        generator,
        normalized_inputs=execution.normalized_input,
        route_factors=execution.route_factor_snapshot,
        operator_outputs=execution.operator_output,
        operator_output_score_gradients=torch.ones_like(execution.operator_output),
    )

    assert capture.invocation_id == generator.config.invocation_id
    assert capture.query_head_count == 2
    assert capture.key_value_head_count == 1
    assert torch.allclose(
        capture.reconstruct_native_operator_outputs(),
        capture.valid_operator_outputs,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_native_attention_bridge_rejects_one_ulp_repeated_value_drift() -> None:
    generator = _exact_attention_generator()
    hidden = torch.randn(
        1,
        5,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(90_003),
    )
    execution = generator(
        hidden,
        _sequence(hidden.shape[1]),
        capture_route_factors=True,
    )
    assert execution.route_factor_snapshot is not None
    repeated = execution.route_factor_snapshot.repeated_values.clone()
    repeated.flatten()[0] = torch.nextafter(
        repeated.flatten()[0],
        torch.tensor(torch.inf, dtype=repeated.dtype),
    )

    with pytest.raises(ValueError, match="native repeated values do not close"):
        attention_transport_capture_from_exact_attention(
            generator,
            normalized_inputs=execution.normalized_input,
            route_factors=replace(
                execution.route_factor_snapshot,
                repeated_values=repeated,
            ),
            operator_outputs=execution.operator_output,
            operator_output_score_gradients=torch.ones_like(
                execution.operator_output
            ),
        )


def test_joint_low_rank_bank_recovers_shared_v_o_transport_on_heldout_rows() -> None:
    fit_captures = tuple(
        _structured_capture(index, activation_seed=91_000 + index)
        for index in range(2)
    )
    evaluation_captures = tuple(
        _structured_capture(index, activation_seed=92_000 + index)
        for index in range(2)
    )
    fit = fit_fisher_attention_transport_generator_bank(
        fit_captures,
        evaluation_captures,
        config=AttentionTransportFitConfig(
            generator_count=2,
            generator_rank=2,
            steps=3,
            learning_rate=1.0e-4,
            coefficient_ridge=0.0,
        ),
    )

    assert fit.initial_fit_metrics.normalized_rmse < 1.0e-12
    assert fit.final_fit_metrics.normalized_rmse < 1.0e-12
    assert fit.evaluation_metrics.normalized_rmse < 1.0e-12
    assert fit.parameter_compression_eligible
    assert fit.compiled_bank_parameter_count < fit.native_v_o_parameter_count
    state = fit.bank.state_dict()
    assert state["native_qk_retained"] is True
    assert state["native_output_rmsnorm_retained"] is True
    assert state["contains_teacher_outputs_or_gradients"] is False
    assert state["contains_native_value_or_output_weights"] is False
    reloaded = fisher_attention_transport_generator_bank_from_state_dict(state)
    assert reloaded.artifact_sha256 == fit.bank.artifact_sha256
    tensor_keys = {
        key for key, value in state.items() if isinstance(value, torch.Tensor)
    }
    assert not any(
        forbidden in key
        for key in tensor_keys
        for forbidden in (
            "operator_outputs",
            "score_gradients",
            "value_weight",
            "output_weight",
        )
    )
    accounting = attention_transport_resource_accounting(
        fit.bank,
        fit_captures,
    )
    assert accounting.parameter_compression_eligible
    assert accounting.logical_speedup_claimed is False


def test_prompt_free_row_batches_preserve_compiler_sufficient_statistics() -> None:
    fit_captures = tuple(
        _structured_capture(index, activation_seed=91_500 + index)
        for index in range(2)
    )
    evaluation_captures = tuple(
        _structured_capture(index, activation_seed=91_600 + index)
        for index in range(2)
    )
    selected = torch.tensor([0, 3, 7, 11], dtype=torch.int64)
    fit_rows = tuple(
        attention_transport_row_batch_from_capture(
            capture,
            row_indices=selected,
        )
        for capture in fit_captures
    )
    evaluation_rows = tuple(
        attention_transport_row_batch_from_capture(
            capture,
            row_indices=selected,
        )
        for capture in evaluation_captures
    )

    fit = fit_fisher_attention_transport_generator_bank(
        fit_rows,
        evaluation_rows,
        config=AttentionTransportFitConfig(
            generator_count=2,
            generator_rank=2,
            steps=2,
            learning_rate=1.0e-4,
            coefficient_ridge=0.0,
        ),
    )
    atlas = fit_frozen_attention_chart_atlas(
        fit_rows,
        chart_count=1,
        chart_rank=1,
        lloyd_iterations=2,
    )

    assert fit.final_fit_metrics.row_count == 8
    assert fit.evaluation_metrics.row_count == 8
    assert fit.final_fit_metrics.normalized_rmse < 1.0e-12
    assert atlas.hidden_width == fit.bank.hidden_width


def test_compact_rows_share_one_authenticated_head_operator_catalog() -> None:
    first = _structured_capture(0, activation_seed=91_700)
    second = _structured_capture(0, activation_seed=91_701)
    catalog = AttentionTransportHeadOperatorCatalog(
        invocation_id=first.invocation_id,
        head_operators=first.head_operators,
        native_v_o_parameter_count=first.native_v_o_parameter_count,
    )

    first_rows = attention_transport_row_batch_from_capture(
        first,
        row_indices=torch.tensor([0, 5], dtype=torch.int64),
        head_operator_catalog=catalog,
    )
    second_rows = attention_transport_row_batch_from_capture(
        second,
        row_indices=torch.tensor([1, 7], dtype=torch.int64),
        head_operator_catalog=catalog,
    )

    assert first_rows.head_operators is catalog.head_operators
    assert second_rows.head_operators is catalog.head_operators
    assert (
        first_rows.head_operator_catalog_identity
        == second_rows.head_operator_catalog_identity
    )
    assert first_rows.head_operators_sha256 == second_rows.head_operators_sha256
    first_rows.validate_head_operator_catalog(rehash=True)

    catalog.head_operators[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="catalog changed"):
        second_rows.validate_head_operator_catalog(rehash=True)


def test_capture_catalog_materializes_head_operators_only_when_requested() -> None:
    captures = (
        _structured_capture(0, activation_seed=91_800),
        _structured_capture(0, activation_seed=91_801),
        _structured_capture(1, activation_seed=91_802),
    )

    values, representatives, head_ids, operators = _capture_catalog(captures)

    assert values == captures
    assert representatives == (captures[0], captures[2])
    assert head_ids == captures[0].head_ids + captures[2].head_ids
    assert operators is None

    _, _, _, materialized = _capture_catalog(
        captures,
        materialize_operators=True,
    )
    assert materialized is not None
    assert torch.equal(
        materialized,
        torch.cat((captures[0].head_operators, captures[2].head_operators)),
    )


def test_joint_bank_aggregates_repeated_invocation_batches_once_per_operator() -> None:
    fit_captures = (
        _structured_capture(0, activation_seed=92_100),
        _structured_capture(0, activation_seed=92_101),
        _structured_capture(1, activation_seed=92_102),
        _structured_capture(1, activation_seed=92_103),
    )
    evaluation_captures = (
        _structured_capture(1, activation_seed=92_200),
        _structured_capture(0, activation_seed=92_201),
        _structured_capture(0, activation_seed=92_202),
    )
    fit = fit_fisher_attention_transport_generator_bank(
        fit_captures,
        evaluation_captures,
        config=AttentionTransportFitConfig(
            generator_count=2,
            generator_rank=2,
            steps=3,
            learning_rate=1.0e-4,
            coefficient_ridge=0.0,
        ),
    )

    expected_head_ids = fit_captures[0].head_ids + fit_captures[2].head_ids
    expected_native_parameters = (
        fit_captures[0].native_v_o_parameter_count
        + fit_captures[2].native_v_o_parameter_count
    )
    assert fit.bank.head_ids == expected_head_ids
    assert fit.bank.head_coefficients.shape[0] == len(expected_head_ids)
    assert fit.final_fit_metrics.row_count == sum(
        capture.row_count for capture in fit_captures
    )
    assert fit.evaluation_metrics.row_count == sum(
        capture.row_count for capture in evaluation_captures
    )
    assert fit.final_fit_metrics.normalized_rmse < 1.0e-12
    assert fit.evaluation_metrics.normalized_rmse < 1.0e-12
    assert fit.native_v_o_parameter_count == expected_native_parameters

    accounting = attention_transport_resource_accounting(
        fit.bank,
        tuple(reversed(fit_captures)),
    )
    assert accounting.native_v_o_parameters == expected_native_parameters


def test_repeated_invocation_batches_require_identical_frozen_v_o_weights() -> None:
    first = _structured_capture(0, activation_seed=92_300)
    conflicting = replace(
        _structured_capture(1, activation_seed=92_301),
        invocation_id=first.invocation_id,
    )

    with pytest.raises(
        ValueError,
        match="identical frozen V/O weights and head geometry",
    ):
        fit_fisher_attention_transport_generator_bank(
            (first, conflicting),
            (first,),
            config=AttentionTransportFitConfig(
                generator_count=2,
                generator_rank=2,
                steps=0,
            ),
        )

    with pytest.raises(
        ValueError,
        match="fit and evaluation captures must share frozen V/O operators",
    ):
        fit_fisher_attention_transport_generator_bank(
            (first,),
            (conflicting,),
            config=AttentionTransportFitConfig(
                generator_count=2,
                generator_rank=2,
                steps=0,
            ),
        )


def test_capture_rejects_a_nonclosing_native_head_decomposition() -> None:
    capture = _structured_capture(0, activation_seed=93_000)
    corrupted = capture.operator_outputs.clone()
    corrupted[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="does not reconstruct operator output"):
        replace(capture, operator_outputs=corrupted)


def test_sparse_fisher_oracle_recovers_signed_generator_coalition() -> None:
    rows = 5
    candidates = torch.zeros(rows, 3, 4, dtype=torch.float64)
    candidates[:, 0, 0] = 1.0
    candidates[:, 1, 1] = 1.0
    candidates[:, 2, 2] = 1.0
    expected = torch.tensor(
        [[1.0, -0.5], [-0.75, 1.25], [0.4, -1.2], [2.0, 0.25], [-1.0, -1.0]],
        dtype=torch.float64,
    )
    targets = torch.zeros(rows, 4, dtype=torch.float64)
    targets[:, :2] = expected
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        torch.zeros_like(targets),
        active_count=2,
        euclidean_floor=0.2,
        coefficient_ridge=0.0,
    )

    assert torch.equal(oracle.support[:, :2], torch.ones(rows, 2, dtype=torch.bool))
    assert not bool(oracle.support[:, 2].any())
    assert torch.allclose(oracle.coefficients[:, :2], expected, atol=1.0e-12)
    assert torch.allclose(
        oracle.objective,
        torch.zeros(rows, dtype=torch.float64),
        atol=1.0e-12,
    )

    degenerate = candidates.clone()
    degenerate[:, 2] = 0.0
    degenerate_oracle = solve_fisher_sparse_transport_oracle(
        degenerate,
        targets,
        torch.zeros_like(targets),
        active_count=3,
        euclidean_floor=0.2,
        coefficient_ridge=0.0,
    )
    assert not bool(degenerate_oracle.support[:, 2].any())
    assert torch.allclose(
        degenerate_oracle.coefficients[:, :2],
        expected,
        atol=1.0e-12,
    )


def test_sparse_fisher_oracle_enumerates_past_a_greedy_decoy() -> None:
    candidates = torch.tensor(
        [
            [
                [1.0, 1.0, 0.2],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        torch.zeros_like(targets),
        active_count=2,
        euclidean_floor=1.0,
        coefficient_ridge=0.0,
    )

    assert oracle.solver_kind == EXACT_SPARSE_ORACLE_SOLVER
    assert oracle.coefficient_ridge == 0.0
    assert torch.equal(
        oracle.support,
        torch.tensor([[False, True, True]]),
    )
    assert torch.allclose(
        oracle.coefficients,
        torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert float(oracle.objective[0]) == pytest.approx(0.0, abs=1.0e-12)


def test_chart_router_is_fit_after_bank_and_uses_topk_signed_amplitudes() -> None:
    bank, capture, atlas = _routing_fixture()
    fit = fit_chart_conditioned_attention_transport_router(
        bank,
        capture,
        atlas,
        config=ChartConditionedRouterConfig(
            active_count=1,
            steps=8,
            learning_rate=1.0e-3,
            oracle_ridge=1.0e-10,
        ),
    )
    contextual_features = bank.router_context_features_from_head_contexts(
        invocation_id=capture.invocation_id,
        head_contexts=capture.head_contexts,
    )
    execution = fit.router.execute(
        capture.valid_chart_inputs,
        contextual_features,
    )

    assert torch.equal(
        execution.active_mask.sum(dim=-1),
        torch.ones(capture.row_count, atlas.chart_count, dtype=torch.int64),
    )
    assert torch.allclose(
        execution.sparse_selection_probabilities.sum(dim=-1),
        torch.ones(capture.row_count, atlas.chart_count, dtype=torch.float64),
    )
    assert bool((execution.signed_amplitudes >= 0.0).all())
    assert fit.oracle_output_metrics.normalized_rmse < 1.0e-8
    assert fit.router_output_metrics.normalized_rmse < 1.0e-3
    accounting = attention_transport_resource_accounting(
        bank,
        (capture,),
        (fit.router,),
    )
    assert accounting.native_qk_retained
    assert accounting.native_output_rmsnorm_retained
    assert accounting.logical_speedup_claimed is False


def test_fitted_atlas_and_multibatch_router_round_trip_and_evaluate() -> None:
    atlas_captures = (
        _structured_capture(0, activation_seed=94_000),
        _structured_capture(1, activation_seed=94_001),
    )
    atlas = fit_frozen_attention_chart_atlas(
        atlas_captures,
        chart_count=2,
        chart_rank=1,
        temperature_scale=1.0,
        lloyd_iterations=2,
    )
    reloaded_atlas = frozen_attention_chart_atlas_from_state_dict(
        atlas.state_dict()
    )
    assert reloaded_atlas.artifact_sha256 == atlas.artifact_sha256
    bank, capture, router_atlas = _routing_fixture()
    fit = fit_chart_conditioned_attention_transport_router(
        bank,
        (capture, capture),
        router_atlas,
        config=ChartConditionedRouterConfig(
            active_count=1,
            steps=2,
            learning_rate=1.0e-3,
            oracle_ridge=1.0e-10,
        ),
    )
    reloaded_router = chart_conditioned_attention_transport_router_from_state_dict(
        fit.router.state_dict(),
        atlas=router_atlas,
    )
    assert reloaded_router.artifact_sha256 == fit.router.artifact_sha256
    evaluation = evaluate_chart_conditioned_attention_transport_router(
        bank,
        reloaded_router,
        (capture, capture),
        euclidean_floor=1.0e-3,
    )
    assert evaluation.output_metrics.row_count == 2 * capture.row_count
    assert evaluation.relative_fisher_loss < 1.0e-4
    assert evaluation.oracle_relative_fisher_loss <= evaluation.relative_fisher_loss
    assert evaluation.active_count_minimum == 1
    assert evaluation.active_count_mean == 1.0
    assert evaluation.active_count_maximum == 1
    assert evaluation.oracle_solver_kind == EXACT_SPARSE_ORACLE_SOLVER
    assert evaluation.oracle_active_count_limit == 1
    assert (
        evaluation.oracle_coefficient_ridge
        == HELDOUT_ORACLE_COEFFICIENT_RIDGE
        == 0.0
    )
    assert evaluation.effective_support_count_minimum == 2
    assert evaluation.effective_support_count_maximum == 2
    assert evaluation.sparse_probability_renormalization_max_error < 1.0e-12
    assert torch.allclose(
        evaluation.generator_usage.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_router_evaluation_reports_the_union_of_per_chart_supports() -> None:
    bank, capture, atlas = _routing_fixture()
    fit = fit_chart_conditioned_attention_transport_router(
        bank,
        capture,
        atlas,
        config=ChartConditionedRouterConfig(
            active_count=1,
            steps=0,
            learning_rate=1.0e-3,
            oracle_ridge=1.0e-10,
        ),
    )
    router = replace(
        fit.router,
        selection_weight=torch.zeros_like(fit.router.selection_weight),
        selection_bias=torch.tensor(
            [[20.0, -20.0], [-20.0, 20.0]],
            dtype=torch.float64,
        ),
        amplitude_weight=torch.zeros_like(fit.router.amplitude_weight),
        amplitude_bias=torch.ones_like(fit.router.amplitude_bias),
    )

    evaluation = evaluate_chart_conditioned_attention_transport_router(
        bank,
        router,
        (capture,),
        euclidean_floor=1.0e-3,
    )

    assert evaluation.active_count_minimum == 1
    assert evaluation.active_count_maximum == 1
    assert evaluation.oracle_active_count_limit == 1
    assert evaluation.effective_support_count_minimum == 2
    assert evaluation.effective_support_count_mean == 2.0
    assert evaluation.effective_support_count_maximum == 2


def test_resource_accounting_counts_a_shared_atlas_once() -> None:
    bank, capture, atlas = _routing_fixture()
    first = fit_chart_conditioned_attention_transport_router(
        bank,
        capture,
        atlas,
        config=ChartConditionedRouterConfig(
            active_count=1,
            steps=0,
            learning_rate=1.0e-3,
            oracle_ridge=1.0e-10,
        ),
    ).router
    second = replace(first, invocation_id="mutation.router.second")

    accounting = attention_transport_resource_accounting(
        bank,
        (capture,),
        (first, second),
    )

    assert accounting.chart_parameters == atlas.stored_parameter_count
    assert accounting.router_parameters == 2 * first.stored_parameter_count
