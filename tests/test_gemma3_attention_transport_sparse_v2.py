from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
    FisherAttentionTransportGeneratorBank,
)
from fisher_graph.gemma3_attention_transport_sparse_v2 import (
    SparseAwareAttentionTransportFitConfig,
    evaluate_sparse_attention_transport_bank,
    fit_sparse_aware_attention_transport_generator_bank,
)


def _row_batch(invocation_id: str, *, seed: int) -> AttentionTransportRowBatch:
    random = torch.Generator().manual_seed(seed)
    rows = 12
    hidden = 4
    heads = 2
    contexts = torch.randn(
        rows,
        heads,
        hidden,
        dtype=torch.float64,
        generator=random,
    )
    operators = torch.zeros(heads, hidden, hidden, dtype=torch.float64)
    operators[0, 0, 0] = 1.0
    operators[0, 1, 1] = 0.75
    operators[1, 2, 2] = 1.25
    operators[1, 3, 3] = 0.5
    outputs = torch.einsum("aoh,nah->no", operators, contexts)
    gradients = torch.randn(
        outputs.shape,
        dtype=torch.float64,
        generator=random,
    )
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=gradients,
        chart_inputs=torch.randn(
            rows,
            hidden,
            dtype=torch.float64,
            generator=random,
        ),
        head_operators=operators,
        native_v_o_parameter_count=128,
    )


def _dense_exact_but_top1_inexact() -> tuple[
    FisherAttentionTransportGeneratorBank,
    AttentionTransportRowBatch,
]:
    hidden = 2
    contexts = torch.tensor(
        [[[1.0, 2.0]], [[-2.0, 1.0]], [[1.5, -0.5]]],
        dtype=torch.float64,
    )
    operator = torch.eye(hidden, dtype=torch.float64)[None]
    outputs = contexts[:, 0].clone()
    capture = AttentionTransportRowBatch(
        invocation_id="mutation.sparse-mismatch",
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=outputs.clone(),
        head_operators=operator,
        native_v_o_parameter_count=16,
    )
    left = torch.zeros(2, hidden, 1, dtype=torch.float64)
    right = torch.zeros_like(left)
    left[0, 0, 0] = right[0, 0, 0] = 1.0
    left[1, 1, 0] = right[1, 1, 0] = 1.0
    bank = FisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.ones(2, 1, dtype=torch.float64),
        right_factors=right,
        head_coefficients=torch.ones(1, 2, dtype=torch.float64),
        head_ids=("mutation.sparse-mismatch/query-head.0",),
    )
    return bank, capture


def test_sparse_aware_config_rejects_impossible_support() -> None:
    with pytest.raises(ValueError, match="active_count exceeds"):
        SparseAwareAttentionTransportFitConfig(
            generator_count=2,
            generator_rank=1,
            active_count=3,
        )


def test_exact_sparse_evaluation_exposes_dense_fit_deployment_mismatch() -> None:
    bank, capture = _dense_exact_but_top1_inexact()
    dense = bank.predict_operator_outputs(capture)
    assert torch.equal(dense, capture.valid_operator_outputs)

    sparse = evaluate_sparse_attention_transport_bank(
        bank,
        (capture,),
        active_count=1,
        euclidean_floor=1.0e-3,
    )

    assert sparse.relative_fisher_loss > 0.0
    assert sparse.active_count_limit == 1
    assert sparse.coefficient_ridge == 0.0


def test_sparse_aware_fit_never_selects_a_worse_fit_round() -> None:
    fit = (
        _row_batch("mutation.0", seed=101),
        _row_batch("mutation.1", seed=102),
    )
    evaluation = (
        _row_batch("mutation.0", seed=201),
        _row_batch("mutation.1", seed=202),
    )
    result = fit_sparse_aware_attention_transport_generator_bank(
        fit,
        config=SparseAwareAttentionTransportFitConfig(
            generator_count=2,
            generator_rank=1,
            active_count=1,
            dense_warm_start_steps=2,
            alternating_rounds=2,
            steps_per_round=3,
            learning_rate=1.0e-3,
        ),
    )
    heldout = evaluate_sparse_attention_transport_bank(
        result.bank,
        evaluation,
        active_count=1,
        euclidean_floor=1.0e-3,
    )

    assert (
        result.final_fit_oracle.relative_fisher_loss
        <= result.initial_fit_oracle.relative_fisher_loss
    )
    assert len(result.fit_round_relative_fisher_losses) == 3
    assert all(
        right <= left
        for left, right in zip(
            result.fit_round_relative_fisher_losses,
            result.fit_round_relative_fisher_losses[1:],
        )
    )
    assert heldout.coefficient_ridge == 0.0
    assert result.bank.generator_count == 2
    assert result.bank.generator_rank == 1
    assert torch.allclose(
        torch.linalg.vector_norm(result.bank.singular_scales, dim=1),
        torch.ones(2, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(result.bank.head_coefficients, dim=0),
        torch.ones(2, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
