from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fisher_graph.gemma3_fisher_local_chart_capture import (
    ATTENTION_SUMMARY_FEATURE_NAMES,
    DEFAULT_MLP_PARTITION_SEED,
    LAYER4_CANONICAL_SITES,
    FisherVJPRowID,
    Layer4OutputFisherVJP,
    SharedGemmaRMSNormScale,
    attention_head_q_contributions,
    build_layer4_sequence_capture,
    causal_attention_summaries,
    causal_nll_output_fisher_vjp,
    detach_layer4_output_as_leaf,
    gemma_rmsnorm,
    make_mlp_probe_partition,
    mlp_block_q_contributions,
    run_with_eager_attention_probability_capture,
)


def _causal_probabilities(
    batch: int,
    heads: int,
    length: int,
    valid: Tensor,
) -> Tensor:
    generator = torch.Generator().manual_seed(91)
    scores = torch.randn(
        batch,
        heads,
        length,
        length,
        generator=generator,
        dtype=torch.float32,
    )
    query = torch.arange(length).view(1, 1, length, 1)
    key = torch.arange(length).view(1, 1, 1, length)
    allowed = (key <= query) & valid[:, None, None, :]
    scores = scores.masked_fill(~allowed, -torch.inf)
    probabilities = scores.softmax(dim=-1)
    return probabilities.masked_fill(~valid[:, None, :, None], 0)


def _dummy_vjp(
    *,
    sequence_ids: tuple[str, ...],
    valid: Tensor,
    positions: Tensor,
    width: int,
) -> Layer4OutputFisherVJP:
    row_mask = torch.zeros_like(valid)
    row_mask[:, :-1] = valid[:, :-1] & valid[:, 1:]
    generator = torch.Generator().manual_seed(113)
    gradients = torch.randn(
        valid.shape[0],
        valid.shape[1],
        width,
        generator=generator,
        dtype=torch.float64,
    ).masked_fill(~row_mask.unsqueeze(-1), 0)
    rows = []
    for batch_index, sequence_id in enumerate(sequence_ids):
        for tensor_index in torch.nonzero(
            row_mask[batch_index],
            as_tuple=False,
        ).flatten():
            index = int(tensor_index.item())
            rows.append(
                FisherVJPRowID(
                    sequence_id=sequence_id,
                    query_position=int(positions[batch_index, index].item()),
                    target_position=int(positions[batch_index, index + 1].item()),
                )
            )
    return Layer4OutputFisherVJP(
        sequence_ids=sequence_ids,
        logical_positions=positions,
        valid_row_mask=row_mask,
        gradients=gradients,
        row_ids=tuple(rows),
        causal_nll_sum=7.0,
        supervised_token_count=int(row_mask.sum().item()),
    )


def _exact_capture(*, scheme: str = "contiguous"):
    generator = torch.Generator().manual_seed(57)
    batch, length, width = 2, 5, 8
    heads, head_dimension, intermediate = 4, 3, 13
    sequence_ids = ("sequence-a", "sequence-b")
    valid = torch.ones(batch, length, dtype=torch.bool)
    positions = torch.arange(length).repeat(batch, 1)

    context = torch.randn(
        batch,
        length,
        heads * head_dimension,
        generator=generator,
    )
    output_weight = torch.randn(
        width,
        heads * head_dimension,
        generator=generator,
    )
    q_A = F.linear(context, output_weight)
    attention_norm_weight = torch.tensor(
        [-0.75, -0.1, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0],
        dtype=torch.float32,
    )
    v_A = gemma_rmsnorm(
        q_A,
        attention_norm_weight,
        epsilon=1.0e-6,
    )
    h = torch.randn(batch, length, width, generator=generator) * 3.0
    h_A = h + v_A
    X4 = gemma_rmsnorm(
        h_A,
        torch.linspace(-0.25, 0.5, width),
        epsilon=1.0e-6,
    )

    down_input = torch.randn(
        batch,
        length,
        intermediate,
        generator=generator,
    )
    down_weight = torch.randn(width, intermediate, generator=generator)
    q_M = F.linear(down_input, down_weight)
    mlp_norm_weight = torch.tensor(
        [1.2, 0.7, 0.3, 0.0, -0.2, -0.5, 1.8, 2.4],
        dtype=torch.float32,
    )
    v_M = gemma_rmsnorm(q_M, mlp_norm_weight, epsilon=1.0e-6)
    h_plus = h_A + v_M
    probabilities = _causal_probabilities(batch, heads, length, valid)
    partition = make_mlp_probe_partition(
        intermediate,
        scheme=scheme,
        seed=DEFAULT_MLP_PARTITION_SEED,
    )
    vjp = _dummy_vjp(
        sequence_ids=sequence_ids,
        valid=valid,
        positions=positions,
        width=width,
    )
    capture = build_layer4_sequence_capture(
        sequence_ids=sequence_ids,
        query_valid_mask=valid,
        key_valid_mask=valid,
        logical_positions=positions,
        key_positions=positions,
        h=h,
        q_A=q_A,
        v_A=v_A,
        h_A=h_A,
        X4=X4,
        q_M=q_M,
        v_M=v_M,
        h_plus=h_plus,
        attention_context=context,
        attention_probabilities=probabilities,
        attention_output_projection_weight=output_weight,
        attention_norm_weight=attention_norm_weight,
        attention_norm_epsilon=1.0e-6,
        query_head_count=heads,
        head_dimension=head_dimension,
        mlp_down_projection_input=down_input,
        mlp_down_projection_weight=down_weight,
        mlp_norm_weight=mlp_norm_weight,
        mlp_norm_epsilon=1.0e-6,
        mlp_partition=partition,
        output_fisher_vjp=vjp,
        runtime_dtypes=(("stage", "torch.float32"),),
        absolute_tolerance=2.0e-5,
        relative_tolerance=2.0e-5,
    )
    return capture, {
        "context": context,
        "output_weight": output_weight,
        "down_input": down_input,
        "down_weight": down_weight,
        "attention_norm_weight": attention_norm_weight,
        "mlp_norm_weight": mlp_norm_weight,
    }


def test_canonical_sites_distinguish_post_attention_from_historical_x4() -> None:
    assert dict(LAYER4_CANONICAL_SITES) == {
        "h": "layer.4.input",
        "q_A": "layer.4.attention.operator_output",
        "v_A": "layer.4.attention.delta",
        "h_A": "layer.4.post_attention",
        "X4": "layer.4.mlp.normalized_input",
        "q_M": "layer.4.mlp.operator_output",
        "v_M": "layer.4.mlp.delta",
        "h_plus": "layer.4.output",
    }
    assert LAYER4_CANONICAL_SITES["h_A"] != LAYER4_CANONICAL_SITES["X4"]


@pytest.mark.parametrize("scheme", ["contiguous", "interleaved"])
def test_stage_and_component_invariants_4_through_9_close(scheme: str) -> None:
    capture, _fixture = _exact_capture(scheme=scheme)

    torch.testing.assert_close(capture.h_A, capture.h + capture.v_A)
    torch.testing.assert_close(capture.h_plus, capture.h_A + capture.v_M)
    torch.testing.assert_close(
        capture.h_plus,
        capture.h + capture.v_A + capture.v_M,
    )
    torch.testing.assert_close(
        capture.q_A.to(torch.float64),
        capture.attention_head_q.to(torch.float64).sum(dim=2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        capture.v_A.to(torch.float64),
        capture.attention_head_v.sum(dim=2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        capture.q_M.to(torch.float64),
        capture.mlp_block_q.to(torch.float64).sum(dim=2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        capture.v_M.to(torch.float64),
        capture.mlp_block_v.sum(dim=2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        capture.h_plus.to(torch.float64) - capture.h.to(torch.float64),
        capture.attention_head_v.sum(dim=2) + capture.mlp_block_v.sum(dim=2),
        atol=3.0e-5,
        rtol=2.0e-5,
    )
    assert len(capture.diagnostics.all) == 8
    assert all(item.relative_l2_error < 2.0e-6 for item in capture.diagnostics.all)


def test_shared_scale_uses_fp32_unit_offset_weight_and_one_total() -> None:
    total = torch.tensor(
        [[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]],
        dtype=torch.float16,
    )
    weight = torch.tensor([-0.5, 0.25, 2.0], dtype=torch.float16)
    scale = SharedGemmaRMSNormScale.from_total(
        total,
        weight,
        epsilon=1.0e-6,
    )
    reciprocal = torch.rsqrt(
        total.float().square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    expected_scale = reciprocal * (1.0 + weight.float())

    assert scale.scale.dtype == torch.float32
    assert scale.compute_dtype == "float32"
    assert scale.scale_parameterization == "unit_offset"
    torch.testing.assert_close(scale.scale, expected_scale)
    torch.testing.assert_close(
        gemma_rmsnorm(total, weight, epsilon=1.0e-6),
        (total.float() * expected_scale).to(total.dtype),
    )


def test_independent_component_rmsnorm_is_not_the_realized_decomposition() -> None:
    capture, fixture = _exact_capture()
    independently_normalized_heads = torch.stack(
        tuple(
            gemma_rmsnorm(
                capture.attention_head_q[:, :, head].float(),
                fixture["attention_norm_weight"],
                epsilon=1.0e-6,
            )
            for head in range(4)
        ),
        dim=2,
    ).sum(dim=2)
    independently_normalized_blocks = torch.stack(
        tuple(
            gemma_rmsnorm(
                capture.mlp_block_q[:, :, block].float(),
                fixture["mlp_norm_weight"],
                epsilon=1.0e-6,
            )
            for block in range(4)
        ),
        dim=2,
    ).sum(dim=2)

    shared_attention_error = torch.linalg.vector_norm(
        capture.attention_head_v.sum(dim=2) - capture.v_A.to(torch.float64)
    )
    independent_attention_error = torch.linalg.vector_norm(
        independently_normalized_heads - capture.v_A
    )
    shared_mlp_error = torch.linalg.vector_norm(
        capture.mlp_block_v.sum(dim=2) - capture.v_M.to(torch.float64)
    )
    independent_mlp_error = torch.linalg.vector_norm(
        independently_normalized_blocks - capture.v_M
    )
    assert independent_attention_error > shared_attention_error * 1_000
    assert independent_mlp_error > shared_mlp_error * 1_000
    assert independent_attention_error > 1.0
    assert independent_mlp_error > 1.0


def test_projection_helpers_slice_exact_head_and_probe_block_weights() -> None:
    capture, fixture = _exact_capture(scheme="interleaved")
    head_q = attention_head_q_contributions(
        fixture["context"],
        fixture["output_weight"],
        query_head_count=4,
        head_dimension=3,
    )
    block_q = mlp_block_q_contributions(
        fixture["down_input"],
        fixture["down_weight"],
        capture.mlp_partition,
    )
    torch.testing.assert_close(head_q, capture.attention_head_q)
    torch.testing.assert_close(block_q, capture.mlp_block_q)


def test_mlp_probe_partitions_are_deterministic_balanced_and_exhaustive() -> None:
    contiguous = make_mlp_probe_partition(
        2048,
        scheme="contiguous",
        seed=123,
    )
    interleaved = make_mlp_probe_partition(
        2048,
        scheme="interleaved",
        seed=123,
    )
    repeated = make_mlp_probe_partition(
        2048,
        scheme="interleaved",
        seed=123,
    )
    different_seed = make_mlp_probe_partition(
        2048,
        scheme="interleaved",
        seed=124,
    )
    general = make_mlp_probe_partition(11, scheme="interleaved", seed=123)

    assert contiguous.block_sizes == (512, 512, 512, 512)
    assert interleaved.block_sizes == (512, 512, 512, 512)
    assert repeated == interleaved
    assert different_seed.channel_indices != interleaved.channel_indices
    assert contiguous.channel_indices != interleaved.channel_indices
    assert general.block_sizes == (3, 3, 3, 2)
    for manifest in (contiguous, interleaved, general):
        flattened = [
            channel
            for block in manifest.channel_indices
            for channel in block
        ]
        assert sorted(flattened) == list(range(manifest.intermediate_width))
        assert len(flattened) == len(set(flattened))


def test_causal_attention_summary_has_fixed_destination_free_features() -> None:
    valid = torch.tensor([[True, True, True, False]])
    positions = torch.tensor([[10, 12, 15, 99]])
    probabilities = _causal_probabilities(1, 4, 4, valid)
    summary = causal_attention_summaries(
        probabilities,
        query_valid_mask=valid,
        key_valid_mask=valid,
        query_positions=positions,
        key_positions=positions,
    )

    assert summary.feature_names == ATTENTION_SUMMARY_FEATURE_NAMES
    assert summary.features.shape == (1, 4, 4, 5)
    assert summary.maximum_future_mass == 0.0
    assert summary.maximum_invalid_key_mass == 0.0
    assert "no_destination_state" in summary.source_feature_contract
    assert torch.equal(summary.features[:, :, 3], torch.zeros(1, 4, 5))
    assert bool((summary.features[:, :, :3, 1:3] <= 1.0 + 1.0e-12).all())


def test_causal_attention_summary_rejects_future_and_padding_leakage() -> None:
    valid = torch.tensor([[True, True, True, False]])
    positions = torch.arange(4).view(1, 4)
    probabilities = _causal_probabilities(1, 4, 4, valid)
    future_leak = probabilities.clone()
    future_leak[0, 0, 1, 2] = 1.0e-3
    future_leak[0, 0, 1, 0] -= 1.0e-3
    with pytest.raises(ValueError, match="future-token mass"):
        causal_attention_summaries(
            future_leak,
            query_valid_mask=valid,
            key_valid_mask=valid,
            query_positions=positions,
            key_positions=positions,
        )

    key_valid = torch.tensor([[True, False, True, True]])
    padding_leak = _causal_probabilities(1, 4, 4, key_valid)
    query_valid = torch.ones_like(key_valid)
    padding_leak[0, 0, 2, 1] = 1.0e-3
    padding_leak[0, 0, 2, 0] -= 1.0e-3
    with pytest.raises(ValueError, match="invalid keys"):
        causal_attention_summaries(
            padding_leak,
            query_valid_mask=query_valid,
            key_valid_mask=key_valid,
            query_positions=positions,
            key_positions=positions,
        )


class _EagerAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        length = values.shape[1]
        probabilities = torch.eye(length).view(1, 1, length, length)
        return values, probabilities


def test_eager_probability_hook_is_removed_on_success_and_exception() -> None:
    attention = _EagerAttention()
    before = len(attention._forward_hooks)
    result, probabilities = run_with_eager_attention_probability_capture(
        attention,
        lambda: attention(torch.ones(1, 3, 2)),
    )
    assert isinstance(result, tuple)
    assert probabilities.shape == (1, 1, 3, 3)
    assert len(attention._forward_hooks) == before

    def fail_after_attention() -> None:
        attention(torch.ones(1, 3, 2))
        raise RuntimeError("teacher failure")

    with pytest.raises(RuntimeError, match="teacher failure"):
        run_with_eager_attention_probability_capture(attention, fail_after_attention)
    assert len(attention._forward_hooks) == before


def test_output_fisher_vjp_uses_detached_leaf_and_exact_valid_row_ids() -> None:
    generator = torch.Generator().manual_seed(201)
    prefix = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
    leaf = detach_layer4_output_as_leaf(prefix * 2.0)
    readout = torch.randn(3, 7, generator=generator)
    logits = leaf @ readout
    token_ids = torch.tensor([[1, 2, 3, 4], [2, 1, 5, 0]])
    valid = torch.tensor(
        [[True, True, True, True], [True, True, True, False]]
    )
    positions = torch.tensor([[4, 5, 6, 7], [20, 21, 22, 23]])

    result = causal_nll_output_fisher_vjp(
        logits,
        leaf,
        token_ids=token_ids,
        query_valid_mask=valid,
        logical_positions=positions,
        sequence_ids=("first", "second"),
    )

    assert leaf.is_leaf
    assert leaf.grad_fn is None
    assert prefix.grad is None
    assert result.supervised_token_count == 5
    assert result.valid_gradients.shape == (5, 3)
    assert result.row_ids == (
        FisherVJPRowID("first", 4, 5),
        FisherVJPRowID("first", 5, 6),
        FisherVJPRowID("first", 6, 7),
        FisherVJPRowID("second", 20, 21),
        FisherVJPRowID("second", 21, 22),
    )
    assert torch.equal(
        result.gradients[~result.valid_row_mask],
        torch.zeros_like(result.gradients[~result.valid_row_mask]),
    )


def test_validated_capture_rejects_a_broken_stage_identity() -> None:
    capture, _fixture = _exact_capture()
    with pytest.raises(ValueError, match="additive identities failed"):
        replace(
            capture,
            h_A=capture.h_A + 1.0,
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-8,
        )
