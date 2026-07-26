import copy

import pytest
import torch

from fisher_graph.adapters.base import (
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.static_transformer_span_executor import (
    StaticTransformerSpanExecutor,
    StaticTransformerSpanExecutorConfig,
)


def _sequence(answer_positions: tuple[int, ...], length: int) -> SequenceContext:
    batch_size = len(answer_positions)
    positions = torch.arange(length, dtype=torch.long).unsqueeze(0).expand(
        batch_size,
        -1,
    )
    tensor_positions = torch.arange(length).unsqueeze(0)
    answers = torch.tensor(answer_positions).unsqueeze(1)
    query_valid = tensor_positions.eq(answers)
    key_valid = tensor_positions.le(answers)
    return SequenceContext(
        query_valid_mask=query_valid,
        key_valid_mask=key_valid,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
        cache_state=None,
        adapter_payload=None,
    )


def _executor(
    *,
    residual_width: int = 6,
    hidden_width: int = 8,
    layer_count: int = 2,
    retained_rank: int = 3,
) -> StaticTransformerSpanExecutor:
    torch.manual_seed(1409)
    decoder, _ = torch.linalg.qr(
        torch.randn(residual_width, retained_rank),
        mode="reduced",
    )
    return StaticTransformerSpanExecutor(
        StaticTransformerSpanExecutorConfig(
            residual_width=residual_width,
            hidden_width=hidden_width,
            layer_count=layer_count,
            head_count=2,
            feed_forward_width=12,
            retained_rank=retained_rank,
        ),
        decoder,
    )


def test_residual_math_and_answer_only_query_passthrough() -> None:
    executor = _executor().eval()
    sequence = _sequence((2, 4), 6)
    hidden = torch.randn(2, 6, 6)
    original = hidden.clone()

    result = executor.forward_components(hidden, sequence)
    flat_input = hidden.reshape(-1, 6)
    selected_input = flat_input.index_select(
        0,
        result.demanded_flat_indices,
    )

    assert result.demanded_coordinates.shape == (2, 3)
    assert result.demanded_delta.shape == (2, 6)
    assert result.dense_prefix_length == 5
    torch.testing.assert_close(
        result.output[sequence.query_valid_mask],
        selected_input + result.demanded_delta,
    )
    assert torch.equal(
        result.output[~sequence.query_valid_mask],
        original[~sequence.query_valid_mask],
    )


def test_two_block_prefix_transport_is_causal_and_future_invariant() -> None:
    executor = _executor(layer_count=2).eval()
    sequence = _sequence((3,), 6)
    baseline = torch.randn(1, 6, 6)

    prefix_changed = baseline.clone()
    prefix_changed[:, 0] += 2.0
    future_changed = baseline.clone()
    future_changed[:, 4:] += 10_000.0

    baseline_output = executor(baseline, sequence)
    prefix_output = executor(prefix_changed, sequence)
    future_output = executor(future_changed, sequence)

    assert not torch.allclose(
        prefix_output[:, 3],
        baseline_output[:, 3],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        future_output[:, 3],
        baseline_output[:, 3],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        future_output[:, 4:],
        future_changed[:, 4:],
    )


def test_every_predicted_delta_is_confined_to_fixed_decoder_span() -> None:
    executor = _executor(
        residual_width=7,
        hidden_width=8,
        retained_rank=2,
    ).eval()
    sequence = _sequence((1, 4, 5), 6)
    hidden = torch.randn(3, 6, 7)
    result = executor.forward_components(hidden, sequence)

    decoder = executor.decoder
    orthogonal_projector = (
        torch.eye(7) - decoder @ decoder.T
    )
    off_span = result.demanded_delta @ orthogonal_projector
    torch.testing.assert_close(
        off_span,
        torch.zeros_like(off_span),
        rtol=0.0,
        atol=2e-6,
    )


def test_parameter_storage_and_logical_mac_counts_are_exact() -> None:
    executor = _executor().eval()
    assert (
        executor.learned_parameter_count
        == executor.expected_learned_parameter_count
    )
    assert executor.fixed_runtime_coefficient_count == 6 * 3
    assert executor.total_runtime_coefficient_count == (
        executor.learned_parameter_count + 18
    )
    assert executor.executor_local_source_free
    assert not executor.owns_source_model_weights
    assert not executor.owns_source_fallback
    assert executor.requires_terminal_query_demand

    sequence = _sequence((2, 4), 6)
    accounting = executor.logical_accounting(sequence)
    assert accounting.valid_key_tokens == 8
    assert accounting.demanded_query_tokens == 2
    assert accounting.logical_causal_key_pairs == 21
    assert accounting.dense_prefix_length == 5
    assert accounting.reference_dense_prefix_rows == 10
    assert accounting.reference_dense_attention_pairs == 50
    assert accounting.logical_total_macs == (
        accounting.input_projection_macs
        + accounting.transformer_trunk_macs
        + accounting.output_head_macs
        + accounting.decoder_macs
    )
    assert (
        accounting.reference_dense_prefix_total_macs
        > accounting.logical_total_macs
    )


def test_strict_weights_only_artifact_roundtrip_and_tamper_rejection(
    tmp_path,
) -> None:
    executor = _executor().eval()
    sequence = _sequence((2, 4), 6)
    hidden = torch.randn(2, 6, 6)
    expected = executor(hidden, sequence)
    artifact = executor.artifact_state_dict()
    artifact_path = tmp_path / "static-transformer-span.pt"
    torch.save(artifact, artifact_path)
    loaded = torch.load(artifact_path, weights_only=True)

    restored = StaticTransformerSpanExecutor.from_artifact_state_dict(
        loaded
    )
    assert restored.execution_fingerprint() == executor.execution_fingerprint()
    torch.testing.assert_close(restored(hidden, sequence), expected)
    assert artifact["contains_source_model_weights"] is False
    assert artifact["contains_source_fallback"] is False

    changed = copy.deepcopy(artifact)
    changed["model_state_dict"]["output_head.bias"][0] += 0.25
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StaticTransformerSpanExecutor.from_artifact_state_dict(changed)

    unknown = copy.deepcopy(artifact)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="artifact fields"):
        StaticTransformerSpanExecutor.from_artifact_state_dict(unknown)

    wrong_shape = copy.deepcopy(artifact)
    wrong_shape["model_state_dict"]["decoder"] = torch.zeros(99, 3)
    with pytest.raises(ValueError, match="decoder"):
        StaticTransformerSpanExecutor.from_artifact_state_dict(wrong_shape)

    unsafe = copy.deepcopy(artifact)
    unsafe["contains_source_fallback"] = True
    with pytest.raises(ValueError, match="unsupported"):
        StaticTransformerSpanExecutor.from_artifact_state_dict(unsafe)
