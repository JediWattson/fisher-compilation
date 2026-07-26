import copy

import pytest
import torch

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.full_width_single_layer_executor import (
    FullWidthSingleLayerExecutor,
    FullWidthSingleLayerExecutorConfig,
)


def _sequence(mask: torch.Tensor) -> SequenceContext:
    positions = torch.arange(mask.shape[1], dtype=torch.long).unsqueeze(
        0
    ).expand(mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
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
    causal_edges_enabled: bool,
) -> FullWidthSingleLayerExecutor:
    torch.manual_seed(9104)
    return FullWidthSingleLayerExecutor(
        FullWidthSingleLayerExecutorConfig(
            residual_width=6,
            hidden_width=8,
            layer_count=2,
            head_count=2,
            feed_forward_width=12,
            causal_edges_enabled=causal_edges_enabled,
        )
    ).eval()


def test_full_width_identity_decoder_and_invalid_row_passthrough() -> None:
    executor = _executor(causal_edges_enabled=True)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    sequence = _sequence(mask)
    hidden = torch.randn(2, 5, 6)
    result = executor.forward_components(hidden, sequence)

    assert executor.retained_rank == executor.width == 6
    assert torch.equal(executor.executor.decoder, torch.eye(6))
    assert result.demanded_coordinates.shape == (7, 6)
    torch.testing.assert_close(
        result.demanded_delta,
        result.demanded_coordinates,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(result.output[~mask], hidden[~mask])
    assert executor.executor_local_source_free
    assert not executor.owns_source_model_weights
    assert not executor.owns_source_fallback


def test_causal_student_reads_prefix_but_not_future() -> None:
    executor = _executor(causal_edges_enabled=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(1, 5, 6)
    baseline = executor(hidden, sequence)

    prefix_changed = hidden.clone()
    prefix_changed[:, 0] += 3.0
    future_changed = hidden.clone()
    future_changed[:, 4] += 1_000.0

    assert not torch.allclose(
        executor(prefix_changed, sequence)[:, 3],
        baseline[:, 3],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        executor(future_changed, sequence)[:, :4],
        baseline[:, :4],
        rtol=0.0,
        atol=0.0,
    )


def test_storage_matched_control_has_no_cross_position_influence() -> None:
    executor = _executor(causal_edges_enabled=False)
    mask = torch.ones(1, 5, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(1, 5, 6)
    changed = hidden.clone()
    changed[:, 0] += 100.0

    baseline = executor(hidden, sequence)
    candidate = executor(changed, sequence)

    torch.testing.assert_close(
        candidate[:, 1:],
        baseline[:, 1:],
        rtol=0.0,
        atol=0.0,
    )
    assert executor.causal_edge_control == (
        "attention_output_zeroed_storage_matched"
    )
    assert executor.learned_parameter_count == _executor(
        causal_edges_enabled=True
    ).learned_parameter_count


def test_strict_artifact_roundtrip_binds_causal_control() -> None:
    executor = _executor(causal_edges_enabled=False)
    mask = torch.ones(2, 4, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(2, 4, 6)
    expected = executor(hidden, sequence)
    artifact = executor.artifact_state_dict()

    restored = FullWidthSingleLayerExecutor.from_artifact_state_dict(
        artifact
    )
    assert not restored.config.causal_edges_enabled
    assert restored.execution_fingerprint() == executor.execution_fingerprint()
    torch.testing.assert_close(
        restored(hidden, sequence),
        expected,
        rtol=0.0,
        atol=0.0,
    )

    changed = copy.deepcopy(artifact)
    changed["causal_edges_enabled"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        FullWidthSingleLayerExecutor.from_artifact_state_dict(changed)

    reduced = copy.deepcopy(artifact)
    reduced["executor"]["config"]["retained_rank"] = 5
    with pytest.raises(ValueError):
        FullWidthSingleLayerExecutor.from_artifact_state_dict(reduced)

    non_identity = copy.deepcopy(artifact)
    non_identity["executor"]["model_state_dict"]["decoder"][0, 0] = 0.0
    with pytest.raises(ValueError):
        FullWidthSingleLayerExecutor.from_artifact_state_dict(non_identity)
