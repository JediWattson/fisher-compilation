import copy
from pathlib import Path
import tempfile

import pytest
import torch

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.dynamic_executor import StatefulCausalModalGraph
from fisher_graph.static_span_executor import StaticSpanBlockExecutor


def _sequence(
    query_mask: torch.Tensor,
    key_mask: torch.Tensor | None = None,
) -> SequenceContext:
    if key_mask is None:
        key_mask = query_mask
    positions = torch.arange(
        query_mask.shape[1],
        dtype=torch.long,
        device=query_mask.device,
    ).unsqueeze(0).expand(query_mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=query_mask,
        key_valid_mask=key_mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _executor(
    *,
    width: int = 5,
    rank: int = 3,
    seed: int = 701,
) -> StaticSpanBlockExecutor:
    torch.manual_seed(seed)
    graph = StatefulCausalModalGraph(
        input_modes=width,
        output_modes=rank,
        state_channels=2,
        routing_width=4,
        activation="tanh",
    )
    decoder = torch.linalg.qr(torch.randn(width, rank)).Q
    return StaticSpanBlockExecutor(
        graph=graph,
        decoder=decoder,
        input_activation_name="layer.0.input",
        output_activation_name="layer.2.output",
    )


def test_residual_decode_and_query_passthrough_are_exact() -> None:
    graph = StatefulCausalModalGraph(
        input_modes=3,
        output_modes=2,
        state_channels=1,
        routing_width=2,
        activation="identity",
    )
    with torch.no_grad():
        for parameter in graph.parameters():
            parameter.zero_()
        graph.output_bias.copy_(torch.tensor([0.5, -1.0]))
    decoder = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, -1.0],
        ]
    )
    executor = StaticSpanBlockExecutor(
        graph=graph,
        decoder=decoder,
        input_activation_name="span.input",
        output_activation_name="span.output",
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    query = torch.tensor([[False, True, False, True]])
    keys = torch.ones_like(query)
    sequence = _sequence(query, keys)

    result = executor.forward_components(hidden, sequence)

    expected_delta = torch.tensor([0.5, -2.0, 1.5])
    torch.testing.assert_close(
        result.modal_delta[query],
        torch.tensor([[0.5, -1.0], [0.5, -1.0]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.output[query],
        hidden[query] + expected_delta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.output[~query],
        hidden[~query],
        rtol=0.0,
        atol=0.0,
    )
    assert executor.width == 3
    assert executor.retained_rank == 2
    assert executor.fixed_runtime_coefficient_count == 6
    assert (
        executor.runtime_stored_coefficient_count
        == executor.learned_parameter_count + 6
    )
    assert executor.executor_local_source_free
    assert executor.supports_query_sparse_prefill
    assert not executor.reference_kernel_physically_query_sparse
    assert not executor.reference_kernel_speed_claim


def test_output_delta_is_structurally_confined_to_decoder_span() -> None:
    executor = _executor().eval()
    hidden = torch.randn(2, 6, executor.width)
    query = torch.tensor(
        [
            [False, False, True, False, False, False],
            [False, False, False, False, True, False],
        ]
    )
    keys = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    result = executor.forward_components(hidden, _sequence(query, keys))
    delta = (result.output - hidden)[query]
    projector = executor.decoder @ executor.decoder.T

    torch.testing.assert_close(
        delta,
        delta @ projector,
        rtol=0.0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.output[~query],
        hidden[~query],
        rtol=0.0,
        atol=0.0,
    )


def test_sparse_answer_query_cannot_read_future_tensor_slots() -> None:
    executor = _executor().eval()
    query = torch.tensor([[False, False, False, True, False, False]])
    keys = torch.ones_like(query)
    sequence = _sequence(query, keys)
    baseline = torch.randn(1, 6, executor.width)
    changed = baseline.clone()
    changed[:, 4:] += 100.0

    first = executor.forward_context(
        baseline,
        sequence=sequence,
        prefix="compiled.span",
    )
    second = executor.forward_context(
        changed,
        sequence=sequence,
        prefix="compiled.span",
    )

    torch.testing.assert_close(
        first[:, 3],
        second[:, 3],
        rtol=0.0,
        atol=0.0,
    )


def test_strict_weights_only_artifact_roundtrip_and_tamper_rejection() -> None:
    executor = _executor().eval()
    state = executor.artifact_state_dict()
    original_decoder = executor.decoder.clone()
    state["decoder"][0, 0] += 10.0
    torch.testing.assert_close(
        executor.decoder,
        original_decoder,
        rtol=0.0,
        atol=0.0,
    )
    state = executor.artifact_state_dict()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "static-span.pt"
        torch.save(state, path)
        weights_only_state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    loaded = StaticSpanBlockExecutor.from_artifact_state_dict(
        weights_only_state
    )

    assert loaded.execution_fingerprint() == executor.execution_fingerprint()
    assert loaded.retained_rank == 3
    assert loaded.width == 5
    hidden = torch.randn(2, 5, 5)
    query = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, False, True, False],
        ]
    )
    keys = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    sequence = _sequence(query, keys)
    torch.testing.assert_close(
        loaded.forward_context(hidden, sequence=sequence),
        executor.forward_context(hidden, sequence=sequence),
        rtol=0.0,
        atol=0.0,
    )
    assert not hasattr(loaded, "source")
    assert not hasattr(loaded, "fallback")
    assert all(
        "source" not in name and "fallback" not in name
        for name in loaded.state_dict()
    )

    changed = copy.deepcopy(state)
    changed["decoder"][0, 0] += 0.25
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StaticSpanBlockExecutor.from_artifact_state_dict(changed)

    changed = copy.deepcopy(state)
    changed["graph_state_dict"]["output_bias"][0] += 0.25
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StaticSpanBlockExecutor.from_artifact_state_dict(changed)

    changed = copy.deepcopy(state)
    changed["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        StaticSpanBlockExecutor.from_artifact_state_dict(changed)


def test_constructor_and_context_guards_fail_closed() -> None:
    graph = StatefulCausalModalGraph(
        input_modes=4,
        output_modes=2,
        state_channels=1,
        routing_width=3,
    )
    with pytest.raises(ValueError, match="decoder rank"):
        StaticSpanBlockExecutor(
            graph=graph,
            decoder=torch.zeros(4, 3),
            input_activation_name="span.input",
            output_activation_name="span.output",
        )
    executor = StaticSpanBlockExecutor(
        graph=graph,
        decoder=torch.zeros(4, 2),
        input_activation_name="span.input",
        output_activation_name="span.output",
    )
    query = torch.ones(1, 3, dtype=torch.bool)
    sequence = _sequence(query)
    sequence.phase = "decode"
    with pytest.raises(ValueError, match="cached decode"):
        executor.forward_context(
            torch.zeros(1, 3, 4),
            sequence=sequence,
        )
