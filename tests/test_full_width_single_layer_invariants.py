import pytest
import torch
from torch import nn

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.full_width_single_layer_executor import (
    FullWidthSingleLayerExecutor,
    FullWidthSingleLayerExecutorConfig,
)
from fisher_graph.gemma3_full_width_single_layer_experiment import (
    _assert_source_independence,
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


def _executor() -> FullWidthSingleLayerExecutor:
    torch.manual_seed(91_104)
    return FullWidthSingleLayerExecutor(
        FullWidthSingleLayerExecutorConfig(
            residual_width=6,
            hidden_width=8,
            layer_count=2,
            head_count=2,
            feed_forward_width=12,
            causal_edges_enabled=True,
        )
    ).eval()


def test_future_positions_have_exactly_zero_gradient_influence() -> None:
    executor = _executor()
    sequence = _sequence(torch.ones(2, 7, dtype=torch.bool))
    hidden = torch.randn(2, 7, 6, requires_grad=True)

    earlier_outputs = executor(hidden, sequence)[:, :3].sum()
    gradient = torch.autograd.grad(earlier_outputs, hidden)[0]

    torch.testing.assert_close(
        gradient[:, 3:],
        torch.zeros_like(gradient[:, 3:]),
        rtol=0.0,
        atol=0.0,
    )


def test_appended_padding_and_its_values_cannot_change_valid_outputs() -> None:
    executor = _executor()
    compact_hidden = torch.randn(2, 4, 6)
    compact_sequence = _sequence(torch.ones(2, 4, dtype=torch.bool))
    compact_output = executor(compact_hidden, compact_sequence)

    padded_mask = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, False, False, False],
        ]
    )
    padded_sequence = _sequence(padded_mask)
    padded_hidden = torch.cat(
        (compact_hidden, torch.randn(2, 3, 6)),
        dim=1,
    )
    perturbed_hidden = padded_hidden.clone()
    perturbed_hidden[:, 4:] = 1_000_000 * torch.randn(2, 3, 6)

    padded_output = executor(padded_hidden, padded_sequence)
    perturbed_output = executor(perturbed_hidden, padded_sequence)

    torch.testing.assert_close(
        padded_output[:, :4],
        compact_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        perturbed_output[:, :4],
        padded_output[:, :4],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(padded_output[:, 4:], padded_hidden[:, 4:])
    assert torch.equal(perturbed_output[:, 4:], perturbed_hidden[:, 4:])


def test_source_independence_rejects_shared_tensor_storage() -> None:
    source = nn.Linear(6, 6)
    independent = nn.Linear(6, 6)
    audit = _assert_source_independence(
        source,
        {"independent": independent},
    )
    assert audit["passed"] is True

    aliased = nn.Linear(6, 6)
    aliased.weight = nn.Parameter(source.weight.detach())
    with pytest.raises(RuntimeError, match="tensor storage"):
        _assert_source_independence(source, {"aliased": aliased})
