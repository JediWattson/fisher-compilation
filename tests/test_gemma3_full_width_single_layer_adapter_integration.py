import pytest
import torch

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.full_width_single_layer_executor import (
    FullWidthSingleLayerExecutor,
    FullWidthSingleLayerExecutorConfig,
)
from fisher_graph.gemma3_full_width_single_layer_experiment import (
    _attention_visibility_contract,
)
from test_gemma3_adapter import FakeGemma3ForCausalLM, FakeGemma3Layer


def test_full_width_executor_uses_real_gemma_boundary_and_mask_abi() -> None:
    torch.manual_seed(71_305)
    model = FakeGemma3ForCausalLM().eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    inputs = {
        "input_ids": torch.tensor(
            [
                [0, 2, 3],
                [4, 5, 6],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [False, True, True],
                [True, True, True],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
    }
    plan = adapter.plan_layer_block(0, 0)
    layer = adapter.layer(plan.layer_ids[0])
    assert layer.attention is not None
    assert layer.attention.kind == "sliding_causal"
    assert layer.attention.window_size == inputs["input_ids"].shape[1]

    captured = adapter.forward(
        inputs,
        capture_sites=plan.activation_sites,
    )
    sequence = adapter.prepare_sequence(inputs)
    embedded = adapter.embed(inputs, sequence).hidden_states
    torch.testing.assert_close(
        captured.activations[plan.activation_sites[0]],
        embedded,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        sequence.query_valid_mask,
        inputs["attention_mask"],
    )
    assert torch.equal(
        sequence.key_valid_mask,
        inputs["attention_mask"],
    )
    assert torch.equal(
        sequence.logical_positions,
        inputs["position_ids"].expand(2, -1),
    )

    native_boundary = adapter.run_segment(
        adapter.segments[plan.start_ordinal],
        embedded,
        sequence,
    ).hidden_states
    torch.testing.assert_close(
        native_boundary,
        captured.activations[plan.activation_sites[-1]],
        rtol=0.0,
        atol=0.0,
    )
    first_layer = model.model.layers[0]
    assert isinstance(first_layer, FakeGemma3Layer)
    additive_mask = first_layer.last_attention_mask
    assert additive_mask is not None
    assert additive_mask.shape == (2, 1, 3, 3)
    minimum = torch.finfo(additive_mask.dtype).min
    assert additive_mask[0, 0, 2, 0].item() == minimum
    assert additive_mask[1, 0, 0, 1].item() == minimum
    assert additive_mask[1, 0, 2, 0].item() == 0.0

    replay = native_boundary
    for segment in adapter.segments[plan.end_ordinal + 1 :]:
        replay = adapter.run_segment(
            segment,
            replay,
            sequence,
        ).hidden_states
    replay_logits = adapter.project_logits(replay, sequence)
    torch.testing.assert_close(
        replay_logits,
        captured.logits,
        rtol=0.0,
        atol=0.0,
    )

    executor = FullWidthSingleLayerExecutor(
        FullWidthSingleLayerExecutorConfig(
            residual_width=plan.widths[0],
            hidden_width=8,
            layer_count=1,
            head_count=2,
            feed_forward_width=12,
        )
    ).eval()
    result = executor.forward_components(
        captured.activations[plan.activation_sites[0]],
        sequence,
    )
    assert result.output.shape == native_boundary.shape
    assert result.demanded_coordinates.shape == (
        int(inputs["attention_mask"].sum().item()),
        plan.widths[-1],
    )
    assert torch.equal(
        result.output[~inputs["attention_mask"]],
        captured.activations[plan.activation_sites[0]][
            ~inputs["attention_mask"]
        ],
    )
    expected_indices = (
        inputs["attention_mask"].reshape(-1)
        .nonzero(as_tuple=False)
        .flatten()
    )
    assert torch.equal(result.demanded_flat_indices, expected_indices)

    accepted = _attention_visibility_contract(
        adapter,
        layer_id=plan.layer_ids[0],
        maximum_length=3,
    )
    assert accepted["passed"] is True
    with pytest.raises(ValueError, match="visibility-equivalent"):
        _attention_visibility_contract(
            adapter,
            layer_id=plan.layer_ids[0],
            maximum_length=4,
        )
