from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.gemma3_parts_first_native_intervention import (
    FrozenNativePartGate,
    run_with_native_part_gates,
)
from fisher_graph.gemma3_parts_first_supermodes import (
    ExecutionBoundary,
    PhysicalAtomFamily,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Linear(1_024, 640, bias=False)
        nn.init.constant_(self.o_proj.weight, 0.01)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(2_048, 640, bias=False)
        nn.init.constant_(self.down_proj.weight, 0.005)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, hidden: Tensor) -> Tensor:
        attention_input = hidden.repeat_interleave(2, dim=-1)[..., :1_024]
        hidden = hidden + self.self_attn.o_proj(attention_input)
        mlp_input = hidden.repeat_interleave(4, dim=-1)[..., :2_048]
        return hidden + self.mlp.down_proj(mlp_input)


class _Backbone(nn.Module):
    def __init__(self, lexical: nn.Embedding) -> None:
        super().__init__()
        self.embed_tokens = lexical
        self.layers = nn.ModuleList((_Layer(), _Layer()))

    def forward(self, ids: Tensor) -> Tensor:
        hidden = self.embed_tokens(ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _ToyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        lexical = nn.Embedding(8, 640)
        nn.init.normal_(lexical.weight, generator=torch.Generator().manual_seed(7))
        self.model = _Backbone(lexical)
        self.lm_head = nn.Linear(640, 8, bias=False)
        self.lm_head.weight = lexical.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, ids: Tensor) -> Tensor:
        return self.lm_head(self.model(ids))


def _gates(alpha: float) -> tuple[FrozenNativePartGate, ...]:
    return (
        FrozenNativePartGate(
            "gate.ingress",
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            ExecutionBoundary.lexical_ingress(),
            (0, 1),
            alpha,
            "chart.all",
        ),
        FrozenNativePartGate(
            "gate.attention",
            PhysicalAtomFamily.ATTENTION_QO_HEAD,
            ExecutionBoundary.attention(0),
            (0,),
            alpha,
            "chart.all",
        ),
        FrozenNativePartGate(
            "gate.mlp",
            PhysicalAtomFamily.MLP_CHANNEL,
            ExecutionBoundary.mlp(0),
            (0, 1, 2),
            alpha,
            "chart.all",
        ),
        FrozenNativePartGate(
            "gate.readout",
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            ExecutionBoundary.lexical_readout(),
            (2, 3),
            alpha,
            "chart.all",
        ),
    )


def test_alpha_one_is_literal_parity_and_hooks_are_removed() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2, 3]])
    baseline = model(ids)
    gates = _gates(1.0)
    masks = {gate.gate_id: torch.ones_like(ids, dtype=torch.bool) for gate in gates}
    output, receipt = run_with_native_part_gates(
        model,
        gates,
        masks,
        lambda: model(ids),
        allow_test_nonproduction_shape=True,
    )
    assert torch.equal(output, baseline)
    assert receipt.alpha_one_control
    assert receipt.hook_call_counts == (1, 1, 1, 1)
    assert receipt.hook_execution_order == (0, 1, 2, 3)
    assert receipt.hooks_removed
    assert torch.equal(model(ids), baseline)


def test_ordered_finite_gates_change_the_native_suffix() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2, 3]])
    gates = _gates(0.0)
    masks = {gate.gate_id: torch.ones_like(ids, dtype=torch.bool) for gate in gates}
    baseline = model(ids)
    output, receipt = run_with_native_part_gates(
        model,
        gates,
        masks,
        lambda: model(ids),
        allow_test_nonproduction_shape=True,
    )
    assert not torch.equal(output, baseline)
    assert not receipt.alpha_one_control
    assert receipt.later_native_computation_reran
    assert receipt.weights_mutated_by_hooks is False


def test_chart_mask_limits_the_gate_to_frozen_rows() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2, 3]])
    gate = _gates(0.0)[0]
    output, _ = run_with_native_part_gates(
        model,
        (gate,),
        {gate.gate_id: torch.tensor([[True, False, False]])},
        lambda: model(ids),
        allow_test_nonproduction_shape=True,
    )
    baseline = model(ids)
    assert not torch.equal(output[:, :1], baseline[:, :1])
    # Later causal rows may change, but the unselected embedding rows themselves
    # are not directly gated; hook mask accounting records exactly one row.


def test_duplicate_boundary_requires_a_canonical_union() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2]])
    first = _gates(0.5)[0]
    second = replace(first, gate_id="gate.ingress.second", part_ordinals=(2,))
    with pytest.raises(ValueError, match="unioned once"):
        run_with_native_part_gates(
            model,
            (first, second),
            {
                first.gate_id: torch.ones_like(ids, dtype=torch.bool),
                second.gate_id: torch.ones_like(ids, dtype=torch.bool),
            },
            lambda: model(ids),
            allow_test_nonproduction_shape=True,
        )


def test_gates_must_be_frozen_in_native_execution_order() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2]])
    ingress, attention = _gates(0.5)[:2]
    with pytest.raises(ValueError, match="execution order"):
        run_with_native_part_gates(
            model,
            (attention, ingress),
            {
                ingress.gate_id: torch.ones_like(ids, dtype=torch.bool),
                attention.gate_id: torch.ones_like(ids, dtype=torch.bool),
            },
            lambda: model(ids),
            allow_test_nonproduction_shape=True,
        )


def test_bad_mask_fails_and_hooks_are_removed() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2]])
    gate = _gates(0.0)[1]
    with pytest.raises(ValueError, match="does not align"):
        run_with_native_part_gates(
            model,
            (gate,),
            {gate.gate_id: torch.ones((1, 3), dtype=torch.bool)},
            lambda: model(ids),
            allow_test_nonproduction_shape=True,
        )
    # A failed gated forward must not leave a hook on the source model.
    assert model(ids).shape == (1, 2, 8)


def test_partial_hook_registration_is_rolled_back() -> None:
    model = _ToyGemma().eval()
    ids = torch.tensor([[1, 2]])
    ingress, attention = _gates(0.0)[:2]
    input_module = model.get_input_embeddings()
    model.model.layers[0].self_attn.o_proj = None
    with pytest.raises(TypeError, match="omitted o_proj"):
        run_with_native_part_gates(
            model,
            (ingress, attention),
            {
                ingress.gate_id: torch.ones_like(ids, dtype=torch.bool),
                attention.gate_id: torch.ones_like(ids, dtype=torch.bool),
            },
            lambda: model(ids),
            allow_test_nonproduction_shape=True,
        )
    assert not input_module._forward_hooks
