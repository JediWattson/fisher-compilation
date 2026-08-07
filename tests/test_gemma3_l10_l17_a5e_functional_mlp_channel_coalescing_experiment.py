from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph import (
    gemma3_all_mode_generator_graph_executor as all_mode_graph,
    gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment as a5e,
)


Gemma3PhysicalCompactMLP = a5e.Gemma3PhysicalCompactMLP
_FrozenResidual = a5e._FrozenResidual
_materialize_all_modes = a5e._materialize_all_modes
_temporary_mlp_overlay = a5e._temporary_mlp_overlay


class _ToyMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 6) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class _FakeAdapter:
    def __init__(self) -> None:
        layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mlp": _ToyMLP(),
                        "post_feedforward_layernorm": nn.Identity(),
                    }
                )
                for _ in range(18)
            ]
        )
        # ModuleDict does not support attribute access, while Gemma layers do.
        materialized = nn.ModuleList()
        for layer in layers:
            wrapper = nn.Module()
            wrapper.mlp = layer["mlp"]
            wrapper.post_feedforward_layernorm = layer[
                "post_feedforward_layernorm"
            ]
            materialized.append(wrapper)
        self.module = SimpleNamespace(model=SimpleNamespace(layers=materialized))


def test_physical_compact_mlp_has_real_narrow_shapes_and_manual_forward() -> None:
    torch.manual_seed(7)
    gate = torch.randn(3, 4)
    up = torch.randn(3, 4)
    down = torch.randn(4, 3)
    candidate = Gemma3PhysicalCompactMLP(
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        activation=nn.GELU(approximate="tanh"),
    )
    inputs = torch.randn(2, 5, 4)
    expected = (nn.GELU(approximate="tanh")(inputs @ gate.T) * (inputs @ up.T)) @ down.T
    assert candidate.gate_proj.weight.shape == (3, 4)
    assert candidate.up_proj.weight.shape == (3, 4)
    assert candidate.down_proj.weight.shape == (4, 3)
    assert candidate.intermediate_width == 3
    assert torch.allclose(candidate(inputs), expected)


def test_all_mode_compilation_is_source_free_full_width_and_exact() -> None:
    torch.manual_seed(11)
    source = _ToyMLP()
    compiled = _materialize_all_modes(source)
    inputs = torch.randn(2, 5, 4)

    assert compiled.mode_count == source.gate_proj.out_features
    assert compiled.interaction_count == 0
    assert compiled.topological_frontier_count == 1
    assert compiled.gate_proj.weight.data_ptr() != source.gate_proj.weight.data_ptr()
    assert compiled.up_proj.weight.data_ptr() != source.up_proj.weight.data_ptr()
    assert compiled.down_proj.weight.data_ptr() != source.down_proj.weight.data_ptr()
    assert torch.equal(compiled(inputs), source(inputs))
    execution = compiled.execute_graph(inputs, capture_modal_states=True)
    expected_states = source.act_fn(source.gate_proj(inputs)) * source.up_proj(
        inputs
    )
    assert execution.node_count == 6
    assert execution.interaction_count == 0
    assert execution.modal_states is not None
    assert torch.equal(execution.modal_states, expected_states)


def test_chart_graph_routes_hidden_state_through_local_edge_messages() -> None:
    base_weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]
    )
    base_bias = torch.tensor([0.1, -0.2, 0.3])
    decoder = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    bases = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    scales = torch.ones(2)
    edges = torch.tensor(
        [
            [[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, -0.2, 0.0], [0.0, 0.5, 0.0]],
        ]
    )
    graph = all_mode_graph.compile_chart_conditioned_all_mode_generator_graph(
        affine_weight=base_weight,
        affine_bias=base_bias,
        decoder_weight=decoder,
        chart_centers=centers,
        chart_bases=bases,
        chart_distance_scales=scales,
        chart_edge_weight=edges,
        chart_temperature=1.0,
        model_fingerprint="1" * 64,
        layer_ordinal=10,
        fit_split_sha256="2" * 64,
    )
    inputs = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    execution = graph.execute_graph(
        inputs,
        capture_modal_states=True,
        capture_chart_states=True,
    )
    displacement = inputs[:, None, :] - centers
    memberships = (
        -0.5 * displacement.square().sum(dim=-1)
    ).softmax(dim=-1)
    coordinates = torch.einsum("nch,chr->ncr", displacement, bases)
    augmented = torch.cat(
        (torch.ones_like(coordinates[..., :1]), coordinates),
        dim=-1,
    )
    messages = torch.einsum("ncr,crk->nck", augmented, edges)
    states = inputs @ base_weight.T + base_bias
    states = states + (memberships[..., None] * messages).sum(dim=1)
    expected = states @ decoder.T

    assert execution.modal_states is not None
    assert execution.chart_memberships is not None
    assert execution.chart_coordinates is not None
    assert torch.allclose(execution.chart_memberships.sum(dim=1), torch.ones(2))
    assert execution.chart_memberships[0, 0] > execution.chart_memberships[0, 1]
    assert execution.chart_memberships[1, 1] > execution.chart_memberships[1, 0]
    assert torch.allclose(execution.modal_states, states)
    assert torch.allclose(execution.output, expected)
    assert execution.node_count == 3
    assert execution.interaction_count == 6
    assert execution.topological_frontier_count == 2
    assert graph.graph_metadata()["all_modes_retained"] is True
    assert graph.graph_metadata()["stored_coefficient_count"] > (
        graph.learned_parameter_count
    )
    assert graph.routing_blend_macs_per_token == 6
    assert graph.estimated_total_macs_per_token == (
        graph.matrix_macs_per_token + 6
    )


def test_native_residual_overlay_calls_original_projections_and_restores() -> None:
    adapter = _FakeAdapter()
    originals = {
        ordinal: adapter.module.model.layers[ordinal].mlp
        for ordinal in (10, 17)
    }
    calls = {
        ordinal: {name: 0 for name in ("gate_proj", "up_proj", "down_proj")}
        for ordinal in originals
    }
    handles = []
    for ordinal, mlp in originals.items():
        for name in calls[ordinal]:
            def increment(
                _module: nn.Module,
                _args: tuple[Tensor, ...],
                _output: Tensor,
                *,
                layer: int = ordinal,
                projection: str = name,
            ) -> None:
                calls[layer][projection] += 1

            handles.append(getattr(mlp, name).register_forward_hook(increment))
    residuals = {
        ordinal: _FrozenResidual(
            input_mean=torch.zeros(1, 4),
            output_mean=torch.zeros(1, 4),
            input_to_latent=torch.zeros(4, 2),
            latent_to_output=torch.zeros(2, 4),
        )
        for ordinal in originals
    }
    inputs = torch.randn(2, 3, 4)
    try:
        with _temporary_mlp_overlay(
            adapter,  # type: ignore[arg-type]
            originals,
            residuals=residuals,
        ):
            for ordinal in originals:
                layer = adapter.module.model.layers[ordinal]
                raw = layer.mlp(inputs)
                observed = layer.post_feedforward_layernorm(raw)
                assert observed.shape == inputs.shape
        assert all(
            adapter.module.model.layers[ordinal].mlp is original
            for ordinal, original in originals.items()
        )
        assert all(
            adapter.module.model.layers[ordinal].post_feedforward_layernorm.__class__
            is nn.Identity
            for ordinal in originals
        )
        assert all(count == 1 for layer in calls.values() for count in layer.values())
    finally:
        for handle in handles:
            handle.remove()


def test_overlay_restores_after_callback_error() -> None:
    adapter = _FakeAdapter()
    original = adapter.module.model.layers[10].mlp
    with pytest.raises(RuntimeError, match="boom"):
        with _temporary_mlp_overlay(  # type: ignore[arg-type]
            adapter,
            {10: _ToyMLP()},
        ):
            raise RuntimeError("boom")
    assert adapter.module.model.layers[10].mlp is original
