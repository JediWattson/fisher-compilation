"""Incremental Gemma execution for computational-coordinate generator graphs.

The generic modal graph executor receives every boundary input at once.  A
decoder-only transformer exposes those boundaries incrementally as its native
layer walk progresses.  This module bridges the two execution models:

* every graph node is authenticated against one modal-generator lowering and
  one parameter-cluster layer fragment;
* graph tensors are copied once into frozen, device-local parameters;
* each affected Gemma MLP physically omits the union of fragment gate/up rows
  and down columns;
* the layer overlay advances an incremental graph session using the native
  normalized MLP input; and
* all graph contributions at that layer are summed with the retained native
  MLP output.

An optional second, non-owning graph can read the same normalized input and
add a zero-mean residual after the live post-feed-forward RMSNorm.  The owning
graph still controls native-channel deletion; omitting the additive graph is
bit-identical to the historical executor path.

The ``deletion`` condition uses the identical compact native MLPs while
suppressing the graph.  It is therefore architecture-matched and never reads
the removed native coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .gemma3_modal_generator_executor import (
    _EmptyLinear,
    _apply_activation,
    _storage_pointers,
    _validate_source_mlp,
)
from .modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphExecution,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
    StateConditionedModalGeneratorInteraction,
    _conditional_outgoing_groups,
)
from .modal_generator_lowering import ModalGeneratorLowering
from .parameter_cluster_fragments import ParameterClusterLayerFragment


_CONDITIONS = frozenset(("generated", "deletion"))
_RUNTIME_STORAGE = "registered_copied_device_local_graph_parameters"
_COMPACT_ROW_CHUNK_SIZE = 2048
_CallbackResult = TypeVar("_CallbackResult")
_DiagnosticCorrectionProvider = Callable[[Tensor], Tensor]


def _frozen_parameter(
    value: Tensor,
    *,
    like: Tensor,
    dtype: torch.dtype | None = None,
) -> nn.Parameter:
    copied = value.detach().to(
        device=like.device,
        dtype=like.dtype if dtype is None else dtype,
    ).contiguous().clone()
    if not bool(torch.isfinite(copied).all()):
        raise ValueError("graph coefficient became non-finite after device cast")
    return nn.Parameter(copied, requires_grad=False)


class _DeviceNodeRuntime(nn.Module):
    """One copied coordinate generator and decoder."""

    def __init__(
        self,
        weights: LinearModalGeneratorNodeWeights,
        *,
        like: Tensor,
    ) -> None:
        super().__init__()
        if weights.state_kind != "computational_mode_coordinates":
            raise ValueError(
                "Gemma graph nodes must expose computational-mode coordinates"
            )
        self.input_factor = _frozen_parameter(
            weights.input_factor,
            like=like,
        )
        self.register_parameter(
            "state_factor",
            (
                None
                if weights.state_factor is None
                else _frozen_parameter(weights.state_factor, like=like)
            ),
        )
        self.output_factor = _frozen_parameter(
            weights.output_factor,
            like=like,
        )
        self.register_parameter(
            "latent_bias",
            (
                None
                if weights.latent_bias is None
                else _frozen_parameter(weights.latent_bias, like=like)
            ),
        )
        self.register_parameter(
            "output_bias",
            (
                None
                if weights.output_bias is None
                else _frozen_parameter(weights.output_bias, like=like)
            ),
        )
        self.input_width = weights.input_width
        self.private_width = weights.private_width
        self.state_width = weights.latent_width
        self.output_width = weights.output_width

    def generate(self, values: Tensor) -> Tensor:
        if (
            values.device != self.input_factor.device
            or values.dtype != self.input_factor.dtype
            or values.shape[-1] != self.input_width
        ):
            raise ValueError(
                "Gemma graph input device, dtype, or width drifted"
            )
        state = values @ self.input_factor
        if self.state_factor is not None:
            state = state @ self.state_factor
        if self.latent_bias is not None:
            state = state + self.latent_bias.to(
                device=state.device,
                dtype=state.dtype,
            )
        return state

    def decode(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.state_width:
            raise ValueError("Gemma modal state width drifted")
        result = state @ self.output_factor
        if self.output_bias is not None:
            result = result + self.output_bias.to(
                device=result.device,
                dtype=result.dtype,
            )
        return result


class _DeviceEdgeRuntime(nn.Module):
    """One copied affine graph interaction."""

    def __init__(
        self,
        interaction: ModalGeneratorInteraction,
        *,
        like: Tensor,
    ) -> None:
        super().__init__()
        self.message_matrix = _frozen_parameter(
            interaction.message_matrix,
            like=like,
        )
        self.message_bias = _frozen_parameter(
            interaction.message_bias,
            like=like,
        )
        self.source_width = interaction.source_width
        self.target_width = interaction.target_width

    def forward(self, source_state: Tensor) -> Tensor:
        if (
            source_state.device != self.message_matrix.device
            or source_state.shape[-1] != self.source_width
        ):
            raise ValueError("Gemma graph interaction runtime state drifted")
        result = source_state @ self.message_matrix
        return result + self.message_bias.to(
            device=result.device,
            dtype=result.dtype,
        )


class _DeviceStateConditionedEdgeRuntime(nn.Module):
    """Copied polynomial proposal and source-only routing coefficients."""

    def __init__(
        self,
        interaction: StateConditionedModalGeneratorInteraction,
        *,
        like: Tensor,
    ) -> None:
        super().__init__()
        self.message_matrix = _frozen_parameter(
            interaction.message_matrix,
            like=like,
        )
        self.message_bias = _frozen_parameter(
            interaction.message_bias,
            like=like,
        )
        routing_dtype = (
            torch.float32
            if like.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else like.dtype
        )
        self.gate_weight = _frozen_parameter(
            interaction.gate_weight,
            like=like,
            dtype=routing_dtype,
        )
        self.gate_bias = _frozen_parameter(
            interaction.gate_bias,
            like=like,
            dtype=routing_dtype,
        )
        for name in (
            "quadratic_left",
            "quadratic_right",
            "quadratic_output",
        ):
            value = getattr(interaction, name)
            self.register_parameter(
                name,
                None if value is None else _frozen_parameter(value, like=like),
            )
        self.source_width = interaction.source_width
        self.target_width = interaction.target_width

    def routing_logit(self, source_state: Tensor) -> Tensor:
        if (
            source_state.device != self.gate_weight.device
            or source_state.shape[-1] != self.source_width
        ):
            raise ValueError("Gemma conditional routing state drifted")
        compute_dtype = (
            torch.float32
            if source_state.dtype in (torch.float16, torch.bfloat16)
            else source_state.dtype
        )
        values = source_state.to(dtype=compute_dtype)
        weight = self.gate_weight.to(dtype=compute_dtype)
        bias = self.gate_bias.to(dtype=compute_dtype)
        with torch.autocast(device_type=values.device.type, enabled=False):
            result = values @ weight + bias[0]
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Gemma conditional routing logit became non-finite")
        return result

    def proposed_message(self, source_state: Tensor) -> Tensor:
        if (
            source_state.device != self.message_matrix.device
            or source_state.shape[-1] != self.source_width
        ):
            raise ValueError("Gemma conditional interaction state drifted")
        result = source_state @ self.message_matrix
        result = result + self.message_bias.to(
            device=result.device,
            dtype=result.dtype,
        )
        if self.quadratic_left is not None:
            result = result + (
                (source_state @ self.quadratic_left)
                * (source_state @ self.quadratic_right)
            ) @ self.quadratic_output
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Gemma conditional proposal became non-finite")
        return result


@dataclass(frozen=True, slots=True)
class _BoundGraphNode:
    node: ModalGeneratorNode
    lowering: ModalGeneratorLowering
    fragment: ParameterClusterLayerFragment


class _DeviceGraphRuntime(nn.Module):
    """Registered, copied graph coefficients with incremental sessions."""

    def __init__(
        self,
        plan: ModalGeneratorGraphPlan,
        bound_nodes: tuple[_BoundGraphNode, ...],
        *,
        like: Tensor,
    ) -> None:
        super().__init__()
        self.node_runtimes = nn.ModuleList(
            _DeviceNodeRuntime(bound.node.weights, like=like)
            for bound in bound_nodes
        )
        self.edge_runtimes = nn.ModuleList(
            (
                _DeviceEdgeRuntime(edge, like=like)
                if isinstance(edge, ModalGeneratorInteraction)
                else _DeviceStateConditionedEdgeRuntime(edge, like=like)
            )
            for edge in plan.interactions
        )
        self._node_index = {
            bound.node.name: index
            for index, bound in enumerate(bound_nodes)
        }
        self._edge_index = {
            edge.key: index
            for index, edge in enumerate(plan.interactions)
        }
        incoming: dict[str, list[ModalGeneratorInteraction]] = defaultdict(list)
        for edge in plan.interactions:
            if isinstance(edge, ModalGeneratorInteraction):
                incoming[edge.target_node].append(edge)
        self._incoming = {
            name: tuple(edges) for name, edges in incoming.items()
        }
        self._conditional_outgoing = _conditional_outgoing_groups(
            plan.interactions
        )
        nodes_by_layer: dict[int, list[ModalGeneratorNode]] = defaultdict(list)
        for bound in bound_nodes:
            nodes_by_layer[bound.fragment.layer_ordinal].append(bound.node)
        self.nodes_by_layer = {
            ordinal: tuple(nodes)
            for ordinal, nodes in sorted(nodes_by_layer.items())
        }
        self.layer_order = tuple(self.nodes_by_layer)

        node_position = {
            node.name: index for index, node in enumerate(plan.nodes)
        }
        last_use = dict(node_position)
        for edge in plan.interactions:
            # Static interactions read the source state when the target runs.
            # Conditional interactions instead materialize their weighted
            # messages immediately after the source state is complete, so the
            # source itself need not remain live until the target layer.
            if isinstance(edge, ModalGeneratorInteraction):
                last_use[edge.source_node] = max(
                    last_use[edge.source_node],
                    node_position[edge.target_node],
                )
        self._node_position = node_position
        self._last_use = last_use
        self.peak_live_modal_width = self._compute_peak_live_width(plan)
        self.requires_grad_(False)
        self.eval()

        runtime_parameters = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if runtime_parameters != plan.parameter_count:
            raise RuntimeError(
                "device graph parameter count does not match graph plan"
            )

    def _compute_peak_live_width(
        self,
        plan: ModalGeneratorGraphPlan,
    ) -> int:
        live: dict[str, int] = {}
        peak = 0
        for index, node in enumerate(plan.nodes):
            live[node.name] = node.latent_width
            peak = max(peak, sum(live.values()))
            for name in tuple(live):
                if self._last_use[name] <= index:
                    del live[name]
        return peak

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def route_conditional_group(
        self,
        source_state: Tensor,
        edges: tuple[StateConditionedModalGeneratorInteraction, ...],
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, int],
        dict[str, Tensor],
    ]:
        runtimes = tuple(
            self.edge_runtimes[self._edge_index[edge.key]] for edge in edges
        )
        if any(
            not isinstance(runtime, _DeviceStateConditionedEdgeRuntime)
            for runtime in runtimes
        ):
            raise RuntimeError("conditional edge runtime kind drifted")
        logits = torch.stack(
            tuple(
                runtime.routing_logit(source_state)
                for runtime in runtimes
            ),
            dim=-1,
        )
        if edges[0].top_k < len(edges):
            order = torch.argsort(
                logits,
                dim=-1,
                descending=True,
                stable=True,
            )
            selected = order[..., : edges[0].top_k]
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, selected, True)
            selected_logits = logits.masked_fill(~mask, float("-inf"))
        else:
            mask = torch.ones_like(logits, dtype=torch.bool)
            selected_logits = logits
        centered_logits = selected_logits - selected_logits.max(
            dim=-1,
            keepdim=True,
        ).values
        scaled = centered_logits / edges[0].temperature
        weights = torch.softmax(scaled, dim=-1)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("Gemma conditional routing weights became non-finite")
        flat_source = source_state.reshape(-1, edges[0].source_width)
        flat_weights = weights.reshape(-1, len(edges))
        flat_mask = mask.reshape(-1, len(edges))
        result: dict[str, Tensor] = {}
        route_weights: dict[str, Tensor] = {}
        evaluated_rows: dict[str, int] = {}
        selected_masks: dict[str, Tensor] = {}
        for index, (edge, runtime) in enumerate(
            zip(edges, runtimes, strict=True)
        ):
            edge_weights = flat_weights[:, index]
            selected_rows = torch.nonzero(
                flat_mask[:, index],
                as_tuple=False,
            ).flatten()
            flat_message = source_state.new_zeros(
                (flat_source.shape[0], edge.target_width)
            )
            if selected_rows.numel():
                selected_source = flat_source.index_select(0, selected_rows)
                proposal = runtime.proposed_message(selected_source)
                selected_weights = edge_weights.index_select(
                    0,
                    selected_rows,
                ).to(dtype=proposal.dtype)
                flat_message.index_copy_(
                    0,
                    selected_rows,
                    proposal * selected_weights[:, None],
                )
            result[edge.key] = flat_message.reshape(
                (*source_state.shape[:-1], edge.target_width)
            )
            route_weights[edge.key] = weights[..., index]
            evaluated_rows[edge.key] = int(selected_rows.numel())
            selected_masks[edge.key] = mask[..., index]
        return result, route_weights, evaluated_rows, selected_masks

    def start(
        self,
        plan: ModalGeneratorGraphPlan,
        *,
        condition: str,
        capture_modal_states: bool,
        capture_edge_messages: bool,
        capture_routing: bool,
        logical_valid_mask: Tensor | None = None,
    ) -> _IncrementalGraphSession:
        return _IncrementalGraphSession(
            runtime=self,
            plan=plan,
            condition=condition,
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
            capture_routing=capture_routing,
            logical_valid_mask=logical_valid_mask,
        )


class _IncrementalGraphSession:
    """One non-reentrant, layer-ordered traversal state."""

    def __init__(
        self,
        *,
        runtime: _DeviceGraphRuntime,
        plan: ModalGeneratorGraphPlan,
        condition: str,
        capture_modal_states: bool,
        capture_edge_messages: bool,
        capture_routing: bool,
        logical_valid_mask: Tensor | None,
    ) -> None:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'generated' or 'deletion'")
        self.runtime = runtime
        self.plan = plan
        self.condition = condition
        self.states: dict[str, Tensor] = {}
        self.outputs: dict[str, Tensor] = {}
        self.conditional_messages: defaultdict[
            str,
            list[tuple[str, Tensor]],
        ] = defaultdict(list)
        self.modal_states = {} if capture_modal_states else None
        self.edge_messages = {} if capture_edge_messages else None
        self.routing_weights = {} if capture_routing else None
        self.evaluated_edge_rows = {} if capture_routing else None
        if logical_valid_mask is not None and (
            not isinstance(logical_valid_mask, Tensor)
            or logical_valid_mask.dtype != torch.bool
            or logical_valid_mask.numel() == 0
        ):
            raise ValueError("logical_valid_mask must be a nonempty bool Tensor")
        self.logical_valid_mask = logical_valid_mask
        self.logical_evaluated_edge_rows: dict[str, int] = defaultdict(int)
        self.called_layers: set[int] = set()
        self.executed_nodes: list[str] = []

    def execute_layer(
        self,
        layer_ordinal: int,
        normalized_input: Tensor,
    ) -> Tensor:
        if layer_ordinal in self.called_layers:
            raise RuntimeError("each affected Gemma MLP may execute only once")
        nodes = self.runtime.nodes_by_layer.get(layer_ordinal)
        if nodes is None:
            raise KeyError("layer is not part of the modal graph")
        expected_layer = self.runtime.layer_order[len(self.called_layers)]
        if layer_ordinal != expected_layer:
            raise RuntimeError(
                "Gemma modal graph layers executed out of causal order"
            )
        self.called_layers.add(layer_ordinal)
        output_boundary = nodes[0].output_boundary
        if self.condition == "deletion":
            output = normalized_input.new_zeros(
                (*normalized_input.shape[:-1], nodes[0].output_width)
            )
            self.outputs[output_boundary] = output
            return output

        output: Tensor | None = None
        for node in nodes:
            index = self.runtime._node_index[node.name]
            node_runtime = self.runtime.node_runtimes[index]
            state = node_runtime.generate(normalized_input)
            for edge in self.runtime._incoming.get(node.name, ()):
                try:
                    source_state = self.states[edge.source_node]
                except KeyError as error:
                    raise RuntimeError(
                        f"source modal state {edge.source_node!r} is not live"
                    ) from error
                edge_runtime = self.runtime.edge_runtimes[
                    self.runtime._edge_index[edge.key]
                ]
                message = edge_runtime(source_state)
                if (
                    message.shape[:-1] != state.shape[:-1]
                    or message.device != state.device
                    or message.dtype != state.dtype
                ):
                    raise ValueError(
                        f"interaction {edge.key!r} runtime batch drifted"
                    )
                if self.edge_messages is not None:
                    self.edge_messages[edge.key] = message.detach().clone()
                state = state + message
            for edge_key, message in self.conditional_messages.pop(
                node.name,
                (),
            ):
                if (
                    message.shape != state.shape
                    or message.device != state.device
                    or message.dtype != state.dtype
                ):
                    raise ValueError(
                        f"interaction {edge_key!r} runtime batch drifted"
                    )
                if self.edge_messages is not None:
                    self.edge_messages[edge_key] = message.detach().clone()
                state = state + message
            if not bool(torch.isfinite(state).all()):
                raise ValueError(
                    f"modal state for node {node.name!r} became non-finite"
                )
            self.states[node.name] = state
            self.executed_nodes.append(node.name)
            if self.modal_states is not None:
                self.modal_states[node.name] = state.detach().clone()
            for group in self.runtime._conditional_outgoing.get(node.name, ()):
                messages, route_weights, evaluated_rows, selected_masks = (
                    self.runtime.route_conditional_group(state, group)
                )
                for edge in group:
                    self.conditional_messages[edge.target_node].append(
                        (edge.key, messages[edge.key])
                    )
                    if self.routing_weights is not None:
                        self.routing_weights[edge.key] = route_weights[
                            edge.key
                        ].detach().clone()
                    if self.evaluated_edge_rows is not None:
                        self.evaluated_edge_rows[edge.key] = evaluated_rows[
                            edge.key
                        ]
                    selected_mask = selected_masks[edge.key]
                    if self.logical_valid_mask is None:
                        logical_rows = int(selected_mask.sum().item())
                    else:
                        valid_mask = self.logical_valid_mask.to(
                            device=selected_mask.device
                        )
                        if valid_mask.shape != selected_mask.shape:
                            raise ValueError(
                                "logical valid mask does not match routed token shape"
                            )
                        logical_rows = int(
                            (selected_mask & valid_mask).sum().item()
                        )
                    self.logical_evaluated_edge_rows[edge.key] += logical_rows
            contribution = node_runtime.decode(state)
            if not bool(torch.isfinite(contribution).all()):
                raise ValueError(
                    f"graph contribution for node {node.name!r} became non-finite"
                )
            output = (
                contribution
                if output is None
                else output + contribution
            )

            position = self.runtime._node_position[node.name]
            for name in tuple(self.states):
                if self.runtime._last_use[name] <= position:
                    del self.states[name]

        if output is None:
            raise RuntimeError("generated graph layer executed no nodes")
        self.outputs[output_boundary] = output
        return output

    def finish(self) -> ModalGeneratorGraphExecution:
        if set(self.called_layers) != set(self.runtime.layer_order):
            raise RuntimeError("not every affected Gemma graph layer executed")
        if self.condition == "generated":
            if tuple(self.executed_nodes) != self.plan.traversal_order:
                raise RuntimeError("incremental traversal order drifted")
            if self.states:
                raise RuntimeError("modal state liveness did not drain")
            if self.conditional_messages:
                raise RuntimeError(
                    "conditional graph messages did not drain"
                )
        return ModalGeneratorGraphExecution(
            outputs=dict(self.outputs),
            traversal_order=(
                self.plan.traversal_order
                if self.condition == "generated"
                else ()
            ),
            modal_states=self.modal_states,
            edge_messages=self.edge_messages,
            routing_weights=self.routing_weights,
            evaluated_edge_rows=self.evaluated_edge_rows,
        )


class Gemma3GraphCompactMLP(nn.Module):
    """Copied native MLP with a declared union of channels physically absent."""

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        removed_mode_indices: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        gate, up, down = _validate_source_mlp(source_mlp, label="source_mlp")
        if source_mlp.training or any(
            parameter.requires_grad for parameter in source_mlp.parameters()
        ):
            raise ValueError("graph compilation requires a frozen eval MLP")
        intermediate_width = gate.out_features
        if (
            type(removed_mode_indices) is not tuple
            or not removed_mode_indices
            or removed_mode_indices
            != tuple(sorted(set(removed_mode_indices)))
            or any(
                type(index) is not int
                or index < 0
                or index >= intermediate_width
                for index in removed_mode_indices
            )
        ):
            raise ValueError("removed mode union is not canonical")
        retained = tuple(
            index
            for index in range(intermediate_width)
            if index not in removed_mode_indices
        )
        retained_tensor = torch.tensor(
            retained,
            dtype=torch.long,
            device=gate.weight.device,
        )
        if retained:
            self.gate_proj: nn.Module = nn.Linear(
                gate.in_features,
                len(retained),
                bias=False,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.up_proj: nn.Module = nn.Linear(
                up.in_features,
                len(retained),
                bias=False,
                dtype=up.weight.dtype,
                device=up.weight.device,
            )
            self.down_proj: nn.Module = nn.Linear(
                len(retained),
                down.out_features,
                bias=False,
                dtype=down.weight.dtype,
                device=down.weight.device,
            )
        else:
            self.gate_proj = _EmptyLinear(
                gate.in_features,
                0,
                dtype=gate.weight.dtype,
                device=gate.weight.device,
            )
            self.up_proj = _EmptyLinear(
                up.in_features,
                0,
                dtype=up.weight.dtype,
                device=up.weight.device,
            )
            self.down_proj = _EmptyLinear(
                0,
                down.out_features,
                dtype=down.weight.dtype,
                device=down.weight.device,
            )
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                gate.weight.index_select(0, retained_tensor)
            )
            self.up_proj.weight.copy_(
                up.weight.index_select(0, retained_tensor)
            )
            self.down_proj.weight.copy_(
                down.weight.index_select(1, retained_tensor)
            )
        self.activation = activation
        self.residual_width = gate.in_features
        self.source_intermediate_width = intermediate_width
        self.removed_mode_indices = removed_mode_indices
        self.retained_mode_indices = retained
        self.removed_parameter_count = (
            len(removed_mode_indices) * 3 * self.residual_width
        )
        self.retained_parameter_count = (
            len(retained) * 3 * self.residual_width
        )
        self.requires_grad_(False)
        self.eval()
        if _storage_pointers(source_mlp) & _storage_pointers(self):
            raise RuntimeError("compact graph MLP aliases source storage")

    @property
    def is_full_native_replacement(self) -> bool:
        return not self.retained_mode_indices

    def forward(self, normalized_input: Tensor) -> Tensor:
        if normalized_input.shape[-1] != self.residual_width:
            raise ValueError("compact MLP input width drifted")
        if not self.retained_mode_indices:
            return normalized_input.new_zeros(normalized_input.shape)
        gated = _apply_activation(
            self.activation,
            self.gate_proj(normalized_input),
        )
        return self.down_proj(gated * self.up_proj(normalized_input))


class _GraphMLPOverlay(nn.Module):
    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        session: _IncrementalGraphSession,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        # Avoid registering the shared graph runtime/session below every MLP.
        object.__setattr__(self, "_session", session)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        generated = self._session.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        return retained + generated


class _PostFeedForwardDeltaBuffer:
    """One pending decoded correction between an MLP and its live RMSNorm."""

    def __init__(
        self,
        diagnostic_scale_by_layer: Mapping[int, float] | None = None,
        diagnostic_correction_provider_by_layer: Mapping[
            int,
            _DiagnosticCorrectionProvider,
        ]
        | None = None,
    ) -> None:
        self._pending: dict[int, Tensor] = {}
        self._diagnostic_scale_by_layer = dict(
            diagnostic_scale_by_layer or {}
        )
        self._diagnostic_correction_provider_by_layer = dict(
            diagnostic_correction_provider_by_layer or {}
        )
        if (
            self._diagnostic_scale_by_layer.keys()
            & self._diagnostic_correction_provider_by_layer.keys()
        ):
            raise ValueError(
                "a post-feed-forward correction cannot be both attenuated "
                "and overridden"
            )

    def store(self, layer_ordinal: int, correction: Tensor) -> None:
        if layer_ordinal in self._pending:
            raise RuntimeError(
                "post-feed-forward correction was not consumed before reuse"
            )
        if not isinstance(correction, Tensor) or not bool(
            torch.isfinite(correction).all()
        ):
            raise ValueError("post-feed-forward correction is invalid")
        provider = self._diagnostic_correction_provider_by_layer.get(
            layer_ordinal
        )
        if provider is not None:
            replacement = provider(correction)
            if not isinstance(replacement, Tensor):
                raise TypeError(
                    "diagnostic correction provider must return a Tensor"
                )
            if replacement.shape != correction.shape:
                raise ValueError(
                    "diagnostic correction replacement shape must exactly "
                    "match the generated correction"
                )
            if replacement.device != correction.device:
                raise ValueError(
                    "diagnostic correction replacement device must exactly "
                    "match the generated correction"
                )
            if replacement.dtype != correction.dtype:
                raise ValueError(
                    "diagnostic correction replacement dtype must exactly "
                    "match the generated correction"
                )
            if not bool(torch.isfinite(replacement).all()):
                raise ValueError(
                    "diagnostic correction replacement must be finite"
                )
            correction = replacement
        self._pending[layer_ordinal] = correction

    def consume(self, layer_ordinal: int, retained_delta: Tensor) -> Tensor:
        try:
            correction = self._pending.pop(layer_ordinal)
        except KeyError as error:
            raise RuntimeError(
                "post-feed-forward normalization ran without a correction"
            ) from error
        if (
            correction.shape != retained_delta.shape
            or correction.device != retained_delta.device
            or correction.dtype != retained_delta.dtype
        ):
            raise ValueError(
                "post-feed-forward retained and generated deltas disagree"
            )
        diagnostic_scale = self._diagnostic_scale_by_layer.get(
            layer_ordinal,
            1.0,
        )
        if diagnostic_scale != 1.0:
            correction = correction * diagnostic_scale
        result = retained_delta + correction
        if not bool(torch.isfinite(result).all()):
            raise ValueError(
                "post-feed-forward corrected delta became non-finite"
            )
        return result

    @property
    def empty(self) -> bool:
        return not self._pending


class _GraphPostFeedForwardMLPOverlay(nn.Module):
    """Compact raw MLP that defers its graph correction until after RMSNorm."""

    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        session: _IncrementalGraphSession,
        buffer: _PostFeedForwardDeltaBuffer,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        generated = self._session.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        self._buffer.store(self.layer_ordinal, generated)
        return retained


class _GraphWithAdditivePostFeedForwardMLPOverlay(nn.Module):
    """Owning source graph plus a non-owning post-RMSNorm residual graph."""

    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        source_session: _IncrementalGraphSession,
        additive_session: _IncrementalGraphSession,
        buffer: _PostFeedForwardDeltaBuffer,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        object.__setattr__(self, "_source_session", source_session)
        object.__setattr__(self, "_additive_session", additive_session)
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        source_generated = self._source_session.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        additive_generated = self._additive_session.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        self._buffer.store(self.layer_ordinal, additive_generated)
        return retained + source_generated


class _PostFeedForwardNormOverlay(nn.Module):
    """Run the live RMSNorm, then add one generated block-space correction."""

    def __init__(
        self,
        source_norm: nn.Module,
        *,
        layer_ordinal: int,
        buffer: _PostFeedForwardDeltaBuffer,
    ) -> None:
        super().__init__()
        self.source_norm = source_norm
        self.layer_ordinal = layer_ordinal
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    def forward(self, compact_raw_output: Tensor) -> Tensor:
        retained_delta = self.source_norm(compact_raw_output)
        if not isinstance(retained_delta, Tensor):
            raise TypeError(
                "post-feed-forward normalization must return a Tensor"
            )
        return self._buffer.consume(self.layer_ordinal, retained_delta)


@dataclass(slots=True)
class _RepeatedGraphOverlayState:
    """One causal graph session per full-model callback forward."""

    runtime: _DeviceGraphRuntime
    plan: ModalGeneratorGraphPlan
    expected_forward_calls: int
    calls: dict[int, int]
    completed_forward_calls: int = 0
    session: _IncrementalGraphSession | None = None

    def execute_layer(
        self,
        layer_ordinal: int,
        normalized_input: Tensor,
    ) -> Tensor:
        calls = self.calls.get(layer_ordinal, 0)
        if calls >= self.expected_forward_calls:
            raise RuntimeError(
                "each callback graph overlay may execute only "
                f"{self.expected_forward_calls} times"
            )
        if self.session is None:
            if layer_ordinal != self.runtime.layer_order[0]:
                raise RuntimeError(
                    "callback graph forward did not begin at its first layer"
                )
            self.session = self.runtime.start(
                self.plan,
                condition="generated",
                capture_modal_states=False,
                capture_edge_messages=False,
                capture_routing=False,
            )
        generated = self.session.execute_layer(
            layer_ordinal,
            normalized_input,
        )
        self.calls[layer_ordinal] = calls + 1
        if layer_ordinal == self.runtime.layer_order[-1]:
            self.session.finish()
            self.session = None
            self.completed_forward_calls += 1
        return generated


class _RepeatedGraphMLPOverlay(nn.Module):
    """Compact MLP plus a graph session shared across one callback forward."""

    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        state: _RepeatedGraphOverlayState,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        # Do not register the already-owned shared runtime below every MLP.
        object.__setattr__(self, "_state", state)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        generated = self._state.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        return retained + generated


class _RepeatedGraphPostFeedForwardMLPOverlay(nn.Module):
    """Repeated compact MLP with a correction applied after the live RMSNorm."""

    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        state: _RepeatedGraphOverlayState,
        buffer: _PostFeedForwardDeltaBuffer,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        generated = self._state.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        self._buffer.store(self.layer_ordinal, generated)
        return retained


class _RepeatedGraphWithAdditivePostFeedForwardMLPOverlay(nn.Module):
    """Repeated owning source graph plus a non-owning post-norm residual."""

    def __init__(
        self,
        compact: Gemma3GraphCompactMLP,
        *,
        layer_ordinal: int,
        source_state: _RepeatedGraphOverlayState,
        additive_state: _RepeatedGraphOverlayState,
        buffer: _PostFeedForwardDeltaBuffer,
    ) -> None:
        super().__init__()
        self.compact = compact
        self.layer_ordinal = layer_ordinal
        object.__setattr__(self, "_source_state", source_state)
        object.__setattr__(self, "_additive_state", additive_state)
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.compact.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.compact.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.compact.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        retained = self.compact(normalized_input)
        source_generated = self._source_state.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        additive_generated = self._additive_state.execute_layer(
            self.layer_ordinal,
            normalized_input,
        )
        self._buffer.store(self.layer_ordinal, additive_generated)
        return retained + source_generated


@dataclass(frozen=True, slots=True)
class Gemma3ModalGeneratorGraphExecution:
    """Model output, graph instrumentation, and logical linear-MAC accounting.

    ``logical_modal_graph_macs`` is the conservative dense-candidate bound.
    ``logical_executed_modal_graph_macs`` uses the actual conditional top-k
    selections on valid tokens. Addition totals retain the graph plan's dense
    linear/proposal bound. Softmax, comparison/sort, gather/scatter, and other
    nonlinear routing operations are deliberately excluded from these
    linear-MAC fields.
    """

    model_output: object
    graph_execution: ModalGeneratorGraphExecution
    condition: str
    replaced_layer_count: int
    graph_node_count: int
    fragment_count: int
    removed_mode_count: int
    source_whole_model_learned_parameters: int
    candidate_whole_model_learned_parameters: int
    native_removed_learned_parameters: int
    modal_graph_learned_parameters: int
    net_stored_parameter_savings: int
    valid_tokens: int
    logical_linear_macs_native_removed: int
    logical_modal_graph_macs: int
    logical_executed_modal_graph_macs: int
    logical_modal_graph_additions: int
    logical_executed_modal_graph_additions: int
    net_logical_macs_saved: int
    peak_live_modal_width: int
    replacement_scope: str
    graph_runtime_storage: str = _RUNTIME_STORAGE
    additive_graph_execution: ModalGeneratorGraphExecution | None = None

    def __post_init__(self) -> None:
        if self.condition not in _CONDITIONS:
            raise ValueError("invalid Gemma modal graph condition")
        if self.graph_runtime_storage != _RUNTIME_STORAGE:
            raise ValueError("graph runtime storage label drifted")
        if self.additive_graph_execution is not None and not isinstance(
            self.additive_graph_execution,
            ModalGeneratorGraphExecution,
        ):
            raise TypeError("additive_graph_execution must be a graph execution")


def _feed_forward_metadata(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
) -> tuple[str, str, str, str, nn.Module]:
    if not 0 <= layer_ordinal < len(adapter.layers):
        raise ValueError("fragment layer ordinal is outside the Gemma model")
    layer = adapter.layers[layer_ordinal]
    transformer = layer.transformer
    if (
        transformer is None
        or transformer.feed_forward is None
        or transformer.operator_sites is None
    ):
        raise ValueError("Gemma layer lacks structured MLP metadata")
    stages = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(stages) != 1:
        raise ValueError("Gemma layer must expose one feed-forward stage")
    source_layer = adapter.source_module(layer.id)
    source_mlp = getattr(source_layer, "mlp", None)
    if not isinstance(source_mlp, nn.Module):
        raise TypeError("Gemma source layer does not expose an MLP")
    return (
        stages[0].normalized_input_site,
        stages[0].operator_output_site,
        transformer.operator_sites.feed_forward_down_input,
        transformer.feed_forward.activation,
        source_mlp,
    )


def _post_feed_forward_metadata(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
) -> tuple[str, nn.Module]:
    """Return the authenticated post-MLP delta site and live RMSNorm."""

    if not 0 <= layer_ordinal < len(adapter.layers):
        raise ValueError("fragment layer ordinal is outside the Gemma model")
    layer = adapter.layers[layer_ordinal]
    transformer = layer.transformer
    if transformer is None:
        raise ValueError("Gemma layer lacks structured MLP metadata")
    stages = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(stages) != 1:
        raise ValueError("Gemma layer must expose one feed-forward stage")
    source_layer = adapter.source_module(layer.id)
    post_norm = getattr(source_layer, "post_feedforward_layernorm", None)
    if not isinstance(post_norm, nn.Module):
        raise TypeError(
            "Gemma source layer does not expose post-feed-forward normalization"
        )
    return stages[0].delta_site, post_norm


def _authenticated_lowering(
    value: ModalGeneratorLowering,
) -> ModalGeneratorLowering:
    if not isinstance(value, ModalGeneratorLowering):
        raise TypeError("lowerings must contain ModalGeneratorLowering values")
    return ModalGeneratorLowering.from_state_dict(value.state_dict())


def _bound_fragment(
    lowering: ModalGeneratorLowering,
) -> ParameterClusterLayerFragment:
    matching = tuple(
        fragment
        for fragment in lowering.fragment_plan.fragments
        if fragment.artifact_sha256 == lowering.selected_fragment_sha256
    )
    if len(matching) != 1:
        raise ValueError("lowering must bind exactly one parameter fragment")
    return matching[0]


@dataclass(frozen=True, slots=True)
class _AuthenticatedAdditivePostFeedForwardGraph:
    plan: ModalGeneratorGraphPlan
    lowerings: tuple[ModalGeneratorLowering, ...]
    bound_nodes: tuple[_BoundGraphNode, ...]
    layer_ordinal: int


def _authenticate_additive_post_feedforward_graph(
    *,
    adapter: Gemma3CausalLMAdapter,
    graph_plan: ModalGeneratorGraphPlan,
    lowerings: Sequence[ModalGeneratorLowering],
    source_model_sha256: str,
    source_parameter_cluster_plan_sha256: str,
    owning_fragment_sha256s: frozenset[str],
) -> _AuthenticatedAdditivePostFeedForwardGraph:
    """Authenticate one non-owning zero-mean graph at a post-FF boundary."""

    graph_plan.validate_integrity()
    plan = ModalGeneratorGraphPlan.from_state_dict(graph_plan.state_dict())
    if (
        plan.model_fingerprint != source_model_sha256
        or plan.parameter_cluster_plan_sha256
        != source_parameter_cluster_plan_sha256
    ):
        raise ValueError(
            "additive post-feed-forward graph differs from the source identity"
        )
    supplied = tuple(_authenticated_lowering(value) for value in lowerings)
    if not supplied or len(supplied) != len(plan.nodes):
        raise ValueError(
            "additive post-feed-forward nodes and lowerings must be one-to-one"
        )
    if any(
        lowering.fragment_plan.artifact_sha256
        != plan.parameter_cluster_plan_sha256
        for lowering in supplied
    ):
        raise ValueError(
            "additive post-feed-forward lowering fragment plan drifted"
        )
    by_weight_hash: dict[str, list[ModalGeneratorLowering]] = defaultdict(list)
    for lowering in supplied:
        by_weight_hash[lowering.graph_weights.artifact_sha256].append(lowering)
    bound_nodes: list[_BoundGraphNode] = []
    used_lowerings: set[str] = set()
    ordinals: set[int] = set()
    for node in plan.nodes:
        matches = by_weight_hash.get(node.weights.artifact_sha256, ())
        if len(matches) != 1:
            raise ValueError(
                "each additive graph node must match exactly one lowering"
            )
        lowering = matches[0]
        if lowering.artifact_sha256 in used_lowerings:
            raise ValueError(
                "an additive lowering cannot back multiple graph nodes"
            )
        used_lowerings.add(lowering.artifact_sha256)
        fragment = _bound_fragment(lowering)
        input_site, operator_output_site, down_site, _, _ = (
            _feed_forward_metadata(adapter, fragment.layer_ordinal)
        )
        delta_site, _ = _post_feed_forward_metadata(
            adapter,
            fragment.layer_ordinal,
        )
        layer = adapter.layers[fragment.layer_ordinal]
        basis = lowering.computational_mode_basis
        if (
            fragment.artifact_sha256 not in owning_fragment_sha256s
            or fragment.source_model_sha256 != source_model_sha256
            or fragment.layer_id != layer.id
            or fragment.activation_site != down_site
            or fragment.output_site != operator_output_site
            or node.input_boundary != input_site
            or node.output_boundary != delta_site
            or node.input_width != layer.residual_width
            or node.output_width != layer.residual_width
            or lowering.coordinate_generator_plan.binding.input_kind
            != "native_layer_input"
            or lowering.coordinate_generator_plan.binding.input_site
            != input_site
            or lowering.coordinate_generator_plan.binding.output_site
            != delta_site
            or basis.binding.output_site != delta_site
        ):
            raise ValueError(
                "additive post-feed-forward graph binding is invalid"
            )
        if bool(torch.count_nonzero(basis.mean_bias)):
            raise ValueError(
                "additive post-feed-forward graph must decode with zero mean"
            )
        ordinals.add(fragment.layer_ordinal)
        bound_nodes.append(
            _BoundGraphNode(node=node, lowering=lowering, fragment=fragment)
        )
    if len(used_lowerings) != len(supplied) or len(ordinals) != 1:
        raise ValueError(
            "additive post-feed-forward graph must cover exactly one layer"
        )
    layer_ordinal = next(iter(ordinals))
    if any(
        edge.source_node not in {node.name for node in plan.nodes}
        or edge.target_node not in {node.name for node in plan.nodes}
        for edge in plan.interactions
    ):
        raise ValueError("additive graph interaction leaves its one-layer graph")
    return _AuthenticatedAdditivePostFeedForwardGraph(
        plan=plan,
        lowerings=supplied,
        bound_nodes=tuple(bound_nodes),
        layer_ordinal=layer_ordinal,
    )


class Gemma3ModalGeneratorGraphExecutor(nn.Module):
    """Overlay an owning graph and optional non-owning residual during Gemma."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        graph_plan: ModalGeneratorGraphPlan,
        lowerings: Sequence[ModalGeneratorLowering],
        *,
        post_feedforward_delta_layer_ordinals: Sequence[int] = (),
        additive_post_feedforward_graph_plan: ModalGeneratorGraphPlan | None = None,
        additive_post_feedforward_lowerings: Sequence[
            ModalGeneratorLowering
        ] = (),
        additive_post_feedforward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be Gemma3CausalLMAdapter")
        if not isinstance(graph_plan, ModalGeneratorGraphPlan):
            raise TypeError("graph_plan must be ModalGeneratorGraphPlan")
        if isinstance(lowerings, (str, bytes)) or not isinstance(
            lowerings,
            Sequence,
        ):
            raise TypeError("lowerings must be a sequence")
        supplied_lowerings = tuple(lowerings)
        if not supplied_lowerings:
            raise ValueError("lowerings cannot be empty")
        if isinstance(
            post_feedforward_delta_layer_ordinals,
            (str, bytes),
        ) or not isinstance(post_feedforward_delta_layer_ordinals, Sequence):
            raise TypeError(
                "post_feedforward_delta_layer_ordinals must be a sequence"
            )
        post_delta_ordinals = tuple(post_feedforward_delta_layer_ordinals)
        if (
            any(
                type(value) is not int or value < 0
                for value in post_delta_ordinals
            )
            or post_delta_ordinals != tuple(sorted(set(post_delta_ordinals)))
        ):
            raise ValueError(
                "post-feed-forward delta layer ordinals must be unique and "
                "in causal order"
            )
        post_delta_set = set(post_delta_ordinals)
        if isinstance(additive_post_feedforward_lowerings, (str, bytes)) or not (
            isinstance(additive_post_feedforward_lowerings, Sequence)
        ):
            raise TypeError(
                "additive_post_feedforward_lowerings must be a sequence"
            )
        supplied_additive_lowerings = tuple(
            additive_post_feedforward_lowerings
        )
        if type(additive_post_feedforward_scale) not in (int, float):
            raise TypeError("additive_post_feedforward_scale must be real")
        additive_scale = float(additive_post_feedforward_scale)
        if not math.isfinite(additive_scale) or not 0.0 < additive_scale <= 1.0:
            raise ValueError(
                "additive_post_feedforward_scale must lie in (0, 1]"
            )
        if additive_post_feedforward_graph_plan is None:
            if supplied_additive_lowerings or additive_scale != 1.0:
                raise ValueError(
                    "additive post-feed-forward lowerings or scale require a graph"
                )
        elif not isinstance(
            additive_post_feedforward_graph_plan,
            ModalGeneratorGraphPlan,
        ):
            raise TypeError(
                "additive_post_feedforward_graph_plan must be a graph plan"
            )
        model = adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError("Gemma graph execution requires a frozen eval model")
        source_model_sha256 = adapter.model_fingerprint()
        graph_plan.validate_integrity()
        plan = ModalGeneratorGraphPlan.from_state_dict(
            graph_plan.state_dict()
        )
        if plan.model_fingerprint != source_model_sha256:
            raise ValueError("graph plan does not bind the live Gemma model")
        copied_lowerings = tuple(
            _authenticated_lowering(value) for value in supplied_lowerings
        )
        if len(copied_lowerings) != len(plan.nodes):
            raise ValueError("graph nodes and lowerings must be one-to-one")
        if any(
            lowering.fragment_plan.artifact_sha256
            != plan.parameter_cluster_plan_sha256
            for lowering in copied_lowerings
        ):
            raise ValueError(
                "lowering fragment plan does not match the graph plan"
            )

        by_weight_hash: dict[str, list[ModalGeneratorLowering]] = defaultdict(
            list
        )
        for lowering in copied_lowerings:
            by_weight_hash[
                lowering.graph_weights.artifact_sha256
            ].append(lowering)
        bound_nodes: list[_BoundGraphNode] = []
        used_lowerings: set[str] = set()
        prior_layer = -1
        for node in plan.nodes:
            matches = by_weight_hash.get(node.weights.artifact_sha256, ())
            if len(matches) != 1:
                raise ValueError(
                    "each graph node weights hash must match one lowering"
                )
            lowering = matches[0]
            if lowering.artifact_sha256 in used_lowerings:
                raise ValueError("a lowering cannot back multiple graph nodes")
            used_lowerings.add(lowering.artifact_sha256)
            fragment = _bound_fragment(lowering)
            if (
                node.weights.artifact_sha256
                != lowering.graph_weights.artifact_sha256
                or node.weights.parameter_cluster_fragment_sha256
                != fragment.artifact_sha256
                or fragment.source_model_sha256 != source_model_sha256
                or fragment.layer_ordinal < prior_layer
            ):
                raise ValueError(
                    "graph node/lowering/fragment causal binding is invalid"
                )
            input_site, operator_output_site, down_site, _, _ = (
                _feed_forward_metadata(adapter, fragment.layer_ordinal)
            )
            delta_site, _ = _post_feed_forward_metadata(
                adapter,
                fragment.layer_ordinal,
            )
            expected_output_site = (
                delta_site
                if fragment.layer_ordinal in post_delta_set
                else operator_output_site
            )
            layer = adapter.layers[fragment.layer_ordinal]
            if (
                fragment.layer_id != layer.id
                or fragment.activation_site != down_site
                or fragment.output_site != operator_output_site
                or node.input_boundary != input_site
                or node.output_boundary != expected_output_site
                or node.input_width != layer.residual_width
                or node.output_width != layer.residual_width
                or lowering.coordinate_generator_plan.binding.input_kind
                != "native_layer_input"
                or lowering.coordinate_generator_plan.binding.input_site
                != input_site
                or lowering.coordinate_generator_plan.binding.output_site
                != expected_output_site
                or lowering.computational_mode_basis.binding.output_site
                != expected_output_site
            ):
                raise ValueError(
                    "graph fragment sites do not match the live Gemma layer"
                )
            prior_layer = fragment.layer_ordinal
            bound_nodes.append(
                _BoundGraphNode(
                    node=node,
                    lowering=lowering,
                    fragment=fragment,
                )
            )
        if len(used_lowerings) != len(copied_lowerings):
            raise ValueError("not every lowering is represented in the graph")

        fragments_by_layer: dict[
            int,
            list[ParameterClusterLayerFragment],
        ] = defaultdict(list)
        for bound in bound_nodes:
            fragments_by_layer[bound.fragment.layer_ordinal].append(
                bound.fragment
            )
        if not post_delta_set.issubset(fragments_by_layer):
            raise ValueError(
                "post-feed-forward delta layers must belong to the graph"
            )
        compact: dict[str, Gemma3GraphCompactMLP] = {}
        source_fingerprints: dict[int, str] = {}
        compact_fingerprints: dict[int, str] = {}
        removed_mode_count = 0
        removed_parameters = 0
        full_layers = 0
        for ordinal, fragments in sorted(fragments_by_layer.items()):
            channels = tuple(
                channel
                for fragment in fragments
                for channel in fragment.channel_indices
            )
            if len(channels) != len(set(channels)):
                raise ValueError(
                    "fragments on one layer may not remove overlapping modes"
                )
            removed = tuple(sorted(channels))
            _, _, _, activation, source_mlp = _feed_forward_metadata(
                adapter,
                ordinal,
            )
            compiled = Gemma3GraphCompactMLP(
                source_mlp,
                removed_mode_indices=removed,
                activation=activation,
            )
            declared_parameters = sum(
                fragment.native_parameter_count for fragment in fragments
            )
            if declared_parameters != compiled.removed_parameter_count:
                raise ValueError(
                    "fragment native parameter accounting does not match Gemma"
                )
            compact[str(ordinal)] = compiled
            source_fingerprints[ordinal] = module_state_fingerprint(source_mlp)
            compact_fingerprints[ordinal] = module_state_fingerprint(compiled)
            removed_mode_count += len(removed)
            removed_parameters += compiled.removed_parameter_count
            full_layers += int(compiled.is_full_native_replacement)

        affected_weights = tuple(
            next(
                _feed_forward_metadata(adapter, ordinal)[4].parameters()
            )
            for ordinal in sorted(fragments_by_layer)
        )
        first_weight = affected_weights[0]
        if any(
            weight.device != first_weight.device
            or weight.dtype != first_weight.dtype
            for weight in affected_weights[1:]
        ):
            raise ValueError(
                "Gemma modal graph currently requires all affected layers "
                "on one device with one parameter dtype"
            )
        graph_runtime = _DeviceGraphRuntime(
            plan,
            tuple(bound_nodes),
            like=first_weight,
        )
        additive: _AuthenticatedAdditivePostFeedForwardGraph | None = None
        additive_runtime: _DeviceGraphRuntime | None = None
        if additive_post_feedforward_graph_plan is not None:
            additive = _authenticate_additive_post_feedforward_graph(
                adapter=adapter,
                graph_plan=additive_post_feedforward_graph_plan,
                lowerings=supplied_additive_lowerings,
                source_model_sha256=source_model_sha256,
                source_parameter_cluster_plan_sha256=(
                    plan.parameter_cluster_plan_sha256
                ),
                owning_fragment_sha256s=frozenset(
                    bound.fragment.artifact_sha256 for bound in bound_nodes
                ),
            )
            if (
                additive.layer_ordinal not in fragments_by_layer
                or additive.layer_ordinal in post_delta_set
            ):
                raise ValueError(
                    "additive post-feed-forward layer must be owned by a "
                    "pre-normalization source graph"
                )
            additive_runtime = _DeviceGraphRuntime(
                additive.plan,
                additive.bound_nodes,
                like=first_weight,
            )
        source_storage = _storage_pointers(model)
        runtime_storage = _storage_pointers(graph_runtime)
        plan_storage = {
            tensor.untyped_storage().data_ptr()
            for node in plan.nodes
            for tensor in (
                node.weights.input_factor,
                node.weights.state_factor,
                node.weights.output_factor,
                node.weights.latent_bias,
                node.weights.output_bias,
            )
            if tensor is not None and tensor.numel()
        } | {
            tensor.untyped_storage().data_ptr()
            for edge in plan.interactions
            for tensor in (
                edge.message_matrix,
                edge.message_bias,
                getattr(edge, "gate_weight", None),
                getattr(edge, "gate_bias", None),
                getattr(edge, "quadratic_left", None),
                getattr(edge, "quadratic_right", None),
                getattr(edge, "quadratic_output", None),
            )
            if tensor is not None and tensor.numel()
        }
        if (
            source_storage & runtime_storage
            or plan_storage & runtime_storage
        ):
            raise RuntimeError("device graph runtime aliases source/artifact")
        if graph_runtime.parameter_count != plan.parameter_count:
            raise RuntimeError("graph runtime storage accounting drifted")
        if additive is not None and additive_runtime is not None:
            additive_runtime_storage = _storage_pointers(additive_runtime)
            additive_plan_storage = {
                tensor.untyped_storage().data_ptr()
                for node in additive.plan.nodes
                for tensor in (
                    node.weights.input_factor,
                    node.weights.state_factor,
                    node.weights.output_factor,
                    node.weights.latent_bias,
                    node.weights.output_bias,
                )
                if tensor is not None and tensor.numel()
            } | {
                tensor.untyped_storage().data_ptr()
                for edge in additive.plan.interactions
                for tensor in (
                    edge.message_matrix,
                    edge.message_bias,
                    getattr(edge, "gate_weight", None),
                    getattr(edge, "gate_bias", None),
                    getattr(edge, "quadratic_left", None),
                    getattr(edge, "quadratic_right", None),
                    getattr(edge, "quadratic_output", None),
                )
                if tensor is not None and tensor.numel()
            }
            if (
                source_storage & additive_runtime_storage
                or runtime_storage & additive_runtime_storage
                or plan_storage & additive_runtime_storage
                or additive_plan_storage & additive_runtime_storage
                or additive_plan_storage & runtime_storage
            ):
                raise RuntimeError(
                    "additive graph runtime aliases source or graph storage"
                )
            if additive_runtime.parameter_count != additive.plan.parameter_count:
                raise RuntimeError(
                    "additive graph runtime storage accounting drifted"
                )

        self.adapter = adapter
        self.graph_plan = plan
        self.compiled_mlps = nn.ModuleDict(compact)
        self.graph_runtime = graph_runtime
        self.additive_post_feedforward_graph_runtime = additive_runtime
        self._lowerings = copied_lowerings
        self._bound_nodes = tuple(bound_nodes)
        self._additive_post_feedforward_graph_plan = (
            None if additive is None else additive.plan
        )
        self._additive_post_feedforward_lowerings = (
            () if additive is None else additive.lowerings
        )
        self._additive_post_feedforward_layer_ordinal = (
            None if additive is None else additive.layer_ordinal
        )
        self._additive_post_feedforward_scale = additive_scale
        self._affected_ordinals = tuple(sorted(fragments_by_layer))
        self._post_feedforward_delta_ordinals = post_delta_ordinals
        self._source_model_sha256 = source_model_sha256
        self._source_fingerprints = source_fingerprints
        self._compact_fingerprints = compact_fingerprints
        self._graph_runtime_fingerprint = module_state_fingerprint(
            graph_runtime
        )
        self._additive_graph_runtime_fingerprint = (
            None
            if additive_runtime is None
            else module_state_fingerprint(additive_runtime)
        )
        self._removed_mode_count = removed_mode_count
        self._removed_parameters = removed_parameters
        self._fragment_count = len(bound_nodes)
        self._active = False
        self._validated_transaction_active = False
        if full_layers == len(self._affected_ordinals):
            self._replacement_scope = (
                "full_native_mlp_replacement_at_selected_layers"
            )
        elif full_layers:
            self._replacement_scope = (
                "mixed_partial_and_full_native_mlp_replacement"
            )
        else:
            self._replacement_scope = (
                "partial_native_mlp_mode_replacement"
            )
        if additive is not None:
            self._replacement_scope += (
                "_with_nonowning_post_feedforward_additive_graph"
            )
        self.requires_grad_(False)
        self.eval()
        self._validate_live_state()

    @property
    def graph_runtime_parameter_count(self) -> int:
        return self.graph_runtime.parameter_count

    @property
    def additive_post_feedforward_graph_runtime_parameter_count(self) -> int:
        runtime = self.additive_post_feedforward_graph_runtime
        return 0 if runtime is None else runtime.parameter_count

    @property
    def additive_post_feedforward_graph_plan(
        self,
    ) -> ModalGeneratorGraphPlan | None:
        """Authenticated non-owning graph, absent for the exact source path."""

        return self._additive_post_feedforward_graph_plan

    @property
    def total_graph_runtime_parameter_count(self) -> int:
        return (
            self.graph_runtime_parameter_count
            + self.additive_post_feedforward_graph_runtime_parameter_count
        )

    @property
    def affected_layer_ordinals(self) -> tuple[int, ...]:
        """Exact causal layer subset replaced by this executor."""

        return self._affected_ordinals

    @property
    def post_feedforward_delta_layer_ordinals(self) -> tuple[int, ...]:
        """Layers whose generated contribution is applied after RMSNorm."""

        return self._post_feedforward_delta_ordinals

    @property
    def additive_post_feedforward_layer_ordinals(self) -> tuple[int, ...]:
        """Layer receiving a non-owning zero-mean post-RMSNorm residual."""

        ordinal = self._additive_post_feedforward_layer_ordinal
        return () if ordinal is None else (ordinal,)

    @property
    def additive_post_feedforward_scale(self) -> float:
        return self._additive_post_feedforward_scale

    @property
    def lowering_artifact_sha256s(self) -> tuple[str, ...]:
        """Lowering commitments in graph traversal order."""

        return tuple(
            bound.lowering.artifact_sha256 for bound in self._bound_nodes
        )

    @property
    def additive_post_feedforward_lowering_artifact_sha256s(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            lowering.artifact_sha256
            for lowering in self._additive_post_feedforward_lowerings
        )

    @property
    def peak_live_modal_width(self) -> int:
        additive = self.additive_post_feedforward_graph_runtime
        return self.graph_runtime.peak_live_modal_width + (
            0 if additive is None else additive.peak_live_modal_width
        )

    def _validate_live_state(self) -> None:
        self.graph_plan.validate_integrity()
        for lowering in self._lowerings:
            ModalGeneratorLowering.from_state_dict(lowering.state_dict())
        additive_plan = self._additive_post_feedforward_graph_plan
        additive_runtime = self.additive_post_feedforward_graph_runtime
        if additive_plan is not None:
            additive_plan.validate_integrity()
            for lowering in self._additive_post_feedforward_lowerings:
                restored = ModalGeneratorLowering.from_state_dict(
                    lowering.state_dict()
                )
                if bool(
                    torch.count_nonzero(
                        restored.computational_mode_basis.mean_bias
                    )
                ):
                    raise ValueError(
                        "additive post-feed-forward lowering gained a mean"
                    )
            if (
                additive_runtime is None
                or self._additive_graph_runtime_fingerprint is None
                or module_state_fingerprint(additive_runtime)
                != self._additive_graph_runtime_fingerprint
            ):
                raise ValueError(
                    "registered additive graph runtime coefficients drifted"
                )
        elif (
            additive_runtime is not None
            or self._additive_post_feedforward_lowerings
            or self._additive_graph_runtime_fingerprint is not None
        ):
            raise ValueError("additive graph runtime state is inconsistent")
        if self.adapter.model_fingerprint() != self._source_model_sha256:
            raise ValueError("live Gemma model fingerprint drifted")
        if module_state_fingerprint(
            self.graph_runtime
        ) != self._graph_runtime_fingerprint:
            raise ValueError("registered graph runtime coefficients drifted")
        model = self.adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError("live Gemma model is no longer frozen eval")
        for ordinal in self._affected_ordinals:
            _, _, _, _, source_mlp = _feed_forward_metadata(
                self.adapter,
                ordinal,
            )
            compact = self.compiled_mlps[str(ordinal)]
            if (
                module_state_fingerprint(source_mlp)
                != self._source_fingerprints[ordinal]
                or module_state_fingerprint(compact)
                != self._compact_fingerprints[ordinal]
            ):
                raise ValueError("source or compact Gemma MLP state drifted")

    @contextmanager
    def validated_transaction(self) -> Iterator[None]:
        """Validate once around a sequence of qualification executions.

        Ordinary executions authenticate live state at every call boundary.
        A frozen qualification invokes the same executor repeatedly without
        permitting mutation between calls; this context preserves that
        integrity boundary while amortizing the expensive whole-model and
        lowering fingerprints across the transaction.

        Transactions are deliberately neither reentrant nor nestable. State
        is fully validated before the context becomes active and again while
        leaving it, including when its body raises.
        """

        if self._active:
            raise RuntimeError(
                "cannot start a validated transaction during graph execution"
            )
        if self._validated_transaction_active:
            raise RuntimeError("validated transactions cannot be nested")
        self._validate_live_state()
        self._validated_transaction_active = True
        try:
            yield
        finally:
            self._validated_transaction_active = False
            self._validate_live_state()

    def execute_graph_inputs(
        self,
        boundary_inputs: Mapping[str, Tensor],
        *,
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
        capture_routing: bool = False,
    ) -> ModalGeneratorGraphExecution:
        """Execute the copied incremental runtime without invoking Gemma."""

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if not self._validated_transaction_active:
            self._validate_live_state()
        if not isinstance(boundary_inputs, Mapping):
            raise TypeError("boundary_inputs must be a mapping")
        if set(boundary_inputs) != set(self.graph_plan.input_boundary_widths):
            raise ValueError("graph boundary input catalog mismatch")
        for boundary, width in self.graph_plan.input_boundary_widths.items():
            value = boundary_inputs[boundary]
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.ndim < 1
                or value.numel() == 0
                or value.shape[-1] != width
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"boundary {boundary!r} must be a finite nonempty "
                    f"floating Tensor with trailing width {width}"
                )
        session = self.graph_runtime.start(
            self.graph_plan,
            condition="generated",
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
            capture_routing=capture_routing,
        )
        for ordinal in self._affected_ordinals:
            nodes = self.graph_runtime.nodes_by_layer[ordinal]
            boundary = nodes[0].input_boundary
            values = boundary_inputs[boundary]
            if any(node.input_boundary != boundary for node in nodes):
                raise RuntimeError("one layer has multiple graph input boundaries")
            session.execute_layer(ordinal, values)
        return session.finish()

    def execute_additive_post_feedforward_graph_inputs(
        self,
        normalized_inputs: Tensor,
        *,
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
        capture_routing: bool = False,
    ) -> ModalGeneratorGraphExecution:
        """Execute only the non-owning additive graph on one layer input."""

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        plan = self._additive_post_feedforward_graph_plan
        runtime = self.additive_post_feedforward_graph_runtime
        ordinal = self._additive_post_feedforward_layer_ordinal
        if plan is None or runtime is None or ordinal is None:
            raise RuntimeError("executor has no additive post-feed-forward graph")
        if (
            not isinstance(normalized_inputs, Tensor)
            or not normalized_inputs.is_floating_point()
            or normalized_inputs.ndim < 1
            or normalized_inputs.numel() == 0
            or not bool(torch.isfinite(normalized_inputs).all())
        ):
            raise ValueError(
                "normalized_inputs must be a finite nonempty floating Tensor"
            )
        boundaries = plan.input_boundary_widths
        if len(boundaries) != 1:
            raise RuntimeError("one-layer additive graph has multiple inputs")
        expected_width = next(iter(boundaries.values()))
        if normalized_inputs.shape[-1] != expected_width:
            raise ValueError("additive graph input width drifted")
        if not self._validated_transaction_active:
            self._validate_live_state()
        session = runtime.start(
            plan,
            condition="generated",
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
            capture_routing=capture_routing,
        )
        session.execute_layer(ordinal, normalized_inputs)
        return session.finish()

    def execute_compact_mlp_rows(
        self,
        layer_ordinal: int,
        normalized_inputs: Tensor,
    ) -> Tensor:
        """Replay one authenticated compact MLP without executing its graph.

        Activation collectors intentionally canonicalize rows as CPU float64
        tensors.  A compact Gemma MLP, however, must be evaluated with the
        dtype and device used by its copied runtime weights if its output is to
        be the exact retained-native term used by the compiled executor.  This
        method owns that cast, processes at most 2,048 rows at once to bound
        device-local MLP intermediates, and returns one ordered detached CPU
        float64 row matrix.

        No modal state is generated and no interaction message is evaluated.
        The live source model, copied compact MLP, and graph runtime are
        authenticated at both call boundaries outside a validated transaction.
        """

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if type(layer_ordinal) is not int:
            raise TypeError("layer_ordinal must be an integer")
        if layer_ordinal not in self._affected_ordinals:
            raise ValueError("layer is not part of the compact Gemma graph")
        if (
            not isinstance(normalized_inputs, Tensor)
            or normalized_inputs.ndim != 2
            or normalized_inputs.shape[0] <= 0
            or not normalized_inputs.is_floating_point()
            or not bool(torch.isfinite(normalized_inputs).all())
        ):
            raise ValueError(
                "normalized_inputs must be a finite nonempty floating row matrix"
            )
        if not self._validated_transaction_active:
            self._validate_live_state()

        compact = self.compiled_mlps[str(layer_ordinal)]
        if normalized_inputs.shape[1] != compact.residual_width:
            raise ValueError("compact MLP row input width drifted")
        runtime_weight = compact.gate_proj.weight
        self._active = True
        try:
            output_chunks: list[Tensor] = []
            with torch.no_grad():
                for start in range(
                    0,
                    normalized_inputs.shape[0],
                    _COMPACT_ROW_CHUNK_SIZE,
                ):
                    stop = min(
                        start + _COMPACT_ROW_CHUNK_SIZE,
                        normalized_inputs.shape[0],
                    )
                    runtime_inputs = normalized_inputs[start:stop].detach().to(
                        device=runtime_weight.device,
                        dtype=runtime_weight.dtype,
                    ).contiguous()
                    runtime_output = compact(runtime_inputs)
                    output_chunks.append(
                        runtime_output.detach().to(
                            device="cpu",
                            dtype=torch.float64,
                        ).contiguous()
                    )
            output = torch.cat(output_chunks, dim=0)
            if (
                output.shape
                != (normalized_inputs.shape[0], compact.residual_width)
                or not bool(torch.isfinite(output).all())
            ):
                raise RuntimeError("compact MLP row output is invalid")
            return output
        finally:
            self._active = False
            if not self._validated_transaction_active:
                self._validate_live_state()

    def execute_compact_post_feedforward_delta_rows(
        self,
        layer_ordinal: int,
        normalized_inputs: Tensor,
    ) -> Tensor:
        """Replay the exact compact MLP followed by the live Gemma RMSNorm.

        Rows are cast to the compact runtime's source dtype/device, evaluated
        in bounded chunks, and returned as detached CPU float64 values.  The
        graph is never executed, so this is the retained post-feed-forward
        delta used by post-norm correction targets and deletion baselines.
        """

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if type(layer_ordinal) is not int:
            raise TypeError("layer_ordinal must be an integer")
        if layer_ordinal not in self._affected_ordinals:
            raise ValueError("layer is not part of the compact Gemma graph")
        if (
            not isinstance(normalized_inputs, Tensor)
            or normalized_inputs.ndim != 2
            or normalized_inputs.shape[0] <= 0
            or not normalized_inputs.is_floating_point()
            or not bool(torch.isfinite(normalized_inputs).all())
        ):
            raise ValueError(
                "normalized_inputs must be a finite nonempty floating row matrix"
            )
        if not self._validated_transaction_active:
            self._validate_live_state()

        compact = self.compiled_mlps[str(layer_ordinal)]
        if normalized_inputs.shape[1] != compact.residual_width:
            raise ValueError("compact MLP row input width drifted")
        _, post_norm = _post_feed_forward_metadata(
            self.adapter,
            layer_ordinal,
        )
        runtime_weight = compact.gate_proj.weight
        self._active = True
        try:
            output_chunks: list[Tensor] = []
            with torch.no_grad():
                for start in range(
                    0,
                    normalized_inputs.shape[0],
                    _COMPACT_ROW_CHUNK_SIZE,
                ):
                    stop = min(
                        start + _COMPACT_ROW_CHUNK_SIZE,
                        normalized_inputs.shape[0],
                    )
                    runtime_inputs = normalized_inputs[start:stop].detach().to(
                        device=runtime_weight.device,
                        dtype=runtime_weight.dtype,
                    ).contiguous()
                    runtime_output = post_norm(compact(runtime_inputs))
                    output_chunks.append(
                        runtime_output.detach().to(
                            device="cpu",
                            dtype=torch.float64,
                        ).contiguous()
                    )
            output = torch.cat(output_chunks, dim=0)
            if (
                output.shape
                != (normalized_inputs.shape[0], compact.residual_width)
                or not bool(torch.isfinite(output).all())
            ):
                raise RuntimeError(
                    "compact post-feed-forward delta row output is invalid"
                )
            return output
        finally:
            self._active = False
            if not self._validated_transaction_active:
                self._validate_live_state()

    def run_with_generated_overlay(
        self,
        callback: Callable[[], _CallbackResult],
        *,
        expected_forward_calls: int = 1,
    ) -> _CallbackResult:
        """Run a synchronous callback under the generated graph overlay.

        This is the instrumentation path used when a downstream activation
        stream must be collected on the compiled trajectory.  Each complete
        model forward receives a fresh incremental graph session, while
        cross-layer messages remain live between affected layers inside that
        forward.  The callback must consume any lazy iterator before returning.
        """

        return self._run_with_generated_overlay(
            callback,
            expected_forward_calls=expected_forward_calls,
            diagnostic_scale_by_layer={},
            diagnostic_correction_provider_by_layer={},
        )

    def run_with_diagnostic_post_feedforward_delta_attenuation(
        self,
        callback: Callable[[], _CallbackResult],
        *,
        layer_ordinal: int,
        alpha: float = 1.0,
        expected_forward_calls: int = 1,
    ) -> _CallbackResult:
        """Trace with one generated post-MLP block correction scaled by alpha.

        This is a diagnostic-only intervention at an authenticated
        ``layer.<ordinal>.mlp.delta`` boundary.  It scales the decoded graph
        correction after the live compact MLP's RMSNorm and before residual
        addition.  It does not mutate or re-hash the graph, lowerings, or
        bundle, and deliberately returns no resource-accounting execution
        record.  ``alpha=1`` takes the exact ordinary overlay arithmetic path.
        """

        if type(layer_ordinal) is not int:
            raise TypeError("layer_ordinal must be an integer")
        if layer_ordinal not in self._post_feedforward_delta_ordinals:
            raise ValueError(
                "diagnostic attenuation requires a declared "
                "post-feed-forward delta layer"
            )
        if type(alpha) not in (int, float):
            raise TypeError("alpha must be a real scalar")
        alpha_value = float(alpha)
        if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
            raise ValueError("alpha must be finite and lie in [0, 1]")
        return self._run_with_generated_overlay(
            callback,
            expected_forward_calls=expected_forward_calls,
            diagnostic_scale_by_layer={layer_ordinal: alpha_value},
            diagnostic_correction_provider_by_layer={},
        )

    def run_with_diagnostic_post_feedforward_delta_override(
        self,
        callback: Callable[[], _CallbackResult],
        *,
        layer_ordinal: int,
        correction_provider: _DiagnosticCorrectionProvider,
        expected_forward_calls: int = 1,
    ) -> _CallbackResult:
        """Trace with one generated post-MLP block correction overridden.

        This is a diagnostic-only intervention at an authenticated
        ``layer.<ordinal>.mlp.delta`` boundary.  The ordinary composed graph
        still executes, then ``correction_provider`` receives its decoded
        correction with the exact live batch, sequence, width, dtype, and
        device.  The provider must return a finite Tensor with those properties
        unchanged.  That ephemeral replacement is added to the retained
        compact post-RMSNorm delta; no artifact is mutated or re-hashed and no
        resource-accounting execution record is produced.

        The provider sees every runtime position, including padding.  It is
        responsible for preserving or replacing those positions explicitly;
        this method performs no row packing, causal omission, or padding fill.
        """

        if type(layer_ordinal) is not int:
            raise TypeError("layer_ordinal must be an integer")
        if layer_ordinal not in self._post_feedforward_delta_ordinals:
            raise ValueError(
                "diagnostic override requires a declared post-feed-forward "
                "delta layer"
            )
        if not callable(correction_provider):
            raise TypeError("correction_provider must be callable")
        return self._run_with_generated_overlay(
            callback,
            expected_forward_calls=expected_forward_calls,
            diagnostic_scale_by_layer={},
            diagnostic_correction_provider_by_layer={
                layer_ordinal: correction_provider
            },
        )

    def _run_with_generated_overlay(
        self,
        callback: Callable[[], _CallbackResult],
        *,
        expected_forward_calls: int,
        diagnostic_scale_by_layer: Mapping[int, float],
        diagnostic_correction_provider_by_layer: Mapping[
            int,
            _DiagnosticCorrectionProvider,
        ],
    ) -> _CallbackResult:

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if (
            type(expected_forward_calls) is not int
            or expected_forward_calls <= 0
        ):
            raise ValueError(
                "expected_forward_calls must be a positive integer"
            )
        if not self._validated_transaction_active:
            self._validate_live_state()

        model = self.adapter.module
        layers = getattr(getattr(model, "model"), "layers")
        original_mlps: dict[int, nn.Module] = {}
        original_post_norms: dict[int, nn.Module] = {}
        additive_ordinal = self._additive_post_feedforward_layer_ordinal
        buffer_scales = dict(diagnostic_scale_by_layer)
        if additive_ordinal is not None:
            if additive_ordinal in buffer_scales:
                raise ValueError(
                    "diagnostic and additive post-feed-forward scales overlap"
                )
            buffer_scales[additive_ordinal] = (
                self._additive_post_feedforward_scale
            )
        post_delta_buffer = _PostFeedForwardDeltaBuffer(
            buffer_scales,
            diagnostic_correction_provider_by_layer,
        )
        state = _RepeatedGraphOverlayState(
            runtime=self.graph_runtime,
            plan=self.graph_plan,
            expected_forward_calls=expected_forward_calls,
            calls={},
        )
        additive_runtime = self.additive_post_feedforward_graph_runtime
        additive_plan = self._additive_post_feedforward_graph_plan
        additive_state = (
            None
            if additive_runtime is None or additive_plan is None
            else _RepeatedGraphOverlayState(
                runtime=additive_runtime,
                plan=additive_plan,
                expected_forward_calls=expected_forward_calls,
                calls={},
            )
        )
        additive_parameters = (
            0 if additive_plan is None else additive_plan.parameter_count
        )
        expected_candidate_parameters = (
            sum(parameter.numel() for parameter in model.parameters())
            - self._removed_parameters
            + self.graph_plan.parameter_count
            + additive_parameters
        )
        self._active = True
        try:
            for ordinal in self._affected_ordinals:
                original = getattr(layers[ordinal], "mlp")
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                original_mlps[ordinal] = original
                if ordinal == additive_ordinal:
                    if additive_state is None:
                        raise RuntimeError(
                            "additive post-feed-forward runtime is unavailable"
                        )
                    original_post_norm = getattr(
                        layers[ordinal],
                        "post_feedforward_layernorm",
                        None,
                    )
                    if not isinstance(original_post_norm, nn.Module):
                        raise TypeError(
                            "live Gemma post-feed-forward norm is invalid"
                        )
                    original_post_norms[ordinal] = original_post_norm
                    layers[ordinal].mlp = (
                        _RepeatedGraphWithAdditivePostFeedForwardMLPOverlay(
                            self.compiled_mlps[str(ordinal)],
                            layer_ordinal=ordinal,
                            source_state=state,
                            additive_state=additive_state,
                            buffer=post_delta_buffer,
                        )
                    )
                    layers[ordinal].post_feedforward_layernorm = (
                        _PostFeedForwardNormOverlay(
                            original_post_norm,
                            layer_ordinal=ordinal,
                            buffer=post_delta_buffer,
                        )
                    )
                elif ordinal in self._post_feedforward_delta_ordinals:
                    original_post_norm = getattr(
                        layers[ordinal],
                        "post_feedforward_layernorm",
                        None,
                    )
                    if not isinstance(original_post_norm, nn.Module):
                        raise TypeError(
                            "live Gemma post-feed-forward norm is invalid"
                        )
                    original_post_norms[ordinal] = original_post_norm
                    layers[ordinal].mlp = (
                        _RepeatedGraphPostFeedForwardMLPOverlay(
                            self.compiled_mlps[str(ordinal)],
                            layer_ordinal=ordinal,
                            state=state,
                            buffer=post_delta_buffer,
                        )
                    )
                    layers[ordinal].post_feedforward_layernorm = (
                        _PostFeedForwardNormOverlay(
                            original_post_norm,
                            layer_ordinal=ordinal,
                            buffer=post_delta_buffer,
                        )
                    )
                else:
                    layers[ordinal].mlp = _RepeatedGraphMLPOverlay(
                        self.compiled_mlps[str(ordinal)],
                        layer_ordinal=ordinal,
                        state=state,
                    )
            compact_model_parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )
            if (
                compact_model_parameters
                + self.graph_plan.parameter_count
                + additive_parameters
                != expected_candidate_parameters
            ):
                raise RuntimeError(
                    "Gemma callback graph parameter accounting drifted"
                )
            result = callback()
            if (
                state.session is not None
                or not post_delta_buffer.empty
                or state.completed_forward_calls != expected_forward_calls
                or set(state.calls) != set(self._affected_ordinals)
                or any(
                    calls != expected_forward_calls
                    for calls in state.calls.values()
                )
                or (
                    additive_state is not None
                    and (
                        additive_state.session is not None
                        or additive_state.completed_forward_calls
                        != expected_forward_calls
                        or set(additive_state.calls) != {additive_ordinal}
                        or any(
                            calls != expected_forward_calls
                            for calls in additive_state.calls.values()
                        )
                    )
                )
            ):
                raise RuntimeError(
                    "not every callback graph overlay executed exactly "
                    f"{expected_forward_calls} times"
                )
            if isinstance(result, Iterator) or inspect.isawaitable(result):
                raise TypeError(
                    "callback result must be fully materialized before the "
                    "generated graph overlay is restored"
                )
        finally:
            for ordinal, original in original_post_norms.items():
                layers[ordinal].post_feedforward_layernorm = original
            for ordinal, original in original_mlps.items():
                layers[ordinal].mlp = original
            self._active = False
            if not self._validated_transaction_active:
                self._validate_live_state()
        return result

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "generated",
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
        capture_routing: bool = False,
    ) -> Gemma3ModalGeneratorGraphExecution:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'generated' or 'deletion'")
        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        if not self._validated_transaction_active:
            self._validate_live_state()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        source_parameters = sum(
            parameter.numel() for parameter in self.adapter.module.parameters()
        )
        additive_plan = self._additive_post_feedforward_graph_plan
        additive_runtime = self.additive_post_feedforward_graph_runtime
        additive_parameters = (
            0 if additive_plan is None else additive_plan.parameter_count
        )
        expected_candidate_parameters = (
            source_parameters
            - self._removed_parameters
            + self.graph_plan.parameter_count
            + additive_parameters
        )
        session = self.graph_runtime.start(
            self.graph_plan,
            condition=condition,
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
            capture_routing=capture_routing,
            logical_valid_mask=context.query_valid_mask,
        )
        additive_session = (
            None
            if additive_plan is None or additive_runtime is None
            else additive_runtime.start(
                additive_plan,
                condition=condition,
                capture_modal_states=capture_modal_states,
                capture_edge_messages=capture_edge_messages,
                capture_routing=capture_routing,
                logical_valid_mask=context.query_valid_mask,
            )
        )
        additive_ordinal = self._additive_post_feedforward_layer_ordinal
        layers = getattr(getattr(self.adapter.module, "model"), "layers")
        original_mlps: dict[int, nn.Module] = {}
        original_post_norms: dict[int, nn.Module] = {}
        post_delta_buffer = _PostFeedForwardDeltaBuffer(
            (
                {}
                if additive_ordinal is None
                else {
                    additive_ordinal: self._additive_post_feedforward_scale
                }
            )
        )
        self._active = True
        try:
            for ordinal in self._affected_ordinals:
                original = getattr(layers[ordinal], "mlp")
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                original_mlps[ordinal] = original
                if ordinal == additive_ordinal:
                    if additive_session is None:
                        raise RuntimeError(
                            "additive post-feed-forward session is unavailable"
                        )
                    original_post_norm = getattr(
                        layers[ordinal],
                        "post_feedforward_layernorm",
                        None,
                    )
                    if not isinstance(original_post_norm, nn.Module):
                        raise TypeError(
                            "live Gemma post-feed-forward norm is invalid"
                        )
                    original_post_norms[ordinal] = original_post_norm
                    layers[ordinal].mlp = (
                        _GraphWithAdditivePostFeedForwardMLPOverlay(
                            self.compiled_mlps[str(ordinal)],
                            layer_ordinal=ordinal,
                            source_session=session,
                            additive_session=additive_session,
                            buffer=post_delta_buffer,
                        )
                    )
                    layers[ordinal].post_feedforward_layernorm = (
                        _PostFeedForwardNormOverlay(
                            original_post_norm,
                            layer_ordinal=ordinal,
                            buffer=post_delta_buffer,
                        )
                    )
                elif ordinal in self._post_feedforward_delta_ordinals:
                    original_post_norm = getattr(
                        layers[ordinal],
                        "post_feedforward_layernorm",
                        None,
                    )
                    if not isinstance(original_post_norm, nn.Module):
                        raise TypeError(
                            "live Gemma post-feed-forward norm is invalid"
                        )
                    original_post_norms[ordinal] = original_post_norm
                    layers[ordinal].mlp = _GraphPostFeedForwardMLPOverlay(
                        self.compiled_mlps[str(ordinal)],
                        layer_ordinal=ordinal,
                        session=session,
                        buffer=post_delta_buffer,
                    )
                    layers[ordinal].post_feedforward_layernorm = (
                        _PostFeedForwardNormOverlay(
                            original_post_norm,
                            layer_ordinal=ordinal,
                            buffer=post_delta_buffer,
                        )
                    )
                else:
                    layers[ordinal].mlp = _GraphMLPOverlay(
                        self.compiled_mlps[str(ordinal)],
                        layer_ordinal=ordinal,
                        session=session,
                    )
            compact_model_parameters = sum(
                parameter.numel()
                for parameter in self.adapter.module.parameters()
            )
            if (
                compact_model_parameters
                + self.graph_plan.parameter_count
                + additive_parameters
                != expected_candidate_parameters
            ):
                raise RuntimeError(
                    "Gemma graph candidate parameter accounting drifted"
                )
            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            model_output = self.adapter.module(**call_inputs)
            graph_execution = session.finish()
            additive_graph_execution = (
                None
                if additive_session is None
                else additive_session.finish()
            )
            if not post_delta_buffer.empty:
                raise RuntimeError(
                    "post-feed-forward graph corrections were not consumed"
                )
        finally:
            for ordinal, original in original_post_norms.items():
                layers[ordinal].post_feedforward_layernorm = original
            for ordinal, original in original_mlps.items():
                layers[ordinal].mlp = original
            self._active = False
            # Validate after restoration on both success and failure.  Keeping
            # this inside ``finally`` closes the same error-path integrity gap
            # already covered by the repeated-callback executor.
            if not self._validated_transaction_active:
                self._validate_live_state()
        removed_macs = valid_tokens * self._removed_parameters
        graph_macs = valid_tokens * (
            self.graph_plan.macs_per_token
            + (0 if additive_plan is None else additive_plan.macs_per_token)
        )
        static_interaction_macs_per_token = sum(
            edge.macs_per_token
            for edge in self.graph_plan.interactions
            if isinstance(edge, ModalGeneratorInteraction)
        )
        always_executed_macs_per_token = (
            self.graph_plan.accounting.node_macs_per_token
            + static_interaction_macs_per_token
            + self.graph_plan.conditional_routing_macs_per_token
        )
        selected_conditional_macs = sum(
            session.logical_evaluated_edge_rows.get(edge.key, 0)
            * edge.message_macs_per_selected_token
            for edge in self.graph_plan.interactions
            if isinstance(edge, StateConditionedModalGeneratorInteraction)
        )
        additive_static_interaction_macs_per_token = (
            0
            if additive_plan is None
            else sum(
                edge.macs_per_token
                for edge in additive_plan.interactions
                if isinstance(edge, ModalGeneratorInteraction)
            )
        )
        additive_always_executed_macs_per_token = (
            0
            if additive_plan is None
            else (
                additive_plan.accounting.node_macs_per_token
                + additive_static_interaction_macs_per_token
                + additive_plan.conditional_routing_macs_per_token
            )
        )
        additive_selected_conditional_macs = (
            0
            if additive_plan is None or additive_session is None
            else sum(
                additive_session.logical_evaluated_edge_rows.get(edge.key, 0)
                * edge.message_macs_per_selected_token
                for edge in additive_plan.interactions
                if isinstance(edge, StateConditionedModalGeneratorInteraction)
            )
        )
        executed_graph_macs = (
            valid_tokens
            * (
                always_executed_macs_per_token
                + additive_always_executed_macs_per_token
            )
            + selected_conditional_macs
            + additive_selected_conditional_macs
            if condition == "generated"
            else 0
        )
        graph_additions = (
            valid_tokens
            * (
                self.graph_plan.accounting.elementwise_additions_per_token
                + (
                    0
                    if additive_plan is None
                    else additive_plan.accounting.elementwise_additions_per_token
                )
            )
        )
        executed_graph_additions = (
            graph_additions if condition == "generated" else 0
        )
        return Gemma3ModalGeneratorGraphExecution(
            model_output=model_output,
            graph_execution=graph_execution,
            condition=condition,
            replaced_layer_count=len(self._affected_ordinals),
            graph_node_count=(
                len(self.graph_plan.nodes)
                + (0 if additive_plan is None else len(additive_plan.nodes))
            ),
            fragment_count=(
                self._fragment_count
                + (0 if additive_plan is None else len(additive_plan.nodes))
            ),
            removed_mode_count=self._removed_mode_count,
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=(
                expected_candidate_parameters
            ),
            native_removed_learned_parameters=self._removed_parameters,
            modal_graph_learned_parameters=(
                self.graph_plan.parameter_count + additive_parameters
            ),
            net_stored_parameter_savings=(
                self._removed_parameters
                - self.graph_plan.parameter_count
                - additive_parameters
            ),
            valid_tokens=valid_tokens,
            logical_linear_macs_native_removed=removed_macs,
            logical_modal_graph_macs=graph_macs,
            logical_executed_modal_graph_macs=executed_graph_macs,
            logical_modal_graph_additions=graph_additions,
            logical_executed_modal_graph_additions=(
                executed_graph_additions
            ),
            net_logical_macs_saved=removed_macs - executed_graph_macs,
            peak_live_modal_width=(
                self.peak_live_modal_width
                if condition == "generated"
                else 0
            ),
            replacement_scope=self._replacement_scope,
            additive_graph_execution=additive_graph_execution,
        )

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3ModalGeneratorGraphExecution:
        return self.run(model_inputs, condition="generated")


__all__ = [
    "Gemma3GraphCompactMLP",
    "Gemma3ModalGeneratorGraphExecution",
    "Gemma3ModalGeneratorGraphExecutor",
]
