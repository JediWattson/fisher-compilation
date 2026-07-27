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

The ``deletion`` condition uses the identical compact native MLPs while
suppressing the graph.  It is therefore architecture-matched and never reads
the removed native coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
)
from .modal_generator_lowering import ModalGeneratorLowering
from .parameter_cluster_fragments import ParameterClusterLayerFragment


_CONDITIONS = frozenset(("generated", "deletion"))
_RUNTIME_STORAGE = "registered_copied_device_local_graph_parameters"


def _frozen_parameter(value: Tensor, *, like: Tensor) -> nn.Parameter:
    copied = value.detach().to(
        device=like.device,
        dtype=like.dtype,
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
            _DeviceEdgeRuntime(edge, like=like)
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
            incoming[edge.target_node].append(edge)
        self._incoming = {
            name: tuple(edges) for name, edges in incoming.items()
        }
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

    def start(
        self,
        plan: ModalGeneratorGraphPlan,
        *,
        condition: str,
        capture_modal_states: bool,
        capture_edge_messages: bool,
    ) -> _IncrementalGraphSession:
        return _IncrementalGraphSession(
            runtime=self,
            plan=plan,
            condition=condition,
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
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
    ) -> None:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'generated' or 'deletion'")
        self.runtime = runtime
        self.plan = plan
        self.condition = condition
        self.states: dict[str, Tensor] = {}
        self.outputs: dict[str, Tensor] = {}
        self.modal_states = {} if capture_modal_states else None
        self.edge_messages = {} if capture_edge_messages else None
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
            if not bool(torch.isfinite(state).all()):
                raise ValueError(
                    f"modal state for node {node.name!r} became non-finite"
                )
            self.states[node.name] = state
            self.executed_nodes.append(node.name)
            if self.modal_states is not None:
                self.modal_states[node.name] = state.detach().clone()
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
        return ModalGeneratorGraphExecution(
            outputs=dict(self.outputs),
            traversal_order=(
                self.plan.traversal_order
                if self.condition == "generated"
                else ()
            ),
            modal_states=self.modal_states,
            edge_messages=self.edge_messages,
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


@dataclass(frozen=True, slots=True)
class Gemma3ModalGeneratorGraphExecution:
    """Model output, graph instrumentation, and exact logical accounting."""

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

    def __post_init__(self) -> None:
        if self.condition not in _CONDITIONS:
            raise ValueError("invalid Gemma modal graph condition")
        if self.graph_runtime_storage != _RUNTIME_STORAGE:
            raise ValueError("graph runtime storage label drifted")


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


class Gemma3ModalGeneratorGraphExecutor(nn.Module):
    """Overlay an authenticated modal graph during the native Gemma layer walk."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        graph_plan: ModalGeneratorGraphPlan,
        lowerings: Sequence[ModalGeneratorLowering],
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
            input_site, output_site, down_site, _, _ = (
                _feed_forward_metadata(adapter, fragment.layer_ordinal)
            )
            layer = adapter.layers[fragment.layer_ordinal]
            if (
                fragment.layer_id != layer.id
                or fragment.activation_site != down_site
                or node.input_boundary != input_site
                or node.output_boundary != output_site
                or node.input_width != layer.residual_width
                or node.output_width != layer.residual_width
                or lowering.coordinate_generator_plan.binding.input_kind
                != "native_layer_input"
                or lowering.coordinate_generator_plan.binding.input_site
                != input_site
                or lowering.coordinate_generator_plan.binding.output_site
                != output_site
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
            for tensor in (edge.message_matrix, edge.message_bias)
            if tensor.numel()
        }
        if (
            source_storage & runtime_storage
            or plan_storage & runtime_storage
        ):
            raise RuntimeError("device graph runtime aliases source/artifact")
        if graph_runtime.parameter_count != plan.parameter_count:
            raise RuntimeError("graph runtime storage accounting drifted")

        self.adapter = adapter
        self.graph_plan = plan
        self.compiled_mlps = nn.ModuleDict(compact)
        self.graph_runtime = graph_runtime
        self._lowerings = copied_lowerings
        self._bound_nodes = tuple(bound_nodes)
        self._affected_ordinals = tuple(sorted(fragments_by_layer))
        self._source_model_sha256 = source_model_sha256
        self._source_fingerprints = source_fingerprints
        self._compact_fingerprints = compact_fingerprints
        self._graph_runtime_fingerprint = module_state_fingerprint(
            graph_runtime
        )
        self._removed_mode_count = removed_mode_count
        self._removed_parameters = removed_parameters
        self._fragment_count = len(bound_nodes)
        self._active = False
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
        self.requires_grad_(False)
        self.eval()
        self._validate_live_state()

    @property
    def graph_runtime_parameter_count(self) -> int:
        return self.graph_runtime.parameter_count

    @property
    def peak_live_modal_width(self) -> int:
        return self.graph_runtime.peak_live_modal_width

    def _validate_live_state(self) -> None:
        self.graph_plan.validate_integrity()
        for lowering in self._lowerings:
            ModalGeneratorLowering.from_state_dict(lowering.state_dict())
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

    def execute_graph_inputs(
        self,
        boundary_inputs: Mapping[str, Tensor],
        *,
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
    ) -> ModalGeneratorGraphExecution:
        """Execute the copied incremental runtime without invoking Gemma."""

        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
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
        )
        for ordinal in self._affected_ordinals:
            nodes = self.graph_runtime.nodes_by_layer[ordinal]
            boundary = nodes[0].input_boundary
            values = boundary_inputs[boundary]
            if any(node.input_boundary != boundary for node in nodes):
                raise RuntimeError("one layer has multiple graph input boundaries")
            session.execute_layer(ordinal, values)
        return session.finish()

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "generated",
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
    ) -> Gemma3ModalGeneratorGraphExecution:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'generated' or 'deletion'")
        if self._active:
            raise RuntimeError("Gemma modal graph execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self._validate_live_state()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        source_parameters = sum(
            parameter.numel() for parameter in self.adapter.module.parameters()
        )
        expected_candidate_parameters = (
            source_parameters
            - self._removed_parameters
            + self.graph_plan.parameter_count
        )
        session = self.graph_runtime.start(
            self.graph_plan,
            condition=condition,
            capture_modal_states=capture_modal_states,
            capture_edge_messages=capture_edge_messages,
        )
        layers = getattr(getattr(self.adapter.module, "model"), "layers")
        originals: dict[int, nn.Module] = {}
        self._active = True
        try:
            for ordinal in self._affected_ordinals:
                original = getattr(layers[ordinal], "mlp")
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                originals[ordinal] = original
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
                compact_model_parameters + self.graph_plan.parameter_count
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
        finally:
            for ordinal, original in originals.items():
                layers[ordinal].mlp = original
            self._active = False
        self._validate_live_state()
        removed_macs = valid_tokens * self._removed_parameters
        graph_macs = valid_tokens * self.graph_plan.macs_per_token
        executed_graph_macs = graph_macs if condition == "generated" else 0
        graph_additions = (
            valid_tokens
            * self.graph_plan.accounting.elementwise_additions_per_token
        )
        executed_graph_additions = (
            graph_additions if condition == "generated" else 0
        )
        return Gemma3ModalGeneratorGraphExecution(
            model_output=model_output,
            graph_execution=graph_execution,
            condition=condition,
            replaced_layer_count=len(self._affected_ordinals),
            graph_node_count=len(self.graph_plan.nodes),
            fragment_count=self._fragment_count,
            removed_mode_count=self._removed_mode_count,
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=(
                expected_candidate_parameters
            ),
            native_removed_learned_parameters=self._removed_parameters,
            modal_graph_learned_parameters=self.graph_plan.parameter_count,
            net_stored_parameter_savings=(
                self._removed_parameters - self.graph_plan.parameter_count
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
