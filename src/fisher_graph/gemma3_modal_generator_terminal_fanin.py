"""Core machinery for a terminal-only multi-fragment Gemma fan-in rung.

This module deliberately stops before CLI, artifact, and final-assessment
orchestration.  It provides the smallest causal interaction experiment whose
offline fitting rows match the physical runtime trajectory:

* select top-Fisher fragments from distinct layers;
* fit/lower each fragment independently outside this module;
* execute those lowerings as an edgeless compact-MLP overlay;
* capture the actual modal states and a native removed-fragment teacher at the
  shifted input seen by each compact MLP; and
* fit interactions only from earlier nodes into the last selected node.

Because interactions target only the terminal compiled node, accepted edges
cannot change any captured upstream source trajectory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .fisher_prompt_clustering import FisherPromptClusterPlan
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_executor import (
    _apply_activation,
    _validate_source_mlp,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .modal_compiler_pipeline import (
    ModalCompilerPipeline,
    ModalSourceReplacementAccounting,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_interaction_fitting import (
    ModalInteractionSelection,
    select_modal_interactions_greedily,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from .parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPParameterGroupCatalog,
)
from .prompt_mode_tracing import PromptModeTrace
from .streaming_analysis import ActivationScoreGradientRows


__all__ = [
    "AlignedFragmentRows",
    "DistinctLayerFragmentSelection",
    "EdgelessTerminalFanInPlan",
    "EdgelessTerminalFanInRows",
    "TerminalFanInCompilation",
    "build_edgeless_terminal_fanin_plan",
    "collect_aligned_fragment_rows",
    "collect_edgeless_terminal_fanin_rows",
    "fit_terminal_fanin_compilation",
    "select_top_distinct_layer_fragments",
]


_ROW_KEY_DOMAIN = b"fisher_graph.gemma3_terminal_fanin.row_keys.v1\0"


def _row_key_sha256(row_keys: tuple[tuple[str, int], ...]) -> str:
    encoded = json.dumps(
        row_keys,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_ROW_KEY_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _canonical_fragments(
    fragments: Sequence[ParameterClusterLayerFragment],
) -> tuple[ParameterClusterLayerFragment, ...]:
    if isinstance(fragments, (str, bytes)) or not isinstance(
        fragments,
        Sequence,
    ):
        raise TypeError("fragments must be a sequence")
    values = tuple(fragments)
    if len(values) < 2 or any(
        not isinstance(value, ParameterClusterLayerFragment)
        for value in values
    ):
        raise ValueError("terminal fan-in requires at least two fragments")
    for value in values:
        value.validate_integrity()
    if len({value.artifact_sha256 for value in values}) != len(values):
        raise ValueError("selected fragments must be unique")
    if len({value.layer_ordinal for value in values}) != len(values):
        raise ValueError("terminal fan-in fragments must occupy distinct layers")
    causal = tuple(
        sorted(values, key=lambda value: (value.layer_ordinal, value.cluster_id))
    )
    return causal


@dataclass(frozen=True, slots=True)
class DistinctLayerFragmentSelection:
    """Fit-Fisher selection order plus its executable causal order."""

    fisher_order: tuple[ParameterClusterLayerFragment, ...]
    causal_order: tuple[ParameterClusterLayerFragment, ...]
    minimum_fragment_modes: int

    def __post_init__(self) -> None:
        causal = _canonical_fragments(self.fisher_order)
        if self.causal_order != causal:
            raise ValueError("causal_order does not sort the Fisher selection")
        if (
            type(self.minimum_fragment_modes) is not int
            or self.minimum_fragment_modes <= 0
            or any(
                fragment.mode_count < self.minimum_fragment_modes
                for fragment in self.fisher_order
            )
        ):
            raise ValueError("selected fragment mode minimum is invalid")

    @property
    def terminal_fragment(self) -> ParameterClusterLayerFragment:
        return self.causal_order[-1]

    @property
    def source_fragments(
        self,
    ) -> tuple[ParameterClusterLayerFragment, ...]:
        return self.causal_order[:-1]


def select_top_distinct_layer_fragments(
    fragment_plan: ParameterClusterLayerFragmentPlan,
    *,
    count: int = 4,
    minimum_fragment_modes: int = 32,
) -> DistinctLayerFragmentSelection:
    """Greedily take top fit-Fisher fragments while allowing one per layer."""

    if not isinstance(fragment_plan, ParameterClusterLayerFragmentPlan):
        raise TypeError("fragment_plan must be a fragment plan")
    fragment_plan.validate_integrity()
    if type(count) is not int or count < 2:
        raise ValueError("count must be at least two")
    if (
        type(minimum_fragment_modes) is not int
        or minimum_fragment_modes <= 0
    ):
        raise ValueError("minimum_fragment_modes must be positive")

    chosen: list[ParameterClusterLayerFragment] = []
    occupied_layers: set[int] = set()
    for fragment in fragment_plan.top_by_fisher_mass(
        fragment_plan.fragment_count
    ):
        if (
            fragment.mode_count < minimum_fragment_modes
            or fragment.layer_ordinal in occupied_layers
        ):
            continue
        chosen.append(fragment)
        occupied_layers.add(fragment.layer_ordinal)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(
            "not enough distinct eligible layers for terminal fan-in"
        )
    fisher_order = tuple(chosen)
    return DistinctLayerFragmentSelection(
        fisher_order=fisher_order,
        causal_order=tuple(
            sorted(
                fisher_order,
                key=lambda value: (
                    value.layer_ordinal,
                    value.cluster_id,
                ),
            )
        ),
        minimum_fragment_modes=minimum_fragment_modes,
    )


@dataclass(frozen=True, slots=True)
class AlignedFragmentRows:
    """Ephemeral native rows sharing exact example/logical-position keys."""

    rows_by_fragment: Mapping[str, LayerFragmentRows]
    row_keys: tuple[tuple[str, int], ...]
    row_key_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rows_by_fragment, Mapping) or not (
            self.rows_by_fragment
        ):
            raise ValueError("rows_by_fragment must be a nonempty mapping")
        copied = dict(self.rows_by_fragment)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(rows, LayerFragmentRows)
            for name, rows in copied.items()
        ):
            raise TypeError("fragment row catalog is invalid")
        observations = {rows.inputs.shape[0] for rows in copied.values()}
        sequences = {rows.sequences for rows in copied.values()}
        if observations != {len(self.row_keys)} or len(sequences) != 1:
            raise ValueError("fragment rows do not share one aligned row axis")
        if (
            not self.row_keys
            or len(self.row_keys) != len(set(self.row_keys))
            or any(
                type(key) is not tuple
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or type(key[1]) is not int
                or key[1] < 0
                for key in self.row_keys
            )
        ):
            raise ValueError("row_keys must be nonempty and unique")
        computed = _row_key_sha256(self.row_keys)
        if self.row_key_sha256 == "":
            object.__setattr__(self, "row_key_sha256", computed)
        elif self.row_key_sha256 != computed:
            raise ValueError("row-key hash mismatch")
        object.__setattr__(
            self,
            "rows_by_fragment",
            MappingProxyType(copied),
        )

    @property
    def observations(self) -> int:
        return len(self.row_keys)

    @property
    def sequences(self) -> int:
        return next(iter(self.rows_by_fragment.values())).sequences


def collect_aligned_fragment_rows(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    fragments: Sequence[ParameterClusterLayerFragment],
    down_projection_weights: Mapping[str, Tensor],
) -> AlignedFragmentRows:
    """Collect several fragment targets from one shared gradient-row replay."""

    selected = _canonical_fragments(fragments)
    if not isinstance(down_projection_weights, Mapping) or set(
        down_projection_weights
    ) != {fragment.fragment_id for fragment in selected}:
        raise ValueError(
            "down_projection_weights must exactly cover selected fragments"
        )
    expected_sites = {
        site
        for fragment in selected
        for site in (fragment.input_site, fragment.activation_site)
    }

    prepared: dict[
        str,
        tuple[ParameterClusterLayerFragment, Tensor, Tensor],
    ] = {}
    for fragment in selected:
        down = down_projection_weights[fragment.fragment_id]
        if (
            not isinstance(down, Tensor)
            or down.ndim != 2
            or not down.is_floating_point()
            or down.shape[0] != fragment.output_width
        ):
            raise ValueError("fragment down projection is invalid")
        canonical_down = down.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        if not bool(torch.isfinite(canonical_down).all()):
            raise ValueError("fragment down projection must be finite")
        index = torch.tensor(
            fragment.removed_mode_indices,
            dtype=torch.long,
        )
        if int(index.max().item()) >= canonical_down.shape[1]:
            raise ValueError("fragment modes exceed down-projection width")
        prepared[fragment.fragment_id] = (
            fragment,
            index,
            canonical_down.index_select(1, index).contiguous(),
        )

    inputs: dict[str, list[Tensor]] = {
        fragment.fragment_id: [] for fragment in selected
    }
    contributions: dict[str, list[Tensor]] = {
        fragment.fragment_id: [] for fragment in selected
    }
    fisher_weights: dict[str, list[Tensor]] = {
        fragment.fragment_id: [] for fragment in selected
    }
    row_keys: list[tuple[str, int]] = []
    seen_row_keys: set[tuple[str, int]] = set()
    sequences = 0
    iterator = iter(rows)
    try:
        for row in iterator:
            if set(row.activations) != expected_sites:
                raise ValueError(
                    "aligned fragment rows do not expose the exact site union"
                )
            if row.example_id is None:
                raise ValueError(
                    "aligned fragment rows require stable example ids"
                )
            keys = tuple(
                (row.example_id, int(position))
                for position in row.logical_positions.tolist()
            )
            if any(key in seen_row_keys for key in keys):
                raise ValueError("aligned fragment row keys are duplicated")
            row_keys.extend(keys)
            seen_row_keys.update(keys)
            for fragment_id, (fragment, index, selected_down) in (
                prepared.items()
            ):
                x = row.activations[fragment.input_site].to(
                    dtype=torch.float64
                )
                z = row.activations[fragment.activation_site].to(
                    dtype=torch.float64
                )
                gradient = row.score_gradients[
                    fragment.activation_site
                ].to(dtype=torch.float64)
                if (
                    x.shape[0] != z.shape[0]
                    or z.shape != gradient.shape
                    or x.shape[1] != fragment.input_width
                    or z.shape[1] <= int(index.max().item())
                ):
                    raise ValueError("aligned fragment row shapes disagree")
                selected_z = z.index_select(1, index)
                selected_gradient = gradient.index_select(1, index)
                inputs[fragment_id].append(x)
                contributions[fragment_id].append(
                    selected_z @ selected_down.T
                )
                fisher_weights[fragment_id].append(
                    (selected_z * selected_gradient).square().sum(dim=1)
                )
            sequences += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if sequences <= 0:
        raise ValueError("aligned fragment row stream cannot be empty")
    result = {
        fragment.fragment_id: LayerFragmentRows(
            inputs=torch.cat(inputs[fragment.fragment_id], dim=0),
            contributions=torch.cat(
                contributions[fragment.fragment_id],
                dim=0,
            ),
            fisher_weights=torch.cat(
                fisher_weights[fragment.fragment_id],
                dim=0,
            ),
            sequences=sequences,
        )
        for fragment in selected
    }
    return AlignedFragmentRows(
        rows_by_fragment=result,
        row_keys=tuple(row_keys),
    )


@dataclass(frozen=True, slots=True)
class EdgelessTerminalFanInPlan:
    selection: DistinctLayerFragmentSelection
    graph_plan: ModalGeneratorGraphPlan
    lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    fragment_id_by_node: Mapping[str, str]
    target_node: str
    source_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        self.graph_plan.validate_integrity()
        if self.graph_plan.interactions:
            raise ValueError("edgeless fan-in plan cannot contain interactions")
        lowerings = {
            name: ModalGeneratorLowering.from_state_dict(value.state_dict())
            for name, value in self.lowerings_by_node.items()
        }
        fragments = dict(self.fragment_id_by_node)
        node_names = tuple(node.name for node in self.graph_plan.nodes)
        if (
            set(lowerings) != set(node_names)
            or set(fragments) != set(node_names)
            or self.target_node != node_names[-1]
            or self.source_nodes != node_names[:-1]
        ):
            raise ValueError("edgeless fan-in node catalogs disagree")
        expected_ids = {
            fragment.fragment_id for fragment in self.selection.causal_order
        }
        if set(fragments.values()) != expected_ids:
            raise ValueError("edgeless graph does not cover selected fragments")
        for node in self.graph_plan.nodes:
            lowering = lowerings[node.name]
            if (
                node.weights.artifact_sha256
                != lowering.graph_weights.artifact_sha256
                or lowering.mode_set_id != fragments[node.name]
            ):
                raise ValueError("edgeless node does not match its lowering")
        object.__setattr__(
            self,
            "lowerings_by_node",
            MappingProxyType(lowerings),
        )
        object.__setattr__(
            self,
            "fragment_id_by_node",
            MappingProxyType(fragments),
        )

    @property
    def lowerings(self) -> tuple[ModalGeneratorLowering, ...]:
        return tuple(
            self.lowerings_by_node[node.name]
            for node in self.graph_plan.nodes
        )


def build_edgeless_terminal_fanin_plan(
    selection: DistinctLayerFragmentSelection,
    *,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    lowerings_by_fragment: Mapping[str, ModalGeneratorLowering],
) -> EdgelessTerminalFanInPlan:
    """Assemble independently fitted lowerings into one causal edgeless graph."""

    if not isinstance(selection, DistinctLayerFragmentSelection):
        raise TypeError("selection must be a distinct-layer selection")
    fragment_plan.validate_integrity()
    selected_ids = {
        fragment.fragment_id for fragment in selection.causal_order
    }
    if not isinstance(lowerings_by_fragment, Mapping) or set(
        lowerings_by_fragment
    ) != selected_ids:
        raise ValueError("lowerings must exactly cover selected fragments")

    nodes = []
    lowerings_by_node: dict[str, ModalGeneratorLowering] = {}
    fragment_id_by_node: dict[str, str] = {}
    model_sha256: str | None = None
    for fragment in selection.causal_order:
        supplied = lowerings_by_fragment[fragment.fragment_id]
        lowering = ModalGeneratorLowering.from_state_dict(
            supplied.state_dict()
        )
        if (
            lowering.fragment_plan.artifact_sha256
            != fragment_plan.artifact_sha256
            or lowering.selected_fragment_sha256
            != fragment.artifact_sha256
        ):
            raise ValueError("lowering does not bind the selected fragment")
        name = f"{lowering.generator_id}.graph-node"
        node = lowering.to_graph_node(
            name=name,
            causal_order=fragment.layer_ordinal,
        )
        if model_sha256 is None:
            model_sha256 = node.weights.source_model_sha256
        elif node.weights.source_model_sha256 != model_sha256:
            raise ValueError("selected lowerings bind different models")
        nodes.append(node)
        lowerings_by_node[name] = lowering
        fragment_id_by_node[name] = fragment.fragment_id
    assert model_sha256 is not None
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=model_sha256,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=tuple(nodes),
        interactions=(),
    )
    names = tuple(node.name for node in graph.nodes)
    return EdgelessTerminalFanInPlan(
        selection=selection,
        graph_plan=graph,
        lowerings_by_node=lowerings_by_node,
        fragment_id_by_node=fragment_id_by_node,
        target_node=names[-1],
        source_nodes=names[:-1],
    )


@dataclass(frozen=True, slots=True)
class EdgelessTerminalFanInRows:
    """Actual edgeless states and shifted-input native teacher coordinates."""

    node_states: Mapping[str, Tensor]
    teacher_coordinates: Mapping[str, Tensor]
    row_keys: tuple[tuple[str, int], ...]
    row_key_sha256: str = ""

    def __post_init__(self) -> None:
        states = dict(self.node_states)
        teachers = dict(self.teacher_coordinates)
        if not states or set(states) != set(teachers):
            raise ValueError("runtime state and teacher node catalogs differ")
        for catalog, label in (
            (states, "node state"),
            (teachers, "teacher coordinate"),
        ):
            for name, value in catalog.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(value, Tensor)
                    or value.ndim != 2
                    or value.shape[0] != len(self.row_keys)
                    or not value.is_floating_point()
                ):
                    raise ValueError(f"{label} rows are invalid")
                canonical = value.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                ).contiguous()
                if not bool(torch.isfinite(canonical).all()):
                    raise ValueError(f"{label} rows must be finite")
                catalog[name] = canonical
        for name in states:
            if states[name].shape != teachers[name].shape:
                raise ValueError(
                    "runtime state and teacher coordinate widths differ"
                )
        if not self.row_keys or len(self.row_keys) != len(set(self.row_keys)):
            raise ValueError("runtime row keys must be nonempty and unique")
        computed = _row_key_sha256(self.row_keys)
        if self.row_key_sha256 == "":
            object.__setattr__(self, "row_key_sha256", computed)
        elif self.row_key_sha256 != computed:
            raise ValueError("runtime row-key hash mismatch")
        object.__setattr__(self, "node_states", MappingProxyType(states))
        object.__setattr__(
            self,
            "teacher_coordinates",
            MappingProxyType(teachers),
        )

    @property
    def observations(self) -> int:
        return len(self.row_keys)


def _batch_row_keys(
    adapter: Gemma3CausalLMAdapter,
    batch: CalibrationBatch,
) -> tuple[tuple[str, int], ...]:
    if batch.example_ids is None:
        raise ValueError("edgeless capture requires stable example ids")
    context = adapter.prepare_sequence(batch.model_inputs)
    valid = batch.valid_positions.to(device=context.logical_positions.device)
    if valid.shape != context.logical_positions.shape or bool(
        (valid & ~context.query_valid_mask).any()
    ):
        raise ValueError("batch positions do not match the adapter sequence")
    result: list[tuple[str, int]] = []
    for index, example_id in enumerate(batch.example_ids):
        positions = context.logical_positions[index, valid[index]]
        result.extend(
            (example_id, int(position))
            for position in positions.detach().to(device="cpu").tolist()
        )
    return tuple(result)


def collect_edgeless_terminal_fanin_rows(
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
    *,
    plan: EdgelessTerminalFanInPlan,
    expected_row_keys: tuple[tuple[str, int], ...],
) -> EdgelessTerminalFanInRows:
    """Capture physical edgeless states and shifted-input native teachers."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    if not isinstance(executor, Gemma3ModalGeneratorGraphExecutor):
        raise TypeError("executor must be a Gemma graph executor")
    if not isinstance(plan, EdgelessTerminalFanInPlan):
        raise TypeError("plan must be an edgeless terminal fan-in plan")
    if (
        executor.graph_plan.artifact_sha256
        != plan.graph_plan.artifact_sha256
        or executor.graph_plan.interactions
    ):
        raise ValueError("executor is not the declared edgeless graph")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError("batches must be a sequence")
    if not batches:
        raise ValueError("edgeless capture batches cannot be empty")

    fragment_by_id = {
        fragment.fragment_id: fragment
        for fragment in plan.selection.causal_order
    }
    hook_targets: dict[
        str,
        tuple[nn.Module, str, Tensor, Tensor, Tensor, Tensor, Tensor],
    ] = {}
    for node in plan.graph_plan.nodes:
        fragment = fragment_by_id[plan.fragment_id_by_node[node.name]]
        source_layer = adapter.source_module(fragment.layer_id)
        source_mlp = getattr(source_layer, "mlp", None)
        if not isinstance(source_mlp, nn.Module):
            raise TypeError("selected Gemma layer does not expose an MLP")
        gate, up, down = _validate_source_mlp(
            source_mlp,
            label="source_mlp",
        )
        transformer = adapter.layers[fragment.layer_ordinal].transformer
        if transformer is None or transformer.feed_forward is None:
            raise ValueError("selected Gemma layer lacks MLP metadata")
        index = torch.tensor(
            fragment.removed_mode_indices,
            device=gate.weight.device,
            dtype=torch.long,
        )
        lowering = plan.lowerings_by_node[node.name]
        basis = lowering.computational_mode_basis
        hook_targets[node.name] = (
            executor.compiled_mlps[str(fragment.layer_ordinal)],
            transformer.feed_forward.activation,
            gate.weight.detach().index_select(0, index).clone(),
            up.weight.detach().index_select(0, index).clone(),
            down.weight.detach().index_select(1, index).clone(),
            basis.mean_bias,
            basis.encoder_basis,
        )

    current_teachers: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for node_name, (
        compact,
        activation,
        gate_weight,
        up_weight,
        down_weight,
        mean,
        basis,
    ) in hook_targets.items():

        def capture_teacher(
            _module: nn.Module,
            arguments: tuple[Tensor, ...],
            *,
            node_name: str = node_name,
            activation: str = activation,
            gate_weight: Tensor = gate_weight,
            up_weight: Tensor = up_weight,
            down_weight: Tensor = down_weight,
            mean: Tensor = mean,
            basis: Tensor = basis,
        ) -> None:
            if len(arguments) != 1 or not isinstance(arguments[0], Tensor):
                raise RuntimeError("compact MLP pre-hook input is invalid")
            if node_name in current_teachers:
                raise RuntimeError("compact MLP executed twice in one batch")
            values = arguments[0]
            gate_values = F.linear(
                values,
                gate_weight.to(device=values.device, dtype=values.dtype),
            )
            up_values = F.linear(
                values,
                up_weight.to(device=values.device, dtype=values.dtype),
            )
            selected_hidden = _apply_activation(
                activation,
                gate_values,
            ) * up_values
            contribution = F.linear(
                selected_hidden,
                down_weight.to(device=values.device, dtype=values.dtype),
            )
            current_teachers[node_name] = (
                contribution
                - mean.to(device=values.device, dtype=values.dtype)
            ) @ basis.to(device=values.device, dtype=values.dtype)

        handles.append(compact.register_forward_pre_hook(capture_teacher))

    state_rows = {name: [] for name in plan.lowerings_by_node}
    teacher_rows = {name: [] for name in plan.lowerings_by_node}
    captured_keys: list[tuple[str, int]] = []
    try:
        for batch in batches:
            current_teachers.clear()
            keys = _batch_row_keys(adapter, batch)
            captured_keys.extend(keys)
            with torch.no_grad():
                execution = executor.run(
                    batch.model_inputs,
                    condition="generated",
                    capture_modal_states=True,
                )
            states = execution.graph_execution.modal_states
            if states is None or set(states) != set(plan.lowerings_by_node):
                raise RuntimeError("edgeless modal-state capture is incomplete")
            if set(current_teachers) != set(plan.lowerings_by_node):
                raise RuntimeError("shifted native-teacher capture is incomplete")
            for name in plan.lowerings_by_node:
                state = states[name]
                teacher = current_teachers[name]
                mask = batch.valid_positions.to(device=state.device)
                if (
                    state.shape[:-1] != mask.shape
                    or teacher.shape[:-1] != mask.shape
                ):
                    raise ValueError("captured runtime row shapes drifted")
                state_rows[name].append(
                    state[mask].detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )
                teacher_rows[name].append(
                    teacher[
                        batch.valid_positions.to(device=teacher.device)
                    ]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                )
    finally:
        for handle in handles:
            handle.remove()
    row_keys = tuple(captured_keys)
    if row_keys != expected_row_keys:
        raise ValueError(
            "edgeless runtime rows do not match native fragment row keys"
        )
    return EdgelessTerminalFanInRows(
        node_states={
            name: torch.cat(values, dim=0)
            for name, values in state_rows.items()
        },
        teacher_coordinates={
            name: torch.cat(values, dim=0)
            for name, values in teacher_rows.items()
        },
        row_keys=row_keys,
    )


@dataclass(frozen=True, slots=True)
class TerminalFanInCompilation:
    edgeless: EdgelessTerminalFanInPlan
    interaction_selection: ModalInteractionSelection
    graph_plan: ModalGeneratorGraphPlan
    source_replacement_accounting: ModalSourceReplacementAccounting
    compiler_pipeline: ModalCompilerPipeline


def fit_terminal_fanin_compilation(
    *,
    edgeless: EdgelessTerminalFanInPlan,
    fit_rows: EdgelessTerminalFanInRows,
    eval_rows: EdgelessTerminalFanInRows,
    target_fisher_weights_fit: Tensor,
    target_fisher_weights_eval: Tensor,
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    fit_split_sha256: str,
    eval_split_sha256: str,
    ridges: Sequence[float] | float = (0.0,),
    minimum_heldout_improvement: float = 1e-3,
    fit_intercept: bool = False,
) -> TerminalFanInCompilation:
    """Fit terminal-only messages and build the authenticated final pipeline."""

    if fit_rows.row_keys == eval_rows.row_keys or set(
        fit_rows.row_keys
    ) & set(eval_rows.row_keys):
        raise ValueError("fan-in fit and interaction-selection rows overlap")
    node_names = tuple(node.name for node in edgeless.graph_plan.nodes)
    if (
        set(fit_rows.node_states) != set(node_names)
        or set(eval_rows.node_states) != set(node_names)
        or set(fit_rows.teacher_coordinates) != set(node_names)
        or set(eval_rows.teacher_coordinates) != set(node_names)
    ):
        raise ValueError("fan-in runtime row node catalogs differ")
    target = edgeless.target_node
    candidate_edges = tuple(
        (source, target) for source in edgeless.source_nodes
    )
    target_residual_fit = (
        fit_rows.teacher_coordinates[target]
        - fit_rows.node_states[target]
    )
    target_residual_eval = (
        eval_rows.teacher_coordinates[target]
        - eval_rows.node_states[target]
    )
    if not math.isfinite(minimum_heldout_improvement) or (
        minimum_heldout_improvement < 0.0
    ):
        raise ValueError("minimum_heldout_improvement must be nonnegative")
    selection = select_modal_interactions_greedily(
        fit_rows.node_states,
        eval_rows.node_states,
        {target: target_residual_fit},
        {target: target_residual_eval},
        node_causal_orders={
            node.name: node.causal_order
            for node in edgeless.graph_plan.nodes
        },
        generator_artifact_sha256s={
            name: lowering.coordinate_generator_plan.artifact_sha256
            for name, lowering in edgeless.lowerings_by_node.items()
        },
        source_model_sha256=edgeless.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        candidate_edges=candidate_edges,
        ridges=ridges,
        fisher_weights_fit=target_fisher_weights_fit,
        fisher_weights_eval=target_fisher_weights_eval,
        fit_intercept=fit_intercept,
        selection_metric="weighted_nrmse",
        minimum_heldout_improvement=minimum_heldout_improvement,
        max_incoming_edges=len(candidate_edges),
        eval_split_role="open_development",
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=edgeless.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=edgeless.graph_plan.nodes,
        interactions=selection.interactions,
    )
    accounting = build_modal_source_replacement_accounting(
        parameter_catalog,
        fragment_plan,
        tuple(
            fragment.fragment_id
            for fragment in edgeless.selection.causal_order
        ),
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=fit_prompt_trace,
        parameter_catalog=parameter_catalog,
        grouped_fisher=fisher_coupling,
        fisher_clusters=parameter_clusters,
        parameter_cluster_fragments=fragment_plan,
        lowerings_by_node=edgeless.lowerings_by_node,
        graph_plan=graph,
        interaction_selection=selection,
        source_replacement_accounting=accounting,
    )
    return TerminalFanInCompilation(
        edgeless=edgeless,
        interaction_selection=selection,
        graph_plan=graph,
        source_replacement_accounting=accounting,
        compiler_pipeline=pipeline,
    )
