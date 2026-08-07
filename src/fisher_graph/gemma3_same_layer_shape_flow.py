"""Same-layer shape-and-flow scaffolding for Gemma modal generators.

This module isolates the first reusable pieces needed to study several
disjoint parameter-cluster fragments at one physical Gemma MLP boundary:

* deterministic top-Fisher fragment selection within one layer;
* an authenticated edgeless graph oriented from the highest-Fisher source,
  with strictly increasing intra-layer causal order for every target; and
* aligned capture of the graph's realized modal states (shape) and the native
  removed-fragment teacher coordinates at the exact compact-MLP input (flow).

The captured rows are ephemeral fit inputs.  No prompt text, token ids,
activation rows, gradients, or native weights are serialized here.
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
from .gemma3_modal_generator_executor import (
    _apply_activation,
    _validate_source_mlp,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
    collect_aligned_fragment_rows,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from .streaming_analysis import ActivationScoreGradientRows


__all__ = [
    "EdgelessSameLayerGraphPlan",
    "SameLayerFragmentSelection",
    "SameLayerShapeFlowRows",
    "build_edgeless_same_layer_graph",
    "collect_aligned_same_layer_fragment_rows",
    "collect_edgeless_same_layer_shape_flow_rows",
    "select_top_fisher_same_layer_fragments",
]


_SELECTION_KIND = "fisher_graph.gemma3_same_layer_fragment_selection"
_SELECTION_VERSION = 1
_SELECTION_DOMAIN = b"fisher_graph.gemma3_same_layer_selection.v1\0"
_ROW_KEY_DOMAIN = b"fisher_graph.gemma3_same_layer_shape_flow.rows.v1\0"
_FIXED_LAYER_POLICY = "fixed_layer_top_fisher"
_MAXIMUM_MASS_POLICY = "maximum_same_layer_top_fisher_mass"
_LAYER_POLICIES = frozenset({_FIXED_LAYER_POLICY, _MAXIMUM_MASS_POLICY})
_INTRA_LAYER_CAUSAL_STRIDE = 1_000_000


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _fisher_order(
    fragments: Sequence[ParameterClusterLayerFragment],
) -> tuple[ParameterClusterLayerFragment, ...]:
    return tuple(
        sorted(
            fragments,
            key=lambda fragment: (
                -fragment.fisher_mass,
                fragment.cluster_id,
                fragment.artifact_sha256,
            ),
        )
    )


def _execution_order(
    fragments: Sequence[ParameterClusterLayerFragment],
) -> tuple[ParameterClusterLayerFragment, ...]:
    # Same-layer fragments have no native temporal order.  Orient the graph
    # from the highest-Fisher fragment so the canonical source has semantic
    # meaning, then use cluster identity as a deterministic target ordering.
    ranked = _fisher_order(fragments)
    source = ranked[0]
    return (
        source,
        *sorted(
            ranked[1:],
            key=lambda fragment: (
                fragment.cluster_id,
                fragment.artifact_sha256,
            ),
        ),
    )


def _select_layer_candidate(
    fragment_plan: ParameterClusterLayerFragmentPlan,
    *,
    layer_ordinal: int,
    count: int,
    minimum_fragment_modes: int,
) -> tuple[ParameterClusterLayerFragment, ...] | None:
    ranked = _fisher_order(
        tuple(
            fragment
            for fragment in fragment_plan.fragments
            if fragment.layer_ordinal == layer_ordinal
            and fragment.mode_count >= minimum_fragment_modes
        )
    )
    selected: list[ParameterClusterLayerFragment] = []
    occupied_channels: set[int] = set()
    for fragment in ranked:
        channels = set(fragment.removed_mode_indices)
        if channels & occupied_channels:
            continue
        selected.append(fragment)
        occupied_channels.update(channels)
        if len(selected) == count:
            return tuple(selected)
    return None


def _selected_candidate(
    fragment_plan: ParameterClusterLayerFragmentPlan,
    *,
    count: int,
    minimum_fragment_modes: int,
    layer_ordinal: int | None,
) -> tuple[str, tuple[ParameterClusterLayerFragment, ...]]:
    if layer_ordinal is not None:
        selected = _select_layer_candidate(
            fragment_plan,
            layer_ordinal=layer_ordinal,
            count=count,
            minimum_fragment_modes=minimum_fragment_modes,
        )
        if selected is None:
            raise ValueError(
                "requested layer lacks enough disjoint eligible fragments"
            )
        return _FIXED_LAYER_POLICY, selected

    candidates: list[tuple[tuple[object, ...], tuple[ParameterClusterLayerFragment, ...]]] = []
    for ordinal in sorted(
        {fragment.layer_ordinal for fragment in fragment_plan.fragments}
    ):
        selected = _select_layer_candidate(
            fragment_plan,
            layer_ordinal=ordinal,
            count=count,
            minimum_fragment_modes=minimum_fragment_modes,
        )
        if selected is None:
            continue
        candidates.append(
            (
                (
                    -math.fsum(fragment.fisher_mass for fragment in selected),
                    ordinal,
                    tuple(fragment.artifact_sha256 for fragment in selected),
                ),
                selected,
            )
        )
    if not candidates:
        raise ValueError(
            "no layer contains enough disjoint eligible fragments"
        )
    candidates.sort(key=lambda item: item[0])
    return _MAXIMUM_MASS_POLICY, candidates[0][1]


@dataclass(frozen=True, slots=True)
class SameLayerFragmentSelection:
    """Authenticated top-Fisher fragments sharing one physical MLP layer."""

    source_fragment_plan_sha256: str
    source_model_sha256: str
    layer_ordinal: int
    layer_id: str
    fisher_order: tuple[ParameterClusterLayerFragment, ...]
    execution_order: tuple[ParameterClusterLayerFragment, ...]
    minimum_fragment_modes: int
    layer_selection_policy: str
    artifact_sha256: str = ""
    artifact_kind: str = _SELECTION_KIND
    format_version: int = _SELECTION_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_fragment_plan_sha256,
            label="source_fragment_plan_sha256",
        )
        _require_sha256(
            self.source_model_sha256,
            label="source_model_sha256",
        )
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise ValueError("layer_id must be nonempty")
        if (
            type(self.minimum_fragment_modes) is not int
            or self.minimum_fragment_modes <= 0
        ):
            raise ValueError("minimum_fragment_modes must be positive")
        if self.layer_selection_policy not in _LAYER_POLICIES:
            raise ValueError("layer_selection_policy is invalid")
        if (
            type(self.fisher_order) is not tuple
            or len(self.fisher_order) < 2
            or any(
                not isinstance(fragment, ParameterClusterLayerFragment)
                for fragment in self.fisher_order
            )
            or type(self.execution_order) is not tuple
        ):
            raise ValueError("same-layer selection requires at least two fragments")
        for fragment in self.fisher_order:
            fragment.validate_integrity()
        hashes = tuple(
            fragment.artifact_sha256 for fragment in self.fisher_order
        )
        if len(hashes) != len(set(hashes)):
            raise ValueError("same-layer fragments must be unique")
        if self.fisher_order != _fisher_order(self.fisher_order):
            raise ValueError("fisher_order is not canonical")
        if self.execution_order != _execution_order(self.fisher_order):
            raise ValueError("execution_order is not canonical")
        if any(
            fragment.layer_ordinal != self.layer_ordinal
            or fragment.layer_id != self.layer_id
            or fragment.source_model_sha256 != self.source_model_sha256
            or fragment.mode_count < self.minimum_fragment_modes
            for fragment in self.fisher_order
        ):
            raise ValueError("selected fragments do not share the declared layer")
        boundary_signatures = {
            (
                fragment.input_site,
                fragment.activation_site,
                fragment.output_site,
                fragment.input_width,
                fragment.output_width,
                fragment.parameter_catalog_sha256,
                fragment.source_cluster_plan_sha256,
                fragment.source_fisher_coupling_sha256,
            )
            for fragment in self.fisher_order
        }
        if len(boundary_signatures) != 1:
            raise ValueError("same-layer fragment boundary metadata differs")
        channels = tuple(
            channel
            for fragment in self.fisher_order
            for channel in fragment.removed_mode_indices
        )
        groups = tuple(
            group
            for fragment in self.fisher_order
            for group in fragment.group_indices
        )
        if (
            len(channels) != len(set(channels))
            or len(groups) != len(set(groups))
        ):
            raise ValueError("same-layer fragments must be mode-disjoint")
        if (
            self.artifact_kind != _SELECTION_KIND
            or self.format_version != _SELECTION_VERSION
        ):
            raise ValueError("same-layer selection artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256,
            label="artifact_sha256",
        ) != computed:
            raise ValueError("same-layer selection artifact hash mismatch")

    @property
    def fragment_count(self) -> int:
        return len(self.fisher_order)

    @property
    def fragment_ids(self) -> tuple[str, ...]:
        return tuple(
            fragment.fragment_id for fragment in self.execution_order
        )

    @property
    def source_fragment(self) -> ParameterClusterLayerFragment:
        """Return the highest-Fisher fragment that orients causal execution."""

        return self.execution_order[0]

    @property
    def target_fragments(
        self,
    ) -> tuple[ParameterClusterLayerFragment, ...]:
        """Return deterministic lower-Fisher targets after the source."""

        return self.execution_order[1:]

    @property
    def removed_mode_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                channel
                for fragment in self.fisher_order
                for channel in fragment.removed_mode_indices
            )
        )

    @property
    def selected_fisher_mass(self) -> float:
        return math.fsum(fragment.fisher_mass for fragment in self.fisher_order)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_fragment_plan_sha256": self.source_fragment_plan_sha256,
            "source_model_sha256": self.source_model_sha256,
            "layer_ordinal": self.layer_ordinal,
            "layer_id": self.layer_id,
            "minimum_fragment_modes": self.minimum_fragment_modes,
            "layer_selection_policy": self.layer_selection_policy,
            "fisher_order_sha256s": tuple(
                fragment.artifact_sha256 for fragment in self.fisher_order
            ),
            "execution_order_sha256s": tuple(
                fragment.artifact_sha256 for fragment in self.execution_order
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_SELECTION_DOMAIN)

    def validate_integrity(self) -> None:
        for fragment in self.fisher_order:
            fragment.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("same-layer selection artifact hash mismatch")

    def validate_against(
        self,
        fragment_plan: ParameterClusterLayerFragmentPlan,
    ) -> None:
        """Require exact membership and deterministic top-Fisher selection."""

        if not isinstance(fragment_plan, ParameterClusterLayerFragmentPlan):
            raise TypeError("fragment_plan must be a fragment plan")
        fragment_plan.validate_integrity()
        self.validate_integrity()
        if (
            fragment_plan.artifact_sha256
            != self.source_fragment_plan_sha256
            or fragment_plan.source_model_sha256 != self.source_model_sha256
        ):
            raise ValueError("same-layer selection does not bind fragment_plan")
        policy, expected = _selected_candidate(
            fragment_plan,
            count=self.fragment_count,
            minimum_fragment_modes=self.minimum_fragment_modes,
            layer_ordinal=(
                self.layer_ordinal
                if self.layer_selection_policy == _FIXED_LAYER_POLICY
                else None
            ),
        )
        if (
            policy != self.layer_selection_policy
            or tuple(fragment.artifact_sha256 for fragment in expected)
            != tuple(
                fragment.artifact_sha256 for fragment in self.fisher_order
            )
        ):
            raise ValueError("same-layer selection is not the declared top set")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "fragment_count": self.fragment_count,
            "fragment_ids": self.fragment_ids,
            "removed_mode_indices": self.removed_mode_indices,
            "selected_fisher_mass": self.selected_fisher_mass,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fisher_order": tuple(
                fragment.state_dict() for fragment in self.fisher_order
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> SameLayerFragmentSelection:
        expected = {
            "artifact_kind",
            "format_version",
            "source_fragment_plan_sha256",
            "source_model_sha256",
            "layer_ordinal",
            "layer_id",
            "minimum_fragment_modes",
            "layer_selection_policy",
            "fisher_order_sha256s",
            "execution_order_sha256s",
            "fisher_order",
            "artifact_sha256",
        }
        _strict_fields(state, expected, label="same-layer selection")
        raw_fragments = state["fisher_order"]
        if not isinstance(raw_fragments, tuple):
            raise TypeError("fisher_order must be a tuple")
        fragments = tuple(
            ParameterClusterLayerFragment.from_state_dict(value)
            for value in raw_fragments  # type: ignore[arg-type]
        )
        result = cls(
            source_fragment_plan_sha256=state[
                "source_fragment_plan_sha256"
            ],  # type: ignore[arg-type]
            source_model_sha256=state[
                "source_model_sha256"
            ],  # type: ignore[arg-type]
            layer_ordinal=state["layer_ordinal"],  # type: ignore[arg-type]
            layer_id=state["layer_id"],  # type: ignore[arg-type]
            fisher_order=fragments,
            execution_order=_execution_order(fragments),
            minimum_fragment_modes=state[
                "minimum_fragment_modes"
            ],  # type: ignore[arg-type]
            layer_selection_policy=state[
                "layer_selection_policy"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            state["fisher_order_sha256s"]
            != tuple(fragment.artifact_sha256 for fragment in fragments)
            or state["execution_order_sha256s"]
            != tuple(
                fragment.artifact_sha256
                for fragment in result.execution_order
            )
        ):
            raise ValueError("serialized same-layer fragment order drifted")
        return result


def select_top_fisher_same_layer_fragments(
    fragment_plan: ParameterClusterLayerFragmentPlan,
    *,
    count: int = 2,
    minimum_fragment_modes: int = 1,
    layer_ordinal: int | None = None,
) -> SameLayerFragmentSelection:
    """Select a deterministic mode-disjoint top-Fisher set on one layer."""

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
    if layer_ordinal is not None and (
        type(layer_ordinal) is not int or layer_ordinal < 0
    ):
        raise ValueError("layer_ordinal must be nonnegative when provided")
    policy, selected = _selected_candidate(
        fragment_plan,
        count=count,
        minimum_fragment_modes=minimum_fragment_modes,
        layer_ordinal=layer_ordinal,
    )
    first = selected[0]
    result = SameLayerFragmentSelection(
        source_fragment_plan_sha256=fragment_plan.artifact_sha256,
        source_model_sha256=fragment_plan.source_model_sha256,
        layer_ordinal=first.layer_ordinal,
        layer_id=first.layer_id,
        fisher_order=selected,
        execution_order=_execution_order(selected),
        minimum_fragment_modes=minimum_fragment_modes,
        layer_selection_policy=policy,
    )
    result.validate_against(fragment_plan)
    return result


@dataclass(frozen=True, slots=True)
class EdgelessSameLayerGraphPlan:
    """One authenticated edgeless graph over same-layer fragments."""

    selection: SameLayerFragmentSelection
    graph_plan: ModalGeneratorGraphPlan
    lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    fragment_id_by_node: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.selection, SameLayerFragmentSelection):
            raise TypeError("selection must be SameLayerFragmentSelection")
        self.selection.validate_integrity()
        if not isinstance(self.graph_plan, ModalGeneratorGraphPlan):
            raise TypeError("graph_plan must be ModalGeneratorGraphPlan")
        graph = ModalGeneratorGraphPlan.from_state_dict(
            self.graph_plan.state_dict()
        )
        if graph.interactions:
            raise ValueError("same-layer base graph must be edgeless")
        if (
            graph.model_fingerprint != self.selection.source_model_sha256
            or graph.parameter_cluster_plan_sha256
            != self.selection.source_fragment_plan_sha256
        ):
            raise ValueError("same-layer graph does not bind its selection")
        if not isinstance(self.lowerings_by_node, Mapping):
            raise TypeError("lowerings_by_node must be a mapping")
        if not isinstance(self.fragment_id_by_node, Mapping):
            raise TypeError("fragment_id_by_node must be a mapping")
        if any(
            not isinstance(value, ModalGeneratorLowering)
            for value in self.lowerings_by_node.values()
        ):
            raise TypeError("lowerings_by_node values are invalid")
        lowerings = {
            name: ModalGeneratorLowering.from_state_dict(value.state_dict())
            for name, value in self.lowerings_by_node.items()
        }
        fragments = dict(self.fragment_id_by_node)
        names = tuple(node.name for node in graph.nodes)
        if (
            len(names) != self.selection.fragment_count
            or set(lowerings) != set(names)
            or set(fragments) != set(names)
            or tuple(fragments[name] for name in names)
            != self.selection.fragment_ids
        ):
            raise ValueError("same-layer graph node catalogs disagree")
        base_order = self.selection.layer_ordinal * _INTRA_LAYER_CAUSAL_STRIDE
        if tuple(node.causal_order for node in graph.nodes) != tuple(
            base_order + position for position in range(len(graph.nodes))
        ):
            raise ValueError("same-layer node causal orders are not canonical")
        selected_by_id = {
            fragment.fragment_id: fragment
            for fragment in self.selection.execution_order
        }
        for node in graph.nodes:
            lowering = lowerings[node.name]
            selected = selected_by_id[fragments[node.name]]
            if (
                node.weights.artifact_sha256
                != lowering.graph_weights.artifact_sha256
                or lowering.mode_set_id != fragments[node.name]
                or lowering.fragment_plan.artifact_sha256
                != self.selection.source_fragment_plan_sha256
                or lowering.selected_fragment_sha256
                != selected.artifact_sha256
            ):
                raise ValueError("same-layer node does not match its lowering")
        object.__setattr__(self, "graph_plan", graph)
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
            self.lowerings_by_node[node.name] for node in self.graph_plan.nodes
        )

    @property
    def node_names(self) -> tuple[str, ...]:
        return self.graph_plan.traversal_order


def build_edgeless_same_layer_graph(
    selection: SameLayerFragmentSelection,
    *,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    lowerings_by_fragment: Mapping[str, ModalGeneratorLowering],
) -> EdgelessSameLayerGraphPlan:
    """Build one edgeless graph with explicit intra-layer causal order."""

    if not isinstance(selection, SameLayerFragmentSelection):
        raise TypeError("selection must be SameLayerFragmentSelection")
    selection.validate_against(fragment_plan)
    expected_ids = set(selection.fragment_ids)
    if not isinstance(lowerings_by_fragment, Mapping):
        raise TypeError("lowerings_by_fragment must be a mapping")
    if set(lowerings_by_fragment) != expected_ids:
        raise ValueError("lowerings must exactly cover selected fragments")
    if any(
        not isinstance(lowering, ModalGeneratorLowering)
        for lowering in lowerings_by_fragment.values()
    ):
        raise TypeError("lowerings must be ModalGeneratorLowering artifacts")
    if selection.fragment_count >= _INTRA_LAYER_CAUSAL_STRIDE:
        raise ValueError("same-layer selection exceeds causal-order stride")

    nodes = []
    lowerings_by_node: dict[str, ModalGeneratorLowering] = {}
    fragment_id_by_node: dict[str, str] = {}
    base_order = selection.layer_ordinal * _INTRA_LAYER_CAUSAL_STRIDE
    for position, fragment in enumerate(selection.execution_order):
        lowering = ModalGeneratorLowering.from_state_dict(
            lowerings_by_fragment[fragment.fragment_id].state_dict()
        )
        if (
            lowering.fragment_plan.artifact_sha256
            != fragment_plan.artifact_sha256
            or lowering.selected_fragment_sha256 != fragment.artifact_sha256
        ):
            raise ValueError("lowering does not bind selected same-layer fragment")
        name = (
            f"{lowering.generator_id}.same-layer-{position}.graph-node"
        )
        node = lowering.to_graph_node(
            name=name,
            causal_order=base_order + position,
        )
        nodes.append(node)
        lowerings_by_node[name] = lowering
        fragment_id_by_node[name] = fragment.fragment_id
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=selection.source_model_sha256,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=tuple(nodes),
        interactions=(),
    )
    return EdgelessSameLayerGraphPlan(
        selection=selection,
        graph_plan=graph,
        lowerings_by_node=lowerings_by_node,
        fragment_id_by_node=fragment_id_by_node,
    )


def collect_aligned_same_layer_fragment_rows(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    selection: SameLayerFragmentSelection,
    down_projection_weights: Mapping[str, Tensor],
) -> AlignedFragmentRows:
    """Collect native fragment rows while preserving one shared row axis."""

    if not isinstance(selection, SameLayerFragmentSelection):
        raise TypeError("selection must be SameLayerFragmentSelection")
    selection.validate_integrity()
    return collect_aligned_fragment_rows(
        rows,
        fragments=selection.execution_order,
        down_projection_weights=down_projection_weights,
        require_distinct_layers=False,
    )


def _row_key_sha256(row_keys: tuple[tuple[str, int], ...]) -> str:
    return _json_sha256(row_keys, domain=_ROW_KEY_DOMAIN)


@dataclass(frozen=True, slots=True)
class SameLayerShapeFlowRows:
    """Ephemeral realized shape states and native target-flow coordinates."""

    node_states: Mapping[str, Tensor]
    teacher_coordinates: Mapping[str, Tensor]
    row_keys: tuple[tuple[str, int], ...]
    row_key_sha256: str = ""

    def __post_init__(self) -> None:
        states = dict(self.node_states)
        teachers = dict(self.teacher_coordinates)
        if not states or set(states) != set(teachers):
            raise ValueError("shape and teacher node catalogs differ")
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
                raise ValueError("shape and teacher coordinate widths differ")
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
            raise ValueError("same-layer row-key hash mismatch")
        object.__setattr__(self, "node_states", MappingProxyType(states))
        object.__setattr__(
            self,
            "teacher_coordinates",
            MappingProxyType(teachers),
        )

    @property
    def observations(self) -> int:
        return len(self.row_keys)

    @property
    def teacher_flows(self) -> Mapping[str, Tensor]:
        """Return teacher minus generated state in each node's coordinates."""

        return MappingProxyType(
            {
                name: self.teacher_coordinates[name] - state
                for name, state in self.node_states.items()
            }
        )


def _batch_row_keys(
    adapter: Gemma3CausalLMAdapter,
    batch: CalibrationBatch,
) -> tuple[tuple[str, int], ...]:
    if batch.example_ids is None:
        raise ValueError("same-layer capture requires stable example ids")
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


def collect_edgeless_same_layer_shape_flow_rows(
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
    *,
    plan: EdgelessSameLayerGraphPlan,
    expected_row_keys: tuple[tuple[str, int], ...],
) -> SameLayerShapeFlowRows:
    """Capture physical same-layer modal states and native teacher flow."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be Gemma3CausalLMAdapter")
    if not isinstance(executor, Gemma3ModalGeneratorGraphExecutor):
        raise TypeError("executor must be a Gemma graph executor")
    if not isinstance(plan, EdgelessSameLayerGraphPlan):
        raise TypeError("plan must be EdgelessSameLayerGraphPlan")
    if (
        executor.graph_plan.artifact_sha256
        != plan.graph_plan.artifact_sha256
        or executor.graph_plan.interactions
    ):
        raise ValueError("executor is not the declared edgeless graph")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError("batches must be a sequence")
    if not batches:
        raise ValueError("same-layer capture batches cannot be empty")

    fragment_by_id = {
        fragment.fragment_id: fragment
        for fragment in plan.selection.execution_order
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
            captured_keys.extend(_batch_row_keys(adapter, batch))
            with torch.no_grad():
                execution = executor.run(
                    batch.model_inputs,
                    condition="generated",
                    capture_modal_states=True,
                )
            states = execution.graph_execution.modal_states
            if states is None or set(states) != set(plan.lowerings_by_node):
                raise RuntimeError("same-layer modal-state capture is incomplete")
            if set(current_teachers) != set(plan.lowerings_by_node):
                raise RuntimeError("same-layer native-teacher capture is incomplete")
            for name in plan.lowerings_by_node:
                state = states[name]
                teacher = current_teachers[name]
                mask = batch.valid_positions.to(device=state.device)
                if (
                    state.shape[:-1] != mask.shape
                    or teacher.shape[:-1] != mask.shape
                ):
                    raise ValueError("captured same-layer row shapes drifted")
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
            "same-layer runtime rows do not match expected row keys"
        )
    return SameLayerShapeFlowRows(
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
