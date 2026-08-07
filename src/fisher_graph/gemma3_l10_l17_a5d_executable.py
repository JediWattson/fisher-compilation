"""Freeze the A5d source-owned graph plus an optional residual overlay.

This module is the executable boundary between family-disjoint A5d selection
and outer-held scoring.  The selected residual is deliberately *additive*:
the authenticated Layer-17 graph and the authenticated Layer-10/Layer-17
composition remain the owning runtimes in every branch.  Alpha zero therefore
contains no residual graph at all, while positive alpha adds one zero-mean
Layer-17 graph after the feed-forward RMSNorm.

The freeze receipt is tensor-free.  Executable tensors remain in separately
authenticated graph and lowering objects, whose identities are committed by a
domain-separated SHA-256 digest before a held batch may be selected.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    _take_first_examples,
)
from .gemma3_l10_l17_a5d_family_residual_cv import (
    A5dFamilyResidualCvSelection,
    validate_a5d_family_residual_cv_receipt,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    FrozenBasisGeneratorFit,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
)
from .modal_generator_lowering import ModalGeneratorLowering


__all__ = [
    "A5dExecutableFreeze",
    "a5d_executable_freeze_sha256",
    "build_a5d_scoring_executors",
    "freeze_a5d_executable",
    "recompute_a5d_executable_freeze_sha256",
    "select_a5d_held_scoring_batch_after_freeze",
]


_FREEZE_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-freeze:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FROZEN = "frozen_source_fallback"
_ADDITIVE = "additive_zero_mean_residual"
_APPLICATION_LAYER_ORDINAL = 17
_APPLICATION_BOUNDARY = "layer.17.mlp.delta"
_APPLICATION_ORDER = (
    "post_feedforward_rmsnorm_then_scaled_additive_residual"
)
_LINEAGE_FIELDS = {
    "a5c_report_sha256",
    "capture_sha256",
    "target_solve_receipt_sha256",
    "coordinate_row_bank_receipt_sha256",
    "breadth_split_receipt_sha256",
    "source_anchored_residual_receipt_sha256",
    "residual_cv_receipt_sha256",
    "layer10_graph_sha256",
    "layer10_lowering_sha256_by_node",
    "matched_double_deletion_graph_sha256",
}
_RESOURCE_FIELDS = (
    "node_count",
    "interaction_count",
    "parameter_count",
    "macs_per_token",
    "additions_per_token",
    "peak_live_modal_width",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_value(value: object) -> object:
    """Return a plain JSON tree without trusting ``json`` to handle proxies."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"freeze metadata contains unsupported value {type(value)!r}")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(_freeze_json(child) for child in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("freeze metadata contains a non-finite float")
        return value
    if isinstance(value, Tensor):
        raise TypeError("freeze metadata must not contain tensors")
    raise TypeError(f"freeze metadata contains unsupported value {type(value)!r}")


def _float(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _authenticate_graph(
    value: ModalGeneratorGraphPlan, *, label: str
) -> ModalGeneratorGraphPlan:
    if not isinstance(value, ModalGeneratorGraphPlan):
        raise TypeError(f"{label} must be a ModalGeneratorGraphPlan")
    value.validate_integrity()
    restored = ModalGeneratorGraphPlan.from_state_dict(value.state_dict())
    if restored.artifact_sha256 != value.artifact_sha256:
        raise ValueError(f"{label} roundtrip identity drifted")
    return restored


def _authenticate_lowering(
    value: ModalGeneratorLowering, *, label: str
) -> ModalGeneratorLowering:
    if not isinstance(value, ModalGeneratorLowering):
        raise TypeError(f"{label} must be a ModalGeneratorLowering")
    restored = ModalGeneratorLowering.from_state_dict(value.state_dict())
    if restored.artifact_sha256 != value.artifact_sha256:
        raise ValueError(f"{label} roundtrip identity drifted")
    return restored


def _authenticate_lowering_mapping(
    graph: ModalGeneratorGraphPlan,
    values: Mapping[str, ModalGeneratorLowering],
    *,
    label: str,
) -> Mapping[str, ModalGeneratorLowering]:
    names = graph.traversal_order
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError(f"{label} catalog differs from its graph")
    nodes = {node.name: node for node in graph.nodes}
    result: dict[str, ModalGeneratorLowering] = {}
    for name in names:
        lowering = _authenticate_lowering(values[name], label=f"{label} {name}")
        if (
            lowering.graph_weights.artifact_sha256
            != nodes[name].weights.artifact_sha256
        ):
            raise ValueError(f"{label} {name} is not paired to its graph node")
        result[name] = lowering
    if len({value.artifact_sha256 for value in result.values()}) != len(result):
        raise ValueError(f"{label} reuses one lowering for multiple graph nodes")
    return MappingProxyType(result)


def _authenticate_lowering_sequence(
    graph: ModalGeneratorGraphPlan,
    values: Sequence[ModalGeneratorLowering],
    *,
    label: str,
) -> tuple[tuple[ModalGeneratorLowering, ...], Mapping[str, ModalGeneratorLowering]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    supplied = tuple(
        _authenticate_lowering(value, label=f"{label} entry") for value in values
    )
    if len(supplied) != len(graph.nodes):
        raise ValueError(f"{label} must cover every graph node exactly once")
    by_weights: dict[str, list[ModalGeneratorLowering]] = defaultdict(list)
    for lowering in supplied:
        by_weights[lowering.graph_weights.artifact_sha256].append(lowering)
    ordered: list[ModalGeneratorLowering] = []
    by_name: dict[str, ModalGeneratorLowering] = {}
    for node in graph.nodes:
        matches = by_weights.get(node.weights.artifact_sha256, ())
        if len(matches) != 1:
            raise ValueError(f"{label} is not one-to-one with graph nodes")
        ordered.append(matches[0])
        by_name[node.name] = matches[0]
    if len({value.artifact_sha256 for value in ordered}) != len(ordered):
        raise ValueError(f"{label} reuses one lowering for multiple graph nodes")
    return tuple(ordered), MappingProxyType(by_name)


def _bound_fragment_layer(lowering: ModalGeneratorLowering) -> int:
    matches = tuple(
        fragment
        for fragment in lowering.fragment_plan.fragments
        if fragment.artifact_sha256 == lowering.selected_fragment_sha256
    )
    if len(matches) != 1:
        raise ValueError("lowering does not select exactly one source fragment")
    return matches[0].layer_ordinal


def _require_layer(
    lowerings: Mapping[str, ModalGeneratorLowering],
    *,
    ordinal: int,
    label: str,
) -> None:
    if any(_bound_fragment_layer(value) != ordinal for value in lowerings.values()):
        raise ValueError(f"{label} must bind canonical Layer {ordinal}")


def _lowering_hashes(
    graph: ModalGeneratorGraphPlan,
    values: Mapping[str, ModalGeneratorLowering],
) -> dict[str, str]:
    return {
        name: values[name].artifact_sha256 for name in graph.traversal_order
    }


def _peak_live_modal_width(graph: ModalGeneratorGraphPlan) -> int:
    """Mirror the executor's exact static-edge latent liveness accounting."""

    positions = {node.name: index for index, node in enumerate(graph.nodes)}
    last_use = dict(positions)
    for edge in graph.interactions:
        if isinstance(edge, ModalGeneratorInteraction):
            last_use[edge.source_node] = max(
                last_use[edge.source_node], positions[edge.target_node]
            )
    live: dict[str, int] = {}
    peak = 0
    for index, node in enumerate(graph.nodes):
        live[node.name] = node.latent_width
        peak = max(peak, sum(live.values()))
        for name in tuple(live):
            if last_use[name] <= index:
                del live[name]
    return peak


def _resources(graph: ModalGeneratorGraphPlan) -> dict[str, int]:
    graph.validate_integrity()
    return {
        "node_count": len(graph.nodes),
        "interaction_count": len(graph.interactions),
        "parameter_count": graph.parameter_count,
        "macs_per_token": graph.macs_per_token,
        "additions_per_token": (
            graph.accounting.elementwise_additions_per_token
        ),
        "peak_live_modal_width": _peak_live_modal_width(graph),
    }


def _sum_resources(
    owner: Mapping[str, int], additive: Mapping[str, int] | None
) -> dict[str, int]:
    if additive is None:
        return {name: int(owner[name]) for name in _RESOURCE_FIELDS}
    return {
        name: int(owner[name]) + int(additive[name])
        for name in _RESOURCE_FIELDS
    }


def _validate_lineage(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _LINEAGE_FIELDS:
        raise ValueError("A5d executable lineage fields are invalid")
    catalog = value["layer10_lowering_sha256_by_node"]
    if not isinstance(catalog, Mapping) or not catalog:
        raise ValueError("A5d executable lineage lacks Layer10 lowerings")
    lowerings: dict[str, str] = {}
    for name, digest in catalog.items():
        if not isinstance(name, str) or not name:
            raise ValueError("A5d Layer10 lineage contains an invalid node name")
        lowerings[name] = _require_sha256(
            digest, label=f"A5d lineage Layer10 lowering {name}"
        )
    result = {
        name: _require_sha256(value[name], label=f"A5d lineage {name}")
        for name in _LINEAGE_FIELDS - {"layer10_lowering_sha256_by_node"}
    }
    result["layer10_lowering_sha256_by_node"] = lowerings
    return result


def _freeze_payload(
    *,
    kind: str,
    selected_alpha: float,
    selected_ridge: float | None,
    lineage: Mapping[str, object],
    source_owner: Mapping[str, object],
    additive_residual: Mapping[str, object] | None,
    selected_resources: Mapping[str, object],
) -> dict[str, object]:
    alpha = _float(selected_alpha, label="A5d selected alpha")
    ridge = (
        None
        if selected_ridge is None
        else _float(selected_ridge, label="A5d selected ridge")
    )
    if kind == _FROZEN:
        if alpha != 0.0 or ridge is not None or additive_residual is not None:
            raise ValueError("A5d alpha-zero freeze retained an additive residual")
    elif kind == _ADDITIVE:
        if alpha <= 0.0 or ridge is None or additive_residual is None:
            raise ValueError("A5d positive freeze lacks its additive residual")
    else:
        raise ValueError("A5d executable kind is invalid")
    return {
        "kind": kind,
        "selected_alpha": alpha,
        "selected_alpha_hex": alpha.hex(),
        "selected_ridge": ridge,
        "selected_ridge_hex": None if ridge is None else ridge.hex(),
        "application_boundary": _APPLICATION_BOUNDARY,
        "application_order": _APPLICATION_ORDER,
        "source_ownership_preserved": True,
        "source_affine_means_reinjected": False,
        "lineage": _json_value(lineage),
        "source_owner": _json_value(source_owner),
        "additive_residual": _json_value(additive_residual),
        "selected_resources": _json_value(selected_resources),
    }


def a5d_executable_freeze_sha256(
    *,
    kind: str,
    selected_alpha: float,
    selected_ridge: float | None,
    lineage: Mapping[str, object],
    source_owner: Mapping[str, object],
    additive_residual: Mapping[str, object] | None,
    selected_resources: Mapping[str, object],
) -> str:
    """Hash every frozen field that can alter A5d graph execution."""

    payload = _freeze_payload(
        kind=kind,
        selected_alpha=selected_alpha,
        selected_ridge=selected_ridge,
        lineage=lineage,
        source_owner=source_owner,
        additive_residual=additive_residual,
        selected_resources=selected_resources,
    )
    return hashlib.sha256(_FREEZE_DOMAIN + _canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class A5dExecutableFreeze:
    """Authenticated source owner and optional non-owning residual overlay."""

    kind: str
    selected_alpha: float
    selected_ridge: float | None
    source_layer17_graph: ModalGeneratorGraphPlan = field(repr=False)
    source_layer17_lowerings_by_node: Mapping[
        str, ModalGeneratorLowering
    ] = field(repr=False)
    source_composition_graph: ModalGeneratorGraphPlan = field(repr=False)
    source_composition_lowerings: tuple[ModalGeneratorLowering, ...] = field(
        repr=False
    )
    additive_residual_graph: ModalGeneratorGraphPlan | None = field(repr=False)
    additive_residual_lowerings_by_node: Mapping[
        str, ModalGeneratorLowering
    ] = field(repr=False)
    lineage: Mapping[str, object]
    source_owner: Mapping[str, object]
    additive_residual: Mapping[str, object] | None
    selected_resources: Mapping[str, object]
    selection_freeze_sha256: str

    @property
    def layer17_graph(self) -> ModalGeneratorGraphPlan:
        return self.source_layer17_graph

    @property
    def layer17_lowerings_by_node(self) -> Mapping[str, ModalGeneratorLowering]:
        return self.source_layer17_lowerings_by_node

    @property
    def composition_graph(self) -> ModalGeneratorGraphPlan:
        return self.source_composition_graph

    @property
    def composition_lowerings(self) -> tuple[ModalGeneratorLowering, ...]:
        return self.source_composition_lowerings

    @property
    def additive_graph(self) -> ModalGeneratorGraphPlan | None:
        return self.additive_residual_graph

    @property
    def additive_lowerings_by_node(
        self,
    ) -> Mapping[str, ModalGeneratorLowering]:
        return self.additive_residual_lowerings_by_node

    def _report_payload(self) -> dict[str, object]:
        return _freeze_payload(
            kind=self.kind,
            selected_alpha=self.selected_alpha,
            selected_ridge=self.selected_ridge,
            lineage=self.lineage,
            source_owner=self.source_owner,
            additive_residual=self.additive_residual,
            selected_resources=self.selected_resources,
        )

    def validate_integrity(self) -> None:
        """Reauthenticate tensors and the tensor-free freeze commitment."""

        layer17 = _authenticate_graph(
            self.source_layer17_graph, label="A5d frozen source Layer17 graph"
        )
        layer17_lowerings = _authenticate_lowering_mapping(
            layer17,
            self.source_layer17_lowerings_by_node,
            label="A5d frozen source Layer17 lowering",
        )
        composition = _authenticate_graph(
            self.source_composition_graph,
            label="A5d frozen source composition graph",
        )
        composition_lowerings, composition_by_node = (
            _authenticate_lowering_sequence(
                composition,
                self.source_composition_lowerings,
                label="A5d frozen source composition lowering",
            )
        )
        del composition_lowerings
        expected_owner = _source_owner_descriptor(
            layer17_graph=layer17,
            layer17_lowerings=layer17_lowerings,
            composition_graph=composition,
            composition_lowerings=composition_by_node,
        )
        if _json_value(self.source_owner) != expected_owner:
            raise ValueError("A5d frozen source owner descriptor drifted")

        additive_descriptor: dict[str, object] | None = None
        additive_resources: dict[str, int] | None = None
        if self.additive_residual_graph is not None:
            additive_graph = _authenticate_graph(
                self.additive_residual_graph,
                label="A5d frozen additive residual graph",
            )
            additive_lowerings = _authenticate_lowering_mapping(
                additive_graph,
                self.additive_residual_lowerings_by_node,
                label="A5d frozen additive residual lowering",
            )
            _validate_additive_against_source(
                source_graph=layer17,
                source_lowerings=layer17_lowerings,
                additive_graph=additive_graph,
                additive_lowerings=additive_lowerings,
            )
            additive_resources = _resources(additive_graph)
            additive_descriptor = _additive_descriptor(
                additive_graph, additive_lowerings
            )
        elif self.additive_residual_lowerings_by_node:
            raise ValueError("A5d fallback retained additive lowerings")
        if _json_value(self.additive_residual) != additive_descriptor:
            raise ValueError("A5d frozen additive descriptor drifted")

        owner = expected_owner
        expected_resources = {
            "layer17_scope": _sum_resources(
                owner["layer17_resources"], additive_resources  # type: ignore[arg-type]
            ),
            "composition_scope": _sum_resources(
                owner["composition_resources"],  # type: ignore[arg-type]
                additive_resources,
            ),
        }
        if _json_value(self.selected_resources) != expected_resources:
            raise ValueError("A5d frozen selected resources drifted")
        _validate_lineage(self.lineage)
        expected_sha = recompute_a5d_executable_freeze_sha256(self)
        if expected_sha != self.selection_freeze_sha256:
            raise ValueError("A5d executable freeze hash drifted")

    def report_section(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._report_payload(),
            "selection_freeze_sha256": self.selection_freeze_sha256,
        }


def _source_owner_descriptor(
    *,
    layer17_graph: ModalGeneratorGraphPlan,
    layer17_lowerings: Mapping[str, ModalGeneratorLowering],
    composition_graph: ModalGeneratorGraphPlan,
    composition_lowerings: Mapping[str, ModalGeneratorLowering],
) -> dict[str, object]:
    return {
        "layer17_graph_sha256": layer17_graph.artifact_sha256,
        "layer17_lowering_sha256_by_node": _lowering_hashes(
            layer17_graph, layer17_lowerings
        ),
        "composition_graph_sha256": composition_graph.artifact_sha256,
        "layer17_resources": _resources(layer17_graph),
        "composition_resources": _resources(composition_graph),
    }


def _additive_descriptor(
    graph: ModalGeneratorGraphPlan,
    lowerings: Mapping[str, ModalGeneratorLowering],
) -> dict[str, object]:
    return {
        "graph_sha256": graph.artifact_sha256,
        "lowering_sha256_by_node": _lowering_hashes(graph, lowerings),
        "application_layer_ordinal": _APPLICATION_LAYER_ORDINAL,
        "basis_means_exactly_zero": True,
        "source_decoders_reused": True,
        "source_affine_means_reinjected": False,
        "resources": _resources(graph),
    }


def _validate_composition_ownership(
    *,
    layer17_graph: ModalGeneratorGraphPlan,
    layer17_lowerings: Mapping[str, ModalGeneratorLowering],
    composition_graph: ModalGeneratorGraphPlan,
    composition_lowerings: Mapping[str, ModalGeneratorLowering],
    lineage: Mapping[str, object],
) -> None:
    layer10_hashes = lineage["layer10_lowering_sha256_by_node"]
    assert isinstance(layer10_hashes, Mapping)
    expected_names = set(layer17_graph.traversal_order) | set(layer10_hashes)
    if set(composition_graph.traversal_order) != expected_names:
        raise ValueError(
            "A5d source composition is not exactly Layer10 plus Layer17"
        )
    composition_nodes = {node.name: node for node in composition_graph.nodes}
    source_nodes = {node.name: node for node in layer17_graph.nodes}
    for name in layer17_graph.traversal_order:
        if (
            composition_nodes[name].artifact_sha256
            != source_nodes[name].artifact_sha256
            or composition_lowerings[name].artifact_sha256
            != layer17_lowerings[name].artifact_sha256
        ):
            raise ValueError("A5d composition replaced a source Layer17 node")
    for name, digest in layer10_hashes.items():
        if composition_lowerings[name].artifact_sha256 != digest:
            raise ValueError("A5d composition Layer10 lowering lineage drifted")
    layers = {
        _bound_fragment_layer(lowering)
        for lowering in composition_lowerings.values()
    }
    if layers != {10, 17}:
        raise ValueError("A5d source composition must bind only Layers 10 and 17")


def _validate_additive_against_source(
    *,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings: Mapping[str, ModalGeneratorLowering],
    additive_graph: ModalGeneratorGraphPlan,
    additive_lowerings: Mapping[str, ModalGeneratorLowering],
) -> None:
    if additive_graph.interactions:
        raise ValueError("A5d additive residual graph must be edgeless")
    if (
        additive_graph.model_fingerprint != source_graph.model_fingerprint
        or additive_graph.parameter_cluster_plan_sha256
        != source_graph.parameter_cluster_plan_sha256
        or additive_graph.traversal_order != source_graph.traversal_order
    ):
        raise ValueError("A5d additive residual graph source identity drifted")
    _require_layer(
        additive_lowerings,
        ordinal=_APPLICATION_LAYER_ORDINAL,
        label="A5d additive residual",
    )
    source_nodes = {node.name: node for node in source_graph.nodes}
    for node in additive_graph.nodes:
        source_node = source_nodes[node.name]
        source = source_lowerings[node.name]
        additive = additive_lowerings[node.name]
        source_basis = source.computational_mode_basis
        additive_basis = additive.computational_mode_basis
        if (
            node.causal_order != source_node.causal_order
            or node.input_boundary != source_node.input_boundary
            or node.input_width != source_node.input_width
            or node.output_width != source_node.output_width
            or node.output_boundary != _APPLICATION_BOUNDARY
            or additive.selected_fragment_sha256
            != source.selected_fragment_sha256
            or bool(torch.count_nonzero(additive_basis.mean_bias))
            or additive_basis.decoder_basis_sha256
            != source_basis.decoder_basis_sha256
            or not torch.equal(
                additive_basis.decoder_basis, source_basis.decoder_basis
            )
        ):
            raise ValueError(
                "A5d additive residual changed zero-mean source-decoder semantics"
            )


def freeze_a5d_executable(
    *,
    selection: A5dFamilyResidualCvSelection,
    source_layer17_graph: ModalGeneratorGraphPlan,
    source_layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    source_composition_graph: ModalGeneratorGraphPlan,
    source_composition_lowerings: Sequence[ModalGeneratorLowering],
    lineage: Mapping[str, object],
) -> A5dExecutableFreeze:
    """Freeze one CV decision without replacing either source-owning graph."""

    if not isinstance(selection, A5dFamilyResidualCvSelection):
        raise TypeError("selection must be A5dFamilyResidualCvSelection")
    receipt = validate_a5d_family_residual_cv_receipt(selection.receipt())
    source_receipt = receipt["source"]
    cv_choice = receipt["selection"]
    final_refit = receipt["final_refit"]
    configuration = receipt["configuration"]
    assert all(
        isinstance(value, Mapping)
        for value in (source_receipt, cv_choice, final_refit, configuration)
    )

    frozen_lineage = _validate_lineage(lineage)
    if (
        frozen_lineage["residual_cv_receipt_sha256"]
        != receipt["receipt_sha256"]
        or frozen_lineage["source_anchored_residual_receipt_sha256"]
        != source_receipt["residual_target_receipt_sha256"]
        or frozen_lineage["coordinate_row_bank_receipt_sha256"]
        != source_receipt["bridge_receipt_sha256"]
    ):
        raise ValueError("A5d executable lineage does not bind its CV sources")

    layer17_graph = _authenticate_graph(
        source_layer17_graph, label="A5d source Layer17 graph"
    )
    if layer17_graph.interactions:
        raise ValueError("A5d source Layer17 graph must be edgeless")
    layer17_lowerings = _authenticate_lowering_mapping(
        layer17_graph,
        source_layer17_lowerings_by_node,
        label="A5d source Layer17 lowering",
    )
    _require_layer(
        layer17_lowerings,
        ordinal=_APPLICATION_LAYER_ORDINAL,
        label="A5d source Layer17",
    )
    source_lowering_hashes = _lowering_hashes(
        layer17_graph, layer17_lowerings
    )
    source_mean_hashes = {
        name: layer17_lowerings[
            name
        ].computational_mode_basis.mean_bias_sha256
        for name in layer17_graph.traversal_order
    }
    source_decoder_hashes = {
        name: layer17_lowerings[
            name
        ].computational_mode_basis.decoder_basis_sha256
        for name in layer17_graph.traversal_order
    }
    if (
        tuple(selection.node_order) != layer17_graph.traversal_order
        or tuple(source_receipt["source_lowering_sha256_by_node"])
        != layer17_graph.traversal_order
        or source_receipt["source_graph_sha256"]
        != layer17_graph.artifact_sha256
        or source_receipt["source_model_sha256"]
        != layer17_graph.model_fingerprint
        or source_receipt["source_graph_parameter_count"]
        != layer17_graph.parameter_count
        or source_receipt["source_graph_macs_per_token"]
        != layer17_graph.macs_per_token
        or dict(source_receipt["source_lowering_sha256_by_node"])
        != source_lowering_hashes
        or dict(source_receipt["source_mean_sha256_by_node"])
        != source_mean_hashes
        or dict(source_receipt["source_decoder_sha256_by_node"])
        != source_decoder_hashes
    ):
        raise ValueError("A5d source Layer17 descriptors differ from CV")

    composition_graph = _authenticate_graph(
        source_composition_graph, label="A5d source composition graph"
    )
    composition_lowerings, composition_by_node = (
        _authenticate_lowering_sequence(
            composition_graph,
            source_composition_lowerings,
            label="A5d source composition lowering",
        )
    )
    _validate_composition_ownership(
        layer17_graph=layer17_graph,
        layer17_lowerings=layer17_lowerings,
        composition_graph=composition_graph,
        composition_lowerings=composition_by_node,
        lineage=frozen_lineage,
    )
    if (
        frozen_lineage["matched_double_deletion_graph_sha256"]
        != composition_graph.artifact_sha256
    ):
        raise ValueError(
            "A5d matched-double-deletion graph differs from source composition"
        )

    if (
        selection.selected_alpha != cv_choice["selected_alpha"]
        or selection.selected_ridge != cv_choice["selected_ridge"]
        or selection.use_frozen_fallback
        is not bool(cv_choice["use_frozen_fallback"])
        or configuration["output_boundary"] != _APPLICATION_BOUNDARY
    ):
        raise ValueError("A5d in-memory selection contradicts its CV receipt")

    additive_graph: ModalGeneratorGraphPlan | None = None
    additive_lowerings: Mapping[str, ModalGeneratorLowering] = MappingProxyType({})
    additive_descriptor: dict[str, object] | None = None
    additive_resources: dict[str, int] | None = None
    if selection.use_frozen_fallback:
        if (
            selection.selected_alpha != 0.0
            or selection.selected_ridge is not None
            or selection.residual_fit is not None
            or final_refit["fit"] is not None
        ):
            raise ValueError("A5d fallback retained residual execution state")
        kind = _FROZEN
    else:
        fit = selection.residual_fit
        if not isinstance(fit, FrozenBasisGeneratorFit):
            raise TypeError("A5d positive selection lacks a frozen residual fit")
        if selection.selected_alpha <= 0.0 or selection.selected_ridge is None:
            raise ValueError("A5d positive selection lacks alpha/ridge")
        final_fit = final_refit["fit"]
        if not isinstance(final_fit, Mapping):
            raise ValueError("A5d positive selection lacks final-refit evidence")
        additive_graph = _authenticate_graph(
            fit.graph_plan, label="A5d additive residual graph"
        )
        additive_lowerings = _authenticate_lowering_mapping(
            additive_graph,
            fit.lowerings_by_node,
            label="A5d additive residual lowering",
        )
        _validate_additive_against_source(
            source_graph=layer17_graph,
            source_lowerings=layer17_lowerings,
            additive_graph=additive_graph,
            additive_lowerings=additive_lowerings,
        )
        additive_hashes = _lowering_hashes(
            additive_graph, additive_lowerings
        )
        additive_zero_hashes = {
            name: additive_lowerings[
                name
            ].computational_mode_basis.mean_bias_sha256
            for name in additive_graph.traversal_order
        }
        if (
            final_fit["graph_sha256"] != additive_graph.artifact_sha256
            or dict(final_fit["lowering_sha256_by_node"])
            != additive_hashes
            or dict(final_fit["zero_mean_sha256_by_node"])
            != additive_zero_hashes
            or dict(final_fit["decoder_sha256_by_node"])
            != source_decoder_hashes
            or final_fit["parameter_count"] != additive_graph.parameter_count
            or final_fit["macs_per_token"] != additive_graph.macs_per_token
        ):
            raise ValueError("A5d additive runtime differs from final refit")
        additive_descriptor = _additive_descriptor(
            additive_graph, additive_lowerings
        )
        additive_resources = _resources(additive_graph)
        kind = _ADDITIVE

    source_owner = _source_owner_descriptor(
        layer17_graph=layer17_graph,
        layer17_lowerings=layer17_lowerings,
        composition_graph=composition_graph,
        composition_lowerings=composition_by_node,
    )
    selected_resources = {
        "layer17_scope": _sum_resources(
            source_owner["layer17_resources"],  # type: ignore[arg-type]
            additive_resources,
        ),
        "composition_scope": _sum_resources(
            source_owner["composition_resources"],  # type: ignore[arg-type]
            additive_resources,
        ),
    }
    freeze_sha = a5d_executable_freeze_sha256(
        kind=kind,
        selected_alpha=selection.selected_alpha,
        selected_ridge=selection.selected_ridge,
        lineage=frozen_lineage,
        source_owner=source_owner,
        additive_residual=additive_descriptor,
        selected_resources=selected_resources,
    )
    executable = A5dExecutableFreeze(
        kind=kind,
        selected_alpha=selection.selected_alpha,
        selected_ridge=selection.selected_ridge,
        source_layer17_graph=layer17_graph,
        source_layer17_lowerings_by_node=layer17_lowerings,
        source_composition_graph=composition_graph,
        source_composition_lowerings=composition_lowerings,
        additive_residual_graph=additive_graph,
        additive_residual_lowerings_by_node=additive_lowerings,
        lineage=_freeze_json(frozen_lineage),  # type: ignore[arg-type]
        source_owner=_freeze_json(source_owner),  # type: ignore[arg-type]
        additive_residual=(
            None
            if additive_descriptor is None
            else _freeze_json(additive_descriptor)  # type: ignore[arg-type]
        ),
        selected_resources=_freeze_json(selected_resources),  # type: ignore[arg-type]
        selection_freeze_sha256=freeze_sha,
    )
    executable.validate_integrity()
    return executable


def recompute_a5d_executable_freeze_sha256(
    executable: A5dExecutableFreeze,
) -> str:
    if not isinstance(executable, A5dExecutableFreeze):
        raise TypeError("executable must be A5dExecutableFreeze")
    return a5d_executable_freeze_sha256(
        kind=executable.kind,
        selected_alpha=executable.selected_alpha,
        selected_ridge=executable.selected_ridge,
        lineage=executable.lineage,
        source_owner=executable.source_owner,
        additive_residual=executable.additive_residual,
        selected_resources=executable.selected_resources,
    )


def select_a5d_held_scoring_batch_after_freeze(
    *,
    blocks: Mapping[str, object],
    held_family_alias: str,
    executable: A5dExecutableFreeze,
) -> tuple[object, ...]:
    """Select one held example only after reauthenticating the full freeze."""

    # Keep this before *all* mapping operations.  Even membership tests may
    # invoke user-defined Mapping code and therefore count as held access.
    expected = recompute_a5d_executable_freeze_sha256(executable)
    if expected != executable.selection_freeze_sha256:
        raise RuntimeError("A5d executable is not frozen before held access")
    executable.validate_integrity()
    if held_family_alias not in blocks:
        raise KeyError("outer-held family is unavailable")
    return tuple(_take_first_examples(blocks[held_family_alias], 1))


def _ordered_lowerings(
    graph: ModalGeneratorGraphPlan,
    by_node: Mapping[str, ModalGeneratorLowering],
) -> tuple[ModalGeneratorLowering, ...]:
    return tuple(by_node[name] for name in graph.traversal_order)


def build_a5d_scoring_executors(
    adapter: Gemma3CausalLMAdapter,
    layer10_graph: ModalGeneratorGraphPlan,
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    executable: A5dExecutableFreeze,
) -> tuple[
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
    Gemma3ModalGeneratorGraphExecutor,
]:
    """Build four independent source-owning A5d scoring executors.

    The return order is Layer10, selected Layer17, frozen composition, selected
    composition.  No primary graph uses the post-delta replacement path.
    Alpha-zero construction omits additive keyword arguments altogether.
    """

    if not isinstance(executable, A5dExecutableFreeze):
        raise TypeError("executable must be A5dExecutableFreeze")
    executable.validate_integrity()
    layer10_graph.validate_integrity()
    if (
        layer10_graph.artifact_sha256
        != executable.lineage["layer10_graph_sha256"]
        or set(layer10_lowerings_by_node) != set(layer10_graph.traversal_order)
        or {
            name: layer10_lowerings_by_node[name].artifact_sha256
            for name in layer10_graph.traversal_order
        }
        != dict(executable.lineage["layer10_lowering_sha256_by_node"])
    ):
        raise ValueError("A5d Layer10 scoring runtime differs from lineage")

    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        _ordered_lowerings(layer10_graph, layer10_lowerings_by_node),
    )
    source_layer17_lowerings = _ordered_lowerings(
        executable.source_layer17_graph,
        executable.source_layer17_lowerings_by_node,
    )
    if executable.additive_residual_graph is None:
        selected_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            executable.source_layer17_graph,
            source_layer17_lowerings,
        )
    else:
        additive_lowerings = _ordered_lowerings(
            executable.additive_residual_graph,
            executable.additive_residual_lowerings_by_node,
        )
        selected_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            executable.source_layer17_graph,
            source_layer17_lowerings,
            additive_post_feedforward_graph_plan=(
                executable.additive_residual_graph
            ),
            additive_post_feedforward_lowerings=additive_lowerings,
            additive_post_feedforward_scale=executable.selected_alpha,
        )
    frozen_composition_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        executable.source_composition_graph,
        executable.source_composition_lowerings,
    )
    if executable.additive_residual_graph is None:
        selected_composition_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            executable.source_composition_graph,
            executable.source_composition_lowerings,
        )
    else:
        selected_composition_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            executable.source_composition_graph,
            executable.source_composition_lowerings,
            additive_post_feedforward_graph_plan=(
                executable.additive_residual_graph
            ),
            additive_post_feedforward_lowerings=_ordered_lowerings(
                executable.additive_residual_graph,
                executable.additive_residual_lowerings_by_node,
            ),
            additive_post_feedforward_scale=executable.selected_alpha,
        )
    executors = (
        layer10_executor,
        selected_layer17_executor,
        frozen_composition_executor,
        selected_composition_executor,
    )
    if len({id(value) for value in executors}) != 4:
        raise RuntimeError("A5d scoring executors are not distinct")
    if any(
        value.post_feedforward_delta_layer_ordinals != ()
        for value in executors
    ):
        raise RuntimeError("A5d used the forbidden primary post-delta path")
    return executors


# Private aliases ease the eventual runner migration without preserving any of
# A5c's replacement-graph semantics.
_FrozenA5dExecutable = A5dExecutableFreeze
_freeze_selected_executable = freeze_a5d_executable
_select_held_scoring_batch_after_freeze = (
    select_a5d_held_scoring_batch_after_freeze
)
_build_scoring_executors = build_a5d_scoring_executors
