"""Analysis-only hierarchy nominations for the strict Gemma causal map.

This module is deliberately a bridge, not a compiler.  It turns the
already-strict-loaded 18-generator causal-map artifact into:

* a complete partition of generator layers into causally contiguous parent
  candidates;
* a separate catalog of potentially noncontiguous parameter-sharing
  hypotheses; and
* an exhaustive classification of each finite-intervention directed response
  as either internal to one nominated parent or crossing a parent boundary.

The source measurements are finite suppression responses.  They are not
Jacobians, activation Fisher rows, execution plans, or mutation authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math
import re

from fisher_graph.gemma3_generator_causal_map_artifact import (
    GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION,
    GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA,
    validate_gemma3_generator_causal_map_payload,
)


__all__ = [
    "FINITE_INTERVENTION_RESPONSE_EVIDENCE",
    "CausalGroupNomination",
    "DirectedEdgeNomination",
    "GeneratorHierarchyNomination",
    "HierarchyParentNomination",
    "SharingFamilyNomination",
    "known_v1_gemma3_generator_hierarchy_specs",
    "nominate_gemma3_generator_hierarchy",
    "nominate_known_v1_gemma3_generator_hierarchy",
]


_LAYER_COUNT = 18
_PAIR_COUNT = _LAYER_COUNT * (_LAYER_COUNT - 1) // 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_DIGEST_DOMAIN = b"fisher_graph.gemma3.generator_hierarchy_nomination.v1\0"
_SPEC_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_hierarchy_nomination.spec.v1\0"
)
_PARENT_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_hierarchy_nomination.parent.v1\0"
)
_FAMILY_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_hierarchy_nomination.family.v1\0"
)
_EDGE_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_hierarchy_nomination.edge.v1\0"
)
_SOURCE_EDGE_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_hierarchy_nomination.source_edge.v1\0"
)

FINITE_INTERVENTION_RESPONSE_EVIDENCE = (
    "finite_upstream_suppression_response_not_jacobian"
)

_AUTHORITY_FIELDS = (
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object, *, domain: bytes) -> str:
    result = hashlib.sha256()
    result.update(domain)
    result.update(_canonical_json_bytes(value))
    return result.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_layer_tuple(
    value: object,
    *,
    label: str,
    minimum_size: int,
    contiguous: bool,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    layers = tuple(value)
    if len(layers) < minimum_size:
        raise ValueError(
            f"{label} must contain at least {minimum_size} layers"
        )
    if any(type(layer) is not int for layer in layers):
        raise TypeError(f"{label} must contain integer layer ordinals")
    if any(layer < 0 or layer >= _LAYER_COUNT for layer in layers):
        raise ValueError(f"{label} layer ordinals must be between 0 and 17")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError(f"{label} must be strictly increasing and unique")
    if contiguous and any(
        right != left + 1 for left, right in zip(layers, layers[1:])
    ):
        raise ValueError(f"{label} must be causally contiguous")
    return layers


def _finite(value: object, *, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return float(value)


def _no_authority(value: Mapping[str, object], *, label: str) -> None:
    for field_name in _AUTHORITY_FIELDS:
        if value.get(field_name) is not False:
            raise ValueError(
                f"{label} carries forbidden optimization authority in "
                f"{field_name}"
            )


@dataclass(frozen=True, slots=True)
class CausalGroupNomination:
    """Caller-declared contiguous child interval nominated for coarsening."""

    parent_id: str
    child_layer_ordinals: tuple[int, ...]
    nomination_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        parent_id = _require_name(self.parent_id, label="parent_id")
        layers = _require_layer_tuple(
            self.child_layer_ordinals,
            label=f"causal group {parent_id}",
            minimum_size=2,
            contiguous=True,
        )
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "child_layer_ordinals", layers)
        object.__setattr__(
            self,
            "nomination_sha256",
            _digest(
                {
                    "parent_id": parent_id,
                    "child_layer_ordinals": layers,
                    "candidate_only": True,
                    "authorizes_execution": False,
                },
                domain=_SPEC_DIGEST_DOMAIN,
            ),
        )


@dataclass(frozen=True, slots=True)
class SharingFamilyNomination:
    """Noncausal shared-template hypothesis; it never contracts graph nodes."""

    family_id: str
    child_layer_ordinals: tuple[int, ...]
    generator_ids: tuple[str, ...] = ()
    observational_only: bool = True
    authorizes_execution: bool = False
    nomination_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        family_id = _require_name(self.family_id, label="family_id")
        layers = _require_layer_tuple(
            self.child_layer_ordinals,
            label=f"sharing family {family_id}",
            minimum_size=2,
            contiguous=False,
        )
        generators = tuple(self.generator_ids)
        if generators and len(generators) != len(layers):
            raise ValueError(
                "sharing family generator_ids must align with its layers"
            )
        for generator_id in generators:
            _require_name(generator_id, label="sharing generator_id")
        if (
            self.observational_only is not True
            or self.authorizes_execution is not False
        ):
            raise ValueError(
                "sharing families are observational and cannot authorize "
                "execution"
            )
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "child_layer_ordinals", layers)
        object.__setattr__(self, "generator_ids", generators)
        object.__setattr__(
            self,
            "nomination_sha256",
            _digest(
                {
                    "family_id": family_id,
                    "child_layer_ordinals": layers,
                    "generator_ids": generators,
                    "observational_only": True,
                    "authorizes_execution": False,
                },
                domain=_FAMILY_DIGEST_DOMAIN,
            ),
        )


@dataclass(frozen=True, slots=True)
class HierarchyParentNomination:
    """One complete-partition parent candidate at the nominated level."""

    parent_id: str
    child_layer_ordinals: tuple[int, ...]
    generator_ids: tuple[str, ...]
    kind: str
    observational_only: bool = True
    authorizes_execution: bool = False
    nomination_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        parent_id = _require_name(self.parent_id, label="parent_id")
        layers = _require_layer_tuple(
            self.child_layer_ordinals,
            label=f"hierarchy parent {parent_id}",
            minimum_size=1,
            contiguous=True,
        )
        generators = tuple(self.generator_ids)
        if len(generators) != len(layers):
            raise ValueError("parent generator_ids must align with its layers")
        for generator_id in generators:
            _require_name(generator_id, label="parent generator_id")
        expected_kind = (
            "singleton_passthrough"
            if len(layers) == 1
            else "causal_contraction_candidate"
        )
        if self.kind != expected_kind:
            raise ValueError("hierarchy parent kind is inconsistent")
        if (
            self.observational_only is not True
            or self.authorizes_execution is not False
        ):
            raise ValueError(
                "hierarchy parents are observational and cannot authorize "
                "execution"
            )
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "child_layer_ordinals", layers)
        object.__setattr__(self, "generator_ids", generators)
        object.__setattr__(
            self,
            "nomination_sha256",
            _digest(
                {
                    "parent_id": parent_id,
                    "child_layer_ordinals": layers,
                    "generator_ids": generators,
                    "kind": expected_kind,
                    "observational_only": True,
                    "authorizes_execution": False,
                },
                domain=_PARENT_DIGEST_DOMAIN,
            ),
        )


@dataclass(frozen=True, slots=True)
class DirectedEdgeNomination:
    """One exhaustive source-edge classification at the parent boundary."""

    edge_ordinal: int
    upstream_layer_ordinal: int
    downstream_layer_ordinal: int
    upstream_parent_id: str
    downstream_parent_id: str
    disposition: str
    source_edge_sha256: str
    mean_directed_response_rms: float
    maximum_directed_response_rms: float
    mean_directed_response_ratio_over_defined: float
    evidence_semantics: str = FINITE_INTERVENTION_RESPONSE_EVIDENCE
    jacobian_estimate: bool = False
    observational_only: bool = True
    authorizes_execution: bool = False
    nomination_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.edge_ordinal) is not int or not (
            0 <= self.edge_ordinal < _PAIR_COUNT
        ):
            raise ValueError("edge_ordinal is invalid")
        upstream = self.upstream_layer_ordinal
        downstream = self.downstream_layer_ordinal
        if (
            type(upstream) is not int
            or type(downstream) is not int
            or not (0 <= upstream < downstream < _LAYER_COUNT)
        ):
            raise ValueError("directed edge layer ordinals are invalid")
        upstream_parent = _require_name(
            self.upstream_parent_id,
            label="upstream_parent_id",
        )
        downstream_parent = _require_name(
            self.downstream_parent_id,
            label="downstream_parent_id",
        )
        expected_disposition = (
            "internal"
            if upstream_parent == downstream_parent
            else "surfaced_cut"
        )
        if self.disposition != expected_disposition:
            raise ValueError("directed edge disposition is inconsistent")
        source_edge_sha256 = _require_sha256(
            self.source_edge_sha256,
            label="source_edge_sha256",
        )
        response_rms = _finite(
            self.mean_directed_response_rms,
            label="mean_directed_response_rms",
        )
        maximum_response_rms = _finite(
            self.maximum_directed_response_rms,
            label="maximum_directed_response_rms",
        )
        response_ratio = _finite(
            self.mean_directed_response_ratio_over_defined,
            label="mean_directed_response_ratio_over_defined",
        )
        if maximum_response_rms < response_rms:
            raise ValueError(
                "maximum directed response RMS cannot be below its mean"
            )
        if (
            self.evidence_semantics
            != FINITE_INTERVENTION_RESPONSE_EVIDENCE
            or self.jacobian_estimate is not False
        ):
            raise ValueError(
                "finite intervention responses cannot be labeled as "
                "Jacobians"
            )
        if (
            self.observational_only is not True
            or self.authorizes_execution is not False
        ):
            raise ValueError(
                "directed response nominations are observational and cannot "
                "authorize execution"
            )
        object.__setattr__(self, "upstream_parent_id", upstream_parent)
        object.__setattr__(self, "downstream_parent_id", downstream_parent)
        object.__setattr__(self, "source_edge_sha256", source_edge_sha256)
        object.__setattr__(self, "mean_directed_response_rms", response_rms)
        object.__setattr__(
            self,
            "maximum_directed_response_rms",
            maximum_response_rms,
        )
        object.__setattr__(
            self,
            "mean_directed_response_ratio_over_defined",
            response_ratio,
        )
        object.__setattr__(
            self,
            "nomination_sha256",
            _digest(
                {
                    "edge_ordinal": self.edge_ordinal,
                    "upstream_layer_ordinal": upstream,
                    "downstream_layer_ordinal": downstream,
                    "upstream_parent_id": upstream_parent,
                    "downstream_parent_id": downstream_parent,
                    "disposition": expected_disposition,
                    "source_edge_sha256": source_edge_sha256,
                    "mean_directed_response_rms": response_rms,
                    "maximum_directed_response_rms": maximum_response_rms,
                    "mean_directed_response_ratio_over_defined": (
                        response_ratio
                    ),
                    "evidence_semantics": (
                        FINITE_INTERVENTION_RESPONSE_EVIDENCE
                    ),
                    "jacobian_estimate": False,
                    "observational_only": True,
                    "authorizes_execution": False,
                },
                domain=_EDGE_DIGEST_DOMAIN,
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneratorHierarchyNomination:
    """Authenticated, immutable analysis-only hierarchy nomination."""

    source_scientific_payload_sha256: str
    parents: tuple[HierarchyParentNomination, ...]
    sharing_families: tuple[SharingFamilyNomination, ...]
    directed_edges: tuple[DirectedEdgeNomination, ...]
    internal_edge_count: int
    surfaced_cut_edge_count: int
    evidence_semantics: str = FINITE_INTERVENTION_RESPONSE_EVIDENCE
    observational_only: bool = True
    authorizes_merge: bool = False
    authorizes_pruning: bool = False
    authorizes_routing: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_mutation: bool = False
    nomination_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        source_digest = _require_sha256(
            self.source_scientific_payload_sha256,
            label="source scientific_payload_sha256",
        )
        parents = tuple(self.parents)
        families = tuple(self.sharing_families)
        edges = tuple(self.directed_edges)
        if not parents:
            raise ValueError("hierarchy nomination must contain parents")
        flattened = tuple(
            layer
            for parent in parents
            for layer in parent.child_layer_ordinals
        )
        if tuple(sorted(flattened)) != tuple(range(_LAYER_COUNT)):
            raise ValueError(
                "hierarchy parents must partition all 18 child layers exactly "
                "once"
            )
        parent_ids = tuple(parent.parent_id for parent in parents)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("hierarchy parent ids must be unique")
        if len(edges) != _PAIR_COUNT:
            raise ValueError("all 153 directed edges must be classified")
        expected_pairs = tuple(combinations(range(_LAYER_COUNT), 2))
        actual_pairs = tuple(
            (
                edge.upstream_layer_ordinal,
                edge.downstream_layer_ordinal,
            )
            for edge in edges
        )
        if actual_pairs != expected_pairs:
            raise ValueError(
                "directed edges must classify each causal pair exactly once"
            )
        internal_count = sum(
            edge.disposition == "internal" for edge in edges
        )
        cut_count = sum(
            edge.disposition == "surfaced_cut" for edge in edges
        )
        if (
            self.internal_edge_count != internal_count
            or self.surfaced_cut_edge_count != cut_count
            or internal_count + cut_count != _PAIR_COUNT
        ):
            raise ValueError("directed edge classification counts are invalid")
        family_ids = tuple(family.family_id for family in families)
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("sharing family ids must be unique")
        if self.evidence_semantics != FINITE_INTERVENTION_RESPONSE_EVIDENCE:
            raise ValueError(
                "finite intervention responses cannot be labeled as "
                "Jacobians"
            )
        if (
            self.observational_only is not True
            or self.authorizes_merge is not False
            or self.authorizes_pruning is not False
            or self.authorizes_routing is not False
            or self.authorizes_compilation is not False
            or self.authorizes_execution is not False
            or self.authorizes_mutation is not False
        ):
            raise ValueError(
                "hierarchy nomination is observational and grants no "
                "optimization or execution authority"
            )
        object.__setattr__(
            self,
            "source_scientific_payload_sha256",
            source_digest,
        )
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "sharing_families", families)
        object.__setattr__(self, "directed_edges", edges)
        object.__setattr__(
            self,
            "nomination_sha256",
            _digest(
                {
                    "source_scientific_payload_sha256": source_digest,
                    "parent_sha256s": tuple(
                        parent.nomination_sha256 for parent in parents
                    ),
                    "sharing_family_sha256s": tuple(
                        family.nomination_sha256 for family in families
                    ),
                    "directed_edge_sha256s": tuple(
                        edge.nomination_sha256 for edge in edges
                    ),
                    "internal_edge_count": internal_count,
                    "surfaced_cut_edge_count": cut_count,
                    "evidence_semantics": (
                        FINITE_INTERVENTION_RESPONSE_EVIDENCE
                    ),
                    "observational_only": True,
                    "authorizes_merge": False,
                    "authorizes_pruning": False,
                    "authorizes_routing": False,
                    "authorizes_compilation": False,
                    "authorizes_execution": False,
                    "authorizes_mutation": False,
                },
                domain=_DIGEST_DOMAIN,
            ),
        )


def _strict_source_catalog(
    causal_map: Mapping[str, object],
) -> tuple[
    str,
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    if not isinstance(causal_map, Mapping):
        raise TypeError("causal_map must be an already strict-loaded mapping")
    validate_gemma3_generator_causal_map_payload(causal_map)
    if (
        causal_map.get("schema") != GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA
        or causal_map.get("format_version")
        != GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION
    ):
        raise ValueError("causal_map schema or format version is invalid")
    source_digest = _require_sha256(
        causal_map.get("scientific_payload_sha256"),
        label="scientific_payload_sha256",
    )
    status = causal_map.get("scientific_status")
    safety = causal_map.get("safety")
    if not isinstance(status, Mapping) or not isinstance(safety, Mapping):
        raise TypeError("causal_map status and safety must be mappings")
    if (
        status.get("observational_metrics_only") is not True
        or safety.get("analysis_only") is not True
    ):
        raise ValueError("causal_map must remain observational analysis only")
    _no_authority(status, label="causal_map scientific_status")
    _no_authority(safety, label="causal_map safety")

    raw_nodes = causal_map.get("generator_nodes")
    raw_edges = causal_map.get("directed_edges")
    if (
        isinstance(raw_nodes, (str, bytes))
        or not isinstance(raw_nodes, Sequence)
        or isinstance(raw_edges, (str, bytes))
        or not isinstance(raw_edges, Sequence)
    ):
        raise TypeError("causal_map nodes and directed edges must be sequences")
    nodes = tuple(raw_nodes)
    edges = tuple(raw_edges)
    if len(nodes) != _LAYER_COUNT:
        raise ValueError("causal_map must contain exactly 18 generator nodes")
    if len(edges) != _PAIR_COUNT:
        raise ValueError("causal_map must contain all 153 directed edges")

    generator_ids: set[str] = set()
    for ordinal, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError("generator node rows must be mappings")
        generator_id = _require_name(
            node.get("generator_id"),
            label=f"generator node {ordinal} id",
        )
        if (
            node.get("layer_ordinal") != ordinal
            or node.get("observational_only") is not True
            or generator_id in generator_ids
        ):
            raise ValueError(
                f"generator node {ordinal} is inconsistent with strict map"
            )
        _no_authority(node, label=f"generator node {ordinal}")
        generator_ids.add(generator_id)

    expected_pairs = tuple(combinations(range(_LAYER_COUNT), 2))
    for pair, edge in zip(expected_pairs, edges):
        if not isinstance(edge, Mapping):
            raise TypeError("directed edge rows must be mappings")
        upstream, downstream = pair
        if (
            edge.get("upstream_layer_ordinal") != upstream
            or edge.get("downstream_layer_ordinal") != downstream
            or edge.get("observational_only") is not True
            or edge.get("strict_upstream_invariance_confirmed") is not True
        ):
            raise ValueError(
                f"directed edge {upstream}->{downstream} is inconsistent "
                "with strict map"
            )
        _no_authority(
            edge,
            label=f"directed edge {upstream}->{downstream}",
        )
        _finite(
            edge.get("mean_directed_response_rms"),
            label=f"edge {upstream}->{downstream} mean response RMS",
        )
        _finite(
            edge.get("maximum_directed_response_rms"),
            label=f"edge {upstream}->{downstream} maximum response RMS",
        )
        _finite(
            edge.get("mean_directed_response_ratio_over_defined"),
            label=f"edge {upstream}->{downstream} mean response ratio",
        )
    return source_digest, nodes, edges


def nominate_gemma3_generator_hierarchy(
    causal_map: Mapping[str, object],
    *,
    causal_groups: Sequence[CausalGroupNomination] = (),
    sharing_families: Sequence[SharingFamilyNomination] = (),
    evidence_semantics: str = FINITE_INTERVENTION_RESPONSE_EVIDENCE,
    jacobian_interpretation: bool = False,
    authorizes_execution: bool = False,
) -> GeneratorHierarchyNomination:
    """Nominate a complete analysis-only parent boundary.

    ``causal_map`` must be the mapping returned by the strict causal-map
    loader.  The source digest is bound into every resulting top-level
    nomination hash, but this bridge intentionally does not replace the
    strict loader or re-authenticate arbitrary mappings.
    """

    if (
        evidence_semantics != FINITE_INTERVENTION_RESPONSE_EVIDENCE
        or jacobian_interpretation is not False
    ):
        raise ValueError(
            "finite intervention responses cannot be labeled as Jacobians"
        )
    if authorizes_execution is not False:
        raise ValueError(
            "an observational hierarchy nomination cannot authorize execution"
        )
    source_digest, nodes, source_edges = _strict_source_catalog(causal_map)

    groups = tuple(causal_groups)
    families = tuple(sharing_families)
    if any(not isinstance(group, CausalGroupNomination) for group in groups):
        raise TypeError(
            "causal_groups must contain CausalGroupNomination values"
        )
    if any(
        not isinstance(family, SharingFamilyNomination)
        for family in families
    ):
        raise TypeError(
            "sharing_families must contain SharingFamilyNomination values"
        )
    group_ids = tuple(group.parent_id for group in groups)
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("causal group parent ids must be unique")
    claimed_layers: set[int] = set()
    for group in groups:
        overlap = claimed_layers.intersection(group.child_layer_ordinals)
        if overlap:
            raise ValueError(
                "causal groups must be disjoint; child layers cannot appear "
                "in more than one parent"
            )
        claimed_layers.update(group.child_layer_ordinals)

    generator_ids = tuple(
        str(node["generator_id"])
        for node in nodes
    )
    group_by_start = {
        group.child_layer_ordinals[0]: group for group in groups
    }
    parent_rows: list[HierarchyParentNomination] = []
    layer_to_parent: dict[int, str] = {}
    layer = 0
    while layer < _LAYER_COUNT:
        group = group_by_start.get(layer)
        if group is None:
            if layer in claimed_layers:
                raise AssertionError("group interval traversal drifted")
            parent_id = f"singleton/L{layer:02d}"
            child_layers = (layer,)
            kind = "singleton_passthrough"
        else:
            parent_id = group.parent_id
            child_layers = group.child_layer_ordinals
            kind = "causal_contraction_candidate"
        parent = HierarchyParentNomination(
            parent_id=parent_id,
            child_layer_ordinals=child_layers,
            generator_ids=tuple(
                generator_ids[child] for child in child_layers
            ),
            kind=kind,
        )
        parent_rows.append(parent)
        for child in child_layers:
            layer_to_parent[child] = parent_id
        layer = child_layers[-1] + 1
    parent_ids = tuple(parent.parent_id for parent in parent_rows)
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(
            "causal group ids cannot collide with generated singleton ids"
        )

    bound_families: list[SharingFamilyNomination] = []
    family_ids: set[str] = set()
    for family in families:
        if family.family_id in family_ids:
            raise ValueError("sharing family ids must be unique")
        if family.generator_ids:
            expected = tuple(
                generator_ids[layer]
                for layer in family.child_layer_ordinals
            )
            if family.generator_ids != expected:
                raise ValueError(
                    f"sharing family {family.family_id} generator catalog "
                    "does not match causal map"
                )
            bound = family
        else:
            bound = SharingFamilyNomination(
                family_id=family.family_id,
                child_layer_ordinals=family.child_layer_ordinals,
                generator_ids=tuple(
                    generator_ids[layer]
                    for layer in family.child_layer_ordinals
                ),
            )
        bound_families.append(bound)
        family_ids.add(family.family_id)

    edge_rows: list[DirectedEdgeNomination] = []
    for edge_ordinal, source_edge in enumerate(source_edges):
        upstream = int(source_edge["upstream_layer_ordinal"])
        downstream = int(source_edge["downstream_layer_ordinal"])
        upstream_parent = layer_to_parent[upstream]
        downstream_parent = layer_to_parent[downstream]
        disposition = (
            "internal"
            if upstream_parent == downstream_parent
            else "surfaced_cut"
        )
        source_edge_sha256 = _digest(
            {
                "source_scientific_payload_sha256": source_digest,
                "directed_edge": source_edge,
            },
            domain=_SOURCE_EDGE_DIGEST_DOMAIN,
        )
        edge_rows.append(
            DirectedEdgeNomination(
                edge_ordinal=edge_ordinal,
                upstream_layer_ordinal=upstream,
                downstream_layer_ordinal=downstream,
                upstream_parent_id=upstream_parent,
                downstream_parent_id=downstream_parent,
                disposition=disposition,
                source_edge_sha256=source_edge_sha256,
                mean_directed_response_rms=float(
                    source_edge["mean_directed_response_rms"]
                ),
                maximum_directed_response_rms=float(
                    source_edge["maximum_directed_response_rms"]
                ),
                mean_directed_response_ratio_over_defined=float(
                    source_edge[
                        "mean_directed_response_ratio_over_defined"
                    ]
                ),
            )
        )

    internal_count = sum(
        edge.disposition == "internal" for edge in edge_rows
    )
    return GeneratorHierarchyNomination(
        source_scientific_payload_sha256=source_digest,
        parents=tuple(parent_rows),
        sharing_families=tuple(bound_families),
        directed_edges=tuple(edge_rows),
        internal_edge_count=internal_count,
        surfaced_cut_edge_count=_PAIR_COUNT - internal_count,
    )


def known_v1_gemma3_generator_hierarchy_specs() -> tuple[
    tuple[CausalGroupNomination, ...],
    tuple[SharingFamilyNomination, ...],
]:
    """Return the current open-development L3/L4 and L12/L15 hypotheses."""

    return (
        (
            CausalGroupNomination(
                parent_id="causal_parent/L03-L04",
                child_layer_ordinals=(3, 4),
            ),
        ),
        (
            SharingFamilyNomination(
                family_id="sharing_family/L12-L15",
                child_layer_ordinals=(12, 15),
            ),
        ),
    )


def nominate_known_v1_gemma3_generator_hierarchy(
    causal_map: Mapping[str, object],
) -> GeneratorHierarchyNomination:
    """Apply the known v1 analysis-only nomination to one strict map."""

    causal_groups, sharing_families = (
        known_v1_gemma3_generator_hierarchy_specs()
    )
    return nominate_gemma3_generator_hierarchy(
        causal_map,
        causal_groups=causal_groups,
        sharing_families=sharing_families,
    )
