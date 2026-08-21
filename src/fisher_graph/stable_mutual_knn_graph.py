"""Deterministic, model-agnostic stable mutual-kNN graph receipts.

The core consumes authenticated directed sufficient statistics for two frozen
panels.  It pools sufficient statistics before forming ratios, recomputes
point, within-family bootstrap, and leave-one-family-out neighbor topology,
and fails closed when a node has a tie at the k-neighbor boundary.

Exact-equivalence classes are atomic graph nodes.  Their logical members are
retained only for exact expansion in the receipt; they never participate as
separate ranking candidates.  Connected components are topology diagnostics
and maximal cliques are overlapping stable-neighbor candidate groups.  This
module does not authorize a mode family, cluster, merge, compression, or
deployment claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Final, Mapping, Sequence

import numpy as np

from fisher_graph.mode_similarity_atlas import (
    WithinFamilyBootstrapConfig,
    parse_within_family_bootstrap_config_payload,
)
from fisher_graph.multi_abi_mode_atlas import (
    DownstreamResponseExamplePairStats,
    DownstreamResponsePairStats,
)


__all__ = [
    "STABLE_MUTUAL_KNN_GRAPH_FORMAT_VERSION",
    "STABLE_MUTUAL_KNN_GRAPH_SCHEMA",
    "STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY",
    "StableMutualKNNConfig",
    "StableMutualKNNExampleStratum",
    "StableMutualKNNGraphReceipt",
    "StableMutualKNNInput",
    "StableMutualKNNNodeSpec",
    "StableMutualKNNPanelInput",
    "StableMutualKNNStratumGroup",
    "build_stable_mutual_knn_graph_receipt",
    "load_stable_mutual_knn_graph_receipt",
    "parse_stable_mutual_knn_graph_receipt_payload",
    "save_stable_mutual_knn_graph_receipt",
    "stable_mutual_knn_candidate_catalog_sha256",
]


STABLE_MUTUAL_KNN_GRAPH_SCHEMA: Final = (
    "fisher_graph.stable_mutual_knn_graph"
)
STABLE_MUTUAL_KNN_GRAPH_FORMAT_VERSION: Final = 1
STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY: Final = (
    "fail_closed_undefined_node_neighbors"
)

_CANDIDATE_CATALOG_DOMAIN = (
    b"fisher_graph.stable_mutual_knn_graph.candidate_catalog.v1\0"
)
_INPUT_DOMAIN = b"fisher_graph.stable_mutual_knn_graph.input.v1\0"
_RECEIPT_DOMAIN = b"fisher_graph.stable_mutual_knn_graph.receipt.v1\0"
_BOOTSTRAP_TOPOLOGY_DOMAIN = (
    b"fisher_graph.stable_mutual_knn_graph.bootstrap_topology.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_MAX_RECEIPT_BYTES = 256 * 1024 * 1024
_ENERGY_EPSILON = 1e-12
_SUFFICIENT_STAT_RELATIVE_TOLERANCE = 1e-7
_POOLED_STRATUM_ID = "pooled"

_SAFETY: dict[str, object] = {
    "contains_model_weights": False,
    "contains_prompt_text": False,
    "contains_raw_activations": False,
    "contains_raw_gradients": False,
    "contains_tensors": False,
    "contains_token_ids": False,
    "contains_only_aggregate_sufficient_statistics": True,
}

_SCIENTIFIC_STATUS: dict[str, object] = {
    "analysis_only": True,
    "candidate_edges_are_preregistered_discovery_results": True,
    "connected_components_are_topology_diagnostics_only": True,
    "fresh_non_candidate_edges_are_exploratory_only": True,
    "maximal_cliques_are_overlapping_stable_neighbor_candidate_groups": True,
    "transitive_similarity_inferred_from_chains": False,
    "authorizes_mode_family_claim": False,
    "authorizes_cluster_claim": False,
    "authorizes_mode_merge": False,
    "authorizes_parameter_compression": False,
    "authorizes_deployment": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_digest(domain: bytes, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable nonempty name")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("an edge must join distinct nodes")
    return (left, right) if left < right else (right, left)


def _edge_payload(
    edge: tuple[str, str],
    member_ids_by_node: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    left, right = edge
    return {
        "left_member_mode_ids": list(member_ids_by_node[left]),
        "left_node_id": left,
        "right_member_mode_ids": list(member_ids_by_node[right]),
        "right_node_id": right,
    }


@dataclass(frozen=True, slots=True)
class StableMutualKNNNodeSpec:
    """One atomic exact-equivalence class on the fixed graph axis."""

    node_id: str
    member_mode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_name(self.node_id, label="node_id")
        if not isinstance(self.member_mode_ids, tuple):
            raise TypeError("member_mode_ids must be a tuple")
        if not self.member_mode_ids:
            raise ValueError("member_mode_ids must not be empty")
        for index, mode_id in enumerate(self.member_mode_ids):
            _require_name(mode_id, label=f"member_mode_ids[{index}]")
        if len(set(self.member_mode_ids)) != len(self.member_mode_ids):
            raise ValueError("member_mode_ids must be unique within a node")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "member_mode_ids": list(self.member_mode_ids),
            "node_id": self.node_id,
        }


@dataclass(frozen=True, slots=True)
class StableMutualKNNExampleStratum:
    """Authenticated stratum label for one family/example cluster."""

    family_id: str
    example_cluster_id: str
    stratum_id: str

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.example_cluster_id, label="example_cluster_id")
        _require_name(self.stratum_id, label="stratum_id")
        if self.stratum_id == _POOLED_STRATUM_ID:
            raise ValueError("pooled is reserved for the synthetic full panel")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "example_cluster_id": self.example_cluster_id,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
        }


@dataclass(frozen=True, slots=True)
class StableMutualKNNStratumGroup:
    """A point-evaluation stratum formed from frozen example labels."""

    group_id: str
    member_stratum_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_name(self.group_id, label="group_id")
        if self.group_id == _POOLED_STRATUM_ID:
            raise ValueError("pooled is reserved for the synthetic full panel")
        if not isinstance(self.member_stratum_ids, tuple):
            raise TypeError("member_stratum_ids must be a tuple")
        if not self.member_stratum_ids:
            raise ValueError("member_stratum_ids must not be empty")
        for index, stratum_id in enumerate(self.member_stratum_ids):
            _require_name(stratum_id, label=f"member_stratum_ids[{index}]")
            if stratum_id == _POOLED_STRATUM_ID:
                raise ValueError("member_stratum_ids must not contain pooled")
        if len(set(self.member_stratum_ids)) != len(
            self.member_stratum_ids
        ):
            raise ValueError("member_stratum_ids must be unique")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "member_stratum_ids": list(self.member_stratum_ids),
        }


@dataclass(frozen=True, slots=True)
class StableMutualKNNConfig:
    """Frozen topology, error-gate, and stability policy."""

    k: int = 2
    rank_boundary_tie_absolute_tolerance: float = 1e-10
    rank_boundary_tie_policy: str = (
        STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY
    )
    energy_epsilon: float = _ENERGY_EPSILON
    max_directional_relative_sse: float = 1.0
    max_absolute_weighted_rmse: float = 0.125
    bootstrap_min_mutual_frequency: float = 0.80
    lofo_min_mutual_count: int = 5
    lofo_replicate_count: int = 6
    bootstrap_resample_count: int = 10_000
    bootstrap_max_degenerate_fraction: float = 0.01
    minimum_clique_size: int = 2
    primary_stratum_id: str = _POOLED_STRATUM_ID
    hard_veto_stratum_ids: tuple[str, ...] = ()
    diagnostic_stratum_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.k, label="k")
        _require_finite(
            self.rank_boundary_tie_absolute_tolerance,
            label="rank_boundary_tie_absolute_tolerance",
            strictly_positive=True,
        )
        if (
            self.rank_boundary_tie_policy
            != STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY
        ):
            raise ValueError("rank_boundary_tie_policy is unsupported")
        _require_finite(
            self.energy_epsilon,
            label="energy_epsilon",
            strictly_positive=True,
        )
        _require_finite(
            self.max_directional_relative_sse,
            label="max_directional_relative_sse",
            minimum=0.0,
        )
        _require_finite(
            self.max_absolute_weighted_rmse,
            label="max_absolute_weighted_rmse",
            minimum=0.0,
        )
        _require_finite(
            self.bootstrap_min_mutual_frequency,
            label="bootstrap_min_mutual_frequency",
            minimum=0.0,
            maximum=1.0,
        )
        _require_positive_int(
            self.lofo_min_mutual_count, label="lofo_min_mutual_count"
        )
        _require_positive_int(
            self.lofo_replicate_count, label="lofo_replicate_count"
        )
        if self.lofo_min_mutual_count > self.lofo_replicate_count:
            raise ValueError(
                "lofo_min_mutual_count must not exceed lofo_replicate_count"
            )
        _require_positive_int(
            self.bootstrap_resample_count,
            label="bootstrap_resample_count",
        )
        _require_finite(
            self.bootstrap_max_degenerate_fraction,
            label="bootstrap_max_degenerate_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_positive_int(
            self.minimum_clique_size, label="minimum_clique_size"
        )
        if self.primary_stratum_id != _POOLED_STRATUM_ID:
            raise ValueError("primary_stratum_id must be pooled")
        for field_name in ("hard_veto_stratum_ids", "diagnostic_stratum_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            for index, value in enumerate(values):
                _require_name(value, label=f"{field_name}[{index}]")
                if value == _POOLED_STRATUM_ID:
                    raise ValueError(
                        f"{field_name} must not contain reserved pooled"
                    )
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique IDs")
        if set(self.hard_veto_stratum_ids) & set(
            self.diagnostic_stratum_ids
        ):
            raise ValueError(
                "hard-veto and diagnostic strata must be disjoint"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bootstrap_max_degenerate_fraction": float(
                self.bootstrap_max_degenerate_fraction
            ),
            "bootstrap_min_mutual_frequency": float(
                self.bootstrap_min_mutual_frequency
            ),
            "bootstrap_resample_count": self.bootstrap_resample_count,
            "diagnostic_stratum_ids": list(self.diagnostic_stratum_ids),
            "energy_epsilon": float(self.energy_epsilon),
            "hard_veto_stratum_ids": list(self.hard_veto_stratum_ids),
            "k": self.k,
            "lofo_min_mutual_count": self.lofo_min_mutual_count,
            "lofo_replicate_count": self.lofo_replicate_count,
            "max_absolute_weighted_rmse": float(
                self.max_absolute_weighted_rmse
            ),
            "max_directional_relative_sse": float(
                self.max_directional_relative_sse
            ),
            "minimum_clique_size": self.minimum_clique_size,
            "primary_stratum_id": self.primary_stratum_id,
            "rank_boundary_tie_absolute_tolerance": float(
                self.rank_boundary_tie_absolute_tolerance
            ),
            "rank_boundary_tie_policy": (
                self.rank_boundary_tie_policy
            ),
        }


@dataclass(frozen=True, slots=True)
class StableMutualKNNPanelInput:
    """Raw sufficient statistics and resampling contract for one panel."""

    panel_role: str
    panel_binding_sha256: str
    family_pair_stats: tuple[DownstreamResponsePairStats, ...]
    family_example_pair_stats: tuple[
        DownstreamResponseExamplePairStats, ...
    ]
    within_family_bootstrap_config: WithinFamilyBootstrapConfig
    example_strata: tuple[StableMutualKNNExampleStratum, ...] = ()
    stratum_groups: tuple[StableMutualKNNStratumGroup, ...] = ()

    def __post_init__(self) -> None:
        if self.panel_role not in {"discovery", "fresh_confirmation"}:
            raise ValueError(
                "panel_role must be discovery or fresh_confirmation"
            )
        _require_sha256(
            self.panel_binding_sha256, label="panel_binding_sha256"
        )
        if not isinstance(self.family_pair_stats, tuple):
            raise TypeError("family_pair_stats must be a tuple")
        if not isinstance(self.family_example_pair_stats, tuple):
            raise TypeError("family_example_pair_stats must be a tuple")
        if not self.family_pair_stats:
            raise ValueError("family_pair_stats must not be empty")
        if not self.family_example_pair_stats:
            raise ValueError("family_example_pair_stats must not be empty")
        if any(
            not isinstance(row, DownstreamResponsePairStats)
            for row in self.family_pair_stats
        ):
            raise TypeError(
                "family_pair_stats must contain DownstreamResponsePairStats"
            )
        if any(
            not isinstance(row, DownstreamResponseExamplePairStats)
            for row in self.family_example_pair_stats
        ):
            raise TypeError(
                "family_example_pair_stats must contain "
                "DownstreamResponseExamplePairStats"
            )
        if not isinstance(
            self.within_family_bootstrap_config,
            WithinFamilyBootstrapConfig,
        ):
            raise TypeError(
                "within_family_bootstrap_config must be a "
                "WithinFamilyBootstrapConfig"
            )
        if not isinstance(self.example_strata, tuple):
            raise TypeError("example_strata must be a tuple")
        if any(
            not isinstance(item, StableMutualKNNExampleStratum)
            for item in self.example_strata
        ):
            raise TypeError(
                "example_strata must contain StableMutualKNNExampleStratum"
            )
        if not isinstance(self.stratum_groups, tuple):
            raise TypeError("stratum_groups must be a tuple")
        if any(
            not isinstance(item, StableMutualKNNStratumGroup)
            for item in self.stratum_groups
        ):
            raise TypeError(
                "stratum_groups must contain StableMutualKNNStratumGroup"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family_example_pair_stats": [
                row.canonical_payload()
                for row in self.family_example_pair_stats
            ],
            "family_pair_stats": [
                row.canonical_payload() for row in self.family_pair_stats
            ],
            "example_strata": [
                item.canonical_payload() for item in self.example_strata
            ],
            "panel_binding_sha256": self.panel_binding_sha256,
            "panel_role": self.panel_role,
            "stratum_groups": [
                item.canonical_payload() for item in self.stratum_groups
            ],
            "within_family_bootstrap_config": (
                self.within_family_bootstrap_config.canonical_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class StableMutualKNNInput:
    """Complete frozen input for a discovery/fresh graph confirmation."""

    node_specs: tuple[StableMutualKNNNodeSpec, ...]
    candidate_edges: tuple[tuple[str, str], ...]
    candidate_catalog_sha256: str
    discovery: StableMutualKNNPanelInput
    fresh_confirmation: StableMutualKNNPanelInput
    config: StableMutualKNNConfig

    def __post_init__(self) -> None:
        if not isinstance(self.node_specs, tuple):
            raise TypeError("node_specs must be a tuple")
        if any(
            not isinstance(item, StableMutualKNNNodeSpec)
            for item in self.node_specs
        ):
            raise TypeError("node_specs must contain StableMutualKNNNodeSpec")
        if not isinstance(self.candidate_edges, tuple):
            raise TypeError("candidate_edges must be a tuple")
        _require_sha256(
            self.candidate_catalog_sha256,
            label="candidate_catalog_sha256",
        )
        if not isinstance(self.discovery, StableMutualKNNPanelInput):
            raise TypeError("discovery must be a StableMutualKNNPanelInput")
        if not isinstance(
            self.fresh_confirmation, StableMutualKNNPanelInput
        ):
            raise TypeError(
                "fresh_confirmation must be a StableMutualKNNPanelInput"
            )
        if self.discovery.panel_role != "discovery":
            raise ValueError("discovery panel has the wrong panel_role")
        if self.fresh_confirmation.panel_role != "fresh_confirmation":
            raise ValueError(
                "fresh_confirmation panel has the wrong panel_role"
            )
        if not isinstance(self.config, StableMutualKNNConfig):
            raise TypeError("config must be a StableMutualKNNConfig")


@dataclass(frozen=True, slots=True)
class StableMutualKNNGraphReceipt:
    """Authenticated raw inputs and deterministic graph confirmation output."""

    graph_input: StableMutualKNNInput
    input_sha256: str
    analysis: Mapping[str, object]
    receipt_sha256: str

    def _payload_without_receipt_sha256(self) -> dict[str, object]:
        return {
            "analysis": dict(self.analysis),
            "candidate_catalog_sha256": (
                self.graph_input.candidate_catalog_sha256
            ),
            "config": self.graph_input.config.canonical_payload(),
            "format_version": STABLE_MUTUAL_KNN_GRAPH_FORMAT_VERSION,
            "input": _graph_input_payload(self.graph_input),
            "input_sha256": self.input_sha256,
            "safety": dict(_SAFETY),
            "schema": STABLE_MUTUAL_KNN_GRAPH_SCHEMA,
            "scientific_status": dict(_SCIENTIFIC_STATUS),
        }

    def canonical_payload(self) -> dict[str, object]:
        payload = self._payload_without_receipt_sha256()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _normalized_candidate_edges(
    candidate_edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for index, edge in enumerate(candidate_edges):
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError(f"candidate_edges[{index}] must be a pair")
        left = _require_name(edge[0], label=f"candidate_edges[{index}][0]")
        right = _require_name(edge[1], label=f"candidate_edges[{index}][1]")
        result.append(_canonical_pair(left, right))
    normalized = tuple(sorted(result))
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate_edges must be unique")
    if not normalized:
        raise ValueError("candidate_edges must not be empty")
    return normalized


def stable_mutual_knn_candidate_catalog_sha256(
    node_specs: Sequence[StableMutualKNNNodeSpec],
    candidate_edges: Sequence[tuple[str, str]],
) -> str:
    """Authenticate the fixed quotient axis and preregistered edge catalog."""

    nodes = tuple(node_specs)
    if any(not isinstance(item, StableMutualKNNNodeSpec) for item in nodes):
        raise TypeError("node_specs must contain StableMutualKNNNodeSpec")
    edges = _normalized_candidate_edges(candidate_edges)
    return _domain_digest(
        _CANDIDATE_CATALOG_DOMAIN,
        [list(edge) for edge in edges],
    )


def _canonical_panel_input(
    panel: StableMutualKNNPanelInput,
) -> StableMutualKNNPanelInput:
    stats = tuple(
        sorted(
            panel.family_pair_stats,
            key=lambda row: (
                row.family_id,
                row.mode_a_id,
                row.mode_b_id,
            ),
        )
    )
    example_stats = tuple(
        sorted(
            panel.family_example_pair_stats,
            key=lambda row: (
                row.family_id,
                row.example_cluster_id,
                row.mode_a_id,
                row.mode_b_id,
            ),
        )
    )
    strata = tuple(
        sorted(
            panel.example_strata,
            key=lambda item: (
                item.family_id,
                item.example_cluster_id,
                item.stratum_id,
            ),
        )
    )
    groups = tuple(sorted(panel.stratum_groups, key=lambda item: item.group_id))
    return StableMutualKNNPanelInput(
        panel_role=panel.panel_role,
        panel_binding_sha256=panel.panel_binding_sha256,
        family_pair_stats=stats,
        family_example_pair_stats=example_stats,
        within_family_bootstrap_config=panel.within_family_bootstrap_config,
        example_strata=strata,
        stratum_groups=groups,
    )


def _graph_input_payload(graph_input: StableMutualKNNInput) -> dict[str, object]:
    return {
        "candidate_edges": [list(edge) for edge in graph_input.candidate_edges],
        "discovery": graph_input.discovery.canonical_payload(),
        "fresh_confirmation": (
            graph_input.fresh_confirmation.canonical_payload()
        ),
        "node_specs": [
            node.canonical_payload() for node in graph_input.node_specs
        ],
    }


@dataclass(frozen=True, slots=True)
class _ValidatedPanel:
    panel: StableMutualKNNPanelInput
    family_ids: tuple[str, ...]
    cluster_ids_by_family: Mapping[str, tuple[str, ...]]
    pair_rows: Mapping[tuple[str, str, str], DownstreamResponsePairStats]
    example_rows: Mapping[
        tuple[str, str, str, str], DownstreamResponseExamplePairStats
    ]
    response_abi_sha256: str
    stratum_by_example: Mapping[tuple[str, str], str]
    member_stratum_ids_by_group: Mapping[str, tuple[str, ...]]


def _validate_pair_quadratic(
    row: DownstreamResponsePairStats,
) -> None:
    expected_sse = (
        float(row.weighted_mode_a_energy)
        + float(row.weighted_mode_b_energy)
        - 2.0 * float(row.weighted_dot)
    )
    scale = max(
        1.0,
        float(row.weighted_sse),
        float(row.weighted_mode_a_energy),
        float(row.weighted_mode_b_energy),
        abs(float(row.weighted_dot)),
    )
    if (
        abs(float(row.weighted_sse) - expected_sse)
        > _SUFFICIENT_STAT_RELATIVE_TOLERANCE * scale
    ):
        raise ValueError(
            "weighted_sse is inconsistent with the pair quadratic"
        )
    bound = math.sqrt(
        float(row.weighted_mode_a_energy)
        * float(row.weighted_mode_b_energy)
    )
    if (
        abs(float(row.weighted_dot))
        > bound + _SUFFICIENT_STAT_RELATIVE_TOLERANCE * scale
    ):
        raise ValueError("weighted_dot violates the Cauchy bound")


def _canonicalize_pair_row(
    row: DownstreamResponsePairStats,
    *,
    logical_index: Mapping[str, int],
) -> DownstreamResponsePairStats:
    if logical_index[row.mode_a_id] < logical_index[row.mode_b_id]:
        return row
    return DownstreamResponsePairStats(
        family_id=row.family_id,
        mode_a_id=row.mode_b_id,
        mode_b_id=row.mode_a_id,
        response_abi_sha256=row.response_abi_sha256,
        weighted_sse=row.weighted_sse,
        weighted_mode_a_energy=row.weighted_mode_b_energy,
        weighted_dot=row.weighted_dot,
        weighted_mode_b_energy=row.weighted_mode_a_energy,
        row_count=row.row_count,
    )


def _canonicalize_example_pair_row(
    row: DownstreamResponseExamplePairStats,
    *,
    logical_index: Mapping[str, int],
) -> DownstreamResponseExamplePairStats:
    pair = _canonicalize_pair_row(
        row.as_pair_stats(), logical_index=logical_index
    )
    return DownstreamResponseExamplePairStats(
        family_id=pair.family_id,
        example_cluster_id=row.example_cluster_id,
        mode_a_id=pair.mode_a_id,
        mode_b_id=pair.mode_b_id,
        response_abi_sha256=pair.response_abi_sha256,
        weighted_sse=pair.weighted_sse,
        weighted_mode_a_energy=pair.weighted_mode_a_energy,
        weighted_dot=pair.weighted_dot,
        weighted_mode_b_energy=pair.weighted_mode_b_energy,
        row_count=pair.row_count,
    )


def _pair_key(
    family_id: str,
    left: str,
    right: str,
    *,
    logical_index: Mapping[str, int],
) -> tuple[str, str, str]:
    if logical_index[left] < logical_index[right]:
        return (family_id, left, right)
    return (family_id, right, left)


def _example_pair_key(
    family_id: str,
    example_cluster_id: str,
    left: str,
    right: str,
    *,
    logical_index: Mapping[str, int],
) -> tuple[str, str, str, str]:
    pair = _pair_key(
        family_id, left, right, logical_index=logical_index
    )
    return (pair[0], example_cluster_id, pair[1], pair[2])


def _oriented_pair_values(
    row: DownstreamResponsePairStats,
    *,
    left: str,
    right: str,
) -> tuple[float, float, float, float, int]:
    if row.mode_a_id == left and row.mode_b_id == right:
        return (
            float(row.weighted_sse),
            float(row.weighted_mode_a_energy),
            float(row.weighted_dot),
            float(row.weighted_mode_b_energy),
            row.row_count,
        )
    if row.mode_a_id == right and row.mode_b_id == left:
        return (
            float(row.weighted_sse),
            float(row.weighted_mode_b_energy),
            float(row.weighted_dot),
            float(row.weighted_mode_a_energy),
            row.row_count,
        )
    raise ValueError("pair row does not contain the requested modes")


def _validate_exact_equivalence_closure(
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    families: tuple[str, ...],
    clusters_by_family: Mapping[str, tuple[str, ...]],
    pair_rows: Mapping[tuple[str, str, str], DownstreamResponsePairStats],
    example_rows: Mapping[
        tuple[str, str, str, str], DownstreamResponseExamplePairStats
    ],
    logical_index: Mapping[str, int],
) -> None:
    def require_exact_identity(
        row: DownstreamResponsePairStats,
        *,
        label: str,
    ) -> None:
        if not (
            float(row.weighted_sse) == 0.0
            and float(row.weighted_mode_a_energy)
            == float(row.weighted_dot)
            == float(row.weighted_mode_b_energy)
        ):
            raise ValueError(f"{label} is not an exact identity pair")

    for node in nodes:
        members = node.member_mode_ids
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                for family_id in families:
                    row = pair_rows[
                        _pair_key(
                            family_id,
                            left,
                            right,
                            logical_index=logical_index,
                        )
                    ]
                    require_exact_identity(
                        row,
                        label=(
                            f"declared equivalence {node.node_id} aggregate"
                        ),
                    )
                    for cluster_id in clusters_by_family[family_id]:
                        example_row = example_rows[
                            _example_pair_key(
                                family_id,
                                cluster_id,
                                left,
                                right,
                                logical_index=logical_index,
                            )
                        ].as_pair_stats()
                        require_exact_identity(
                            example_row,
                            label=(
                                f"declared equivalence {node.node_id} example"
                            ),
                        )

    def require_substitution_equal(
        baseline: tuple[float, float, float, float, int],
        candidate: tuple[float, float, float, float, int],
        *,
        label: str,
    ) -> None:
        if candidate != baseline:
            raise ValueError(
                f"{label} violates exact member-substitution closure"
            )

    for left_node_index, left_node in enumerate(nodes):
        for right_node in nodes[left_node_index + 1 :]:
            representative_left = left_node.member_mode_ids[0]
            representative_right = right_node.member_mode_ids[0]
            for family_id in families:
                baseline_row = pair_rows[
                    _pair_key(
                        family_id,
                        representative_left,
                        representative_right,
                        logical_index=logical_index,
                    )
                ]
                baseline = _oriented_pair_values(
                    baseline_row,
                    left=representative_left,
                    right=representative_right,
                )
                for left in left_node.member_mode_ids:
                    for right in right_node.member_mode_ids:
                        row = pair_rows[
                            _pair_key(
                                family_id,
                                left,
                                right,
                                logical_index=logical_index,
                            )
                        ]
                        require_substitution_equal(
                            baseline,
                            _oriented_pair_values(
                                row, left=left, right=right
                            ),
                            label=(
                                f"aggregate {left_node.node_id}/"
                                f"{right_node.node_id}"
                            ),
                        )
                for cluster_id in clusters_by_family[family_id]:
                    baseline_example = example_rows[
                        _example_pair_key(
                            family_id,
                            cluster_id,
                            representative_left,
                            representative_right,
                            logical_index=logical_index,
                        )
                    ].as_pair_stats()
                    baseline_values = _oriented_pair_values(
                        baseline_example,
                        left=representative_left,
                        right=representative_right,
                    )
                    for left in left_node.member_mode_ids:
                        for right in right_node.member_mode_ids:
                            row = example_rows[
                                _example_pair_key(
                                    family_id,
                                    cluster_id,
                                    left,
                                    right,
                                    logical_index=logical_index,
                                )
                            ].as_pair_stats()
                            require_substitution_equal(
                                baseline_values,
                                _oriented_pair_values(
                                    row, left=left, right=right
                                ),
                                label=(
                                    f"example {left_node.node_id}/"
                                    f"{right_node.node_id}"
                                ),
                            )


def _validate_panel(
    panel: StableMutualKNNPanelInput,
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    logical_axis: tuple[str, ...],
    config: StableMutualKNNConfig,
) -> _ValidatedPanel:
    logical_set = set(logical_axis)
    logical_index = {
        mode_id: index for index, mode_id in enumerate(logical_axis)
    }
    bootstrap = panel.within_family_bootstrap_config
    if bootstrap.resample_count != config.bootstrap_resample_count:
        raise ValueError(
            f"{panel.panel_role} bootstrap resample count does not match config"
        )
    if not _close(
        bootstrap.maximum_degenerate_fraction,
        config.bootstrap_max_degenerate_fraction,
        tolerance=0.0,
    ):
        raise ValueError(
            f"{panel.panel_role} bootstrap degenerate fraction does not match "
            "config"
        )
    families = bootstrap.ordered_family_ids
    if len(families) != config.lofo_replicate_count:
        raise ValueError(
            f"{panel.panel_role} family count must equal lofo_replicate_count"
        )
    clusters_by_family = dict(bootstrap.family_example_cluster_ids)

    pair_rows: dict[tuple[str, str, str], DownstreamResponsePairStats] = {}
    abi_sha256s: set[str] = set()
    for raw_row in panel.family_pair_stats:
        if (
            raw_row.mode_a_id not in logical_set
            or raw_row.mode_b_id not in logical_set
        ):
            raise ValueError(
                f"{panel.panel_role} aggregate row references unknown mode"
            )
        row = _canonicalize_pair_row(raw_row, logical_index=logical_index)
        _validate_pair_quadratic(row)
        key = (row.family_id, row.mode_a_id, row.mode_b_id)
        if key in pair_rows:
            raise ValueError(
                f"{panel.panel_role} has duplicate family pair statistics"
            )
        pair_rows[key] = row
        abi_sha256s.add(row.response_abi_sha256)
    expected_pair_keys = {
        (family_id, logical_axis[left], logical_axis[right])
        for family_id in families
        for left in range(len(logical_axis))
        for right in range(left + 1, len(logical_axis))
    }
    if set(pair_rows) != expected_pair_keys:
        raise ValueError(
            f"{panel.panel_role} aggregate statistics must be the complete "
            "logical-mode unordered matrix"
        )
    if len(abi_sha256s) != 1:
        raise ValueError(
            f"{panel.panel_role} must use one response ABI SHA-256"
        )
    response_abi_sha256 = next(iter(abi_sha256s))

    example_rows: dict[
        tuple[str, str, str, str], DownstreamResponseExamplePairStats
    ] = {}
    for raw_row in panel.family_example_pair_stats:
        if (
            raw_row.mode_a_id not in logical_set
            or raw_row.mode_b_id not in logical_set
        ):
            raise ValueError(
                f"{panel.panel_role} example row references unknown mode"
            )
        row = _canonicalize_example_pair_row(
            raw_row, logical_index=logical_index
        )
        if row.response_abi_sha256 != response_abi_sha256:
            raise ValueError(
                f"{panel.panel_role} example row response ABI does not match"
            )
        _validate_pair_quadratic(row.as_pair_stats())
        key = (
            row.family_id,
            row.example_cluster_id,
            row.mode_a_id,
            row.mode_b_id,
        )
        if key in example_rows:
            raise ValueError(
                f"{panel.panel_role} has duplicate example pair statistics"
            )
        example_rows[key] = row
    expected_example_keys = {
        (
            family_id,
            cluster_id,
            logical_axis[left],
            logical_axis[right],
        )
        for family_id in families
        for cluster_id in clusters_by_family[family_id]
        for left in range(len(logical_axis))
        for right in range(left + 1, len(logical_axis))
    }
    if set(example_rows) != expected_example_keys:
        raise ValueError(
            f"{panel.panel_role} example statistics must be the complete "
            "family-example-logical-mode unordered matrix"
        )

    for key, aggregate in pair_rows.items():
        family_id, mode_a_id, mode_b_id = key
        components = [
            example_rows[(family_id, cluster_id, mode_a_id, mode_b_id)]
            for cluster_id in clusters_by_family[family_id]
        ]
        if sum(row.row_count for row in components) != aggregate.row_count:
            raise ValueError(
                f"{panel.panel_role} aggregate row_count is not the example "
                "sum"
            )
        for field in (
            "weighted_sse",
            "weighted_mode_a_energy",
            "weighted_dot",
            "weighted_mode_b_energy",
        ):
            component_sum = math.fsum(
                float(getattr(row, field)) for row in components
            )
            if not _close(
                component_sum,
                float(getattr(aggregate, field)),
                tolerance=_SUFFICIENT_STAT_RELATIVE_TOLERANCE,
            ):
                raise ValueError(
                    f"{panel.panel_role} aggregate {field} is not the example "
                    "sum"
                )

    # A response mode's energy and coordinate row count cannot depend on its
    # comparison partner.  Enforcing this catches coordinated row rewrites
    # that remain locally quadratic but no longer describe one fixed response.
    aggregate_invariants: dict[tuple[str, str], tuple[int, float]] = {}
    example_invariants: dict[
        tuple[str, str, str], tuple[int, float]
    ] = {}
    for row in pair_rows.values():
        for mode_id, energy in (
            (row.mode_a_id, row.weighted_mode_a_energy),
            (row.mode_b_id, row.weighted_mode_b_energy),
        ):
            key = (row.family_id, mode_id)
            current = (row.row_count, float(energy))
            prior = aggregate_invariants.setdefault(key, current)
            if prior[0] != current[0] or not _close(
                prior[1],
                current[1],
                tolerance=_SUFFICIENT_STAT_RELATIVE_TOLERANCE,
            ):
                raise ValueError(
                    f"{panel.panel_role} mode energy/row_count is partner "
                    "dependent"
                )
    for row in example_rows.values():
        for mode_id, energy in (
            (row.mode_a_id, row.weighted_mode_a_energy),
            (row.mode_b_id, row.weighted_mode_b_energy),
        ):
            key = (row.family_id, row.example_cluster_id, mode_id)
            current = (row.row_count, float(energy))
            prior = example_invariants.setdefault(key, current)
            if prior[0] != current[0] or not _close(
                prior[1],
                current[1],
                tolerance=_SUFFICIENT_STAT_RELATIVE_TOLERANCE,
            ):
                raise ValueError(
                    f"{panel.panel_role} example mode energy/row_count is "
                    "partner dependent"
                )

    strata_by_example: dict[tuple[str, str], str] = {}
    for item in panel.example_strata:
        key = (item.family_id, item.example_cluster_id)
        if key in strata_by_example:
            raise ValueError(f"{panel.panel_role} duplicate example stratum")
        strata_by_example[key] = item.stratum_id
    expected_example_catalog = {
        (family_id, cluster_id)
        for family_id in families
        for cluster_id in clusters_by_family[family_id]
    }
    group_ids = [item.group_id for item in panel.stratum_groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError(f"{panel.panel_role} duplicate stratum group ID")
    member_stratum_ids_by_group = {
        item.group_id: item.member_stratum_ids for item in panel.stratum_groups
    }
    if panel.panel_role == "fresh_confirmation":
        required_groups = set(config.hard_veto_stratum_ids) | set(
            config.diagnostic_stratum_ids
        )
        if set(group_ids) != required_groups:
            raise ValueError(
                "fresh stratum group IDs must exactly match the configured "
                "hard-veto and diagnostic groups"
            )
    if panel.stratum_groups or panel.example_strata:
        if set(strata_by_example) != expected_example_catalog:
            raise ValueError(
                f"{panel.panel_role} example strata must label the complete "
                "bootstrap example catalog"
            )
        grouped_member_ids = [
            stratum_id
            for item in panel.stratum_groups
            for stratum_id in item.member_stratum_ids
        ]
        if len(set(grouped_member_ids)) != len(grouped_member_ids):
            raise ValueError(
                f"{panel.panel_role} stratum groups must not overlap"
            )
        if set(grouped_member_ids) != set(strata_by_example.values()):
            raise ValueError(
                f"{panel.panel_role} stratum groups must exactly cover raw "
                "example stratum labels"
            )
        for family_id in families:
            family_labels = {
                strata_by_example[(family_id, cluster_id)]
                for cluster_id in clusters_by_family[family_id]
            }
            for group in panel.stratum_groups:
                if not family_labels.intersection(group.member_stratum_ids):
                    raise ValueError(
                        f"{panel.panel_role} every family must contribute to "
                        f"stratum group {group.group_id}"
                    )
    elif panel.panel_role == "fresh_confirmation" and (
        config.hard_veto_stratum_ids or config.diagnostic_stratum_ids
    ):
        raise ValueError("fresh configured strata require labels and groups")

    _validate_exact_equivalence_closure(
        nodes=nodes,
        families=families,
        clusters_by_family=clusters_by_family,
        pair_rows=pair_rows,
        example_rows=example_rows,
        logical_index=logical_index,
    )

    canonical_panel = StableMutualKNNPanelInput(
        panel_role=panel.panel_role,
        panel_binding_sha256=panel.panel_binding_sha256,
        family_pair_stats=tuple(pair_rows[key] for key in sorted(pair_rows)),
        family_example_pair_stats=tuple(
            example_rows[key] for key in sorted(example_rows)
        ),
        within_family_bootstrap_config=bootstrap,
        example_strata=tuple(
            sorted(
                panel.example_strata,
                key=lambda item: (
                    item.family_id,
                    item.example_cluster_id,
                    item.stratum_id,
                ),
            )
        ),
        stratum_groups=tuple(
            sorted(panel.stratum_groups, key=lambda item: item.group_id)
        ),
    )
    return _ValidatedPanel(
        panel=canonical_panel,
        family_ids=families,
        cluster_ids_by_family=clusters_by_family,
        pair_rows=pair_rows,
        example_rows=example_rows,
        response_abi_sha256=response_abi_sha256,
        stratum_by_example=strata_by_example,
        member_stratum_ids_by_group=member_stratum_ids_by_group,
    )


def _neighbor_topology(
    distance: np.ndarray,
    *,
    node_axis: tuple[str, ...],
    config: StableMutualKNNConfig,
) -> tuple[list[dict[str, object]], set[tuple[str, str]], np.ndarray]:
    size = len(node_axis)
    if distance.shape != (size, size):
        raise ValueError("distance matrix shape does not match node axis")
    if not np.isfinite(distance).all() or (distance < 0.0).any():
        raise ValueError("distance matrix must be finite and nonnegative")
    if config.k >= size:
        raise ValueError("k must be smaller than the node count")
    neighbor_mask = np.zeros((size, size), dtype=bool)
    receipts: list[dict[str, object]] = []
    for node_index, node_id in enumerate(node_axis):
        candidate_indices = np.asarray(
            [index for index in range(size) if index != node_index],
            dtype=np.int64,
        )
        order = np.argsort(
            distance[node_index, candidate_indices], kind="stable"
        )
        ordered = candidate_indices[order]
        boundary_undefined = False
        kth_distance = float(distance[node_index, ordered[config.k - 1]])
        kplus1_distance: float | None = None
        if len(ordered) > config.k:
            kplus1_distance = float(distance[node_index, ordered[config.k]])
            boundary_undefined = (
                abs(kplus1_distance - kth_distance)
                <= config.rank_boundary_tie_absolute_tolerance
            )
        if boundary_undefined:
            neighbor_ids: list[str] = []
            status = "undefined_rank_boundary_tie"
        else:
            selected = ordered[: config.k]
            neighbor_mask[node_index, selected] = True
            neighbor_ids = sorted(node_axis[int(index)] for index in selected)
            status = "defined"
        receipts.append(
            {
                "k_plus_one_distance": kplus1_distance,
                "kth_distance": kth_distance,
                "neighbor_node_ids": neighbor_ids,
                "node_id": node_id,
                "status": status,
            }
        )
    mutual_mask = neighbor_mask & neighbor_mask.T
    edges = {
        _canonical_pair(node_axis[left], node_axis[right])
        for left in range(size)
        for right in range(left + 1, size)
        if bool(mutual_mask[left, right])
    }
    return receipts, edges, mutual_mask


def _batch_mutual_masks(
    distance: np.ndarray,
    *,
    config: StableMutualKNNConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if distance.ndim != 3 or distance.shape[1] != distance.shape[2]:
        raise ValueError("batched distance must have shape [batch,node,node]")
    batch_size, size, _ = distance.shape
    if config.k >= size:
        raise ValueError("k must be smaller than the node count")
    ranked_distance = np.array(distance, copy=True)
    diagonal = np.arange(size)
    ranked_distance[:, diagonal, diagonal] = np.inf
    order = np.argsort(ranked_distance, axis=2, kind="stable")
    defined = np.ones((batch_size, size), dtype=bool)
    if size - 1 > config.k:
        kth = np.take_along_axis(
            ranked_distance, order[:, :, config.k - 1 : config.k], axis=2
        )[:, :, 0]
        kplus1 = np.take_along_axis(
            ranked_distance, order[:, :, config.k : config.k + 1], axis=2
        )[:, :, 0]
        defined &= (
            np.abs(kplus1 - kth)
            > config.rank_boundary_tie_absolute_tolerance
        )
    neighbors = np.zeros((batch_size, size, size), dtype=bool)
    np.put_along_axis(neighbors, order[:, :, : config.k], True, axis=2)
    neighbors &= defined[:, :, None]
    mutual = neighbors & np.swapaxes(neighbors, 1, 2)
    valid = np.isfinite(distance).all(axis=(1, 2)) & (
        distance >= 0.0
    ).all(axis=(1, 2))
    valid &= defined.all(axis=1)
    return neighbors, mutual, valid, defined


def _point_analysis(
    panel: _ValidatedPanel,
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    logical_index: Mapping[str, int],
    config: StableMutualKNNConfig,
    stratum_id: str,
) -> dict[str, object]:
    node_axis = tuple(node.node_id for node in nodes)
    size = len(nodes)
    directed = np.zeros((size, size), dtype=np.float64)
    absolute_rmse = np.zeros((size, size), dtype=np.float64)
    selected_clusters_by_family: dict[str, tuple[str, ...]] = {}
    if stratum_id == _POOLED_STRATUM_ID:
        selected_clusters_by_family = {
            family_id: panel.cluster_ids_by_family[family_id]
            for family_id in panel.family_ids
        }
    else:
        try:
            member_stratum_ids = set(
                panel.member_stratum_ids_by_group[stratum_id]
            )
        except KeyError as error:
            raise ValueError(f"unknown point stratum group {stratum_id}") from error
        for family_id in panel.family_ids:
            selected = tuple(
                cluster_id
                for cluster_id in panel.cluster_ids_by_family[family_id]
                if panel.stratum_by_example[(family_id, cluster_id)]
                in member_stratum_ids
            )
            if not selected:
                raise ValueError(
                    f"stratum {stratum_id} has no examples in {family_id}"
                )
            selected_clusters_by_family[family_id] = selected

    for left_index, left_node in enumerate(nodes):
        left_mode = left_node.member_mode_ids[0]
        for right_index in range(left_index + 1, size):
            right_node = nodes[right_index]
            right_mode = right_node.member_mode_ids[0]
            relative_left_to_right: list[float] = []
            relative_right_to_left: list[float] = []
            squared_rmse_by_family: list[float] = []
            for family_id in panel.family_ids:
                if stratum_id == _POOLED_STRATUM_ID:
                    row = panel.pair_rows[
                        _pair_key(
                            family_id,
                            left_mode,
                            right_mode,
                            logical_index=logical_index,
                        )
                    ]
                    (
                        weighted_sse,
                        left_energy,
                        _,
                        right_energy,
                        row_count,
                    ) = _oriented_pair_values(
                        row, left=left_mode, right=right_mode
                    )
                else:
                    rows = [
                        panel.example_rows[
                            _example_pair_key(
                                family_id,
                                cluster_id,
                                left_mode,
                                right_mode,
                                logical_index=logical_index,
                            )
                        ].as_pair_stats()
                        for cluster_id in selected_clusters_by_family[family_id]
                    ]
                    oriented = [
                        _oriented_pair_values(
                            row, left=left_mode, right=right_mode
                        )
                        for row in rows
                    ]
                    weighted_sse = math.fsum(item[0] for item in oriented)
                    left_energy = math.fsum(item[1] for item in oriented)
                    right_energy = math.fsum(item[3] for item in oriented)
                    row_count = sum(item[4] for item in oriented)
                relative_left_to_right.append(
                    weighted_sse / max(left_energy, config.energy_epsilon)
                )
                relative_right_to_left.append(
                    weighted_sse / max(right_energy, config.energy_epsilon)
                )
                squared_rmse_by_family.append(weighted_sse / row_count)
            directed[left_index, right_index] = float(
                np.mean(relative_left_to_right)
            )
            directed[right_index, left_index] = float(
                np.mean(relative_right_to_left)
            )
            rmse = math.sqrt(float(np.mean(squared_rmse_by_family)))
            absolute_rmse[left_index, right_index] = rmse
            absolute_rmse[right_index, left_index] = rmse
    symmetric = 0.5 * (directed + directed.T)
    np.fill_diagonal(symmetric, 0.0)
    neighbor_receipts, mutual_edges, _ = _neighbor_topology(
        symmetric, node_axis=node_axis, config=config
    )
    member_ids_by_node = {
        node.node_id: node.member_mode_ids for node in nodes
    }
    accidental_equalities = sorted(
        edge
        for edge in (
            _canonical_pair(node_axis[left], node_axis[right])
            for left in range(size)
            for right in range(left + 1, size)
        )
        if (
            directed[
                node_axis.index(edge[0]), node_axis.index(edge[1])
            ]
            == 0.0
            and directed[
                node_axis.index(edge[1]), node_axis.index(edge[0])
            ]
            == 0.0
        )
    )
    return {
        "absolute_weighted_rmse": _matrix_payload(absolute_rmse),
        "accidental_exact_equality_edges": [
            {
                **_edge_payload(edge, member_ids_by_node),
                "diagnostic_only": True,
                "node_universe_changed": False,
            }
            for edge in accidental_equalities
        ],
        "directed_relative_sse": _matrix_payload(directed),
        "equal_family_weighting": True,
        "example_cluster_count_by_family": [
            {
                "example_cluster_count": len(
                    selected_clusters_by_family[family_id]
                ),
                "family_id": family_id,
            }
            for family_id in panel.family_ids
        ],
        "mutual_edges": [
            _edge_payload(edge, member_ids_by_node)
            for edge in sorted(mutual_edges)
        ],
        "neighbor_sets": neighbor_receipts,
        "pooled_sufficient_statistics_before_ratios": True,
        "stratum_id": stratum_id,
        "symmetric_distance": _matrix_payload(symmetric),
    }


def _bootstrap_example_ordinal(
    *,
    bootstrap: WithinFamilyBootstrapConfig,
    resample_index: int,
    family_ordinal: int,
    draw_ordinal: int,
    example_count: int,
) -> int:
    preimage = b"".join(
        (
            bootstrap.draw_protocol_domain,
            bootstrap.seed.to_bytes(8, "big", signed=False),
            resample_index.to_bytes(8, "big", signed=False),
            family_ordinal.to_bytes(4, "big", signed=False),
            draw_ordinal.to_bytes(4, "big", signed=False),
        )
    )
    return int.from_bytes(
        hashlib.sha256(preimage).digest()[:8], "big", signed=False
    ) % example_count


def _bootstrap_analysis(
    panel: _ValidatedPanel,
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    logical_index: Mapping[str, int],
    config: StableMutualKNNConfig,
) -> dict[str, object]:
    bootstrap = panel.panel.within_family_bootstrap_config
    size = len(nodes)
    family_count = len(panel.family_ids)
    mutual_counts = np.zeros((size, size), dtype=np.int64)
    degenerate_count = 0
    bootstrap_payload = bootstrap.canonical_payload()
    catalog_domain_hex = bootstrap_payload[
        "draw_catalog_digest_domain_hex"
    ]
    if not isinstance(catalog_domain_hex, str):
        raise TypeError("bootstrap catalog digest domain must be hex text")
    draw_digest = hashlib.sha256()
    draw_digest.update(bytes.fromhex(catalog_domain_hex))
    draw_digest.update(_canonical_json_bytes(bootstrap_payload))
    topology_digest = hashlib.sha256()
    topology_digest.update(_BOOTSTRAP_TOPOLOGY_DOMAIN)
    topology_digest.update(_canonical_json_bytes(config.canonical_payload()))
    topology_digest.update(panel.panel.panel_binding_sha256.encode("ascii"))

    draw_ordinals = [
        np.empty(
            (bootstrap.resample_count, bootstrap.draws_per_family),
            dtype=np.int32,
        )
        for _ in bootstrap.family_example_cluster_ids
    ]
    # Match the established catalog order exactly: resample, family, draw.
    for resample_index in range(bootstrap.resample_count):
        for family_ordinal, (_, cluster_ids) in enumerate(
            bootstrap.family_example_cluster_ids
        ):
            for draw_ordinal in range(bootstrap.draws_per_family):
                example_ordinal = _bootstrap_example_ordinal(
                    bootstrap=bootstrap,
                    resample_index=resample_index,
                    family_ordinal=family_ordinal,
                    draw_ordinal=draw_ordinal,
                    example_count=len(cluster_ids),
                )
                draw_ordinals[family_ordinal][
                    resample_index, draw_ordinal
                ] = example_ordinal
                draw_digest.update(
                    resample_index.to_bytes(8, "big", signed=False)
                )
                draw_digest.update(
                    family_ordinal.to_bytes(4, "big", signed=False)
                )
                draw_digest.update(
                    draw_ordinal.to_bytes(4, "big", signed=False)
                )
                draw_digest.update(
                    example_ordinal.to_bytes(4, "big", signed=False)
                )

    family_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for family_id, cluster_ids in bootstrap.family_example_cluster_ids:
        cluster_count = len(cluster_ids)
        sse = np.zeros((cluster_count, size, size), dtype=np.float64)
        source_energy = np.ones_like(sse)
        for cluster_index, cluster_id in enumerate(cluster_ids):
            for left_index, left_node in enumerate(nodes):
                left_mode = left_node.member_mode_ids[0]
                for right_index in range(left_index + 1, size):
                    right_mode = nodes[right_index].member_mode_ids[0]
                    row = panel.example_rows[
                        _example_pair_key(
                            family_id,
                            cluster_id,
                            left_mode,
                            right_mode,
                            logical_index=logical_index,
                        )
                    ].as_pair_stats()
                    values = _oriented_pair_values(
                        row, left=left_mode, right=right_mode
                    )
                    sse[cluster_index, left_index, right_index] = values[0]
                    sse[cluster_index, right_index, left_index] = values[0]
                    source_energy[
                        cluster_index, left_index, right_index
                    ] = values[1]
                    source_energy[
                        cluster_index, right_index, left_index
                    ] = values[3]
        family_arrays.append((sse, source_energy))

    chunk_size = 128
    for chunk_start in range(0, bootstrap.resample_count, chunk_size):
        chunk_stop = min(
            bootstrap.resample_count, chunk_start + chunk_size
        )
        chunk_count = chunk_stop - chunk_start
        directed = np.zeros((chunk_count, size, size), dtype=np.float64)
        for family_ordinal, (_, cluster_ids) in enumerate(
            bootstrap.family_example_cluster_ids
        ):
            counts = np.zeros(
                (chunk_count, len(cluster_ids)), dtype=np.float64
            )
            for local_index, resample_index in enumerate(range(chunk_start, chunk_stop)):
                for example_ordinal in draw_ordinals[family_ordinal][
                    resample_index
                ]:
                    counts[local_index, example_ordinal] += 1.0
            family_sse, family_energy = family_arrays[family_ordinal]
            pooled_sse = np.einsum(
                "bc,cij->bij", counts, family_sse, optimize=True
            )
            pooled_energy = np.einsum(
                "bc,cij->bij", counts, family_energy, optimize=True
            )
            directed += pooled_sse / np.maximum(
                pooled_energy, config.energy_epsilon
            )
        directed /= family_count
        symmetric = 0.5 * (directed + np.swapaxes(directed, 1, 2))
        diagonal = np.arange(size)
        symmetric[:, diagonal, diagonal] = 0.0
        neighbors, mutual, valid, defined = _batch_mutual_masks(
            symmetric, config=config
        )
        for local_index, resample_index in enumerate(
            range(chunk_start, chunk_stop)
        ):
            topology_digest.update(
                resample_index.to_bytes(8, "big", signed=False)
            )
            topology_digest.update(b"\x01" if valid[local_index] else b"\x00")
            topology_digest.update(
                np.packbits(defined[local_index]).tobytes()
            )
            if not valid[local_index]:
                degenerate_count += 1
                continue
            topology_digest.update(
                np.packbits(neighbors[local_index].reshape(-1)).tobytes()
            )
            mutual_counts += mutual[local_index].astype(np.int64)

    maximum_degenerate_count = math.floor(
        config.bootstrap_max_degenerate_fraction
        * bootstrap.resample_count
        + 1e-12
    )
    if degenerate_count > maximum_degenerate_count:
        raise ValueError(
            f"{panel.panel.panel_role} bootstrap exceeded maximum degenerate "
            "fraction"
        )
    valid_count = bootstrap.resample_count - degenerate_count
    if valid_count <= 0:
        raise ValueError("bootstrap has no valid resamples")
    frequency = mutual_counts.astype(np.float64) / valid_count
    return {
        "degenerate_fraction": degenerate_count / bootstrap.resample_count,
        "degenerate_resample_count": degenerate_count,
        "draw_catalog_sha256": draw_digest.hexdigest(),
        "frequency_denominator": "valid_resamples_only",
        "mutual_count": [
            [int(value) for value in row] for row in mutual_counts
        ],
        "mutual_frequency": _matrix_payload(frequency),
        "point_error_gates_recomputed_per_resample": False,
        "resample_count": bootstrap.resample_count,
        "topology_catalog_encoding": (
            "resample_u64be||valid_u8||packbits(defined_node_mask)||"
            "packbits(row_major_directed_neighbor_mask_if_valid)"
        ),
        "topology_catalog_sha256": topology_digest.hexdigest(),
        "valid_resample_count": valid_count,
    }


def _lofo_analysis(
    panel: _ValidatedPanel,
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    logical_index: Mapping[str, int],
    config: StableMutualKNNConfig,
) -> dict[str, object]:
    node_axis = tuple(node.node_id for node in nodes)
    member_ids_by_node = {
        node.node_id: node.member_mode_ids for node in nodes
    }
    size = len(nodes)
    mutual_counts = np.zeros((size, size), dtype=np.int64)
    replicates: list[dict[str, object]] = []
    for omitted_family_id in panel.family_ids:
        included = tuple(
            family_id
            for family_id in panel.family_ids
            if family_id != omitted_family_id
        )
        directed = np.zeros((size, size), dtype=np.float64)
        for left_index, left_node in enumerate(nodes):
            left_mode = left_node.member_mode_ids[0]
            for right_index in range(left_index + 1, size):
                right_mode = nodes[right_index].member_mode_ids[0]
                left_values: list[float] = []
                right_values: list[float] = []
                for family_id in included:
                    row = panel.pair_rows[
                        _pair_key(
                            family_id,
                            left_mode,
                            right_mode,
                            logical_index=logical_index,
                        )
                    ]
                    values = _oriented_pair_values(
                        row, left=left_mode, right=right_mode
                    )
                    left_values.append(
                        values[0] / max(values[1], config.energy_epsilon)
                    )
                    right_values.append(
                        values[0] / max(values[3], config.energy_epsilon)
                    )
                directed[left_index, right_index] = float(
                    np.mean(left_values)
                )
                directed[right_index, left_index] = float(
                    np.mean(right_values)
                )
        symmetric = 0.5 * (directed + directed.T)
        neighbor_receipts, mutual_edges, mutual_mask = _neighbor_topology(
            symmetric, node_axis=node_axis, config=config
        )
        mutual_counts += mutual_mask.astype(np.int64)
        replicates.append(
            {
                "mutual_edges": [
                    _edge_payload(edge, member_ids_by_node)
                    for edge in sorted(mutual_edges)
                ],
                "neighbor_sets": neighbor_receipts,
                "omitted_family_id": omitted_family_id,
            }
        )
    return {
        "mutual_count": [
            [int(value) for value in row] for row in mutual_counts
        ],
        "point_error_gates_recomputed_per_replicate": False,
        "replicate_count": len(replicates),
        "replicates": replicates,
    }


def _panel_analysis(
    panel: _ValidatedPanel,
    *,
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    logical_index: Mapping[str, int],
    config: StableMutualKNNConfig,
) -> dict[str, object]:
    stratum_ids = (_POOLED_STRATUM_ID,) + tuple(
        sorted(panel.member_stratum_ids_by_group)
    )
    point_evaluations = [
        _point_analysis(
            panel,
            nodes=nodes,
            logical_index=logical_index,
            config=config,
            stratum_id=stratum_id,
        )
        for stratum_id in stratum_ids
    ]
    return {
        "bootstrap": _bootstrap_analysis(
            panel,
            nodes=nodes,
            logical_index=logical_index,
            config=config,
        ),
        "family_ids": list(panel.family_ids),
        "leave_one_family_out": _lofo_analysis(
            panel,
            nodes=nodes,
            logical_index=logical_index,
            config=config,
        ),
        "panel_binding_sha256": panel.panel.panel_binding_sha256,
        "panel_role": panel.panel.panel_role,
        "point_evaluations": point_evaluations,
        "response_abi_sha256": panel.response_abi_sha256,
    }


def _point_by_stratum(
    panel_analysis: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = panel_analysis["point_evaluations"]
    assert isinstance(raw, list)
    return {
        str(item["stratum_id"]): item
        for item in raw
        if isinstance(item, Mapping)
    }


def _matrix_at(
    point: Mapping[str, object],
    field: str,
    left_index: int,
    right_index: int,
) -> float:
    matrix = point[field]
    assert isinstance(matrix, list)
    row = matrix[left_index]
    assert isinstance(row, list)
    return float(row[right_index])


def _mutual_edge_set(
    point: Mapping[str, object],
) -> set[tuple[str, str]]:
    raw = point["mutual_edges"]
    assert isinstance(raw, list)
    result: set[tuple[str, str]] = set()
    for item in raw:
        assert isinstance(item, Mapping)
        result.add(
            _canonical_pair(
                str(item["left_node_id"]), str(item["right_node_id"])
            )
        )
    return result


def _asymmetry_diagnostic(
    left_to_right: float,
    right_to_left: float,
) -> dict[str, object]:
    minimum = min(left_to_right, right_to_left)
    maximum = max(left_to_right, right_to_left)
    if maximum == 0.0:
        return {"ratio": 1.0, "status": "both_zero"}
    if minimum == 0.0:
        return {"ratio": None, "status": "one_direction_zero"}
    return {"ratio": maximum / minimum, "status": "finite"}


def _components(
    node_axis: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], ...]:
    adjacency = {node_id: set() for node_id in node_axis}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    axis_index = {node_id: index for index, node_id in enumerate(node_axis)}
    seen: set[str] = set()
    result: list[tuple[str, ...]] = []
    for start in node_axis:
        if start in seen:
            continue
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen.update(component)
        result.append(tuple(sorted(component, key=axis_index.__getitem__)))
    return tuple(result)


def _maximal_cliques(
    node_axis: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    *,
    minimum_size: int,
) -> tuple[tuple[str, ...], ...]:
    adjacency = {node_id: set() for node_id in node_axis}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    axis_index = {node_id: index for index, node_id in enumerate(node_axis)}
    cliques: list[tuple[str, ...]] = []

    def visit(r: set[str], p: set[str], x: set[str]) -> None:
        if not p and not x:
            if len(r) >= minimum_size:
                cliques.append(
                    tuple(sorted(r, key=axis_index.__getitem__))
                )
            return
        pivot_candidates = p | x
        pivot = None
        if pivot_candidates:
            pivot = max(
                pivot_candidates,
                key=lambda item: (
                    len(p & adjacency[item]),
                    -axis_index[item],
                ),
            )
        candidates = p - (set() if pivot is None else adjacency[pivot])
        for vertex in sorted(candidates, key=axis_index.__getitem__):
            visit(
                r | {vertex},
                p & adjacency[vertex],
                x & adjacency[vertex],
            )
            p.remove(vertex)
            x.add(vertex)

    visit(set(), set(node_axis), set())
    return tuple(
        sorted(
            set(cliques),
            key=lambda clique: tuple(axis_index[item] for item in clique),
        )
    )


def _group_payload(
    node_ids: tuple[str, ...],
    *,
    member_ids_by_node: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    return {
        "member_mode_ids": [
            mode_id
            for node_id in node_ids
            for mode_id in member_ids_by_node[node_id]
        ],
        "node_ids": list(node_ids),
    }


def _candidate_edge_receipts(
    *,
    candidate_edges: tuple[tuple[str, str], ...],
    nodes: tuple[StableMutualKNNNodeSpec, ...],
    discovery_analysis: Mapping[str, object],
    fresh_analysis: Mapping[str, object],
    config: StableMutualKNNConfig,
) -> tuple[dict[str, object], ...]:
    node_axis = tuple(node.node_id for node in nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_axis)}
    members = {node.node_id: node.member_mode_ids for node in nodes}
    discovery_points = _point_by_stratum(discovery_analysis)
    fresh_points = _point_by_stratum(fresh_analysis)
    discovery_pooled = discovery_points[_POOLED_STRATUM_ID]
    fresh_pooled = fresh_points[_POOLED_STRATUM_ID]
    discovery_mutual = _mutual_edge_set(discovery_pooled)
    fresh_mutual = _mutual_edge_set(fresh_pooled)
    discovery_bootstrap = discovery_analysis["bootstrap"]
    fresh_bootstrap = fresh_analysis["bootstrap"]
    discovery_lofo = discovery_analysis["leave_one_family_out"]
    fresh_lofo = fresh_analysis["leave_one_family_out"]
    assert isinstance(discovery_bootstrap, Mapping)
    assert isinstance(fresh_bootstrap, Mapping)
    assert isinstance(discovery_lofo, Mapping)
    assert isinstance(fresh_lofo, Mapping)

    def integer_matrix_at(
        container: Mapping[str, object],
        field: str,
        left: int,
        right: int,
    ) -> int:
        matrix = container[field]
        assert isinstance(matrix, list)
        row = matrix[left]
        assert isinstance(row, list)
        return int(row[right])

    receipts: list[dict[str, object]] = []
    for edge in candidate_edges:
        node_a_id, node_b_id = edge
        left = node_index[node_a_id]
        right = node_index[node_b_id]
        discovery_a_to_b = _matrix_at(
            discovery_pooled, "directed_relative_sse", left, right
        )
        discovery_b_to_a = _matrix_at(
            discovery_pooled, "directed_relative_sse", right, left
        )
        discovery_rmse = _matrix_at(
            discovery_pooled, "absolute_weighted_rmse", left, right
        )
        discovery_symmetric_distance = _matrix_at(
            discovery_pooled, "symmetric_distance", left, right
        )
        fresh_a_to_b = _matrix_at(
            fresh_pooled, "directed_relative_sse", left, right
        )
        fresh_b_to_a = _matrix_at(
            fresh_pooled, "directed_relative_sse", right, left
        )
        fresh_rmse = _matrix_at(
            fresh_pooled, "absolute_weighted_rmse", left, right
        )
        fresh_symmetric_distance = _matrix_at(
            fresh_pooled, "symmetric_distance", left, right
        )
        pooled_relative_gate = (
            fresh_a_to_b <= config.max_directional_relative_sse
            and fresh_b_to_a <= config.max_directional_relative_sse
        )
        pooled_absolute_gate = (
            fresh_rmse <= config.max_absolute_weighted_rmse
        )
        fresh_bootstrap_count = integer_matrix_at(
            fresh_bootstrap, "mutual_count", left, right
        )
        fresh_bootstrap_valid = int(
            fresh_bootstrap["valid_resample_count"]
        )
        fresh_bootstrap_frequency = (
            fresh_bootstrap_count / fresh_bootstrap_valid
        )
        fresh_bootstrap_gate = (
            fresh_bootstrap_frequency
            >= config.bootstrap_min_mutual_frequency
        )
        fresh_lofo_count = integer_matrix_at(
            fresh_lofo, "mutual_count", left, right
        )
        fresh_lofo_gate = fresh_lofo_count >= config.lofo_min_mutual_count

        discovery_bootstrap_count = integer_matrix_at(
            discovery_bootstrap, "mutual_count", left, right
        )
        discovery_bootstrap_valid = int(
            discovery_bootstrap["valid_resample_count"]
        )
        discovery_lofo_count = integer_matrix_at(
            discovery_lofo, "mutual_count", left, right
        )

        hard_veto_receipts: list[dict[str, object]] = []
        hard_veto_pass = True
        for stratum_id in config.hard_veto_stratum_ids:
            point = fresh_points[stratum_id]
            mutual = edge in _mutual_edge_set(point)
            a_to_b = _matrix_at(
                point, "directed_relative_sse", left, right
            )
            b_to_a = _matrix_at(
                point, "directed_relative_sse", right, left
            )
            rmse = _matrix_at(
                point, "absolute_weighted_rmse", left, right
            )
            symmetric_distance = _matrix_at(
                point, "symmetric_distance", left, right
            )
            relative_gate = (
                a_to_b <= config.max_directional_relative_sse
                and b_to_a <= config.max_directional_relative_sse
            )
            absolute_gate = rmse <= config.max_absolute_weighted_rmse
            passed = mutual and relative_gate and absolute_gate
            hard_veto_pass = hard_veto_pass and passed
            hard_veto_receipts.append(
                {
                    "absolute_gate_pass": absolute_gate,
                    "absolute_weighted_rmse": rmse,
                    "absolute_weighted_rmse_a_to_b": rmse,
                    "absolute_weighted_rmse_b_to_a": rmse,
                    "asymmetry_ratio": _asymmetry_diagnostic(
                        a_to_b, b_to_a
                    ),
                    "directed_relative_sse_a_to_b": a_to_b,
                    "directed_relative_sse_b_to_a": b_to_a,
                    "point_mutual": mutual,
                    "relative_gate_pass": relative_gate,
                    "stratum_id": stratum_id,
                    "symmetric_distance": symmetric_distance,
                    "veto_pass": passed,
                }
            )

        diagnostic_receipts: list[dict[str, object]] = []
        for stratum_id in config.diagnostic_stratum_ids:
            point = fresh_points[stratum_id]
            a_to_b = _matrix_at(
                point, "directed_relative_sse", left, right
            )
            b_to_a = _matrix_at(
                point, "directed_relative_sse", right, left
            )
            diagnostic_rmse = _matrix_at(
                point, "absolute_weighted_rmse", left, right
            )
            diagnostic_symmetric_distance = _matrix_at(
                point, "symmetric_distance", left, right
            )
            diagnostic_receipts.append(
                {
                    "absolute_weighted_rmse": diagnostic_rmse,
                    "absolute_weighted_rmse_a_to_b": diagnostic_rmse,
                    "absolute_weighted_rmse_b_to_a": diagnostic_rmse,
                    "asymmetry_ratio": _asymmetry_diagnostic(
                        a_to_b, b_to_a
                    ),
                    "directed_relative_sse_a_to_b": a_to_b,
                    "directed_relative_sse_b_to_a": b_to_a,
                    "point_mutual": edge in _mutual_edge_set(point),
                    "stratum_id": stratum_id,
                    "symmetric_distance": diagnostic_symmetric_distance,
                }
            )

        confirmed = (
            edge in fresh_mutual
            and pooled_relative_gate
            and pooled_absolute_gate
            and fresh_bootstrap_gate
            and fresh_lofo_gate
            and hard_veto_pass
        )
        payload: dict[str, object] = {
            "bootstrap_gate_pass": fresh_bootstrap_gate,
            "bootstrap_mutual_count": fresh_bootstrap_count,
            "bootstrap_mutual_frequency": fresh_bootstrap_frequency,
            "bootstrap_valid_count": fresh_bootstrap_valid,
            "confirmed": confirmed,
            "diagnostic_strata": diagnostic_receipts,
            "discovery_absolute_weighted_rmse": discovery_rmse,
            "discovery_absolute_weighted_rmse_a_to_b": discovery_rmse,
            "discovery_absolute_weighted_rmse_b_to_a": discovery_rmse,
            "discovery_asymmetry_ratio": _asymmetry_diagnostic(
                discovery_a_to_b, discovery_b_to_a
            ),
            "discovery_bootstrap_mutual_count": (
                discovery_bootstrap_count
            ),
            "discovery_bootstrap_mutual_frequency": (
                discovery_bootstrap_count / discovery_bootstrap_valid
            ),
            "discovery_directed_relative_sse_a_to_b": discovery_a_to_b,
            "discovery_directed_relative_sse_b_to_a": discovery_b_to_a,
            "discovery_lofo_mutual_count": discovery_lofo_count,
            "discovery_point_mutual": edge in discovery_mutual,
            "discovery_symmetric_distance": discovery_symmetric_distance,
            "fresh_hard_veto_strata": hard_veto_receipts,
            "fresh_pooled_absolute_weighted_rmse": fresh_rmse,
            "fresh_pooled_absolute_weighted_rmse_a_to_b": fresh_rmse,
            "fresh_pooled_absolute_weighted_rmse_b_to_a": fresh_rmse,
            "fresh_pooled_asymmetry_ratio": _asymmetry_diagnostic(
                fresh_a_to_b, fresh_b_to_a
            ),
            "fresh_pooled_directed_relative_sse_a_to_b": fresh_a_to_b,
            "fresh_pooled_directed_relative_sse_b_to_a": fresh_b_to_a,
            "fresh_pooled_point_mutual": edge in fresh_mutual,
            "fresh_pooled_symmetric_distance": fresh_symmetric_distance,
            "lofo_gate_pass": fresh_lofo_gate,
            "lofo_mutual_count": fresh_lofo_count,
            "lofo_mutual_frequency": (
                fresh_lofo_count / config.lofo_replicate_count
            ),
            "lofo_replicate_count": config.lofo_replicate_count,
            "node_a_id": node_a_id,
            "node_a_member_mode_ids": list(members[node_a_id]),
            "node_b_id": node_b_id,
            "node_b_member_mode_ids": list(members[node_b_id]),
            "pooled_absolute_gate_pass": pooled_absolute_gate,
            "pooled_relative_gate_pass": pooled_relative_gate,
        }
        long_receipt = next(
            (
                item
                for item in hard_veto_receipts
                if item["stratum_id"] == "long"
            ),
            None,
        )
        if long_receipt is not None:
            payload.update(
                {
                    "fresh_long_absolute_weighted_rmse": (
                        long_receipt["absolute_weighted_rmse"]
                    ),
                    "fresh_long_absolute_weighted_rmse_a_to_b": (
                        long_receipt["absolute_weighted_rmse"]
                    ),
                    "fresh_long_absolute_weighted_rmse_b_to_a": (
                        long_receipt["absolute_weighted_rmse"]
                    ),
                    "fresh_long_asymmetry_ratio": (
                        long_receipt["asymmetry_ratio"]
                    ),
                    "fresh_long_directed_relative_sse_a_to_b": (
                        long_receipt["directed_relative_sse_a_to_b"]
                    ),
                    "fresh_long_directed_relative_sse_b_to_a": (
                        long_receipt["directed_relative_sse_b_to_a"]
                    ),
                    "fresh_long_point_mutual": long_receipt["point_mutual"],
                    "fresh_long_symmetric_distance": (
                        long_receipt["symmetric_distance"]
                    ),
                    "long_absolute_gate_pass": (
                        long_receipt["absolute_gate_pass"]
                    ),
                    "long_relative_gate_pass": (
                        long_receipt["relative_gate_pass"]
                    ),
                }
            )
        receipts.append(payload)
    return tuple(receipts)


def build_stable_mutual_knn_graph_receipt(
    graph_input: StableMutualKNNInput,
) -> StableMutualKNNGraphReceipt:
    """Build one deterministic stable mutual-kNN confirmation receipt."""

    if not isinstance(graph_input, StableMutualKNNInput):
        raise TypeError("graph_input must be a StableMutualKNNInput")
    nodes = tuple(graph_input.node_specs)
    if len(nodes) < 3:
        raise ValueError("the graph requires at least three atomic nodes")
    node_axis = tuple(node.node_id for node in nodes)
    if len(set(node_axis)) != len(node_axis):
        raise ValueError("node IDs must be unique")
    logical_axis = tuple(
        mode_id for node in nodes for mode_id in node.member_mode_ids
    )
    if len(set(logical_axis)) != len(logical_axis):
        raise ValueError("logical member mode IDs must be globally unique")
    if graph_input.config.k >= len(nodes):
        raise ValueError("k must be smaller than the atomic node count")
    candidate_edges = _normalized_candidate_edges(graph_input.candidate_edges)
    node_id_set = set(node_axis)
    for edge in candidate_edges:
        if edge[0] not in node_id_set or edge[1] not in node_id_set:
            raise ValueError("candidate edge references an unknown node")
    expected_candidate_sha256 = stable_mutual_knn_candidate_catalog_sha256(
        nodes, candidate_edges
    )
    if graph_input.candidate_catalog_sha256 != expected_candidate_sha256:
        raise ValueError(
            "candidate_catalog_sha256 does not authenticate candidate edges"
        )

    config = graph_input.config
    discovery = _validate_panel(
        graph_input.discovery,
        nodes=nodes,
        logical_axis=logical_axis,
        config=config,
    )
    fresh = _validate_panel(
        graph_input.fresh_confirmation,
        nodes=nodes,
        logical_axis=logical_axis,
        config=config,
    )
    if discovery.response_abi_sha256 != fresh.response_abi_sha256:
        raise ValueError("discovery and fresh panels must use one response ABI")
    logical_index = {
        mode_id: index for index, mode_id in enumerate(logical_axis)
    }
    canonical_input = StableMutualKNNInput(
        node_specs=nodes,
        candidate_edges=candidate_edges,
        candidate_catalog_sha256=expected_candidate_sha256,
        discovery=discovery.panel,
        fresh_confirmation=fresh.panel,
        config=config,
    )
    discovery_analysis = _panel_analysis(
        discovery,
        nodes=nodes,
        logical_index=logical_index,
        config=config,
    )
    fresh_analysis = _panel_analysis(
        fresh,
        nodes=nodes,
        logical_index=logical_index,
        config=config,
    )
    discovery_pooled_mutual = _mutual_edge_set(
        _point_by_stratum(discovery_analysis)[_POOLED_STRATUM_ID]
    )
    missing_discovery_mutual = set(candidate_edges) - discovery_pooled_mutual
    if missing_discovery_mutual:
        raise ValueError(
            "every preregistered candidate edge must be discovery pooled "
            "mutual-kNN"
        )
    candidate_receipts = _candidate_edge_receipts(
        candidate_edges=candidate_edges,
        nodes=nodes,
        discovery_analysis=discovery_analysis,
        fresh_analysis=fresh_analysis,
        config=config,
    )
    confirmed_edges = tuple(
        (str(item["node_a_id"]), str(item["node_b_id"]))
        for item in candidate_receipts
        if item["confirmed"] is True
    )
    components = _components(node_axis, confirmed_edges)
    nontrivial_components = tuple(
        component for component in components if len(component) >= 2
    )
    singleton_node_ids = tuple(
        component[0] for component in components if len(component) == 1
    )
    cliques = _maximal_cliques(
        node_axis,
        confirmed_edges,
        minimum_size=config.minimum_clique_size,
    )
    members = {node.node_id: node.member_mode_ids for node in nodes}
    fresh_pooled = _point_by_stratum(fresh_analysis)[_POOLED_STRATUM_ID]
    fresh_non_candidate = tuple(
        sorted(_mutual_edge_set(fresh_pooled) - set(candidate_edges))
    )
    claims = {
        "cluster_claimed": False,
        "confirmed_neighborhood_edge_claimed": bool(confirmed_edges),
        "deployment_claimed": False,
        "generator_family_claimed": False,
        "merge_or_compression_claimed": False,
        "stable_neighbor_group_claimed": bool(cliques),
    }
    analysis: dict[str, object] = {
        "candidate_edge_receipts": list(candidate_receipts),
        "claim_flags": claims,
        "confirmed_edges": [
            _edge_payload(edge, members) for edge in confirmed_edges
        ],
        "connected_components": [
            {
                **_group_payload(component, member_ids_by_node=members),
                "topology_diagnostic_only": True,
            }
            for component in nontrivial_components
        ],
        "discovery": discovery_analysis,
        "fresh_confirmation": fresh_analysis,
        "fresh_non_candidate_mutual_edges": [
            {
                **_edge_payload(edge, members),
                "exploratory_only": True,
            }
            for edge in fresh_non_candidate
        ],
        "maximal_cliques": [
            {
                **_group_payload(clique, member_ids_by_node=members),
                "overlap_allowed": True,
                "stable_neighbor_candidate_group": True,
                "transitive_chain_inference": False,
            }
            for clique in cliques
        ],
        "method_contract": {
            "absolute_weighted_rmse_formula": (
                "sqrt(equal_family_mean(weighted_sse_family/"
                "coordinate_row_count_family))"
            ),
            "candidate_edges_frozen_from_discovery": True,
            "discovery_evidence_role": "diagnostic_only",
            "directional_relative_sse_formula": (
                "equal_family_mean(weighted_sse_family/"
                "max(source_energy_family,energy_epsilon))"
            ),
            "fresh_confirmation_only": True,
            "new_exact_equivalence_discovery_on_fresh_panel": False,
            "point_error_gates_recomputed_per_resample": False,
            "pooled_sufficient_statistics_before_ratios": True,
            "rank_boundary_tie_policy": config.rank_boundary_tie_policy,
        },
        "node_axis": list(node_axis),
        "node_specs": [node.canonical_payload() for node in nodes],
        "response_abi_sha256": discovery.response_abi_sha256,
        "singleton_node_ids": list(singleton_node_ids),
    }
    input_payload = {
        "candidate_catalog_sha256": expected_candidate_sha256,
        "config": config.canonical_payload(),
        **_graph_input_payload(canonical_input),
    }
    input_sha256 = _domain_digest(_INPUT_DOMAIN, input_payload)
    provisional = StableMutualKNNGraphReceipt(
        graph_input=canonical_input,
        input_sha256=input_sha256,
        analysis=analysis,
        receipt_sha256="0" * 64,
    )
    receipt_sha256 = _domain_digest(
        _RECEIPT_DOMAIN, provisional._payload_without_receipt_sha256()
    )
    return StableMutualKNNGraphReceipt(
        graph_input=canonical_input,
        input_sha256=input_sha256,
        analysis=analysis,
        receipt_sha256=receipt_sha256,
    )


def _parse_config(value: object) -> StableMutualKNNConfig:
    raw = _require_exact_fields(
        value,
        {
            "bootstrap_max_degenerate_fraction",
            "bootstrap_min_mutual_frequency",
            "bootstrap_resample_count",
            "diagnostic_stratum_ids",
            "energy_epsilon",
            "hard_veto_stratum_ids",
            "k",
            "lofo_min_mutual_count",
            "lofo_replicate_count",
            "max_absolute_weighted_rmse",
            "max_directional_relative_sse",
            "minimum_clique_size",
            "primary_stratum_id",
            "rank_boundary_tie_absolute_tolerance",
            "rank_boundary_tie_policy",
        },
        label="config",
    )
    hard_raw = raw["hard_veto_stratum_ids"]
    diagnostic_raw = raw["diagnostic_stratum_ids"]
    if not isinstance(hard_raw, list):
        raise TypeError("config.hard_veto_stratum_ids must be a list")
    if not isinstance(diagnostic_raw, list):
        raise TypeError("config.diagnostic_stratum_ids must be a list")
    return StableMutualKNNConfig(
        k=_require_positive_int(raw["k"], label="config.k"),
        rank_boundary_tie_absolute_tolerance=_require_finite(
            raw["rank_boundary_tie_absolute_tolerance"],
            label="config.rank_boundary_tie_absolute_tolerance",
            strictly_positive=True,
        ),
        rank_boundary_tie_policy=_require_name(
            raw["rank_boundary_tie_policy"],
            label="config.rank_boundary_tie_policy",
        ),
        energy_epsilon=_require_finite(
            raw["energy_epsilon"],
            label="config.energy_epsilon",
            strictly_positive=True,
        ),
        max_directional_relative_sse=_require_finite(
            raw["max_directional_relative_sse"],
            label="config.max_directional_relative_sse",
            minimum=0.0,
        ),
        max_absolute_weighted_rmse=_require_finite(
            raw["max_absolute_weighted_rmse"],
            label="config.max_absolute_weighted_rmse",
            minimum=0.0,
        ),
        bootstrap_min_mutual_frequency=_require_finite(
            raw["bootstrap_min_mutual_frequency"],
            label="config.bootstrap_min_mutual_frequency",
            minimum=0.0,
            maximum=1.0,
        ),
        lofo_min_mutual_count=_require_positive_int(
            raw["lofo_min_mutual_count"],
            label="config.lofo_min_mutual_count",
        ),
        lofo_replicate_count=_require_positive_int(
            raw["lofo_replicate_count"],
            label="config.lofo_replicate_count",
        ),
        bootstrap_resample_count=_require_positive_int(
            raw["bootstrap_resample_count"],
            label="config.bootstrap_resample_count",
        ),
        bootstrap_max_degenerate_fraction=_require_finite(
            raw["bootstrap_max_degenerate_fraction"],
            label="config.bootstrap_max_degenerate_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_clique_size=_require_positive_int(
            raw["minimum_clique_size"],
            label="config.minimum_clique_size",
        ),
        primary_stratum_id=_require_name(
            raw["primary_stratum_id"],
            label="config.primary_stratum_id",
        ),
        hard_veto_stratum_ids=tuple(
            _require_name(
                item,
                label=f"config.hard_veto_stratum_ids[{index}]",
            )
            for index, item in enumerate(hard_raw)
        ),
        diagnostic_stratum_ids=tuple(
            _require_name(
                item,
                label=f"config.diagnostic_stratum_ids[{index}]",
            )
            for index, item in enumerate(diagnostic_raw)
        ),
    )


def _parse_node(value: object, *, index: int) -> StableMutualKNNNodeSpec:
    label = f"input.node_specs[{index}]"
    raw = _require_exact_fields(
        value, {"member_mode_ids", "node_id"}, label=label
    )
    members_raw = raw["member_mode_ids"]
    if not isinstance(members_raw, list):
        raise TypeError(f"{label}.member_mode_ids must be a list")
    return StableMutualKNNNodeSpec(
        node_id=_require_name(raw["node_id"], label=f"{label}.node_id"),
        member_mode_ids=tuple(
            _require_name(
                item, label=f"{label}.member_mode_ids[{member_index}]"
            )
            for member_index, item in enumerate(members_raw)
        ),
    )


def _parse_pair_row(
    value: object,
    *,
    label: str,
) -> DownstreamResponsePairStats:
    raw = _require_exact_fields(
        value,
        {
            "family_id",
            "mode_a_id",
            "mode_b_id",
            "response_abi_sha256",
            "row_count",
            "weighted_dot",
            "weighted_mode_a_energy",
            "weighted_mode_b_energy",
            "weighted_sse",
        },
        label=label,
    )
    return DownstreamResponsePairStats(
        family_id=_require_name(
            raw["family_id"], label=f"{label}.family_id"
        ),
        mode_a_id=_require_name(
            raw["mode_a_id"], label=f"{label}.mode_a_id"
        ),
        mode_b_id=_require_name(
            raw["mode_b_id"], label=f"{label}.mode_b_id"
        ),
        response_abi_sha256=_require_sha256(
            raw["response_abi_sha256"],
            label=f"{label}.response_abi_sha256",
        ),
        weighted_sse=_require_finite(
            raw["weighted_sse"],
            label=f"{label}.weighted_sse",
            minimum=0.0,
        ),
        weighted_mode_a_energy=_require_finite(
            raw["weighted_mode_a_energy"],
            label=f"{label}.weighted_mode_a_energy",
            minimum=0.0,
        ),
        weighted_dot=_require_finite(
            raw["weighted_dot"], label=f"{label}.weighted_dot"
        ),
        weighted_mode_b_energy=_require_finite(
            raw["weighted_mode_b_energy"],
            label=f"{label}.weighted_mode_b_energy",
            minimum=0.0,
        ),
        row_count=_require_positive_int(
            raw["row_count"], label=f"{label}.row_count"
        ),
    )


def _parse_example_pair_row(
    value: object,
    *,
    label: str,
) -> DownstreamResponseExamplePairStats:
    raw = _require_exact_fields(
        value,
        {
            "example_cluster_id",
            "family_id",
            "mode_a_id",
            "mode_b_id",
            "response_abi_sha256",
            "row_count",
            "weighted_dot",
            "weighted_mode_a_energy",
            "weighted_mode_b_energy",
            "weighted_sse",
        },
        label=label,
    )
    pair = _parse_pair_row(
        {key: item for key, item in raw.items() if key != "example_cluster_id"},
        label=label,
    )
    return DownstreamResponseExamplePairStats(
        family_id=pair.family_id,
        example_cluster_id=_require_name(
            raw["example_cluster_id"],
            label=f"{label}.example_cluster_id",
        ),
        mode_a_id=pair.mode_a_id,
        mode_b_id=pair.mode_b_id,
        response_abi_sha256=pair.response_abi_sha256,
        weighted_sse=pair.weighted_sse,
        weighted_mode_a_energy=pair.weighted_mode_a_energy,
        weighted_dot=pair.weighted_dot,
        weighted_mode_b_energy=pair.weighted_mode_b_energy,
        row_count=pair.row_count,
    )


def _parse_panel(value: object, *, label: str) -> StableMutualKNNPanelInput:
    raw = _require_exact_fields(
        value,
        {
            "example_strata",
            "family_example_pair_stats",
            "family_pair_stats",
            "panel_binding_sha256",
            "panel_role",
            "stratum_groups",
            "within_family_bootstrap_config",
        },
        label=label,
    )
    pairs_raw = raw["family_pair_stats"]
    examples_raw = raw["family_example_pair_stats"]
    strata_raw = raw["example_strata"]
    groups_raw = raw["stratum_groups"]
    if not isinstance(pairs_raw, list):
        raise TypeError(f"{label}.family_pair_stats must be a list")
    if not isinstance(examples_raw, list):
        raise TypeError(f"{label}.family_example_pair_stats must be a list")
    if not isinstance(strata_raw, list):
        raise TypeError(f"{label}.example_strata must be a list")
    if not isinstance(groups_raw, list):
        raise TypeError(f"{label}.stratum_groups must be a list")
    strata: list[StableMutualKNNExampleStratum] = []
    for index, value_item in enumerate(strata_raw):
        item_label = f"{label}.example_strata[{index}]"
        item = _require_exact_fields(
            value_item,
            {"example_cluster_id", "family_id", "stratum_id"},
            label=item_label,
        )
        strata.append(
            StableMutualKNNExampleStratum(
                family_id=_require_name(
                    item["family_id"], label=f"{item_label}.family_id"
                ),
                example_cluster_id=_require_name(
                    item["example_cluster_id"],
                    label=f"{item_label}.example_cluster_id",
                ),
                stratum_id=_require_name(
                    item["stratum_id"], label=f"{item_label}.stratum_id"
                ),
            )
        )
    groups: list[StableMutualKNNStratumGroup] = []
    for index, value_item in enumerate(groups_raw):
        item_label = f"{label}.stratum_groups[{index}]"
        item = _require_exact_fields(
            value_item,
            {"group_id", "member_stratum_ids"},
            label=item_label,
        )
        members_raw = item["member_stratum_ids"]
        if not isinstance(members_raw, list):
            raise TypeError(f"{item_label}.member_stratum_ids must be a list")
        groups.append(
            StableMutualKNNStratumGroup(
                group_id=_require_name(
                    item["group_id"], label=f"{item_label}.group_id"
                ),
                member_stratum_ids=tuple(
                    _require_name(
                        member,
                        label=(
                            f"{item_label}.member_stratum_ids[{member_index}]"
                        ),
                    )
                    for member_index, member in enumerate(members_raw)
                ),
            )
        )
    return StableMutualKNNPanelInput(
        panel_role=_require_name(
            raw["panel_role"], label=f"{label}.panel_role"
        ),
        panel_binding_sha256=_require_sha256(
            raw["panel_binding_sha256"],
            label=f"{label}.panel_binding_sha256",
        ),
        family_pair_stats=tuple(
            _parse_pair_row(
                item, label=f"{label}.family_pair_stats[{index}]"
            )
            for index, item in enumerate(pairs_raw)
        ),
        family_example_pair_stats=tuple(
            _parse_example_pair_row(
                item,
                label=f"{label}.family_example_pair_stats[{index}]",
            )
            for index, item in enumerate(examples_raw)
        ),
        within_family_bootstrap_config=(
            parse_within_family_bootstrap_config_payload(
                raw["within_family_bootstrap_config"]
            )
        ),
        example_strata=tuple(strata),
        stratum_groups=tuple(groups),
    )


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid non-finite JSON constant {value}")


def _rebuild_from_payload(
    raw: Mapping[str, object],
) -> StableMutualKNNGraphReceipt:
    _require_exact_fields(
        raw,
        {
            "analysis",
            "candidate_catalog_sha256",
            "config",
            "format_version",
            "input",
            "input_sha256",
            "receipt_sha256",
            "safety",
            "schema",
            "scientific_status",
        },
        label="receipt",
    )
    if raw["schema"] != STABLE_MUTUAL_KNN_GRAPH_SCHEMA:
        raise ValueError("unsupported stable mutual-kNN graph schema")
    if raw["format_version"] != STABLE_MUTUAL_KNN_GRAPH_FORMAT_VERSION:
        raise ValueError("unsupported stable mutual-kNN graph format version")
    if raw["safety"] != _SAFETY:
        raise ValueError("safety is not the frozen contract")
    if raw["scientific_status"] != _SCIENTIFIC_STATUS:
        raise ValueError("scientific_status is not the frozen contract")
    input_raw = _require_exact_fields(
        raw["input"],
        {
            "candidate_edges",
            "discovery",
            "fresh_confirmation",
            "node_specs",
        },
        label="input",
    )
    node_specs_raw = input_raw["node_specs"]
    edges_raw = input_raw["candidate_edges"]
    if not isinstance(node_specs_raw, list):
        raise TypeError("input.node_specs must be a list")
    if not isinstance(edges_raw, list):
        raise TypeError("input.candidate_edges must be a list")
    edges: list[tuple[str, str]] = []
    for index, item in enumerate(edges_raw):
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(f"input.candidate_edges[{index}] must be a pair")
        edges.append(
            (
                _require_name(
                    item[0], label=f"input.candidate_edges[{index}][0]"
                ),
                _require_name(
                    item[1], label=f"input.candidate_edges[{index}][1]"
                ),
            )
        )
    rebuilt = build_stable_mutual_knn_graph_receipt(
        StableMutualKNNInput(
            node_specs=tuple(
                _parse_node(item, index=index)
                for index, item in enumerate(node_specs_raw)
            ),
            candidate_edges=tuple(edges),
            candidate_catalog_sha256=_require_sha256(
                raw["candidate_catalog_sha256"],
                label="candidate_catalog_sha256",
            ),
            discovery=_parse_panel(
                input_raw["discovery"], label="input.discovery"
            ),
            fresh_confirmation=_parse_panel(
                input_raw["fresh_confirmation"],
                label="input.fresh_confirmation",
            ),
            config=_parse_config(raw["config"]),
        )
    )
    if _canonical_json_bytes(raw) != _canonical_json_bytes(
        rebuilt.canonical_payload()
    ):
        raise ValueError(
            "receipt does not exactly match deterministic reconstruction"
        )
    return rebuilt


def parse_stable_mutual_knn_graph_receipt_payload(
    value: Mapping[str, object],
) -> StableMutualKNNGraphReceipt:
    """Strictly authenticate and reconstruct one in-memory receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("stable mutual-kNN graph payload must be a mapping")
    return _rebuild_from_payload(value)


def save_stable_mutual_knn_graph_receipt(
    path: Path | str,
    receipt: StableMutualKNNGraphReceipt,
) -> None:
    """Exclusively save canonical JSON, then strict-load persisted bytes."""

    if not isinstance(receipt, StableMutualKNNGraphReceipt):
        raise TypeError("receipt must be a StableMutualKNNGraphReceipt")
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("stable mutual-kNN graph receipt must use .json")
    if output.is_symlink() or output.exists():
        raise FileExistsError("refusing to overwrite graph receipt")
    payload = _canonical_json_bytes(receipt.canonical_payload()) + b"\n"
    if not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise ValueError("graph receipt has an invalid byte size")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
            linked = True
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite graph receipt"
            ) from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        persisted = load_stable_mutual_knn_graph_receipt(output)
        if _canonical_json_bytes(persisted.canonical_payload()) != (
            _canonical_json_bytes(receipt.canonical_payload())
        ):
            raise ValueError("persisted graph receipt failed strict reload")
    except BaseException:
        if linked:
            output.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def load_stable_mutual_knn_graph_receipt(
    path: Path | str,
) -> StableMutualKNNGraphReceipt:
    """Strict-load canonical JSON and reconstruct all graph evidence."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("refusing to load graph receipt through a symlink")
    raw_bytes = source.read_bytes()
    if not raw_bytes or len(raw_bytes) > _MAX_RECEIPT_BYTES:
        raise ValueError("graph receipt has an invalid byte size")
    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("graph receipt must be UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("graph receipt must be a JSON object")
    if raw_bytes != _canonical_json_bytes(raw) + b"\n":
        raise ValueError("graph receipt bytes are not canonical JSON")
    return parse_stable_mutual_knn_graph_receipt_payload(raw)
