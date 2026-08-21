"""Model-independent neighbor graphs over generalized chart energy.

This core consumes aggregate chart sufficient statistics.  It never imports a
model runtime and never sees weights, token IDs, prompts, or raw activations.
Every mode is projected onto one frozen comparison surface:
``carrier_chart_energy.h640.v1`` over aligned
``(example_id, prediction_query_position, target_label_position)`` rows with
``target_label_position = prediction_query_position + 1``.

Native ABIs remain authenticated provenance.  They deliberately do not block
cross-kind comparison after a mode has an authenticated adapter onto the
common carrier surface.  Attention additionally carries causal-route receipts
as supplemental evidence; those receipts never replace carrier-energy
distance.  Lexical readout is admissible only through an authenticated
softcap/readout covector pullback into terminal h640 carrier space.

For a directed source-to-target transport, each family contributes
``sum(residual energy) / sum(target Fisher chart energy)``.  The directed
distance is the equal-family mean of those ratios, preventing long families
from dominating solely by row count.  The unordered distance is the arithmetic
mean of both directions.  A global pooled ratio is emitted as a diagnostic
only.  Transport fit and neighbor evaluation are required to use authenticated,
disjoint roles.

Mutual-kNN topology, per-family topology, leave-one-family-out topology, and
leave-one-example-out topology are descriptive stability hooks.  They do not
authorize a mode family, cluster, merge, compression, or deployment claim.
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


__all__ = [
    "CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION",
    "CHART_ENERGY_NEIGHBOR_GRAPH_SCHEMA",
    "CHART_ENERGY_RANK_BOUNDARY_TIE_POLICY",
    "CAUSAL_SEQUENCE_NATIVE_ABI",
    "DIRECT_H640_MUTATION_RESIDUAL_ADAPTER",
    "EMBEDDING_CARRIER_H640_ADAPTER",
    "FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL",
    "LEXICAL_INGRESS_NATIVE_ABI",
    "LEXICAL_READOUT_NATIVE_ABI",
    "SOFTCAP_READOUT_PULLBACK_H640_ADAPTER",
    "TOKEN_LOCAL_NATIVE_ABI",
    "ChartComparisonABI",
    "ChartEnergyNeighborConfig",
    "ChartEnergyDiagnosticSubsetReceipt",
    "ChartEnergyNeighborInput",
    "ChartEnergyNeighborReceipt",
    "ChartExampleStats",
    "ChartModeSpec",
    "ChartWithinFamilyBootstrapConfig",
    "DirectedChartTransportStats",
    "NativeChartABI",
    "build_chart_energy_neighbor_receipt",
    "build_chart_energy_diagnostic_subset_receipt",
    "canonical_carrier_comparison_abi",
    "load_chart_energy_neighbor_receipt",
    "parse_chart_energy_neighbor_input_payload",
    "parse_chart_energy_neighbor_receipt_payload",
    "parse_chart_energy_diagnostic_subset_receipt_payload",
    "save_chart_energy_neighbor_receipt",
]


CHART_ENERGY_NEIGHBOR_GRAPH_SCHEMA: Final = (
    "fisher_graph.chart_energy_neighbor_graph"
)
CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION: Final = 1
CHART_ENERGY_DIAGNOSTIC_SUBSET_SCHEMA: Final = (
    "fisher_graph.chart_energy_neighbor_graph.diagnostic_subset"
)
CHART_ENERGY_RANK_BOUNDARY_TIE_POLICY: Final = (
    "fail_closed_undefined_node_neighbors"
)

TOKEN_LOCAL_NATIVE_ABI: Final = "token_local"
CAUSAL_SEQUENCE_NATIVE_ABI: Final = "causal_sequence"
LEXICAL_INGRESS_NATIVE_ABI: Final = "lexical_ingress"
LEXICAL_READOUT_NATIVE_ABI: Final = "lexical_readout"
DIRECT_H640_MUTATION_RESIDUAL_ADAPTER: Final = (
    "direct_h640_mutation_residual"
)
EMBEDDING_CARRIER_H640_ADAPTER: Final = "embedding_carrier_h640"
SOFTCAP_READOUT_PULLBACK_H640_ADAPTER: Final = (
    "softcap_readout_pullback_covector_h640"
)
FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL: Final = (
    "frozen_A_fit_directed_chart_transport_v1"
)

_NATIVE_ABI_KINDS = frozenset(
    {
        TOKEN_LOCAL_NATIVE_ABI,
        CAUSAL_SEQUENCE_NATIVE_ABI,
        LEXICAL_INGRESS_NATIVE_ABI,
        LEXICAL_READOUT_NATIVE_ABI,
    }
)
_COMPARISON_ABI_NAME = "carrier_chart_energy.h640.v1"
_COMPARISON_CARRIER_WIDTH = 640
_COMPARISON_ROW_SEMANTICS = (
    "aligned_example_prediction_query_and_target_label_positions"
)
_COMPARISON_CONTRIBUTION_SEMANTICS = (
    "value_hadamard_summed_next_token_nll_score_gradient"
)
_COMPARISON_METRIC_SEMANTICS = (
    "shared_membership_spd_tangent_inverse_scale_plus_isotropic_orthogonal"
)
_COMPARISON_ENERGY_SEMANTICS = (
    "membership_weighted_spd_self_cross_mismatch"
)
_ADAPTER_BY_NATIVE_ABI: dict[str, str] = {
    TOKEN_LOCAL_NATIVE_ABI: DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    CAUSAL_SEQUENCE_NATIVE_ABI: DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    LEXICAL_INGRESS_NATIVE_ABI: EMBEDDING_CARRIER_H640_ADAPTER,
    LEXICAL_READOUT_NATIVE_ABI: SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
}
_MODE_KIND_BY_NATIVE_ABI: dict[str, str] = {
    TOKEN_LOCAL_NATIVE_ABI: "mlp",
    CAUSAL_SEQUENCE_NATIVE_ABI: "attention",
    LEXICAL_INGRESS_NATIVE_ABI: "lexical_ingress",
    LEXICAL_READOUT_NATIVE_ABI: "lexical_egress",
}
_ABI_NAME_BY_NATIVE_ABI: dict[str, str] = {
    TOKEN_LOCAL_NATIVE_ABI: "mlp.residual_h640.token_local.v1",
    CAUSAL_SEQUENCE_NATIVE_ABI: (
        "attention.residual_h640.causal_sequence.v1"
    ),
    LEXICAL_INGRESS_NATIVE_ABI: "lexical_ingress.lookup_to_h640.v1",
    LEXICAL_READOUT_NATIVE_ABI: (
        "lexical_egress.tied_vocab_softcap_readout.v1"
    ),
}

_FIT_ROLE = "A_chart_transport_fit"
_EVALUATION_ROLES = frozenset(
    {
        "B_neighbor_discovery",
        "C_neighbor_confirmation",
        "D_response_validation",
    }
)
_TRANSPORT_PROTOCOL = FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL
_INPUT_DOMAIN = b"fisher_graph.chart_energy_neighbor_graph.input.v1\0"
_RECEIPT_DOMAIN = b"fisher_graph.chart_energy_neighbor_graph.receipt.v1\0"
_DIAGNOSTIC_SUBSET_CATALOG_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.diagnostic_subset_catalog.v1\0"
)
_DIAGNOSTIC_SUBSET_RECEIPT_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.diagnostic_subset_receipt.v1\0"
)
_COMPARISON_ABI_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.comparison_abi.v1\0"
)
_NATIVE_ABI_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.native_abi.v1\0"
)
_CHART_STATS_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.chart_stats.v1\0"
)
_TRANSPORT_STATS_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.transport_stats.v1\0"
)
_BOOTSTRAP_TOPOLOGY_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.bootstrap_topology.v1\0"
)
_BOOTSTRAP_DRAW_DOMAIN = (
    b"fisher_graph:chart-energy-neighbor-bootstrap-draw:v1\0"
)
_BOOTSTRAP_CATALOG_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.bootstrap_catalog.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_BYTES = re.compile(r"^(?:[0-9a-f]{2}){1,256}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_MAX_RECEIPT_BYTES = 256 * 1024 * 1024
_RELATIVE_TOLERANCE = 1e-9

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
    "activation_energy_is_diagnostic_only": True,
    "attention_route_distance_is_primary": False,
    "common_carrier_energy_distance_is_primary": True,
    "global_pooled_ratio_is_diagnostic_only": True,
    "native_abi_partitions_are_diagnostic_only": True,
    "stability_topologies_are_diagnostic_hooks": True,
    "point_gated_edges_are_candidate_nominations_only": True,
    "confirmation_core_edges_require_frozen_B_catalog_intersection": True,
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


def _reject_duplicate_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"nonstandard JSON constant {value} is forbidden")


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable nonempty name")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_optional_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label=label)


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
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


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _RELATIVE_TOLERANCE * max(
        1.0, abs(left), abs(right)
    )


@dataclass(frozen=True, slots=True)
class ChartComparisonABI:
    """The exact common chart surface on which graph distance is defined."""

    abi_name: str
    carrier_width: int
    row_semantics: str
    generalized_contribution_semantics: str
    metric_semantics: str
    energy_semantics: str

    def __post_init__(self) -> None:
        _require_name(self.abi_name, label="abi_name")
        _require_positive_int(self.carrier_width, label="carrier_width")
        _require_name(self.row_semantics, label="row_semantics")
        _require_name(
            self.generalized_contribution_semantics,
            label="generalized_contribution_semantics",
        )
        _require_name(self.metric_semantics, label="metric_semantics")
        _require_name(self.energy_semantics, label="energy_semantics")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "abi_name": self.abi_name,
            "carrier_width": self.carrier_width,
            "energy_semantics": self.energy_semantics,
            "generalized_contribution_semantics": (
                self.generalized_contribution_semantics
            ),
            "metric_semantics": self.metric_semantics,
            "row_semantics": self.row_semantics,
        }

    @property
    def sha256(self) -> str:
        return _domain_digest(
            _COMPARISON_ABI_DOMAIN, self.canonical_payload()
        )


def canonical_carrier_comparison_abi() -> ChartComparisonABI:
    """Return the one admissible cross-kind comparison ABI."""

    return ChartComparisonABI(
        abi_name=_COMPARISON_ABI_NAME,
        carrier_width=_COMPARISON_CARRIER_WIDTH,
        row_semantics=_COMPARISON_ROW_SEMANTICS,
        generalized_contribution_semantics=(
            _COMPARISON_CONTRIBUTION_SEMANTICS
        ),
        metric_semantics=_COMPARISON_METRIC_SEMANTICS,
        energy_semantics=_COMPARISON_ENERGY_SEMANTICS,
    )


@dataclass(frozen=True, slots=True)
class NativeChartABI:
    """Authenticated native execution semantics retained as provenance."""

    abi_kind: str
    abi_name: str
    abi_version: int
    input_signature: str
    output_signature: str
    context_signature: str

    def __post_init__(self) -> None:
        if self.abi_kind not in _NATIVE_ABI_KINDS:
            raise ValueError("abi_kind is not a supported native chart ABI")
        _require_name(self.abi_name, label="abi_name")
        if self.abi_name != _ABI_NAME_BY_NATIVE_ABI[self.abi_kind]:
            raise ValueError("abi_name is incompatible with native ABI kind")
        _require_positive_int(self.abi_version, label="abi_version")
        if self.abi_version != 1:
            raise ValueError("native ABI version must be 1")
        _require_name(self.input_signature, label="input_signature")
        _require_name(self.output_signature, label="output_signature")
        _require_name(self.context_signature, label="context_signature")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "abi_kind": self.abi_kind,
            "abi_name": self.abi_name,
            "abi_version": self.abi_version,
            "context_signature": self.context_signature,
            "input_signature": self.input_signature,
            "output_signature": self.output_signature,
        }

    @property
    def sha256(self) -> str:
        return _domain_digest(_NATIVE_ABI_DOMAIN, self.canonical_payload())


@dataclass(frozen=True, slots=True)
class ChartModeSpec:
    """One mode and its authenticated map onto the common chart surface."""

    mode_id: str
    mode_kind: str
    state_fingerprint_sha256: str
    event_id: str
    event_index: int
    context_site_id: str
    contribution_site_id: str
    site_placement_sha256: str
    native_abi: NativeChartABI
    comparison_abi: ChartComparisonABI
    comparison_adapter_kind: str
    comparison_adapter_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_name(self.mode_id, label="mode_id")
        _require_name(self.mode_kind, label="mode_kind")
        _require_sha256(
            self.state_fingerprint_sha256,
            label="state_fingerprint_sha256",
        )
        _require_name(self.event_id, label="event_id")
        _require_nonnegative_int(self.event_index, label="event_index")
        _require_name(self.context_site_id, label="context_site_id")
        _require_name(
            self.contribution_site_id, label="contribution_site_id"
        )
        _require_sha256(
            self.site_placement_sha256, label="site_placement_sha256"
        )
        if not isinstance(self.native_abi, NativeChartABI):
            raise TypeError("native_abi must be a NativeChartABI")
        if not isinstance(self.comparison_abi, ChartComparisonABI):
            raise TypeError("comparison_abi must be a ChartComparisonABI")
        expected_mode_kind = _MODE_KIND_BY_NATIVE_ABI[
            self.native_abi.abi_kind
        ]
        if self.mode_kind != expected_mode_kind:
            raise ValueError("mode_kind is incompatible with native ABI")
        if self.native_abi.abi_kind in {
            TOKEN_LOCAL_NATIVE_ABI,
            CAUSAL_SEQUENCE_NATIVE_ABI,
        } and self.context_site_id == self.contribution_site_id:
            raise ValueError(
                "body modes require distinct context and contribution sites"
            )
        expected_adapter = _ADAPTER_BY_NATIVE_ABI[self.native_abi.abi_kind]
        if self.comparison_adapter_kind != expected_adapter:
            raise ValueError(
                "comparison_adapter_kind is incompatible with native ABI "
                f"{self.native_abi.abi_kind}"
            )
        _require_sha256(
            self.comparison_adapter_receipt_sha256,
            label="comparison_adapter_receipt_sha256",
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "comparison_abi": {
                **self.comparison_abi.canonical_payload(),
                "sha256": self.comparison_abi.sha256,
            },
            "comparison_adapter_kind": self.comparison_adapter_kind,
            "comparison_adapter_receipt_sha256": (
                self.comparison_adapter_receipt_sha256
            ),
            "context_site_id": self.context_site_id,
            "contribution_site_id": self.contribution_site_id,
            "event_id": self.event_id,
            "event_index": self.event_index,
            "mode_id": self.mode_id,
            "mode_kind": self.mode_kind,
            "native_abi": {
                **self.native_abi.canonical_payload(),
                "sha256": self.native_abi.sha256,
            },
            "site_placement_sha256": self.site_placement_sha256,
            "state_fingerprint_sha256": self.state_fingerprint_sha256,
        }


@dataclass(frozen=True, slots=True)
class ChartExampleStats:
    """Per-mode chart sufficient statistics over one aligned example row set."""

    family_id: str
    example_id: str
    mode_id: str
    comparison_abi_sha256: str
    chart_atlas_sha256: str
    aligned_row_catalog_sha256: str
    chart_energy_vector_catalog_sha256: str
    row_count: int
    occupied_row_count: int
    minimum_row_fisher_energy: float
    weighted_activation_energy: float
    weighted_chart_tangent_energy: float
    weighted_chart_orthogonal_energy: float
    weighted_fisher_energy: float
    supplemental_route_catalog_sha256: str | None = None
    supplemental_route_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.example_id, label="example_id")
        _require_name(self.mode_id, label="mode_id")
        _require_sha256(
            self.comparison_abi_sha256,
            label="comparison_abi_sha256",
        )
        _require_sha256(self.chart_atlas_sha256, label="chart_atlas_sha256")
        _require_sha256(
            self.aligned_row_catalog_sha256,
            label="aligned_row_catalog_sha256",
        )
        _require_sha256(
            self.chart_energy_vector_catalog_sha256,
            label="chart_energy_vector_catalog_sha256",
        )
        _require_positive_int(self.row_count, label="row_count")
        _require_nonnegative_int(
            self.occupied_row_count, label="occupied_row_count"
        )
        if self.occupied_row_count > self.row_count:
            raise ValueError("occupied_row_count must not exceed row_count")
        minimum_row_energy = _require_finite(
            self.minimum_row_fisher_energy,
            label="minimum_row_fisher_energy",
            minimum=0.0,
        )
        _require_finite(
            self.weighted_activation_energy,
            label="weighted_activation_energy",
            minimum=0.0,
        )
        tangent = _require_finite(
            self.weighted_chart_tangent_energy,
            label="weighted_chart_tangent_energy",
            minimum=0.0,
        )
        orthogonal = _require_finite(
            self.weighted_chart_orthogonal_energy,
            label="weighted_chart_orthogonal_energy",
            minimum=0.0,
        )
        _require_finite(
            self.weighted_fisher_energy,
            label="weighted_fisher_energy",
            minimum=0.0,
        )
        if not _close(tangent + orthogonal, self.weighted_fisher_energy):
            raise ValueError(
                "tangent plus orthogonal chart energy must equal total Fisher "
                "chart energy"
            )
        if (
            self.weighted_fisher_energy
            + _RELATIVE_TOLERANCE
            < minimum_row_energy * self.row_count
        ):
            raise ValueError(
                "minimum row Fisher energy is inconsistent with total energy"
            )
        _require_optional_sha256(
            self.supplemental_route_catalog_sha256,
            label="supplemental_route_catalog_sha256",
        )
        _require_optional_sha256(
            self.supplemental_route_receipt_sha256,
            label="supplemental_route_receipt_sha256",
        )
        if (self.supplemental_route_catalog_sha256 is None) != (
            self.supplemental_route_receipt_sha256 is None
        ):
            raise ValueError(
                "supplemental route catalog and receipt must appear together"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "aligned_row_catalog_sha256": self.aligned_row_catalog_sha256,
            "chart_atlas_sha256": self.chart_atlas_sha256,
            "chart_energy_vector_catalog_sha256": (
                self.chart_energy_vector_catalog_sha256
            ),
            "comparison_abi_sha256": self.comparison_abi_sha256,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "mode_id": self.mode_id,
            "minimum_row_fisher_energy": float(
                self.minimum_row_fisher_energy
            ),
            "occupied_row_count": self.occupied_row_count,
            "row_count": self.row_count,
            "supplemental_route_catalog_sha256": (
                self.supplemental_route_catalog_sha256
            ),
            "supplemental_route_receipt_sha256": (
                self.supplemental_route_receipt_sha256
            ),
            "weighted_activation_energy": float(
                self.weighted_activation_energy
            ),
            "weighted_chart_orthogonal_energy": float(
                self.weighted_chart_orthogonal_energy
            ),
            "weighted_chart_tangent_energy": float(
                self.weighted_chart_tangent_energy
            ),
            "weighted_fisher_energy": float(self.weighted_fisher_energy),
        }

    @property
    def sha256(self) -> str:
        return _domain_digest(_CHART_STATS_DOMAIN, self.canonical_payload())


@dataclass(frozen=True, slots=True)
class DirectedChartTransportStats:
    """Held-evaluation residual statistics for source-to-target transport."""

    family_id: str
    example_id: str
    source_mode_id: str
    target_mode_id: str
    comparison_abi_sha256: str
    aligned_row_catalog_sha256: str
    source_chart_stats_sha256: str
    target_chart_stats_sha256: str
    row_count: int
    transported_row_count: int
    transport_fit_role: str
    transport_evaluation_role: str
    transport_fit_catalog_sha256: str
    transport_evaluation_catalog_sha256: str
    transport_evaluation_panel_sha256: str
    transport_split_receipt_sha256: str
    transport_fit_receipt_sha256: str
    row_shuffle_catalog_sha256: str
    transport_protocol: str
    weighted_transport_residual_energy: float
    weighted_predicted_fisher_energy: float
    weighted_target_fisher_energy: float
    weighted_predicted_target_cross_inner_product: float
    weighted_zero_prediction_residual_energy: float
    weighted_row_shuffled_transport_residual_energy: float

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.example_id, label="example_id")
        _require_name(self.source_mode_id, label="source_mode_id")
        _require_name(self.target_mode_id, label="target_mode_id")
        if self.source_mode_id == self.target_mode_id:
            raise ValueError("directed transport must compare distinct modes")
        for field_name in (
            "comparison_abi_sha256",
            "aligned_row_catalog_sha256",
            "source_chart_stats_sha256",
            "target_chart_stats_sha256",
            "transport_fit_catalog_sha256",
            "transport_evaluation_catalog_sha256",
            "transport_evaluation_panel_sha256",
            "transport_split_receipt_sha256",
            "transport_fit_receipt_sha256",
            "row_shuffle_catalog_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        _require_positive_int(self.row_count, label="row_count")
        _require_nonnegative_int(
            self.transported_row_count, label="transported_row_count"
        )
        if self.transport_fit_role != _FIT_ROLE:
            raise ValueError(f"transport_fit_role must be {_FIT_ROLE}")
        if self.transport_evaluation_role not in _EVALUATION_ROLES:
            raise ValueError("transport_evaluation_role is unsupported")
        if self.transport_protocol != _TRANSPORT_PROTOCOL:
            raise ValueError(
                "transport_protocol must be " + _TRANSPORT_PROTOCOL
            )
        if (
            self.transport_fit_catalog_sha256
            == self.transport_evaluation_catalog_sha256
        ):
            raise ValueError(
                "transport fit and evaluation catalogs must be distinct"
            )
        residual = _require_finite(
            self.weighted_transport_residual_energy,
            label="weighted_transport_residual_energy",
            minimum=0.0,
        )
        predicted_energy = _require_finite(
            self.weighted_predicted_fisher_energy,
            label="weighted_predicted_fisher_energy",
            minimum=0.0,
        )
        target_energy = _require_finite(
            self.weighted_target_fisher_energy,
            label="weighted_target_fisher_energy",
            minimum=0.0,
        )
        cross = _require_finite(
            self.weighted_predicted_target_cross_inner_product,
            label="weighted_predicted_target_cross_inner_product",
        )
        zero_residual = _require_finite(
            self.weighted_zero_prediction_residual_energy,
            label="weighted_zero_prediction_residual_energy",
            minimum=0.0,
        )
        _require_finite(
            self.weighted_row_shuffled_transport_residual_energy,
            label="weighted_row_shuffled_transport_residual_energy",
            minimum=0.0,
        )
        if not _close(zero_residual, target_energy):
            raise ValueError(
                "zero-prediction residual must equal exact target Fisher energy"
            )
        expected_residual = predicted_energy + target_energy - 2.0 * cross
        if not _close(residual, expected_residual):
            raise ValueError(
                "transport residual is inconsistent with the shared-SPD "
                "energy quadratic"
            )
        scale = max(1.0, predicted_energy, target_energy, abs(cross))
        bound = math.sqrt(predicted_energy * target_energy)
        if abs(cross) > bound + _RELATIVE_TOLERANCE * scale:
            raise ValueError(
                "weighted_predicted_target_cross_inner_product violates the "
                "Cauchy bound"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "aligned_row_catalog_sha256": self.aligned_row_catalog_sha256,
            "comparison_abi_sha256": self.comparison_abi_sha256,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "row_count": self.row_count,
            "row_shuffle_catalog_sha256": self.row_shuffle_catalog_sha256,
            "source_chart_stats_sha256": self.source_chart_stats_sha256,
            "source_mode_id": self.source_mode_id,
            "target_chart_stats_sha256": self.target_chart_stats_sha256,
            "target_mode_id": self.target_mode_id,
            "transport_evaluation_catalog_sha256": (
                self.transport_evaluation_catalog_sha256
            ),
            "transport_evaluation_panel_sha256": (
                self.transport_evaluation_panel_sha256
            ),
            "transport_evaluation_role": self.transport_evaluation_role,
            "transport_fit_catalog_sha256": (
                self.transport_fit_catalog_sha256
            ),
            "transport_fit_receipt_sha256": (
                self.transport_fit_receipt_sha256
            ),
            "transport_fit_role": self.transport_fit_role,
            "transport_protocol": self.transport_protocol,
            "transport_split_receipt_sha256": (
                self.transport_split_receipt_sha256
            ),
            "transported_row_count": self.transported_row_count,
            "weighted_predicted_fisher_energy": float(
                self.weighted_predicted_fisher_energy
            ),
            "weighted_predicted_target_cross_inner_product": float(
                self.weighted_predicted_target_cross_inner_product
            ),
            "weighted_row_shuffled_transport_residual_energy": float(
                self.weighted_row_shuffled_transport_residual_energy
            ),
            "weighted_target_fisher_energy": float(
                self.weighted_target_fisher_energy
            ),
            "weighted_transport_residual_energy": float(
                self.weighted_transport_residual_energy
            ),
            "weighted_zero_prediction_residual_energy": float(
                self.weighted_zero_prediction_residual_energy
            ),
        }

    @property
    def sha256(self) -> str:
        return _domain_digest(
            _TRANSPORT_STATS_DOMAIN,
            self.canonical_payload(),
        )


@dataclass(frozen=True, slots=True)
class ChartWithinFamilyBootstrapConfig:
    """Frozen whole-example bootstrap catalog generated by SHA-256 counters."""

    seed: int
    resample_count: int
    draws_per_family: int
    maximum_degenerate_fraction: float
    family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    draw_protocol_domain: bytes = _BOOTSTRAP_DRAW_DOMAIN

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**64:
            raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
        _require_positive_int(self.resample_count, label="resample_count")
        _require_positive_int(self.draws_per_family, label="draws_per_family")
        _require_finite(
            self.maximum_degenerate_fraction,
            label="maximum_degenerate_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if self.draw_protocol_domain != _BOOTSTRAP_DRAW_DOMAIN:
            raise ValueError("bootstrap draw protocol domain is unsupported")
        if not isinstance(self.family_example_ids, tuple):
            raise TypeError("family_example_ids must be a tuple")
        if not self.family_example_ids:
            raise ValueError("family_example_ids must not be empty")
        seen_families: set[str] = set()
        for family_index, entry in enumerate(self.family_example_ids):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    f"family_example_ids[{family_index}] must be a pair"
                )
            family_id = _require_name(
                entry[0], label=f"family_example_ids[{family_index}].family_id"
            )
            if family_id in seen_families:
                raise ValueError("bootstrap family IDs must be unique")
            seen_families.add(family_id)
            example_ids = entry[1]
            if not isinstance(example_ids, tuple) or not example_ids:
                raise TypeError(
                    "each bootstrap family must contain a nonempty tuple of "
                    "example IDs"
                )
            for example_index, example_id in enumerate(example_ids):
                _require_name(
                    example_id,
                    label=(
                        f"family_example_ids[{family_index}]"
                        f".example_ids[{example_index}]"
                    ),
                )
            if len(set(example_ids)) != len(example_ids):
                raise ValueError(
                    "example IDs must be unique within each bootstrap family"
                )

    @property
    def ordered_family_ids(self) -> tuple[str, ...]:
        return tuple(family_id for family_id, _ in self.family_example_ids)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "draw_catalog_digest_encoding": (
                "sha256(catalog_domain||canonical_config||"
                "repeated(resample_u64be,family_u32be,draw_u32be,"
                "example_u32be))"
            ),
            "draw_catalog_digest_domain_hex": _BOOTSTRAP_CATALOG_DOMAIN.hex(),
            "draw_digest_selection": (
                "first_8_sha256_bytes_u64be_mod_example_count"
            ),
            "draw_preimage_encoding": (
                "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
            ),
            "draw_protocol": "sha256_counter_modulo_example_count_v1",
            "draw_protocol_domain_hex": self.draw_protocol_domain.hex(),
            "draws_per_family": self.draws_per_family,
            "families": [
                {
                    "example_ids": list(example_ids),
                    "family_id": family_id,
                }
                for family_id, example_ids in self.family_example_ids
            ],
            "maximum_degenerate_fraction": float(
                self.maximum_degenerate_fraction
            ),
            "resample_count": self.resample_count,
            "resample_within_each_family_with_replacement": True,
            "sampling_unit": "whole_example_with_all_its_target_token_rows",
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ChartEnergyNeighborConfig:
    """Frozen topology and degeneracy policy; no scientific merge threshold."""

    k: int = 2
    rank_boundary_tie_absolute_tolerance: float = 1e-10
    rank_boundary_tie_policy: str = CHART_ENERGY_RANK_BOUNDARY_TIE_POLICY
    fisher_energy_epsilon: float = 1e-12
    maximum_directional_normalized_residual: float = 1.0
    bootstrap_minimum_mutual_frequency: float = 0.80
    lofo_minimum_mutual_count: int = 5
    lofo_expected_replicate_count: int = 6

    def __post_init__(self) -> None:
        _require_positive_int(self.k, label="k")
        _require_finite(
            self.rank_boundary_tie_absolute_tolerance,
            label="rank_boundary_tie_absolute_tolerance",
            strictly_positive=True,
        )
        if (
            self.rank_boundary_tie_policy
            != CHART_ENERGY_RANK_BOUNDARY_TIE_POLICY
        ):
            raise ValueError("rank_boundary_tie_policy is unsupported")
        _require_finite(
            self.fisher_energy_epsilon,
            label="fisher_energy_epsilon",
            strictly_positive=True,
        )
        _require_finite(
            self.maximum_directional_normalized_residual,
            label="maximum_directional_normalized_residual",
            minimum=0.0,
        )
        _require_finite(
            self.bootstrap_minimum_mutual_frequency,
            label="bootstrap_minimum_mutual_frequency",
            minimum=0.0,
            maximum=1.0,
        )
        _require_positive_int(
            self.lofo_minimum_mutual_count,
            label="lofo_minimum_mutual_count",
        )
        _require_positive_int(
            self.lofo_expected_replicate_count,
            label="lofo_expected_replicate_count",
        )
        if self.lofo_minimum_mutual_count > self.lofo_expected_replicate_count:
            raise ValueError(
                "lofo_minimum_mutual_count must not exceed expected replicates"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bootstrap_minimum_mutual_frequency": float(
                self.bootstrap_minimum_mutual_frequency
            ),
            "fisher_energy_epsilon": float(self.fisher_energy_epsilon),
            "k": self.k,
            "lofo_expected_replicate_count": (
                self.lofo_expected_replicate_count
            ),
            "lofo_minimum_mutual_count": self.lofo_minimum_mutual_count,
            "maximum_directional_normalized_residual": float(
                self.maximum_directional_normalized_residual
            ),
            "rank_boundary_tie_absolute_tolerance": float(
                self.rank_boundary_tie_absolute_tolerance
            ),
            "rank_boundary_tie_policy": self.rank_boundary_tie_policy,
        }


@dataclass(frozen=True, slots=True)
class ChartEnergyNeighborInput:
    """Complete authenticated sufficient-statistic input to the graph core."""

    panel_role: str
    panel_binding_sha256: str
    mode_specs: tuple[ChartModeSpec, ...]
    example_stats: tuple[ChartExampleStats, ...]
    directed_transport_stats: tuple[DirectedChartTransportStats, ...]
    within_family_bootstrap_config: ChartWithinFamilyBootstrapConfig
    config: ChartEnergyNeighborConfig

    def __post_init__(self) -> None:
        if self.panel_role not in _EVALUATION_ROLES:
            raise ValueError("panel_role is unsupported")
        _require_sha256(
            self.panel_binding_sha256, label="panel_binding_sha256"
        )
        if not isinstance(self.mode_specs, tuple):
            raise TypeError("mode_specs must be a tuple")
        if not self.mode_specs or any(
            not isinstance(item, ChartModeSpec) for item in self.mode_specs
        ):
            raise TypeError("mode_specs must contain ChartModeSpec")
        if not isinstance(self.example_stats, tuple):
            raise TypeError("example_stats must be a tuple")
        if not self.example_stats or any(
            not isinstance(item, ChartExampleStats)
            for item in self.example_stats
        ):
            raise TypeError("example_stats must contain ChartExampleStats")
        if not isinstance(self.directed_transport_stats, tuple):
            raise TypeError("directed_transport_stats must be a tuple")
        if not self.directed_transport_stats or any(
            not isinstance(item, DirectedChartTransportStats)
            for item in self.directed_transport_stats
        ):
            raise TypeError(
                "directed_transport_stats must contain "
                "DirectedChartTransportStats"
            )
        if not isinstance(
            self.within_family_bootstrap_config,
            ChartWithinFamilyBootstrapConfig,
        ):
            raise TypeError(
                "within_family_bootstrap_config must be a "
                "ChartWithinFamilyBootstrapConfig"
            )
        if not isinstance(self.config, ChartEnergyNeighborConfig):
            raise TypeError("config must be a ChartEnergyNeighborConfig")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "config": self.config.canonical_payload(),
            "directed_transport_stats": [
                row.canonical_payload()
                for row in self.directed_transport_stats
            ],
            "example_stats": [
                row.canonical_payload() for row in self.example_stats
            ],
            "mode_specs": [
                mode.canonical_payload() for mode in self.mode_specs
            ],
            "panel_binding_sha256": self.panel_binding_sha256,
            "panel_role": self.panel_role,
            "within_family_bootstrap_config": (
                self.within_family_bootstrap_config.canonical_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class ChartEnergyNeighborReceipt:
    """Authenticated input and deterministic chart-neighbor analysis."""

    graph_input: ChartEnergyNeighborInput
    input_sha256: str
    analysis: Mapping[str, object]
    receipt_sha256: str

    def _payload_without_receipt_sha256(self) -> dict[str, object]:
        return {
            "analysis": dict(self.analysis),
            "format_version": CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION,
            "input": self.graph_input.canonical_payload(),
            "input_sha256": self.input_sha256,
            "safety": dict(_SAFETY),
            "schema": CHART_ENERGY_NEIGHBOR_GRAPH_SCHEMA,
            "scientific_status": dict(_SCIENTIFIC_STATUS),
        }

    def canonical_payload(self) -> dict[str, object]:
        payload = self._payload_without_receipt_sha256()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ChartEnergyDiagnosticSubsetReceipt:
    """Non-authorizing point analysis over an exact example subset.

    The source input is supplied separately and authenticated by its canonical
    input digest.  This keeps a parent report from duplicating the complete
    sufficient-statistic matrix for every diagnostic stratum while still
    allowing strict deterministic reconstruction against that source input.
    """

    subset_id: str
    source_input_sha256: str
    source_panel_role: str
    source_panel_binding_sha256: str
    family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    subset_catalog_sha256: str
    analysis: Mapping[str, object]
    receipt_sha256: str

    def _payload_without_receipt_sha256(self) -> dict[str, object]:
        return {
            "analysis": dict(self.analysis),
            "authorizes_edges": False,
            "family_example_ids": [
                {
                    "example_ids": list(example_ids),
                    "family_id": family_id,
                }
                for family_id, example_ids in self.family_example_ids
            ],
            "format_version": CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION,
            "safety": dict(_SAFETY),
            "schema": CHART_ENERGY_DIAGNOSTIC_SUBSET_SCHEMA,
            "source_input_sha256": self.source_input_sha256,
            "source_panel_binding_sha256": (
                self.source_panel_binding_sha256
            ),
            "source_panel_role": self.source_panel_role,
            "subset_catalog_sha256": self.subset_catalog_sha256,
            "subset_id": self.subset_id,
        }

    def canonical_payload(self) -> dict[str, object]:
        payload = self._payload_without_receipt_sha256()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class _ValidatedInput:
    graph_input: ChartEnergyNeighborInput
    mode_axis: tuple[str, ...]
    mode_by_id: Mapping[str, ChartModeSpec]
    family_ids: tuple[str, ...]
    example_ids_by_family: Mapping[str, tuple[str, ...]]
    chart_rows: Mapping[tuple[str, str, str], ChartExampleStats]
    transport_rows: Mapping[
        tuple[str, str, str, str], DirectedChartTransportStats
    ]
    comparison_abi: ChartComparisonABI
    chart_atlas_sha256: str
    row_shuffle_catalog_sha256: str
    transport_fit_catalog_sha256: str
    transport_split_receipt_sha256: str


def _canonicalize_input(
    graph_input: ChartEnergyNeighborInput,
) -> ChartEnergyNeighborInput:
    return ChartEnergyNeighborInput(
        panel_role=graph_input.panel_role,
        panel_binding_sha256=graph_input.panel_binding_sha256,
        mode_specs=tuple(
            sorted(graph_input.mode_specs, key=lambda item: item.mode_id)
        ),
        example_stats=tuple(
            sorted(
                graph_input.example_stats,
                key=lambda row: (row.family_id, row.example_id, row.mode_id),
            )
        ),
        directed_transport_stats=tuple(
            sorted(
                graph_input.directed_transport_stats,
                key=lambda row: (
                    row.family_id,
                    row.example_id,
                    row.source_mode_id,
                    row.target_mode_id,
                ),
            )
        ),
        within_family_bootstrap_config=(
            graph_input.within_family_bootstrap_config
        ),
        config=graph_input.config,
    )


def _validate_input(graph_input: ChartEnergyNeighborInput) -> _ValidatedInput:
    canonical = _canonicalize_input(graph_input)
    mode_axis = tuple(mode.mode_id for mode in canonical.mode_specs)
    if len(set(mode_axis)) != len(mode_axis):
        raise ValueError("mode_specs must have unique mode_id values")
    event_ids = tuple(mode.event_id for mode in canonical.mode_specs)
    event_indices = tuple(mode.event_index for mode in canonical.mode_specs)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("mode_specs must have unique event_id values")
    if len(set(event_indices)) != len(event_indices):
        raise ValueError("mode_specs must have unique event_index values")
    if canonical.config.k >= len(mode_axis):
        raise ValueError("k must be smaller than the mode count")
    mode_by_id = {mode.mode_id: mode for mode in canonical.mode_specs}

    expected_comparison = canonical_carrier_comparison_abi()
    for mode in canonical.mode_specs:
        if mode.comparison_abi != expected_comparison:
            raise ValueError(
                "every mode must use exact carrier_chart_energy.h640.v1 ABI"
            )

    chart_rows: dict[tuple[str, str, str], ChartExampleStats] = {}
    sample_keys: set[tuple[str, str]] = set()
    chart_atlas_sha256s: set[str] = set()
    for row in canonical.example_stats:
        if row.mode_id not in mode_by_id:
            raise ValueError("example stats reference an unknown mode")
        key = (row.family_id, row.example_id, row.mode_id)
        if key in chart_rows:
            raise ValueError("duplicate per-example chart statistics")
        mode = mode_by_id[row.mode_id]
        if row.comparison_abi_sha256 != expected_comparison.sha256:
            raise ValueError("chart stats use an incompatible comparison ABI")
        route_present = row.supplemental_route_receipt_sha256 is not None
        route_required = (
            mode.native_abi.abi_kind == CAUSAL_SEQUENCE_NATIVE_ABI
        )
        if route_present != route_required:
            raise ValueError(
                "causal-sequence modes require route catalog and receipt; "
                "other native ABIs must not claim them"
            )
        chart_rows[key] = row
        sample_keys.add((row.family_id, row.example_id))
        chart_atlas_sha256s.add(row.chart_atlas_sha256)

    expected_chart_keys = {
        (family_id, example_id, mode_id)
        for family_id, example_id in sample_keys
        for mode_id in mode_axis
    }
    if set(chart_rows) != expected_chart_keys:
        raise ValueError(
            "per-example chart statistics must form a complete mode matrix"
        )
    if not sample_keys:
        raise ValueError("the chart panel must contain examples")
    if len(chart_atlas_sha256s) != 1:
        raise ValueError("all modes and examples must use one shared chart atlas")

    raw_example_ids_by_family = {
        family_id: tuple(
            sorted(
                example_id
                for row_family, example_id in sample_keys
                if row_family == family_id
            )
        )
        for family_id in sorted({key[0] for key in sample_keys})
    }
    bootstrap = canonical.within_family_bootstrap_config
    bootstrap_example_ids_by_family = dict(
        bootstrap.family_example_ids
    )
    if set(bootstrap_example_ids_by_family) != set(
        raw_example_ids_by_family
    ):
        raise ValueError(
            "bootstrap family catalog must exactly match chart families"
        )
    family_ids = bootstrap.ordered_family_ids
    example_ids_by_family = {
        family_id: bootstrap_example_ids_by_family[family_id]
        for family_id in family_ids
    }
    if len(family_ids) != canonical.config.lofo_expected_replicate_count:
        raise ValueError(
            "family count must equal lofo_expected_replicate_count"
        )
    for family_id, example_ids in example_ids_by_family.items():
        if set(example_ids) != set(raw_example_ids_by_family[family_id]):
            raise ValueError(
                "bootstrap example catalog must exactly match chart examples"
            )
        if bootstrap.draws_per_family != len(example_ids):
            raise ValueError(
                "bootstrap draws_per_family must equal examples per family"
            )
        if not example_ids:
            raise ValueError(f"family {family_id} has no examples")
        for example_id in example_ids:
            rows = [
                chart_rows[(family_id, example_id, mode_id)]
                for mode_id in mode_axis
            ]
            aligned = {
                (row.aligned_row_catalog_sha256, row.row_count)
                for row in rows
            }
            if len(aligned) != 1:
                raise ValueError(
                    "all modes must use exactly aligned token rows for each "
                    "family/example"
                )

    transport_rows: dict[
        tuple[str, str, str, str], DirectedChartTransportStats
    ] = {}
    fit_catalogs: set[str] = set()
    split_receipts: set[str] = set()
    evaluation_catalogs: set[str] = set()
    shuffle_catalogs: set[str] = set()
    real_fit_receipts_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in canonical.directed_transport_stats:
        if (
            row.source_mode_id not in mode_by_id
            or row.target_mode_id not in mode_by_id
        ):
            raise ValueError("transport stats reference an unknown mode")
        if (row.family_id, row.example_id) not in sample_keys:
            raise ValueError("transport stats reference an unknown example")
        key = (
            row.family_id,
            row.example_id,
            row.source_mode_id,
            row.target_mode_id,
        )
        if key in transport_rows:
            raise ValueError("duplicate directed transport statistics")
        source = chart_rows[
            (row.family_id, row.example_id, row.source_mode_id)
        ]
        target = chart_rows[
            (row.family_id, row.example_id, row.target_mode_id)
        ]
        if row.comparison_abi_sha256 != expected_comparison.sha256:
            raise ValueError("transport stats use an incompatible comparison ABI")
        if (
            row.transport_evaluation_panel_sha256
            != canonical.panel_binding_sha256
        ):
            raise ValueError(
                "transport evaluation panel binding does not match graph panel"
            )
        if row.transport_evaluation_role != canonical.panel_role:
            raise ValueError(
                "transport evaluation role does not match graph panel role"
            )
        if (
            row.aligned_row_catalog_sha256
            != source.aligned_row_catalog_sha256
            or row.aligned_row_catalog_sha256
            != target.aligned_row_catalog_sha256
            or row.transport_evaluation_catalog_sha256
            != row.aligned_row_catalog_sha256
        ):
            raise ValueError(
                "transport evaluation must use the exact aligned chart rows"
            )
        if row.source_chart_stats_sha256 != source.sha256:
            raise ValueError("transport source chart-stat binding is invalid")
        if row.target_chart_stats_sha256 != target.sha256:
            raise ValueError("transport target chart-stat binding is invalid")
        if row.row_count != source.row_count or row.row_count != target.row_count:
            raise ValueError("transport row count does not match chart rows")
        if row.transported_row_count != row.row_count:
            raise ValueError("transport must cover every aligned evaluation row")
        if not _close(
            row.weighted_target_fisher_energy,
            target.weighted_fisher_energy,
        ):
            raise ValueError(
                "transport target Fisher energy does not match target chart "
                "statistics"
            )
        transport_rows[key] = row
        fit_catalogs.add(row.transport_fit_catalog_sha256)
        split_receipts.add(row.transport_split_receipt_sha256)
        evaluation_catalogs.add(row.transport_evaluation_catalog_sha256)
        shuffle_catalogs.add(row.row_shuffle_catalog_sha256)
        pair = (row.source_mode_id, row.target_mode_id)
        real_fit_receipts_by_pair.setdefault(pair, set()).add(
            row.transport_fit_receipt_sha256
        )

    expected_transport_keys = {
        (family_id, example_id, source, target)
        for family_id, example_id in sample_keys
        for source in mode_axis
        for target in mode_axis
        if source != target
    }
    if set(transport_rows) != expected_transport_keys:
        raise ValueError(
            "directed transport statistics must form a complete ordered "
            "mode matrix for every example"
        )
    if len(fit_catalogs) != 1:
        raise ValueError("all transports must use one frozen fit catalog")
    if len(split_receipts) != 1:
        raise ValueError("all transports must use one frozen split receipt")
    if len(shuffle_catalogs) != 1:
        raise ValueError(
            "all transports must use one frozen within-family row-shuffle "
            "permutation catalog"
        )
    fit_catalog = next(iter(fit_catalogs))
    if fit_catalog == canonical.panel_binding_sha256:
        raise ValueError(
            "transport fit catalog must differ from evaluation panel binding"
        )
    if fit_catalog in evaluation_catalogs:
        raise ValueError(
            "transport fit catalog must be disjoint from every evaluation "
            "row catalog"
        )

    expected_pairs = {
        (source, target)
        for source in mode_axis
        for target in mode_axis
        if source != target
    }
    if set(real_fit_receipts_by_pair) != expected_pairs:
        raise ValueError("directed fit-receipt catalog is incomplete")
    if any(len(values) != 1 for values in real_fit_receipts_by_pair.values()):
        raise ValueError(
            "each directed pair must use one frozen real transport fit receipt"
        )
    real_receipts = {
        next(iter(values)) for values in real_fit_receipts_by_pair.values()
    }
    if len(real_receipts) != len(expected_pairs):
        raise ValueError("real transport fit receipts must be pair-specific")

    # Every family denominator must be nondegenerate for every directed target.
    for family_id in family_ids:
        for target in mode_axis:
            energy = math.fsum(
                chart_rows[(family_id, example_id, target)].weighted_fisher_energy
                for example_id in example_ids_by_family[family_id]
            )
            if (
                not math.isfinite(energy)
                or energy <= canonical.config.fisher_energy_epsilon
            ):
                raise ValueError(
                    "every family/mode Fisher chart-energy denominator must "
                    "be nondegenerate"
                )

    return _ValidatedInput(
        graph_input=canonical,
        mode_axis=mode_axis,
        mode_by_id=mode_by_id,
        family_ids=family_ids,
        example_ids_by_family=example_ids_by_family,
        chart_rows=chart_rows,
        transport_rows=transport_rows,
        comparison_abi=expected_comparison,
        chart_atlas_sha256=next(iter(chart_atlas_sha256s)),
        row_shuffle_catalog_sha256=next(iter(shuffle_catalogs)),
        transport_fit_catalog_sha256=fit_catalog,
        transport_split_receipt_sha256=next(iter(split_receipts)),
    )


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("an edge must join distinct modes")
    return (left, right) if left < right else (right, left)


def _matrix_payload(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _selected_examples(
    validated: _ValidatedInput,
    *,
    include_families: Sequence[str] | None = None,
    excluded_example: tuple[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    families = (
        tuple(include_families)
        if include_families is not None
        else validated.family_ids
    )
    result: dict[str, tuple[str, ...]] = {}
    for family_id in families:
        examples = tuple(
            example_id
            for example_id in validated.example_ids_by_family[family_id]
            if excluded_example != (family_id, example_id)
        )
        if not examples:
            raise ValueError("a stability replicate removed an entire family")
        result[family_id] = examples
    return result


def _distance_analysis(
    validated: _ValidatedInput,
    selected: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    axis = validated.mode_axis
    size = len(axis)
    directed = [[0.0] * size for _ in range(size)]
    symmetric = [[0.0] * size for _ in range(size)]
    global_ratio = [[0.0] * size for _ in range(size)]
    zero_control = [[0.0] * size for _ in range(size)]
    row_shuffle_control = [[0.0] * size for _ in range(size)]
    real_gain_over_zero = [[0.0] * size for _ in range(size)]
    row_shuffle_gain_over_zero = [[0.0] * size for _ in range(size)]
    real_gain_over_row_shuffle: list[list[float | None]] = [
        [None] * size for _ in range(size)
    ]
    per_family: dict[str, list[list[float]]] = {
        family_id: [[0.0] * size for _ in range(size)]
        for family_id in selected
    }
    per_family_zero: dict[str, list[list[float]]] = {
        family_id: [[0.0] * size for _ in range(size)]
        for family_id in selected
    }
    per_family_shuffle: dict[str, list[list[float]]] = {
        family_id: [[0.0] * size for _ in range(size)]
        for family_id in selected
    }

    for source_index, source in enumerate(axis):
        for target_index, target in enumerate(axis):
            if source == target:
                continue
            ratios: list[float] = []
            zero_ratios: list[float] = []
            shuffle_ratios: list[float] = []
            total_residual = 0.0
            total_energy = 0.0
            for family_id, example_ids in selected.items():
                residual = math.fsum(
                    validated.transport_rows[
                        (family_id, example_id, source, target)
                    ].weighted_transport_residual_energy
                    for example_id in example_ids
                )
                target_energy = math.fsum(
                    validated.transport_rows[
                        (family_id, example_id, source, target)
                    ].weighted_target_fisher_energy
                    for example_id in example_ids
                )
                zero_residual = math.fsum(
                    validated.transport_rows[
                        (family_id, example_id, source, target)
                    ].weighted_zero_prediction_residual_energy
                    for example_id in example_ids
                )
                shuffle_residual = math.fsum(
                    validated.transport_rows[
                        (family_id, example_id, source, target)
                    ].weighted_row_shuffled_transport_residual_energy
                    for example_id in example_ids
                )
                if (
                    not math.isfinite(target_energy)
                    or target_energy
                    <= validated.graph_input.config.fisher_energy_epsilon
                ):
                    raise ValueError(
                        "selected family has a degenerate target Fisher-energy "
                        "denominator"
                    )
                ratio = residual / target_energy
                zero_ratio = zero_residual / target_energy
                shuffle_ratio = shuffle_residual / target_energy
                if not math.isfinite(ratio) or ratio < 0.0:
                    raise ValueError("directed chart distance is invalid")
                if (
                    not math.isfinite(zero_ratio)
                    or zero_ratio <= 0.0
                    or not math.isfinite(shuffle_ratio)
                    or shuffle_ratio < 0.0
                ):
                    raise ValueError("directed control distance is invalid")
                per_family[family_id][source_index][target_index] = ratio
                per_family_zero[family_id][source_index][
                    target_index
                ] = zero_ratio
                per_family_shuffle[family_id][source_index][
                    target_index
                ] = shuffle_ratio
                ratios.append(ratio)
                zero_ratios.append(zero_ratio)
                shuffle_ratios.append(shuffle_ratio)
                total_residual += residual
                total_energy += target_energy
            directed[source_index][target_index] = math.fsum(ratios) / len(ratios)
            zero_control[source_index][target_index] = (
                math.fsum(zero_ratios) / len(zero_ratios)
            )
            row_shuffle_control[source_index][target_index] = (
                math.fsum(shuffle_ratios) / len(shuffle_ratios)
            )
            real_gain_over_zero[source_index][target_index] = 1.0 - (
                directed[source_index][target_index]
                / zero_control[source_index][target_index]
            )
            row_shuffle_gain_over_zero[source_index][target_index] = 1.0 - (
                row_shuffle_control[source_index][target_index]
                / zero_control[source_index][target_index]
            )
            if (
                row_shuffle_control[source_index][target_index]
                > validated.graph_input.config.fisher_energy_epsilon
            ):
                real_gain_over_row_shuffle[source_index][target_index] = (
                    1.0
                    - directed[source_index][target_index]
                    / row_shuffle_control[source_index][target_index]
                )
            global_ratio[source_index][target_index] = (
                total_residual / total_energy
            )

    for left in range(size):
        for right in range(left + 1, size):
            value = 0.5 * (
                directed[left][right] + directed[right][left]
            )
            symmetric[left][right] = value
            symmetric[right][left] = value

    return {
        "directional_equal_family_distance_matrix": _matrix_payload(directed),
        "directional_global_pooled_ratio_matrix_diagnostic": _matrix_payload(
            global_ratio
        ),
        "directional_per_family_distance_matrices": {
            family_id: _matrix_payload(matrix)
            for family_id, matrix in sorted(per_family.items())
        },
        "directional_per_family_row_shuffle_control_ratio_matrices": {
            family_id: _matrix_payload(matrix)
            for family_id, matrix in sorted(per_family_shuffle.items())
        },
        "directional_per_family_zero_prediction_control_ratio_matrices": {
            family_id: _matrix_payload(matrix)
            for family_id, matrix in sorted(per_family_zero.items())
        },
        "directional_real_gain_over_row_shuffle_matrix": (
            real_gain_over_row_shuffle
        ),
        "directional_real_gain_over_zero_matrix": _matrix_payload(
            real_gain_over_zero
        ),
        "directional_row_shuffle_control_ratio_matrix": _matrix_payload(
            row_shuffle_control
        ),
        "directional_row_shuffle_gain_over_zero_matrix": _matrix_payload(
            row_shuffle_gain_over_zero
        ),
        "directional_zero_prediction_control_ratio_matrix": _matrix_payload(
            zero_control
        ),
        "symmetric_distance_matrix": _matrix_payload(symmetric),
    }


def _neighbor_topology(
    distance: Sequence[Sequence[float]],
    *,
    mode_axis: tuple[str, ...],
    config: ChartEnergyNeighborConfig,
) -> dict[str, object]:
    size = len(mode_axis)
    if len(distance) != size or any(len(row) != size for row in distance):
        raise ValueError("distance matrix shape does not match mode axis")
    for row in distance:
        for value in row:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("distance matrix must be finite and nonnegative")

    neighbor_sets: dict[str, set[str]] = {}
    receipts: list[dict[str, object]] = []
    for mode_index, mode_id in enumerate(mode_axis):
        ordered = sorted(
            (
                (float(distance[mode_index][other_index]), other_index)
                for other_index in range(size)
                if other_index != mode_index
            ),
            key=lambda item: (item[0], item[1]),
        )
        kth_distance = ordered[config.k - 1][0]
        k_plus_one_distance = (
            ordered[config.k][0] if len(ordered) > config.k else None
        )
        undefined = (
            k_plus_one_distance is not None
            and abs(k_plus_one_distance - kth_distance)
            <= config.rank_boundary_tie_absolute_tolerance
        )
        if undefined:
            neighbors: set[str] = set()
            status = "undefined_rank_boundary_tie"
        else:
            neighbors = {
                mode_axis[index] for _, index in ordered[: config.k]
            }
            status = "defined"
        neighbor_sets[mode_id] = neighbors
        receipts.append(
            {
                "k_plus_one_distance": k_plus_one_distance,
                "kth_distance": kth_distance,
                "mode_id": mode_id,
                "neighbor_mode_ids": sorted(neighbors),
                "status": status,
            }
        )

    mutual_edges = sorted(
        _canonical_pair(left, right)
        for left_index, left in enumerate(mode_axis)
        for right in mode_axis[left_index + 1 :]
        if right in neighbor_sets[left] and left in neighbor_sets[right]
    )
    return {
        "mutual_edges": [list(edge) for edge in mutual_edges],
        "neighbor_receipts": receipts,
    }


def _bootstrap_example_ordinal(
    *,
    bootstrap: ChartWithinFamilyBootstrapConfig,
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


def _bootstrap_analysis(validated: _ValidatedInput) -> dict[str, object]:
    """Recompute mutual-kNN on deterministic whole-example resamples."""

    bootstrap = validated.graph_input.within_family_bootstrap_config
    config = validated.graph_input.config
    axis = validated.mode_axis
    size = len(axis)
    family_count = len(validated.family_ids)
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
    topology_digest.update(
        _canonical_json_bytes(validated.graph_input.config.canonical_payload())
    )
    topology_digest.update(
        validated.graph_input.panel_binding_sha256.encode("ascii")
    )

    draw_ordinals = [
        np.empty(
            (bootstrap.resample_count, bootstrap.draws_per_family),
            dtype=np.int32,
        )
        for _ in validated.family_ids
    ]
    for resample_index in range(bootstrap.resample_count):
        for family_ordinal, family_id in enumerate(validated.family_ids):
            examples = validated.example_ids_by_family[family_id]
            for draw_ordinal in range(bootstrap.draws_per_family):
                example_ordinal = _bootstrap_example_ordinal(
                    bootstrap=bootstrap,
                    resample_index=resample_index,
                    family_ordinal=family_ordinal,
                    draw_ordinal=draw_ordinal,
                    example_count=len(examples),
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
    for family_id in validated.family_ids:
        examples = validated.example_ids_by_family[family_id]
        residual = np.zeros((len(examples), size, size), dtype=np.float64)
        target_energy = np.ones_like(residual)
        for example_ordinal, example_id in enumerate(examples):
            for source_index, source in enumerate(axis):
                for target_index, target in enumerate(axis):
                    if source == target:
                        continue
                    row = validated.transport_rows[
                        (family_id, example_id, source, target)
                    ]
                    residual[example_ordinal, source_index, target_index] = (
                        row.weighted_transport_residual_energy
                    )
                    target_energy[
                        example_ordinal, source_index, target_index
                    ] = row.weighted_target_fisher_energy
        family_arrays.append((residual, target_energy))

    mutual_count = np.zeros((size, size), dtype=np.int64)
    degenerate_count = 0
    chunk_size = 128
    diagonal = np.arange(size)
    for chunk_start in range(0, bootstrap.resample_count, chunk_size):
        chunk_stop = min(
            bootstrap.resample_count, chunk_start + chunk_size
        )
        chunk_count = chunk_stop - chunk_start
        directed = np.zeros((chunk_count, size, size), dtype=np.float64)
        valid = np.ones(chunk_count, dtype=bool)
        for family_ordinal, (family_residual, family_energy) in enumerate(
            family_arrays
        ):
            example_count = family_residual.shape[0]
            counts = np.zeros(
                (chunk_count, example_count), dtype=np.float64
            )
            for local_index, resample_index in enumerate(
                range(chunk_start, chunk_stop)
            ):
                for example_ordinal in draw_ordinals[family_ordinal][
                    resample_index
                ]:
                    counts[local_index, int(example_ordinal)] += 1.0
            pooled_residual = np.einsum(
                "bc,cij->bij", counts, family_residual, optimize=True
            )
            pooled_energy = np.einsum(
                "bc,cij->bij", counts, family_energy, optimize=True
            )
            off_diagonal = np.ones((size, size), dtype=bool)
            off_diagonal[diagonal, diagonal] = False
            valid &= (
                np.isfinite(pooled_energy[:, off_diagonal]).all(axis=1)
                & (
                    pooled_energy[:, off_diagonal]
                    > config.fisher_energy_epsilon
                ).all(axis=1)
            )
            directed += np.divide(
                pooled_residual,
                pooled_energy,
                out=np.zeros_like(pooled_residual),
                where=pooled_energy > config.fisher_energy_epsilon,
            )
        directed /= family_count
        symmetric = 0.5 * (directed + np.swapaxes(directed, 1, 2))
        symmetric[:, diagonal, diagonal] = 0.0
        valid &= np.isfinite(symmetric).all(axis=(1, 2))
        valid &= (symmetric >= 0.0).all(axis=(1, 2))

        ranked = np.array(symmetric, copy=True)
        ranked[:, diagonal, diagonal] = np.inf
        order = np.argsort(ranked, axis=2, kind="stable")
        defined = np.ones((chunk_count, size), dtype=bool)
        if size - 1 > config.k:
            kth = np.take_along_axis(
                ranked, order[:, :, config.k - 1 : config.k], axis=2
            )[:, :, 0]
            k_plus_one = np.take_along_axis(
                ranked, order[:, :, config.k : config.k + 1], axis=2
            )[:, :, 0]
            defined &= (
                np.abs(k_plus_one - kth)
                > config.rank_boundary_tie_absolute_tolerance
            )
        valid &= defined.all(axis=1)
        neighbors = np.zeros((chunk_count, size, size), dtype=bool)
        np.put_along_axis(neighbors, order[:, :, : config.k], True, axis=2)
        neighbors &= defined[:, :, None]
        mutual = neighbors & np.swapaxes(neighbors, 1, 2)

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
            mutual_count += mutual[local_index].astype(np.int64)

    maximum_degenerate_count = math.floor(
        bootstrap.maximum_degenerate_fraction
        * bootstrap.resample_count
        + 1e-12
    )
    if degenerate_count > maximum_degenerate_count:
        raise ValueError(
            "bootstrap exceeded maximum degenerate resample fraction"
        )
    valid_count = bootstrap.resample_count - degenerate_count
    if valid_count <= 0:
        raise ValueError("bootstrap has no valid resamples")
    frequency = mutual_count.astype(np.float64) / valid_count
    return {
        "degenerate_fraction": degenerate_count / bootstrap.resample_count,
        "degenerate_resample_count": degenerate_count,
        "draw_catalog_sha256": draw_digest.hexdigest(),
        "frequency_denominator": "valid_resamples_only",
        "minimum_mutual_frequency": (
            config.bootstrap_minimum_mutual_frequency
        ),
        "mutual_count_matrix": [
            [int(value) for value in row] for row in mutual_count
        ],
        "mutual_frequency_matrix": _matrix_payload(frequency),
        "resample_count": bootstrap.resample_count,
        "sampling_unit": "whole_example_with_all_its_target_token_rows",
        "topology_catalog_encoding": (
            "resample_u64be||valid_u8||packbits(defined_node_mask)||"
            "packbits(row_major_directed_neighbor_mask_if_valid)"
        ),
        "topology_catalog_sha256": topology_digest.hexdigest(),
        "valid_resample_count": valid_count,
    }


def _replicate_summary(
    *,
    mode_axis: tuple[str, ...],
    replicates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    size = len(mode_axis)
    mutual_count = [[0] * size for _ in range(size)]
    joint_defined_count = [[0] * size for _ in range(size)]
    mode_index = {mode_id: index for index, mode_id in enumerate(mode_axis)}
    for replicate in replicates:
        topology = replicate["topology"]
        assert isinstance(topology, Mapping)
        receipts = topology["neighbor_receipts"]
        assert isinstance(receipts, list)
        defined = {
            str(item["mode_id"]): item["status"] == "defined"
            for item in receipts
            if isinstance(item, Mapping)
        }
        edges = {
            (str(edge[0]), str(edge[1]))
            for edge in topology["mutual_edges"]
            if isinstance(edge, list) and len(edge) == 2
        }
        for left in range(size):
            for right in range(left + 1, size):
                if defined[mode_axis[left]] and defined[mode_axis[right]]:
                    joint_defined_count[left][right] += 1
                    joint_defined_count[right][left] += 1
                if _canonical_pair(mode_axis[left], mode_axis[right]) in edges:
                    mutual_count[left][right] += 1
                    mutual_count[right][left] += 1

    frequency: list[list[float | None]] = [
        [None] * size for _ in range(size)
    ]
    for left in range(size):
        frequency[left][left] = 1.0
        for right in range(left + 1, size):
            denominator = joint_defined_count[left][right]
            value = (
                mutual_count[left][right] / denominator
                if denominator > 0
                else None
            )
            frequency[left][right] = value
            frequency[right][left] = value
    return {
        "joint_defined_count_matrix": joint_defined_count,
        "mutual_count_matrix": mutual_count,
        "mutual_frequency_given_joint_defined_matrix": frequency,
        "replicate_count": len(replicates),
        "replicates": list(replicates),
    }


def _topology_for_selected(
    validated: _ValidatedInput,
    selected: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, object], dict[str, object]]:
    distances = _distance_analysis(validated, selected)
    topology = _neighbor_topology(
        distances["symmetric_distance_matrix"],
        mode_axis=validated.mode_axis,
        config=validated.graph_input.config,
    )
    return distances, topology


def _stability_analysis(validated: _ValidatedInput) -> dict[str, object]:
    family_replicates: list[dict[str, object]] = []
    for family_id in validated.family_ids:
        selected = _selected_examples(
            validated, include_families=(family_id,)
        )
        distances, topology = _topology_for_selected(validated, selected)
        family_replicates.append(
            {
                "distance": distances,
                "family_id": family_id,
                "topology": topology,
            }
        )

    lofo_replicates: list[dict[str, object]] = []
    if len(validated.family_ids) >= 2:
        for heldout_family_id in validated.family_ids:
            included = tuple(
                family_id
                for family_id in validated.family_ids
                if family_id != heldout_family_id
            )
            selected = _selected_examples(
                validated, include_families=included
            )
            distances, topology = _topology_for_selected(validated, selected)
            lofo_replicates.append(
                {
                    "distance": distances,
                    "heldout_family_id": heldout_family_id,
                    "topology": topology,
                }
            )

    loeo_replicates: list[dict[str, object]] = []
    loeo_available = all(
        len(example_ids) >= 2
        for example_ids in validated.example_ids_by_family.values()
    )
    if loeo_available:
        for family_id in validated.family_ids:
            for example_id in validated.example_ids_by_family[family_id]:
                selected = _selected_examples(
                    validated, excluded_example=(family_id, example_id)
                )
                distances, topology = _topology_for_selected(
                    validated, selected
                )
                loeo_replicates.append(
                    {
                        "distance": distances,
                        "heldout_example_id": example_id,
                        "heldout_family_id": family_id,
                        "topology": topology,
                    }
                )

    return {
        "bootstrap": _bootstrap_analysis(validated),
        "by_family": _replicate_summary(
            mode_axis=validated.mode_axis, replicates=family_replicates
        ),
        "leave_one_example_out": {
            "status": (
                "available"
                if loeo_available
                else "unavailable_family_with_single_example"
            ),
            **_replicate_summary(
                mode_axis=validated.mode_axis,
                replicates=loeo_replicates,
            ),
        },
        "leave_one_family_out": {
            "status": (
                "available"
                if lofo_replicates
                else "unavailable_single_family"
            ),
            **_replicate_summary(
                mode_axis=validated.mode_axis,
                replicates=lofo_replicates,
            ),
        },
    }


def _mode_diagnostics(validated: _ValidatedInput) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for mode_id in validated.mode_axis:
        mode = validated.mode_by_id[mode_id]
        family_receipts: list[dict[str, object]] = []
        total_rows = 0
        occupied_rows = 0
        activation_energy = 0.0
        tangent_energy = 0.0
        orthogonal_energy = 0.0
        fisher_energy = 0.0
        route_catalogs: set[str] = set()
        route_receipts: set[str] = set()
        for family_id in validated.family_ids:
            rows = [
                validated.chart_rows[(family_id, example_id, mode_id)]
                for example_id in validated.example_ids_by_family[family_id]
            ]
            family_rows = sum(row.row_count for row in rows)
            family_occupied = sum(row.occupied_row_count for row in rows)
            family_activation = math.fsum(
                row.weighted_activation_energy for row in rows
            )
            family_fisher = math.fsum(
                row.weighted_fisher_energy for row in rows
            )
            family_tangent = math.fsum(
                row.weighted_chart_tangent_energy for row in rows
            )
            family_orthogonal = math.fsum(
                row.weighted_chart_orthogonal_energy for row in rows
            )
            family_receipts.append(
                {
                    "family_id": family_id,
                    "occupied_row_count": family_occupied,
                    "occupancy_fraction": family_occupied / family_rows,
                    "row_count": family_rows,
                    "weighted_activation_energy": family_activation,
                    "weighted_chart_orthogonal_energy": family_orthogonal,
                    "weighted_chart_tangent_energy": family_tangent,
                    "weighted_fisher_energy": family_fisher,
                }
            )
            total_rows += family_rows
            occupied_rows += family_occupied
            activation_energy += family_activation
            tangent_energy += family_tangent
            orthogonal_energy += family_orthogonal
            fisher_energy += family_fisher
            for row in rows:
                if row.supplemental_route_catalog_sha256 is not None:
                    route_catalogs.add(row.supplemental_route_catalog_sha256)
                    assert row.supplemental_route_receipt_sha256 is not None
                    route_receipts.add(row.supplemental_route_receipt_sha256)
        result.append(
            {
                "comparison_adapter_kind": mode.comparison_adapter_kind,
                "family_diagnostics": family_receipts,
                "mode_id": mode_id,
                "mode_kind": mode.mode_kind,
                "native_abi_kind": mode.native_abi.abi_kind,
                "occupied_row_count": occupied_rows,
                "occupancy_fraction": occupied_rows / total_rows,
                "row_count": total_rows,
                "supplemental_route_catalog_sha256s": sorted(route_catalogs),
                "supplemental_route_receipt_sha256s": sorted(route_receipts),
                "weighted_activation_energy": activation_energy,
                "weighted_chart_orthogonal_energy": orthogonal_energy,
                "weighted_chart_tangent_energy": tangent_energy,
                "weighted_fisher_energy": fisher_energy,
            }
        )
    return result


def _native_partition_diagnostics(
    validated: _ValidatedInput,
    pooled_topology: Mapping[str, object],
) -> dict[str, object]:
    partitions = {
        abi_kind: tuple(
            mode_id
            for mode_id in validated.mode_axis
            if validated.mode_by_id[mode_id].native_abi.abi_kind == abi_kind
        )
        for abi_kind in sorted(_NATIVE_ABI_KINDS)
    }
    raw_edges = pooled_topology["mutual_edges"]
    assert isinstance(raw_edges, list)
    within: dict[str, list[list[str]]] = {
        abi_kind: [] for abi_kind in sorted(_NATIVE_ABI_KINDS)
    }
    cross: list[dict[str, object]] = []
    for raw_edge in raw_edges:
        assert isinstance(raw_edge, list) and len(raw_edge) == 2
        left, right = str(raw_edge[0]), str(raw_edge[1])
        left_kind = validated.mode_by_id[left].native_abi.abi_kind
        right_kind = validated.mode_by_id[right].native_abi.abi_kind
        if left_kind == right_kind:
            within[left_kind].append([left, right])
        else:
            cross.append(
                {
                    "left_mode_id": left,
                    "left_native_abi_kind": left_kind,
                    "right_mode_id": right,
                    "right_native_abi_kind": right_kind,
                }
            )
    return {
        "cross_native_abi_mutual_edges": cross,
        "partitions": [
            {
                "mode_ids": list(partitions[abi_kind]),
                "native_abi_kind": abi_kind,
                "within_partition_mutual_edges": within[abi_kind],
            }
            for abi_kind in sorted(_NATIVE_ABI_KINDS)
        ],
    }


def _point_gate_receipts(
    *,
    validated: _ValidatedInput,
    pooled_distance: Mapping[str, object],
    pooled_topology: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build the shared point/error/control gate vector for every pair."""

    axis = validated.mode_axis
    index = {mode_id: ordinal for ordinal, mode_id in enumerate(axis)}
    point_edges = {
        tuple(edge)
        for edge in pooled_topology["mutual_edges"]
        if isinstance(edge, list)
    }
    directed = pooled_distance["directional_equal_family_distance_matrix"]
    symmetric = pooled_distance["symmetric_distance_matrix"]
    global_ratio = pooled_distance[
        "directional_global_pooled_ratio_matrix_diagnostic"
    ]
    gain_zero = pooled_distance["directional_real_gain_over_zero_matrix"]
    gain_shuffle = pooled_distance[
        "directional_real_gain_over_row_shuffle_matrix"
    ]
    shuffle_gain_zero = pooled_distance[
        "directional_row_shuffle_gain_over_zero_matrix"
    ]
    assert isinstance(directed, list)
    assert isinstance(symmetric, list)
    assert isinstance(global_ratio, list)
    assert isinstance(gain_zero, list)
    assert isinstance(gain_shuffle, list)
    assert isinstance(shuffle_gain_zero, list)

    result: list[dict[str, object]] = []
    for left_index, left in enumerate(axis):
        for right in axis[left_index + 1 :]:
            right_index = index[right]
            point_mutual = (left, right) in point_edges
            left_distance = float(directed[left_index][right_index])
            right_distance = float(directed[right_index][left_index])
            left_gain_zero = float(gain_zero[left_index][right_index])
            right_gain_zero = float(gain_zero[right_index][left_index])
            left_gain_shuffle = gain_shuffle[left_index][right_index]
            right_gain_shuffle = gain_shuffle[right_index][left_index]
            left_directional_gate = (
                left_distance
                <= validated.graph_input.config.maximum_directional_normalized_residual
            )
            right_directional_gate = (
                right_distance
                <= validated.graph_input.config.maximum_directional_normalized_residual
            )
            left_zero_gate = left_gain_zero > 0.0
            right_zero_gate = right_gain_zero > 0.0
            left_shuffle_gate = (
                isinstance(left_gain_shuffle, (int, float))
                and not isinstance(left_gain_shuffle, bool)
                and float(left_gain_shuffle) > 0.0
            )
            right_shuffle_gate = (
                isinstance(right_gain_shuffle, (int, float))
                and not isinstance(right_gain_shuffle, bool)
                and float(right_gain_shuffle) > 0.0
            )
            point_gate = (
                point_mutual
                and left_directional_gate
                and right_directional_gate
                and left_zero_gate
                and right_zero_gate
                and left_shuffle_gate
                and right_shuffle_gate
            )
            result.append(
                {
                    "left_mode_id": left,
                    "left_to_right_directional_distance": left_distance,
                    "left_to_right_directional_residual_gate_pass": (
                        left_directional_gate
                    ),
                    "left_to_right_real_gain_over_row_shuffle": (
                        left_gain_shuffle
                    ),
                    "left_to_right_real_gain_over_row_shuffle_gate_pass": (
                        left_shuffle_gate
                    ),
                    "left_to_right_real_gain_over_zero": left_gain_zero,
                    "left_to_right_real_gain_over_zero_gate_pass": (
                        left_zero_gate
                    ),
                    "left_to_right_row_shuffle_gain_over_zero": (
                        shuffle_gain_zero[left_index][right_index]
                    ),
                    "left_to_right_global_pooled_ratio_diagnostic": (
                        global_ratio[left_index][right_index]
                    ),
                    "point_gate_pass": point_gate,
                    "point_mutual_knn": point_mutual,
                    "right_mode_id": right,
                    "right_to_left_directional_distance": right_distance,
                    "right_to_left_directional_residual_gate_pass": (
                        right_directional_gate
                    ),
                    "right_to_left_real_gain_over_row_shuffle": (
                        right_gain_shuffle
                    ),
                    "right_to_left_real_gain_over_row_shuffle_gate_pass": (
                        right_shuffle_gate
                    ),
                    "right_to_left_real_gain_over_zero": right_gain_zero,
                    "right_to_left_real_gain_over_zero_gate_pass": (
                        right_zero_gate
                    ),
                    "right_to_left_row_shuffle_gain_over_zero": (
                        shuffle_gain_zero[right_index][left_index]
                    ),
                    "right_to_left_global_pooled_ratio_diagnostic": (
                        global_ratio[right_index][left_index]
                    ),
                    "symmetric_distance": symmetric[left_index][right_index],
                }
            )
    return result


def _edge_stability_receipts(
    *,
    validated: _ValidatedInput,
    pooled_distance: Mapping[str, object],
    pooled_topology: Mapping[str, object],
    stability: Mapping[str, object],
) -> list[dict[str, object]]:
    axis = validated.mode_axis
    index = {mode_id: ordinal for ordinal, mode_id in enumerate(axis)}
    point_edges = {
        tuple(edge)
        for edge in pooled_topology["mutual_edges"]
        if isinstance(edge, list)
    }

    def matrix_value(section: str, field: str, left: int, right: int) -> object:
        raw_section = stability[section]
        assert isinstance(raw_section, Mapping)
        matrix = raw_section[field]
        assert isinstance(matrix, list)
        row = matrix[left]
        assert isinstance(row, list)
        return row[right]

    directed = pooled_distance["directional_equal_family_distance_matrix"]
    symmetric = pooled_distance["symmetric_distance_matrix"]
    global_ratio = pooled_distance[
        "directional_global_pooled_ratio_matrix_diagnostic"
    ]
    gain_zero = pooled_distance["directional_real_gain_over_zero_matrix"]
    gain_shuffle = pooled_distance[
        "directional_real_gain_over_row_shuffle_matrix"
    ]
    shuffle_gain_zero = pooled_distance[
        "directional_row_shuffle_gain_over_zero_matrix"
    ]
    assert isinstance(directed, list)
    assert isinstance(symmetric, list)
    assert isinstance(global_ratio, list)
    assert isinstance(gain_zero, list)
    assert isinstance(gain_shuffle, list)
    assert isinstance(shuffle_gain_zero, list)
    result: list[dict[str, object]] = []
    for left_index, left in enumerate(axis):
        for right in axis[left_index + 1 :]:
            right_index = index[right]
            point_mutual = (left, right) in point_edges
            left_distance = float(directed[left_index][right_index])
            right_distance = float(directed[right_index][left_index])
            left_gain_zero = float(gain_zero[left_index][right_index])
            right_gain_zero = float(gain_zero[right_index][left_index])
            left_gain_shuffle = gain_shuffle[left_index][right_index]
            right_gain_shuffle = gain_shuffle[right_index][left_index]
            left_directional_gate = (
                left_distance
                <= validated.graph_input.config.maximum_directional_normalized_residual
            )
            right_directional_gate = (
                right_distance
                <= validated.graph_input.config.maximum_directional_normalized_residual
            )
            left_zero_gate = left_gain_zero > 0.0
            right_zero_gate = right_gain_zero > 0.0
            left_shuffle_gate = (
                isinstance(left_gain_shuffle, (int, float))
                and not isinstance(left_gain_shuffle, bool)
                and float(left_gain_shuffle) > 0.0
            )
            right_shuffle_gate = (
                isinstance(right_gain_shuffle, (int, float))
                and not isinstance(right_gain_shuffle, bool)
                and float(right_gain_shuffle) > 0.0
            )
            point_gate = (
                point_mutual
                and left_directional_gate
                and right_directional_gate
                and left_zero_gate
                and right_zero_gate
                and left_shuffle_gate
                and right_shuffle_gate
            )
            bootstrap_frequency = float(
                matrix_value(
                    "bootstrap",
                    "mutual_frequency_matrix",
                    left_index,
                    right_index,
                )
            )
            bootstrap_count = int(
                matrix_value(
                    "bootstrap",
                    "mutual_count_matrix",
                    left_index,
                    right_index,
                )
            )
            bootstrap_gate = (
                bootstrap_frequency
                >= validated.graph_input.config.bootstrap_minimum_mutual_frequency
            )
            lofo_count = int(
                matrix_value(
                    "leave_one_family_out",
                    "mutual_count_matrix",
                    left_index,
                    right_index,
                )
            )
            lofo_gate = (
                lofo_count
                >= validated.graph_input.config.lofo_minimum_mutual_count
            )
            result.append(
                {
                    "bootstrap_mutual_count": bootstrap_count,
                    "bootstrap_mutual_frequency": bootstrap_frequency,
                    "bootstrap_mutual_frequency_gate_pass": bootstrap_gate,
                    "by_family_joint_defined_count": matrix_value(
                        "by_family",
                        "joint_defined_count_matrix",
                        left_index,
                        right_index,
                    ),
                    "by_family_mutual_count": matrix_value(
                        "by_family",
                        "mutual_count_matrix",
                        left_index,
                        right_index,
                    ),
                    "by_family_mutual_frequency_given_joint_defined": (
                        matrix_value(
                            "by_family",
                            "mutual_frequency_given_joint_defined_matrix",
                            left_index,
                            right_index,
                        )
                    ),
                    "left_mode_id": left,
                    "left_to_right_directional_distance": left_distance,
                    "left_to_right_directional_residual_gate_pass": (
                        left_directional_gate
                    ),
                    "left_to_right_real_gain_over_row_shuffle": (
                        left_gain_shuffle
                    ),
                    "left_to_right_real_gain_over_row_shuffle_gate_pass": (
                        left_shuffle_gate
                    ),
                    "left_to_right_real_gain_over_zero": left_gain_zero,
                    "left_to_right_real_gain_over_zero_gate_pass": (
                        left_zero_gate
                    ),
                    "left_to_right_row_shuffle_gain_over_zero": (
                        shuffle_gain_zero[left_index][right_index]
                    ),
                    "left_to_right_global_pooled_ratio_diagnostic": (
                        global_ratio[left_index][right_index]
                    ),
                    "leave_one_example_out_mutual_count": matrix_value(
                        "leave_one_example_out",
                        "mutual_count_matrix",
                        left_index,
                        right_index,
                    ),
                    "leave_one_family_out_mutual_count": matrix_value(
                        "leave_one_family_out",
                        "mutual_count_matrix",
                        left_index,
                        right_index,
                    ),
                    "leave_one_family_out_mutual_count_gate_pass": lofo_gate,
                    "point_gate_pass": point_gate,
                    "point_mutual_knn": point_mutual,
                    "right_mode_id": right,
                    "right_to_left_directional_distance": right_distance,
                    "right_to_left_directional_residual_gate_pass": (
                        right_directional_gate
                    ),
                    "right_to_left_real_gain_over_row_shuffle": (
                        right_gain_shuffle
                    ),
                    "right_to_left_real_gain_over_row_shuffle_gate_pass": (
                        right_shuffle_gate
                    ),
                    "right_to_left_real_gain_over_zero": right_gain_zero,
                    "right_to_left_real_gain_over_zero_gate_pass": (
                        right_zero_gate
                    ),
                    "right_to_left_row_shuffle_gain_over_zero": (
                        shuffle_gain_zero[right_index][left_index]
                    ),
                    "right_to_left_global_pooled_ratio_diagnostic": (
                        global_ratio[right_index][left_index]
                    ),
                    "symmetric_distance": symmetric[left_index][right_index],
                    "topology_stability_gate_pass": (
                        bootstrap_gate and lofo_gate
                    ),
                    "confirmation_core_gate_pass": (
                        point_gate and bootstrap_gate and lofo_gate
                    ),
                }
            )
    return result


def build_chart_energy_neighbor_receipt(
    graph_input: ChartEnergyNeighborInput,
) -> ChartEnergyNeighborReceipt:
    """Validate sufficient statistics and build a deterministic receipt."""

    if not isinstance(graph_input, ChartEnergyNeighborInput):
        raise TypeError("graph_input must be a ChartEnergyNeighborInput")
    validated = _validate_input(graph_input)
    canonical_input = validated.graph_input
    input_payload = canonical_input.canonical_payload()
    input_sha256 = _domain_digest(_INPUT_DOMAIN, input_payload)

    selected = _selected_examples(validated)
    pooled_distance, pooled_topology = _topology_for_selected(
        validated, selected
    )
    stability = _stability_analysis(validated)
    analysis: dict[str, object] = {
        "comparison_abi": {
            **validated.comparison_abi.canonical_payload(),
            "sha256": validated.comparison_abi.sha256,
        },
        "chart_atlas_sha256": validated.chart_atlas_sha256,
        "distance_contract": {
            "directed_distance": (
                "equal_family_mean_of_pooled_within_family_"
                "transport_residual_over_target_fisher_chart_energy"
            ),
            "global_pooled_ratio_role": "diagnostic_only",
            "symmetric_distance": (
                "arithmetic_mean_of_both_directed_distances"
            ),
            "transport_direction": (
                "source_chart_transported_to_target_chart"
            ),
            "shared_spd_mismatch": (
                "predicted_energy_plus_target_energy_minus_two_cross_inner_"
                "product"
            ),
        },
        "edge_stability_receipts": [],
        "confirmation_core_edges": [],
        "mode_axis": list(validated.mode_axis),
        "mode_diagnostics": _mode_diagnostics(validated),
        "native_abi_diagnostics": {},
        "pooled": {
            "distance": pooled_distance,
            "topology": pooled_topology,
        },
        "point_gate_contract": {
            "both_directional_normalized_residual_at_most": (
                canonical_input.config.maximum_directional_normalized_residual
            ),
            "both_directions_positive_gain_over_row_shuffle": True,
            "both_directions_positive_gain_over_zero_prediction": True,
            "mutual_knn_required": True,
        },
        "point_gated_candidate_edges": [],
        "point_gated_candidate_edges_status": (
            "applicable"
            if canonical_input.panel_role == "B_neighbor_discovery"
            else "not_applicable_for_panel_role"
        ),
        "confirmation_core_edges_status": (
            "applicable"
            if canonical_input.panel_role == "C_neighbor_confirmation"
            else "not_applicable_for_panel_role"
        ),
        "row_shuffle_catalog_sha256": (
            validated.row_shuffle_catalog_sha256
        ),
        "stability": stability,
        "transport_fit_catalog_sha256": (
            validated.transport_fit_catalog_sha256
        ),
        "transport_fit_role": _FIT_ROLE,
        "transport_evaluation_role": canonical_input.panel_role,
        "transport_split_receipt_sha256": (
            validated.transport_split_receipt_sha256
        ),
    }
    analysis["native_abi_diagnostics"] = _native_partition_diagnostics(
        validated, pooled_topology
    )
    edge_receipts = _edge_stability_receipts(
        validated=validated,
        pooled_distance=pooled_distance,
        pooled_topology=pooled_topology,
        stability=stability,
    )
    analysis["edge_stability_receipts"] = edge_receipts
    if canonical_input.panel_role == "B_neighbor_discovery":
        analysis["point_gated_candidate_edges"] = [
            [item["left_mode_id"], item["right_mode_id"]]
            for item in edge_receipts
            if item["point_gate_pass"] is True
        ]
    if canonical_input.panel_role == "C_neighbor_confirmation":
        analysis["confirmation_core_edges"] = [
            [item["left_mode_id"], item["right_mode_id"]]
            for item in edge_receipts
            if item["confirmation_core_gate_pass"] is True
        ]
    provisional = ChartEnergyNeighborReceipt(
        graph_input=canonical_input,
        input_sha256=input_sha256,
        analysis=analysis,
        receipt_sha256="0" * 64,
    )
    receipt_sha256 = _domain_digest(
        _RECEIPT_DOMAIN, provisional._payload_without_receipt_sha256()
    )
    return ChartEnergyNeighborReceipt(
        graph_input=canonical_input,
        input_sha256=input_sha256,
        analysis=analysis,
        receipt_sha256=receipt_sha256,
    )


def _canonical_diagnostic_subset_examples(
    validated: _ValidatedInput,
    family_example_ids: Sequence[tuple[str, Sequence[str]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(family_example_ids, (tuple, list)):
        raise TypeError("family_example_ids must be a sequence")
    if len(family_example_ids) != len(validated.family_ids):
        raise ValueError("diagnostic subset must include every family")
    result: list[tuple[str, tuple[str, ...]]] = []
    for family_ordinal, expected_family in enumerate(validated.family_ids):
        entry = family_example_ids[family_ordinal]
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("each diagnostic subset family must be a pair")
        family_id = _require_name(
            entry[0], label="diagnostic subset family_id"
        )
        if family_id != expected_family:
            raise ValueError(
                "diagnostic subset family order must match the source input"
            )
        raw_examples = entry[1]
        if not isinstance(raw_examples, (tuple, list)) or not raw_examples:
            raise TypeError(
                "each diagnostic subset family needs example IDs"
            )
        example_ids = tuple(
            _require_name(
                example_id,
                label="diagnostic subset example_id",
            )
            for example_id in raw_examples
        )
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("diagnostic subset example IDs must be unique")
        source_examples = validated.example_ids_by_family[family_id]
        expected_order = tuple(
            example_id
            for example_id in source_examples
            if example_id in set(example_ids)
        )
        if example_ids != expected_order:
            raise ValueError(
                "diagnostic subset examples must be an ordered source subset"
            )
        result.append((family_id, example_ids))
    return tuple(result)


def build_chart_energy_diagnostic_subset_receipt(
    graph_input: ChartEnergyNeighborInput,
    *,
    subset_id: str,
    family_example_ids: Sequence[tuple[str, Sequence[str]]],
) -> ChartEnergyDiagnosticSubsetReceipt:
    """Recompute non-authorizing point geometry on an exact row subset.

    Bootstrap and leave-out stability are intentionally absent.  This surface
    is for preregistered short/boundary/transition diagnostics and the C
    hard-long point/error veto.  It cannot nominate or confirm an edge.
    """

    if not isinstance(graph_input, ChartEnergyNeighborInput):
        raise TypeError("graph_input must be a ChartEnergyNeighborInput")
    canonical_subset_id = _require_name(subset_id, label="subset_id")
    validated = _validate_input(graph_input)
    canonical_input = validated.graph_input
    source_input_sha256 = _domain_digest(
        _INPUT_DOMAIN, canonical_input.canonical_payload()
    )
    canonical_examples = _canonical_diagnostic_subset_examples(
        validated, family_example_ids
    )
    selected = dict(canonical_examples)
    pooled_distance, pooled_topology = _topology_for_selected(
        validated, selected
    )
    point_receipts = _point_gate_receipts(
        validated=validated,
        pooled_distance=pooled_distance,
        pooled_topology=pooled_topology,
    )
    subset_catalog_payload = {
        "family_example_ids": [
            {
                "example_ids": list(example_ids),
                "family_id": family_id,
            }
            for family_id, example_ids in canonical_examples
        ],
        "source_input_sha256": source_input_sha256,
        "source_panel_binding_sha256": canonical_input.panel_binding_sha256,
        "source_panel_role": canonical_input.panel_role,
        "subset_id": canonical_subset_id,
    }
    subset_catalog_sha256 = _domain_digest(
        _DIAGNOSTIC_SUBSET_CATALOG_DOMAIN,
        subset_catalog_payload,
    )
    first_mode = validated.mode_axis[0]
    family_receipts: list[dict[str, object]] = []
    for family_id, example_ids in canonical_examples:
        row_count = sum(
            validated.chart_rows[
                (family_id, example_id, first_mode)
            ].row_count
            for example_id in example_ids
        )
        family_receipts.append(
            {
                "example_count": len(example_ids),
                "example_ids": list(example_ids),
                "family_id": family_id,
                "prediction_row_count": row_count,
            }
        )
    analysis: dict[str, object] = {
        "authorizes_edges": False,
        "comparison_abi": {
            **validated.comparison_abi.canonical_payload(),
            "sha256": validated.comparison_abi.sha256,
        },
        "distance_contract": {
            "directed_distance": (
                "equal_family_mean_of_pooled_within_family_"
                "transport_residual_over_target_fisher_chart_energy"
            ),
            "global_pooled_ratio_role": "diagnostic_only",
            "symmetric_distance": (
                "arithmetic_mean_of_both_directed_distances"
            ),
        },
        "family_receipts": family_receipts,
        "mode_axis": list(validated.mode_axis),
        "native_abi_diagnostics": _native_partition_diagnostics(
            validated, pooled_topology
        ),
        "point_gate_contract": {
            "both_directional_normalized_residual_at_most": (
                canonical_input.config.maximum_directional_normalized_residual
            ),
            "both_directions_positive_gain_over_row_shuffle": True,
            "both_directions_positive_gain_over_zero_prediction": True,
            "mutual_knn_required": True,
        },
        "point_gate_receipts": point_receipts,
        "point_gated_edges": [
            [item["left_mode_id"], item["right_mode_id"]]
            for item in point_receipts
            if item["point_gate_pass"] is True
        ],
        "pooled": {
            "distance": pooled_distance,
            "topology": pooled_topology,
        },
        "stability_status": (
            "not_computed_for_non_authorizing_diagnostic_subset"
        ),
        "subset_catalog_sha256": subset_catalog_sha256,
        "subset_id": canonical_subset_id,
    }
    provisional = ChartEnergyDiagnosticSubsetReceipt(
        subset_id=canonical_subset_id,
        source_input_sha256=source_input_sha256,
        source_panel_role=canonical_input.panel_role,
        source_panel_binding_sha256=canonical_input.panel_binding_sha256,
        family_example_ids=canonical_examples,
        subset_catalog_sha256=subset_catalog_sha256,
        analysis=analysis,
        receipt_sha256="0" * 64,
    )
    receipt_sha256 = _domain_digest(
        _DIAGNOSTIC_SUBSET_RECEIPT_DOMAIN,
        provisional._payload_without_receipt_sha256(),
    )
    return ChartEnergyDiagnosticSubsetReceipt(
        subset_id=canonical_subset_id,
        source_input_sha256=source_input_sha256,
        source_panel_role=canonical_input.panel_role,
        source_panel_binding_sha256=canonical_input.panel_binding_sha256,
        family_example_ids=canonical_examples,
        subset_catalog_sha256=subset_catalog_sha256,
        analysis=analysis,
        receipt_sha256=receipt_sha256,
    )


def _parse_comparison_abi(value: object) -> ChartComparisonABI:
    raw = _require_exact_fields(
        value,
        {
            "abi_name",
            "carrier_width",
            "energy_semantics",
            "generalized_contribution_semantics",
            "metric_semantics",
            "row_semantics",
            "sha256",
        },
        label="comparison_abi",
    )
    abi = ChartComparisonABI(
        abi_name=_require_name(raw["abi_name"], label="abi_name"),
        carrier_width=_require_positive_int(
            raw["carrier_width"], label="carrier_width"
        ),
        energy_semantics=_require_name(
            raw["energy_semantics"], label="energy_semantics"
        ),
        generalized_contribution_semantics=_require_name(
            raw["generalized_contribution_semantics"],
            label="generalized_contribution_semantics",
        ),
        metric_semantics=_require_name(
            raw["metric_semantics"], label="metric_semantics"
        ),
        row_semantics=_require_name(
            raw["row_semantics"], label="row_semantics"
        ),
    )
    if _require_sha256(raw["sha256"], label="comparison_abi.sha256") != abi.sha256:
        raise ValueError("comparison ABI digest is invalid")
    return abi


def _parse_native_abi(value: object) -> NativeChartABI:
    raw = _require_exact_fields(
        value,
        {
            "abi_kind",
            "abi_name",
            "abi_version",
            "context_signature",
            "input_signature",
            "output_signature",
            "sha256",
        },
        label="native_abi",
    )
    abi = NativeChartABI(
        abi_kind=_require_name(raw["abi_kind"], label="abi_kind"),
        abi_name=_require_name(raw["abi_name"], label="abi_name"),
        abi_version=_require_positive_int(
            raw["abi_version"], label="abi_version"
        ),
        context_signature=_require_name(
            raw["context_signature"], label="context_signature"
        ),
        input_signature=_require_name(
            raw["input_signature"], label="input_signature"
        ),
        output_signature=_require_name(
            raw["output_signature"], label="output_signature"
        ),
    )
    if _require_sha256(raw["sha256"], label="native_abi.sha256") != abi.sha256:
        raise ValueError("native ABI digest is invalid")
    return abi


def _parse_mode_spec(value: object) -> ChartModeSpec:
    raw = _require_exact_fields(
        value,
        {
            "comparison_abi",
            "comparison_adapter_kind",
            "comparison_adapter_receipt_sha256",
            "context_site_id",
            "contribution_site_id",
            "event_id",
            "event_index",
            "mode_id",
            "mode_kind",
            "native_abi",
            "site_placement_sha256",
            "state_fingerprint_sha256",
        },
        label="mode_spec",
    )
    return ChartModeSpec(
        mode_id=_require_name(raw["mode_id"], label="mode_id"),
        mode_kind=_require_name(raw["mode_kind"], label="mode_kind"),
        state_fingerprint_sha256=_require_sha256(
            raw["state_fingerprint_sha256"],
            label="state_fingerprint_sha256",
        ),
        event_id=_require_name(
            raw["event_id"], label="event_id"
        ),
        event_index=_require_nonnegative_int(
            raw["event_index"], label="event_index"
        ),
        context_site_id=_require_name(
            raw["context_site_id"], label="context_site_id"
        ),
        contribution_site_id=_require_name(
            raw["contribution_site_id"], label="contribution_site_id"
        ),
        site_placement_sha256=_require_sha256(
            raw["site_placement_sha256"], label="site_placement_sha256"
        ),
        native_abi=_parse_native_abi(raw["native_abi"]),
        comparison_abi=_parse_comparison_abi(raw["comparison_abi"]),
        comparison_adapter_kind=_require_name(
            raw["comparison_adapter_kind"],
            label="comparison_adapter_kind",
        ),
        comparison_adapter_receipt_sha256=_require_sha256(
            raw["comparison_adapter_receipt_sha256"],
            label="comparison_adapter_receipt_sha256",
        ),
    )


def _parse_example_stats(value: object) -> ChartExampleStats:
    fields = {
        "aligned_row_catalog_sha256",
        "chart_atlas_sha256",
        "chart_energy_vector_catalog_sha256",
        "comparison_abi_sha256",
        "example_id",
        "family_id",
        "minimum_row_fisher_energy",
        "mode_id",
        "occupied_row_count",
        "row_count",
        "supplemental_route_catalog_sha256",
        "supplemental_route_receipt_sha256",
        "weighted_activation_energy",
        "weighted_chart_orthogonal_energy",
        "weighted_chart_tangent_energy",
        "weighted_fisher_energy",
    }
    raw = _require_exact_fields(value, fields, label="example_stats")
    return ChartExampleStats(
        family_id=_require_name(raw["family_id"], label="family_id"),
        example_id=_require_name(raw["example_id"], label="example_id"),
        mode_id=_require_name(raw["mode_id"], label="mode_id"),
        comparison_abi_sha256=_require_sha256(
            raw["comparison_abi_sha256"], label="comparison_abi_sha256"
        ),
        chart_atlas_sha256=_require_sha256(
            raw["chart_atlas_sha256"], label="chart_atlas_sha256"
        ),
        aligned_row_catalog_sha256=_require_sha256(
            raw["aligned_row_catalog_sha256"],
            label="aligned_row_catalog_sha256",
        ),
        chart_energy_vector_catalog_sha256=_require_sha256(
            raw["chart_energy_vector_catalog_sha256"],
            label="chart_energy_vector_catalog_sha256",
        ),
        row_count=_require_positive_int(raw["row_count"], label="row_count"),
        occupied_row_count=_require_nonnegative_int(
            raw["occupied_row_count"], label="occupied_row_count"
        ),
        minimum_row_fisher_energy=_require_finite(
            raw["minimum_row_fisher_energy"],
            label="minimum_row_fisher_energy",
            minimum=0.0,
        ),
        weighted_activation_energy=_require_finite(
            raw["weighted_activation_energy"],
            label="weighted_activation_energy",
            minimum=0.0,
        ),
        weighted_chart_tangent_energy=_require_finite(
            raw["weighted_chart_tangent_energy"],
            label="weighted_chart_tangent_energy",
            minimum=0.0,
        ),
        weighted_chart_orthogonal_energy=_require_finite(
            raw["weighted_chart_orthogonal_energy"],
            label="weighted_chart_orthogonal_energy",
            minimum=0.0,
        ),
        weighted_fisher_energy=_require_finite(
            raw["weighted_fisher_energy"],
            label="weighted_fisher_energy",
            minimum=0.0,
        ),
        supplemental_route_catalog_sha256=_require_optional_sha256(
            raw["supplemental_route_catalog_sha256"],
            label="supplemental_route_catalog_sha256",
        ),
        supplemental_route_receipt_sha256=_require_optional_sha256(
            raw["supplemental_route_receipt_sha256"],
            label="supplemental_route_receipt_sha256",
        ),
    )


def _parse_transport_stats(value: object) -> DirectedChartTransportStats:
    fields = {
        "aligned_row_catalog_sha256",
        "comparison_abi_sha256",
        "example_id",
        "family_id",
        "row_count",
        "source_chart_stats_sha256",
        "source_mode_id",
        "target_chart_stats_sha256",
        "target_mode_id",
        "transport_evaluation_catalog_sha256",
        "transport_evaluation_panel_sha256",
        "transport_evaluation_role",
        "transport_fit_catalog_sha256",
        "transport_fit_receipt_sha256",
        "transport_fit_role",
        "transport_protocol",
        "transport_split_receipt_sha256",
        "transported_row_count",
        "row_shuffle_catalog_sha256",
        "weighted_predicted_fisher_energy",
        "weighted_predicted_target_cross_inner_product",
        "weighted_row_shuffled_transport_residual_energy",
        "weighted_target_fisher_energy",
        "weighted_transport_residual_energy",
        "weighted_zero_prediction_residual_energy",
    }
    raw = _require_exact_fields(value, fields, label="transport_stats")
    kwargs: dict[str, object] = {}
    for name in (
        "family_id",
        "example_id",
        "source_mode_id",
        "target_mode_id",
        "transport_fit_role",
        "transport_evaluation_role",
        "transport_protocol",
    ):
        kwargs[name] = _require_name(raw[name], label=name)
    for name in (
        "comparison_abi_sha256",
        "aligned_row_catalog_sha256",
        "source_chart_stats_sha256",
        "target_chart_stats_sha256",
        "transport_fit_catalog_sha256",
        "transport_evaluation_catalog_sha256",
        "transport_evaluation_panel_sha256",
        "transport_split_receipt_sha256",
        "transport_fit_receipt_sha256",
        "row_shuffle_catalog_sha256",
    ):
        kwargs[name] = _require_sha256(raw[name], label=name)
    kwargs["row_count"] = _require_positive_int(
        raw["row_count"], label="row_count"
    )
    kwargs["transported_row_count"] = _require_nonnegative_int(
        raw["transported_row_count"], label="transported_row_count"
    )
    kwargs["weighted_transport_residual_energy"] = _require_finite(
        raw["weighted_transport_residual_energy"],
        label="weighted_transport_residual_energy",
        minimum=0.0,
    )
    kwargs["weighted_predicted_fisher_energy"] = _require_finite(
        raw["weighted_predicted_fisher_energy"],
        label="weighted_predicted_fisher_energy",
        minimum=0.0,
    )
    kwargs["weighted_target_fisher_energy"] = _require_finite(
        raw["weighted_target_fisher_energy"],
        label="weighted_target_fisher_energy",
        minimum=0.0,
    )
    kwargs["weighted_predicted_target_cross_inner_product"] = _require_finite(
        raw["weighted_predicted_target_cross_inner_product"],
        label="weighted_predicted_target_cross_inner_product",
    )
    kwargs["weighted_zero_prediction_residual_energy"] = _require_finite(
        raw["weighted_zero_prediction_residual_energy"],
        label="weighted_zero_prediction_residual_energy",
        minimum=0.0,
    )
    kwargs["weighted_row_shuffled_transport_residual_energy"] = _require_finite(
        raw["weighted_row_shuffled_transport_residual_energy"],
        label="weighted_row_shuffled_transport_residual_energy",
        minimum=0.0,
    )
    return DirectedChartTransportStats(**kwargs)  # type: ignore[arg-type]


def _parse_bootstrap_config(
    value: object,
) -> ChartWithinFamilyBootstrapConfig:
    fields = {
        "draw_catalog_digest_encoding",
        "draw_catalog_digest_domain_hex",
        "draw_digest_selection",
        "draw_preimage_encoding",
        "draw_protocol",
        "draw_protocol_domain_hex",
        "draws_per_family",
        "families",
        "maximum_degenerate_fraction",
        "resample_count",
        "resample_within_each_family_with_replacement",
        "sampling_unit",
        "seed",
    }
    raw = _require_exact_fields(value, fields, label="bootstrap_config")
    fixed = {
        "draw_catalog_digest_encoding": (
            "sha256(catalog_domain||canonical_config||"
            "repeated(resample_u64be,family_u32be,draw_u32be,"
            "example_u32be))"
        ),
        "draw_catalog_digest_domain_hex": _BOOTSTRAP_CATALOG_DOMAIN.hex(),
        "draw_digest_selection": (
            "first_8_sha256_bytes_u64be_mod_example_count"
        ),
        "draw_preimage_encoding": (
            "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
        ),
        "draw_protocol": "sha256_counter_modulo_example_count_v1",
        "draw_protocol_domain_hex": _BOOTSTRAP_DRAW_DOMAIN.hex(),
        "resample_within_each_family_with_replacement": True,
        "sampling_unit": "whole_example_with_all_its_target_token_rows",
    }
    for field_name, expected in fixed.items():
        if raw[field_name] != expected:
            raise ValueError(f"bootstrap_config.{field_name} is unsupported")
    families_raw = raw["families"]
    if not isinstance(families_raw, list):
        raise TypeError("bootstrap_config.families must be a list")
    families: list[tuple[str, tuple[str, ...]]] = []
    for family_index, value in enumerate(families_raw):
        family = _require_exact_fields(
            value,
            {"example_ids", "family_id"},
            label=f"bootstrap_config.families[{family_index}]",
        )
        example_ids = family["example_ids"]
        if not isinstance(example_ids, list):
            raise TypeError("bootstrap family example_ids must be a list")
        families.append(
            (
                _require_name(family["family_id"], label="family_id"),
                tuple(
                    _require_name(example_id, label="example_id")
                    for example_id in example_ids
                ),
            )
        )
    seed = raw["seed"]
    if type(seed) is not int:
        raise ValueError("bootstrap_config.seed must be an integer")
    domain_hex = raw["draw_protocol_domain_hex"]
    if not isinstance(domain_hex, str) or _HEX_BYTES.fullmatch(domain_hex) is None:
        raise ValueError("bootstrap draw domain must be canonical lowercase hex")
    return ChartWithinFamilyBootstrapConfig(
        seed=seed,
        resample_count=_require_positive_int(
            raw["resample_count"], label="resample_count"
        ),
        draws_per_family=_require_positive_int(
            raw["draws_per_family"], label="draws_per_family"
        ),
        maximum_degenerate_fraction=_require_finite(
            raw["maximum_degenerate_fraction"],
            label="maximum_degenerate_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        family_example_ids=tuple(families),
        draw_protocol_domain=bytes.fromhex(domain_hex),
    )


def _parse_config(value: object) -> ChartEnergyNeighborConfig:
    raw = _require_exact_fields(
        value,
        {
            "bootstrap_minimum_mutual_frequency",
            "fisher_energy_epsilon",
            "k",
            "lofo_expected_replicate_count",
            "lofo_minimum_mutual_count",
            "maximum_directional_normalized_residual",
            "rank_boundary_tie_absolute_tolerance",
            "rank_boundary_tie_policy",
        },
        label="config",
    )
    return ChartEnergyNeighborConfig(
        k=_require_positive_int(raw["k"], label="k"),
        rank_boundary_tie_absolute_tolerance=_require_finite(
            raw["rank_boundary_tie_absolute_tolerance"],
            label="rank_boundary_tie_absolute_tolerance",
            strictly_positive=True,
        ),
        rank_boundary_tie_policy=_require_name(
            raw["rank_boundary_tie_policy"],
            label="rank_boundary_tie_policy",
        ),
        fisher_energy_epsilon=_require_finite(
            raw["fisher_energy_epsilon"],
            label="fisher_energy_epsilon",
            strictly_positive=True,
        ),
        maximum_directional_normalized_residual=_require_finite(
            raw["maximum_directional_normalized_residual"],
            label="maximum_directional_normalized_residual",
            minimum=0.0,
        ),
        bootstrap_minimum_mutual_frequency=_require_finite(
            raw["bootstrap_minimum_mutual_frequency"],
            label="bootstrap_minimum_mutual_frequency",
            minimum=0.0,
            maximum=1.0,
        ),
        lofo_minimum_mutual_count=_require_positive_int(
            raw["lofo_minimum_mutual_count"],
            label="lofo_minimum_mutual_count",
        ),
        lofo_expected_replicate_count=_require_positive_int(
            raw["lofo_expected_replicate_count"],
            label="lofo_expected_replicate_count",
        ),
    )


def _parse_input(value: object) -> ChartEnergyNeighborInput:
    raw = _require_exact_fields(
        value,
        {
            "config",
            "directed_transport_stats",
            "example_stats",
            "mode_specs",
            "panel_binding_sha256",
            "panel_role",
            "within_family_bootstrap_config",
        },
        label="input",
    )
    for field in ("mode_specs", "example_stats", "directed_transport_stats"):
        if not isinstance(raw[field], list):
            raise TypeError(f"input.{field} must be a list")
    return ChartEnergyNeighborInput(
        panel_role=_require_name(raw["panel_role"], label="panel_role"),
        panel_binding_sha256=_require_sha256(
            raw["panel_binding_sha256"], label="panel_binding_sha256"
        ),
        mode_specs=tuple(_parse_mode_spec(item) for item in raw["mode_specs"]),
        example_stats=tuple(
            _parse_example_stats(item) for item in raw["example_stats"]
        ),
        directed_transport_stats=tuple(
            _parse_transport_stats(item)
            for item in raw["directed_transport_stats"]
        ),
        within_family_bootstrap_config=(
            _parse_bootstrap_config(raw["within_family_bootstrap_config"])
        ),
        config=_parse_config(raw["config"]),
    )


def parse_chart_energy_neighbor_input_payload(
    value: object,
) -> ChartEnergyNeighborInput:
    """Strictly parse and validate one canonical sufficient-stat input.

    This surface intentionally stops before bootstrap/LOFO analysis.  It lets
    a larger authenticated report store one all-example input, derive an exact
    authorized subset, and then rebuild the primary and diagnostic receipts
    without duplicating the sufficient-stat rows.
    """

    parsed = _parse_input(value)
    if _canonical_json_bytes(value) != _canonical_json_bytes(
        parsed.canonical_payload()
    ):
        raise ValueError(
            "chart-energy neighbor input does not match strict reconstruction"
        )
    return _validate_input(parsed).graph_input


def canonicalize_chart_energy_neighbor_input(
    graph_input: ChartEnergyNeighborInput,
) -> ChartEnergyNeighborInput:
    """Validate and return the canonical sufficient-stat input ordering.

    This typed surface intentionally stops before bootstrap/LOFO analysis.  It
    lets an authenticated outer worker canonicalize freshly constructed inputs
    without first serializing and reparsing them.
    """

    if not isinstance(graph_input, ChartEnergyNeighborInput):
        raise TypeError("graph_input must be a ChartEnergyNeighborInput")
    return _validate_input(graph_input).graph_input


def parse_chart_energy_neighbor_receipt_payload(
    value: object,
) -> ChartEnergyNeighborReceipt:
    """Strictly parse and semantically rebuild a receipt payload."""

    raw = _require_exact_fields(
        value,
        {
            "analysis",
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
    if raw["schema"] != CHART_ENERGY_NEIGHBOR_GRAPH_SCHEMA:
        raise ValueError("unsupported chart-energy neighbor schema")
    if (
        type(raw["format_version"]) is not int
        or raw["format_version"]
        != CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION
    ):
        raise ValueError("unsupported chart-energy neighbor format version")
    if raw["safety"] != _SAFETY:
        raise ValueError("chart-energy neighbor safety contract drifted")
    if raw["scientific_status"] != _SCIENTIFIC_STATUS:
        raise ValueError("chart-energy scientific-status contract drifted")
    _require_sha256(raw["input_sha256"], label="input_sha256")
    _require_sha256(raw["receipt_sha256"], label="receipt_sha256")
    graph_input = _parse_input(raw["input"])
    rebuilt = build_chart_energy_neighbor_receipt(graph_input)
    if _canonical_json_bytes(raw) != _canonical_json_bytes(
        rebuilt.canonical_payload()
    ):
        raise ValueError(
            "chart-energy neighbor receipt does not match deterministic rebuild"
        )
    return rebuilt


def parse_chart_energy_diagnostic_subset_receipt_payload(
    value: object,
    *,
    source_graph_input: ChartEnergyNeighborInput,
) -> ChartEnergyDiagnosticSubsetReceipt:
    """Strictly parse and rebuild a subset receipt against its source input."""

    raw = _require_exact_fields(
        value,
        {
            "analysis",
            "authorizes_edges",
            "family_example_ids",
            "format_version",
            "receipt_sha256",
            "safety",
            "schema",
            "source_input_sha256",
            "source_panel_binding_sha256",
            "source_panel_role",
            "subset_catalog_sha256",
            "subset_id",
        },
        label="diagnostic_subset_receipt",
    )
    if raw["schema"] != CHART_ENERGY_DIAGNOSTIC_SUBSET_SCHEMA:
        raise ValueError("unsupported chart-energy diagnostic subset schema")
    if (
        type(raw["format_version"]) is not int
        or raw["format_version"]
        != CHART_ENERGY_NEIGHBOR_GRAPH_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported chart-energy diagnostic subset format version"
        )
    if raw["authorizes_edges"] is not False:
        raise ValueError("diagnostic subset receipts cannot authorize edges")
    if raw["safety"] != _SAFETY:
        raise ValueError("diagnostic subset safety contract drifted")
    for field in (
        "receipt_sha256",
        "source_input_sha256",
        "source_panel_binding_sha256",
        "subset_catalog_sha256",
    ):
        _require_sha256(raw[field], label=field)
    family_rows = raw["family_example_ids"]
    if not isinstance(family_rows, list):
        raise TypeError("diagnostic subset family_example_ids must be a list")
    family_example_ids: list[tuple[str, tuple[str, ...]]] = []
    for family_ordinal, family_value in enumerate(family_rows):
        family = _require_exact_fields(
            family_value,
            {"example_ids", "family_id"},
            label=(
                "diagnostic_subset_receipt.family_example_ids"
                f"[{family_ordinal}]"
            ),
        )
        examples = family["example_ids"]
        if not isinstance(examples, list):
            raise TypeError("diagnostic subset example_ids must be a list")
        family_example_ids.append(
            (
                _require_name(family["family_id"], label="family_id"),
                tuple(
                    _require_name(example_id, label="example_id")
                    for example_id in examples
                ),
            )
        )
    rebuilt = build_chart_energy_diagnostic_subset_receipt(
        source_graph_input,
        subset_id=_require_name(raw["subset_id"], label="subset_id"),
        family_example_ids=tuple(family_example_ids),
    )
    if _canonical_json_bytes(raw) != _canonical_json_bytes(
        rebuilt.canonical_payload()
    ):
        raise ValueError(
            "diagnostic subset receipt does not match deterministic rebuild"
        )
    return rebuilt


def save_chart_energy_neighbor_receipt(
    path: str | Path,
    receipt: ChartEnergyNeighborReceipt,
) -> None:
    """Atomically persist one canonical receipt without overwriting a file."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("chart-energy neighbor receipt must use .json")
    if output.is_symlink():
        raise ValueError("refusing to write through a symlink")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not isinstance(receipt, ChartEnergyNeighborReceipt):
        raise TypeError("receipt must be a ChartEnergyNeighborReceipt")
    payload = receipt.canonical_payload()
    parse_chart_energy_neighbor_receipt_payload(payload)
    encoded = _canonical_json_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def load_chart_energy_neighbor_receipt(
    path: str | Path,
) -> ChartEnergyNeighborReceipt:
    """Strictly load canonical JSON with a bounded size and no symlinks."""

    source = Path(path)
    if source.suffix != ".json":
        raise ValueError("chart-energy neighbor receipt must use .json")
    if source.is_symlink():
        raise ValueError("refusing to load through a symlink")
    size = source.stat().st_size
    if size > _MAX_RECEIPT_BYTES:
        raise ValueError("chart-energy neighbor receipt exceeds size limit")
    encoded = source.read_bytes()
    if len(encoded) != size:
        raise ValueError("chart-energy neighbor receipt changed while reading")
    try:
        raw = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid chart-energy neighbor JSON") from error
    if encoded != _canonical_json_bytes(raw):
        raise ValueError("chart-energy neighbor JSON must be canonical")
    return parse_chart_energy_neighbor_receipt_payload(raw)
