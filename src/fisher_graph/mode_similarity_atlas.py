"""Model-agnostic similarity atlases for compatible computational modes.

The atlas consumes aggregate, directed cross-evaluation statistics.  It never
looks at transformer layer ordinals, parameter names, or execution position.
An MLP, attention operator, lexical operator, or future mode may participate
when (and only when) every compared mode declares the exact same comparison
ABI.

For a reference mode ``Gi`` and candidate mode ``Gj``, one directed row holds
the sufficient statistics for ``Gi(x_i)`` versus ``Gj(x_i)`` inside one frozen
example family.  The scientific distance is the equal-family mean of
``weighted_sse / max(weighted_reference_energy, epsilon)``.  An unordered
distance is the arithmetic mean of both directions.  Dot products and
candidate energies are retained only for cosine diagnostics; they never
change the ordering.
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
    "MODE_SIMILARITY_ATLAS_FORMAT_VERSION",
    "MODE_SIMILARITY_ATLAS_SCHEMA",
    "AtlasConfig",
    "DirectedFamilyExampleStats",
    "DirectedFamilyStats",
    "FrozenFamilyResample",
    "ModeABI",
    "ModeSimilarityAtlasReceipt",
    "ModeSpec",
    "WithinFamilyBootstrapConfig",
    "build_mode_similarity_atlas_receipt",
    "load_mode_similarity_atlas_receipt",
    "parse_atlas_config_payload",
    "parse_frozen_family_resample_payload",
    "parse_mode_similarity_atlas_receipt_payload",
    "parse_within_family_bootstrap_config_payload",
    "save_mode_similarity_atlas_receipt",
]


MODE_SIMILARITY_ATLAS_SCHEMA: Final = "fisher_graph.mode_similarity_atlas"
MODE_SIMILARITY_ATLAS_FORMAT_VERSION: Final = 1

_ABI_DIGEST_DOMAIN = b"fisher_graph.mode_similarity_atlas.abi.v1\0"
_STATS_DIGEST_DOMAIN = b"fisher_graph.mode_similarity_atlas.stats.v1\0"
_RECEIPT_DIGEST_DOMAIN = b"fisher_graph.mode_similarity_atlas.receipt.v1\0"
_BOOTSTRAP_DRAW_DOMAIN = (
    b"fisher_graph:gemma3-mode-atlas-bootstrap-draw:v1\0"
)
_BOOTSTRAP_CATALOG_DIGEST_DOMAIN = (
    b"fisher_graph.mode_similarity_atlas.bootstrap_catalog.v1\0"
)
_BOOTSTRAP_ORDER_DIGEST_DOMAIN = (
    b"fisher_graph.mode_similarity_atlas.bootstrap_orders.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_BYTES = re.compile(r"^(?:[0-9a-f]{2}){1,256}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_RANDOM_PERMUTATION_BASELINE_COUNT = 10_000
_RANDOM_PERMUTATION_BASELINE_SEED = 0

_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "frozen_mode_similarity_atlas",
    "analysis_only": True,
    "adaptive_compression_selection": False,
    "heldout_confirmation": False,
    "ordinal_features_used": False,
    "source_parameter_values_enter_distance_or_affinity": False,
    "state_fingerprint_used_only_for_axis_and_reversal": True,
    "raw_rows_used_by_atlas_core": False,
    "authorizes_mode_merge": False,
    "authorizes_parameter_compression": False,
    "authorizes_deployment": False,
}

_SAFETY: dict[str, object] = {
    "contains_tensors": False,
    "contains_raw_activations": False,
    "contains_raw_gradients": False,
    "contains_model_weights": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_only_aggregate_sufficient_statistics": True,
}


class _SeriationDegeneracy(ValueError):
    """A valid distance draw whose one-dimensional seriation is undefined."""


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


@dataclass(frozen=True, slots=True)
class ModeABI:
    """Exact interface on which two modes may be cross-evaluated.

    Signatures are semantic, versioned names rather than Python or framework
    types.  A future adapter may expose attention or lexical machinery through
    the same ABI, but a label such as ``mode_kind='attention'`` never bypasses
    exact ABI equality.
    """

    abi_name: str
    abi_version: int
    input_signature: str
    output_signature: str
    context_signature: str

    def __post_init__(self) -> None:
        _require_name(self.abi_name, label="abi_name")
        _require_positive_int(self.abi_version, label="abi_version")
        _require_name(self.input_signature, label="input_signature")
        _require_name(self.output_signature, label="output_signature")
        _require_name(self.context_signature, label="context_signature")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "abi_name": self.abi_name,
            "abi_version": self.abi_version,
            "context_signature": self.context_signature,
            "input_signature": self.input_signature,
            "output_signature": self.output_signature,
        }

    @property
    def sha256(self) -> str:
        return _domain_digest(_ABI_DIGEST_DOMAIN, self.canonical_payload())


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """One frozen computational mode; execution position is intentionally absent."""

    mode_id: str
    mode_kind: str
    state_fingerprint_sha256: str
    compatibility_abi: ModeABI

    def __post_init__(self) -> None:
        _require_name(self.mode_id, label="mode_id")
        _require_name(self.mode_kind, label="mode_kind")
        _require_sha256(
            self.state_fingerprint_sha256,
            label="state_fingerprint_sha256",
        )
        if not isinstance(self.compatibility_abi, ModeABI):
            raise TypeError("compatibility_abi must be a ModeABI")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "compatibility_abi": {
                **self.compatibility_abi.canonical_payload(),
                "sha256": self.compatibility_abi.sha256,
            },
            "mode_id": self.mode_id,
            "mode_kind": self.mode_kind,
            "state_fingerprint_sha256": self.state_fingerprint_sha256,
        }


@dataclass(frozen=True, slots=True)
class DirectedFamilyStats:
    """Aggregate statistics for ``Gi(x_i)`` versus ``Gj(x_i)`` in one family."""

    family_id: str
    reference_mode_id: str
    candidate_mode_id: str
    compatibility_abi_sha256: str
    weighted_sse: float
    weighted_reference_energy: float
    weighted_dot: float
    weighted_candidate_energy: float
    row_count: int

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.reference_mode_id, label="reference_mode_id")
        _require_name(self.candidate_mode_id, label="candidate_mode_id")
        if self.reference_mode_id == self.candidate_mode_id:
            raise ValueError("directed statistics must compare distinct modes")
        _require_sha256(
            self.compatibility_abi_sha256,
            label="compatibility_abi_sha256",
        )
        _require_finite(self.weighted_sse, label="weighted_sse", minimum=0.0)
        _require_finite(
            self.weighted_reference_energy,
            label="weighted_reference_energy",
            minimum=0.0,
        )
        _require_finite(self.weighted_dot, label="weighted_dot")
        _require_finite(
            self.weighted_candidate_energy,
            label="weighted_candidate_energy",
            minimum=0.0,
        )
        _require_positive_int(self.row_count, label="row_count")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "candidate_mode_id": self.candidate_mode_id,
            "compatibility_abi_sha256": self.compatibility_abi_sha256,
            "family_id": self.family_id,
            "reference_mode_id": self.reference_mode_id,
            "row_count": self.row_count,
            "weighted_candidate_energy": float(
                self.weighted_candidate_energy
            ),
            "weighted_dot": float(self.weighted_dot),
            "weighted_reference_energy": float(
                self.weighted_reference_energy
            ),
            "weighted_sse": float(self.weighted_sse),
        }


@dataclass(frozen=True, slots=True)
class DirectedFamilyExampleStats:
    """One example-plus-probes cluster of directed quadratic statistics."""

    family_id: str
    example_cluster_id: str
    reference_mode_id: str
    candidate_mode_id: str
    compatibility_abi_sha256: str
    weighted_sse: float
    weighted_reference_energy: float
    weighted_dot: float
    weighted_candidate_energy: float
    row_count: int

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.example_cluster_id, label="example_cluster_id")
        # Reuse the aggregate row's exact scalar validation contract.
        DirectedFamilyStats(
            family_id=self.family_id,
            reference_mode_id=self.reference_mode_id,
            candidate_mode_id=self.candidate_mode_id,
            compatibility_abi_sha256=self.compatibility_abi_sha256,
            weighted_sse=self.weighted_sse,
            weighted_reference_energy=self.weighted_reference_energy,
            weighted_dot=self.weighted_dot,
            weighted_candidate_energy=self.weighted_candidate_energy,
            row_count=self.row_count,
        )

    def as_family_stats(self) -> DirectedFamilyStats:
        return DirectedFamilyStats(
            family_id=self.family_id,
            reference_mode_id=self.reference_mode_id,
            candidate_mode_id=self.candidate_mode_id,
            compatibility_abi_sha256=self.compatibility_abi_sha256,
            weighted_sse=self.weighted_sse,
            weighted_reference_energy=self.weighted_reference_energy,
            weighted_dot=self.weighted_dot,
            weighted_candidate_energy=self.weighted_candidate_energy,
            row_count=self.row_count,
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = self.as_family_stats().canonical_payload()
        payload["example_cluster_id"] = self.example_cluster_id
        return payload


@dataclass(frozen=True, slots=True)
class FrozenFamilyResample:
    """Externally frozen family multiplicities; the atlas performs no RNG."""

    resample_id: str
    family_multiplicities: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _require_name(self.resample_id, label="resample_id")
        if not isinstance(self.family_multiplicities, tuple):
            raise TypeError("family_multiplicities must be a tuple")
        if not self.family_multiplicities:
            raise ValueError("family_multiplicities must not be empty")
        seen: set[str] = set()
        for index, entry in enumerate(self.family_multiplicities):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    f"family_multiplicities[{index}] must be a pair"
                )
            family_id = _require_name(
                entry[0], label=f"family_multiplicities[{index}].family_id"
            )
            _require_positive_int(
                entry[1],
                label=f"family_multiplicities[{index}].multiplicity",
            )
            if family_id in seen:
                raise ValueError("family multiplicities must be unique")
            seen.add(family_id)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family_multiplicities": [
                {"family_id": family_id, "multiplicity": multiplicity}
                for family_id, multiplicity in sorted(
                    self.family_multiplicities
                )
            ],
            "resample_id": self.resample_id,
        }


@dataclass(frozen=True, slots=True)
class WithinFamilyBootstrapConfig:
    """Frozen clustered bootstrap catalog generated by SHA-256 counters."""

    seed: int
    resample_count: int
    draws_per_family: int
    maximum_degenerate_fraction: float
    family_example_cluster_ids: tuple[tuple[str, tuple[str, ...]], ...]
    draw_protocol_domain: bytes = _BOOTSTRAP_DRAW_DOMAIN

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**64:
            raise ValueError("bootstrap seed must be an unsigned 64-bit integer")
        _require_positive_int(self.resample_count, label="resample_count")
        _require_positive_int(self.draws_per_family, label="draws_per_family")
        fraction = _require_finite(
            self.maximum_degenerate_fraction,
            label="maximum_degenerate_fraction",
            minimum=0.0,
        )
        if fraction > 1.0:
            raise ValueError("maximum_degenerate_fraction must be <= 1")
        if (
            not isinstance(self.draw_protocol_domain, bytes)
            or not self.draw_protocol_domain
            or len(self.draw_protocol_domain) > 256
            or not self.draw_protocol_domain.endswith(b"\0")
        ):
            raise ValueError(
                "draw_protocol_domain must be 1..256 bytes ending in NUL"
            )
        if not isinstance(self.family_example_cluster_ids, tuple):
            raise TypeError("family_example_cluster_ids must be a tuple")
        if not self.family_example_cluster_ids:
            raise ValueError("family_example_cluster_ids must not be empty")
        seen_families: set[str] = set()
        for family_index, entry in enumerate(self.family_example_cluster_ids):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    f"family_example_cluster_ids[{family_index}] must be a pair"
                )
            family_id = _require_name(
                entry[0],
                label=f"family_example_cluster_ids[{family_index}].family_id",
            )
            if family_id in seen_families:
                raise ValueError("bootstrap family IDs must be unique")
            seen_families.add(family_id)
            cluster_ids = entry[1]
            if not isinstance(cluster_ids, tuple) or not cluster_ids:
                raise TypeError(
                    "each bootstrap family must contain a nonempty tuple of "
                    "example cluster IDs"
                )
            for cluster_index, cluster_id in enumerate(cluster_ids):
                _require_name(
                    cluster_id,
                    label=(
                        f"family_example_cluster_ids[{family_index}]"
                        f".example_cluster_ids[{cluster_index}]"
                    ),
                )
            if len(set(cluster_ids)) != len(cluster_ids):
                raise ValueError(
                    "example cluster IDs must be unique within each family"
                )

    @property
    def ordered_family_ids(self) -> tuple[str, ...]:
        return tuple(
            family_id for family_id, _ in self.family_example_cluster_ids
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "draw_catalog_digest_encoding": (
                "sha256(catalog_domain||canonical_config||"
                "repeated(resample_u64be,family_u32be,draw_u32be,"
                "example_u32be))"
            ),
            "draw_catalog_digest_domain_hex": (
                _BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
            ),
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
                    "example_cluster_ids": list(cluster_ids),
                    "family_id": family_id,
                }
                for family_id, cluster_ids in self.family_example_cluster_ids
            ],
            "maximum_degenerate_fraction": float(
                self.maximum_degenerate_fraction
            ),
            "resample_count": self.resample_count,
            "seed": self.seed,
            "sampling_unit": "example_plus_selected_probes_cluster",
            "with_replacement": True,
        }


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    """Numerical contract for distance, affinity, and fail-closed seriation."""

    energy_epsilon: float = 1e-12
    sufficient_stat_relative_tolerance: float = 1e-7
    connectivity_tolerance: float = 1e-12
    fiedler_eigengap_relative_tolerance: float = 1e-8
    fiedler_coordinate_tie_tolerance: float = 1e-10
    nearest_neighbor_tie_tolerance: float = 1e-10
    affinity_max_exponent: float = 700.0

    def __post_init__(self) -> None:
        for name in (
            "energy_epsilon",
            "sufficient_stat_relative_tolerance",
            "connectivity_tolerance",
            "fiedler_eigengap_relative_tolerance",
            "fiedler_coordinate_tie_tolerance",
            "nearest_neighbor_tie_tolerance",
            "affinity_max_exponent",
        ):
            _require_finite(
                getattr(self, name), label=name, strictly_positive=True
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "affinity_kernel": "exp_neg_distance_over_positive_median_v1",
            "affinity_max_exponent": float(self.affinity_max_exponent),
            "connectivity_tolerance": float(self.connectivity_tolerance),
            "cosine_role": "diagnostic_only",
            "distance_metric": (
                "bidirectional_equal_family_mean_relative_sse_v1"
            ),
            "energy_epsilon": float(self.energy_epsilon),
            "fiedler_coordinate_tie_tolerance": float(
                self.fiedler_coordinate_tie_tolerance
            ),
            "fiedler_eigengap_relative_tolerance": float(
                self.fiedler_eigengap_relative_tolerance
            ),
            "laplacian": "symmetric_normalized_v1",
            "mode_id_role": "receipt_label_only",
            "mode_kind_role": "receipt_label_only",
            "nearest_neighbor_tie_tolerance": float(
                self.nearest_neighbor_tie_tolerance
            ),
            "ordering_quality_random_permutation_count": (
                _RANDOM_PERMUTATION_BASELINE_COUNT
            ),
            "ordering_quality_random_permutation_seed": (
                _RANDOM_PERMUTATION_BASELINE_SEED
            ),
            "state_fingerprint_role": (
                "canonical_axis_and_global_reversal_only"
            ),
            "sufficient_stat_relative_tolerance": float(
                self.sufficient_stat_relative_tolerance
            ),
        }


@dataclass(frozen=True, slots=True)
class _SpectralOrdering:
    affinity_scale: float
    affinity: tuple[tuple[float, ...], ...]
    laplacian_eigenvalues: tuple[float, ...]
    fiedler_eigenvalue: float
    fiedler_eigengap: float
    fiedler_coordinates: tuple[float, ...]
    order: tuple[str, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "affinity": [list(row) for row in self.affinity],
            "affinity_scale": self.affinity_scale,
            "fiedler_coordinates": list(self.fiedler_coordinates),
            "fiedler_eigengap": self.fiedler_eigengap,
            "fiedler_eigenvalue": self.fiedler_eigenvalue,
            "laplacian_eigenvalues": list(self.laplacian_eigenvalues),
            "seriation_order": list(self.order),
        }


@dataclass(frozen=True, slots=True)
class _StabilityReplicate:
    replicate_id: str
    omitted_family_id: str | None
    family_multiplicities: tuple[tuple[str, int], ...]
    order: tuple[str, ...]
    adjacency_retention: float
    nearest_neighbor_retention: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "adjacency_retention": self.adjacency_retention,
            "family_multiplicities": [
                {"family_id": family_id, "multiplicity": multiplicity}
                for family_id, multiplicity in self.family_multiplicities
            ],
            "nearest_neighbor_retention": self.nearest_neighbor_retention,
            "omitted_family_id": self.omitted_family_id,
            "replicate_id": self.replicate_id,
            "seriation_order": list(self.order),
        }


@dataclass(frozen=True, slots=True)
class _StabilitySummary:
    replicates: tuple[_StabilityReplicate, ...]
    adjacency_frequency: tuple[tuple[float, ...], ...]
    nearest_neighbor_frequency: tuple[tuple[float, ...], ...]
    mean_adjacency_retention: float | None
    minimum_adjacency_retention: float | None
    mean_nearest_neighbor_retention: float | None
    minimum_nearest_neighbor_retention: float | None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "adjacency_frequency": [
                list(row) for row in self.adjacency_frequency
            ],
            "mean_adjacency_retention": self.mean_adjacency_retention,
            "mean_nearest_neighbor_retention": (
                self.mean_nearest_neighbor_retention
            ),
            "minimum_adjacency_retention": self.minimum_adjacency_retention,
            "minimum_nearest_neighbor_retention": (
                self.minimum_nearest_neighbor_retention
            ),
            "nearest_neighbor_frequency": [
                list(row) for row in self.nearest_neighbor_frequency
            ],
            "replicates": [
                replicate.canonical_payload()
                for replicate in self.replicates
            ],
        }


@dataclass(frozen=True, slots=True)
class _WithinFamilyBootstrapSummary:
    draw_catalog_sha256: str
    order_catalog_sha256: str
    valid_resample_count: int
    degenerate_resample_count: int
    degenerate_fraction: float
    adjacency_frequency: tuple[tuple[float, ...], ...]
    nearest_neighbor_frequency: tuple[tuple[float, ...], ...]
    mean_adjacency_retention: float
    minimum_adjacency_retention: float
    p05_adjacency_retention: float
    mean_nearest_neighbor_retention: float
    minimum_nearest_neighbor_retention: float
    p05_nearest_neighbor_retention: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "adjacency_frequency": [
                list(row) for row in self.adjacency_frequency
            ],
            "degenerate_fraction": self.degenerate_fraction,
            "degenerate_resample_count": self.degenerate_resample_count,
            "draw_catalog_sha256": self.draw_catalog_sha256,
            "mean_adjacency_retention": self.mean_adjacency_retention,
            "mean_nearest_neighbor_retention": (
                self.mean_nearest_neighbor_retention
            ),
            "minimum_adjacency_retention": self.minimum_adjacency_retention,
            "minimum_nearest_neighbor_retention": (
                self.minimum_nearest_neighbor_retention
            ),
            "nearest_neighbor_frequency": [
                list(row) for row in self.nearest_neighbor_frequency
            ],
            "order_catalog_sha256": self.order_catalog_sha256,
            "p05_adjacency_retention": self.p05_adjacency_retention,
            "p05_nearest_neighbor_retention": (
                self.p05_nearest_neighbor_retention
            ),
            "valid_resample_count": self.valid_resample_count,
        }


@dataclass(frozen=True, slots=True)
class _AtlasAnalysis:
    mode_axis: tuple[str, ...]
    directed_relative_sse: tuple[tuple[float, ...], ...]
    directed_cosine_similarity: tuple[tuple[float, ...], ...]
    symmetric_distance: tuple[tuple[float, ...], ...]
    symmetric_cosine_distance: tuple[tuple[float, ...], ...]
    spectral: _SpectralOrdering
    fiedler_path_cost: float
    random_path_cost_mean: float
    random_path_cost_std: float
    random_path_cost_minimum: float
    random_path_cost_maximum: float
    fiedler_to_random_mean_cost_ratio: float
    fiedler_random_percentile_fraction: float
    fiedler_random_improvement_z: float | None
    nearest_neighbors: tuple[tuple[str, str, float], ...]
    leave_one_family_out: _StabilitySummary
    frozen_resamples: _StabilitySummary
    within_family_bootstrap: _WithinFamilyBootstrapSummary | None

    def canonical_payload(self) -> dict[str, object]:
        payload = {
            "directed_cosine_similarity": [
                list(row) for row in self.directed_cosine_similarity
            ],
            "directed_relative_sse": [
                list(row) for row in self.directed_relative_sse
            ],
            "frozen_resample_stability": (
                self.frozen_resamples.canonical_payload()
            ),
            "ordering_quality": {
                "fiedler_path_cost": self.fiedler_path_cost,
                "fiedler_random_improvement_z": (
                    self.fiedler_random_improvement_z
                ),
                "fiedler_random_percentile_fraction": (
                    self.fiedler_random_percentile_fraction
                ),
                "fiedler_to_random_mean_cost_ratio": (
                    self.fiedler_to_random_mean_cost_ratio
                ),
                "path_cost_definition": (
                    "sum_of_adjacent_symmetric_distances"
                ),
                "random_path_cost_maximum": self.random_path_cost_maximum,
                "random_path_cost_mean": self.random_path_cost_mean,
                "random_path_cost_minimum": self.random_path_cost_minimum,
                "random_path_cost_std": self.random_path_cost_std,
                "random_permutation_count": (
                    _RANDOM_PERMUTATION_BASELINE_COUNT
                ),
                "random_permutation_seed": (
                    _RANDOM_PERMUTATION_BASELINE_SEED
                ),
                "role": "diagnostic_only",
            },
            "leave_one_family_out_stability": (
                self.leave_one_family_out.canonical_payload()
            ),
            "mode_axis": list(self.mode_axis),
            "nearest_neighbors": [
                {
                    "distance": distance,
                    "mode_id": mode_id,
                    "neighbor_mode_id": neighbor_mode_id,
                }
                for mode_id, neighbor_mode_id, distance in self.nearest_neighbors
            ],
            "symmetric_cosine_distance": [
                list(row) for row in self.symmetric_cosine_distance
            ],
            "symmetric_distance": [
                list(row) for row in self.symmetric_distance
            ],
            "within_family_bootstrap_stability": (
                None
                if self.within_family_bootstrap is None
                else self.within_family_bootstrap.canonical_payload()
            ),
        }
        payload.update(self.spectral.canonical_payload())
        return payload


@dataclass(frozen=True, slots=True)
class ModeSimilarityAtlasReceipt:
    """Authenticated inputs and deterministic atlas outputs."""

    modes: tuple[ModeSpec, ...]
    directed_stats: tuple[DirectedFamilyStats, ...]
    directed_example_stats: tuple[DirectedFamilyExampleStats, ...]
    config: AtlasConfig
    frozen_resamples: tuple[FrozenFamilyResample, ...]
    within_family_bootstrap_config: WithinFamilyBootstrapConfig | None
    family_ids: tuple[str, ...]
    compatibility_abi_sha256: str
    input_statistics_sha256: str
    analysis: _AtlasAnalysis
    receipt_sha256: str

    @property
    def seriation_order(self) -> tuple[str, ...]:
        return self.analysis.spectral.order

    @property
    def symmetric_distance(self) -> tuple[tuple[float, ...], ...]:
        return self.analysis.symmetric_distance

    def _payload_without_receipt_sha256(self) -> dict[str, object]:
        return {
            "analysis": self.analysis.canonical_payload(),
            "atlas_config": self.config.canonical_payload(),
            "compatibility_abi_sha256": self.compatibility_abi_sha256,
            "directed_family_stats": [
                stats.canonical_payload() for stats in self.directed_stats
            ],
            "directed_family_example_stats": [
                stats.canonical_payload()
                for stats in self.directed_example_stats
            ],
            "family_ids": list(self.family_ids),
            "format_version": MODE_SIMILARITY_ATLAS_FORMAT_VERSION,
            "frozen_family_resamples": [
                resample.canonical_payload()
                for resample in self.frozen_resamples
            ],
            "input_statistics_sha256": self.input_statistics_sha256,
            "mode_specs": [mode.canonical_payload() for mode in self.modes],
            "safety": dict(_SAFETY),
            "schema": MODE_SIMILARITY_ATLAS_SCHEMA,
            "scientific_status": dict(_SCIENTIFIC_STATUS),
            "within_family_bootstrap_config": (
                None
                if self.within_family_bootstrap_config is None
                else self.within_family_bootstrap_config.canonical_payload()
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        payload = self._payload_without_receipt_sha256()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _validate_sufficient_statistics(
    stats: DirectedFamilyStats,
    *,
    tolerance: float,
) -> None:
    expected_sse = (
        float(stats.weighted_reference_energy)
        + float(stats.weighted_candidate_energy)
        - 2.0 * float(stats.weighted_dot)
    )
    scale = max(
        1.0,
        float(stats.weighted_sse),
        float(stats.weighted_reference_energy),
        float(stats.weighted_candidate_energy),
        abs(float(stats.weighted_dot)),
    )
    if abs(float(stats.weighted_sse) - expected_sse) > tolerance * scale:
        raise ValueError(
            "weighted_sse is inconsistent with the quadratic statistics"
        )
    norm_product = math.sqrt(
        float(stats.weighted_reference_energy)
        * float(stats.weighted_candidate_energy)
    )
    if abs(float(stats.weighted_dot)) > norm_product + tolerance * scale:
        raise ValueError("weighted_dot violates the Cauchy bound")


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _distance_matrices(
    *,
    axis: tuple[str, ...],
    stats_by_key: Mapping[tuple[str, str, str], DirectedFamilyStats],
    family_multiplicities: Mapping[str, int],
    config: AtlasConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(axis)
    directed = np.zeros((size, size), dtype=np.float64)
    directed_cosine = np.ones((size, size), dtype=np.float64)
    total_multiplicity = sum(family_multiplicities.values())
    if total_multiplicity <= 0:
        raise ValueError("at least one family must have positive multiplicity")

    for reference_index, reference_id in enumerate(axis):
        for candidate_index, candidate_id in enumerate(axis):
            if reference_index == candidate_index:
                continue
            relative_sse = 0.0
            cosine = 0.0
            for family_id, multiplicity in family_multiplicities.items():
                row = stats_by_key[(family_id, reference_id, candidate_id)]
                reference_energy = max(
                    float(row.weighted_reference_energy),
                    config.energy_epsilon,
                )
                relative_sse += multiplicity * (
                    float(row.weighted_sse) / reference_energy
                )
                cosine_denominator = math.sqrt(
                    max(
                        float(row.weighted_reference_energy),
                        config.energy_epsilon,
                    )
                    * max(
                        float(row.weighted_candidate_energy),
                        config.energy_epsilon,
                    )
                )
                raw_cosine = float(row.weighted_dot) / cosine_denominator
                if raw_cosine < -1.0 - config.sufficient_stat_relative_tolerance:
                    raise ValueError("cosine diagnostic is below -1")
                if raw_cosine > 1.0 + config.sufficient_stat_relative_tolerance:
                    raise ValueError("cosine diagnostic exceeds 1")
                cosine += multiplicity * min(1.0, max(-1.0, raw_cosine))
            directed[reference_index, candidate_index] = (
                relative_sse / total_multiplicity
            )
            directed_cosine[reference_index, candidate_index] = (
                cosine / total_multiplicity
            )

    symmetric = 0.5 * (directed + directed.T)
    np.fill_diagonal(symmetric, 0.0)
    symmetric_cosine = 1.0 - 0.5 * (
        directed_cosine + directed_cosine.T
    )
    np.fill_diagonal(symmetric_cosine, 0.0)
    if not np.isfinite(symmetric).all() or (symmetric < 0.0).any():
        raise ValueError("symmetric distance must be finite and nonnegative")
    return directed, directed_cosine, symmetric, symmetric_cosine


def _spectral_ordering(
    *,
    distance: np.ndarray,
    axis: tuple[str, ...],
    state_fingerprints: Mapping[str, str],
    config: AtlasConfig,
) -> _SpectralOrdering:
    size = len(axis)
    off_diagonal = distance[np.triu_indices(size, k=1)]
    positive = off_diagonal[off_diagonal > 0.0]
    if positive.size == 0:
        raise _SeriationDegeneracy(
            "all pairwise distances are zero; seriation is undefined"
        )
    affinity_scale = float(np.median(positive))
    if not math.isfinite(affinity_scale) or affinity_scale <= 0.0:
        raise _SeriationDegeneracy(
            "affinity scale must be finite and positive"
        )

    exponent = np.minimum(
        distance / affinity_scale,
        config.affinity_max_exponent,
    )
    affinity = np.exp(-exponent)
    np.fill_diagonal(affinity, 0.0)
    if not np.isfinite(affinity).all():
        raise _SeriationDegeneracy("affinity matrix must be finite")
    off_affinity = affinity[~np.eye(size, dtype=bool)]
    if (off_affinity <= 0.0).any():
        raise _SeriationDegeneracy("affinity graph must remain dense")

    degree = affinity.sum(axis=1)
    if (degree <= 0.0).any() or not np.isfinite(degree).all():
        raise _SeriationDegeneracy("affinity graph has an invalid degree")
    inverse_root_degree = 1.0 / np.sqrt(degree)
    laplacian = np.eye(size, dtype=np.float64) - (
        inverse_root_degree[:, None]
        * affinity
        * inverse_root_degree[None, :]
    )
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if not np.isfinite(eigenvalues).all():
        raise _SeriationDegeneracy("laplacian eigenvalues must be finite")
    fiedler_eigenvalue = float(eigenvalues[1])
    if fiedler_eigenvalue <= config.connectivity_tolerance:
        raise _SeriationDegeneracy(
            "affinity graph is disconnected or numerically singular"
        )
    fiedler_eigengap = float(eigenvalues[2] - eigenvalues[1])
    eigengap_scale = max(
        1.0, abs(float(eigenvalues[1])), abs(float(eigenvalues[2]))
    )
    if (
        fiedler_eigengap
        <= config.fiedler_eigengap_relative_tolerance * eigengap_scale
    ):
        raise _SeriationDegeneracy("Fiedler eigenvalue is degenerate")

    coordinates = np.asarray(eigenvectors[:, 1], dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise _SeriationDegeneracy("Fiedler coordinates must be finite")
    raw_order = np.argsort(coordinates, kind="stable")
    ordered_coordinates = coordinates[raw_order]
    coordinate_scale = max(1.0, float(np.max(np.abs(coordinates))))
    if np.any(
        np.diff(ordered_coordinates)
        <= config.fiedler_coordinate_tie_tolerance * coordinate_scale
    ):
        raise _SeriationDegeneracy("Fiedler coordinates contain a tie")

    forward_ids = tuple(axis[int(index)] for index in raw_order)
    reverse_ids = tuple(reversed(forward_ids))
    forward_fingerprints = tuple(
        state_fingerprints[mode_id] for mode_id in forward_ids
    )
    reverse_fingerprints = tuple(
        state_fingerprints[mode_id] for mode_id in reverse_ids
    )
    if reverse_fingerprints < forward_fingerprints:
        coordinates = -coordinates
        raw_order = np.argsort(coordinates, kind="stable")
    order = tuple(axis[int(index)] for index in raw_order)
    return _SpectralOrdering(
        affinity_scale=affinity_scale,
        affinity=_matrix_tuple(affinity),
        laplacian_eigenvalues=tuple(float(value) for value in eigenvalues),
        fiedler_eigenvalue=fiedler_eigenvalue,
        fiedler_eigengap=fiedler_eigengap,
        fiedler_coordinates=tuple(float(value) for value in coordinates),
        order=order,
    )


def _nearest_neighbors(
    *,
    distance: np.ndarray,
    axis: tuple[str, ...],
    tie_tolerance: float,
) -> tuple[tuple[str, str, float], ...]:
    result: list[tuple[str, str, float]] = []
    for index, mode_id in enumerate(axis):
        candidate_indices = [other for other in range(len(axis)) if other != index]
        candidate_indices.sort(key=lambda other: float(distance[index, other]))
        first = candidate_indices[0]
        second = candidate_indices[1]
        first_distance = float(distance[index, first])
        second_distance = float(distance[index, second])
        scale = max(1.0, abs(first_distance), abs(second_distance))
        if second_distance - first_distance <= tie_tolerance * scale:
            raise _SeriationDegeneracy(
                f"nearest neighbor is tied for mode {mode_id}"
            )
        result.append((mode_id, axis[first], first_distance))
    return tuple(result)


def _adjacencies(order: Sequence[str]) -> set[frozenset[str]]:
    return {
        frozenset((left, right))
        for left, right in zip(order, order[1:])
    }


def _ordering_quality(
    *,
    distance: np.ndarray,
    axis: tuple[str, ...],
    order: tuple[str, ...],
) -> tuple[float, float, float, float, float, float, float, float | None]:
    axis_index = {mode_id: index for index, mode_id in enumerate(axis)}
    order_indices = np.asarray(
        [axis_index[mode_id] for mode_id in order], dtype=np.int64
    )
    fiedler_cost = float(
        distance[order_indices[:-1], order_indices[1:]].sum()
    )
    generator = np.random.Generator(
        np.random.PCG64(_RANDOM_PERMUTATION_BASELINE_SEED)
    )
    # Sorting independent continuous draws gives an exactly uniform random
    # permutation without letting an input/provenance order enter the score.
    permutations = np.argsort(
        generator.random(
            (_RANDOM_PERMUTATION_BASELINE_COUNT, len(axis)),
            dtype=np.float64,
        ),
        axis=1,
        kind="stable",
    )
    costs = distance[permutations[:, :-1], permutations[:, 1:]].sum(axis=1)
    mean = float(np.mean(costs))
    standard_deviation = float(np.std(costs))
    minimum = float(np.min(costs))
    maximum = float(np.max(costs))
    if not all(
        math.isfinite(value)
        for value in (
            fiedler_cost,
            mean,
            standard_deviation,
            minimum,
            maximum,
        )
    ):
        raise ValueError("ordering quality diagnostics must be finite")
    if mean <= 0.0:
        raise ValueError("random path-cost mean must be positive")
    ratio = fiedler_cost / mean
    percentile = float(np.mean(costs <= fiedler_cost))
    improvement_z = (
        (mean - fiedler_cost) / standard_deviation
        if standard_deviation > 0.0
        else None
    )
    return (
        fiedler_cost,
        mean,
        standard_deviation,
        minimum,
        maximum,
        ratio,
        percentile,
        improvement_z,
    )


def _stability_summary(
    *,
    specifications: Sequence[
        tuple[str, str | None, tuple[tuple[str, int], ...]]
    ],
    axis: tuple[str, ...],
    state_fingerprints: Mapping[str, str],
    stats_by_key: Mapping[tuple[str, str, str], DirectedFamilyStats],
    config: AtlasConfig,
    base_order: tuple[str, ...],
    base_nearest_neighbors: tuple[tuple[str, str, float], ...],
) -> _StabilitySummary:
    size = len(axis)
    base_edges = _adjacencies(base_order)
    base_nearest = {
        mode_id: neighbor for mode_id, neighbor, _ in base_nearest_neighbors
    }
    replicates: list[_StabilityReplicate] = []
    adjacency_counts = np.zeros((size, size), dtype=np.float64)
    nearest_counts = np.zeros((size, size), dtype=np.float64)
    axis_index = {mode_id: index for index, mode_id in enumerate(axis)}

    for replicate_id, omitted_family_id, multiplicities in specifications:
        directed, directed_cosine, distance, symmetric_cosine = (
            _distance_matrices(
                axis=axis,
                stats_by_key=stats_by_key,
                family_multiplicities=dict(multiplicities),
                config=config,
            )
        )
        del directed, directed_cosine, symmetric_cosine
        spectral = _spectral_ordering(
            distance=distance,
            axis=axis,
            state_fingerprints=state_fingerprints,
            config=config,
        )
        nearest = _nearest_neighbors(
            distance=distance,
            axis=axis,
            tie_tolerance=config.nearest_neighbor_tie_tolerance,
        )
        edges = _adjacencies(spectral.order)
        adjacency_retention = len(base_edges & edges) / len(base_edges)
        nearest_map = {
            mode_id: neighbor for mode_id, neighbor, _ in nearest
        }
        nearest_retention = sum(
            nearest_map[mode_id] == base_nearest[mode_id]
            for mode_id in axis
        ) / size

        for edge in edges:
            left, right = tuple(edge)
            left_index = axis_index[left]
            right_index = axis_index[right]
            adjacency_counts[left_index, right_index] += 1.0
            adjacency_counts[right_index, left_index] += 1.0
        for mode_id, neighbor_id, _ in nearest:
            nearest_counts[axis_index[mode_id], axis_index[neighbor_id]] += 1.0
        replicates.append(
            _StabilityReplicate(
                replicate_id=replicate_id,
                omitted_family_id=omitted_family_id,
                family_multiplicities=multiplicities,
                order=spectral.order,
                adjacency_retention=float(adjacency_retention),
                nearest_neighbor_retention=float(nearest_retention),
            )
        )

    if replicates:
        adjacency_counts /= len(replicates)
        nearest_counts /= len(replicates)
        adjacency_retentions = [
            replicate.adjacency_retention for replicate in replicates
        ]
        nearest_retentions = [
            replicate.nearest_neighbor_retention for replicate in replicates
        ]
        mean_adjacency = float(np.mean(adjacency_retentions))
        minimum_adjacency = float(min(adjacency_retentions))
        mean_nearest = float(np.mean(nearest_retentions))
        minimum_nearest = float(min(nearest_retentions))
    else:
        mean_adjacency = None
        minimum_adjacency = None
        mean_nearest = None
        minimum_nearest = None
    return _StabilitySummary(
        replicates=tuple(replicates),
        adjacency_frequency=_matrix_tuple(adjacency_counts),
        nearest_neighbor_frequency=_matrix_tuple(nearest_counts),
        mean_adjacency_retention=mean_adjacency,
        minimum_adjacency_retention=minimum_adjacency,
        mean_nearest_neighbor_retention=mean_nearest,
        minimum_nearest_neighbor_retention=minimum_nearest,
    )


def _bootstrap_example_ordinal(
    *,
    domain: bytes,
    seed: int,
    resample_index: int,
    family_ordinal: int,
    draw_ordinal: int,
    example_count: int,
) -> int:
    preimage = b"".join(
        (
            domain,
            # The configured domain is authenticated in the receipt.  The
            # default is the frozen Gemma-3 MLP-first setup domain, while the
            # atlas math remains model-agnostic.
            seed.to_bytes(8, "big", signed=False),
            resample_index.to_bytes(8, "big", signed=False),
            family_ordinal.to_bytes(4, "big", signed=False),
            draw_ordinal.to_bytes(4, "big", signed=False),
        )
    )
    draw = int.from_bytes(
        hashlib.sha256(preimage).digest()[:8], "big", signed=False
    )
    return draw % example_count


def _within_family_bootstrap_summary(
    *,
    bootstrap_config: WithinFamilyBootstrapConfig,
    axis: tuple[str, ...],
    state_fingerprints: Mapping[str, str],
    example_stats_by_key: Mapping[
        tuple[str, str, str, str], DirectedFamilyExampleStats
    ],
    config: AtlasConfig,
    base_order: tuple[str, ...],
    base_nearest_neighbors: tuple[tuple[str, str, float], ...],
) -> _WithinFamilyBootstrapSummary:
    resample_count = bootstrap_config.resample_count
    size = len(axis)
    directed_batch = np.zeros(
        (resample_count, size, size), dtype=np.float64
    )
    draw_catalog_digest = hashlib.sha256()
    draw_catalog_digest.update(_BOOTSTRAP_CATALOG_DIGEST_DOMAIN)
    draw_catalog_digest.update(
        _canonical_json_bytes(bootstrap_config.canonical_payload())
    )

    family_counts = [
        np.zeros((resample_count, len(cluster_ids)), dtype=np.float64)
        for _, cluster_ids in bootstrap_config.family_example_cluster_ids
    ]
    # Catalog order is part of the protocol: resample, then configured family,
    # then draw.  The hash preimage independently carries all three ordinals.
    for resample_index in range(resample_count):
        for family_ordinal, (_, cluster_ids) in enumerate(
            bootstrap_config.family_example_cluster_ids
        ):
            for draw_ordinal in range(bootstrap_config.draws_per_family):
                example_ordinal = _bootstrap_example_ordinal(
                    domain=bootstrap_config.draw_protocol_domain,
                    seed=bootstrap_config.seed,
                    resample_index=resample_index,
                    family_ordinal=family_ordinal,
                    draw_ordinal=draw_ordinal,
                    example_count=len(cluster_ids),
                )
                family_counts[family_ordinal][
                    resample_index, example_ordinal
                ] += 1.0
                draw_catalog_digest.update(
                    resample_index.to_bytes(8, "big", signed=False)
                )
                draw_catalog_digest.update(
                    family_ordinal.to_bytes(4, "big", signed=False)
                )
                draw_catalog_digest.update(
                    draw_ordinal.to_bytes(4, "big", signed=False)
                )
                draw_catalog_digest.update(
                    example_ordinal.to_bytes(4, "big", signed=False)
                )

    for family_ordinal, (family_id, cluster_ids) in enumerate(
        bootstrap_config.family_example_cluster_ids
    ):
        cluster_count = len(cluster_ids)
        counts = family_counts[family_ordinal]
        family_sse = np.zeros(
            (cluster_count, size, size), dtype=np.float64
        )
        family_reference_energy = np.zeros_like(family_sse)
        for cluster_index, cluster_id in enumerate(cluster_ids):
            for reference_index, reference_id in enumerate(axis):
                for candidate_index, candidate_id in enumerate(axis):
                    if reference_index == candidate_index:
                        family_reference_energy[
                            cluster_index, reference_index, candidate_index
                        ] = 1.0
                        continue
                    row = example_stats_by_key[
                        (family_id, cluster_id, reference_id, candidate_id)
                    ]
                    family_sse[
                        cluster_index, reference_index, candidate_index
                    ] = float(row.weighted_sse)
                    family_reference_energy[
                        cluster_index, reference_index, candidate_index
                    ] = float(row.weighted_reference_energy)
        pooled_sse = np.einsum(
            "rc,cij->rij", counts, family_sse, optimize=True
        )
        pooled_reference_energy = np.einsum(
            "rc,cij->rij", counts, family_reference_energy, optimize=True
        )
        directed_batch += pooled_sse / np.maximum(
            pooled_reference_energy, config.energy_epsilon
        )
    del family_counts

    directed_batch /= len(bootstrap_config.family_example_cluster_ids)
    symmetric_batch = 0.5 * (
        directed_batch + np.swapaxes(directed_batch, 1, 2)
    )
    diagonal = np.arange(size)
    symmetric_batch[:, diagonal, diagonal] = 0.0
    if not np.isfinite(symmetric_batch).all() or (
        symmetric_batch < 0.0
    ).any():
        raise ValueError("bootstrap distance matrices must be finite and nonnegative")

    del directed_batch
    base_edges = _adjacencies(base_order)
    base_nearest = {
        mode_id: neighbor for mode_id, neighbor, _ in base_nearest_neighbors
    }
    axis_index = {mode_id: index for index, mode_id in enumerate(axis)}
    adjacency_counts = np.zeros((size, size), dtype=np.float64)
    nearest_counts = np.zeros((size, size), dtype=np.float64)
    adjacency_retentions: list[float] = []
    nearest_retentions: list[float] = []
    degenerate_count = 0
    order_catalog_digest = hashlib.sha256()
    order_catalog_digest.update(_BOOTSTRAP_ORDER_DIGEST_DOMAIN)
    order_catalog_digest.update(
        _canonical_json_bytes(bootstrap_config.canonical_payload())
    )
    for resample_index in range(resample_count):
        distance = symmetric_batch[resample_index]
        try:
            spectral = _spectral_ordering(
                distance=distance,
                axis=axis,
                state_fingerprints=state_fingerprints,
                config=config,
            )
            nearest = _nearest_neighbors(
                distance=distance,
                axis=axis,
                tie_tolerance=config.nearest_neighbor_tie_tolerance,
            )
        except _SeriationDegeneracy:
            degenerate_count += 1
            order_catalog_digest.update(
                _canonical_json_bytes(
                    {
                        "degenerate": True,
                        "resample_index": resample_index,
                    }
                )
            )
            order_catalog_digest.update(b"\n")
            continue
        order_catalog_digest.update(
            _canonical_json_bytes(
                {
                    "degenerate": False,
                    "resample_index": resample_index,
                    "seriation_order": list(spectral.order),
                }
            )
        )
        order_catalog_digest.update(b"\n")
        edges = _adjacencies(spectral.order)
        adjacency_retentions.append(len(base_edges & edges) / len(base_edges))
        nearest_map = {
            mode_id: neighbor for mode_id, neighbor, _ in nearest
        }
        nearest_retentions.append(
            sum(
                nearest_map[mode_id] == base_nearest[mode_id]
                for mode_id in axis
            )
            / size
        )
        for edge in edges:
            left, right = tuple(edge)
            left_index = axis_index[left]
            right_index = axis_index[right]
            adjacency_counts[left_index, right_index] += 1.0
            adjacency_counts[right_index, left_index] += 1.0
        for mode_id, neighbor_id, _ in nearest:
            nearest_counts[axis_index[mode_id], axis_index[neighbor_id]] += 1.0

    maximum_degenerate_count = math.floor(
        bootstrap_config.maximum_degenerate_fraction * resample_count + 1e-12
    )
    if degenerate_count > maximum_degenerate_count:
        raise ValueError(
            "within-family bootstrap exceeded maximum degenerate fraction "
            f"({degenerate_count}/{resample_count})"
        )
    valid_count = resample_count - degenerate_count
    if valid_count <= 0:
        raise ValueError("within-family bootstrap has no valid seriations")
    adjacency_counts /= valid_count
    nearest_counts /= valid_count
    adjacency_values = np.asarray(adjacency_retentions, dtype=np.float64)
    nearest_values = np.asarray(nearest_retentions, dtype=np.float64)
    return _WithinFamilyBootstrapSummary(
        draw_catalog_sha256=draw_catalog_digest.hexdigest(),
        order_catalog_sha256=order_catalog_digest.hexdigest(),
        valid_resample_count=valid_count,
        degenerate_resample_count=degenerate_count,
        degenerate_fraction=degenerate_count / resample_count,
        adjacency_frequency=_matrix_tuple(adjacency_counts),
        nearest_neighbor_frequency=_matrix_tuple(nearest_counts),
        mean_adjacency_retention=float(np.mean(adjacency_values)),
        minimum_adjacency_retention=float(np.min(adjacency_values)),
        p05_adjacency_retention=float(
            np.quantile(adjacency_values, 0.05, method="linear")
        ),
        mean_nearest_neighbor_retention=float(np.mean(nearest_values)),
        minimum_nearest_neighbor_retention=float(np.min(nearest_values)),
        p05_nearest_neighbor_retention=float(
            np.quantile(nearest_values, 0.05, method="linear")
        ),
    )


def build_mode_similarity_atlas_receipt(
    modes: Sequence[ModeSpec],
    directed_stats: Sequence[DirectedFamilyStats],
    *,
    config: AtlasConfig | None = None,
    frozen_resamples: Sequence[FrozenFamilyResample] = (),
    directed_example_stats: Sequence[DirectedFamilyExampleStats] = (),
    within_family_bootstrap_config: WithinFamilyBootstrapConfig | None = None,
) -> ModeSimilarityAtlasReceipt:
    """Build one deterministic, same-ABI mode similarity atlas receipt."""

    if config is None:
        config = AtlasConfig()
    if not isinstance(config, AtlasConfig):
        raise TypeError("config must be an AtlasConfig")
    mode_tuple = tuple(modes)
    if len(mode_tuple) < 3:
        raise ValueError("an atlas requires at least three modes")
    if any(not isinstance(mode, ModeSpec) for mode in mode_tuple):
        raise TypeError("modes must contain only ModeSpec values")
    mode_ids = [mode.mode_id for mode in mode_tuple]
    if len(set(mode_ids)) != len(mode_ids):
        raise ValueError("mode IDs must be unique")
    fingerprints = [mode.state_fingerprint_sha256 for mode in mode_tuple]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("mode state fingerprints must be unique")
    abi_sha256s = {mode.compatibility_abi.sha256 for mode in mode_tuple}
    if len(abi_sha256s) != 1:
        raise ValueError(
            "cross-comparison requires one exact compatibility ABI"
        )
    compatibility_abi_sha256 = next(iter(abi_sha256s))
    mode_tuple = tuple(
        sorted(mode_tuple, key=lambda mode: mode.state_fingerprint_sha256)
    )
    axis = tuple(mode.mode_id for mode in mode_tuple)
    mode_id_set = set(axis)
    state_fingerprints = {
        mode.mode_id: mode.state_fingerprint_sha256 for mode in mode_tuple
    }

    stats_tuple = tuple(directed_stats)
    if not stats_tuple:
        raise ValueError("directed_stats must not be empty")
    if any(not isinstance(row, DirectedFamilyStats) for row in stats_tuple):
        raise TypeError(
            "directed_stats must contain only DirectedFamilyStats values"
        )
    stats_by_key: dict[tuple[str, str, str], DirectedFamilyStats] = {}
    family_ids: set[str] = set()
    reference_invariants: dict[tuple[str, str], tuple[int, float]] = {}
    for row in stats_tuple:
        if row.reference_mode_id not in mode_id_set:
            raise ValueError("statistics reference an unknown reference mode")
        if row.candidate_mode_id not in mode_id_set:
            raise ValueError("statistics reference an unknown candidate mode")
        if row.compatibility_abi_sha256 != compatibility_abi_sha256:
            raise ValueError("statistics compatibility ABI does not match modes")
        _validate_sufficient_statistics(
            row,
            tolerance=config.sufficient_stat_relative_tolerance,
        )
        key = (row.family_id, row.reference_mode_id, row.candidate_mode_id)
        if key in stats_by_key:
            raise ValueError("duplicate directed family statistics")
        stats_by_key[key] = row
        family_ids.add(row.family_id)
        invariant_key = (row.family_id, row.reference_mode_id)
        invariant = reference_invariants.get(invariant_key)
        current = (row.row_count, float(row.weighted_reference_energy))
        if invariant is None:
            reference_invariants[invariant_key] = current
        elif invariant[0] != current[0] or not _close(
            invariant[1],
            current[1],
            tolerance=config.sufficient_stat_relative_tolerance,
        ):
            raise ValueError(
                "reference row count and energy must be candidate-invariant"
            )

    if (
        within_family_bootstrap_config is not None
        and not isinstance(
            within_family_bootstrap_config, WithinFamilyBootstrapConfig
        )
    ):
        raise TypeError(
            "within_family_bootstrap_config must be a "
            "WithinFamilyBootstrapConfig"
        )
    if within_family_bootstrap_config is None:
        family_axis = tuple(sorted(family_ids))
    else:
        family_axis = within_family_bootstrap_config.ordered_family_ids
        if set(family_axis) != family_ids:
            raise ValueError(
                "bootstrap family catalog must exactly match aggregate families"
            )
    if len(family_axis) < 2:
        raise ValueError(
            "at least two families are required for leave-one-family-out"
        )
    expected_keys = {
        (family_id, reference_id, candidate_id)
        for family_id in family_axis
        for reference_id in axis
        for candidate_id in axis
        if reference_id != candidate_id
    }
    actual_keys = set(stats_by_key)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        extra = len(actual_keys - expected_keys)
        raise ValueError(
            "directed statistics must form a complete family-by-mode matrix "
            f"(missing={missing}, extra={extra})"
        )
    stats_tuple = tuple(
        sorted(
            stats_tuple,
            key=lambda row: (
                row.family_id,
                state_fingerprints[row.reference_mode_id],
                state_fingerprints[row.candidate_mode_id],
            ),
        )
    )

    resample_tuple = tuple(frozen_resamples)
    if any(
        not isinstance(resample, FrozenFamilyResample)
        for resample in resample_tuple
    ):
        raise TypeError(
            "frozen_resamples must contain only FrozenFamilyResample values"
        )
    if len({item.resample_id for item in resample_tuple}) != len(resample_tuple):
        raise ValueError("frozen resample IDs must be unique")
    family_id_set = set(family_axis)
    for resample in resample_tuple:
        resample_families = {
            family_id for family_id, _ in resample.family_multiplicities
        }
        unknown = resample_families - family_id_set
        if unknown:
            raise ValueError(
                f"frozen resample references unknown families {sorted(unknown)}"
            )
    resample_tuple = tuple(
        sorted(resample_tuple, key=lambda item: item.resample_id)
    )

    example_stats_tuple = tuple(directed_example_stats)
    if any(
        not isinstance(row, DirectedFamilyExampleStats)
        for row in example_stats_tuple
    ):
        raise TypeError(
            "directed_example_stats must contain only "
            "DirectedFamilyExampleStats values"
        )
    if within_family_bootstrap_config is not None and not example_stats_tuple:
        raise ValueError(
            "within-family bootstrap requires directed example statistics"
        )
    example_stats_by_key: dict[
        tuple[str, str, str, str], DirectedFamilyExampleStats
    ] = {}
    example_reference_invariants: dict[
        tuple[str, str, str], tuple[int, float]
    ] = {}
    cluster_ids_by_family: dict[str, set[str]] = {
        family_id: set() for family_id in family_axis
    }
    for row in example_stats_tuple:
        if row.family_id not in family_id_set:
            raise ValueError("example statistics reference an unknown family")
        if row.reference_mode_id not in mode_id_set:
            raise ValueError(
                "example statistics reference an unknown reference mode"
            )
        if row.candidate_mode_id not in mode_id_set:
            raise ValueError(
                "example statistics reference an unknown candidate mode"
            )
        if row.compatibility_abi_sha256 != compatibility_abi_sha256:
            raise ValueError(
                "example statistics compatibility ABI does not match modes"
            )
        family_row = row.as_family_stats()
        _validate_sufficient_statistics(
            family_row,
            tolerance=config.sufficient_stat_relative_tolerance,
        )
        key = (
            row.family_id,
            row.example_cluster_id,
            row.reference_mode_id,
            row.candidate_mode_id,
        )
        if key in example_stats_by_key:
            raise ValueError("duplicate directed family example statistics")
        example_stats_by_key[key] = row
        cluster_ids_by_family[row.family_id].add(row.example_cluster_id)
        invariant_key = (
            row.family_id,
            row.example_cluster_id,
            row.reference_mode_id,
        )
        invariant = example_reference_invariants.get(invariant_key)
        current = (row.row_count, float(row.weighted_reference_energy))
        if invariant is None:
            example_reference_invariants[invariant_key] = current
        elif invariant[0] != current[0] or not _close(
            invariant[1],
            current[1],
            tolerance=config.sufficient_stat_relative_tolerance,
        ):
            raise ValueError(
                "example reference row count and energy must be "
                "candidate-invariant"
            )

    if example_stats_tuple:
        if any(not cluster_ids for cluster_ids in cluster_ids_by_family.values()):
            raise ValueError("every family must have example statistics")
        expected_example_keys = {
            (family_id, cluster_id, reference_id, candidate_id)
            for family_id, cluster_ids in cluster_ids_by_family.items()
            for cluster_id in cluster_ids
            for reference_id in axis
            for candidate_id in axis
            if reference_id != candidate_id
        }
        if set(example_stats_by_key) != expected_example_keys:
            missing = len(expected_example_keys - set(example_stats_by_key))
            extra = len(set(example_stats_by_key) - expected_example_keys)
            raise ValueError(
                "example statistics must form a complete "
                "family-by-example-by-mode matrix "
                f"(missing={missing}, extra={extra})"
            )
        for family_id in family_axis:
            for reference_id in axis:
                for candidate_id in axis:
                    if reference_id == candidate_id:
                        continue
                    aggregate = stats_by_key[
                        (family_id, reference_id, candidate_id)
                    ]
                    component_rows = [
                        example_stats_by_key[
                            (
                                family_id,
                                cluster_id,
                                reference_id,
                                candidate_id,
                            )
                        ]
                        for cluster_id in cluster_ids_by_family[family_id]
                    ]
                    summed_row_count = sum(
                        row.row_count for row in component_rows
                    )
                    if summed_row_count != aggregate.row_count:
                        raise ValueError(
                            "family aggregate row_count does not equal its "
                            "example-cluster sum"
                        )
                    for field in (
                        "weighted_sse",
                        "weighted_reference_energy",
                        "weighted_dot",
                        "weighted_candidate_energy",
                    ):
                        component_sum = math.fsum(
                            float(getattr(row, field)) for row in component_rows
                        )
                        if not _close(
                            component_sum,
                            float(getattr(aggregate, field)),
                            tolerance=config.sufficient_stat_relative_tolerance,
                        ):
                            raise ValueError(
                                f"family aggregate {field} does not equal its "
                                "example-cluster sum"
                            )

    if within_family_bootstrap_config is not None:
        configured_family_ids = (
            within_family_bootstrap_config.ordered_family_ids
        )
        if set(configured_family_ids) != family_id_set:
            raise ValueError(
                "bootstrap family catalog must exactly match aggregate families"
            )
        for family_id, configured_cluster_ids in (
            within_family_bootstrap_config.family_example_cluster_ids
        ):
            if set(configured_cluster_ids) != cluster_ids_by_family[family_id]:
                raise ValueError(
                    "bootstrap example cluster catalog must exactly match "
                    "example statistics"
                )
    example_stats_tuple = tuple(
        sorted(
            example_stats_tuple,
            key=lambda row: (
                row.family_id,
                row.example_cluster_id,
                state_fingerprints[row.reference_mode_id],
                state_fingerprints[row.candidate_mode_id],
            ),
        )
    )

    full_multiplicities = {family_id: 1 for family_id in family_axis}
    directed, directed_cosine, symmetric, symmetric_cosine = (
        _distance_matrices(
            axis=axis,
            stats_by_key=stats_by_key,
            family_multiplicities=full_multiplicities,
            config=config,
        )
    )
    spectral = _spectral_ordering(
        distance=symmetric,
        axis=axis,
        state_fingerprints=state_fingerprints,
        config=config,
    )
    nearest_neighbors = _nearest_neighbors(
        distance=symmetric,
        axis=axis,
        tie_tolerance=config.nearest_neighbor_tie_tolerance,
    )
    (
        fiedler_path_cost,
        random_path_cost_mean,
        random_path_cost_std,
        random_path_cost_minimum,
        random_path_cost_maximum,
        fiedler_to_random_mean_cost_ratio,
        fiedler_random_percentile_fraction,
        fiedler_random_improvement_z,
    ) = _ordering_quality(
        distance=symmetric,
        axis=axis,
        order=spectral.order,
    )

    lofo_specs = tuple(
        (
            f"leave-out-{family_id}",
            family_id,
            tuple(
                (included_family_id, 1)
                for included_family_id in family_axis
                if included_family_id != family_id
            ),
        )
        for family_id in family_axis
    )
    lofo = _stability_summary(
        specifications=lofo_specs,
        axis=axis,
        state_fingerprints=state_fingerprints,
        stats_by_key=stats_by_key,
        config=config,
        base_order=spectral.order,
        base_nearest_neighbors=nearest_neighbors,
    )
    frozen_specs = tuple(
        (
            resample.resample_id,
            None,
            tuple(sorted(resample.family_multiplicities)),
        )
        for resample in resample_tuple
    )
    frozen_stability = _stability_summary(
        specifications=frozen_specs,
        axis=axis,
        state_fingerprints=state_fingerprints,
        stats_by_key=stats_by_key,
        config=config,
        base_order=spectral.order,
        base_nearest_neighbors=nearest_neighbors,
    )
    within_family_bootstrap = (
        None
        if within_family_bootstrap_config is None
        else _within_family_bootstrap_summary(
            bootstrap_config=within_family_bootstrap_config,
            axis=axis,
            state_fingerprints=state_fingerprints,
            example_stats_by_key=example_stats_by_key,
            config=config,
            base_order=spectral.order,
            base_nearest_neighbors=nearest_neighbors,
        )
    )
    analysis = _AtlasAnalysis(
        mode_axis=axis,
        directed_relative_sse=_matrix_tuple(directed),
        directed_cosine_similarity=_matrix_tuple(directed_cosine),
        symmetric_distance=_matrix_tuple(symmetric),
        symmetric_cosine_distance=_matrix_tuple(symmetric_cosine),
        spectral=spectral,
        fiedler_path_cost=fiedler_path_cost,
        random_path_cost_mean=random_path_cost_mean,
        random_path_cost_std=random_path_cost_std,
        random_path_cost_minimum=random_path_cost_minimum,
        random_path_cost_maximum=random_path_cost_maximum,
        fiedler_to_random_mean_cost_ratio=(
            fiedler_to_random_mean_cost_ratio
        ),
        fiedler_random_percentile_fraction=(
            fiedler_random_percentile_fraction
        ),
        fiedler_random_improvement_z=fiedler_random_improvement_z,
        nearest_neighbors=nearest_neighbors,
        leave_one_family_out=lofo,
        frozen_resamples=frozen_stability,
        within_family_bootstrap=within_family_bootstrap,
    )
    input_statistics_sha256 = _domain_digest(
        _STATS_DIGEST_DOMAIN,
        {
            "directed_family_example_stats": [
                row.canonical_payload() for row in example_stats_tuple
            ],
            "directed_family_stats": [
                row.canonical_payload() for row in stats_tuple
            ],
        },
    )
    provisional = ModeSimilarityAtlasReceipt(
        modes=mode_tuple,
        directed_stats=stats_tuple,
        directed_example_stats=example_stats_tuple,
        config=config,
        frozen_resamples=resample_tuple,
        within_family_bootstrap_config=within_family_bootstrap_config,
        family_ids=family_axis,
        compatibility_abi_sha256=compatibility_abi_sha256,
        input_statistics_sha256=input_statistics_sha256,
        analysis=analysis,
        receipt_sha256="0" * 64,
    )
    receipt_sha256 = _domain_digest(
        _RECEIPT_DIGEST_DOMAIN,
        provisional._payload_without_receipt_sha256(),
    )
    return ModeSimilarityAtlasReceipt(
        modes=provisional.modes,
        directed_stats=provisional.directed_stats,
        directed_example_stats=provisional.directed_example_stats,
        config=provisional.config,
        frozen_resamples=provisional.frozen_resamples,
        within_family_bootstrap_config=(
            provisional.within_family_bootstrap_config
        ),
        family_ids=provisional.family_ids,
        compatibility_abi_sha256=provisional.compatibility_abi_sha256,
        input_statistics_sha256=provisional.input_statistics_sha256,
        analysis=provisional.analysis,
        receipt_sha256=receipt_sha256,
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


def _parse_abi(value: object, *, label: str) -> ModeABI:
    raw = _require_exact_fields(
        value,
        {
            "abi_name",
            "abi_version",
            "context_signature",
            "input_signature",
            "output_signature",
            "sha256",
        },
        label=label,
    )
    abi = ModeABI(
        abi_name=_require_name(raw["abi_name"], label=f"{label}.abi_name"),
        abi_version=_require_positive_int(
            raw["abi_version"], label=f"{label}.abi_version"
        ),
        input_signature=_require_name(
            raw["input_signature"], label=f"{label}.input_signature"
        ),
        output_signature=_require_name(
            raw["output_signature"], label=f"{label}.output_signature"
        ),
        context_signature=_require_name(
            raw["context_signature"], label=f"{label}.context_signature"
        ),
    )
    if raw["sha256"] != abi.sha256:
        raise ValueError(f"{label}.sha256 does not authenticate the ABI")
    return abi


def _parse_mode(value: object, *, index: int) -> ModeSpec:
    label = f"mode_specs[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "compatibility_abi",
            "mode_id",
            "mode_kind",
            "state_fingerprint_sha256",
        },
        label=label,
    )
    return ModeSpec(
        mode_id=_require_name(raw["mode_id"], label=f"{label}.mode_id"),
        mode_kind=_require_name(
            raw["mode_kind"], label=f"{label}.mode_kind"
        ),
        state_fingerprint_sha256=_require_sha256(
            raw["state_fingerprint_sha256"],
            label=f"{label}.state_fingerprint_sha256",
        ),
        compatibility_abi=_parse_abi(
            raw["compatibility_abi"], label=f"{label}.compatibility_abi"
        ),
    )


def _parse_stats(value: object, *, index: int) -> DirectedFamilyStats:
    label = f"directed_family_stats[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "candidate_mode_id",
            "compatibility_abi_sha256",
            "family_id",
            "reference_mode_id",
            "row_count",
            "weighted_candidate_energy",
            "weighted_dot",
            "weighted_reference_energy",
            "weighted_sse",
        },
        label=label,
    )
    return DirectedFamilyStats(
        family_id=_require_name(
            raw["family_id"], label=f"{label}.family_id"
        ),
        reference_mode_id=_require_name(
            raw["reference_mode_id"],
            label=f"{label}.reference_mode_id",
        ),
        candidate_mode_id=_require_name(
            raw["candidate_mode_id"],
            label=f"{label}.candidate_mode_id",
        ),
        compatibility_abi_sha256=_require_sha256(
            raw["compatibility_abi_sha256"],
            label=f"{label}.compatibility_abi_sha256",
        ),
        weighted_sse=_require_finite(
            raw["weighted_sse"],
            label=f"{label}.weighted_sse",
            minimum=0.0,
        ),
        weighted_reference_energy=_require_finite(
            raw["weighted_reference_energy"],
            label=f"{label}.weighted_reference_energy",
            minimum=0.0,
        ),
        weighted_dot=_require_finite(
            raw["weighted_dot"], label=f"{label}.weighted_dot"
        ),
        weighted_candidate_energy=_require_finite(
            raw["weighted_candidate_energy"],
            label=f"{label}.weighted_candidate_energy",
            minimum=0.0,
        ),
        row_count=_require_positive_int(
            raw["row_count"], label=f"{label}.row_count"
        ),
    )


def _parse_example_stats(
    value: object,
    *,
    index: int,
) -> DirectedFamilyExampleStats:
    label = f"directed_family_example_stats[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "candidate_mode_id",
            "compatibility_abi_sha256",
            "example_cluster_id",
            "family_id",
            "reference_mode_id",
            "row_count",
            "weighted_candidate_energy",
            "weighted_dot",
            "weighted_reference_energy",
            "weighted_sse",
        },
        label=label,
    )
    return DirectedFamilyExampleStats(
        family_id=_require_name(
            raw["family_id"], label=f"{label}.family_id"
        ),
        example_cluster_id=_require_name(
            raw["example_cluster_id"],
            label=f"{label}.example_cluster_id",
        ),
        reference_mode_id=_require_name(
            raw["reference_mode_id"],
            label=f"{label}.reference_mode_id",
        ),
        candidate_mode_id=_require_name(
            raw["candidate_mode_id"],
            label=f"{label}.candidate_mode_id",
        ),
        compatibility_abi_sha256=_require_sha256(
            raw["compatibility_abi_sha256"],
            label=f"{label}.compatibility_abi_sha256",
        ),
        weighted_sse=_require_finite(
            raw["weighted_sse"],
            label=f"{label}.weighted_sse",
            minimum=0.0,
        ),
        weighted_reference_energy=_require_finite(
            raw["weighted_reference_energy"],
            label=f"{label}.weighted_reference_energy",
            minimum=0.0,
        ),
        weighted_dot=_require_finite(
            raw["weighted_dot"], label=f"{label}.weighted_dot"
        ),
        weighted_candidate_energy=_require_finite(
            raw["weighted_candidate_energy"],
            label=f"{label}.weighted_candidate_energy",
            minimum=0.0,
        ),
        row_count=_require_positive_int(
            raw["row_count"], label=f"{label}.row_count"
        ),
    )
def _parse_config(value: object) -> AtlasConfig:
    raw = _require_exact_fields(
        value,
        {
            "affinity_kernel",
            "affinity_max_exponent",
            "connectivity_tolerance",
            "cosine_role",
            "distance_metric",
            "energy_epsilon",
            "fiedler_coordinate_tie_tolerance",
            "fiedler_eigengap_relative_tolerance",
            "laplacian",
            "mode_id_role",
            "mode_kind_role",
            "nearest_neighbor_tie_tolerance",
            "ordering_quality_random_permutation_count",
            "ordering_quality_random_permutation_seed",
            "state_fingerprint_role",
            "sufficient_stat_relative_tolerance",
        },
        label="atlas_config",
    )
    fixed = {
        "affinity_kernel": "exp_neg_distance_over_positive_median_v1",
        "cosine_role": "diagnostic_only",
        "distance_metric": "bidirectional_equal_family_mean_relative_sse_v1",
        "laplacian": "symmetric_normalized_v1",
        "mode_id_role": "receipt_label_only",
        "mode_kind_role": "receipt_label_only",
        "ordering_quality_random_permutation_count": (
            _RANDOM_PERMUTATION_BASELINE_COUNT
        ),
        "ordering_quality_random_permutation_seed": (
            _RANDOM_PERMUTATION_BASELINE_SEED
        ),
        "state_fingerprint_role": "canonical_axis_and_global_reversal_only",
    }
    for field, expected in fixed.items():
        if raw[field] != expected:
            raise ValueError(f"atlas_config.{field} is unsupported")
    return AtlasConfig(
        energy_epsilon=_require_finite(
            raw["energy_epsilon"],
            label="atlas_config.energy_epsilon",
            strictly_positive=True,
        ),
        sufficient_stat_relative_tolerance=_require_finite(
            raw["sufficient_stat_relative_tolerance"],
            label="atlas_config.sufficient_stat_relative_tolerance",
            strictly_positive=True,
        ),
        connectivity_tolerance=_require_finite(
            raw["connectivity_tolerance"],
            label="atlas_config.connectivity_tolerance",
            strictly_positive=True,
        ),
        fiedler_eigengap_relative_tolerance=_require_finite(
            raw["fiedler_eigengap_relative_tolerance"],
            label="atlas_config.fiedler_eigengap_relative_tolerance",
            strictly_positive=True,
        ),
        fiedler_coordinate_tie_tolerance=_require_finite(
            raw["fiedler_coordinate_tie_tolerance"],
            label="atlas_config.fiedler_coordinate_tie_tolerance",
            strictly_positive=True,
        ),
        nearest_neighbor_tie_tolerance=_require_finite(
            raw["nearest_neighbor_tie_tolerance"],
            label="atlas_config.nearest_neighbor_tie_tolerance",
            strictly_positive=True,
        ),
        affinity_max_exponent=_require_finite(
            raw["affinity_max_exponent"],
            label="atlas_config.affinity_max_exponent",
            strictly_positive=True,
        ),
    )


def _parse_resample(value: object, *, index: int) -> FrozenFamilyResample:
    label = f"frozen_family_resamples[{index}]"
    raw = _require_exact_fields(
        value,
        {"family_multiplicities", "resample_id"},
        label=label,
    )
    multiplicities_raw = raw["family_multiplicities"]
    if not isinstance(multiplicities_raw, list):
        raise TypeError(f"{label}.family_multiplicities must be a list")
    multiplicities: list[tuple[str, int]] = []
    for item_index, item in enumerate(multiplicities_raw):
        item_label = f"{label}.family_multiplicities[{item_index}]"
        item_raw = _require_exact_fields(
            item,
            {"family_id", "multiplicity"},
            label=item_label,
        )
        multiplicities.append(
            (
                _require_name(
                    item_raw["family_id"], label=f"{item_label}.family_id"
                ),
                _require_positive_int(
                    item_raw["multiplicity"],
                    label=f"{item_label}.multiplicity",
                ),
            )
        )
    return FrozenFamilyResample(
        resample_id=_require_name(
            raw["resample_id"], label=f"{label}.resample_id"
        ),
        family_multiplicities=tuple(multiplicities),
    )


def _parse_within_family_bootstrap_config(
    value: object,
) -> WithinFamilyBootstrapConfig:
    raw = _require_exact_fields(
        value,
        {
            "draw_protocol",
            "draw_catalog_digest_encoding",
            "draw_catalog_digest_domain_hex",
            "draw_digest_selection",
            "draw_preimage_encoding",
            "draw_protocol_domain_hex",
            "draws_per_family",
            "families",
            "maximum_degenerate_fraction",
            "resample_count",
            "sampling_unit",
            "seed",
            "with_replacement",
        },
        label="within_family_bootstrap_config",
    )
    fixed = {
        "draw_catalog_digest_encoding": (
            "sha256(catalog_domain||canonical_config||"
            "repeated(resample_u64be,family_u32be,draw_u32be,"
            "example_u32be))"
        ),
        "draw_catalog_digest_domain_hex": (
            _BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
        ),
        "draw_digest_selection": (
            "first_8_sha256_bytes_u64be_mod_example_count"
        ),
        "draw_preimage_encoding": (
            "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
        ),
        "draw_protocol": "sha256_counter_modulo_example_count_v1",
        "sampling_unit": "example_plus_selected_probes_cluster",
        "with_replacement": True,
    }
    for field, expected in fixed.items():
        if raw[field] != expected:
            raise ValueError(
                f"within_family_bootstrap_config.{field} is unsupported"
            )
    families_raw = raw["families"]
    if not isinstance(families_raw, list):
        raise TypeError("within_family_bootstrap_config.families must be a list")
    families: list[tuple[str, tuple[str, ...]]] = []
    for family_index, family_value in enumerate(families_raw):
        family_label = (
            f"within_family_bootstrap_config.families[{family_index}]"
        )
        family_raw = _require_exact_fields(
            family_value,
            {"example_cluster_ids", "family_id"},
            label=family_label,
        )
        cluster_ids_raw = family_raw["example_cluster_ids"]
        if not isinstance(cluster_ids_raw, list):
            raise TypeError(f"{family_label}.example_cluster_ids must be a list")
        families.append(
            (
                _require_name(
                    family_raw["family_id"], label=f"{family_label}.family_id"
                ),
                tuple(
                    _require_name(
                        cluster_id,
                        label=(
                            f"{family_label}.example_cluster_ids[{cluster_index}]"
                        ),
                    )
                    for cluster_index, cluster_id in enumerate(cluster_ids_raw)
                ),
            )
        )
    seed = raw["seed"]
    if type(seed) is not int:
        raise ValueError("within_family_bootstrap_config.seed must be an integer")
    domain_hex = raw["draw_protocol_domain_hex"]
    if not isinstance(domain_hex, str) or _HEX_BYTES.fullmatch(domain_hex) is None:
        raise ValueError(
            "within_family_bootstrap_config.draw_protocol_domain_hex must be "
            "canonical lowercase hex"
        )
    return WithinFamilyBootstrapConfig(
        seed=seed,
        resample_count=_require_positive_int(
            raw["resample_count"],
            label="within_family_bootstrap_config.resample_count",
        ),
        draws_per_family=_require_positive_int(
            raw["draws_per_family"],
            label="within_family_bootstrap_config.draws_per_family",
        ),
        maximum_degenerate_fraction=_require_finite(
            raw["maximum_degenerate_fraction"],
            label=(
                "within_family_bootstrap_config.maximum_degenerate_fraction"
            ),
            minimum=0.0,
        ),
        family_example_cluster_ids=tuple(families),
        draw_protocol_domain=bytes.fromhex(domain_hex),
    )


def parse_atlas_config_payload(value: object) -> AtlasConfig:
    """Strictly parse the canonical ``atlas_config`` setup mapping."""

    return _parse_config(value)


def parse_frozen_family_resample_payload(
    value: object,
) -> FrozenFamilyResample:
    """Strictly parse one canonical aggregate-family resample mapping."""

    return _parse_resample(value, index=0)


def parse_within_family_bootstrap_config_payload(
    value: object,
) -> WithinFamilyBootstrapConfig:
    """Strictly parse the canonical clustered-bootstrap setup mapping."""

    return _parse_within_family_bootstrap_config(value)


def _rebuild_from_payload(raw: Mapping[str, object]) -> ModeSimilarityAtlasReceipt:
    fields = {
        "analysis",
        "atlas_config",
        "compatibility_abi_sha256",
        "directed_family_example_stats",
        "directed_family_stats",
        "family_ids",
        "format_version",
        "frozen_family_resamples",
        "input_statistics_sha256",
        "mode_specs",
        "receipt_sha256",
        "safety",
        "schema",
        "scientific_status",
        "within_family_bootstrap_config",
    }
    _require_exact_fields(raw, fields, label="receipt")
    if raw["schema"] != MODE_SIMILARITY_ATLAS_SCHEMA:
        raise ValueError("unsupported mode similarity atlas schema")
    if raw["format_version"] != MODE_SIMILARITY_ATLAS_FORMAT_VERSION:
        raise ValueError("unsupported mode similarity atlas format version")
    if raw["scientific_status"] != _SCIENTIFIC_STATUS:
        raise ValueError("scientific_status is not the frozen contract")
    if raw["safety"] != _SAFETY:
        raise ValueError("safety is not the frozen contract")
    modes_raw = raw["mode_specs"]
    stats_raw = raw["directed_family_stats"]
    example_stats_raw = raw["directed_family_example_stats"]
    resamples_raw = raw["frozen_family_resamples"]
    if not isinstance(modes_raw, list):
        raise TypeError("mode_specs must be a list")
    if not isinstance(stats_raw, list):
        raise TypeError("directed_family_stats must be a list")
    if not isinstance(example_stats_raw, list):
        raise TypeError("directed_family_example_stats must be a list")
    if not isinstance(resamples_raw, list):
        raise TypeError("frozen_family_resamples must be a list")
    rebuilt = build_mode_similarity_atlas_receipt(
        tuple(_parse_mode(value, index=index) for index, value in enumerate(modes_raw)),
        tuple(
            _parse_stats(value, index=index)
            for index, value in enumerate(stats_raw)
        ),
        config=_parse_config(raw["atlas_config"]),
        frozen_resamples=tuple(
            _parse_resample(value, index=index)
            for index, value in enumerate(resamples_raw)
        ),
        directed_example_stats=tuple(
            _parse_example_stats(value, index=index)
            for index, value in enumerate(example_stats_raw)
        ),
        within_family_bootstrap_config=(
            None
            if raw["within_family_bootstrap_config"] is None
            else _parse_within_family_bootstrap_config(
                raw["within_family_bootstrap_config"]
            )
        ),
    )
    if _canonical_json_bytes(raw) != _canonical_json_bytes(
        rebuilt.canonical_payload()
    ):
        raise ValueError(
            "receipt does not exactly match deterministic reconstruction"
        )
    return rebuilt


def parse_mode_similarity_atlas_receipt_payload(
    value: Mapping[str, object],
) -> ModeSimilarityAtlasReceipt:
    """Strictly authenticate a canonical receipt mapping already in memory."""

    if not isinstance(value, Mapping):
        raise TypeError("mode similarity atlas receipt payload must be a mapping")
    return _rebuild_from_payload(value)


def save_mode_similarity_atlas_receipt(
    path: Path | str,
    receipt: ModeSimilarityAtlasReceipt,
) -> None:
    """Exclusively and atomically save one canonical JSON receipt."""

    if not isinstance(receipt, ModeSimilarityAtlasReceipt):
        raise TypeError("receipt must be a ModeSimilarityAtlasReceipt")
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("mode similarity atlas receipt must use .json")
    if output.is_symlink() or output.exists():
        raise FileExistsError("refusing to overwrite atlas receipt")
    rebuilt = build_mode_similarity_atlas_receipt(
        receipt.modes,
        receipt.directed_stats,
        config=receipt.config,
        frozen_resamples=receipt.frozen_resamples,
        directed_example_stats=receipt.directed_example_stats,
        within_family_bootstrap_config=(
            receipt.within_family_bootstrap_config
        ),
    )
    if _canonical_json_bytes(
        receipt.canonical_payload()
    ) != _canonical_json_bytes(rebuilt.canonical_payload()):
        raise ValueError("receipt is not a valid deterministic build product")
    payload = _canonical_json_bytes(receipt.canonical_payload()) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite atlas receipt") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_mode_similarity_atlas_receipt(
    path: Path | str,
) -> ModeSimilarityAtlasReceipt:
    """Strict-load, authenticate, and reconstruct one canonical receipt."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("refusing to load an atlas receipt through a symlink")
    raw_bytes = source.read_bytes()
    if not raw_bytes or len(raw_bytes) > _MAX_RECEIPT_BYTES:
        raise ValueError("atlas receipt has an invalid byte size")
    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("atlas receipt must be UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("atlas receipt must be a JSON object")
    if raw_bytes != _canonical_json_bytes(raw) + b"\n":
        raise ValueError("atlas receipt bytes are not canonical JSON")
    return parse_mode_similarity_atlas_receipt_payload(raw)
