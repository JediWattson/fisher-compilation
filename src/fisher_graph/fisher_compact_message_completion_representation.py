"""Pre-tune representation qualification for the synthetic CMC campaign.

The objects in this module turn the fitted synthetic selection geometry into
literal, hashable P4 measurements.  Only ``fit_train`` rows are consulted.
Tune and development-confirm tensors are deliberately outside every helper in
this file, and all results remain synthetic qualification evidence rather than
scientific or guard evidence.

The layer records more detail than the current external artifact schemas can
carry.  In particular, every tensor manifest is assembled from named,
dtype-sensitive, shape-sensitive identities, and the exact fitted matching
calipers, donor boundaries, coefficient epsilons, and analytic operator cells
remain available on the returned aggregate even where the external schema
contains only their commitment hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import struct
from typing import Final, Literal, Protocol

import numpy as np
import torch
from torch import Tensor

from .fisher_compact_message_completion_qualification import (
    CODEC_LADDER,
    FittedCodecCandidate,
    FittedLocator,
    FittedMessageSketch,
    O1Qualification,
    PrincipalRepresentation,
    SelectionGeometry,
    row_keys,
    subset_partition_by_units,
)
from .fisher_compact_message_completion_science import (
    HierarchicalRowKey,
    OperatorAggregateSummary,
    OperatorObservation,
    OperatorPairSummary,
    OperatorReferenceResult,
    ScientificContractError,
    aggregate_operator_pair_summaries,
    hierarchical_row_weights,
    positive_operator_reference,
    stable_rank,
    summarize_operator_pairs,
    weighted_linear_quantile,
    weighted_median,
)
from .fisher_compact_message_completion_synthetic import (
    SYNTHETIC_EVIDENCE_SCOPE,
    SyntheticCampaignData,
    SyntheticPartition,
)


__all__ = [
    "AnalyticOperatorCell",
    "CodecRepresentationQualification",
    "DirectedMatchingCaliperEstimate",
    "DonorQuintileBoundaries",
    "MatchingCaliperRecord",
    "MatchingCaliperTable",
    "NamedTensorIdentity",
    "LocatorAtlasEvidence",
    "PreTuneRepresentationQualification",
    "RepresentationQualificationError",
    "analytic_ambient_operator",
    "build_pre_tune_representation_qualification",
    "directed_matching_caliper",
    "identify_tensor",
    "tensor_manifest_sha256",
]


_ATTEMPTED_LOCATOR_K: Final = (4, 2)
_MATCHING_QUANTILE: Final = 0.25
_DONOR_QUANTILES: Final = (0.2, 0.4, 0.6, 0.8)
_TENSOR_IDENTITY_DOMAIN: Final = b"fisher-cmc/named-tensor-identity/v1"
_TENSOR_MANIFEST_DOMAIN: Final = b"fisher-cmc/named-tensor-manifest/v1"
_CANONICAL_JSON_DOMAIN: Final = b"fisher-cmc/canonical-json/v1"


class RepresentationQualificationError(RuntimeError):
    """A fitted object cannot form a valid frozen pre-tune representation."""


class _FullRankSmoke(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def frozen_tolerances_sha256(self) -> str: ...

    def as_artifact_body(self) -> dict[str, object]: ...


def _canonical_json_sha256(domain: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


def _array(value: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(value, Tensor):
        materialized = value.detach().to(device="cpu").contiguous().numpy()
    elif isinstance(value, np.ndarray):
        materialized = np.ascontiguousarray(value)
    else:  # pragma: no cover - public type guard
        raise TypeError("tensor identity input must be a numpy array or Tensor")
    if materialized.dtype.hasobject:
        raise RepresentationQualificationError("object tensors cannot be identified")
    if materialized.dtype.kind in {"f", "c"} and not bool(
        np.isfinite(materialized).all()
    ):
        raise RepresentationQualificationError("tensor identities require finite data")
    return materialized


@dataclass(frozen=True, slots=True)
class NamedTensorIdentity:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


def identify_tensor(name: str, value: np.ndarray | Tensor) -> NamedTensorIdentity:
    """Return a name-, dtype-, shape-, and byte-sensitive tensor identity."""

    if not isinstance(name, str) or not name:
        raise ValueError("tensor identity name must be a nonempty string")
    materialized = _array(value)
    dtype = str(materialized.dtype)
    shape = tuple(int(dimension) for dimension in materialized.shape)
    name_bytes = name.encode("utf-8")
    dtype_bytes = materialized.dtype.str.encode("ascii")
    digest = hashlib.sha256(_TENSOR_IDENTITY_DOMAIN + b"\0")
    digest.update(struct.pack(">I", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack(">I", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack(">I", len(shape)))
    for dimension in shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(materialized.tobytes(order="C"))
    return NamedTensorIdentity(
        name=name,
        dtype=dtype,
        shape=shape,
        byte_count=materialized.nbytes,
        sha256=digest.hexdigest(),
    )


def tensor_manifest_sha256(
    identities: Sequence[NamedTensorIdentity],
    *,
    namespace: str,
) -> str:
    """Commit an exact, canonically ordered inventory of named tensors."""

    rows = tuple(identities)
    if not rows or any(not isinstance(row, NamedTensorIdentity) for row in rows):
        raise RepresentationQualificationError(
            "tensor manifests require at least one named tensor identity"
        )
    if len({row.name for row in rows}) != len(rows):
        raise RepresentationQualificationError("tensor manifest names must be unique")
    ordered = sorted(rows, key=lambda row: row.name.encode("utf-8"))
    body = {
        "namespace": namespace,
        "tensors": [row.as_protocol_record() for row in ordered],
    }
    return _canonical_json_sha256(_TENSOR_MANIFEST_DOMAIN, body)


def _representation_identities(
    prefix: str,
    representation: PrincipalRepresentation,
) -> tuple[NamedTensorIdentity, ...]:
    return (
        identify_tensor(f"{prefix}.mean", representation.mean),
        identify_tensor(f"{prefix}.basis", representation.basis),
        identify_tensor(
            f"{prefix}.coordinate_mean", representation.coordinate_mean
        ),
        identify_tensor(
            f"{prefix}.coordinate_scale", representation.coordinate_scale
        ),
    )


def _locator_identities(locator: FittedLocator) -> tuple[NamedTensorIdentity, ...]:
    prefix = f"locator.{locator.atlas_name}"
    rows: list[NamedTensorIdentity] = list(
        _representation_identities(
            f"{prefix}.global", locator.global_representation
        )
    )
    rows.extend(
        (
            identify_tensor(f"{prefix}.centers", locator.centers),
            identify_tensor(
                f"{prefix}.sigma_perp_squared", locator.sigma_perp_squared
            ),
            identify_tensor(
                f"{prefix}.parallel_support_limits",
                locator.parallel_support_limits,
            ),
            identify_tensor(
                f"{prefix}.perpendicular_support_limits",
                locator.perpendicular_support_limits,
            ),
            identify_tensor(
                f"{prefix}.fit_train_unit_counts",
                np.asarray(locator.fit_train_unit_counts, dtype=np.int64),
            ),
            identify_tensor(
                f"{prefix}.runtime_fit_train_unit_counts",
                np.asarray(locator.runtime_fit_train_unit_counts, dtype=np.int64),
            ),
            identify_tensor(
                f"{prefix}.fit_monitor_row_counts",
                np.asarray(locator.fit_monitor_row_counts, dtype=np.int64),
            ),
        )
    )
    for chart, representation in enumerate(locator.local_representations):
        rows.extend(
            _representation_identities(
                f"{prefix}.chart{chart}.local", representation
            )
        )
        covariance = locator.support_covariances[chart]
        rows.extend(
            (
                identify_tensor(
                    f"{prefix}.chart{chart}.support_covariance",
                    covariance.symmetric_covariance,
                ),
                identify_tensor(
                    f"{prefix}.chart{chart}.support_pseudoinverse",
                    covariance.pseudoinverse,
                ),
                identify_tensor(
                    f"{prefix}.chart{chart}.support_active_eigenspace",
                    covariance.active,
                ),
            )
        )
    return tuple(rows)


def _message_sketch_identities(
    sketch: FittedMessageSketch,
) -> tuple[NamedTensorIdentity, ...]:
    rows: list[NamedTensorIdentity] = []
    for head, representation in enumerate(sketch.representations):
        rows.extend(_representation_identities(f"message_sketch.head{head}", representation))
    return tuple(rows)


def _codec_tensor_identities(
    candidate: FittedCodecCandidate,
) -> tuple[NamedTensorIdentity, ...]:
    rows: list[NamedTensorIdentity] = []
    for kind, codecs in (("q", candidate.q_codecs), ("v", candidate.v_codecs)):
        for chart, codec in enumerate(codecs):
            prefix = f"codec.{candidate.codec_id}.{kind}.chart{chart}"
            for field in (
                "carrier_vectors",
                "carrier_eigenvalues",
                "encoder",
                "decoder",
                "projector",
                "supported_fisher",
                "codec_eigenvalues",
            ):
                rows.append(identify_tensor(f"{prefix}.{field}", getattr(codec, field)))
            rows.append(
                identify_tensor(
                    f"{prefix}.regularization",
                    np.asarray([codec.regularization], dtype=np.float64),
                )
            )
    rows.extend(
        (
            identify_tensor(
                f"codec.{candidate.codec_id}.epsilon_completion",
                candidate.epsilon_completion_by_chart_head,
            ),
            identify_tensor(
                f"codec.{candidate.codec_id}.epsilon_J0",
                candidate.epsilon_j0_by_chart_head,
            ),
        )
    )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class DirectedMatchingCaliperEstimate:
    status: Literal["fitted", "invalid_no_eligible_pairs"]
    caliper: float | None
    contributing_target_count: int
    directed_pair_count: int
    contributing_target_weight: float
    normalized_pair_weight_sum: float


def directed_matching_caliper(
    standardized_coordinates: Tensor | np.ndarray,
    keys: Sequence[HierarchicalRowKey],
    row_weights: Tensor | Sequence[float],
) -> DirectedMatchingCaliperEstimate:
    """Fit F18/G7's directed, target-normalized q=.25 caliper."""

    coordinates = torch.as_tensor(
        standardized_coordinates, dtype=torch.float64, device="cpu"
    )
    resolved_keys = tuple(keys)
    weights = torch.as_tensor(row_weights, dtype=torch.float64, device="cpu").flatten()
    if coordinates.ndim != 2 or coordinates.shape[0] != len(resolved_keys):
        raise RepresentationQualificationError(
            "matching coordinates and row keys are not aligned"
        )
    if weights.shape != (len(resolved_keys),):
        raise RepresentationQualificationError(
            "matching row weights and row keys are not aligned"
        )
    if not bool(torch.isfinite(coordinates).all()) or not bool(
        torch.isfinite(weights).all()
    ):
        raise RepresentationQualificationError("matching inputs must be finite")
    if not bool((weights > 0).all()):
        raise RepresentationQualificationError(
            "matching row weights must be strictly positive"
        )

    distances: list[float] = []
    pair_weights: list[float] = []
    tie_keys: list[bytes] = []
    target_count = 0
    target_weight = 0.0
    for target, target_key in enumerate(resolved_keys):
        donors = [
            donor
            for donor, donor_key in enumerate(resolved_keys)
            if donor_key.analysis_unit_commitment
            != target_key.analysis_unit_commitment
            and donor_key.canonical_sequence_hash
            != target_key.canonical_sequence_hash
        ]
        if not donors:
            continue
        target_count += 1
        materialized_target_weight = float(weights[target])
        target_weight = math.fsum((target_weight, materialized_target_weight))
        per_pair_weight = materialized_target_weight / len(donors)
        for donor in donors:
            distances.append(
                float(torch.linalg.vector_norm(coordinates[target] - coordinates[donor]))
            )
            pair_weights.append(per_pair_weight)
            tie_keys.append(
                b"fisher-cmc/matching-pair/v1\0"
                + target_key.order_bytes
                + resolved_keys[donor].order_bytes
            )
    if not distances:
        return DirectedMatchingCaliperEstimate(
            status="invalid_no_eligible_pairs",
            caliper=None,
            contributing_target_count=0,
            directed_pair_count=0,
            contributing_target_weight=0.0,
            normalized_pair_weight_sum=0.0,
        )
    normalized = torch.tensor(pair_weights, dtype=torch.float64)
    normalized /= normalized.sum()
    caliper = weighted_linear_quantile(
        distances,
        normalized,
        _MATCHING_QUANTILE,
        tie_keys=tie_keys,
    )
    return DirectedMatchingCaliperEstimate(
        status="fitted",
        caliper=caliper,
        contributing_target_count=target_count,
        directed_pair_count=len(distances),
        contributing_target_weight=target_weight,
        normalized_pair_weight_sum=1.0,
    )


@dataclass(frozen=True, slots=True)
class MatchingCaliperRecord:
    chart_id: int
    head_id: int
    status: Literal["fitted", "invalid_no_eligible_pairs"]
    caliper: float | None
    fit_train_row_count: int
    fit_train_analysis_unit_count: int
    contributing_target_count: int
    directed_pair_count: int
    contributing_target_weight: float

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "chart_id": self.chart_id,
            "head_id": self.head_id,
            "status": self.status,
            "caliper": self.caliper,
            "quantile": _MATCHING_QUANTILE,
            "fit_train_row_count": self.fit_train_row_count,
            "fit_train_analysis_unit_count": self.fit_train_analysis_unit_count,
            "contributing_target_count": self.contributing_target_count,
            "directed_pair_count": self.directed_pair_count,
            "contributing_target_weight": self.contributing_target_weight,
            "pair_weight_rule": "target_weight_divided_by_eligible_donor_count",
        }


@dataclass(frozen=True, slots=True)
class MatchingCaliperTable:
    records: tuple[MatchingCaliperRecord, ...]

    @property
    def passed(self) -> bool:
        return bool(self.records) and all(
            row.status == "fitted" and row.caliper is not None for row in self.records
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            b"fisher-cmc/fit-train-matching-calipers/v1",
            self.as_protocol_body(),
        )

    def as_protocol_body(self) -> dict[str, object]:
        return {
            "fit_partition": "fit_train",
            "coordinate_system": "standardized_per_head_r",
            "directed_pairs": True,
            "quantile": _MATCHING_QUANTILE,
            "quantile_interpolation": "weighted_linear",
            "records": [row.as_protocol_record() for row in self.records],
            "all_calipers_fitted": self.passed,
        }


@dataclass(frozen=True, slots=True)
class DonorQuintileBoundaries:
    quantiles: tuple[float, ...]
    values: tuple[float, ...]
    fit_train_row_count: int
    fit_train_analysis_unit_count: int

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            b"fisher-cmc/fit-train-donor-quintiles/v1",
            self.as_protocol_body(),
        )

    def as_protocol_body(self) -> dict[str, object]:
        return {
            "fit_partition": "fit_train",
            "retained_message_norm": "sqrt(sum_head(||r_head||_2^2))",
            "quantiles": list(self.quantiles),
            "boundaries": list(self.values),
            "quantile_interpolation": "weighted_linear",
            "fit_train_row_count": self.fit_train_row_count,
            "fit_train_analysis_unit_count": self.fit_train_analysis_unit_count,
            "position_buckets": [
                "[0,8)",
                "[8,16)",
                "[16,32)",
                "[32,64)",
                "[64,128)",
                "[128,infinity)",
            ],
            "interaction_donor_seed": 271828,
        }


@dataclass(frozen=True, slots=True)
class LocatorAtlasEvidence:
    """The available result of the frozen K=4 then K=2 locator ladder."""

    atlas_name: str
    selected_k: int
    attempted_k_values: tuple[int, ...]
    lower_k_fallback_used: bool
    fit_train_unit_counts: tuple[int, ...]
    runtime_fit_train_unit_counts: tuple[int, ...]
    fit_monitor_row_counts: tuple[int, ...]
    kmeans_iterations: int
    kmeans_stopping_condition: str
    tensor_manifest_sha256: str

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "atlas_name": self.atlas_name,
            "selected_K": self.selected_k,
            "attempted_K_values": list(self.attempted_k_values),
            "lower_K_fallback_used": self.lower_k_fallback_used,
            "fit_train_unit_counts": list(self.fit_train_unit_counts),
            "runtime_fit_train_unit_counts": list(
                self.runtime_fit_train_unit_counts
            ),
            "fit_monitor_row_counts": list(self.fit_monitor_row_counts),
            "kmeans_iterations": self.kmeans_iterations,
            "kmeans_stopping_condition": self.kmeans_stopping_condition,
            "tensor_manifest_sha256": self.tensor_manifest_sha256,
        }


def _fit_train(campaign: SyntheticCampaignData) -> SyntheticPartition:
    return subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.right_units,
        partition_role="fit",
    )


def _fit_matching_calipers(
    fit_train: SyntheticPartition,
    geometry: SelectionGeometry,
) -> MatchingCaliperTable:
    keys = row_keys(fit_train)
    charts = geometry.locator.assign(fit_train.tensors["h"])
    retained = geometry.message_sketch.transform(fit_train.tensors["message"])
    records: list[MatchingCaliperRecord] = []
    for chart in range(geometry.locator.chart_count):
        mask = charts == chart
        indexes = torch.nonzero(mask, as_tuple=False).flatten()
        chart_keys = tuple(keys[int(index)] for index in indexes)
        if not chart_keys:
            for head in range(geometry.message_sketch.head_count):
                records.append(
                    MatchingCaliperRecord(
                        chart,
                        head,
                        "invalid_no_eligible_pairs",
                        None,
                        0,
                        0,
                        0,
                        0,
                        0.0,
                    )
                )
            continue
        weights = hierarchical_row_weights(chart_keys)
        unit_count = len({key.analysis_unit_commitment for key in chart_keys})
        for head in range(geometry.message_sketch.head_count):
            estimate = directed_matching_caliper(
                retained[indexes, head, :], chart_keys, weights
            )
            records.append(
                MatchingCaliperRecord(
                    chart_id=chart,
                    head_id=head,
                    status=estimate.status,
                    caliper=estimate.caliper,
                    fit_train_row_count=len(chart_keys),
                    fit_train_analysis_unit_count=unit_count,
                    contributing_target_count=estimate.contributing_target_count,
                    directed_pair_count=estimate.directed_pair_count,
                    contributing_target_weight=estimate.contributing_target_weight,
                )
            )
    return MatchingCaliperTable(tuple(records))


def _fit_donor_boundaries(
    fit_train: SyntheticPartition,
    geometry: SelectionGeometry,
) -> DonorQuintileBoundaries:
    keys = row_keys(fit_train)
    weights = hierarchical_row_weights(keys)
    retained = geometry.message_sketch.transform(fit_train.tensors["message"])
    norms = torch.sqrt(retained.square().sum(dim=(1, 2)))
    ties = tuple(key.order_bytes for key in keys)
    values = tuple(
        weighted_linear_quantile(norms, weights, quantile, tie_keys=ties)
        for quantile in _DONOR_QUANTILES
    )
    return DonorQuintileBoundaries(
        quantiles=_DONOR_QUANTILES,
        values=values,
        fit_train_row_count=len(keys),
        fit_train_analysis_unit_count=len(
            {key.analysis_unit_commitment for key in keys}
        ),
    )


def analytic_ambient_operator(
    q_codec: object,
    output_weight: Tensor | np.ndarray,
    message_basis: Tensor | np.ndarray,
) -> Tensor:
    """Return ``D_q E_q W_O S S.T`` in the full message ambient domain."""

    decoder = torch.as_tensor(getattr(q_codec, "decoder"), dtype=torch.float64)
    encoder = torch.as_tensor(getattr(q_codec, "encoder"), dtype=torch.float64)
    output = torch.as_tensor(output_weight, dtype=torch.float64)
    basis = torch.as_tensor(message_basis, dtype=torch.float64)
    if decoder.ndim != 2 or encoder.ndim != 2 or output.ndim != 2 or basis.ndim != 2:
        raise RepresentationQualificationError("analytic operator inputs must be matrices")
    if (
        decoder.shape[1] != encoder.shape[0]
        or encoder.shape[1] != output.shape[0]
        or output.shape[1] != basis.shape[0]
    ):
        raise RepresentationQualificationError(
            "analytic operator matrix dimensions are incompatible"
        )
    result = decoder @ encoder @ output @ basis @ basis.T
    if not bool(torch.isfinite(result).all()):
        raise RepresentationQualificationError("analytic operator is nonfinite")
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class AnalyticOperatorCell:
    chart_id: int
    head_id: int
    fit_train_row_count: int
    fit_train_analysis_unit_count: int
    ambient_operator_norm: float
    h4_surrogate_operator_norm: float
    ambient_operator_stable_rank: float
    ambient_operator_sha256: str

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "chart_id": self.chart_id,
            "head_id": self.head_id,
            "fit_train_row_count": self.fit_train_row_count,
            "fit_train_analysis_unit_count": self.fit_train_analysis_unit_count,
            "ambient_operator_norm": self.ambient_operator_norm,
            "h4_surrogate_operator_norm": self.h4_surrogate_operator_norm,
            "ambient_operator_stable_rank": self.ambient_operator_stable_rank,
            "ambient_operator_sha256": self.ambient_operator_sha256,
        }


def _epsilon_values(value: Tensor) -> tuple[tuple[float, ...], ...]:
    materialized = value.detach().to(device="cpu", dtype=torch.float64)
    return tuple(tuple(float(cell) for cell in row) for row in materialized)


@dataclass(frozen=True, slots=True)
class CodecRepresentationQualification:
    codec_id: str
    q_rank: int
    v_rank: int
    regularization_ratio: float
    orthogonality_error: float
    idempotence_error: float
    encoder_construction_error: float
    status: Literal[
        "eligible",
        "invalid_numerical_identity",
        "invalid_no_positive_operator_reference",
        "invalid_support",
    ]
    tensor_identities: tuple[NamedTensorIdentity, ...]
    tensor_manifest_sha256: str
    epsilon_completion_values: tuple[tuple[float, ...], ...]
    epsilon_j0_values: tuple[tuple[float, ...], ...]
    epsilon_completion_sha256: str
    epsilon_j0_sha256: str
    operator_cells: tuple[AnalyticOperatorCell, ...]
    operator_pair_summaries: tuple[OperatorPairSummary, ...]
    operator_aggregate: OperatorAggregateSummary | None
    operator_reference: OperatorReferenceResult
    constant_manifest_sha256: str

    def as_artifact_record(self, *, locator_candidate_id: str) -> dict[str, object]:
        return {
            "codec_id": self.codec_id,
            "locator_candidate_id": locator_candidate_id,
            "q_rank": self.q_rank,
            "v_rank": self.v_rank,
            "lambda_ratio": self.regularization_ratio,
            "orthogonality_error": self.orthogonality_error,
            "idempotence_error": self.idempotence_error,
            "encoder_construction_error": self.encoder_construction_error,
            "operator_reference_norm": self.operator_reference.operator_reference_norm,
            "tau_R": self.operator_reference.tau_r,
            "status": self.status,
            "tensor_manifest_sha256": self.tensor_manifest_sha256,
        }

    def constants_protocol_body(self) -> dict[str, object]:
        return {
            "codec_id": self.codec_id,
            "epsilon_completion": [list(row) for row in self.epsilon_completion_values],
            "epsilon_completion_sha256": self.epsilon_completion_sha256,
            "epsilon_J0": [list(row) for row in self.epsilon_j0_values],
            "epsilon_J0_sha256": self.epsilon_j0_sha256,
            "operator_cells": [cell.as_protocol_record() for cell in self.operator_cells],
            "operator_reference": {
                "status": self.operator_reference.status,
                "operator_reference_norm": self.operator_reference.operator_reference_norm,
                "tau_R": self.operator_reference.tau_r,
                "positive_pair_count": self.operator_reference.positive_pair_count,
                "zero_pair_count": self.operator_reference.zero_pair_count,
            },
            "constant_manifest_sha256": self.constant_manifest_sha256,
        }


def _operator_qualification(
    campaign: SyntheticCampaignData,
    fit_train: SyntheticPartition,
    geometry: SelectionGeometry,
    candidate: FittedCodecCandidate,
    maximum_o1_relative_tolerance: float,
) -> tuple[
    tuple[AnalyticOperatorCell, ...],
    tuple[OperatorPairSummary, ...],
    OperatorAggregateSummary | None,
    OperatorReferenceResult,
]:
    keys = row_keys(fit_train)
    charts, supported = geometry.locator.assign_with_support(fit_train.tensors["h"])
    expected_pairs = tuple(
        (chart, head)
        for chart in range(geometry.locator.chart_count)
        for head in range(geometry.message_sketch.head_count)
    )
    raw_operators: dict[tuple[int, int], Tensor] = {}
    h4_norms: dict[tuple[int, int], float] = {}
    for chart, head in expected_pairs:
        operator = analytic_ambient_operator(
            candidate.q_codecs[chart],
            campaign.source.output_weights[head],
            geometry.message_sketch.representations[head].basis,
        )
        raw_operators[(chart, head)] = operator
        codec = candidate.q_codecs[chart]
        carrier = codec.carrier_vectors
        eigenvalues = codec.carrier_eigenvalues
        h4_surrogate = eigenvalues.sqrt().unsqueeze(1) * (carrier.T @ operator)
        h4_norms[(chart, head)] = float(
            torch.linalg.matrix_norm(h4_surrogate, ord=2)
        )

    def observations(tolerance: float) -> tuple[OperatorObservation, ...]:
        rows: list[OperatorObservation] = []
        for index, key in enumerate(keys):
            if not bool(supported[index]):
                continue
            chart = int(charts[index])
            for head in range(geometry.message_sketch.head_count):
                operator = raw_operators[(chart, head)]
                rows.append(
                    OperatorObservation(
                        row_key=key,
                        chart_id=chart,
                        head_id=head,
                        ambient_operator_norm=float(
                            torch.linalg.matrix_norm(operator, ord=2)
                        ),
                        h4_surrogate_operator_norm=h4_norms[(chart, head)],
                        ambient_operator_stable_rank=stable_rank(
                            operator, tolerance=tolerance
                        ),
                    )
                )
        return tuple(rows)

    initial_summaries = summarize_operator_pairs(
        observations(0.0), expected_pairs=expected_pairs
    )
    reference = positive_operator_reference(
        initial_summaries,
        maximum_frozen_o1_relative_tolerance=maximum_o1_relative_tolerance,
    )
    final_summaries = summarize_operator_pairs(
        observations(reference.tau_r or 0.0), expected_pairs=expected_pairs
    )
    try:
        aggregate = aggregate_operator_pair_summaries(final_summaries)
    except ScientificContractError as error:
        if "no chart/head pair has sufficient support" not in str(error):
            raise
        aggregate = None

    cells: list[AnalyticOperatorCell] = []
    for chart, head in expected_pairs:
        mask = (charts == chart) & supported
        pair_keys = tuple(
            key for key, keep in zip(keys, mask, strict=True) if bool(keep)
        )
        operator = raw_operators[(chart, head)]
        cells.append(
            AnalyticOperatorCell(
                chart_id=chart,
                head_id=head,
                fit_train_row_count=len(pair_keys),
                fit_train_analysis_unit_count=len(
                    {key.analysis_unit_commitment for key in pair_keys}
                ),
                ambient_operator_norm=float(
                    torch.linalg.matrix_norm(operator, ord=2)
                ),
                h4_surrogate_operator_norm=h4_norms[(chart, head)],
                ambient_operator_stable_rank=stable_rank(
                    operator, tolerance=reference.tau_r or 0.0
                ),
                ambient_operator_sha256=identify_tensor(
                    f"codec.{candidate.codec_id}.operator.chart{chart}.head{head}",
                    operator,
                ).sha256,
            )
        )
    return tuple(cells), final_summaries, aggregate, reference


def _codec_qualification(
    campaign: SyntheticCampaignData,
    fit_train: SyntheticPartition,
    geometry: SelectionGeometry,
    candidate: FittedCodecCandidate,
    maximum_o1_relative_tolerance: float,
) -> CodecRepresentationQualification:
    identities = _codec_tensor_identities(candidate)
    manifest_sha256 = tensor_manifest_sha256(
        identities, namespace=f"codec_candidate_{candidate.codec_id}"
    )
    all_codecs = candidate.q_codecs + candidate.v_codecs
    orthogonality = max(codec.invariants.err_orth for codec in all_codecs)
    idempotence = max(codec.invariants.err_idem for codec in all_codecs)
    encoder_error = max(codec.invariants.err_encoder for codec in all_codecs)
    cells, pair_summaries, aggregate, reference = _operator_qualification(
        campaign,
        fit_train,
        geometry,
        candidate,
        maximum_o1_relative_tolerance,
    )
    numerical_passed = (
        orthogonality <= 1e-6
        and idempotence <= 1e-6
        and encoder_error <= 1e-10
    )
    if not numerical_passed:
        status: Literal[
            "eligible",
            "invalid_numerical_identity",
            "invalid_no_positive_operator_reference",
            "invalid_support",
        ] = "invalid_numerical_identity"
    elif not any(row.status == "supported" for row in pair_summaries):
        status = "invalid_support"
    elif reference.status != "valid":
        status = "invalid_no_positive_operator_reference"
    else:
        status = "eligible"

    completion_identity = next(
        row for row in identities if row.name.endswith(".epsilon_completion")
    )
    j0_identity = next(row for row in identities if row.name.endswith(".epsilon_J0"))
    epsilon_completion_values = _epsilon_values(
        candidate.epsilon_completion_by_chart_head
    )
    epsilon_j0_values = _epsilon_values(candidate.epsilon_j0_by_chart_head)
    constant_body = {
        "codec_id": candidate.codec_id,
        "epsilon_completion": [list(row) for row in epsilon_completion_values],
        "epsilon_completion_sha256": completion_identity.sha256,
        "epsilon_J0": [list(row) for row in epsilon_j0_values],
        "epsilon_J0_sha256": j0_identity.sha256,
        "operator_cells": [cell.as_protocol_record() for cell in cells],
        "operator_reference": {
            "status": reference.status,
            "operator_reference_norm": reference.operator_reference_norm,
            "tau_R": reference.tau_r,
            "positive_pair_count": reference.positive_pair_count,
            "zero_pair_count": reference.zero_pair_count,
        },
    }
    constant_sha256 = _canonical_json_sha256(
        b"fisher-cmc/codec-representation-constants/v1", constant_body
    )
    return CodecRepresentationQualification(
        codec_id=candidate.codec_id,
        q_rank=candidate.q_rank,
        v_rank=candidate.v_rank,
        regularization_ratio=candidate.regularization_ratio,
        orthogonality_error=orthogonality,
        idempotence_error=idempotence,
        encoder_construction_error=encoder_error,
        status=status,
        tensor_identities=identities,
        tensor_manifest_sha256=manifest_sha256,
        epsilon_completion_values=epsilon_completion_values,
        epsilon_j0_values=epsilon_j0_values,
        epsilon_completion_sha256=completion_identity.sha256,
        epsilon_j0_sha256=j0_identity.sha256,
        operator_cells=cells,
        operator_pair_summaries=pair_summaries,
        operator_aggregate=aggregate,
        operator_reference=reference,
        constant_manifest_sha256=constant_sha256,
    )


def _freeze_epsilon_down(campaign: SyntheticCampaignData) -> float:
    fit_train = _fit_train(campaign)
    keys = row_keys(fit_train)
    displacement = torch.as_tensor(
        fit_train.tensors["h_plus"].astype(np.float64)
        - fit_train.tensors["h"].astype(np.float64),
        dtype=torch.float64,
    )
    fisher = torch.as_tensor(
        campaign.source.h4_fisher_diagonal, dtype=torch.float64
    )
    row_energy = torch.einsum("nd,d,nd->n", displacement, fisher, displacement)
    sequence_values: dict[tuple[bytes, bytes], list[int]] = {}
    for index, key in enumerate(keys):
        sequence_values.setdefault(
            (key.analysis_unit_commitment, key.canonical_sequence_hash), []
        ).append(index)
    unit_values: dict[bytes, list[float]] = {}
    for (unit, _), indexes in sequence_values.items():
        unit_values.setdefault(unit, []).append(float(row_energy[indexes].sum()))
    ordered_values = [
        math.fsum(values) / len(values) for _, values in sorted(unit_values.items())
    ]
    reference = weighted_median(ordered_values, [1.0] * len(ordered_values))
    return max(1e-12, 1e-6 * reference)


def _chart_support(
    fit_train: SyntheticPartition,
    locator: FittedLocator,
) -> tuple[dict[str, object], ...]:
    assignments = locator.assign(fit_train.tensors["h"])
    rows = []
    for chart in range(locator.chart_count):
        metadata = tuple(
            item
            for item, assigned in zip(
                fit_train.metadata, assignments.tolist(), strict=True
            )
            if int(assigned) == chart
        )
        unit_count = len({item.analysis_unit_commitment for item in metadata})
        rows.append(
            {
                "chart_id": chart,
                "analysis_unit_count": unit_count,
                "sequence_count": len(
                    {item.canonical_sequence_hash for item in metadata}
                ),
                "row_count": len(metadata),
                "supported": (
                    unit_count >= 8
                    and locator.fit_monitor_row_counts[chart] > 0
                ),
            }
        )
    return tuple(rows)


def _message_head_records(
    sketch: FittedMessageSketch,
) -> tuple[dict[str, object], ...]:
    rows = []
    for head, representation in enumerate(sketch.representations):
        center = identify_tensor(f"message_sketch.head{head}.mean", representation.mean)
        basis = identify_tensor(f"message_sketch.head{head}.basis", representation.basis)
        normalization = (
            identify_tensor(
                f"message_sketch.head{head}.coordinate_mean",
                representation.coordinate_mean,
            ),
            identify_tensor(
                f"message_sketch.head{head}.coordinate_scale",
                representation.coordinate_scale,
            ),
        )
        rows.append(
            {
                "head_id": head,
                "message_dimension": int(representation.basis.shape[0]),
                "center_sha256": center.sha256,
                "basis_sha256": basis.sha256,
                "normalization_sha256": tensor_manifest_sha256(
                    normalization,
                    namespace=f"message_sketch_head_{head}_normalization",
                ),
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class PreTuneRepresentationQualification:
    selected_locator_k: int
    atlas_selected_k_values: tuple[tuple[str, int], ...]
    attempted_locator_k_values: tuple[int, ...]
    locator_atlas_evidence: tuple[LocatorAtlasEvidence, ...]
    locator_tensor_identities: tuple[NamedTensorIdentity, ...]
    locator_tensor_manifest_sha256: str
    locator_chart_support: tuple[dict[str, object], ...]
    message_sketch_tensor_identities: tuple[NamedTensorIdentity, ...]
    message_sketch_tensor_manifest_sha256: str
    message_sketch_head_records: tuple[dict[str, object], ...]
    codec_qualifications: tuple[CodecRepresentationQualification, ...]
    matching_calipers: MatchingCaliperTable
    donor_boundaries: DonorQuintileBoundaries
    epsilon_down: float
    maximum_o1_relative_tolerance: float
    full_rank_tolerances_sha256: str
    full_rank_artifact_body: Mapping[str, object]
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def selected_locator_candidate_id(self) -> str:
        return f"K{self.selected_locator_k}"

    @property
    def lower_k_fallback_used(self) -> bool:
        return self.selected_locator_k != self.attempted_locator_k_values[0]

    @property
    def lower_k_fallback_exhausted(self) -> bool:
        return self.lower_k_fallback_used and not any(
            codec.status == "eligible" for codec in self.codec_qualifications
        )

    @property
    def requires_lower_k_operator_retry(self) -> bool:
        """Whether J4 requires rebuilding the representation at frozen K=2."""

        return (
            self.selected_locator_k == self.attempted_locator_k_values[0]
            and bool(self.codec_qualifications)
            and all(
                codec.status == "invalid_no_positive_operator_reference"
                for codec in self.codec_qualifications
            )
        )

    @property
    def operator_reference_fallback_exhausted(self) -> bool:
        """Whether every codec still lacks a positive reference at lower K."""

        return (
            self.lower_k_fallback_used
            and bool(self.codec_qualifications)
            and all(
                codec.status == "invalid_no_positive_operator_reference"
                for codec in self.codec_qualifications
            )
        )

    @property
    def passed(self) -> bool:
        return (
            self.matching_calipers.passed
            and all(bool(row["supported"]) for row in self.locator_chart_support)
            and any(codec.status == "eligible" for codec in self.codec_qualifications)
            and bool(self.full_rank_artifact_body["all_eligible_records_passed"])
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            b"fisher-cmc/pre-tune-representation-qualification/v1",
            {
                "selected_locator_candidate_id": self.selected_locator_candidate_id,
                "atlas_selected_k_values": [
                    {"atlas_name": name, "selected_K": selected_k}
                    for name, selected_k in self.atlas_selected_k_values
                ],
                "locator_atlas_evidence": [
                    row.as_protocol_record()
                    for row in self.locator_atlas_evidence
                ],
                "attempted_locator_k_values": list(self.attempted_locator_k_values),
                "locator_tensor_manifest_sha256": self.locator_tensor_manifest_sha256,
                "message_sketch_tensor_manifest_sha256": (
                    self.message_sketch_tensor_manifest_sha256
                ),
                "codec_tensor_manifests": [
                    codec.tensor_manifest_sha256
                    for codec in self.codec_qualifications
                ],
                "codec_constant_manifests": [
                    codec.constant_manifest_sha256
                    for codec in self.codec_qualifications
                ],
                "matching_calipers_sha256": self.matching_calipers.identity_sha256,
                "donor_boundaries_sha256": self.donor_boundaries.identity_sha256,
                "epsilon_down": self.epsilon_down,
                "maximum_o1_relative_tolerance": (
                    self.maximum_o1_relative_tolerance
                ),
                "full_rank_tolerances_sha256": self.full_rank_tolerances_sha256,
            },
        )

    def locator_protocol_body(self) -> dict[str, object]:
        return {
            "atlas_names": ["A_in", "A_mid", "A_out"],
            "membership_atlas": "A_in",
            "global_pca_rank": 32,
            "local_locator_rank": 32,
            "candidate_K_values": list(self.attempted_locator_k_values),
            "minimum_fit_train_units_per_chart": 8,
            "kmeans_seed": 1701,
            "maximum_center_updates": 100,
            "centroid_shift_tolerance": 1e-6,
            "support_quantile": 0.99,
            "support_rcond": 1e-8,
            "activation_only_membership": True,
            "teacher_fallback": False,
        }

    def locator_candidates_body(
        self, *, locator_protocol_sha256: str
    ) -> dict[str, object]:
        return {
            "locator_protocol_sha256": locator_protocol_sha256,
            "candidates": [
                {
                    "candidate_id": self.selected_locator_candidate_id,
                    "K": self.selected_locator_k,
                    "status": "eligible",
                    "chart_support": [dict(row) for row in self.locator_chart_support],
                    "tensor_manifest_sha256": self.locator_tensor_manifest_sha256,
                }
            ],
            "selected_locator_candidate_id": self.selected_locator_candidate_id,
        }

    def codec_protocol_body(
        self, *, carrier_identity_sha256: str
    ) -> dict[str, object]:
        return {
            "carrier_identity_sha256": carrier_identity_sha256,
            "codec_candidates": [
                {
                    "codec_id": codec_id,
                    "q_rank": q_rank,
                    "v_rank": v_rank,
                    "lambda_ratio": ratio,
                }
                for codec_id, q_rank, v_rank, ratio in CODEC_LADDER
            ],
            "uncentered_second_moments": True,
            "linear_encoders": True,
            "shared_q_decoder_across_heads": True,
            "orthogonality_threshold": 1e-6,
            "idempotence_threshold": 1e-6,
            "encoder_construction_threshold": 1e-10,
            "epsilon_codec": 1e-12,
            "positive_only_operator_reference": True,
        }

    def codec_candidate_manifests_body(
        self, *, codec_protocol_sha256: str
    ) -> dict[str, object]:
        return {
            "codec_protocol_sha256": codec_protocol_sha256,
            "candidates": [
                codec.as_artifact_record(
                    locator_candidate_id=self.selected_locator_candidate_id
                )
                for codec in self.codec_qualifications
            ],
            "lower_K_fallback_exhausted": self.lower_k_fallback_exhausted,
        }

    def message_sketch_manifest_body(self) -> dict[str, object]:
        return {
            "message_rank_per_head": 32,
            "fit_partition": "fit_train",
            "centered_covariance": True,
            "deterministic_svd": True,
            "sign_rule": "largest_magnitude_loading_positive_smallest_index_tie",
            "heads": [dict(row) for row in self.message_sketch_head_records],
        }

    def representation_numeric_constants_body(self) -> dict[str, object]:
        return {
            "variance_floor": 1e-6,
            "support_rcond": 1e-8,
            "support_absolute_floor": 1e-8,
            "epsilon_codec": 1e-12,
            "epsilon_down": self.epsilon_down,
            "rho_denominator_floor": 1e-12,
            "codec_constant_manifests": [
                {
                    "name": codec.codec_id,
                    "artifact_sha256": codec.constant_manifest_sha256,
                }
                for codec in self.codec_qualifications
            ],
            "matching_calipers_sha256": self.matching_calipers.identity_sha256,
            "donor_boundaries_sha256": self.donor_boundaries.identity_sha256,
            "full_rank_tolerances_sha256": self.full_rank_tolerances_sha256,
        }

    def full_rank_identity_smoke_body(self) -> dict[str, object]:
        return copy.deepcopy(dict(self.full_rank_artifact_body))

    def selection_representation_manifest_body(self) -> dict[str, object]:
        scalar_hash = _canonical_json_sha256(
            b"fisher-cmc/epsilon-down/v1", {"epsilon_down": self.epsilon_down}
        )
        bindings = [
            {
                "object_name": "locator_atlases_A_in_A_mid_A_out",
                "artifact_sha256": self.locator_tensor_manifest_sha256,
                "immutable": True,
            },
            {
                "object_name": "message_sketch_p32",
                "artifact_sha256": self.message_sketch_tensor_manifest_sha256,
                "immutable": True,
            },
            {
                "object_name": "matching_calipers",
                "artifact_sha256": self.matching_calipers.identity_sha256,
                "immutable": True,
            },
            {
                "object_name": "interaction_donor_boundaries",
                "artifact_sha256": self.donor_boundaries.identity_sha256,
                "immutable": True,
            },
            {
                "object_name": "epsilon_down",
                "artifact_sha256": scalar_hash,
                "immutable": True,
            },
            {
                "object_name": "full_rank_tolerances",
                "artifact_sha256": self.full_rank_tolerances_sha256,
                "immutable": True,
            },
        ]
        for codec in self.codec_qualifications:
            bindings.extend(
                (
                    {
                        "object_name": f"codec_candidate_{codec.codec_id}",
                        "artifact_sha256": codec.tensor_manifest_sha256,
                        "immutable": True,
                    },
                    {
                        "object_name": f"codec_constants_{codec.codec_id}",
                        "artifact_sha256": codec.constant_manifest_sha256,
                        "immutable": True,
                    },
                )
            )
        return {
            "namespace": "selection_representation",
            "fit_partition": "fit_train",
            "selected_locator_candidate_id": self.selected_locator_candidate_id,
            "bindings": bindings,
            "chart_id_order": list(range(self.selected_locator_k)),
            "immutable_after_tune_access": True,
        }

    def artifact_bodies(
        self,
        *,
        locator_protocol_sha256: str,
        codec_protocol_sha256: str,
        carrier_identity_sha256: str,
    ) -> dict[str, dict[str, object]]:
        """Return every measured P4 representation body under current schemas."""

        return {
            "locator_protocol.json": self.locator_protocol_body(),
            "locator_candidates.json": self.locator_candidates_body(
                locator_protocol_sha256=locator_protocol_sha256
            ),
            "codec_protocol.json": self.codec_protocol_body(
                carrier_identity_sha256=carrier_identity_sha256
            ),
            "codec_candidate_manifests.json": (
                self.codec_candidate_manifests_body(
                    codec_protocol_sha256=codec_protocol_sha256
                )
            ),
            "message_sketch_manifest_p32.json": (
                self.message_sketch_manifest_body()
            ),
            "representation_numeric_constants.json": (
                self.representation_numeric_constants_body()
            ),
            "full_rank_identity_smoke.json": self.full_rank_identity_smoke_body(),
            "selection_representation_manifest.json": (
                self.selection_representation_manifest_body()
            ),
        }


def build_pre_tune_representation_qualification(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    o1: O1Qualification,
    full_rank_smoke: _FullRankSmoke,
) -> PreTuneRepresentationQualification:
    """Measure and freeze all representation-derived P4 values.

    The caller must construct ``geometry``, ``o1``, and ``full_rank_smoke``
    before any tune access.  This function then reads only the campaign's
    fit-train rows and source parameters.
    """

    if not o1.passed:
        raise RepresentationQualificationError(
            "O1 must pass before representation preregistration"
        )
    if not full_rank_smoke.passed:
        raise RepresentationQualificationError(
            "the full-rank fit-monitor smoke must pass before preregistration"
        )
    ladder = tuple(
        (
            candidate.codec_id,
            candidate.q_rank,
            candidate.v_rank,
            candidate.regularization_ratio,
        )
        for candidate in geometry.codec_candidates
    )
    if ladder != CODEC_LADDER:
        raise RepresentationQualificationError(
            "fitted codec inventory differs from the preregistered ladder"
        )
    locators = (geometry.locator, geometry.mid_locator, geometry.out_locator)
    if tuple(locator.atlas_name for locator in locators) != (
        "A_in",
        "A_mid",
        "A_out",
    ):
        raise RepresentationQualificationError("activation atlas inventory differs")
    if any(locator.chart_count not in _ATTEMPTED_LOCATOR_K for locator in locators):
        raise RepresentationQualificationError(
            "an activation atlas selected a locator K that was not preregistered"
        )
    if geometry.message_sketch.rank != 32:
        raise RepresentationQualificationError("message sketch is not frozen at p=32")

    fit_train = _fit_train(campaign)
    maximum_o1_relative_tolerance = max(row.tau_rel for row in o1.tolerances)
    locator_identities = tuple(
        identity for locator in locators for identity in _locator_identities(locator)
    )
    locator_atlas_evidence = tuple(
        LocatorAtlasEvidence(
            atlas_name=locator.atlas_name,
            selected_k=locator.chart_count,
            attempted_k_values=_ATTEMPTED_LOCATOR_K,
            lower_k_fallback_used=(
                locator.chart_count != _ATTEMPTED_LOCATOR_K[0]
            ),
            fit_train_unit_counts=locator.fit_train_unit_counts,
            runtime_fit_train_unit_counts=locator.runtime_fit_train_unit_counts,
            fit_monitor_row_counts=locator.fit_monitor_row_counts,
            kmeans_iterations=locator.kmeans_iterations,
            kmeans_stopping_condition=locator.kmeans_stopping_condition,
            tensor_manifest_sha256=tensor_manifest_sha256(
                _locator_identities(locator),
                namespace=f"activation_locator_{locator.atlas_name}",
            ),
        )
        for locator in locators
    )
    locator_manifest = tensor_manifest_sha256(
        locator_identities, namespace="activation_locator_atlases"
    )
    sketch_identities = _message_sketch_identities(geometry.message_sketch)
    sketch_manifest = tensor_manifest_sha256(
        sketch_identities, namespace="message_sketch_p32"
    )
    codec_qualifications = tuple(
        _codec_qualification(
            campaign,
            fit_train,
            geometry,
            candidate,
            maximum_o1_relative_tolerance,
        )
        for candidate in geometry.codec_candidates
    )
    matching_calipers = _fit_matching_calipers(fit_train, geometry)
    donor_boundaries = _fit_donor_boundaries(fit_train, geometry)
    result = PreTuneRepresentationQualification(
        selected_locator_k=geometry.locator.chart_count,
        atlas_selected_k_values=tuple(
            (locator.atlas_name, locator.chart_count) for locator in locators
        ),
        attempted_locator_k_values=_ATTEMPTED_LOCATOR_K,
        locator_atlas_evidence=locator_atlas_evidence,
        locator_tensor_identities=locator_identities,
        locator_tensor_manifest_sha256=locator_manifest,
        locator_chart_support=_chart_support(fit_train, geometry.locator),
        message_sketch_tensor_identities=sketch_identities,
        message_sketch_tensor_manifest_sha256=sketch_manifest,
        message_sketch_head_records=_message_head_records(geometry.message_sketch),
        codec_qualifications=codec_qualifications,
        matching_calipers=matching_calipers,
        donor_boundaries=donor_boundaries,
        epsilon_down=_freeze_epsilon_down(campaign),
        maximum_o1_relative_tolerance=maximum_o1_relative_tolerance,
        full_rank_tolerances_sha256=full_rank_smoke.frozen_tolerances_sha256,
        full_rank_artifact_body=full_rank_smoke.as_artifact_body(),
    )
    if not math.isfinite(result.epsilon_down) or result.epsilon_down < 1e-12:
        raise RepresentationQualificationError("epsilon_down is invalid")
    if not matching_calipers.passed:
        raise RepresentationQualificationError(
            "one or more chart/head matching calipers has no eligible pair"
        )
    return result
