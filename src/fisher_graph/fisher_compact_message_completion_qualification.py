"""Synthetic scientific qualification for compact-message completion.

This layer connects the model-free numerical kernels to the deterministic
synthetic backend.  It fits only the frozen fit-train representation, evaluates
the seam and codec ceilings, and prepares fixed inputs for the paired J-model
bank.  All outputs remain synthetic qualification evidence and are unsuitable
for a scientific or guard claim.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Final

import numpy as np
import torch
from torch import Tensor

from .fisher_compact_message_completion_science import (
    FidelityMetrics,
    FidelityPromptObservation,
    FisherCodec,
    HierarchicalRowKey,
    KMeansSupportError,
    O2CodecEvaluation,
    O2SelectionDecision,
    QContributionCodec,
    StabilizedCovariance,
    WeightedKMeansConfig,
    VFlowCodec,
    canonicalize_tied_eigenvectors,
    combined_qv_second_moment,
    evaluate_five_fidelity_gates,
    fit_q_contribution_codec,
    fit_v_flow_codec,
    fit_weighted_kmeans,
    hierarchical_row_weights,
    select_o2q_codec,
    stabilize_support_covariance,
    summarize_fidelity,
    tensor_weighted_median,
    weighted_mean_and_covariance,
    weighted_linear_quantile,
    weighted_relative_l2,
)
from .fisher_compact_message_completion_synthetic import (
    SyntheticCampaignData,
    SyntheticPartition,
    exact_synthetic_mlp,
    exact_synthetic_mlp_blocks,
    synthetic_rmsnorm,
)


__all__ = [
    "CODEC_LADDER",
    "DevelopmentSupportAssessment",
    "FittedCodecCandidate",
    "FittedLocator",
    "FittedMessageSketch",
    "O1Qualification",
    "O1SeamResult",
    "O1Tolerance",
    "PreparedCompletionPanel",
    "PrincipalRepresentation",
    "QualificationError",
    "SelectionGeometry",
    "build_o1_qualification",
    "assess_development_support",
    "evaluate_fidelity",
    "evaluate_o2",
    "evaluate_o2q_replay",
    "fit_selection_geometry",
    "fidelity_gate_records",
    "prepare_completion_panel",
    "row_keys",
    "subset_partition_by_units",
]


CODEC_LADDER: Final = (
    ("C0", 32, 32, 1e-3),
    ("C1", 64, 64, 1e-3),
    ("C2", 32, 32, 1e-2),
    ("C3", 64, 64, 1e-2),
)
_O1_SEAMS: Final = (
    "attention_q_head_sum",
    "attention_q_total",
    "attention_h_A",
    "mlp_q_block_sum",
    "mlp_q_total",
    "mlp_h_plus",
)


class QualificationError(RuntimeError):
    """The synthetic fixture violates a frozen scientific precondition."""


def row_keys(partition: SyntheticPartition) -> tuple[HierarchicalRowKey, ...]:
    return tuple(
        HierarchicalRowKey(
            bytes.fromhex(item.analysis_unit_commitment),
            bytes.fromhex(item.canonical_sequence_hash),
            item.token_position,
        )
        for item in partition.metadata
    )


def _selector(
    partition: SyntheticPartition, unit_commitments: Sequence[str]
) -> np.ndarray:
    allowed = frozenset(unit_commitments)
    result = np.asarray(
        [item.analysis_unit_commitment in allowed for item in partition.metadata],
        dtype=bool,
    )
    if not bool(result.any()):
        raise QualificationError("unit selection produced no rows")
    return result


def _tensor(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.asarray(value).copy()).to(dtype=torch.float64)


def _oriented_eigenvectors(covariance: Tensor, rank: int) -> Tensor:
    values, vectors = torch.linalg.eigh(covariance)
    vectors = canonicalize_tied_eigenvectors(values, vectors)
    order = torch.argsort(values, descending=True, stable=True)
    vectors = vectors[:, order[:rank]].clone()
    for column in range(vectors.shape[1]):
        pivot = int(torch.argmax(vectors[:, column].abs()))
        if float(vectors[pivot, column]) < 0.0:
            vectors[:, column].neg_()
    return vectors.contiguous()


def _hash_tensor_sequence(domain: bytes, tensors: Sequence[Tensor]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for tensor in tensors:
        value = tensor.detach().cpu().to(dtype=torch.float64).contiguous().numpy()
        digest.update(struct.pack(">I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(value.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PrincipalRepresentation:
    mean: Tensor
    basis: Tensor
    coordinate_mean: Tensor
    coordinate_scale: Tensor

    def raw_transform(self, values: np.ndarray | Tensor) -> Tensor:
        """Project into the unstandardized PCA coordinates used by F14."""

        rows = _tensor(values) if isinstance(values, np.ndarray) else values.to(torch.float64)
        return (rows - self.mean) @ self.basis

    def transform(self, values: np.ndarray | Tensor) -> Tensor:
        coordinates = self.raw_transform(values)
        return (coordinates - self.coordinate_mean) / self.coordinate_scale

    def reconstruct(self, standardized: Tensor) -> Tensor:
        coordinates = self.coordinate_mean + standardized.to(torch.float64) * self.coordinate_scale
        return self.mean + coordinates @ self.basis.T


def _fit_principal_representation(
    rows: Tensor,
    keys: Sequence[HierarchicalRowKey],
    *,
    rank: int,
) -> PrincipalRepresentation:
    if rows.ndim != 2 or rows.shape[0] != len(keys):
        raise QualificationError("PCA rows and keys are not aligned")
    weights = hierarchical_row_weights(keys)
    mean, covariance = weighted_mean_and_covariance(rows, weights)
    basis = _oriented_eigenvectors(covariance, rank)
    coordinates = (rows - mean) @ basis
    coordinate_mean = torch.einsum("n,nd->d", weights, coordinates)
    centered = coordinates - coordinate_mean
    variance = torch.einsum("n,nd,nd->d", weights, centered, centered)
    scale = torch.sqrt(torch.clamp(variance, min=1e-6))
    return PrincipalRepresentation(mean, basis, coordinate_mean, scale)


@dataclass(frozen=True, slots=True)
class FittedLocator:
    atlas_name: str
    global_representation: PrincipalRepresentation
    centers: Tensor
    local_representations: tuple[PrincipalRepresentation, ...]
    support_covariances: tuple[StabilizedCovariance, ...]
    sigma_perp_squared: Tensor
    parallel_support_limits: Tensor
    perpendicular_support_limits: Tensor
    fit_train_unit_counts: tuple[int, ...]
    runtime_fit_train_unit_counts: tuple[int, ...]
    fit_monitor_row_counts: tuple[int, ...]
    kmeans_iterations: int
    kmeans_stopping_condition: str

    @property
    def chart_count(self) -> int:
        return self.centers.shape[0]

    def assign(self, h: np.ndarray | Tensor) -> Tensor:
        total, _, _ = self.distances(h)
        return torch.argmin(total, dim=1)

    def distances(
        self, h: np.ndarray | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return total, tangential, and normal-space distances per chart."""

        rows = _tensor(h) if isinstance(h, np.ndarray) else h.to(torch.float64)
        parallel: list[Tensor] = []
        perpendicular: list[Tensor] = []
        for chart, representation in enumerate(self.local_representations):
            centered = rows - representation.mean
            raw = centered @ representation.basis
            parallel.append(
                torch.einsum(
                    "ni,ij,nj->n",
                    raw,
                    self.support_covariances[chart].pseudoinverse,
                    raw,
                )
            )
            residual = centered - raw @ representation.basis.T
            perpendicular.append(
                residual.square().sum(-1) / self.sigma_perp_squared[chart]
            )
        parallel_tensor = torch.stack(parallel, dim=1)
        perpendicular_tensor = torch.stack(perpendicular, dim=1)
        return (
            parallel_tensor + perpendicular_tensor,
            parallel_tensor,
            perpendicular_tensor,
        )

    def assign_with_support(
        self, h: np.ndarray | Tensor
    ) -> tuple[Tensor, Tensor]:
        """Choose the minimum-distance chart and apply both 99th-percentile limits."""

        total, parallel, perpendicular = self.distances(h)
        assignments = torch.argmin(total, dim=1)
        indexes = torch.arange(assignments.numel(), dtype=torch.int64)
        supported = (
            parallel[indexes, assignments]
            <= self.parallel_support_limits[assignments]
        ) & (
            perpendicular[indexes, assignments]
            <= self.perpendicular_support_limits[assignments]
        )
        return assignments, supported

    def transform(self, h: np.ndarray | Tensor) -> tuple[Tensor, Tensor]:
        rows = _tensor(h) if isinstance(h, np.ndarray) else h.to(torch.float64)
        assignments = self.assign(rows)
        result = torch.empty(
            (rows.shape[0], self.local_representations[0].basis.shape[1]),
            dtype=torch.float64,
        )
        for chart, representation in enumerate(self.local_representations):
            mask = assignments == chart
            if bool(mask.any()):
                result[mask] = representation.transform(rows[mask])
        return assignments, result

    def transform_with_support(
        self, h: np.ndarray | Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return frozen chart IDs, standardized local coordinates, and support."""

        rows = _tensor(h) if isinstance(h, np.ndarray) else h.to(torch.float64)
        assignments, supported = self.assign_with_support(rows)
        result = torch.empty(
            (rows.shape[0], self.local_representations[0].basis.shape[1]),
            dtype=torch.float64,
        )
        for chart, representation in enumerate(self.local_representations):
            mask = assignments == chart
            if bool(mask.any()):
                result[mask] = representation.transform(rows[mask])
        return assignments, result, supported

    @property
    def identity_sha256(self) -> str:
        tensors: list[Tensor] = [
            self.global_representation.mean,
            self.global_representation.basis,
            self.global_representation.coordinate_mean,
            self.global_representation.coordinate_scale,
            self.centers,
            self.sigma_perp_squared,
            self.parallel_support_limits,
            self.perpendicular_support_limits,
            torch.tensor(self.fit_train_unit_counts, dtype=torch.int64),
            torch.tensor(self.runtime_fit_train_unit_counts, dtype=torch.int64),
            torch.tensor(self.fit_monitor_row_counts, dtype=torch.int64),
        ]
        for representation in self.local_representations:
            tensors.extend(
                (
                    representation.mean,
                    representation.basis,
                    representation.coordinate_mean,
                    representation.coordinate_scale,
                )
            )
        for covariance in self.support_covariances:
            tensors.extend(
                (
                    covariance.symmetric_covariance,
                    covariance.pseudoinverse,
                    covariance.active,
                )
            )
        return _hash_tensor_sequence(
            f"fisher-cmc/synthetic-locator/{self.atlas_name}/v1".encode(),
            tensors,
        )


@dataclass(frozen=True, slots=True)
class FittedMessageSketch:
    representations: tuple[PrincipalRepresentation, ...]

    @property
    def head_count(self) -> int:
        return len(self.representations)

    @property
    def rank(self) -> int:
        return self.representations[0].basis.shape[1]

    def transform(self, messages: np.ndarray | Tensor) -> Tensor:
        rows = _tensor(messages) if isinstance(messages, np.ndarray) else messages.to(torch.float64)
        if rows.ndim != 3 or rows.shape[1] != self.head_count:
            raise QualificationError("messages must have shape [rows, heads, width]")
        return torch.stack(
            [
                representation.transform(rows[:, head, :])
                for head, representation in enumerate(self.representations)
            ],
            dim=1,
        )

    def reconstruct(self, standardized: Tensor) -> Tensor:
        if standardized.ndim != 3 or standardized.shape[1] != self.head_count:
            raise QualificationError("message coordinates have invalid shape")
        return torch.stack(
            [
                representation.reconstruct(standardized[:, head, :])
                for head, representation in enumerate(self.representations)
            ],
            dim=1,
        )

    @property
    def identity_sha256(self) -> str:
        tensors: list[Tensor] = []
        for representation in self.representations:
            tensors.extend(
                (
                    representation.mean,
                    representation.basis,
                    representation.coordinate_mean,
                    representation.coordinate_scale,
                )
            )
        return _hash_tensor_sequence(b"fisher-cmc/synthetic-message-sketch/v1", tensors)


@dataclass(frozen=True, slots=True)
class FittedCodecCandidate:
    codec_id: str
    q_rank: int
    v_rank: int
    regularization_ratio: float
    q_codecs: tuple[QContributionCodec, ...]
    v_codecs: tuple[VFlowCodec, ...]
    epsilon_completion_by_chart_head: Tensor
    epsilon_j0_by_chart_head: Tensor

    def __post_init__(self) -> None:
        if not self.q_codecs or len(self.q_codecs) != len(self.v_codecs):
            raise QualificationError(
                "Q-contribution and V-flow codec inventories must align"
            )
        if any(not isinstance(codec, QContributionCodec) for codec in self.q_codecs):
            raise QualificationError(
                "q_codecs must contain only QContributionCodec values"
            )
        if any(not isinstance(codec, VFlowCodec) for codec in self.v_codecs):
            raise QualificationError("v_codecs must contain only VFlowCodec values")
        if any(codec.rank != self.q_rank for codec in self.q_codecs):
            raise QualificationError("Q-contribution codec rank is inconsistent")
        if any(codec.rank != self.v_rank for codec in self.v_codecs):
            raise QualificationError("V-flow codec rank is inconsistent")

    def project_q(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codec(values, chart_ids, self.q_codecs, "project")

    def project_v(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codec(values, chart_ids, self.v_codecs, "project")

    def encode_q(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codec(values, chart_ids, self.q_codecs, "encode")

    def decode_q(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codec(values, chart_ids, self.q_codecs, "decode")

    @property
    def identity_sha256(self) -> str:
        tensors: list[Tensor] = []
        for codec in self.q_codecs + self.v_codecs:
            tensors.extend((codec.encoder, codec.decoder, codec.projector))
        tensors.extend(
            (
                self.epsilon_completion_by_chart_head,
                self.epsilon_j0_by_chart_head,
            )
        )
        return _hash_tensor_sequence(
            f"fisher-cmc/synthetic-codec/{self.codec_id}/v1".encode(), tensors
        )


def _apply_chart_codec(
    values: Tensor,
    chart_ids: Tensor,
    codecs: Sequence[FisherCodec],
    operation: str,
) -> Tensor:
    rows = values.to(torch.float64)
    charts = chart_ids.to(torch.int64)
    if rows.shape[0] != charts.numel():
        raise QualificationError("codec values and chart IDs are misaligned")
    output_width = (
        codecs[0].rank if operation == "encode" else codecs[0].decoder.shape[0]
    )
    result = torch.empty(rows.shape[:-1] + (output_width,), dtype=torch.float64)
    for chart, codec in enumerate(codecs):
        mask = charts == chart
        if not bool(mask.any()):
            continue
        method = getattr(codec, operation)
        result[mask] = method(rows[mask])
    return result


@dataclass(frozen=True, slots=True)
class SelectionGeometry:
    locator: FittedLocator
    mid_locator: FittedLocator
    out_locator: FittedLocator
    message_sketch: FittedMessageSketch
    codec_candidates: tuple[FittedCodecCandidate, ...]

    def codec(self, codec_id: str) -> FittedCodecCandidate:
        for candidate in self.codec_candidates:
            if candidate.codec_id == codec_id:
                return candidate
        raise KeyError(codec_id)

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"fisher-cmc/synthetic-selection-geometry/v1\0")
        digest.update(bytes.fromhex(self.locator.identity_sha256))
        digest.update(bytes.fromhex(self.mid_locator.identity_sha256))
        digest.update(bytes.fromhex(self.out_locator.identity_sha256))
        digest.update(bytes.fromhex(self.message_sketch.identity_sha256))
        for candidate in self.codec_candidates:
            digest.update(bytes.fromhex(candidate.identity_sha256))
        return digest.hexdigest()


def _fit_locator(
    partition: SyntheticPartition,
    selected: np.ndarray,
    *,
    rank: int,
    tensor_name: str,
    atlas_name: str,
    attempted_k: Sequence[int] = (4, 2),
) -> FittedLocator:
    h = _tensor(partition.tensors[tensor_name][selected])
    keys_all = row_keys(partition)
    keys = tuple(key for key, keep in zip(keys_all, selected, strict=True) if keep)
    weights = hierarchical_row_weights(keys)
    global_representation = _fit_principal_representation(h, keys, rank=rank)
    # F14 clusters the centered PCA scores themselves.  Standardization is a
    # later model-input contract and would change the k-means metric.
    global_coordinates = global_representation.raw_transform(h)
    unit_ids = np.asarray(
        [
            item.analysis_unit_commitment
            for item, keep in zip(partition.metadata, selected, strict=True)
            if keep
        ]
    )
    last_error: Exception | None = None
    for k in attempted_k:
        try:
            clustering = fit_weighted_kmeans(
                global_coordinates,
                keys,
                weights,
                WeightedKMeansConfig(k),
            )
            unit_counts = tuple(
                len(set(unit_ids[clustering.assignments.numpy() == chart]))
                for chart in range(k)
            )
            if min(unit_counts) < 8:
                raise QualificationError(
                    f"K={k} has fewer than eight fit-train units in a chart"
                )
            local: list[PrincipalRepresentation] = []
            support_covariances: list[StabilizedCovariance] = []
            sigma_perp: list[float] = []
            parallel_limits: list[float] = []
            perpendicular_limits: list[float] = []
            for chart in range(k):
                mask = clustering.assignments == chart
                chart_rows = h[mask]
                chart_keys = tuple(
                    key
                    for key, keep in zip(keys, mask, strict=True)
                    if bool(keep)
                )
                representation = _fit_principal_representation(
                    chart_rows, chart_keys, rank=rank
                )
                local.append(representation)
                chart_weights = hierarchical_row_weights(chart_keys)
                centered = chart_rows - representation.mean
                raw = centered @ representation.basis
                _, covariance = weighted_mean_and_covariance(raw, chart_weights)
                stabilized = stabilize_support_covariance(covariance)
                support_covariances.append(stabilized)
                residual = centered - raw @ representation.basis.T
                residual_energy = residual.square().sum(-1)
                sigma = max(
                    float(torch.dot(chart_weights, residual_energy)),
                    1e-12,
                )
                sigma_perp.append(sigma)
                parallel_distance = torch.einsum(
                    "ni,ij,nj->n", raw, stabilized.pseudoinverse, raw
                )
                perpendicular_distance = residual_energy / sigma
                tie_keys = tuple(key.order_bytes for key in chart_keys)
                parallel_limits.append(
                    weighted_linear_quantile(
                        parallel_distance,
                        chart_weights,
                        0.99,
                        tie_keys=tie_keys,
                    )
                )
                perpendicular_limits.append(
                    weighted_linear_quantile(
                        perpendicular_distance,
                        chart_weights,
                        0.99,
                        tie_keys=tie_keys,
                    )
                )
            fitted = FittedLocator(
                atlas_name=atlas_name,
                global_representation=global_representation,
                centers=clustering.centers,
                local_representations=tuple(local),
                support_covariances=tuple(support_covariances),
                sigma_perp_squared=torch.tensor(sigma_perp, dtype=torch.float64),
                parallel_support_limits=torch.tensor(
                    parallel_limits, dtype=torch.float64
                ),
                perpendicular_support_limits=torch.tensor(
                    perpendicular_limits, dtype=torch.float64
                ),
                fit_train_unit_counts=unit_counts,
                runtime_fit_train_unit_counts=unit_counts,
                fit_monitor_row_counts=tuple(0 for _ in range(k)),
                kmeans_iterations=clustering.iteration_count,
                kmeans_stopping_condition=clustering.stopping_condition,
            )
            final_assignments = fitted.assign(h).numpy()
            final_unit_counts = tuple(
                len(set(unit_ids[final_assignments == chart]))
                for chart in range(k)
            )
            if min(final_unit_counts) < 8:
                raise QualificationError(
                    f"K={k} has fewer than eight fit-train units after locator assignment"
                )
            monitor_h = _tensor(partition.tensors[tensor_name][~selected])
            monitor_assignments = fitted.assign(monitor_h).numpy()
            monitor_row_counts = tuple(
                int(np.count_nonzero(monitor_assignments == chart))
                for chart in range(k)
            )
            if min(monitor_row_counts) == 0:
                raise QualificationError(
                    f"K={k} has an active chart with zero fit-monitor validation rows"
                )
            return FittedLocator(
                atlas_name=fitted.atlas_name,
                global_representation=fitted.global_representation,
                centers=fitted.centers,
                local_representations=fitted.local_representations,
                support_covariances=fitted.support_covariances,
                sigma_perp_squared=fitted.sigma_perp_squared,
                parallel_support_limits=fitted.parallel_support_limits,
                perpendicular_support_limits=fitted.perpendicular_support_limits,
                fit_train_unit_counts=fitted.fit_train_unit_counts,
                runtime_fit_train_unit_counts=final_unit_counts,
                fit_monitor_row_counts=monitor_row_counts,
                kmeans_iterations=fitted.kmeans_iterations,
                kmeans_stopping_condition=fitted.kmeans_stopping_condition,
            )
        except (QualificationError, KMeansSupportError) as error:
            # These are the two preregistered lower-K fallback conditions:
            # inadequate chart support and an unrepairable empty cluster.
            last_error = error
    raise QualificationError("lower-K locator fallback exhausted") from last_error


def _fit_message_sketch(
    partition: SyntheticPartition,
    selected: np.ndarray,
    *,
    rank: int,
) -> FittedMessageSketch:
    messages = _tensor(partition.tensors["message"][selected])
    keys_all = row_keys(partition)
    keys = tuple(key for key, keep in zip(keys_all, selected, strict=True) if keep)
    return FittedMessageSketch(
        tuple(
            _fit_principal_representation(messages[:, head, :], keys, rank=rank)
            for head in range(messages.shape[1])
        )
    )


def _fit_codec_candidates(
    campaign: SyntheticCampaignData,
    selected: np.ndarray,
    locator: FittedLocator,
    message_sketch: FittedMessageSketch,
) -> tuple[FittedCodecCandidate, ...]:
    partition = campaign.fit
    selected_h = _tensor(partition.tensors["h"][selected])
    charts = locator.assign(selected_h)
    q_total = _tensor(partition.tensors["q_A"][selected])
    q_heads = _tensor(partition.tensors["q_A_head"][selected])
    v_total = _tensor(partition.tensors["v_A"][selected])
    v_heads = _tensor(partition.tensors["v_A_head"][selected])
    keys_all = row_keys(partition)
    keys = tuple(key for key, keep in zip(keys_all, selected, strict=True) if keep)
    carrier_vectors = _tensor(campaign.source.q_carrier)
    carrier_eigenvalues = _tensor(campaign.source.h4_fisher_diagonal)
    messages = _tensor(partition.tensors["message"][selected])
    retained_messages = message_sketch.transform(messages)
    reconstructed_messages = message_sketch.reconstruct(retained_messages)
    q_base = torch.einsum(
        "ahm,nam->nah",
        _tensor(campaign.source.output_weights),
        reconstructed_messages,
    )
    candidates: list[FittedCodecCandidate] = []
    for codec_id, q_rank, v_rank, ratio in CODEC_LADDER:
        q_codecs: list[QContributionCodec] = []
        v_codecs: list[VFlowCodec] = []
        for chart in range(locator.chart_count):
            mask = charts == chart
            chart_keys = tuple(key for key, keep in zip(keys, mask, strict=True) if bool(keep))
            weights = hierarchical_row_weights(chart_keys)
            q_moment = combined_qv_second_moment(
                q_total[mask], q_heads[mask], weights
            )
            v_moment = combined_qv_second_moment(
                v_total[mask], v_heads[mask], weights
            )
            q_codecs.append(
                fit_q_contribution_codec(
                    q_moment,
                    carrier_vectors,
                    carrier_eigenvalues,
                    rank=q_rank,
                    regularization_ratio=ratio,
                )
            )
            v_codecs.append(
                fit_v_flow_codec(
                    v_moment,
                    carrier_vectors,
                    carrier_eigenvalues,
                    rank=v_rank,
                    regularization_ratio=ratio,
                )
            )
        frozen_q_codecs = tuple(q_codecs)
        c_base = _apply_chart_codec(q_base, charts, frozen_q_codecs, "encode")
        c_teacher = _apply_chart_codec(q_heads, charts, frozen_q_codecs, "encode")
        delta = c_teacher - c_base
        epsilon_completion = torch.empty(
            (locator.chart_count, q_heads.shape[1]), dtype=torch.float64
        )
        epsilon_j0 = torch.empty_like(epsilon_completion)
        for chart in range(locator.chart_count):
            mask = charts == chart
            chart_keys = tuple(
                key for key, keep in zip(keys, mask, strict=True) if bool(keep)
            )
            weights = hierarchical_row_weights(chart_keys)
            for head in range(q_heads.shape[1]):
                completion_energy = delta[mask, head, :].square().sum(-1)
                full_energy = c_teacher[mask, head, :].square().sum(-1)
                epsilon_completion[chart, head] = max(
                    1e-12,
                    1e-4
                    * tensor_weighted_median(
                        completion_energy[:, None], weights, chart_keys
                    ),
                )
                epsilon_j0[chart, head] = max(
                    1e-12,
                    1e-4
                    * tensor_weighted_median(
                        full_energy[:, None], weights, chart_keys
                    ),
                )
        candidates.append(
            FittedCodecCandidate(
                codec_id,
                q_rank,
                v_rank,
                ratio,
                frozen_q_codecs,
                tuple(v_codecs),
                epsilon_completion,
                epsilon_j0,
            )
        )
    return tuple(candidates)


def fit_selection_geometry(
    campaign: SyntheticCampaignData,
    *,
    attempted_locator_k: Sequence[int] = (4, 2),
) -> SelectionGeometry:
    """Fit the immutable fit-train representation under one K ladder.

    The full ``(4, 2)`` ladder is used for the initial representation attempt.
    A caller may pass ``(2,)`` only when the preregistered lower-K retry has
    already been triggered by a representation-level support failure (for
    example, J4's no-positive-operator-reference condition).
    """

    attempted = tuple(attempted_locator_k)
    if attempted not in {(4, 2), (2,)}:
        raise QualificationError(
            "attempted_locator_k must be the frozen (4, 2) ladder or lower-K (2,) retry"
        )
    selected = _selector(campaign.fit, campaign.fit_split.right_units)
    locator = _fit_locator(
        campaign.fit,
        selected,
        rank=campaign.config.compact_dimension,
        tensor_name="h",
        atlas_name="A_in",
        attempted_k=attempted,
    )
    mid_locator = _fit_locator(
        campaign.fit,
        selected,
        rank=campaign.config.compact_dimension,
        tensor_name="h_A",
        atlas_name="A_mid",
        attempted_k=attempted,
    )
    out_locator = _fit_locator(
        campaign.fit,
        selected,
        rank=campaign.config.compact_dimension,
        tensor_name="h_plus",
        atlas_name="A_out",
        attempted_k=attempted,
    )
    sketch = _fit_message_sketch(
        campaign.fit,
        selected,
        rank=campaign.config.compact_dimension,
    )
    candidates = _fit_codec_candidates(campaign, selected, locator, sketch)
    return SelectionGeometry(locator, mid_locator, out_locator, sketch, candidates)


@dataclass(frozen=True, slots=True)
class O1Tolerance:
    seam: str
    reference_scale: float
    calibration_max_abs_error: float
    calibration_weighted_relative_l2: float
    cap_abs: float
    cap_rel: float
    tau_abs: float
    tau_rel: float
    calibration_passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "seam": self.seam,
            "reference_scale": self.reference_scale,
            "calibration_max_abs_error": self.calibration_max_abs_error,
            "calibration_weighted_relative_l2": self.calibration_weighted_relative_l2,
            "cap_abs": self.cap_abs,
            "cap_rel": self.cap_rel,
            "tau_abs": self.tau_abs,
            "tau_rel": self.tau_rel,
            "calibration_passed": self.calibration_passed,
        }


@dataclass(frozen=True, slots=True)
class O1SeamResult:
    seam: str
    observed_max_abs_error: float
    observed_weighted_relative_l2: float
    tau_abs: float
    tau_rel: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "seam": self.seam,
            "observed_max_abs_error": self.observed_max_abs_error,
            "observed_weighted_relative_l2": self.observed_weighted_relative_l2,
            "tau_abs": self.tau_abs,
            "tau_rel": self.tau_rel,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class O1Qualification:
    tolerances: tuple[O1Tolerance, ...]
    validation: tuple[O1SeamResult, ...]
    complete_fit: tuple[O1SeamResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.calibration_passed for item in self.tolerances) and all(
            item.passed for item in self.validation + self.complete_fit
        )


def _o1_pairs(
    campaign: SyntheticCampaignData,
) -> Mapping[str, tuple[Tensor, Tensor]]:
    tensors = campaign.fit.tensors
    source = campaign.source
    message = tensors["message"].astype(np.float64)
    recomputed_heads = np.einsum(
        "ahm,nam->nah", source.output_weights, message, optimize=True
    )
    # einsum labels above produce [n,a,h].
    q_from_heads = recomputed_heads.sum(axis=1)
    v_from_q = synthetic_rmsnorm(q_from_heads, source.post_attention_norm_weight)
    h_a_from_q = tensors["h"].astype(np.float64) + v_from_q
    blocks = exact_synthetic_mlp_blocks(tensors["h_A"], source)
    q_m_from_blocks = tensors["q_M_block"].astype(np.float64).sum(axis=1)
    q_m_exact = blocks.sum(axis=1)
    _, v_m = exact_synthetic_mlp(tensors["h_A"], source)
    h_plus = tensors["h_A"].astype(np.float64) + v_m
    return {
        "attention_q_head_sum": (_tensor(tensors["q_A"]), _tensor(tensors["q_A_head"].sum(axis=1))),
        "attention_q_total": (_tensor(tensors["q_A"]), _tensor(q_from_heads)),
        "attention_h_A": (_tensor(tensors["h_A"]), _tensor(h_a_from_q)),
        "mlp_q_block_sum": (_tensor(tensors["q_M"]), _tensor(q_m_from_blocks)),
        "mlp_q_total": (_tensor(tensors["q_M"]), _tensor(q_m_exact)),
        "mlp_h_plus": (_tensor(tensors["h_plus"]), _tensor(h_plus)),
    }


def _seam_observation(
    reference: Tensor,
    reconstruction: Tensor,
    keys: Sequence[HierarchicalRowKey],
) -> tuple[float, float, float]:
    weights = hierarchical_row_weights(keys)
    error = float((reconstruction - reference).abs().max())
    relative = weighted_relative_l2(reference, reconstruction, weights)
    scale = max(
        1e-8,
        tensor_weighted_median(reference, weights, keys, absolute=True),
    )
    return error, relative, scale


def build_o1_qualification(campaign: SyntheticCampaignData) -> O1Qualification:
    pairs = _o1_pairs(campaign)
    keys_all = row_keys(campaign.fit)
    calibration_mask = _selector(campaign.fit, campaign.o1_split.left_units)
    validation_mask = _selector(campaign.fit, campaign.o1_split.right_units)

    tolerances: list[O1Tolerance] = []
    validation: list[O1SeamResult] = []
    complete: list[O1SeamResult] = []
    epsilon64 = float(torch.finfo(torch.float64).eps)
    for seam in _O1_SEAMS:
        reference, reconstruction = pairs[seam]
        calibration_keys = tuple(
            key for key, keep in zip(keys_all, calibration_mask, strict=True) if keep
        )
        cal_error, cal_relative, scale = _seam_observation(
            reference[calibration_mask],
            reconstruction[calibration_mask],
            calibration_keys,
        )
        cap_abs = 1e-4 * scale + 1e-6
        cap_rel = 1e-4
        tau_abs = min(
            cap_abs,
            max(10.0 * cal_error, 100.0 * epsilon64 * scale, 1e-12),
        )
        tau_rel = min(
            cap_rel,
            max(10.0 * cal_relative, 100.0 * epsilon64, 1e-12),
        )
        tolerance = O1Tolerance(
            seam,
            scale,
            cal_error,
            cal_relative,
            cap_abs,
            cap_rel,
            tau_abs,
            tau_rel,
            cal_error <= cap_abs and cal_relative <= cap_rel,
        )
        tolerances.append(tolerance)
        for mask, destination in (
            (validation_mask, validation),
            (np.ones(len(keys_all), dtype=bool), complete),
        ):
            keys = tuple(key for key, keep in zip(keys_all, mask, strict=True) if keep)
            error, relative, _ = _seam_observation(
                reference[mask], reconstruction[mask], keys
            )
            destination.append(
                O1SeamResult(
                    seam,
                    error,
                    relative,
                    tau_abs,
                    tau_rel,
                    error <= tau_abs and relative <= tau_rel,
                )
            )
    return O1Qualification(tuple(tolerances), tuple(validation), tuple(complete))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def evaluate_fidelity(
    partition: SyntheticPartition,
    candidate_h_plus: np.ndarray | Tensor,
    campaign: SyntheticCampaignData,
) -> FidelityMetrics:
    candidate = (
        candidate_h_plus.detach().cpu().numpy()
        if isinstance(candidate_h_plus, Tensor)
        else np.asarray(candidate_h_plus)
    ).astype(np.float64)
    teacher_h = partition.tensors["h_plus"].astype(np.float64)
    if candidate.shape != teacher_h.shape:
        raise QualificationError("candidate h_plus shape differs from teacher")
    teacher_logits = teacher_h @ campaign.source.lm_head.T
    candidate_logits = candidate @ campaign.source.lm_head.T
    teacher_log_prob = _log_softmax(teacher_logits)
    candidate_log_prob = _log_softmax(candidate_logits)
    teacher_prob = np.exp(teacher_log_prob)
    target = partition.tensors["target_token_id"].astype(np.int64)
    source_nll = -teacher_log_prob[np.arange(len(target)), target]
    candidate_nll = -candidate_log_prob[np.arange(len(target)), target]
    kl = np.sum(
        teacher_prob * (teacher_log_prob - candidate_log_prob), axis=-1
    )
    matches = np.argmax(teacher_logits, axis=-1) == np.argmax(candidate_logits, axis=-1)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(partition.metadata):
        grouped[metadata.canonical_sequence_hash].append(index)
    observations = []
    for sequence_hash in sorted(grouped):
        indexes = np.asarray(grouped[sequence_hash], dtype=np.int64)
        observations.append(
            FidelityPromptObservation(
                prompt_id=sequence_hash,
                supervised_tokens=len(indexes),
                source_summed_nll=float(source_nll[indexes].sum()),
                candidate_summed_nll=float(candidate_nll[indexes].sum()),
                source_to_candidate_summed_kl=max(0.0, float(kl[indexes].sum())),
                top1_matches=int(matches[indexes].sum()),
            )
        )
    return summarize_fidelity(observations)


def _candidate_h_plus_from_q(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
    candidate: FittedCodecCandidate,
    locator: FittedLocator,
) -> Tensor:
    h = _tensor(partition.tensors["h"])
    charts = locator.assign(h)
    q = candidate.project_q(_tensor(partition.tensors["q_A"]), charts)
    v = _tensor(synthetic_rmsnorm(q.numpy(), campaign.source.post_attention_norm_weight))
    h_a = h + v
    _, v_m = exact_synthetic_mlp(h_a.numpy(), campaign.source)
    return h_a + _tensor(v_m)


def _candidate_h_plus_from_v(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
    candidate: FittedCodecCandidate,
    locator: FittedLocator,
) -> Tensor:
    h = _tensor(partition.tensors["h"])
    charts = locator.assign(h)
    v = candidate.project_v(_tensor(partition.tensors["v_A"]), charts)
    h_a = h + v
    _, v_m = exact_synthetic_mlp(h_a.numpy(), campaign.source)
    return h_a + _tensor(v_m)


def evaluate_o2(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> tuple[tuple[O2CodecEvaluation, ...], O2SelectionDecision]:
    fit_monitor_mask = _selector(campaign.fit, campaign.fit_split.left_units)
    # Preserve metadata alignment by creating a lightweight selected partition.
    monitor = _subset_partition(campaign.fit, fit_monitor_mask, "fit")
    evaluations: list[O2CodecEvaluation] = []
    for candidate in geometry.codec_candidates:
        monitor_q = evaluate_fidelity(
            monitor,
            _candidate_h_plus_from_q(monitor, campaign, candidate, geometry.locator),
            campaign,
        )
        tune_q = evaluate_fidelity(
            campaign.tune,
            _candidate_h_plus_from_q(
                campaign.tune, campaign, candidate, geometry.locator
            ),
            campaign,
        )
        monitor_v = evaluate_fidelity(
            monitor,
            _candidate_h_plus_from_v(monitor, campaign, candidate, geometry.locator),
            campaign,
        )
        tune_v = evaluate_fidelity(
            campaign.tune,
            _candidate_h_plus_from_v(
                campaign.tune, campaign, candidate, geometry.locator
            ),
            campaign,
        )
        evaluations.append(
            O2CodecEvaluation(
                candidate.codec_id,
                candidate.q_rank,
                candidate.v_rank,
                candidate.regularization_ratio,
                monitor_q,
                tune_q,
                monitor_v,
                tune_v,
            )
        )
    frozen = tuple(evaluations)
    return frozen, select_o2q_codec(frozen)


def evaluate_o2q_replay(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec_id: str,
) -> FidelityMetrics:
    """Replay the frozen selected q codec on one later complete panel."""

    candidate = geometry.codec(codec_id)
    return evaluate_fidelity(
        partition,
        _candidate_h_plus_from_q(partition, campaign, candidate, geometry.locator),
        campaign,
    )


def _subset_partition(
    partition: SyntheticPartition, mask: np.ndarray, role: str
) -> SyntheticPartition:
    from .fisher_compact_message_completion_protocol import build_row_index_payload

    selected_rows = tuple(
        row for row, keep in zip(partition.row_index.rows, mask, strict=True) if keep
    )
    row_index = build_row_index_payload(
        role,
        partition.row_index.row_derivation_protocol_hash,
        selected_rows,
    )
    original_by_id = {
        metadata.row_id: index for index, metadata in enumerate(partition.metadata)
    }
    indexes = np.asarray([original_by_id[row_id] for row_id in row_index.row_ids])
    return SyntheticPartition(
        role,
        row_index,
        tuple(partition.metadata[index] for index in indexes),
        {
            name: np.ascontiguousarray(value[indexes])
            for name, value in partition.tensors.items()
        },
    )


def subset_partition_by_units(
    partition: SyntheticPartition,
    unit_commitments: Sequence[str],
    *,
    partition_role: str | None = None,
) -> SyntheticPartition:
    """Return a canonical, row-aligned view containing only whole units."""

    return _subset_partition(
        partition,
        _selector(partition, unit_commitments),
        partition_role or partition.partition_role,
    )


@dataclass(frozen=True, slots=True)
class PreparedCompletionPanel:
    partition_role: str
    chart_ids: Tensor
    active_chart_ids: tuple[int, ...]
    support_mask: Tensor
    z: Tensor
    r: Tensor
    c_base: Tensor
    c_teacher: Tensor
    delta_c: Tensor
    epsilon_completion: Tensor
    epsilon_j0: Tensor
    row_keys: tuple[HierarchicalRowKey, ...]
    h: Tensor
    h_plus: Tensor


@dataclass(frozen=True, slots=True)
class DevelopmentSupportAssessment:
    evaluable_analysis_unit_count: int
    chart_analysis_unit_counts: tuple[tuple[int, int], ...]
    chart_token_fractions: tuple[tuple[int, float], ...]
    out_of_support_token_count: int
    out_of_support_rate: float
    out_of_support_passed: bool
    aggregate_unit_support_passed: bool
    per_chart_unit_support_passed: bool
    underpowered_chart_ids: tuple[int, ...]
    materially_underpowered_chart_ids: tuple[int, ...]

    @property
    def independent_support_passed(self) -> bool:
        return (
            self.aggregate_unit_support_passed
            and self.per_chart_unit_support_passed
        )

    @property
    def insufficient_independent_support(self) -> bool:
        return (
            not self.aggregate_unit_support_passed
            or bool(self.materially_underpowered_chart_ids)
        )

    @property
    def development_support_passed(self) -> bool:
        return self.out_of_support_passed and self.independent_support_passed


def assess_development_support(
    panel: PreparedCompletionPanel,
) -> DevelopmentSupportAssessment:
    """Apply the frozen development support and out-of-support thresholds."""

    if panel.chart_ids.numel() == 0 or panel.support_mask.shape != panel.chart_ids.shape:
        raise QualificationError("development support panel is empty or misaligned")
    unit_ids = tuple(key.analysis_unit_commitment for key in panel.row_keys)
    if len(unit_ids) != panel.chart_ids.numel():
        raise QualificationError("development support row keys are misaligned")
    chart_values = tuple(int(value) for value in panel.chart_ids.tolist())
    active = panel.active_chart_ids
    if not active or tuple(sorted(set(active))) != active:
        raise QualificationError("active development chart inventory is invalid")
    supported_values = tuple(bool(value) for value in panel.support_mask.tolist())
    chart_counts = tuple(
        (
            chart,
            len(
                {
                    unit
                    for unit, assigned, supported in zip(
                        unit_ids, chart_values, supported_values, strict=True
                    )
                    if assigned == chart and supported
                }
            ),
        )
        for chart in active
    )
    total_rows = len(chart_values)
    chart_fractions = tuple(
        (
            chart,
            sum(assigned == chart for assigned in chart_values) / total_rows,
        )
        for chart in active
    )
    underpowered = tuple(chart for chart, count in chart_counts if count < 4)
    fraction_by_chart = dict(chart_fractions)
    material = tuple(
        chart for chart in underpowered if fraction_by_chart[chart] > 0.10
    )
    unsupported = int((~panel.support_mask).sum())
    out_rate = unsupported / total_rows
    evaluable_units = {
        unit
        for unit, supported in zip(unit_ids, supported_values, strict=True)
        if supported
    }
    return DevelopmentSupportAssessment(
        evaluable_analysis_unit_count=len(evaluable_units),
        chart_analysis_unit_counts=chart_counts,
        chart_token_fractions=chart_fractions,
        out_of_support_token_count=unsupported,
        out_of_support_rate=out_rate,
        out_of_support_passed=out_rate <= 0.10,
        aggregate_unit_support_passed=len(evaluable_units) >= 18,
        per_chart_unit_support_passed=not material,
        underpowered_chart_ids=underpowered,
        materially_underpowered_chart_ids=material,
    )


def prepare_completion_panel(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec_id: str,
) -> PreparedCompletionPanel:
    codec = geometry.codec(codec_id)
    h = _tensor(partition.tensors["h"])
    charts, z, support_mask = geometry.locator.transform_with_support(h)
    messages = _tensor(partition.tensors["message"])
    r = geometry.message_sketch.transform(messages)
    reconstructed = geometry.message_sketch.reconstruct(r)
    q_base_ambient = torch.einsum(
        "ahm,nam->nah",
        _tensor(campaign.source.output_weights),
        reconstructed,
    )
    q_teacher = _tensor(partition.tensors["q_A_head"])
    c_base = codec.encode_q(q_base_ambient, charts)
    c_teacher = codec.encode_q(q_teacher, charts)
    delta = c_teacher - c_base
    epsilon = codec.epsilon_completion_by_chart_head[charts]
    epsilon_j0 = codec.epsilon_j0_by_chart_head[charts]
    keys = row_keys(partition)
    return PreparedCompletionPanel(
        partition.partition_role,
        charts,
        tuple(range(geometry.locator.chart_count)),
        support_mask,
        z,
        r,
        c_base,
        c_teacher,
        delta,
        epsilon,
        epsilon_j0,
        keys,
        h,
        _tensor(partition.tensors["h_plus"]),
    )


def fidelity_gate_records(metrics: FidelityMetrics) -> list[dict[str, object]]:
    decision = evaluate_five_fidelity_gates(metrics)
    return [
        {
            "gate_id": "absolute_global_delta_nll_per_token",
            "observed": abs(metrics.delta_nll_per_token),
            "threshold": 0.05,
            "comparison": "<=",
            "passed": decision.absolute_delta_nll_passed,
        },
        {
            "gate_id": "global_kl_per_token",
            "observed": metrics.source_to_candidate_kl_per_token,
            "threshold": 0.05,
            "comparison": "<=",
            "passed": decision.kl_passed,
        },
        {
            "gate_id": "global_top1_agreement",
            "observed": metrics.top1_agreement,
            "threshold": 0.95,
            "comparison": ">=",
            "passed": decision.top1_passed,
        },
        {
            "gate_id": "p90_absolute_prompt_delta_nll_per_token",
            "observed": metrics.prompt_p90_absolute_delta_nll_per_token,
            "threshold": 0.10,
            "comparison": "<=",
            "passed": decision.prompt_p90_absolute_delta_nll_passed,
        },
        {
            "gate_id": "p10_prompt_top1_agreement",
            "observed": metrics.prompt_p10_top1_agreement,
            "threshold": 0.90,
            "comparison": ">=",
            "passed": decision.prompt_p10_top1_passed,
        },
    ]
