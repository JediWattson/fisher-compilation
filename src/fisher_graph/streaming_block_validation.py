"""Multi-layer geometry and bounded-memory transport for Fisher modes.

A low overlap between modes at two depths has two very different possible
causes:

* the modes at one or both boundaries are not reproducible across data splits;
* both boundaries are reproducible, but the represented subspace changes
  coherently as the residual stream passes through the intervening block.

This module keeps those measurements separate.  Split-replicate principal
angles measure *static boundary stability*.  Adjacent-depth principal angles
measure direct subspace drift in the shared residual coordinate system.
Finally, a streaming paired-row estimator projects transient rows into the two
modal bases and measures their whitened canonical correlations.  High
within-boundary stability, low direct depth overlap, and high paired canonical
correlation are evidence for a stable transported rotation rather than a noisy
boundary estimate.

Principal-angle geometry, canonical-correlation quality, and frozen-map
prediction error are invariant to eigenvector signs and orthogonal changes of
coordinates when their maps and bases transform together.  The optional
Procrustes transport matrix is gauge-covariant.  By contrast, the diagnostic
identity-coordinate baseline and its derived rotation gain depend on the
chosen ordered Fisher coordinate gauges; metadata labels that distinction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor

from .streaming_fisher import StreamingFisherResult


_GEOMETRY_FORMAT_VERSION = 1
_TRANSPORT_FORMAT_VERSION = 1
_TRANSPORT_ALGORITHM = "streaming_modal_whitened_cross_moment"
_TRANSPORT_ALGORITHM_VERSION = 1
_FROZEN_TRANSPORT_FORMAT_VERSION = 1
_FROZEN_TRANSPORT_ALGORITHM = "frozen_whitened_orthogonal_procrustes"
_FROZEN_TRANSPORT_ALGORITHM_VERSION = 1
_FROZEN_EVALUATION_FORMAT_VERSION = 2


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _basis_sha256(vectors: Tensor) -> str:
    canonical = vectors.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.modal_transport_basis.v1\0")
    digest.update(
        f"{canonical.shape[0]}x{canonical.shape[1]}\0float64\0".encode()
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validated_ranks(
    ranks: Iterable[int],
    *,
    maximum: int,
) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        values = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of positive integers"
        ) from error
    if not values:
        raise ValueError("ranks cannot be empty")
    if any(type(rank) is not int or rank <= 0 for rank in values):
        raise ValueError("ranks must contain positive integers")
    if any(rank > maximum for rank in values):
        raise ValueError(f"ranks cannot exceed the available mode count {maximum}")
    return tuple(sorted(set(values)))


def _validate_orthonormal_columns(
    vectors: Tensor,
    *,
    label: str,
) -> None:
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError(f"{label} must have shape [width, modes]")
    if vectors.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{label} must use float32 or float64")
    if not torch.isfinite(vectors).all():
        raise ValueError(f"{label} must be finite")
    converted = vectors.detach().to(device="cpu", dtype=torch.float64)
    identity = torch.eye(converted.shape[1], dtype=torch.float64)
    tolerance = 2e-5 if vectors.dtype == torch.float32 else 1e-10
    if not torch.allclose(
        converted.T @ converted,
        identity,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(f"{label} must have orthonormal columns")


def _orthonormalized_prefix(vectors: Tensor, rank: int) -> Tensor:
    prefix = vectors[:, :rank].detach().to(device="cpu", dtype=torch.float64)
    _validate_orthonormal_columns(prefix, label="Fisher basis prefix")
    orthonormal, triangular = torch.linalg.qr(prefix, mode="reduced")
    signs = triangular.diagonal().sign()
    signs[signs == 0] = 1
    return (orthonormal * signs).contiguous()


def _principal_cosines(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError("subspace bases must have the same shape")
    _validate_orthonormal_columns(left, label="left subspace")
    _validate_orthonormal_columns(right, label="right subspace")
    return torch.linalg.svdvals(
        left.to(dtype=torch.float64).T @ right.to(dtype=torch.float64)
    ).clamp(0, 1).cpu()


def _alignment_properties(cosines: Tensor) -> dict[str, float]:
    overlap = cosines.square().mean().item()
    minimum = cosines.min().item()
    return {
        "mean_squared_overlap": overlap,
        "minimum_principal_cosine": minimum,
        "largest_principal_angle_degrees": math.degrees(
            math.acos(min(max(minimum, 0.0), 1.0))
        ),
        "normalized_projection_distance": math.sqrt(
            max(0.0, 1.0 - overlap)
        ),
    }


@dataclass(frozen=True, slots=True)
class RankedSubspaceAlignment:
    """Sign- and rotation-invariant subspace alignment at one prefix rank."""

    rank: int
    principal_cosines: Tensor

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if self.principal_cosines.shape != (self.rank,):
            raise ValueError("principal_cosines must have shape [rank]")
        if self.principal_cosines.device.type != "cpu":
            raise ValueError("principal_cosines must be on CPU")
        if self.principal_cosines.dtype not in (torch.float32, torch.float64):
            raise ValueError("principal_cosines must be float32 or float64")
        if not torch.isfinite(self.principal_cosines).all():
            raise ValueError("principal_cosines must be finite")
        tolerance = 32 * torch.finfo(self.principal_cosines.dtype).eps
        if (self.principal_cosines < -tolerance).any() or (
            self.principal_cosines > 1 + tolerance
        ).any():
            raise ValueError("principal_cosines must lie in [0, 1]")

    @property
    def mean_squared_overlap(self) -> float:
        return _alignment_properties(self.principal_cosines)[
            "mean_squared_overlap"
        ]

    @property
    def minimum_principal_cosine(self) -> float:
        return _alignment_properties(self.principal_cosines)[
            "minimum_principal_cosine"
        ]

    @property
    def largest_principal_angle_degrees(self) -> float:
        return _alignment_properties(self.principal_cosines)[
            "largest_principal_angle_degrees"
        ]

    @property
    def normalized_projection_distance(self) -> float:
        return _alignment_properties(self.principal_cosines)[
            "normalized_projection_distance"
        ]

    def metadata(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "principal_cosines": self.principal_cosines.tolist(),
            **_alignment_properties(self.principal_cosines),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "principal_cosines": self.principal_cosines,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> RankedSubspaceAlignment:
        if set(state) != {"rank", "principal_cosines"}:
            raise ValueError("ranked subspace alignment fields are invalid")
        cosines = state["principal_cosines"]
        if not isinstance(cosines, Tensor):
            raise TypeError("principal_cosines must be a Tensor")
        return cls(rank=int(state["rank"]), principal_cosines=cosines)


@dataclass(frozen=True, slots=True)
class ModalSubspaceAlignment:
    """Rank curve for either split stability or adjacent-depth alignment."""

    relation: str
    source_layer: str
    target_layer: str
    source_observations: int
    target_observations: int
    points: tuple[RankedSubspaceAlignment, ...]

    def __post_init__(self) -> None:
        if self.relation not in {"split_replicate", "adjacent_depth"}:
            raise ValueError("unsupported subspace alignment relation")
        for label, value in (
            ("source_layer", self.source_layer),
            ("target_layer", self.target_layer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if self.relation == "split_replicate" and (
            self.source_layer != self.target_layer
        ):
            raise ValueError("split replicate alignment must name one layer")
        if self.relation == "adjacent_depth" and (
            self.source_layer == self.target_layer
        ):
            raise ValueError("adjacent depth alignment requires two layers")
        for label, value in (
            ("source_observations", self.source_observations),
            ("target_observations", self.target_observations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if not self.points:
            raise ValueError("subspace alignment points cannot be empty")
        ranks = tuple(point.rank for point in self.points)
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("subspace alignment ranks must be unique and sorted")

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(point.rank for point in self.points)

    def at_rank(self, rank: int) -> RankedSubspaceAlignment:
        for point in self.points:
            if point.rank == rank:
                return point
        raise KeyError(f"rank {rank} is not present")

    def metadata(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "source_observations": self.source_observations,
            "target_observations": self.target_observations,
            "points": [point.metadata() for point in self.points],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "source_observations": self.source_observations,
            "target_observations": self.target_observations,
            "points": [point.state_dict() for point in self.points],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalSubspaceAlignment:
        expected = {
            "relation",
            "source_layer",
            "target_layer",
            "source_observations",
            "target_observations",
            "points",
        }
        if set(state) != expected:
            raise ValueError("modal subspace alignment fields are invalid")
        raw_points = state["points"]
        if not isinstance(raw_points, list):
            raise TypeError("modal subspace alignment points must be a list")
        points = []
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping):
                raise TypeError("ranked alignment state must be a mapping")
            points.append(RankedSubspaceAlignment.from_state_dict(raw_point))
        return cls(
            relation=str(state["relation"]),
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            source_observations=int(state["source_observations"]),
            target_observations=int(state["target_observations"]),
            points=tuple(points),
        )


@dataclass(frozen=True, slots=True)
class ModalTrajectoryGeometry:
    """Static split stability and direct depth geometry for ordered layers."""

    layers: tuple[str, ...]
    ranks: tuple[int, ...]
    boundary_stability: tuple[ModalSubspaceAlignment, ...]
    depth_alignment: tuple[ModalSubspaceAlignment, ...]
    scope: str
    score_reduction: str
    normalizer: str

    def __post_init__(self) -> None:
        if not self.layers or any(
            not isinstance(layer, str) or not layer for layer in self.layers
        ):
            raise ValueError("layers must contain nonempty names")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layer names must be unique")
        if (
            not self.ranks
            or self.ranks != tuple(sorted(set(self.ranks)))
            or any(type(rank) is not int or rank <= 0 for rank in self.ranks)
        ):
            raise ValueError("ranks must be positive, unique, and sorted")
        if len(self.depth_alignment) != max(len(self.layers) - 1, 0):
            raise ValueError("depth alignments must connect every adjacent layer")
        expected_depth_pairs = tuple(zip(self.layers, self.layers[1:]))
        actual_depth_pairs = tuple(
            (item.source_layer, item.target_layer)
            for item in self.depth_alignment
        )
        if actual_depth_pairs != expected_depth_pairs:
            raise ValueError("depth alignments do not follow layer order")
        if any(
            item.relation != "adjacent_depth" or item.ranks != self.ranks
            for item in self.depth_alignment
        ):
            raise ValueError("depth alignment semantics are inconsistent")
        if self.boundary_stability:
            if tuple(
                item.source_layer for item in self.boundary_stability
            ) != self.layers:
                raise ValueError(
                    "boundary stability must contain every ordered layer"
                )
            if any(
                item.relation != "split_replicate"
                or item.ranks != self.ranks
                for item in self.boundary_stability
            ):
                raise ValueError("boundary stability semantics are inconsistent")
        for label, value in (
            ("scope", self.scope),
            ("score_reduction", self.score_reduction),
            ("normalizer", self.normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")

    def boundary(self, layer: str) -> ModalSubspaceAlignment:
        for item in self.boundary_stability:
            if item.source_layer == layer:
                return item
        raise KeyError(f"no split replicate is available for {layer!r}")

    def transition(
        self,
        source_layer: str,
        target_layer: str,
    ) -> ModalSubspaceAlignment:
        for item in self.depth_alignment:
            if (
                item.source_layer == source_layer
                and item.target_layer == target_layer
            ):
                return item
        raise KeyError(
            f"no adjacent-depth alignment for {source_layer!r} -> "
            f"{target_layer!r}"
        )

    def metadata(self) -> dict[str, object]:
        return {
            "layers": self.layers,
            "ranks": self.ranks,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "boundary_stability": [
                item.metadata() for item in self.boundary_stability
            ],
            "depth_alignment": [
                item.metadata() for item in self.depth_alignment
            ],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _GEOMETRY_FORMAT_VERSION,
            "layers": self.layers,
            "ranks": self.ranks,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "boundary_stability": [
                item.state_dict() for item in self.boundary_stability
            ],
            "depth_alignment": [
                item.state_dict() for item in self.depth_alignment
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalTrajectoryGeometry:
        expected = {
            "format_version",
            "layers",
            "ranks",
            "scope",
            "score_reduction",
            "normalizer",
            "boundary_stability",
            "depth_alignment",
        }
        if set(state) != expected:
            raise ValueError("modal trajectory geometry fields are invalid")
        if state["format_version"] != _GEOMETRY_FORMAT_VERSION:
            raise ValueError("unsupported modal trajectory geometry format")
        raw_boundary = state["boundary_stability"]
        raw_depth = state["depth_alignment"]
        if not isinstance(raw_boundary, list) or not isinstance(raw_depth, list):
            raise TypeError("trajectory alignments must be lists")
        boundary = []
        for item in raw_boundary:
            if not isinstance(item, Mapping):
                raise TypeError("boundary alignment state must be a mapping")
            boundary.append(ModalSubspaceAlignment.from_state_dict(item))
        depth = []
        for item in raw_depth:
            if not isinstance(item, Mapping):
                raise TypeError("depth alignment state must be a mapping")
            depth.append(ModalSubspaceAlignment.from_state_dict(item))
        raw_layers = state["layers"]
        raw_ranks = state["ranks"]
        if not isinstance(raw_layers, tuple) or not isinstance(raw_ranks, tuple):
            raise TypeError("trajectory layers and ranks must be tuples")
        return cls(
            layers=raw_layers,
            ranks=raw_ranks,
            boundary_stability=tuple(boundary),
            depth_alignment=tuple(depth),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
        )


def analyze_modal_subspace_trajectory(
    reference: Sequence[StreamingFisherResult],
    *,
    ranks: Iterable[int],
    replicate: Sequence[StreamingFisherResult] | None = None,
    layer_names: Sequence[str] | None = None,
) -> ModalTrajectoryGeometry:
    """Analyze reproducibility and direct subspace drift across ordered layers.

    ``reference`` supplies one Fisher basis per depth.  ``replicate`` may
    supply an independently calibrated basis for each same boundary.  Direct
    depth angles require adjacent residual widths to match; paired transport
    across unequal widths can still be measured separately with
    :class:`StreamingModalTransportEstimator`.
    """

    references = tuple(reference)
    if not references:
        raise ValueError("reference bases cannot be empty")
    if any(not isinstance(item, StreamingFisherResult) for item in references):
        raise TypeError("reference must contain StreamingFisherResult values")
    names = (
        tuple(item.activation_name for item in references)
        if layer_names is None
        else tuple(layer_names)
    )
    if len(names) != len(references):
        raise ValueError("layer_names must match the reference basis count")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("layer_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("layer_names must be unique")

    replicates: tuple[StreamingFisherResult, ...] | None = None
    if replicate is not None:
        replicates = tuple(replicate)
        if len(replicates) != len(references) or any(
            not isinstance(item, StreamingFisherResult)
            for item in replicates
        ):
            raise TypeError(
                "replicate must contain one StreamingFisherResult per layer"
            )

    maximum = min(
        item.modes
        for item in (
            references
            if replicates is None
            else references + replicates
        )
    )
    requested_ranks = _validated_ranks(ranks, maximum=maximum)
    maximum_rank = max(requested_ranks)
    provenance = (
        references[0].scope,
        references[0].score_reduction,
        references[0].normalizer,
    )
    for result in (
        references if replicates is None else references + replicates
    ):
        if (
            result.scope,
            result.score_reduction,
            result.normalizer,
        ) != provenance:
            raise ValueError("Fisher bases disagree on scientific provenance")
        _validate_orthonormal_columns(
            result.vectors[:, :maximum_rank],
            label=f"{result.activation_name!r} Fisher vectors",
        )
    if replicates is not None:
        for left, right in zip(references, replicates):
            if (
                left.activation_name != right.activation_name
                or left.width != right.width
            ):
                raise ValueError(
                    "replicate Fisher bases must match their reference boundary"
                )

    def alignment_points(
        left: StreamingFisherResult,
        right: StreamingFisherResult,
    ) -> tuple[RankedSubspaceAlignment, ...]:
        return tuple(
            RankedSubspaceAlignment(
                rank=rank,
                principal_cosines=_principal_cosines(
                    left.vectors[:, :rank],
                    right.vectors[:, :rank],
                ),
            )
            for rank in requested_ranks
        )

    boundary = ()
    if replicates is not None:
        boundary = tuple(
            ModalSubspaceAlignment(
                relation="split_replicate",
                source_layer=name,
                target_layer=name,
                source_observations=left.observations,
                target_observations=right.observations,
                points=alignment_points(left, right),
            )
            for name, left, right in zip(names, references, replicates)
        )
    depth = []
    for index, (left, right) in enumerate(zip(references, references[1:])):
        if left.width != right.width:
            raise ValueError(
                "adjacent basis widths must match for direct depth geometry"
            )
        depth.append(
            ModalSubspaceAlignment(
                relation="adjacent_depth",
                source_layer=names[index],
                target_layer=names[index + 1],
                source_observations=left.observations,
                target_observations=right.observations,
                points=alignment_points(left, right),
            )
        )
    return ModalTrajectoryGeometry(
        layers=names,
        ranks=requested_ranks,
        boundary_stability=boundary,
        depth_alignment=tuple(depth),
        scope=provenance[0],
        score_reduction=provenance[1],
        normalizer=provenance[2],
    )


def _symmetric_psd_inverse_sqrt(
    matrix: Tensor,
    *,
    relative_cutoff: float,
) -> tuple[Tensor, int]:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    largest = max(eigenvalues.max().item(), 0.0)
    tolerance = max(
        largest * relative_cutoff,
        torch.finfo(matrix.dtype).eps * max(matrix.shape[0], 1) * 32,
    )
    if eigenvalues.min().item() < -tolerance:
        raise ValueError("modal second moment is not positive semidefinite")
    supported = eigenvalues > tolerance
    inverse = torch.zeros_like(eigenvalues)
    inverse[supported] = eigenvalues[supported].rsqrt()
    return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T, int(
        supported.sum().item()
    )


def _symmetric_psd_sqrt(
    matrix: Tensor,
    *,
    relative_cutoff: float,
) -> Tensor:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    largest = max(eigenvalues.max().item(), 0.0)
    tolerance = max(
        largest * relative_cutoff,
        torch.finfo(matrix.dtype).eps * max(matrix.shape[0], 1) * 32,
    )
    if eigenvalues.min().item() < -tolerance:
        raise ValueError("modal second moment is not positive semidefinite")
    roots = eigenvalues.clamp_min(0).sqrt()
    return (eigenvectors * roots.unsqueeze(0)) @ eigenvectors.T


@dataclass(frozen=True, slots=True)
class ModalTransportPoint:
    """Whitened paired-row transport diagnostics at one modal prefix."""

    rank: int
    canonical_correlations: Tensor
    orthogonal_transport: Tensor
    source_effective_rank: int
    target_effective_rank: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.canonical_correlations.shape != (self.rank,):
            raise ValueError("canonical_correlations must have shape [rank]")
        if self.orthogonal_transport.shape != (self.rank, self.rank):
            raise ValueError("orthogonal_transport must have shape [rank, rank]")
        for tensor in (
            self.canonical_correlations,
            self.orthogonal_transport,
        ):
            if tensor.device.type != "cpu" or tensor.dtype not in (
                torch.float32,
                torch.float64,
            ):
                raise ValueError("transport tensors must be CPU float32 or float64")
            if not torch.isfinite(tensor).all():
                raise ValueError("transport tensors must be finite")
        if self.orthogonal_transport.dtype != self.canonical_correlations.dtype:
            raise ValueError("transport tensors must share one dtype")
        tolerance = 64 * torch.finfo(
            self.canonical_correlations.dtype
        ).eps
        if (self.canonical_correlations < -tolerance).any() or (
            self.canonical_correlations > 1 + tolerance
        ).any():
            raise ValueError("canonical correlations must lie in [0, 1]")
        for label, value in (
            ("source_effective_rank", self.source_effective_rank),
            ("target_effective_rank", self.target_effective_rank),
        ):
            if type(value) is not int or not 0 <= value <= self.rank:
                raise ValueError(f"{label} is out of range")
        identity = torch.eye(
            self.rank,
            dtype=self.orthogonal_transport.dtype,
        )
        orthogonal_tolerance = (
            2e-5
            if self.orthogonal_transport.dtype == torch.float32
            else 1e-8
        )
        if not torch.allclose(
            self.orthogonal_transport.T @ self.orthogonal_transport,
            identity,
            rtol=orthogonal_tolerance,
            atol=orthogonal_tolerance,
        ):
            raise ValueError("orthogonal_transport must be orthogonal")

    @property
    def mean_squared_canonical_correlation(self) -> float:
        """Coherent paired transport over all requested modal directions."""

        return self.canonical_correlations.square().mean().item()

    @property
    def supported_mean_squared_canonical_correlation(self) -> float:
        """Transport coherence over the jointly supported modal rank."""

        supported = min(
            self.source_effective_rank,
            self.target_effective_rank,
        )
        if supported == 0:
            return 0.0
        return (
            self.canonical_correlations[:supported]
            .square()
            .mean()
            .item()
        )

    def metadata(
        self,
        *,
        include_transport: bool = False,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "rank": self.rank,
            "canonical_correlations": self.canonical_correlations.tolist(),
            "source_effective_rank": self.source_effective_rank,
            "target_effective_rank": self.target_effective_rank,
            "mean_squared_canonical_correlation": (
                self.mean_squared_canonical_correlation
            ),
            "supported_mean_squared_canonical_correlation": (
                self.supported_mean_squared_canonical_correlation
            ),
        }
        if include_transport:
            metadata["orthogonal_transport"] = (
                self.orthogonal_transport.tolist()
            )
        return metadata


@dataclass(frozen=True, slots=True)
class StreamingModalTransportResult:
    """Sufficient statistics for rank-prefix paired modal transport."""

    source_layer: str
    target_layer: str
    row_kind: str
    centered: bool
    source_width: int
    target_width: int
    source_basis_sha256: str
    target_basis_sha256: str
    observations: int
    rows_seen: int
    source_sum: Tensor
    target_sum: Tensor
    source_gram_sum: Tensor
    target_gram_sum: Tensor
    cross_sum: Tensor
    accumulation_dtype: str
    relative_eigenvalue_cutoff: float
    scope: str
    score_reduction: str
    normalizer: str
    algorithm: str = _TRANSPORT_ALGORITHM
    algorithm_version: int = _TRANSPORT_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("source_layer", self.source_layer),
            ("target_layer", self.target_layer),
            ("row_kind", self.row_kind),
            ("accumulation_dtype", self.accumulation_dtype),
            ("scope", self.scope),
            ("score_reduction", self.score_reduction),
            ("normalizer", self.normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if self.source_layer == self.target_layer:
            raise ValueError("transport must connect distinct layers")
        if type(self.centered) is not bool:
            raise TypeError("centered must be a bool")
        for label, value in (
            ("source_width", self.source_width),
            ("target_width", self.target_width),
            ("observations", self.observations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if type(self.rows_seen) is not int or self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        rank = self.source_sum.numel()
        if rank == 0 or self.source_sum.shape != (rank,):
            raise ValueError("source_sum must be a nonempty vector")
        if self.target_sum.shape != (rank,):
            raise ValueError("target_sum must have shape [rank]")
        for label, tensor in (
            ("source_gram_sum", self.source_gram_sum),
            ("target_gram_sum", self.target_gram_sum),
            ("cross_sum", self.cross_sum),
        ):
            if tensor.shape != (rank, rank):
                raise ValueError(f"{label} must have shape [rank, rank]")
        tensors = (
            self.source_sum,
            self.target_sum,
            self.source_gram_sum,
            self.target_gram_sum,
            self.cross_sum,
        )
        if any(
            tensor.device.type != "cpu"
            or tensor.dtype not in (torch.float32, torch.float64)
            for tensor in tensors
        ):
            raise ValueError("transport statistics must be CPU float32 or float64")
        if any(tensor.dtype != self.source_sum.dtype for tensor in tensors):
            raise ValueError("transport statistics must share one dtype")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("transport statistics must be finite")
        expected_dtype = str(self.source_sum.dtype).removeprefix("torch.")
        if self.accumulation_dtype != expected_dtype:
            raise ValueError("accumulation_dtype does not match statistics")
        if not torch.allclose(
            self.source_gram_sum,
            self.source_gram_sum.T,
        ) or not torch.allclose(
            self.target_gram_sum,
            self.target_gram_sum.T,
        ):
            raise ValueError("modal Gram sums must be symmetric")
        if not _is_sha256(self.source_basis_sha256) or not _is_sha256(
            self.target_basis_sha256
        ):
            raise ValueError("basis digests must be lowercase SHA-256 values")
        if (
            not isinstance(self.relative_eigenvalue_cutoff, float)
            or not math.isfinite(self.relative_eigenvalue_cutoff)
            or not 0 < self.relative_eigenvalue_cutoff < 1
        ):
            raise ValueError("relative_eigenvalue_cutoff must lie in (0, 1)")
        if self.algorithm != _TRANSPORT_ALGORITHM:
            raise ValueError("unsupported modal transport algorithm")
        if self.algorithm_version != _TRANSPORT_ALGORITHM_VERSION:
            raise ValueError("unsupported modal transport algorithm version")

    @property
    def rank(self) -> int:
        return self.source_sum.numel()

    def _moments(self, rank: int) -> tuple[Tensor, Tensor, Tensor]:
        if type(rank) is not int or not 1 <= rank <= self.rank:
            raise ValueError(f"rank must be between 1 and {self.rank}")
        source = self.source_gram_sum[:rank, :rank] / self.observations
        target = self.target_gram_sum[:rank, :rank] / self.observations
        cross = self.cross_sum[:rank, :rank] / self.observations
        if self.centered:
            source_mean = self.source_sum[:rank] / self.observations
            target_mean = self.target_sum[:rank] / self.observations
            source = source - torch.outer(source_mean, source_mean)
            target = target - torch.outer(target_mean, target_mean)
            cross = cross - torch.outer(source_mean, target_mean)
        return (source + source.T) / 2, (target + target.T) / 2, cross

    def point(self, rank: int | None = None) -> ModalTransportPoint:
        """Compute exact prefix CCA/Procrustes metrics from sufficient stats."""

        resolved_rank = self.rank if rank is None else rank
        source, target, cross = self._moments(resolved_rank)
        source_inverse, source_effective = _symmetric_psd_inverse_sqrt(
            source,
            relative_cutoff=self.relative_eigenvalue_cutoff,
        )
        target_inverse, target_effective = _symmetric_psd_inverse_sqrt(
            target,
            relative_cutoff=self.relative_eigenvalue_cutoff,
        )
        whitened_cross = source_inverse @ cross @ target_inverse
        left, correlations, right = torch.linalg.svd(
            whitened_cross,
            full_matrices=True,
        )
        correlation_tolerance = (
            128
            * torch.finfo(correlations.dtype).eps
            * max(resolved_rank, 1)
        )
        if correlations.numel() and correlations.max().item() > (
            1 + correlation_tolerance
        ):
            raise ValueError(
                "paired moments violate the canonical-correlation bound"
            )
        correlations = correlations.clamp(0, 1)
        return ModalTransportPoint(
            rank=resolved_rank,
            canonical_correlations=correlations.cpu(),
            orthogonal_transport=(left @ right).cpu(),
            source_effective_rank=source_effective,
            target_effective_rank=target_effective,
        )

    def metadata(
        self,
        *,
        ranks: Iterable[int] | None = None,
    ) -> dict[str, object]:
        requested = (
            (self.rank,)
            if ranks is None
            else _validated_ranks(ranks, maximum=self.rank)
        )
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "centered": self.centered,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "rank": self.rank,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "accumulation_dtype": self.accumulation_dtype,
            "relative_eigenvalue_cutoff": self.relative_eigenvalue_cutoff,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "points": [self.point(rank).metadata() for rank in requested],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _TRANSPORT_FORMAT_VERSION,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "centered": self.centered,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "source_sum": self.source_sum,
            "target_sum": self.target_sum,
            "source_gram_sum": self.source_gram_sum,
            "target_gram_sum": self.target_gram_sum,
            "cross_sum": self.cross_sum,
            "accumulation_dtype": self.accumulation_dtype,
            "relative_eigenvalue_cutoff": self.relative_eigenvalue_cutoff,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StreamingModalTransportResult:
        expected = {
            "format_version",
            "source_layer",
            "target_layer",
            "row_kind",
            "centered",
            "source_width",
            "target_width",
            "source_basis_sha256",
            "target_basis_sha256",
            "observations",
            "rows_seen",
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
            "accumulation_dtype",
            "relative_eigenvalue_cutoff",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
        }
        if set(state) != expected:
            raise ValueError("streaming modal transport fields are invalid")
        if state["format_version"] != _TRANSPORT_FORMAT_VERSION:
            raise ValueError("unsupported streaming modal transport format")
        tensor_names = {
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
        }
        if any(not isinstance(state[name], Tensor) for name in tensor_names):
            raise TypeError("transport sufficient statistics must be Tensors")
        return cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            centered=state["centered"],  # type: ignore[arg-type]
            source_width=int(state["source_width"]),
            target_width=int(state["target_width"]),
            source_basis_sha256=str(state["source_basis_sha256"]),
            target_basis_sha256=str(state["target_basis_sha256"]),
            observations=int(state["observations"]),
            rows_seen=int(state["rows_seen"]),
            source_sum=state["source_sum"],  # type: ignore[arg-type]
            target_sum=state["target_sum"],  # type: ignore[arg-type]
            source_gram_sum=state["source_gram_sum"],  # type: ignore[arg-type]
            target_gram_sum=state["target_gram_sum"],  # type: ignore[arg-type]
            cross_sum=state["cross_sum"],  # type: ignore[arg-type]
            accumulation_dtype=str(state["accumulation_dtype"]),
            relative_eigenvalue_cutoff=float(
                state["relative_eigenvalue_cutoff"]
            ),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )


class StreamingModalTransportEstimator:
    """Accumulate paired full-width rows into modal transport statistics."""

    def __init__(
        self,
        source: StreamingFisherResult,
        target: StreamingFisherResult,
        *,
        rank: int,
        source_layer: str | None = None,
        target_layer: str | None = None,
        row_kind: str = "score_gradient",
        centered: bool = False,
        accumulation_dtype: torch.dtype = torch.float64,
        relative_eigenvalue_cutoff: float = 1e-10,
    ) -> None:
        if not isinstance(source, StreamingFisherResult) or not isinstance(
            target,
            StreamingFisherResult,
        ):
            raise TypeError("source and target must be StreamingFisherResult")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(source.modes, target.modes):
            raise ValueError("rank exceeds the available Fisher modes")
        if (
            source.scope,
            source.score_reduction,
            source.normalizer,
        ) != (
            target.scope,
            target.score_reduction,
            target.normalizer,
        ):
            raise ValueError("source and target Fisher provenance disagree")
        resolved_source_layer = (
            source.activation_name if source_layer is None else source_layer
        )
        resolved_target_layer = (
            target.activation_name if target_layer is None else target_layer
        )
        for label, value in (
            ("source_layer", resolved_source_layer),
            ("target_layer", resolved_target_layer),
            ("row_kind", row_kind),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if resolved_source_layer == resolved_target_layer:
            raise ValueError("source and target layers must differ")
        if type(centered) is not bool:
            raise TypeError("centered must be a bool")
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be float32 or float64")
        if (
            not isinstance(relative_eigenvalue_cutoff, float)
            or not math.isfinite(relative_eigenvalue_cutoff)
            or not 0 < relative_eigenvalue_cutoff < 1
        ):
            raise ValueError("relative_eigenvalue_cutoff must lie in (0, 1)")

        source_basis = _orthonormalized_prefix(source.vectors, rank).to(
            dtype=accumulation_dtype
        )
        target_basis = _orthonormalized_prefix(target.vectors, rank).to(
            dtype=accumulation_dtype
        )
        self.source = source
        self.target = target
        self.source_layer = resolved_source_layer
        self.target_layer = resolved_target_layer
        self.row_kind = row_kind
        self.centered = centered
        self.accumulation_dtype = accumulation_dtype
        self.relative_eigenvalue_cutoff = relative_eigenvalue_cutoff
        self._source_basis = source_basis.contiguous()
        self._target_basis = target_basis.contiguous()
        self._source_basis_sha256 = _basis_sha256(self._source_basis)
        self._target_basis_sha256 = _basis_sha256(self._target_basis)
        self._source_sum = torch.zeros(rank, dtype=accumulation_dtype)
        self._target_sum = torch.zeros(rank, dtype=accumulation_dtype)
        self._source_gram_sum = torch.zeros(
            (rank, rank), dtype=accumulation_dtype
        )
        self._target_gram_sum = torch.zeros(
            (rank, rank), dtype=accumulation_dtype
        )
        self._cross_sum = torch.zeros(
            (rank, rank), dtype=accumulation_dtype
        )
        self._observations = 0
        self._rows_seen = 0

    @property
    def rank(self) -> int:
        return self._source_basis.shape[1]

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def rows_seen(self) -> int:
        return self._rows_seen

    @property
    def storage_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "source_basis": tuple(self._source_basis.shape),
            "target_basis": tuple(self._target_basis.shape),
            "source_sum": tuple(self._source_sum.shape),
            "target_sum": tuple(self._target_sum.shape),
            "source_gram_sum": tuple(self._source_gram_sum.shape),
            "target_gram_sum": tuple(self._target_gram_sum.shape),
            "cross_sum": tuple(self._cross_sum.shape),
        }

    def update(
        self,
        source_rows: Tensor,
        target_rows: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> StreamingModalTransportEstimator:
        """Add paired ``[observations, width]`` transient rows."""

        if not isinstance(source_rows, Tensor) or not isinstance(
            target_rows,
            Tensor,
        ):
            raise TypeError("source_rows and target_rows must be Tensors")
        if source_rows.ndim != 2 or target_rows.ndim != 2:
            raise ValueError("paired rows must have shape [observations, width]")
        if source_rows.shape[0] != target_rows.shape[0]:
            raise ValueError("paired rows must have the same observation count")
        if source_rows.shape[1] != self.source.width:
            raise ValueError("source row width does not match the Fisher basis")
        if target_rows.shape[1] != self.target.width:
            raise ValueError("target row width does not match the Fisher basis")
        if not source_rows.is_floating_point() or not (
            target_rows.is_floating_point()
        ):
            raise ValueError("paired rows must use real floating dtypes")
        rows = source_rows.shape[0]
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise TypeError("mask must be a Tensor")
            if mask.shape != (rows,):
                raise ValueError("mask must have shape [observations]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must use boolean dtype")
            source_selected = source_rows[
                mask.to(device=source_rows.device)
            ]
            target_selected = target_rows[
                mask.to(device=target_rows.device)
            ]
        else:
            source_selected = source_rows
            target_selected = target_rows
        source_selected = source_selected.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        target_selected = target_selected.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        if not torch.isfinite(source_selected).all() or not torch.isfinite(
            target_selected
        ).all():
            raise ValueError("selected paired rows must be finite")

        self._rows_seen += rows
        if source_selected.shape[0] == 0:
            return self
        source_modal = source_selected @ self._source_basis
        target_modal = target_selected @ self._target_basis
        self._source_sum += source_modal.sum(dim=0)
        self._target_sum += target_modal.sum(dim=0)
        self._source_gram_sum += source_modal.T @ source_modal
        self._target_gram_sum += target_modal.T @ target_modal
        self._cross_sum += source_modal.T @ target_modal
        self._observations += source_modal.shape[0]
        return self

    def finalize(self) -> StreamingModalTransportResult:
        if self._observations == 0:
            raise ValueError("cannot finalize without selected observations")
        return StreamingModalTransportResult(
            source_layer=self.source_layer,
            target_layer=self.target_layer,
            row_kind=self.row_kind,
            centered=self.centered,
            source_width=self.source.width,
            target_width=self.target.width,
            source_basis_sha256=self._source_basis_sha256,
            target_basis_sha256=self._target_basis_sha256,
            observations=self._observations,
            rows_seen=self._rows_seen,
            source_sum=self._source_sum.clone(),
            target_sum=self._target_sum.clone(),
            source_gram_sum=self._source_gram_sum.clone(),
            target_gram_sum=self._target_gram_sum.clone(),
            cross_sum=self._cross_sum.clone(),
            accumulation_dtype=str(self.accumulation_dtype).removeprefix(
                "torch."
            ),
            relative_eigenvalue_cutoff=self.relative_eigenvalue_cutoff,
            scope=self.source.scope,
            score_reduction=self.source.score_reduction,
            normalizer=self.source.normalizer,
        )


@dataclass(frozen=True, slots=True)
class FrozenModalTransport:
    """Calibration-fit map from one frozen modal coordinate system to another."""

    source_layer: str
    target_layer: str
    row_kind: str
    centered: bool
    source_width: int
    target_width: int
    source_basis_sha256: str
    target_basis_sha256: str
    basis_rank: int
    rank: int
    source_mean: Tensor
    target_mean: Tensor
    matrix: Tensor
    calibration_observations: int
    relative_eigenvalue_cutoff: float
    accumulation_dtype: str
    scope: str
    score_reduction: str
    normalizer: str
    algorithm: str = _FROZEN_TRANSPORT_ALGORITHM
    algorithm_version: int = _FROZEN_TRANSPORT_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("source_layer", self.source_layer),
            ("target_layer", self.target_layer),
            ("row_kind", self.row_kind),
            ("accumulation_dtype", self.accumulation_dtype),
            ("scope", self.scope),
            ("score_reduction", self.score_reduction),
            ("normalizer", self.normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if self.source_layer == self.target_layer:
            raise ValueError("transport must connect distinct layers")
        if type(self.centered) is not bool:
            raise TypeError("centered must be a bool")
        for label, value in (
            ("source_width", self.source_width),
            ("target_width", self.target_width),
            ("basis_rank", self.basis_rank),
            ("rank", self.rank),
            ("calibration_observations", self.calibration_observations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.rank > self.basis_rank:
            raise ValueError("transport rank cannot exceed its bound basis rank")
        if self.basis_rank > min(self.source_width, self.target_width):
            raise ValueError("transport rank cannot exceed a boundary width")
        if self.source_mean.shape != (self.rank,):
            raise ValueError("source_mean must have shape [rank]")
        if self.target_mean.shape != (self.rank,):
            raise ValueError("target_mean must have shape [rank]")
        if self.matrix.shape != (self.rank, self.rank):
            raise ValueError("transport matrix must have shape [rank, rank]")
        tensors = (self.source_mean, self.target_mean, self.matrix)
        if any(
            tensor.device.type != "cpu"
            or tensor.dtype not in (torch.float32, torch.float64)
            for tensor in tensors
        ):
            raise ValueError("frozen transport tensors must be CPU float tensors")
        if any(tensor.dtype != self.matrix.dtype for tensor in tensors):
            raise ValueError("frozen transport tensors must share one dtype")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("frozen transport tensors must be finite")
        if self.accumulation_dtype != str(self.matrix.dtype).removeprefix(
            "torch."
        ):
            raise ValueError("accumulation_dtype does not match transport tensors")
        if not _is_sha256(self.source_basis_sha256) or not _is_sha256(
            self.target_basis_sha256
        ):
            raise ValueError("basis digests must be lowercase SHA-256 values")
        if (
            not isinstance(self.relative_eigenvalue_cutoff, float)
            or not math.isfinite(self.relative_eigenvalue_cutoff)
            or not 0 < self.relative_eigenvalue_cutoff < 1
        ):
            raise ValueError("relative_eigenvalue_cutoff must lie in (0, 1)")
        if self.algorithm != _FROZEN_TRANSPORT_ALGORITHM:
            raise ValueError("unsupported frozen modal transport algorithm")
        if self.algorithm_version != _FROZEN_TRANSPORT_ALGORITHM_VERSION:
            raise ValueError(
                "unsupported frozen modal transport algorithm version"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _FROZEN_TRANSPORT_FORMAT_VERSION,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "centered": self.centered,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "basis_rank": self.basis_rank,
            "rank": self.rank,
            "source_mean": self.source_mean,
            "target_mean": self.target_mean,
            "matrix": self.matrix,
            "calibration_observations": self.calibration_observations,
            "relative_eigenvalue_cutoff": self.relative_eigenvalue_cutoff,
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FrozenModalTransport:
        expected = {
            "format_version",
            "source_layer",
            "target_layer",
            "row_kind",
            "centered",
            "source_width",
            "target_width",
            "source_basis_sha256",
            "target_basis_sha256",
            "basis_rank",
            "rank",
            "source_mean",
            "target_mean",
            "matrix",
            "calibration_observations",
            "relative_eigenvalue_cutoff",
            "accumulation_dtype",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
        }
        if set(state) != expected:
            raise ValueError("frozen modal transport fields are invalid")
        if state["format_version"] != _FROZEN_TRANSPORT_FORMAT_VERSION:
            raise ValueError("unsupported frozen modal transport format")
        for name in ("source_mean", "target_mean", "matrix"):
            if not isinstance(state[name], Tensor):
                raise TypeError("frozen modal transport values must be Tensors")
        return cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            centered=state["centered"],  # type: ignore[arg-type]
            source_width=int(state["source_width"]),
            target_width=int(state["target_width"]),
            source_basis_sha256=str(state["source_basis_sha256"]),
            target_basis_sha256=str(state["target_basis_sha256"]),
            basis_rank=int(state["basis_rank"]),
            rank=int(state["rank"]),
            source_mean=state["source_mean"],  # type: ignore[arg-type]
            target_mean=state["target_mean"],  # type: ignore[arg-type]
            matrix=state["matrix"],  # type: ignore[arg-type]
            calibration_observations=int(state["calibration_observations"]),
            relative_eigenvalue_cutoff=float(
                state["relative_eigenvalue_cutoff"]
            ),
            accumulation_dtype=str(state["accumulation_dtype"]),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )


def freeze_modal_transport(
    result: StreamingModalTransportResult,
    *,
    rank: int | None = None,
) -> FrozenModalTransport:
    """Fit a whitened orthogonal Procrustes map from calibration moments."""

    if not isinstance(result, StreamingModalTransportResult):
        raise TypeError("result must be a StreamingModalTransportResult")
    resolved_rank = result.rank if rank is None else rank
    if type(resolved_rank) is not int or not 1 <= resolved_rank <= result.rank:
        raise ValueError(f"rank must be between 1 and {result.rank}")
    source_covariance, target_covariance, _ = result._moments(resolved_rank)
    source_inverse, _ = _symmetric_psd_inverse_sqrt(
        source_covariance,
        relative_cutoff=result.relative_eigenvalue_cutoff,
    )
    target_root = _symmetric_psd_sqrt(
        target_covariance,
        relative_cutoff=result.relative_eigenvalue_cutoff,
    )
    orthogonal = result.point(resolved_rank).orthogonal_transport
    dtype = result.source_sum.dtype
    zeros = torch.zeros(resolved_rank, dtype=dtype)
    source_mean = (
        result.source_sum[:resolved_rank] / result.observations
        if result.centered
        else zeros
    )
    target_mean = (
        result.target_sum[:resolved_rank] / result.observations
        if result.centered
        else zeros
    )
    return FrozenModalTransport(
        source_layer=result.source_layer,
        target_layer=result.target_layer,
        row_kind=result.row_kind,
        centered=result.centered,
        source_width=result.source_width,
        target_width=result.target_width,
        source_basis_sha256=result.source_basis_sha256,
        target_basis_sha256=result.target_basis_sha256,
        basis_rank=result.rank,
        rank=resolved_rank,
        source_mean=source_mean.clone(),
        target_mean=target_mean.clone(),
        matrix=(source_inverse @ orthogonal @ target_root).cpu(),
        calibration_observations=result.observations,
        relative_eigenvalue_cutoff=result.relative_eigenvalue_cutoff,
        accumulation_dtype=result.accumulation_dtype,
        scope=result.scope,
        score_reduction=result.score_reduction,
        normalizer=result.normalizer,
    )


@dataclass(frozen=True, slots=True)
class FrozenModalTransportEvaluation:
    """Exact held-out predictive accounting for one frozen modal map."""

    source_layer: str
    target_layer: str
    row_kind: str
    centered: bool
    rank: int
    observations: int
    rows_seen: int
    source_sum: Tensor
    target_sum: Tensor
    source_gram_sum: Tensor
    target_gram_sum: Tensor
    cross_sum: Tensor
    target_baseline_squared_error: float
    identity_squared_error: float
    transport_squared_error: float
    transport_target_dot_sum: float
    transport_squared_norm_sum: float
    target_squared_norm_sum: float
    source_basis_sha256: str
    target_basis_sha256: str
    accumulation_dtype: str
    algorithm: str = _FROZEN_TRANSPORT_ALGORITHM
    algorithm_version: int = _FROZEN_TRANSPORT_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("source_layer", self.source_layer),
            ("target_layer", self.target_layer),
            ("row_kind", self.row_kind),
            ("accumulation_dtype", self.accumulation_dtype),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if type(self.centered) is not bool:
            raise TypeError("centered must be a bool")
        for label, value in (
            ("rank", self.rank),
            ("observations", self.observations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if type(self.rows_seen) is not int or self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        for label, tensor, shape in (
            ("source_sum", self.source_sum, (self.rank,)),
            ("target_sum", self.target_sum, (self.rank,)),
            ("source_gram_sum", self.source_gram_sum, (self.rank, self.rank)),
            ("target_gram_sum", self.target_gram_sum, (self.rank, self.rank)),
            ("cross_sum", self.cross_sum, (self.rank, self.rank)),
        ):
            if not isinstance(tensor, Tensor) or tensor.shape != shape:
                raise ValueError(f"{label} must have shape {list(shape)}")
            if tensor.device.type != "cpu" or tensor.dtype not in (
                torch.float32,
                torch.float64,
            ):
                raise ValueError(
                    "evaluation moments must be CPU float32 or float64 tensors"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError("evaluation moments must be finite")
        moment_tensors = (
            self.source_sum,
            self.target_sum,
            self.source_gram_sum,
            self.target_gram_sum,
            self.cross_sum,
        )
        if any(
            tensor.dtype != self.source_sum.dtype
            for tensor in moment_tensors
        ):
            raise ValueError("evaluation moments must share one dtype")
        expected_dtype = str(self.source_sum.dtype).removeprefix("torch.")
        if self.accumulation_dtype != expected_dtype:
            raise ValueError(
                "accumulation_dtype does not match evaluation moments"
            )
        if not torch.allclose(
            self.source_gram_sum,
            self.source_gram_sum.T,
        ) or not torch.allclose(
            self.target_gram_sum,
            self.target_gram_sum.T,
        ):
            raise ValueError("evaluation Gram sums must be symmetric")
        for label, value in (
            ("target_baseline_squared_error", self.target_baseline_squared_error),
            ("identity_squared_error", self.identity_squared_error),
            ("transport_squared_error", self.transport_squared_error),
            ("transport_squared_norm_sum", self.transport_squared_norm_sum),
            ("target_squared_norm_sum", self.target_squared_norm_sum),
        ):
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
        if not isinstance(self.transport_target_dot_sum, float) or not (
            math.isfinite(self.transport_target_dot_sum)
        ):
            raise ValueError("transport_target_dot_sum must be finite")
        if not _is_sha256(self.source_basis_sha256) or not _is_sha256(
            self.target_basis_sha256
        ):
            raise ValueError("basis digests must be lowercase SHA-256 values")
        if self.algorithm != _FROZEN_TRANSPORT_ALGORITHM:
            raise ValueError("unsupported frozen modal transport algorithm")
        if self.algorithm_version != _FROZEN_TRANSPORT_ALGORITHM_VERSION:
            raise ValueError(
                "unsupported frozen modal transport algorithm version"
            )

    def _relative_score(self, squared_error: float) -> float | None:
        if self.target_baseline_squared_error == 0:
            return 1.0 if squared_error == 0 else None
        return 1.0 - squared_error / self.target_baseline_squared_error

    @property
    def baseline_kind(self) -> str:
        return "calibration_mean" if self.centered else "zero"

    @property
    def identity_explained_fraction(self) -> float | None:
        """Gauge-dependent score for copying the same modal coordinates."""

        return self._relative_score(self.identity_squared_error)

    @property
    def transport_explained_fraction(self) -> float | None:
        return self._relative_score(self.transport_squared_error)

    @property
    def identity_r_squared(self) -> float | None:
        """Out-of-sample R-squared when the fit used a centered baseline."""

        return self.identity_explained_fraction if self.centered else None

    @property
    def transport_r_squared(self) -> float | None:
        """Out-of-sample R-squared when the fit used a centered baseline."""

        return self.transport_explained_fraction if self.centered else None

    @property
    def identity_normalized_rmse(self) -> float | None:
        if self.target_baseline_squared_error == 0:
            return 0.0 if self.identity_squared_error == 0 else None
        return math.sqrt(
            self.identity_squared_error / self.target_baseline_squared_error
        )

    @property
    def transport_normalized_rmse(self) -> float | None:
        if self.target_baseline_squared_error == 0:
            return 0.0 if self.transport_squared_error == 0 else None
        return math.sqrt(
            self.transport_squared_error / self.target_baseline_squared_error
        )

    @property
    def transport_target_cosine(self) -> float | None:
        denominator = math.sqrt(
            self.transport_squared_norm_sum * self.target_squared_norm_sum
        )
        if denominator == 0:
            return None
        return max(
            -1.0,
            min(1.0, self.transport_target_dot_sum / denominator),
        )

    @property
    def rotation_gain(self) -> float | None:
        """Gain over the gauge-dependent identity-coordinate baseline."""

        identity = self.identity_explained_fraction
        transport = self.transport_explained_fraction
        if identity is None or transport is None:
            return None
        return transport - identity

    def metadata(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "centered": self.centered,
            "baseline_kind": self.baseline_kind,
            "rank": self.rank,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "identity_explained_fraction": (
                self.identity_explained_fraction
            ),
            "transport_explained_fraction": (
                self.transport_explained_fraction
            ),
            "identity_r_squared": self.identity_r_squared,
            "transport_r_squared": self.transport_r_squared,
            "rotation_gain": self.rotation_gain,
            "identity_coordinate_baseline_gauge_dependent": True,
            "rotation_gain_gauge_dependent": True,
            "identity_normalized_rmse": self.identity_normalized_rmse,
            "transport_normalized_rmse": self.transport_normalized_rmse,
            "transport_target_cosine": self.transport_target_cosine,
            "target_baseline_squared_error": (
                self.target_baseline_squared_error
            ),
            "identity_squared_error": self.identity_squared_error,
            "transport_squared_error": self.transport_squared_error,
            "accumulation_dtype": self.accumulation_dtype,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _FROZEN_EVALUATION_FORMAT_VERSION,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "centered": self.centered,
            "rank": self.rank,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "source_sum": self.source_sum,
            "target_sum": self.target_sum,
            "source_gram_sum": self.source_gram_sum,
            "target_gram_sum": self.target_gram_sum,
            "cross_sum": self.cross_sum,
            "target_baseline_squared_error": (
                self.target_baseline_squared_error
            ),
            "identity_squared_error": self.identity_squared_error,
            "transport_squared_error": self.transport_squared_error,
            "transport_target_dot_sum": self.transport_target_dot_sum,
            "transport_squared_norm_sum": self.transport_squared_norm_sum,
            "target_squared_norm_sum": self.target_squared_norm_sum,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "accumulation_dtype": self.accumulation_dtype,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FrozenModalTransportEvaluation:
        expected = {
            "format_version",
            "source_layer",
            "target_layer",
            "row_kind",
            "centered",
            "rank",
            "observations",
            "rows_seen",
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
            "target_baseline_squared_error",
            "identity_squared_error",
            "transport_squared_error",
            "transport_target_dot_sum",
            "transport_squared_norm_sum",
            "target_squared_norm_sum",
            "source_basis_sha256",
            "target_basis_sha256",
            "accumulation_dtype",
            "algorithm",
            "algorithm_version",
        }
        if set(state) != expected:
            raise ValueError("frozen modal evaluation fields are invalid")
        if state["format_version"] != _FROZEN_EVALUATION_FORMAT_VERSION:
            raise ValueError("unsupported frozen modal evaluation format")
        for name in (
            "source_sum",
            "target_sum",
            "source_gram_sum",
            "target_gram_sum",
            "cross_sum",
        ):
            if not isinstance(state[name], Tensor):
                raise TypeError(
                    "frozen modal evaluation moments must be Tensors"
                )
        return cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            centered=state["centered"],  # type: ignore[arg-type]
            rank=int(state["rank"]),
            observations=int(state["observations"]),
            rows_seen=int(state["rows_seen"]),
            source_sum=state["source_sum"],  # type: ignore[arg-type]
            target_sum=state["target_sum"],  # type: ignore[arg-type]
            source_gram_sum=state["source_gram_sum"],  # type: ignore[arg-type]
            target_gram_sum=state["target_gram_sum"],  # type: ignore[arg-type]
            cross_sum=state["cross_sum"],  # type: ignore[arg-type]
            target_baseline_squared_error=float(
                state["target_baseline_squared_error"]
            ),
            identity_squared_error=float(state["identity_squared_error"]),
            transport_squared_error=float(state["transport_squared_error"]),
            transport_target_dot_sum=float(
                state["transport_target_dot_sum"]
            ),
            transport_squared_norm_sum=float(
                state["transport_squared_norm_sum"]
            ),
            target_squared_norm_sum=float(
                state["target_squared_norm_sum"]
            ),
            source_basis_sha256=str(state["source_basis_sha256"]),
            target_basis_sha256=str(state["target_basis_sha256"]),
            accumulation_dtype=str(state["accumulation_dtype"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )


def _clamp_nonnegative_moment_scalar(
    value: Tensor,
    *,
    label: str,
    scale: float,
    rank: int,
) -> float:
    resolved = value.item()
    tolerance = (
        2048
        * torch.finfo(value.dtype).eps
        * max(scale, 1.0)
        * max(rank, 1)
    )
    if resolved < -tolerance:
        raise ValueError(f"{label} implied by held-out moments is negative")
    return max(resolved, 0.0)


def evaluate_frozen_modal_transport_from_moments(
    transport: FrozenModalTransport,
    *,
    observations: int,
    rows_seen: int,
    source_sum: Tensor,
    target_sum: Tensor,
    source_gram_sum: Tensor,
    target_gram_sum: Tensor,
    cross_sum: Tensor,
) -> FrozenModalTransportEvaluation:
    """Recompute exact held-out errors from bounded sufficient statistics."""

    if not isinstance(transport, FrozenModalTransport):
        raise TypeError("transport must be a FrozenModalTransport")
    if type(observations) is not int or observations <= 0:
        raise ValueError("observations must be positive")
    if type(rows_seen) is not int or rows_seen < observations:
        raise ValueError("rows_seen cannot be smaller than observations")
    rank = transport.rank
    # Evaluation is always accumulated and reconstructed in float64.  A
    # rank-scaled float32 cancellation tolerance is large enough to hide
    # materially impossible negative SSEs at useful ranks.
    dtype = torch.float64
    tensors: dict[str, Tensor] = {}
    for label, value, shape in (
        ("source_sum", source_sum, (rank,)),
        ("target_sum", target_sum, (rank,)),
        ("source_gram_sum", source_gram_sum, (rank, rank)),
        ("target_gram_sum", target_gram_sum, (rank, rank)),
        ("cross_sum", cross_sum, (rank, rank)),
    ):
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"{label} must have shape {list(shape)}")
        converted = value.detach().to(device="cpu", dtype=dtype)
        if not torch.isfinite(converted).all():
            raise ValueError("held-out transport moments must be finite")
        tensors[label] = converted
    source_sum = tensors["source_sum"]
    target_sum = tensors["target_sum"]
    source_gram_sum = tensors["source_gram_sum"]
    target_gram_sum = tensors["target_gram_sum"]
    cross_sum = tensors["cross_sum"]
    if not torch.allclose(source_gram_sum, source_gram_sum.T) or not (
        torch.allclose(target_gram_sum, target_gram_sum.T)
    ):
        raise ValueError("held-out transport Gram sums must be symmetric")

    source_mean = transport.source_mean.to(dtype=dtype)
    target_mean = transport.target_mean.to(dtype=dtype)
    matrix = transport.matrix.to(dtype=dtype)
    count = float(observations)
    source_centered_sum = source_sum - count * source_mean
    source_centered_gram = (
        source_gram_sum
        - torch.outer(source_mean, source_sum)
        - torch.outer(source_sum, source_mean)
        + count * torch.outer(source_mean, source_mean)
    )
    target_centered_gram = (
        target_gram_sum
        - torch.outer(target_mean, target_sum)
        - torch.outer(target_sum, target_mean)
        + count * torch.outer(target_mean, target_mean)
    )
    centered_cross = (
        cross_sum
        - torch.outer(source_mean, target_sum)
        - torch.outer(source_sum, target_mean)
        + count * torch.outer(source_mean, target_mean)
    )
    source_centered_gram = (
        source_centered_gram + source_centered_gram.T
    ) / 2
    target_centered_gram = (
        target_centered_gram + target_centered_gram.T
    ) / 2
    target_centered_norm = torch.trace(target_centered_gram)
    source_centered_norm = torch.trace(source_centered_gram)
    identity_cross = torch.trace(centered_cross)
    transported_centered_norm = torch.trace(
        matrix.T @ source_centered_gram @ matrix
    )
    transported_centered_target_dot = torch.trace(
        matrix.T @ centered_cross
    )
    raw_target_norm = torch.trace(target_gram_sum)
    raw_prediction_norm = (
        transported_centered_norm
        + 2 * torch.dot(source_centered_sum @ matrix, target_mean)
        + count * torch.dot(target_mean, target_mean)
    )
    raw_prediction_target_dot = (
        torch.trace(
            matrix.T
            @ (
                cross_sum
                - torch.outer(source_mean, target_sum)
            )
        )
        + torch.dot(target_mean, target_sum)
    )
    scale = max(
        abs(raw_target_norm.item()),
        abs(raw_prediction_norm.item()),
        abs(target_centered_norm.item()),
        abs(source_centered_norm.item()),
        1.0,
    )
    baseline_error = _clamp_nonnegative_moment_scalar(
        target_centered_norm,
        label="target baseline squared error",
        scale=scale,
        rank=rank,
    )
    identity_error = _clamp_nonnegative_moment_scalar(
        target_centered_norm
        + source_centered_norm
        - 2 * identity_cross,
        label="identity squared error",
        scale=scale,
        rank=rank,
    )
    transport_error = _clamp_nonnegative_moment_scalar(
        target_centered_norm
        + transported_centered_norm
        - 2 * transported_centered_target_dot,
        label="transport squared error",
        scale=scale,
        rank=rank,
    )
    prediction_norm = _clamp_nonnegative_moment_scalar(
        raw_prediction_norm,
        label="transport prediction squared norm",
        scale=scale,
        rank=rank,
    )
    target_norm = _clamp_nonnegative_moment_scalar(
        raw_target_norm,
        label="target squared norm",
        scale=scale,
        rank=rank,
    )
    prediction_target_dot = raw_prediction_target_dot.item()
    cauchy_tolerance = (
        4096
        * torch.finfo(dtype).eps
        * max(prediction_norm * target_norm, 1.0)
        * max(rank, 1)
    )
    if prediction_target_dot**2 > (
        prediction_norm * target_norm + cauchy_tolerance
    ):
        raise ValueError(
            "held-out transport moments violate the prediction-target "
            "Cauchy bound"
        )
    return FrozenModalTransportEvaluation(
        source_layer=transport.source_layer,
        target_layer=transport.target_layer,
        row_kind=transport.row_kind,
        centered=transport.centered,
        rank=rank,
        observations=observations,
        rows_seen=rows_seen,
        source_sum=source_sum.clone(),
        target_sum=target_sum.clone(),
        source_gram_sum=source_gram_sum.clone(),
        target_gram_sum=target_gram_sum.clone(),
        cross_sum=cross_sum.clone(),
        target_baseline_squared_error=baseline_error,
        identity_squared_error=identity_error,
        transport_squared_error=transport_error,
        transport_target_dot_sum=prediction_target_dot,
        transport_squared_norm_sum=prediction_norm,
        target_squared_norm_sum=target_norm,
        source_basis_sha256=transport.source_basis_sha256,
        target_basis_sha256=transport.target_basis_sha256,
        accumulation_dtype="float64",
    )


class StreamingFrozenModalTransportEvaluator:
    """Evaluate one calibration-frozen transport without retaining held-out rows."""

    def __init__(
        self,
        transport: FrozenModalTransport,
        source: StreamingFisherResult,
        target: StreamingFisherResult,
    ) -> None:
        if not isinstance(transport, FrozenModalTransport):
            raise TypeError("transport must be a FrozenModalTransport")
        if not isinstance(source, StreamingFisherResult) or not isinstance(
            target,
            StreamingFisherResult,
        ):
            raise TypeError("source and target must be StreamingFisherResult")
        if source.width != transport.source_width:
            raise ValueError("source basis width does not match frozen transport")
        if target.width != transport.target_width:
            raise ValueError("target basis width does not match frozen transport")
        if transport.basis_rank > min(source.modes, target.modes):
            raise ValueError("frozen rank exceeds the supplied Fisher modes")
        transport_dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }.get(transport.accumulation_dtype)
        if transport_dtype is None:
            raise ValueError("unsupported frozen transport accumulation dtype")
        accumulation_dtype = torch.float64
        source_binding_basis = _orthonormalized_prefix(
            source.vectors,
            transport.basis_rank,
        ).to(dtype=accumulation_dtype)
        target_binding_basis = _orthonormalized_prefix(
            target.vectors,
            transport.basis_rank,
        ).to(dtype=accumulation_dtype)
        if (
            _basis_sha256(source_binding_basis)
            != transport.source_basis_sha256
        ):
            raise ValueError("source basis does not match frozen transport")
        if (
            _basis_sha256(target_binding_basis)
            != transport.target_basis_sha256
        ):
            raise ValueError("target basis does not match frozen transport")
        source_basis = source_binding_basis[:, : transport.rank].contiguous()
        target_basis = target_binding_basis[:, : transport.rank].contiguous()
        self.transport = transport
        self.source = source
        self.target = target
        self._source_basis = source_basis
        self._target_basis = target_basis
        self._observations = 0
        self._rows_seen = 0
        rank = transport.rank
        self._source_sum = torch.zeros(
            rank,
            dtype=accumulation_dtype,
        )
        self._target_sum = torch.zeros(
            rank,
            dtype=accumulation_dtype,
        )
        self._source_gram_sum = torch.zeros(
            (rank, rank),
            dtype=accumulation_dtype,
        )
        self._target_gram_sum = torch.zeros(
            (rank, rank),
            dtype=accumulation_dtype,
        )
        self._cross_sum = torch.zeros(
            (rank, rank),
            dtype=accumulation_dtype,
        )

    def update(
        self,
        source_rows: Tensor,
        target_rows: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> StreamingFrozenModalTransportEvaluator:
        if not isinstance(source_rows, Tensor) or not isinstance(
            target_rows,
            Tensor,
        ):
            raise TypeError("source_rows and target_rows must be Tensors")
        if source_rows.ndim != 2 or target_rows.ndim != 2:
            raise ValueError("paired rows must have shape [observations, width]")
        if source_rows.shape[0] != target_rows.shape[0]:
            raise ValueError("paired rows must have the same observation count")
        if source_rows.shape[1] != self.transport.source_width:
            raise ValueError("source row width does not match frozen transport")
        if target_rows.shape[1] != self.transport.target_width:
            raise ValueError("target row width does not match frozen transport")
        if not source_rows.is_floating_point() or not (
            target_rows.is_floating_point()
        ):
            raise ValueError("paired rows must use real floating dtypes")
        rows = source_rows.shape[0]
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise TypeError("mask must be a Tensor")
            if mask.shape != (rows,):
                raise ValueError("mask must have shape [observations]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must use boolean dtype")
            source_rows = source_rows[mask.to(device=source_rows.device)]
            target_rows = target_rows[mask.to(device=target_rows.device)]
        source_rows = source_rows.detach().to(
            device="cpu",
            dtype=self._source_basis.dtype,
        )
        target_rows = target_rows.detach().to(
            device="cpu",
            dtype=self._target_basis.dtype,
        )
        if not torch.isfinite(source_rows).all() or not torch.isfinite(
            target_rows
        ).all():
            raise ValueError("selected paired rows must be finite")
        self._rows_seen += rows
        if source_rows.shape[0] == 0:
            return self
        source_modal = source_rows @ self._source_basis
        target_modal = target_rows @ self._target_basis
        self._source_sum += source_modal.sum(dim=0)
        self._target_sum += target_modal.sum(dim=0)
        self._source_gram_sum += source_modal.T @ source_modal
        self._target_gram_sum += target_modal.T @ target_modal
        self._cross_sum += source_modal.T @ target_modal
        self._observations += source_rows.shape[0]
        return self

    def finalize(self) -> FrozenModalTransportEvaluation:
        if self._observations == 0:
            raise ValueError("cannot finalize without selected observations")
        return evaluate_frozen_modal_transport_from_moments(
            self.transport,
            observations=self._observations,
            rows_seen=self._rows_seen,
            source_sum=self._source_sum,
            target_sum=self._target_sum,
            source_gram_sum=self._source_gram_sum,
            target_gram_sum=self._target_gram_sum,
            cross_sum=self._cross_sum,
        )


@dataclass(frozen=True, slots=True)
class ModalTransitionEvidence:
    """Combined evidence distinguishing instability from transported rotation."""

    source_layer: str
    target_layer: str
    rank: int
    source_boundary_overlap: float | None
    target_boundary_overlap: float | None
    direct_depth_overlap: float
    paired_transport_coherence: float | None

    def __post_init__(self) -> None:
        for label, value in (
            ("source_layer", self.source_layer),
            ("target_layer", self.target_layer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if self.source_layer == self.target_layer:
            raise ValueError("transition layers must differ")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be positive")
        for label, value in (
            ("source_boundary_overlap", self.source_boundary_overlap),
            ("target_boundary_overlap", self.target_boundary_overlap),
            ("paired_transport_coherence", self.paired_transport_coherence),
        ):
            if value is not None and (
                not isinstance(value, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{label} must lie in [0, 1] when present")
        if (
            not isinstance(self.direct_depth_overlap, float)
            or not math.isfinite(self.direct_depth_overlap)
            or not 0 <= self.direct_depth_overlap <= 1
        ):
            raise ValueError("direct_depth_overlap must lie in [0, 1]")

    @property
    def boundary_stability_floor(self) -> float | None:
        if (
            self.source_boundary_overlap is None
            or self.target_boundary_overlap is None
        ):
            return None
        return min(
            self.source_boundary_overlap,
            self.target_boundary_overlap,
        )

    def classify(
        self,
        *,
        stable_boundary_overlap: float = 0.9,
        aligned_depth_overlap: float = 0.8,
        coherent_transport: float = 0.8,
    ) -> str:
        """Classify with explicit, caller-visible diagnostic thresholds."""

        for label, value in (
            ("stable_boundary_overlap", stable_boundary_overlap),
            ("aligned_depth_overlap", aligned_depth_overlap),
            ("coherent_transport", coherent_transport),
        ):
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{label} must lie in [0, 1]")
        floor = self.boundary_stability_floor
        if floor is None:
            return "boundary_stability_unmeasured"
        if floor < stable_boundary_overlap:
            return "static_boundary_instability"
        if self.direct_depth_overlap >= aligned_depth_overlap:
            return "stable_depth_aligned"
        if self.paired_transport_coherence is None:
            return "stable_depth_rotation_unmeasured"
        if self.paired_transport_coherence >= coherent_transport:
            return "stable_transported_rotation"
        return "stable_depth_drift_without_coherent_transport"

    def metadata(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "rank": self.rank,
            "source_boundary_overlap": self.source_boundary_overlap,
            "target_boundary_overlap": self.target_boundary_overlap,
            "boundary_stability_floor": self.boundary_stability_floor,
            "direct_depth_overlap": self.direct_depth_overlap,
            "paired_transport_coherence": self.paired_transport_coherence,
        }


def assess_modal_transition(
    geometry: ModalTrajectoryGeometry,
    *,
    source_layer: str,
    target_layer: str,
    rank: int,
    transport: StreamingModalTransportResult | None = None,
) -> ModalTransitionEvidence:
    """Combine split, depth, and optional paired-row evidence."""

    if not isinstance(geometry, ModalTrajectoryGeometry):
        raise TypeError("geometry must be a ModalTrajectoryGeometry")
    depth = geometry.transition(source_layer, target_layer).at_rank(rank)
    source_overlap = None
    target_overlap = None
    if geometry.boundary_stability:
        source_overlap = (
            geometry.boundary(source_layer)
            .at_rank(rank)
            .mean_squared_overlap
        )
        target_overlap = (
            geometry.boundary(target_layer)
            .at_rank(rank)
            .mean_squared_overlap
        )
    transport_score = None
    if transport is not None:
        if not isinstance(transport, StreamingModalTransportResult):
            raise TypeError("transport must be a StreamingModalTransportResult")
        if (
            transport.source_layer != source_layer
            or transport.target_layer != target_layer
        ):
            raise ValueError("transport does not match the geometry transition")
        transport_score = (
            transport.point(rank).mean_squared_canonical_correlation
        )
    return ModalTransitionEvidence(
        source_layer=source_layer,
        target_layer=target_layer,
        rank=rank,
        source_boundary_overlap=source_overlap,
        target_boundary_overlap=target_overlap,
        direct_depth_overlap=depth.mean_squared_overlap,
        paired_transport_coherence=transport_score,
    )


__all__ = [
    "FrozenModalTransport",
    "FrozenModalTransportEvaluation",
    "ModalSubspaceAlignment",
    "ModalTrajectoryGeometry",
    "ModalTransitionEvidence",
    "ModalTransportPoint",
    "RankedSubspaceAlignment",
    "StreamingFrozenModalTransportEvaluator",
    "StreamingModalTransportEstimator",
    "StreamingModalTransportResult",
    "analyze_modal_subspace_trajectory",
    "assess_modal_transition",
    "evaluate_frozen_modal_transport_from_moments",
    "freeze_modal_transport",
]
