"""Bounded-memory validation for streamed activation-Fisher modes.

Two separate questions matter after extracting a low-rank Fisher basis:

* Does another calibration split recover the same *subspace*?
* How much exact Fisher energy does the frozen basis capture on held-out rows?

The first question is answered with principal angles.  For two rank-``k``
orthonormal bases ``U`` and ``V``, the singular values of ``U.T @ V`` are the
principal cosines.  Their mean square is a normalized overlap in ``[0, 1]``
that is invariant to mode signs and rotations within the subspaces.

The second question is answered without constructing a width-by-width Fisher
matrix and without retaining score-gradient rows.  For held-out rows ``G`` and
a frozen orthonormal basis ``V``, this module accumulates

``trace(V.T @ F_validation @ V) = ||G @ V||_F^2 / N``

alongside the exact held-out trace ``||G||_F^2 / N``.  Storage is linear in
``width * modes`` and does not grow with the number of observations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .streaming_fisher import StreamingFisherResult


_RAYLEIGH_ALGORITHM = "exact_streaming_frozen_basis_rayleigh"
_RAYLEIGH_ALGORITHM_VERSION = 1


def _basis_sha256(vectors: Tensor) -> str:
    """Hash the exact ordered basis used by a Rayleigh estimator."""

    canonical = vectors.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.rayleigh_basis.v1\0")
    digest.update(
        f"{canonical.shape[0]}x{canonical.shape[1]}\0float64\0".encode()
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_orthonormal_columns(
    vectors: Tensor,
    *,
    label: str,
    source_dtype: torch.dtype | None = None,
) -> None:
    modes = vectors.shape[1]
    identity = torch.eye(modes, dtype=torch.float64)
    gram = vectors.to(dtype=torch.float64).T @ vectors.to(dtype=torch.float64)
    resolved_dtype = vectors.dtype if source_dtype is None else source_dtype
    tolerance = 2e-5 if resolved_dtype == torch.float32 else 1e-10
    if not torch.allclose(
        gram,
        identity,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(f"{label} vectors must have orthonormal columns")


def _validate_ranks(
    ranks: Iterable[int] | None,
    *,
    maximum: int,
) -> tuple[int, ...]:
    if ranks is None:
        return (maximum,)
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        requested = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of positive integers"
        ) from error
    if not requested:
        raise ValueError("ranks cannot be empty")
    if any(type(rank) is not int or rank <= 0 for rank in requested):
        raise ValueError("ranks must contain positive integers")
    if any(rank > maximum for rank in requested):
        raise ValueError(f"ranks cannot exceed the shared mode count {maximum}")
    return tuple(sorted(set(requested)))


@dataclass(frozen=True, slots=True)
class FisherSubspaceStabilityPoint:
    """Principal-angle agreement between two Fisher subspaces at one rank."""

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
        """Normalized projection overlap; one is identical, zero orthogonal."""

        return self.principal_cosines.square().mean().item()

    @property
    def minimum_principal_cosine(self) -> float:
        """Agreement of the least-aligned direction in the two subspaces."""

        return self.principal_cosines.min().item()

    @property
    def largest_principal_angle_degrees(self) -> float:
        """Largest principal angle, in degrees, after numerical clamping."""

        cosine = self.principal_cosines.clamp(0, 1).min().item()
        return math.degrees(math.acos(cosine))

    @property
    def normalized_projection_distance(self) -> float:
        """Projection distance normalized to the interval ``[0, 1]``."""

        return math.sqrt(max(0.0, 1.0 - self.mean_squared_overlap))

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serializable view of the stability point."""

        return {
            "rank": self.rank,
            "principal_cosines": self.principal_cosines.tolist(),
            "mean_squared_overlap": self.mean_squared_overlap,
            "minimum_principal_cosine": self.minimum_principal_cosine,
            "largest_principal_angle_degrees": (
                self.largest_principal_angle_degrees
            ),
            "normalized_projection_distance": (
                self.normalized_projection_distance
            ),
        }


@dataclass(frozen=True, slots=True)
class FisherSubspaceStability:
    """Rank-wise geometric agreement between two streamed Fisher results."""

    activation_name: str
    width: int
    left_observations: int
    right_observations: int
    points: tuple[FisherSubspaceStabilityPoint, ...]
    scope: str
    score_reduction: str
    normalizer: str

    def __post_init__(self) -> None:
        if not isinstance(self.activation_name, str) or not self.activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be positive")
        if type(self.left_observations) is not int or self.left_observations <= 0:
            raise ValueError("left_observations must be positive")
        if (
            type(self.right_observations) is not int
            or self.right_observations <= 0
        ):
            raise ValueError("right_observations must be positive")
        if not self.points:
            raise ValueError("stability report cannot be empty")
        ranks = tuple(point.rank for point in self.points)
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("stability ranks must be unique and ascending")
        for label, value in (
            ("scope", self.scope),
            ("score_reduction", self.score_reduction),
            ("normalizer", self.normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(point.rank for point in self.points)

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable geometry and provenance."""

        return {
            "activation_name": self.activation_name,
            "width": self.width,
            "left_observations": self.left_observations,
            "right_observations": self.right_observations,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "metric": "principal_angle_projection_overlap",
            "points": [point.metadata() for point in self.points],
        }


def compare_fisher_subspaces(
    left: StreamingFisherResult,
    right: StreamingFisherResult,
    *,
    ranks: Iterable[int] | None = None,
) -> FisherSubspaceStability:
    """Compare two split-specific Fisher subspaces at requested prefix ranks.

    This compares subspace geometry, not individual eigenvector identity.
    Consequently sign flips and rotations inside a degenerate eigenspace do
    not lower the score.  A small eigengap can still make a particular
    rank-``k`` boundary unstable; inspecting the curve over several ranks is
    more informative than relying on one mode-by-mode correlation.
    """

    if not isinstance(left, StreamingFisherResult) or not isinstance(
        right,
        StreamingFisherResult,
    ):
        raise TypeError("left and right must be StreamingFisherResult values")
    if left.activation_name != right.activation_name:
        raise ValueError("Fisher results must name the same activation")
    if left.width != right.width:
        raise ValueError("Fisher results must have the same activation width")
    for field in ("scope", "score_reduction", "normalizer"):
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"Fisher results disagree on {field}")
    _validate_orthonormal_columns(left.vectors, label="left Fisher")
    _validate_orthonormal_columns(right.vectors, label="right Fisher")

    maximum = min(left.modes, right.modes)
    requested = _validate_ranks(ranks, maximum=maximum)
    points = []
    for rank in requested:
        left_vectors = left.vectors[:, :rank].to(dtype=torch.float64)
        right_vectors = right.vectors[:, :rank].to(dtype=torch.float64)
        cosines = torch.linalg.svdvals(left_vectors.T @ right_vectors)
        # The inputs are orthonormal up to their accumulation precision.  Clamp
        # the tiny overshoot from the SVD before deriving angles and overlap.
        cosines = cosines.clamp(0, 1).cpu()
        points.append(
            FisherSubspaceStabilityPoint(
                rank=rank,
                principal_cosines=cosines,
            )
        )
    return FisherSubspaceStability(
        activation_name=left.activation_name,
        width=left.width,
        left_observations=left.observations,
        right_observations=right.observations,
        points=tuple(points),
        scope=left.scope,
        score_reduction=left.score_reduction,
        normalizer=left.normalizer,
    )


@dataclass(frozen=True, slots=True)
class StreamingRayleighEnergyResult:
    """Exact held-out Fisher energy measured in a frozen orthonormal basis."""

    activation_name: str
    width: int
    mode_energies: Tensor
    observations: int
    nonzero_observations: int
    rows_seen: int
    squared_gradient_norm_sum: float
    fisher_trace: float
    accumulation_dtype: str
    basis_sha256: str
    scope: str = "width_pooled"
    score_reduction: str = "sum"
    normalizer: str = "valid_activation_positions"
    algorithm: str = _RAYLEIGH_ALGORITHM
    algorithm_version: int = _RAYLEIGH_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.activation_name, str) or not self.activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be positive")
        if self.mode_energies.ndim != 1 or self.mode_energies.numel() == 0:
            raise ValueError("mode_energies must be a nonempty vector")
        if self.mode_energies.numel() > self.width:
            raise ValueError("mode count cannot exceed activation width")
        if self.mode_energies.device.type != "cpu":
            raise ValueError("mode_energies must be on CPU")
        if self.mode_energies.dtype not in (torch.float32, torch.float64):
            raise ValueError("mode_energies must be float32 or float64")
        if not torch.isfinite(self.mode_energies).all():
            raise ValueError("mode_energies must be finite")
        if (self.mode_energies < 0).any():
            raise ValueError("mode_energies cannot be negative")
        if type(self.observations) is not int or self.observations <= 0:
            raise ValueError("observations must be positive")
        if (
            type(self.nonzero_observations) is not int
            or not 0 <= self.nonzero_observations <= self.observations
        ):
            raise ValueError("nonzero_observations is out of range")
        if type(self.rows_seen) is not int or self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        trace_values = torch.tensor(
            [self.squared_gradient_norm_sum, self.fisher_trace],
            dtype=torch.float64,
        )
        if not torch.isfinite(trace_values).all():
            raise ValueError("Fisher trace values must be finite")
        if self.squared_gradient_norm_sum < 0 or self.fisher_trace < 0:
            raise ValueError("Fisher trace values cannot be negative")
        expected_trace = self.squared_gradient_norm_sum / self.observations
        if not torch.isclose(
            torch.tensor(self.fisher_trace, dtype=torch.float64),
            torch.tensor(expected_trace, dtype=torch.float64),
        ):
            raise ValueError(
                "fisher_trace must equal squared_gradient_norm_sum / observations"
            )
        tolerance = 128 * torch.finfo(self.mode_energies.dtype).eps * max(
            self.fisher_trace,
            1.0,
        )
        if self.mode_energies.sum().item() > self.fisher_trace + tolerance:
            raise ValueError(
                "orthonormal basis energy cannot exceed the Fisher trace"
            )
        for label, value in (
            ("accumulation_dtype", self.accumulation_dtype),
            ("basis_sha256", self.basis_sha256),
            ("scope", self.scope),
            ("score_reduction", self.score_reduction),
            ("normalizer", self.normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        expected_accumulation_dtype = str(
            self.mode_energies.dtype
        ).removeprefix("torch.")
        if self.accumulation_dtype != expected_accumulation_dtype:
            raise ValueError(
                "accumulation_dtype must match the mode-energy dtype"
            )
        if (
            len(self.basis_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.basis_sha256
            )
        ):
            raise ValueError("basis_sha256 must be a lowercase SHA-256 digest")
        if self.algorithm != _RAYLEIGH_ALGORITHM:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")
        if self.algorithm_version != _RAYLEIGH_ALGORITHM_VERSION:
            raise ValueError(
                f"unsupported algorithm version: {self.algorithm_version}"
            )

    @property
    def modes(self) -> int:
        return self.mode_energies.numel()

    @property
    def cumulative_energies(self) -> Tensor:
        return self.mode_energies.cumsum(dim=0)

    def retained_trace(self, rank: int | None = None) -> float:
        """Return exact validation Fisher trace inside the first ``rank`` modes."""

        resolved = self.modes if rank is None else rank
        if type(resolved) is not int or not 1 <= resolved <= self.modes:
            raise ValueError(f"rank must be between 1 and {self.modes}")
        return self.cumulative_energies[resolved - 1].item()

    def retained_fraction(self, rank: int | None = None) -> float:
        """Return exact held-out subspace energy divided by held-out trace."""

        if self.fisher_trace == 0:
            return 0.0
        return min(self.retained_trace(rank) / self.fisher_trace, 1.0)

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable exact replay metrics and provenance."""

        cumulative = self.cumulative_energies
        fractions = (
            torch.zeros_like(cumulative)
            if self.fisher_trace == 0
            else (cumulative / self.fisher_trace).clamp(max=1)
        )
        return {
            "activation_name": self.activation_name,
            "width": self.width,
            "modes": self.modes,
            "observations": self.observations,
            "nonzero_observations": self.nonzero_observations,
            "rows_seen": self.rows_seen,
            "squared_gradient_norm_sum": self.squared_gradient_norm_sum,
            "fisher_trace": self.fisher_trace,
            "basis_sha256": self.basis_sha256,
            "mode_energies": self.mode_energies.tolist(),
            "cumulative_retained_trace": cumulative.tolist(),
            "cumulative_retained_fraction": fractions.tolist(),
            "retained_trace_semantics": (
                "exact_validation_rayleigh_trace_over_exact_validation_trace"
            ),
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        """Return a strict analysis-only serialization mapping."""

        return {
            "format_version": 1,
            "activation_name": self.activation_name,
            "width": self.width,
            "mode_energies": self.mode_energies,
            "observations": self.observations,
            "nonzero_observations": self.nonzero_observations,
            "rows_seen": self.rows_seen,
            "squared_gradient_norm_sum": self.squared_gradient_norm_sum,
            "fisher_trace": self.fisher_trace,
            "accumulation_dtype": self.accumulation_dtype,
            "basis_sha256": self.basis_sha256,
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
    ) -> StreamingRayleighEnergyResult:
        """Restore a result produced by :meth:`state_dict`."""

        expected = {
            "format_version",
            "activation_name",
            "width",
            "mode_energies",
            "observations",
            "nonzero_observations",
            "rows_seen",
            "squared_gradient_norm_sum",
            "fisher_trace",
            "accumulation_dtype",
            "basis_sha256",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
        }
        if set(state) != expected:
            raise ValueError(
                "streaming Rayleigh result fields do not match format version 1"
            )
        if state["format_version"] != 1:
            raise ValueError("unsupported streaming Rayleigh result format")
        mode_energies = state["mode_energies"]
        if not isinstance(mode_energies, Tensor):
            raise TypeError("mode_energies must be a Tensor")
        return cls(
            activation_name=str(state["activation_name"]),
            width=int(state["width"]),
            mode_energies=mode_energies,
            observations=int(state["observations"]),
            nonzero_observations=int(state["nonzero_observations"]),
            rows_seen=int(state["rows_seen"]),
            squared_gradient_norm_sum=float(
                state["squared_gradient_norm_sum"]
            ),
            fisher_trace=float(state["fisher_trace"]),
            accumulation_dtype=str(state["accumulation_dtype"]),
            basis_sha256=str(state["basis_sha256"]),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )


class StreamingRayleighEnergyEstimator:
    """Measure held-out Fisher energy in fixed modes with bounded memory."""

    def __init__(
        self,
        *,
        activation_name: str,
        basis_vectors: Tensor,
        accumulation_dtype: torch.dtype = torch.float64,
        scope: str = "width_pooled",
        score_reduction: str = "sum",
        normalizer: str = "valid_activation_positions",
    ) -> None:
        if not isinstance(activation_name, str) or not activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if not isinstance(basis_vectors, Tensor):
            raise TypeError("basis_vectors must be a Tensor")
        if basis_vectors.ndim != 2:
            raise ValueError("basis_vectors must have shape [width, modes]")
        width, modes = basis_vectors.shape
        if width <= 0 or modes <= 0:
            raise ValueError("basis_vectors dimensions must be positive")
        if modes > width:
            raise ValueError("basis mode count cannot exceed its width")
        if basis_vectors.dtype not in (torch.float32, torch.float64):
            raise ValueError("basis_vectors must use float32 or float64")
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be float32 or float64")
        for label, value in (
            ("scope", scope),
            ("score_reduction", score_reduction),
            ("normalizer", normalizer),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")

        basis = basis_vectors.detach().to(
            device="cpu",
            dtype=accumulation_dtype,
        )
        if not torch.isfinite(basis).all():
            raise ValueError("basis_vectors must be finite")
        # Fisher bases may have originally been accumulated in float32, so use
        # a tolerance that accepts normal decomposition roundoff but rejects a
        # meaningfully scaled or correlated basis.
        _validate_orthonormal_columns(
            basis,
            label="basis_vectors",
            source_dtype=basis_vectors.dtype,
        )
        # Reorthonormalization removes source-precision drift without changing
        # the span of any leading column prefix.  The diagonal signs align the
        # resulting columns with the supplied modes.
        orthonormal, triangular = torch.linalg.qr(basis, mode="reduced")
        signs = triangular.diagonal().sign()
        signs[signs == 0] = 1
        basis = orthonormal * signs

        self.activation_name = activation_name
        self.accumulation_dtype = accumulation_dtype
        self.scope = scope
        self.score_reduction = score_reduction
        self.normalizer = normalizer
        self._basis = basis.contiguous()
        self._basis_sha256 = _basis_sha256(self._basis)
        self._mode_squared_norm_sums = torch.zeros(
            modes,
            dtype=accumulation_dtype,
        )
        self._observations = 0
        self._nonzero_observations = 0
        self._rows_seen = 0
        self._squared_norm_sum = 0.0

    @classmethod
    def from_fisher_result(
        cls,
        basis: StreamingFisherResult,
        *,
        rank: int | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> StreamingRayleighEnergyEstimator:
        """Freeze a prefix of modes from a streamed Fisher result."""

        if not isinstance(basis, StreamingFisherResult):
            raise TypeError("basis must be a StreamingFisherResult")
        resolved_rank = basis.modes if rank is None else rank
        if (
            type(resolved_rank) is not int
            or not 1 <= resolved_rank <= basis.modes
        ):
            raise ValueError(f"rank must be between 1 and {basis.modes}")
        return cls(
            activation_name=basis.activation_name,
            basis_vectors=basis.vectors[:, :resolved_rank],
            accumulation_dtype=accumulation_dtype,
            scope=basis.scope,
            score_reduction=basis.score_reduction,
            normalizer=basis.normalizer,
        )

    @property
    def width(self) -> int:
        return self._basis.shape[0]

    @property
    def modes(self) -> int:
        return self._basis.shape[1]

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def basis_sha256(self) -> str:
        """Digest binding results to the exact ordered frozen basis."""

        return self._basis_sha256

    @property
    def rows_seen(self) -> int:
        return self._rows_seen

    @property
    def storage_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Shapes of persistent tensors, independent of observation count."""

        return tuple(self._basis.shape), tuple(
            self._mode_squared_norm_sums.shape
        )

    def update(
        self,
        score_vectors: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> StreamingRayleighEnergyEstimator:
        """Add held-out ``[observations, width]`` score-gradient rows."""

        if not isinstance(score_vectors, Tensor):
            raise TypeError("score_vectors must be a Tensor")
        if score_vectors.ndim != 2:
            raise ValueError(
                "score_vectors must have shape [observations, width]"
            )
        if not score_vectors.is_floating_point():
            raise ValueError("score_vectors must use a real floating dtype")
        rows, width = score_vectors.shape
        if width != self.width:
            raise ValueError(
                f"expected score vector width {self.width}, got {width}"
            )
        if mask is not None:
            if not isinstance(mask, Tensor):
                raise TypeError("mask must be a Tensor")
            if mask.shape != (rows,):
                raise ValueError("mask must have shape [observations]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must use boolean dtype")
            selected = score_vectors[mask.to(device=score_vectors.device)]
        else:
            selected = score_vectors
        selected = selected.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        if not torch.isfinite(selected).all():
            raise ValueError("selected score vectors must be finite")

        # Validate before changing accounting, matching the Fisher estimator's
        # transactional behavior for rejected batches.
        self._rows_seen += rows
        if selected.shape[0] == 0:
            return self
        squared_norms = selected.square().sum(dim=1)
        projected = selected @ self._basis
        self._observations += selected.shape[0]
        self._nonzero_observations += int(
            torch.count_nonzero(squared_norms > 0).item()
        )
        self._squared_norm_sum += squared_norms.sum().item()
        self._mode_squared_norm_sums += projected.square().sum(dim=0)
        return self

    def finalize(self) -> StreamingRayleighEnergyResult:
        """Return exact fixed-basis validation energies."""

        if self._observations == 0:
            raise ValueError("cannot finalize without any selected observations")
        return StreamingRayleighEnergyResult(
            activation_name=self.activation_name,
            width=self.width,
            mode_energies=(
                self._mode_squared_norm_sums / self._observations
            ).clone(),
            observations=self._observations,
            nonzero_observations=self._nonzero_observations,
            rows_seen=self._rows_seen,
            squared_gradient_norm_sum=self._squared_norm_sum,
            fisher_trace=self._squared_norm_sum / self._observations,
            accumulation_dtype=str(self.accumulation_dtype).removeprefix(
                "torch."
            ),
            basis_sha256=self.basis_sha256,
            scope=self.scope,
            score_reduction=self.score_reduction,
            normalizer=self.normalizer,
        )


__all__ = [
    "FisherSubspaceStability",
    "FisherSubspaceStabilityPoint",
    "StreamingRayleighEnergyEstimator",
    "StreamingRayleighEnergyResult",
    "compare_fisher_subspaces",
]
