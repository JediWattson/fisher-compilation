"""Bounded-memory reverse-causal transport between Fisher modal bases.

For a causal transformer, an activation at logical position ``s`` can affect
the same and later output positions.  Reverse-mode gradients therefore flow in
the opposite triangular direction: an upstream gradient at ``s`` can depend on
downstream gradients at ``s + lag``.  A same-row transport omits those terms.

This module fits the finite-lag diagnostic

``upstream[s] ~= sum(downstream[s + lag] @ W_lag)``

in frozen Fisher coordinates.  Rows never leave the sequence that produced
them, missing logical positions are not compressed into false neighbours, and
only sufficient statistics are retained.  The lag-zero member is a nested
row-local ridge baseline for the wider reverse-causal models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import struct

import torch
from torch import Tensor

from .streaming_block_validation import (
    _basis_sha256,
    _is_sha256,
    _orthonormalized_prefix,
)
from .streaming_fisher import StreamingFisherResult


_RESULT_FORMAT_VERSION = 1
_RESULT_ALGORITHM = "streaming_exact_lag_reverse_causal_modal_moments"
_RESULT_ALGORITHM_VERSION = 1
_FROZEN_FORMAT_VERSION = 1
_FROZEN_ALGORITHM = "homogeneous_modal_ridge"
_FROZEN_ALGORITHM_VERSION = 1
_EVALUATION_FORMAT_VERSION = 1


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _validate_relative_ridge(value: float) -> None:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("relative_ridge must be finite and nonnegative")


def _validate_positions(
    positions: Tensor,
    *,
    rows: int,
    max_lag: int,
) -> Tensor:
    if (
        not isinstance(positions, Tensor)
        or positions.shape != (rows,)
        or positions.device.type != "cpu"
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "positions must be a CPU integer vector with one entry per row"
        )
    converted = positions.detach().to(dtype=torch.int64).contiguous()
    if (converted < 0).any():
        raise ValueError("positions must be nonnegative")
    if rows > 1 and not torch.all(converted[1:] > converted[:-1]):
        raise ValueError("positions must be strictly increasing")
    if converted[-1].item() > torch.iinfo(torch.int64).max - max_lag:
        raise ValueError("positions plus max_lag would overflow int64")
    return converted


def _prefix_feature_indices(
    *,
    basis_rank: int,
    rank: int,
    max_lag: int,
) -> Tensor:
    return torch.cat(
        [
            torch.arange(
                lag * basis_rank,
                lag * basis_rank + rank,
                dtype=torch.int64,
            )
            for lag in range(max_lag + 1)
        ]
    )


def _clamp_nonnegative(
    value: Tensor,
    *,
    label: str,
    scale: float,
    dimension: int,
) -> float:
    resolved = value.item()
    tolerance = (
        2048
        * torch.finfo(value.dtype).eps
        * max(scale, 1.0)
        * max(dimension, 1)
    )
    if resolved < -tolerance:
        raise ValueError(f"{label} implied by moments is negative")
    return max(resolved, 0.0)


def _validate_psd(matrix: Tensor, *, label: str) -> None:
    """Reject moment matrices with a materially negative eigenvalue."""

    symmetric = (matrix + matrix.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    scale = max(float(eigenvalues.abs().max().item()), 1.0)
    tolerance = (
        4096
        * torch.finfo(matrix.dtype).eps
        * scale
        * max(matrix.shape[0], 1)
    )
    if eigenvalues[0].item() < -tolerance:
        raise ValueError(f"{label} must be positive semidefinite")


@dataclass(frozen=True, slots=True)
class StreamingCausalModalTransportResult:
    """Sequence-scoped sufficient statistics for exact-lag modal transport."""

    source_layer: str
    target_layer: str
    row_kind: str
    source_width: int
    target_width: int
    source_basis_sha256: str
    target_basis_sha256: str
    rank: int
    max_lag: int
    visibility_window: int | None
    observations: int
    rows_seen: int
    sequences: int
    lag_pair_counts: tuple[int, ...]
    position_schedule_sha256: str
    feature_gram_sum: Tensor
    feature_target_cross_sum: Tensor
    target_gram_sum: Tensor
    accumulation_dtype: str
    scope: str
    score_reduction: str
    normalizer: str
    algorithm: str = _RESULT_ALGORITHM
    algorithm_version: int = _RESULT_ALGORITHM_VERSION

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
        if self.row_kind != "score_gradient":
            raise ValueError("causal modal transport is gradient-only")
        for label, value in (
            ("source_width", self.source_width),
            ("target_width", self.target_width),
            ("rank", self.rank),
            ("observations", self.observations),
            ("rows_seen", self.rows_seen),
            ("sequences", self.sequences),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        if type(self.max_lag) is not int or self.max_lag < 0:
            raise ValueError("max_lag must be nonnegative")
        if self.visibility_window is not None and (
            type(self.visibility_window) is not int
            or self.visibility_window <= 0
        ):
            raise ValueError("visibility_window must be positive or None")
        if self.rank > min(self.source_width, self.target_width):
            raise ValueError("rank cannot exceed a boundary width")
        if self.sequences > self.observations:
            raise ValueError("sequences cannot exceed observations")
        if (
            not isinstance(self.lag_pair_counts, tuple)
            or len(self.lag_pair_counts) != self.max_lag + 1
            or any(
                type(count) is not int
                or count < 0
                or count > self.observations
                for count in self.lag_pair_counts
            )
            or self.lag_pair_counts[0] != self.observations
        ):
            raise ValueError("lag_pair_counts are invalid")
        if not _is_sha256(self.position_schedule_sha256):
            raise ValueError("position schedule digest must be lowercase SHA-256")
        if not _is_sha256(self.source_basis_sha256) or not _is_sha256(
            self.target_basis_sha256
        ):
            raise ValueError("basis digests must be lowercase SHA-256 values")
        feature_width = (self.max_lag + 1) * self.rank
        for label, tensor, shape in (
            (
                "feature_gram_sum",
                self.feature_gram_sum,
                (feature_width, feature_width),
            ),
            (
                "feature_target_cross_sum",
                self.feature_target_cross_sum,
                (feature_width, self.rank),
            ),
            (
                "target_gram_sum",
                self.target_gram_sum,
                (self.rank, self.rank),
            ),
        ):
            if not isinstance(tensor, Tensor) or tensor.shape != shape:
                raise ValueError(f"{label} must have shape {list(shape)}")
            if tensor.device.type != "cpu" or tensor.dtype != torch.float64:
                raise ValueError(
                    "causal transport moments must be CPU float64 tensors"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError("causal transport moments must be finite")
        if any(
            tensor.dtype != self.feature_gram_sum.dtype
            for tensor in (
                self.feature_target_cross_sum,
                self.target_gram_sum,
            )
        ):
            raise ValueError("causal transport moments must share one dtype")
        if self.accumulation_dtype != _dtype_name(
            self.feature_gram_sum.dtype
        ):
            raise ValueError("accumulation_dtype does not match moments")
        if not torch.allclose(
            self.feature_gram_sum,
            self.feature_gram_sum.T,
        ) or not torch.allclose(
            self.target_gram_sum,
            self.target_gram_sum.T,
        ):
            raise ValueError("causal transport Gram sums must be symmetric")
        joint = torch.cat(
            (
                torch.cat(
                    (
                        self.feature_gram_sum,
                        self.feature_target_cross_sum,
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        self.feature_target_cross_sum.T,
                        self.target_gram_sum,
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        _validate_psd(joint, label="joint causal transport moments")
        if self.visibility_window is not None:
            for lag in range(self.visibility_window, self.max_lag + 1):
                block = slice(lag * self.rank, (lag + 1) * self.rank)
                if (
                    self.lag_pair_counts[lag] != 0
                    or torch.count_nonzero(
                        self.feature_gram_sum[block, :]
                    ).item()
                    or torch.count_nonzero(
                        self.feature_gram_sum[:, block]
                    ).item()
                    or torch.count_nonzero(
                        self.feature_target_cross_sum[block, :]
                    ).item()
                ):
                    raise ValueError(
                        "moments outside structural visibility must be zero"
                    )
        if self.algorithm != _RESULT_ALGORITHM:
            raise ValueError("unsupported causal modal transport algorithm")
        if self.algorithm_version != _RESULT_ALGORITHM_VERSION:
            raise ValueError(
                "unsupported causal modal transport algorithm version"
            )

    @property
    def feature_width(self) -> int:
        return (self.max_lag + 1) * self.rank

    def prefix_moments(
        self,
        *,
        rank: int,
        max_lag: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if type(rank) is not int or not 1 <= rank <= self.rank:
            raise ValueError(f"rank must be between 1 and {self.rank}")
        if (
            type(max_lag) is not int
            or not 0 <= max_lag <= self.max_lag
        ):
            raise ValueError(f"max_lag must be between 0 and {self.max_lag}")
        indices = _prefix_feature_indices(
            basis_rank=self.rank,
            rank=rank,
            max_lag=max_lag,
        )
        feature_gram = self.feature_gram_sum.index_select(
            0, indices
        ).index_select(1, indices)
        feature_target = self.feature_target_cross_sum.index_select(
            0, indices
        )[:, :rank]
        target_gram = self.target_gram_sum[:rank, :rank]
        return feature_gram, feature_target, target_gram

    def metadata(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "rank": self.rank,
            "max_lag": self.max_lag,
            "visibility_window": self.visibility_window,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "sequences": self.sequences,
            "lag_pair_counts": self.lag_pair_counts,
            "position_schedule_sha256": self.position_schedule_sha256,
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _RESULT_FORMAT_VERSION,
            **self.metadata(),
            "feature_gram_sum": self.feature_gram_sum,
            "feature_target_cross_sum": self.feature_target_cross_sum,
            "target_gram_sum": self.target_gram_sum,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StreamingCausalModalTransportResult:
        expected = {
            "format_version",
            "source_layer",
            "target_layer",
            "row_kind",
            "source_width",
            "target_width",
            "source_basis_sha256",
            "target_basis_sha256",
            "rank",
            "max_lag",
            "visibility_window",
            "observations",
            "rows_seen",
            "sequences",
            "lag_pair_counts",
            "position_schedule_sha256",
            "feature_gram_sum",
            "feature_target_cross_sum",
            "target_gram_sum",
            "accumulation_dtype",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("streaming causal modal transport fields are invalid")
        if state["format_version"] != _RESULT_FORMAT_VERSION:
            raise ValueError("unsupported streaming causal transport format")
        for name in (
            "feature_gram_sum",
            "feature_target_cross_sum",
            "target_gram_sum",
        ):
            if not isinstance(state[name], Tensor):
                raise TypeError("causal transport moments must be Tensors")
        lag_pair_counts = state["lag_pair_counts"]
        if not isinstance(lag_pair_counts, tuple):
            raise TypeError("lag_pair_counts must be a tuple")
        return cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            source_width=int(state["source_width"]),
            target_width=int(state["target_width"]),
            source_basis_sha256=str(state["source_basis_sha256"]),
            target_basis_sha256=str(state["target_basis_sha256"]),
            rank=int(state["rank"]),
            max_lag=int(state["max_lag"]),
            visibility_window=(
                None
                if state["visibility_window"] is None
                else int(state["visibility_window"])
            ),
            observations=int(state["observations"]),
            rows_seen=int(state["rows_seen"]),
            sequences=int(state["sequences"]),
            lag_pair_counts=lag_pair_counts,
            position_schedule_sha256=str(
                state["position_schedule_sha256"]
            ),
            feature_gram_sum=state["feature_gram_sum"],  # type: ignore[arg-type]
            feature_target_cross_sum=state[
                "feature_target_cross_sum"
            ],  # type: ignore[arg-type]
            target_gram_sum=state["target_gram_sum"],  # type: ignore[arg-type]
            accumulation_dtype=str(state["accumulation_dtype"]),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )


class StreamingCausalModalTransportEstimator:
    """Accumulate sequence-local exact-lag modal regression moments."""

    def __init__(
        self,
        source_fisher: StreamingFisherResult,
        target_fisher: StreamingFisherResult,
        *,
        rank: int,
        max_lag: int,
        row_kind: str = "score_gradient",
        source_layer: str | None = None,
        target_layer: str | None = None,
        visibility_window: int | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> None:
        if not isinstance(
            source_fisher, StreamingFisherResult
        ) or not isinstance(target_fisher, StreamingFisherResult):
            raise TypeError(
                "source_fisher and target_fisher must be "
                "StreamingFisherResult values"
            )
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(source_fisher.modes, target_fisher.modes):
            raise ValueError("rank exceeds the available Fisher modes")
        if type(max_lag) is not int or max_lag < 0:
            raise ValueError("max_lag must be nonnegative")
        if visibility_window is not None and (
            type(visibility_window) is not int or visibility_window <= 0
        ):
            raise ValueError("visibility_window must be positive or None")
        if row_kind != "score_gradient":
            raise ValueError("causal modal transport is gradient-only")
        if accumulation_dtype != torch.float64:
            raise ValueError("accumulation_dtype must be float64")
        if (
            source_fisher.scope,
            source_fisher.score_reduction,
            source_fisher.normalizer,
        ) != (
            target_fisher.scope,
            target_fisher.score_reduction,
            target_fisher.normalizer,
        ):
            raise ValueError("source and target Fisher provenance disagree")
        resolved_source = (
            source_fisher.activation_name
            if source_layer is None
            else source_layer
        )
        resolved_target = (
            target_fisher.activation_name
            if target_layer is None
            else target_layer
        )
        for label, value in (
            ("source_layer", resolved_source),
            ("target_layer", resolved_target),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if resolved_source == resolved_target:
            raise ValueError("source and target layers must differ")

        self.source_fisher = source_fisher
        self.target_fisher = target_fisher
        self.source_layer = resolved_source
        self.target_layer = resolved_target
        self.row_kind = row_kind
        self.max_lag = max_lag
        self.visibility_window = visibility_window
        self.accumulation_dtype = accumulation_dtype
        self._source_basis = _orthonormalized_prefix(
            source_fisher.vectors, rank
        ).to(dtype=accumulation_dtype)
        self._target_basis = _orthonormalized_prefix(
            target_fisher.vectors, rank
        ).to(dtype=accumulation_dtype)
        self._source_basis_sha256 = _basis_sha256(self._source_basis)
        self._target_basis_sha256 = _basis_sha256(self._target_basis)
        feature_width = (max_lag + 1) * rank
        self._feature_gram_sum = torch.zeros(
            (feature_width, feature_width),
            dtype=accumulation_dtype,
        )
        self._feature_target_cross_sum = torch.zeros(
            (feature_width, rank),
            dtype=accumulation_dtype,
        )
        self._target_gram_sum = torch.zeros(
            (rank, rank),
            dtype=accumulation_dtype,
        )
        self._lag_pair_counts = [0] * (max_lag + 1)
        self._position_digest = hashlib.sha256()
        self._position_digest.update(
            b"fisher_graph.causal_modal_position_schedule.v1\0"
        )
        self._observations = 0
        self._rows_seen = 0
        self._sequences = 0

    @property
    def rank(self) -> int:
        return self._source_basis.shape[1]

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def sequences(self) -> int:
        return self._sequences

    @property
    def source_basis_sha256(self) -> str:
        return self._source_basis_sha256

    @property
    def target_basis_sha256(self) -> str:
        return self._target_basis_sha256

    @property
    def storage_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "source_basis": tuple(self._source_basis.shape),
            "target_basis": tuple(self._target_basis.shape),
            "feature_gram_sum": tuple(self._feature_gram_sum.shape),
            "feature_target_cross_sum": tuple(
                self._feature_target_cross_sum.shape
            ),
            "target_gram_sum": tuple(self._target_gram_sum.shape),
            "lag_pair_counts": (len(self._lag_pair_counts),),
        }

    def update_sequence(
        self,
        source_rows: Tensor,
        target_rows: Tensor,
        positions: Tensor,
    ) -> StreamingCausalModalTransportEstimator:
        """Add one independent sequence without retaining its transient rows."""

        if not isinstance(source_rows, Tensor) or not isinstance(
            target_rows, Tensor
        ):
            raise TypeError("source_rows and target_rows must be Tensors")
        if source_rows.ndim != 2 or target_rows.ndim != 2:
            raise ValueError("sequence rows must have shape [positions, width]")
        if source_rows.shape[0] != target_rows.shape[0]:
            raise ValueError("sequence rows must have the same position count")
        if source_rows.shape[0] <= 0:
            raise ValueError("a sequence must contain at least one row")
        if source_rows.shape[1] != self.source_fisher.width:
            raise ValueError("source row width does not match the Fisher basis")
        if target_rows.shape[1] != self.target_fisher.width:
            raise ValueError("target row width does not match the Fisher basis")
        if not source_rows.is_floating_point() or not (
            target_rows.is_floating_point()
        ):
            raise ValueError("sequence rows must use real floating dtypes")
        rows = source_rows.shape[0]
        positions = _validate_positions(
            positions,
            rows=rows,
            max_lag=self.max_lag,
        )
        source_rows = source_rows.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        target_rows = target_rows.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        if not torch.isfinite(source_rows).all() or not torch.isfinite(
            target_rows
        ).all():
            raise ValueError("sequence rows must be finite")

        source_modal = source_rows @ self._source_basis
        target_modal = target_rows @ self._target_basis
        features = torch.zeros(
            (rows, (self.max_lag + 1) * self.rank),
            dtype=self.accumulation_dtype,
        )
        for lag in range(self.max_lag + 1):
            if (
                self.visibility_window is not None
                and lag >= self.visibility_window
            ):
                continue
            desired = positions + lag
            indices = torch.searchsorted(positions, desired)
            in_range = indices < rows
            valid = torch.zeros(rows, dtype=torch.bool)
            if in_range.any():
                candidate_rows = torch.nonzero(
                    in_range, as_tuple=False
                ).flatten()
                candidate_indices = indices[in_range]
                valid[candidate_rows] = (
                    positions[candidate_indices] == desired[candidate_rows]
                )
            if valid.any():
                features[
                    valid,
                    lag * self.rank : (lag + 1) * self.rank,
                ] = source_modal[indices[valid]]
            self._lag_pair_counts[lag] += int(valid.sum().item())

        self._feature_gram_sum += features.T @ features
        self._feature_target_cross_sum += features.T @ target_modal
        self._target_gram_sum += target_modal.T @ target_modal
        self._observations += rows
        self._rows_seen += rows
        self._sequences += 1
        canonical_positions = positions.contiguous().numpy().tobytes(order="C")
        self._position_digest.update(struct.pack("<Q", rows))
        self._position_digest.update(canonical_positions)
        return self

    def finalize(self) -> StreamingCausalModalTransportResult:
        if self._observations == 0:
            raise ValueError("cannot finalize without sequence observations")
        return StreamingCausalModalTransportResult(
            source_layer=self.source_layer,
            target_layer=self.target_layer,
            row_kind=self.row_kind,
            source_width=self.source_fisher.width,
            target_width=self.target_fisher.width,
            source_basis_sha256=self._source_basis_sha256,
            target_basis_sha256=self._target_basis_sha256,
            rank=self.rank,
            max_lag=self.max_lag,
            visibility_window=self.visibility_window,
            observations=self._observations,
            rows_seen=self._rows_seen,
            sequences=self._sequences,
            lag_pair_counts=tuple(self._lag_pair_counts),
            position_schedule_sha256=self._position_digest.hexdigest(),
            feature_gram_sum=self._feature_gram_sum.clone(),
            feature_target_cross_sum=(
                self._feature_target_cross_sum.clone()
            ),
            target_gram_sum=self._target_gram_sum.clone(),
            accumulation_dtype=_dtype_name(self.accumulation_dtype),
            scope=self.source_fisher.scope,
            score_reduction=self.source_fisher.score_reduction,
            normalizer=self.source_fisher.normalizer,
        )


@dataclass(frozen=True, slots=True)
class FrozenCausalModalTransport:
    """Calibration-fit homogeneous ridge map for one rank and lag window."""

    source_layer: str
    target_layer: str
    row_kind: str
    source_width: int
    target_width: int
    source_basis_sha256: str
    target_basis_sha256: str
    basis_rank: int
    rank: int
    basis_max_lag: int
    max_lag: int
    visibility_window: int | None
    matrix: Tensor
    relative_ridge: float
    ridge_penalty: float
    feature_effective_rank: int
    feature_condition_number: float | None
    calibration_observations: int
    calibration_sequences: int
    calibration_lag_pair_counts: tuple[int, ...]
    calibration_position_schedule_sha256: str
    accumulation_dtype: str
    scope: str
    score_reduction: str
    normalizer: str
    algorithm: str = _FROZEN_ALGORITHM
    algorithm_version: int = _FROZEN_ALGORITHM_VERSION

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
        if self.row_kind != "score_gradient":
            raise ValueError("causal modal transport is gradient-only")
        for label, value in (
            ("source_width", self.source_width),
            ("target_width", self.target_width),
            ("basis_rank", self.basis_rank),
            ("rank", self.rank),
            ("calibration_observations", self.calibration_observations),
            ("calibration_sequences", self.calibration_sequences),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.rank > self.basis_rank:
            raise ValueError("rank cannot exceed basis_rank")
        if self.basis_rank > min(self.source_width, self.target_width):
            raise ValueError("basis_rank cannot exceed a boundary width")
        if self.calibration_sequences > self.calibration_observations:
            raise ValueError(
                "calibration_sequences cannot exceed observations"
            )
        if type(self.basis_max_lag) is not int or self.basis_max_lag < 0:
            raise ValueError("basis_max_lag must be nonnegative")
        if (
            type(self.max_lag) is not int
            or not 0 <= self.max_lag <= self.basis_max_lag
        ):
            raise ValueError("max_lag must be within the fitted lag basis")
        if self.visibility_window is not None and (
            type(self.visibility_window) is not int
            or self.visibility_window <= 0
        ):
            raise ValueError("visibility_window must be positive or None")
        if self.matrix.shape != (
            (self.max_lag + 1) * self.rank,
            self.rank,
        ):
            raise ValueError("matrix has the wrong causal transport shape")
        if (
            self.matrix.device.type != "cpu"
            or self.matrix.dtype != torch.float64
            or not torch.isfinite(self.matrix).all()
        ):
            raise ValueError("matrix must be a finite CPU float64 tensor")
        if self.accumulation_dtype != _dtype_name(self.matrix.dtype):
            raise ValueError("accumulation_dtype does not match matrix")
        _validate_relative_ridge(self.relative_ridge)
        if (
            not isinstance(self.ridge_penalty, float)
            or not math.isfinite(self.ridge_penalty)
            or self.ridge_penalty < 0
        ):
            raise ValueError("ridge_penalty must be finite and nonnegative")
        active_lags = (
            self.max_lag + 1
            if self.visibility_window is None
            else min(self.max_lag + 1, self.visibility_window)
        )
        feature_width = active_lags * self.rank
        if (
            type(self.feature_effective_rank) is not int
            or not 0 <= self.feature_effective_rank <= feature_width
        ):
            raise ValueError("feature_effective_rank is invalid")
        if self.feature_condition_number is None:
            if self.feature_effective_rank != 0:
                raise ValueError(
                    "nonzero effective rank requires a condition number"
                )
        elif (
            not isinstance(self.feature_condition_number, float)
            or not math.isfinite(self.feature_condition_number)
            or self.feature_condition_number < 1
            or self.feature_effective_rank == 0
        ):
            raise ValueError("feature_condition_number is invalid")
        if (
            not isinstance(self.calibration_lag_pair_counts, tuple)
            or len(self.calibration_lag_pair_counts) != self.max_lag + 1
            or any(
                type(count) is not int
                or count < 0
                or count > self.calibration_observations
                for count in self.calibration_lag_pair_counts
            )
            or self.calibration_lag_pair_counts[0]
            != self.calibration_observations
        ):
            raise ValueError("calibration lag-pair counts are invalid")
        if self.visibility_window is not None:
            for lag in range(self.visibility_window, self.max_lag + 1):
                block = slice(lag * self.rank, (lag + 1) * self.rank)
                if (
                    self.calibration_lag_pair_counts[lag] != 0
                    or torch.count_nonzero(self.matrix[block]).item()
                ):
                    raise ValueError(
                        "frozen coefficients outside visibility must be zero"
                    )
        if not _is_sha256(
            self.calibration_position_schedule_sha256
        ) or not _is_sha256(self.source_basis_sha256) or not _is_sha256(
            self.target_basis_sha256
        ):
            raise ValueError("frozen transport digests are invalid")
        if self.algorithm != _FROZEN_ALGORITHM:
            raise ValueError("unsupported frozen causal transport algorithm")
        if self.algorithm_version != _FROZEN_ALGORITHM_VERSION:
            raise ValueError(
                "unsupported frozen causal transport algorithm version"
            )

    @property
    def lag_matrices(self) -> Tensor:
        return self.matrix.reshape(
            self.max_lag + 1,
            self.rank,
            self.rank,
        )

    @property
    def lag_matrix_norms(self) -> tuple[float, ...]:
        return tuple(
            float(torch.linalg.vector_norm(matrix).item())
            for matrix in self.lag_matrices
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "basis_rank": self.basis_rank,
            "rank": self.rank,
            "basis_max_lag": self.basis_max_lag,
            "max_lag": self.max_lag,
            "visibility_window": self.visibility_window,
            "lag_matrix_norms": self.lag_matrix_norms,
            "relative_ridge": self.relative_ridge,
            "ridge_penalty": self.ridge_penalty,
            "feature_effective_rank": self.feature_effective_rank,
            "feature_condition_number": self.feature_condition_number,
            "calibration_observations": self.calibration_observations,
            "calibration_sequences": self.calibration_sequences,
            "calibration_lag_pair_counts": (
                self.calibration_lag_pair_counts
            ),
            "calibration_position_schedule_sha256": (
                self.calibration_position_schedule_sha256
            ),
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _FROZEN_FORMAT_VERSION,
            **self.metadata(),
            "matrix": self.matrix,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FrozenCausalModalTransport:
        expected = {
            "format_version",
            "source_layer",
            "target_layer",
            "row_kind",
            "source_width",
            "target_width",
            "source_basis_sha256",
            "target_basis_sha256",
            "basis_rank",
            "rank",
            "basis_max_lag",
            "max_lag",
            "visibility_window",
            "lag_matrix_norms",
            "relative_ridge",
            "ridge_penalty",
            "feature_effective_rank",
            "feature_condition_number",
            "calibration_observations",
            "calibration_sequences",
            "calibration_lag_pair_counts",
            "calibration_position_schedule_sha256",
            "accumulation_dtype",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
            "matrix",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("frozen causal modal transport fields are invalid")
        if state["format_version"] != _FROZEN_FORMAT_VERSION:
            raise ValueError("unsupported frozen causal transport format")
        if not isinstance(state["matrix"], Tensor):
            raise TypeError("frozen causal transport matrix must be a Tensor")
        lag_counts = state["calibration_lag_pair_counts"]
        if not isinstance(lag_counts, tuple):
            raise TypeError("calibration_lag_pair_counts must be a tuple")
        result = cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            source_width=int(state["source_width"]),
            target_width=int(state["target_width"]),
            source_basis_sha256=str(state["source_basis_sha256"]),
            target_basis_sha256=str(state["target_basis_sha256"]),
            basis_rank=int(state["basis_rank"]),
            rank=int(state["rank"]),
            basis_max_lag=int(state["basis_max_lag"]),
            max_lag=int(state["max_lag"]),
            visibility_window=(
                None
                if state["visibility_window"] is None
                else int(state["visibility_window"])
            ),
            matrix=state["matrix"],  # type: ignore[arg-type]
            relative_ridge=float(state["relative_ridge"]),
            ridge_penalty=float(state["ridge_penalty"]),
            feature_effective_rank=int(state["feature_effective_rank"]),
            feature_condition_number=(
                None
                if state["feature_condition_number"] is None
                else float(state["feature_condition_number"])
            ),
            calibration_observations=int(
                state["calibration_observations"]
            ),
            calibration_sequences=int(state["calibration_sequences"]),
            calibration_lag_pair_counts=lag_counts,
            calibration_position_schedule_sha256=str(
                state["calibration_position_schedule_sha256"]
            ),
            accumulation_dtype=str(state["accumulation_dtype"]),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )
        stored_norms = state["lag_matrix_norms"]
        if (
            not isinstance(stored_norms, tuple)
            or len(stored_norms) != result.max_lag + 1
            or any(
                not isinstance(value, float) or not math.isfinite(value)
                for value in stored_norms
            )
            or not all(
                math.isclose(
                    actual,
                    expected_value,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                for actual, expected_value in zip(
                    stored_norms, result.lag_matrix_norms
                )
            )
        ):
            raise ValueError("lag matrix norms do not match the matrix")
        return result


def freeze_causal_modal_transport(
    result: StreamingCausalModalTransportResult,
    *,
    rank: int | None = None,
    max_lag: int | None = None,
    relative_ridge: float = 1e-2,
) -> FrozenCausalModalTransport:
    """Fit one rank/lag-prefix homogeneous ridge map from streamed moments."""

    if not isinstance(result, StreamingCausalModalTransportResult):
        raise TypeError(
            "result must be a StreamingCausalModalTransportResult"
        )
    resolved_rank = result.rank if rank is None else rank
    resolved_lag = result.max_lag if max_lag is None else max_lag
    _validate_relative_ridge(relative_ridge)
    feature_gram, feature_target, _ = result.prefix_moments(
        rank=resolved_rank,
        max_lag=resolved_lag,
    )
    dtype = feature_gram.dtype
    dimension = feature_gram.shape[0]
    active_lags = (
        resolved_lag + 1
        if result.visibility_window is None
        else min(resolved_lag + 1, result.visibility_window)
    )
    active_dimension = active_lags * resolved_rank
    scale = float(
        feature_gram.diagonal().sum().item() / active_dimension
    )
    eigenvalues = torch.linalg.eigvalsh(
        (feature_gram + feature_gram.T) / 2
    )
    largest = max(float(eigenvalues[-1].item()), 0.0)
    supported = eigenvalues > max(largest * 1e-10, 0.0)
    feature_effective_rank = int(supported.sum().item())
    feature_condition_number = (
        None
        if feature_effective_rank == 0
        else float(
            largest / eigenvalues[supported].min().item()
        )
    )
    if scale == 0:
        penalty = 0.0
        matrix = torch.zeros(
            (dimension, resolved_rank),
            dtype=dtype,
        )
    else:
        penalty = relative_ridge * scale
        numerical_resolution = (
            torch.finfo(dtype).eps
            * max(largest, scale)
            * max(dimension, 1)
        )
        if penalty <= numerical_resolution:
            penalty = 0.0
        regularized = feature_gram + penalty * torch.eye(
            dimension,
            dtype=dtype,
        )
        if penalty == 0:
            matrix = torch.linalg.pinv(
                regularized,
                rtol=1e-12,
                atol=0.0,
            ) @ feature_target
        else:
            matrix = torch.linalg.solve(regularized, feature_target)
    return FrozenCausalModalTransport(
        source_layer=result.source_layer,
        target_layer=result.target_layer,
        row_kind=result.row_kind,
        source_width=result.source_width,
        target_width=result.target_width,
        source_basis_sha256=result.source_basis_sha256,
        target_basis_sha256=result.target_basis_sha256,
        basis_rank=result.rank,
        rank=resolved_rank,
        basis_max_lag=result.max_lag,
        max_lag=resolved_lag,
        visibility_window=result.visibility_window,
        matrix=matrix.cpu(),
        relative_ridge=relative_ridge,
        ridge_penalty=float(penalty),
        feature_effective_rank=feature_effective_rank,
        feature_condition_number=feature_condition_number,
        calibration_observations=result.observations,
        calibration_sequences=result.sequences,
        calibration_lag_pair_counts=result.lag_pair_counts[
            : resolved_lag + 1
        ],
        calibration_position_schedule_sha256=(
            result.position_schedule_sha256
        ),
        accumulation_dtype=result.accumulation_dtype,
        scope=result.scope,
        score_reduction=result.score_reduction,
        normalizer=result.normalizer,
    )


@dataclass(frozen=True, slots=True)
class FrozenCausalModalTransportEvaluation:
    """Held-out zero-baseline accounting for a frozen causal modal map."""

    source_layer: str
    target_layer: str
    row_kind: str
    source_width: int
    target_width: int
    rank: int
    max_lag: int
    visibility_window: int | None
    observations: int
    rows_seen: int
    sequences: int
    lag_pair_counts: tuple[int, ...]
    position_schedule_sha256: str
    target_baseline_squared_error: float
    transport_squared_error: float
    transport_target_dot_sum: float
    transport_squared_norm_sum: float
    target_squared_norm_sum: float
    source_basis_sha256: str
    target_basis_sha256: str
    accumulation_dtype: str
    scope: str
    score_reduction: str
    normalizer: str
    algorithm: str = _FROZEN_ALGORITHM
    algorithm_version: int = _FROZEN_ALGORITHM_VERSION

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
        if self.row_kind != "score_gradient":
            raise ValueError("causal modal transport is gradient-only")
        for label, value in (
            ("source_width", self.source_width),
            ("target_width", self.target_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        for label, value in (
            ("rank", self.rank),
            ("observations", self.observations),
            ("rows_seen", self.rows_seen),
            ("sequences", self.sequences),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        if self.sequences > self.observations:
            raise ValueError("sequences cannot exceed observations")
        if self.rank > min(self.source_width, self.target_width):
            raise ValueError("rank cannot exceed a boundary width")
        if type(self.max_lag) is not int or self.max_lag < 0:
            raise ValueError("max_lag must be nonnegative")
        if self.visibility_window is not None and (
            type(self.visibility_window) is not int
            or self.visibility_window <= 0
        ):
            raise ValueError("visibility_window must be positive or None")
        if (
            not isinstance(self.lag_pair_counts, tuple)
            or len(self.lag_pair_counts) != self.max_lag + 1
            or any(
                type(count) is not int
                or count < 0
                or count > self.observations
                for count in self.lag_pair_counts
            )
            or self.lag_pair_counts[0] != self.observations
        ):
            raise ValueError("held-out lag-pair counts are invalid")
        for label, value in (
            (
                "target_baseline_squared_error",
                self.target_baseline_squared_error,
            ),
            ("transport_squared_error", self.transport_squared_error),
            (
                "transport_squared_norm_sum",
                self.transport_squared_norm_sum,
            ),
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
        if not _is_sha256(self.position_schedule_sha256) or not _is_sha256(
            self.source_basis_sha256
        ) or not _is_sha256(self.target_basis_sha256):
            raise ValueError("held-out causal transport digests are invalid")
        if self.accumulation_dtype != "float64":
            raise ValueError("accumulation_dtype must be float64")
        if self.visibility_window is not None and any(
            self.lag_pair_counts[lag] != 0
            for lag in range(self.visibility_window, self.max_lag + 1)
        ):
            raise ValueError(
                "held-out pairs outside structural visibility must be zero"
            )
        accounting_dimension = (
            (self.max_lag + 1) * self.rank * self.rank
        )
        accounting_scale = max(
            self.target_baseline_squared_error,
            self.transport_squared_error,
            self.transport_squared_norm_sum,
            self.target_squared_norm_sum,
            abs(2 * self.transport_target_dot_sum),
            1.0,
        )
        accounting_tolerance = (
            4096
            * torch.finfo(torch.float64).eps
            * accounting_scale
            * max(accounting_dimension, 1)
        )
        expected_error = (
            self.target_squared_norm_sum
            + self.transport_squared_norm_sum
            - 2 * self.transport_target_dot_sum
        )
        if abs(
            self.target_baseline_squared_error
            - self.target_squared_norm_sum
        ) > accounting_tolerance or abs(
            self.transport_squared_error - expected_error
        ) > accounting_tolerance:
            raise ValueError(
                "held-out causal transport accounting is inconsistent"
            )
        cauchy_tolerance = (
            4096
            * torch.finfo(torch.float64).eps
            * max(
                self.transport_squared_norm_sum
                * self.target_squared_norm_sum,
                1.0,
            )
            * max(accounting_dimension, 1)
        )
        if self.transport_target_dot_sum**2 > (
            self.transport_squared_norm_sum
            * self.target_squared_norm_sum
            + cauchy_tolerance
        ):
            raise ValueError(
                "held-out causal transport violates the Cauchy bound"
            )
        if self.algorithm != _FROZEN_ALGORITHM:
            raise ValueError("unsupported frozen causal transport algorithm")
        if self.algorithm_version != _FROZEN_ALGORITHM_VERSION:
            raise ValueError(
                "unsupported frozen causal transport algorithm version"
            )

    @property
    def baseline_kind(self) -> str:
        return "zero"

    @property
    def data_split_independence(self) -> str:
        return "caller_responsibility"

    @property
    def transport_explained_fraction(self) -> float | None:
        if self.target_baseline_squared_error == 0:
            return 1.0 if self.transport_squared_error == 0 else None
        return (
            1.0
            - self.transport_squared_error
            / self.target_baseline_squared_error
        )

    @property
    def transport_r_squared(self) -> None:
        """R-squared is intentionally undefined for a zero baseline."""

        return None

    @property
    def transport_normalized_rmse(self) -> float | None:
        if self.target_baseline_squared_error == 0:
            return 0.0 if self.transport_squared_error == 0 else None
        return math.sqrt(
            self.transport_squared_error
            / self.target_baseline_squared_error
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

    def metadata(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "row_kind": self.row_kind,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "rank": self.rank,
            "max_lag": self.max_lag,
            "visibility_window": self.visibility_window,
            "observations": self.observations,
            "rows_seen": self.rows_seen,
            "sequences": self.sequences,
            "lag_pair_counts": self.lag_pair_counts,
            "position_schedule_sha256": self.position_schedule_sha256,
            "baseline_kind": self.baseline_kind,
            "data_split_independence": self.data_split_independence,
            "transport_explained_fraction": (
                self.transport_explained_fraction
            ),
            "transport_r_squared": self.transport_r_squared,
            "transport_normalized_rmse": (
                self.transport_normalized_rmse
            ),
            "transport_target_cosine": self.transport_target_cosine,
            "target_baseline_squared_error": (
                self.target_baseline_squared_error
            ),
            "transport_squared_error": self.transport_squared_error,
            "transport_target_dot_sum": self.transport_target_dot_sum,
            "transport_squared_norm_sum": (
                self.transport_squared_norm_sum
            ),
            "target_squared_norm_sum": self.target_squared_norm_sum,
            "source_basis_sha256": self.source_basis_sha256,
            "target_basis_sha256": self.target_basis_sha256,
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _EVALUATION_FORMAT_VERSION,
            **self.metadata(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FrozenCausalModalTransportEvaluation:
        metadata_only = {
            "source_layer",
            "target_layer",
            "row_kind",
            "source_width",
            "target_width",
            "rank",
            "max_lag",
            "visibility_window",
            "observations",
            "rows_seen",
            "sequences",
            "lag_pair_counts",
            "position_schedule_sha256",
            "baseline_kind",
            "data_split_independence",
            "transport_explained_fraction",
            "transport_r_squared",
            "transport_normalized_rmse",
            "transport_target_cosine",
            "target_baseline_squared_error",
            "transport_squared_error",
            "transport_target_dot_sum",
            "transport_squared_norm_sum",
            "target_squared_norm_sum",
            "source_basis_sha256",
            "target_basis_sha256",
            "accumulation_dtype",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
        }
        if not isinstance(state, Mapping) or set(state) != (
            metadata_only | {"format_version"}
        ):
            raise ValueError("causal modal evaluation fields are invalid")
        if state["format_version"] != _EVALUATION_FORMAT_VERSION:
            raise ValueError("unsupported causal modal evaluation format")
        lag_counts = state["lag_pair_counts"]
        if not isinstance(lag_counts, tuple):
            raise TypeError("lag_pair_counts must be a tuple")
        result = cls(
            source_layer=str(state["source_layer"]),
            target_layer=str(state["target_layer"]),
            row_kind=str(state["row_kind"]),
            source_width=int(state["source_width"]),
            target_width=int(state["target_width"]),
            rank=int(state["rank"]),
            max_lag=int(state["max_lag"]),
            visibility_window=(
                None
                if state["visibility_window"] is None
                else int(state["visibility_window"])
            ),
            observations=int(state["observations"]),
            rows_seen=int(state["rows_seen"]),
            sequences=int(state["sequences"]),
            lag_pair_counts=lag_counts,
            position_schedule_sha256=str(
                state["position_schedule_sha256"]
            ),
            target_baseline_squared_error=float(
                state["target_baseline_squared_error"]
            ),
            transport_squared_error=float(
                state["transport_squared_error"]
            ),
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
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )
        expected = result.metadata()
        if state["baseline_kind"] != expected["baseline_kind"]:
            raise ValueError(
                "causal modal evaluation derived fields are invalid"
            )
        if (
            state["data_split_independence"]
            != expected["data_split_independence"]
        ):
            raise ValueError(
                "causal modal evaluation provenance fields are invalid"
            )
        for name in (
            "transport_explained_fraction",
            "transport_r_squared",
            "transport_normalized_rmse",
            "transport_target_cosine",
        ):
            actual_value = state[name]
            expected_value = expected[name]
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    raise ValueError(
                        "causal modal evaluation derived fields are invalid"
                    )
            elif not (
                isinstance(actual_value, float)
                and math.isfinite(actual_value)
                and math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "causal modal evaluation derived fields are invalid"
                )
        return result


def evaluate_frozen_causal_modal_transport(
    transport: FrozenCausalModalTransport,
    heldout: StreamingCausalModalTransportResult,
) -> FrozenCausalModalTransportEvaluation:
    """Evaluate a frozen map exactly from held-out sufficient statistics.

    The moment objects authenticate geometry and Fisher-basis compatibility,
    not dataset independence.  The caller must keep fit and held-out streams
    disjoint; experiment artifacts should bind their external split
    provenance separately.
    """

    if not isinstance(transport, FrozenCausalModalTransport):
        raise TypeError("transport must be a FrozenCausalModalTransport")
    if not isinstance(heldout, StreamingCausalModalTransportResult):
        raise TypeError(
            "heldout must be a StreamingCausalModalTransportResult"
        )
    bindings = (
        ("source_layer", transport.source_layer, heldout.source_layer),
        ("target_layer", transport.target_layer, heldout.target_layer),
        ("row_kind", transport.row_kind, heldout.row_kind),
        (
            "source_basis_sha256",
            transport.source_basis_sha256,
            heldout.source_basis_sha256,
        ),
        (
            "target_basis_sha256",
            transport.target_basis_sha256,
            heldout.target_basis_sha256,
        ),
        ("scope", transport.scope, heldout.scope),
        (
            "score_reduction",
            transport.score_reduction,
            heldout.score_reduction,
        ),
        ("normalizer", transport.normalizer, heldout.normalizer),
        (
            "visibility_window",
            transport.visibility_window,
            heldout.visibility_window,
        ),
    )
    for label, expected, actual in bindings:
        if expected != actual:
            raise ValueError(f"held-out {label} does not match frozen transport")
    if transport.basis_rank != heldout.rank:
        raise ValueError("held-out basis rank does not match frozen transport")
    if transport.basis_max_lag != heldout.max_lag:
        raise ValueError(
            "held-out lag basis does not match frozen transport"
        )

    feature_gram, feature_target, target_gram = heldout.prefix_moments(
        rank=transport.rank,
        max_lag=transport.max_lag,
    )
    feature_gram = feature_gram.to(dtype=torch.float64)
    feature_target = feature_target.to(dtype=torch.float64)
    target_gram = target_gram.to(dtype=torch.float64)
    matrix = transport.matrix.to(dtype=torch.float64)
    prediction_norm_raw = torch.trace(
        matrix.T @ feature_gram @ matrix
    )
    target_norm_raw = torch.trace(target_gram)
    prediction_target_dot = torch.trace(matrix.T @ feature_target).item()
    scale = max(
        abs(prediction_norm_raw.item()),
        abs(target_norm_raw.item()),
        abs(2 * prediction_target_dot),
        1.0,
    )
    prediction_norm = _clamp_nonnegative(
        prediction_norm_raw,
        label="prediction squared norm",
        scale=scale,
        dimension=matrix.numel(),
    )
    target_norm = _clamp_nonnegative(
        target_norm_raw,
        label="target squared norm",
        scale=scale,
        dimension=transport.rank,
    )
    squared_error = _clamp_nonnegative(
        target_norm_raw
        + prediction_norm_raw
        - 2 * torch.tensor(
            prediction_target_dot,
            dtype=torch.float64,
        ),
        label="transport squared error",
        scale=scale,
        dimension=matrix.numel(),
    )
    cauchy_tolerance = (
        2048
        * torch.finfo(torch.float64).eps
        * max(prediction_norm * target_norm, 1.0)
        * max(matrix.numel(), 1)
    )
    if prediction_target_dot**2 > (
        prediction_norm * target_norm + cauchy_tolerance
    ):
        raise ValueError(
            "held-out causal moments violate the prediction-target "
            "Cauchy bound"
        )
    return FrozenCausalModalTransportEvaluation(
        source_layer=transport.source_layer,
        target_layer=transport.target_layer,
        row_kind=transport.row_kind,
        source_width=transport.source_width,
        target_width=transport.target_width,
        rank=transport.rank,
        max_lag=transport.max_lag,
        visibility_window=transport.visibility_window,
        observations=heldout.observations,
        rows_seen=heldout.rows_seen,
        sequences=heldout.sequences,
        lag_pair_counts=heldout.lag_pair_counts[
            : transport.max_lag + 1
        ],
        position_schedule_sha256=heldout.position_schedule_sha256,
        target_baseline_squared_error=target_norm,
        transport_squared_error=squared_error,
        transport_target_dot_sum=prediction_target_dot,
        transport_squared_norm_sum=prediction_norm,
        target_squared_norm_sum=target_norm,
        source_basis_sha256=transport.source_basis_sha256,
        target_basis_sha256=transport.target_basis_sha256,
        accumulation_dtype="float64",
        scope=transport.scope,
        score_reduction=transport.score_reduction,
        normalizer=transport.normalizer,
    )


__all__ = [
    "FrozenCausalModalTransport",
    "FrozenCausalModalTransportEvaluation",
    "StreamingCausalModalTransportEstimator",
    "StreamingCausalModalTransportResult",
    "evaluate_frozen_causal_modal_transport",
    "freeze_causal_modal_transport",
]
