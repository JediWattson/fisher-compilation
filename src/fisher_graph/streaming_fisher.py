"""Bounded-memory low-rank empirical activation-Fisher estimation.

This module consumes rows of activation score gradients.  If ``g_i`` is the
gradient of a per-sequence score (typically summed negative log-likelihood)
with respect to one valid activation position, the width-wise empirical
activation Fisher used by this project is

``F = (1 / N) * sum_i g_i g_i.T``.

Materializing ``F`` costs quadratic memory in the activation width.  The
estimator below instead maintains a deterministic Frequent Directions sketch
``B`` and diagonalizes ``B.T @ B / N`` only through the thin sketch SVD.  Its
stored state is linear in width.  Frequent Directions is a covariance sketch:
the returned eigenvalues are conservative approximations, while the exact
trace of ``F`` is tracked separately from the row squared norms.

The estimator does not compute gradients itself.  Callers retain control over
the score definition and must stream two-dimensional ``[observations, width]``
gradient batches in the same normalization convention they intend to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


_ALGORITHM = "frequent_directions"
_ALGORITHM_VERSION = 1


def _canonicalize_vector_signs(vectors: Tensor) -> Tensor:
    """Choose deterministic signs for the columns of an orthonormal basis."""

    if vectors.numel() == 0:
        return vectors
    pivots = vectors.abs().argmax(dim=0)
    columns = torch.arange(vectors.shape[1])
    signs = vectors[pivots, columns].sign()
    signs[signs == 0] = 1
    return vectors * signs


def _complete_orthonormal_columns(vectors: Tensor, columns: int) -> Tensor:
    """Complete a thin basis without allocating a width-by-width identity."""

    width = vectors.shape[0]
    basis = [vectors[:, index].clone() for index in range(vectors.shape[1])]
    tolerance = torch.finfo(vectors.dtype).eps * max(width, 1) * 16
    for axis in range(width):
        if len(basis) == columns:
            break
        candidate = torch.zeros(width, dtype=vectors.dtype)
        candidate[axis] = 1
        # Reorthogonalize to keep the completion stable for nearly aligned
        # singular vectors.
        for _ in range(2):
            for existing in basis:
                candidate -= torch.dot(existing, candidate) * existing
        norm = torch.linalg.vector_norm(candidate)
        if norm > tolerance:
            basis.append(candidate / norm)
    if len(basis) != columns:
        raise RuntimeError("could not construct an orthonormal basis completion")
    return torch.stack(basis, dim=1)


@dataclass(frozen=True, slots=True)
class StreamingFisherResult:
    """Leading modes from a streaming empirical activation-Fisher sketch.

    ``eigenvalues`` and ``vectors`` describe the rank-limited positive
    semidefinite approximation ``vectors @ diag(eigenvalues) @ vectors.T``.
    ``fisher_trace`` is the exact trace of the unsketched empirical Fisher;
    ``retained_trace`` is the trace represented by the returned modes.
    Tensors are CPU float32 or float64 values so the state dictionary is
    portable and can be written directly with :func:`torch.save`.
    """

    activation_name: str
    eigenvalues: Tensor
    vectors: Tensor
    observations: int
    nonzero_observations: int
    rows_seen: int
    requested_rank: int
    sketch_rows: int
    squared_gradient_norm_sum: float
    fisher_trace: float
    accumulation_dtype: str
    scope: str = "width_pooled"
    score_reduction: str = "sum"
    normalizer: str = "valid_activation_positions"
    algorithm: str = _ALGORITHM
    algorithm_version: int = _ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.activation_name, str) or not self.activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if self.eigenvalues.ndim != 1 or self.eigenvalues.numel() == 0:
            raise ValueError("eigenvalues must be a nonempty vector")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must have shape [width, modes]")
        if self.vectors.shape[1] != self.eigenvalues.numel():
            raise ValueError("vectors and eigenvalues disagree on mode count")
        if self.eigenvalues.device.type != "cpu" or self.vectors.device.type != "cpu":
            raise ValueError("streaming Fisher result tensors must be on CPU")
        if self.eigenvalues.dtype not in (torch.float32, torch.float64):
            raise ValueError("eigenvalues must use float32 or float64")
        if self.vectors.dtype != self.eigenvalues.dtype:
            raise ValueError("vectors and eigenvalues must use the same dtype")
        if not torch.isfinite(self.eigenvalues).all():
            raise ValueError("eigenvalues must be finite")
        if not torch.isfinite(self.vectors).all():
            raise ValueError("vectors must be finite")
        if (self.eigenvalues < 0).any():
            raise ValueError("eigenvalues cannot be negative")
        if type(self.observations) is not int or self.observations <= 0:
            raise ValueError("observations must be positive")
        if (
            type(self.nonzero_observations) is not int
            or not 0 <= self.nonzero_observations <= self.observations
        ):
            raise ValueError("nonzero_observations is out of range")
        if type(self.rows_seen) is not int or self.rows_seen < self.observations:
            raise ValueError("rows_seen cannot be smaller than observations")
        if type(self.requested_rank) is not int or self.requested_rank <= 0:
            raise ValueError("requested_rank must be positive")
        if self.eigenvalues.numel() > self.requested_rank:
            raise ValueError("result has more modes than requested")
        if (
            type(self.sketch_rows) is not int
            or self.sketch_rows <= self.requested_rank
        ):
            raise ValueError("sketch_rows must be greater than requested_rank")
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
        if self.algorithm != _ALGORITHM:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise ValueError(
                f"unsupported algorithm version: {self.algorithm_version}"
            )

    @property
    def width(self) -> int:
        return self.vectors.shape[0]

    @property
    def modes(self) -> int:
        return self.eigenvalues.numel()

    @property
    def retained_trace(self) -> float:
        return self.eigenvalues.sum().item()

    @property
    def retained_trace_fraction(self) -> float:
        """Conservative sketch trace divided by the exact Fisher trace."""

        if self.fisher_trace == 0:
            return 0.0
        return min(self.retained_trace / self.fisher_trace, 1.0)

    @property
    def sketch_retained_trace_fraction(self) -> float:
        """Explicit alias distinguishing sketch energy from Rayleigh energy."""

        return self.retained_trace_fraction

    def approximate_matrix(self) -> Tensor:
        """Materialize the rank-limited Fisher approximation on request."""

        return (self.vectors * self.eigenvalues.unsqueeze(0)) @ self.vectors.T

    def metadata(self) -> dict[str, str | int | float]:
        """Return JSON-serializable metadata without tensor payloads."""

        return {
            "activation_name": self.activation_name,
            "width": self.width,
            "modes": self.modes,
            "observations": self.observations,
            "nonzero_observations": self.nonzero_observations,
            "rows_seen": self.rows_seen,
            "requested_rank": self.requested_rank,
            "sketch_rows": self.sketch_rows,
            "squared_gradient_norm_sum": self.squared_gradient_norm_sum,
            "fisher_trace": self.fisher_trace,
            "retained_trace": self.retained_trace,
            "retained_trace_fraction": self.retained_trace_fraction,
            "retained_trace_semantics": (
                "frequent_directions_sketch_trace_over_exact_trace"
            ),
            "accumulation_dtype": self.accumulation_dtype,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        """Return a serialization-friendly tensor and metadata mapping."""

        return {
            **self.metadata(),
            "eigenvalues": self.eigenvalues,
            "vectors": self.vectors,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StreamingFisherResult:
        """Restore a result produced by :meth:`state_dict`."""

        expected = {
            "activation_name",
            "width",
            "modes",
            "observations",
            "nonzero_observations",
            "rows_seen",
            "requested_rank",
            "sketch_rows",
            "squared_gradient_norm_sum",
            "fisher_trace",
            "retained_trace",
            "retained_trace_fraction",
            "retained_trace_semantics",
            "accumulation_dtype",
            "scope",
            "score_reduction",
            "normalizer",
            "algorithm",
            "algorithm_version",
            "eigenvalues",
            "vectors",
        }
        if set(state) != expected:
            raise ValueError(
                "streaming Fisher result fields do not match format version 1"
            )
        eigenvalues = state["eigenvalues"]
        vectors = state["vectors"]
        if not isinstance(eigenvalues, Tensor) or not isinstance(vectors, Tensor):
            raise TypeError("eigenvalues and vectors must be tensors")
        result = cls(
            activation_name=str(state["activation_name"]),
            eigenvalues=eigenvalues,
            vectors=vectors,
            observations=int(state["observations"]),
            nonzero_observations=int(state["nonzero_observations"]),
            rows_seen=int(state["rows_seen"]),
            requested_rank=int(state["requested_rank"]),
            sketch_rows=int(state["sketch_rows"]),
            squared_gradient_norm_sum=float(
                state["squared_gradient_norm_sum"]
            ),
            fisher_trace=float(state["fisher_trace"]),
            accumulation_dtype=str(state["accumulation_dtype"]),
            scope=str(state["scope"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
            algorithm=str(state["algorithm"]),
            algorithm_version=int(state["algorithm_version"]),
        )
        if int(state["width"]) != result.width:
            raise ValueError("serialized Fisher width does not match vectors")
        if int(state["modes"]) != result.modes:
            raise ValueError("serialized Fisher mode count does not match tensors")
        if (
            str(state["retained_trace_semantics"])
            != "frequent_directions_sketch_trace_over_exact_trace"
        ):
            raise ValueError("unsupported retained Fisher trace semantics")
        retained_trace = float(state["retained_trace"])
        retained_fraction = float(state["retained_trace_fraction"])
        if not torch.isclose(
            torch.tensor(retained_trace, dtype=torch.float64),
            torch.tensor(result.retained_trace, dtype=torch.float64),
        ):
            raise ValueError("serialized retained trace does not match modes")
        if not torch.isclose(
            torch.tensor(retained_fraction, dtype=torch.float64),
            torch.tensor(
                result.retained_trace_fraction,
                dtype=torch.float64,
            ),
        ):
            raise ValueError(
                "serialized retained trace fraction does not match modes"
            )
        return result


class StreamingActivationFisherEstimator:
    """Stream a low-rank empirical activation Fisher with bounded memory.

    The retained Frequent Directions sketch has ``sketch_rows`` rows and uses
    a ``2 * sketch_rows`` merge buffer.  Thus storage is
    ``O(sketch_rows * width)`` and does not grow with observation count.
    ``sketch_rows`` must be larger than the requested output rank; using about
    twice the requested rank is a practical default.

    Input batches may live on any PyTorch device and use any real floating
    dtype.  Selected rows are detached and accumulated on CPU in float64 by
    default so this analysis does not retain model autograd graphs or GPU
    activation history.
    """

    def __init__(
        self,
        *,
        activation_name: str,
        rank: int,
        width: int | None = None,
        sketch_rows: int | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
        score_reduction: str = "sum",
        normalizer: str = "valid_activation_positions",
    ) -> None:
        if not isinstance(activation_name, str) or not activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if width is not None and (type(width) is not int or width <= 0):
            raise ValueError("width must be a positive integer")
        resolved_sketch_rows = sketch_rows if sketch_rows is not None else 2 * rank
        if (
            type(resolved_sketch_rows) is not int
            or resolved_sketch_rows <= rank
        ):
            raise ValueError("sketch_rows must be an integer greater than rank")
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be float32 or float64")
        if not isinstance(score_reduction, str) or not score_reduction:
            raise ValueError("score_reduction must be a nonempty string")
        if not isinstance(normalizer, str) or not normalizer:
            raise ValueError("normalizer must be a nonempty string")

        self.activation_name = activation_name
        self.rank = rank
        self.sketch_rows = resolved_sketch_rows
        self.accumulation_dtype = accumulation_dtype
        self.score_reduction = score_reduction
        self.normalizer = normalizer
        self._width = width
        self._buffer: Tensor | None = None
        self._filled = 0
        self._observations = 0
        self._nonzero_observations = 0
        self._rows_seen = 0
        self._squared_norm_sum = 0.0
        if width is not None:
            self._allocate_buffer(width)

    @property
    def width(self) -> int | None:
        return self._width

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def nonzero_observations(self) -> int:
        return self._nonzero_observations

    @property
    def rows_seen(self) -> int:
        return self._rows_seen

    @property
    def storage_shape(self) -> tuple[int, int] | None:
        if self._buffer is None:
            return None
        return tuple(self._buffer.shape)

    def _allocate_buffer(self, width: int) -> None:
        self._buffer = torch.zeros(
            (2 * self.sketch_rows, width),
            dtype=self.accumulation_dtype,
            device="cpu",
        )

    def _compress(self) -> None:
        assert self._buffer is not None
        if self._filled < self.sketch_rows:
            return
        _, singular_values, right_vectors = torch.linalg.svd(
            self._buffer[: self._filled],
            full_matrices=False,
        )
        if singular_values.numel() < self.sketch_rows:
            shrunk = singular_values
            retained_vectors = right_vectors
        else:
            delta = singular_values[self.sketch_rows - 1].square()
            shrunk = (
                singular_values[: self.sketch_rows].square() - delta
            ).clamp_min(0).sqrt()
            retained_vectors = right_vectors[: self.sketch_rows]

        positive = int(torch.count_nonzero(shrunk > 0).item())
        self._buffer.zero_()
        if positive:
            self._buffer[:positive].copy_(
                shrunk[:positive].unsqueeze(1)
                * retained_vectors[:positive]
            )
        self._filled = positive

    def _append_rows(self, rows: Tensor) -> None:
        assert self._buffer is not None
        cursor = 0
        while cursor < rows.shape[0]:
            available = self._buffer.shape[0] - self._filled
            count = min(available, rows.shape[0] - cursor)
            self._buffer[self._filled : self._filled + count].copy_(
                rows[cursor : cursor + count]
            )
            self._filled += count
            cursor += count
            if self._filled == self._buffer.shape[0]:
                self._compress()

    def update(
        self,
        score_vectors: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> StreamingActivationFisherEstimator:
        """Add a ``[observations, width]`` batch of score-gradient rows.

        ``mask`` is an optional one-dimensional boolean tensor.  Masked-out
        rows are ignored completely, including for normalization.  Selected
        zero rows do count as observations because they are valid zero-gradient
        Fisher samples, but they consume no sketch capacity.
        """

        if not isinstance(score_vectors, Tensor):
            raise TypeError("score_vectors must be a Tensor")
        if score_vectors.ndim != 2:
            raise ValueError(
                "score_vectors must have shape [observations, width]"
            )
        if not score_vectors.is_floating_point():
            raise ValueError("score_vectors must use a real floating dtype")
        rows, width = score_vectors.shape
        if width <= 0:
            raise ValueError("score vector width must be positive")
        if self._width is not None and width != self._width:
            raise ValueError(
                f"expected score vector width {self._width}, got {width}"
            )

        if mask is not None:
            if not isinstance(mask, Tensor):
                raise TypeError("mask must be a Tensor")
            if mask.shape != (rows,):
                raise ValueError("mask must have shape [observations]")
            if mask.dtype != torch.bool:
                raise ValueError("mask must use boolean dtype")
            selected = score_vectors[
                mask.to(device=score_vectors.device)
            ]
        else:
            selected = score_vectors
        selected = selected.detach().to(
            device="cpu",
            dtype=self.accumulation_dtype,
        )
        if not torch.isfinite(selected).all():
            raise ValueError("selected score vectors must be finite")

        # Mutate the estimator only after the complete input batch has passed
        # validation, so a rejected update cannot change its accounting.
        if self._width is None:
            self._width = width
            self._allocate_buffer(width)
        self._rows_seen += rows
        if selected.shape[0] == 0:
            return self

        squared_norms = selected.square().sum(dim=1)
        nonzero = squared_norms > 0
        self._observations += selected.shape[0]
        self._nonzero_observations += int(nonzero.sum().item())
        self._squared_norm_sum += squared_norms.sum().item()
        if nonzero.any():
            self._append_rows(selected[nonzero])
        return self

    def finalize(self) -> StreamingFisherResult:
        """Return the current leading Fisher modes without mutating the sketch."""

        if self._observations == 0:
            raise ValueError("cannot finalize without any selected observations")
        assert self._width is not None
        assert self._buffer is not None
        modes = min(self.rank, self._width)

        if self._filled:
            _, singular_values, right_vectors = torch.linalg.svd(
                self._buffer[: self._filled],
                full_matrices=False,
            )
            available = min(modes, singular_values.numel())
            eigenvalues = (
                singular_values[:available].square() / self._observations
            )
            vectors = right_vectors[:available].T.contiguous()
        else:
            available = 0
            eigenvalues = torch.empty(
                0,
                dtype=self.accumulation_dtype,
            )
            vectors = torch.empty(
                (self._width, 0),
                dtype=self.accumulation_dtype,
            )

        if available < modes:
            vectors = _complete_orthonormal_columns(vectors, modes)
            eigenvalues = torch.cat(
                (
                    eigenvalues,
                    torch.zeros(
                        modes - available,
                        dtype=self.accumulation_dtype,
                    ),
                )
            )
        vectors = _canonicalize_vector_signs(vectors)
        return StreamingFisherResult(
            activation_name=self.activation_name,
            eigenvalues=eigenvalues.clamp_min(0).cpu(),
            vectors=vectors.cpu(),
            observations=self._observations,
            nonzero_observations=self._nonzero_observations,
            rows_seen=self._rows_seen,
            requested_rank=self.rank,
            sketch_rows=self.sketch_rows,
            squared_gradient_norm_sum=self._squared_norm_sum,
            fisher_trace=self._squared_norm_sum / self._observations,
            accumulation_dtype=str(self.accumulation_dtype).removeprefix(
                "torch."
            ),
            score_reduction=self.score_reduction,
            normalizer=self.normalizer,
        )
