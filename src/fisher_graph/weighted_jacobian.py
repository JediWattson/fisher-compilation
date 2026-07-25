"""Causal low-rank factors of activation- and Fisher-weighted Jacobians.

This module is a deliberately small reference implementation for a constant
modal graph.  It does not average positions into one global operator.  For
every output position ``t`` it forms the independent prefix operator

```
M_t = F_t^(1/2) [J_(t,0) C_0^(1/2) ... J_(t,t) C_t^(1/2)]
```

and factors ``M_t`` with its own SVD.  Consequently, neither the stored factor
nor the executor has a slot through which output ``t`` could read an input
position later than ``t``.

The input covariance is block-local by source position: v1 deliberately omits
cross-position covariance blocks.  Besides keeping the reference
implementation compact and scalable, this makes weighted energy attributable
to individual causal position edges.  A future full-prefix metric can extend
the covariance model without changing the causal prefix boundary.  The factors
are computed in float64 and use support pseudoinverses for singular covariance
or Fisher matrices.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


_ALGORITHM = "causal_prefix_weighted_jacobian_svd"
_ALGORITHM_VERSION = 1
_FORMAT_VERSION = 1


def _cpu_float64(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _close(
    actual: Tensor,
    expected: Tensor,
    *,
    scale: float = 1.0,
) -> bool:
    return torch.allclose(
        actual,
        expected,
        rtol=2e-10,
        atol=2e-11 * max(scale, 1.0),
    )


def _validate_relative_cutoff(value: float, *, label: str) -> None:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{label} must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class PSDSupportFactor:
    """A symmetric PSD square root and inverse restricted to its support.

    All tensors are detached CPU float64 values.  ``inverse_square_root`` is
    zero on unsupported eigendirections, and its product with ``square_root``
    is therefore the orthogonal ``support_projector`` rather than necessarily
    the identity.
    """

    matrix: Tensor
    square_root: Tensor
    inverse_square_root: Tensor
    support_projector: Tensor
    eigenvalues: Tensor
    support_rank: int
    cutoff: float

    def __post_init__(self) -> None:
        tensors = {
            "matrix": self.matrix,
            "square_root": self.square_root,
            "inverse_square_root": self.inverse_square_root,
            "support_projector": self.support_projector,
            "eigenvalues": self.eigenvalues,
        }
        for label, tensor in tensors.items():
            if (
                not isinstance(tensor, Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float64
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    f"{label} must be a finite CPU float64 Tensor"
                )
            object.__setattr__(self, label, tensor.detach().clone())

        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("PSD matrix must be square")
        width = self.matrix.shape[0]
        if width == 0:
            raise ValueError("PSD matrix cannot be empty")
        if self.eigenvalues.shape != (width,):
            raise ValueError("PSD eigenvalues must have shape [width]")
        for label in (
            "square_root",
            "inverse_square_root",
            "support_projector",
        ):
            if getattr(self, label).shape != (width, width):
                raise ValueError(f"{label} must have shape [width, width]")
        if type(self.support_rank) is not int or not (
            0 <= self.support_rank <= width
        ):
            raise ValueError("support_rank is outside the PSD width")
        if (
            not isinstance(self.cutoff, float)
            or not math.isfinite(self.cutoff)
            or self.cutoff < 0.0
        ):
            raise ValueError("PSD cutoff must be finite and nonnegative")

        scale = max(float(self.matrix.abs().max().item()), 1.0)
        for label in (
            "matrix",
            "square_root",
            "inverse_square_root",
            "support_projector",
        ):
            tensor = getattr(self, label)
            if not _close(tensor, tensor.T, scale=scale):
                raise ValueError(f"{label} must be symmetric")
        if (self.eigenvalues < 0).any():
            raise ValueError("PSD eigenvalues cannot be negative")
        if self.eigenvalues.numel() > 1 and (
            self.eigenvalues[1:] < self.eigenvalues[:-1]
        ).any():
            raise ValueError("PSD eigenvalues must be sorted ascending")
        supported = self.eigenvalues > self.cutoff
        if int(supported.sum().item()) != self.support_rank:
            raise ValueError("support_rank does not match the PSD cutoff")

        identity_on_support = (
            self.square_root @ self.inverse_square_root
        )
        if not _close(
            self.square_root @ self.square_root,
            self.matrix,
            scale=scale,
        ):
            raise ValueError("PSD square root does not reconstruct the matrix")
        if not _close(
            identity_on_support,
            self.support_projector,
            scale=1.0,
        ):
            raise ValueError(
                "PSD inverse square root does not match the support"
            )
        if not _close(
            self.support_projector @ self.support_projector,
            self.support_projector,
            scale=1.0,
        ):
            raise ValueError("support_projector must be idempotent")


def factor_psd_support(
    matrix: Tensor,
    *,
    relative_cutoff: float = 1e-12,
) -> PSDSupportFactor:
    """Factor a floating symmetric PSD matrix in CPU float64.

    Small negative eigenvalues are tolerated only relative to a positive
    spectral scale.  A materially indefinite or asymmetric matrix is rejected
    rather than silently repaired.
    """

    _validate_relative_cutoff(
        relative_cutoff,
        label="relative_cutoff",
    )
    value = _cpu_float64(matrix, label="matrix")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    if value.shape[0] == 0:
        raise ValueError("matrix cannot be empty")

    magnitude = float(value.abs().max().item())
    symmetry_tolerance = (
        max(magnitude, torch.finfo(torch.float64).tiny)
        * torch.finfo(torch.float64).eps
        * max(value.shape[0], 1)
        * 128
    )
    if float((value - value.T).abs().max().item()) > symmetry_tolerance:
        raise ValueError("matrix must be symmetric")
    symmetric = (value + value.T) / 2
    raw_eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    positive_scale = max(float(raw_eigenvalues.max().item()), 0.0)
    spectral_scale = max(
        positive_scale,
        float(raw_eigenvalues.abs().max().item()),
    )
    numerical_relative = (
        torch.finfo(torch.float64).eps * max(value.shape[0], 1) * 128
    )
    negative_tolerance = spectral_scale * max(
        relative_cutoff,
        numerical_relative,
    )
    if float(raw_eigenvalues.min().item()) < -negative_tolerance:
        raise ValueError("matrix must be positive semidefinite")

    eigenvalues = raw_eigenvalues.clamp_min(0.0)
    cutoff = positive_scale * max(relative_cutoff, numerical_relative)
    supported = eigenvalues > cutoff
    roots = eigenvalues.sqrt()
    inverse_roots = torch.zeros_like(roots)
    inverse_roots[supported] = roots[supported].reciprocal()
    square_root = (eigenvectors * roots.unsqueeze(0)) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * inverse_roots.unsqueeze(0)
    ) @ eigenvectors.T
    supported_vectors = eigenvectors[:, supported]
    projector = supported_vectors @ supported_vectors.T
    reconstructed = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
    return PSDSupportFactor(
        matrix=reconstructed,
        square_root=square_root,
        inverse_square_root=inverse_square_root,
        support_projector=projector,
        eigenvalues=eigenvalues,
        support_rank=int(supported.sum().item()),
        cutoff=float(cutoff),
    )


@dataclass(frozen=True, slots=True)
class CausalWeightedJacobianFactor:
    """One signed, low-rank output-position prefix factor.

    Axes:

    - ``weighted_left_vectors``: ``[output_width, spectrum_rank]``
    - ``weighted_right_vectors``:
      ``[prefix_length * input_width, spectrum_rank]``
    - ``input_inverse_square_root``:
      ``[prefix_length, input_width, input_width]``
    - ``output_inverse_square_root``: ``[output_width, output_width]``
    - ``input_factor``: ``[retained_rank, prefix_length, input_width]``
    - ``output_factor``: ``[output_width, retained_rank]``
    - ``source_edge_energy``: ``[prefix_length]``

    The executor computes ``output_factor @ input_factor @ centered_prefix``.
    Both execution factors are raw signed factors; RMS magnitudes are not used
    as executable edge weights.
    """

    output_position: int
    input_width: int
    output_width: int
    retained_rank: int
    singular_values: Tensor
    weighted_left_vectors: Tensor
    weighted_right_vectors: Tensor
    input_inverse_square_root: Tensor
    output_inverse_square_root: Tensor
    input_factor: Tensor
    output_factor: Tensor
    source_edge_energy: Tensor
    singular_tolerance: float
    input_support_ranks: tuple[int, ...]
    output_support_rank: int

    def __post_init__(self) -> None:
        for label, value in (
            ("output_position", self.output_position),
            ("input_width", self.input_width),
            ("output_width", self.output_width),
            ("retained_rank", self.retained_rank),
            ("output_support_rank", self.output_support_rank),
        ):
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
        if self.output_position < 0:
            raise ValueError("output_position must be nonnegative")
        if self.input_width <= 0 or self.output_width <= 0:
            raise ValueError("factor widths must be positive")
        prefix_length = self.output_position + 1
        spectrum_rank = min(
            self.output_width,
            prefix_length * self.input_width,
        )
        if not 0 <= self.retained_rank <= spectrum_rank:
            raise ValueError("retained_rank is outside the prefix spectrum")
        if not 0 <= self.output_support_rank <= self.output_width:
            raise ValueError("output_support_rank is outside output width")
        if (
            not isinstance(self.input_support_ranks, tuple)
            or len(self.input_support_ranks) != prefix_length
            or any(
                type(rank) is not int or not 0 <= rank <= self.input_width
                for rank in self.input_support_ranks
            )
        ):
            raise ValueError(
                "input_support_ranks must match the causal prefix"
            )
        if (
            not isinstance(self.singular_tolerance, float)
            or not math.isfinite(self.singular_tolerance)
            or self.singular_tolerance < 0.0
        ):
            raise ValueError(
                "singular_tolerance must be finite and nonnegative"
            )

        expected_shapes = {
            "singular_values": (spectrum_rank,),
            "weighted_left_vectors": (
                self.output_width,
                spectrum_rank,
            ),
            "weighted_right_vectors": (
                prefix_length * self.input_width,
                spectrum_rank,
            ),
            "input_inverse_square_root": (
                prefix_length,
                self.input_width,
                self.input_width,
            ),
            "output_inverse_square_root": (
                self.output_width,
                self.output_width,
            ),
            "input_factor": (
                self.retained_rank,
                prefix_length,
                self.input_width,
            ),
            "output_factor": (
                self.output_width,
                self.retained_rank,
            ),
            "source_edge_energy": (prefix_length,),
        }
        for label, expected_shape in expected_shapes.items():
            tensor = getattr(self, label)
            if (
                not isinstance(tensor, Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float64
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    f"{label} must be a finite CPU float64 Tensor"
                )
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{label} must have shape {expected_shape}"
                )
            object.__setattr__(self, label, tensor.detach().clone())

        spectral_scale = max(
            float(self.singular_values.abs().max().item())
            if self.singular_values.numel()
            else 0.0,
            1.0,
        )
        if (self.singular_values < 0).any():
            raise ValueError("singular_values cannot be negative")
        if self.singular_values.numel() > 1 and (
            self.singular_values[1:] > self.singular_values[:-1]
        ).any():
            raise ValueError("singular_values must be sorted descending")
        if (self.source_edge_energy < 0).any():
            raise ValueError("source_edge_energy cannot be negative")

        identity = torch.eye(spectrum_rank, dtype=torch.float64)
        if not _close(
            self.weighted_left_vectors.T
            @ self.weighted_left_vectors,
            identity,
            scale=1.0,
        ):
            raise ValueError("weighted left vectors must be orthonormal")
        if not _close(
            self.weighted_right_vectors.T
            @ self.weighted_right_vectors,
            identity,
            scale=1.0,
        ):
            raise ValueError("weighted right vectors must be orthonormal")
        for label in (
            "output_inverse_square_root",
        ):
            tensor = getattr(self, label)
            if not _close(tensor, tensor.T, scale=float(tensor.abs().max())):
                raise ValueError(f"{label} must be symmetric")
        for position, tensor in enumerate(self.input_inverse_square_root):
            if not _close(
                tensor,
                tensor.T,
                scale=float(tensor.abs().max()),
            ):
                raise ValueError(
                    "input_inverse_square_root blocks must be symmetric "
                    f"(source position {position})"
                )

        weighted_matrix = (
            self.weighted_left_vectors
            * self.singular_values.unsqueeze(0)
        ) @ self.weighted_right_vectors.T
        expected_edge_energy = (
            weighted_matrix.reshape(
                self.output_width,
                prefix_length,
                self.input_width,
            )
            .square()
            .sum(dim=(0, 2))
        )
        if not _close(
            self.source_edge_energy,
            expected_edge_energy,
            scale=spectral_scale**2,
        ):
            raise ValueError(
                "source_edge_energy does not account for weighted edges"
            )
        if not _close(
            self.source_edge_energy.sum(),
            self.singular_values.square().sum(),
            scale=spectral_scale**2,
        ):
            raise ValueError(
                "edge energy does not equal singular-value energy"
            )

        root = self.singular_values[: self.retained_rank].sqrt()
        expected_output = self.output_inverse_square_root @ (
            self.weighted_left_vectors[:, : self.retained_rank]
            * root.unsqueeze(0)
        )
        block_inverse = torch.block_diag(
            *tuple(self.input_inverse_square_root)
        )
        expected_input = (
            root.unsqueeze(1)
            * self.weighted_right_vectors[
                :, : self.retained_rank
            ].T
        ) @ block_inverse
        expected_input = expected_input.reshape(
            self.retained_rank,
            prefix_length,
            self.input_width,
        )
        execution_scale = max(
            float(expected_output.abs().max().item())
            if expected_output.numel()
            else 0.0,
            float(expected_input.abs().max().item())
            if expected_input.numel()
            else 0.0,
            1.0,
        )
        if not _close(
            self.output_factor,
            expected_output,
            scale=execution_scale,
        ) or not _close(
            self.input_factor,
            expected_input,
            scale=execution_scale,
        ):
            raise ValueError(
                "execution factors do not match the weighted SVD"
            )

    @property
    def prefix_length(self) -> int:
        return self.output_position + 1

    @property
    def spectrum_rank(self) -> int:
        return self.singular_values.numel()

    @property
    def effective_rank(self) -> int:
        return int(
            (self.singular_values > self.singular_tolerance).sum().item()
        )

    @property
    def total_weighted_energy(self) -> float:
        return float(self.singular_values.square().sum().item())

    @property
    def retained_weighted_energy(self) -> float:
        return float(
            self.singular_values[: self.retained_rank]
            .square()
            .sum()
            .item()
        )

    @property
    def discarded_weighted_energy(self) -> float:
        return float(
            self.singular_values[self.retained_rank :]
            .square()
            .sum()
            .item()
        )

    @property
    def weighted_energy_curve(self) -> Tensor:
        """Cumulative weighted energy for ranks ``0..spectrum_rank``."""

        return torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                self.singular_values.square().cumsum(dim=0),
            )
        )

    @property
    def weighted_tail_curve(self) -> Tensor:
        """Optimal weighted reconstruction error for every retained rank."""

        return self.total_weighted_energy - self.weighted_energy_curve

    def reconstructed_jacobian(self) -> Tensor:
        """Return signed blocks ``[output, prefix, input]``."""

        return torch.einsum(
            "or,rsi->osi",
            self.output_factor,
            self.input_factor,
        )

    @property
    def retained_source_edge_energy(self) -> Tensor:
        """Weighted energy of the retained approximation by source edge."""

        retained_weighted = (
            self.weighted_left_vectors[:, : self.retained_rank]
            * self.singular_values[: self.retained_rank].unsqueeze(0)
        ) @ self.weighted_right_vectors[:, : self.retained_rank].T
        return (
            retained_weighted.reshape(
                self.output_width,
                self.prefix_length,
                self.input_width,
            )
            .square()
            .sum(dim=(0, 2))
        )

    def truncate(self, retained_rank: int) -> CausalWeightedJacobianFactor:
        if type(retained_rank) is not int or not (
            0 <= retained_rank <= self.spectrum_rank
        ):
            raise ValueError("retained_rank is outside the prefix spectrum")
        root = self.singular_values[:retained_rank].sqrt()
        output_factor = self.output_inverse_square_root @ (
            self.weighted_left_vectors[:, :retained_rank]
            * root.unsqueeze(0)
        )
        block_inverse = torch.block_diag(
            *tuple(self.input_inverse_square_root)
        )
        input_factor = (
            root.unsqueeze(1)
            * self.weighted_right_vectors[:, :retained_rank].T
        ) @ block_inverse
        return CausalWeightedJacobianFactor(
            output_position=self.output_position,
            input_width=self.input_width,
            output_width=self.output_width,
            retained_rank=retained_rank,
            singular_values=self.singular_values,
            weighted_left_vectors=self.weighted_left_vectors,
            weighted_right_vectors=self.weighted_right_vectors,
            input_inverse_square_root=self.input_inverse_square_root,
            output_inverse_square_root=self.output_inverse_square_root,
            input_factor=input_factor.reshape(
                retained_rank,
                self.prefix_length,
                self.input_width,
            ),
            output_factor=output_factor,
            source_edge_energy=self.source_edge_energy,
            singular_tolerance=self.singular_tolerance,
            input_support_ranks=self.input_support_ranks,
            output_support_rank=self.output_support_rank,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "output_position": self.output_position,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "retained_rank": self.retained_rank,
            "singular_values": self.singular_values.clone(),
            "weighted_left_vectors": self.weighted_left_vectors.clone(),
            "weighted_right_vectors": self.weighted_right_vectors.clone(),
            "input_inverse_square_root": (
                self.input_inverse_square_root.clone()
            ),
            "output_inverse_square_root": (
                self.output_inverse_square_root.clone()
            ),
            "input_factor": self.input_factor.clone(),
            "output_factor": self.output_factor.clone(),
            "source_edge_energy": self.source_edge_energy.clone(),
            "singular_tolerance": self.singular_tolerance,
            "input_support_ranks": list(self.input_support_ranks),
            "output_support_rank": self.output_support_rank,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CausalWeightedJacobianFactor:
        expected = {
            "output_position",
            "input_width",
            "output_width",
            "retained_rank",
            "singular_values",
            "weighted_left_vectors",
            "weighted_right_vectors",
            "input_inverse_square_root",
            "output_inverse_square_root",
            "input_factor",
            "output_factor",
            "source_edge_energy",
            "singular_tolerance",
            "input_support_ranks",
            "output_support_rank",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("weighted Jacobian factor fields are invalid")
        tensor_fields = {
            "singular_values",
            "weighted_left_vectors",
            "weighted_right_vectors",
            "input_inverse_square_root",
            "output_inverse_square_root",
            "input_factor",
            "output_factor",
            "source_edge_energy",
        }
        if any(not isinstance(state[name], Tensor) for name in tensor_fields):
            raise TypeError("weighted Jacobian factor arrays must be Tensors")
        raw_support = state["input_support_ranks"]
        if not isinstance(raw_support, list) or any(
            type(rank) is not int for rank in raw_support
        ):
            raise TypeError("input_support_ranks must be a list of integers")
        integer_fields = {
            "output_position",
            "input_width",
            "output_width",
            "retained_rank",
            "output_support_rank",
        }
        if any(type(state[name]) is not int for name in integer_fields):
            raise TypeError("weighted Jacobian factor dimensions must be integers")
        if type(state["singular_tolerance"]) is not float:
            raise TypeError("singular_tolerance must be a float")
        return cls(
            output_position=state["output_position"],
            input_width=state["input_width"],
            output_width=state["output_width"],
            retained_rank=state["retained_rank"],
            singular_values=state["singular_values"],
            weighted_left_vectors=state["weighted_left_vectors"],
            weighted_right_vectors=state["weighted_right_vectors"],
            input_inverse_square_root=state[
                "input_inverse_square_root"
            ],
            output_inverse_square_root=state[
                "output_inverse_square_root"
            ],
            input_factor=state["input_factor"],
            output_factor=state["output_factor"],
            source_edge_energy=state["source_edge_energy"],
            singular_tolerance=state["singular_tolerance"],
            input_support_ranks=tuple(raw_support),
            output_support_rank=state["output_support_rank"],
        )


@dataclass(frozen=True, slots=True)
class CausalWeightedJacobianResult:
    """All independent prefix factors for one constant causal Jacobian."""

    input_mean: Tensor
    output_mean: Tensor
    factors: tuple[CausalWeightedJacobianFactor, ...]
    relative_eigenvalue_cutoff: float
    relative_singular_value_cutoff: float
    algorithm: str = _ALGORITHM
    algorithm_version: int = _ALGORITHM_VERSION

    def __post_init__(self) -> None:
        _validate_relative_cutoff(
            self.relative_eigenvalue_cutoff,
            label="relative_eigenvalue_cutoff",
        )
        _validate_relative_cutoff(
            self.relative_singular_value_cutoff,
            label="relative_singular_value_cutoff",
        )
        if self.algorithm != _ALGORITHM:
            raise ValueError("unsupported weighted Jacobian algorithm")
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise ValueError("unsupported weighted Jacobian algorithm version")
        if not isinstance(self.factors, tuple) or not self.factors:
            raise ValueError("factors must be a nonempty tuple")
        if any(
            not isinstance(factor, CausalWeightedJacobianFactor)
            for factor in self.factors
        ):
            raise TypeError("all factors must be causal weighted factors")
        first = self.factors[0]
        expected_positions = tuple(range(len(self.factors)))
        if tuple(
            factor.output_position for factor in self.factors
        ) != expected_positions:
            raise ValueError(
                "factors must contain each output position in causal order"
            )
        if any(
            factor.input_width != first.input_width
            or factor.output_width != first.output_width
            for factor in self.factors
        ):
            raise ValueError("all factors must share input and output widths")

        input_mean = _cpu_float64(self.input_mean, label="input_mean")
        output_mean = _cpu_float64(self.output_mean, label="output_mean")
        expected_input = (len(self.factors), first.input_width)
        expected_output = (len(self.factors), first.output_width)
        if input_mean.shape != expected_input:
            raise ValueError(f"input_mean must have shape {expected_input}")
        if output_mean.shape != expected_output:
            raise ValueError(f"output_mean must have shape {expected_output}")
        object.__setattr__(self, "input_mean", input_mean)
        object.__setattr__(self, "output_mean", output_mean)

    @property
    def sequence_length(self) -> int:
        return len(self.factors)

    @property
    def input_width(self) -> int:
        return self.factors[0].input_width

    @property
    def output_width(self) -> int:
        return self.factors[0].output_width

    @property
    def retained_ranks(self) -> tuple[int, ...]:
        return tuple(factor.retained_rank for factor in self.factors)

    @property
    def total_weighted_energy(self) -> float:
        return sum(
            factor.total_weighted_energy for factor in self.factors
        )

    @property
    def retained_weighted_energy(self) -> float:
        return sum(
            factor.retained_weighted_energy for factor in self.factors
        )

    @property
    def discarded_weighted_energy(self) -> float:
        return sum(
            factor.discarded_weighted_energy for factor in self.factors
        )

    @property
    def dense_causal_coefficient_count(self) -> int:
        """Signed coefficients in the unfactored lower-triangular Jacobian."""

        causal_position_pairs = (
            self.sequence_length * (self.sequence_length + 1) // 2
        )
        return (
            causal_position_pairs
            * self.input_width
            * self.output_width
        )

    @property
    def dense_causal_mac_count(self) -> int:
        """Reference MACs to apply the dense causal Jacobian once."""

        return self.dense_causal_coefficient_count

    @property
    def factor_coefficient_count(self) -> int:
        """Executable signed factor coefficients, excluding affine means.

        For output position ``t`` and retained rank ``r_t``, the input factor
        stores ``r_t * (t + 1) * input_width`` coefficients and the output
        factor stores ``output_width * r_t`` coefficients.  SVD diagnostics,
        PSD audit tensors, singular values, and affine means are deliberately
        excluded because they are not signed execution edges.
        """

        return sum(
            factor.retained_rank
            * (
                factor.prefix_length * self.input_width
                + self.output_width
            )
            for factor in self.factors
        )

    @property
    def factor_mac_count(self) -> int:
        """Reference MACs to apply both signed factors once."""

        return self.factor_coefficient_count

    @property
    def input_mean_coefficient_count(self) -> int:
        """Stored affine input-center values, outside factor accounting."""

        return self.input_mean.numel()

    @property
    def output_mean_coefficient_count(self) -> int:
        """Stored affine output-center values, outside factor accounting."""

        return self.output_mean.numel()

    @property
    def affine_mean_coefficient_count(self) -> int:
        """Total separately stored affine state."""

        return (
            self.input_mean_coefficient_count
            + self.output_mean_coefficient_count
        )

    @property
    def compression_ratio(self) -> float:
        """Executable factor coefficients divided by dense coefficients.

        This is a compressed-to-dense ratio, so smaller is better and a
        rank-zero executor has ratio zero.
        """

        return (
            self.factor_coefficient_count
            / self.dense_causal_coefficient_count
        )

    @property
    def mac_ratio(self) -> float:
        """Factored execution MACs divided by dense causal execution MACs."""

        return self.factor_mac_count / self.dense_causal_mac_count

    @property
    def factor_to_dense_coefficient_ratio(self) -> float:
        """Unambiguous alias for :attr:`compression_ratio`."""

        return self.compression_ratio

    @property
    def factor_to_dense_mac_ratio(self) -> float:
        """Unambiguous alias for :attr:`mac_ratio`."""

        return self.mac_ratio

    @property
    def edge_weighted_energy(self) -> Tensor:
        """Lower-triangular ``[output_position, source_position]`` energy."""

        energy = torch.zeros(
            self.sequence_length,
            self.sequence_length,
            dtype=torch.float64,
        )
        for factor in self.factors:
            energy[
                factor.output_position,
                : factor.prefix_length,
            ] = factor.source_edge_energy
        return energy

    @property
    def retained_edge_weighted_energy(self) -> Tensor:
        """Lower-triangular retained-approximation energy by position edge."""

        energy = torch.zeros(
            self.sequence_length,
            self.sequence_length,
            dtype=torch.float64,
        )
        for factor in self.factors:
            energy[
                factor.output_position,
                : factor.prefix_length,
            ] = factor.retained_source_edge_energy
        return energy

    @staticmethod
    def _energy_by_lag(edge_energy: Tensor) -> Tensor:
        sequence_length = edge_energy.shape[0]
        return torch.stack(
            tuple(
                torch.diagonal(
                    edge_energy,
                    offset=-lag,
                ).sum()
                for lag in range(sequence_length)
            )
        )

    @property
    def weighted_energy_by_lag(self) -> Tensor:
        """Full synthetic-reference energy for logical lags ``0..T-1``."""

        return self._energy_by_lag(self.edge_weighted_energy)

    @property
    def retained_weighted_energy_by_lag(self) -> Tensor:
        """Retained approximation energy for logical lags ``0..T-1``."""

        return self._energy_by_lag(self.retained_edge_weighted_energy)

    @property
    def weighted_energy_fraction_by_lag(self) -> Tensor:
        """Each lag's share of full synthetic-reference weighted energy."""

        energy = self.weighted_energy_by_lag
        total = energy.sum()
        if total.item() == 0.0:
            return torch.zeros_like(energy)
        return energy / total

    @property
    def retained_weighted_energy_fraction_by_lag(self) -> Tensor:
        """Each lag's share of retained-approximation weighted energy."""

        energy = self.retained_weighted_energy_by_lag
        total = energy.sum()
        if total.item() == 0.0:
            return torch.zeros_like(energy)
        return energy / total

    @property
    def weighted_energy_curve(self) -> Tensor:
        """Total energy kept by a uniform per-prefix rank cap."""

        maximum = max(factor.spectrum_rank for factor in self.factors)
        curve = torch.zeros(maximum + 1, dtype=torch.float64)
        for rank in range(maximum + 1):
            curve[rank] = sum(
                float(
                    factor.singular_values[
                        : min(rank, factor.spectrum_rank)
                    ]
                    .square()
                    .sum()
                    .item()
                )
                for factor in self.factors
            )
        return curve

    @property
    def weighted_tail_curve(self) -> Tensor:
        return self.total_weighted_energy - self.weighted_energy_curve

    def ranks_for_prefix_energy_fraction(
        self,
        fraction: float,
    ) -> tuple[int, ...]:
        """Return each prefix's minimal rank retaining ``fraction`` of energy.

        The threshold is applied independently to every output-position
        prefix, rather than to one global spectrum.  A prefix whose total
        weighted energy is exactly zero requires rank zero for every positive
        requested fraction.
        """

        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("fraction must lie in (0, 1]")
        resolved_fraction = float(fraction)
        ranks = []
        for factor in self.factors:
            curve = factor.weighted_energy_curve
            total = curve[-1]
            if total.item() == 0.0:
                ranks.append(0)
                continue
            threshold = total * resolved_fraction
            # Search the rank-1..rank-q curve. ``searchsorted`` returns the
            # first satisfying entry, which makes the selected rank minimal.
            zero_based = int(
                torch.searchsorted(
                    curve[1:],
                    threshold,
                    right=False,
                ).item()
            )
            ranks.append(min(zero_based + 1, factor.spectrum_rank))
        return tuple(ranks)

    def truncate_for_prefix_energy_fraction(
        self,
        fraction: float,
    ) -> CausalWeightedJacobianResult:
        """Truncate to the minimal independent prefix-energy ranks."""

        return self.truncate(
            self.ranks_for_prefix_energy_fraction(fraction)
        )

    def truncate(
        self,
        retained_ranks: int | Sequence[int],
    ) -> CausalWeightedJacobianResult:
        ranks = _resolve_retained_ranks(
            retained_ranks,
            spectrum_ranks=tuple(
                factor.spectrum_rank for factor in self.factors
            ),
        )
        return CausalWeightedJacobianResult(
            input_mean=self.input_mean,
            output_mean=self.output_mean,
            factors=tuple(
                factor.truncate(rank)
                for factor, rank in zip(self.factors, ranks, strict=True)
            ),
            relative_eigenvalue_cutoff=self.relative_eigenvalue_cutoff,
            relative_singular_value_cutoff=(
                self.relative_singular_value_cutoff
            ),
        )

    def executor(self) -> CausalWeightedJacobianExecutor:
        return CausalWeightedJacobianExecutor(self)

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "relative_eigenvalue_cutoff": (
                self.relative_eigenvalue_cutoff
            ),
            "relative_singular_value_cutoff": (
                self.relative_singular_value_cutoff
            ),
            "input_mean": self.input_mean.clone(),
            "output_mean": self.output_mean.clone(),
            "factors": [factor.state_dict() for factor in self.factors],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CausalWeightedJacobianResult:
        expected = {
            "format_version",
            "algorithm",
            "algorithm_version",
            "relative_eigenvalue_cutoff",
            "relative_singular_value_cutoff",
            "input_mean",
            "output_mean",
            "factors",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("weighted Jacobian result fields are invalid")
        if state["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported weighted Jacobian result format")
        if not isinstance(state["input_mean"], Tensor) or not isinstance(
            state["output_mean"], Tensor
        ):
            raise TypeError("weighted Jacobian means must be Tensors")
        raw_factors = state["factors"]
        if not isinstance(raw_factors, list):
            raise TypeError("weighted Jacobian factors must be a list")
        factors = []
        for raw_factor in raw_factors:
            if not isinstance(raw_factor, Mapping):
                raise TypeError("weighted Jacobian factor must be a mapping")
            factors.append(
                CausalWeightedJacobianFactor.from_state_dict(raw_factor)
            )
        if type(state["algorithm"]) is not str:
            raise TypeError("weighted Jacobian algorithm must be a string")
        if type(state["algorithm_version"]) is not int:
            raise TypeError(
                "weighted Jacobian algorithm_version must be an integer"
            )
        for label in (
            "relative_eigenvalue_cutoff",
            "relative_singular_value_cutoff",
        ):
            if type(state[label]) is not float:
                raise TypeError(f"{label} must be a float")
        return cls(
            input_mean=state["input_mean"],
            output_mean=state["output_mean"],
            factors=tuple(factors),
            relative_eigenvalue_cutoff=state[
                "relative_eigenvalue_cutoff"
            ],
            relative_singular_value_cutoff=state[
                "relative_singular_value_cutoff"
            ],
            algorithm=state["algorithm"],
            algorithm_version=state["algorithm_version"],
        )


class CausalWeightedJacobianExecutor(nn.Module):
    """Execute signed prefix factors without any future-position parameters."""

    def __init__(self, result: CausalWeightedJacobianResult) -> None:
        super().__init__()
        if not isinstance(result, CausalWeightedJacobianResult):
            raise TypeError("result must be a CausalWeightedJacobianResult")
        self.sequence_length = result.sequence_length
        self.input_width = result.input_width
        self.output_width = result.output_width
        self.retained_ranks = result.retained_ranks
        self.register_buffer("input_mean", result.input_mean.clone())
        self.register_buffer("output_mean", result.output_mean.clone())
        for factor in result.factors:
            position = factor.output_position
            self.register_buffer(
                f"input_factor_{position}",
                factor.input_factor.clone(),
            )
            self.register_buffer(
                f"output_factor_{position}",
                factor.output_factor.clone(),
            )

    @property
    def causal_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (source, target)
            for target in range(self.sequence_length)
            for source in range(target + 1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating Tensor")
        if (
            inputs.ndim != 3
            or inputs.shape[-1] != self.input_width
            or not 1 <= inputs.shape[1] <= self.sequence_length
        ):
            raise ValueError(
                "inputs must have shape [batch, sequence, input_width] "
                "within the compiled sequence length"
            )
        if inputs.device != self.input_mean.device:
            raise ValueError("inputs and executor must be on the same device")
        original_dtype = inputs.dtype
        compute = inputs.to(dtype=self.input_mean.dtype)
        outputs = []
        for target in range(inputs.shape[1]):
            prefix = (
                compute[:, : target + 1]
                - self.input_mean[: target + 1]
            )
            input_factor = getattr(self, f"input_factor_{target}")
            output_factor = getattr(self, f"output_factor_{target}")
            routed = torch.einsum("bsi,rsi->br", prefix, input_factor)
            delta = torch.einsum("br,or->bo", routed, output_factor)
            outputs.append(delta + self.output_mean[target])
        return torch.stack(outputs, dim=1).to(dtype=original_dtype)


def _resolve_retained_ranks(
    retained_ranks: int | Sequence[int] | None,
    *,
    spectrum_ranks: tuple[int, ...],
) -> tuple[int, ...]:
    if retained_ranks is None:
        return spectrum_ranks
    if type(retained_ranks) is int:
        ranks = (retained_ranks,) * len(spectrum_ranks)
    elif isinstance(retained_ranks, Sequence) and not isinstance(
        retained_ranks,
        (str, bytes),
    ):
        ranks = tuple(retained_ranks)
        if len(ranks) != len(spectrum_ranks):
            raise ValueError(
                "retained_ranks must have one value per output position"
            )
    else:
        raise TypeError(
            "retained_ranks must be an integer, a sequence, or None"
        )
    for position, (rank, maximum) in enumerate(
        zip(ranks, spectrum_ranks, strict=True)
    ):
        if type(rank) is not int or not 0 <= rank <= maximum:
            raise ValueError(
                f"retained rank at output position {position} must be "
                f"between 0 and {maximum}"
            )
    return ranks


def factor_causal_weighted_jacobian(
    jacobian: Tensor,
    input_covariance: Tensor,
    output_fisher: Tensor,
    *,
    input_mean: Tensor | None = None,
    output_mean: Tensor | None = None,
    retained_ranks: int | Sequence[int] | None = None,
    relative_eigenvalue_cutoff: float = 1e-12,
    relative_singular_value_cutoff: float = 1e-12,
    causal_relative_tolerance: float = 1e-12,
) -> CausalWeightedJacobianResult:
    """Build independent causal prefix factors for a constant Jacobian.

    Args:
        jacobian:
            Signed blocks with axes ``[output_position, output_width,
            input_position, input_width]``.
        input_covariance:
            Per-source-position blocks with axes ``[input_position,
            input_width, input_width]``.
        output_fisher:
            Per-target-position blocks with axes ``[output_position,
            output_width, output_width]``.
        input_mean:
            Optional affine center ``[sequence, input_width]``.
        output_mean:
            Optional affine image ``[sequence, output_width]``.
        retained_ranks:
            One rank shared by every prefix, one rank per position, or ``None``
            for the complete supported factorization.

    Full-rank factors reconstruct ``J`` when all covariance and Fisher blocks
    are positive definite.  With singular blocks they reconstruct
    ``P_F J P_C``, the component visible on both PSD supports.
    """

    _validate_relative_cutoff(
        relative_eigenvalue_cutoff,
        label="relative_eigenvalue_cutoff",
    )
    _validate_relative_cutoff(
        relative_singular_value_cutoff,
        label="relative_singular_value_cutoff",
    )
    if (
        not isinstance(causal_relative_tolerance, float)
        or not math.isfinite(causal_relative_tolerance)
        or not 0.0 <= causal_relative_tolerance < 1.0
    ):
        raise ValueError(
            "causal_relative_tolerance must lie in [0, 1)"
        )
    raw_jacobian = _cpu_float64(jacobian, label="jacobian")
    raw_covariance = _cpu_float64(
        input_covariance,
        label="input_covariance",
    )
    raw_fisher = _cpu_float64(output_fisher, label="output_fisher")
    if raw_jacobian.ndim != 4:
        raise ValueError(
            "jacobian must have axes [output_position, output_width, "
            "input_position, input_width]"
        )
    sequence_length, output_width, source_length, input_width = (
        raw_jacobian.shape
    )
    if sequence_length == 0 or output_width == 0 or input_width == 0:
        raise ValueError("jacobian dimensions must be nonempty")
    if source_length != sequence_length:
        raise ValueError(
            "jacobian input and output position axes must have equal length"
        )
    if raw_covariance.shape != (
        sequence_length,
        input_width,
        input_width,
    ):
        raise ValueError(
            "input_covariance must have shape "
            "[sequence, input_width, input_width]"
        )
    if raw_fisher.shape != (
        sequence_length,
        output_width,
        output_width,
    ):
        raise ValueError(
            "output_fisher must have shape "
            "[sequence, output_width, output_width]"
        )

    jacobian_scale = float(raw_jacobian.abs().max().item())
    causal_tolerance = jacobian_scale * max(
        causal_relative_tolerance,
        torch.finfo(torch.float64).eps
        * max(sequence_length, input_width, output_width)
        * 128,
    )
    for target in range(sequence_length):
        if target + 1 == sequence_length:
            continue
        future = raw_jacobian[target, :, target + 1 :, :]
        if (
            future.numel()
            and float(future.abs().max().item()) > causal_tolerance
        ):
            raise ValueError(
                "jacobian contains a noncausal future-position edge "
                f"at output position {target}"
            )

    covariance_factors = tuple(
        factor_psd_support(
            raw_covariance[position],
            relative_cutoff=relative_eigenvalue_cutoff,
        )
        for position in range(sequence_length)
    )
    fisher_factors = tuple(
        factor_psd_support(
            raw_fisher[position],
            relative_cutoff=relative_eigenvalue_cutoff,
        )
        for position in range(sequence_length)
    )
    spectrum_ranks = tuple(
        min(output_width, (position + 1) * input_width)
        for position in range(sequence_length)
    )
    ranks = _resolve_retained_ranks(
        retained_ranks,
        spectrum_ranks=spectrum_ranks,
    )

    factors = []
    for target, retained_rank in enumerate(ranks):
        prefix_length = target + 1
        output_psd = fisher_factors[target]
        weighted_blocks = tuple(
            output_psd.square_root
            @ raw_jacobian[target, :, source, :]
            @ covariance_factors[source].square_root
            for source in range(prefix_length)
        )
        weighted_matrix = torch.cat(weighted_blocks, dim=1)
        left, singular_values, right_h = torch.linalg.svd(
            weighted_matrix,
            full_matrices=False,
        )
        singular_scale = (
            float(singular_values[0].item())
            if singular_values.numel()
            else 0.0
        )
        numerical_relative = (
            torch.finfo(torch.float64).eps
            * max(weighted_matrix.shape)
            * 128
        )
        singular_tolerance = singular_scale * max(
            relative_singular_value_cutoff,
            numerical_relative,
        )
        root = singular_values[:retained_rank].sqrt()
        output_factor = output_psd.inverse_square_root @ (
            left[:, :retained_rank] * root.unsqueeze(0)
        )
        prefix_inverse = torch.stack(
            tuple(
                covariance_factors[source].inverse_square_root
                for source in range(prefix_length)
            )
        )
        block_inverse = torch.block_diag(*tuple(prefix_inverse))
        input_factor = (
            root.unsqueeze(1) * right_h[:retained_rank]
        ) @ block_inverse
        source_edge_energy = (
            weighted_matrix.reshape(
                output_width,
                prefix_length,
                input_width,
            )
            .square()
            .sum(dim=(0, 2))
        )
        factors.append(
            CausalWeightedJacobianFactor(
                output_position=target,
                input_width=input_width,
                output_width=output_width,
                retained_rank=retained_rank,
                singular_values=singular_values,
                weighted_left_vectors=left,
                weighted_right_vectors=right_h.T,
                input_inverse_square_root=prefix_inverse,
                output_inverse_square_root=(
                    output_psd.inverse_square_root
                ),
                input_factor=input_factor.reshape(
                    retained_rank,
                    prefix_length,
                    input_width,
                ),
                output_factor=output_factor,
                source_edge_energy=source_edge_energy,
                singular_tolerance=float(singular_tolerance),
                input_support_ranks=tuple(
                    covariance_factors[source].support_rank
                    for source in range(prefix_length)
                ),
                output_support_rank=output_psd.support_rank,
            )
        )

    resolved_input_mean = (
        torch.zeros(
            sequence_length,
            input_width,
            dtype=torch.float64,
        )
        if input_mean is None
        else _cpu_float64(input_mean, label="input_mean")
    )
    resolved_output_mean = (
        torch.zeros(
            sequence_length,
            output_width,
            dtype=torch.float64,
        )
        if output_mean is None
        else _cpu_float64(output_mean, label="output_mean")
    )
    return CausalWeightedJacobianResult(
        input_mean=resolved_input_mean,
        output_mean=resolved_output_mean,
        factors=tuple(factors),
        relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
        relative_singular_value_cutoff=(
            relative_singular_value_cutoff
        ),
    )


__all__ = [
    "CausalWeightedJacobianExecutor",
    "CausalWeightedJacobianFactor",
    "CausalWeightedJacobianResult",
    "PSDSupportFactor",
    "factor_causal_weighted_jacobian",
    "factor_psd_support",
]
