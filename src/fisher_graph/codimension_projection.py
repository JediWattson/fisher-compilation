"""Inspectable orthogonal projections for residual block deltas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


def canonical_unit_direction(
    value: Tensor,
    *,
    label: str = "direction",
) -> Tensor:
    """Return a detached FP64 unit vector with deterministic sign."""

    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim != 1
        or value.numel() == 0
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite floating vector")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone()
    norm = torch.linalg.vector_norm(result)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        raise ValueError(f"{label} must have nonzero norm")
    result /= norm
    pivot = int(result.abs().argmax().item())
    if float(result[pivot].item()) < 0.0:
        result.neg_()
    return result.contiguous()


def canonical_orthonormal_basis(
    value: Tensor,
    *,
    label: str = "basis",
) -> Tensor:
    """Return canonical-sign FP64 orthonormal columns on CPU."""

    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim != 2
        or value.shape[0] < 2
        or not 1 <= value.shape[1] < value.shape[0]
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{label} must be a finite floating [width, codimension] "
            "matrix with 1 <= codimension < width"
        )
    result = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).clone()
    identity = torch.eye(result.shape[1], dtype=torch.float64)
    if not torch.allclose(
        result.T @ result,
        identity,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"{label} columns must be orthonormal")
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0.0:
            result[:, column_index].neg_()
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class CodimensionOneDeltaProjector:
    """Remove one shared Euclidean direction from a block delta."""

    normal: Tensor

    def __post_init__(self) -> None:
        canonical = canonical_unit_direction(
            self.normal,
            label="projector normal",
        )
        supplied = self.normal.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.allclose(
            supplied,
            canonical,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "projector normal must already be unit length with "
                "canonical sign"
            )
        object.__setattr__(self, "normal", canonical)

    @property
    def width(self) -> int:
        return int(self.normal.numel())

    def project_delta(self, delta: Tensor) -> Tensor:
        if (
            not isinstance(delta, Tensor)
            or not delta.is_floating_point()
            or delta.ndim < 1
            or delta.shape[-1] != self.width
        ):
            raise ValueError(
                "delta must be floating with the projector width "
                "on its final axis"
            )
        compute_dtype = (
            torch.float32
            if delta.dtype in (torch.float16, torch.bfloat16)
            else delta.dtype
        )
        values = delta.to(dtype=compute_dtype)
        normal = self.normal.to(
            device=delta.device,
            dtype=compute_dtype,
        )
        removed = (values @ normal).unsqueeze(-1) * normal
        return (values - removed).to(dtype=delta.dtype)

    def project_output(
        self,
        source: Tensor,
        target: Tensor,
        *,
        valid_positions: Tensor,
    ) -> Tensor:
        if (
            not isinstance(source, Tensor)
            or not isinstance(target, Tensor)
            or source.shape != target.shape
            or source.ndim != 3
            or source.shape[-1] != self.width
            or not source.is_floating_point()
            or not target.is_floating_point()
        ):
            raise ValueError(
                "source and target must be aligned floating "
                "[batch, sequence, width] tensors"
            )
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
            or valid_positions.shape != source.shape[:2]
        ):
            raise ValueError(
                "valid_positions must be a matching boolean "
                "[batch, sequence] tensor"
            )
        projected = source + self.project_delta(target - source)
        return torch.where(
            valid_positions.to(device=source.device).unsqueeze(-1),
            projected,
            target,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "normal": self.normal.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CodimensionOneDeltaProjector:
        if set(state) != {"format_version", "normal"}:
            raise ValueError("codimension projector fields are invalid")
        if type(state["format_version"]) is not int or (
            state["format_version"] != 1
        ):
            raise ValueError("unsupported codimension projector format")
        normal = state["normal"]
        if not isinstance(normal, Tensor):
            raise TypeError("codimension projector normal must be a Tensor")
        return cls(normal=normal)


@dataclass(frozen=True, slots=True)
class OrthogonalDeltaProjector:
    """Remove an inspectable orthonormal subspace from a block delta."""

    omitted_basis: Tensor

    def __post_init__(self) -> None:
        canonical = canonical_orthonormal_basis(
            self.omitted_basis,
            label="omitted basis",
        )
        supplied = self.omitted_basis.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.allclose(
            supplied,
            canonical,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "omitted basis must already use canonical column signs"
            )
        object.__setattr__(self, "omitted_basis", canonical)

    @property
    def width(self) -> int:
        return int(self.omitted_basis.shape[0])

    @property
    def removed_dimensions(self) -> int:
        return int(self.omitted_basis.shape[1])

    @property
    def retained_rank(self) -> int:
        return self.width - self.removed_dimensions

    def project_delta(self, delta: Tensor) -> Tensor:
        if (
            not isinstance(delta, Tensor)
            or not delta.is_floating_point()
            or delta.ndim < 1
            or delta.shape[-1] != self.width
        ):
            raise ValueError(
                "delta must be floating with the projector width "
                "on its final axis"
            )
        compute_dtype = (
            torch.float32
            if delta.dtype in (torch.float16, torch.bfloat16)
            else delta.dtype
        )
        values = delta.to(dtype=compute_dtype)
        basis = self.omitted_basis.to(
            device=delta.device,
            dtype=compute_dtype,
        )
        removed = (values @ basis) @ basis.T
        return (values - removed).to(dtype=delta.dtype)

    def project_output(
        self,
        source: Tensor,
        target: Tensor,
        *,
        valid_positions: Tensor,
    ) -> Tensor:
        if (
            not isinstance(source, Tensor)
            or not isinstance(target, Tensor)
            or source.shape != target.shape
            or source.ndim != 3
            or source.shape[-1] != self.width
            or not source.is_floating_point()
            or not target.is_floating_point()
        ):
            raise ValueError(
                "source and target must be aligned floating "
                "[batch, sequence, width] tensors"
            )
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
            or valid_positions.shape != source.shape[:2]
        ):
            raise ValueError(
                "valid_positions must be a matching boolean "
                "[batch, sequence] tensor"
            )
        projected = source + self.project_delta(target - source)
        return torch.where(
            valid_positions.to(device=source.device).unsqueeze(-1),
            projected,
            target,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "omitted_basis": self.omitted_basis.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> OrthogonalDeltaProjector:
        if set(state) != {"format_version", "omitted_basis"}:
            raise ValueError("orthogonal projector fields are invalid")
        if (
            type(state["format_version"]) is not int
            or state["format_version"] != 1
        ):
            raise ValueError("unsupported orthogonal projector format")
        basis = state["omitted_basis"]
        if not isinstance(basis, Tensor):
            raise TypeError("orthogonal projector basis must be a Tensor")
        return cls(omitted_basis=basis)


__all__ = [
    "CodimensionOneDeltaProjector",
    "OrthogonalDeltaProjector",
    "canonical_orthonormal_basis",
    "canonical_unit_direction",
]
