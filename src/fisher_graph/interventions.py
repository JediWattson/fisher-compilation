"""Causal interventions in Fisher-mode coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .modes import FisherModeBasis


@dataclass(frozen=True, slots=True)
class FisherModeSuppression:
    """Scale selected centered Fisher-mode coordinates toward zero.

    ``suppression_fraction=0`` leaves the activation unchanged and ``1`` fully
    removes the selected mode components. By default the same intervention is
    applied at every token position.
    """

    basis: FisherModeBasis
    mode_indices: tuple[int, ...]
    suppression_fraction: float = 1.0
    positions: tuple[int, ...] | None = None
    centering: Literal["position", "pooled"] = "position"

    def __post_init__(self) -> None:
        if not 0.0 <= self.suppression_fraction <= 1.0:
            raise ValueError("suppression_fraction must be in [0, 1]")
        if len(set(self.mode_indices)) != len(self.mode_indices):
            raise ValueError("mode_indices cannot contain duplicates")
        if any(
            index < 0 or index >= self.basis.width
            for index in self.mode_indices
        ):
            raise ValueError("mode index is outside the Fisher basis")
        if self.positions is not None:
            if len(set(self.positions)) != len(self.positions):
                raise ValueError("positions cannot contain duplicates")
            if any(position < 0 for position in self.positions):
                raise ValueError("positions cannot be negative")
        if self.centering not in ("position", "pooled"):
            raise ValueError("centering must be 'position' or 'pooled'")

    def __call__(self, activation: Tensor) -> Tensor:
        if activation.shape[-1] != self.basis.width:
            raise ValueError(
                f"activation width {activation.shape[-1]} does not match "
                f"Fisher basis width {self.basis.width}"
            )
        if not self.mode_indices or self.suppression_fraction == 0.0:
            return activation

        vectors = self.basis.vectors[:, self.mode_indices].to(
            device=activation.device,
            dtype=activation.dtype,
        )
        if self.centering == "position":
            if self.basis.position_means is None:
                raise ValueError(
                    "position centering requires position means in the basis"
                )
            if activation.ndim < 2:
                raise ValueError(
                    "position centering requires a sequence dimension"
                )
            if activation.shape[-2] != self.basis.position_means.shape[0]:
                raise ValueError(
                    "activation sequence length does not match position means"
                )
            mean = self.basis.position_means.to(
                device=activation.device,
                dtype=activation.dtype,
            )
        else:
            mean = self.basis.mean.to(
                device=activation.device,
                dtype=activation.dtype,
            )
        centered = activation - mean
        selected_component = (
            centered @ vectors
        ) @ vectors.transpose(0, 1)
        delta = -self.suppression_fraction * selected_component

        if self.positions is None:
            return activation + delta
        if activation.ndim < 2:
            raise ValueError(
                "position-specific suppression requires a sequence dimension"
            )
        sequence_length = activation.shape[-2]
        if any(position >= sequence_length for position in self.positions):
            raise ValueError("position is outside the activation sequence")
        position_mask = torch.zeros(
            sequence_length,
            device=activation.device,
            dtype=activation.dtype,
        )
        position_mask[list(self.positions)] = 1
        mask_shape = (1,) * (activation.ndim - 2) + (
            sequence_length,
            1,
        )
        return activation + delta * position_mask.view(mask_shape)


def top_mode_indices(basis: FisherModeBasis, count: int) -> tuple[int, ...]:
    if not 0 <= count <= basis.width:
        raise ValueError(f"count must be between 0 and {basis.width}")
    return tuple(range(count))


def bottom_mode_indices(
    basis: FisherModeBasis, count: int
) -> tuple[int, ...]:
    if not 0 <= count <= basis.width:
        raise ValueError(f"count must be between 0 and {basis.width}")
    return tuple(range(basis.width - count, basis.width))


def random_mode_indices(
    basis: FisherModeBasis,
    count: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    if not 0 <= count <= basis.width:
        raise ValueError(f"count must be between 0 and {basis.width}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(
        basis.width,
        generator=generator,
    )[:count]
    return tuple(sorted(indices.tolist()))
