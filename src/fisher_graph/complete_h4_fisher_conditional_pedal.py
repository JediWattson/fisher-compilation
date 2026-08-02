"""Source-free Fisher-XY directions with a learned bounded confidence pedal.

This provider is the pointwise-bounded successor to the globally projected
Fisher-XY square.  The offline fit retains the same two-coordinate router and
rank-k multi-affine direction

``q = [c1 * p, c2 * p, c1 * c2 * p] @ left @ right``.

Instead of shrinking the direction factors globally, every runtime row is
normalized into the relative trust ball

``b = q * min(1, beta * ||p|| / ||q||)``

with fixed ``beta = 0.25``.  This clips directions outside the trust ball but
never amplifies a direction already inside it.  A source-free
pedal ``a = clamp(bias + [c1, c2, c1*c2] @ weight, 0, 1)`` then emits
``delta = a * b``.  Consequently ``||delta|| <= beta * ||p||`` pointwise,
including exact-zero behavior when the parent or direction is zero.  This is
an amplitude certificate, not a Jacobian or Lipschitz certificate.

The conditional pedal is distilled from the train-only analytic scalar
``<residual, b> / ||b||^2`` using row weights proportional to the existing
Fisher-weighted fit weights times ``||b||^2``.  Runtime clamping produces the
per-row constrained oracle; both raw and clipped targets are authenticated.
The direction itself is fit against a train-only residual target clipped into
that same trust ball.
Constant-optimal and unit-pedal controls store the same tensors and report
identical resource geometry.  Native H4, reverse-VJP gradients, axes, and
family IDs remain strictly fit-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import math
import os
from pathlib import Path
import stat
import tempfile

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
    _SEQUENCE_DOMAIN,
    _bounded_vjp_multipliers,
    _float_tensor,
    _ordered_sequences,
    _require_sha256,
    _sha256,
    _tensor_sha256,
    autonomous_complete_h4_residual_provider_from_state_dict,
    autonomous_complete_h4_residual_provider_state_dict,
)
from .complete_h4_fisher_conditional_residual import (
    COORDINATE_OBJECTIVES,
    FISHER_XY_COORDINATE_COUNT,
    _bounded_coordinate_target_r2,
    _coordinate_axes_and_targets,
    _finite_float,
    _finite_tuple,
    _float_tuple,
    _nonnegative_float,
    _positive_float,
    _training_parent_modal,
    _weighted_ridge,
    fisher_xy_bounded_coordinates,
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .radial_finite_displacement_correction import family_balanced_row_weights


__all__ = [
    "FISHER_XY_PEDAL_MODES",
    "FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR",
    "FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR",
    "FISHER_XY_PEDAL_TRUST_FRACTION",
    "AutonomousCompleteH4FisherXYPedalProvider",
    "FisherXYPedalRuntimeReplay",
    "autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict",
    "autonomous_complete_h4_fisher_xy_pedal_provider_state_dict",
    "fisher_xy_pedal_features",
    "fisher_xy_pedal_fit_support_mask",
    "fisher_xy_pointwise_bounded_direction",
    "fit_autonomous_complete_h4_fisher_xy_pedal",
    "load_autonomous_complete_h4_fisher_xy_pedal_provider",
    "replay_autonomous_complete_h4_fisher_xy_pedal",
    "save_autonomous_complete_h4_fisher_xy_pedal_provider",
    "validate_fisher_xy_pedal_runtime_replay_metadata",
]


FISHER_XY_PEDAL_TRUST_FRACTION = 0.25
FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR = torch.finfo(torch.float64).eps
FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR = torch.finfo(torch.float64).tiny
FISHER_XY_PEDAL_MODES = frozenset(
    {"conditional", "constant_optimal", "unit"}
)


def _validate_training_sequence_integrity(
    sequence: AutonomousCompleteH4TrainingSequence,
) -> None:
    """Fail closed if an upstream canonical sequence was mutated in place."""

    if not isinstance(sequence, AutonomousCompleteH4TrainingSequence):
        raise TypeError("Fisher-XY pedal training sequence type differs")
    gradient = sequence.reverse_vjp_gradients
    payload = {
        "example_id": sequence.example_id,
        "family_id": sequence.family_id,
        "tensor_sha256s": {
            name: _tensor_sha256(getattr(sequence, name))
            for name in (
                "source_modes",
                "logical_positions",
                "valid_mask",
                "source_mask",
                "support_mask",
                "base_h4",
                "native_h4",
            )
        }
        | {
            "reverse_vjp_gradients": (
                None if gradient is None else _tensor_sha256(gradient)
            )
        },
    }
    embedded = _require_sha256(
        sequence.artifact_sha256,
        label="Fisher-XY pedal training sequence artifact",
    )
    if _sha256(_SEQUENCE_DOMAIN, payload) != embedded:
        raise RuntimeError("Fisher-XY pedal training sequence payload drifted")

_H4_SITE = "layer.4.output"
_FEATURE_BLOCK_COUNT = 3
_PEDAL_FEATURE_COUNT = 3
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-xy-pedal:provider:v1\0"
)
_FIT_RECEIPT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-xy-pedal:fit:v1\0"
)
_REPLAY_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-xy-pedal:replay:v1\0"
)
_STATE_SCHEMA = (
    "fisher_graph.autonomous_complete_h4_fisher_xy_pedal_provider_tensor.v1"
)
_STATE_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "provider_artifact_sha256",
        "bridge_binding_sha256",
        "parent_provider_artifact_sha256",
        "router_ridge",
        "direction_ridge",
        "pedal_ridge",
        "trust_fraction",
        "fit_row_count",
        "fit_family_ids",
        "fit_sequence_sha256s",
        "coordinate_objective",
        "coordinate_axes_sha256",
        "coordinate_axis_values",
        "fit_weight_sha256",
        "coordinate_target_weighted_rmse",
        "bounded_coordinate_geometry_sha256",
        "bounded_coordinate_covariance_eigenvalues",
        "bounded_coordinate_lambda2_over_lambda1",
        "bounded_coordinate_abs_correlation",
        "bounded_coordinate_target_r2",
        "residual_second_coordinate_energy_fraction",
        "pedal_mode",
        "pedal_unclipped_target_sha256",
        "pedal_target_sha256",
        "pedal_fit_weight_sha256",
        "pedal_support_mask_sha256",
        "pedal_supported_row_count",
        "pedal_unclipped_target_weighted_mean",
        "pedal_unclipped_target_weighted_rmse",
        "pedal_target_weighted_mean",
        "pedal_target_weighted_rmse",
        "pedal_weighted_mean",
        "pedal_weighted_std",
        "pedal_min",
        "pedal_max",
        "pedal_effective_weighted_mean",
        "pedal_effective_weighted_std",
        "pedal_effective_min",
        "pedal_effective_max",
        "pedal_zero_fraction",
        "pedal_one_fraction",
        "pedal_target_clipped_fraction",
        "bounded_target_clipped_fraction",
        "bounded_target_mean_clip_scale",
        "direction_clipped_fraction",
        "direction_mean_clip_scale",
        "weighted_bounded_target_rmse_before",
        "weighted_bounded_target_rmse_after",
        "weighted_residual_rmse_before",
        "weighted_residual_rmse_constant",
        "weighted_residual_rmse_unit",
        "weighted_residual_rmse_oracle",
        "weighted_residual_rmse_after",
        "fit_bounded_direction_ratio_quantiles",
        "fit_emitted_delta_ratio_quantiles",
        "fit_max_bounded_direction_ratio",
        "fit_max_emitted_delta_ratio",
        "zero_parent_row_count",
        "zero_direction_row_count",
        "fit_receipt_sha256",
        "parent_provider_state",
        "tensors",
    }
)
_TENSOR_KEYS = frozenset(
    {
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
        "pedal_weight",
        "pedal_bias",
    }
)


def _fraction(value: object, *, label: str) -> float:
    result = _nonnegative_float(value, label=label)
    if result > 1.0:
        raise ValueError(f"{label} must not exceed one")
    return result


def fisher_xy_pedal_features(coordinates: Tensor) -> Tensor:
    """Return ``[c1, c2, c1*c2]`` without detaching live activations."""

    if (
        not isinstance(coordinates, Tensor)
        or coordinates.ndim < 2
        or coordinates.shape[-1] != FISHER_XY_COORDINATE_COUNT
        or not coordinates.is_floating_point()
        or not bool(torch.isfinite(coordinates).all())
        or bool((coordinates.abs() >= 1.0).any())
    ):
        raise ValueError("pedal coordinates must be finite and lie in (-1, 1)")
    values = coordinates.to(dtype=torch.float64)
    return torch.cat(
        (values[..., :1], values[..., 1:], values[..., :1] * values[..., 1:]),
        dim=-1,
    ).contiguous()


def fisher_xy_pedal_fit_support_mask(
    parent_modal: Tensor,
    bounded_direction: Tensor,
) -> Tensor:
    """Select rows whose direction energy safely supports analytic division."""

    if (
        not isinstance(parent_modal, Tensor)
        or not isinstance(bounded_direction, Tensor)
        or parent_modal.shape != bounded_direction.shape
        or parent_modal.ndim != 2
        or not parent_modal.is_floating_point()
        or not bounded_direction.is_floating_point()
        or not bool(torch.isfinite(parent_modal).all())
        or not bool(torch.isfinite(bounded_direction).all())
    ):
        raise ValueError("Fisher-XY pedal fit support tensors differ")
    parent = parent_modal.to(dtype=torch.float64)
    bounded = bounded_direction.to(device=parent.device, dtype=torch.float64)
    bounded_energy = bounded.square().sum(dim=1)
    parent_energy = parent.square().sum(dim=1)
    if (
        not bool(torch.isfinite(bounded_energy).all())
        or not bool(torch.isfinite(parent_energy).all())
    ):
        raise ValueError("Fisher-XY pedal fit support energy became nonfinite")
    energy_floor = torch.maximum(
        torch.full_like(
            bounded_energy,
            FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR,
        ),
        FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR * parent_energy,
    )
    return (bounded_energy > energy_floor).contiguous()


def fisher_xy_pointwise_bounded_direction(
    parent_modal: Tensor,
    direction: Tensor,
    *,
    trust_fraction: float = FISHER_XY_PEDAL_TRUST_FRACTION,
) -> Tensor:
    """Normalize a direction into ``||b|| <= beta * ||parent||`` per row.

    The two zero cases are explicit: a zero parent or zero direction returns
    an exactly zero row.  A nonzero direction already within the budget is
    returned unchanged rather than amplified to the trust-ball boundary.
    """

    if (
        not isinstance(parent_modal, Tensor)
        or not isinstance(direction, Tensor)
        or parent_modal.shape != direction.shape
        or parent_modal.ndim < 2
        or not parent_modal.is_floating_point()
        or not direction.is_floating_point()
        or not bool(torch.isfinite(parent_modal).all())
        or not bool(torch.isfinite(direction).all())
    ):
        raise ValueError("parent modal and direction must be matching finite tensors")
    beta = _positive_float(trust_fraction, label="trust_fraction")
    if beta > 1.0:
        raise ValueError("trust_fraction must not exceed one")

    parent = parent_modal.to(dtype=torch.float64)
    candidate = direction.to(device=parent.device, dtype=torch.float64)
    parent_norm = torch.linalg.vector_norm(parent, dim=-1, keepdim=True)
    direction_norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
    active = (parent_norm > 0.0) & (direction_norm > 0.0)
    safe_direction_norm = torch.where(
        active,
        direction_norm,
        torch.ones_like(direction_norm),
    )
    scale = torch.where(
        active,
        torch.minimum(
            torch.ones_like(direction_norm),
            beta * parent_norm / safe_direction_norm,
        ),
        torch.zeros_like(direction_norm),
    )
    bounded = candidate * scale
    # Multiplication by an exact zero is exact in IEEE arithmetic, but assign
    # the inactive rows as well so NaN-producing exotic backends cannot leak.
    bounded = bounded.masked_fill(~active, 0.0).contiguous()
    if (
        not bool(torch.isfinite(bounded).all())
        or bool((bounded[~active.squeeze(-1)] != 0.0).any())
    ):
        raise RuntimeError("pointwise Fisher-XY direction escaped its trust ball")
    return bounded


@dataclass(frozen=True, slots=True)
class FisherXYPedalRuntimeReplay:
    """Canonical support-row replay from serving-available fields only."""

    provider_artifact_sha256: str
    parent_provider_artifact_sha256: str
    sequence_artifact_sha256: str
    trust_fraction: float
    parent_modal: Tensor
    bounded_coordinates: Tensor
    unbounded_direction: Tensor
    bounded_direction: Tensor
    pedal: Tensor
    emitted_delta: Tensor
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.provider_artifact_sha256,
            label="replay provider artifact",
        )
        _require_sha256(
            self.parent_provider_artifact_sha256,
            label="replay parent provider artifact",
        )
        _require_sha256(
            self.sequence_artifact_sha256,
            label="replay sequence artifact",
        )
        trust = _positive_float(self.trust_fraction, label="replay trust_fraction")
        if trust != FISHER_XY_PEDAL_TRUST_FRACTION:
            raise ValueError("Fisher-XY pedal replay trust_fraction must be 0.25")
        object.__setattr__(self, "trust_fraction", trust)
        parent = _float_tensor(self.parent_modal, label="replay parent modal", ndim=2)
        coordinates = _float_tensor(
            self.bounded_coordinates,
            label="replay bounded coordinates",
            ndim=2,
        )
        direction = _float_tensor(
            self.unbounded_direction,
            label="replay unbounded direction",
            ndim=2,
        )
        bounded = _float_tensor(
            self.bounded_direction,
            label="replay bounded direction",
            ndim=2,
        )
        pedal = _float_tensor(self.pedal, label="replay pedal", ndim=1)
        delta = _float_tensor(self.emitted_delta, label="replay emitted delta", ndim=2)
        rows, rank = parent.shape
        if (
            coordinates.shape != (rows, FISHER_XY_COORDINATE_COUNT)
            or direction.shape != (rows, rank)
            or bounded.shape != (rows, rank)
            or pedal.shape != (rows,)
            or delta.shape != (rows, rank)
            or bool((coordinates.abs() >= 1.0).any())
            or bool((pedal < 0.0).any())
            or bool((pedal > 1.0).any())
            or not torch.equal(delta, pedal.unsqueeze(1) * bounded)
        ):
            raise ValueError("Fisher-XY pedal replay geometry differs")
        parent_norm = torch.linalg.vector_norm(parent, dim=1)
        bounded_norm = torch.linalg.vector_norm(bounded, dim=1)
        delta_norm = torch.linalg.vector_norm(delta, dim=1)
        tolerance = 64.0 * torch.finfo(torch.float64).eps * torch.maximum(
            trust * parent_norm,
            torch.ones_like(parent_norm),
        )
        if (
            bool((bounded_norm > trust * parent_norm + tolerance).any())
            or bool((delta_norm > trust * parent_norm + tolerance).any())
            or bool((bounded[parent_norm == 0.0] != 0.0).any())
            or bool((delta[parent_norm == 0.0] != 0.0).any())
        ):
            raise ValueError("Fisher-XY pedal replay escaped pointwise trust")
        for name, value in (
            ("parent_modal", parent),
            ("bounded_coordinates", coordinates),
            ("unbounded_direction", direction),
            ("bounded_direction", bounded),
            ("pedal", pedal),
            ("emitted_delta", delta),
        ):
            object.__setattr__(self, name, value)
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="Fisher-XY pedal replay",
            ) != computed:
                raise ValueError("Fisher-XY pedal replay artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def row_count(self) -> int:
        return int(self.parent_modal.shape[0])

    def _payload(self) -> dict[str, object]:
        parent_norm = torch.linalg.vector_norm(self.parent_modal, dim=1)
        bounded_norm = torch.linalg.vector_norm(self.bounded_direction, dim=1)
        delta_norm = torch.linalg.vector_norm(self.emitted_delta, dim=1)
        positive_parent = parent_norm > 0.0
        bounded_ratio = torch.where(
            positive_parent,
            bounded_norm / torch.where(
                positive_parent,
                parent_norm,
                torch.ones_like(parent_norm),
            ),
            torch.zeros_like(parent_norm),
        )
        delta_ratio = torch.where(
            positive_parent,
            delta_norm / torch.where(
                positive_parent,
                parent_norm,
                torch.ones_like(parent_norm),
            ),
            torch.zeros_like(parent_norm),
        )
        return {
            "schema": "fisher_graph.fisher_xy_pedal_runtime_replay.v1",
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "parent_provider_artifact_sha256": self.parent_provider_artifact_sha256,
            "sequence_artifact_sha256": self.sequence_artifact_sha256,
            "trust_fraction": self.trust_fraction,
            "row_count": self.row_count,
            "rank": int(self.parent_modal.shape[1]),
            "tensor_sha256s": {
                "parent_modal": _tensor_sha256(self.parent_modal),
                "bounded_coordinates": _tensor_sha256(self.bounded_coordinates),
                "unbounded_direction": _tensor_sha256(self.unbounded_direction),
                "bounded_direction": _tensor_sha256(self.bounded_direction),
                "pedal": _tensor_sha256(self.pedal),
                "emitted_delta": _tensor_sha256(self.emitted_delta),
            },
            "zero_parent_row_count": int((parent_norm == 0.0).sum()),
            "zero_direction_row_count": int(
                (torch.linalg.vector_norm(self.unbounded_direction, dim=1) == 0.0).sum()
            ),
            "zero_pedal_row_count": int((self.pedal == 0.0).sum()),
            "unit_pedal_row_count": int((self.pedal == 1.0).sum()),
            "pedal_min": float(self.pedal.min()),
            "pedal_mean": float(self.pedal.mean()),
            "pedal_max": float(self.pedal.max()),
            "max_bounded_direction_to_parent_norm_ratio": float(bounded_ratio.max()),
            "max_emitted_delta_to_parent_norm_ratio": float(delta_ratio.max()),
            "pointwise_trust_certificate_passed": bool(
                float(bounded_ratio.max()) <= self.trust_fraction + 1.0e-14
                and float(delta_ratio.max()) <= self.trust_fraction + 1.0e-14
            ),
            "runtime_field_semantics": "support_rows_from_source_prefix_and_base_h4_only",
        }

    def _computed_sha256(self) -> str:
        return _sha256(_REPLAY_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("Fisher-XY pedal replay payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        result = {**self._payload(), "artifact_sha256": self.artifact_sha256}
        validate_fisher_xy_pedal_runtime_replay_metadata(result)
        return result


def validate_fisher_xy_pedal_runtime_replay_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate an authenticated tensor-free replay receipt from a report."""

    expected_keys = frozenset(
        {
            "schema",
            "provider_artifact_sha256",
            "parent_provider_artifact_sha256",
            "sequence_artifact_sha256",
            "trust_fraction",
            "row_count",
            "rank",
            "tensor_sha256s",
            "zero_parent_row_count",
            "zero_direction_row_count",
            "zero_pedal_row_count",
            "unit_pedal_row_count",
            "pedal_min",
            "pedal_mean",
            "pedal_max",
            "max_bounded_direction_to_parent_norm_ratio",
            "max_emitted_delta_to_parent_norm_ratio",
            "pointwise_trust_certificate_passed",
            "runtime_field_semantics",
            "artifact_sha256",
        }
    )
    if not isinstance(metadata, Mapping) or set(metadata) != expected_keys:
        raise ValueError("Fisher-XY pedal replay metadata fields differ")
    payload = dict(metadata)
    artifact = _require_sha256(
        payload.pop("artifact_sha256"),
        label="Fisher-XY pedal replay metadata artifact",
    )
    if payload.get("schema") != "fisher_graph.fisher_xy_pedal_runtime_replay.v1":
        raise ValueError("Fisher-XY pedal replay metadata schema differs")
    for name in (
        "provider_artifact_sha256",
        "parent_provider_artifact_sha256",
        "sequence_artifact_sha256",
    ):
        _require_sha256(payload.get(name), label=f"replay metadata {name}")
    trust = _positive_float(
        payload.get("trust_fraction"),
        label="replay metadata trust_fraction",
    )
    if trust != FISHER_XY_PEDAL_TRUST_FRACTION:
        raise ValueError("Fisher-XY pedal replay metadata trust differs")
    row_count = payload.get("row_count")
    rank = payload.get("rank")
    if type(row_count) is not int or row_count <= 0 or type(rank) is not int or rank <= 0:
        raise ValueError("Fisher-XY pedal replay metadata geometry differs")
    hashes = payload.get("tensor_sha256s")
    expected_hash_keys = {
        "parent_modal",
        "bounded_coordinates",
        "unbounded_direction",
        "bounded_direction",
        "pedal",
        "emitted_delta",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hash_keys:
        raise ValueError("Fisher-XY pedal replay tensor hashes differ")
    for name in expected_hash_keys:
        _require_sha256(hashes.get(name), label=f"replay tensor {name}")
    for name in (
        "zero_parent_row_count",
        "zero_direction_row_count",
        "zero_pedal_row_count",
        "unit_pedal_row_count",
    ):
        count = payload.get(name)
        if type(count) is not int or count < 0 or count > row_count:
            raise ValueError(f"Fisher-XY pedal replay metadata {name} differs")
    pedal_min = _fraction(payload.get("pedal_min"), label="replay pedal_min")
    pedal_mean = _fraction(payload.get("pedal_mean"), label="replay pedal_mean")
    pedal_max = _fraction(payload.get("pedal_max"), label="replay pedal_max")
    if not pedal_min <= pedal_mean <= pedal_max:
        raise ValueError("Fisher-XY pedal replay pedal range differs")
    bounded_ratio = _nonnegative_float(
        payload.get("max_bounded_direction_to_parent_norm_ratio"),
        label="replay bounded direction ratio",
    )
    emitted_ratio = _nonnegative_float(
        payload.get("max_emitted_delta_to_parent_norm_ratio"),
        label="replay emitted delta ratio",
    )
    if (
        bounded_ratio > trust + 1.0e-14
        or emitted_ratio > trust + 1.0e-14
        or payload.get("pointwise_trust_certificate_passed") is not True
        or payload.get("runtime_field_semantics")
        != "support_rows_from_source_prefix_and_base_h4_only"
    ):
        raise ValueError("Fisher-XY pedal replay metadata trust differs")
    if _sha256(_REPLAY_DOMAIN, payload) != artifact:
        raise ValueError("Fisher-XY pedal replay metadata artifact hash mismatch")
    return {**payload, "artifact_sha256": artifact}


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherXYPedalProvider(Gemma3L3L4CorrectionProvider):
    """Immutable source-free Fisher-XY direction plus confidence pedal."""

    parent_provider: AutonomousCompleteH4ResidualProvider
    router_weight: Tensor
    router_bias: Tensor
    coordinate_scales: Tensor
    direction_left: Tensor
    direction_right: Tensor
    pedal_weight: Tensor
    pedal_bias: Tensor
    router_ridge: float
    direction_ridge: float
    pedal_ridge: float
    trust_fraction: float
    fit_row_count: int
    fit_family_ids: tuple[str, ...]
    fit_sequence_sha256s: tuple[str, ...]
    coordinate_objective: str
    coordinate_axes_sha256: str
    coordinate_axis_values: tuple[float, float]
    fit_weight_sha256: str
    coordinate_target_weighted_rmse: float
    bounded_coordinate_geometry_sha256: str
    bounded_coordinate_covariance_eigenvalues: tuple[float, float]
    bounded_coordinate_lambda2_over_lambda1: float
    bounded_coordinate_abs_correlation: float
    bounded_coordinate_target_r2: tuple[float, float]
    residual_second_coordinate_energy_fraction: float
    pedal_mode: str
    pedal_unclipped_target_sha256: str
    pedal_target_sha256: str
    pedal_fit_weight_sha256: str
    pedal_support_mask_sha256: str
    pedal_supported_row_count: int
    pedal_unclipped_target_weighted_mean: float
    pedal_unclipped_target_weighted_rmse: float
    pedal_target_weighted_mean: float
    pedal_target_weighted_rmse: float
    pedal_weighted_mean: float
    pedal_weighted_std: float
    pedal_min: float
    pedal_max: float
    pedal_effective_weighted_mean: float
    pedal_effective_weighted_std: float
    pedal_effective_min: float
    pedal_effective_max: float
    pedal_zero_fraction: float
    pedal_one_fraction: float
    pedal_target_clipped_fraction: float
    bounded_target_clipped_fraction: float
    bounded_target_mean_clip_scale: float
    direction_clipped_fraction: float
    direction_mean_clip_scale: float
    weighted_bounded_target_rmse_before: float
    weighted_bounded_target_rmse_after: float
    weighted_residual_rmse_before: float
    weighted_residual_rmse_constant: float
    weighted_residual_rmse_unit: float
    weighted_residual_rmse_oracle: float
    weighted_residual_rmse_after: float
    fit_bounded_direction_ratio_quantiles: tuple[float, float, float]
    fit_emitted_delta_ratio_quantiles: tuple[float, float, float]
    fit_max_bounded_direction_ratio: float
    fit_max_emitted_delta_ratio: float
    zero_parent_row_count: int
    zero_direction_row_count: int
    fit_receipt_sha256: str
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.parent_provider, AutonomousCompleteH4ResidualProvider):
            raise TypeError("Fisher-XY pedal parent must be autonomous complete-H4")
        self.parent_provider.validate_integrity()
        router_weight = _float_tensor(
            self.router_weight,
            label="router_weight",
            ndim=2,
        )
        router_bias = _float_tensor(self.router_bias, label="router_bias", ndim=1)
        coordinate_scales = _float_tensor(
            self.coordinate_scales,
            label="coordinate_scales",
            ndim=1,
        )
        left = _float_tensor(self.direction_left, label="direction_left", ndim=2)
        right = _float_tensor(self.direction_right, label="direction_right", ndim=2)
        pedal_weight = _float_tensor(
            self.pedal_weight,
            label="pedal_weight",
            ndim=1,
        )
        pedal_bias = _float_tensor(self.pedal_bias, label="pedal_bias", ndim=1)
        rank = self.parent_provider.rank
        if (
            router_weight.shape != (rank, FISHER_XY_COORDINATE_COUNT)
            or router_bias.shape != (FISHER_XY_COORDINATE_COUNT,)
            or coordinate_scales.shape != (FISHER_XY_COORDINATE_COUNT,)
            or bool((coordinate_scales <= 0.0).any())
            or left.shape[0] != _FEATURE_BLOCK_COUNT * rank
            or right.shape[1] != rank
            or left.shape[1] != right.shape[0]
            or pedal_weight.shape != (_PEDAL_FEATURE_COUNT,)
            or pedal_bias.shape != (1,)
        ):
            raise ValueError("Fisher-XY pedal provider tensor geometry differs")
        for name, value in (
            ("router_weight", router_weight),
            ("router_bias", router_bias),
            ("coordinate_scales", coordinate_scales),
            ("direction_left", left),
            ("direction_right", right),
            ("pedal_weight", pedal_weight),
            ("pedal_bias", pedal_bias),
        ):
            object.__setattr__(self, name, value)
        for name in ("router_ridge", "direction_ridge", "pedal_ridge"):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), label=name),
            )
        trust = _positive_float(self.trust_fraction, label="trust_fraction")
        if trust != FISHER_XY_PEDAL_TRUST_FRACTION:
            raise ValueError("Fisher-XY pedal trust_fraction is frozen at 0.25")
        object.__setattr__(self, "trust_fraction", trust)
        if type(self.fit_row_count) is not int or self.fit_row_count <= 0:
            raise ValueError("Fisher-XY pedal fit_row_count must be positive")
        if (
            type(self.fit_family_ids) is not tuple
            or not self.fit_family_ids
            or self.fit_family_ids != tuple(sorted(set(self.fit_family_ids)))
            or type(self.fit_sequence_sha256s) is not tuple
            or not self.fit_sequence_sha256s
            or self.fit_sequence_sha256s
            != tuple(sorted(set(self.fit_sequence_sha256s)))
        ):
            raise ValueError("Fisher-XY pedal fit ownership must be canonical")
        for value in self.fit_sequence_sha256s:
            _require_sha256(value, label="Fisher-XY pedal fit sequence")
        if self.coordinate_objective not in COORDINATE_OBJECTIVES:
            raise ValueError("Fisher-XY pedal coordinate objective differs")
        if self.pedal_mode not in FISHER_XY_PEDAL_MODES:
            raise ValueError("Fisher-XY pedal mode differs")
        if self.pedal_mode == "unit" and (
            bool((pedal_weight != 0.0).any())
            or not torch.equal(pedal_bias, torch.ones_like(pedal_bias))
        ):
            raise ValueError("unit Fisher-XY pedal tensors differ")
        if self.pedal_mode == "constant_optimal" and (
            bool((pedal_weight != 0.0).any())
            or bool((pedal_bias < 0.0).any())
            or bool((pedal_bias > 1.0).any())
        ):
            raise ValueError("constant-optimal Fisher-XY pedal tensors differ")
        for name in (
            "coordinate_axes_sha256",
            "fit_weight_sha256",
            "bounded_coordinate_geometry_sha256",
            "pedal_unclipped_target_sha256",
            "pedal_target_sha256",
            "pedal_fit_weight_sha256",
            "pedal_support_mask_sha256",
            "fit_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "coordinate_axis_values",
            _float_tuple(
                self.coordinate_axis_values,
                count=2,
                label="coordinate_axis_values",
                positive=True,
            ),
        )
        covariance = _float_tuple(
            self.bounded_coordinate_covariance_eigenvalues,
            count=2,
            label="bounded_coordinate_covariance_eigenvalues",
        )
        if covariance[0] < covariance[1]:
            raise ValueError("bounded coordinate covariance must be descending")
        object.__setattr__(
            self,
            "bounded_coordinate_covariance_eigenvalues",
            covariance,
        )
        ratio = _fraction(
            self.bounded_coordinate_lambda2_over_lambda1,
            label="bounded_coordinate_lambda2_over_lambda1",
        )
        expected_ratio = 0.0 if covariance[0] == 0.0 else covariance[1] / covariance[0]
        if not math.isclose(ratio, expected_ratio, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError("bounded coordinate covariance ratio differs")
        object.__setattr__(self, "bounded_coordinate_lambda2_over_lambda1", ratio)
        object.__setattr__(
            self,
            "bounded_coordinate_abs_correlation",
            _fraction(
                self.bounded_coordinate_abs_correlation,
                label="bounded_coordinate_abs_correlation",
            ),
        )
        target_r2 = _finite_tuple(
            self.bounded_coordinate_target_r2,
            count=2,
            label="bounded_coordinate_target_r2",
        )
        if any(value > 1.0 for value in target_r2):
            raise ValueError("bounded coordinate target R-squared exceeds one")
        object.__setattr__(self, "bounded_coordinate_target_r2", target_r2)
        object.__setattr__(
            self,
            "residual_second_coordinate_energy_fraction",
            _fraction(
                self.residual_second_coordinate_energy_fraction,
                label="residual_second_coordinate_energy_fraction",
            ),
        )
        object.__setattr__(
            self,
            "pedal_unclipped_target_weighted_mean",
            _finite_float(
                self.pedal_unclipped_target_weighted_mean,
                label="pedal_unclipped_target_weighted_mean",
            ),
        )
        if (
            type(self.pedal_supported_row_count) is not int
            or self.pedal_supported_row_count <= 0
            or self.pedal_supported_row_count > self.fit_row_count
        ):
            raise ValueError("pedal_supported_row_count differs")
        for name in (
            "coordinate_target_weighted_rmse",
            "pedal_unclipped_target_weighted_rmse",
            "pedal_target_weighted_mean",
            "pedal_target_weighted_rmse",
            "pedal_weighted_mean",
            "pedal_weighted_std",
            "pedal_min",
            "pedal_max",
            "pedal_effective_weighted_mean",
            "pedal_effective_weighted_std",
            "pedal_effective_min",
            "pedal_effective_max",
            "pedal_zero_fraction",
            "pedal_one_fraction",
            "pedal_target_clipped_fraction",
            "bounded_target_clipped_fraction",
            "bounded_target_mean_clip_scale",
            "direction_clipped_fraction",
            "direction_mean_clip_scale",
            "weighted_bounded_target_rmse_before",
            "weighted_bounded_target_rmse_after",
            "weighted_residual_rmse_before",
            "weighted_residual_rmse_constant",
            "weighted_residual_rmse_unit",
            "weighted_residual_rmse_oracle",
            "weighted_residual_rmse_after",
            "fit_max_bounded_direction_ratio",
            "fit_max_emitted_delta_ratio",
        ):
            value = _nonnegative_float(getattr(self, name), label=name)
            if name in (
                "pedal_target_weighted_mean",
                "pedal_weighted_mean",
                "pedal_min",
                "pedal_max",
                "pedal_effective_weighted_mean",
                "pedal_effective_min",
                "pedal_effective_max",
                "pedal_zero_fraction",
                "pedal_one_fraction",
                "pedal_target_clipped_fraction",
                "bounded_target_clipped_fraction",
                "bounded_target_mean_clip_scale",
                "direction_clipped_fraction",
                "direction_mean_clip_scale",
            ) and value > 1.0:
                fraction_tolerance = 128.0 * torch.finfo(torch.float64).eps
                if value > 1.0 + fraction_tolerance:
                    raise ValueError(f"{name} must not exceed one")
                value = 1.0
            object.__setattr__(self, name, value)
        for mean_name, minimum_name, maximum_name, label in (
            ("pedal_weighted_mean", "pedal_min", "pedal_max", "fitted pedal"),
            (
                "pedal_effective_weighted_mean",
                "pedal_effective_min",
                "pedal_effective_max",
                "effective fitted pedal",
            ),
        ):
            mean = getattr(self, mean_name)
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            range_tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
                1.0,
                abs(minimum),
                abs(maximum),
            )
            if mean < minimum - range_tolerance or mean > maximum + range_tolerance:
                raise ValueError(f"{label} range does not contain its weighted mean")
            object.__setattr__(self, mean_name, min(max(mean, minimum), maximum))
        for name in (
            "fit_bounded_direction_ratio_quantiles",
            "fit_emitted_delta_ratio_quantiles",
        ):
            quantiles = _float_tuple(
                getattr(self, name),
                count=3,
                label=name,
            )
            if quantiles != tuple(sorted(quantiles)):
                raise ValueError(f"{name} must be ascending")
            object.__setattr__(self, name, quantiles)
        tolerance = 64.0 * torch.finfo(torch.float64).eps
        if (
            self.fit_max_bounded_direction_ratio > trust + tolerance
            or self.fit_max_emitted_delta_ratio > trust + tolerance
            or not math.isclose(
                self.fit_bounded_direction_ratio_quantiles[-1],
                self.fit_max_bounded_direction_ratio,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                self.fit_emitted_delta_ratio_quantiles[-1],
                self.fit_max_emitted_delta_ratio,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("Fisher-XY pedal fit escaped pointwise trust")
        for name in ("zero_parent_row_count", "zero_direction_row_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > self.fit_row_count:
                raise ValueError(f"{name} differs")
        control_tolerance = 64.0 * torch.finfo(torch.float64).eps
        if self.pedal_mode == "unit":
            unit_diagnostics = (
                self.pedal_weighted_mean == 1.0
                and self.pedal_weighted_std == 0.0
                and self.pedal_min == 1.0
                and self.pedal_max == 1.0
                and self.pedal_effective_weighted_mean == 1.0
                and self.pedal_effective_weighted_std == 0.0
                and self.pedal_effective_min == 1.0
                and self.pedal_effective_max == 1.0
                and self.pedal_zero_fraction == 0.0
                and self.pedal_one_fraction == 1.0
                and self.weighted_residual_rmse_after
                == self.weighted_residual_rmse_unit
                and self.fit_emitted_delta_ratio_quantiles
                == self.fit_bounded_direction_ratio_quantiles
                and self.fit_max_emitted_delta_ratio
                == self.fit_max_bounded_direction_ratio
            )
            if not unit_diagnostics:
                raise ValueError("unit Fisher-XY pedal diagnostics differ")
        elif self.pedal_mode == "constant_optimal":
            constant = float(self.pedal_bias[0])

            def close(value: float, expected: float) -> bool:
                return math.isclose(
                    value,
                    expected,
                    rel_tol=0.0,
                    abs_tol=control_tolerance,
                )

            expected_zero = 1.0 if constant == 0.0 else 0.0
            expected_one = 1.0 if constant == 1.0 else 0.0
            constant_diagnostics = (
                all(
                    close(value, constant)
                    for value in (
                        self.pedal_weighted_mean,
                        self.pedal_min,
                        self.pedal_max,
                        self.pedal_effective_weighted_mean,
                        self.pedal_effective_min,
                        self.pedal_effective_max,
                    )
                )
                and self.pedal_weighted_std <= control_tolerance
                and self.pedal_effective_weighted_std <= control_tolerance
                and close(self.pedal_zero_fraction, expected_zero)
                and close(self.pedal_one_fraction, expected_one)
                and close(
                    self.weighted_residual_rmse_after,
                    self.weighted_residual_rmse_constant,
                )
                and all(
                    close(emitted, constant * bounded)
                    for emitted, bounded in zip(
                        self.fit_emitted_delta_ratio_quantiles,
                        self.fit_bounded_direction_ratio_quantiles,
                        strict=True,
                    )
                )
                and close(
                    self.fit_max_emitted_delta_ratio,
                    constant * self.fit_max_bounded_direction_ratio,
                )
            )
            if not constant_diagnostics:
                raise ValueError(
                    "constant-optimal Fisher-XY pedal diagnostics differ"
                )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="Fisher-XY pedal provider",
            ) != computed:
                raise ValueError("Fisher-XY pedal provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def bridge_binding_sha256(self) -> str:
        return self.parent_provider.bridge_binding_sha256

    @property
    def rank(self) -> int:
        return self.parent_provider.rank

    @property
    def conditional_rank(self) -> int:
        return int(self.direction_right.shape[0])

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return int(
            self.router_weight.numel()
            + self.router_bias.numel()
            + self.coordinate_scales.numel()
            + self.direction_left.numel()
            + self.direction_right.numel()
            + self.pedal_weight.numel()
            + self.pedal_bias.numel()
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_provider.prepared_float_scalar_count
            + self.incremental_prepared_float_scalar_count
        )

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        return int(
            FISHER_XY_COORDINATE_COUNT * self.rank
            + _FEATURE_BLOCK_COUNT * self.rank * self.conditional_rank
            + self.conditional_rank * self.rank
            + _PEDAL_FEATURE_COUNT
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_provider.logical_macs_per_token_upper_bound
            + self.incremental_logical_macs_per_token_upper_bound
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.autonomous_complete_h4_fisher_xy_pedal_provider.v1",
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "tensor_sha256s": {
                "router_weight": _tensor_sha256(self.router_weight),
                "router_bias": _tensor_sha256(self.router_bias),
                "coordinate_scales": _tensor_sha256(self.coordinate_scales),
                "direction_left": _tensor_sha256(self.direction_left),
                "direction_right": _tensor_sha256(self.direction_right),
                "pedal_weight": _tensor_sha256(self.pedal_weight),
                "pedal_bias": _tensor_sha256(self.pedal_bias),
            },
            "router_ridge": self.router_ridge,
            "direction_ridge": self.direction_ridge,
            "pedal_ridge": self.pedal_ridge,
            "trust_fraction": self.trust_fraction,
            "fit_row_count": self.fit_row_count,
            "fit_family_ids": self.fit_family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "coordinate_objective": self.coordinate_objective,
            "coordinate_axes_sha256": self.coordinate_axes_sha256,
            "coordinate_axis_values": self.coordinate_axis_values,
            "fit_weight_sha256": self.fit_weight_sha256,
            "coordinate_target_weighted_rmse": self.coordinate_target_weighted_rmse,
            "bounded_coordinate_geometry_sha256": self.bounded_coordinate_geometry_sha256,
            "bounded_coordinate_covariance_eigenvalues": self.bounded_coordinate_covariance_eigenvalues,
            "bounded_coordinate_lambda2_over_lambda1": self.bounded_coordinate_lambda2_over_lambda1,
            "bounded_coordinate_abs_correlation": self.bounded_coordinate_abs_correlation,
            "bounded_coordinate_target_r2": self.bounded_coordinate_target_r2,
            "residual_second_coordinate_energy_fraction": self.residual_second_coordinate_energy_fraction,
            "pedal_mode": self.pedal_mode,
            "pedal_unclipped_target_sha256": self.pedal_unclipped_target_sha256,
            "pedal_target_sha256": self.pedal_target_sha256,
            "pedal_fit_weight_sha256": self.pedal_fit_weight_sha256,
            "pedal_support_mask_sha256": self.pedal_support_mask_sha256,
            "pedal_supported_row_count": self.pedal_supported_row_count,
            "pedal_unclipped_target_weighted_mean": self.pedal_unclipped_target_weighted_mean,
            "pedal_unclipped_target_weighted_rmse": self.pedal_unclipped_target_weighted_rmse,
            "pedal_target_weighted_mean": self.pedal_target_weighted_mean,
            "pedal_target_weighted_rmse": self.pedal_target_weighted_rmse,
            "pedal_weighted_mean": self.pedal_weighted_mean,
            "pedal_weighted_std": self.pedal_weighted_std,
            "pedal_min": self.pedal_min,
            "pedal_max": self.pedal_max,
            "pedal_effective_weighted_mean": self.pedal_effective_weighted_mean,
            "pedal_effective_weighted_std": self.pedal_effective_weighted_std,
            "pedal_effective_min": self.pedal_effective_min,
            "pedal_effective_max": self.pedal_effective_max,
            "pedal_zero_fraction": self.pedal_zero_fraction,
            "pedal_one_fraction": self.pedal_one_fraction,
            "pedal_target_clipped_fraction": self.pedal_target_clipped_fraction,
            "bounded_target_clipped_fraction": self.bounded_target_clipped_fraction,
            "bounded_target_mean_clip_scale": self.bounded_target_mean_clip_scale,
            "direction_clipped_fraction": self.direction_clipped_fraction,
            "direction_mean_clip_scale": self.direction_mean_clip_scale,
            "weighted_bounded_target_rmse_before": self.weighted_bounded_target_rmse_before,
            "weighted_bounded_target_rmse_after": self.weighted_bounded_target_rmse_after,
            "weighted_residual_rmse_before": self.weighted_residual_rmse_before,
            "weighted_residual_rmse_constant": self.weighted_residual_rmse_constant,
            "weighted_residual_rmse_unit": self.weighted_residual_rmse_unit,
            "weighted_residual_rmse_oracle": self.weighted_residual_rmse_oracle,
            "weighted_residual_rmse_after": self.weighted_residual_rmse_after,
            "fit_bounded_direction_ratio_quantiles": self.fit_bounded_direction_ratio_quantiles,
            "fit_emitted_delta_ratio_quantiles": self.fit_emitted_delta_ratio_quantiles,
            "fit_max_bounded_direction_ratio": self.fit_max_bounded_direction_ratio,
            "fit_max_emitted_delta_ratio": self.fit_max_emitted_delta_ratio,
            "zero_parent_row_count": self.zero_parent_row_count,
            "zero_direction_row_count": self.zero_direction_row_count,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": (
                "native_h4",
                "targets",
                "logits",
                "gradients",
                "coordinate_axes",
                "family_ids",
            ),
            "coordinate_semantics": "u_div_scale_plus_abs_u",
            "direction_feature_semantics": "c1_p_c2_p_c1_c2_p",
            "bounded_direction_semantics": (
                "q_times_min_one_beta_parent_norm_over_q_norm_zero_safe_no_amplification"
            ),
            "pedal_semantics": "clamp_bias_plus_c1_c2_c1c2_dot_weight_zero_one",
            "pedal_fit_semantics": (
                "raw_residual_projection_scalar_weighted_by_fit_weight_times_b_norm_"
                "squared_with_centered_ridge_slopes_and_unpenalized_intercept"
            ),
            "pedal_support_semantics": (
                "bounded_direction_energy_strictly_above_max_float64_tiny_and_"
                "float64_epsilon_times_parent_modal_energy"
            ),
            "pedal_relative_energy_floor": FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR,
            "pedal_absolute_energy_floor": FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR,
            "certificate_scope": (
                "pointwise_emitted_modal_amplitude_not_full_nonlinear_jacobian_or_lipschitz"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        self.parent_provider.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("Fisher-XY pedal provider tensor payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": self.incremental_prepared_float_scalar_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": (
                self.incremental_prepared_float_scalar_count * 8
            ),
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "runtime_parameter_bytes_float64_scope": (
                "prepared_float_payload_not_peak_runtime_memory"
            ),
            "incremental_logical_macs_per_token_upper_bound": (
                self.incremental_logical_macs_per_token_upper_bound
            ),
            "logical_macs_per_token_upper_bound": self.logical_macs_per_token_upper_bound,
            "logical_macs_accounting_scope": (
                "matrix_multiply_accumulates_excluding_norms_integrity_hashes_"
                "device_transfers_and_temporary_workspace"
            ),
            "runtime_state_float_scalars_per_sequence": 0,
            "router_bias_additions_per_token": 2,
            "pedal_bias_additions_per_token": 1,
            "rational_coordinate_abs_ops_per_token": 2,
            "rational_coordinate_add_ops_per_token": 2,
            "rational_coordinate_division_ops_per_token": 2,
            "rational_coordinate_open_bound_clamp_ops_per_token": 4,
            "direction_feature_scalar_multiplications_per_token": 3 * self.rank + 1,
            "pointwise_norm_square_multiplications_per_token": 2 * self.rank,
            "pointwise_norm_reduction_additions_per_token": 2 * max(0, self.rank - 1),
            "pointwise_norm_square_roots_per_token": 2,
            "pointwise_bound_divisions_per_token": 1,
            "pointwise_bound_radius_multiplications_per_token": 1,
            "pointwise_bound_minimum_ops_per_token": 1,
            "pointwise_direction_scale_multiplications_per_token": self.rank,
            "pedal_feature_scalar_multiplications_per_token": 1,
            "pedal_clamp_ops_per_token": 2,
            "emitted_delta_scalar_multiplications_per_token": self.rank,
            "modal_additions_per_token": self.rank,
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("Fisher-XY pedal modal router input differs")
        values = parent_modal.to(dtype=torch.float64)
        flat = values.reshape(-1, self.rank)
        raw = flat @ self.router_weight.to(flat.device) + self.router_bias.to(flat.device)
        bounded = fisher_xy_bounded_coordinates(
            raw,
            self.coordinate_scales.to(raw.device),
        )
        return bounded.reshape(
            *parent_modal.shape[:-1],
            FISHER_XY_COORDINATE_COUNT,
        )

    def unbounded_direction(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("Fisher-XY pedal parent modal differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded_coordinates = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if (
            bounded_coordinates.shape
            != (*parent.shape[:-1], FISHER_XY_COORDINATE_COUNT)
            or not bool(torch.isfinite(bounded_coordinates).all())
            or bool((bounded_coordinates.abs() >= 1.0).any())
        ):
            raise ValueError("Fisher-XY pedal coordinate geometry differs")
        c1 = bounded_coordinates[..., :1]
        c2 = bounded_coordinates[..., 1:]
        features = torch.cat((c1 * parent, c2 * parent, c1 * c2 * parent), dim=-1)
        direction = (
            features @ self.direction_left.to(features.device)
        ) @ self.direction_right.to(features.device)
        if not bool(torch.isfinite(direction).all()):
            raise RuntimeError("Fisher-XY pedal direction became nonfinite")
        return direction.contiguous()

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        features = fisher_xy_pedal_features(coordinates)
        values = (
            features @ self.pedal_weight.to(features.device)
            + self.pedal_bias.to(features.device)[0]
        ).clamp(0.0, 1.0)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("Fisher-XY pedal became nonfinite")
        return values.contiguous()

    def _modal_terms(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        parent = self.parent_provider.modal_correction(prefix, realized_state)
        coordinates = self.bounded_coordinates(parent)
        direction = self.unbounded_direction(parent, coordinates)
        bounded = fisher_xy_pointwise_bounded_direction(
            parent,
            direction,
            trust_fraction=self.trust_fraction,
        )
        pedal = self.pedal_values(coordinates)
        delta = pedal.unsqueeze(-1) * bounded
        return parent, coordinates, direction, bounded, pedal, delta

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        prefix_sha = prefix.artifact_sha256
        realized_sha = _tensor_sha256(realized_state)
        parent, _coordinates, _direction, _bounded, _pedal, delta = (
            self._modal_terms(prefix, realized_state)
        )
        modal = parent + delta
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
        if (
            prefix.artifact_sha256 != prefix_sha
            or _tensor_sha256(realized_state) != realized_sha
        ):
            raise RuntimeError("Fisher-XY pedal provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("Fisher-XY pedal modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("Fisher-XY pedal modal correction escaped support")
        self.validate_integrity()
        prefix.validate_integrity()
        return modal.contiguous()

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        return self.parent_provider.decode_modal(
            prefix,
            self.modal_correction(prefix, realized_state),
            like=realized_state,
        )


def replay_autonomous_complete_h4_fisher_xy_pedal(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
    sequence: AutonomousCompleteH4TrainingSequence,
) -> FisherXYPedalRuntimeReplay:
    """Replay support rows using only source modes, masks, positions, and base H4."""

    if not isinstance(provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("provider must be autonomous complete-H4 Fisher-XY pedal")
    if not isinstance(sequence, AutonomousCompleteH4TrainingSequence):
        raise TypeError("sequence must be autonomous complete-H4 training data")
    _validate_training_sequence_integrity(sequence)
    provider.validate_integrity()
    parent = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent)
    direction = provider.unbounded_direction(parent, coordinates)
    bounded = fisher_xy_pointwise_bounded_direction(
        parent,
        direction,
        trust_fraction=provider.trust_fraction,
    )
    pedal = provider.pedal_values(coordinates)
    delta = pedal.unsqueeze(1) * bounded
    support = sequence.support_mask
    replay = FisherXYPedalRuntimeReplay(
        provider_artifact_sha256=provider.artifact_sha256,
        parent_provider_artifact_sha256=provider.parent_provider.artifact_sha256,
        sequence_artifact_sha256=sequence.artifact_sha256,
        trust_fraction=provider.trust_fraction,
        parent_modal=parent[support],
        bounded_coordinates=coordinates[support],
        unbounded_direction=direction[support],
        bounded_direction=bounded[support],
        pedal=pedal[support],
        emitted_delta=delta[support],
    )
    replay.validate_integrity()
    provider.validate_integrity()
    return replay


def _clip_scales(
    parent_modal: Tensor,
    candidate: Tensor,
    *,
    trust_fraction: float,
) -> Tensor:
    parent_norm = torch.linalg.vector_norm(parent_modal, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=1)
    nonzero = candidate_norm > 0.0
    safe_norm = torch.where(nonzero, candidate_norm, torch.ones_like(candidate_norm))
    scales = torch.where(
        nonzero,
        torch.minimum(
            torch.ones_like(candidate_norm),
            trust_fraction * parent_norm / safe_norm,
        ),
        torch.ones_like(candidate_norm),
    )
    return scales.contiguous()


def _weighted_modal_rmse(error: Tensor, weights: Tensor, *, rank: int) -> float:
    value = torch.sqrt((weights.unsqueeze(1) * error.square()).sum() / rank)
    if not bool(torch.isfinite(value)):
        raise RuntimeError("Fisher-XY pedal weighted RMSE became nonfinite")
    return float(value)


def _realized_convex_weighted_mean(
    values: Tensor,
    weights: Tensor,
    *,
    label: str,
) -> tuple[float, Tensor]:
    """Normalize by the realized weight sum and contain only roundoff drift."""

    weight_total = weights.sum()
    if not bool(torch.isfinite(weight_total)) or not bool(weight_total > 0.0):
        raise RuntimeError(f"{label} weight total became invalid")
    mean = float((weights * values).sum() / weight_total)
    minimum = float(values.min())
    maximum = float(values.max())
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        1.0,
        abs(minimum),
        abs(maximum),
    )
    if mean < minimum - tolerance or mean > maximum + tolerance:
        raise RuntimeError(f"{label} escaped its observed convex range")
    return min(max(mean, minimum), maximum), weight_total


def _relative_norms(candidate: Tensor, parent: Tensor) -> Tensor:
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=1)
    nonzero = parent_norm > 0.0
    ratios = torch.where(
        nonzero,
        candidate_norm
        / torch.where(nonzero, parent_norm, torch.ones_like(parent_norm)),
        torch.zeros_like(parent_norm),
    )
    return ratios.contiguous()


def fit_autonomous_complete_h4_fisher_xy_pedal(
    *,
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    parent_provider: AutonomousCompleteH4ResidualProvider,
    conditional_rank: int = 16,
    coordinate_objective: str = "reverse_vjp_fisher",
    pedal_mode: str = "conditional",
    router_ridge: float = 1.0e-4,
    direction_ridge: float = 1.0e-4,
    pedal_ridge: float = 1.0e-4,
    trust_fraction: float = FISHER_XY_PEDAL_TRUST_FRACTION,
    vjp_weight_floor: float = 0.5,
    vjp_weight_ceiling: float = 2.0,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    """Fit one deterministic family-balanced pointwise-bounded pedal provider."""

    if (
        isinstance(sequences, (str, bytes))
        or not isinstance(sequences, Sequence)
        or not sequences
    ):
        raise ValueError("fit requires autonomous complete-H4 training sequences")
    if any(
        not isinstance(value, AutonomousCompleteH4TrainingSequence)
        for value in sequences
    ):
        raise TypeError("fit sequence type differs")
    for sequence in sequences:
        _validate_training_sequence_integrity(sequence)
    ordered = _ordered_sequences(sequences)
    if not isinstance(parent_provider, AutonomousCompleteH4ResidualProvider):
        raise TypeError("parent_provider must be autonomous complete-H4")
    parent_provider.validate_integrity()
    expected_sequence_ids = tuple(sorted(value.artifact_sha256 for value in ordered))
    expected_families = tuple(sorted({value.family_id for value in ordered}))
    if (
        parent_provider.fit_sequence_sha256s != expected_sequence_ids
        or parent_provider.fit_family_ids != expected_families
    ):
        raise ValueError("Fisher-XY pedal parent and fit ownership differ")
    if (
        type(conditional_rank) is not int
        or conditional_rank <= 0
        or conditional_rank > parent_provider.rank
    ):
        raise ValueError("conditional_rank must lie in [1, parent rank]")
    if coordinate_objective not in COORDINATE_OBJECTIVES:
        raise ValueError("Fisher-XY pedal coordinate objective differs")
    if pedal_mode not in FISHER_XY_PEDAL_MODES:
        raise ValueError("pedal_mode must be conditional, constant_optimal, or unit")
    selected_router_ridge = _positive_float(router_ridge, label="router_ridge")
    selected_direction_ridge = _positive_float(
        direction_ridge,
        label="direction_ridge",
    )
    selected_pedal_ridge = _positive_float(pedal_ridge, label="pedal_ridge")
    selected_trust = _positive_float(trust_fraction, label="trust_fraction")
    if selected_trust != FISHER_XY_PEDAL_TRUST_FRACTION:
        raise ValueError("Fisher-XY pedal trust_fraction is frozen at 0.25")
    if not (0.0 < vjp_weight_floor <= 1.0 <= vjp_weight_ceiling):
        raise ValueError("VJP multiplier bounds must bracket one")

    modal_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    gradient_modal_rows: list[Tensor] = []
    gradient_h4_rows: list[Tensor] = []
    families: list[str] = []
    examples: list[str] = []
    for sequence in ordered:
        if sequence.reverse_vjp_gradients is None:
            raise ValueError(
                "Fisher-XY pedal and matched PCA fits require every reverse-VJP gradient"
            )
        selected = sequence.support_mask
        parent_modal = _training_parent_modal(parent_provider, sequence)
        target = (
            sequence.native_h4 - sequence.base_h4
        ) @ parent_provider.output_decoder.T
        gradient_modal = (
            sequence.reverse_vjp_gradients @ parent_provider.output_decoder.T
        )
        modal_rows.append(parent_modal[selected])
        target_rows.append(target[selected])
        gradient_modal_rows.append(gradient_modal[selected])
        gradient_h4_rows.append(sequence.reverse_vjp_gradients[selected])
        count = int(selected.sum())
        families.extend([sequence.family_id] * count)
        examples.extend([sequence.example_id] * count)

    modal = torch.cat(modal_rows, dim=0).to(torch.float64)
    target = torch.cat(target_rows, dim=0).to(torch.float64)
    gradient_modal = torch.cat(gradient_modal_rows, dim=0).to(torch.float64)
    gradient_h4 = torch.cat(gradient_h4_rows, dim=0).to(torch.float64)
    family_tuple = tuple(families)
    example_tuple = tuple(examples)
    base_weights = family_balanced_row_weights(family_tuple, example_tuple).to(
        torch.float64
    )
    axes, coordinate_targets, axis_values = _coordinate_axes_and_targets(
        coordinate_objective=coordinate_objective,
        modal=modal,
        gradient_modal=gradient_modal,
        weights=base_weights,
    )
    router_design = torch.cat(
        (modal, torch.ones((modal.shape[0], 1), dtype=torch.float64)),
        dim=1,
    )
    router_coefficients, _router_scales = _weighted_ridge(
        router_design,
        coordinate_targets,
        base_weights,
        ridge=selected_router_ridge,
    )
    raw_coordinates = router_design @ router_coefficients
    coordinate_scales = torch.sqrt(
        (base_weights.unsqueeze(1) * raw_coordinates.square()).sum(dim=0)
    )
    coordinate_scales = torch.where(
        coordinate_scales > math.sqrt(torch.finfo(torch.float64).eps),
        coordinate_scales,
        torch.ones_like(coordinate_scales),
    ).contiguous()
    coordinates = fisher_xy_bounded_coordinates(raw_coordinates, coordinate_scales)
    c1 = coordinates[:, :1]
    c2 = coordinates[:, 1:]
    direction_features = torch.cat(
        (c1 * modal, c2 * modal, c1 * c2 * modal),
        dim=1,
    )
    residual = target - modal
    bounded_target = fisher_xy_pointwise_bounded_direction(
        modal,
        residual,
        trust_fraction=selected_trust,
    )
    multipliers = _bounded_vjp_multipliers(
        gradient_h4,
        example_tuple,
        floor=vjp_weight_floor,
        ceiling=vjp_weight_ceiling,
    )
    fit_weights = (base_weights * multipliers).contiguous()
    fit_weights = fit_weights / fit_weights.sum()

    dense_direction, _direction_scales = _weighted_ridge(
        direction_features,
        bounded_target,
        fit_weights,
        ridge=selected_direction_ridge,
    )
    u, singular, vh = torch.linalg.svd(dense_direction, full_matrices=False)
    if conditional_rank > int(singular.numel()):
        raise ValueError("conditional_rank exceeds fitted direction matrix rank bound")
    left = (u[:, :conditional_rank] * singular[:conditional_rank]).contiguous()
    right = vh[:conditional_rank].contiguous()
    for index in range(conditional_rank):
        pivot = int(right[index].abs().argmax())
        if float(right[index, pivot]) < 0.0:
            right[index].neg_()
            left[:, index].neg_()
    unbounded_direction = (direction_features @ left) @ right
    bounded_direction = fisher_xy_pointwise_bounded_direction(
        modal,
        unbounded_direction,
        trust_fraction=selected_trust,
    )

    bounded_norm_squared = bounded_direction.square().sum(dim=1)
    pedal_supported = fisher_xy_pedal_fit_support_mask(
        modal,
        bounded_direction,
    )
    if not bool(pedal_supported.any()):
        raise ValueError("Fisher-XY pedal fit requires nonzero bounded direction energy")
    analytic_numerator = (residual * bounded_direction).sum(dim=1)
    safe_energy = torch.where(
        pedal_supported,
        bounded_norm_squared,
        torch.ones_like(bounded_norm_squared),
    )
    pedal_unclipped_target = torch.where(
        pedal_supported,
        analytic_numerator / safe_energy,
        torch.zeros_like(analytic_numerator),
    )
    if not bool(torch.isfinite(pedal_unclipped_target).all()):
        raise ValueError("Fisher-XY analytic pedal target became nonfinite")
    pedal_target = pedal_unclipped_target.clamp(0.0, 1.0)
    raw_pedal_fit_weights = torch.where(
        pedal_supported,
        fit_weights * bounded_norm_squared,
        torch.zeros_like(fit_weights),
    )
    pedal_fit_weight_total = raw_pedal_fit_weights.sum()
    if not bool(torch.isfinite(pedal_fit_weight_total)) or float(
        pedal_fit_weight_total
    ) <= 0.0:
        raise ValueError("Fisher-XY pedal fit weight has no positive mass")
    pedal_fit_weights = (
        raw_pedal_fit_weights / pedal_fit_weight_total
    ).contiguous()
    realized_pedal_fit_weight_total = pedal_fit_weights.sum()
    if not bool(torch.isfinite(realized_pedal_fit_weight_total)) or not bool(
        realized_pedal_fit_weight_total > 0.0
    ):
        raise RuntimeError("realized Fisher-XY pedal fit weight became invalid")
    pedal_features = fisher_xy_pedal_features(coordinates)
    if pedal_mode == "conditional":
        selected_rows = pedal_fit_weights > 0.0
        selected_weights = pedal_fit_weights[selected_rows]
        selected_weights = selected_weights / selected_weights.sum()
        selected_features = pedal_features[selected_rows]
        selected_target = pedal_unclipped_target[selected_rows]
        feature_mean = (
            selected_weights.unsqueeze(1) * selected_features
        ).sum(dim=0)
        target_mean = (selected_weights * selected_target).sum()
        centered_features = selected_features - feature_mean
        centered_target = selected_target - target_mean
        pedal_coefficients, _pedal_scales = _weighted_ridge(
            centered_features,
            centered_target.unsqueeze(1),
            selected_weights,
            ridge=selected_pedal_ridge,
        )
        pedal_weight = pedal_coefficients[:, 0].contiguous()
        pedal_bias = (
            target_mean - feature_mean @ pedal_weight
        ).reshape(1).contiguous()
    elif pedal_mode == "constant_optimal":
        pedal_weight = torch.zeros(_PEDAL_FEATURE_COUNT, dtype=torch.float64)
        pedal_bias = torch.tensor(
            [
                float(
                    (
                        (pedal_fit_weights * pedal_unclipped_target).sum()
                        / realized_pedal_fit_weight_total
                    )
                    .clamp(0.0, 1.0)
                )
            ],
            dtype=torch.float64,
        )
    else:
        pedal_weight = torch.zeros(_PEDAL_FEATURE_COUNT, dtype=torch.float64)
        pedal_bias = torch.ones(1, dtype=torch.float64)
    pedal = (pedal_features @ pedal_weight + pedal_bias[0]).clamp(0.0, 1.0)
    constant_optimal_pedal = float(
        (
            (pedal_fit_weights * pedal_unclipped_target).sum()
            / realized_pedal_fit_weight_total
        ).clamp(0.0, 1.0)
    )
    emitted_delta = pedal.unsqueeze(1) * bounded_direction
    constant_delta = constant_optimal_pedal * bounded_direction
    oracle_delta = pedal_target.unsqueeze(1) * bounded_direction

    bounded_coordinate_targets = fisher_xy_bounded_coordinates(
        coordinate_targets,
        coordinate_scales,
    )
    coordinate_geometry = summarize_fisher_xy_bounded_coordinate_geometry(
        coordinates,
        fit_weights,
    )
    target_r2 = _bounded_coordinate_target_r2(
        coordinates,
        bounded_coordinate_targets,
        fit_weights,
    )
    router_error = coordinate_targets - raw_coordinates
    router_rmse = torch.sqrt(
        (base_weights.unsqueeze(1) * router_error.square()).sum()
        / FISHER_XY_COORDINATE_COUNT
    )
    pedal_error = pedal - pedal_target
    pedal_unclipped_error = pedal - pedal_unclipped_target
    pedal_unclipped_target_mean = float(
        (pedal_fit_weights * pedal_unclipped_target).sum()
        / realized_pedal_fit_weight_total
    )
    pedal_unclipped_target_rmse = float(
        torch.sqrt(
            (pedal_fit_weights * pedal_unclipped_error.square()).sum()
            / realized_pedal_fit_weight_total
        )
    )
    pedal_target_mean = float(
        (pedal_fit_weights * pedal_target).sum()
        / realized_pedal_fit_weight_total
    )
    pedal_target_rmse = float(
        torch.sqrt(
            (pedal_fit_weights * pedal_error.square()).sum()
            / realized_pedal_fit_weight_total
        )
    )
    effective_pedal = pedal[pedal_supported]
    if pedal_mode == "conditional":
        pedal_weighted_mean, realized_fit_weight_total = (
            _realized_convex_weighted_mean(
                pedal,
                fit_weights,
                label="conditional pedal",
            )
        )
        pedal_weighted_std = float(
            torch.sqrt(
                (
                    fit_weights * (pedal - pedal_weighted_mean).square()
                ).sum()
                / realized_fit_weight_total
            )
        )
        pedal_effective_weighted_mean, realized_pedal_weight_total = (
            _realized_convex_weighted_mean(
                effective_pedal,
                pedal_fit_weights[pedal_supported],
                label="conditional effective pedal",
            )
        )
        pedal_effective_weighted_std = float(
            torch.sqrt(
                (
                    pedal_fit_weights[pedal_supported]
                    * (
                        effective_pedal - pedal_effective_weighted_mean
                    ).square()
                ).sum()
                / realized_pedal_weight_total
            )
        )
        pedal_min = float(pedal.min())
        pedal_max = float(pedal.max())
        pedal_effective_min = float(effective_pedal.min())
        pedal_effective_max = float(effective_pedal.max())
        pedal_zero_fraction = min(
            max(
                float(
                    (fit_weights * (pedal == 0.0)).sum()
                    / realized_fit_weight_total
                ),
                0.0,
            ),
            1.0,
        )
        pedal_one_fraction = min(
            max(
                float(
                    (fit_weights * (pedal == 1.0)).sum()
                    / realized_fit_weight_total
                ),
                0.0,
            ),
            1.0,
        )
    else:
        # These modes are constant by construction.  Derive their descriptive
        # statistics from that exact serving value rather than a normalized
        # floating-weight sum whose total can differ microscopically from one.
        exact_pedal = 1.0 if pedal_mode == "unit" else float(pedal_bias[0])
        pedal_weighted_mean = exact_pedal
        pedal_weighted_std = 0.0
        pedal_effective_weighted_mean = exact_pedal
        pedal_effective_weighted_std = 0.0
        pedal_min = pedal_max = exact_pedal
        pedal_effective_min = pedal_effective_max = exact_pedal
        pedal_zero_fraction = 1.0 if exact_pedal == 0.0 else 0.0
        pedal_one_fraction = 1.0 if exact_pedal == 1.0 else 0.0
    target_clip_scales = _clip_scales(
        modal,
        residual,
        trust_fraction=selected_trust,
    )
    direction_clip_scales = _clip_scales(
        modal,
        unbounded_direction,
        trust_fraction=selected_trust,
    )
    bounded_ratios = _relative_norms(bounded_direction, modal)
    emitted_ratios = _relative_norms(emitted_delta, modal)
    quantile_points = torch.tensor((0.5, 0.9, 1.0), dtype=torch.float64)
    bounded_ratio_quantiles = tuple(
        float(value) for value in torch.quantile(bounded_ratios, quantile_points)
    )
    emitted_ratio_quantiles = tuple(
        float(value) for value in torch.quantile(emitted_ratios, quantile_points)
    )
    fit_max_bounded_ratio = bounded_ratio_quantiles[-1]
    fit_max_delta_ratio = emitted_ratio_quantiles[-1]
    tolerance = 64.0 * torch.finfo(torch.float64).eps
    if (
        fit_max_bounded_ratio > selected_trust + tolerance
        or fit_max_delta_ratio > selected_trust + tolerance
    ):
        raise RuntimeError("Fisher-XY pedal fit escaped pointwise trust")

    fit_receipt = {
        "parent_provider_artifact_sha256": parent_provider.artifact_sha256,
        "fit_sequence_sha256s": expected_sequence_ids,
        "fit_family_ids": expected_families,
        "coordinate_objective": coordinate_objective,
        "coordinate_axes_sha256": _tensor_sha256(axes),
        "coordinate_axis_values": axis_values,
        "router_ridge": selected_router_ridge,
        "direction_ridge": selected_direction_ridge,
        "pedal_ridge": selected_pedal_ridge,
        "conditional_rank": conditional_rank,
        "pedal_mode": pedal_mode,
        "trust_fraction": selected_trust,
        "vjp_weight_floor": float(vjp_weight_floor),
        "vjp_weight_ceiling": float(vjp_weight_ceiling),
        "fit_weight_sha256": _tensor_sha256(fit_weights),
        "pedal_unclipped_target_sha256": _tensor_sha256(
            pedal_unclipped_target
        ),
        "pedal_target_sha256": _tensor_sha256(pedal_target),
        "pedal_fit_weight_sha256": _tensor_sha256(pedal_fit_weights),
        "pedal_support_mask_sha256": _tensor_sha256(pedal_supported),
        "pedal_supported_row_count": int(pedal_supported.sum()),
        "pedal_relative_energy_floor": FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR,
        "pedal_absolute_energy_floor": FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR,
        "bounded_target_sha256": _tensor_sha256(bounded_target),
        "unbounded_direction_sha256": _tensor_sha256(unbounded_direction),
        "bounded_direction_sha256": _tensor_sha256(bounded_direction),
        "bounded_coordinate_geometry_sha256": coordinate_geometry.artifact_sha256,
        "bounded_direction_semantics": (
            "q_times_min_one_beta_parent_norm_over_q_norm_zero_safe_no_amplification"
        ),
        "pedal_target_semantics": (
            "raw_residual_dot_b_over_b_norm_squared_train_only_with_clipped_"
            "target_retained_as_diagnostic"
        ),
        "pedal_fit_weight_semantics": "normalized_fit_weight_times_b_norm_squared",
        "conditional_pedal_fit_semantics": (
            "weighted_centered_raw_target_ridge_slopes_unpenalized_intercept"
        ),
    }
    return AutonomousCompleteH4FisherXYPedalProvider(
        parent_provider=parent_provider,
        router_weight=router_coefficients[:-1],
        router_bias=router_coefficients[-1],
        coordinate_scales=coordinate_scales,
        direction_left=left,
        direction_right=right,
        pedal_weight=pedal_weight,
        pedal_bias=pedal_bias,
        router_ridge=selected_router_ridge,
        direction_ridge=selected_direction_ridge,
        pedal_ridge=selected_pedal_ridge,
        trust_fraction=selected_trust,
        fit_row_count=int(modal.shape[0]),
        fit_family_ids=expected_families,
        fit_sequence_sha256s=expected_sequence_ids,
        coordinate_objective=coordinate_objective,
        coordinate_axes_sha256=_tensor_sha256(axes),
        coordinate_axis_values=axis_values,
        fit_weight_sha256=_tensor_sha256(fit_weights),
        coordinate_target_weighted_rmse=float(router_rmse),
        bounded_coordinate_geometry_sha256=coordinate_geometry.artifact_sha256,
        bounded_coordinate_covariance_eigenvalues=coordinate_geometry.covariance_eigenvalues,
        bounded_coordinate_lambda2_over_lambda1=coordinate_geometry.lambda2_over_lambda1,
        bounded_coordinate_abs_correlation=coordinate_geometry.abs_correlation,
        bounded_coordinate_target_r2=target_r2,
        residual_second_coordinate_energy_fraction=(
            coordinate_geometry.residual_second_coordinate_energy_fraction
        ),
        pedal_mode=pedal_mode,
        pedal_unclipped_target_sha256=_tensor_sha256(pedal_unclipped_target),
        pedal_target_sha256=_tensor_sha256(pedal_target),
        pedal_fit_weight_sha256=_tensor_sha256(pedal_fit_weights),
        pedal_support_mask_sha256=_tensor_sha256(pedal_supported),
        pedal_supported_row_count=int(pedal_supported.sum()),
        pedal_unclipped_target_weighted_mean=pedal_unclipped_target_mean,
        pedal_unclipped_target_weighted_rmse=pedal_unclipped_target_rmse,
        pedal_target_weighted_mean=pedal_target_mean,
        pedal_target_weighted_rmse=pedal_target_rmse,
        pedal_weighted_mean=pedal_weighted_mean,
        pedal_weighted_std=pedal_weighted_std,
        pedal_min=pedal_min,
        pedal_max=pedal_max,
        pedal_effective_weighted_mean=pedal_effective_weighted_mean,
        pedal_effective_weighted_std=pedal_effective_weighted_std,
        pedal_effective_min=pedal_effective_min,
        pedal_effective_max=pedal_effective_max,
        pedal_zero_fraction=pedal_zero_fraction,
        pedal_one_fraction=pedal_one_fraction,
        pedal_target_clipped_fraction=float(
            (
                pedal_fit_weights
                * ((pedal_unclipped_target < 0.0) | (pedal_unclipped_target > 1.0))
            ).sum()
            / realized_pedal_fit_weight_total
        ),
        bounded_target_clipped_fraction=float(
            (fit_weights * (target_clip_scales < 1.0)).sum()
        ),
        bounded_target_mean_clip_scale=float(
            (fit_weights * target_clip_scales).sum()
        ),
        direction_clipped_fraction=float(
            (fit_weights * (direction_clip_scales < 1.0)).sum()
        ),
        direction_mean_clip_scale=float(
            (fit_weights * direction_clip_scales).sum()
        ),
        weighted_bounded_target_rmse_before=_weighted_modal_rmse(
            bounded_target,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_bounded_target_rmse_after=_weighted_modal_rmse(
            bounded_target - bounded_direction,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_residual_rmse_before=_weighted_modal_rmse(
            residual,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_residual_rmse_constant=_weighted_modal_rmse(
            residual - constant_delta,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_residual_rmse_unit=_weighted_modal_rmse(
            residual - bounded_direction,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_residual_rmse_oracle=_weighted_modal_rmse(
            residual - oracle_delta,
            fit_weights,
            rank=parent_provider.rank,
        ),
        weighted_residual_rmse_after=_weighted_modal_rmse(
            residual - emitted_delta,
            fit_weights,
            rank=parent_provider.rank,
        ),
        fit_bounded_direction_ratio_quantiles=bounded_ratio_quantiles,
        fit_emitted_delta_ratio_quantiles=emitted_ratio_quantiles,
        fit_max_bounded_direction_ratio=fit_max_bounded_ratio,
        fit_max_emitted_delta_ratio=fit_max_delta_ratio,
        zero_parent_row_count=int(
            (torch.linalg.vector_norm(modal, dim=1) == 0.0).sum()
        ),
        zero_direction_row_count=int(
            (torch.linalg.vector_norm(unbounded_direction, dim=1) == 0.0).sum()
        ),
        fit_receipt_sha256=_sha256(_FIT_RECEIPT_DOMAIN, fit_receipt),
    )


def autonomous_complete_h4_fisher_xy_pedal_provider_state_dict(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
) -> dict[str, object]:
    if not isinstance(provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("provider must be AutonomousCompleteH4FisherXYPedalProvider")
    provider.validate_integrity()
    return {
        "schema": _STATE_SCHEMA,
        "format_version": 1,
        "provider_artifact_sha256": provider.artifact_sha256,
        "bridge_binding_sha256": provider.bridge_binding_sha256,
        "parent_provider_artifact_sha256": provider.parent_provider.artifact_sha256,
        "router_ridge": provider.router_ridge,
        "direction_ridge": provider.direction_ridge,
        "pedal_ridge": provider.pedal_ridge,
        "trust_fraction": provider.trust_fraction,
        "fit_row_count": provider.fit_row_count,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "coordinate_objective": provider.coordinate_objective,
        "coordinate_axes_sha256": provider.coordinate_axes_sha256,
        "coordinate_axis_values": provider.coordinate_axis_values,
        "fit_weight_sha256": provider.fit_weight_sha256,
        "coordinate_target_weighted_rmse": provider.coordinate_target_weighted_rmse,
        "bounded_coordinate_geometry_sha256": provider.bounded_coordinate_geometry_sha256,
        "bounded_coordinate_covariance_eigenvalues": provider.bounded_coordinate_covariance_eigenvalues,
        "bounded_coordinate_lambda2_over_lambda1": provider.bounded_coordinate_lambda2_over_lambda1,
        "bounded_coordinate_abs_correlation": provider.bounded_coordinate_abs_correlation,
        "bounded_coordinate_target_r2": provider.bounded_coordinate_target_r2,
        "residual_second_coordinate_energy_fraction": provider.residual_second_coordinate_energy_fraction,
        "pedal_mode": provider.pedal_mode,
        "pedal_unclipped_target_sha256": provider.pedal_unclipped_target_sha256,
        "pedal_target_sha256": provider.pedal_target_sha256,
        "pedal_fit_weight_sha256": provider.pedal_fit_weight_sha256,
        "pedal_support_mask_sha256": provider.pedal_support_mask_sha256,
        "pedal_supported_row_count": provider.pedal_supported_row_count,
        "pedal_unclipped_target_weighted_mean": provider.pedal_unclipped_target_weighted_mean,
        "pedal_unclipped_target_weighted_rmse": provider.pedal_unclipped_target_weighted_rmse,
        "pedal_target_weighted_mean": provider.pedal_target_weighted_mean,
        "pedal_target_weighted_rmse": provider.pedal_target_weighted_rmse,
        "pedal_weighted_mean": provider.pedal_weighted_mean,
        "pedal_weighted_std": provider.pedal_weighted_std,
        "pedal_min": provider.pedal_min,
        "pedal_max": provider.pedal_max,
        "pedal_effective_weighted_mean": provider.pedal_effective_weighted_mean,
        "pedal_effective_weighted_std": provider.pedal_effective_weighted_std,
        "pedal_effective_min": provider.pedal_effective_min,
        "pedal_effective_max": provider.pedal_effective_max,
        "pedal_zero_fraction": provider.pedal_zero_fraction,
        "pedal_one_fraction": provider.pedal_one_fraction,
        "pedal_target_clipped_fraction": provider.pedal_target_clipped_fraction,
        "bounded_target_clipped_fraction": provider.bounded_target_clipped_fraction,
        "bounded_target_mean_clip_scale": provider.bounded_target_mean_clip_scale,
        "direction_clipped_fraction": provider.direction_clipped_fraction,
        "direction_mean_clip_scale": provider.direction_mean_clip_scale,
        "weighted_bounded_target_rmse_before": provider.weighted_bounded_target_rmse_before,
        "weighted_bounded_target_rmse_after": provider.weighted_bounded_target_rmse_after,
        "weighted_residual_rmse_before": provider.weighted_residual_rmse_before,
        "weighted_residual_rmse_constant": provider.weighted_residual_rmse_constant,
        "weighted_residual_rmse_unit": provider.weighted_residual_rmse_unit,
        "weighted_residual_rmse_oracle": provider.weighted_residual_rmse_oracle,
        "weighted_residual_rmse_after": provider.weighted_residual_rmse_after,
        "fit_bounded_direction_ratio_quantiles": provider.fit_bounded_direction_ratio_quantiles,
        "fit_emitted_delta_ratio_quantiles": provider.fit_emitted_delta_ratio_quantiles,
        "fit_max_bounded_direction_ratio": provider.fit_max_bounded_direction_ratio,
        "fit_max_emitted_delta_ratio": provider.fit_max_emitted_delta_ratio,
        "zero_parent_row_count": provider.zero_parent_row_count,
        "zero_direction_row_count": provider.zero_direction_row_count,
        "fit_receipt_sha256": provider.fit_receipt_sha256,
        "parent_provider_state": autonomous_complete_h4_residual_provider_state_dict(
            provider.parent_provider
        ),
        "tensors": {
            "router_weight": provider.router_weight.detach().clone(),
            "router_bias": provider.router_bias.detach().clone(),
            "coordinate_scales": provider.coordinate_scales.detach().clone(),
            "direction_left": provider.direction_left.detach().clone(),
            "direction_right": provider.direction_right.detach().clone(),
            "pedal_weight": provider.pedal_weight.detach().clone(),
            "pedal_bias": provider.pedal_bias.detach().clone(),
        },
    }


def autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict(
    state: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("Fisher-XY pedal provider state fields differ")
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected Fisher-XY pedal provider",
    )
    embedded_artifact = _require_sha256(
        state.get("provider_artifact_sha256"),
        label="embedded Fisher-XY pedal provider",
    )
    if embedded_artifact != expected_artifact:
        raise ValueError("Fisher-XY pedal state artifact differs from expected")
    bridge = _require_sha256(
        state.get("bridge_binding_sha256"),
        label="Fisher-XY pedal bridge binding",
    )
    if expected_bridge_binding_sha256 is not None and bridge != _require_sha256(
        expected_bridge_binding_sha256,
        label="expected Fisher-XY pedal bridge binding",
    ):
        raise ValueError("Fisher-XY pedal bridge binding differs from expected")
    parent_artifact = _require_sha256(
        state.get("parent_provider_artifact_sha256"),
        label="Fisher-XY pedal parent provider",
    )
    parent_state = state.get("parent_provider_state")
    if not isinstance(parent_state, Mapping):
        raise ValueError("Fisher-XY pedal parent state must be a mapping")
    parent = autonomous_complete_h4_residual_provider_from_state_dict(
        parent_state,
        expected_artifact_sha256=parent_artifact,
        expected_bridge_binding_sha256=bridge,
    )
    tensors = state.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("Fisher-XY pedal state tensor fields differ")
    if any(not isinstance(tensors.get(name), Tensor) for name in _TENSOR_KEYS):
        raise ValueError("Fisher-XY pedal state tensors must be tensors")
    if (
        state.get("schema") != _STATE_SCHEMA
        or state.get("format_version") != 1
        or type(state.get("fit_row_count")) is not int
        or type(state.get("fit_family_ids")) is not tuple
        or type(state.get("fit_sequence_sha256s")) is not tuple
        or type(state.get("zero_parent_row_count")) is not int
        or type(state.get("zero_direction_row_count")) is not int
        or type(state.get("pedal_supported_row_count")) is not int
        or not isinstance(state.get("coordinate_objective"), str)
        or not isinstance(state.get("pedal_mode"), str)
    ):
        raise ValueError("Fisher-XY pedal state scalar contract differs")
    return AutonomousCompleteH4FisherXYPedalProvider(
        parent_provider=parent,
        router_weight=tensors["router_weight"],  # type: ignore[arg-type]
        router_bias=tensors["router_bias"],  # type: ignore[arg-type]
        coordinate_scales=tensors["coordinate_scales"],  # type: ignore[arg-type]
        direction_left=tensors["direction_left"],  # type: ignore[arg-type]
        direction_right=tensors["direction_right"],  # type: ignore[arg-type]
        pedal_weight=tensors["pedal_weight"],  # type: ignore[arg-type]
        pedal_bias=tensors["pedal_bias"],  # type: ignore[arg-type]
        router_ridge=_positive_float(state.get("router_ridge"), label="router_ridge"),
        direction_ridge=_positive_float(
            state.get("direction_ridge"),
            label="direction_ridge",
        ),
        pedal_ridge=_positive_float(state.get("pedal_ridge"), label="pedal_ridge"),
        trust_fraction=_positive_float(
            state.get("trust_fraction"),
            label="trust_fraction",
        ),
        fit_row_count=state["fit_row_count"],  # type: ignore[arg-type]
        fit_family_ids=state["fit_family_ids"],  # type: ignore[arg-type]
        fit_sequence_sha256s=state["fit_sequence_sha256s"],  # type: ignore[arg-type]
        coordinate_objective=state["coordinate_objective"],  # type: ignore[arg-type]
        coordinate_axes_sha256=_require_sha256(
            state.get("coordinate_axes_sha256"),
            label="coordinate_axes_sha256",
        ),
        coordinate_axis_values=_float_tuple(
            state.get("coordinate_axis_values"),
            count=2,
            label="coordinate_axis_values",
            positive=True,
        ),  # type: ignore[arg-type]
        fit_weight_sha256=_require_sha256(
            state.get("fit_weight_sha256"),
            label="fit_weight_sha256",
        ),
        coordinate_target_weighted_rmse=_nonnegative_float(
            state.get("coordinate_target_weighted_rmse"),
            label="coordinate_target_weighted_rmse",
        ),
        bounded_coordinate_geometry_sha256=_require_sha256(
            state.get("bounded_coordinate_geometry_sha256"),
            label="bounded_coordinate_geometry_sha256",
        ),
        bounded_coordinate_covariance_eigenvalues=_float_tuple(
            state.get("bounded_coordinate_covariance_eigenvalues"),
            count=2,
            label="bounded_coordinate_covariance_eigenvalues",
        ),  # type: ignore[arg-type]
        bounded_coordinate_lambda2_over_lambda1=_nonnegative_float(
            state.get("bounded_coordinate_lambda2_over_lambda1"),
            label="bounded_coordinate_lambda2_over_lambda1",
        ),
        bounded_coordinate_abs_correlation=_nonnegative_float(
            state.get("bounded_coordinate_abs_correlation"),
            label="bounded_coordinate_abs_correlation",
        ),
        bounded_coordinate_target_r2=_finite_tuple(
            state.get("bounded_coordinate_target_r2"),
            count=2,
            label="bounded_coordinate_target_r2",
        ),  # type: ignore[arg-type]
        residual_second_coordinate_energy_fraction=_nonnegative_float(
            state.get("residual_second_coordinate_energy_fraction"),
            label="residual_second_coordinate_energy_fraction",
        ),
        pedal_mode=state["pedal_mode"],  # type: ignore[arg-type]
        pedal_unclipped_target_sha256=_require_sha256(
            state.get("pedal_unclipped_target_sha256"),
            label="pedal_unclipped_target_sha256",
        ),
        pedal_target_sha256=_require_sha256(
            state.get("pedal_target_sha256"),
            label="pedal_target_sha256",
        ),
        pedal_fit_weight_sha256=_require_sha256(
            state.get("pedal_fit_weight_sha256"),
            label="pedal_fit_weight_sha256",
        ),
        pedal_support_mask_sha256=_require_sha256(
            state.get("pedal_support_mask_sha256"),
            label="pedal_support_mask_sha256",
        ),
        pedal_supported_row_count=state["pedal_supported_row_count"],  # type: ignore[arg-type]
        pedal_unclipped_target_weighted_mean=_finite_float(
            state.get("pedal_unclipped_target_weighted_mean"),
            label="pedal_unclipped_target_weighted_mean",
        ),
        pedal_unclipped_target_weighted_rmse=_nonnegative_float(
            state.get("pedal_unclipped_target_weighted_rmse"),
            label="pedal_unclipped_target_weighted_rmse",
        ),
        pedal_target_weighted_mean=_nonnegative_float(
            state.get("pedal_target_weighted_mean"),
            label="pedal_target_weighted_mean",
        ),
        pedal_target_weighted_rmse=_nonnegative_float(
            state.get("pedal_target_weighted_rmse"),
            label="pedal_target_weighted_rmse",
        ),
        pedal_weighted_mean=_nonnegative_float(
            state.get("pedal_weighted_mean"),
            label="pedal_weighted_mean",
        ),
        pedal_weighted_std=_nonnegative_float(
            state.get("pedal_weighted_std"),
            label="pedal_weighted_std",
        ),
        pedal_min=_nonnegative_float(
            state.get("pedal_min"),
            label="pedal_min",
        ),
        pedal_max=_nonnegative_float(
            state.get("pedal_max"),
            label="pedal_max",
        ),
        pedal_effective_weighted_mean=_nonnegative_float(
            state.get("pedal_effective_weighted_mean"),
            label="pedal_effective_weighted_mean",
        ),
        pedal_effective_weighted_std=_nonnegative_float(
            state.get("pedal_effective_weighted_std"),
            label="pedal_effective_weighted_std",
        ),
        pedal_effective_min=_nonnegative_float(
            state.get("pedal_effective_min"),
            label="pedal_effective_min",
        ),
        pedal_effective_max=_nonnegative_float(
            state.get("pedal_effective_max"),
            label="pedal_effective_max",
        ),
        pedal_zero_fraction=_nonnegative_float(
            state.get("pedal_zero_fraction"),
            label="pedal_zero_fraction",
        ),
        pedal_one_fraction=_nonnegative_float(
            state.get("pedal_one_fraction"),
            label="pedal_one_fraction",
        ),
        pedal_target_clipped_fraction=_nonnegative_float(
            state.get("pedal_target_clipped_fraction"),
            label="pedal_target_clipped_fraction",
        ),
        bounded_target_clipped_fraction=_nonnegative_float(
            state.get("bounded_target_clipped_fraction"),
            label="bounded_target_clipped_fraction",
        ),
        bounded_target_mean_clip_scale=_nonnegative_float(
            state.get("bounded_target_mean_clip_scale"),
            label="bounded_target_mean_clip_scale",
        ),
        direction_clipped_fraction=_nonnegative_float(
            state.get("direction_clipped_fraction"),
            label="direction_clipped_fraction",
        ),
        direction_mean_clip_scale=_nonnegative_float(
            state.get("direction_mean_clip_scale"),
            label="direction_mean_clip_scale",
        ),
        weighted_bounded_target_rmse_before=_nonnegative_float(
            state.get("weighted_bounded_target_rmse_before"),
            label="weighted_bounded_target_rmse_before",
        ),
        weighted_bounded_target_rmse_after=_nonnegative_float(
            state.get("weighted_bounded_target_rmse_after"),
            label="weighted_bounded_target_rmse_after",
        ),
        weighted_residual_rmse_before=_nonnegative_float(
            state.get("weighted_residual_rmse_before"),
            label="weighted_residual_rmse_before",
        ),
        weighted_residual_rmse_constant=_nonnegative_float(
            state.get("weighted_residual_rmse_constant"),
            label="weighted_residual_rmse_constant",
        ),
        weighted_residual_rmse_unit=_nonnegative_float(
            state.get("weighted_residual_rmse_unit"),
            label="weighted_residual_rmse_unit",
        ),
        weighted_residual_rmse_oracle=_nonnegative_float(
            state.get("weighted_residual_rmse_oracle"),
            label="weighted_residual_rmse_oracle",
        ),
        weighted_residual_rmse_after=_nonnegative_float(
            state.get("weighted_residual_rmse_after"),
            label="weighted_residual_rmse_after",
        ),
        fit_bounded_direction_ratio_quantiles=_float_tuple(
            state.get("fit_bounded_direction_ratio_quantiles"),
            count=3,
            label="fit_bounded_direction_ratio_quantiles",
        ),  # type: ignore[arg-type]
        fit_emitted_delta_ratio_quantiles=_float_tuple(
            state.get("fit_emitted_delta_ratio_quantiles"),
            count=3,
            label="fit_emitted_delta_ratio_quantiles",
        ),  # type: ignore[arg-type]
        fit_max_bounded_direction_ratio=_nonnegative_float(
            state.get("fit_max_bounded_direction_ratio"),
            label="fit_max_bounded_direction_ratio",
        ),
        fit_max_emitted_delta_ratio=_nonnegative_float(
            state.get("fit_max_emitted_delta_ratio"),
            label="fit_max_emitted_delta_ratio",
        ),
        zero_parent_row_count=state["zero_parent_row_count"],  # type: ignore[arg-type]
        zero_direction_row_count=state["zero_direction_row_count"],  # type: ignore[arg-type]
        fit_receipt_sha256=_require_sha256(
            state.get("fit_receipt_sha256"),
            label="fit_receipt_sha256",
        ),
        artifact_sha256=embedded_artifact,
    )


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            "Fisher-XY pedal path is not a readable regular file"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError("Fisher-XY pedal path must be a nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise RuntimeError("Fisher-XY pedal provider file changed while reading")
    return payload


def _provider_from_bytes(
    payload: bytes,
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Fisher-XY pedal tensor payload is invalid") from error
    if not isinstance(state, Mapping):
        raise ValueError("Fisher-XY pedal payload must contain a mapping")
    return autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict(
        state,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )


def save_autonomous_complete_h4_fisher_xy_pedal_provider(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
    path: Path | str,
) -> dict[str, object]:
    if not isinstance(provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("provider must be AutonomousCompleteH4FisherXYPedalProvider")
    provider.validate_integrity()
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("Fisher-XY pedal provider output must use .pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite Fisher-XY pedal provider")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    stage = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                autonomous_complete_h4_fisher_xy_pedal_provider_state_dict(
                    provider
                ),
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        payload = _read_regular_file(stage)
        restored = _provider_from_bytes(
            payload,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_bridge_binding_sha256=provider.bridge_binding_sha256,
        )
        if restored.metadata() != provider.metadata():
            raise RuntimeError("staged Fisher-XY pedal provider roundtrip drifted")
        try:
            os.link(stage, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite Fisher-XY pedal provider"
            ) from error
        published = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        destination_mode = stat.S_IMODE(destination.stat().st_mode)
        if destination_mode != (stat.S_IRUSR | stat.S_IWUSR):
            raise RuntimeError("Fisher-XY pedal provider file mode is not 0600")
        return {
            "path": destination.as_posix(),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "file_bytes": len(payload),
            "file_mode_octal": "0600",
            "provider_artifact_sha256": provider.artifact_sha256,
            "bridge_binding_sha256": provider.bridge_binding_sha256,
        }
    except BaseException:
        if published:
            raise RuntimeError(
                "Fisher-XY pedal provider publication durability is uncertain"
            )
        raise
    finally:
        stage.unlink(missing_ok=True)


def load_autonomous_complete_h4_fisher_xy_pedal_provider(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str | None = None,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    source = Path(path)
    payload = _read_regular_file(source)
    if stat.S_IMODE(source.stat().st_mode) != (stat.S_IRUSR | stat.S_IWUSR):
        raise ValueError("Fisher-XY pedal provider file mode must be 0600")
    if expected_file_sha256 is not None:
        expected_file = _require_sha256(
            expected_file_sha256,
            label="expected Fisher-XY pedal tensor file",
        )
        if hashlib.sha256(payload).hexdigest() != expected_file:
            raise ValueError("Fisher-XY pedal tensor file differs from expected")
    return _provider_from_bytes(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )
