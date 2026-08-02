"""Finite-objective joint direction/pedal refits for complete-H4 providers.

This module deliberately leaves the V18 provider and schema untouched.  A
V19 provider starts from an authenticated V18 Fisher-XY pedal provider, keeps
its parent and two-coordinate router exactly, and replaces only the low-rank
direction factors and sigmoid-pedal parameters.  The serving payload remains
the same seven tensors as V18:: router weight/bias, coordinate scales,
direction left/right, and pedal weight/bias.

For parent modal row ``p`` and bounded Fisher coordinates ``c`` the provider
computes::

    phi = [c1 * p, c2 * p, c1 * c2 * p]
    q   = phi @ direction_left @ direction_right
    b   = q * min(1, 0.25 * ||p|| / ||q||)
    a   = sigmoid(pedal_bias + [c1, c2, c1*c2] @ pedal_weight)
    d   = a * b

The intercept ablation stores zero pedal slopes.  The unit ablation stores
zero pedal tensors and emits ``a = 1`` exactly.  All modes retain identical
prepared tensor and matched upper-bound matrix-MAC geometry.  Fit protocol
and evidence enter only through authenticated SHA-256 receipts; fit examples,
logits, gradients, and optimizer state are never serialized into the serving
provider.
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
    _float_tensor,
    _require_sha256,
    _sha256,
    _tensor_sha256,
    autonomous_complete_h4_residual_provider_from_state_dict,
    autonomous_complete_h4_residual_provider_state_dict,
)
from .complete_h4_fisher_conditional_pedal import (
    FISHER_XY_PEDAL_TRUST_FRACTION,
    AutonomousCompleteH4FisherXYPedalProvider,
    _training_parent_modal,
    _validate_training_sequence_integrity,
    fisher_xy_bounded_coordinates,
    fisher_xy_pedal_features,
    fisher_xy_pointwise_bounded_direction,
)
from .complete_h4_fisher_conditional_residual import (
    COORDINATE_OBJECTIVES,
    FISHER_XY_COORDINATE_COUNT,
    _positive_float,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_FINITE_JOINT_PEDAL_MODES",
    "FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION",
    "AutonomousCompleteH4FisherFiniteJointPedalProvider",
    "FisherFiniteJointPedalRuntimeReplay",
    "autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict",
    "autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict",
    "canonical_balanced_rank_svd_retraction",
    "dense_direction_descent_proposal",
    "fisher_finite_joint_direction_features",
    "fisher_finite_joint_matched_resource_geometry",
    "fisher_finite_joint_modal_delta",
    "fisher_finite_joint_modal_terms",
    "fisher_finite_joint_pedal_control",
    "initialize_autonomous_complete_h4_fisher_finite_joint_pedal",
    "interpolate_fisher_finite_joint_pedal_parameters",
    "load_autonomous_complete_h4_fisher_finite_joint_pedal_provider",
    "refit_autonomous_complete_h4_fisher_finite_joint_pedal",
    "replay_autonomous_complete_h4_fisher_finite_joint_pedal",
    "save_autonomous_complete_h4_fisher_finite_joint_pedal_provider",
    "validate_fisher_finite_joint_pedal_runtime_replay_metadata",
]


FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION = FISHER_XY_PEDAL_TRUST_FRACTION
FISHER_FINITE_JOINT_PEDAL_MODES = frozenset(
    {"conditional", "intercept", "unit"}
)

_H4_SITE = "layer.4.output"
_DIRECTION_FEATURE_BLOCK_COUNT = 3
_PEDAL_FEATURE_COUNT = 3
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-finite-joint-pedal:provider:v1\0"
)
_FIT_RECEIPT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-finite-joint-pedal:fit:v1\0"
)
_REPLAY_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-finite-joint-pedal:replay:v1\0"
)
_STATE_SCHEMA = (
    "fisher_graph.autonomous_complete_h4_fisher_finite_joint_pedal_provider_tensor.v1"
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
_STATE_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "provider_artifact_sha256",
        "bridge_binding_sha256",
        "parent_provider_artifact_sha256",
        "start_provider_artifact_sha256",
        "fit_protocol_sha256",
        "fit_evidence_sha256",
        "fit_receipt_sha256",
        "trust_fraction",
        "fit_row_count",
        "fit_family_ids",
        "fit_sequence_sha256s",
        "coordinate_objective",
        "pedal_mode",
        "parent_provider_state",
        "tensors",
    }
)


def _finite_runtime_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    """Validate a live tensor without detaching its autograd graph."""

    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or value.layout != torch.strided
        or value.device.type == "meta"
        or any(int(width) <= 0 for width in value.shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite materialized rank-{ndim} tensor")
    return value.to(dtype=torch.float64)


def fisher_finite_joint_direction_features(
    parent_modal: Tensor,
    coordinates: Tensor,
) -> Tensor:
    """Return differentiable ``[c1*p, c2*p, c1*c2*p]`` features."""

    parent = _finite_runtime_tensor(
        parent_modal,
        label="finite-joint parent modal",
        ndim=2,
    )
    bounded = _finite_runtime_tensor(
        coordinates,
        label="finite-joint bounded coordinates",
        ndim=2,
    ).to(parent.device)
    if (
        bounded.shape != (parent.shape[0], FISHER_XY_COORDINATE_COUNT)
        or bool((bounded.abs() >= 1.0).any())
    ):
        raise ValueError("finite-joint coordinate geometry differs")
    c1 = bounded[:, :1]
    c2 = bounded[:, 1:]
    return torch.cat((c1 * parent, c2 * parent, c1 * c2 * parent), dim=1)


def fisher_finite_joint_modal_terms(
    parent_modal: Tensor,
    coordinates: Tensor,
    direction_left: Tensor,
    direction_right: Tensor,
    pedal_weight: Tensor,
    pedal_bias: Tensor,
    *,
    pedal_mode: str = "conditional",
    trust_fraction: float = FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return differentiable ``(q, b, a, a*b)`` finite-joint terms."""

    if pedal_mode not in FISHER_FINITE_JOINT_PEDAL_MODES:
        raise ValueError("finite-joint pedal mode differs")
    parent = _finite_runtime_tensor(
        parent_modal,
        label="finite-joint parent modal",
        ndim=2,
    )
    features = fisher_finite_joint_direction_features(parent, coordinates)
    left = _finite_runtime_tensor(
        direction_left,
        label="finite-joint direction left",
        ndim=2,
    ).to(parent.device)
    right = _finite_runtime_tensor(
        direction_right,
        label="finite-joint direction right",
        ndim=2,
    ).to(parent.device)
    beta = _finite_runtime_tensor(
        pedal_weight,
        label="finite-joint pedal weight",
        ndim=1,
    ).to(parent.device)
    bias = _finite_runtime_tensor(
        pedal_bias,
        label="finite-joint pedal bias",
        ndim=1,
    ).to(parent.device)
    rank = int(parent.shape[1])
    if (
        left.shape[0] != _DIRECTION_FEATURE_BLOCK_COUNT * rank
        or left.shape[1] != right.shape[0]
        or right.shape[1] != rank
        or beta.shape != (_PEDAL_FEATURE_COUNT,)
        or bias.shape != (1,)
    ):
        raise ValueError("finite-joint serving tensor geometry differs")
    if pedal_mode == "intercept" and bool((beta != 0.0).any()):
        raise ValueError("finite-joint intercept pedal slopes must be zero")
    if pedal_mode == "unit" and (
        bool((beta != 0.0).any()) or bool((bias != 0.0).any())
    ):
        raise ValueError("finite-joint unit pedal tensors must be zero")

    direction = (features @ left) @ right
    bounded = fisher_xy_pointwise_bounded_direction(
        parent,
        direction,
        trust_fraction=trust_fraction,
    )
    if pedal_mode == "unit":
        pedal = torch.ones(
            parent.shape[0],
            dtype=parent.dtype,
            device=parent.device,
        )
    else:
        pedal_features = fisher_xy_pedal_features(coordinates).to(parent.device)
        pedal = torch.sigmoid(pedal_features @ beta + bias[0])
    delta = pedal.unsqueeze(1) * bounded
    if (
        not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(pedal).all())
        or not bool(torch.isfinite(delta).all())
        or bool((pedal < 0.0).any())
        or bool((pedal > 1.0).any())
    ):
        raise RuntimeError("finite-joint modal terms became invalid")
    return (
        direction.contiguous(),
        bounded.contiguous(),
        pedal.contiguous(),
        delta.contiguous(),
    )


def fisher_finite_joint_modal_delta(
    parent_modal: Tensor,
    coordinates: Tensor,
    direction_left: Tensor,
    direction_right: Tensor,
    pedal_weight: Tensor,
    pedal_bias: Tensor,
    *,
    pedal_mode: str = "conditional",
    trust_fraction: float = FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
) -> Tensor:
    """Return only the differentiable emitted modal delta."""

    return fisher_finite_joint_modal_terms(
        parent_modal,
        coordinates,
        direction_left,
        direction_right,
        pedal_weight,
        pedal_bias,
        pedal_mode=pedal_mode,
        trust_fraction=trust_fraction,
    )[-1]


def fisher_finite_joint_matched_resource_geometry(
    *,
    parent_prepared_float_scalar_count: int,
    parent_logical_macs_per_token_upper_bound: int,
    rank: int,
    conditional_rank: int,
    pedal_mode: str,
) -> dict[str, int]:
    """Return the mode-matched serving geometry without constructing tensors."""

    for name, value in (
        ("parent_prepared_float_scalar_count", parent_prepared_float_scalar_count),
        (
            "parent_logical_macs_per_token_upper_bound",
            parent_logical_macs_per_token_upper_bound,
        ),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name, value in (("rank", rank), ("conditional_rank", conditional_rank)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if pedal_mode not in FISHER_FINITE_JOINT_PEDAL_MODES:
        raise ValueError("finite-joint pedal mode differs")
    incremental_scalars = int(
        FISHER_XY_COORDINATE_COUNT * rank
        + FISHER_XY_COORDINATE_COUNT
        + FISHER_XY_COORDINATE_COUNT
        + _DIRECTION_FEATURE_BLOCK_COUNT * rank * conditional_rank
        + conditional_rank * rank
        + _PEDAL_FEATURE_COUNT
        + 1
    )
    incremental_macs = int(
        FISHER_XY_COORDINATE_COUNT * rank
        + _DIRECTION_FEATURE_BLOCK_COUNT * rank * conditional_rank
        + conditional_rank * rank
        + _PEDAL_FEATURE_COUNT
    )
    return {
        "incremental_prepared_float_scalar_count": incremental_scalars,
        "prepared_float_scalar_count": (
            parent_prepared_float_scalar_count + incremental_scalars
        ),
        "incremental_logical_macs_per_token_upper_bound": incremental_macs,
        "logical_macs_per_token_upper_bound": (
            parent_logical_macs_per_token_upper_bound + incremental_macs
        ),
    }


def dense_direction_descent_proposal(
    dense_direction: Tensor,
    dense_gradient: Tensor,
    *,
    step_size: float,
) -> Tensor:
    """Apply one pure dense objective-gradient descent proposal."""

    dense = _float_tensor(
        dense_direction,
        label="dense direction",
        ndim=2,
    )
    gradient = _float_tensor(
        dense_gradient,
        label="dense direction gradient",
        ndim=2,
    )
    if gradient.shape != dense.shape:
        raise ValueError("dense direction and gradient geometry differ")
    if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
        raise TypeError("step_size must be a finite nonnegative scalar")
    step = float(step_size)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be a finite nonnegative scalar")
    result = dense - step * gradient
    if not bool(torch.isfinite(result).all()):
        raise ValueError("dense direction proposal became nonfinite")
    return result.contiguous()


def canonical_balanced_rank_svd_retraction(
    dense_direction: Tensor,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Retract a dense direction to canonical balanced rank-``rank`` factors."""

    dense = _float_tensor(
        dense_direction,
        label="dense direction retraction",
        ndim=2,
    )
    if type(rank) is not int or rank <= 0 or rank > min(dense.shape):
        raise ValueError("retraction rank differs from the dense geometry")
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    u = u[:, :rank].contiguous()
    singular = singular[:rank].contiguous()
    vh = vh[:rank].contiguous()
    for component in range(rank):
        pivot = int(vh[component].abs().argmax().item())
        if float(vh[component, pivot]) < 0.0:
            u[:, component].neg_()
            vh[component].neg_()
    root = torch.sqrt(singular)
    left = (u * root.unsqueeze(0)).contiguous()
    right = (root.unsqueeze(1) * vh).contiguous()
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise RuntimeError("balanced SVD retraction became nonfinite")
    return left, right


def interpolate_fisher_finite_joint_pedal_parameters(
    start_weight: Tensor,
    start_bias: Tensor,
    proposed_weight: Tensor,
    proposed_bias: Tensor,
    *,
    fraction: float,
) -> tuple[Tensor, Tensor]:
    """Linearly interpolate finite sigmoid-pedal logits in parameter space."""

    start_w = _float_tensor(start_weight, label="start pedal weight", ndim=1)
    start_b = _float_tensor(start_bias, label="start pedal bias", ndim=1)
    proposed_w = _float_tensor(
        proposed_weight,
        label="proposed pedal weight",
        ndim=1,
    )
    proposed_b = _float_tensor(
        proposed_bias,
        label="proposed pedal bias",
        ndim=1,
    )
    if (
        start_w.shape != (_PEDAL_FEATURE_COUNT,)
        or start_b.shape != (1,)
        or proposed_w.shape != start_w.shape
        or proposed_b.shape != start_b.shape
    ):
        raise ValueError("pedal interpolation geometry differs")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("pedal interpolation fraction must be numeric")
    alpha = float(fraction)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("pedal interpolation fraction must lie in [0, 1]")
    return (
        ((1.0 - alpha) * start_w + alpha * proposed_w).contiguous(),
        ((1.0 - alpha) * start_b + alpha * proposed_b).contiguous(),
    )


@dataclass(frozen=True, slots=True)
class FisherFiniteJointPedalRuntimeReplay:
    """Authenticated serving-only support-row replay for a V19 provider."""

    provider_artifact_sha256: str
    start_provider_artifact_sha256: str
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
        for name in (
            "provider_artifact_sha256",
            "start_provider_artifact_sha256",
            "parent_provider_artifact_sha256",
            "sequence_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"finite-joint replay {name}")
        trust = _positive_float(self.trust_fraction, label="replay trust_fraction")
        if trust != FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION:
            raise ValueError("finite-joint replay trust_fraction must be 0.25")
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
            raise ValueError("finite-joint replay geometry differs")
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
            raise ValueError("finite-joint replay escaped pointwise trust")
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
            if _require_sha256(self.artifact_sha256, label="finite-joint replay") != computed:
                raise ValueError("finite-joint replay artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def row_count(self) -> int:
        return int(self.parent_modal.shape[0])

    def _payload(self) -> dict[str, object]:
        parent_norm = torch.linalg.vector_norm(self.parent_modal, dim=1)
        bounded_norm = torch.linalg.vector_norm(self.bounded_direction, dim=1)
        delta_norm = torch.linalg.vector_norm(self.emitted_delta, dim=1)
        positive = parent_norm > 0.0
        divisor = torch.where(positive, parent_norm, torch.ones_like(parent_norm))
        bounded_ratio = torch.where(positive, bounded_norm / divisor, 0.0)
        delta_ratio = torch.where(positive, delta_norm / divisor, 0.0)
        return {
            "schema": "fisher_graph.fisher_finite_joint_pedal_runtime_replay.v1",
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "start_provider_artifact_sha256": self.start_provider_artifact_sha256,
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
            "pedal_min": float(self.pedal.min()),
            "pedal_mean": float(self.pedal.mean()),
            "pedal_max": float(self.pedal.max()),
            "max_bounded_direction_to_parent_norm_ratio": float(bounded_ratio.max()),
            "max_emitted_delta_to_parent_norm_ratio": float(delta_ratio.max()),
            "pointwise_trust_certificate_passed": bool(
                float(bounded_ratio.max()) <= self.trust_fraction + 1.0e-14
                and float(delta_ratio.max()) <= self.trust_fraction + 1.0e-14
            ),
            "runtime_field_semantics": (
                "support_rows_from_source_prefix_and_base_h4_only"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_REPLAY_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("finite-joint replay payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        result = {**self._payload(), "artifact_sha256": self.artifact_sha256}
        return validate_fisher_finite_joint_pedal_runtime_replay_metadata(result)


def validate_fisher_finite_joint_pedal_runtime_replay_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate a tensor-free V19 replay receipt."""

    expected = frozenset(
        {
            "schema",
            "provider_artifact_sha256",
            "start_provider_artifact_sha256",
            "parent_provider_artifact_sha256",
            "sequence_artifact_sha256",
            "trust_fraction",
            "row_count",
            "rank",
            "tensor_sha256s",
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
    if not isinstance(metadata, Mapping) or set(metadata) != expected:
        raise ValueError("finite-joint replay metadata fields differ")
    payload = dict(metadata)
    artifact = _require_sha256(
        payload.pop("artifact_sha256"),
        label="finite-joint replay metadata artifact",
    )
    if payload.get("schema") != "fisher_graph.fisher_finite_joint_pedal_runtime_replay.v1":
        raise ValueError("finite-joint replay metadata schema differs")
    for name in (
        "provider_artifact_sha256",
        "start_provider_artifact_sha256",
        "parent_provider_artifact_sha256",
        "sequence_artifact_sha256",
    ):
        _require_sha256(payload.get(name), label=f"replay metadata {name}")
    if payload.get("trust_fraction") != FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION:
        raise ValueError("finite-joint replay metadata trust differs")
    row_count = payload.get("row_count")
    rank = payload.get("rank")
    if type(row_count) is not int or row_count <= 0 or type(rank) is not int or rank <= 0:
        raise ValueError("finite-joint replay metadata geometry differs")
    hashes = payload.get("tensor_sha256s")
    expected_hashes = {
        "parent_modal",
        "bounded_coordinates",
        "unbounded_direction",
        "bounded_direction",
        "pedal",
        "emitted_delta",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hashes:
        raise ValueError("finite-joint replay tensor hashes differ")
    for name in expected_hashes:
        _require_sha256(hashes.get(name), label=f"replay tensor {name}")
    pedal_values = []
    for name in ("pedal_min", "pedal_mean", "pedal_max"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("finite-joint replay pedal range differs")
        converted = float(value)
        if not math.isfinite(converted) or converted < 0.0 or converted > 1.0:
            raise ValueError("finite-joint replay pedal range differs")
        pedal_values.append(converted)
    if pedal_values != sorted(pedal_values):
        raise ValueError("finite-joint replay pedal range differs")
    for name in (
        "max_bounded_direction_to_parent_norm_ratio",
        "max_emitted_delta_to_parent_norm_ratio",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("finite-joint replay trust ratio differs")
        converted = float(value)
        if (
            not math.isfinite(converted)
            or converted < 0.0
            or converted > FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION + 1.0e-14
        ):
            raise ValueError("finite-joint replay trust ratio differs")
    if (
        payload.get("pointwise_trust_certificate_passed") is not True
        or payload.get("runtime_field_semantics")
        != "support_rows_from_source_prefix_and_base_h4_only"
    ):
        raise ValueError("finite-joint replay trust certificate differs")
    if _sha256(_REPLAY_DOMAIN, payload) != artifact:
        raise ValueError("finite-joint replay metadata artifact hash mismatch")
    return {**payload, "artifact_sha256": artifact}


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherFiniteJointPedalProvider(
    Gemma3L3L4CorrectionProvider
):
    """Immutable V19 finite-objective direction plus sigmoid-pedal provider."""

    parent_provider: AutonomousCompleteH4ResidualProvider
    router_weight: Tensor
    router_bias: Tensor
    coordinate_scales: Tensor
    direction_left: Tensor
    direction_right: Tensor
    pedal_weight: Tensor
    pedal_bias: Tensor
    start_provider_artifact_sha256: str
    fit_protocol_sha256: str
    fit_evidence_sha256: str
    fit_receipt_sha256: str
    trust_fraction: float
    fit_row_count: int
    fit_family_ids: tuple[str, ...]
    fit_sequence_sha256s: tuple[str, ...]
    coordinate_objective: str
    pedal_mode: str
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.parent_provider, AutonomousCompleteH4ResidualProvider):
            raise TypeError("finite-joint parent must be autonomous complete-H4")
        self.parent_provider.validate_integrity()
        tensors = {
            "router_weight": _float_tensor(
                self.router_weight,
                label="router_weight",
                ndim=2,
            ),
            "router_bias": _float_tensor(self.router_bias, label="router_bias", ndim=1),
            "coordinate_scales": _float_tensor(
                self.coordinate_scales,
                label="coordinate_scales",
                ndim=1,
            ),
            "direction_left": _float_tensor(
                self.direction_left,
                label="direction_left",
                ndim=2,
            ),
            "direction_right": _float_tensor(
                self.direction_right,
                label="direction_right",
                ndim=2,
            ),
            "pedal_weight": _float_tensor(
                self.pedal_weight,
                label="pedal_weight",
                ndim=1,
            ),
            "pedal_bias": _float_tensor(self.pedal_bias, label="pedal_bias", ndim=1),
        }
        rank = self.parent_provider.rank
        if (
            tensors["router_weight"].shape
            != (rank, FISHER_XY_COORDINATE_COUNT)
            or tensors["router_bias"].shape != (FISHER_XY_COORDINATE_COUNT,)
            or tensors["coordinate_scales"].shape
            != (FISHER_XY_COORDINATE_COUNT,)
            or bool((tensors["coordinate_scales"] <= 0.0).any())
            or tensors["direction_left"].shape[0]
            != _DIRECTION_FEATURE_BLOCK_COUNT * rank
            or tensors["direction_left"].shape[1]
            != tensors["direction_right"].shape[0]
            or tensors["direction_right"].shape[1] != rank
            or tensors["pedal_weight"].shape != (_PEDAL_FEATURE_COUNT,)
            or tensors["pedal_bias"].shape != (1,)
        ):
            raise ValueError("finite-joint provider tensor geometry differs")
        if self.pedal_mode not in FISHER_FINITE_JOINT_PEDAL_MODES:
            raise ValueError("finite-joint pedal mode differs")
        if self.pedal_mode == "intercept" and bool(
            (tensors["pedal_weight"] != 0.0).any()
        ):
            raise ValueError("finite-joint intercept pedal slopes must be zero")
        if self.pedal_mode == "unit" and (
            bool((tensors["pedal_weight"] != 0.0).any())
            or bool((tensors["pedal_bias"] != 0.0).any())
        ):
            raise ValueError("finite-joint unit pedal tensors must be zero")
        for name, value in tensors.items():
            object.__setattr__(self, name, value)
        for name in (
            "start_provider_artifact_sha256",
            "fit_protocol_sha256",
            "fit_evidence_sha256",
            "fit_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        trust = _positive_float(self.trust_fraction, label="trust_fraction")
        if trust != FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION:
            raise ValueError("finite-joint trust_fraction is frozen at 0.25")
        object.__setattr__(self, "trust_fraction", trust)
        if type(self.fit_row_count) is not int or self.fit_row_count <= 0:
            raise ValueError("finite-joint fit_row_count must be positive")
        if (
            type(self.fit_family_ids) is not tuple
            or not self.fit_family_ids
            or self.fit_family_ids != tuple(sorted(set(self.fit_family_ids)))
            or type(self.fit_sequence_sha256s) is not tuple
            or not self.fit_sequence_sha256s
            or self.fit_sequence_sha256s
            != tuple(sorted(set(self.fit_sequence_sha256s)))
        ):
            raise ValueError("finite-joint fit ownership must be canonical")
        for value in self.fit_sequence_sha256s:
            _require_sha256(value, label="finite-joint fit sequence")
        if self.coordinate_objective not in COORDINATE_OBJECTIVES:
            raise ValueError("finite-joint coordinate objective differs")
        expected_receipt = _sha256(_FIT_RECEIPT_DOMAIN, self._fit_receipt_payload())
        if expected_receipt != self.fit_receipt_sha256:
            raise ValueError("finite-joint fit receipt hash mismatch")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="finite-joint provider",
            ) != computed:
                raise ValueError("finite-joint provider artifact hash mismatch")
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
        result = fisher_finite_joint_matched_resource_geometry(
            parent_prepared_float_scalar_count=0,
            parent_logical_macs_per_token_upper_bound=0,
            rank=self.rank,
            conditional_rank=self.conditional_rank,
            pedal_mode=self.pedal_mode,
        )
        return result["incremental_prepared_float_scalar_count"]

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_provider.prepared_float_scalar_count
            + self.incremental_prepared_float_scalar_count
        )

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        result = fisher_finite_joint_matched_resource_geometry(
            parent_prepared_float_scalar_count=0,
            parent_logical_macs_per_token_upper_bound=0,
            rank=self.rank,
            conditional_rank=self.conditional_rank,
            pedal_mode=self.pedal_mode,
        )
        return result["incremental_logical_macs_per_token_upper_bound"]

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_provider.logical_macs_per_token_upper_bound
            + self.incremental_logical_macs_per_token_upper_bound
        )

    def _tensor_sha256s(self) -> dict[str, str]:
        return {name: _tensor_sha256(getattr(self, name)) for name in sorted(_TENSOR_KEYS)}

    def _fit_receipt_payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.fisher_finite_joint_pedal_fit_receipt.v1",
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.start_provider_artifact_sha256,
            "fit_protocol_sha256": self.fit_protocol_sha256,
            "fit_evidence_sha256": self.fit_evidence_sha256,
            "fit_row_count": self.fit_row_count,
            "fit_family_ids": self.fit_family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "coordinate_objective": self.coordinate_objective,
            "pedal_mode": self.pedal_mode,
            "trust_fraction": self.trust_fraction,
            "tensor_sha256s": self._tensor_sha256s(),
            "fit_tensors_serialized": False,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_finite_joint_pedal_provider.v1"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.start_provider_artifact_sha256,
            "fit_protocol_sha256": self.fit_protocol_sha256,
            "fit_evidence_sha256": self.fit_evidence_sha256,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "trust_fraction": self.trust_fraction,
            "fit_row_count": self.fit_row_count,
            "fit_family_ids": self.fit_family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "coordinate_objective": self.coordinate_objective,
            "pedal_mode": self.pedal_mode,
            "tensor_sha256s": self._tensor_sha256s(),
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": (
                "native_h4",
                "targets",
                "logits",
                "gradients",
                "coordinate_axes",
                "family_ids",
                "fit_evidence",
                "optimizer_state",
            ),
            "coordinate_semantics": "u_div_scale_plus_abs_u",
            "direction_feature_semantics": "c1_p_c2_p_c1_c2_p",
            "bounded_direction_semantics": (
                "q_times_min_one_beta_parent_norm_over_q_norm_zero_safe_no_amplification"
            ),
            "pedal_semantics": (
                "conditional_sigmoid_logit_intercept_sigmoid_bias_unit_exact_one"
            ),
            "certificate_scope": (
                "pointwise_emitted_modal_amplitude_not_full_nonlinear_jacobian_or_lipschitz"
            ),
            "fit_tensors_serialized": False,
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        self.parent_provider.validate_integrity()
        if _sha256(_FIT_RECEIPT_DOMAIN, self._fit_receipt_payload()) != self.fit_receipt_sha256:
            raise RuntimeError("finite-joint fit receipt payload drifted")
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("finite-joint provider tensor payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
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
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "logical_macs_accounting_scope": (
                "matched_matrix_multiply_upper_bound_including_ablated_pedal_path_"
                "excluding_norms_sigmoid_hashes_transfers_and_workspace"
            ),
            "runtime_state_float_scalars_per_sequence": 0,
            "router_bias_additions_per_token": 2,
            "pedal_bias_additions_per_token_upper_bound": 1,
            "pedal_sigmoid_ops_per_token_upper_bound": 1,
            "pointwise_norm_square_roots_per_token": 2,
            "pointwise_bound_divisions_per_token": 1,
            "pointwise_direction_scale_multiplications_per_token": self.rank,
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
            raise ValueError("finite-joint modal router input differs")
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
            raise ValueError("finite-joint parent modal differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, FISHER_XY_COORDINATE_COUNT)
        features = fisher_finite_joint_direction_features(
            flat_parent,
            flat_coordinates,
        )
        direction = (
            features @ self.direction_left.to(features.device)
        ) @ self.direction_right.to(features.device)
        if not bool(torch.isfinite(direction).all()):
            raise RuntimeError("finite-joint direction became nonfinite")
        return direction.reshape(original_shape).contiguous()

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(coordinates, Tensor)
            or coordinates.ndim < 2
            or coordinates.shape[-1] != FISHER_XY_COORDINATE_COUNT
            or not coordinates.is_floating_point()
            or not bool(torch.isfinite(coordinates).all())
            or bool((coordinates.abs() >= 1.0).any())
        ):
            raise ValueError("finite-joint pedal coordinates differ")
        if self.pedal_mode == "unit":
            return torch.ones(
                coordinates.shape[:-1],
                dtype=torch.float64,
                device=coordinates.device,
            ).contiguous()
        flat = coordinates.reshape(-1, FISHER_XY_COORDINATE_COUNT)
        features = fisher_xy_pedal_features(flat)
        values = torch.sigmoid(
            features @ self.pedal_weight.to(features.device)
            + self.pedal_bias.to(features.device)[0]
        )
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("finite-joint sigmoid pedal became nonfinite")
        return values.reshape(coordinates.shape[:-1]).contiguous()

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
            raise RuntimeError("finite-joint provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("finite-joint modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("finite-joint modal correction escaped support")
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


def _fit_receipt_sha256(
    *,
    parent_provider: AutonomousCompleteH4ResidualProvider,
    router_weight: Tensor,
    router_bias: Tensor,
    coordinate_scales: Tensor,
    direction_left: Tensor,
    direction_right: Tensor,
    pedal_weight: Tensor,
    pedal_bias: Tensor,
    start_provider_artifact_sha256: str,
    fit_protocol_sha256: str,
    fit_evidence_sha256: str,
    fit_row_count: int,
    fit_family_ids: tuple[str, ...],
    fit_sequence_sha256s: tuple[str, ...],
    coordinate_objective: str,
    pedal_mode: str,
    trust_fraction: float,
) -> str:
    tensor_values = {
        "router_weight": router_weight,
        "router_bias": router_bias,
        "coordinate_scales": coordinate_scales,
        "direction_left": direction_left,
        "direction_right": direction_right,
        "pedal_weight": pedal_weight,
        "pedal_bias": pedal_bias,
    }
    return _sha256(
        _FIT_RECEIPT_DOMAIN,
        {
            "schema": "fisher_graph.fisher_finite_joint_pedal_fit_receipt.v1",
            "bridge_binding_sha256": parent_provider.bridge_binding_sha256,
            "parent_provider_artifact_sha256": parent_provider.artifact_sha256,
            "start_provider_artifact_sha256": start_provider_artifact_sha256,
            "fit_protocol_sha256": fit_protocol_sha256,
            "fit_evidence_sha256": fit_evidence_sha256,
            "fit_row_count": fit_row_count,
            "fit_family_ids": fit_family_ids,
            "fit_sequence_sha256s": fit_sequence_sha256s,
            "coordinate_objective": coordinate_objective,
            "pedal_mode": pedal_mode,
            "trust_fraction": trust_fraction,
            "tensor_sha256s": {
                name: _tensor_sha256(value)
                for name, value in sorted(tensor_values.items())
            },
            "fit_tensors_serialized": False,
        },
    )


def refit_autonomous_complete_h4_fisher_finite_joint_pedal(
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
    *,
    direction_left: Tensor,
    direction_right: Tensor,
    pedal_weight: Tensor,
    pedal_bias: Tensor,
    fit_protocol_sha256: str,
    fit_evidence_sha256: str,
    pedal_mode: str = "conditional",
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    """Build a V19 serving provider from an authenticated V18 start."""

    if not isinstance(start_provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("finite-joint start must be a V18 Fisher-XY pedal provider")
    start_provider.validate_integrity()
    if start_provider.pedal_mode != "conditional":
        raise ValueError("finite-joint V18 start pedal mode must be conditional")
    protocol = _require_sha256(fit_protocol_sha256, label="fit_protocol_sha256")
    evidence = _require_sha256(fit_evidence_sha256, label="fit_evidence_sha256")
    if pedal_mode not in FISHER_FINITE_JOINT_PEDAL_MODES:
        raise ValueError("finite-joint pedal mode differs")
    left = _float_tensor(direction_left, label="direction_left", ndim=2)
    right = _float_tensor(direction_right, label="direction_right", ndim=2)
    weight = _float_tensor(pedal_weight, label="pedal_weight", ndim=1)
    bias = _float_tensor(pedal_bias, label="pedal_bias", ndim=1)
    if pedal_mode == "intercept":
        weight = torch.zeros_like(weight)
    elif pedal_mode == "unit":
        weight = torch.zeros_like(weight)
        bias = torch.zeros_like(bias)
    receipt = _fit_receipt_sha256(
        parent_provider=start_provider.parent_provider,
        router_weight=start_provider.router_weight,
        router_bias=start_provider.router_bias,
        coordinate_scales=start_provider.coordinate_scales,
        direction_left=left,
        direction_right=right,
        pedal_weight=weight,
        pedal_bias=bias,
        start_provider_artifact_sha256=start_provider.artifact_sha256,
        fit_protocol_sha256=protocol,
        fit_evidence_sha256=evidence,
        fit_row_count=start_provider.fit_row_count,
        fit_family_ids=start_provider.fit_family_ids,
        fit_sequence_sha256s=start_provider.fit_sequence_sha256s,
        coordinate_objective=start_provider.coordinate_objective,
        pedal_mode=pedal_mode,
        trust_fraction=FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
    )
    return AutonomousCompleteH4FisherFiniteJointPedalProvider(
        parent_provider=start_provider.parent_provider,
        router_weight=start_provider.router_weight,
        router_bias=start_provider.router_bias,
        coordinate_scales=start_provider.coordinate_scales,
        direction_left=left,
        direction_right=right,
        pedal_weight=weight,
        pedal_bias=bias,
        start_provider_artifact_sha256=start_provider.artifact_sha256,
        fit_protocol_sha256=protocol,
        fit_evidence_sha256=evidence,
        fit_receipt_sha256=receipt,
        trust_fraction=FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
        fit_row_count=start_provider.fit_row_count,
        fit_family_ids=start_provider.fit_family_ids,
        fit_sequence_sha256s=start_provider.fit_sequence_sha256s,
        coordinate_objective=start_provider.coordinate_objective,
        pedal_mode=pedal_mode,
    )


def initialize_autonomous_complete_h4_fisher_finite_joint_pedal(
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
    *,
    fit_protocol_sha256: str,
    fit_evidence_sha256: str,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    """Initialize V19 at ``2 * V18 direction`` and sigmoid pedal ``0.5``.

    Before radial clipping, multiplying the V18 direction by two and applying
    a zero-logit sigmoid pedal reproduces the V18 unit-direction amplitude.
    The dense matrix is retracted to canonical balanced factors at the V18
    conditional rank.
    """

    if not isinstance(start_provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("finite-joint start must be a V18 Fisher-XY pedal provider")
    start_provider.validate_integrity()
    if start_provider.pedal_mode != "conditional":
        raise ValueError("finite-joint V18 start pedal mode must be conditional")
    dense = 2.0 * (start_provider.direction_left @ start_provider.direction_right)
    left, right = canonical_balanced_rank_svd_retraction(
        dense,
        rank=start_provider.conditional_rank,
    )
    return refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start_provider,
        direction_left=left,
        direction_right=right,
        pedal_weight=torch.zeros(_PEDAL_FEATURE_COUNT, dtype=torch.float64),
        pedal_bias=torch.zeros(1, dtype=torch.float64),
        fit_protocol_sha256=fit_protocol_sha256,
        fit_evidence_sha256=fit_evidence_sha256,
        pedal_mode="conditional",
    )


def fisher_finite_joint_pedal_control(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    pedal_mode: str,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    """Return a resource-matched intercept or unit ablation."""

    if not isinstance(provider, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("finite-joint control requires a V19 provider")
    provider.validate_integrity()
    if pedal_mode not in FISHER_FINITE_JOINT_PEDAL_MODES:
        raise ValueError("finite-joint pedal mode differs")
    weight = provider.pedal_weight
    bias = provider.pedal_bias
    if pedal_mode == "intercept":
        weight = torch.zeros_like(weight)
    elif pedal_mode == "unit":
        weight = torch.zeros_like(weight)
        bias = torch.zeros_like(bias)
    receipt = _fit_receipt_sha256(
        parent_provider=provider.parent_provider,
        router_weight=provider.router_weight,
        router_bias=provider.router_bias,
        coordinate_scales=provider.coordinate_scales,
        direction_left=provider.direction_left,
        direction_right=provider.direction_right,
        pedal_weight=weight,
        pedal_bias=bias,
        start_provider_artifact_sha256=provider.start_provider_artifact_sha256,
        fit_protocol_sha256=provider.fit_protocol_sha256,
        fit_evidence_sha256=provider.fit_evidence_sha256,
        fit_row_count=provider.fit_row_count,
        fit_family_ids=provider.fit_family_ids,
        fit_sequence_sha256s=provider.fit_sequence_sha256s,
        coordinate_objective=provider.coordinate_objective,
        pedal_mode=pedal_mode,
        trust_fraction=provider.trust_fraction,
    )
    return AutonomousCompleteH4FisherFiniteJointPedalProvider(
        parent_provider=provider.parent_provider,
        router_weight=provider.router_weight,
        router_bias=provider.router_bias,
        coordinate_scales=provider.coordinate_scales,
        direction_left=provider.direction_left,
        direction_right=provider.direction_right,
        pedal_weight=weight,
        pedal_bias=bias,
        start_provider_artifact_sha256=provider.start_provider_artifact_sha256,
        fit_protocol_sha256=provider.fit_protocol_sha256,
        fit_evidence_sha256=provider.fit_evidence_sha256,
        fit_receipt_sha256=receipt,
        trust_fraction=provider.trust_fraction,
        fit_row_count=provider.fit_row_count,
        fit_family_ids=provider.fit_family_ids,
        fit_sequence_sha256s=provider.fit_sequence_sha256s,
        coordinate_objective=provider.coordinate_objective,
        pedal_mode=pedal_mode,
    )


def replay_autonomous_complete_h4_fisher_finite_joint_pedal(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    sequence: AutonomousCompleteH4TrainingSequence,
) -> FisherFiniteJointPedalRuntimeReplay:
    """Replay support rows using only serving-available sequence fields."""

    if not isinstance(provider, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("provider must be finite-joint complete-H4")
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
    replay = FisherFiniteJointPedalRuntimeReplay(
        provider_artifact_sha256=provider.artifact_sha256,
        start_provider_artifact_sha256=provider.start_provider_artifact_sha256,
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


def autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
) -> dict[str, object]:
    """Return the exact V19 serving state; no fit tensors are included."""

    if not isinstance(provider, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("provider must be finite-joint complete-H4")
    provider.validate_integrity()
    return {
        "schema": _STATE_SCHEMA,
        "format_version": 1,
        "provider_artifact_sha256": provider.artifact_sha256,
        "bridge_binding_sha256": provider.bridge_binding_sha256,
        "parent_provider_artifact_sha256": provider.parent_provider.artifact_sha256,
        "start_provider_artifact_sha256": provider.start_provider_artifact_sha256,
        "fit_protocol_sha256": provider.fit_protocol_sha256,
        "fit_evidence_sha256": provider.fit_evidence_sha256,
        "fit_receipt_sha256": provider.fit_receipt_sha256,
        "trust_fraction": provider.trust_fraction,
        "fit_row_count": provider.fit_row_count,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "coordinate_objective": provider.coordinate_objective,
        "pedal_mode": provider.pedal_mode,
        "parent_provider_state": autonomous_complete_h4_residual_provider_state_dict(
            provider.parent_provider
        ),
        "tensors": {
            name: getattr(provider, name).detach().clone()
            for name in sorted(_TENSOR_KEYS)
        },
    }


def autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
    state: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None = None,
    expected_start_provider_artifact_sha256: str | None = None,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    """Restore a V19 serving provider with fail-closed bindings."""

    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("finite-joint provider state fields differ")
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected finite-joint provider",
    )
    embedded_artifact = _require_sha256(
        state.get("provider_artifact_sha256"),
        label="embedded finite-joint provider",
    )
    if embedded_artifact != expected_artifact:
        raise ValueError("finite-joint state artifact differs from expected")
    bridge = _require_sha256(
        state.get("bridge_binding_sha256"),
        label="finite-joint bridge binding",
    )
    if expected_bridge_binding_sha256 is not None and bridge != _require_sha256(
        expected_bridge_binding_sha256,
        label="expected finite-joint bridge binding",
    ):
        raise ValueError("finite-joint bridge binding differs from expected")
    start = _require_sha256(
        state.get("start_provider_artifact_sha256"),
        label="finite-joint start provider",
    )
    if (
        expected_start_provider_artifact_sha256 is not None
        and start
        != _require_sha256(
            expected_start_provider_artifact_sha256,
            label="expected finite-joint start provider",
        )
    ):
        raise ValueError("finite-joint start provider differs from expected")
    parent_artifact = _require_sha256(
        state.get("parent_provider_artifact_sha256"),
        label="finite-joint parent provider",
    )
    parent_state = state.get("parent_provider_state")
    if not isinstance(parent_state, Mapping):
        raise ValueError("finite-joint parent state must be a mapping")
    parent = autonomous_complete_h4_residual_provider_from_state_dict(
        parent_state,
        expected_artifact_sha256=parent_artifact,
        expected_bridge_binding_sha256=bridge,
    )
    tensors = state.get("tensors")
    if (
        not isinstance(tensors, Mapping)
        or set(tensors) != _TENSOR_KEYS
        or any(not isinstance(tensors.get(name), Tensor) for name in _TENSOR_KEYS)
    ):
        raise ValueError("finite-joint state tensor fields differ")
    if (
        state.get("schema") != _STATE_SCHEMA
        or state.get("format_version") != 1
        or type(state.get("fit_row_count")) is not int
        or type(state.get("fit_family_ids")) is not tuple
        or type(state.get("fit_sequence_sha256s")) is not tuple
        or not isinstance(state.get("coordinate_objective"), str)
        or not isinstance(state.get("pedal_mode"), str)
    ):
        raise ValueError("finite-joint state scalar contract differs")
    return AutonomousCompleteH4FisherFiniteJointPedalProvider(
        parent_provider=parent,
        router_weight=tensors["router_weight"],  # type: ignore[arg-type]
        router_bias=tensors["router_bias"],  # type: ignore[arg-type]
        coordinate_scales=tensors["coordinate_scales"],  # type: ignore[arg-type]
        direction_left=tensors["direction_left"],  # type: ignore[arg-type]
        direction_right=tensors["direction_right"],  # type: ignore[arg-type]
        pedal_weight=tensors["pedal_weight"],  # type: ignore[arg-type]
        pedal_bias=tensors["pedal_bias"],  # type: ignore[arg-type]
        start_provider_artifact_sha256=start,
        fit_protocol_sha256=_require_sha256(
            state.get("fit_protocol_sha256"),
            label="fit_protocol_sha256",
        ),
        fit_evidence_sha256=_require_sha256(
            state.get("fit_evidence_sha256"),
            label="fit_evidence_sha256",
        ),
        fit_receipt_sha256=_require_sha256(
            state.get("fit_receipt_sha256"),
            label="fit_receipt_sha256",
        ),
        trust_fraction=_positive_float(
            state.get("trust_fraction"),
            label="trust_fraction",
        ),
        fit_row_count=state["fit_row_count"],  # type: ignore[arg-type]
        fit_family_ids=state["fit_family_ids"],  # type: ignore[arg-type]
        fit_sequence_sha256s=state["fit_sequence_sha256s"],  # type: ignore[arg-type]
        coordinate_objective=state["coordinate_objective"],  # type: ignore[arg-type]
        pedal_mode=state["pedal_mode"],  # type: ignore[arg-type]
        artifact_sha256=embedded_artifact,
    )


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("finite-joint path is not a readable regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError("finite-joint path must be a nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise RuntimeError("finite-joint provider file changed while reading")
    return payload


def _provider_from_bytes(
    payload: bytes,
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None,
    expected_start_provider_artifact_sha256: str | None,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("finite-joint tensor payload is invalid") from error
    if not isinstance(state, Mapping):
        raise ValueError("finite-joint payload must contain a mapping")
    return autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
        state,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
        expected_start_provider_artifact_sha256=(
            expected_start_provider_artifact_sha256
        ),
    )


def save_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    path: Path | str,
) -> dict[str, object]:
    """Publish a write-once, owner-only V19 tensor sidecar."""

    if not isinstance(provider, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("provider must be finite-joint complete-H4")
    provider.validate_integrity()
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("finite-joint provider output must use .pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite finite-joint provider")
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
                autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict(
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
            expected_start_provider_artifact_sha256=(
                provider.start_provider_artifact_sha256
            ),
        )
        if restored.metadata() != provider.metadata():
            raise RuntimeError("staged finite-joint provider roundtrip drifted")
        try:
            os.link(stage, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite finite-joint provider"
            ) from error
        published = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if stat.S_IMODE(destination.stat().st_mode) != (
            stat.S_IRUSR | stat.S_IWUSR
        ):
            raise RuntimeError("finite-joint provider file mode is not 0600")
        return {
            "path": destination.as_posix(),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "file_bytes": len(payload),
            "file_mode_octal": "0600",
            "provider_artifact_sha256": provider.artifact_sha256,
            "start_provider_artifact_sha256": (
                provider.start_provider_artifact_sha256
            ),
            "bridge_binding_sha256": provider.bridge_binding_sha256,
        }
    except BaseException:
        if published:
            raise RuntimeError(
                "finite-joint provider publication durability is uncertain"
            )
        raise
    finally:
        stage.unlink(missing_ok=True)


def load_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str | None = None,
    expected_bridge_binding_sha256: str | None = None,
    expected_start_provider_artifact_sha256: str | None = None,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    """Load a V19 sidecar while requiring its logical artifact binding."""

    source = Path(path)
    payload = _read_regular_file(source)
    if stat.S_IMODE(source.stat().st_mode) != (stat.S_IRUSR | stat.S_IWUSR):
        raise ValueError("finite-joint provider file mode must be 0600")
    if expected_file_sha256 is not None:
        expected_file = _require_sha256(
            expected_file_sha256,
            label="expected finite-joint tensor file",
        )
        if hashlib.sha256(payload).hexdigest() != expected_file:
            raise ValueError("finite-joint tensor file differs from expected")
    return _provider_from_bytes(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
        expected_start_provider_artifact_sha256=(
            expected_start_provider_artifact_sha256
        ),
    )
