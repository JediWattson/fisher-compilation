"""Bounded Fisher-XY conditioning for autonomous complete-H4 providers.

The offline fit uses either reverse-VJP rows to define two modal Fisher
directions, or activation PCA as an exactly parameter-matched control, and
distils their signed scores into a source-free linear router over the parent
provider's modal prediction.  At runtime the two router outputs are bounded in
``(-1, 1)`` with ``u / (scale + abs(u))`` and drive the multi-affine feature
bank ``[c1*p, c2*p, c1*c2*p]``.

The conditional map is stored as one rank-k factorization.  Its four corner
operators are globally projected to spectral norm 0.25.  Because the map is
affine in either coordinate while the other is held fixed, every interior
operator is a convex interpolation of the four corners and inherits the same
pointwise correction-amplitude bound.  This is not a bound on the Jacobian of
the complete nonlinear router-plus-correction map.  Native H4, logits, targets,
gradients, Fisher axes, and family labels are fit-only; the serving ABI remains
``(one_pass_prefix, realized_h4)``.
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
    _H4_WIDTH,
    _SOURCE_RANK,
    _bounded_vjp_multipliers,
    _float_tensor,
    _ordered_sequences,
    _require_sha256,
    _sha256,
    _tensor_sha256,
    autonomous_complete_h4_residual_provider_from_state_dict,
    autonomous_complete_h4_residual_provider_state_dict,
)
from .conditional_quadratic_edge import build_causal_lagged_modal_design
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .radial_finite_displacement_correction import (
    family_balanced_row_weights,
)


__all__ = [
    "COORDINATE_OBJECTIVES",
    "FISHER_XY_COORDINATE_COUNT",
    "FISHER_XY_OPERATOR_NORM_BOUND",
    "AutonomousCompleteH4FisherXYProvider",
    "FisherXYBoundedCoordinateGeometry",
    "fisher_xy_bounded_coordinates",
    "fit_autonomous_complete_h4_fisher_xy_residual",
    "load_autonomous_complete_h4_fisher_xy_provider",
    "project_fisher_xy_conditional_factors",
    "replay_autonomous_complete_h4_fisher_xy_bounded_coordinates",
    "summarize_fisher_xy_bounded_coordinate_geometry",
    "autonomous_complete_h4_fisher_xy_provider_from_state_dict",
    "autonomous_complete_h4_fisher_xy_provider_state_dict",
    "save_autonomous_complete_h4_fisher_xy_provider",
]


FISHER_XY_COORDINATE_COUNT = 2
FISHER_XY_OPERATOR_NORM_BOUND = 0.25
COORDINATE_OBJECTIVES = frozenset(
    {"reverse_vjp_fisher", "activation_pca"}
)
_H4_SITE = "layer.4.output"
_FEATURE_BLOCK_COUNT = 3
_PROVIDER_DOMAIN = b"fisher-graph:autonomous-complete-h4-fisher-xy:provider:v1\0"
_FIT_RECEIPT_DOMAIN = b"fisher-graph:autonomous-complete-h4-fisher-xy:fit:v1\0"
_GEOMETRY_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-xy:bounded-geometry:v1\0"
)
_STATE_SCHEMA = (
    "fisher_graph.autonomous_complete_h4_fisher_xy_provider_tensor.v1"
)
_STATE_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "provider_artifact_sha256",
        "bridge_binding_sha256",
        "parent_provider_artifact_sha256",
        "router_ridge",
        "conditional_ridge",
        "operator_norm_bound",
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
        "weighted_residual_rmse_before",
        "weighted_residual_rmse_after",
        "pre_projection_corner_operator_norms",
        "post_projection_corner_operator_norms",
        "trust_projection_scale",
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
        "conditional_left",
        "conditional_right",
    }
)


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _positive_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} values")
    values = tuple(
        _positive_float(item, label=f"{label}[{index}]")
        if positive
        else _nonnegative_float(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return values


def _finite_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} values")
    return tuple(
        _finite_float(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def fisher_xy_bounded_coordinates(raw: Tensor, scales: Tensor) -> Tensor:
    """Apply the exact rational map into the open coordinate square."""

    if (
        not isinstance(raw, Tensor)
        or raw.ndim != 2
        or not raw.is_floating_point()
        or not bool(torch.isfinite(raw).all())
        or not isinstance(scales, Tensor)
        or scales.ndim != 1
        or not scales.is_floating_point()
        or not bool(torch.isfinite(scales).all())
    ):
        raise ValueError("Fisher-XY coordinates must be finite floating tensors")
    # This helper is part of the runtime path.  Unlike the serialization
    # canonicalizer, it must not detach or silently migrate live activations to
    # CPU.  Float64 is the provider's arithmetic contract, on the input device.
    values = raw.to(dtype=torch.float64)
    scale = scales.to(device=values.device, dtype=torch.float64)
    if (
        values.shape[1] != FISHER_XY_COORDINATE_COUNT
        or scale.shape != (FISHER_XY_COORDINATE_COUNT,)
        or bool((scale <= 0).any())
    ):
        raise ValueError("Fisher-XY coordinate geometry differs")
    bounded = values / (scale.unsqueeze(0) + values.abs())
    # At extreme finite magnitudes, IEEE addition can round ``scale + |u|``
    # back to ``|u|``.  Clamp that representational endpoint to the closest
    # float64 strictly inside the interval promised by the mathematical map.
    one = torch.ones((), dtype=torch.float64, device=values.device)
    interior = torch.nextafter(one, torch.zeros_like(one))
    bounded = bounded.clamp(min=-interior, max=interior)
    if (
        not bool(torch.isfinite(bounded).all())
        or bool((bounded.abs() >= 1.0).any())
    ):
        raise RuntimeError("Fisher-XY rational coordinates escaped (-1, 1)")
    return bounded.contiguous()


@dataclass(frozen=True, slots=True)
class FisherXYBoundedCoordinateGeometry:
    """Authenticated scalar geometry of actual bounded runtime coordinates."""

    row_count: int
    bounded_coordinates_sha256: str
    row_weight_sha256: str
    covariance_eigenvalues: tuple[float, float]
    lambda2_over_lambda1: float
    abs_correlation: float
    residual_second_coordinate_energy_fraction: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("bounded coordinate geometry row_count must be positive")
        for name in ("bounded_coordinates_sha256", "row_weight_sha256"):
            _require_sha256(getattr(self, name), label=name)
        eigenvalues = _float_tuple(
            self.covariance_eigenvalues,
            count=2,
            label="covariance_eigenvalues",
        )
        if eigenvalues[0] < eigenvalues[1]:
            raise ValueError("bounded coordinate eigenvalues must be descending")
        object.__setattr__(self, "covariance_eigenvalues", eigenvalues)
        ratio = _nonnegative_float(
            self.lambda2_over_lambda1,
            label="lambda2_over_lambda1",
        )
        expected_ratio = (
            0.0 if eigenvalues[0] == 0.0 else eigenvalues[1] / eigenvalues[0]
        )
        if ratio > 1.0 or not math.isclose(
            ratio,
            expected_ratio,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("bounded coordinate eigenvalue ratio differs")
        object.__setattr__(self, "lambda2_over_lambda1", ratio)
        correlation = _nonnegative_float(
            self.abs_correlation,
            label="abs_correlation",
        )
        if correlation > 1.0:
            raise ValueError("bounded coordinate absolute correlation exceeds one")
        object.__setattr__(self, "abs_correlation", correlation)
        residual_fraction = _nonnegative_float(
            self.residual_second_coordinate_energy_fraction,
            label="residual_second_coordinate_energy_fraction",
        )
        if residual_fraction > 1.0:
            raise ValueError("bounded coordinate residual energy exceeds one")
        object.__setattr__(
            self,
            "residual_second_coordinate_energy_fraction",
            residual_fraction,
        )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="bounded coordinate geometry artifact",
            ) != computed:
                raise ValueError("bounded coordinate geometry artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.fisher_xy_bounded_coordinate_geometry.v1",
            "row_count": self.row_count,
            "bounded_coordinates_sha256": self.bounded_coordinates_sha256,
            "row_weight_sha256": self.row_weight_sha256,
            "covariance_eigenvalues": self.covariance_eigenvalues,
            "lambda2_over_lambda1": self.lambda2_over_lambda1,
            "abs_correlation": self.abs_correlation,
            "residual_second_coordinate_energy_fraction": (
                self.residual_second_coordinate_energy_fraction
            ),
            "weight_semantics": "positive_normalized_row_weights",
            "centering_semantics": "fit_weighted_coordinate_mean",
            "residual_semantics": (
                "weighted_centered_coordinate_2_after_linear_projection_on_"
                "centered_coordinate_1"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_GEOMETRY_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("bounded coordinate geometry payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def summarize_fisher_xy_bounded_coordinate_geometry(
    coordinates: Tensor,
    weights: Tensor | None = None,
) -> FisherXYBoundedCoordinateGeometry:
    """Return a tensor-free authenticated summary of bounded coordinate rows."""

    coordinate_rows = _float_tensor(
        coordinates,
        label="bounded coordinates",
        ndim=2,
    )
    if coordinate_rows.shape[1] != FISHER_XY_COORDINATE_COUNT or bool(
        (coordinate_rows.abs() >= 1.0).any()
    ):
        raise ValueError("bounded coordinates must have finite shape [rows, 2]")
    if weights is None:
        row_weights = torch.full(
            (coordinate_rows.shape[0],),
            1.0 / coordinate_rows.shape[0],
            dtype=torch.float64,
        )
    else:
        row_weights = _float_tensor(weights, label="row weights", ndim=1)
        if row_weights.shape != (coordinate_rows.shape[0],) or bool(
            (row_weights <= 0).any()
        ):
            raise ValueError("bounded coordinate row weights differ")
        row_weights = (row_weights / row_weights.sum()).contiguous()

    coordinate_mean = (row_weights.unsqueeze(1) * coordinate_rows).sum(dim=0)
    centered = coordinate_rows - coordinate_mean
    covariance = centered.T @ (row_weights.unsqueeze(1) * centered)
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    lambda_1 = float(eigenvalues[0])
    lambda_2 = float(eigenvalues[1])
    eigenvalue_ratio = 0.0 if lambda_1 == 0.0 else lambda_2 / lambda_1

    variance_1 = float(covariance[0, 0])
    variance_2 = float(covariance[1, 1])
    covariance_12 = float(covariance[0, 1])
    if variance_1 > 0.0 and variance_2 > 0.0:
        absolute_correlation = abs(covariance_12) / math.sqrt(
            variance_1 * variance_2
        )
        absolute_correlation = max(0.0, min(1.0, absolute_correlation))
    else:
        absolute_correlation = 0.0

    if variance_2 == 0.0:
        residual_energy_fraction = 0.0
    elif variance_1 == 0.0:
        residual_energy_fraction = 1.0
    else:
        projection = covariance_12 / variance_1
        residual = centered[:, 1] - projection * centered[:, 0]
        residual_energy = float((row_weights * residual.square()).sum())
        residual_energy_fraction = max(
            0.0,
            min(1.0, residual_energy / variance_2),
        )

    return FisherXYBoundedCoordinateGeometry(
        row_count=int(coordinate_rows.shape[0]),
        bounded_coordinates_sha256=_tensor_sha256(coordinate_rows),
        row_weight_sha256=_tensor_sha256(row_weights),
        covariance_eigenvalues=(lambda_1, lambda_2),
        lambda2_over_lambda1=eigenvalue_ratio,
        abs_correlation=absolute_correlation,
        residual_second_coordinate_energy_fraction=residual_energy_fraction,
    )


def _bounded_coordinate_target_r2(
    coordinates: Tensor,
    bounded_targets: Tensor,
    weights: Tensor,
) -> tuple[float, float]:
    if coordinates.shape != bounded_targets.shape or coordinates.shape[1] != 2:
        raise ValueError("bounded coordinate target geometry differs")
    coordinate_rows = coordinates.to(device="cpu", dtype=torch.float64)
    target_rows = bounded_targets.to(device="cpu", dtype=torch.float64)
    row_weights = weights.to(device="cpu", dtype=torch.float64)
    row_weights = row_weights / row_weights.sum()
    target_mean = (row_weights.unsqueeze(1) * target_rows).sum(dim=0)
    centered_target = target_rows - target_mean
    values: list[float] = []
    for index in range(FISHER_XY_COORDINATE_COUNT):
        squared_error = float(
            (
                row_weights
                * (coordinate_rows[:, index] - target_rows[:, index]).square()
            ).sum()
        )
        target_energy = float(
            (row_weights * centered_target[:, index].square()).sum()
        )
        r2 = (
            (1.0 if squared_error == 0.0 else 0.0)
            if target_energy == 0.0
            else 1.0 - squared_error / target_energy
        )
        if not math.isfinite(r2):
            raise RuntimeError("bounded coordinate target R-squared became nonfinite")
        values.append(r2)
    return values[0], values[1]


def _conditional_matrices(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if (
        not isinstance(left, Tensor)
        or not isinstance(right, Tensor)
        or left.ndim != 2
        or right.ndim != 2
        or not left.is_floating_point()
        or not right.is_floating_point()
        or left.shape[1] != right.shape[0]
        or right.shape[1] <= 0
        or left.shape[0] != _FEATURE_BLOCK_COUNT * right.shape[1]
    ):
        raise ValueError("Fisher-XY conditional factor geometry differs")
    rank = int(right.shape[1])
    blocks = left.to(dtype=torch.float64).reshape(
        _FEATURE_BLOCK_COUNT,
        rank,
        int(right.shape[0]),
    )
    output = right.to(device=blocks.device, dtype=torch.float64)
    return tuple(blocks[index] @ output for index in range(3))  # type: ignore[return-value]


def _corner_operator_norms(left: Tensor, right: Tensor) -> tuple[float, float, float, float]:
    a_x, a_y, a_xy = _conditional_matrices(left, right)
    norms = tuple(
        float(torch.linalg.svdvals(sx * a_x + sy * a_y + sx * sy * a_xy).max())
        for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )
    if any(not math.isfinite(value) for value in norms):
        raise ValueError("Fisher-XY corner operator norm is nonfinite")
    return norms  # type: ignore[return-value]


def project_fisher_xy_conditional_factors(
    left: Tensor,
    right: Tensor,
    *,
    operator_norm_bound: float = FISHER_XY_OPERATOR_NORM_BOUND,
) -> tuple[Tensor, Tensor, tuple[float, float, float, float], tuple[float, float, float, float], float]:
    """Globally project a factor pair by its four square-corner norms."""

    bound = _positive_float(operator_norm_bound, label="operator norm bound")
    canonical_left = _float_tensor(left, label="conditional left", ndim=2)
    canonical_right = _float_tensor(right, label="conditional right", ndim=2)
    pre = _corner_operator_norms(canonical_left, canonical_right)
    maximum = max(pre)
    scale = 1.0 if maximum <= bound else bound * (1.0 - 1.0e-12) / maximum
    projected_right = (canonical_right * scale).contiguous()
    post = _corner_operator_norms(canonical_left, projected_right)
    if max(post) > bound + 1.0e-12:
        raise RuntimeError("Fisher-XY four-corner projection failed")
    return canonical_left, projected_right, pre, post, scale


def _weighted_ridge(
    design: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    if (
        design.ndim != 2
        or target.ndim != 2
        or design.shape[0] != target.shape[0]
        or weights.shape != (design.shape[0],)
        or not bool(torch.isfinite(design).all())
        or not bool(torch.isfinite(target).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
    ):
        raise ValueError("Fisher-XY ridge geometry differs")
    selected_ridge = _positive_float(ridge, label="ridge")
    rms = torch.sqrt((weights.unsqueeze(1) * design.square()).sum(dim=0))
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    standardized = design / scales
    root = weights.sqrt().unsqueeze(1)
    weighted_design = standardized * root
    weighted_target = target * root
    gram = weighted_design.T @ weighted_design
    coefficients = torch.linalg.solve(
        gram
        + selected_ridge
        * torch.eye(gram.shape[0], dtype=torch.float64),
        weighted_design.T @ weighted_target,
    ) / scales.unsqueeze(1)
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("Fisher-XY ridge fit became nonfinite")
    return coefficients.contiguous(), scales.contiguous()


def _canonical_symmetric_axes(
    matrix: Tensor,
    weights: Tensor,
    *,
    center: bool,
    label: str,
) -> tuple[Tensor, tuple[float, float]]:
    rows = matrix
    if center:
        rows = rows - (weights.unsqueeze(1) * rows).sum(dim=0)
    moment = rows.T @ (weights.unsqueeze(1) * rows)
    moment = ((moment + moment.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    order = torch.argsort(eigenvalues, descending=True)
    selected_values = eigenvalues.index_select(0, order[:2])
    axes = eigenvectors.index_select(1, order[:2]).T.contiguous()
    tolerance = max(float(selected_values[0]), 1.0) * 1.0e-12
    if (
        axes.shape != (FISHER_XY_COORDINATE_COUNT, matrix.shape[1])
        or float(selected_values[1]) <= tolerance
    ):
        raise ValueError(f"Fisher-XY fit requires two supported {label} axes")
    for row in range(FISHER_XY_COORDINATE_COUNT):
        pivot = int(axes[row].abs().argmax())
        if float(axes[row, pivot]) < 0.0:
            axes[row].neg_()
    return axes, (float(selected_values[0]), float(selected_values[1]))


def _coordinate_axes_and_targets(
    *,
    coordinate_objective: str,
    modal: Tensor,
    gradient_modal: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor, tuple[float, float]]:
    if coordinate_objective == "reverse_vjp_fisher":
        axes, values = _canonical_symmetric_axes(
            gradient_modal,
            weights,
            center=False,
            label="reverse-VJP Fisher",
        )
        return axes, gradient_modal @ axes.T, values
    if coordinate_objective == "activation_pca":
        axes, values = _canonical_symmetric_axes(
            modal,
            weights,
            center=True,
            label="activation-PCA",
        )
        mean = (weights.unsqueeze(1) * modal).sum(dim=0)
        return axes, (modal - mean) @ axes.T, values
    raise ValueError(
        "coordinate_objective must be 'reverse_vjp_fisher' or 'activation_pca'"
    )


def _training_parent_modal(
    parent: AutonomousCompleteH4ResidualProvider,
    sequence: AutonomousCompleteH4TrainingSequence,
) -> Tensor:
    source_design = build_causal_lagged_modal_design(
        sequence.source_modes,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        lag_count=parent.lag_count,
    )
    encoder = parent.output_decoder if parent.state_encoder is None else parent.state_encoder
    state_modes = sequence.base_h4 @ encoder.T
    modal = (
        source_design @ parent.lag_source_kernel.reshape(-1, parent.rank)
        + state_modes @ parent.state_kernel
        + parent.bias
    )
    modal = modal.masked_fill((~sequence.support_mask).unsqueeze(-1), 0.0)
    if not bool(torch.isfinite(modal[sequence.support_mask]).all()):
        raise RuntimeError("parent training modal prediction became nonfinite")
    return modal.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherXYProvider(Gemma3L3L4CorrectionProvider):
    """One immutable source-free Fisher-XY conditional H4 provider."""

    parent_provider: AutonomousCompleteH4ResidualProvider
    router_weight: Tensor
    router_bias: Tensor
    coordinate_scales: Tensor
    conditional_left: Tensor
    conditional_right: Tensor
    router_ridge: float
    conditional_ridge: float
    operator_norm_bound: float
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
    weighted_residual_rmse_before: float
    weighted_residual_rmse_after: float
    pre_projection_corner_operator_norms: tuple[float, float, float, float]
    post_projection_corner_operator_norms: tuple[float, float, float, float]
    trust_projection_scale: float
    fit_receipt_sha256: str
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.parent_provider, AutonomousCompleteH4ResidualProvider):
            raise TypeError("Fisher-XY parent must be autonomous complete-H4")
        self.parent_provider.validate_integrity()
        router_weight = _float_tensor(self.router_weight, label="router_weight", ndim=2)
        router_bias = _float_tensor(self.router_bias, label="router_bias", ndim=1)
        coordinate_scales = _float_tensor(
            self.coordinate_scales,
            label="coordinate_scales",
            ndim=1,
        )
        left = _float_tensor(self.conditional_left, label="conditional_left", ndim=2)
        right = _float_tensor(self.conditional_right, label="conditional_right", ndim=2)
        rank = self.parent_provider.rank
        if (
            router_weight.shape != (rank, FISHER_XY_COORDINATE_COUNT)
            or router_bias.shape != (FISHER_XY_COORDINATE_COUNT,)
            or coordinate_scales.shape != (FISHER_XY_COORDINATE_COUNT,)
            or bool((coordinate_scales <= 0).any())
            or left.shape[0] != _FEATURE_BLOCK_COUNT * rank
            or right.shape[1] != rank
            or left.shape[1] != right.shape[0]
        ):
            raise ValueError("Fisher-XY provider tensor geometry differs")
        for name, value in (
            ("router_weight", router_weight),
            ("router_bias", router_bias),
            ("coordinate_scales", coordinate_scales),
            ("conditional_left", left),
            ("conditional_right", right),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "router_ridge", _positive_float(self.router_ridge, label="router_ridge"))
        object.__setattr__(self, "conditional_ridge", _positive_float(self.conditional_ridge, label="conditional_ridge"))
        object.__setattr__(self, "operator_norm_bound", _positive_float(self.operator_norm_bound, label="operator_norm_bound"))
        if type(self.fit_row_count) is not int or self.fit_row_count <= 0:
            raise ValueError("Fisher-XY fit_row_count must be positive")
        if (
            type(self.fit_family_ids) is not tuple
            or not self.fit_family_ids
            or self.fit_family_ids != tuple(sorted(set(self.fit_family_ids)))
            or type(self.fit_sequence_sha256s) is not tuple
            or not self.fit_sequence_sha256s
            or self.fit_sequence_sha256s != tuple(sorted(set(self.fit_sequence_sha256s)))
        ):
            raise ValueError("Fisher-XY fit ownership must be canonical")
        for value in self.fit_sequence_sha256s:
            _require_sha256(value, label="Fisher-XY fit sequence")
        if (
            not isinstance(self.coordinate_objective, str)
            or self.coordinate_objective not in COORDINATE_OBJECTIVES
        ):
            raise ValueError("Fisher-XY coordinate objective differs")
        for name in (
            "coordinate_axes_sha256",
            "fit_weight_sha256",
            "bounded_coordinate_geometry_sha256",
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
        for name in (
            "coordinate_target_weighted_rmse",
            "weighted_residual_rmse_before",
            "weighted_residual_rmse_after",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), label=name),
            )
        covariance_eigenvalues = _float_tuple(
            self.bounded_coordinate_covariance_eigenvalues,
            count=2,
            label="bounded_coordinate_covariance_eigenvalues",
        )
        if covariance_eigenvalues[0] < covariance_eigenvalues[1]:
            raise ValueError(
                "bounded coordinate covariance eigenvalues must be descending"
            )
        object.__setattr__(
            self,
            "bounded_coordinate_covariance_eigenvalues",
            covariance_eigenvalues,
        )
        lambda_ratio = _nonnegative_float(
            self.bounded_coordinate_lambda2_over_lambda1,
            label="bounded_coordinate_lambda2_over_lambda1",
        )
        expected_ratio = (
            0.0
            if covariance_eigenvalues[0] == 0.0
            else covariance_eigenvalues[1] / covariance_eigenvalues[0]
        )
        if (
            lambda_ratio > 1.0
            or not math.isclose(
                lambda_ratio,
                expected_ratio,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("bounded coordinate covariance ratio differs")
        object.__setattr__(
            self,
            "bounded_coordinate_lambda2_over_lambda1",
            lambda_ratio,
        )
        absolute_correlation = _nonnegative_float(
            self.bounded_coordinate_abs_correlation,
            label="bounded_coordinate_abs_correlation",
        )
        if absolute_correlation > 1.0:
            raise ValueError("bounded coordinate absolute correlation exceeds one")
        object.__setattr__(
            self,
            "bounded_coordinate_abs_correlation",
            absolute_correlation,
        )
        target_r2 = _finite_tuple(
            self.bounded_coordinate_target_r2,
            count=2,
            label="bounded_coordinate_target_r2",
        )
        if any(value > 1.0 for value in target_r2):
            raise ValueError("bounded coordinate target R-squared exceeds one")
        object.__setattr__(self, "bounded_coordinate_target_r2", target_r2)
        residual_fraction = _nonnegative_float(
            self.residual_second_coordinate_energy_fraction,
            label="residual_second_coordinate_energy_fraction",
        )
        if residual_fraction > 1.0:
            raise ValueError(
                "residual second-coordinate energy fraction exceeds one"
            )
        object.__setattr__(
            self,
            "residual_second_coordinate_energy_fraction",
            residual_fraction,
        )
        pre = _float_tuple(
            self.pre_projection_corner_operator_norms,
            count=4,
            label="pre_projection_corner_operator_norms",
        )
        post = _float_tuple(
            self.post_projection_corner_operator_norms,
            count=4,
            label="post_projection_corner_operator_norms",
        )
        object.__setattr__(self, "pre_projection_corner_operator_norms", pre)
        object.__setattr__(self, "post_projection_corner_operator_norms", post)
        scale = _positive_float(self.trust_projection_scale, label="trust_projection_scale")
        if scale > 1.0:
            raise ValueError("Fisher-XY trust projection scale cannot exceed one")
        object.__setattr__(self, "trust_projection_scale", scale)
        observed = _corner_operator_norms(left, right)
        tolerance = max(self.operator_norm_bound, 1.0) * 1.0e-10
        if (
            max(observed) > self.operator_norm_bound + 1.0e-12
            or any(abs(a - b) > tolerance for a, b in zip(observed, post, strict=True))
            or any(abs(a * scale - b) > tolerance for a, b in zip(pre, post, strict=True))
        ):
            raise ValueError("Fisher-XY four-corner trust receipt differs")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(self.artifact_sha256, label="Fisher-XY provider") != computed:
                raise ValueError("Fisher-XY provider artifact hash mismatch")
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
        return int(self.conditional_right.shape[0])

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return int(
            self.router_weight.numel()
            + self.router_bias.numel()
            + self.coordinate_scales.numel()
            + self.conditional_left.numel()
            + self.conditional_right.numel()
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
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_provider.logical_macs_per_token_upper_bound
            + self.incremental_logical_macs_per_token_upper_bound
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.autonomous_complete_h4_fisher_xy_provider.v1",
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "tensor_sha256s": {
                "router_weight": _tensor_sha256(self.router_weight),
                "router_bias": _tensor_sha256(self.router_bias),
                "coordinate_scales": _tensor_sha256(self.coordinate_scales),
                "conditional_left": _tensor_sha256(self.conditional_left),
                "conditional_right": _tensor_sha256(self.conditional_right),
            },
            "router_ridge": self.router_ridge,
            "conditional_ridge": self.conditional_ridge,
            "operator_norm_bound": self.operator_norm_bound,
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
            "weighted_residual_rmse_before": self.weighted_residual_rmse_before,
            "weighted_residual_rmse_after": self.weighted_residual_rmse_after,
            "pre_projection_corner_operator_norms": self.pre_projection_corner_operator_norms,
            "post_projection_corner_operator_norms": self.post_projection_corner_operator_norms,
            "trust_projection_scale": self.trust_projection_scale,
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
            "conditional_feature_semantics": "c1_p_c2_p_c1_c2_p",
            "interior_bound_proof": "multi_affine_convex_interpolation_of_four_corners",
            "corner_certificate_scope": (
                "pointwise_conditional_correction_amplitude_operator_bound_"
                "not_full_nonlinear_jacobian_or_lipschitz_bound"
            ),
            "bounded_coordinate_diagnostic_semantics": (
                "fit_weighted_actual_bounded_runtime_router_coordinates_v1"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        self.parent_provider.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("Fisher-XY provider tensor payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": self.incremental_prepared_float_scalar_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": self.incremental_prepared_float_scalar_count * 8,
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "runtime_parameter_bytes_float64_scope": (
                "prepared_float_payload_not_peak_runtime_memory"
            ),
            "incremental_logical_macs_per_token_upper_bound": self.incremental_logical_macs_per_token_upper_bound,
            "logical_macs_per_token_upper_bound": self.logical_macs_per_token_upper_bound,
            "logical_macs_accounting_scope": (
                "matrix_multiply_accumulates_excluding_integrity_hashes_"
                "device_transfers_and_temporary_workspace"
            ),
            "runtime_state_float_scalars_per_sequence": 0,
            "rational_coordinate_abs_ops_per_token": 2,
            "rational_coordinate_add_ops_per_token": 2,
            "rational_coordinate_division_ops_per_token": 2,
            "rational_coordinate_open_bound_clamp_ops_per_token": 4,
            "router_bias_additions_per_token": 2,
            "conditional_gate_scalar_multiplications_per_token": 3 * self.rank + 1,
            "conditional_modal_additions_per_token": self.rank,
        }

    def bounded_coordinates(self, modal: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(modal, Tensor)
            or modal.ndim != 3
            or modal.shape[-1] != self.rank
            or not modal.is_floating_point()
        ):
            raise ValueError("Fisher-XY modal router input differs")
        flat = modal.to(dtype=torch.float64).reshape(-1, self.rank)
        raw = flat @ self.router_weight.to(flat.device) + self.router_bias.to(flat.device)
        bounded = fisher_xy_bounded_coordinates(
            raw,
            self.coordinate_scales.to(raw.device),
        )
        return bounded.reshape(*modal.shape[:2], FISHER_XY_COORDINATE_COUNT)

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        prefix_sha = prefix.artifact_sha256
        realized_sha = _tensor_sha256(realized_state)
        parent_modal = self.parent_provider.modal_correction(prefix, realized_state)
        coordinates = self.bounded_coordinates(parent_modal)
        c_x = coordinates[..., 0].unsqueeze(-1)
        c_y = coordinates[..., 1].unsqueeze(-1)
        features = torch.cat(
            (c_x * parent_modal, c_y * parent_modal, c_x * c_y * parent_modal),
            dim=-1,
        )
        conditional = (
            features @ self.conditional_left.to(features.device)
        ) @ self.conditional_right.to(features.device)
        modal = parent_modal + conditional
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
        if (
            prefix.artifact_sha256 != prefix_sha
            or _tensor_sha256(realized_state) != realized_sha
        ):
            raise RuntimeError("Fisher-XY provider mutated a runtime input")
        if (
            bool(support.any())
            and not bool(torch.isfinite(modal[support]).all())
        ):
            raise RuntimeError("Fisher-XY modal correction became nonfinite")
        if bool((modal[~support] != 0).any()):
            raise RuntimeError("Fisher-XY modal correction escaped support")
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


def replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
    provider: AutonomousCompleteH4FisherXYProvider,
    sequence: AutonomousCompleteH4TrainingSequence,
) -> Tensor:
    """Replay support-row coordinates from serving-available trace fields only.

    This offline audit path reads source modes, logical positions, execution
    masks, and base H4.  It deliberately never reads native H4, reverse-VJP
    gradients, targets, or logits and does not require fit ownership, so it may
    be used on a held family.
    """

    if not isinstance(provider, AutonomousCompleteH4FisherXYProvider):
        raise TypeError("provider must be autonomous complete-H4 Fisher-XY")
    if not isinstance(sequence, AutonomousCompleteH4TrainingSequence):
        raise TypeError("sequence must be autonomous complete-H4 training data")
    provider.validate_integrity()
    parent_modal = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent_modal.unsqueeze(0))[0]
    support = sequence.support_mask.to(coordinates.device)
    result = (
        coordinates[support]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if (
        result.ndim != 2
        or result.shape[1] != FISHER_XY_COORDINATE_COUNT
        or result.shape[0] != int(sequence.support_mask.sum())
        or not bool(torch.isfinite(result).all())
        or bool((result.abs() >= 1.0).any())
    ):
        raise RuntimeError("held Fisher-XY coordinate replay became invalid")
    provider.validate_integrity()
    return result


def fit_autonomous_complete_h4_fisher_xy_residual(
    *,
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    parent_provider: AutonomousCompleteH4ResidualProvider,
    conditional_rank: int = 16,
    coordinate_objective: str = "reverse_vjp_fisher",
    router_ridge: float = 1.0e-4,
    conditional_ridge: float = 1.0e-4,
    operator_norm_bound: float = FISHER_XY_OPERATOR_NORM_BOUND,
    vjp_weight_floor: float = 0.5,
    vjp_weight_ceiling: float = 2.0,
) -> AutonomousCompleteH4FisherXYProvider:
    """Fit one deterministic family-balanced Fisher-XY provider."""

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
        raise ValueError("Fisher-XY parent and fit ownership differ")
    if (
        type(conditional_rank) is not int
        or conditional_rank <= 0
        or conditional_rank > parent_provider.rank
    ):
        raise ValueError("conditional_rank must lie in [1, parent rank]")
    selected_router_ridge = _positive_float(router_ridge, label="router_ridge")
    selected_conditional_ridge = _positive_float(
        conditional_ridge,
        label="conditional_ridge",
    )
    selected_bound = _positive_float(
        operator_norm_bound,
        label="operator_norm_bound",
    )
    if coordinate_objective not in COORDINATE_OBJECTIVES:
        raise ValueError(
            "coordinate_objective must be 'reverse_vjp_fisher' or 'activation_pca'"
        )
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
                "Fisher-XY and parameter-matched PCA fits require every reverse-VJP gradient"
            )
        selected = sequence.support_mask
        modal = _training_parent_modal(parent_provider, sequence)
        target = (sequence.native_h4 - sequence.base_h4) @ parent_provider.output_decoder.T
        gradient_modal = sequence.reverse_vjp_gradients @ parent_provider.output_decoder.T
        modal_rows.append(modal[selected])
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
    base_weights = family_balanced_row_weights(family_tuple, example_tuple).to(torch.float64)
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
    features = torch.cat(
        (
            coordinates[:, :1] * modal,
            coordinates[:, 1:] * modal,
            coordinates[:, :1] * coordinates[:, 1:] * modal,
        ),
        dim=1,
    )
    residual = target - modal
    multipliers = _bounded_vjp_multipliers(
        gradient_h4,
        example_tuple,
        floor=vjp_weight_floor,
        ceiling=vjp_weight_ceiling,
    )
    fit_weights = (base_weights * multipliers).contiguous()
    fit_weights = fit_weights / fit_weights.sum()
    bounded_coordinate_targets = fisher_xy_bounded_coordinates(
        coordinate_targets,
        coordinate_scales,
    )
    bounded_coordinate_geometry = (
        summarize_fisher_xy_bounded_coordinate_geometry(
            coordinates,
            fit_weights,
        )
    )
    bounded_coordinate_target_r2 = _bounded_coordinate_target_r2(
        coordinates,
        bounded_coordinate_targets,
        fit_weights,
    )
    bounded_coordinate_covariance_eigenvalues = (
        bounded_coordinate_geometry.covariance_eigenvalues
    )
    bounded_coordinate_lambda2_over_lambda1 = (
        bounded_coordinate_geometry.lambda2_over_lambda1
    )
    bounded_coordinate_abs_correlation = (
        bounded_coordinate_geometry.abs_correlation
    )
    residual_second_coordinate_energy_fraction = (
        bounded_coordinate_geometry.residual_second_coordinate_energy_fraction
    )
    dense, _feature_scales = _weighted_ridge(
        features,
        residual,
        fit_weights,
        ridge=selected_conditional_ridge,
    )
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    if conditional_rank > int(singular.numel()):
        raise ValueError("conditional_rank exceeds fitted matrix rank bound")
    left = (u[:, :conditional_rank] * singular[:conditional_rank]).contiguous()
    right = vh[:conditional_rank].contiguous()
    for index in range(conditional_rank):
        pivot = int(right[index].abs().argmax())
        if float(right[index, pivot]) < 0.0:
            right[index].neg_()
            left[:, index].neg_()
    left, right, pre, post, trust_scale = project_fisher_xy_conditional_factors(
        left,
        right,
        operator_norm_bound=selected_bound,
    )
    before = torch.sqrt(
        (fit_weights.unsqueeze(1) * residual.square()).sum()
        / parent_provider.rank
    )
    remaining = residual - (features @ left) @ right
    after = torch.sqrt(
        (fit_weights.unsqueeze(1) * remaining.square()).sum()
        / parent_provider.rank
    )
    router_error = coordinate_targets - raw_coordinates
    router_rmse = torch.sqrt(
        (base_weights.unsqueeze(1) * router_error.square()).sum()
        / FISHER_XY_COORDINATE_COUNT
    )
    fit_receipt = {
        "parent_provider_artifact_sha256": parent_provider.artifact_sha256,
        "fit_sequence_sha256s": expected_sequence_ids,
        "fit_family_ids": expected_families,
        "coordinate_objective": coordinate_objective,
        "coordinate_axes_sha256": _tensor_sha256(axes),
        "coordinate_axis_values": axis_values,
        "router_ridge": selected_router_ridge,
        "conditional_ridge": selected_conditional_ridge,
        "conditional_rank": conditional_rank,
        "operator_norm_bound": selected_bound,
        "vjp_weight_floor": float(vjp_weight_floor),
        "vjp_weight_ceiling": float(vjp_weight_ceiling),
        "fit_weight_sha256": _tensor_sha256(fit_weights),
        "bounded_coordinate_geometry_sha256": (
            bounded_coordinate_geometry.artifact_sha256
        ),
        "bounded_coordinate_covariance_eigenvalues": (
            bounded_coordinate_covariance_eigenvalues
        ),
        "bounded_coordinate_lambda2_over_lambda1": (
            bounded_coordinate_lambda2_over_lambda1
        ),
        "bounded_coordinate_abs_correlation": (
            bounded_coordinate_abs_correlation
        ),
        "bounded_coordinate_target_r2": bounded_coordinate_target_r2,
        "residual_second_coordinate_energy_fraction": (
            residual_second_coordinate_energy_fraction
        ),
        "coordinate_semantics": "u_div_scale_plus_abs_u",
        "conditional_feature_semantics": "c1_p_c2_p_c1_c2_p",
    }
    return AutonomousCompleteH4FisherXYProvider(
        parent_provider=parent_provider,
        router_weight=router_coefficients[:-1],
        router_bias=router_coefficients[-1],
        coordinate_scales=coordinate_scales,
        conditional_left=left,
        conditional_right=right,
        router_ridge=selected_router_ridge,
        conditional_ridge=selected_conditional_ridge,
        operator_norm_bound=selected_bound,
        fit_row_count=int(modal.shape[0]),
        fit_family_ids=expected_families,
        fit_sequence_sha256s=expected_sequence_ids,
        coordinate_objective=coordinate_objective,
        coordinate_axes_sha256=_tensor_sha256(axes),
        coordinate_axis_values=axis_values,
        fit_weight_sha256=_tensor_sha256(fit_weights),
        coordinate_target_weighted_rmse=float(router_rmse),
        bounded_coordinate_geometry_sha256=(
            bounded_coordinate_geometry.artifact_sha256
        ),
        bounded_coordinate_covariance_eigenvalues=(
            bounded_coordinate_covariance_eigenvalues
        ),
        bounded_coordinate_lambda2_over_lambda1=(
            bounded_coordinate_lambda2_over_lambda1
        ),
        bounded_coordinate_abs_correlation=(
            bounded_coordinate_abs_correlation
        ),
        bounded_coordinate_target_r2=bounded_coordinate_target_r2,
        residual_second_coordinate_energy_fraction=(
            residual_second_coordinate_energy_fraction
        ),
        weighted_residual_rmse_before=float(before),
        weighted_residual_rmse_after=float(after),
        pre_projection_corner_operator_norms=pre,
        post_projection_corner_operator_norms=post,
        trust_projection_scale=trust_scale,
        fit_receipt_sha256=_sha256(_FIT_RECEIPT_DOMAIN, fit_receipt),
    )


def autonomous_complete_h4_fisher_xy_provider_state_dict(
    provider: AutonomousCompleteH4FisherXYProvider,
) -> dict[str, object]:
    if not isinstance(provider, AutonomousCompleteH4FisherXYProvider):
        raise TypeError("provider must be AutonomousCompleteH4FisherXYProvider")
    provider.validate_integrity()
    return {
        "schema": _STATE_SCHEMA,
        "format_version": 1,
        "provider_artifact_sha256": provider.artifact_sha256,
        "bridge_binding_sha256": provider.bridge_binding_sha256,
        "parent_provider_artifact_sha256": provider.parent_provider.artifact_sha256,
        "router_ridge": provider.router_ridge,
        "conditional_ridge": provider.conditional_ridge,
        "operator_norm_bound": provider.operator_norm_bound,
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
        "weighted_residual_rmse_before": provider.weighted_residual_rmse_before,
        "weighted_residual_rmse_after": provider.weighted_residual_rmse_after,
        "pre_projection_corner_operator_norms": provider.pre_projection_corner_operator_norms,
        "post_projection_corner_operator_norms": provider.post_projection_corner_operator_norms,
        "trust_projection_scale": provider.trust_projection_scale,
        "fit_receipt_sha256": provider.fit_receipt_sha256,
        "parent_provider_state": autonomous_complete_h4_residual_provider_state_dict(
            provider.parent_provider
        ),
        "tensors": {
            "router_weight": provider.router_weight.detach().clone(),
            "router_bias": provider.router_bias.detach().clone(),
            "coordinate_scales": provider.coordinate_scales.detach().clone(),
            "conditional_left": provider.conditional_left.detach().clone(),
            "conditional_right": provider.conditional_right.detach().clone(),
        },
    }


def autonomous_complete_h4_fisher_xy_provider_from_state_dict(
    state: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4FisherXYProvider:
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("Fisher-XY provider state fields differ")
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected Fisher-XY provider",
    )
    embedded_artifact = _require_sha256(
        state.get("provider_artifact_sha256"),
        label="embedded Fisher-XY provider",
    )
    if embedded_artifact != expected_artifact:
        raise ValueError("Fisher-XY state artifact differs from expected")
    bridge = _require_sha256(
        state.get("bridge_binding_sha256"),
        label="Fisher-XY bridge binding",
    )
    if expected_bridge_binding_sha256 is not None and bridge != _require_sha256(
        expected_bridge_binding_sha256,
        label="expected Fisher-XY bridge binding",
    ):
        raise ValueError("Fisher-XY bridge binding differs from expected")
    parent_artifact = _require_sha256(
        state.get("parent_provider_artifact_sha256"),
        label="Fisher-XY parent provider",
    )
    parent_state = state.get("parent_provider_state")
    if not isinstance(parent_state, Mapping):
        raise ValueError("Fisher-XY parent state must be a mapping")
    parent = autonomous_complete_h4_residual_provider_from_state_dict(
        parent_state,
        expected_artifact_sha256=parent_artifact,
        expected_bridge_binding_sha256=bridge,
    )
    tensors = state.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("Fisher-XY state tensor fields differ")
    if any(not isinstance(tensors.get(name), Tensor) for name in _TENSOR_KEYS):
        raise ValueError("Fisher-XY state tensors must be tensors")
    if (
        state.get("schema") != _STATE_SCHEMA
        or state.get("format_version") != 1
        or type(state.get("fit_row_count")) is not int
        or type(state.get("fit_family_ids")) is not tuple
        or type(state.get("fit_sequence_sha256s")) is not tuple
        or not isinstance(state.get("coordinate_objective"), str)
    ):
        raise ValueError("Fisher-XY state scalar contract differs")
    provider = AutonomousCompleteH4FisherXYProvider(
        parent_provider=parent,
        router_weight=tensors["router_weight"],
        router_bias=tensors["router_bias"],
        coordinate_scales=tensors["coordinate_scales"],
        conditional_left=tensors["conditional_left"],
        conditional_right=tensors["conditional_right"],
        router_ridge=_positive_float(state.get("router_ridge"), label="router_ridge"),
        conditional_ridge=_positive_float(state.get("conditional_ridge"), label="conditional_ridge"),
        operator_norm_bound=_positive_float(state.get("operator_norm_bound"), label="operator_norm_bound"),
        fit_row_count=state["fit_row_count"],  # type: ignore[arg-type]
        fit_family_ids=state["fit_family_ids"],  # type: ignore[arg-type]
        fit_sequence_sha256s=state["fit_sequence_sha256s"],  # type: ignore[arg-type]
        coordinate_objective=state.get("coordinate_objective"),  # type: ignore[arg-type]
        coordinate_axes_sha256=_require_sha256(state.get("coordinate_axes_sha256"), label="coordinate_axes_sha256"),
        coordinate_axis_values=_float_tuple(state.get("coordinate_axis_values"), count=2, label="coordinate_axis_values", positive=True),  # type: ignore[arg-type]
        fit_weight_sha256=_require_sha256(state.get("fit_weight_sha256"), label="fit_weight_sha256"),
        coordinate_target_weighted_rmse=_nonnegative_float(state.get("coordinate_target_weighted_rmse"), label="coordinate_target_weighted_rmse"),
        bounded_coordinate_geometry_sha256=_require_sha256(state.get("bounded_coordinate_geometry_sha256"), label="bounded_coordinate_geometry_sha256"),
        bounded_coordinate_covariance_eigenvalues=_float_tuple(state.get("bounded_coordinate_covariance_eigenvalues"), count=2, label="bounded_coordinate_covariance_eigenvalues"),  # type: ignore[arg-type]
        bounded_coordinate_lambda2_over_lambda1=_nonnegative_float(state.get("bounded_coordinate_lambda2_over_lambda1"), label="bounded_coordinate_lambda2_over_lambda1"),
        bounded_coordinate_abs_correlation=_nonnegative_float(state.get("bounded_coordinate_abs_correlation"), label="bounded_coordinate_abs_correlation"),
        bounded_coordinate_target_r2=_finite_tuple(state.get("bounded_coordinate_target_r2"), count=2, label="bounded_coordinate_target_r2"),  # type: ignore[arg-type]
        residual_second_coordinate_energy_fraction=_nonnegative_float(state.get("residual_second_coordinate_energy_fraction"), label="residual_second_coordinate_energy_fraction"),
        weighted_residual_rmse_before=_nonnegative_float(state.get("weighted_residual_rmse_before"), label="weighted_residual_rmse_before"),
        weighted_residual_rmse_after=_nonnegative_float(state.get("weighted_residual_rmse_after"), label="weighted_residual_rmse_after"),
        pre_projection_corner_operator_norms=_float_tuple(state.get("pre_projection_corner_operator_norms"), count=4, label="pre_projection_corner_operator_norms"),  # type: ignore[arg-type]
        post_projection_corner_operator_norms=_float_tuple(state.get("post_projection_corner_operator_norms"), count=4, label="post_projection_corner_operator_norms"),  # type: ignore[arg-type]
        trust_projection_scale=_positive_float(state.get("trust_projection_scale"), label="trust_projection_scale"),
        fit_receipt_sha256=_require_sha256(state.get("fit_receipt_sha256"), label="fit_receipt_sha256"),
        artifact_sha256=embedded_artifact,
    )
    provider.validate_integrity()
    return provider


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Fisher-XY path is not a readable regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError("Fisher-XY path must be a nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise RuntimeError("Fisher-XY provider file changed while reading")
    return payload


def _provider_from_bytes(
    payload: bytes,
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None,
) -> AutonomousCompleteH4FisherXYProvider:
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Fisher-XY provider tensor payload is invalid") from error
    if not isinstance(state, Mapping):
        raise ValueError("Fisher-XY provider payload must contain a mapping")
    return autonomous_complete_h4_fisher_xy_provider_from_state_dict(
        state,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )


def save_autonomous_complete_h4_fisher_xy_provider(
    provider: AutonomousCompleteH4FisherXYProvider,
    path: Path | str,
) -> dict[str, object]:
    if not isinstance(provider, AutonomousCompleteH4FisherXYProvider):
        raise TypeError("provider must be AutonomousCompleteH4FisherXYProvider")
    provider.validate_integrity()
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("Fisher-XY provider output must use .pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite Fisher-XY provider")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    stage = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(autonomous_complete_h4_fisher_xy_provider_state_dict(provider), handle)
            handle.flush()
            os.fsync(handle.fileno())
        payload = _read_regular_file(stage)
        restored = _provider_from_bytes(
            payload,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_bridge_binding_sha256=provider.bridge_binding_sha256,
        )
        if restored.metadata() != provider.metadata():
            raise RuntimeError("staged Fisher-XY provider roundtrip drifted")
        try:
            os.link(stage, destination)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite Fisher-XY provider") from error
        published = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return {
            "path": destination.as_posix(),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "file_bytes": len(payload),
            "provider_artifact_sha256": provider.artifact_sha256,
            "bridge_binding_sha256": provider.bridge_binding_sha256,
        }
    except BaseException:
        if published:
            raise RuntimeError("Fisher-XY provider publication durability is uncertain")
        raise
    finally:
        stage.unlink(missing_ok=True)


def load_autonomous_complete_h4_fisher_xy_provider(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str | None = None,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4FisherXYProvider:
    payload = _read_regular_file(Path(path))
    if expected_file_sha256 is not None:
        expected_file = _require_sha256(
            expected_file_sha256,
            label="expected Fisher-XY tensor file",
        )
        if hashlib.sha256(payload).hexdigest() != expected_file:
            raise ValueError("Fisher-XY tensor file differs from expected")
    return _provider_from_bytes(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )
