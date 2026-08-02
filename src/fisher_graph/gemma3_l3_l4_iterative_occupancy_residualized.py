"""Development-only fold-local residualization for Iteration-5 occupancy.

The two occupancy Jacobian columns are projected away from the four existing
shared/balance columns using only the training families in a fold:

``A = (sqrt(W) B)^+ sqrt(W) O`` and ``R = O - B A``.

Ridge fitting happens in ``[B, R]`` coordinates.  The fitted coefficients are
then mapped back into the unchanged six-coordinate runtime basis:

``theta_B = gamma_B - A gamma_O`` and ``theta_O = gamma_O``.

The projection matrix is fit-only metadata.  Serving still evaluates the
original occupancy route with six learned scalars and four causal state
scalars; this module adds no runtime operation or state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor

from .gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    GemmaCausalTop2OccupancyConformalRouteH4Provider,
    GemmaIterativeOccupancyConformalRouteFitRecord,
    GemmaIterativeOccupancyConformalRouteFoldFit,
    OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
    OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
    OCCUPANCY_FIT_COORDINATE_RESIDUALIZED,
    OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE,
    OCCUPANCY_RESIDUAL_SVD_ABSOLUTE_TOLERANCE,
    OCCUPANCY_STANDARDIZED_RIDGE,
    _COLUMN_SUPPORT_EPSILON,
    _condition_number,
    _kind,
    _project_coefficients,
    _record,
)
from .gemma3_l3_l4_two_head_lowerer import GemmaCausalResidualHead


__all__ = [
    "OCCUPANCY_RESIDUAL_MAP_TOLERANCE",
    "fit_gemma_iterative_residualized_occupancy_route_fold",
    "fit_gemma_iterative_residualized_occupancy_route_fold_provider",
    "fit_gemma_iterative_residualized_occupancy_route_full_provider",
    "occupancy_residual_basis_coefficients",
    "occupancy_residual_projection_matrix",
]


OCCUPANCY_RESIDUAL_MAP_TOLERANCE = 1.0e-10
_FULL_FIT = "__full_fit__"


def _canonical_training_records(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> tuple[GemmaIterativeOccupancyConformalRouteFitRecord, ...]:
    selected = tuple(
        sorted((_record(value) for value in records), key=lambda x: x.example_id)
    )
    if not selected or len({row.example_id for row in selected}) != len(
        selected
    ):
        raise ValueError("fit records must be nonempty and unique")
    family_counts = Counter(row.family_id for row in selected)
    if held_family_id != _FULL_FIT and held_family_id in family_counts:
        raise ValueError("the held family leaked into residualized training")
    if (
        len({row.parent_h4_artifact_sha256 for row in selected}) != 1
        or len({row.top_mode_indices for row in selected}) != 1
        or len({row.top_mode_norms for row in selected}) != 1
    ):
        raise ValueError(
            "fit records belong to different residualized occupancy features"
        )
    return selected


def _family_balanced_weights(
    records: Sequence[GemmaIterativeOccupancyConformalRouteFitRecord],
) -> Tensor:
    counts = Counter(row.family_id for row in records)
    family_mass = 1.0 / len(counts)
    weights = torch.tensor(
        [family_mass / counts[row.family_id] for row in records],
        dtype=torch.float64,
    )
    if abs(float(weights.sum()) - 1.0) > 1.0e-12:
        raise RuntimeError("residualized family weights do not sum to one")
    return weights


def _weighted_base_projection(
    base: Tensor,
    occupancy: Tensor,
    weights: Tensor,
) -> tuple[Tensor, int]:
    """Return the raw-coordinate minimum-norm weighted map ``B -> O``."""

    base_scales = torch.sqrt((weights[:, None] * base.square()).sum(dim=0))
    supported = tuple(
        index
        for index in range(base.shape[1])
        if float(base_scales[index]) > _COLUMN_SUPPORT_EPSILON
    )
    projection = torch.zeros(
        (base.shape[1], occupancy.shape[1]),
        dtype=torch.float64,
    )
    if not supported:
        return projection, 0

    indices = torch.tensor(supported, dtype=torch.int64)
    selected_scales = base_scales.index_select(0, indices)
    normalized_base = base.index_select(1, indices) / selected_scales
    weighted_base = weights.sqrt().unsqueeze(1) * normalized_base
    weighted_occupancy = weights.sqrt().unsqueeze(1) * occupancy
    u, singular, vh = torch.linalg.svd(weighted_base, full_matrices=False)
    if not bool(torch.isfinite(singular).all()):
        raise RuntimeError("residualized base SVD became nonfinite")
    maximum = float(singular.max()) if singular.numel() else 0.0
    tolerance = max(
        OCCUPANCY_RESIDUAL_SVD_ABSOLUTE_TOLERANCE,
        torch.finfo(torch.float64).eps
        * max(weighted_base.shape)
        * maximum,
    )
    rank = int((singular > tolerance).sum())
    if rank:
        scaled_projection = (
            (vh[:rank].T / singular[:rank])
            @ (u[:, :rank].T @ weighted_occupancy)
        )
        raw_projection = scaled_projection / selected_scales.unsqueeze(1)
        projection.index_copy_(0, indices, raw_projection)
    if not bool(torch.isfinite(projection).all()):
        raise RuntimeError("residualized occupancy projection became nonfinite")
    return projection.contiguous(), rank


def _maximum_weighted_cross_correlation(
    base: Tensor,
    residual: Tensor,
    weights: Tensor,
) -> float:
    base_scales = torch.sqrt((weights[:, None] * base.square()).sum(dim=0))
    residual_scales = torch.sqrt(
        (weights[:, None] * residual.square()).sum(dim=0)
    )
    denominator = base_scales[:, None] * residual_scales[None, :]
    cross = base.T @ (weights[:, None] * residual)
    supported = denominator > _COLUMN_SUPPORT_EPSILON
    if not bool(supported.any()):
        return 0.0
    result = float((cross[supported].abs() / denominator[supported]).max())
    if not math.isfinite(result):
        raise RuntimeError("weighted base-residual correlation is nonfinite")
    return result


def occupancy_residual_projection_matrix(
    fit: GemmaIterativeOccupancyConformalRouteFoldFit,
) -> Tensor:
    """Restore the authenticated 4-by-2 raw-coordinate projection."""

    fit.validate_integrity()
    if fit.fit_coordinate_system != OCCUPANCY_FIT_COORDINATE_RESIDUALIZED:
        raise ValueError("fold fit is not occupancy-residualized")
    return torch.tensor(
        fit.occupancy_projection_on_base_by_base_and_occupancy_coordinate,
        dtype=torch.float64,
    ).reshape(4, 2)


def occupancy_residual_basis_coefficients(
    fit: GemmaIterativeOccupancyConformalRouteFoldFit,
) -> tuple[float, float, float, float, float, float]:
    """Recover projected ``gamma`` from mapped runtime ``theta`` and ``A``."""

    projection = occupancy_residual_projection_matrix(fit)
    theta = torch.tensor(
        fit.coefficients_by_occupancy_conformal_coefficient,
        dtype=torch.float64,
    )
    occupancy = theta[4:]
    base = theta[:4] + projection @ occupancy
    gamma = torch.cat((base, occupancy))
    if not bool(torch.isfinite(gamma).all()):
        raise RuntimeError("residual-basis coefficients became nonfinite")
    return tuple(float(value) for value in gamma)  # type: ignore[return-value]


def fit_gemma_iterative_residualized_occupancy_route_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
    occupancy_kind: str,
) -> GemmaIterativeOccupancyConformalRouteFoldFit:
    """Fit one leakage-safe occupancy arm in fold-local residual coordinates."""

    kind = _kind(occupancy_kind)
    selected = _canonical_training_records(
        records,
        held_family_id=held_family_id,
    )
    family_counts = Counter(row.family_id for row in selected)
    jacobian_field = (
        "jacobian_by_cumulative_occupancy_conformal_coefficient"
        if kind == CENTERED_CUMULATIVE_OCCUPANCY
        else "jacobian_by_ew_occupancy_conformal_coefficient"
    )
    original_design = torch.tensor(
        [getattr(row, jacobian_field) for row in selected],
        dtype=torch.float64,
    )
    target = -torch.tensor(
        [row.parent_signed_delta_nll_per_token for row in selected],
        dtype=torch.float64,
    )
    weights = _family_balanced_weights(selected)
    base = original_design[:, :4]
    occupancy = original_design[:, 4:]
    projection, base_projection_rank = _weighted_base_projection(
        base,
        occupancy,
        weights,
    )
    residual = occupancy - base @ projection
    residualized_design = torch.cat((base, residual), dim=1).contiguous()
    if (
        not bool(torch.isfinite(residual).all())
        or not bool(torch.isfinite(residualized_design).all())
    ):
        raise RuntimeError("residualized occupancy design became nonfinite")

    maximum_correlation = _maximum_weighted_cross_correlation(
        base,
        residual,
        weights,
    )
    if (
        maximum_correlation
        > OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE
    ):
        raise RuntimeError(
            "occupancy residual failed weighted orthogonality"
        )

    occupancy_energy = (
        weights[:, None] * occupancy.square()
    ).sum(dim=0)
    residual_energy = (
        weights[:, None] * residual.square()
    ).sum(dim=0)
    occupancy_scales = occupancy_energy.sqrt()
    residual_energy_fraction = torch.where(
        occupancy_energy > _COLUMN_SUPPORT_EPSILON**2,
        residual_energy / occupancy_energy,
        torch.zeros_like(occupancy_energy),
    )
    if (
        not bool(torch.isfinite(residual_energy_fraction).all())
        or bool((residual_energy_fraction < -1.0e-12).any())
        or bool((residual_energy_fraction > 1.0 + 1.0e-10).any())
    ):
        raise RuntimeError("occupancy residual-energy receipt is invalid")

    scales = torch.sqrt(
        (weights[:, None] * residualized_design.square()).sum(dim=0)
    )
    supported = tuple(
        index
        for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
        if float(scales[index]) > _COLUMN_SUPPORT_EPSILON
    )
    raw_rank, raw_condition = _condition_number(
        residualized_design,
        weights,
    )
    gamma = torch.zeros(
        OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        dtype=torch.float64,
    )
    standardized_rank = 0
    standardized_condition = 0.0
    if supported:
        indices = torch.tensor(supported, dtype=torch.int64)
        standardized = residualized_design.index_select(
            1, indices
        ) / scales.index_select(0, indices)
        standardized_rank, standardized_condition = _condition_number(
            standardized,
            weights,
        )
        normal = standardized.T @ (weights[:, None] * standardized)
        beta = torch.linalg.solve(
            normal
            + OCCUPANCY_STANDARDIZED_RIDGE
            * torch.eye(len(supported), dtype=torch.float64),
            standardized.T @ (weights * target),
        )
        solved = beta / scales.index_select(0, indices)
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("residualized standardized ridge is nonfinite")
        gamma[indices] = solved

    runtime_coefficients = torch.cat(
        (
            gamma[:4] - projection @ gamma[4:],
            gamma[4:],
        )
    )
    transformed_prediction = residualized_design @ gamma
    runtime_prediction = original_design @ runtime_coefficients
    map_denominator = max(
        1.0,
        float(transformed_prediction.abs().max()),
        float(runtime_prediction.abs().max()),
    )
    map_error = float(
        (transformed_prediction - runtime_prediction).abs().max()
    ) / map_denominator
    if map_error > OCCUPANCY_RESIDUAL_MAP_TOLERANCE:
        raise RuntimeError("occupancy residual map-back identity failed")

    (
        runtime_coefficients,
        pre,
        post,
        projection_scale,
        projection_applied,
    ) = _project_coefficients(runtime_coefficients, supported=supported)
    gamma = (gamma * projection_scale).contiguous()
    projected_residual_prediction = residualized_design @ gamma
    projected_runtime_prediction = original_design @ runtime_coefficients
    projected_denominator = max(
        1.0,
        float(projected_residual_prediction.abs().max()),
        float(projected_runtime_prediction.abs().max()),
    )
    projected_map_error = float(
        (
            projected_residual_prediction - projected_runtime_prediction
        ).abs().max()
    ) / projected_denominator
    if projected_map_error > OCCUPANCY_RESIDUAL_MAP_TOLERANCE:
        raise RuntimeError(
            "trust projection failed to commute with occupancy map-back"
        )

    before = float(torch.sqrt((weights * target.square()).sum()))
    after = float(
        torch.sqrt(
            (
                weights
                * (projected_runtime_prediction - target).square()
            ).sum()
        )
    )
    return GemmaIterativeOccupancyConformalRouteFoldFit(
        occupancy_kind=kind,
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_occupancy_conformal_coefficient=tuple(
            float(value) for value in runtime_coefficients
        ),  # type: ignore[arg-type]
        unsupported_occupancy_conformal_coefficient_indices=tuple(
            index
            for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
            if index not in supported
        ),
        active_row_count=sum(row.active_row_count for row in selected),
        weighted_column_scale_by_occupancy_conformal_coefficient=tuple(
            float(value) for value in scales
        ),  # type: ignore[arg-type]
        raw_weighted_design_rank=raw_rank,
        standardized_weighted_design_rank=standardized_rank,
        raw_normal_condition_number=raw_condition,
        standardized_normal_condition_number=standardized_condition,
        pre_projection_corner_operator_norms=pre,
        post_projection_corner_operator_norms=post,
        trust_projection_scale=projection_scale,
        linearized_rmse_before=before,
        linearized_rmse_after=after,
        trust_projection_applied=projection_applied,
        fit_coordinate_system=OCCUPANCY_FIT_COORDINATE_RESIDUALIZED,
        occupancy_projection_on_base_by_base_and_occupancy_coordinate=tuple(
            float(value) for value in projection.reshape(-1)
        ),  # type: ignore[arg-type]
        residualization_base_weighted_design_rank=base_projection_rank,
        pre_residualization_weighted_occupancy_column_scales=tuple(
            float(value) for value in occupancy_scales
        ),  # type: ignore[arg-type]
        occupancy_residual_energy_fraction_by_coordinate=tuple(
            float(value) for value in residual_energy_fraction
        ),  # type: ignore[arg-type]
        maximum_absolute_weighted_base_residual_correlation=(
            maximum_correlation
        ),
    )


def fit_gemma_iterative_residualized_occupancy_route_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    occupancy_kind: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id=held_family,
        occupancy_kind=occupancy_kind,
    )
    return GemmaCausalTop2OccupancyConformalRouteH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_residualized_occupancy_route_full_provider(
    *,
    records: Sequence[object],
    occupancy_kind: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_residualized_occupancy_route_fold_provider(
        records=records,
        held_family=_FULL_FIT,
        occupancy_kind=occupancy_kind,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )
