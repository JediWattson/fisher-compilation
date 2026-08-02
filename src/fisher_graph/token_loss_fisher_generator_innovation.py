"""Nested family-held-out screen for fixed-basis generator innovation.

This module compares four preregistered linearized controls:

``parent``
    No compensation.

``legacy_shared``
    The first two coordinates of the original six-coordinate route.

``static_generator``
    The two shared coordinates in a frozen two-generator Fisher basis.

``conditional_generator``
    The same two shared coordinates plus one causal innovation coefficient per
    generator.

The legacy and generator score matrices are represented by separate
:class:`~fisher_graph.token_loss_fisher.TokenLossFisherPromptRecord` panels.
They must have exactly matching prompt identities, family identities,
supervised-token counts, compensation-target hashes, and target second
moments.  Thus the arms can use different directional-score coordinates while
being scored against the same authenticated target.

Every outer held-family prediction uses normalization and coefficients learned
only from the remaining families.  The conditional ridge is chosen inside
that training partition by another whole-family-held-out loop.  The ``inf``
candidate is exactly the static-generator arm, and the one-standard-error rule
selects the strongest eligible shrinkage.  Family moments are averaged as
family -> prompt -> token, never as a pooled bag of tokens.

The frozen 6-by-2 basis maps four fitted generator coefficients back to the
existing six-coordinate route.  A single radial scale enforces the route's
operator bound over all 16 corners of
``(balance, occupancy, innovation_real, innovation_imag) in {-1, +1}^4``.

This remains a linearized development screen.  It does not compile a runtime
provider and does not authorize opening finite-displacement evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

import torch
from torch import Tensor

from .gemma3_l3_l4_iterative_occupancy_route import (
    OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
)
from .token_loss_fisher import (
    TokenLossFisherPromptRecord,
    _canonical_records,
    _coefficient_cosine,
    _family_moments,
    _mean_moments,
    _median,
    _rank_and_condition,
    _relative_improvement,
    _residual_rmse,
)


__all__ = [
    "GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS",
    "GENERATOR_INNOVATION_GATE_CONFIG",
    "GENERATOR_INNOVATION_SCHEMA",
    "GENERATOR_INNOVATION_SHARED_RIDGE",
    "build_generator_innovation_nested_lofo_report",
    "generator_innovation_corner_operator_norms",
    "project_generator_innovation_coefficients",
    "replay_generator_innovation_nested_lofo_report",
    "validate_generator_innovation_nested_lofo_report",
]


GENERATOR_INNOVATION_SCHEMA = (
    "fisher_graph.token_loss_fisher_generator_innovation.v1"
)
GENERATOR_INNOVATION_SHARED_RIDGE = 1.0e-6
GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS = ("0.1", "1", "10", "inf")
GENERATOR_INNOVATION_GATE_CONFIG = {
    "required_outer_standardized_rank": 4,
    "maximum_median_outer_standardized_condition": 100.0,
    "minimum_mean_pairwise_standardized_coefficient_cosine": 0.90,
    "minimum_conditional_residual_design_energy_fraction": 0.05,
    "minimum_family_macro_relative_rmse_improvement_vs_parent": 0.02,
    "minimum_parent_family_win_fraction": 0.75,
    "maximum_worst_family_relative_rmse_regression_vs_parent": 0.02,
    "minimum_family_macro_relative_rmse_improvement_vs_static_generator": (
        0.005
    ),
    "minimum_material_family_relative_rmse_improvement_vs_static_generator": (
        0.001
    ),
    "minimum_material_static_family_win_fraction": 0.625,
    "minimum_materially_nonzero_conditional_fold_fraction": 0.625,
    "minimum_standardized_conditional_coefficient_fraction_for_active": 0.01,
    "minimum_family_macro_relative_rmse_improvement_vs_legacy_shared": 0.0,
    "minimum_fixed_basis_fisher_trace_coverage": 0.50,
}

_LEGACY_WIDTH = 6
_LEGACY_SHARED_WIDTH = 2
_GENERATOR_WIDTH = 4
_GENERATOR_SHARED_WIDTH = 2
_SUPPORT_EPSILON = 1.0e-12
_NUMBER_RELATIVE_TOLERANCE = 1.0e-12
_NUMBER_ABSOLUTE_TOLERANCE = 1.0e-15
_BASIS_ORTHONORMAL_TOLERANCE = 1.0e-8
_REPORT_DOMAIN = (
    b"fisher-graph:token-loss-fisher-generator-innovation-report:v1\0"
)
_FOLD_DOMAIN = (
    b"fisher-graph:token-loss-fisher-generator-innovation-fold:v1\0"
)
_BASIS_DOMAIN = (
    b"fisher-graph:token-loss-fisher-generator-innovation-basis:v1\0"
)
_INNER_CANDIDATE_FIELDS = {
    "conditional_ridge_label",
    "inner_held_rmse_ratio_by_family",
    "mean_inner_held_rmse_ratio",
    "standard_error_of_mean_inner_held_rmse_ratio",
    "within_best_one_standard_error",
    "selected",
}
_GENERATOR_FIT_FIELDS = {
    "coefficients",
    "standardized_coefficients",
    "column_scales",
    "raw_normal_rank",
    "standardized_normal_rank",
    "standardized_positive_spectrum_condition_number",
    "effective_degrees_of_freedom_before_projection",
    "pre_projection_corner_operator_norms",
    "post_projection_corner_operator_norms",
    "trust_projection_scale",
    "trust_projection_applied",
    "standardized_conditional_coefficient_l2_norm",
    "standardized_conditional_coefficient_l2_fraction",
    "conditional_active",
}
_LEGACY_FIT_FIELDS = {
    "coefficients",
    "standardized_coefficients",
    "column_scales",
    "raw_normal_rank",
    "standardized_normal_rank",
    "standardized_positive_spectrum_condition_number",
    "effective_degrees_of_freedom_before_projection",
    "pre_projection_corner_operator_norms",
    "post_projection_corner_operator_norms",
    "trust_projection_scale",
    "trust_projection_applied",
}
_FOLD_FIELDS = {
    "held_family_id",
    "train_family_ids",
    "train_example_ids",
    "held_example_ids",
    "train_generator_prompt_record_sha256s",
    "held_generator_prompt_record_sha256s",
    "train_legacy_prompt_record_sha256s",
    "held_legacy_prompt_record_sha256s",
    "selected_conditional_ridge_label",
    "inner_one_standard_error_threshold",
    "inner_ridge_candidates",
    "conditional_fit",
    "static_generator_fit",
    "legacy_shared_fit",
    "conditional_residual_design_energy_fraction",
    "train_parent_rmse",
    "train_conditional_rmse",
    "held_parent_rmse",
    "held_conditional_rmse",
    "held_static_generator_rmse",
    "held_legacy_shared_rmse",
    "held_relative_rmse_improvement_vs_parent",
    "held_relative_rmse_improvement_vs_static_generator",
    "held_relative_rmse_improvement_vs_legacy_shared",
    "fold_sha256",
}
_RECIPE = {
    "fit": "nested_family_lofo_fixed_basis_generator_innovation_v1",
    "arms": (
        "parent",
        "legacy_shared",
        "static_generator",
        "conditional_generator",
    ),
    "legacy_shared_coordinate_count": _LEGACY_SHARED_WIDTH,
    "generator_shared_coordinate_count": _GENERATOR_SHARED_WIDTH,
    "generator_conditional_coordinate_count": (
        _GENERATOR_WIDTH - _GENERATOR_SHARED_WIDTH
    ),
    "shared_ridge": GENERATOR_INNOVATION_SHARED_RIDGE,
    "conditional_ridge_labels": (
        GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS
    ),
    "selection_rule": (
        "largest_conditional_ridge_within_one_standard_error_"
        "of_lowest_inner_family_mean_rmse_ratio"
    ),
    "static_generator_label": "inf",
    "family_prompt_token_weighting": (
        "equal_family_then_equal_prompt_then_equal_token"
    ),
    "normalization_scope": "training_families_only_per_fit",
    "conditional_residual_design_energy": (
        "training_fold_standardized_schur_trace_fraction"
    ),
    "trust_projection": (
        "one_global_radial_scale_over_16_balance_occupancy_"
        "innovation_corners"
    ),
    "tokens_used_as_independent_split_units": False,
    "held_family_used_for_ridge_selection": False,
    "fixed_basis_refit_inside_outer_or_inner_folds": False,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _ridge_value(label: str) -> float:
    if label not in GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS:
        raise ValueError("generator conditional ridge label is invalid")
    return math.inf if label == "inf" else float(label)


def _same_number(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_NUMBER_RELATIVE_TOLERANCE,
        abs_tol=_NUMBER_ABSOLUTE_TOLERANCE,
    )


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    result = _identifier(value, label=label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _canonical_identifiers(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(
        _identifier(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if (
        (not allow_empty and not result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _basis_tensor(
    basis: Sequence[Sequence[float]],
) -> Tensor:
    if (
        isinstance(basis, (str, bytes))
        or not isinstance(basis, Sequence)
        or len(basis) != _LEGACY_WIDTH
    ):
        raise ValueError("fixed generator basis must have shape (6, 2)")
    rows: list[tuple[float, float]] = []
    for row_index, row in enumerate(basis):
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != 2
        ):
            raise ValueError("fixed generator basis must have shape (6, 2)")
        selected: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "fixed generator basis entries must be numeric; "
                    f"row={row_index} column={column_index}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("fixed generator basis entries must be finite")
            selected.append(number)
        rows.append((selected[0], selected[1]))
    result = torch.tensor(rows, dtype=torch.float64)
    gram = result.T @ result
    identity = torch.eye(2, dtype=torch.float64)
    if float((gram - identity).abs().max()) > _BASIS_ORTHONORMAL_TOLERANCE:
        raise ValueError("fixed generator basis columns must be orthonormal")
    return result


def _basis_payload(basis: Tensor) -> dict[str, object]:
    rows = tuple(
        tuple(float(value) for value in row)
        for row in basis
    )
    return {
        "shape": (_LEGACY_WIDTH, _GENERATOR_SHARED_WIDTH),
        "rows": rows,
        "column_gram": tuple(
            tuple(float(value) for value in row)
            for row in basis.T @ basis
        ),
        "basis_sha256": _sha256(_BASIS_DOMAIN, rows),
    }


def _aligned_records(
    legacy_records: Sequence[object],
    generator_records: Sequence[object],
) -> tuple[
    tuple[TokenLossFisherPromptRecord, ...],
    tuple[TokenLossFisherPromptRecord, ...],
]:
    legacy = _canonical_records(legacy_records)
    generator = _canonical_records(generator_records)
    if len(legacy[0].coordinate_names) != _LEGACY_WIDTH:
        raise ValueError("legacy score records must have six coordinates")
    if len(generator[0].coordinate_names) != _GENERATOR_WIDTH:
        raise ValueError("generator score records must have four coordinates")
    legacy_by_example = {row.example_id: row for row in legacy}
    generator_by_example = {row.example_id: row for row in generator}
    if set(legacy_by_example) != set(generator_by_example):
        raise ValueError("legacy and generator prompt identities differ")
    for example_id in sorted(legacy_by_example):
        left = legacy_by_example[example_id]
        right = generator_by_example[example_id]
        if (
            left.family_id != right.family_id
            or left.supervised_tokens != right.supervised_tokens
            or left.compensation_target_sha256
            != right.compensation_target_sha256
            or not _same_number(
                left.target_second_moment,
                right.target_second_moment,
            )
        ):
            raise ValueError(
                "legacy and generator target bindings differ for "
                f"{example_id}"
            )
    return legacy, generator


def _route_corner_norms(
    coefficients: Tensor,
    basis: Tensor,
) -> tuple[float, ...]:
    if coefficients.shape != (_GENERATOR_WIDTH,):
        raise ValueError("generator coefficients must contain four values")
    shared = coefficients[:2]
    innovation = coefficients[2:]
    norms: list[float] = []
    for balance in (-1.0, 1.0):
        for occupancy in (-1.0, 1.0):
            for innovation_real in (-1.0, 1.0):
                for innovation_imag in (-1.0, 1.0):
                    amplitudes = shared + torch.tensor(
                        (innovation_real, innovation_imag),
                        dtype=torch.float64,
                    ) * innovation
                    route = basis @ amplitudes
                    output = torch.stack(
                        (
                            route[0]
                            + balance * route[2]
                            + occupancy * route[4],
                            route[1]
                            + balance * route[3]
                            + occupancy * route[5],
                        )
                    )
                    norms.append(float(torch.linalg.vector_norm(output)))
    if len(norms) != 16:
        raise RuntimeError("generator trust-corner enumeration drifted")
    return tuple(norms)


def _project_generator_coefficients(
    coefficients: Tensor,
    basis: Tensor,
) -> tuple[Tensor, tuple[float, ...], tuple[float, ...], float, bool]:
    pre = _route_corner_norms(coefficients, basis)
    maximum = max(pre)
    bound = float(OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND)
    scale = 1.0 if maximum <= bound or maximum == 0.0 else bound / maximum
    projected = (coefficients * scale).contiguous()
    post = _route_corner_norms(projected, basis)
    if max(post) > bound * (1.0 + 1.0e-12):
        raise RuntimeError("generator 16-corner trust projection failed")
    return projected, pre, post, scale, scale < 1.0


def generator_innovation_corner_operator_norms(
    coefficients: Sequence[float],
    *,
    fixed_basis: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Return the frozen route norm at every one of the 16 trust corners."""

    if (
        isinstance(coefficients, (str, bytes))
        or not isinstance(coefficients, Sequence)
        or len(coefficients) != _GENERATOR_WIDTH
    ):
        raise ValueError("generator coefficients must contain four values")
    values = torch.tensor(tuple(coefficients), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("generator coefficients must be finite")
    return _route_corner_norms(values, _basis_tensor(fixed_basis))


def project_generator_innovation_coefficients(
    coefficients: Sequence[float],
    *,
    fixed_basis: Sequence[Sequence[float]],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    bool,
]:
    """Apply the frozen one-scale 16-corner operator-norm projection."""

    if (
        isinstance(coefficients, (str, bytes))
        or not isinstance(coefficients, Sequence)
        or len(coefficients) != _GENERATOR_WIDTH
    ):
        raise ValueError("generator coefficients must contain four values")
    values = torch.tensor(tuple(coefficients), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("generator coefficients must be finite")
    projected, pre, post, scale, applied = _project_generator_coefficients(
        values,
        _basis_tensor(fixed_basis),
    )
    return (
        tuple(float(value) for value in projected),
        pre,
        post,
        scale,
        applied,
    )


def _project_legacy_shared_coefficients(
    coefficients: Tensor,
) -> tuple[Tensor, tuple[float, ...], tuple[float, ...], float, bool]:
    if coefficients.shape != (_LEGACY_SHARED_WIDTH,):
        raise ValueError("legacy shared coefficients must contain two values")
    norm = float(torch.linalg.vector_norm(coefficients))
    pre = tuple(norm for _ in range(16))
    bound = float(OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND)
    scale = 1.0 if norm <= bound or norm == 0.0 else bound / norm
    projected = (coefficients * scale).contiguous()
    post_norm = float(torch.linalg.vector_norm(projected))
    post = tuple(post_norm for _ in range(16))
    return projected, pre, post, scale, scale < 1.0


def _full_standardized_geometry(
    fisher: Tensor,
) -> tuple[Tensor, Tensor, int, int, float]:
    scales = torch.sqrt(torch.clamp(torch.diag(fisher), min=0.0))
    safe_scales = torch.where(
        scales > _SUPPORT_EPSILON,
        scales,
        torch.ones_like(scales),
    )
    standardized = fisher / torch.outer(safe_scales, safe_scales)
    standardized = ((standardized + standardized.T) * 0.5).contiguous()
    unsupported = scales <= _SUPPORT_EPSILON
    standardized[unsupported, :] = 0.0
    standardized[:, unsupported] = 0.0
    raw_rank, _raw_condition, _raw_vectors, _raw_tolerance = (
        _rank_and_condition(fisher)
    )
    (
        standardized_rank,
        standardized_condition,
        _standardized_vectors,
        _standardized_tolerance,
    ) = _rank_and_condition(standardized)
    return (
        scales,
        standardized,
        raw_rank,
        standardized_rank,
        standardized_condition,
    )


def _fit_generator(
    moments: object,
    *,
    basis: Tensor,
    conditional_ridge_label: str,
) -> dict[str, object]:
    fisher = getattr(moments, "fisher", None)
    cross = getattr(moments, "cross", None)
    if (
        not isinstance(fisher, Tensor)
        or fisher.shape != (_GENERATOR_WIDTH, _GENERATOR_WIDTH)
        or not isinstance(cross, Tensor)
        or cross.shape != (_GENERATOR_WIDTH,)
    ):
        raise ValueError("generator moments must have four coordinates")
    scales, standardized, raw_rank, standardized_rank, condition = (
        _full_standardized_geometry(fisher)
    )
    conditional_ridge = _ridge_value(conditional_ridge_label)
    support = tuple(
        index
        for index, scale in enumerate(scales)
        if float(scale) > _SUPPORT_EPSILON
    )
    active = tuple(
        index
        for index in support
        if index < _GENERATOR_SHARED_WIDTH
        or not math.isinf(conditional_ridge)
    )
    coefficients = torch.zeros(_GENERATOR_WIDTH, dtype=torch.float64)
    effective_df = 0.0
    if active:
        selected = torch.tensor(active, dtype=torch.int64)
        chosen_scales = scales.index_select(0, selected)
        chosen_standardized = standardized.index_select(
            0, selected
        ).index_select(1, selected)
        standardized_cross = cross.index_select(0, selected) / chosen_scales
        penalties = torch.tensor(
            tuple(
                GENERATOR_INNOVATION_SHARED_RIDGE
                if index < _GENERATOR_SHARED_WIDTH
                else conditional_ridge
                for index in active
            ),
            dtype=torch.float64,
        )
        regularized = (
            chosen_standardized + torch.diag(penalties)
        ).contiguous()
        inverse = torch.linalg.solve(
            regularized,
            torch.eye(len(active), dtype=torch.float64),
        )
        standardized_coefficients = inverse @ standardized_cross
        coefficients[selected] = standardized_coefficients / chosen_scales
        effective_df = float(torch.trace(chosen_standardized @ inverse))
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("generator innovation ridge fit became nonfinite")
    projected, pre, post, scale, applied = (
        _project_generator_coefficients(coefficients, basis)
    )
    standardized_projected = projected * scales
    total_norm = float(torch.linalg.vector_norm(standardized_projected))
    conditional_norm = float(
        torch.linalg.vector_norm(
            standardized_projected[_GENERATOR_SHARED_WIDTH:]
        )
    )
    conditional_fraction = (
        0.0 if total_norm == 0.0 else conditional_norm / total_norm
    )
    return {
        "coefficients": tuple(float(value) for value in projected),
        "standardized_coefficients": tuple(
            float(value) for value in standardized_projected
        ),
        "column_scales": tuple(float(value) for value in scales),
        "raw_normal_rank": raw_rank,
        "standardized_normal_rank": standardized_rank,
        "standardized_positive_spectrum_condition_number": condition,
        "effective_degrees_of_freedom_before_projection": effective_df,
        "pre_projection_corner_operator_norms": pre,
        "post_projection_corner_operator_norms": post,
        "trust_projection_scale": scale,
        "trust_projection_applied": applied,
        "standardized_conditional_coefficient_l2_norm": conditional_norm,
        "standardized_conditional_coefficient_l2_fraction": (
            conditional_fraction
        ),
        "conditional_active": (
            not math.isinf(conditional_ridge)
            and conditional_fraction
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_standardized_conditional_coefficient_"
                    "fraction_for_active"
                )
            ]
        ),
    }


def _fit_legacy_shared(moments: object) -> dict[str, object]:
    fisher = getattr(moments, "fisher", None)
    cross = getattr(moments, "cross", None)
    if (
        not isinstance(fisher, Tensor)
        or fisher.shape != (_LEGACY_SHARED_WIDTH, _LEGACY_SHARED_WIDTH)
        or not isinstance(cross, Tensor)
        or cross.shape != (_LEGACY_SHARED_WIDTH,)
    ):
        raise ValueError("legacy shared moments must have two coordinates")
    scales, standardized, raw_rank, standardized_rank, condition = (
        _full_standardized_geometry(fisher)
    )
    support = tuple(
        index
        for index, scale in enumerate(scales)
        if float(scale) > _SUPPORT_EPSILON
    )
    coefficients = torch.zeros(_LEGACY_SHARED_WIDTH, dtype=torch.float64)
    effective_df = 0.0
    if support:
        selected = torch.tensor(support, dtype=torch.int64)
        chosen = standardized.index_select(0, selected).index_select(
            1, selected
        )
        regularized = chosen + torch.eye(
            len(support), dtype=torch.float64
        ) * GENERATOR_INNOVATION_SHARED_RIDGE
        inverse = torch.linalg.solve(
            regularized,
            torch.eye(len(support), dtype=torch.float64),
        )
        standardized_cross = cross.index_select(0, selected) / (
            scales.index_select(0, selected)
        )
        standardized_coefficients = inverse @ standardized_cross
        coefficients[selected] = standardized_coefficients / (
            scales.index_select(0, selected)
        )
        effective_df = float(torch.trace(chosen @ inverse))
    projected, pre, post, scale, applied = (
        _project_legacy_shared_coefficients(coefficients)
    )
    return {
        "coefficients": tuple(float(value) for value in projected),
        "standardized_coefficients": tuple(
            float(value) for value in projected * scales
        ),
        "column_scales": tuple(float(value) for value in scales),
        "raw_normal_rank": raw_rank,
        "standardized_normal_rank": standardized_rank,
        "standardized_positive_spectrum_condition_number": condition,
        "effective_degrees_of_freedom_before_projection": effective_df,
        "pre_projection_corner_operator_norms": pre,
        "post_projection_corner_operator_norms": post,
        "trust_projection_scale": scale,
        "trust_projection_applied": applied,
    }


def _conditional_residual_design_energy_fraction(fisher: Tensor) -> float:
    _scales, standardized, _raw_rank, _rank, _condition = (
        _full_standardized_geometry(fisher)
    )
    base = standardized[:2, :2]
    cross = standardized[:2, 2:]
    added = standardized[2:, 2:]
    projection = torch.linalg.pinv(
        (base + base.T) * 0.5,
        rtol=1.0e-10,
        atol=1.0e-12,
        hermitian=True,
    )
    residual = added - cross.T @ projection @ cross
    residual = ((residual + residual.T) * 0.5).contiguous()
    denominator = float(torch.trace(added))
    if denominator <= _SUPPORT_EPSILON:
        return 0.0
    numerator = float(torch.trace(residual))
    tolerance = 1.0e-10 * max(denominator, 1.0)
    if numerator < -tolerance:
        raise RuntimeError("conditional residual Fisher energy is not PSD")
    return min(max(numerator / denominator, 0.0), 1.0)


def _mean_and_standard_error(
    values: Sequence[float],
) -> tuple[float, float]:
    if not values:
        raise ValueError("generator ridge selection needs family scores")
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    squared = math.fsum((value - mean) ** 2 for value in values)
    sample_variance = squared / (len(values) - 1)
    return mean, math.sqrt(sample_variance / len(values))


def _inner_selection(
    generator_family_moments: Mapping[str, object],
    train_family_ids: Sequence[str],
    *,
    basis: Tensor,
) -> tuple[str, tuple[dict[str, object], ...], float]:
    families = tuple(sorted(train_family_ids))
    if len(families) < 3:
        raise ValueError(
            "nested generator selection requires at least three families"
        )
    candidates: list[dict[str, object]] = []
    for label in GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS:
        ratios: list[float] = []
        for held_family in families:
            inner_train = _mean_moments(
                tuple(
                    generator_family_moments[family]
                    for family in families
                    if family != held_family
                )
            )
            fit = _fit_generator(
                inner_train,
                basis=basis,
                conditional_ridge_label=label,
            )
            coefficients = torch.tensor(
                fit["coefficients"], dtype=torch.float64
            )
            held = generator_family_moments[held_family]
            before = _residual_rmse(
                held,
                torch.zeros(_GENERATOR_WIDTH, dtype=torch.float64),
            )
            after = _residual_rmse(held, coefficients)
            ratios.append(0.0 if before == 0.0 else after / before)
        mean, standard_error = _mean_and_standard_error(ratios)
        candidates.append(
            {
                "conditional_ridge_label": label,
                "inner_held_rmse_ratio_by_family": {
                    family: ratio
                    for family, ratio in zip(families, ratios, strict=True)
                },
                "mean_inner_held_rmse_ratio": mean,
                "standard_error_of_mean_inner_held_rmse_ratio": (
                    standard_error
                ),
            }
        )
    best = min(
        candidates,
        key=lambda row: (
            float(row["mean_inner_held_rmse_ratio"]),
            -_ridge_value(str(row["conditional_ridge_label"])),
        ),
    )
    threshold = (
        float(best["mean_inner_held_rmse_ratio"])
        + float(best["standard_error_of_mean_inner_held_rmse_ratio"])
    )
    selected = max(
        (
            row
            for row in candidates
            if float(row["mean_inner_held_rmse_ratio"])
            <= threshold + _NUMBER_ABSOLUTE_TOLERANCE
        ),
        key=lambda row: _ridge_value(str(row["conditional_ridge_label"])),
    )
    selected_label = str(selected["conditional_ridge_label"])
    finalized = tuple(
        {
            **row,
            "within_best_one_standard_error": (
                float(row["mean_inner_held_rmse_ratio"])
                <= threshold + _NUMBER_ABSOLUTE_TOLERANCE
            ),
            "selected": str(row["conditional_ridge_label"])
            == selected_label,
        }
        for row in candidates
    )
    return selected_label, finalized, threshold


def _fold(
    legacy_records: Sequence[TokenLossFisherPromptRecord],
    generator_records: Sequence[TokenLossFisherPromptRecord],
    *,
    legacy_family_moments: Mapping[str, object],
    generator_family_moments: Mapping[str, object],
    basis: Tensor,
    held_family_id: str,
) -> dict[str, object]:
    families = tuple(sorted(generator_family_moments))
    train_families = tuple(
        family for family in families if family != held_family_id
    )
    selected_label, inner_candidates, threshold = _inner_selection(
        generator_family_moments,
        train_families,
        basis=basis,
    )
    train_generator = _mean_moments(
        tuple(generator_family_moments[family] for family in train_families)
    )
    held_generator = generator_family_moments[held_family_id]
    train_legacy = _mean_moments(
        tuple(legacy_family_moments[family] for family in train_families)
    )
    held_legacy = legacy_family_moments[held_family_id]
    conditional = _fit_generator(
        train_generator,
        basis=basis,
        conditional_ridge_label=selected_label,
    )
    static = _fit_generator(
        train_generator,
        basis=basis,
        conditional_ridge_label="inf",
    )
    legacy = _fit_legacy_shared(train_legacy)
    conditional_coefficients = torch.tensor(
        conditional["coefficients"], dtype=torch.float64
    )
    static_coefficients = torch.tensor(
        static["coefficients"], dtype=torch.float64
    )
    legacy_coefficients = torch.tensor(
        legacy["coefficients"], dtype=torch.float64
    )
    zero_generator = torch.zeros(_GENERATOR_WIDTH, dtype=torch.float64)
    zero_legacy = torch.zeros(_LEGACY_SHARED_WIDTH, dtype=torch.float64)
    train_parent = _residual_rmse(train_generator, zero_generator)
    held_parent = _residual_rmse(held_generator, zero_generator)
    train_conditional = _residual_rmse(
        train_generator, conditional_coefficients
    )
    held_conditional = _residual_rmse(
        held_generator, conditional_coefficients
    )
    held_static = _residual_rmse(held_generator, static_coefficients)
    legacy_parent = _residual_rmse(held_legacy, zero_legacy)
    if not _same_number(held_parent, legacy_parent):
        raise RuntimeError("aligned parent target RMSE differs between arms")
    held_legacy_after = _residual_rmse(held_legacy, legacy_coefficients)
    train_records = tuple(
        row for row in generator_records if row.family_id != held_family_id
    )
    held_records = tuple(
        row for row in generator_records if row.family_id == held_family_id
    )
    legacy_by_example = {row.example_id: row for row in legacy_records}
    payload = {
        "held_family_id": held_family_id,
        "train_family_ids": train_families,
        "train_example_ids": tuple(
            sorted(row.example_id for row in train_records)
        ),
        "held_example_ids": tuple(
            sorted(row.example_id for row in held_records)
        ),
        "train_generator_prompt_record_sha256s": tuple(
            sorted(row.prompt_record_sha256 for row in train_records)
        ),
        "held_generator_prompt_record_sha256s": tuple(
            sorted(row.prompt_record_sha256 for row in held_records)
        ),
        "train_legacy_prompt_record_sha256s": tuple(
            sorted(
                legacy_by_example[row.example_id].prompt_record_sha256
                for row in train_records
            )
        ),
        "held_legacy_prompt_record_sha256s": tuple(
            sorted(
                legacy_by_example[row.example_id].prompt_record_sha256
                for row in held_records
            )
        ),
        "selected_conditional_ridge_label": selected_label,
        "inner_one_standard_error_threshold": threshold,
        "inner_ridge_candidates": inner_candidates,
        "conditional_fit": conditional,
        "static_generator_fit": static,
        "legacy_shared_fit": legacy,
        "conditional_residual_design_energy_fraction": (
            _conditional_residual_design_energy_fraction(
                train_generator.fisher
            )
        ),
        "train_parent_rmse": train_parent,
        "train_conditional_rmse": train_conditional,
        "held_parent_rmse": held_parent,
        "held_conditional_rmse": held_conditional,
        "held_static_generator_rmse": held_static,
        "held_legacy_shared_rmse": held_legacy_after,
        "held_relative_rmse_improvement_vs_parent": _relative_improvement(
            held_parent, held_conditional
        ),
        "held_relative_rmse_improvement_vs_static_generator": (
            _relative_improvement(held_static, held_conditional)
        ),
        "held_relative_rmse_improvement_vs_legacy_shared": (
            _relative_improvement(held_legacy_after, held_conditional)
        ),
    }
    return {**payload, "fold_sha256": _sha256(_FOLD_DOMAIN, payload)}


def _fixed_basis_trace_coverage(
    legacy_family_moments: Mapping[str, object],
    basis: Tensor,
) -> float:
    aggregate = _mean_moments(
        tuple(legacy_family_moments[family] for family in legacy_family_moments)
    )
    fisher = getattr(aggregate, "fisher")
    denominator = float(torch.trace(fisher))
    if denominator <= _SUPPORT_EPSILON:
        return 0.0
    numerator = float(torch.trace(basis.T @ fisher @ basis))
    return min(max(numerator / denominator, 0.0), 1.0)


def _aggregates(
    folds: Sequence[Mapping[str, object]],
    *,
    family_count: int,
    basis_trace_coverage: float,
) -> tuple[dict[str, object], tuple[tuple[str, bool], ...], bool]:
    parent = math.fsum(float(row["held_parent_rmse"]) for row in folds) / len(
        folds
    )
    conditional = math.fsum(
        float(row["held_conditional_rmse"]) for row in folds
    ) / len(folds)
    static = math.fsum(
        float(row["held_static_generator_rmse"]) for row in folds
    ) / len(folds)
    legacy = math.fsum(
        float(row["held_legacy_shared_rmse"]) for row in folds
    ) / len(folds)
    parent_improvements = tuple(
        float(row["held_relative_rmse_improvement_vs_parent"])
        for row in folds
    )
    static_improvements = tuple(
        float(row["held_relative_rmse_improvement_vs_static_generator"])
        for row in folds
    )
    legacy_improvements = tuple(
        float(row["held_relative_rmse_improvement_vs_legacy_shared"])
        for row in folds
    )
    standardized_coefficients = tuple(
        tuple(
            float(value)
            for value in row["conditional_fit"]["standardized_coefficients"]
        )
        for row in folds
    )
    cosines = tuple(
        _coefficient_cosine(
            standardized_coefficients[left],
            standardized_coefficients[right],
        )
        for left in range(len(folds))
        for right in range(left + 1, len(folds))
    )
    mean_cosine = 1.0 if not cosines else math.fsum(cosines) / len(cosines)
    conditions = tuple(
        float(
            row["conditional_fit"][
                "standardized_positive_spectrum_condition_number"
            ]
        )
        for row in folds
    )
    design_energies = tuple(
        float(row["conditional_residual_design_energy_fraction"])
        for row in folds
    )
    parent_win_count = sum(value > 0.0 for value in parent_improvements)
    material_static_win_count = sum(
        value
        >= GENERATOR_INNOVATION_GATE_CONFIG[
            (
                "minimum_material_family_relative_rmse_improvement_"
                "vs_static_generator"
            )
        ]
        for value in static_improvements
    )
    active_count = sum(
        bool(row["conditional_fit"]["conditional_active"]) for row in folds
    )
    required_parent_wins = math.ceil(
        family_count
        * GENERATOR_INNOVATION_GATE_CONFIG[
            "minimum_parent_family_win_fraction"
        ]
    )
    required_static_wins = math.ceil(
        family_count
        * GENERATOR_INNOVATION_GATE_CONFIG[
            "minimum_material_static_family_win_fraction"
        ]
    )
    required_active = math.ceil(
        family_count
        * GENERATOR_INNOVATION_GATE_CONFIG[
            "minimum_materially_nonzero_conditional_fold_fraction"
        ]
    )
    parent_macro = _relative_improvement(parent, conditional)
    static_macro = _relative_improvement(static, conditional)
    legacy_macro = _relative_improvement(legacy, conditional)
    gates = {
        "all_outer_standardized_ranks_equal_required": all(
            int(row["conditional_fit"]["standardized_normal_rank"])
            == GENERATOR_INNOVATION_GATE_CONFIG[
                "required_outer_standardized_rank"
            ]
            for row in folds
        ),
        "median_outer_standardized_condition_at_most_maximum": (
            _median(conditions)
            <= GENERATOR_INNOVATION_GATE_CONFIG[
                "maximum_median_outer_standardized_condition"
            ]
        ),
        "mean_pairwise_standardized_coefficient_cosine_at_least_minimum": (
            mean_cosine
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_mean_pairwise_standardized_"
                    "coefficient_cosine"
                )
            ]
        ),
        "minimum_conditional_residual_design_energy_at_least_minimum": (
            min(design_energies)
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                "minimum_conditional_residual_design_energy_fraction"
            ]
        ),
        "family_macro_improvement_vs_parent_at_least_minimum": (
            parent_macro
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_family_macro_relative_rmse_"
                    "improvement_vs_parent"
                )
            ]
        ),
        "parent_family_win_count_at_least_minimum": (
            parent_win_count >= required_parent_wins
        ),
        "worst_family_regression_vs_parent_at_most_maximum": (
            min(parent_improvements)
            >= -GENERATOR_INNOVATION_GATE_CONFIG[
                "maximum_worst_family_relative_rmse_regression_vs_parent"
            ]
        ),
        "family_macro_improvement_vs_static_generator_at_least_minimum": (
            static_macro
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_family_macro_relative_rmse_"
                    "improvement_vs_static_generator"
                )
            ]
        ),
        "material_static_family_win_count_at_least_minimum": (
            material_static_win_count >= required_static_wins
        ),
        "materially_nonzero_conditional_fold_count_at_least_minimum": (
            active_count >= required_active
        ),
        "family_macro_improvement_vs_legacy_shared_at_least_minimum": (
            legacy_macro
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_family_macro_relative_rmse_"
                    "improvement_vs_legacy_shared"
                )
            ]
        ),
        "fixed_basis_fisher_trace_coverage_at_least_minimum": (
            basis_trace_coverage
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                "minimum_fixed_basis_fisher_trace_coverage"
            ]
        ),
    }
    metrics = {
        "family_macro_parent_rmse": parent,
        "family_macro_conditional_rmse": conditional,
        "family_macro_static_generator_rmse": static,
        "family_macro_legacy_shared_rmse": legacy,
        "family_macro_relative_rmse_improvement_vs_parent": parent_macro,
        "family_macro_relative_rmse_improvement_vs_static_generator": (
            static_macro
        ),
        "family_macro_relative_rmse_improvement_vs_legacy_shared": (
            legacy_macro
        ),
        "parent_family_win_count": parent_win_count,
        "required_parent_family_win_count": required_parent_wins,
        "material_static_family_win_count": material_static_win_count,
        "required_material_static_family_win_count": required_static_wins,
        "materially_nonzero_conditional_fold_count": active_count,
        "required_materially_nonzero_conditional_fold_count": required_active,
        "worst_family_relative_rmse_improvement_vs_parent": min(
            parent_improvements
        ),
        "worst_family_relative_rmse_improvement_vs_static_generator": min(
            static_improvements
        ),
        "worst_family_relative_rmse_improvement_vs_legacy_shared": min(
            legacy_improvements
        ),
        "median_outer_standardized_condition": _median(conditions),
        "mean_pairwise_standardized_coefficient_cosine": mean_cosine,
        "minimum_conditional_residual_design_energy_fraction": min(
            design_energies
        ),
        "fixed_basis_fisher_trace_coverage": basis_trace_coverage,
        "selected_conditional_ridge_label_by_held_family": {
            str(row["held_family_id"]): str(
                row["selected_conditional_ridge_label"]
            )
            for row in folds
        },
    }
    canonical_gates = tuple(sorted(gates.items()))
    return metrics, canonical_gates, all(gates.values())


def build_generator_innovation_nested_lofo_report(
    legacy_records: Sequence[object],
    generator_records: Sequence[object],
    *,
    fixed_basis: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Build the frozen four-arm generator-innovation development screen."""

    legacy, generator = _aligned_records(legacy_records, generator_records)
    families = tuple(sorted({row.family_id for row in generator}))
    if len(families) < 4:
        raise ValueError(
            "nested generator innovation LOFO requires at least four families"
        )
    basis = _basis_tensor(fixed_basis)
    legacy_full_family_moments = _family_moments(
        legacy, tuple(range(_LEGACY_WIDTH))
    )
    legacy_shared_family_moments = _family_moments(
        legacy, tuple(range(_LEGACY_SHARED_WIDTH))
    )
    generator_family_moments = _family_moments(
        generator, tuple(range(_GENERATOR_WIDTH))
    )
    if (
        tuple(legacy_full_family_moments)
        != tuple(generator_family_moments)
        or tuple(legacy_shared_family_moments)
        != tuple(generator_family_moments)
    ):
        raise ValueError("legacy and generator family geometry differs")
    folds = tuple(
        _fold(
            legacy,
            generator,
            legacy_family_moments=legacy_shared_family_moments,
            generator_family_moments=generator_family_moments,
            basis=basis,
            held_family_id=family,
        )
        for family in families
    )
    basis_coverage = _fixed_basis_trace_coverage(
        legacy_full_family_moments, basis
    )
    metrics, gates, passed = _aggregates(
        folds,
        family_count=len(families),
        basis_trace_coverage=basis_coverage,
    )
    generator_by_example = {row.example_id: row for row in generator}
    payload = {
        "schema": GENERATOR_INNOVATION_SCHEMA,
        "legacy_coordinate_names": legacy[0].coordinate_names,
        "generator_coordinate_names": generator[0].coordinate_names,
        "fixed_basis": _basis_payload(basis),
        "family_ids": families,
        "example_ids": tuple(sorted(generator_by_example)),
        "family_id_by_example_id": {
            example_id: generator_by_example[example_id].family_id
            for example_id in sorted(generator_by_example)
        },
        "target_sha256_by_example_id": {
            example_id: generator_by_example[
                example_id
            ].compensation_target_sha256
            for example_id in sorted(generator_by_example)
        },
        "legacy_prompt_record_sha256_by_example_id": {
            row.example_id: row.prompt_record_sha256
            for row in sorted(legacy, key=lambda item: item.example_id)
        },
        "generator_prompt_record_sha256_by_example_id": {
            row.example_id: row.prompt_record_sha256
            for row in sorted(generator, key=lambda item: item.example_id)
        },
        "recipe": dict(_RECIPE),
        "gate_config": dict(GENERATOR_INNOVATION_GATE_CONFIG),
        "folds": folds,
        "metrics": metrics,
        "gate_results": gates,
        "passed": passed,
    }
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def _validate_fit_receipt(
    value: object,
    *,
    label: str,
    basis: Tensor,
    conditional_ridge_label: str | None,
) -> None:
    generator = conditional_ridge_label is not None
    expected_fields = (
        _GENERATOR_FIT_FIELDS if generator else _LEGACY_FIT_FIELDS
    )
    width = _GENERATOR_WIDTH if generator else _LEGACY_SHARED_WIDTH
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{label} fit fields differ")
    coefficients = _float_tuple(
        value["coefficients"],
        count=width,
        label=f"{label} coefficients",
    )
    standardized = _float_tuple(
        value["standardized_coefficients"],
        count=width,
        label=f"{label} standardized coefficients",
    )
    scales = _float_tuple(
        value["column_scales"],
        count=width,
        label=f"{label} column scales",
    )
    if any(scale < 0.0 for scale in scales):
        raise ValueError(f"{label} column scales must be nonnegative")
    expected_standardized = tuple(
        coefficient * scale
        for coefficient, scale in zip(coefficients, scales, strict=True)
    )
    if any(
        not _same_number(actual, expected)
        for actual, expected in zip(
            standardized, expected_standardized, strict=True
        )
    ):
        raise ValueError(f"{label} standardized coefficients differ")
    for field in ("raw_normal_rank", "standardized_normal_rank"):
        rank = value[field]
        if type(rank) is not int or not 0 <= rank <= width:
            raise ValueError(f"{label} {field} is invalid")
    condition = _finite(
        value["standardized_positive_spectrum_condition_number"],
        label=f"{label} standardized condition",
    )
    effective_df = _finite(
        value["effective_degrees_of_freedom_before_projection"],
        label=f"{label} effective degrees of freedom",
    )
    if condition < 0.0 or not 0.0 <= effective_df <= width + 1.0e-10:
        raise ValueError(f"{label} fit geometry is invalid")
    pre = _float_tuple(
        value["pre_projection_corner_operator_norms"],
        count=16,
        label=f"{label} pre-projection corners",
    )
    post = _float_tuple(
        value["post_projection_corner_operator_norms"],
        count=16,
        label=f"{label} post-projection corners",
    )
    if any(corner < 0.0 for corner in (*pre, *post)):
        raise ValueError(f"{label} corner norms must be nonnegative")
    scale = _finite(
        value["trust_projection_scale"],
        label=f"{label} trust projection scale",
    )
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"{label} trust projection scale is invalid")
    if type(value["trust_projection_applied"]) is not bool:
        raise TypeError(f"{label} trust projection flag must be boolean")
    if value["trust_projection_applied"] is not (scale < 1.0):
        raise ValueError(f"{label} trust projection flag differs")
    if any(
        not _same_number(after, before * scale)
        for before, after in zip(pre, post, strict=True)
    ):
        raise ValueError(f"{label} trust projection corner scaling differs")
    if max(post) > float(OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND) * (
        1.0 + 1.0e-12
    ):
        raise ValueError(f"{label} exceeds the frozen trust bound")
    projected = torch.tensor(coefficients, dtype=torch.float64)
    unprojected = projected / scale
    if generator:
        expected_pre = _route_corner_norms(unprojected, basis)
        expected_post = _route_corner_norms(projected, basis)
    else:
        expected_pre_norm = float(torch.linalg.vector_norm(unprojected))
        expected_post_norm = float(torch.linalg.vector_norm(projected))
        expected_pre = tuple(expected_pre_norm for _ in range(16))
        expected_post = tuple(expected_post_norm for _ in range(16))
    if any(
        not _same_number(actual, expected)
        for actual, expected in zip(pre, expected_pre, strict=True)
    ) or any(
        not _same_number(actual, expected)
        for actual, expected in zip(post, expected_post, strict=True)
    ):
        raise ValueError(f"{label} corner receipts differ")
    if generator:
        conditional_norm = _finite(
            value["standardized_conditional_coefficient_l2_norm"],
            label=f"{label} conditional norm",
        )
        conditional_fraction = _finite(
            value["standardized_conditional_coefficient_l2_fraction"],
            label=f"{label} conditional norm fraction",
        )
        expected_conditional_norm = math.hypot(
            standardized[2], standardized[3]
        )
        total_norm = math.sqrt(math.fsum(item * item for item in standardized))
        expected_fraction = (
            0.0
            if total_norm == 0.0
            else expected_conditional_norm / total_norm
        )
        if (
            conditional_norm < 0.0
            or not 0.0 <= conditional_fraction <= 1.0
            or not _same_number(conditional_norm, expected_conditional_norm)
            or not _same_number(conditional_fraction, expected_fraction)
        ):
            raise ValueError(f"{label} conditional norm receipt differs")
        if type(value["conditional_active"]) is not bool:
            raise TypeError(f"{label} conditional-active flag must be boolean")
        expected_active = (
            conditional_ridge_label != "inf"
            and conditional_fraction
            >= GENERATOR_INNOVATION_GATE_CONFIG[
                (
                    "minimum_standardized_conditional_coefficient_"
                    "fraction_for_active"
                )
            ]
        )
        if value["conditional_active"] is not expected_active:
            raise ValueError(f"{label} conditional-active flag differs")
        if conditional_ridge_label == "inf" and (
            coefficients[2:] != (0.0, 0.0)
            or standardized[2:] != (0.0, 0.0)
        ):
            raise ValueError(f"{label} static arm has conditional coefficients")


def _validate_inner_selection(
    fold: Mapping[str, object],
    *,
    train_families: tuple[str, ...],
) -> None:
    selected_label = str(fold["selected_conditional_ridge_label"])
    _ridge_value(selected_label)
    candidates_value = fold["inner_ridge_candidates"]
    if not isinstance(candidates_value, (tuple, list)) or len(
        candidates_value
    ) != len(GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS):
        raise ValueError("generator inner ridge candidates differ")
    statistics: list[tuple[str, float, float]] = []
    for expected_label, candidate in zip(
        GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS,
        candidates_value,
        strict=True,
    ):
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != _INNER_CANDIDATE_FIELDS
            or candidate["conditional_ridge_label"] != expected_label
        ):
            raise ValueError("generator inner ridge candidate fields differ")
        ratios_value = candidate["inner_held_rmse_ratio_by_family"]
        if not isinstance(ratios_value, Mapping) or set(
            ratios_value
        ) != set(train_families):
            raise ValueError("generator inner held-family ratios differ")
        ratios = tuple(
            _finite(
                ratios_value[family],
                label=f"{expected_label} inner ratio for {family}",
            )
            for family in train_families
        )
        if any(value < 0.0 for value in ratios):
            raise ValueError("generator inner RMSE ratios must be nonnegative")
        expected_mean, expected_standard_error = _mean_and_standard_error(
            ratios
        )
        mean = _finite(
            candidate["mean_inner_held_rmse_ratio"],
            label="generator inner mean RMSE ratio",
        )
        standard_error = _finite(
            candidate["standard_error_of_mean_inner_held_rmse_ratio"],
            label="generator inner RMSE standard error",
        )
        if (
            standard_error < 0.0
            or not _same_number(mean, expected_mean)
            or not _same_number(standard_error, expected_standard_error)
            or type(candidate["within_best_one_standard_error"]) is not bool
            or type(candidate["selected"]) is not bool
        ):
            raise ValueError("generator inner candidate statistics differ")
        statistics.append((expected_label, mean, standard_error))
    _best_label, best_mean, best_standard_error = min(
        statistics,
        key=lambda row: (row[1], -_ridge_value(row[0])),
    )
    expected_threshold = best_mean + best_standard_error
    threshold = _finite(
        fold["inner_one_standard_error_threshold"],
        label="generator inner one-standard-error threshold",
    )
    eligible = tuple(
        label
        for label, mean, _standard_error in statistics
        if mean <= expected_threshold + _NUMBER_ABSOLUTE_TOLERANCE
    )
    expected_selected = max(eligible, key=_ridge_value)
    if (
        not _same_number(threshold, expected_threshold)
        or selected_label != expected_selected
    ):
        raise ValueError("generator inner one-standard-error selection differs")
    for candidate, (label, mean, _standard_error) in zip(
        candidates_value, statistics, strict=True
    ):
        if candidate["within_best_one_standard_error"] is not (
            mean <= expected_threshold + _NUMBER_ABSOLUTE_TOLERANCE
        ) or candidate["selected"] is not (label == expected_selected):
            raise ValueError("generator inner selection flags differ")


def validate_generator_innovation_nested_lofo_report(
    report: object,
) -> None:
    """Validate canonical partitions, fit receipts, gates, and hashes.

    Source-derived normal equations are additionally checked by
    :func:`replay_generator_innovation_nested_lofo_report`.
    """

    if not isinstance(report, Mapping):
        raise TypeError("generator innovation report must be a mapping")
    expected_fields = {
        "schema",
        "legacy_coordinate_names",
        "generator_coordinate_names",
        "fixed_basis",
        "family_ids",
        "example_ids",
        "family_id_by_example_id",
        "target_sha256_by_example_id",
        "legacy_prompt_record_sha256_by_example_id",
        "generator_prompt_record_sha256_by_example_id",
        "recipe",
        "gate_config",
        "folds",
        "metrics",
        "gate_results",
        "passed",
        "report_sha256",
    }
    if set(report) != expected_fields:
        raise ValueError("generator innovation report fields differ")
    if report["schema"] != GENERATOR_INNOVATION_SCHEMA:
        raise ValueError("generator innovation report schema differs")
    if _canonical_bytes(report["recipe"]) != _canonical_bytes(_RECIPE):
        raise ValueError("generator innovation recipe differs")
    if _canonical_bytes(report["gate_config"]) != _canonical_bytes(
        GENERATOR_INNOVATION_GATE_CONFIG
    ):
        raise ValueError("generator innovation gate configuration differs")
    basis_value = report["fixed_basis"]
    if not isinstance(basis_value, Mapping):
        raise TypeError("generator innovation fixed basis must be a mapping")
    if set(basis_value) != {
        "shape",
        "rows",
        "column_gram",
        "basis_sha256",
    }:
        raise ValueError("generator innovation fixed-basis fields differ")
    basis = _basis_tensor(basis_value["rows"])  # type: ignore[arg-type]
    if _canonical_bytes(basis_value) != _canonical_bytes(
        _basis_payload(basis)
    ):
        raise ValueError("generator innovation fixed-basis receipt differs")
    legacy_names = _canonical_identifiers(
        tuple(sorted(report["legacy_coordinate_names"]))
        if isinstance(report["legacy_coordinate_names"], (tuple, list))
        else report["legacy_coordinate_names"],
        label="legacy coordinate names",
    )
    generator_names = _canonical_identifiers(
        tuple(sorted(report["generator_coordinate_names"]))
        if isinstance(report["generator_coordinate_names"], (tuple, list))
        else report["generator_coordinate_names"],
        label="generator coordinate names",
    )
    if (
        len(legacy_names) != _LEGACY_WIDTH
        or len(generator_names) != _GENERATOR_WIDTH
        or not isinstance(report["legacy_coordinate_names"], (tuple, list))
        or len(set(report["legacy_coordinate_names"])) != _LEGACY_WIDTH
        or not isinstance(report["generator_coordinate_names"], (tuple, list))
        or len(set(report["generator_coordinate_names"])) != _GENERATOR_WIDTH
    ):
        raise ValueError("generator innovation coordinate systems are invalid")
    families = _canonical_identifiers(
        report["family_ids"], label="generator innovation family IDs"
    )
    if len(families) < 4:
        raise ValueError("generator innovation family IDs are invalid")
    examples = _canonical_identifiers(
        report["example_ids"], label="generator innovation example IDs"
    )
    maps: dict[str, Mapping[str, object]] = {}
    for field in (
        "family_id_by_example_id",
        "target_sha256_by_example_id",
        "legacy_prompt_record_sha256_by_example_id",
        "generator_prompt_record_sha256_by_example_id",
    ):
        value = report[field]
        if not isinstance(value, Mapping) or set(value) != set(examples):
            raise ValueError(f"generator innovation {field} differs")
        maps[field] = value
    family_by_example = {
        example: _identifier(
            maps["family_id_by_example_id"][example],
            label=f"family for {example}",
        )
        for example in examples
    }
    if (
        set(family_by_example.values()) != set(families)
        or any(family not in families for family in family_by_example.values())
    ):
        raise ValueError("generator innovation example-family map differs")
    for field in (
        "target_sha256_by_example_id",
        "legacy_prompt_record_sha256_by_example_id",
        "generator_prompt_record_sha256_by_example_id",
    ):
        for example in examples:
            _require_sha256(
                maps[field][example],
                label=f"{field} for {example}",
            )
    folds_value = report["folds"]
    if (
        not isinstance(folds_value, (tuple, list))
        or len(folds_value) != len(families)
    ):
        raise ValueError("generator innovation fold geometry differs")
    for expected_family, fold in zip(
        families, folds_value, strict=True
    ):
        if not isinstance(fold, Mapping):
            raise TypeError("generator innovation folds must be mappings")
        if set(fold) != _FOLD_FIELDS:
            raise ValueError("generator innovation fold fields differ")
        payload = dict(fold)
        receipt = payload.pop("fold_sha256", None)
        if fold.get("held_family_id") != expected_family:
            raise ValueError("generator innovation held-family order differs")
        if receipt != _sha256(_FOLD_DOMAIN, payload):
            raise ValueError("generator innovation fold hash mismatch")
        train_families = _canonical_identifiers(
            fold["train_family_ids"],
            label=f"{expected_family} train family IDs",
        )
        if train_families != tuple(
            family for family in families if family != expected_family
        ):
            raise ValueError("generator innovation train-family partition differs")
        train_examples = _canonical_identifiers(
            fold["train_example_ids"],
            label=f"{expected_family} train example IDs",
        )
        held_examples = _canonical_identifiers(
            fold["held_example_ids"],
            label=f"{expected_family} held example IDs",
        )
        expected_held_examples = tuple(
            example
            for example in examples
            if family_by_example[example] == expected_family
        )
        expected_train_examples = tuple(
            example
            for example in examples
            if family_by_example[example] != expected_family
        )
        if (
            held_examples != expected_held_examples
            or train_examples != expected_train_examples
        ):
            raise ValueError("generator innovation fold prompt partition differs")
        for arm in ("generator", "legacy"):
            for split, split_examples in (
                ("train", train_examples),
                ("held", held_examples),
            ):
                field = f"{split}_{arm}_prompt_record_sha256s"
                receipts = _canonical_identifiers(
                    fold[field],
                    label=f"{expected_family} {field}",
                )
                expected_receipts = tuple(
                    sorted(
                        maps[
                            f"{arm}_prompt_record_sha256_by_example_id"
                        ][example]
                        for example in split_examples
                    )
                )
                if receipts != expected_receipts:
                    raise ValueError(
                        "generator innovation fold prompt receipts differ"
                    )
        _validate_inner_selection(fold, train_families=train_families)
        selected_label = str(fold["selected_conditional_ridge_label"])
        _validate_fit_receipt(
            fold["conditional_fit"],
            label=f"{expected_family} conditional",
            basis=basis,
            conditional_ridge_label=selected_label,
        )
        _validate_fit_receipt(
            fold["static_generator_fit"],
            label=f"{expected_family} static generator",
            basis=basis,
            conditional_ridge_label="inf",
        )
        _validate_fit_receipt(
            fold["legacy_shared_fit"],
            label=f"{expected_family} legacy shared",
            basis=basis,
            conditional_ridge_label=None,
        )
        if selected_label == "inf" and _canonical_bytes(
            fold["conditional_fit"]
        ) != _canonical_bytes(fold["static_generator_fit"]):
            raise ValueError("generator inf candidate differs from static arm")
        design_energy = _finite(
            fold["conditional_residual_design_energy_fraction"],
            label=f"{expected_family} conditional residual design energy",
        )
        if not 0.0 <= design_energy <= 1.0:
            raise ValueError(
                "generator conditional residual design energy is invalid"
            )
        rmse = {
            name: _finite(fold[name], label=f"{expected_family} {name}")
            for name in (
                "train_parent_rmse",
                "train_conditional_rmse",
                "held_parent_rmse",
                "held_conditional_rmse",
                "held_static_generator_rmse",
                "held_legacy_shared_rmse",
            )
        }
        if any(value < 0.0 for value in rmse.values()):
            raise ValueError("generator fold RMSE values must be nonnegative")
        expected_improvements = {
            "held_relative_rmse_improvement_vs_parent": _relative_improvement(
                rmse["held_parent_rmse"], rmse["held_conditional_rmse"]
            ),
            (
                "held_relative_rmse_improvement_vs_static_generator"
            ): _relative_improvement(
                rmse["held_static_generator_rmse"],
                rmse["held_conditional_rmse"],
            ),
            (
                "held_relative_rmse_improvement_vs_legacy_shared"
            ): _relative_improvement(
                rmse["held_legacy_shared_rmse"],
                rmse["held_conditional_rmse"],
            ),
        }
        for name, expected in expected_improvements.items():
            actual = _finite(
                fold[name], label=f"{expected_family} {name}"
            )
            if not _same_number(actual, expected):
                raise ValueError("generator fold improvement receipt differs")
    metrics = report["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("generator innovation metrics must be a mapping")
    basis_coverage = _finite(
        metrics.get("fixed_basis_fisher_trace_coverage"),
        label="fixed-basis Fisher trace coverage",
    )
    if not 0.0 <= basis_coverage <= 1.0:
        raise ValueError("fixed-basis Fisher trace coverage is invalid")
    expected_metrics, expected_gates, expected_passed = _aggregates(
        folds_value,
        family_count=len(families),
        basis_trace_coverage=basis_coverage,
    )
    if _canonical_bytes(metrics) != _canonical_bytes(expected_metrics):
        raise ValueError("generator innovation aggregate metrics differ")
    if _canonical_bytes(report["gate_results"]) != _canonical_bytes(
        expected_gates
    ):
        raise ValueError("generator innovation gate results differ")
    if type(report["passed"]) is not bool or report["passed"] is not (
        expected_passed
    ):
        raise ValueError("generator innovation pass flag differs")
    payload = dict(report)
    receipt = payload.pop("report_sha256", None)
    if receipt != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("generator innovation report hash mismatch")


def replay_generator_innovation_nested_lofo_report(
    legacy_records: Sequence[object],
    generator_records: Sequence[object],
    *,
    fixed_basis: Sequence[Sequence[float]],
    expected_report: Mapping[str, object],
) -> dict[str, object]:
    """Recompute every split, fit, projection, metric, gate, and receipt."""

    validate_generator_innovation_nested_lofo_report(expected_report)
    rebuilt = build_generator_innovation_nested_lofo_report(
        legacy_records,
        generator_records,
        fixed_basis=fixed_basis,
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(expected_report):
        raise ValueError("generator innovation report replay differs")
    return rebuilt
