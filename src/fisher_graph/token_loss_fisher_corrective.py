"""Nested family-held-out shrinkage for token-loss Fisher route fits.

The six-coordinate Gemma occupancy route is already token conditioned:

```
C_t = C(shared + balance_t * balance_delta
                 + occupancy_t * occupancy_delta)
```

The exact token-loss Fisher development rung showed that the four conditional
coordinates are observable, but their fitted coefficients rotate across held
families.  This module tests the smallest corrective hypothesis that can be
answered from the existing prompt sufficient statistics: retain the first two
shared coordinates and shrink the four conditional deviations toward zero.

Every outer held-family score uses a deviation ridge selected by an inner
whole-family-held-out loop.  The exact shared-only fit is a member of the
fixed grid and the conservative one-standard-error rule chooses the strongest
eligible shrinkage.  Tokens are never treated as independent split units.

This is a linearized adaptive-development screen.  It does not orient Fisher
couplings, compile a provider, or authorize graph traversal.
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
    project_occupancy_route_coefficients,
)
from .token_loss_fisher import (
    TokenLossFisherPromptRecord,
    _canonical_records,
    _coefficient_cosine,
    _coordinate_view,
    _family_moments,
    _mean_moments,
    _median,
    _rank_and_condition,
    _relative_improvement,
    _residual_rmse,
)


__all__ = [
    "TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS",
    "TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG",
    "TOKEN_LOSS_FISHER_CORRECTIVE_SCHEMA",
    "TOKEN_LOSS_FISHER_CORRECTIVE_SHARED_RIDGE",
    "build_token_loss_fisher_corrective_report",
    "replay_token_loss_fisher_corrective_report",
    "validate_token_loss_fisher_corrective_report",
]


TOKEN_LOSS_FISHER_CORRECTIVE_SCHEMA = (
    "fisher_graph.token_loss_fisher_corrective.v1"
)
TOKEN_LOSS_FISHER_CORRECTIVE_SHARED_RIDGE = 1.0e-6
TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS = (
    "0.1",
    "1",
    "10",
    "inf",
)
TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG = {
    "maximum_median_standardized_condition": 100.0,
    "minimum_mean_pairwise_coefficient_cosine": 0.90,
    "minimum_family_macro_relative_rmse_improvement": 0.02,
    "minimum_family_win_fraction": 0.75,
    "maximum_worst_family_relative_rmse_regression": 0.02,
    "minimum_conditional_family_macro_relative_rmse_improvement_vs_shared_only": (
        0.005
    ),
    "minimum_conditional_family_relative_rmse_improvement_for_win": 0.001,
    "minimum_incremental_family_win_fraction": 0.625,
}

_WIDTH = 6
_SHARED_WIDTH = 2
_SUPPORT_EPSILON = 1.0e-12
_REPORT_DOMAIN = b"fisher-graph:token-loss-fisher-corrective-report:v1\0"
_FOLD_DOMAIN = b"fisher-graph:token-loss-fisher-corrective-fold:v1\0"
_FOLD_FIELDS = {
    "held_family_id",
    "train_family_ids",
    "train_example_ids",
    "held_example_ids",
    "train_prompt_record_sha256s",
    "held_prompt_record_sha256s",
    "selected_deviation_ridge_label",
    "inner_one_standard_error_threshold",
    "inner_ridge_candidates",
    "coefficients",
    "column_scales",
    "raw_normal_rank",
    "standardized_normal_rank",
    "standardized_positive_spectrum_condition_number",
    "effective_degrees_of_freedom_before_projection",
    "pre_projection_corner_operator_norms",
    "post_projection_corner_operator_norms",
    "trust_projection_scale",
    "trust_projection_applied",
    "conditional_active",
    "standardized_conditional_coefficient_l2_norm",
    "standardized_conditional_coefficient_l2_fraction",
    "shared_only_coefficients",
    "shared_only_effective_degrees_of_freedom_before_projection",
    "train_rmse_before",
    "train_rmse_after",
    "held_rmse_before",
    "held_rmse_after",
    "held_shared_only_rmse_after",
    "held_relative_rmse_improvement",
    "held_shared_only_relative_rmse_improvement",
    "held_conditional_relative_rmse_improvement_vs_shared_only",
    "fold_sha256",
}
_INNER_CANDIDATE_FIELDS = {
    "deviation_ridge_label",
    "inner_held_rmse_ratio_by_family",
    "mean_inner_held_rmse_ratio",
    "standard_error_of_mean_inner_held_rmse_ratio",
    "within_best_one_standard_error",
    "selected",
}
_RECIPE = {
    "fit": (
        "nested_family_lofo_standardized_shared_plus_"
        "shrunk_conditional_v1"
    ),
    "shared_coordinate_count": _SHARED_WIDTH,
    "shared_ridge": TOKEN_LOSS_FISHER_CORRECTIVE_SHARED_RIDGE,
    "deviation_ridge_labels": (
        TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS
    ),
    "selection_rule": (
        "largest_deviation_ridge_within_one_standard_error_"
        "of_lowest_inner_family_mean_rmse_ratio"
    ),
    "shared_only_label": "inf",
    "family_prompt_token_weighting": (
        "equal_family_then_equal_prompt_then_equal_token"
    ),
    "trust_projection": (
        "existing_six_coordinate_four_corner_operator_bound"
    ),
    "tokens_used_as_independent_split_units": False,
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
    if label not in TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS:
        raise ValueError("corrective deviation ridge label is invalid")
    return math.inf if label == "inf" else float(label)


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


def _canonical_identifiers(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(
        _identifier(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be nonempty, sorted, and unique")
    return result


def _canonical_sha256s(value: object, *, label: str) -> tuple[str, ...]:
    result = _canonical_identifiers(value, label=label)
    for index, item in enumerate(result):
        _require_sha256(item, label=f"{label}[{index}]")
    return result


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _same_number(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _corner_operator_norms(
    coefficients: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(coefficients) != _WIDTH:
        raise ValueError("corrective corner receipt requires six coefficients")
    a0, b0, ag, bg, ao, bo = coefficients
    return tuple(
        math.hypot(
            a0 + balance * ag + occupancy * ao,
            b0 + balance * bg + occupancy * bo,
        )
        for balance, occupancy in (
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        )
    )  # type: ignore[return-value]


def _fit(
    moments: object,
    *,
    deviation_ridge_label: str,
) -> dict[str, object]:
    fisher = getattr(moments, "fisher", None)
    cross = getattr(moments, "cross", None)
    if (
        not isinstance(fisher, Tensor)
        or fisher.shape != (_WIDTH, _WIDTH)
        or not isinstance(cross, Tensor)
        or cross.shape != (_WIDTH,)
    ):
        raise ValueError("corrective moments must have six coordinates")
    scales = torch.sqrt(torch.clamp(torch.diag(fisher), min=0.0))
    support = tuple(
        index
        for index, scale in enumerate(scales)
        if float(scale) > _SUPPORT_EPSILON
    )
    coefficients = torch.zeros(_WIDTH, dtype=torch.float64)
    effective_df = 0.0
    deviation_ridge = _ridge_value(deviation_ridge_label)
    active = tuple(
        index
        for index in support
        if index < _SHARED_WIDTH or not math.isinf(deviation_ridge)
    )
    if active:
        selected = torch.tensor(active, dtype=torch.int64)
        chosen_scales = scales.index_select(0, selected)
        chosen_fisher = fisher.index_select(0, selected).index_select(
            1, selected
        )
        standardized = chosen_fisher / torch.outer(
            chosen_scales, chosen_scales
        )
        standardized = ((standardized + standardized.T) * 0.5).contiguous()
        standardized_cross = cross.index_select(0, selected) / chosen_scales
        penalties = torch.tensor(
            tuple(
                TOKEN_LOSS_FISHER_CORRECTIVE_SHARED_RIDGE
                if index < _SHARED_WIDTH
                else deviation_ridge
                for index in active
            ),
            dtype=torch.float64,
        )
        regularized = standardized + torch.diag(penalties)
        regularized = (
            (regularized + regularized.T) * 0.5
        ).contiguous()
        inverse = torch.linalg.solve(
            regularized,
            torch.eye(len(active), dtype=torch.float64),
        )
        standardized_coefficients = inverse @ standardized_cross
        coefficients[selected] = standardized_coefficients / chosen_scales
        effective_df = float(torch.trace(standardized @ inverse))
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("corrective ridge fit became nonfinite")

    projected, pre_corners, post_corners, scale, applied = (
        project_occupancy_route_coefficients(
            coefficients, supported=support
        )
    )
    full_scales = torch.where(
        scales > _SUPPORT_EPSILON,
        scales,
        torch.ones_like(scales),
    )
    standardized_full = fisher / torch.outer(full_scales, full_scales)
    standardized_full = (
        (standardized_full + standardized_full.T) * 0.5
    ).contiguous()
    standardized_full[
        scales <= _SUPPORT_EPSILON, :
    ] = 0.0
    standardized_full[
        :, scales <= _SUPPORT_EPSILON
    ] = 0.0
    raw_rank, _raw_condition, _raw_vectors, _raw_tolerance = (
        _rank_and_condition(fisher)
    )
    (
        standardized_rank,
        standardized_condition,
        _standardized_vectors,
        _standardized_tolerance,
    ) = _rank_and_condition(standardized_full)
    standardized_projected = projected * scales
    total_norm = float(torch.linalg.vector_norm(standardized_projected))
    deviation_norm = float(
        torch.linalg.vector_norm(
            standardized_projected[_SHARED_WIDTH:]
        )
    )
    return {
        "coefficients": tuple(float(value) for value in projected),
        "column_scales": tuple(float(value) for value in scales),
        "raw_normal_rank": raw_rank,
        "standardized_normal_rank": standardized_rank,
        "standardized_positive_spectrum_condition_number": (
            standardized_condition
        ),
        "effective_degrees_of_freedom_before_projection": effective_df,
        "pre_projection_corner_operator_norms": pre_corners,
        "post_projection_corner_operator_norms": post_corners,
        "trust_projection_scale": scale,
        "trust_projection_applied": applied,
        "conditional_active": (
            not math.isinf(deviation_ridge) and deviation_norm > 0.0
        ),
        "standardized_conditional_coefficient_l2_norm": deviation_norm,
        "standardized_conditional_coefficient_l2_fraction": (
            0.0 if total_norm == 0.0 else deviation_norm / total_norm
        ),
    }


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("corrective ridge selection needs held-family scores")
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    squared = math.fsum((value - mean) ** 2 for value in values)
    sample_variance = squared / (len(values) - 1)
    return mean, math.sqrt(sample_variance / len(values))


def _inner_selection(
    family_moments: Mapping[str, object],
    train_family_ids: Sequence[str],
) -> tuple[str, tuple[dict[str, object], ...], float]:
    families = tuple(sorted(train_family_ids))
    if len(families) < 3:
        raise ValueError(
            "nested corrective selection requires at least three families"
        )
    candidates: list[dict[str, object]] = []
    for label in TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS:
        ratios = []
        for held in families:
            inner_train = _mean_moments(
                tuple(
                    family_moments[family]
                    for family in families
                    if family != held
                )
            )
            fit = _fit(
                inner_train,
                deviation_ridge_label=label,
            )
            coefficients = torch.tensor(
                fit["coefficients"], dtype=torch.float64
            )
            held_moments = family_moments[held]
            before = _residual_rmse(
                held_moments,
                torch.zeros(_WIDTH, dtype=torch.float64),
            )
            after = _residual_rmse(held_moments, coefficients)
            if before == 0.0:
                if after != 0.0:
                    raise RuntimeError(
                        "corrective fit introduced error from a zero baseline"
                    )
                ratios.append(0.0)
            else:
                ratios.append(after / before)
        mean, standard_error = _mean_and_standard_error(ratios)
        candidates.append(
            {
                "deviation_ridge_label": label,
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
            -_ridge_value(str(row["deviation_ridge_label"])),
        ),
    )
    threshold = (
        float(best["mean_inner_held_rmse_ratio"])
        + float(
            best[
                "standard_error_of_mean_inner_held_rmse_ratio"
            ]
        )
    )
    eligible = tuple(
        row
        for row in candidates
        if float(row["mean_inner_held_rmse_ratio"])
        <= threshold + 1.0e-15
    )
    selected = max(
        eligible,
        key=lambda row: _ridge_value(str(row["deviation_ridge_label"])),
    )
    selected_label = str(selected["deviation_ridge_label"])
    finalized = tuple(
        {
            **row,
            "within_best_one_standard_error": (
                float(row["mean_inner_held_rmse_ratio"])
                <= threshold + 1.0e-15
            ),
            "selected": (
                str(row["deviation_ridge_label"]) == selected_label
            ),
        }
        for row in candidates
    )
    return selected_label, finalized, threshold


def _fold(
    records: Sequence[TokenLossFisherPromptRecord],
    *,
    family_moments: Mapping[str, object],
    held_family_id: str,
) -> dict[str, object]:
    families = tuple(sorted(family_moments))
    if held_family_id not in family_moments:
        raise ValueError("corrective held family is absent")
    train_families = tuple(
        family for family in families if family != held_family_id
    )
    selected_label, candidates, threshold = _inner_selection(
        family_moments,
        train_families,
    )
    train = _mean_moments(
        tuple(family_moments[family] for family in train_families)
    )
    held = family_moments[held_family_id]
    candidate = _fit(train, deviation_ridge_label=selected_label)
    shared = _fit(train, deviation_ridge_label="inf")
    candidate_coefficients = torch.tensor(
        candidate["coefficients"], dtype=torch.float64
    )
    shared_coefficients = torch.tensor(
        shared["coefficients"], dtype=torch.float64
    )
    zero = torch.zeros(_WIDTH, dtype=torch.float64)
    train_before = _residual_rmse(train, zero)
    train_after = _residual_rmse(train, candidate_coefficients)
    held_before = _residual_rmse(held, zero)
    held_after = _residual_rmse(held, candidate_coefficients)
    held_shared = _residual_rmse(held, shared_coefficients)
    train_records = tuple(
        row for row in records if row.family_id != held_family_id
    )
    held_records = tuple(
        row for row in records if row.family_id == held_family_id
    )
    payload = {
        "held_family_id": held_family_id,
        "train_family_ids": train_families,
        "train_example_ids": tuple(
            sorted(row.example_id for row in train_records)
        ),
        "held_example_ids": tuple(
            sorted(row.example_id for row in held_records)
        ),
        "train_prompt_record_sha256s": tuple(
            sorted(row.prompt_record_sha256 for row in train_records)
        ),
        "held_prompt_record_sha256s": tuple(
            sorted(row.prompt_record_sha256 for row in held_records)
        ),
        "selected_deviation_ridge_label": selected_label,
        "inner_one_standard_error_threshold": threshold,
        "inner_ridge_candidates": candidates,
        **candidate,
        "shared_only_coefficients": tuple(
            float(value) for value in shared_coefficients
        ),
        "shared_only_effective_degrees_of_freedom_before_projection": (
            shared["effective_degrees_of_freedom_before_projection"]
        ),
        "train_rmse_before": train_before,
        "train_rmse_after": train_after,
        "held_rmse_before": held_before,
        "held_rmse_after": held_after,
        "held_shared_only_rmse_after": held_shared,
        "held_relative_rmse_improvement": _relative_improvement(
            held_before, held_after
        ),
        "held_shared_only_relative_rmse_improvement": (
            _relative_improvement(held_before, held_shared)
        ),
        "held_conditional_relative_rmse_improvement_vs_shared_only": (
            _relative_improvement(held_shared, held_after)
        ),
    }
    return {
        **payload,
        "fold_sha256": _sha256(_FOLD_DOMAIN, payload),
    }


def build_token_loss_fisher_corrective_report(
    records: Sequence[object],
    *,
    coordinate_indices: Sequence[int],
) -> dict[str, object]:
    """Build the strict nested-LOFO partially pooled corrective screen."""

    selected = _canonical_records(records)
    names = selected[0].coordinate_names
    view = _coordinate_view(len(names), coordinate_indices)
    if len(view) != _WIDTH:
        raise ValueError("corrective screen requires exactly six coordinates")
    if view != tuple(sorted(view)):
        raise ValueError(
            "corrective coordinate view must retain canonical source order"
        )
    view_names = tuple(names[index] for index in view)
    family_moments = _family_moments(selected, view)
    families = tuple(sorted(family_moments))
    if len(families) < 4:
        raise ValueError(
            "nested corrective LOFO requires at least four families"
        )
    folds = tuple(
        _fold(
            selected,
            family_moments=family_moments,
            held_family_id=family,
        )
        for family in families
    )
    metrics, gate_results, passed = _aggregates_from_serialized_folds(
        folds,
        family_count=len(families),
    )
    payload = {
        "schema": TOKEN_LOSS_FISHER_CORRECTIVE_SCHEMA,
        "coordinate_indices": view,
        "coordinate_names": view_names,
        "family_ids": families,
        "prompt_record_sha256s": tuple(
            sorted(row.prompt_record_sha256 for row in selected)
        ),
        "prompt_record_sha256_by_example_id": {
            row.example_id: row.prompt_record_sha256
            for row in sorted(selected, key=lambda item: item.example_id)
        },
        "recipe": dict(_RECIPE),
        "gate_config": dict(TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG),
        "folds": folds,
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def _aggregates_from_serialized_folds(
    folds: Sequence[Mapping[str, object]],
    *,
    family_count: int,
) -> tuple[dict[str, object], tuple[tuple[str, bool], ...], bool]:
    before = math.fsum(float(row["held_rmse_before"]) for row in folds) / len(
        folds
    )
    after = math.fsum(float(row["held_rmse_after"]) for row in folds) / len(
        folds
    )
    shared_after = math.fsum(
        float(row["held_shared_only_rmse_after"]) for row in folds
    ) / len(folds)
    improvements = tuple(
        float(row["held_relative_rmse_improvement"]) for row in folds
    )
    incremental = tuple(
        float(
            row[
                "held_conditional_relative_rmse_improvement_vs_shared_only"
            ]
        )
        for row in folds
    )
    coefficient_rows = tuple(
        tuple(float(value) for value in row["coefficients"])
        for row in folds
    )
    pairwise = tuple(
        _coefficient_cosine(
            coefficient_rows[left], coefficient_rows[right]
        )
        for left in range(len(folds))
        for right in range(left + 1, len(folds))
    )
    cosine = 1.0 if not pairwise else math.fsum(pairwise) / len(pairwise)
    conditions = tuple(
        float(
            row[
                "standardized_positive_spectrum_condition_number"
            ]
        )
        for row in folds
    )
    wins = sum(value > 0.0 for value in improvements)
    incremental_wins = sum(
        value
        >= TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
            "minimum_conditional_family_relative_rmse_improvement_for_win"
        ]
        for value in incremental
    )
    minimum_wins = math.ceil(
        TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
            "minimum_family_win_fraction"
        ]
        * family_count
    )
    minimum_incremental_wins = math.ceil(
        TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
            "minimum_incremental_family_win_fraction"
        ]
        * family_count
    )
    macro_improvement = _relative_improvement(before, after)
    shared_improvement = _relative_improvement(before, shared_after)
    incremental_improvement = _relative_improvement(shared_after, after)
    gates = {
        "all_fold_raw_normal_ranks_full": all(
            int(row["raw_normal_rank"]) == _WIDTH for row in folds
        ),
        "all_fold_standardized_normal_ranks_full": all(
            int(row["standardized_normal_rank"]) == _WIDTH for row in folds
        ),
        "median_fold_standardized_condition_at_most_maximum": (
            _median(conditions)
            <= TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
                "maximum_median_standardized_condition"
            ]
        ),
        "mean_pairwise_fold_coefficient_cosine_at_least_minimum": (
            cosine
            >= TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
                "minimum_mean_pairwise_coefficient_cosine"
            ]
        ),
        "family_macro_relative_rmse_improvement_at_least_minimum": (
            macro_improvement
            >= TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
                "minimum_family_macro_relative_rmse_improvement"
            ]
        ),
        "family_win_count_at_least_minimum": wins >= minimum_wins,
        "worst_family_relative_rmse_regression_at_most_maximum": (
            min(improvements)
            >= -TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
                "maximum_worst_family_relative_rmse_regression"
            ]
        ),
        (
            "conditional_family_macro_relative_rmse_improvement_"
            "vs_shared_only_at_least_minimum"
        ): (
            incremental_improvement
            >= TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG[
                (
                    "minimum_conditional_family_macro_relative_rmse_"
                    "improvement_vs_shared_only"
                )
            ]
        ),
        "conditional_family_win_count_vs_shared_only_at_least_minimum": (
            incremental_wins >= minimum_incremental_wins
        ),
        "at_least_one_outer_fold_selected_conditional_deviations": any(
            bool(row["conditional_active"]) for row in folds
        ),
    }
    metrics = {
        "family_macro_rmse_before": before,
        "family_macro_rmse_after": after,
        "shared_only_family_macro_rmse_after": shared_after,
        "family_macro_relative_rmse_improvement": macro_improvement,
        "shared_only_family_macro_relative_rmse_improvement": (
            shared_improvement
        ),
        "conditional_family_macro_relative_rmse_improvement_vs_shared_only": (
            incremental_improvement
        ),
        "family_win_count": wins,
        "minimum_family_win_count": minimum_wins,
        "conditional_family_win_count_vs_shared_only": incremental_wins,
        "minimum_conditional_family_win_count_vs_shared_only": (
            minimum_incremental_wins
        ),
        "worst_family_relative_rmse_improvement": min(improvements),
        "median_fold_standardized_condition": _median(conditions),
        "mean_pairwise_fold_coefficient_cosine": cosine,
        "conditional_active_fold_count": sum(
            bool(row["conditional_active"]) for row in folds
        ),
        "selected_deviation_ridge_label_by_held_family": {
            str(row["held_family_id"]): str(
                row["selected_deviation_ridge_label"]
            )
            for row in folds
        },
    }
    canonical_gates = tuple(sorted(gates.items()))
    return metrics, canonical_gates, all(gates.values())


def validate_token_loss_fisher_corrective_report(
    report: object,
) -> None:
    """Validate one serialized corrective report without source records."""

    if not isinstance(report, Mapping):
        raise TypeError("corrective report must be a mapping")
    expected = {
        "schema",
        "coordinate_indices",
        "coordinate_names",
        "family_ids",
        "prompt_record_sha256s",
        "prompt_record_sha256_by_example_id",
        "recipe",
        "gate_config",
        "folds",
        "metrics",
        "gate_results",
        "passed",
        "report_sha256",
    }
    if set(report) != expected:
        raise ValueError("corrective report fields differ")
    if report["schema"] != TOKEN_LOSS_FISHER_CORRECTIVE_SCHEMA:
        raise ValueError("corrective report schema differs")
    if not isinstance(report["coordinate_indices"], (tuple, list)):
        raise TypeError("corrective coordinate indices must be a sequence")
    coordinate_indices = tuple(report["coordinate_indices"])
    if (
        len(coordinate_indices) != _WIDTH
        or any(type(index) is not int or index < 0 for index in coordinate_indices)
        or coordinate_indices != tuple(sorted(set(coordinate_indices)))
    ):
        raise ValueError("corrective coordinate view is invalid")
    if not isinstance(report["coordinate_names"], (tuple, list)):
        raise TypeError("corrective coordinate names must be a sequence")
    coordinate_names = tuple(
        _identifier(name, label=f"corrective coordinate name {index}")
        for index, name in enumerate(report["coordinate_names"])
    )
    if len(coordinate_names) != _WIDTH or len(set(coordinate_names)) != _WIDTH:
        raise ValueError("corrective coordinate names are invalid")
    families = _canonical_identifiers(
        report["family_ids"], label="corrective family ids"
    )
    if len(families) < 4:
        raise ValueError("corrective family/fold geometry differs")
    if not isinstance(report["folds"], (tuple, list)):
        raise TypeError("corrective folds must be a sequence")
    folds = tuple(report["folds"])
    if len(folds) != len(families):
        raise ValueError("corrective family/fold geometry differs")
    prompt_receipts = _canonical_sha256s(
        report["prompt_record_sha256s"],
        label="corrective prompt record receipts",
    )
    prompt_receipt_map_value = report[
        "prompt_record_sha256_by_example_id"
    ]
    if not isinstance(prompt_receipt_map_value, Mapping):
        raise TypeError(
            "corrective example-to-prompt receipt map must be a mapping"
        )
    prompt_example_ids = _canonical_identifiers(
        tuple(prompt_receipt_map_value),
        label="corrective prompt example ids",
    )
    prompt_receipt_map = {
        example_id: _require_sha256(
            prompt_receipt_map_value[example_id],
            label=f"corrective prompt receipt for {example_id}",
        )
        for example_id in prompt_example_ids
    }
    if (
        len(set(prompt_receipt_map.values())) != len(prompt_receipt_map)
        or tuple(sorted(prompt_receipt_map.values())) != prompt_receipts
    ):
        raise ValueError("corrective example-to-prompt receipt map differs")
    if _canonical_bytes(report["recipe"]) != _canonical_bytes(_RECIPE):
        raise ValueError("corrective recipe differs")
    if _canonical_bytes(report["gate_config"]) != _canonical_bytes(
        TOKEN_LOSS_FISHER_CORRECTIVE_GATE_CONFIG
    ):
        raise ValueError("corrective gate configuration differs")
    held_examples_by_family: dict[str, tuple[str, ...]] = {}
    held_receipts_by_family: dict[str, tuple[str, ...]] = {}
    for family, fold in zip(families, folds, strict=True):
        if not isinstance(fold, Mapping):
            raise TypeError("corrective folds must be mappings")
        if set(fold) != _FOLD_FIELDS:
            raise ValueError("corrective fold fields differ")
        payload = dict(fold)
        receipt = payload.pop("fold_sha256", None)
        if fold.get("held_family_id") != family:
            raise ValueError("corrective held-family order differs")
        _require_sha256(receipt, label="corrective fold")
        if receipt != _sha256(_FOLD_DOMAIN, payload):
            raise ValueError("corrective fold hash mismatch")

        train_families = _canonical_identifiers(
            fold["train_family_ids"],
            label=f"{family} corrective train family ids",
        )
        expected_train_families = tuple(
            candidate for candidate in families if candidate != family
        )
        if train_families != expected_train_families:
            raise ValueError("corrective train-family partition differs")
        train_examples = _canonical_identifiers(
            fold["train_example_ids"],
            label=f"{family} corrective train example ids",
        )
        held_examples = _canonical_identifiers(
            fold["held_example_ids"],
            label=f"{family} corrective held example ids",
        )
        train_receipts = _canonical_sha256s(
            fold["train_prompt_record_sha256s"],
            label=f"{family} corrective train prompt receipts",
        )
        held_receipts = _canonical_sha256s(
            fold["held_prompt_record_sha256s"],
            label=f"{family} corrective held prompt receipts",
        )
        if (
            set(train_examples) & set(held_examples)
            or set(train_receipts) & set(held_receipts)
            or tuple(sorted((*train_receipts, *held_receipts)))
            != prompt_receipts
            or not set((*train_examples, *held_examples)).issubset(
                prompt_receipt_map
            )
            or tuple(
                sorted(prompt_receipt_map[item] for item in train_examples)
            )
            != train_receipts
            or tuple(
                sorted(prompt_receipt_map[item] for item in held_examples)
            )
            != held_receipts
        ):
            raise ValueError("corrective fold prompt partition differs")
        held_examples_by_family[family] = held_examples
        held_receipts_by_family[family] = held_receipts

        label = str(fold.get("selected_deviation_ridge_label"))
        _ridge_value(label)
        threshold = _finite(
            fold["inner_one_standard_error_threshold"],
            label="corrective inner one-standard-error threshold",
        )
        if threshold < 0.0:
            raise ValueError("corrective inner threshold is negative")
        if not isinstance(fold["inner_ridge_candidates"], (tuple, list)):
            raise TypeError("corrective inner ridge candidates must be a sequence")
        candidates = tuple(fold["inner_ridge_candidates"])
        if len(candidates) != len(
            TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS
        ):
            raise ValueError("corrective inner ridge receipt differs")
        candidate_statistics: list[tuple[str, float, float]] = []
        for expected_label, candidate in zip(
            TOKEN_LOSS_FISHER_CORRECTIVE_DEVIATION_RIDGE_LABELS,
            candidates,
            strict=True,
        ):
            if not isinstance(candidate, Mapping):
                raise TypeError("corrective inner candidates must be mappings")
            if set(candidate) != _INNER_CANDIDATE_FIELDS:
                raise ValueError("corrective inner candidate fields differ")
            candidate_label = str(candidate["deviation_ridge_label"])
            if candidate_label != expected_label:
                raise ValueError("corrective inner ridge order differs")
            ratios_value = candidate["inner_held_rmse_ratio_by_family"]
            if not isinstance(ratios_value, Mapping) or set(
                ratios_value
            ) != set(train_families):
                raise ValueError("corrective inner held-family ratios differ")
            ratios = tuple(
                _finite(
                    ratios_value[inner_family],
                    label=(
                        f"{family} {candidate_label} inner ratio "
                        f"for {inner_family}"
                    ),
                )
                for inner_family in train_families
            )
            if any(value < 0.0 for value in ratios):
                raise ValueError("corrective inner RMSE ratio is negative")
            expected_mean, expected_standard_error = (
                _mean_and_standard_error(ratios)
            )
            mean = _finite(
                candidate["mean_inner_held_rmse_ratio"],
                label="corrective inner mean RMSE ratio",
            )
            standard_error = _finite(
                candidate[
                    "standard_error_of_mean_inner_held_rmse_ratio"
                ],
                label="corrective inner RMSE ratio standard error",
            )
            if (
                standard_error < 0.0
                or not _same_number(mean, expected_mean)
                or not _same_number(
                    standard_error, expected_standard_error
                )
            ):
                raise ValueError("corrective inner candidate statistics differ")
            if type(candidate["within_best_one_standard_error"]) is not bool:
                raise TypeError(
                    "corrective one-standard-error flag must be boolean"
                )
            if type(candidate["selected"]) is not bool:
                raise TypeError("corrective selected flag must be boolean")
            candidate_statistics.append(
                (candidate_label, mean, standard_error)
            )
        best_label, best_mean, best_standard_error = min(
            candidate_statistics,
            key=lambda row: (row[1], -_ridge_value(row[0])),
        )
        del best_label
        expected_threshold = best_mean + best_standard_error
        eligible_labels = tuple(
            candidate_label
            for candidate_label, mean, _standard_error in candidate_statistics
            if mean <= expected_threshold + 1.0e-15
        )
        expected_label = max(eligible_labels, key=_ridge_value)
        if (
            not _same_number(threshold, expected_threshold)
            or label != expected_label
        ):
            raise ValueError("corrective one-standard-error selection differs")
        for candidate, (candidate_label, mean, _standard_error) in zip(
            candidates,
            candidate_statistics,
            strict=True,
        ):
            if candidate["within_best_one_standard_error"] is not (
                mean <= expected_threshold + 1.0e-15
            ) or candidate["selected"] is not (
                candidate_label == expected_label
            ):
                raise ValueError("corrective inner selection flags differ")

        coefficients = _float_tuple(
            fold["coefficients"],
            count=_WIDTH,
            label="corrective coefficients",
        )
        shared_coefficients = _float_tuple(
            fold["shared_only_coefficients"],
            count=_WIDTH,
            label="corrective shared-only coefficients",
        )
        column_scales = _float_tuple(
            fold["column_scales"],
            count=_WIDTH,
            label="corrective column scales",
        )
        if any(scale < 0.0 for scale in column_scales):
            raise ValueError("corrective column scale is negative")
        if any(
            coefficient != 0.0
            for coefficient, scale in zip(
                coefficients, column_scales, strict=True
            )
            if scale <= _SUPPORT_EPSILON
        ):
            raise ValueError(
                "unsupported corrective coefficients must remain zero"
            )
        if any(value != 0.0 for value in shared_coefficients[_SHARED_WIDTH:]):
            raise ValueError("corrective shared-only deviations must be zero")

        for key in ("raw_normal_rank", "standardized_normal_rank"):
            rank = fold[key]
            if type(rank) is not int or not 0 <= rank <= _WIDTH:
                raise ValueError(f"corrective {key} is invalid")
        condition = _finite(
            fold["standardized_positive_spectrum_condition_number"],
            label="corrective standardized condition",
        )
        if condition < 0.0:
            raise ValueError("corrective standardized condition is negative")
        effective_df = _finite(
            fold["effective_degrees_of_freedom_before_projection"],
            label="corrective effective degrees of freedom",
        )
        shared_effective_df = _finite(
            fold[
                "shared_only_effective_degrees_of_freedom_before_projection"
            ],
            label="corrective shared-only effective degrees of freedom",
        )
        if (
            not -1.0e-12 <= effective_df <= _WIDTH + 1.0e-12
            or not -1.0e-12
            <= shared_effective_df
            <= _SHARED_WIDTH + 1.0e-12
        ):
            raise ValueError(
                "corrective effective degrees of freedom are invalid"
            )

        pre_corners = _float_tuple(
            fold["pre_projection_corner_operator_norms"],
            count=4,
            label="corrective pre-projection corner norms",
        )
        post_corners = _float_tuple(
            fold["post_projection_corner_operator_norms"],
            count=4,
            label="corrective post-projection corner norms",
        )
        if any(value < 0.0 for value in (*pre_corners, *post_corners)):
            raise ValueError("corrective corner operator norm is negative")
        projection_scale = _finite(
            fold["trust_projection_scale"],
            label="corrective trust projection scale",
        )
        projection_applied = fold["trust_projection_applied"]
        if (
            type(projection_applied) is not bool
            or not 0.0 < projection_scale <= 1.0
        ):
            raise ValueError("corrective trust projection receipt is invalid")
        expected_post_corners = _corner_operator_norms(coefficients)
        if any(
            not _same_number(actual, expected_value)
            for actual, expected_value in zip(
                post_corners, expected_post_corners, strict=True
            )
        ):
            raise ValueError("corrective projected corner norms differ")
        maximum_pre = max(pre_corners)
        expected_projection_applied = (
            maximum_pre > OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
        )
        expected_projection_scale = (
            OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
            * (1.0 - 1.0e-12)
            / maximum_pre
            if expected_projection_applied
            else 1.0
        )
        if (
            projection_applied is not expected_projection_applied
            or not _same_number(
                projection_scale, expected_projection_scale
            )
            or any(
                not _same_number(post, pre * projection_scale)
                for pre, post in zip(
                    pre_corners, post_corners, strict=True
                )
            )
            or max(post_corners)
            > OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12
        ):
            raise ValueError("corrective trust projection receipt differs")

        standardized_coefficients = tuple(
            coefficient * scale
            for coefficient, scale in zip(
                coefficients, column_scales, strict=True
            )
        )
        expected_total_norm = math.sqrt(
            math.fsum(value * value for value in standardized_coefficients)
        )
        expected_conditional_norm = math.sqrt(
            math.fsum(
                value * value
                for value in standardized_coefficients[_SHARED_WIDTH:]
            )
        )
        conditional_norm = _finite(
            fold["standardized_conditional_coefficient_l2_norm"],
            label="corrective conditional coefficient norm",
        )
        conditional_fraction = _finite(
            fold["standardized_conditional_coefficient_l2_fraction"],
            label="corrective conditional coefficient fraction",
        )
        expected_fraction = (
            0.0
            if expected_total_norm == 0.0
            else expected_conditional_norm / expected_total_norm
        )
        conditional_active = fold["conditional_active"]
        if (
            type(conditional_active) is not bool
            or conditional_norm < 0.0
            or not 0.0 <= conditional_fraction <= 1.0 + 1.0e-12
            or not _same_number(
                conditional_norm, expected_conditional_norm
            )
            or not _same_number(conditional_fraction, expected_fraction)
            or conditional_active
            is not (label != "inf" and conditional_norm > 0.0)
        ):
            raise ValueError("corrective conditional-active receipt differs")

        rmse = {
            key: _finite(fold[key], label=f"corrective {key}")
            for key in (
                "train_rmse_before",
                "train_rmse_after",
                "held_rmse_before",
                "held_rmse_after",
                "held_shared_only_rmse_after",
            )
        }
        if any(value < 0.0 for value in rmse.values()):
            raise ValueError("corrective RMSE receipt is negative")
        expected_improvements = {
            "held_relative_rmse_improvement": _relative_improvement(
                rmse["held_rmse_before"], rmse["held_rmse_after"]
            ),
            "held_shared_only_relative_rmse_improvement": (
                _relative_improvement(
                    rmse["held_rmse_before"],
                    rmse["held_shared_only_rmse_after"],
                )
            ),
            "held_conditional_relative_rmse_improvement_vs_shared_only": (
                _relative_improvement(
                    rmse["held_shared_only_rmse_after"],
                    rmse["held_rmse_after"],
                )
            ),
        }
        if any(
            not _same_number(
                _finite(fold[key], label=f"corrective {key}"),
                expected_value,
            )
            for key, expected_value in expected_improvements.items()
        ):
            raise ValueError("corrective relative improvement receipt differs")
        if label == "inf" and (
            coefficients != shared_coefficients
            or conditional_active
            or conditional_norm != 0.0
            or conditional_fraction != 0.0
            or not _same_number(effective_df, shared_effective_df)
            or not _same_number(
                rmse["held_rmse_after"],
                rmse["held_shared_only_rmse_after"],
            )
        ):
            raise ValueError("corrective shared-only fallback differs")

    all_held_examples = tuple(
        item
        for family in families
        for item in held_examples_by_family[family]
    )
    all_held_receipts = tuple(
        item
        for family in families
        for item in held_receipts_by_family[family]
    )
    if (
        len(set(all_held_examples)) != len(all_held_examples)
        or len(set(all_held_receipts)) != len(all_held_receipts)
        or tuple(sorted(all_held_receipts)) != prompt_receipts
    ):
        raise ValueError("corrective held partitions are not disjoint")
    for family, fold in zip(families, folds, strict=True):
        expected_train_examples = tuple(
            sorted(
                item
                for other_family in families
                if other_family != family
                for item in held_examples_by_family[other_family]
            )
        )
        expected_train_receipts = tuple(
            sorted(
                item
                for other_family in families
                if other_family != family
                for item in held_receipts_by_family[other_family]
            )
        )
        if (
            tuple(fold["train_example_ids"]) != expected_train_examples
            or tuple(fold["train_prompt_record_sha256s"])
            != expected_train_receipts
        ):
            raise ValueError("corrective outer fold partition differs")

    metrics, gate_results, passed = _aggregates_from_serialized_folds(
        folds,
        family_count=len(families),
    )
    if _canonical_bytes(report["metrics"]) != _canonical_bytes(metrics):
        raise ValueError("corrective aggregate metrics differ")
    if _canonical_bytes(report["gate_results"]) != _canonical_bytes(
        gate_results
    ):
        raise ValueError("corrective gate results differ")
    if type(report["passed"]) is not bool or report["passed"] is not passed:
        raise ValueError("corrective pass decision differs from gates")
    payload = dict(report)
    receipt = payload.pop("report_sha256")
    _require_sha256(receipt, label="corrective report")
    if receipt != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("corrective report hash mismatch")


def replay_token_loss_fisher_corrective_report(
    records: Sequence[object],
    report: object,
) -> dict[str, object]:
    """Rebuild a report from prompt moments and require byte equality."""

    validate_token_loss_fisher_corrective_report(report)
    if not isinstance(report, Mapping):
        raise TypeError("corrective report must be a mapping")
    rebuilt = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=tuple(report["coordinate_indices"]),
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(report):
        raise ValueError("corrective report does not replay from prompt moments")
    return rebuilt
