"""Replayable development screen for fold-local occupancy residualization.

This rung is deliberately restricted to the reusable 16-by-8 calibration-A
fit panel.  It recomputes every direct and residualized leave-one-family-out
fit from authenticated scalar records, scores held prompts in the original
six-coordinate runtime basis, and never opens or authorizes a selection
panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

from .gemma3_l3_l4_iterative_occupancy_development import (
    _arm_scientific_gates,
    _canonical_folds,
    _canonical_records,
    _coefficient_cosine,
    _family_macro,
    _median,
)
from .gemma3_l3_l4_iterative_occupancy_residualized import (
    fit_gemma_iterative_residualized_occupancy_route_fold,
)
from .gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
    OCCUPANCY_FIT_COORDINATE_DIRECT,
    OCCUPANCY_FIT_COORDINATE_RESIDUALIZED,
    GemmaIterativeOccupancyConformalRouteFoldFit,
    _record,
    fit_gemma_iterative_occupancy_conformal_route_fold,
)
from .gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)


__all__ = [
    "OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR",
    "build_gemma_iterative_residualized_occupancy_development_report",
    "validate_gemma_iterative_residualized_occupancy_development_report",
]


OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR = 0.05
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_"
    "residualized_development"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-residualized-development:v1\0"
)
_ARMS = (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)
_KIND_BY_ARM = {
    CUMULATIVE_OCCUPANCY_ARM: CENTERED_CUMULATIVE_OCCUPANCY,
    EW_OCCUPANCY_ARM: CENTERED_EW_OCCUPANCY,
}
_JACOBIAN_FIELD_BY_ARM = {
    CUMULATIVE_OCCUPANCY_ARM: (
        "jacobian_by_cumulative_occupancy_conformal_coefficient"
    ),
    EW_OCCUPANCY_ARM: (
        "jacobian_by_ew_occupancy_conformal_coefficient"
    ),
}
_OCCUPANCY_STD_FIELD_BY_ARM = {
    CUMULATIVE_OCCUPANCY_ARM: "cumulative_occupancy_feature_std",
    EW_OCCUPANCY_ARM: "ew_occupancy_feature_std",
}
_EXPECTED_FAMILIES = 8
_EXPECTED_FOLD_PAIRS = 28
_MAX_MEDIAN_CONDITION = 100.0
_MIN_MEAN_FOLD_COSINE = 0.90
_MIN_FEATURE_STD = 0.05
_MIN_TOP2_ENERGY = 0.5
_RESIDUAL_BOOLEAN_GATES = (
    "all_fold_raw_weighted_design_ranks_exactly_6",
    "all_fold_standardized_weighted_design_ranks_exactly_6",
    "all_6_residualized_coordinates_supported_in_every_fold",
    "residual_block_rank_exactly_2_in_every_fold",
    "median_fold_standardized_normal_condition_number_at_most_100",
    "mean_pairwise_mapped_runtime_coefficient_cosine_at_least_0_90",
    "all_fold_occupancy_residual_energy_fractions_at_least_0_05",
    "all_fold_weighted_base_residual_orthogonality_checks_passed",
    "family_macro_balance_feature_std_at_least_0_05",
    "family_macro_occupancy_feature_std_at_least_0_05",
    "family_macro_top2_modal_energy_fraction_at_least_0_5",
    "both_occupancy_signs_seen",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _float6(value: object, *, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
    ):
        raise ValueError(f"{label} must contain exactly six values")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _fold_object(value: Mapping[str, object]) -> (
    GemmaIterativeOccupancyConformalRouteFoldFit
):
    expected = set(
        GemmaIterativeOccupancyConformalRouteFoldFit.__dataclass_fields__
    )
    if set(value) != expected:
        raise ValueError("occupancy fold fields differ")
    payload = dict(value)
    receipt = payload.pop("fold_receipt_sha256")
    fit = GemmaIterativeOccupancyConformalRouteFoldFit(
        **payload,  # type: ignore[arg-type]
    )
    if fit.fold_receipt_sha256 != receipt:
        raise ValueError("occupancy fold receipt hash differs")
    return fit


def _parent_family_macro_absolute(
    records: Sequence[Mapping[str, object]],
) -> float:
    by_family: dict[str, list[float]] = {}
    for row in records:
        value = float(row["parent_signed_delta_nll_per_token"])
        if not math.isfinite(value):
            raise ValueError("parent delta NLL must be finite")
        by_family.setdefault(str(row["family_id"]), []).append(abs(value))
    return math.fsum(
        math.fsum(values) / len(values)
        for _, values in sorted(by_family.items())
    ) / len(by_family)


def _replay_direct_folds(
    *,
    records: Sequence[Mapping[str, object]],
    folds_by_arm: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    record_objects = tuple(_record(row) for row in records)
    for arm_id in _ARMS:
        for fold_row in folds_by_arm[arm_id]:
            held_family = str(fold_row["held_family_id"])
            training = tuple(
                row
                for row in record_objects
                if row.family_id != held_family
            )
            replayed = fit_gemma_iterative_occupancy_conformal_route_fold(
                training,
                held_family_id=held_family,
                occupancy_kind=_KIND_BY_ARM[arm_id],
            )
            if _canonical_bytes(replayed.to_dict()) != _canonical_bytes(
                fold_row
            ):
                raise ValueError(
                    f"{arm_id} direct fold does not replay from training rows"
                )


def _residualized_folds(
    records: Sequence[Mapping[str, object]],
) -> dict[str, tuple[GemmaIterativeOccupancyConformalRouteFoldFit, ...]]:
    record_objects = tuple(_record(row) for row in records)
    families = tuple(sorted({row.family_id for row in record_objects}))
    result: dict[
        str,
        tuple[GemmaIterativeOccupancyConformalRouteFoldFit, ...],
    ] = {}
    for arm_id in _ARMS:
        folds = []
        for held_family in families:
            training = tuple(
                row
                for row in record_objects
                if row.family_id != held_family
            )
            folds.append(
                fit_gemma_iterative_residualized_occupancy_route_fold(
                    training,
                    held_family_id=held_family,
                    occupancy_kind=_KIND_BY_ARM[arm_id],
                )
            )
        result[arm_id] = tuple(folds)
    return result


def _residualized_arm_metrics(
    *,
    arm_id: str,
    records: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
    direct_metrics: Mapping[str, object],
    parent_macro: float,
) -> dict[str, object]:
    folds_by_family = {
        str(row["held_family_id"]): row
        for row in folds
    }
    predicted_by_family: dict[str, list[float]] = {}
    for record in records:
        family_id = str(record["family_id"])
        fold = folds_by_family[family_id]
        jacobian = _float6(
            record[_JACOBIAN_FIELD_BY_ARM[arm_id]],
            label=f"{arm_id} held Jacobian",
        )
        coefficients = _float6(
            fold["coefficients_by_occupancy_conformal_coefficient"],
            label=f"{arm_id} mapped runtime coefficients",
        )
        predicted = float(record["parent_signed_delta_nll_per_token"])
        predicted += math.fsum(
            left * right
            for left, right in zip(jacobian, coefficients, strict=True)
        )
        if not math.isfinite(predicted):
            raise ValueError("residualized predicted delta NLL is nonfinite")
        predicted_by_family.setdefault(family_id, []).append(abs(predicted))
    predicted_macro = math.fsum(
        math.fsum(values) / len(values)
        for _, values in sorted(predicted_by_family.items())
    ) / len(predicted_by_family)

    raw_ranks = {
        str(row["held_family_id"]): int(row["raw_weighted_design_rank"])
        for row in folds
    }
    standardized_ranks = {
        str(row["held_family_id"]): int(
            row["standardized_weighted_design_rank"]
        )
        for row in folds
    }
    supported_counts = {
        index: sum(
            index
            not in tuple(
                row[
                    "unsupported_occupancy_conformal_coefficient_indices"
                ]
            )
            for row in folds
        )
        for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
    }
    standardized_conditions = tuple(
        float(row["standardized_normal_condition_number"])
        for row in folds
    )
    raw_conditions = tuple(
        float(row["raw_normal_condition_number"])
        for row in folds
    )
    median_standardized_condition = _median(standardized_conditions)
    median_raw_condition = _median(raw_conditions)
    coefficient_rows = tuple(
        _float6(
            row["coefficients_by_occupancy_conformal_coefficient"],
            label=f"{arm_id} mapped runtime coefficients",
        )
        for row in folds
    )
    pairwise_cosines = tuple(
        _coefficient_cosine(coefficient_rows[left], coefficient_rows[right])
        for left in range(len(coefficient_rows))
        for right in range(left + 1, len(coefficient_rows))
    )
    if len(pairwise_cosines) != _EXPECTED_FOLD_PAIRS:
        raise ValueError("residualized fold stability requires 28 pairs")
    mean_cosine = math.fsum(pairwise_cosines) / len(pairwise_cosines)
    residual_energy = tuple(
        float(value)
        for row in folds
        for value in row[
            "occupancy_residual_energy_fraction_by_coordinate"
        ]
    )
    correlations = tuple(
        float(
            row[
                "maximum_absolute_weighted_base_residual_correlation"
            ]
        )
        for row in folds
    )
    macro_balance_std = _family_macro(
        records,
        field_name="balance_feature_std",
    )
    macro_occupancy_std = _family_macro(
        records,
        field_name=_OCCUPANCY_STD_FIELD_BY_ARM[arm_id],
    )
    macro_top2_energy = _family_macro(
        records,
        field_name="top2_modal_energy_fraction",
    )
    negative_rows = sum(
        int(row["negative_balance_row_count"]) for row in records
    )
    nonnegative_rows = sum(
        int(row["nonnegative_balance_row_count"]) for row in records
    )
    direct_condition = float(
        direct_metrics[
            "median_fold_standardized_normal_condition_number"
        ]
    )
    direct_cosine = float(
        direct_metrics["mean_pairwise_fold_coefficient_cosine"]
    )
    direct_predicted = float(
        direct_metrics[
            "predicted_family_macro_mean_absolute_delta_nll_per_token"
        ]
    )
    result: dict[str, object] = {
        "parent_family_macro_mean_absolute_delta_nll_per_token": parent_macro,
        "predicted_family_macro_mean_absolute_delta_nll_per_token": (
            predicted_macro
        ),
        "predicted_fraction_of_parent_absolute_delta": (
            predicted_macro / parent_macro if parent_macro else 0.0
        ),
        "predicted_delta_vs_direct": predicted_macro - direct_predicted,
        "raw_weighted_design_rank_by_held_family": raw_ranks,
        "standardized_weighted_design_rank_by_held_family": (
            standardized_ranks
        ),
        "all_fold_raw_weighted_design_ranks_exactly_6": all(
            rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            for rank in raw_ranks.values()
        ),
        "all_fold_standardized_weighted_design_ranks_exactly_6": all(
            rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            for rank in standardized_ranks.values()
        ),
        "supported_fold_count_by_residualized_coordinate": supported_counts,
        "all_6_residualized_coordinates_supported_in_every_fold": all(
            count == _EXPECTED_FAMILIES
            for count in supported_counts.values()
        ),
        "residual_block_rank_exactly_2_in_every_fold": all(
            4
            not in tuple(
                row[
                    "unsupported_occupancy_conformal_coefficient_indices"
                ]
            )
            and 5
            not in tuple(
                row[
                    "unsupported_occupancy_conformal_coefficient_indices"
                ]
            )
            for row in folds
        ),
        "median_fold_raw_normal_condition_number": median_raw_condition,
        "median_fold_standardized_normal_condition_number": (
            median_standardized_condition
        ),
        "median_fold_standardized_normal_condition_number_at_most_100": (
            median_standardized_condition <= _MAX_MEDIAN_CONDITION
        ),
        "standardized_condition_ratio_vs_direct": (
            median_standardized_condition / direct_condition
            if direct_condition
            else 0.0
        ),
        "mean_pairwise_mapped_runtime_coefficient_cosine": mean_cosine,
        "mean_pairwise_mapped_runtime_coefficient_cosine_at_least_0_90": (
            mean_cosine >= _MIN_MEAN_FOLD_COSINE
        ),
        "mapped_runtime_coefficient_cosine_delta_vs_direct": (
            mean_cosine - direct_cosine
        ),
        "minimum_fold_occupancy_residual_energy_fraction": min(
            residual_energy
        ),
        "median_fold_occupancy_residual_energy_fraction": _median(
            residual_energy
        ),
        "all_fold_occupancy_residual_energy_fractions_at_least_0_05": all(
            value >= OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR
            for value in residual_energy
        ),
        "maximum_fold_absolute_weighted_base_residual_correlation": max(
            correlations
        ),
        "all_fold_weighted_base_residual_orthogonality_checks_passed": all(
            value <= 1.0e-10 for value in correlations
        ),
        "family_macro_balance_feature_std": macro_balance_std,
        "family_macro_balance_feature_std_at_least_0_05": (
            macro_balance_std >= _MIN_FEATURE_STD
        ),
        "family_macro_occupancy_feature_std": macro_occupancy_std,
        "family_macro_occupancy_feature_std_at_least_0_05": (
            macro_occupancy_std >= _MIN_FEATURE_STD
        ),
        "family_macro_top2_modal_energy_fraction": macro_top2_energy,
        "family_macro_top2_modal_energy_fraction_at_least_0_5": (
            macro_top2_energy >= _MIN_TOP2_ENERGY
        ),
        "negative_balance_row_count": negative_rows,
        "nonnegative_balance_row_count": nonnegative_rows,
        "both_occupancy_signs_seen": (
            negative_rows > 0 and nonnegative_rows > 0
        ),
    }
    result["passed"] = all(
        bool(result[field]) for field in _RESIDUAL_BOOLEAN_GATES
    )
    return result


def _assemble(
    *,
    fit_records: Sequence[object],
    direct_fold_receipts_by_arm: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    records = _canonical_records(fit_records)
    if set(direct_fold_receipts_by_arm) != set(_ARMS):
        raise ValueError("direct occupancy development arms differ")
    direct_folds = {
        arm_id: _canonical_folds(
            direct_fold_receipts_by_arm[arm_id],
            arm_id=arm_id,
            records=records,
        )
        for arm_id in _ARMS
    }
    if any(
        row["fit_coordinate_system"] != OCCUPANCY_FIT_COORDINATE_DIRECT
        for rows in direct_folds.values()
        for row in rows
    ):
        raise ValueError("direct baseline uses the wrong fit coordinates")
    _replay_direct_folds(records=records, folds_by_arm=direct_folds)

    residual_objects = _residualized_folds(records)
    residual_folds = {
        arm_id: tuple(fit.to_dict() for fit in rows)
        for arm_id, rows in residual_objects.items()
    }
    if any(
        row["fit_coordinate_system"]
        != OCCUPANCY_FIT_COORDINATE_RESIDUALIZED
        for rows in residual_folds.values()
        for row in rows
    ):
        raise RuntimeError("residualized fold uses direct fit coordinates")

    direct_metrics = {
        arm_id: _arm_scientific_gates(
            arm_id=arm_id,
            records=records,
            folds=direct_folds[arm_id],
        )
        for arm_id in _ARMS
    }
    parent_macro = _parent_family_macro_absolute(records)
    residual_metrics = {
        arm_id: _residualized_arm_metrics(
            arm_id=arm_id,
            records=records,
            folds=residual_folds[arm_id],
            direct_metrics=direct_metrics[arm_id],
            parent_macro=parent_macro,
        )
        for arm_id in _ARMS
    }
    eligible = tuple(
        arm_id
        for arm_id in _ARMS
        if residual_metrics[arm_id]["passed"] is True
    )
    selected = (
        min(
            eligible,
            key=lambda arm_id: (
                float(
                    residual_metrics[arm_id][
                        "predicted_family_macro_mean_absolute_delta_nll_"
                        "per_token"
                    ]
                ),
                0 if arm_id == CUMULATIVE_OCCUPANCY_ARM else 1,
            ),
        )
        if eligible
        else None
    )
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 1,
        "development_role": "calibration_a_fit_only",
        "fit_coordinate_system": (
            OCCUPANCY_FIT_COORDINATE_RESIDUALIZED
        ),
        "residual_retained_energy_floor": (
            OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR
        ),
        "fit_records": records,
        "direct_fold_receipts_by_arm": direct_folds,
        "residualized_fold_receipts_by_arm": residual_folds,
        "direct_scientific_metrics_by_arm": direct_metrics,
        "residualized_scientific_metrics_by_arm": residual_metrics,
        "selected_residualized_arm_id": selected,
        "viable_on_reusable_development": selected is not None,
        "selection_panel_authorized": False,
        "selection_panel_opened": False,
        "selection_claim_created": False,
        "fresh_confirmation_requires_new_blinded_panel": True,
        "next_action": (
            "freeze_new_blinded_selection_plan"
            if selected is not None
            else "stop_residualized_occupancy_without_fresh_run"
        ),
    }
    payload["report_sha256"] = _sha256(payload)
    return payload


def build_gemma_iterative_residualized_occupancy_development_report(
    *,
    fit_records: Sequence[object],
    direct_fold_receipts_by_arm: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    """Build a prompt-free, replayable, development-only result."""

    report = _assemble(
        fit_records=fit_records,
        direct_fold_receipts_by_arm=direct_fold_receipts_by_arm,
    )
    validate_gemma_iterative_residualized_occupancy_development_report(
        report
    )
    return report


def validate_gemma_iterative_residualized_occupancy_development_report(
    value: Mapping[str, object],
) -> None:
    """Replay every fit and reject any changed scalar or receipt."""

    row = dict(_mapping(value, label="residualized occupancy report"))
    observed_hash = row.pop("report_sha256", None)
    if observed_hash != _sha256(row):
        raise ValueError("residualized occupancy report hash differs")
    if (
        row.get("schema") != _SCHEMA
        or row.get("format_version") != 1
        or row.get("development_role") != "calibration_a_fit_only"
        or row.get("selection_panel_authorized") is not False
        or row.get("selection_panel_opened") is not False
        or row.get("selection_claim_created") is not False
        or row.get("fresh_confirmation_requires_new_blinded_panel")
        is not True
    ):
        raise ValueError("residualized occupancy development boundary differs")
    direct = _mapping(
        row.get("direct_fold_receipts_by_arm"),
        label="direct fold receipts by arm",
    )
    expected = _assemble(
        fit_records=tuple(row.get("fit_records", ())),
        direct_fold_receipts_by_arm={
            arm_id: tuple(direct[arm_id])  # type: ignore[arg-type]
            for arm_id in direct
        },
    )
    if _canonical_bytes(expected) != _canonical_bytes(value):
        raise ValueError("residualized occupancy report does not replay")
