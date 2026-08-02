"""Development-only arm selection for Iteration-5 occupancy routing.

The selector consumes the shared, dual-Jacobian fit records and two complete
sets of family-blocked leave-one-family-out fold fits.  It performs no model
execution and retains no tensors.  Each held prompt is replayed at the parent
linearization point with the fold that excluded its family:

``predicted_delta = parent_delta + occupancy_jacobian @ fold_coefficients``.

Only arms that pass the frozen scientific gates participate in selection.
The lower family-macro absolute predicted delta-NLL wins; an exact tie is
resolved in favour of cumulative occupancy.  This decision is frozen before
the fresh one-shot selection panel is opened.  Raw conditioning is retained
as a diagnostic, but only standardized conditioning gates eligibility: the
small occupancy column is precisely why the fit is column-standardized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
import math

from .gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
    GemmaIterativeOccupancyConformalRouteFitRecord,
    GemmaIterativeOccupancyConformalRouteFoldFit,
)
from .gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)


__all__ = [
    "OCCUPANCY_DEVELOPMENT_SELECTION_RULE",
    "build_gemma_iterative_occupancy_development_selection",
    "validate_gemma_iterative_occupancy_development_selection",
]


OCCUPANCY_DEVELOPMENT_SELECTION_RULE = (
    "minimum_family_macro_predicted_absolute_delta_nll"
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
_COEFFICIENT_ORDER = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "occupancy_contrast_real",
    "occupancy_contrast_imag",
)
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PER_FAMILY = 2
_EXPECTED_FOLD_PAIRS = 28
_MAX_MEDIAN_CONDITION = 100.0
_MIN_MEAN_FOLD_COSINE = 0.90
_MIN_FEATURE_STD = 0.05
_MIN_TOP2_ENERGY = 0.5

_ARM_METRIC_FIELDS = frozenset(
    {
        "predicted_family_macro_mean_absolute_delta_nll_per_token",
        "raw_weighted_design_rank_by_held_family",
        "standardized_weighted_design_rank_by_held_family",
        "all_fold_raw_weighted_design_ranks_exactly_6",
        "all_fold_standardized_weighted_design_ranks_exactly_6",
        "supported_fold_count_by_occupancy_conformal_coordinate",
        "all_6_occupancy_conformal_coordinates_supported_in_every_fold",
        "median_fold_raw_normal_condition_number",
        "median_fold_raw_normal_condition_number_at_most_100",
        "median_fold_standardized_normal_condition_number",
        "median_fold_standardized_normal_condition_number_at_most_100",
        "mean_pairwise_fold_coefficient_cosine",
        "mean_pairwise_fold_coefficient_cosine_at_least_0_90",
        "family_macro_balance_feature_std",
        "family_macro_balance_feature_std_at_least_0_05",
        "family_macro_occupancy_feature_std",
        "family_macro_occupancy_feature_std_at_least_0_05",
        "family_macro_top2_modal_energy_fraction",
        "family_macro_top2_modal_energy_fraction_at_least_0_5",
        "negative_balance_row_count",
        "nonnegative_balance_row_count",
        "both_occupancy_signs_seen",
        "passed",
    }
)
_BOOLEAN_GATE_FIELDS = (
    "all_fold_raw_weighted_design_ranks_exactly_6",
    "all_fold_standardized_weighted_design_ranks_exactly_6",
    "all_6_occupancy_conformal_coordinates_supported_in_every_fold",
    "median_fold_standardized_normal_condition_number_at_most_100",
    "mean_pairwise_fold_coefficient_cosine_at_least_0_90",
    "family_macro_balance_feature_std_at_least_0_05",
    "family_macro_occupancy_feature_std_at_least_0_05",
    "family_macro_top2_modal_energy_fraction_at_least_0_5",
    "both_occupancy_signs_seen",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float6(value: object, *, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
    ):
        raise ValueError(f"{label} must contain exactly six values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _assert_scalar_hash_only(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _assert_scalar_hash_only(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_scalar_hash_only(nested, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _authenticated_dataclass_dict(
    value: object,
    *,
    expected_type: type,
    label: str,
) -> dict[str, object]:
    row = (
        value.to_dict()
        if isinstance(value, expected_type)
        else dict(_mapping(value, label=label))
    )
    dataclass_fields = fields(expected_type)
    expected_fields = {item.name for item in dataclass_fields}
    if set(row) != expected_fields:
        raise ValueError(f"{label} fields differ")
    replayed = expected_type(
        **{
            item.name: row[item.name]
            for item in dataclass_fields
            if item.init
        }
    )
    result = replayed.to_dict()
    if _canonical_json_bytes(result) != _canonical_json_bytes(row):
        raise ValueError(f"{label} hash mismatch")
    _assert_scalar_hash_only(result, path=label)
    return result


def _canonical_records(
    values: Sequence[object],
) -> tuple[dict[str, object], ...]:
    records = tuple(
        sorted(
            (
                _authenticated_dataclass_dict(
                    value,
                    expected_type=(
                        GemmaIterativeOccupancyConformalRouteFitRecord
                    ),
                    label="occupancy development fit record",
                )
                for value in values
            ),
            key=lambda row: str(row["example_id"]),
        )
    )
    family_counts = Counter(str(row["family_id"]) for row in records)
    if (
        len(records) != _EXPECTED_EXAMPLES
        or len({row["example_id"] for row in records})
        != _EXPECTED_EXAMPLES
        or len(family_counts) != _EXPECTED_FAMILIES
        or set(family_counts.values()) != {_EXPECTED_PER_FAMILY}
    ):
        raise ValueError(
            "occupancy development records must be a strict 16-by-8 panel"
        )
    return records


def _canonical_folds(
    values: Sequence[object],
    *,
    arm_id: str,
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    folds = tuple(
        sorted(
            (
                _authenticated_dataclass_dict(
                    value,
                    expected_type=(
                        GemmaIterativeOccupancyConformalRouteFoldFit
                    ),
                    label=f"{arm_id} occupancy development fold",
                )
                for value in values
            ),
            key=lambda row: str(row["held_family_id"]),
        )
    )
    families = {str(row["family_id"]) for row in records}
    if (
        len(folds) != _EXPECTED_FAMILIES
        or {row["held_family_id"] for row in folds} != families
    ):
        raise ValueError(
            f"{arm_id} folds must cover the same eight held families"
        )
    by_example = {str(row["example_id"]): row for row in records}
    for fold in folds:
        held = str(fold["held_family_id"])
        if fold["occupancy_kind"] != _KIND_BY_ARM[arm_id]:
            raise ValueError(f"{arm_id} fold has the wrong occupancy kind")
        expected_examples = tuple(
            sorted(
                example_id
                for example_id, row in by_example.items()
                if row["family_id"] != held
            )
        )
        expected_families = tuple(sorted(families - {held}))
        expected_hashes = tuple(
            sorted(
                str(by_example[example_id]["fit_record_sha256"])
                for example_id in expected_examples
            )
        )
        if (
            tuple(fold["train_example_ids"]) != expected_examples
            or tuple(fold["train_family_ids"]) != expected_families
            or tuple(fold["train_fit_record_sha256s"]) != expected_hashes
            or len(expected_examples) != 14
            or len(expected_families) != 7
        ):
            raise ValueError(
                f"{arm_id} fold leaks, omits, or reorders development data"
            )
        expected_active_rows = sum(
            int(by_example[example_id]["active_row_count"])
            for example_id in expected_examples
        )
        if fold["active_row_count"] != expected_active_rows:
            raise ValueError(f"{arm_id} fold active-row count differs")
    return folds


def _median(values: Sequence[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("median requires values")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _coefficient_cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise ValueError("fold coefficient cosine must be finite")
    return max(-1.0, min(1.0, value))


def _family_macro(
    records: Sequence[Mapping[str, object]],
    *,
    field_name: str,
) -> float:
    values_by_family: dict[str, list[float]] = {}
    for record in records:
        values_by_family.setdefault(str(record["family_id"]), []).append(
            _finite(record[field_name], label=field_name)
        )
    family_means = tuple(
        math.fsum(values) / len(values)
        for _, values in sorted(values_by_family.items())
    )
    return math.fsum(family_means) / len(family_means)


def _arm_scientific_gates(
    *,
    arm_id: str,
    records: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    folds_by_family = {
        str(fold["held_family_id"]): fold
        for fold in folds
    }
    predicted_by_family: dict[str, list[float]] = {}
    for record in records:
        family_id = str(record["family_id"])
        fold = folds_by_family[family_id]
        jacobian = _float6(
            record[_JACOBIAN_FIELD_BY_ARM[arm_id]],
            label=f"{arm_id} held-prompt Jacobian",
        )
        coefficients = _float6(
            fold[
                "coefficients_by_occupancy_conformal_coefficient"
            ],
            label=f"{arm_id} fold coefficients",
        )
        predicted = _finite(
            record["parent_signed_delta_nll_per_token"],
            label="parent signed delta NLL",
        ) + math.fsum(
            left * right
            for left, right in zip(jacobian, coefficients, strict=True)
        )
        if not math.isfinite(predicted):
            raise ValueError(f"{arm_id} predicted delta NLL is nonfinite")
        predicted_by_family.setdefault(family_id, []).append(abs(predicted))
    predicted_macro = math.fsum(
        math.fsum(values) / len(values)
        for _, values in sorted(predicted_by_family.items())
    ) / len(predicted_by_family)

    raw_rank_by_family = {
        str(fold["held_family_id"]): int(
            fold["raw_weighted_design_rank"]
        )
        for fold in folds
    }
    standardized_rank_by_family = {
        str(fold["held_family_id"]): int(
            fold["standardized_weighted_design_rank"]
        )
        for fold in folds
    }
    supported_counts = {
        coordinate: sum(
            index
            not in tuple(
                fold[
                    "unsupported_occupancy_conformal_coefficient_indices"
                ]
            )
            for fold in folds
        )
        for index, coordinate in enumerate(_COEFFICIENT_ORDER)
    }
    raw_conditions = tuple(
        _finite(
            fold["raw_normal_condition_number"],
            label=f"{arm_id} raw normal condition",
        )
        for fold in folds
    )
    standardized_conditions = tuple(
        _finite(
            fold["standardized_normal_condition_number"],
            label=f"{arm_id} standardized normal condition",
        )
        for fold in folds
    )
    median_raw_condition = _median(raw_conditions)
    median_standardized_condition = _median(standardized_conditions)
    coefficient_rows = tuple(
        _float6(
            fold[
                "coefficients_by_occupancy_conformal_coefficient"
            ],
            label=f"{arm_id} fold coefficients",
        )
        for fold in folds
    )
    pairwise_cosines = tuple(
        _coefficient_cosine(coefficient_rows[left], coefficient_rows[right])
        for left in range(len(coefficient_rows))
        for right in range(left + 1, len(coefficient_rows))
    )
    if len(pairwise_cosines) != _EXPECTED_FOLD_PAIRS:
        raise ValueError("occupancy fold stability requires 28 pairs")
    mean_cosine = math.fsum(pairwise_cosines) / len(pairwise_cosines)

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
        int(record["negative_balance_row_count"])
        for record in records
    )
    nonnegative_rows = sum(
        int(record["nonnegative_balance_row_count"])
        for record in records
    )

    result: dict[str, object] = {
        "predicted_family_macro_mean_absolute_delta_nll_per_token": (
            predicted_macro
        ),
        "raw_weighted_design_rank_by_held_family": raw_rank_by_family,
        "standardized_weighted_design_rank_by_held_family": (
            standardized_rank_by_family
        ),
        "all_fold_raw_weighted_design_ranks_exactly_6": all(
            rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            for rank in raw_rank_by_family.values()
        ),
        "all_fold_standardized_weighted_design_ranks_exactly_6": all(
            rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            for rank in standardized_rank_by_family.values()
        ),
        "supported_fold_count_by_occupancy_conformal_coordinate": (
            supported_counts
        ),
        "all_6_occupancy_conformal_coordinates_supported_in_every_fold": all(
            count == _EXPECTED_FAMILIES
            for count in supported_counts.values()
        ),
        "median_fold_raw_normal_condition_number": median_raw_condition,
        "median_fold_raw_normal_condition_number_at_most_100": (
            median_raw_condition <= _MAX_MEDIAN_CONDITION
        ),
        "median_fold_standardized_normal_condition_number": (
            median_standardized_condition
        ),
        "median_fold_standardized_normal_condition_number_at_most_100": (
            median_standardized_condition <= _MAX_MEDIAN_CONDITION
        ),
        "mean_pairwise_fold_coefficient_cosine": mean_cosine,
        "mean_pairwise_fold_coefficient_cosine_at_least_0_90": (
            mean_cosine >= _MIN_MEAN_FOLD_COSINE
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
    result["passed"] = all(bool(result[field]) for field in _BOOLEAN_GATE_FIELDS)
    return result


def _selected_arm(
    scientific_gates_by_arm: Mapping[str, Mapping[str, object]],
) -> str:
    eligible = tuple(
        arm_id
        for arm_id in _ARMS
        if scientific_gates_by_arm[arm_id]["passed"] is True
    )
    if not eligible:
        raise ValueError(
            "no occupancy arm passes the development gates: "
            + _canonical_json_bytes(scientific_gates_by_arm).decode("ascii")
        )
    return min(
        eligible,
        key=lambda arm_id: (
            float(
                scientific_gates_by_arm[arm_id][
                    "predicted_family_macro_mean_absolute_delta_nll_per_token"
                ]
            ),
            0 if arm_id == CUMULATIVE_OCCUPANCY_ARM else 1,
        ),
    )


def build_gemma_iterative_occupancy_development_selection(
    *,
    fit_records: Sequence[object],
    fold_receipts_by_arm: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    """Freeze one scientifically supported arm without opening fresh inputs."""

    records = _canonical_records(fit_records)
    raw_folds = _mapping(
        fold_receipts_by_arm,
        label="occupancy development folds by arm",
    )
    if set(raw_folds) != set(_ARMS):
        raise ValueError("occupancy development fold arms differ")
    folds = {
        arm_id: _canonical_folds(
            raw_folds[arm_id],
            arm_id=arm_id,
            records=records,
        )
        for arm_id in _ARMS
    }
    if {
        tuple(str(row["held_family_id"]) for row in arm_folds)
        for arm_folds in folds.values()
    } != {
        tuple(sorted({str(row["family_id"]) for row in records}))
    }:
        raise ValueError(
            "occupancy arms do not cover the same held families"
        )
    gates = {
        arm_id: _arm_scientific_gates(
            arm_id=arm_id,
            records=records,
            folds=folds[arm_id],
        )
        for arm_id in _ARMS
    }
    result: dict[str, object] = {
        "selected_arm_id": _selected_arm(gates),
        "selection_opened": False,
        "selection_rule_frozen": True,
        "scientific_gates_by_arm": gates,
        "selection_rule": OCCUPANCY_DEVELOPMENT_SELECTION_RULE,
    }
    validate_gemma_iterative_occupancy_development_selection(result)
    return result


def _rank_map(
    value: object,
    *,
    label: str,
) -> dict[str, int]:
    row = _mapping(value, label=label)
    if len(row) != _EXPECTED_FAMILIES:
        raise ValueError(f"{label} must cover eight families")
    result: dict[str, int] = {}
    for family_id, rank in row.items():
        if (
            not isinstance(family_id, str)
            or not family_id
            or type(rank) is not int
            or not 0
            <= rank
            <= OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
        ):
            raise ValueError(f"{label} is invalid")
        result[family_id] = rank
    return result


def validate_gemma_iterative_occupancy_development_selection(
    value: Mapping[str, object],
) -> None:
    """Validate the closed, deterministic scalar selection receipt."""

    row = dict(_mapping(value, label="occupancy development selection"))
    expected = {
        "selected_arm_id",
        "selection_opened",
        "selection_rule_frozen",
        "scientific_gates_by_arm",
        "selection_rule",
    }
    if set(row) != expected:
        raise ValueError("occupancy development selection fields differ")
    _assert_scalar_hash_only(row, path="occupancy development selection")
    if (
        row["selection_opened"] is not False
        or row["selection_rule_frozen"] is not True
        or row["selection_rule"] != OCCUPANCY_DEVELOPMENT_SELECTION_RULE
    ):
        raise ValueError("occupancy development selection was not frozen")
    raw_gates = _mapping(
        row["scientific_gates_by_arm"],
        label="occupancy development scientific gates",
    )
    if set(raw_gates) != set(_ARMS):
        raise ValueError("occupancy development scientific arms differ")
    gates: dict[str, dict[str, object]] = {}
    for arm_id in _ARMS:
        arm = dict(
            _mapping(
                raw_gates[arm_id],
                label=f"{arm_id} scientific gates",
            )
        )
        if set(arm) != _ARM_METRIC_FIELDS:
            raise ValueError(f"{arm_id} scientific gate fields differ")
        raw_ranks = _rank_map(
            arm["raw_weighted_design_rank_by_held_family"],
            label=f"{arm_id} raw ranks",
        )
        standardized_ranks = _rank_map(
            arm["standardized_weighted_design_rank_by_held_family"],
            label=f"{arm_id} standardized ranks",
        )
        if set(raw_ranks) != set(standardized_ranks):
            raise ValueError(f"{arm_id} rank families differ")
        support = _mapping(
            arm[
                "supported_fold_count_by_occupancy_conformal_coordinate"
            ],
            label=f"{arm_id} coordinate support",
        )
        if set(support) != set(_COEFFICIENT_ORDER) or any(
            type(count) is not int or not 0 <= count <= _EXPECTED_FAMILIES
            for count in support.values()
        ):
            raise ValueError(f"{arm_id} coordinate support is invalid")
        float_fields = (
            "predicted_family_macro_mean_absolute_delta_nll_per_token",
            "median_fold_raw_normal_condition_number",
            "median_fold_standardized_normal_condition_number",
            "mean_pairwise_fold_coefficient_cosine",
            "family_macro_balance_feature_std",
            "family_macro_occupancy_feature_std",
            "family_macro_top2_modal_energy_fraction",
        )
        metrics = {
            name: _finite(arm[name], label=f"{arm_id} {name}")
            for name in float_fields
        }
        if (
            metrics[
                "predicted_family_macro_mean_absolute_delta_nll_per_token"
            ]
            < 0.0
            or metrics["median_fold_raw_normal_condition_number"] < 0.0
            or metrics[
                "median_fold_standardized_normal_condition_number"
            ]
            < 0.0
            or not -1.0
            <= metrics["mean_pairwise_fold_coefficient_cosine"]
            <= 1.0
            or metrics["family_macro_balance_feature_std"] < 0.0
            or metrics["family_macro_occupancy_feature_std"] < 0.0
            or not 0.0
            <= metrics["family_macro_top2_modal_energy_fraction"]
            <= 1.0
        ):
            raise ValueError(f"{arm_id} scientific metric is invalid")
        for name in (*_BOOLEAN_GATE_FIELDS, "passed"):
            if type(arm[name]) is not bool:
                raise ValueError(f"{arm_id} gate {name} is not boolean")
        negative_rows = arm["negative_balance_row_count"]
        nonnegative_rows = arm["nonnegative_balance_row_count"]
        if (
            type(negative_rows) is not int
            or type(nonnegative_rows) is not int
            or negative_rows < 0
            or nonnegative_rows < 0
            or negative_rows + nonnegative_rows <= 0
        ):
            raise ValueError(f"{arm_id} occupancy sign counts are invalid")
        expected_gates = {
            "all_fold_raw_weighted_design_ranks_exactly_6": all(
                rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
                for rank in raw_ranks.values()
            ),
            "all_fold_standardized_weighted_design_ranks_exactly_6": all(
                rank == OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
                for rank in standardized_ranks.values()
            ),
            (
                "all_6_occupancy_conformal_coordinates_"
                "supported_in_every_fold"
            ): all(
                count == _EXPECTED_FAMILIES
                for count in support.values()
            ),
            "median_fold_raw_normal_condition_number_at_most_100": (
                metrics["median_fold_raw_normal_condition_number"]
                <= _MAX_MEDIAN_CONDITION
            ),
            (
                "median_fold_standardized_normal_condition_number_"
                "at_most_100"
            ): (
                metrics[
                    "median_fold_standardized_normal_condition_number"
                ]
                <= _MAX_MEDIAN_CONDITION
            ),
            "mean_pairwise_fold_coefficient_cosine_at_least_0_90": (
                metrics["mean_pairwise_fold_coefficient_cosine"]
                >= _MIN_MEAN_FOLD_COSINE
            ),
            "family_macro_balance_feature_std_at_least_0_05": (
                metrics["family_macro_balance_feature_std"]
                >= _MIN_FEATURE_STD
            ),
            "family_macro_occupancy_feature_std_at_least_0_05": (
                metrics["family_macro_occupancy_feature_std"]
                >= _MIN_FEATURE_STD
            ),
            "family_macro_top2_modal_energy_fraction_at_least_0_5": (
                metrics["family_macro_top2_modal_energy_fraction"]
                >= _MIN_TOP2_ENERGY
            ),
            "both_occupancy_signs_seen": (
                negative_rows > 0 and nonnegative_rows > 0
            ),
        }
        if any(
            arm[name] is not expected
            for name, expected in expected_gates.items()
        ):
            raise ValueError(f"{arm_id} scientific gate contradicts metrics")
        expected_passed = all(
            bool(arm[name]) for name in _BOOLEAN_GATE_FIELDS
        )
        if arm["passed"] is not expected_passed:
            raise ValueError(f"{arm_id} scientific decision differs")
        gates[arm_id] = arm
    selected = _selected_arm(gates)
    if row["selected_arm_id"] != selected:
        raise ValueError("occupancy development selected arm differs")
