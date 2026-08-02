from __future__ import annotations

import copy
import hashlib
import math

import pytest

from fisher_graph.gemma3_l3_l4_iterative_occupancy_development import (
    OCCUPANCY_DEVELOPMENT_SELECTION_RULE,
    build_gemma_iterative_occupancy_development_selection,
    validate_gemma_iterative_occupancy_development_selection,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
    OCCUPANCY_STANDARDIZED_RIDGE,
    GemmaIterativeOccupancyConformalRouteFitRecord,
    GemmaIterativeOccupancyConformalRouteFoldFit,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _records(
    *,
    ew_occupancy_jacobian: float = -0.5,
    negative_rows: int = 4,
) -> tuple[GemmaIterativeOccupancyConformalRouteFitRecord, ...]:
    records = []
    for index in range(16):
        shared = tuple(
            -1.0 if index & (1 << coordinate) else 1.0
            for coordinate in range(4)
        )
        records.append(
            GemmaIterativeOccupancyConformalRouteFitRecord(
                example_id=f"development-example-{index:02d}",
                family_id=f"development-family-{index % 8}",
                model_inputs_sha256=_sha(f"inputs-{index}"),
                parent_execution_sha256=_sha(f"parent-execution-{index}"),
                parent_observation_sha256=_sha(
                    f"parent-observation-{index}"
                ),
                parent_h4_artifact_sha256=_sha("parent-h4"),
                prefix_sha256=_sha(f"prefix-{index}"),
                gradient_sha256=_sha(f"gradient-{index}"),
                parent_modal_sha256=_sha(f"parent-modal-{index}"),
                balance_feature_sha256=_sha(f"balance-{index}"),
                cumulative_occupancy_feature_sha256=_sha(
                    f"cumulative-occupancy-{index}"
                ),
                ew_occupancy_feature_sha256=_sha(f"ew-occupancy-{index}"),
                shared_feature_sha256=_sha(f"shared-{index}"),
                balance_contrast_feature_sha256=_sha(
                    f"balance-contrast-{index}"
                ),
                cumulative_occupancy_contrast_feature_sha256=_sha(
                    f"cumulative-contrast-{index}"
                ),
                ew_occupancy_contrast_feature_sha256=_sha(
                    f"ew-contrast-{index}"
                ),
                supervised_tokens=10,
                parent_signed_delta_nll_per_token=0.2,
                jacobian_by_cumulative_occupancy_conformal_coefficient=(
                    *shared,
                    -1.0,
                    -1.0,
                ),
                jacobian_by_ew_occupancy_conformal_coefficient=(
                    *shared,
                    ew_occupancy_jacobian,
                    ew_occupancy_jacobian,
                ),
                active_row_count=10,
                negative_balance_row_count=negative_rows,
                nonnegative_balance_row_count=10 - negative_rows,
                top_mode_indices=(0, 1),
                top_mode_norms=(2.0, 1.0),
                balance_feature_std=0.2,
                cumulative_occupancy_feature_std=0.2,
                ew_occupancy_feature_std=0.2,
                top2_modal_energy_fraction=0.8,
            )
        )
    return tuple(records)


def _corner_norms(
    coefficients: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
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
    )


def _folds(
    records: tuple[
        GemmaIterativeOccupancyConformalRouteFitRecord, ...
    ],
    *,
    occupancy_kind: str,
    raw_condition: float = 10.0,
    standardized_condition: float = 10.0,
    coefficients: tuple[
        float, float, float, float, float, float
    ] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
) -> tuple[GemmaIterativeOccupancyConformalRouteFoldFit, ...]:
    corners = _corner_norms(coefficients)
    result = []
    for held_index in range(8):
        held = f"development-family-{held_index}"
        training = tuple(
            sorted(
                (record for record in records if record.family_id != held),
                key=lambda record: record.example_id,
            )
        )
        result.append(
            GemmaIterativeOccupancyConformalRouteFoldFit(
                occupancy_kind=occupancy_kind,
                held_family_id=held,
                train_example_ids=tuple(
                    record.example_id for record in training
                ),
                train_family_ids=tuple(
                    sorted({record.family_id for record in training})
                ),
                train_fit_record_sha256s=tuple(
                    sorted(record.fit_record_sha256 for record in training)
                ),
                coefficients_by_occupancy_conformal_coefficient=coefficients,
                unsupported_occupancy_conformal_coefficient_indices=(),
                active_row_count=sum(
                    record.active_row_count for record in training
                ),
                weighted_column_scale_by_occupancy_conformal_coefficient=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                raw_weighted_design_rank=6,
                standardized_weighted_design_rank=6,
                raw_normal_condition_number=raw_condition,
                standardized_normal_condition_number=(
                    standardized_condition
                ),
                pre_projection_corner_operator_norms=corners,
                post_projection_corner_operator_norms=corners,
                trust_projection_scale=1.0,
                linearized_rmse_before=0.2,
                linearized_rmse_after=0.1,
                trust_projection_applied=False,
                ridge=OCCUPANCY_STANDARDIZED_RIDGE,
                operator_norm_bound=OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
            )
        )
    return tuple(result)


def _selection(
    records: tuple[
        GemmaIterativeOccupancyConformalRouteFitRecord, ...
    ],
    *,
    cumulative_raw_condition: float = 10.0,
    cumulative_standardized_condition: float = 10.0,
) -> dict[str, object]:
    return build_gemma_iterative_occupancy_development_selection(
        fit_records=records,
        fold_receipts_by_arm={
            CUMULATIVE_OCCUPANCY_ARM: _folds(
                records,
                occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
                raw_condition=cumulative_raw_condition,
                standardized_condition=cumulative_standardized_condition,
            ),
            EW_OCCUPANCY_ARM: _folds(
                records,
                occupancy_kind=CENTERED_EW_OCCUPANCY,
            ),
        },
    )


def test_development_selection_replays_lofo_predictions_and_freezes_arm() -> None:
    selection = _selection(_records())

    validate_gemma_iterative_occupancy_development_selection(selection)
    assert set(selection) == {
        "selected_arm_id",
        "selection_opened",
        "selection_rule_frozen",
        "scientific_gates_by_arm",
        "selection_rule",
    }
    assert selection["selected_arm_id"] == CUMULATIVE_OCCUPANCY_ARM
    assert selection["selection_opened"] is False
    assert selection["selection_rule_frozen"] is True
    assert (
        selection["selection_rule"]
        == OCCUPANCY_DEVELOPMENT_SELECTION_RULE
    )
    gates = selection["scientific_gates_by_arm"]
    cumulative = gates[CUMULATIVE_OCCUPANCY_ARM]  # type: ignore[index]
    ew = gates[EW_OCCUPANCY_ARM]  # type: ignore[index]
    assert cumulative["passed"] is True
    assert ew["passed"] is True
    assert cumulative[
        "predicted_family_macro_mean_absolute_delta_nll_per_token"
    ] < ew["predicted_family_macro_mean_absolute_delta_nll_per_token"]
    assert cumulative[
        "mean_pairwise_fold_coefficient_cosine"
    ] == pytest.approx(1.0)


def test_selection_is_order_invariant_and_cumulative_wins_exact_tie() -> None:
    records = _records(ew_occupancy_jacobian=-1.0)
    cumulative = _folds(
        records,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
    )
    ew = _folds(records, occupancy_kind=CENTERED_EW_OCCUPANCY)

    first = build_gemma_iterative_occupancy_development_selection(
        fit_records=records,
        fold_receipts_by_arm={
            CUMULATIVE_OCCUPANCY_ARM: cumulative,
            EW_OCCUPANCY_ARM: ew,
        },
    )
    reversed_input = build_gemma_iterative_occupancy_development_selection(
        fit_records=tuple(reversed(records)),
        fold_receipts_by_arm={
            CUMULATIVE_OCCUPANCY_ARM: tuple(reversed(cumulative)),
            EW_OCCUPANCY_ARM: tuple(reversed(ew)),
        },
    )

    assert first == reversed_input
    assert first["selected_arm_id"] == CUMULATIVE_OCCUPANCY_ARM


def test_raw_condition_is_diagnostic_but_standardized_condition_is_a_gate() -> None:
    raw_ill_conditioned = _selection(
        _records(),
        cumulative_raw_condition=1_000.0,
    )
    cumulative = raw_ill_conditioned[
        "scientific_gates_by_arm"
    ][CUMULATIVE_OCCUPANCY_ARM]  # type: ignore[index]
    assert cumulative[
        "median_fold_raw_normal_condition_number_at_most_100"
    ] is False
    assert cumulative["passed"] is True
    assert (
        raw_ill_conditioned["selected_arm_id"]
        == CUMULATIVE_OCCUPANCY_ARM
    )

    standardized_ill_conditioned = _selection(
        _records(),
        cumulative_standardized_condition=1_000.0,
    )
    cumulative = standardized_ill_conditioned[
        "scientific_gates_by_arm"
    ][CUMULATIVE_OCCUPANCY_ARM]  # type: ignore[index]
    assert cumulative[
        "median_fold_standardized_normal_condition_number_at_most_100"
    ] is False
    assert cumulative["passed"] is False
    assert standardized_ill_conditioned["selected_arm_id"] == EW_OCCUPANCY_ARM


def test_selection_requires_both_occupancy_signs() -> None:
    with pytest.raises(ValueError, match="no occupancy arm passes"):
        _selection(_records(negative_rows=0))


def test_selection_rejects_fold_leakage_and_hash_tampering() -> None:
    records = _records()
    cumulative = list(
        _folds(
            records,
            occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
        )
    )
    ew = _folds(records, occupancy_kind=CENTERED_EW_OCCUPANCY)
    tampered = copy.deepcopy(cumulative[0].to_dict())
    tampered["fold_receipt_sha256"] = _sha("tampered")
    cumulative[0] = tampered  # type: ignore[list-item]

    with pytest.raises(ValueError, match="hash mismatch"):
        build_gemma_iterative_occupancy_development_selection(
            fit_records=records,
            fold_receipts_by_arm={
                CUMULATIVE_OCCUPANCY_ARM: cumulative,
                EW_OCCUPANCY_ARM: ew,
            },
        )

    leaked = copy.deepcopy(ew[0].to_dict())
    leaked["held_family_id"] = "development-family-1"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_gemma_iterative_occupancy_development_selection(
            fit_records=records,
            fold_receipts_by_arm={
                CUMULATIVE_OCCUPANCY_ARM: _folds(
                    records,
                    occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
                ),
                EW_OCCUPANCY_ARM: (leaked, *ew[1:]),
            },
        )
