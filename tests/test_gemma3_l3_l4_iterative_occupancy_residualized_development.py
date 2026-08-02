from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
import math

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_occupancy_residualized_development
    as residualized_development,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_residualized_development import (
    OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR,
    build_gemma_iterative_residualized_occupancy_development_report,
    validate_gemma_iterative_residualized_occupancy_development_report,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_route import (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
    GemmaIterativeOccupancyConformalRouteFitRecord,
    GemmaIterativeOccupancyConformalRouteFoldFit,
    fit_gemma_iterative_occupancy_conformal_route_fold,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)


_ARMS_AND_KINDS = (
    (CUMULATIVE_OCCUPANCY_ARM, CENTERED_CUMULATIVE_OCCUPANCY),
    (EW_OCCUPANCY_ARM, CENTERED_EW_OCCUPANCY),
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _walsh(row: int, mask: int) -> float:
    return -1.0 if (row & mask).bit_count() % 2 else 1.0


def _records(
    *,
    occupancy_novelty: float = 0.5,
    occupancy_collinearity: float | None = None,
    unstable_target: bool = False,
) -> tuple[GemmaIterativeOccupancyConformalRouteFitRecord, ...]:
    """Make a strict 16-by-8 panel with controllable occupancy novelty."""

    result = []
    for index in range(16):
        base = (
            _walsh(index, 0),
            _walsh(index, 1),
            _walsh(index, 2),
            _walsh(index, 4),
        )
        cumulative_novel = (
            _walsh(index, 8),
            _walsh(index, 3),
        )
        ew_novel = (
            _walsh(index, 5),
            _walsh(index, 9),
        )
        if occupancy_collinearity is None:
            cumulative_occupancy = (
                base[0] + occupancy_novelty * cumulative_novel[0],
                base[1] + occupancy_novelty * cumulative_novel[1],
            )
            ew_occupancy = (
                base[2] + occupancy_novelty * ew_novel[0],
                base[3] + occupancy_novelty * ew_novel[1],
            )
        else:
            cumulative_occupancy = (
                base[0] + occupancy_novelty * cumulative_novel[0],
                base[1]
                + occupancy_novelty
                * (
                    cumulative_novel[0]
                    + occupancy_collinearity * cumulative_novel[1]
                ),
            )
            ew_occupancy = (
                base[2] + occupancy_novelty * ew_novel[0],
                base[3]
                + occupancy_novelty
                * (
                    ew_novel[0]
                    + occupancy_collinearity * ew_novel[1]
                ),
            )
        parent_delta = -(
            0.030 * base[0]
            - 0.020 * base[1]
            + 0.015 * base[2]
            + 0.010 * base[3]
        )
        if unstable_target:
            family_sign = 1.0 if index % 8 < 4 else -1.0
            duplicate_sign = 1.0 if index < 8 else -1.0
            parent_delta += 0.12 * family_sign * duplicate_sign
        result.append(
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
                parent_signed_delta_nll_per_token=parent_delta,
                jacobian_by_cumulative_occupancy_conformal_coefficient=(
                    *base,
                    *cumulative_occupancy,
                ),
                jacobian_by_ew_occupancy_conformal_coefficient=(
                    *base,
                    *ew_occupancy,
                ),
                active_row_count=10,
                negative_balance_row_count=4,
                nonnegative_balance_row_count=6,
                top_mode_indices=(0, 1),
                top_mode_norms=(2.0, 1.0),
                balance_feature_std=0.2,
                cumulative_occupancy_feature_std=0.2,
                ew_occupancy_feature_std=0.2,
                top2_modal_energy_fraction=0.8,
            )
        )
    return tuple(result)


def _direct_folds(
    records: tuple[
        GemmaIterativeOccupancyConformalRouteFitRecord, ...
    ],
) -> dict[
    str,
    tuple[GemmaIterativeOccupancyConformalRouteFoldFit, ...],
]:
    families = tuple(sorted({record.family_id for record in records}))
    return {
        arm_id: tuple(
            fit_gemma_iterative_occupancy_conformal_route_fold(
                tuple(
                    record
                    for record in records
                    if record.family_id != held_family
                ),
                held_family_id=held_family,
                occupancy_kind=occupancy_kind,
            )
            for held_family in families
        )
        for arm_id, occupancy_kind in _ARMS_AND_KINDS
    }


def _report(
    *,
    occupancy_novelty: float = 0.5,
    occupancy_collinearity: float | None = None,
    unstable_target: bool = False,
) -> tuple[
    dict[str, object],
    tuple[GemmaIterativeOccupancyConformalRouteFitRecord, ...],
    dict[
        str,
        tuple[GemmaIterativeOccupancyConformalRouteFoldFit, ...],
    ],
]:
    records = _records(
        occupancy_novelty=occupancy_novelty,
        occupancy_collinearity=occupancy_collinearity,
        unstable_target=unstable_target,
    )
    direct = _direct_folds(records)
    report = (
        build_gemma_iterative_residualized_occupancy_development_report(
            fit_records=records,
            direct_fold_receipts_by_arm=direct,
        )
    )
    return report, records, direct


def _rebuild_fold(
    value: GemmaIterativeOccupancyConformalRouteFoldFit
    | dict[str, object],
    **changes: object,
) -> GemmaIterativeOccupancyConformalRouteFoldFit:
    row = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    payload = {
        item.name: row[item.name]
        for item in fields(GemmaIterativeOccupancyConformalRouteFoldFit)
        if item.init
    }
    payload.update(changes)
    return GemmaIterativeOccupancyConformalRouteFoldFit(
        **payload,  # type: ignore[arg-type]
    )


def _rehash_report(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("report_sha256", None)
    value["report_sha256"] = residualized_development._sha256(payload)


def _coefficient_cosine(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    left_norm = math.sqrt(math.fsum(item * item for item in left))
    right_norm = math.sqrt(math.fsum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return math.fsum(
        x * y for x, y in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def test_residualized_report_exactly_replays_and_authenticates_hashes() -> None:
    report, records, direct = _report()

    validate_gemma_iterative_residualized_occupancy_development_report(
        report
    )
    disk_replay = json.loads(json.dumps(report, sort_keys=True))
    validate_gemma_iterative_residualized_occupancy_development_report(
        disk_replay
    )
    reordered = (
        build_gemma_iterative_residualized_occupancy_development_report(
            fit_records=tuple(reversed(records)),
            direct_fold_receipts_by_arm={
                arm_id: tuple(reversed(folds))
                for arm_id, folds in direct.items()
            },
        )
    )
    assert reordered == report

    hash_tampered = copy.deepcopy(report)
    hash_tampered["viable_on_reusable_development"] = False
    with pytest.raises(ValueError, match="report hash differs"):
        validate_gemma_iterative_residualized_occupancy_development_report(
            hash_tampered
        )

    replay_tampered = copy.deepcopy(report)
    folds_by_arm = replay_tampered[
        "residualized_fold_receipts_by_arm"
    ]
    cumulative = list(folds_by_arm[CUMULATIVE_OCCUPANCY_ARM])  # type: ignore[index]
    first = cumulative[0]
    changed = _rebuild_fold(
        first,  # type: ignore[arg-type]
        linearized_rmse_after=(
            float(first["linearized_rmse_after"]) + 0.001  # type: ignore[index]
        ),
    )
    cumulative[0] = changed.to_dict()
    folds_by_arm[CUMULATIVE_OCCUPANCY_ARM] = cumulative  # type: ignore[index]
    _rehash_report(replay_tampered)
    with pytest.raises(ValueError, match="does not replay"):
        validate_gemma_iterative_residualized_occupancy_development_report(
            replay_tampered
        )

    direct_hash_tampered = {
        arm_id: list(folds) for arm_id, folds in direct.items()
    }
    first_direct = direct[CUMULATIVE_OCCUPANCY_ARM][0].to_dict()
    first_direct["fold_receipt_sha256"] = _sha("tampered-direct-fold")
    direct_hash_tampered[CUMULATIVE_OCCUPANCY_ARM][0] = first_direct
    with pytest.raises(ValueError, match="hash mismatch"):
        build_gemma_iterative_residualized_occupancy_development_report(
            fit_records=records,
            direct_fold_receipts_by_arm=direct_hash_tampered,
        )


def test_no_passing_arm_fails_closed_without_opening_selection() -> None:
    report, _, _ = _report(occupancy_novelty=0.2)

    assert report["selected_residualized_arm_id"] is None
    assert report["viable_on_reusable_development"] is False
    assert report["selection_panel_authorized"] is False
    assert report["selection_panel_opened"] is False
    assert report["selection_claim_created"] is False
    assert report["next_action"] == (
        "stop_residualized_occupancy_without_fresh_run"
    )
    metrics = report["residualized_scientific_metrics_by_arm"]
    for arm_id, _ in _ARMS_AND_KINDS:
        assert metrics[arm_id]["passed"] is False  # type: ignore[index]


def test_five_percent_residual_energy_gate_is_frozen() -> None:
    passing, _, _ = _report(occupancy_novelty=0.35)
    failing, _, _ = _report(occupancy_novelty=0.2)

    assert OCCUPANCY_RESIDUAL_RETAINED_ENERGY_FLOOR == 0.05
    assert passing["residual_retained_energy_floor"] == 0.05
    assert failing["residual_retained_energy_floor"] == 0.05
    for report, expected in ((passing, True), (failing, False)):
        metrics = report["residualized_scientific_metrics_by_arm"]
        for arm_id, _ in _ARMS_AND_KINDS:
            arm = metrics[arm_id]  # type: ignore[index]
            assert arm[
                "all_fold_occupancy_residual_energy_fractions_at_least_0_05"
            ] is expected
            if expected:
                assert arm[
                    "minimum_fold_occupancy_residual_energy_fraction"
                ] >= 0.05
            else:
                assert arm[
                    "minimum_fold_occupancy_residual_energy_fraction"
                ] < 0.05


def test_condition_and_mapped_runtime_cosine_gate_independently() -> None:
    ill_conditioned, _, _ = _report(
        occupancy_novelty=0.5,
        occupancy_collinearity=0.1,
    )
    unstable, _, _ = _report(
        occupancy_novelty=0.5,
        unstable_target=True,
    )

    ill_metrics = ill_conditioned[
        "residualized_scientific_metrics_by_arm"
    ]
    unstable_metrics = unstable[
        "residualized_scientific_metrics_by_arm"
    ]
    for arm_id, _ in _ARMS_AND_KINDS:
        condition = ill_metrics[arm_id]  # type: ignore[index]
        assert condition[
            "median_fold_standardized_normal_condition_number"
        ] > 100.0
        assert condition[
            "median_fold_standardized_normal_condition_number_at_most_100"
        ] is False
        assert condition[
            "mean_pairwise_mapped_runtime_coefficient_cosine_at_least_0_90"
        ] is True

        cosine = unstable_metrics[arm_id]  # type: ignore[index]
        assert cosine[
            "median_fold_standardized_normal_condition_number_at_most_100"
        ] is True
        assert cosine[
            "mean_pairwise_mapped_runtime_coefficient_cosine"
        ] < 0.90
        assert cosine[
            "mean_pairwise_mapped_runtime_coefficient_cosine_at_least_0_90"
        ] is False

        fold_rows = unstable[
            "residualized_fold_receipts_by_arm"
        ][arm_id]  # type: ignore[index]
        mapped = tuple(
            tuple(
                float(item)
                for item in row[
                    "coefficients_by_occupancy_conformal_coefficient"
                ]
            )
            for row in fold_rows
        )
        pairwise = tuple(
            _coefficient_cosine(mapped[left], mapped[right])
            for left in range(len(mapped))
            for right in range(left + 1, len(mapped))
        )
        assert len(pairwise) == 28
        assert cosine[
            "mean_pairwise_mapped_runtime_coefficient_cosine"
        ] == pytest.approx(math.fsum(pairwise) / len(pairwise))


def test_direct_fold_leakage_and_recomputed_fit_drift_are_rejected() -> None:
    records = _records()
    direct = _direct_folds(records)

    leaking = {
        arm_id: list(folds) for arm_id, folds in direct.items()
    }
    leaking[CUMULATIVE_OCCUPANCY_ARM][0] = _rebuild_fold(
        direct[CUMULATIVE_OCCUPANCY_ARM][1],
        held_family_id="development-family-0",
    )
    with pytest.raises(ValueError, match="leaks, omits, or reorders"):
        build_gemma_iterative_residualized_occupancy_development_report(
            fit_records=records,
            direct_fold_receipts_by_arm=leaking,
        )

    drifted = {
        arm_id: list(folds) for arm_id, folds in direct.items()
    }
    drifted[CUMULATIVE_OCCUPANCY_ARM][0] = _rebuild_fold(
        direct[CUMULATIVE_OCCUPANCY_ARM][0],
        coefficients_by_occupancy_conformal_coefficient=(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        pre_projection_corner_operator_norms=(0.0, 0.0, 0.0, 0.0),
        post_projection_corner_operator_norms=(0.0, 0.0, 0.0, 0.0),
        trust_projection_scale=1.0,
        trust_projection_applied=False,
    )
    with pytest.raises(ValueError, match="does not replay from training"):
        build_gemma_iterative_residualized_occupancy_development_report(
            fit_records=records,
            direct_fold_receipts_by_arm=drifted,
        )


def test_report_freezes_development_boundary_flags() -> None:
    report, _, _ = _report()

    assert report["development_role"] == "calibration_a_fit_only"
    assert report["viable_on_reusable_development"] is True
    assert (
        report["selected_residualized_arm_id"]
        == CUMULATIVE_OCCUPANCY_ARM
    )
    assert report["selection_panel_authorized"] is False
    assert report["selection_panel_opened"] is False
    assert report["selection_claim_created"] is False
    assert report["fresh_confirmation_requires_new_blinded_panel"] is True
    assert report["next_action"] == "freeze_new_blinded_selection_plan"

    boundary_tampered = copy.deepcopy(report)
    boundary_tampered["selection_panel_opened"] = True
    _rehash_report(boundary_tampered)
    with pytest.raises(ValueError, match="development boundary differs"):
        validate_gemma_iterative_residualized_occupancy_development_report(
            boundary_tampered
        )
