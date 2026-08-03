from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

import fisher_graph.complete_h4_fisher_soft_polarity_trust_region_fit as fit_core
import fisher_graph.complete_h4_fisher_soft_polarity_fit as v20f_fit_core
from fisher_graph.complete_h4_fisher_soft_polarity import (
    FISHER_SOFT_POLARITY_ETA_MAX_ABS,
    fisher_soft_polarity_box_certificate,
)
from fisher_graph.complete_h4_fisher_soft_polarity_trust_region_fit import (
    SOFT_POLARITY_FIT_ALPHAS,
    SOFT_POLARITY_FIT_ETA_MAX_ABS,
    SOFT_POLARITY_FIT_PROTOCOL_SHA256,
    build_soft_polarity_candidate_receipt,
    build_soft_polarity_direction_receipt,
    build_soft_polarity_fold_receipt,
    build_soft_polarity_oof_qualification,
    soft_polarity_work_accounting,
    validate_soft_polarity_candidate_receipt,
    validate_soft_polarity_direction_receipt,
    validate_soft_polarity_fold_receipt,
    validate_soft_polarity_oof_qualification,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
ARMS = ("base", "fixed_plus", "fixed_minus", "soft_router")
TARGET_ALPHA = 1.0 / 2.0


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gradient_rows(held: str):
    rows = {}
    for index, family in enumerate(FAMILIES):
        if family == held:
            continue
        rows[family] = {
            f"{family}:0": (
                1.0 + 0.03 * index,
                0.4 - 0.01 * index,
                -0.2 + 0.02 * index,
                0.1 + 0.01 * index,
            ),
            f"{family}:1": (
                0.8 + 0.02 * index,
                0.3 + 0.01 * index,
                -0.1 - 0.01 * index,
                0.25 - 0.01 * index,
            ),
        }
    return rows


def _all_gradient_rows():
    return {
        family: rows
        for family in FAMILIES
        for rows in [_gradient_rows(next(item for item in FAMILIES if item != family))[family]]
    }


def _direction(held: str):
    return build_soft_polarity_direction_receipt(
        source_artifact_sha256s={
            "endpoint_receipt": _sha(f"endpoint-receipt:{held}"),
            "endpoint_evidence": _sha(f"endpoint-evidence:{held}"),
            "gradient_evidence": _sha(f"gradient-evidence:{held}"),
        },
        all_development_family_ids=FAMILIES,
        held_family_id=held,
        gradient_rows_by_family=_gradient_rows(held),
        gradient_evidence_sha256=_sha(f"gradient-evidence:{held}"),
    )


def _training_objectives(direction, alpha: float):
    return {
        family: {
            f"{family}:score:0": 1.0
            + (alpha - TARGET_ALPHA) ** 2
            + 0.001 * index,
            f"{family}:score:1": 1.01
            + (alpha - TARGET_ALPHA) ** 2
            + 0.001 * index,
        }
        for index, family in enumerate(direction["training_family_ids"])
    }


def _candidates(direction):
    return tuple(
        build_soft_polarity_candidate_receipt(
            direction_receipt=direction,
            alpha=alpha,
            exact_train_objectives_by_family=_training_objectives(
                direction, alpha
            ),
            execution_receipt_sha256=_sha(
                f"candidate:{direction['held_family_id']}:{alpha}"
            ),
            exact_execution=True,
        )
        for alpha in SOFT_POLARITY_FIT_ALPHAS
    )


def _held_objectives(held: str, *, outcome: str = "pass"):
    index = FAMILIES.index(held)
    base = 1.0 + 0.01 * index
    fixed_plus = base - 0.02
    fixed_minus = base + 0.03
    if outcome == "pass":
        soft = base - 0.04
    elif outcome == "lose_fixed":
        soft = base - 0.01
    elif outcome == "lose_base":
        soft = base + 0.01
    else:
        raise ValueError(outcome)
    return {
        arm: {
            f"{held}:held:0": value,
            f"{held}:held:1": value + 0.002,
        }
        for arm, value in {
            "base": base,
            "fixed_plus": fixed_plus,
            "fixed_minus": fixed_minus,
            "soft_router": soft,
        }.items()
    }


def _fold(held: str, *, outcome: str = "pass", healthy: bool = True):
    direction = _direction(held)
    candidates = _candidates(direction)
    fold = build_soft_polarity_fold_receipt(
        direction_receipt=direction,
        candidate_receipts=candidates,
        held_objectives_by_arm=_held_objectives(held, outcome=outcome),
        held_execution_receipt_sha256s_by_arm={
            arm: _sha(f"held-execution:{held}:{arm}") for arm in ARMS
        },
        held_trace_evidence_sha256=_sha(f"held-trace:{held}"),
        response_gain_min=-0.4 if healthy else 0.0,
        response_gain_max=0.5 if healthy else 0.0,
        response_gain_distinct_count=8 if healthy else 1,
        finite=True,
        pointwise_trust_passed=True,
        exact_execution=True,
    )
    return direction, candidates, fold


def test_direction_is_family_equal_damped_natural_descent() -> None:
    direction = _direction(FAMILIES[0])

    validate_soft_polarity_direction_receipt(direction)
    assert direction["protocol_sha256"] == SOFT_POLARITY_FIT_PROTOCOL_SHA256
    assert len(direction["training_family_ids"]) == 7
    assert direction["held_family_id"] not in direction["training_family_ids"]
    assert len(direction["natural_direction"]) == 4
    assert direction["damping"] > 0.0
    assert direction["directional_derivative"] < 0.0
    assert direction["strict_descent"] is True
    assert direction["normalization_scale"] > 0.0
    assert direction["normalized_box_logit_max_abs"] == pytest.approx(1.0)
    assert max(abs(item) for item in direction["normalized_corner_logits"]) == (
        pytest.approx(1.0)
    )
    assert direction["directional_derivative"] == pytest.approx(
        direction["raw_directional_derivative"] / direction["normalization_scale"]
    )
    assert direction["data_boundary"]["role"] == (
        "historically_reused_calibration_a_fit_A16_development_only"
    )
    assert direction["data_boundary"]["calibration_b_opened"] is False
    assert direction["data_boundary"]["test_opened"] is False


def test_direction_requires_exact_seven_family_boundary() -> None:
    held = FAMILIES[0]
    rows = _gradient_rows(held)
    rows.pop(FAMILIES[1])
    with pytest.raises(ValueError, match="seven training families"):
        build_soft_polarity_direction_receipt(
            source_artifact_sha256s={"source": _sha("source")},
            all_development_family_ids=FAMILIES,
            held_family_id=held,
            gradient_rows_by_family=rows,
            gradient_evidence_sha256=_sha("gradient-evidence"),
        )


def test_v20f_direction_receipt_is_not_a_v20g_trust_region_receipt() -> None:
    held = FAMILIES[0]
    v20f_direction = v20f_fit_core.build_soft_polarity_direction_receipt(
        source_artifact_sha256s={"source": _sha("source")},
        all_development_family_ids=FAMILIES,
        held_family_id=held,
        gradient_rows_by_family=_gradient_rows(held),
        gradient_evidence_sha256=_sha("gradient-evidence"),
    )

    with pytest.raises(ValueError, match="direction receipt content drifted"):
        validate_soft_polarity_direction_receipt(v20f_direction)


def test_candidate_binds_alpha_eta_and_exact_family_equal_objective() -> None:
    direction = _direction(FAMILIES[0])
    candidate = build_soft_polarity_candidate_receipt(
        direction_receipt=direction,
        alpha=TARGET_ALPHA,
        exact_train_objectives_by_family=_training_objectives(
            direction, TARGET_ALPHA
        ),
        execution_receipt_sha256=_sha("candidate"),
        exact_execution=True,
    )

    validate_soft_polarity_candidate_receipt(
        candidate, direction_receipt=direction
    )
    assert candidate["eta"] == pytest.approx(
        tuple(TARGET_ALPHA * value for value in direction["natural_direction"])
    )
    assert candidate["execution_changed_from_base"] is True
    assert candidate["box_logit_bound"] == TARGET_ALPHA
    assert candidate["box_logit_max_abs"] <= TARGET_ALPHA + 1.0e-15
    assert candidate["family_equal_train_objective"] == pytest.approx(1.008)
    with pytest.raises(ValueError, match="frozen ladder"):
        build_soft_polarity_candidate_receipt(
            direction_receipt=direction,
            alpha=0.03,
            exact_train_objectives_by_family=_training_objectives(
                direction, 0.03
            ),
            execution_receipt_sha256=_sha("off-ladder"),
            exact_execution=True,
        )


def test_full_fit_uses_all_eight_families_and_matches_runtime_eta_domain() -> None:
    direction = build_soft_polarity_direction_receipt(
        source_artifact_sha256s={"source": _sha("full-fit-source")},
        all_development_family_ids=FAMILIES,
        held_family_id=None,
        gradient_rows_by_family=_all_gradient_rows(),
        gradient_evidence_sha256=_sha("full-fit-gradient-evidence"),
    )
    candidate = build_soft_polarity_candidate_receipt(
        direction_receipt=direction,
        alpha=TARGET_ALPHA,
        exact_train_objectives_by_family=_training_objectives(
            direction, TARGET_ALPHA
        ),
        execution_receipt_sha256=_sha("full-fit-candidate"),
        exact_execution=True,
    )

    assert direction["held_family_id"] is None
    assert tuple(direction["training_family_ids"]) == FAMILIES
    assert len(candidate["family_mean_train_objectives"]) == 8
    assert SOFT_POLARITY_FIT_ETA_MAX_ABS == FISHER_SOFT_POLARITY_ETA_MAX_ABS
    certificate = fisher_soft_polarity_box_certificate(
        torch.tensor(candidate["eta"], dtype=torch.float64)
    )
    assert certificate["eta_max_abs"] == SOFT_POLARITY_FIT_ETA_MAX_ABS
    assert max(abs(value) for value in candidate["eta"]) <= (
        SOFT_POLARITY_FIT_ETA_MAX_ABS
    )
    assert certificate["gain_max_abs"] == 1.0


def test_fold_selection_is_training_only_and_health_is_trace_bound() -> None:
    direction, candidates, fold = _fold(FAMILIES[2])

    validate_soft_polarity_fold_receipt(
        fold,
        direction_receipt=direction,
        candidate_receipts=candidates,
    )
    assert fold["selected_alpha"] == TARGET_ALPHA
    assert fold["selection_frozen_before_held_scores"] is True
    assert set(fold["held_family_mean_objectives"]) == set(ARMS)
    assert fold["soft_response_health_passed"] is True

    _, _, unhealthy = _fold(FAMILIES[3], healthy=False)
    assert unhealthy["selected_alpha"] == TARGET_ALPHA
    assert unhealthy["response_nonconstant"] is False
    assert unhealthy["soft_response_health_passed"] is False


def test_fold_validator_rejects_duplicate_or_incomplete_candidate_ladder() -> None:
    direction, candidates, fold = _fold(FAMILIES[2])
    selected = next(
        candidate
        for candidate in candidates
        if candidate["artifact_sha256"]
        == fold["selected_candidate_artifact_sha256"]
    )

    with pytest.raises(ValueError, match="duplicate fold alpha"):
        validate_soft_polarity_fold_receipt(
            fold,
            direction_receipt=direction,
            candidate_receipts=(selected,) * len(SOFT_POLARITY_FIT_ALPHAS),
        )


def test_fold_rejects_candidate_scoring_geometry_that_changes_by_alpha() -> None:
    direction = _direction(FAMILIES[2])
    candidates = list(_candidates(direction))
    alpha = float(candidates[-1]["alpha"])
    changed = {
        family: {
            f"{example}:different-alpha": value
            for example, value in _training_objectives(direction, alpha)[family].items()
        }
        for family in direction["training_family_ids"]
    }
    candidates[-1] = build_soft_polarity_candidate_receipt(
        direction_receipt=direction,
        alpha=alpha,
        exact_train_objectives_by_family=changed,
        execution_receipt_sha256=_sha("different-candidate-geometry"),
        exact_execution=True,
    )

    with pytest.raises(ValueError, match="scoring geometry differs"):
        build_soft_polarity_fold_receipt(
            direction_receipt=direction,
            candidate_receipts=candidates,
            held_objectives_by_arm=_held_objectives(FAMILIES[2]),
            held_execution_receipt_sha256s_by_arm={
                arm: _sha(f"held-execution:{FAMILIES[2]}:{arm}")
                for arm in ARMS
            },
            held_trace_evidence_sha256=_sha("held-trace:different-geometry"),
            response_gain_min=-0.4,
            response_gain_max=0.5,
            response_gain_distinct_count=8,
            finite=True,
            pointwise_trust_passed=True,
            exact_execution=True,
        )


def test_oof_passes_only_when_router_beats_base_and_fixed_plus() -> None:
    bundles = [_fold(family) for family in FAMILIES]
    folds = tuple(bundle[2] for bundle in bundles)
    qualification = build_soft_polarity_oof_qualification(
        fold_receipts=folds
    )

    validate_soft_polarity_oof_qualification(
        qualification, fold_receipts=folds
    )
    assert qualification["soft_vs_base_family_win_count"] == 8
    assert qualification["soft_vs_fixed_plus_family_win_count"] == 8
    assert qualification["full_refit_authorized"] is True
    assert qualification["source_artifact_sha256s_by_family"] == {
        family: bundles[index][0]["source_artifact_sha256s"]
        for index, family in enumerate(FAMILIES)
    }
    assert "source_artifact_sha256s" not in qualification
    assert qualification["calibration_b_eligibility_gate_passed"] is True
    # Passing development gates authorizes the all-eight refit, but the
    # historically frozen Calibration-B role still needs a new policy claim.
    assert qualification["calibration_b_eligible"] is False
    assert qualification["calibration_b_opened"] is False
    assert qualification["fresh_family_disjoint_scoring_performed"] is False


def test_oof_rolls_back_when_fixed_plus_wins_or_health_fails() -> None:
    outcomes = ["pass"] * 6 + ["lose_fixed"] * 2
    folds = tuple(
        _fold(family, outcome=outcome)[2]
        for family, outcome in zip(FAMILIES, outcomes, strict=True)
    )
    qualification = build_soft_polarity_oof_qualification(
        fold_receipts=folds
    )
    assert qualification["soft_vs_base_family_win_count"] == 8
    assert qualification["soft_vs_fixed_plus_family_win_count"] == 6
    assert qualification["full_refit_authorized"] is True

    failing_outcomes = ["pass"] * 5 + ["lose_fixed"] * 3
    failing = build_soft_polarity_oof_qualification(
        fold_receipts=tuple(
            _fold(family, outcome=outcome)[2]
            for family, outcome in zip(
                FAMILIES, failing_outcomes, strict=True
            )
        )
    )
    assert failing["soft_vs_fixed_plus_family_win_count"] == 5
    assert failing["full_refit_authorized"] is False
    assert failing["calibration_b_eligibility_gate_passed"] is False
    assert failing["calibration_b_eligible"] is False
    assert failing["rollback_to_base"] is True

    unhealthy_folds = tuple(
        _fold(family, healthy=family != FAMILIES[-1])[2]
        for family in FAMILIES
    )
    unhealthy = build_soft_polarity_oof_qualification(
        fold_receipts=unhealthy_folds
    )
    assert unhealthy["gates"]["all_eight_fold_health_passed"] is False
    assert unhealthy["passed"] is False


def test_receipts_roundtrip_and_detect_tampering() -> None:
    direction, candidates, fold = _fold(FAMILIES[0])
    qualification_folds = tuple(_fold(family)[2] for family in FAMILIES)
    qualification = build_soft_polarity_oof_qualification(
        fold_receipts=qualification_folds
    )

    direction_json = json.loads(json.dumps(direction))
    candidate_json = json.loads(json.dumps(candidates[0]))
    fold_json = json.loads(json.dumps(fold))
    qualification_json = json.loads(json.dumps(qualification))
    validate_soft_polarity_direction_receipt(direction_json)
    validate_soft_polarity_candidate_receipt(
        candidate_json, direction_receipt=direction_json
    )
    validate_soft_polarity_fold_receipt(
        fold_json,
        direction_receipt=direction_json,
        candidate_receipts=tuple(json.loads(json.dumps(item)) for item in candidates),
    )
    validate_soft_polarity_oof_qualification(
        qualification_json,
        fold_receipts=tuple(
            json.loads(json.dumps(item)) for item in qualification_folds
        ),
    )

    bad_direction = copy.deepcopy(direction)
    bad_direction["damping"] *= 2.0
    with pytest.raises(ValueError):
        validate_soft_polarity_direction_receipt(bad_direction)
    bad_candidate = copy.deepcopy(candidates[0])
    bad_candidate["family_equal_train_objective"] += 1.0
    with pytest.raises(ValueError):
        validate_soft_polarity_candidate_receipt(
            bad_candidate, direction_receipt=direction
        )
    bad_fold = copy.deepcopy(fold)
    bad_fold["held_family_mean_objectives"]["soft_router"] += 1.0
    with pytest.raises(ValueError):
        validate_soft_polarity_fold_receipt(
            bad_fold,
            direction_receipt=direction,
            candidate_receipts=candidates,
        )
    bad_oof = copy.deepcopy(qualification)
    bad_oof["calibration_b_eligibility_gate_passed"] = False
    with pytest.raises(ValueError):
        validate_soft_polarity_oof_qualification(
            bad_oof, fold_receipts=qualification_folds
        )


def test_oof_rejects_stale_fold_hash_and_recomputed_lineage() -> None:
    folds = tuple(_fold(family)[2] for family in FAMILIES)

    stale = list(copy.deepcopy(folds))
    stale[0]["held_family_mean_objectives"]["soft_router"] += 0.5
    with pytest.raises(ValueError, match="fold receipt hash drifted"):
        build_soft_polarity_oof_qualification(fold_receipts=stale)

    qualification = build_soft_polarity_oof_qualification(
        fold_receipts=folds
    )
    spliced_folds = list(copy.deepcopy(folds))
    spliced_family = str(spliced_folds[-1]["held_family_id"])
    spliced_folds[-1]["source_artifact_sha256s"] = {
        name: _sha(f"spliced:{spliced_family}:{name}")
        for name in spliced_folds[-1]["source_artifact_sha256s"]
    }
    spliced_folds[-1]["artifact_sha256"] = fit_core._hash(
        fit_core._FOLD_DOMAIN,
        {
            key: item
            for key, item in spliced_folds[-1].items()
            if key != "artifact_sha256"
        },
    )
    with pytest.raises(
        ValueError, match="source_artifact_sha256s_by_family"
    ):
        validate_soft_polarity_oof_qualification(
            qualification, fold_receipts=spliced_folds
        )

    rebuilt = build_soft_polarity_oof_qualification(
        fold_receipts=spliced_folds
    )
    assert rebuilt["source_artifact_sha256s_by_family"][spliced_family] == (
        spliced_folds[-1]["source_artifact_sha256s"]
    )
    validate_soft_polarity_oof_qualification(
        rebuilt, fold_receipts=spliced_folds
    )

    forged_oof = copy.deepcopy(qualification)
    forged_oof["source_artifact_sha256s_by_family"][FAMILIES[0]][
        "endpoint_receipt"
    ] = _sha("forged-endpoint-receipt")
    forged_oof["artifact_sha256"] = fit_core._hash(
        fit_core._OOF_DOMAIN,
        {
            key: item
            for key, item in forged_oof.items()
            if key != "artifact_sha256"
        },
    )
    with pytest.raises(
        ValueError, match="source_artifact_sha256s_by_family"
    ):
        validate_soft_polarity_oof_qualification(
            forged_oof, fold_receipts=folds
        )


def test_work_accounting_has_zero_protected_role_access() -> None:
    bundles = [_fold(family) for family in FAMILIES]
    directions = tuple(bundle[0] for bundle in bundles)
    candidates = tuple(
        candidate for bundle in bundles for candidate in bundle[1]
    )
    folds = tuple(bundle[2] for bundle in bundles)
    work = soft_polarity_work_accounting(
        direction_receipts=directions,
        candidate_receipts=candidates,
        fold_receipts=folds,
        full_refit_performed=False,
    )

    assert work["outer_fold_count"] == 8
    assert work["alpha_count"] == len(SOFT_POLARITY_FIT_ALPHAS)
    assert work["trust_radius_count"] == len(SOFT_POLARITY_FIT_ALPHAS)
    assert work["box_direction_normalization_count"] == 8
    assert work["box_direction_corner_evaluation_count"] == 32
    assert work["candidate_trust_certificate_count"] == 80
    assert work["positive_candidate_trust_certificate_count"] == 72
    assert work["outer_fold_direction_solve_count"] == 8
    assert work["full_refit_direction_solve_count"] == 0
    assert work["unique_development_gradient_row_count"] == 16
    assert work["unique_development_candidate_objective_row_count"] == 16
    assert work["logical_fold_gradient_row_use_count"] == 112
    assert work["logical_full_refit_gradient_row_use_count"] == 0
    assert work["logical_total_gradient_row_use_count"] == 112
    assert work["family_opg_outer_product_evaluation_count"] == 112
    assert work["outer_fold_training_candidate_objective_score_count"] == 1120
    assert work["full_refit_training_candidate_objective_score_count"] == 0
    assert work["exact_training_candidate_objective_score_count"] == 1120
    assert work["calibration_b_example_access_count"] == 0
    assert work["validation_example_access_count"] == 0
    assert work["test_example_access_count"] == 0
    assert work["fresh_family_disjoint_score_count"] == 0


def test_work_accounting_includes_full_refit_rows_and_candidate_scores() -> None:
    bundles = [_fold(family) for family in FAMILIES]
    work = soft_polarity_work_accounting(
        direction_receipts=tuple(bundle[0] for bundle in bundles),
        candidate_receipts=tuple(
            candidate for bundle in bundles for candidate in bundle[1]
        ),
        fold_receipts=tuple(bundle[2] for bundle in bundles),
        full_refit_performed=True,
    )

    assert work["natural_direction_solve_count"] == 9
    assert work["box_direction_normalization_count"] == 9
    assert work["box_direction_corner_evaluation_count"] == 36
    assert work["candidate_trust_certificate_count"] == 90
    assert work["positive_candidate_trust_certificate_count"] == 81
    assert work["full_refit_direction_solve_count"] == 1
    assert work["logical_fold_gradient_row_use_count"] == 112
    assert work["logical_full_refit_gradient_row_use_count"] == 16
    assert work["logical_total_gradient_row_use_count"] == 128
    assert work["family_opg_outer_product_evaluation_count"] == 128
    assert work["outer_fold_training_candidate_objective_score_count"] == 1120
    assert work["full_refit_training_candidate_objective_score_count"] == 160
    assert work["exact_training_candidate_objective_score_count"] == 1280


def test_work_accounting_reauthenticates_the_complete_fold_campaign() -> None:
    bundles = [_fold(family) for family in FAMILIES]
    candidates = [
        candidate for bundle in bundles for candidate in bundle[1]
    ]
    candidates[-1] = candidates[-2]

    with pytest.raises(ValueError, match="duplicate fold alpha"):
        soft_polarity_work_accounting(
            direction_receipts=tuple(bundle[0] for bundle in bundles),
            candidate_receipts=candidates,
            fold_receipts=tuple(bundle[2] for bundle in bundles),
            full_refit_performed=True,
        )
