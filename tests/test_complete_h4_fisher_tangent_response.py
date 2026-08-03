from __future__ import annotations

import copy
import hashlib
import json
import random

import pytest

import fisher_graph.complete_h4_fisher_tangent_response as tangent_core

from fisher_graph.complete_h4_fisher_tangent_response import (
    TANGENT_RESPONSE_ARMS,
    TANGENT_RESPONSE_FRACTIONS,
    TANGENT_RESPONSE_INITIAL_WEIGHTS,
    bilinear_box_certificate,
    bilinear_corner_values,
    build_tangent_response_cv_candidate,
    build_tangent_response_cv_receipt,
    build_tangent_response_gradient_bank_receipt,
    build_tangent_response_direction_from_gradient_bank_receipt,
    build_tangent_response_direction_receipt,
    build_tangent_response_final_candidate_receipt,
    build_tangent_response_fit_receipt,
    build_tangent_response_held_arm_score,
    build_tangent_response_held_role_receipt,
    build_tangent_response_pair_qualification,
    build_tangent_response_ray_receipt,
    build_tangent_response_two_fit_bundle_receipt,
    tangent_response_fraction_proposal,
    tangent_response_gain,
    tangent_response_work_accounting,
    validate_tangent_response_cv_receipt,
    validate_tangent_response_gradient_bank_receipt,
    validate_tangent_response_direction_receipt,
    validate_tangent_response_fit_receipt,
    validate_tangent_response_pair_qualification,
    validate_tangent_response_two_fit_bundle_receipt,
)


FAMILIES = (
    "family_a",
    "family_b",
    "family_c",
    "family_d",
    "family_e",
    "family_f",
    "reed",
    "sundial",
)
EXCLUDED = ("reed", "sundial")
FIT_FAMILIES = FAMILIES[:6]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gradients(law: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    law_scale = 1.0 if law == "signed_log" else 1.3
    return {
        family: {
            f"{family}:0": (0.0, law_scale * (1.0 + 0.01 * index), 0.0),
            f"{family}:1": (0.0, law_scale * (1.1 + 0.01 * index), 0.0),
        }
        for index, family in enumerate(FIT_FAMILIES)
    }


def _direction(
    law: str,
    *,
    validation_family: str | None,
    gradients: dict[str, dict[str, tuple[float, float, float]]] | None = None,
):
    all_gradients = _gradients(law) if gradients is None else gradients
    return build_tangent_response_direction_receipt(
        source_artifact_sha256s={"v20c": _sha("source-v20c"), "v20d": _sha("source-v20d")},
        family_ids=FAMILIES,
        excluded_family_ids=EXCLUDED,
        validation_family_id=validation_family,
        fit_gradients_by_family=all_gradients,
        base_provider_artifact_sha256=_sha("shared-endpoint-base-provider"),
        proposal_provider_artifact_sha256=_sha("shared-endpoint-proposal-provider"),
        gradient_evidence_sha256=_sha(f"{law}:gradient-capability"),
        response_law=law,
    )


def _bank(law: str, *, gradients=None):
    return build_tangent_response_gradient_bank_receipt(
        source_artifact_sha256s={
            "v20c": _sha("source-v20c"),
            "v20d": _sha("source-v20d"),
        },
        family_ids=FAMILIES,
        excluded_family_ids=EXCLUDED,
        fit_gradients_by_family=_gradients(law) if gradients is None else gradients,
        base_provider_artifact_sha256=_sha("shared-endpoint-base-provider"),
        proposal_provider_artifact_sha256=_sha("shared-endpoint-proposal-provider"),
        response_law=law,
    )


def _cv_candidate(
    law: str,
    validation: str,
    direction,
    ray,
    fraction: float,
    objective: float,
):
    example_ids = (f"{validation}:0", f"{validation}:1")
    provider_label = (
        f"{law}:initial-provider"
        if fraction == 0.0
        else f"{law}:{validation}:{fraction}:provider"
    )
    return build_tangent_response_cv_candidate(
        direction_receipt=direction,
        ray_receipt=ray,
        fraction=fraction,
        provider_artifact_sha256=_sha(provider_label),
        validation_example_ids=example_ids,
        validation_objectives_by_example={example: objective for example in example_ids},
        validation_execution_receipt_sha256s_by_example={
            example: _sha(f"{law}:{validation}:{fraction}:{example}:execution")
            for example in example_ids
        },
        execution_evidence_sha256=_sha(f"{law}:{validation}:{fraction}:evidence"),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
        execution_exact=True,
        execution_changed_from_baseline=fraction > 0.0,
    )


def _cv(law: str, *, improved_folds: int = 6):
    directions = []
    rays = []
    candidates = []
    target = 1.0 / 32.0
    for fold_index, validation in enumerate(FIT_FAMILIES):
        direction = _direction(law, validation_family=validation)
        ray = build_tangent_response_ray_receipt(direction_receipt=direction)
        directions.append(direction)
        rays.append(ray)
        baseline = 1.0 + 0.01 * fold_index
        for fraction in TANGENT_RESPONSE_FRACTIONS:
            if fraction == 0.0:
                objective = baseline
            elif fraction == target:
                # With improved_folds=3, macro still improves: the three wins
                # are deliberately larger than the three losses.
                objective = (
                    baseline - 0.20
                    if fold_index < improved_folds
                    else baseline + 0.01
                )
            elif fraction == 1.0 / 64.0:
                objective = baseline - 0.04 if fold_index < improved_folds else baseline + 0.02
            else:
                objective = baseline + 0.05 + fraction
            candidates.append(
                _cv_candidate(law, validation, direction, ray, fraction, objective)
            )
    return build_tangent_response_cv_receipt(
        direction_receipts=directions,
        ray_receipts=rays,
        candidates=candidates,
    )


def _fit(law: str):
    cv = _cv(law)
    direction = _direction(law, validation_family=None)
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    ids = {
        family: (f"{family}:0", f"{family}:1")
        for family in FIT_FAMILIES
    }
    traces = {
        family: {
            example: _sha(f"{law}:{family}:{example}:provider-trace")
            for example in examples
        }
        for family, examples in ids.items()
    }
    final = build_tangent_response_final_candidate_receipt(
        cv_receipt=cv,
        direction_receipt=direction,
        ray_receipt=ray,
        selected_provider_artifact_sha256=_sha(f"{law}:selected-provider"),
        fit_support_example_ids_by_family=ids,
        fit_support_provider_trace_receipt_sha256s_by_family=traces,
        fit_support_gain_trace_sha256s_by_family={
            family: _sha(f"{law}:{family}:gain-trace") for family in FIT_FAMILIES
        },
        fit_support_gain_min_by_family={family: -0.5 + 0.01 * index for index, family in enumerate(FIT_FAMILIES)},
        fit_support_gain_max_by_family={family: 0.5 + 0.01 * index for index, family in enumerate(FIT_FAMILIES)},
        fit_support_gain_distinct_count_by_family={family: 2 for family in FIT_FAMILIES},
        provider_trace_evidence_sha256=_sha(f"{law}:provider-trace-evidence"),
        provider_trace_finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
        provider_trace_exact=True,
        provider_trace_changed_from_initial=True,
    )
    return build_tangent_response_fit_receipt(
        cv_receipt=cv,
        final_direction_receipt=direction,
        final_ray_receipt=ray,
        final_candidate_receipt=final,
    )


def _bundle():
    return build_tangent_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=_fit("signed_log"),
        linear_fit_receipt=_fit("linear"),
    )


def _roles(bundle):
    objective = {
        "base": 2.0,
        "constant_plus_one": 1.8,
        "fixed_signed_log": 1.7,
        "fixed_linear": 1.75,
        "learned_signed_log": 1.6,
        "learned_linear": 1.68,
        "learned_signed_log_sign_flip": 1.9,
    }
    semantics = {
        "base": ("base", 0),
        "constant_plus_one": ("constant", 1),
        "fixed_signed_log": ("signed_log", 1),
        "fixed_linear": ("linear", 1),
        "learned_signed_log": ("signed_log", 1),
        "learned_linear": ("linear", 1),
        "learned_signed_log_sign_flip": ("signed_log", -1),
    }
    learned = bundle["selected_provider_artifact_sha256s_by_law"]
    roles = []
    for outer, held in (("reed", "sundial"), ("sundial", "reed")):
        scores = []
        for arm in TANGENT_RESPONSE_ARMS:
            law, polarity = semantics[arm]
            provider = (
                learned["signed_log"]
                if arm == "learned_signed_log"
                else learned["linear"]
                if arm == "learned_linear"
                else _sha(f"{outer}:{held}:{arm}:provider")
            )
            weight_hash = (
                bundle["selected_weight_sha256s_by_law"]["signed_log"]
                if arm in ("learned_signed_log", "learned_signed_log_sign_flip")
                else bundle["selected_weight_sha256s_by_law"]["linear"]
                if arm == "learned_linear"
                else _sha(f"{outer}:{held}:{arm}:weights")
            )
            scores.append(
                build_tangent_response_held_arm_score(
                    fit_bundle_receipt=bundle,
                    outer_held_family_id=outer,
                    held_family_id=held,
                    arm=arm,
                    response_law=law,
                    response_polarity=polarity,
                    response_weight_sha256=weight_hash,
                    objective=objective[arm],
                    provider_artifact_sha256=provider,
                    execution_receipt_sha256=_sha(f"{outer}:{held}:{arm}:execution"),
                    finite=True,
                    pointwise_trust_passed=True,
                    rank_is_16=True,
                    execution_changed_from_base=arm != "base",
                    response_nonconstant=arm.startswith("learned_"),
                )
            )
        roles.append(
            build_tangent_response_held_role_receipt(
                fit_bundle_receipt=bundle, arm_scores=scores
            )
        )
    return tuple(roles)


def test_bilinear_box_certificate_is_exact_l1_norm_and_gain_is_bounded():
    for weights in ((0.2, -0.3, 0.1), (-0.1, 0.7, -0.2), (0.0, 1.0, 0.0)):
        assert bilinear_box_certificate(weights) == pytest.approx(sum(abs(item) for item in weights))
        assert max(abs(item) for item in bilinear_corner_values(weights)) == bilinear_box_certificate(weights)
    assert tangent_response_gain((0.0, 1.0, 0.0), 0.4, -0.7, law="linear") == pytest.approx(-0.7)
    assert -1.0 <= tangent_response_gain((0.0, 1.0, 0.0), 1.0, -1.0) <= 1.0


def test_family_equal_gradient_and_opg_average_committed_family_summaries():
    gradients = {
        family: {
            f"{family}:0": (0.0, 0.0 if family == "family_a" else 1.0, 0.0),
            f"{family}:1": (0.0, 0.0 if family == "family_a" else 1.0, 0.0),
        }
        for family in FIT_FAMILIES
    }
    receipt = _direction("signed_log", validation_family=None, gradients=gradients)
    assert receipt["gradient_mean"] == pytest.approx((0.0, 5.0 / 6.0, 0.0))
    assert receipt["empirical_fisher"][1][1] == pytest.approx(5.0 / 6.0)
    assert receipt["fisher_semantics"] == "family_equal_response_parameter_empirical_gradient_Fisher_OPG"


def test_gradient_bank_roundtrip_reuses_exactly_twelve_outer_products(monkeypatch):
    calls = 0
    original_outer = tangent_core._outer

    def counted_outer(vector):
        nonlocal calls
        calls += 1
        return original_outer(vector)

    monkeypatch.setattr(tangent_core, "_outer", counted_outer)
    bank = _bank("signed_log")
    assert bank["unique_gradient_row_count"] == 12
    assert bank["empirical_fisher_outer_product_evaluation_count"] == 12
    assert calls == 12
    restored = json.loads(json.dumps(bank))
    assert validate_tangent_response_gradient_bank_receipt(restored) == bank
    for validation in (*FIT_FAMILIES, None):
        from_bank = build_tangent_response_direction_from_gradient_bank_receipt(
            gradient_bank_receipt=bank,
            gradient_evidence_sha256=_sha("signed_log:gradient-capability"),
            validation_family_id=validation,
        )
        direct = _direction("signed_log", validation_family=validation)
        assert from_bank == direct
    assert calls == 12 + 7 * 12
    # The seven bank-based directions themselves performed no new outers; the
    # additional calls above came only from the compatibility direct builder.


def test_gradient_bank_requires_two_rows_and_rejects_indefinite_summary():
    one_row = _gradients("linear")
    one_row["family_a"] = {"family_a:0": (0.0, 1.0, 0.0)}
    with pytest.raises(ValueError, match="exactly two"):
        _bank("linear", gradients=one_row)

    tampered = copy.deepcopy(_bank("linear"))
    summary = tampered["family_gradient_summaries_by_family"]["family_a"]
    summary["empirical_fisher"] = (
        (1.0, 2.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_tangent_response_gradient_bank_receipt(tampered)


def test_tangent_qp_has_exact_constraints_kkt_and_comparable_ray():
    direction = _direction("signed_log", validation_family=None)
    assert direction["strict_descent_direction"] is True
    assert max(direction["tangent_inequality_values"]) <= 0.0
    assert direction["kkt_certificate_passed"] is True
    assert direction["kkt_stationarity_max_abs"] <= direction["solver_feasibility_tolerance"]
    assert direction["kkt_active_equality_max_abs"] <= direction["solver_feasibility_tolerance"]
    assert direction["kkt_primal_violation_max"] <= direction["solver_feasibility_tolerance"]
    assert direction["kkt_dual_violation_max"] <= direction["solver_feasibility_tolerance"]
    assert abs(direction["gradient_curvature_identity_residual"]) <= direction["solver_feasibility_tolerance"]
    assert direction["inward_roundoff_correction"] <= direction["solver_feasibility_tolerance"]
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    assert ray["radial_projection_used"] is False
    assert ray["analytical_ray_max_step"] == pytest.approx(2.0 / direction["direction_l1_norm"])
    assert ray["endpoint_displacement_l1"] == pytest.approx(2.0)
    assert ray["endpoint_box_certificate"] <= 1.0
    assert ray["ray_comparability_invariant_passed"] is True


def test_random_empirical_fisher_directions_survive_sub_ulp_inward_repairs():
    for seed in range(64):
        rng = random.Random(seed)
        gradients = {
            family: {
                f"{family}:{example}": tuple(
                    rng.uniform(-2.0, 2.0) for _ in range(3)
                )
                for example in range(2)
            }
            for family in FIT_FAMILIES
        }
        direction = _direction(
            "signed_log", validation_family=None, gradients=gradients
        )
        assert max(direction["tangent_inequality_values"]) <= 0.0
        assert direction["inward_roundoff_correction"] <= direction["solver_feasibility_tolerance"]
        if direction["strict_descent_direction"]:
            ray = build_tangent_response_ray_receipt(direction_receipt=direction)
            assert ray["endpoint_box_certificate"] <= 1.0
            assert ray["endpoint_displacement_l1"] == pytest.approx(2.0, abs=1e-12)


def test_tiny_fraction_is_provider_ready_and_has_fixed_l1_displacement():
    direction = _direction("linear", validation_family="family_a")
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    proposal = tangent_response_fraction_proposal(
        direction_receipt=direction, ray_receipt=ray, fraction=1.0 / 256.0
    )
    assert proposal["displacement_l1"] == pytest.approx(1.0 / 128.0)
    assert proposal["expected_displacement_l1"] == pytest.approx(1.0 / 128.0)
    assert proposal["box_certificate"] <= 1.0
    assert proposal["radial_projection_used"] is False
    assert proposal["comparability_invariant_passed"] is True


def test_cv_selects_oof_fraction_and_json_roundtrips():
    cv = _cv("signed_log")
    assert cv["selected_fraction"] == 1.0 / 32.0
    assert cv["selected_fold_improvement_count"] == 6
    assert cv["macro_improved"] is True
    assert cv["cv_selection_authorized"] is True
    assert cv["held_score_authorized"] is False
    assert cv["positive_provider_hashes_unique_across_fold_fraction"] is True
    assert cv["zero_fraction_provider_reused_across_folds"] is True
    restored = json.loads(json.dumps(cv))
    assert validate_tangent_response_cv_receipt(restored) == cv


def test_cv_rejects_macro_improvement_when_only_three_folds_improve():
    cv = _cv("signed_log", improved_folds=3)
    assert cv["macro_objectives_by_fraction"][repr(1.0 / 32.0)] < cv["baseline_macro_objective"]
    assert cv["fold_improvement_counts_by_fraction"][repr(1.0 / 32.0)] == 3
    assert cv["selected_fraction"] == 0.0
    assert cv["cv_selection_authorized"] is False
    assert cv["rollback_to_zero_fraction"] is True


def test_cv_rejects_fold_gradient_bank_drift_and_validation_row_drift():
    cv = _cv("signed_log")
    directions = list(cv["direction_receipts_by_validation_family"].values())
    rays = list(cv["ray_receipts_by_validation_family"].values())
    candidates = [
        item
        for family_items in cv["candidate_receipts_by_validation_family"].values()
        for item in family_items
    ]
    drift_gradients = _gradients("signed_log")
    drift_gradients["family_a"] = {
        "family_a:0": (0.1, 1.0, 0.0),
        "family_a:1": (0.1, 1.1, 0.0),
    }
    drift_direction = _direction(
        "signed_log", validation_family="family_b", gradients=drift_gradients
    )
    directions = [
        drift_direction
        if item["validation_family_id"] == "family_b"
        else item
        for item in directions
    ]
    with pytest.raises(
        ValueError,
        match="shared gradient bank|gradient_bank_artifact_sha256 lineage",
    ):
        build_tangent_response_cv_receipt(
            direction_receipts=directions, ray_receipts=rays, candidates=candidates
        )

    original_direction = cv["direction_receipts_by_validation_family"]["family_a"]
    original_ray = cv["ray_receipts_by_validation_family"]["family_a"]
    row_drift_candidate = build_tangent_response_cv_candidate(
        direction_receipt=original_direction,
        ray_receipt=original_ray,
        fraction=1.0 / 256.0,
        provider_artifact_sha256=_sha("signed_log:family_a:row-drift:provider"),
        validation_example_ids=("family_a:other0", "family_a:other1"),
        validation_objectives_by_example={"family_a:other0": 1.0, "family_a:other1": 1.0},
        validation_execution_receipt_sha256s_by_example={
            "family_a:other0": _sha("row-drift:execution:0"),
            "family_a:other1": _sha("row-drift:execution:1"),
        },
        execution_evidence_sha256=_sha("row-drift:evidence"),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
        execution_exact=True,
        execution_changed_from_baseline=True,
    )
    original_candidates = list(candidates)
    original_candidates = [
        row_drift_candidate
        if item["validation_family_id"] == "family_a"
        and item["fraction"] == 1.0 / 256.0
        else item
        for item in original_candidates
    ]
    with pytest.raises(ValueError, match="shared gradient bank"):
        build_tangent_response_cv_receipt(
            direction_receipts=list(cv["direction_receipts_by_validation_family"].values()),
            ray_receipts=rays,
            candidates=original_candidates,
        )


def test_final_candidate_rejects_a_different_all_six_bank():
    law = "linear"
    cv = _cv(law)
    gradients = _gradients(law)
    gradients["family_c"] = {
        "family_c:0": (0.2, 1.0, 0.0),
        "family_c:1": (0.2, 1.1, 0.0),
    }
    direction = _direction(law, validation_family=None, gradients=gradients)
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    ids = {family: (f"{family}:0", f"{family}:1") for family in FIT_FAMILIES}
    with pytest.raises(
        ValueError,
        match="canonical CV gradient bank|gradient_bank_artifact_sha256 lineage",
    ):
        build_tangent_response_final_candidate_receipt(
            cv_receipt=cv,
            direction_receipt=direction,
            ray_receipt=ray,
            selected_provider_artifact_sha256=_sha("linear:different-bank:provider"),
            fit_support_example_ids_by_family=ids,
            fit_support_provider_trace_receipt_sha256s_by_family={
                family: {
                    example: _sha(f"different:{family}:{example}:trace")
                    for example in examples
                }
                for family, examples in ids.items()
            },
            fit_support_gain_trace_sha256s_by_family={family: _sha(f"different:{family}:gains") for family in FIT_FAMILIES},
            fit_support_gain_min_by_family={family: -0.5 for family in FIT_FAMILIES},
            fit_support_gain_max_by_family={family: 0.5 for family in FIT_FAMILIES},
            fit_support_gain_distinct_count_by_family={family: 2 for family in FIT_FAMILIES},
            provider_trace_evidence_sha256=_sha("different-bank:evidence"),
            provider_trace_finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
            provider_trace_exact=True,
            provider_trace_changed_from_initial=True,
        )


def test_final_candidate_rejects_fit_support_ids_outside_gradient_bank():
    fit = _fit("signed_log")
    cv = fit["cv_receipt"]
    direction = fit["final_direction_receipt"]
    ray = fit["final_ray_receipt"]
    original = fit["final_candidate_receipt"]
    ids = copy.deepcopy(original["fit_support_example_ids_by_family"])
    ids["family_a"] = ("family_a:other0", "family_a:other1")
    traces = copy.deepcopy(original["fit_support_provider_trace_receipt_sha256s_by_family"])
    traces["family_a"] = {
        example: _sha(f"outside-bank:{example}:trace") for example in ids["family_a"]
    }
    with pytest.raises(ValueError, match="fit-support IDs"):
        build_tangent_response_final_candidate_receipt(
            cv_receipt=cv,
            direction_receipt=direction,
            ray_receipt=ray,
            selected_provider_artifact_sha256=original["selected_provider_artifact_sha256"],
            fit_support_example_ids_by_family=ids,
            fit_support_provider_trace_receipt_sha256s_by_family=traces,
            fit_support_gain_trace_sha256s_by_family=original["fit_support_gain_trace_sha256s_by_family"],
            fit_support_gain_min_by_family=original["fit_support_gain_min_by_family"],
            fit_support_gain_max_by_family=original["fit_support_gain_max_by_family"],
            fit_support_gain_distinct_count_by_family=original["fit_support_gain_distinct_count_by_family"],
            provider_trace_evidence_sha256=original["provider_trace_evidence_sha256"],
            provider_trace_finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
            provider_trace_exact=True,
            provider_trace_changed_from_initial=True,
        )


def test_final_fit_is_structural_without_post_cv_all_six_rescore():
    fit = _fit("signed_log")
    candidate = fit["final_candidate_receipt"]
    assert fit["selected_fraction"] == 1.0 / 32.0
    assert candidate["final_exact_fit_objectives_used"] is False
    assert fit["post_cv_all_six_exact_rescore_performed"] is False
    assert candidate["response_nonconstant_on_fit_support"] is True
    assert candidate["radial_projection_used"] is False
    assert candidate["final_candidate_authorized"] is True
    assert fit["held_score_authorized"] is True
    restored = json.loads(json.dumps(fit))
    assert validate_tangent_response_fit_receipt(restored) == fit


def test_oof_no_selection_produces_valid_fit_rollback_without_held_capability():
    law = "signed_log"
    cv = _cv(law, improved_folds=3)
    direction = _direction(law, validation_family=None)
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    ids = {family: (f"{family}:0", f"{family}:1") for family in FIT_FAMILIES}
    final = build_tangent_response_final_candidate_receipt(
        cv_receipt=cv,
        direction_receipt=direction,
        ray_receipt=ray,
        selected_provider_artifact_sha256=_sha(f"{law}:initial-provider"),
        fit_support_example_ids_by_family=ids,
        fit_support_provider_trace_receipt_sha256s_by_family={
            family: {
                example: _sha(f"rollback:{law}:{family}:{example}:trace")
                for example in examples
            }
            for family, examples in ids.items()
        },
        fit_support_gain_trace_sha256s_by_family={
            family: _sha(f"rollback:{law}:{family}:gains") for family in FIT_FAMILIES
        },
        fit_support_gain_min_by_family={family: -1.0 for family in FIT_FAMILIES},
        fit_support_gain_max_by_family={family: 1.0 for family in FIT_FAMILIES},
        fit_support_gain_distinct_count_by_family={family: 2 for family in FIT_FAMILIES},
        provider_trace_evidence_sha256=_sha("rollback:provider-trace-evidence"),
        provider_trace_finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
        provider_trace_exact=True,
        provider_trace_changed_from_initial=False,
    )
    fit = build_tangent_response_fit_receipt(
        cv_receipt=cv,
        final_direction_receipt=direction,
        final_ray_receipt=ray,
        final_candidate_receipt=final,
    )
    assert fit["selected_fraction"] == 0.0
    assert fit["learned_candidate_authorized"] is False
    assert fit["rollback_to_initial_weights"] is True
    assert fit["held_score_authorized"] is False
    assert fit["selected_provider_artifact_sha256"] is None


def test_two_law_bundle_binds_laws_and_rejects_swap_or_tamper():
    bundle = _bundle()
    assert bundle["both_cv_selections_authorized"] is True
    assert bundle["both_final_fit_support_responses_nonconstant"] is True
    assert bundle["held_score_authorized"] is True
    assert validate_tangent_response_two_fit_bundle_receipt(json.loads(json.dumps(bundle))) == bundle
    fits = bundle["fit_receipts_by_law"]
    with pytest.raises(ValueError, match="wrong response law"):
        build_tangent_response_two_fit_bundle_receipt(
            signed_log_fit_receipt=fits["linear"],
            linear_fit_receipt=fits["signed_log"],
        )
    tampered = copy.deepcopy(bundle)
    tampered["selected_fractions_by_law"]["signed_log"] = 1.0
    with pytest.raises(ValueError, match="drifted"):
        validate_tangent_response_two_fit_bundle_receipt(tampered)


def test_direction_hash_roundtrip_and_statistic_tamper_rejection():
    direction = _direction("linear", validation_family="family_b")
    restored = json.loads(json.dumps(direction))
    assert validate_tangent_response_direction_receipt(restored) == direction
    tampered = copy.deepcopy(restored)
    tampered["gradient_mean"][1] += 0.1
    with pytest.raises(ValueError, match="drifted"):
        validate_tangent_response_direction_receipt(tampered)


def test_degenerate_zero_gradient_direction_is_a_clean_rollback_geometry():
    zero = {
        family: {
            f"{family}:0": (0.0, 0.0, 0.0),
            f"{family}:1": (0.0, 0.0, 0.0),
        }
        for family in FIT_FAMILIES
    }
    direction = _direction("signed_log", validation_family=None, gradients=zero)
    assert direction["tangent_direction"] == (0.0, 0.0, 0.0)
    assert direction["strict_descent_direction"] is False
    ray = build_tangent_response_ray_receipt(direction_receipt=direction)
    assert ray["direction_degenerate"] is True
    assert ray["feasible_ray_max_step"] == 0.0
    assert ray["endpoint_weights"] == TANGENT_RESPONSE_INITIAL_WEIGHTS
    assert ray["endpoint_displacement_l1"] == 0.0
    assert ray["endpoint_displacement_l1_error_from_two"] == -2.0
    assert ray["ray_comparability_invariant_applicable"] is False
    assert ray["ray_comparability_invariant_passed"] is False


def test_positive_unchanged_cv_observations_persist_and_force_clean_rollback():
    cv = _cv("linear")
    directions = list(cv["direction_receipts_by_validation_family"].values())
    rays = list(cv["ray_receipts_by_validation_family"].values())
    rebuilt_candidates = []
    for family in FIT_FAMILIES:
        direction = cv["direction_receipts_by_validation_family"][family]
        ray = cv["ray_receipts_by_validation_family"][family]
        for candidate in cv["candidate_receipts_by_validation_family"][family]:
            rebuilt_candidates.append(
                build_tangent_response_cv_candidate(
                    direction_receipt=direction,
                    ray_receipt=ray,
                    fraction=candidate["fraction"],
                    provider_artifact_sha256=candidate["provider_artifact_sha256"],
                    validation_example_ids=candidate["validation_example_ids"],
                    validation_objectives_by_example=candidate["validation_objectives_by_example"],
                    validation_execution_receipt_sha256s_by_example=candidate["validation_execution_receipt_sha256s_by_example"],
                    execution_evidence_sha256=candidate["execution_evidence_sha256"],
                    finite=True,
                    pointwise_trust_passed=True,
                    rank_is_16=True,
                    execution_exact=True,
                    execution_changed_from_baseline=False,
                )
            )
    rebuilt = build_tangent_response_cv_receipt(
        direction_receipts=directions, ray_receipts=rays, candidates=rebuilt_candidates
    )
    assert rebuilt["selected_fraction"] == 0.0
    assert rebuilt["cv_selection_authorized"] is False
    assert rebuilt["rollback_to_zero_fraction"] is True


def test_held_qualification_preserves_v20d_scientific_gates():
    bundle = _bundle()
    roles = _roles(bundle)
    qualification = build_tangent_response_pair_qualification(
        fit_bundle_receipt=bundle, roles=roles
    )
    assert qualification["passed"] is True
    assert qualification["learned_signed_log_improves_both_roles"] is True
    assert qualification["learned_signed_log_beats_learned_linear_macro"] is True
    assert qualification["learned_signed_log_beats_mirror_both_roles"] is True
    assert validate_tangent_response_pair_qualification(
        json.loads(json.dumps(qualification)),
        fit_bundle_receipt=bundle,
        roles=roles,
    ) == qualification


def test_held_constant_learned_response_is_recorded_as_failed_qualification():
    bundle = _bundle()
    roles = list(_roles(bundle))
    first_role = roles[0]
    original_scores = list(first_role["arm_scores"])
    learned = next(item for item in original_scores if item["arm"] == "learned_signed_log")
    constant_score = build_tangent_response_held_arm_score(
        fit_bundle_receipt=bundle,
        outer_held_family_id=learned["outer_held_family_id"],
        held_family_id=learned["held_family_id"],
        arm=learned["arm"],
        response_law=learned["response_law"],
        response_polarity=learned["response_polarity"],
        response_weight_sha256=learned["response_weight_sha256"],
        objective=learned["objective"],
        provider_artifact_sha256=learned["provider_artifact_sha256"],
        execution_receipt_sha256=learned["execution_receipt_sha256"],
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
        execution_changed_from_base=True,
        response_nonconstant=False,
    )
    roles[0] = build_tangent_response_held_role_receipt(
        fit_bundle_receipt=bundle,
        arm_scores=[constant_score if item["arm"] == "learned_signed_log" else item for item in original_scores],
    )
    qualification = build_tangent_response_pair_qualification(
        fit_bundle_receipt=bundle, roles=roles
    )
    assert qualification["learned_signed_log_nonconstant_both_roles"] is False
    assert qualification["passed"] is False


def test_work_accounting_matches_frozen_cv_terminal_and_full_paths():
    common = dict(
        collection_forward_count=32,
        collection_backward_count=16,
        endpoint_forward_count=12,
        endpoint_backward_count=12,
        endpoint_local_contraction_count=12,
        endpoint_teacher_access_count=12,
        law_count=2,
        fit_prompt_count=12,
        cv_fold_count=6,
        validation_prompts_per_fold=2,
        fraction_count=10,
        fraction_zero_vjp_reused=True,
        held_role_count=2,
        held_arm_count=7,
        held_prompts_per_role=2,
    )
    terminal = tangent_response_work_accounting(**common, held_scoring_executed=False)
    assert terminal["total_forward_count"] == 284
    assert terminal["total_backward_count"] == 52
    assert terminal["total_local_contraction_count"] == 36
    assert terminal["total_backward_or_local_gradient_call_count"] == 88
    assert terminal["teacher_access_count"] == 252
    assert terminal["unique_empirical_fisher_gradient_row_count"] == 24
    assert terminal["empirical_fisher_outer_product_evaluation_count"] == 24
    assert "empirical_fisher_outer_product_count" not in terminal
    full = tangent_response_work_accounting(**common, held_scoring_executed=True)
    assert full["total_forward_count"] == 312
    assert full["total_backward_count"] == 52
    assert full["total_local_contraction_count"] == 36
    assert full["teacher_access_count"] == 280
    assert full["held_score_count"] == 14
    assert full["unique_empirical_fisher_gradient_row_count"] == 24
    assert full["empirical_fisher_outer_product_evaluation_count"] == 24
    with pytest.raises(ValueError, match="beta-zero VJP reuse"):
        tangent_response_work_accounting(
            **{**common, "fraction_zero_vjp_reused": False},
            held_scoring_executed=False,
        )
    with pytest.raises(ValueError, match="phase allocation"):
        tangent_response_work_accounting(
            **{**common, "collection_forward_count": 31},
            held_scoring_executed=False,
        )


def test_receipts_are_json_scalar_and_hash_only():
    receipt = _bundle()
    encoded = json.dumps(receipt, allow_nan=False)
    assert '"raw_gradients":' not in encoded
    assert '"gradient":' not in encoded
    assert '"logits":' not in encoded
    assert '"targets":' not in encoded
