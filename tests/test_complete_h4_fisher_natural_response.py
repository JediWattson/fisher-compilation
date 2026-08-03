from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from fisher_graph.complete_h4_fisher_natural_response import (
    NATURAL_RESPONSE_ALPHAS,
    NATURAL_RESPONSE_ARMS,
    NATURAL_RESPONSE_INITIAL_WEIGHTS,
    bilinear_box_certificate,
    bilinear_corner_values,
    build_natural_response_alpha_candidate,
    build_natural_response_direction_receipt,
    build_natural_response_fit_receipt,
    build_natural_response_held_arm_score,
    build_natural_response_held_role_receipt,
    build_natural_response_pair_qualification,
    build_natural_response_two_fit_bundle_receipt,
    natural_response_gain,
    natural_response_work_accounting,
    radially_project_bilinear_weights,
    signed_log_response,
    validate_natural_response_alpha_candidate,
    validate_natural_response_direction_receipt,
    validate_natural_response_fit_receipt,
    validate_natural_response_held_role_receipt,
    validate_natural_response_pair_qualification,
    validate_natural_response_two_fit_bundle_receipt,
)


FAMILIES = tuple(f"family_{index}" for index in range(8))
EXCLUDED = FAMILIES[:2]
FIT_FAMILIES = FAMILIES[2:]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _standard_gradients() -> dict[str, dict[str, tuple[float, float, float]]]:
    result = {}
    for index, family in enumerate(FIT_FAMILIES):
        result[family] = {
            "example_a": (0.4 + 0.05 * index, 0.7 + 0.03 * index, 0.2 - 0.02 * index),
            "example_b": (0.5 + 0.04 * index, 0.6 + 0.02 * index, 0.1 + 0.01 * index),
        }
    return result


def _direction(
    law: str,
    *,
    gradients: dict[str, dict[str, tuple[float, float, float]]] | None = None,
    source_label: str = "shared-source",
) -> dict[str, object]:
    return build_natural_response_direction_receipt(
        v20c_source_sha256s={"complete_h4": _sha(source_label)},
        family_ids=FAMILIES,
        excluded_family_ids=EXCLUDED,
        fit_gradients_by_family=gradients or _standard_gradients(),
        base_provider_artifact_sha256=_sha("base-provider"),
        proposal_provider_artifact_sha256=_sha("proposal-provider"),
        gradient_evidence_sha256=_sha(f"{law}-{source_label}-gradient-evidence"),
        response_law=law,
    )


def _candidate(
    direction: dict[str, object], alpha: float, objective: float
) -> dict[str, object]:
    law = str(direction["response_law"])
    ids = direction["fit_example_ids_by_family"]
    objectives = {
        family: {example: objective for example in ids[family]}
        for family in direction["fit_family_ids"]
    }
    executions = {
        family: {
            example: _sha(f"{law}-{alpha}-{family}-{example}-execution")
            for example in ids[family]
        }
        for family in direction["fit_family_ids"]
    }
    return build_natural_response_alpha_candidate(
        direction_receipt=direction,
        alpha=alpha,
        provider_artifact_sha256=_sha(f"{law}-{alpha}-provider"),
        exact_fit_objectives_by_family=objectives,
        fit_execution_receipt_sha256s_by_family=executions,
    )


def _fit(
    law: str,
    *,
    gradients: dict[str, dict[str, tuple[float, float, float]]] | None = None,
    source_label: str = "shared-source",
    objectives: dict[float, float] | None = None,
) -> dict[str, object]:
    direction = _direction(law, gradients=gradients, source_label=source_label)
    schedule = objectives or {
        0.0: 1.0,
        1.0 / 16.0: 0.95,
        1.0 / 8.0: 0.93,
        1.0 / 4.0: 0.90,
        1.0 / 2.0: 0.91,
        1.0: 0.92,
    }
    return build_natural_response_fit_receipt(
        direction_receipt=direction,
        candidates=tuple(_candidate(direction, alpha, schedule[alpha]) for alpha in NATURAL_RESPONSE_ALPHAS),
    )


def _bundle() -> dict[str, object]:
    return build_natural_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=_fit("signed_log"),
        linear_fit_receipt=_fit("linear"),
    )


def _arm_providers(bundle: dict[str, object]) -> dict[str, str]:
    learned = bundle["selected_provider_artifact_sha256s_by_law"]
    return {
        "base": _sha("held-base-provider"),
        "constant_plus_one": _sha("held-constant-provider"),
        "fixed_signed_log": _sha("held-fixed-log-provider"),
        "fixed_linear": _sha("held-fixed-linear-provider"),
        "learned_signed_log": learned["signed_log"],
        "learned_linear": learned["linear"],
        "learned_signed_log_sign_flip": _sha("held-learned-mirror-provider"),
    }


def _role(
    bundle: dict[str, object],
    *,
    outer: str,
    held: str,
    objectives: dict[str, float],
) -> dict[str, object]:
    providers = _arm_providers(bundle)
    scores = tuple(
        build_natural_response_held_arm_score(
            fit_bundle_receipt=bundle,
            outer_held_family_id=outer,
            held_family_id=held,
            arm=arm,
            objective=objectives[arm],
            provider_artifact_sha256=providers[arm],
            execution_receipt_sha256=_sha(f"{outer}-{held}-{arm}-execution"),
            finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
            execution_changed_from_base=arm != "base",
            response_nonconstant=arm != "constant_plus_one",
        )
        for arm in NATURAL_RESPONSE_ARMS
    )
    return build_natural_response_held_role_receipt(
        fit_bundle_receipt=bundle,
        arm_scores=scores,
    )


def _passing_roles(bundle: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    first = {
        "base": 1.0,
        "constant_plus_one": 0.97,
        "fixed_signed_log": 0.96,
        "fixed_linear": 0.98,
        "learned_signed_log": 0.94,
        "learned_linear": 0.95,
        "learned_signed_log_sign_flip": 1.0,
    }
    second = {
        "base": 1.1,
        "constant_plus_one": 1.07,
        "fixed_signed_log": 1.05,
        "fixed_linear": 1.08,
        "learned_signed_log": 1.0,
        "learned_linear": 1.02,
        "learned_signed_log_sign_flip": 1.08,
    }
    return (
        _role(bundle, outer=EXCLUDED[0], held=EXCLUDED[1], objectives=first),
        _role(bundle, outer=EXCLUDED[1], held=EXCLUDED[0], objectives=second),
    )


def test_bilinear_corner_certificate_dominates_dense_box_and_projection_is_exact() -> None:
    weights = (0.41, -0.27, 0.19)
    corners = bilinear_corner_values(weights)
    assert corners == pytest.approx((0.05, -0.87, 0.49, 0.33))
    certificate = bilinear_box_certificate(weights)
    dense = max(
        abs(natural_response_gain(weights, left / 40.0, right / 40.0, law="linear"))
        for left in range(-40, 41)
        for right in range(-40, 41)
    )
    assert dense <= certificate + 1.0e-15

    for index in range(1, 200):
        proposal = (index / 17.0, -index / 29.0, index / 31.0)
        projected = radially_project_bilinear_weights(proposal)
        assert projected["box_certificate"] <= 1.0
        assert bilinear_box_certificate(projected["weights"]) <= 1.0
        assert projected["projection_semantics"] == "radial_scaling_not_euclidean_projection"


def test_signed_log_is_odd_and_has_frozen_endpoints() -> None:
    assert signed_log_response(0.0) == 0.0
    assert signed_log_response(1.0) == pytest.approx(1.0)
    assert signed_log_response(-1.0) == pytest.approx(-1.0)
    for value in (0.01, 0.2, 0.73):
        assert signed_log_response(-value) == pytest.approx(-signed_log_response(value))


def test_direction_uses_family_equal_mean_and_opg_with_unequal_family_sizes() -> None:
    gradients = {
        FIT_FAMILIES[0]: {"a": (1.0, 0.0, 0.0), "b": (3.0, 0.0, 0.0)},
        **{
            family: {"only": (0.0, 1.0, 0.0)}
            for family in FIT_FAMILIES[1:]
        },
    }
    receipt = _direction("linear", gradients=gradients)
    assert receipt["gradient_mean"] == pytest.approx((2.0 / 6.0, 5.0 / 6.0, 0.0))
    expected_fisher = (
        (5.0 / 6.0, 0.0, 0.0),
        (0.0, 5.0 / 6.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    for actual, expected in zip(receipt["empirical_fisher"], expected_fisher):
        assert actual == pytest.approx(expected)
    assert receipt["damping"] == pytest.approx(1.0e-3 * (5.0 / 3.0) / 3.0)
    assert receipt["fisher_semantics"].endswith("Fisher_OPG")
    assert receipt["strict_descent_direction"] is True
    assert receipt["gradient_dot_natural_direction"] < 0.0


def test_direction_rejects_held_gradient_use_and_receipts_are_json_roundtrip_safe() -> None:
    with pytest.raises(ValueError, match="held objectives or gradients"):
        build_natural_response_direction_receipt(
            v20c_source_sha256s={"complete_h4": _sha("source")},
            family_ids=FAMILIES,
            excluded_family_ids=EXCLUDED,
            fit_gradients_by_family=_standard_gradients(),
            base_provider_artifact_sha256=_sha("base-provider"),
            proposal_provider_artifact_sha256=_sha("proposal-provider"),
            gradient_evidence_sha256=_sha("gradient-evidence"),
            response_law="signed_log",
            held_objectives_or_gradients_used=True,
        )

    direction = _direction("signed_log")
    candidate = _candidate(direction, 0.25, 0.9)
    fit = _fit("signed_log")
    assert validate_natural_response_direction_receipt(json.loads(json.dumps(direction))) == direction
    assert validate_natural_response_alpha_candidate(
        json.loads(json.dumps(candidate)), direction_receipt=json.loads(json.dumps(direction))
    ) == candidate
    assert validate_natural_response_fit_receipt(json.loads(json.dumps(fit))) == fit
    serialized = json.dumps(fit)
    assert '"logits":' not in serialized and '"targets":' not in serialized


def test_fit_selects_valid_exact_improvement_and_persists_full_ladder() -> None:
    fit = _fit("signed_log")
    assert fit["response_law"] == "signed_log"
    assert fit["selected_alpha"] == 0.25
    assert fit["selected_objective"] == pytest.approx(0.9)
    assert fit["learned_candidate_authorized"] is True
    assert fit["rollback_to_initial_weights"] is False
    assert fit["selected_weight_hash_changed"] is True
    assert fit["selected_box_certificate"] <= 1.0
    assert fit["selected_projected_displacement_dot_gradient"] < 0.0
    assert len(fit["candidate_receipts"]) == 6
    assert all(len(candidate["corner_values"]) == 4 for candidate in fit["candidate_receipts"])


def test_fit_tie_uses_smaller_alpha_before_weight_hash() -> None:
    schedule = {
        0.0: 1.0,
        1.0 / 16.0: 0.95,
        1.0 / 8.0: 0.90,
        1.0 / 4.0: 0.90,
        1.0 / 2.0: 0.92,
        1.0: 0.93,
    }
    fit = _fit("linear", objectives=schedule)
    assert fit["selected_alpha"] == 1.0 / 8.0
    assert fit["line_search_tie_order"] == "objective_then_smaller_alpha_then_canonical_weight_hash"


def test_invalid_global_objective_minimum_does_not_mask_valid_candidate() -> None:
    rows = (
        (0.9012580017130718, -0.781287818061025, 0.06539083627986098),
        (0.8117548781259958, -0.7985726527723889, 0.23898126951798515),
        (-0.16390898905162832, 0.15068359595695857, -0.16588126216920895),
        (-0.2221145615519926, 0.13717000131814538, 0.6191847137553148),
        (-0.28203211929756344, 0.565408900155399, -0.7223240764304162),
        (-0.21171489750732575, 0.2898787282074473, -0.7866515717977913),
        (0.8268966079898592, -0.8065028408250399, -0.34344668386273636),
        (-0.17514959182095335, -0.069210989767716, 0.860068296349221),
        (0.15072663432991784, 0.39061578295960375, -0.5368072617481208),
        (0.8128018460079272, -0.651876448317954, -0.40205610065154773),
        (-0.3765524323203886, 0.36620968681107424, 0.7966011422086077),
        (-0.6477835649552126, 0.5429725570906438, 0.36884245867277254),
    )
    gradients = {
        family: {"a": rows[2 * index], "b": rows[2 * index + 1]}
        for index, family in enumerate(FIT_FAMILIES)
    }
    fit = _fit(
        "signed_log",
        gradients=gradients,
        objectives={
            0.0: 1.0,
            1.0 / 16.0: 0.95,
            1.0 / 8.0: 0.90,
            1.0 / 4.0: 0.80,
            1.0 / 2.0: 0.85,
            1.0: 0.10,
        },
    )
    by_alpha = {candidate["alpha"]: candidate for candidate in fit["candidate_receipts"]}
    assert by_alpha[1.0]["family_equal_objective"] == pytest.approx(0.1)
    assert by_alpha[1.0]["projected_displacement_dot_gradient"] > 0.0
    assert fit["selected_alpha"] == 0.25
    assert fit["selected_objective"] == pytest.approx(0.8)
    assert fit["learned_candidate_authorized"] is True


def test_zero_gradient_is_valid_direction_and_deterministic_rollback() -> None:
    zeros = {
        family: {"a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0)}
        for family in FIT_FAMILIES
    }
    fit = _fit(
        "linear",
        gradients=zeros,
        objectives={alpha: (1.0 if alpha == 0.0 else 0.5) for alpha in NATURAL_RESPONSE_ALPHAS},
    )
    direction = fit["direction_receipt"]
    assert direction["strict_descent_direction"] is False
    assert direction["natural_direction"] == (0.0, 0.0, 0.0)
    assert fit["learned_candidate_authorized"] is False
    assert fit["rollback_to_initial_weights"] is True
    assert fit["selected_alpha"] == 0.0
    assert fit["selected_weights"] == NATURAL_RESPONSE_INITIAL_WEIGHTS
    assert fit["selected_provider_artifact_sha256"] is None


def test_fit_requires_complete_alpha_ladder() -> None:
    direction = _direction("signed_log")
    candidates = [_candidate(direction, alpha, 1.0 - alpha / 10.0) for alpha in NATURAL_RESPONSE_ALPHAS]
    with pytest.raises(ValueError, match="complete six-alpha ladder"):
        build_natural_response_fit_receipt(direction_receipt=direction, candidates=candidates[:-1])


def test_two_fit_bundle_binds_laws_and_rejects_cross_lineage() -> None:
    signed = _fit("signed_log")
    linear = _fit("linear")
    bundle = build_natural_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=signed,
        linear_fit_receipt=linear,
    )
    assert bundle["both_fits_authorized"] is True
    assert bundle["held_score_authorized"] is True
    assert bundle["fit_receipts_by_law"]["signed_log"]["response_law"] == "signed_log"
    assert bundle["fit_receipts_by_law"]["linear"]["response_law"] == "linear"
    assert validate_natural_response_two_fit_bundle_receipt(json.loads(json.dumps(bundle))) == bundle

    with pytest.raises(ValueError, match="wrong response law"):
        build_natural_response_two_fit_bundle_receipt(
            signed_log_fit_receipt=linear,
            linear_fit_receipt=signed,
        )
    with pytest.raises(ValueError, match="lineage differs"):
        build_natural_response_two_fit_bundle_receipt(
            signed_log_fit_receipt=signed,
            linear_fit_receipt=_fit("linear", source_label="different-source"),
        )


def test_two_fit_bundle_seals_rollback_and_forbids_held_access() -> None:
    zeros = {
        family: {"only": (0.0, 0.0, 0.0)}
        for family in FIT_FAMILIES
    }
    bundle = build_natural_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=_fit("signed_log"),
        linear_fit_receipt=_fit("linear", gradients=zeros),
    )
    assert bundle["both_fits_authorized"] is False
    assert bundle["held_score_authorized"] is False
    with pytest.raises(ValueError, match="did not authorize"):
        build_natural_response_held_arm_score(
            fit_bundle_receipt=bundle,
            outer_held_family_id=EXCLUDED[0],
            held_family_id=EXCLUDED[1],
            arm="base",
            objective=1.0,
            provider_artifact_sha256=_sha("provider"),
            execution_receipt_sha256=_sha("execution"),
            finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
            execution_changed_from_base=False,
            response_nonconstant=True,
        )


def test_held_roles_and_pair_qualification_enforce_all_gates_and_roundtrip() -> None:
    bundle = _bundle()
    roles = _passing_roles(bundle)
    qualification = build_natural_response_pair_qualification(
        fit_bundle_receipt=bundle,
        roles=roles,
    )
    assert qualification["passed"] is True
    assert qualification["learned_signed_log_base_macro_relative_improvement"] >= 0.01
    assert qualification["learned_signed_log_fixed_log_macro_relative_improvement"] >= 0.001
    assert qualification["learned_signed_log_improves_both_roles"] is True
    assert qualification["learned_signed_log_beats_learned_linear_macro"] is True
    assert qualification["learned_signed_log_beats_mirror_both_roles"] is True

    decoded_bundle = json.loads(json.dumps(bundle))
    decoded_roles = json.loads(json.dumps(roles))
    for role in decoded_roles:
        validate_natural_response_held_role_receipt(role, fit_bundle_receipt=decoded_bundle)
    assert validate_natural_response_pair_qualification(
        json.loads(json.dumps(qualification)),
        fit_bundle_receipt=decoded_bundle,
        roles=decoded_roles,
    ) == qualification


def test_held_zero_reference_objectives_fail_finitely_not_by_division() -> None:
    bundle = _bundle()
    zero_objectives = {arm: 0.0 for arm in NATURAL_RESPONSE_ARMS}
    roles = (
        _role(bundle, outer=EXCLUDED[0], held=EXCLUDED[1], objectives=zero_objectives),
        _role(bundle, outer=EXCLUDED[1], held=EXCLUDED[0], objectives=zero_objectives),
    )
    qualification = build_natural_response_pair_qualification(
        fit_bundle_receipt=bundle,
        roles=roles,
    )
    assert qualification["base_macro_denominator_valid"] is False
    assert qualification["fixed_log_macro_denominator_valid"] is False
    assert qualification["learned_signed_log_base_macro_relative_improvement"] == 0.0
    assert qualification["learned_signed_log_fixed_log_macro_relative_improvement"] == 0.0
    assert qualification["passed"] is False
    json.dumps(qualification, allow_nan=False)


def test_learned_held_arm_provider_is_bound_to_its_response_law() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="signed-log arm provider"):
        build_natural_response_held_arm_score(
            fit_bundle_receipt=bundle,
            outer_held_family_id=EXCLUDED[0],
            held_family_id=EXCLUDED[1],
            arm="learned_signed_log",
            objective=1.0,
            provider_artifact_sha256=bundle["selected_provider_artifact_sha256s_by_law"]["linear"],
            execution_receipt_sha256=_sha("wrong-law-execution"),
            finite=True,
            pointwise_trust_passed=True,
            rank_is_16=True,
            execution_changed_from_base=True,
            response_nonconstant=True,
        )


def test_work_accounting_has_full_and_fit_terminal_totals() -> None:
    common = {
        "collection_forward_count": 32,
        "collection_backward_count": 16,
        "endpoint_forward_count": 12,
        "endpoint_backward_count": 12,
        "endpoint_local_contraction_count": 12,
        "law_count": 2,
        "fit_prompt_count": 12,
        "alpha_count": 6,
        "alpha_zero_vjp_reused": True,
        "held_role_count": 2,
        "held_arm_count": 7,
        "held_prompts_per_role": 2,
    }
    full = natural_response_work_accounting(**common, held_scoring_executed=True)
    assert full["total_model_forward_count"] == 216
    assert full["total_model_backward_count"] == 52
    assert full["total_local_contraction_count"] == 36
    assert full["total_backward_or_local_contraction_count"] == 88
    assert full["teacher_h4_logit_access_count"] == 184
    assert full["fit_candidate_count"] == 12
    assert full["held_role_arm_score_count"] == 14

    terminal = natural_response_work_accounting(**common, held_scoring_executed=False)
    assert terminal["total_model_forward_count"] == 188
    assert terminal["total_model_backward_count"] == 52
    assert terminal["total_local_contraction_count"] == 36
    assert terminal["total_backward_or_local_contraction_count"] == 88
    assert terminal["teacher_h4_logit_access_count"] == 156
    assert terminal["held_forward_count"] == 0
    assert terminal["held_role_arm_score_count"] == 0


def test_hash_tampering_is_rejected() -> None:
    bundle = _bundle()
    tampered = copy.deepcopy(bundle)
    weights = list(tampered["selected_weights_by_law"]["signed_log"])
    weights[0] += 0.01
    tampered["selected_weights_by_law"]["signed_log"] = weights
    with pytest.raises(ValueError, match="drifted"):
        validate_natural_response_two_fit_bundle_receipt(tampered)
