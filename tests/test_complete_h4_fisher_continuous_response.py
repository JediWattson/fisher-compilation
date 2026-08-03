from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from fisher_graph.complete_h4_fisher_continuous_response import (
    CONTINUOUS_RESPONSE_ARMS,
    CONTINUOUS_RESPONSE_KAPPA,
    build_continuous_response_arm_score,
    build_continuous_response_law_receipt,
    build_continuous_response_role_receipt,
    build_continuous_response_sentinel_receipt,
    continuous_response_value,
    fit_coordinate_statistics,
    select_strongest_fisher_coordinate,
    signed_log,
    validate_continuous_response_law_receipt,
    validate_continuous_response_sentinel_receipt,
)


def h(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


FAMILIES = tuple(f"family-{index}" for index in range(8))
OUTER = FAMILIES[0]
SOURCES = {"nested_panel": h("panel"), "nested_result": h("result")}


def coordinates(seed: int = 0):
    return tuple(
        (
            0.01 * (index + seed),
            (-1.0 if index % 2 else 1.0) * min(0.95, 0.08 * (index + 1 + seed)),
            0.2 * ((index % 3) - 1),
        )
        for index in range(12)
    )


def law(held: str, *, signed_lambda=1.0, linear_lambda=1.0, **kwargs):
    excluded = {OUTER, held}
    fit_families = tuple(family for family in FAMILIES if family not in excluded)
    return build_continuous_response_law_receipt(
        v20b_source_sha256s=SOURCES,
        family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        held_family_id=held,
        coordinate_source_family_ids=fit_families,
        fit_coordinates_by_family={
            family: coordinates(FAMILIES.index(family)) for family in fit_families
        },
        base_provider_artifact_sha256=h((held, "base")),
        proposal_provider_artifact_sha256=h((held, "proposal")),
        lambda_fit_evidence_sha256=h((held, "lambda-fit")),
        signed_log_lambda=signed_lambda,
        linear_lambda=linear_lambda,
        **kwargs,
    )


def score(law_receipt, arm, objective, *, changed=None, **kwargs):
    if changed is None:
        changed = arm != "base"
    return build_continuous_response_arm_score(
        law_receipt=law_receipt,
        arm=arm,
        objective=objective,
        execution_receipt_sha256=h((law_receipt["held_family_id"], arm, "exec")),
        response_trace_sha256=h((law_receipt["held_family_id"], arm, "trace")),
        finite=kwargs.pop("finite", True),
        pointwise_trust_passed=kwargs.pop("trust", True),
        rank_is_16=kwargs.pop("rank", True),
        execution_changed_from_base=changed,
        **kwargs,
    )


def role(held: str, *, objectives=None, changed=None):
    law_receipt = law(held)
    values = {
        "base": 1.0,
        "signed_log": 0.97,
        "constant_plus_one": 0.99,
        "signed_log_sign_flip": 1.02,
        "linear": 0.985,
    }
    values.update(objectives or {})
    scores = tuple(
        score(
            law_receipt,
            arm,
            values[arm],
            changed=(arm != "base" if changed is None else changed.get(arm, arm != "base")),
        )
        for arm in CONTINUOUS_RESPONSE_ARMS
    )
    return build_continuous_response_role_receipt(law_receipt=law_receipt, arm_scores=scores)


def roles():
    return tuple(role(held) for held in FAMILIES[1:])


def sentinel(selected_roles=None):
    return build_continuous_response_sentinel_receipt(
        family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        v20b_source_sha256s=SOURCES,
        roles=roles() if selected_roles is None else selected_roles,
    )


def test_signed_log_is_smooth_odd_zero_safe_and_fixed_at_kappa_nine():
    assert CONTINUOUS_RESPONSE_KAPPA == 9.0
    assert signed_log(0.0) == 0.0
    assert signed_log(1.0) == pytest.approx(1.0)
    assert signed_log(-0.37) == pytest.approx(-signed_log(0.37))
    epsilon = 1.0e-7
    left = (signed_log(0.0) - signed_log(-epsilon)) / epsilon
    right = (signed_log(epsilon) - signed_log(0.0)) / epsilon
    assert left == pytest.approx(right, rel=1.0e-6)


def test_coordinate_selection_uses_fit_variance_and_lowest_index_tie_only():
    rows = ((-0.5, 0.75, 0.5), (0.5, -0.75, -0.5))
    assert select_strongest_fisher_coordinate((rows[:1], rows[1:])) == 1
    tied = ((-0.5, 0.5), (0.5, -0.5))
    assert select_strongest_fisher_coordinate((tied[:1], tied[1:])) == 0
    stats = fit_coordinate_statistics((rows[:1], rows[1:]))
    assert stats["selected_coordinate_index"] == 1
    assert stats["selected_coordinate_center"] == 0.0
    assert stats["selected_coordinate_scale"] == 0.75


def test_coordinate_variance_is_family_equal_not_token_population_weighted():
    long_family = tuple((0.6 if index % 2 else -0.6, 0.0) for index in range(100))
    short_family = ((0.0, -1.0), (0.0, 1.0))
    # Token-population weighting would select axis 0 because the long family
    # has fifty times more rows.  Family-equal weighting correctly selects 1.
    assert select_strongest_fisher_coordinate((long_family, short_family)) == 1


def test_law_is_scalar_only_json_safe_and_controls_are_exact():
    value = law(FAMILIES[1])
    assert validate_continuous_response_law_receipt(json.loads(json.dumps(value))) == value
    assert value["scientific_status"] == "development_only_reused_a16"
    assert value["fresh_family_disjoint_claim_authorized"] is False
    assert value["raw_coordinates_or_objectives_serialized"] is False
    assert value["coordinate_statistics_usage"] == "diagnostic_only_not_applied"
    assert continuous_response_value(0.0, value, arm="signed_log") == 0.0
    signed = continuous_response_value(0.5, value, arm="signed_log")
    assert signed == signed_log(0.5)
    assert continuous_response_value(0.5, value, arm="linear") == 0.5
    assert continuous_response_value(0.5, value, arm="base") == 0.0
    assert continuous_response_value(0.5, value, arm="constant_plus_one") == 1.0
    assert continuous_response_value(0.5, value, arm="signed_log_sign_flip") == -signed


def test_law_rejects_outer_access_and_held_derived_statistics_marker():
    held = FAMILIES[1]
    with pytest.raises(ValueError, match="exactly the six fit families"):
        build_continuous_response_law_receipt(
            v20b_source_sha256s=SOURCES,
            family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            held_family_id=held,
            coordinate_source_family_ids=FAMILIES[:6],
            fit_coordinates_by_family={family: coordinates() for family in FAMILIES[:6]},
            base_provider_artifact_sha256=h("base"),
            proposal_provider_artifact_sha256=h("proposal"),
            lambda_fit_evidence_sha256=h("fit"),
            signed_log_lambda=0.2,
            linear_lambda=0.1,
        )
    with pytest.raises(ValueError, match="held data may not define"):
        law(held, held_data_used_for_statistics_or_fit=True)


def test_law_rejects_lambda_and_statistics_tampering():
    value = law(FAMILIES[1])
    with pytest.raises(ValueError, match="fixed at exactly 1"):
        law(FAMILIES[1], signed_lambda=0.9)
    tampered = copy.deepcopy(value)
    tampered["signed_log_lambda"] = 9.0
    with pytest.raises(ValueError, match="fixed at exactly 1"):
        validate_continuous_response_law_receipt(tampered)
    tampered = copy.deepcopy(value)
    tampered["coordinate_statistics_scope"] = "held_plus_fit"
    with pytest.raises(ValueError, match="fit-only"):
        validate_continuous_response_law_receipt(tampered)
    tampered = copy.deepcopy(value)
    tampered["coordinate_statistics"]["selected_coordinate_center"] += 0.125
    with pytest.raises(ValueError, match="coordinate center differs"):
        validate_continuous_response_law_receipt(tampered)


def test_expected_source_and_endpoint_lineage_rejects_wrong_receipts():
    value = law(FAMILIES[1])
    with pytest.raises(ValueError, match="source lineage"):
        validate_continuous_response_law_receipt(
            value, expected_v20b_source_sha256s={"nested_panel": h("wrong")}
        )
    with pytest.raises(ValueError, match="base endpoint"):
        validate_continuous_response_law_receipt(
            value, expected_base_provider_artifact_sha256=h("wrong-base")
        )
    with pytest.raises(ValueError, match="proposal endpoint"):
        validate_continuous_response_law_receipt(
            value, expected_proposal_provider_artifact_sha256=h("wrong-proposal")
        )


def test_predicted_only_scores_and_wrong_mirror_are_rejected():
    value = law(FAMILIES[1])
    with pytest.raises(ValueError, match="exact finite execution"):
        score(value, "signed_log", 0.9, score_source="curve_prediction")
    values = [score(value, arm, 1.0) for arm in CONTINUOUS_RESPONSE_ARMS]
    mirror_index = CONTINUOUS_RESPONSE_ARMS.index("signed_log_sign_flip")
    wrong = copy.deepcopy(values[mirror_index])
    wrong["arm_definition"]["response_sign"] = 1.0
    values[mirror_index] = wrong
    with pytest.raises(ValueError, match="drifted"):
        build_continuous_response_role_receipt(law_receipt=value, arm_scores=values)


def test_complete_family_equal_sentinel_passes_every_frozen_gate():
    value = sentinel()
    qualification = value["qualification"]
    assert qualification["passed"] is True
    assert qualification["base_win_count"] == 7
    assert qualification["constant_win_count"] == 7
    assert qualification["mirror_win_count"] == 7
    assert qualification["signed_log_beats_linear_macro_by_numerical_floor"] is True
    assert qualification["all_finite_trusted_rank16_changed_exact"] is True
    assert value["sentinel_passed"] is True
    assert value["next_full_reused_panel_screen_authorized"] is True
    assert value["fresh_family_disjoint_claim_authorized"] is False


def test_sentinel_rejects_partial_roles_and_outer_role_access():
    complete = roles()
    with pytest.raises(ValueError, match="seven non-outer"):
        sentinel(complete[:-1])
    outer_law = law(FAMILIES[1])
    mutated = copy.deepcopy(outer_law)
    mutated["fit_family_ids"] = tuple((*mutated["fit_family_ids"][:-1], OUTER))
    # Hash tampering is caught before a role carrying an outer fit can exist.
    with pytest.raises(ValueError, match="drifted"):
        validate_continuous_response_law_receipt(mutated)


def test_sentinel_external_endpoint_binding_and_hash_tampering():
    value = sentinel()
    endpoints = value["endpoint_artifact_sha256s_by_held_family"]
    assert validate_continuous_response_sentinel_receipt(
        json.loads(json.dumps(value)),
        expected_v20b_source_sha256s=SOURCES,
        expected_endpoint_artifact_sha256s_by_held_family=endpoints,
    ) == value
    wrong = copy.deepcopy(endpoints)
    wrong[FAMILIES[1]]["proposal"] = h("wrong")
    with pytest.raises(ValueError, match="endpoint lineage"):
        validate_continuous_response_sentinel_receipt(
            value,
            expected_endpoint_artifact_sha256s_by_held_family=wrong,
        )


@pytest.mark.parametrize(
    ("objectives", "changed", "failed_field"),
    (
        ({"signed_log": 0.995}, None, "passed"),
        ({"constant_plus_one": 0.96}, None, "signed_log_beats_constant_macro_by_numerical_floor"),
        ({"signed_log_sign_flip": 0.96}, None, "signed_log_beats_mirror_macro_by_numerical_floor"),
        ({"linear": 0.96}, None, "signed_log_beats_linear_macro_by_numerical_floor"),
        (None, {"linear": False}, "all_finite_trusted_rank16_changed_exact"),
    ),
)
def test_qualification_fails_materiality_controls_and_execution_health(objectives, changed, failed_field):
    selected = tuple(
        role(held, objectives=objectives, changed=changed)
        for held in FAMILIES[1:]
    )
    result = sentinel(selected)["qualification"]
    assert result[failed_field] is False
    assert result["passed"] is False


def test_response_preserves_raw_bounded_domain_without_clamping():
    value = law(FAMILIES[1])
    for coordinate in (-1.0, -0.5, 0.0, 0.5, 1.0):
        assert math.isfinite(continuous_response_value(coordinate, value))
    with pytest.raises(ValueError, match="raw bounded"):
        continuous_response_value(1.0001, value)
