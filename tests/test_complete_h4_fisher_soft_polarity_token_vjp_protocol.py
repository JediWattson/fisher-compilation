from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_token_vjp_protocol as core
from fisher_graph.complete_h4_fisher_soft_polarity_token_vjp_protocol import (
    SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER,
    SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS,
    SOFT_POLARITY_TOKEN_VJP_CANDIDATE_LIBRARY,
    SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS,
    SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS,
    SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID,
    SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
    SOFT_POLARITY_TOKEN_VJP_PROTOCOL_SHA256,
    SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER,
    SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS,
    build_soft_polarity_token_vjp_all_seven_refit_receipt,
    build_soft_polarity_token_vjp_candidate_receipt,
    build_soft_polarity_token_vjp_inner_oof_selection_receipt,
    build_soft_polarity_token_vjp_natural_direction_output,
    build_soft_polarity_token_vjp_protocol_receipt,
    build_soft_polarity_token_vjp_scalar_fit_output,
    validate_soft_polarity_token_vjp_all_seven_refit_receipt,
    validate_soft_polarity_token_vjp_candidate_receipt,
    validate_soft_polarity_token_vjp_inner_oof_selection_receipt,
    validate_soft_polarity_token_vjp_protocol_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]
SPEC_BY_ID = {
    row[0]: {
        "candidate_id": row[0],
        "role": row[1],
        "feature_id": row[2],
        "seed_sign": row[3],
        "seed_b": row[4],
        "seed_a": row[5],
        "ridge": row[6],
        "alpha": row[7],
        "fixed_b": row[8],
        "fixed_a": row[9],
    }
    for row in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_LIBRARY
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _direction_metadata(
    *,
    feature_id: str,
    held_family_id: str,
    training_family_ids: tuple[str, ...],
    seed_b: float,
    ridge: float,
    direction_b: float = 1.0,
    direction_a: float = 0.5,
    method: str = "mean_kl_natural_opg_trace_scaled_ridge_linf_direction",
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate_hash = _sha(
        f"aggregate:{feature_id}:{held_family_id}:{ridge.hex()}"
    )
    aggregate = {
        "artifact_sha256": aggregate_hash,
        "feature_id": feature_id,
        "held_family_id": held_family_id,
        "training_family_ids": training_family_ids,
        "mean_parameter_gradient_sha256": _sha(
            f"g:{feature_id}:{held_family_id}:{ridge.hex()}"
        ),
        "gradient_gram_sha256": _sha(
            f"F:{feature_id}:{held_family_id}:{ridge.hex()}"
        ),
    }
    direction = {
        "artifact_sha256": _sha(
            f"direction:{feature_id}:{held_family_id}:{ridge.hex()}"
        ),
        "aggregate_artifact_sha256": aggregate_hash,
        "method": method,
        "parameter_order": ("field_bias", "field_slope"),
        "feature_id": feature_id,
        "held_family_id": held_family_id,
        "reference_b": seed_b,
        "reference_a": 0.0,
        "ridge_multiplier": ridge,
        "gradient_gram_trace": 2.0,
        "tau": 1.0,
        "damping": ridge,
        "direction_b": direction_b,
        "direction_a": direction_a,
        "direction_linf": 1.0,
        "predicted_derivative": -0.25,
        "no_op": False,
    }
    return direction, aggregate


def _stability(*, passed: bool = True) -> dict[str, object]:
    cosine = (1.0, 1.0 if passed else 0.5)
    return {
        "cosine_by_parameter": cosine,
        "audit_to_primary_norm_ratio_by_parameter": (1.0, 1.0),
        "passed": passed,
    }


def _fit_output(
    *,
    feature_id: str,
    held_family_id: str,
    training_family_ids: tuple[str, ...],
    seed_b: float,
    ridge: float,
    direction_b: float = 1.0,
    direction_a: float = 0.5,
    passed: bool = True,
) -> dict[str, object]:
    direction, aggregate = _direction_metadata(
        feature_id=feature_id,
        held_family_id=held_family_id,
        training_family_ids=training_family_ids,
        seed_b=seed_b,
        ridge=ridge,
        direction_b=direction_b,
        direction_a=direction_a,
    )
    return build_soft_polarity_token_vjp_scalar_fit_output(
        direction_metadata=direction,
        aggregate_metadata=aggregate,
        primary_secant_receipt_sha256=_sha(
            f"primary:{feature_id}:{held_family_id}"
        ),
        audit_secant_receipt_sha256=_sha(
            f"audit:{feature_id}:{held_family_id}"
        ),
        secant_stability=_stability(passed=passed),
    )


def _candidate(
    protocol: dict[str, object], inner: str, candidate_id: str
) -> dict[str, object]:
    spec = SPEC_BY_ID[candidate_id]
    kwargs: dict[str, object] = {}
    if spec["role"] == "token_vjp_fit":
        kwargs["scalar_fit_output"] = _fit_output(
            feature_id=str(spec["feature_id"]),
            held_family_id=inner,
            training_family_ids=tuple(
                family for family in INNER if family != inner
            ),
            seed_b=float(spec["seed_b"]),
            ridge=float(spec["ridge"]),
        )
    elif spec["role"] == "v20p_incumbent":
        kwargs.update(
            incumbent_feature_id="source_z",
            incumbent_b=1.0,
            incumbent_a=0.0,
            incumbent_fit_receipt_sha256=_sha("v20p-incumbent"),
        )
    return build_soft_polarity_token_vjp_candidate_receipt(
        protocol_receipt=protocol,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        inner_held_family_id=inner,
        candidate_id=candidate_id,
        candidate_provider_sha256=_sha(f"provider:{inner}:{candidate_id}"),
        **kwargs,
    )


@pytest.fixture(scope="module")
def campaign():
    protocol = build_soft_polarity_token_vjp_protocol_receipt()
    receipts = {
        inner: {
            candidate_id: _candidate(protocol, inner, candidate_id)
            for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
        }
        for inner in INNER
    }
    objectives = {
        inner: {
            candidate_id: 1.2
            for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
        }
        for inner in INNER
    }
    for inner in INNER:
        objectives[inner][SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID] = 1.0
    return protocol, receipts, objectives


def _selection(protocol, receipts, objectives):
    return build_soft_polarity_token_vjp_inner_oof_selection_receipt(
        protocol_receipt=protocol,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        candidate_receipts_by_inner_family=receipts,
        exact_objectives_by_inner_family_and_candidate=objectives,
    )


def test_protocol_freezes_174_logical_candidates_and_constants() -> None:
    assert SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS == (
        "c1",
        "c2",
        "c1_times_c2",
        "source_z",
    )
    assert SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS == (-1, 1)
    assert SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP == 1.0 / 64.0
    assert SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP == 1.0 / 128.0
    assert SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER == (0.1, 1.0, 10.0)
    assert SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER == (
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    )
    assert len(SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS) == 168
    assert len(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS) == 174
    assert len(set(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS)) == 174
    assert SOFT_POLARITY_TOKEN_VJP_PROTOCOL_SHA256 == (
        "2f54b2e4679dcfb11a90580a46bc4e7b1612a34a7b1636b3fbdf406b769bf2aa"
    )


def test_protocol_receipt_is_replayable_and_tensor_free() -> None:
    receipt = build_soft_polarity_token_vjp_protocol_receipt()
    validate_soft_polarity_token_vjp_protocol_receipt(receipt)
    roundtrip = json.loads(json.dumps(receipt))
    validate_soft_polarity_token_vjp_protocol_receipt(roundtrip)
    assert receipt["fit_candidate_count"] == 168
    assert receipt["candidate_count"] == 174
    assert receipt["runtime"] == "unchanged_v20p_local_signed_field_provider"
    tampered = copy.deepcopy(receipt)
    tampered["minimum_secant_cosine"] = 0.0
    with pytest.raises(ValueError):
        validate_soft_polarity_token_vjp_protocol_receipt(tampered)


def test_scalar_direction_bridge_rejects_residual_gn_l2_fit() -> None:
    direction, aggregate = _direction_metadata(
        feature_id="c1",
        held_family_id=INNER[0],
        training_family_ids=INNER[1:],
        seed_b=-0.5,
        ridge=0.1,
        method="one_step_damped_residual_Gauss_Newton_with_OPG",
    )
    with pytest.raises(ValueError, match="natural-OPG"):
        build_soft_polarity_token_vjp_scalar_fit_output(
            direction_metadata=direction,
            aggregate_metadata=aggregate,
            primary_secant_receipt_sha256=_sha("primary"),
            audit_secant_receipt_sha256=_sha("audit"),
            secant_stability=_stability(),
        )


def test_tensor_free_natural_direction_builder_solves_and_binds_g_f() -> None:
    g = (2.0, -1.0)
    gram = ((4.0, 1.0), (1.0, 2.0))
    aggregate = {
        "artifact_sha256": _sha("typed-aggregate"),
        "feature_id": "c1",
        "held_family_id": INNER[0],
        "reference_b": -0.5,
        "reference_a": 0.0,
        "mean_parameter_gradient_sha256": core._float64_tensor_sha256(g, (2,)),
        "gradient_gram_sha256": core._float64_tensor_sha256(
            (4.0, 1.0, 1.0, 2.0), (2, 2)
        ),
    }
    direction = build_soft_polarity_token_vjp_natural_direction_output(
        aggregate_metadata=aggregate,
        mean_gradient=g,
        gradient_gram=gram,
        ridge_multiplier=0.1,
    )
    assert direction["gradient_gram_trace"] == 6.0
    assert direction["tau"] == 3.0
    assert direction["damping"] == pytest.approx(0.3)
    assert direction["direction_linf"] == 1.0
    assert direction["predicted_derivative"] < 0.0
    assert direction["raw_model_tensors_serialized"] is False

    forged = dict(aggregate)
    forged["mean_parameter_gradient_sha256"] = _sha("wrong-g")
    with pytest.raises(ValueError, match="do not replay"):
        build_soft_polarity_token_vjp_natural_direction_output(
            aggregate_metadata=forged,
            mean_gradient=g,
            gradient_gram=gram,
            ridge_multiplier=0.1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tau", 0.5, "trace scale"),
        ("damping", 0.5, "ridge times tau"),
        ("direction_linf", 0.5, "L-infinity"),
        ("predicted_derivative", 0.0, "descending"),
    ],
)
def test_scalar_direction_bridge_enforces_natural_direction_geometry(
    field, value, message
) -> None:
    direction, aggregate = _direction_metadata(
        feature_id="c1",
        held_family_id=INNER[0],
        training_family_ids=INNER[1:],
        seed_b=-0.5,
        ridge=0.1,
    )
    direction[field] = value
    with pytest.raises(ValueError, match=message):
        build_soft_polarity_token_vjp_scalar_fit_output(
            direction_metadata=direction,
            aggregate_metadata=aggregate,
            primary_secant_receipt_sha256=_sha("primary"),
            audit_secant_receipt_sha256=_sha("audit"),
            secant_stability=_stability(),
        )


def test_candidate_materializes_alpha_child_and_enforces_six_one(campaign) -> None:
    protocol, receipts, _ = campaign
    candidate_id = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[0]
    receipt = receipts[INNER[0]][candidate_id]
    assert receipt["training_family_count"] == 6
    assert tuple(receipt["training_family_ids"]) == INNER[1:]
    assert receipt["b"] == -0.5 + 1.0 / 64.0
    assert receipt["a"] == 0.5 / 64.0
    assert receipt["inner_held_objective_consumed"] is False
    validate_soft_polarity_token_vjp_candidate_receipt(
        receipt,
        protocol_receipt=protocol,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        inner_held_family_id=INNER[0],
    )

    bad_output = _fit_output(
        feature_id="c1",
        held_family_id=INNER[0],
        training_family_ids=INNER[2:],
        seed_b=-0.5,
        ridge=0.1,
    )
    with pytest.raises(ValueError, match="candidate index"):
        build_soft_polarity_token_vjp_candidate_receipt(
            protocol_receipt=protocol,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            inner_held_family_id=INNER[0],
            candidate_id=candidate_id,
            candidate_provider_sha256=_sha("bad-provider"),
            scalar_fit_output=bad_output,
        )


def test_candidate_rejects_failed_secant_stability(campaign) -> None:
    protocol, _, _ = campaign
    candidate_id = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[0]
    spec = SPEC_BY_ID[candidate_id]
    output = _fit_output(
        feature_id=str(spec["feature_id"]),
        held_family_id=INNER[0],
        training_family_ids=INNER[1:],
        seed_b=float(spec["seed_b"]),
        ridge=float(spec["ridge"]),
        passed=False,
    )
    with pytest.raises(ValueError, match="stable and descending"):
        build_soft_polarity_token_vjp_candidate_receipt(
            protocol_receipt=protocol,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            inner_held_family_id=INNER[0],
            candidate_id=candidate_id,
            candidate_provider_sha256=_sha("unstable-provider"),
            scalar_fit_output=output,
        )


def test_paired_one_se_rule_selects_smaller_alpha_recipe(campaign) -> None:
    protocol, receipts, base_objectives = campaign
    objectives = copy.deepcopy(base_objectives)
    smaller_alpha = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[0]
    exact_best = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[6]
    best_deltas = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1)
    for family, delta in zip(INNER, best_deltas, strict=True):
        objectives[family][exact_best] = 1.0 + delta
        objectives[family][smaller_alpha] = 0.85
    selection = _selection(protocol, receipts, objectives)
    assert selection["exact_best_candidate_id"] == exact_best
    assert smaller_alpha in selection["eligible_candidate_ids"]
    assert selection["selected_candidate_id"] == smaller_alpha
    assert selection["aggregate_by_candidate"][exact_best][
        "paired_delta_standard_error"
    ] == pytest.approx(math.sqrt(0.28 / 6.0 / 7.0))


def test_one_se_rule_rolls_back_to_incumbent_when_improvement_is_uncertain(
    campaign,
) -> None:
    protocol, receipts, base_objectives = campaign
    objectives = copy.deepcopy(base_objectives)
    candidate = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[-1]
    deltas = (-0.32, -0.22, -0.12, -0.02, 0.08, 0.18, 0.28)
    for family, delta in zip(INNER, deltas, strict=True):
        objectives[family][candidate] = 1.0 + delta
    selection = _selection(protocol, receipts, objectives)
    assert selection["exact_best_candidate_id"] == candidate
    assert SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID in selection[
        "eligible_candidate_ids"
    ]
    assert selection["selected_candidate_id"] == (
        SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID
    )


def test_selection_requires_every_inner_family_and_logical_candidate(campaign) -> None:
    protocol, receipts, objectives = campaign
    missing_family = dict(receipts)
    missing_family.pop(INNER[-1])
    with pytest.raises(ValueError, match="family geometry"):
        _selection(protocol, missing_family, objectives)

    missing_candidate = copy.deepcopy(receipts)
    missing_candidate[INNER[0]].pop(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS[-1])
    with pytest.raises(ValueError, match="candidate receipt geometry"):
        _selection(protocol, missing_candidate, objectives)


def test_selection_json_replay_and_rehashed_tamper_fail_closed(campaign) -> None:
    protocol, receipts, objectives = campaign
    selection = _selection(protocol, receipts, objectives)
    roundtrip = json.loads(json.dumps(selection))
    validate_soft_polarity_token_vjp_inner_oof_selection_receipt(
        roundtrip,
        protocol_receipt=protocol,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        candidate_receipts_by_inner_family=receipts,
        exact_objectives_by_inner_family_and_candidate=objectives,
    )
    tampered = copy.deepcopy(selection)
    tampered["selected_candidate_id"] = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[0]
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = core._hash(core._SELECTION_DOMAIN, tampered)
    with pytest.raises(ValueError, match="content differs"):
        validate_soft_polarity_token_vjp_inner_oof_selection_receipt(
            tampered,
            protocol_receipt=protocol,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            candidate_receipts_by_inner_family=receipts,
            exact_objectives_by_inner_family_and_candidate=objectives,
        )


def test_all_seven_refit_recomputes_selected_direction(campaign) -> None:
    protocol, receipts, base_objectives = campaign
    objectives = copy.deepcopy(base_objectives)
    selected_id = SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS[0]
    for inner in INNER:
        objectives[inner][selected_id] = 0.5
    selection = _selection(protocol, receipts, objectives)
    assert selection["selected_candidate_id"] == selected_id
    spec = SPEC_BY_ID[selected_id]
    final_output = _fit_output(
        feature_id=str(spec["feature_id"]),
        held_family_id=OUTER,
        training_family_ids=INNER,
        seed_b=float(spec["seed_b"]),
        ridge=float(spec["ridge"]),
        direction_b=-1.0,
        direction_a=0.25,
    )
    final = build_soft_polarity_token_vjp_all_seven_refit_receipt(
        protocol_receipt=protocol,
        selection_receipt=selection,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        final_candidate_provider_sha256=_sha("final-provider"),
        scalar_fit_output=final_output,
    )
    assert final["training_family_count"] == 7
    assert final["all_seven_refit_completed"] is True
    assert final["b"] == -0.5 - 1.0 / 64.0
    assert final["a"] == 0.25 / 64.0
    assert final["outer_held_objective_consumed"] is False
    validate_soft_polarity_token_vjp_all_seven_refit_receipt(
        json.loads(json.dumps(final)),
        protocol_receipt=protocol,
        selection_receipt=selection,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
    )


def test_all_seven_incumbent_selection_is_exact_control_reuse(campaign) -> None:
    protocol, receipts, objectives = campaign
    selection = _selection(protocol, receipts, objectives)
    assert selection["selected_candidate_id"] == (
        SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID
    )
    final = build_soft_polarity_token_vjp_all_seven_refit_receipt(
        protocol_receipt=protocol,
        selection_receipt=selection,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        final_candidate_provider_sha256=_sha("incumbent-provider"),
        incumbent_feature_id="source_z",
        incumbent_b=1.0,
        incumbent_a=0.0,
        incumbent_fit_receipt_sha256=_sha("v20p-incumbent"),
    )
    assert final["all_seven_refit_completed"] is False
    assert final["selected_fixed_control_replayed"] is True
    assert final["b"] == 1.0
    assert final["a"] == 0.0


@pytest.mark.parametrize(
    ("selected_id", "expected_role", "expected_b"),
    [
        ("v20q_anchor_minus", "exact_anchor", -1.0),
        ("v20q_seed_plus", "smooth_seed", 0.5),
    ],
)
def test_all_seven_fixed_anchor_and_seed_are_exact_control_replays(
    campaign,
    selected_id,
    expected_role,
    expected_b,
) -> None:
    protocol, receipts, base_objectives = campaign
    objectives = copy.deepcopy(base_objectives)
    for inner in INNER:
        objectives[inner][selected_id] = 0.5
    selection = _selection(protocol, receipts, objectives)
    assert selection["selected_candidate_id"] == selected_id

    final = build_soft_polarity_token_vjp_all_seven_refit_receipt(
        protocol_receipt=protocol,
        selection_receipt=selection,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        final_candidate_provider_sha256=_sha(f"final-control:{selected_id}"),
    )
    assert final["selected_candidate_spec"]["role"] == expected_role
    assert final["feature_id"] == "source_z"
    assert final["b"] == expected_b
    assert final["a"] == 0.0
    assert final["scalar_fit_output"] is None
    assert final["incumbent_fit_receipt_sha256"] is None
    assert final["all_seven_refit_completed"] is False
    assert final["selected_fixed_control_replayed"] is True
    assert final["provider_frozen_before_outer_held_objective"] is True
    assert final["outer_held_objective_consumed"] is False

    roundtrip = json.loads(json.dumps(final))
    validate_soft_polarity_token_vjp_all_seven_refit_receipt(
        roundtrip,
        protocol_receipt=protocol,
        selection_receipt=selection,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
    )

    tampered = copy.deepcopy(final)
    tampered["b"] = math.nextafter(expected_b, math.inf)
    tampered["b_hex"] = tampered["b"].hex()
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = core._hash(core._FINAL_DOMAIN, tampered)
    with pytest.raises(ValueError, match="content differs"):
        validate_soft_polarity_token_vjp_all_seven_refit_receipt(
            tampered,
            protocol_receipt=protocol,
            selection_receipt=selection,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
        )


def test_public_policy_surface_has_no_model_tensor_or_outer_score_arguments() -> None:
    forbidden = {
        "model",
        "tensor",
        "h4",
        "logits",
        "prompt",
        "prompt_text",
        "outer_held_objective",
        "outer_held_score",
        "calibration_b",
    }
    for function in (
        build_soft_polarity_token_vjp_protocol_receipt,
        build_soft_polarity_token_vjp_scalar_fit_output,
        build_soft_polarity_token_vjp_candidate_receipt,
        build_soft_polarity_token_vjp_inner_oof_selection_receipt,
        build_soft_polarity_token_vjp_all_seven_refit_receipt,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1.0"])
def test_exact_kl_objectives_require_finite_strict_floats(campaign, bad) -> None:
    protocol, receipts, base_objectives = campaign
    objectives = copy.deepcopy(base_objectives)
    objectives[INNER[0]][SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS[0]] = bad
    with pytest.raises((TypeError, ValueError)):
        _selection(protocol, receipts, objectives)
