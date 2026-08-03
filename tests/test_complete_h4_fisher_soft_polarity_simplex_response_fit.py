from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_simplex_response_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_response_fit import (
    SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
    SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256,
    SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER,
    SOFT_POLARITY_SIMPLEX_RESPONSE_TRIPLES,
    build_soft_polarity_simplex_response_fit_receipt,
    build_soft_polarity_simplex_response_inner_oof_selection_receipt,
    build_soft_polarity_simplex_response_ladder_receipt,
    soft_polarity_simplex_response_q,
    validate_soft_polarity_simplex_response_fit_receipt,
    validate_soft_polarity_simplex_response_inner_oof_selection_receipt,
    validate_soft_polarity_simplex_response_ladder_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]
EXPECTED_PROTOCOL_SHA256 = (
    "c9e269207e2c7824fad12c934e1e619d921ae69d439938f4124a29d1a53b934f"
)


def _objectives(target_index: int = 12) -> dict[str, dict[str, float]]:
    return {
        family: {
            candidate_id: (
                0.01
                + (index - target_index) ** 2 * 0.001
                + family_index * 0.0001
            )
            for index, candidate_id in enumerate(
                SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS
            )
        }
        for family_index, family in enumerate(INNER)
    }


def _selection(
    objectives: dict[str, dict[str, float]] | None = None,
):
    ladder = build_soft_polarity_simplex_response_ladder_receipt()
    scores = _objectives() if objectives is None else objectives
    selection = build_soft_polarity_simplex_response_inner_oof_selection_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=scores,
    )
    return ladder, scores, selection


def _rehash(
    receipt: dict[str, object], *, domain: bytes
) -> dict[str, object]:
    result = copy.deepcopy(receipt)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = core._hash(domain, result)
    return result


def _all_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            item
            for key, nested in value.items()
            for item in (*_all_strings(key), *_all_strings(nested))
        )
    if isinstance(value, (tuple, list)):
        return tuple(item for nested in value for item in _all_strings(nested))
    return ()


def _tied(indices: tuple[int, ...]) -> dict[str, dict[str, float]]:
    selected = set(indices)
    return {
        family: {
            candidate_id: (0.0 if index in selected else 1.0)
            for index, candidate_id in enumerate(
                SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS
            )
        }
        for family in INNER
    }


def test_fixed_nineteen_candidate_ladder_is_authenticated() -> None:
    expected = (
        (0.0, 0.0, 0.0),
        (1.0 / 8.0, 0.0, 0.0),
        (1.0 / 8.0, 1.0 / 8.0, -1.0 / 8.0),
        (1.0 / 8.0, 1.0 / 8.0, -1.0 / 16.0),
        (1.0 / 8.0, 1.0 / 8.0, 0.0),
        (1.0 / 8.0, 1.0 / 8.0, 1.0 / 16.0),
        (1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0),
        (1.0 / 8.0, 1.0 / 4.0, -1.0 / 8.0),
        (1.0 / 8.0, 1.0 / 4.0, 0.0),
        (1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0),
        (1.0 / 4.0, 0.0, 0.0),
        (1.0 / 4.0, 1.0 / 8.0, -1.0 / 8.0),
        (1.0 / 4.0, 1.0 / 8.0, -1.0 / 16.0),
        (1.0 / 4.0, 1.0 / 8.0, 0.0),
        (1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0),
        (1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0),
        (1.0 / 4.0, 1.0 / 4.0, -1.0 / 8.0),
        (1.0 / 4.0, 1.0 / 4.0, 0.0),
        (1.0 / 4.0, 1.0 / 4.0, 1.0 / 8.0),
    )
    assert SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER == expected
    assert SOFT_POLARITY_SIMPLEX_RESPONSE_TRIPLES == expected
    assert SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS == tuple(
        f"simplex_response_{index:02d}" for index in range(19)
    )
    assert SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256 == (
        EXPECTED_PROTOCOL_SHA256
    )

    ladder = build_soft_polarity_simplex_response_ladder_receipt()
    validate_soft_polarity_simplex_response_ladder_receipt(ladder)
    validate_soft_polarity_simplex_response_ladder_receipt(
        json.loads(json.dumps(ladder))
    )
    assert ladder["candidate_count"] == 19
    assert ladder["zero_control_count"] == 1
    assert tuple(
        tuple(triple) for triple in ladder["candidate_triples_r_u_v"]
    ) == expected
    for index, candidate in enumerate(ladder["candidate_receipts"]):
        r, u, v = expected[index]
        assert candidate["candidate_id"] == f"simplex_response_{index:02d}"
        assert (candidate["r"], candidate["u"], candidate["v"]) == (r, u, v)
        assert candidate["ladder_index"] == index
        assert candidate["zero_control"] is (index == 0)
        assert candidate["coefficient_constraints_satisfied"] is True
        assert candidate["formula"] == (
            "q(z)=(1-u*z^2)*tanh(r*z)+v*z^2"
        )


def test_q_is_exact_continuous_simplex_formula_and_bounded_on_unit_interval() -> None:
    for r, u, v in SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER:
        for z in (-1.0, -0.25, 0.0, 0.25, 1.0):
            expected = (1.0 - u * z**2) * math.tanh(r * z) + v * z**2
            actual = soft_polarity_simplex_response_q(
                z, r=r, u=u, v=v
            )
            assert actual == pytest.approx(expected)
            assert -1.0 <= actual <= 1.0
    assert soft_polarity_simplex_response_q(
        0.0, r=0.25, u=0.5, v=0.5
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs,expected",
    (
        ({"z": math.nan, "r": 0.0, "u": 0.0, "v": 0.0}, "finite"),
        ({"z": 1.0, "r": math.inf, "u": 0.0, "v": 0.0}, "finite"),
        ({"z": True, "r": 0.0, "u": 0.0, "v": 0.0}, "numeric"),
        ({"z": 1.0, "r": -0.1, "u": 0.0, "v": 0.0}, "nonnegative"),
        ({"z": 1.0, "r": 0.1, "u": -0.1, "v": 0.0}, "inside"),
        ({"z": 1.0, "r": 0.1, "u": 0.5001, "v": 0.0}, "inside"),
        ({"z": 1.0, "r": 0.1, "u": 0.1, "v": 0.1001}, "abs\\(v\\)<=u"),
        ({"z": 1.0, "r": 0.1, "u": 0.1, "v": -0.1001}, "abs\\(v\\)<=u"),
        ({"z": 1.0, "r": 0.1, "u": True, "v": 0.0}, "numeric"),
        (
            {
                "z": float.fromhex("0x1.fffffffffffffp+1023"),
                "r": 0.1,
                "u": 0.5,
                "v": 0.5,
            },
            "intermediate must be finite",
        ),
    ),
)
def test_q_rejects_invalid_coefficients_and_nontotal_inputs(
    kwargs: dict[str, float], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        soft_polarity_simplex_response_q(**kwargs)


def test_inner_selection_is_family_equal_exact_kl_and_replayable() -> None:
    ladder, objectives, selection = _selection()
    validate_soft_polarity_simplex_response_inner_oof_selection_receipt(
        selection,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    validate_soft_polarity_simplex_response_fit_receipt(
        json.loads(json.dumps(selection)),
        ladder_receipt=json.loads(json.dumps(ladder)),
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    assert build_soft_polarity_simplex_response_fit_receipt is (
        build_soft_polarity_simplex_response_inner_oof_selection_receipt
    )
    target = "simplex_response_12"
    assert selection["selected_candidate_id"] == target
    assert selection["selected_ladder_index"] == 12
    assert selection["selected_r"] == 1.0 / 4.0
    assert selection["selected_u"] == 1.0 / 8.0
    assert selection["selected_v"] == -1.0 / 16.0
    expected = math.fsum(objectives[family][target] for family in INNER) / 7
    assert selection["selected_family_equal_exact_kl"] == pytest.approx(expected)
    assert tuple(selection["inner_oof_family_order"]) == INNER
    assert OUTER not in selection["exact_kl_objective_by_family_and_candidate"]


def test_ties_follow_u_abs_v_r_index_and_hash_order() -> None:
    ladder = build_soft_polarity_simplex_response_ladder_receipt()

    cases = (
        ((2, 7), "simplex_response_02"),  # smaller u
        ((2, 3), "simplex_response_03"),  # smaller |v|
        ((2, 11), "simplex_response_02"),  # smaller r
        ((2, 6), "simplex_response_02"),  # fixed index before hash
    )
    for indices, expected in cases:
        selected = build_soft_polarity_simplex_response_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=_tied(indices),
        )
        assert selected["selected_candidate_id"] == expected

    all_tied = _tied(tuple(range(19)))
    selected = build_soft_polarity_simplex_response_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=all_tied,
    )
    assert selected["selected_candidate_id"] == "simplex_response_00"
    assert (
        selected["selected_r"],
        selected["selected_u"],
        selected["selected_v"],
    ) == (0.0, 0.0, 0.0)
    assert selected["selection_key"] == (
        "family_equal_token_mean_exact_float64_full_vocabulary_"
        "kl_teacher_to_candidate_then_smaller_u_then_smaller_abs_v_then_"
        "smaller_r_then_fixed_ladder_index_then_candidate_artifact_sha256"
    )


def test_selection_commits_geometry_and_scientific_boundary() -> None:
    ladder, objectives, selection = _selection()
    assert set(selection["family_oof_receipts"]) == set(INNER)
    assert set(selection["family_equal_exact_kl_by_candidate"]) == set(
        SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS
    )
    for family in INNER:
        receipt = selection["family_oof_receipts"][family]
        assert receipt["exact_kl_objective_by_candidate"] == objectives[family]
        assert receipt["conditional_inner_leave_one_family_out"] is True
        assert receipt["fixed_seven_family_endpoint_shared_across_inner_scores"] is True
        assert receipt["inner_held_family_used_for_fixed_endpoint_fit"] is True
        assert receipt["outer_held_family_used"] is False
        assert receipt["exact_float64_execution"] is True
        assert receipt["full_vocabulary_evaluated"] is True
        assert receipt["raw_model_tensors_serialized"] is False
    assert selection["inner_endpoint_refit_per_inner_held_family"] is False
    assert selection["fully_nested_endpoint_selection_claimed"] is False
    assert selection["outer_held_family_used_for_fit_or_selection"] is False
    assert selection["compression_claim_authorized"] is False
    assert selection["speed_claim_authorized"] is False
    assert selection["serving_authorized"] is False
    assert selection["data_boundary"]["calibration_b_opened"] is False
    assert selection["data_boundary"]["validation_opened"] is False
    assert selection["data_boundary"]["test_opened"] is False
    assert selection["ladder_artifact_sha256"] == ladder["artifact_sha256"]


def test_protocol_and_receipts_never_mislabel_exact_kl_as_nll() -> None:
    ladder, _, selection = _selection()
    strings = "\n".join(
        (*_all_strings(core._PROTOCOL), *_all_strings(ladder), *_all_strings(selection))
    ).lower()
    assert ("negative_log_" + "likelihood") not in strings
    assert ("negative log " + "likelihood") not in strings
    assert ("n" + "ll") not in strings
    assert "kl_teacher_to_candidate" in strings
    assert "full_vocabulary" in strings
    assert "exact_float64" in strings


@pytest.mark.parametrize("missing_kind", ("family", "candidate"))
def test_selection_rejects_incomplete_or_extra_exact_geometry(
    missing_kind: str,
) -> None:
    ladder = build_soft_polarity_simplex_response_ladder_receipt()
    objectives = _objectives()
    if missing_kind == "family":
        objectives.pop(INNER[0])
        expected = "family geometry"
    else:
        objectives[INNER[0]].pop(SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS[0])
        expected = "candidate geometry"
    with pytest.raises(ValueError, match=expected):
        build_soft_polarity_simplex_response_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )

    extra = _objectives()
    extra[INNER[0]]["not_a_candidate"] = 1.0
    with pytest.raises(ValueError, match="candidate geometry"):
        build_soft_polarity_simplex_response_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=extra,
        )


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, True, "1"))
def test_selection_rejects_nonfinite_or_nonnumeric_objectives(
    bad_value: object,
) -> None:
    ladder = build_soft_polarity_simplex_response_ladder_receipt()
    objectives = _objectives()
    objectives[INNER[0]][SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS[0]] = bad_value
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        build_soft_polarity_simplex_response_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_ladder_rejects_rehashed_candidate_drift_and_unknown_fields() -> None:
    ladder = build_soft_polarity_simplex_response_ladder_receipt()
    forged = copy.deepcopy(ladder)
    candidate = forged["candidate_receipts"][0]
    candidate["u"] = 0.125
    forged["candidate_receipts"] = (
        _rehash(candidate, domain=core._CANDIDATE_DOMAIN),
        *forged["candidate_receipts"][1:],
    )
    forged["candidate_artifact_sha256s"]["simplex_response_00"] = forged[
        "candidate_receipts"
    ][0]["artifact_sha256"]
    forged = _rehash(forged, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt content drifted"):
        validate_soft_polarity_simplex_response_ladder_receipt(forged)

    unknown = copy.deepcopy(ladder)
    candidate = unknown["candidate_receipts"][0]
    candidate["unknown"] = False
    unknown["candidate_receipts"] = (
        _rehash(candidate, domain=core._CANDIDATE_DOMAIN),
        *unknown["candidate_receipts"][1:],
    )
    unknown = _rehash(unknown, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt key set drifted"):
        validate_soft_polarity_simplex_response_ladder_receipt(unknown)


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("selection_frozen_before_outer_held_score", False),
        ("inner_endpoint_refit_per_inner_held_family", True),
        ("fully_nested_endpoint_selection_claimed", True),
        ("outer_held_family_used_for_fit_or_selection", True),
        ("all_objectives_token_mean_exact_float64_full_vocabulary_kl", False),
        ("compression_claim_authorized", True),
        ("speed_claim_authorized", True),
        ("serving_authorized", True),
    ),
)
def test_rehashed_selection_cannot_forge_boundary_or_authority_flags(
    field: str, bad_value: bool
) -> None:
    ladder, objectives, selection = _selection()
    forged = copy.deepcopy(selection)
    forged[field] = bad_value
    forged = _rehash(forged, domain=core._SELECTION_DOMAIN)
    with pytest.raises(ValueError, match="selection receipt content drifted"):
        validate_soft_polarity_simplex_response_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_fully_rehashed_objective_forgery_fails_authoritative_replay() -> None:
    ladder, objectives, _ = _selection()
    forged_objectives = copy.deepcopy(objectives)
    forged_objectives[INNER[0]]["simplex_response_00"] = -100.0
    forged = build_soft_polarity_simplex_response_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=forged_objectives,
    )
    assert forged["selected_candidate_id"] == "simplex_response_00"
    with pytest.raises(ValueError, match="selection receipt content drifted"):
        validate_soft_polarity_simplex_response_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_public_selection_api_has_no_outer_score_or_later_split_surface() -> None:
    for function in (
        build_soft_polarity_simplex_response_fit_receipt,
        validate_soft_polarity_simplex_response_fit_receipt,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters.intersection(
            {
                "outer_held_objectives",
                "held_objectives",
                "calibration_b",
                "validation",
                "test",
                "prompts",
                "logits",
                "h4",
                "serving",
            }
        )
