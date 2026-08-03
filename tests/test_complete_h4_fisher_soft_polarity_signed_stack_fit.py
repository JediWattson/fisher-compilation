from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_signed_stack_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_signed_stack_fit import (
    SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS,
    SOFT_POLARITY_SIGNED_STACK_FIT_PROTOCOL_SHA256,
    SOFT_POLARITY_SIGNED_STACK_LADDER,
    SOFT_POLARITY_SIGNED_STACK_PAIRS,
    build_soft_polarity_signed_stack_fit_receipt,
    build_soft_polarity_signed_stack_inner_oof_selection_receipt,
    build_soft_polarity_signed_stack_ladder_receipt,
    soft_polarity_signed_stack_q,
    validate_soft_polarity_signed_stack_fit_receipt,
    validate_soft_polarity_signed_stack_inner_oof_selection_receipt,
    validate_soft_polarity_signed_stack_ladder_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]
EXPECTED_PROTOCOL_SHA256 = (
    "952b9a7ac1704113597efe5612686c05178c0700f5070aeb904d49dbdab6b390"
)


def _objectives(target_index: int = 9) -> dict[str, dict[str, float]]:
    return {
        family: {
            candidate_id: (
                0.01
                + (index - target_index) ** 2 * 0.001
                + family_index * 0.0001
            )
            for index, candidate_id in enumerate(
                SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
            )
        }
        for family_index, family in enumerate(INNER)
    }


def _selection(
    objectives: dict[str, dict[str, float]] | None = None,
):
    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    scores = _objectives() if objectives is None else objectives
    selection = build_soft_polarity_signed_stack_inner_oof_selection_receipt(
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


def test_fixed_ladder_protocol_and_candidate_receipts_are_authenticated() -> None:
    assert SOFT_POLARITY_SIGNED_STACK_LADDER == (
        (0.0, 0.0),
        (1.0 / 8.0, 0.0),
        (1.0 / 8.0, -1.0 / 8.0),
        (1.0 / 8.0, 1.0 / 8.0),
        (1.0 / 8.0, -1.0 / 4.0),
        (1.0 / 8.0, 1.0 / 4.0),
        (1.0 / 8.0, -1.0 / 2.0),
        (1.0 / 8.0, 1.0 / 2.0),
        (1.0 / 4.0, 0.0),
        (1.0 / 4.0, -1.0 / 8.0),
        (1.0 / 4.0, 1.0 / 8.0),
        (1.0 / 4.0, -1.0 / 4.0),
        (1.0 / 4.0, 1.0 / 4.0),
        (1.0 / 4.0, -1.0 / 2.0),
        (1.0 / 4.0, 1.0 / 2.0),
    )
    assert SOFT_POLARITY_SIGNED_STACK_PAIRS == SOFT_POLARITY_SIGNED_STACK_LADDER
    assert SOFT_POLARITY_SIGNED_STACK_FIT_PROTOCOL_SHA256 == (
        EXPECTED_PROTOCOL_SHA256
    )
    assert SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS == tuple(
        f"signed_stack_{index:02d}" for index in range(15)
    )

    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    validate_soft_polarity_signed_stack_ladder_receipt(ladder)
    validate_soft_polarity_signed_stack_ladder_receipt(
        json.loads(json.dumps(ladder))
    )
    assert tuple(ladder["candidate_order"]) == (
        SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
    )
    assert tuple(
        tuple(pair) for pair in ladder["candidate_pairs_r_s"]
    ) == SOFT_POLARITY_SIGNED_STACK_LADDER
    assert ladder["candidate_count"] == 15
    assert ladder["zero_control_count"] == 1
    assert (
        ladder[
            "ladder_frozen_before_any_conditional_inner_lofo_objective"
        ]
        is True
    )
    assert ladder["outer_held_objectives_consumed_before_freeze"] is False
    assert ladder["raw_model_tensors_serialized"] is False

    for index, candidate in enumerate(ladder["candidate_receipts"]):
        r, s = SOFT_POLARITY_SIGNED_STACK_LADDER[index]
        assert candidate["candidate_id"] == (
            SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS[index]
        )
        assert candidate["ladder_index"] == index
        assert candidate["r"] == r
        assert candidate["s"] == s
        assert candidate["zero_control"] is (index == 0)
        assert candidate["formula"] == (
            "q(z)=(1-abs(s)*z^2)*tanh(r*z)+s*z^2"
        )
        assert len(candidate["artifact_sha256"]) == 64


def test_q_is_the_exact_continuous_signed_stack_formula() -> None:
    for r, s in SOFT_POLARITY_SIGNED_STACK_LADDER:
        for z in (-2.0, -0.25, 0.0, 0.25, 2.0):
            expected = (
                (1.0 - abs(s) * z**2) * math.tanh(r * z)
                + s * z**2
            )
            actual = soft_polarity_signed_stack_q(z, r=r, s=s)
            assert actual == pytest.approx(expected)
            assert soft_polarity_signed_stack_q(
                -z, r=r, s=-s
            ) == pytest.approx(-actual)
    assert soft_polarity_signed_stack_q(0.0, r=0.25, s=0.5) == 0.0


@pytest.mark.parametrize(
    "kwargs,expected",
    (
        ({"z": math.nan, "r": 0.0, "s": 0.0}, "must be finite"),
        ({"z": 1.0, "r": math.inf, "s": 0.0}, "must be finite"),
        ({"z": True, "r": 0.0, "s": 0.0}, "must be numeric"),
        ({"z": 1.0, "r": -0.1, "s": 0.0}, "nonnegative"),
        ({"z": 1.0, "r": 0.1, "s": -0.5001}, "inside"),
        ({"z": 1.0, "r": 0.1, "s": 0.5001}, "inside"),
        ({"z": 1.0, "r": 0.1, "s": True}, "must be numeric"),
        (
            {"z": float.fromhex("0x1.fffffffffffffp+1023"), "r": 0.1, "s": 0.5},
            "intermediate must be finite",
        ),
    ),
)
def test_q_rejects_invalid_or_nontotal_inputs(
    kwargs: dict[str, float], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        soft_polarity_signed_stack_q(**kwargs)


def test_inner_selection_is_family_equal_exact_kl_and_replayable() -> None:
    ladder, objectives, selection = _selection()
    validate_soft_polarity_signed_stack_inner_oof_selection_receipt(
        selection,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    validate_soft_polarity_signed_stack_inner_oof_selection_receipt(
        json.loads(json.dumps(selection)),
        ladder_receipt=json.loads(json.dumps(ladder)),
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    validate_soft_polarity_signed_stack_fit_receipt(
        selection,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    assert build_soft_polarity_signed_stack_fit_receipt is (
        build_soft_polarity_signed_stack_inner_oof_selection_receipt
    )
    target = SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS[9]
    assert selection["selected_candidate_id"] == target
    assert selection["selected_ladder_index"] == 9
    assert selection["selected_r"] == 1.0 / 4.0
    assert selection["selected_s"] == -1.0 / 8.0
    expected = math.fsum(objectives[family][target] for family in INNER) / 7
    assert selection["selected_family_equal_exact_kl"] == pytest.approx(
        expected
    )
    assert tuple(selection["inner_oof_family_order"]) == INNER
    assert OUTER not in selection["exact_kl_objective_by_family_and_candidate"]
    assert selection["objective_kind"] == (
        "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
    )
    assert selection["aggregate_objective_kind"] == (
        "family_equal_token_mean_exact_float64_full_vocabulary_"
        "kl_teacher_to_candidate"
    )


def test_ties_prefer_smaller_abs_s_then_smaller_r_then_fixed_order() -> None:
    ladder = build_soft_polarity_signed_stack_ladder_receipt()

    abs_s_tie = {
        family: {
            candidate_id: (0.0 if index in (2, 4) else 1.0)
            for index, candidate_id in enumerate(
                SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
            )
        }
        for family in INNER
    }
    selected = build_soft_polarity_signed_stack_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=abs_s_tie,
    )
    assert selected["selected_candidate_id"] == "signed_stack_02"
    assert selected["selected_s"] == -0.125
    assert selected["selected_r"] == 0.125

    r_tie = copy.deepcopy(abs_s_tie)
    for family in INNER:
        for candidate_id in SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS:
            r_tie[family][candidate_id] = 1.0
        r_tie[family]["signed_stack_02"] = 0.0
        r_tie[family]["signed_stack_09"] = 0.0
    selected = build_soft_polarity_signed_stack_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=r_tie,
    )
    assert selected["selected_candidate_id"] == "signed_stack_02"
    assert selected["selected_s"] == -0.125
    assert selected["selected_r"] == 0.125

    index_tie = copy.deepcopy(r_tie)
    for family in INNER:
        for candidate_id in SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS:
            index_tie[family][candidate_id] = 1.0
        index_tie[family]["signed_stack_02"] = 0.0
        index_tie[family]["signed_stack_03"] = 0.0
    selected = build_soft_polarity_signed_stack_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=index_tie,
    )
    assert selected["selected_candidate_id"] == "signed_stack_02"
    assert selected["selected_s"] == -0.125

    all_tied = {
        family: {
            candidate_id: 1.0
            for candidate_id in SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
        }
        for family in INNER
    }
    selected = build_soft_polarity_signed_stack_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=all_tied,
    )
    assert selected["selected_candidate_id"] == "signed_stack_00"
    assert selected["selected_r"] == 0.0
    assert selected["selected_s"] == 0.0


def test_selection_commits_geometry_and_scientific_boundary() -> None:
    ladder, objectives, selection = _selection()
    assert set(selection["family_oof_receipts"]) == set(INNER)
    assert set(selection["family_oof_receipt_sha256s"]) == set(INNER)
    assert set(selection["family_equal_exact_kl_by_candidate"]) == set(
        SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
    )
    assert set(selection["aggregate_artifact_sha256_by_candidate"]) == set(
        SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS
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
        assert selection["family_oof_receipt_sha256s"][family] == (
            receipt["artifact_sha256"]
        )
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


def test_protocol_and_receipts_never_mislabel_the_exact_kl_objective() -> None:
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
    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    objectives = _objectives()
    if missing_kind == "family":
        objectives.pop(INNER[0])
        expected = "family geometry"
    else:
        objectives[INNER[0]].pop(SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS[0])
        expected = "candidate geometry"
    with pytest.raises(ValueError, match=expected):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )

    extra = _objectives()
    extra[INNER[0]]["not_a_candidate"] = 1.0
    with pytest.raises(ValueError, match="candidate geometry"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=extra,
        )


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, True, "1"))
def test_selection_rejects_nonfinite_or_nonnumeric_objectives(
    bad_value: object,
) -> None:
    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    objectives = _objectives()
    objectives[INNER[0]][SOFT_POLARITY_SIGNED_STACK_CANDIDATE_IDS[0]] = bad_value
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_selection_requires_exact_outer_and_conditional_inner_geometry() -> None:
    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    with pytest.raises(ValueError, match="exactly eight"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES[:-1],
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=_objectives(),
        )
    with pytest.raises(ValueError, match="outer held family"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id="unknown_family",
            exact_objectives_by_family_and_candidate=_objectives(),
        )
    with pytest.raises(ValueError, match="canonical nonempty"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=(*FAMILIES[:-1], " bad "),
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=_objectives(),
        )
    leaked = _objectives()
    leaked[OUTER] = leaked.pop(INNER[0])
    with pytest.raises(ValueError, match="family geometry"):
        build_soft_polarity_signed_stack_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=leaked,
        )


def test_ladder_rejects_rehashed_candidate_drift_and_unknown_fields() -> None:
    ladder = build_soft_polarity_signed_stack_ladder_receipt()
    forged = copy.deepcopy(ladder)
    candidate = forged["candidate_receipts"][0]
    candidate["r"] = 0.125
    forged["candidate_receipts"] = (
        _rehash(candidate, domain=core._CANDIDATE_DOMAIN),
        *forged["candidate_receipts"][1:],
    )
    forged["candidate_artifact_sha256s"]["signed_stack_00"] = forged[
        "candidate_receipts"
    ][0]["artifact_sha256"]
    forged = _rehash(forged, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt content drifted"):
        validate_soft_polarity_signed_stack_ladder_receipt(forged)

    unknown = copy.deepcopy(ladder)
    candidate = unknown["candidate_receipts"][0]
    candidate["unknown"] = False
    unknown["candidate_receipts"] = (
        _rehash(candidate, domain=core._CANDIDATE_DOMAIN),
        *unknown["candidate_receipts"][1:],
    )
    unknown = _rehash(unknown, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt key set drifted"):
        validate_soft_polarity_signed_stack_ladder_receipt(unknown)


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
        validate_soft_polarity_signed_stack_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_fully_rehashed_objective_forgery_cannot_bypass_authoritative_replay() -> None:
    ladder, objectives, _ = _selection()
    forged_objectives = copy.deepcopy(objectives)
    forged_objectives[INNER[0]]["signed_stack_00"] = -100.0
    forged = build_soft_polarity_signed_stack_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=forged_objectives,
    )
    assert forged["selected_candidate_id"] == "signed_stack_00"
    with pytest.raises(ValueError, match="selection receipt content drifted"):
        validate_soft_polarity_signed_stack_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_rehashed_unknown_selection_field_is_rejected() -> None:
    ladder, objectives, selection = _selection()
    forged = copy.deepcopy(selection)
    forged["unknown"] = False
    forged = _rehash(forged, domain=core._SELECTION_DOMAIN)
    with pytest.raises(ValueError, match="selection receipt key set drifted"):
        validate_soft_polarity_signed_stack_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_public_selection_api_has_no_outer_score_or_later_split_surface() -> None:
    for function in (
        build_soft_polarity_signed_stack_fit_receipt,
        validate_soft_polarity_signed_stack_fit_receipt,
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
