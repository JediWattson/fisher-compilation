from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_confidence_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_confidence_fit import (
    SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
    SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256,
    SOFT_POLARITY_CONFIDENCE_LADDER,
    SOFT_POLARITY_CONFIDENCE_PAIRS,
    build_soft_polarity_confidence_fit_receipt,
    build_soft_polarity_confidence_inner_oof_selection_receipt,
    build_soft_polarity_confidence_ladder_receipt,
    soft_polarity_confidence_q,
    validate_soft_polarity_confidence_fit_receipt,
    validate_soft_polarity_confidence_inner_oof_selection_receipt,
    validate_soft_polarity_confidence_ladder_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]
EXPECTED_PROTOCOL_SHA256 = (
    "7dcd1300cf3433400283bf1b241d28175aad84853cb2734d9e3f88c05d4e97f2"
)


def _objectives(target_index: int = 5) -> dict[str, dict[str, float]]:
    return {
        family: {
            candidate_id: (
                1.0
                + (index - target_index) ** 2 * 0.01
                + family_index * 0.001
            )
            for index, candidate_id in enumerate(
                SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
            )
        }
        for family_index, family in enumerate(INNER)
    }


def _selection(
    objectives: dict[str, dict[str, float]] | None = None,
):
    ladder = build_soft_polarity_confidence_ladder_receipt()
    scores = _objectives() if objectives is None else objectives
    selection = build_soft_polarity_confidence_inner_oof_selection_receipt(
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


def test_fixed_ladder_protocol_and_candidate_receipts_are_authenticated() -> None:
    assert SOFT_POLARITY_CONFIDENCE_LADDER == (
        (0.0, 0.0),
        (1.0 / 8.0, 0.0),
        (1.0 / 4.0, 0.0),
        (0.0, 1.0 / 2.0),
        (1.0 / 8.0, 1.0 / 2.0),
        (1.0 / 4.0, 1.0 / 2.0),
        (0.0, 2.0),
        (1.0 / 8.0, 2.0),
        (1.0 / 4.0, 2.0),
        (0.0, 8.0),
        (1.0 / 8.0, 8.0),
        (1.0 / 4.0, 8.0),
    )
    assert SOFT_POLARITY_CONFIDENCE_PAIRS == SOFT_POLARITY_CONFIDENCE_LADDER
    assert SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256 == (
        EXPECTED_PROTOCOL_SHA256
    )
    ladder = build_soft_polarity_confidence_ladder_receipt()
    validate_soft_polarity_confidence_ladder_receipt(ladder)
    validate_soft_polarity_confidence_ladder_receipt(
        json.loads(json.dumps(ladder))
    )

    assert tuple(ladder["candidate_order"]) == (
        SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
    )
    assert tuple(tuple(pair) for pair in ladder["candidate_pairs"]) == (
        SOFT_POLARITY_CONFIDENCE_LADDER
    )
    assert ladder["candidate_count"] == 12
    assert ladder["ladder_frozen_before_any_inner_oof_objective"] is True
    assert ladder["outer_held_objectives_consumed_before_freeze"] is False
    assert ladder["raw_objectives_or_model_tensors_serialized"] is False
    for index, candidate in enumerate(ladder["candidate_receipts"]):
        a, b = SOFT_POLARITY_CONFIDENCE_LADDER[index]
        assert candidate["candidate_id"] == (
            SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS[index]
        )
        assert candidate["ladder_index"] == index
        assert candidate["a"] == a
        assert candidate["b"] == b
        assert candidate["a_plus_b"] == math.fsum((a, b))
        assert candidate["formula"] == "q(z)=tanh(a*z+b*z^3)"
        assert len(candidate["artifact_sha256"]) == 64


def test_q_is_the_exact_continuous_odd_confidence_formula() -> None:
    for a, b in SOFT_POLARITY_CONFIDENCE_LADDER:
        for z in (-2.0, -0.25, 0.0, 0.25, 2.0):
            expected = math.tanh(a * z + b * z**3)
            assert soft_polarity_confidence_q(z, a=a, b=b) == pytest.approx(
                expected
            )
            assert soft_polarity_confidence_q(-z, a=a, b=b) == pytest.approx(
                -soft_polarity_confidence_q(z, a=a, b=b)
            )
    assert soft_polarity_confidence_q(0.0, a=0.25, b=8.0) == 0.0
    with pytest.raises(ValueError, match="must be finite"):
        soft_polarity_confidence_q(math.nan, a=0.0, b=0.0)
    with pytest.raises(ValueError, match="intermediate must be finite"):
        soft_polarity_confidence_q(1.0e308, a=0.25, b=8.0)
    with pytest.raises(TypeError, match="must be numeric"):
        soft_polarity_confidence_q(True, a=0.0, b=0.0)
    with pytest.raises(ValueError, match="must be nonnegative"):
        soft_polarity_confidence_q(1.0, a=-0.1, b=0.0)


def test_inner_oof_selection_is_family_equal_exact_and_uses_full_tie_key() -> None:
    ladder, objectives, selection = _selection()
    validate_soft_polarity_confidence_inner_oof_selection_receipt(
        selection,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    validate_soft_polarity_confidence_inner_oof_selection_receipt(
        json.loads(json.dumps(selection)),
        ladder_receipt=json.loads(json.dumps(ladder)),
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    validate_soft_polarity_confidence_fit_receipt(
        selection,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    assert build_soft_polarity_confidence_fit_receipt is (
        build_soft_polarity_confidence_inner_oof_selection_receipt
    )
    target = SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS[5]
    assert selection["selected_candidate_id"] == target
    assert selection["selected_ladder_index"] == 5
    assert selection["selected_a"] == 1.0 / 4.0
    assert selection["selected_b"] == 1.0 / 2.0
    expected = math.fsum(objectives[family][target] for family in INNER) / 7
    assert selection["selected_family_equal_exact_nll"] == pytest.approx(
        expected
    )
    assert tuple(selection["inner_oof_family_order"]) == INNER
    assert OUTER not in selection["exact_objective_by_family_and_candidate"]
    assert selection["selection_key"] == (
        "family_equal_exact_nll_then_a_plus_b_then_b_then_ladder_index_"
        "then_candidate_artifact_sha256"
    )

    tied = {
        family: {
            candidate_id: 1.0
            for candidate_id in SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
        }
        for family in INNER
    }
    tied_selection = build_soft_polarity_confidence_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=tied,
    )
    assert tied_selection["selected_candidate_id"] == "confidence_00"
    assert tied_selection["selected_a"] == 0.0
    assert tied_selection["selected_b"] == 0.0


def test_selection_commits_every_family_candidate_and_freeze_boundary() -> None:
    ladder, objectives, selection = _selection()
    assert set(selection["family_oof_receipts"]) == set(INNER)
    assert set(selection["family_oof_receipt_sha256s"]) == set(INNER)
    assert set(selection["family_equal_exact_nll_by_candidate"]) == set(
        SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
    )
    assert set(selection["aggregate_artifact_sha256_by_candidate"]) == set(
        SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
    )
    for family in INNER:
        family_receipt = selection["family_oof_receipts"][family]
        assert tuple(family_receipt["candidate_order"]) == (
            SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
        )
        assert family_receipt["exact_objective_by_candidate"] == (
            objectives[family]
        )
        assert family_receipt["all_candidates_frozen_before_family_score"] is True
        assert family_receipt["outer_held_family_used"] is False
        assert family_receipt["exact_execution"] is True
        assert selection["family_oof_receipt_sha256s"][family] == (
            family_receipt["artifact_sha256"]
        )
        assert all(
            len(value) == 64
            for value in family_receipt[
                "exact_objective_sha256_by_candidate"
            ].values()
        )
    assert selection[
        "candidate_ladder_frozen_before_any_inner_oof_objective"
    ] is True
    assert selection["selection_frozen_before_outer_held_score"] is True
    assert selection["outer_held_family_used_for_fit_or_selection"] is False
    assert selection["raw_objectives_or_model_tensors_serialized"] is False
    assert selection["serving_authorized"] is False
    assert selection["data_boundary"]["calibration_b_opened"] is False
    assert selection["data_boundary"]["validation_opened"] is False
    assert selection["data_boundary"]["test_opened"] is False
    assert selection["ladder_artifact_sha256"] == ladder["artifact_sha256"]


@pytest.mark.parametrize("missing_kind", ("family", "candidate"))
def test_selection_rejects_incomplete_or_extra_exact_geometry(
    missing_kind: str,
) -> None:
    ladder = build_soft_polarity_confidence_ladder_receipt()
    objectives = _objectives()
    if missing_kind == "family":
        objectives.pop(INNER[0])
        expected = "family geometry"
    else:
        objectives[INNER[0]].pop(SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS[0])
        expected = "candidate geometry"
    with pytest.raises(ValueError, match=expected):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )

    extra = _objectives()
    extra[INNER[0]]["not_a_candidate"] = 1.0
    with pytest.raises(ValueError, match="candidate geometry"):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=extra,
        )


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, True))
def test_selection_rejects_nonfinite_or_nonnumeric_objectives(
    bad_value: float | bool,
) -> None:
    ladder = build_soft_polarity_confidence_ladder_receipt()
    objectives = _objectives()
    objectives[INNER[0]][SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS[0]] = bad_value
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_selection_requires_exact_eight_outer_seven_inner_geometry() -> None:
    ladder = build_soft_polarity_confidence_ladder_receipt()
    with pytest.raises(ValueError, match="exactly eight"):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES[:-1],
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=_objectives(),
        )
    with pytest.raises(ValueError, match="outer held family"):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id="unknown_family",
            exact_objectives_by_family_and_candidate=_objectives(),
        )
    leaked = _objectives()
    leaked[OUTER] = leaked.pop(INNER[0])
    with pytest.raises(ValueError, match="family geometry"):
        build_soft_polarity_confidence_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=leaked,
        )


def test_ladder_rejects_rehashed_candidate_drift_and_unknown_fields() -> None:
    ladder = build_soft_polarity_confidence_ladder_receipt()
    forged = copy.deepcopy(ladder)
    candidate = forged["candidate_receipts"][0]
    candidate["a"] = 0.125
    forged["candidate_receipts"] = (
        _rehash(candidate, domain=core._CANDIDATE_DOMAIN),
        *forged["candidate_receipts"][1:],
    )
    forged["candidate_artifact_sha256s"]["confidence_00"] = forged[
        "candidate_receipts"
    ][0]["artifact_sha256"]
    forged = _rehash(forged, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt content drifted"):
        validate_soft_polarity_confidence_ladder_receipt(forged)

    unknown = copy.deepcopy(ladder)
    unknown_candidate = unknown["candidate_receipts"][0]
    unknown_candidate["unknown"] = False
    unknown["candidate_receipts"] = (
        _rehash(unknown_candidate, domain=core._CANDIDATE_DOMAIN),
        *unknown["candidate_receipts"][1:],
    )
    unknown = _rehash(unknown, domain=core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate receipt key set drifted"):
        validate_soft_polarity_confidence_ladder_receipt(unknown)


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("selection_frozen_before_outer_held_score", False),
        ("outer_held_family_used_for_fit_or_selection", True),
        ("all_objectives_exact", False),
        ("serving_authorized", True),
    ),
)
def test_rehashed_selection_cannot_forge_freeze_or_no_held_flags(
    field: str, bad_value: bool
) -> None:
    ladder, objectives, selection = _selection()
    forged = copy.deepcopy(selection)
    forged[field] = bad_value
    forged = _rehash(forged, domain=core._SELECTION_DOMAIN)
    with pytest.raises(ValueError, match="selection receipt content drifted"):
        validate_soft_polarity_confidence_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_fully_rehashed_objective_forgery_cannot_bypass_authoritative_replay() -> None:
    ladder, objectives, _ = _selection()
    forged_objectives = copy.deepcopy(objectives)
    forged_objectives[INNER[0]]["confidence_00"] = -100.0
    forged = build_soft_polarity_confidence_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=forged_objectives,
    )
    assert forged["selected_candidate_id"] == "confidence_00"
    with pytest.raises(ValueError, match="selection receipt content drifted"):
        validate_soft_polarity_confidence_fit_receipt(
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
        validate_soft_polarity_confidence_fit_receipt(
            forged,
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_public_selection_api_has_no_outer_score_or_calibration_surface() -> None:
    for function in (
        build_soft_polarity_confidence_fit_receipt,
        validate_soft_polarity_confidence_fit_receipt,
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
            }
        )
