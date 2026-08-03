from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_local_signed_field_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_local_signed_field_fit import (
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
    SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY,
    build_soft_polarity_local_signed_field_fit_receipt,
    build_soft_polarity_local_signed_field_ladder_receipt,
    soft_polarity_local_signed_field_scalar,
    validate_soft_polarity_local_signed_field_fit_receipt,
    validate_soft_polarity_local_signed_field_ladder_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]


def _objectives(default: float = 10.0) -> dict[str, dict[str, float]]:
    return {
        family: {
            candidate_id: default
            for candidate_id in SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
        }
        for family in INNER
    }


def _fit(objectives=None):
    selected = _objectives() if objectives is None else objectives
    ladder = build_soft_polarity_local_signed_field_ladder_receipt()
    receipt = build_soft_polarity_local_signed_field_fit_receipt(
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=selected,
    )
    return selected, ladder, receipt


def _rehash(receipt: dict[str, object], domain: bytes) -> dict[str, object]:
    result = copy.deepcopy(receipt)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = core._hash(domain, result)
    return result


def test_protocol_freezes_exact_twenty_seven_candidate_library() -> None:
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS == (
        "c1",
        "c2",
        "c1_times_c2",
        "source_z",
    )
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES == (
        -0.5,
        0.0,
        0.5,
    )
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES == (-1.0, 1.0)
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES == (-1.0, 0.0, 1.0)
    assert len(SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY) == 27
    assert len(set(SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY)) == 27
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY[:6] == (
        ("c1", -0.5, -1.0),
        ("c1", -0.5, 1.0),
        ("c1", 0.0, -1.0),
        ("c1", 0.0, 1.0),
        ("c1", 0.5, -1.0),
        ("c1", 0.5, 1.0),
    )
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY[-3:] == (
        ("source_z", -1.0, 0.0),
        ("source_z", 0.0, 0.0),
        ("source_z", 1.0, 0.0),
    )
    assert SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256 == (
        "e6691edf332f4c25ff17a85b0b754e426fe92b5fe1e0335b04112ce5381f7c95"
    )


def test_ladder_authenticates_candidate_order_and_exact_anchors() -> None:
    ladder = build_soft_polarity_local_signed_field_ladder_receipt()
    validate_soft_polarity_local_signed_field_ladder_receipt(ladder)
    assert ladder["candidate_count"] == 27
    assert ladder["adaptive_candidate_count"] == 24
    assert ladder["exact_nonadaptive_anchor_count"] == 3
    assert tuple(ladder["candidate_order"]) == (
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
    )
    anchors = ladder["candidate_receipts"][-3:]
    assert [(item["feature_id"], item["b"], item["a"]) for item in anchors] == [
        ("source_z", -1.0, 0.0),
        ("source_z", 0.0, 0.0),
        ("source_z", 1.0, 0.0),
    ]
    assert all(item["exact_nonadaptive_anchor"] for item in anchors)


@pytest.mark.parametrize(
    ("feature_id", "c1", "c2", "source_z", "b", "a", "expected"),
    [
        ("c1", 0.25, -0.75, 0.5, 0.5, 1.0, 0.75),
        ("c2", 0.25, -0.75, 0.5, 0.0, -1.0, 0.75),
        ("c1_times_c2", 0.5, -0.5, 0.1, 0.0, 1.0, -0.25),
        ("source_z", 0.25, -0.75, 0.5, -0.5, -1.0, -1.0),
        ("source_z", 0.0, 0.0, 99.0, 1.0, 0.0, 1.0),
        ("c1", 9.0, 0.0, 0.0, 0.5, 1.0, 1.0),
        ("c1", 9.0, 0.0, 0.0, -0.5, -1.0, -1.0),
    ],
)
def test_scalar_evaluator_uses_feature_and_clamps(
    feature_id, c1, c2, source_z, b, a, expected
) -> None:
    assert soft_polarity_local_signed_field_scalar(
        feature_id=feature_id,
        c1=c1,
        c2=c2,
        source_z=source_z,
        b=b,
        a=a,
    ) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_id": "unknown"},
        {"c1": float("nan")},
        {"c2": float("inf")},
        {"source_z": True},
        {"b": "0"},
        {"a": float("-inf")},
        {"c1": 1e308, "c2": 1e308, "feature_id": "c1_times_c2"},
    ],
)
def test_scalar_evaluator_fails_closed(kwargs) -> None:
    values = {
        "feature_id": "c1",
        "c1": 0.0,
        "c2": 0.0,
        "source_z": 0.0,
        "b": 0.0,
        "a": 1.0,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        soft_polarity_local_signed_field_scalar(**values)


def test_exact_seven_family_rows_and_all_candidates_are_required() -> None:
    objectives = _objectives()
    objectives.pop(INNER[-1])
    ladder = build_soft_polarity_local_signed_field_ladder_receipt()
    with pytest.raises(ValueError, match="family geometry"):
        build_soft_polarity_local_signed_field_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )
    objectives = _objectives()
    objectives[INNER[0]].pop(SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0])
    with pytest.raises(ValueError, match="candidate geometry"):
        build_soft_polarity_local_signed_field_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1.0"])
def test_nonfinite_or_nonnumeric_objective_rejected(bad) -> None:
    objectives = _objectives()
    objectives[INNER[0]][SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]] = bad
    ladder = build_soft_polarity_local_signed_field_ladder_receipt()
    with pytest.raises((TypeError, ValueError)):
        build_soft_polarity_local_signed_field_fit_receipt(
            ladder_receipt=ladder,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_objectives_by_family_and_candidate=objectives,
        )


def test_family_equal_aggregation_selects_adaptive_field() -> None:
    objectives = _objectives()
    winner = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[7]
    for index, family in enumerate(INNER):
        objectives[family][winner] = float(index + 1)
    _, _, receipt = _fit(objectives)
    assert receipt["family_equal_exact_kl_by_candidate"][winner] == 4.0
    assert receipt["selected_candidate_id"] == winner
    assert receipt["selected_candidate_index"] == 7
    assert receipt["selected_adaptive"] is True
    assert receipt["selected_feature_id"] == "c2"
    assert receipt["selected_b"] == -0.5
    assert receipt["selected_a"] == 1.0


def test_equal_objectives_prefer_nonadaptive_then_smaller_abs_b() -> None:
    _, _, receipt = _fit(_objectives(1.0))
    zero_anchor = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[25]
    assert receipt["selected_candidate_id"] == zero_anchor
    assert receipt["selected_adaptive"] is False
    assert receipt["selected_b"] == 0.0
    assert receipt["selected_a"] == 0.0
    assert receipt["candidate_ranking"][:3] == (
        zero_anchor,
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[24],
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[26],
    )


def test_lower_adaptive_objective_beats_simpler_anchor() -> None:
    objectives = _objectives(1.0)
    adaptive = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]
    for family in INNER:
        objectives[family][adaptive] = math.nextafter(1.0, 0.0)
    _, _, receipt = _fit(objectives)
    assert receipt["selected_candidate_id"] == adaptive


def test_adaptive_tie_uses_abs_coefficients_then_fixed_index() -> None:
    objectives = _objectives(2.0)
    first = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]
    second = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[1]
    for family in INNER:
        objectives[family][first] = 1.0
        objectives[family][second] = 1.0
    _, _, receipt = _fit(objectives)
    assert receipt["candidate_ranking"][:2] == (first, second)
    assert receipt["selected_candidate_id"] == first


def test_receipt_authenticates_rows_aggregates_ranking_and_selection() -> None:
    objectives, ladder, receipt = _fit()
    validate_soft_polarity_local_signed_field_fit_receipt(
        receipt,
        ladder_receipt=ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    mutations = []
    family_tamper = copy.deepcopy(receipt)
    family_tamper["family_oof_receipts"][INNER[0]][
        "exact_kl_objective_by_candidate"
    ][SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]] = 0.0
    mutations.append(family_tamper)
    aggregate_tamper = copy.deepcopy(receipt)
    aggregate_tamper["aggregate_receipts"][
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]
    ]["family_equal_exact_kl"] = 0.0
    mutations.append(aggregate_tamper)
    ranking_tamper = copy.deepcopy(receipt)
    ranking_tamper["ranking_receipt"]["candidate_ranking"] = tuple(
        reversed(ranking_tamper["ranking_receipt"]["candidate_ranking"])
    )
    mutations.append(ranking_tamper)
    selection_tamper = copy.deepcopy(receipt)
    selection_tamper["selected_candidate_id"] = (
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[0]
    )
    mutations.append(selection_tamper)
    for tampered in mutations:
        forged = _rehash(tampered, core._SELECTION_DOMAIN)
        with pytest.raises(ValueError, match="selection receipt"):
            validate_soft_polarity_local_signed_field_fit_receipt(
                forged,
                ladder_receipt=ladder,
                all_development_family_ids=FAMILIES,
                outer_held_family_id=OUTER,
                exact_objectives_by_family_and_candidate=objectives,
            )


def test_ladder_validator_rejects_rehashed_candidate_tamper() -> None:
    ladder = build_soft_polarity_local_signed_field_ladder_receipt()
    tampered = copy.deepcopy(ladder)
    candidates = list(tampered["candidate_receipts"])
    candidate = candidates[0]
    candidate["b"] = 0.5
    candidates[0] = _rehash(candidate, core._CANDIDATE_DOMAIN)
    tampered["candidate_receipts"] = tuple(candidates)
    forged = _rehash(tampered, core._LADDER_DOMAIN)
    with pytest.raises(ValueError, match="candidate content"):
        validate_soft_polarity_local_signed_field_ladder_receipt(forged)


def test_complete_fit_survives_json_roundtrip_and_exact_replay() -> None:
    objectives = _objectives()
    winner = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[20]
    for family in INNER:
        objectives[family][winner] = 0.25
    _, ladder, receipt = _fit(objectives)
    roundtrip_ladder = json.loads(json.dumps(ladder))
    roundtrip = json.loads(json.dumps(receipt))
    validate_soft_polarity_local_signed_field_fit_receipt(
        roundtrip,
        ladder_receipt=roundtrip_ladder,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_objectives_by_family_and_candidate=objectives,
    )
    assert roundtrip["selected_candidate_id"] == winner


def test_public_fit_api_exposes_only_training_scalar_objectives() -> None:
    forbidden = {
        "model",
        "prompt",
        "prompt_text",
        "logits",
        "h4",
        "raw_h4",
        "outer_held_objective",
        "outer_held_score",
        "calibration_b",
    }
    for function in (
        build_soft_polarity_local_signed_field_ladder_receipt,
        build_soft_polarity_local_signed_field_fit_receipt,
        validate_soft_polarity_local_signed_field_fit_receipt,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_data_boundary_is_training_only_and_fail_closed() -> None:
    _, ladder, receipt = _fit()
    for boundary in (ladder["data_boundary"], receipt["data_boundary"]):
        assert boundary["candidate_library_frozen_before_any_objective"] is True
        assert boundary["outer_held_objectives_consumed"] is False
        assert boundary["prompt_text_consumed"] is False
        assert boundary["raw_logits_consumed"] is False
        assert boundary["raw_h4_consumed"] is False
        assert boundary["calibration_b_opened"] is False
        assert boundary["compression_claim_authorized"] is False
        assert boundary["serving_authorized"] is False
