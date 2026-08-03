from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_signed_continuum_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_signed_continuum_fit import (
    SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS,
    SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES,
    SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
    build_soft_polarity_signed_continuum_anchor_receipt,
    build_soft_polarity_signed_continuum_fit_receipt,
    build_soft_polarity_signed_continuum_quadratic_proposal_receipt,
    build_soft_polarity_signed_continuum_selection_receipt,
    build_soft_polarity_signed_continuum_vertex_score_receipt,
    validate_soft_polarity_signed_continuum_anchor_receipt,
    validate_soft_polarity_signed_continuum_fit_receipt,
    validate_soft_polarity_signed_continuum_quadratic_proposal_receipt,
    validate_soft_polarity_signed_continuum_selection_receipt,
    validate_soft_polarity_signed_continuum_vertex_score_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]


def _anchors(
    y_minus: float = 1.5625,
    y_zero: float = 1.0625,
    y_plus: float = 1.5625,
) -> dict[str, dict[str, float]]:
    return {
        family: {
            "signed_minus_one": y_minus,
            "signed_zero": y_zero,
            "signed_plus_one": y_plus,
        }
        for family in INNER
    }


def _vertex(value: float = 1.0) -> dict[str, float]:
    return {family: value for family in INNER}


def _chain(anchors=None, vertex=None):
    anchor_values = _anchors() if anchors is None else anchors
    vertex_values = _vertex() if vertex is None else vertex
    anchor_receipt = build_soft_polarity_signed_continuum_anchor_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchor_values,
    )
    proposal_receipt = (
        build_soft_polarity_signed_continuum_quadratic_proposal_receipt(
            anchor_receipt=anchor_receipt
        )
    )
    vertex_receipt = build_soft_polarity_signed_continuum_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=vertex_values,
    )
    selection_receipt = build_soft_polarity_signed_continuum_selection_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_receipt,
    )
    return (
        anchor_values,
        vertex_values,
        anchor_receipt,
        proposal_receipt,
        vertex_receipt,
        selection_receipt,
    )


def _rehash(receipt: dict[str, object], domain: bytes) -> dict[str, object]:
    result = copy.deepcopy(receipt)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = core._hash(domain, result)
    return result


def test_protocol_freezes_signed_geometry() -> None:
    assert SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS == (
        "signed_minus_one",
        "signed_zero",
        "signed_plus_one",
    )
    assert SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES == (-1.0, 0.0, 1.0)
    assert SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256 == (
        "e246b03fba02e65aae0346c9175528e292911962e33cad43c393ef416f3e9632"
    )
    _, _, anchors, _, _, _ = _chain()
    assert anchors["anchor_values"] == (-1.0, 0.0, 1.0)
    assert anchors["outer_held_family_used"] is False


def test_quadratic_recovers_exact_signed_vertex() -> None:
    # y(s)=2*s^2 + 0.5*s + 1 => vertex -0.125.
    anchors = _anchors(y_minus=2.5, y_zero=1.0, y_plus=3.5)
    _, _, _, proposal, _, _ = _chain(anchors=anchors)
    assert proposal["quadratic_a"] == 2.0
    assert proposal["quadratic_b"] == 0.5
    assert proposal["quadratic_c"] == 1.0
    assert proposal["raw_vertex"] == -0.125
    assert proposal["proposed_signed_scalar"] == -0.125
    assert proposal["proposal_reason"] == "positive_curvature_clipped_vertex"


@pytest.mark.parametrize(
    ("values", "expected", "reason"),
    [
        ((4.0, 1.0, 0.0), 1.0, "positive_curvature_clipped_vertex"),
        ((0.0, 1.0, 4.0), -1.0, "positive_curvature_clipped_vertex"),
        ((0.5, 1.0, 0.5), -1.0, "nonpositive_curvature_best_exact_anchor"),
        ((2.0, 0.25, 2.0), 0.0, "positive_curvature_clipped_vertex"),
    ],
)
def test_proposal_clips_or_uses_best_anchor(values, expected, reason) -> None:
    anchors = _anchors(*values)
    _, _, _, proposal, _, _ = _chain(anchors=anchors)
    assert proposal["proposed_signed_scalar"] == expected
    assert proposal["proposal_reason"] == reason


def test_stage_2_uses_exact_score_not_quadratic_prediction() -> None:
    anchors = _anchors(y_minus=2.5, y_zero=1.0, y_plus=3.5)
    _, _, _, proposal, vertex, selection = _chain(
        anchors=anchors, vertex=_vertex(9.0)
    )
    assert proposal["proposed_signed_scalar"] == -0.125
    assert vertex["family_equal_exact_kl"] == 9.0
    assert vertex["quadratic_prediction_used_as_stage_2_objective"] is False
    assert selection["selected_signed_scalar"] == 0.0


def test_final_selection_ties_objective_abs_signed_then_hash() -> None:
    anchors = _anchors(1.0, 1.0, 1.0)
    _, _, _, proposal, _, selection = _chain(
        anchors=anchors, vertex=_vertex(1.0)
    )
    assert proposal["proposed_signed_scalar"] == 0.0
    assert selection["selected_signed_scalar"] == 0.0
    candidates = [
        item
        for item in selection["candidate_receipts"]
        if item["signed_scalar"] == 0.0
    ]
    assert len(candidates) == 2
    expected = min(candidates, key=lambda item: item["artifact_sha256"])
    assert selection["selected_candidate_id"] == expected["candidate_id"]


def test_negative_wins_signed_tie_after_equal_absolute_value() -> None:
    anchors = _anchors(0.5, 2.0, 0.5)
    _, _, _, _, _, selection = _chain(anchors=anchors, vertex=_vertex(3.0))
    assert selection["selected_signed_scalar"] == -1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1.0"])
def test_nonfinite_or_nonnumeric_objective_rejected(bad) -> None:
    anchors = _anchors()
    anchors[INNER[0]]["signed_zero"] = bad
    with pytest.raises((TypeError, ValueError)):
        build_soft_polarity_signed_continuum_anchor_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
        )


def test_authoritative_validators_fail_closed_on_tampering() -> None:
    values, vertex_values, anchors, proposal, vertex, selection = _chain()
    forged_anchor = copy.deepcopy(anchors)
    forged_anchor["family_equal_exact_kl_by_anchor"]["signed_zero"] = 0.0
    forged_anchor = _rehash(forged_anchor, core._ANCHOR_DOMAIN)
    with pytest.raises(ValueError):
        validate_soft_polarity_signed_continuum_anchor_receipt(
            forged_anchor,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=values,
        )
    forged_vertex = copy.deepcopy(vertex)
    forged_vertex["family_equal_exact_kl"] = 0.0
    forged_vertex = _rehash(forged_vertex, core._VERTEX_DOMAIN)
    with pytest.raises(ValueError):
        validate_soft_polarity_signed_continuum_vertex_score_receipt(
            forged_vertex,
            anchor_receipt=anchors,
            proposal_receipt=proposal,
            exact_vertex_objectives_by_family=vertex_values,
        )
    forged_selection = copy.deepcopy(selection)
    forged_selection["selected_signed_scalar"] = 1.0
    forged_selection = _rehash(forged_selection, core._SELECTION_DOMAIN)
    with pytest.raises(ValueError):
        validate_soft_polarity_signed_continuum_selection_receipt(
            forged_selection,
            anchor_receipt=anchors,
            proposal_receipt=proposal,
            vertex_score_receipt=vertex,
        )


def test_complete_fit_survives_json_roundtrip_and_replays() -> None:
    anchors = _anchors(2.5, 1.0, 3.5)
    vertex = _vertex(0.75)
    receipt = build_soft_polarity_signed_continuum_fit_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchors,
        exact_vertex_objectives_by_family=vertex,
    )
    roundtrip = json.loads(json.dumps(receipt))
    validate_soft_polarity_signed_continuum_fit_receipt(
        roundtrip,
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchors,
        exact_vertex_objectives_by_family=vertex,
    )
    assert roundtrip["selected_signed_scalar"] == -0.125


def test_public_fit_api_has_no_model_or_outer_objective_surface() -> None:
    for function in (
        build_soft_polarity_signed_continuum_anchor_receipt,
        build_soft_polarity_signed_continuum_quadratic_proposal_receipt,
        build_soft_polarity_signed_continuum_vertex_score_receipt,
        build_soft_polarity_signed_continuum_selection_receipt,
        build_soft_polarity_signed_continuum_fit_receipt,
    ):
        names = set(inspect.signature(function).parameters)
        assert "model" not in names
        assert "outer_held_objective" not in names
        assert "prompt" not in names


def test_data_boundary_is_training_only() -> None:
    _, _, anchors, proposal, vertex, selection = _chain()
    for receipt in (anchors, proposal, vertex, selection):
        boundary = receipt["data_boundary"]
        assert boundary["outer_held_objectives_consumed"] is False
        assert boundary["compression_claim_authorized"] is False
        assert boundary["serving_authorized"] is False
        assert boundary["prompt_text_consumed"] is False
        assert not any(
            isinstance(value, float) and not math.isfinite(value)
            for value in receipt.values()
        )
