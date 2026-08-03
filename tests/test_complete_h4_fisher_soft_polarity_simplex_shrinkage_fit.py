from __future__ import annotations

import copy
import inspect
import json
import math

import pytest

import fisher_graph.complete_h4_fisher_soft_polarity_simplex_shrinkage_fit as core
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_shrinkage_fit import (
    SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS,
    SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS,
    SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256,
    build_soft_polarity_simplex_shrinkage_anchor_receipt,
    build_soft_polarity_simplex_shrinkage_fit_receipt,
    build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt,
    build_soft_polarity_simplex_shrinkage_selection_receipt,
    build_soft_polarity_simplex_shrinkage_vertex_score_receipt,
    validate_soft_polarity_simplex_shrinkage_anchor_receipt,
    validate_soft_polarity_simplex_shrinkage_fit_receipt,
    validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt,
    validate_soft_polarity_simplex_shrinkage_selection_receipt,
    validate_soft_polarity_simplex_shrinkage_vertex_score_receipt,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
OUTER = FAMILIES[0]
INNER = FAMILIES[1:]
EXPECTED_PROTOCOL_SHA256 = (
    "87f723f7f8d3b5a9c921c570288261a84d2e21d73a5009d610a04c84b4f5d674"
)


def _anchors(
    y0: float = 1.0625,
    yhalf: float = 1.0625,
    y1: float = 1.5625,
) -> dict[str, dict[str, float]]:
    return {
        family: {
            "lambda_0": y0,
            "lambda_half": yhalf,
            "lambda_1": y1,
        }
        for family in INNER
    }


def _vertex(value: float = 1.0) -> dict[str, float]:
    return {family: value for family in INNER}


def _chain(
    anchors: dict[str, dict[str, float]] | None = None,
    vertex: dict[str, float] | None = None,
):
    anchor_values = _anchors() if anchors is None else anchors
    vertex_values = _vertex() if vertex is None else vertex
    anchor_receipt = build_soft_polarity_simplex_shrinkage_anchor_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchor_values,
    )
    proposal_receipt = (
        build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
            anchor_receipt=anchor_receipt
        )
    )
    vertex_receipt = (
        build_soft_polarity_simplex_shrinkage_vertex_score_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            exact_vertex_objectives_by_family=vertex_values,
        )
    )
    selection_receipt = (
        build_soft_polarity_simplex_shrinkage_selection_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            vertex_score_receipt=vertex_receipt,
        )
    )
    return (
        anchor_values,
        vertex_values,
        anchor_receipt,
        proposal_receipt,
        vertex_receipt,
        selection_receipt,
    )


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


def test_protocol_freezes_exact_three_anchors_and_stable_hash() -> None:
    assert SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS == (
        "lambda_0",
        "lambda_half",
        "lambda_1",
    )
    assert SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS == (0.0, 0.5, 1.0)
    assert SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256 == (
        EXPECTED_PROTOCOL_SHA256
    )

    _, _, receipt, _, _, _ = _chain()
    assert receipt["anchor_order"] == SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS
    assert receipt["anchor_lambdas"] == (
        SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS
    )
    assert receipt["anchor_lambda_hex"] == (
        "0x0.0p+0",
        "0x1.0000000000000p-1",
        "0x1.0000000000000p+0",
    )
    assert receipt["anchor_lambdas_frozen_before_any_stage_1_objective"] is True
    assert len(receipt["anchor_definition_receipts"]) == 3


def test_anchor_objective_is_exact_family_equal_float64_kl() -> None:
    anchors = _anchors()
    for index, family in enumerate(INNER):
        shift = index / 64.0
        for anchor_id in SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS:
            anchors[family][anchor_id] += shift
    receipt = build_soft_polarity_simplex_shrinkage_anchor_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchors,
    )
    for anchor_id in SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS:
        expected = math.fsum(
            anchors[family][anchor_id] for family in INNER
        ) / len(INNER)
        assert receipt["family_equal_exact_kl_by_anchor"][anchor_id] == expected
        assert (
            receipt["family_equal_exact_kl_hex_by_anchor"][anchor_id]
            == expected.hex()
        )
    assert OUTER not in receipt["exact_kl_objective_by_family_and_anchor"]
    validate_soft_polarity_simplex_shrinkage_anchor_receipt(
        json.loads(json.dumps(receipt)),
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchors,
    )


def test_quadratic_coefficients_and_interior_vertex_are_exact_formula() -> None:
    _, _, _, proposal, _, selection = _chain()
    y0, yhalf, y1 = 1.0625, 1.0625, 1.5625
    expected_a = 2.0 * (y1 - y0) - 4.0 * (yhalf - y0)
    expected_b = (y1 - y0) - expected_a
    assert expected_a == 1.0
    assert expected_b == -0.5
    assert proposal["quadratic_a"] == expected_a
    assert proposal["quadratic_b"] == expected_b
    assert proposal["raw_vertex"] == 0.25
    assert proposal["proposed_lambda"] == 0.25
    assert proposal["proposal_reason"] == "positive_curvature_clipped_vertex"
    assert proposal["quadratic_a_hex"] == expected_a.hex()
    assert proposal["quadratic_b_hex"] == expected_b.hex()
    assert proposal["proposed_lambda_hex"] == (0.25).hex()
    assert selection["selected_candidate_id"] == "lambda_vertex_stage2"
    assert selection["selected_lambda"] == 0.25


@pytest.mark.parametrize(
    "values,expected_a,expected_b,expected_lambda,expected_reason",
    (
        (
            (1.0, 1.75, 3.0),
            1.0,
            1.0,
            0.0,
            "positive_curvature_clipped_vertex",
        ),
        (
            (4.0, 2.75, 2.0),
            1.0,
            -3.0,
            1.0,
            "positive_curvature_clipped_vertex",
        ),
        (
            (1.0, 0.75, 0.5),
            0.0,
            -0.5,
            1.0,
            "nonpositive_curvature_better_endpoint",
        ),
        (
            (0.5, 0.75, 1.0),
            0.0,
            0.5,
            0.0,
            "nonpositive_curvature_better_endpoint",
        ),
        (
            (1.0, 1.25, 1.0),
            -1.0,
            1.0,
            0.0,
            "nonpositive_curvature_better_endpoint",
        ),
    ),
)
def test_proposal_clips_convex_vertex_or_uses_better_endpoint(
    values: tuple[float, float, float],
    expected_a: float,
    expected_b: float,
    expected_lambda: float,
    expected_reason: str,
) -> None:
    anchor_receipt = build_soft_polarity_simplex_shrinkage_anchor_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=_anchors(*values),
    )
    proposal = build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        anchor_receipt=anchor_receipt
    )
    assert proposal["quadratic_a"] == expected_a
    assert proposal["quadratic_b"] == expected_b
    assert proposal["proposed_lambda"] == expected_lambda
    assert proposal["proposal_reason"] == expected_reason


def test_stage_2_uses_supplied_exact_score_not_quadratic_prediction() -> None:
    anchors, vertices, anchor, proposal, vertex, selection = _chain()
    assert vertex["lambda"] == proposal["proposed_lambda"] == 0.25
    assert vertex["family_equal_exact_kl"] == 1.0
    assert vertex["family_equal_exact_kl_hex"] == (1.0).hex()
    assert vertex["objective_supplied_by_exact_stage_2_scoring"] is True
    assert vertex["quadratic_prediction_used_as_stage_2_objective"] is False
    assert selection["selected_family_equal_exact_kl"] == 1.0

    validate_soft_polarity_simplex_shrinkage_vertex_score_receipt(
        json.loads(json.dumps(vertex)),
        anchor_receipt=json.loads(json.dumps(anchor)),
        proposal_receipt=json.loads(json.dumps(proposal)),
        exact_vertex_objectives_by_family=vertices,
    )
    validate_soft_polarity_simplex_shrinkage_fit_receipt(
        json.loads(json.dumps(selection)),
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=anchors,
        exact_vertex_objectives_by_family=vertices,
    )


def test_final_selection_uses_exact_scores_and_ties_smaller_lambda_then_hash() -> None:
    selection = build_soft_polarity_simplex_shrinkage_fit_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=_anchors(),
        exact_vertex_objectives_by_family=_vertex(2.0),
    )
    assert selection["selected_candidate_id"] == "lambda_0_anchor"
    assert selection["selected_lambda"] == 0.0

    tie = build_soft_polarity_simplex_shrinkage_fit_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=_anchors(),
        exact_vertex_objectives_by_family=_vertex(1.0625),
    )
    assert tie["candidate_ranking"][:2] == (
        "lambda_0_anchor",
        "lambda_vertex_stage2",
    )

    same_lambda = build_soft_polarity_simplex_shrinkage_fit_receipt(
        all_development_family_ids=FAMILIES,
        outer_held_family_id=OUTER,
        exact_anchor_objectives_by_family_and_anchor=_anchors(1.0, 1.75, 3.0),
        exact_vertex_objectives_by_family=_vertex(1.0),
    )
    assert same_lambda["lambda_by_candidate"]["lambda_0_anchor"] == 0.0
    assert same_lambda["lambda_by_candidate"]["lambda_vertex_stage2"] == 0.0
    same_lambda_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in same_lambda["candidate_receipts"]
        if candidate["lambda"] == 0.0
    }
    expected = min(
        same_lambda_candidates.values(),
        key=lambda candidate: candidate["artifact_sha256"],
    )
    assert same_lambda["selected_candidate_id"] == expected["candidate_id"]
    assert same_lambda["selected_fixed_candidate_index"] == expected[
        "fixed_candidate_index"
    ]


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, True, "1"))
def test_fit_rejects_nonnumeric_or_nonfinite_anchor_and_vertex_scores(
    bad_value: object,
) -> None:
    anchors = _anchors()
    anchors[INNER[0]]["lambda_0"] = bad_value
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        build_soft_polarity_simplex_shrinkage_fit_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
            exact_vertex_objectives_by_family=_vertex(),
        )

    vertices = _vertex()
    vertices[INNER[0]] = bad_value
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        build_soft_polarity_simplex_shrinkage_fit_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=_anchors(),
            exact_vertex_objectives_by_family=vertices,
        )


def test_fit_rejects_incomplete_or_extra_two_stage_geometry() -> None:
    anchors = _anchors()
    anchors.pop(INNER[0])
    with pytest.raises(ValueError, match="anchor objective family geometry"):
        build_soft_polarity_simplex_shrinkage_fit_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
            exact_vertex_objectives_by_family=_vertex(),
        )

    anchors = _anchors()
    anchors[INNER[0]]["extra"] = 0.0
    with pytest.raises(ValueError, match="anchor objective geometry"):
        build_soft_polarity_simplex_shrinkage_fit_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
            exact_vertex_objectives_by_family=_vertex(),
        )

    vertices = _vertex()
    vertices.pop(INNER[0])
    with pytest.raises(ValueError, match="vertex objective family geometry"):
        build_soft_polarity_simplex_shrinkage_fit_receipt(
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=_anchors(),
            exact_vertex_objectives_by_family=vertices,
        )


def test_rehashed_anchor_and_proposal_forgeries_fail_replay() -> None:
    anchors, _, anchor, proposal, _, _ = _chain()
    forged_anchor = copy.deepcopy(anchor)
    forged_anchor["family_equal_exact_kl_by_anchor"]["lambda_0"] = -100.0
    forged_anchor["family_equal_exact_kl_hex_by_anchor"]["lambda_0"] = (
        (-100.0).hex()
    )
    forged_anchor = _rehash(forged_anchor, domain=core._ANCHOR_DOMAIN)
    with pytest.raises(ValueError, match="anchor receipt content drifted"):
        validate_soft_polarity_simplex_shrinkage_anchor_receipt(
            forged_anchor,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
        )

    forged_proposal = copy.deepcopy(proposal)
    forged_proposal["quadratic_a"] = 99.0
    forged_proposal["quadratic_a_hex"] = (99.0).hex()
    forged_proposal = _rehash(forged_proposal, domain=core._PROPOSAL_DOMAIN)
    with pytest.raises(ValueError, match="proposal receipt content drifted"):
        validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
            forged_proposal, anchor_receipt=anchor
        )


def test_rehashed_vertex_and_selection_forgeries_fail_authoritative_replay() -> None:
    anchors, vertices, _, _, vertex, selection = _chain()
    forged_vertex = copy.deepcopy(vertex)
    forged_vertex["family_equal_exact_kl"] = -100.0
    forged_vertex["family_equal_exact_kl_hex"] = (-100.0).hex()
    forged_vertex = _rehash(forged_vertex, domain=core._VERTEX_DOMAIN)
    forged_selection = copy.deepcopy(selection)
    forged_selection["vertex_score_receipt"] = forged_vertex
    forged_selection["vertex_score_receipt_sha256"] = forged_vertex[
        "artifact_sha256"
    ]
    forged_selection = _rehash(forged_selection, domain=core._SELECTION_DOMAIN)
    with pytest.raises(ValueError, match="fit receipt content drifted"):
        validate_soft_polarity_simplex_shrinkage_fit_receipt(
            forged_selection,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
            exact_vertex_objectives_by_family=vertices,
        )

    forged_authority = copy.deepcopy(selection)
    forged_authority["serving_authorized"] = True
    forged_authority = _rehash(
        forged_authority, domain=core._SELECTION_DOMAIN
    )
    with pytest.raises(ValueError, match="fit receipt content drifted"):
        validate_soft_polarity_simplex_shrinkage_fit_receipt(
            forged_authority,
            all_development_family_ids=FAMILIES,
            outer_held_family_id=OUTER,
            exact_anchor_objectives_by_family_and_anchor=anchors,
            exact_vertex_objectives_by_family=vertices,
        )


def test_receipts_commit_training_only_two_stage_boundary() -> None:
    _, _, anchor, proposal, vertex, selection = _chain()
    assert anchor["stage"] == 1
    assert proposal["stage"] == 1
    assert vertex["stage"] == 2
    assert proposal["proposal_frozen_before_any_stage_2_vertex_objective"] is True
    assert selection["anchors_frozen_before_stage_1_scoring"] is True
    assert selection["proposal_frozen_before_stage_2_scoring"] is True
    assert selection["final_selection_uses_only_exact_objectives"] is True
    assert selection["outer_held_family_used_for_fit_or_selection"] is False
    assert selection["compression_claim_authorized"] is False
    assert selection["speed_claim_authorized"] is False
    assert selection["serving_authorized"] is False
    boundary = selection["data_boundary"]
    assert boundary["outer_held_objectives_consumed"] is False
    assert boundary["calibration_b_opened"] is False
    assert boundary["validation_opened"] is False
    assert boundary["test_opened"] is False
    assert boundary["quadratic_prediction_used_as_final_objective"] is False


def test_public_fit_api_has_no_held_or_model_execution_surface() -> None:
    public_functions = (
        build_soft_polarity_simplex_shrinkage_anchor_receipt,
        build_soft_polarity_simplex_shrinkage_fit_receipt,
        build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt,
        build_soft_polarity_simplex_shrinkage_selection_receipt,
        build_soft_polarity_simplex_shrinkage_vertex_score_receipt,
        validate_soft_polarity_simplex_shrinkage_anchor_receipt,
        validate_soft_polarity_simplex_shrinkage_fit_receipt,
        validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt,
        validate_soft_polarity_simplex_shrinkage_selection_receipt,
        validate_soft_polarity_simplex_shrinkage_vertex_score_receipt,
    )
    forbidden = {
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
    for function in public_functions:
        assert not set(inspect.signature(function).parameters).intersection(
            forbidden
        )

    _, _, anchor, proposal, vertex, selection = _chain()
    strings = "\n".join(
        (
            *_all_strings(core._PROTOCOL),
            *_all_strings(anchor),
            *_all_strings(proposal),
            *_all_strings(vertex),
            *_all_strings(selection),
        )
    ).lower()
    assert "kl_teacher_to_candidate" in strings
    assert "exact_float64" in strings
    assert "full_vocabulary" in strings
    assert ("negative_log_" + "likelihood") not in strings
    assert ("n" + "ll") not in strings
