"""Pure two-stage V20o fit for the signed continuum scalar.

Stage 1 freezes and scores the exact ``s=-1``, ``s=0``, and ``s=+1``
providers on seven inner-OOF families.  Their family-equal objectives define
the unique quadratic

``y(s) = a*s**2 + b*s + c``

with ``c=y0``, ``b=(y_plus-y_minus)/2``, and
``a=(y_plus+y_minus)/2-y0``.  Positive curvature proposes the vertex clipped
to ``[-1,1]``.  Nonpositive curvature proposes the best exact anchor, with the
same deterministic tie key used by final selection.

The proposal is frozen before stage 2 opens an independently supplied exact
score at that signed value.  Final selection compares the exact scores for
``-1``, the stage-2 proposal, ``0``, and ``+1``.  Ties sort by objective,
smaller ``abs(s)``, signed value, and then candidate artifact SHA-256.

No model execution or outer-held score is available through this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re


__all__ = [
    "SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS",
    "SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES",
    "SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_SIMPLEX_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256",
    "build_soft_polarity_signed_continuum_anchor_receipt",
    "build_soft_polarity_signed_continuum_fit_receipt",
    "build_soft_polarity_signed_continuum_quadratic_proposal_receipt",
    "build_soft_polarity_signed_continuum_selection_receipt",
    "build_soft_polarity_signed_continuum_vertex_score_receipt",
    "validate_soft_polarity_signed_continuum_anchor_receipt",
    "validate_soft_polarity_signed_continuum_fit_receipt",
    "validate_soft_polarity_signed_continuum_quadratic_proposal_receipt",
    "validate_soft_polarity_signed_continuum_selection_receipt",
    "validate_soft_polarity_signed_continuum_vertex_score_receipt",
]


SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS = (
    "signed_minus_one",
    "signed_zero",
    "signed_plus_one",
)
SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES = (-1.0, 0.0, 1.0)

_ANCHOR_VALUE_BY_ID = dict(
    zip(
        SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS,
        SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES,
        strict=True,
    )
)
_FINAL_CANDIDATE_IDS = (
    "signed_minus_one_anchor",
    "signed_vertex_stage2",
    "signed_zero_anchor",
    "signed_plus_one_anchor",
)
_DEVELOPMENT_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORMULA = (
    "g_s(c)=e(c)*((1-abs(s))+abs(s)*q(sign_nonnegative_zero(s)*z))"
)
_OBJECTIVE_KIND = (
    "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
)
_AGGREGATE_OBJECTIVE_KIND = (
    "family_equal_token_mean_exact_float64_full_vocabulary_"
    "kl_teacher_to_candidate"
)
_QUADRATIC_FORMULA = (
    "c=y_zero;b=(y_plus-y_minus)/2;"
    "a=(y_plus+y_minus)/2-y_zero"
)
_SELECTION_KEY = (
    "exact_family_equal_kl_then_smaller_abs_signed_scalar_then_signed_scalar_"
    "then_candidate_artifact_sha256"
)

_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"protocol:v20o\0"
)
_ANCHOR_DEFINITION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"anchor-definition:v20o\0"
)
_OBJECTIVE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"objective:v20o\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"family:v20o\0"
)
_ANCHOR_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"anchors:v20o\0"
)
_PROPOSAL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"proposal:v20o\0"
)
_VERTEX_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"vertex-candidate:v20o\0"
)
_VERTEX_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"vertex-score:v20o\0"
)
_FINAL_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"final-candidate:v20o\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-signed-continuum-fit:"
    b"selection:v20o\0"
)

_DATA_BOUNDARY = {
    "role": "v20o_training_only_two_stage_signed_continuum_fit",
    "development_family_count": _DEVELOPMENT_FAMILY_COUNT,
    "inner_oof_family_count": _INNER_FAMILY_COUNT,
    "stage_1_anchor_values_frozen_before_anchor_objectives": True,
    "stage_1_uses_only_three_anchor_objectives": True,
    "stage_1_quadratic_is_proposal_only": True,
    "stage_2_vertex_value_frozen_before_vertex_objectives": True,
    "stage_2_vertex_objective_supplied_by_exact_scoring": True,
    "quadratic_prediction_used_as_final_objective": False,
    "outer_held_family_absent_from_both_fit_stages": True,
    "outer_held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "logits_or_h4_serialized": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_shadow_consumed": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
    "serving_authorized": False,
}
_PROTOCOL = {
    "protocol": "v20o_training_only_two_stage_signed_continuum_selection",
    "scientific_status": "development_only_after_v20m",
    "formula": _FORMULA,
    "signed_scalar_constraint": "minus_one_le_s_le_plus_one",
    "anchor_order": SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS,
    "anchor_values": SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES,
    "anchor_value_hex": tuple(
        value.hex() for value in SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES
    ),
    "quadratic_formula": _QUADRATIC_FORMULA,
    "proposal_rule": (
        "positive_curvature_clipped_vertex_else_best_exact_anchor_by_"
        "objective_abs_signed_scalar_signed_scalar"
    ),
    "two_stage_freeze": (
        "anchors_before_stage_1_scores_then_proposal_before_stage_2_score"
    ),
    "stage_1_objective": _AGGREGATE_OBJECTIVE_KIND,
    "stage_2_objective": _AGGREGATE_OBJECTIVE_KIND,
    "final_candidate_order": _FINAL_CANDIDATE_IDS,
    "selection_key": _SELECTION_KEY,
    "authorization": (
        "training_fit_receipt_only_no_outer_score_compression_speed_or_"
        "serving_authority"
    ),
    "data_boundary": _DATA_BOUNDARY,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256 = _hash(
    _PROTOCOL_DOMAIN, _PROTOCOL
)
SOFT_POLARITY_SIMPLEX_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256 = (
    SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
)


def _finish(
    schema: str, domain: bytes, payload: Mapping[str, object]
) -> dict[str, object]:
    result = {"schema": schema, **dict(payload)}
    result["artifact_sha256"] = _hash(domain, result)
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _float_hex(value: float) -> str:
    return (0.0 if value == 0.0 else value).hex()


def _finite_operation(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return 0.0 if value == 0.0 else value


def _family_equal(values: Sequence[float], label: str) -> float:
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite") from error
    return _finite_operation(result, label)


def _families(
    all_development_family_ids: Sequence[str], outer_held_family_id: str
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    families = tuple(
        sorted(
            _identifier(item, "development family id")
            for item in _sequence(
                all_development_family_ids, "development family ids"
            )
        )
    )
    if (
        len(families) != _DEVELOPMENT_FAMILY_COUNT
        or len(set(families)) != _DEVELOPMENT_FAMILY_COUNT
    ):
        raise ValueError("exactly eight distinct development families are required")
    outer = _identifier(outer_held_family_id, "outer held family id")
    if outer not in families:
        raise ValueError("outer held family must be one development family")
    inner = tuple(family for family in families if family != outer)
    return families, outer, inner


def _anchor_definition(index: int) -> dict[str, object]:
    anchor_id = SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS[index]
    signed_scalar = SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES[index]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
        "anchor_definition.v20o",
        _ANCHOR_DEFINITION_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "anchor_id": anchor_id,
            "fixed_anchor_index": index,
            "signed_scalar": signed_scalar,
            "signed_scalar_hex": _float_hex(signed_scalar),
            "provider_frozen_before_any_stage_1_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
        },
    )


def _normalized_anchor_objectives(
    raw: Mapping[str, Mapping[str, float]],
    *,
    inner_families: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    supplied = _mapping(raw, "exact anchor objectives")
    if set(supplied) != set(inner_families):
        raise ValueError("anchor objective family geometry drifted")
    objectives: dict[str, dict[str, float]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for family in inner_families:
        row = _mapping(supplied[family], f"{family} anchor objectives")
        if set(row) != set(SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS):
            raise ValueError(f"{family} anchor objective geometry drifted")
        objectives[family] = {}
        hashes[family] = {}
        for anchor_id in SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS:
            objective = _number(row[anchor_id], f"{family} {anchor_id} objective")
            objectives[family][anchor_id] = objective
            hashes[family][anchor_id] = _hash(
                _OBJECTIVE_DOMAIN,
                {
                    "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
                    "stage": 1,
                    "family_id": family,
                    "anchor_id": anchor_id,
                    "signed_scalar": _ANCHOR_VALUE_BY_ID[anchor_id],
                    "objective": objective,
                    "objective_hex": _float_hex(objective),
                    "objective_kind": _OBJECTIVE_KIND,
                    "teacher_distribution_is_first_kl_argument": True,
                    "candidate_distribution_is_second_kl_argument": True,
                    "token_equal_within_family": True,
                    "exact_float64_execution": True,
                    "full_vocabulary_evaluated": True,
                },
            )
    return objectives, hashes


def _build_anchor_receipt(
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    families, outer, inner = _families(
        all_development_family_ids, outer_held_family_id
    )
    objectives, objective_hashes = _normalized_anchor_objectives(
        exact_anchor_objectives_by_family_and_anchor,
        inner_families=inner,
    )
    definitions = tuple(_anchor_definition(index) for index in range(3))
    family_receipts = {
        family: _finish(
            "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
            "anchor_family.v20o",
            _FAMILY_DOMAIN,
            {
                "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
                "stage": 1,
                "family_id": family,
                "outer_held_family_id": outer,
                "anchor_order": SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS,
                "exact_kl_objective_by_anchor": objectives[family],
                "exact_kl_objective_sha256_by_anchor": objective_hashes[family],
                "anchors_frozen_before_family_score": True,
                "outer_held_family_used": False,
                "raw_model_tensors_serialized": False,
            },
        )
        for family in inner
    }
    aggregate = {
        anchor_id: _family_equal(
            tuple(objectives[family][anchor_id] for family in inner),
            f"{anchor_id} family-equal objective",
        )
        for anchor_id in SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS
    }
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_anchors.v20o",
        _ANCHOR_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "stage": 1,
            "formula": _FORMULA,
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "inner_oof_family_order": inner,
            "anchor_order": SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_IDS,
            "anchor_values": SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES,
            "anchor_value_hex": tuple(
                _float_hex(value)
                for value in SOFT_POLARITY_SIGNED_CONTINUUM_ANCHOR_VALUES
            ),
            "anchor_definition_receipts": definitions,
            "anchor_definition_sha256s": {
                item["anchor_id"]: item["artifact_sha256"] for item in definitions
            },
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family_and_anchor": objectives,
            "exact_kl_objective_sha256_by_family_and_anchor": objective_hashes,
            "family_receipts": family_receipts,
            "family_receipt_sha256s": {
                family: family_receipts[family]["artifact_sha256"]
                for family in inner
            },
            "family_equal_exact_kl_by_anchor": aggregate,
            "family_equal_exact_kl_hex_by_anchor": {
                key: _float_hex(value) for key, value in aggregate.items()
            },
            "anchor_values_frozen_before_any_stage_1_objective": True,
            "stage_1_complete_before_quadratic_proposal": True,
            "outer_held_family_used": False,
            "exact_scalar_objectives_serialized": True,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_signed_continuum_anchor_receipt(
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    result = _build_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    validate_soft_polarity_signed_continuum_anchor_receipt(
        result,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    return result


def validate_soft_polarity_signed_continuum_anchor_receipt(
    value: Mapping[str, object],
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    receipt = _mapping(value, "signed-continuum anchor receipt")
    expected = _build_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    _sha(receipt.get("artifact_sha256"), "signed-continuum anchor artifact")
    if set(receipt) != set(expected) or _canonical(receipt) != _canonical(expected):
        raise ValueError("signed-continuum anchor receipt drifted")


def _authenticate_anchor_chain(receipt: Mapping[str, object]) -> None:
    anchor = _mapping(receipt, "signed-continuum anchor receipt")
    embedded_families = _sequence(
        anchor.get("all_development_family_ids"), "embedded development families"
    )
    embedded_outer = _identifier(
        anchor.get("outer_held_family_id"), "embedded outer family"
    )
    embedded_objectives = _mapping(
        anchor.get("exact_kl_objective_by_family_and_anchor"),
        "embedded anchor objectives",
    )
    expected = _build_anchor_receipt(
        all_development_family_ids=embedded_families,
        outer_held_family_id=embedded_outer,
        exact_anchor_objectives_by_family_and_anchor=embedded_objectives,
    )
    if _canonical(anchor) != _canonical(expected):
        raise ValueError("signed-continuum anchor chain drifted")


def _quadratic(anchor_receipt: Mapping[str, object]) -> tuple[object, ...]:
    scores = _mapping(
        anchor_receipt.get("family_equal_exact_kl_by_anchor"),
        "family-equal anchor objectives",
    )
    y_minus = _number(scores["signed_minus_one"], "minus-one objective")
    y_zero = _number(scores["signed_zero"], "zero objective")
    y_plus = _number(scores["signed_plus_one"], "plus-one objective")
    b = _finite_operation((y_plus - y_minus) / 2.0, "quadratic b")
    a = _finite_operation(
        (y_plus + y_minus) / 2.0 - y_zero, "quadratic a"
    )
    c = y_zero
    if a > 0.0:
        raw_vertex = _finite_operation(-b / (2.0 * a), "quadratic raw vertex")
        proposal = min(1.0, max(-1.0, raw_vertex))
        reason = "positive_curvature_clipped_vertex"
    else:
        objective, _absolute, proposal = min(
            (y_minus, 1.0, -1.0),
            (y_zero, 0.0, 0.0),
            (y_plus, 1.0, 1.0),
        )
        del objective
        raw_vertex = proposal
        reason = "nonpositive_curvature_best_exact_anchor"
    proposal = 0.0 if proposal == 0.0 else proposal
    return y_minus, y_zero, y_plus, a, b, c, raw_vertex, proposal, reason


def _build_quadratic_proposal_receipt(
    *, anchor_receipt: Mapping[str, object]
) -> dict[str, object]:
    _authenticate_anchor_chain(anchor_receipt)
    y_minus, y_zero, y_plus, a, b, c, raw, proposal, reason = _quadratic(
        anchor_receipt
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
        "quadratic_proposal.v20o",
        _PROPOSAL_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "stage": 1,
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "quadratic_formula": _QUADRATIC_FORMULA,
            "y_minus_one": y_minus,
            "y_minus_one_hex": _float_hex(y_minus),
            "y_zero": y_zero,
            "y_zero_hex": _float_hex(y_zero),
            "y_plus_one": y_plus,
            "y_plus_one_hex": _float_hex(y_plus),
            "quadratic_a": a,
            "quadratic_a_hex": _float_hex(a),
            "quadratic_b": b,
            "quadratic_b_hex": _float_hex(b),
            "quadratic_c": c,
            "quadratic_c_hex": _float_hex(c),
            "raw_vertex": raw,
            "raw_vertex_hex": _float_hex(raw),
            "proposed_signed_scalar": proposal,
            "proposed_signed_scalar_hex": _float_hex(proposal),
            "proposal_reason": reason,
            "proposal_inside_closed_signed_interval": -1.0 <= proposal <= 1.0,
            "proposal_uses_only_stage_1_family_equal_anchor_scores": True,
            "proposal_frozen_before_any_stage_2_vertex_objective": True,
            "quadratic_prediction_is_not_an_exact_stage_2_objective": True,
            "outer_held_family_used": False,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_signed_continuum_quadratic_proposal_receipt(
    *, anchor_receipt: Mapping[str, object]
) -> dict[str, object]:
    result = _build_quadratic_proposal_receipt(anchor_receipt=anchor_receipt)
    validate_soft_polarity_signed_continuum_quadratic_proposal_receipt(
        result, anchor_receipt=anchor_receipt
    )
    return result


def validate_soft_polarity_signed_continuum_quadratic_proposal_receipt(
    value: Mapping[str, object], *, anchor_receipt: Mapping[str, object]
) -> None:
    receipt = _mapping(value, "signed-continuum proposal receipt")
    expected = _build_quadratic_proposal_receipt(anchor_receipt=anchor_receipt)
    _sha(receipt.get("artifact_sha256"), "signed-continuum proposal artifact")
    if set(receipt) != set(expected) or _canonical(receipt) != _canonical(expected):
        raise ValueError("signed-continuum proposal receipt drifted")


def _vertex_candidate_receipt(
    proposal_receipt: Mapping[str, object]
) -> dict[str, object]:
    signed = _number(
        proposal_receipt.get("proposed_signed_scalar"), "proposed signed value"
    )
    if not -1.0 <= signed <= 1.0:
        raise ValueError("proposed signed value must be inside [-1,1]")
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
        "vertex_candidate.v20o",
        _VERTEX_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "candidate_id": "signed_vertex_stage2",
            "fixed_candidate_index": 1,
            "signed_scalar": signed,
            "signed_scalar_hex": _float_hex(signed),
            "frozen_before_any_stage_2_vertex_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
        },
    )


def _build_vertex_score_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> dict[str, object]:
    _authenticate_anchor_chain(anchor_receipt)
    validate_soft_polarity_signed_continuum_quadratic_proposal_receipt(
        proposal_receipt, anchor_receipt=anchor_receipt
    )
    inner = tuple(
        _identifier(item, "inner family")
        for item in _sequence(
            anchor_receipt.get("inner_oof_family_order"), "inner family order"
        )
    )
    raw = _mapping(exact_vertex_objectives_by_family, "vertex objectives")
    if set(raw) != set(inner):
        raise ValueError("vertex objective family geometry drifted")
    candidate = _vertex_candidate_receipt(proposal_receipt)
    signed = float(candidate["signed_scalar"])
    objectives = {
        family: _number(raw[family], f"{family} vertex objective")
        for family in inner
    }
    objective_hashes = {
        family: _hash(
            _OBJECTIVE_DOMAIN,
            {
                "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
                "stage": 2,
                "family_id": family,
                "candidate_id": "signed_vertex_stage2",
                "candidate_artifact_sha256": candidate["artifact_sha256"],
                "signed_scalar": signed,
                "objective": objectives[family],
                "objective_hex": _float_hex(objectives[family]),
                "objective_kind": _OBJECTIVE_KIND,
                "teacher_distribution_is_first_kl_argument": True,
                "candidate_distribution_is_second_kl_argument": True,
                "token_equal_within_family": True,
                "exact_float64_execution": True,
                "full_vocabulary_evaluated": True,
            },
        )
        for family in inner
    }
    aggregate = _family_equal(
        tuple(objectives[family] for family in inner),
        "vertex family-equal objective",
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
        "vertex_score.v20o",
        _VERTEX_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "stage": 2,
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "vertex_candidate_receipt": candidate,
            "vertex_candidate_artifact_sha256": candidate["artifact_sha256"],
            "signed_scalar": signed,
            "signed_scalar_hex": _float_hex(signed),
            "inner_oof_family_order": inner,
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family": objectives,
            "exact_kl_objective_sha256_by_family": objective_hashes,
            "family_equal_exact_kl": aggregate,
            "family_equal_exact_kl_hex": _float_hex(aggregate),
            "proposal_frozen_before_any_stage_2_vertex_objective": True,
            "objective_supplied_by_exact_stage_2_scoring": True,
            "quadratic_prediction_used_as_stage_2_objective": False,
            "outer_held_family_used": False,
            "exact_float64_execution": True,
            "full_vocabulary_evaluated": True,
            "exact_scalar_objectives_serialized": True,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_signed_continuum_vertex_score_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> dict[str, object]:
    result = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    validate_soft_polarity_signed_continuum_vertex_score_receipt(
        result,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    return result


def validate_soft_polarity_signed_continuum_vertex_score_receipt(
    value: Mapping[str, object],
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> None:
    receipt = _mapping(value, "signed-continuum vertex receipt")
    expected = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    _sha(receipt.get("artifact_sha256"), "signed-continuum vertex artifact")
    if set(receipt) != set(expected) or _canonical(receipt) != _canonical(expected):
        raise ValueError("signed-continuum vertex receipt drifted")


def _authenticate_vertex_chain(
    *,
    vertex_score_receipt: Mapping[str, object],
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
) -> None:
    vertex = _mapping(vertex_score_receipt, "signed-continuum vertex receipt")
    embedded = _mapping(
        vertex.get("exact_kl_objective_by_family"), "embedded vertex objectives"
    )
    expected = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=embedded,
    )
    if _canonical(vertex) != _canonical(expected):
        raise ValueError("signed-continuum vertex chain drifted")


def _final_candidate_receipt(
    *,
    candidate_id: str,
    fixed_index: int,
    signed_scalar: float,
    objective: float,
    source_artifact_sha256: str,
) -> dict[str, object]:
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_"
        "final_candidate.v20o",
        _FINAL_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "candidate_id": candidate_id,
            "fixed_candidate_index": fixed_index,
            "signed_scalar": signed_scalar,
            "signed_scalar_hex": _float_hex(signed_scalar),
            "family_equal_exact_kl": objective,
            "family_equal_exact_kl_hex": _float_hex(objective),
            "source_artifact_sha256": _sha(
                source_artifact_sha256, "candidate source artifact"
            ),
            "objective_is_exact_not_quadratic_prediction": True,
        },
    )


def _build_selection_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    vertex_score_receipt: Mapping[str, object],
) -> dict[str, object]:
    _authenticate_anchor_chain(anchor_receipt)
    validate_soft_polarity_signed_continuum_quadratic_proposal_receipt(
        proposal_receipt, anchor_receipt=anchor_receipt
    )
    _authenticate_vertex_chain(
        vertex_score_receipt=vertex_score_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
    )
    scores = _mapping(
        anchor_receipt.get("family_equal_exact_kl_by_anchor"), "anchor scores"
    )
    definitions = {
        str(item["anchor_id"]): _mapping(item, "anchor definition")
        for item in _sequence(
            anchor_receipt.get("anchor_definition_receipts"), "anchor definitions"
        )
    }
    vertex = _mapping(vertex_score_receipt, "vertex score receipt")
    specs = (
        (
            "signed_minus_one_anchor",
            0,
            -1.0,
            _number(scores["signed_minus_one"], "minus-one objective"),
            str(definitions["signed_minus_one"]["artifact_sha256"]),
        ),
        (
            "signed_vertex_stage2",
            1,
            _number(vertex["signed_scalar"], "vertex signed value"),
            _number(vertex["family_equal_exact_kl"], "vertex objective"),
            str(vertex["artifact_sha256"]),
        ),
        (
            "signed_zero_anchor",
            2,
            0.0,
            _number(scores["signed_zero"], "zero objective"),
            str(definitions["signed_zero"]["artifact_sha256"]),
        ),
        (
            "signed_plus_one_anchor",
            3,
            1.0,
            _number(scores["signed_plus_one"], "plus-one objective"),
            str(definitions["signed_plus_one"]["artifact_sha256"]),
        ),
    )
    candidates = tuple(
        _final_candidate_receipt(
            candidate_id=candidate_id,
            fixed_index=index,
            signed_scalar=signed,
            objective=objective,
            source_artifact_sha256=source,
        )
        for candidate_id, index, signed, objective, source in specs
    )
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["family_equal_exact_kl"]),
            abs(float(item["signed_scalar"])),
            float(item["signed_scalar"]),
            str(item["artifact_sha256"]),
        ),
    )
    ranking = tuple(str(item["candidate_id"]) for item in ordered)
    selected = ordered[0]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_signed_continuum_selection.v20o",
        _SELECTION_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256,
            "anchor_receipt": dict(anchor_receipt),
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "proposal_receipt": dict(proposal_receipt),
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "vertex_score_receipt": dict(vertex_score_receipt),
            "vertex_score_receipt_sha256": vertex_score_receipt["artifact_sha256"],
            "candidate_order": _FINAL_CANDIDATE_IDS,
            "candidate_receipts": candidates,
            "candidate_artifact_sha256s": {
                str(item["candidate_id"]): item["artifact_sha256"]
                for item in candidates
            },
            "exact_family_equal_kl_by_candidate": {
                str(item["candidate_id"]): item["family_equal_exact_kl"]
                for item in candidates
            },
            "signed_scalar_by_candidate": {
                str(item["candidate_id"]): item["signed_scalar"]
                for item in candidates
            },
            "candidate_ranking": ranking,
            "selection_key": _SELECTION_KEY,
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_artifact_sha256": selected["artifact_sha256"],
            "selected_fixed_candidate_index": selected["fixed_candidate_index"],
            "selected_signed_scalar": selected["signed_scalar"],
            "selected_signed_scalar_hex": selected["signed_scalar_hex"],
            "selected_family_equal_exact_kl": selected["family_equal_exact_kl"],
            "selected_family_equal_exact_kl_hex": selected[
                "family_equal_exact_kl_hex"
            ],
            "anchors_frozen_before_stage_1_scoring": True,
            "proposal_frozen_before_stage_2_scoring": True,
            "final_selection_uses_only_exact_objectives": True,
            "outer_held_family_used_for_fit_or_selection": False,
            "exact_scalar_objectives_serialized": True,
            "raw_model_tensors_serialized": False,
            "compression_claim_authorized": False,
            "speed_claim_authorized": False,
            "serving_authorized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_signed_continuum_selection_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    vertex_score_receipt: Mapping[str, object],
) -> dict[str, object]:
    result = _build_selection_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    validate_soft_polarity_signed_continuum_selection_receipt(
        result,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    return result


def validate_soft_polarity_signed_continuum_selection_receipt(
    value: Mapping[str, object],
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    vertex_score_receipt: Mapping[str, object],
) -> None:
    receipt = _mapping(value, "signed-continuum selection receipt")
    expected = _build_selection_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    _sha(receipt.get("artifact_sha256"), "signed-continuum selection artifact")
    if set(receipt) != set(expected) or _canonical(receipt) != _canonical(expected):
        raise ValueError("signed-continuum selection receipt drifted")


def build_soft_polarity_signed_continuum_fit_receipt(
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> dict[str, object]:
    anchors = build_soft_polarity_signed_continuum_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    proposal = build_soft_polarity_signed_continuum_quadratic_proposal_receipt(
        anchor_receipt=anchors
    )
    vertex = build_soft_polarity_signed_continuum_vertex_score_receipt(
        anchor_receipt=anchors,
        proposal_receipt=proposal,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    return build_soft_polarity_signed_continuum_selection_receipt(
        anchor_receipt=anchors,
        proposal_receipt=proposal,
        vertex_score_receipt=vertex,
    )


def validate_soft_polarity_signed_continuum_fit_receipt(
    value: Mapping[str, object],
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> None:
    receipt = _mapping(value, "signed-continuum fit receipt")
    expected = build_soft_polarity_signed_continuum_fit_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    _sha(receipt.get("artifact_sha256"), "signed-continuum fit artifact")
    if set(receipt) != set(expected) or _canonical(receipt) != _canonical(expected):
        raise ValueError("signed-continuum fit receipt drifted")
