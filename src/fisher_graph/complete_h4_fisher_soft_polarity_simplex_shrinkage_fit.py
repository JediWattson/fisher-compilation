"""Pure V20n two-stage continuous simplex-shrinkage fit protocol.

For one already-selected V20m response ``(r, u, v)``, V20n shrinks the
nonlinear terms continuously with

``q_lambda(z) = (1-lambda*u*z^2)*tanh(r*z) + lambda*v*z^2``.

This module performs training-only selection of ``lambda``.  Stage 1 freezes
the three anchors ``0``, ``1/2``, and ``1`` before opening their seven
family-equal, token-mean, exact-float64, full-vocabulary
``KL(teacher || candidate)`` scores.  If those aggregate scores are ``y0``,
``yhalf``, and ``y1``, the unique interpolating quadratic is represented by

``a = 2*(y1-y0) - 4*(yhalf-y0)`` and ``b = (y1-y0) - a``.

When ``a > 0``, stage 1 proposes the clipped vertex ``-b/(2*a)``; otherwise it
proposes the better endpoint, breaking an endpoint tie toward zero.  The
proposal is frozen before stage 2 opens a separately supplied exact score at
that lambda.  Final selection compares only the exact objectives at lambda
zero, the frozen proposal, and lambda one.  Ties prefer smaller lambda, then
the candidate artifact hash.

The outer-held family and all later splits are absent from both stages.  The
module performs no model execution and grants no outer-score, compression,
speed, serving, or held-evaluation authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

__all__ = [
    "SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS",
    "SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS",
    "SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256",
    "build_soft_polarity_simplex_shrinkage_anchor_receipt",
    "build_soft_polarity_simplex_shrinkage_fit_receipt",
    "build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt",
    "build_soft_polarity_simplex_shrinkage_selection_receipt",
    "build_soft_polarity_simplex_shrinkage_vertex_score_receipt",
    "validate_soft_polarity_simplex_shrinkage_anchor_receipt",
    "validate_soft_polarity_simplex_shrinkage_fit_receipt",
    "validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt",
    "validate_soft_polarity_simplex_shrinkage_selection_receipt",
    "validate_soft_polarity_simplex_shrinkage_vertex_score_receipt",
]


SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS = (
    "lambda_0",
    "lambda_half",
    "lambda_1",
)
SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS = (0.0, 0.5, 1.0)

_ANCHOR_LAMBDA_BY_ID = dict(
    zip(
        SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS,
        SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS,
        strict=True,
    )
)
_FINAL_CANDIDATE_IDS = (
    "lambda_0_anchor",
    "lambda_vertex_stage2",
    "lambda_1_anchor",
)
_DEVELOPMENT_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORMULA = (
    "q_lambda(z)=(1-lambda*u*z^2)*tanh(r*z)+lambda*v*z^2"
)
_OBJECTIVE_KIND = (
    "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
)
_AGGREGATE_OBJECTIVE_KIND = (
    "family_equal_token_mean_exact_float64_full_vocabulary_"
    "kl_teacher_to_candidate"
)
_QUADRATIC_FORMULA = (
    "a=2*(y1-y0)-4*(yhalf-y0);b=(y1-y0)-a"
)
_SELECTION_KEY = (
    "exact_family_equal_kl_then_smaller_lambda_then_candidate_artifact_sha256"
)

_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"protocol:v20n\0"
)
_ANCHOR_DEFINITION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"anchor-definition:v20n\0"
)
_OBJECTIVE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"objective:v20n\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"family:v20n\0"
)
_ANCHOR_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"anchors:v20n\0"
)
_PROPOSAL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"proposal:v20n\0"
)
_VERTEX_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"vertex-candidate:v20n\0"
)
_VERTEX_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"vertex-score:v20n\0"
)
_FINAL_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"final-candidate:v20n\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-shrinkage-fit:"
    b"selection:v20n\0"
)

_DATA_BOUNDARY = {
    "role": "v20n_training_only_two_stage_continuous_simplex_shrinkage_fit",
    "development_family_count": _DEVELOPMENT_FAMILY_COUNT,
    "inner_oof_family_count": _INNER_FAMILY_COUNT,
    "stage_1_anchor_lambdas_frozen_before_anchor_objectives": True,
    "stage_1_uses_only_three_anchor_objectives": True,
    "stage_1_quadratic_is_proposal_only": True,
    "stage_2_vertex_lambda_frozen_before_vertex_objectives": True,
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
    "protocol": (
        "v20n_training_only_two_stage_continuous_simplex_shrinkage_selection"
    ),
    "scientific_status": "development_only_after_v20m",
    "formula": _FORMULA,
    "lambda_constraint": "0<=lambda<=1",
    "anchor_order": SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS,
    "anchor_lambdas": SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS,
    "anchor_lambda_hex": tuple(
        value.hex()
        for value in SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS
    ),
    "quadratic_formula": _QUADRATIC_FORMULA,
    "proposal_rule": (
        "if_a_positive_clip_negative_b_over_two_a_else_better_endpoint_"
        "with_endpoint_tie_to_zero"
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


SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256 = _hash(
    _PROTOCOL_DOMAIN, _PROTOCOL
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
    if len(inner) != _INNER_FAMILY_COUNT:
        raise ValueError("exactly seven inner-OOF families are required")
    return families, outer, inner


def _anchor_definition(index: int) -> dict[str, object]:
    anchor_id = SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS[index]
    shrinkage = SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS[index]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_"
        "anchor_definition.v20n",
        _ANCHOR_DEFINITION_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "anchor_id": anchor_id,
            "anchor_index": index,
            "lambda": shrinkage,
            "lambda_hex": _float_hex(shrinkage),
            "lambda_inside_closed_unit_interval": 0.0 <= shrinkage <= 1.0,
            "frozen_before_any_stage_1_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
        },
    )


def _anchor_objectives(
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
    *,
    inner_families: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, object]],
]:
    raw_families = _mapping(
        exact_anchor_objectives_by_family_and_anchor,
        "stage-1 exact anchor objectives",
    )
    if set(raw_families) != set(inner_families):
        raise ValueError("stage-1 anchor objective family geometry drifted")

    objectives: dict[str, dict[str, float]] = {}
    objective_hex: dict[str, dict[str, str]] = {}
    objective_hashes: dict[str, dict[str, str]] = {}
    family_receipts: dict[str, dict[str, object]] = {}
    definitions = {
        anchor_id: _anchor_definition(index)
        for index, anchor_id in enumerate(
            SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS
        )
    }
    for family in inner_families:
        raw_anchors = _mapping(
            raw_families[family], f"{family} stage-1 exact anchor objectives"
        )
        if set(raw_anchors) != set(
            SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS
        ):
            raise ValueError("stage-1 anchor objective geometry drifted")
        objectives[family] = {}
        objective_hex[family] = {}
        objective_hashes[family] = {}
        for index, anchor_id in enumerate(
            SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS
        ):
            objective = _number(
                raw_anchors[anchor_id],
                f"{family} {anchor_id} exact KL objective",
            )
            objectives[family][anchor_id] = objective
            objective_hex[family][anchor_id] = _float_hex(objective)
            objective_hashes[family][anchor_id] = _hash(
                _OBJECTIVE_DOMAIN,
                {
                    "protocol_sha256": (
                        SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
                    ),
                    "stage": 1,
                    "family_id": family,
                    "anchor_id": anchor_id,
                    "anchor_index": index,
                    "anchor_artifact_sha256": definitions[anchor_id][
                        "artifact_sha256"
                    ],
                    "lambda": _ANCHOR_LAMBDA_BY_ID[anchor_id],
                    "lambda_hex": _float_hex(
                        _ANCHOR_LAMBDA_BY_ID[anchor_id]
                    ),
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
        family_receipts[family] = _finish(
            "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_"
            "anchor_family.v20n",
            _FAMILY_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
                ),
                "stage": 1,
                "family_id": family,
                "anchor_order": (
                    SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS
                ),
                "exact_kl_objective_by_anchor": objectives[family],
                "exact_kl_objective_hex_by_anchor": objective_hex[family],
                "exact_kl_objective_sha256_by_anchor": (
                    objective_hashes[family]
                ),
                "anchors_frozen_before_family_score": True,
                "outer_held_family_used": False,
                "exact_float64_execution": True,
                "full_vocabulary_evaluated": True,
                "raw_model_tensors_serialized": False,
            },
        )
    return objectives, objective_hex, objective_hashes, family_receipts


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
    objectives, objective_hex, objective_hashes, family_receipts = (
        _anchor_objectives(
            exact_anchor_objectives_by_family_and_anchor,
            inner_families=inner,
        )
    )
    definitions = tuple(
        _anchor_definition(index)
        for index in range(
            len(SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS)
        )
    )
    definition_hashes = {
        definition["anchor_id"]: definition["artifact_sha256"]
        for definition in definitions
    }
    family_equal: dict[str, float] = {}
    family_equal_hex: dict[str, str] = {}
    for anchor_id in SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS:
        aggregate = _family_equal(
            tuple(objectives[family][anchor_id] for family in inner),
            f"{anchor_id} family-equal exact KL objective",
        )
        family_equal[anchor_id] = aggregate
        family_equal_hex[anchor_id] = _float_hex(aggregate)
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_anchors.v20n",
        _ANCHOR_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "stage": 1,
            "formula": _FORMULA,
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "inner_oof_family_order": inner,
            "anchor_order": SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_IDS,
            "anchor_lambdas": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS
            ),
            "anchor_lambda_hex": tuple(
                _float_hex(value)
                for value in SOFT_POLARITY_SIMPLEX_SHRINKAGE_ANCHOR_LAMBDAS
            ),
            "anchor_definition_receipts": definitions,
            "anchor_definition_sha256s": definition_hashes,
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family_and_anchor": objectives,
            "exact_kl_objective_hex_by_family_and_anchor": objective_hex,
            "exact_kl_objective_sha256_by_family_and_anchor": (
                objective_hashes
            ),
            "family_receipts": family_receipts,
            "family_receipt_sha256s": {
                family: family_receipts[family]["artifact_sha256"]
                for family in inner
            },
            "family_equal_exact_kl_by_anchor": family_equal,
            "family_equal_exact_kl_hex_by_anchor": family_equal_hex,
            "anchor_lambdas_frozen_before_any_stage_1_objective": True,
            "stage_1_complete_before_quadratic_proposal": True,
            "outer_held_family_used": False,
            "exact_scalar_objectives_serialized": True,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_simplex_shrinkage_anchor_receipt(
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Build the authenticated stage-1 three-anchor score receipt."""

    result = _build_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    validate_soft_polarity_simplex_shrinkage_anchor_receipt(
        result,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    return result


def validate_soft_polarity_simplex_shrinkage_anchor_receipt(
    value: Mapping[str, object],
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    """Replay stage 1 from authoritative family-by-anchor objectives."""

    receipt = _mapping(value, "simplex-shrinkage anchor receipt")
    expected = _build_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-shrinkage anchor receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-shrinkage anchor artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-shrinkage anchor receipt content drifted")


def _authenticate_anchor_chain(receipt: Mapping[str, object]) -> None:
    anchor = _mapping(receipt, "simplex-shrinkage anchor receipt")
    _sha(anchor.get("artifact_sha256"), "simplex-shrinkage anchor artifact")
    artifact = dict(anchor)
    supplied = artifact.pop("artifact_sha256")
    if supplied != _hash(_ANCHOR_DOMAIN, artifact):
        raise ValueError("simplex-shrinkage anchor artifact hash drifted")
    if anchor.get("protocol_sha256") != (
        SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
    ):
        raise ValueError("simplex-shrinkage anchor protocol drifted")
    embedded_families = _sequence(
        anchor.get("all_development_family_ids"),
        "embedded development family ids",
    )
    embedded_outer = _identifier(
        anchor.get("outer_held_family_id"), "embedded outer held family id"
    )
    embedded_objectives = _mapping(
        anchor.get("exact_kl_objective_by_family_and_anchor"),
        "embedded stage-1 anchor objectives",
    )
    expected = _build_anchor_receipt(
        all_development_family_ids=embedded_families,
        outer_held_family_id=embedded_outer,
        exact_anchor_objectives_by_family_and_anchor=embedded_objectives,
    )
    if _canonical(anchor) != _canonical(expected):
        raise ValueError("simplex-shrinkage anchor chain content drifted")


def _quadratic(anchor_receipt: Mapping[str, object]) -> tuple[float, ...]:
    aggregates = _mapping(
        anchor_receipt.get("family_equal_exact_kl_by_anchor"),
        "family-equal anchor objectives",
    )
    y0 = _number(aggregates["lambda_0"], "lambda zero objective")
    yhalf = _number(aggregates["lambda_half"], "lambda half objective")
    y1 = _number(aggregates["lambda_1"], "lambda one objective")
    delta_1 = _finite_operation(y1 - y0, "quadratic endpoint delta")
    delta_half = _finite_operation(
        yhalf - y0, "quadratic midpoint delta"
    )
    a = _finite_operation(
        2.0 * delta_1 - 4.0 * delta_half,
        "quadratic coefficient a",
    )
    b = _finite_operation(delta_1 - a, "quadratic coefficient b")
    if a > 0.0:
        raw_vertex = _finite_operation(
            -b / (2.0 * a), "quadratic raw vertex"
        )
        proposal = min(1.0, max(0.0, raw_vertex))
        reason = "positive_curvature_clipped_vertex"
    else:
        proposal = 0.0 if y0 <= y1 else 1.0
        raw_vertex = proposal
        reason = "nonpositive_curvature_better_endpoint"
    proposal = 0.0 if proposal == 0.0 else proposal
    return y0, yhalf, y1, a, b, raw_vertex, proposal, reason


def _build_quadratic_proposal_receipt(
    *, anchor_receipt: Mapping[str, object]
) -> dict[str, object]:
    _authenticate_anchor_chain(anchor_receipt)
    y0, yhalf, y1, a, b, raw_vertex, proposal, reason = _quadratic(
        anchor_receipt
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_"
        "quadratic_proposal.v20n",
        _PROPOSAL_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "stage": 1,
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "quadratic_formula": _QUADRATIC_FORMULA,
            "y0": y0,
            "y0_hex": _float_hex(y0),
            "yhalf": yhalf,
            "yhalf_hex": _float_hex(yhalf),
            "y1": y1,
            "y1_hex": _float_hex(y1),
            "quadratic_a": a,
            "quadratic_a_hex": _float_hex(a),
            "quadratic_b": b,
            "quadratic_b_hex": _float_hex(b),
            "raw_vertex": raw_vertex,
            "raw_vertex_hex": _float_hex(raw_vertex),
            "proposed_lambda": proposal,
            "proposed_lambda_hex": _float_hex(proposal),
            "proposal_reason": reason,
            "proposal_inside_closed_unit_interval": 0.0 <= proposal <= 1.0,
            "proposal_uses_only_stage_1_family_equal_anchor_scores": True,
            "proposal_frozen_before_any_stage_2_vertex_objective": True,
            "quadratic_prediction_is_not_an_exact_stage_2_objective": True,
            "outer_held_family_used": False,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
    *, anchor_receipt: Mapping[str, object]
) -> dict[str, object]:
    """Freeze the deterministic quadratic proposal before stage-2 scoring."""

    result = _build_quadratic_proposal_receipt(anchor_receipt=anchor_receipt)
    validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        result, anchor_receipt=anchor_receipt
    )
    return result


def validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
    value: Mapping[str, object],
    *,
    anchor_receipt: Mapping[str, object],
) -> None:
    """Replay the proposal exactly from its authenticated stage-1 anchors."""

    receipt = _mapping(value, "simplex-shrinkage proposal receipt")
    expected = _build_quadratic_proposal_receipt(
        anchor_receipt=anchor_receipt
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-shrinkage proposal receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-shrinkage proposal artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-shrinkage proposal receipt content drifted")


def _vertex_candidate_receipt(
    proposal_receipt: Mapping[str, object]
) -> dict[str, object]:
    shrinkage = _number(
        proposal_receipt.get("proposed_lambda"), "proposed lambda"
    )
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("proposed lambda must be inside [0,1]")
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_"
        "vertex_candidate.v20n",
        _VERTEX_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "candidate_id": "lambda_vertex_stage2",
            "fixed_candidate_index": 1,
            "lambda": shrinkage,
            "lambda_hex": _float_hex(shrinkage),
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
    validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        proposal_receipt, anchor_receipt=anchor_receipt
    )
    inner = tuple(
        _identifier(family, "inner family id")
        for family in _sequence(
            anchor_receipt.get("inner_oof_family_order"),
            "inner family order",
        )
    )
    raw_objectives = _mapping(
        exact_vertex_objectives_by_family,
        "stage-2 exact vertex objectives",
    )
    if set(raw_objectives) != set(inner):
        raise ValueError("stage-2 vertex objective family geometry drifted")
    candidate = _vertex_candidate_receipt(proposal_receipt)
    shrinkage = float(candidate["lambda"])
    objectives: dict[str, float] = {}
    objective_hex: dict[str, str] = {}
    objective_hashes: dict[str, str] = {}
    for family in inner:
        objective = _number(
            raw_objectives[family], f"{family} stage-2 exact vertex objective"
        )
        objectives[family] = objective
        objective_hex[family] = _float_hex(objective)
        objective_hashes[family] = _hash(
            _OBJECTIVE_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
                ),
                "stage": 2,
                "family_id": family,
                "candidate_id": "lambda_vertex_stage2",
                "candidate_artifact_sha256": candidate["artifact_sha256"],
                "lambda": shrinkage,
                "lambda_hex": _float_hex(shrinkage),
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
    aggregate = _family_equal(
        tuple(objectives[family] for family in inner),
        "stage-2 vertex family-equal exact KL objective",
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_vertex_score.v20n",
        _VERTEX_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "stage": 2,
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "vertex_candidate_receipt": candidate,
            "vertex_candidate_artifact_sha256": candidate["artifact_sha256"],
            "lambda": shrinkage,
            "lambda_hex": _float_hex(shrinkage),
            "inner_oof_family_order": inner,
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family": objectives,
            "exact_kl_objective_hex_by_family": objective_hex,
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


def build_soft_polarity_simplex_shrinkage_vertex_score_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> dict[str, object]:
    """Bind the frozen proposal to its supplied exact stage-2 score."""

    result = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    validate_soft_polarity_simplex_shrinkage_vertex_score_receipt(
        result,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    return result


def validate_soft_polarity_simplex_shrinkage_vertex_score_receipt(
    value: Mapping[str, object],
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> None:
    """Replay stage 2 from its frozen proposal and authoritative scores."""

    receipt = _mapping(value, "simplex-shrinkage vertex score receipt")
    expected = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-shrinkage vertex receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-shrinkage vertex artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-shrinkage vertex receipt content drifted")


def _authenticate_vertex_chain(
    *,
    vertex_score_receipt: Mapping[str, object],
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
) -> None:
    vertex = _mapping(
        vertex_score_receipt, "simplex-shrinkage vertex score receipt"
    )
    _sha(vertex.get("artifact_sha256"), "simplex-shrinkage vertex artifact")
    embedded_objectives = _mapping(
        vertex.get("exact_kl_objective_by_family"),
        "embedded stage-2 vertex objectives",
    )
    expected = _build_vertex_score_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=embedded_objectives,
    )
    if _canonical(vertex) != _canonical(expected):
        raise ValueError("simplex-shrinkage vertex chain content drifted")


def _final_candidate_receipt(
    *,
    candidate_id: str,
    fixed_index: int,
    shrinkage: float,
    objective: float,
    source_artifact_sha256: str,
) -> dict[str, object]:
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_final_candidate.v20n",
        _FINAL_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "candidate_id": candidate_id,
            "fixed_candidate_index": fixed_index,
            "lambda": shrinkage,
            "lambda_hex": _float_hex(shrinkage),
            "family_equal_exact_kl": objective,
            "family_equal_exact_kl_hex": _float_hex(objective),
            "source_artifact_sha256": _sha(
                source_artifact_sha256, "final candidate source artifact"
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
    validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        proposal_receipt, anchor_receipt=anchor_receipt
    )
    _authenticate_vertex_chain(
        vertex_score_receipt=vertex_score_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
    )
    vertex = _mapping(
        vertex_score_receipt, "simplex-shrinkage vertex score receipt"
    )

    anchor_scores = _mapping(
        anchor_receipt.get("family_equal_exact_kl_by_anchor"),
        "family-equal anchor objectives",
    )
    anchor_definitions = {
        str(item["anchor_id"]): _mapping(item, "anchor definition")
        for item in _sequence(
            anchor_receipt.get("anchor_definition_receipts"),
            "anchor definitions",
        )
    }
    candidate_specs = (
        (
            "lambda_0_anchor",
            0,
            0.0,
            _number(anchor_scores["lambda_0"], "lambda zero objective"),
            str(anchor_definitions["lambda_0"]["artifact_sha256"]),
        ),
        (
            "lambda_vertex_stage2",
            1,
            _number(vertex.get("lambda"), "vertex lambda"),
            _number(
                vertex.get("family_equal_exact_kl"),
                "vertex family-equal exact KL",
            ),
            str(vertex["artifact_sha256"]),
        ),
        (
            "lambda_1_anchor",
            2,
            1.0,
            _number(anchor_scores["lambda_1"], "lambda one objective"),
            str(anchor_definitions["lambda_1"]["artifact_sha256"]),
        ),
    )
    candidates = tuple(
        _final_candidate_receipt(
            candidate_id=candidate_id,
            fixed_index=index,
            shrinkage=shrinkage,
            objective=objective,
            source_artifact_sha256=source_hash,
        )
        for candidate_id, index, shrinkage, objective, source_hash in candidate_specs
    )
    ranking = tuple(
        candidate["candidate_id"]
        for candidate in sorted(
            candidates,
            key=lambda item: (
                float(item["family_equal_exact_kl"]),
                float(item["lambda"]),
                str(item["artifact_sha256"]),
            ),
        )
    )
    by_id = {str(item["candidate_id"]): item for item in candidates}
    selected = by_id[ranking[0]]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_selection.v20n",
        _SELECTION_DOMAIN,
        {
            "protocol_sha256": (
                SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "anchor_receipt": dict(anchor_receipt),
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "proposal_receipt": dict(proposal_receipt),
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "vertex_score_receipt": dict(vertex_score_receipt),
            "vertex_score_receipt_sha256": vertex_score_receipt[
                "artifact_sha256"
            ],
            "candidate_order": _FINAL_CANDIDATE_IDS,
            "candidate_receipts": candidates,
            "candidate_artifact_sha256s": {
                str(candidate["candidate_id"]): candidate["artifact_sha256"]
                for candidate in candidates
            },
            "exact_family_equal_kl_by_candidate": {
                str(candidate["candidate_id"]): candidate[
                    "family_equal_exact_kl"
                ]
                for candidate in candidates
            },
            "exact_family_equal_kl_hex_by_candidate": {
                str(candidate["candidate_id"]): candidate[
                    "family_equal_exact_kl_hex"
                ]
                for candidate in candidates
            },
            "lambda_by_candidate": {
                str(candidate["candidate_id"]): candidate["lambda"]
                for candidate in candidates
            },
            "lambda_hex_by_candidate": {
                str(candidate["candidate_id"]): candidate["lambda_hex"]
                for candidate in candidates
            },
            "candidate_ranking": ranking,
            "selection_key": _SELECTION_KEY,
            "selected_candidate_id": selected["candidate_id"],
            "selected_candidate_artifact_sha256": selected[
                "artifact_sha256"
            ],
            "selected_fixed_candidate_index": selected[
                "fixed_candidate_index"
            ],
            "selected_lambda": selected["lambda"],
            "selected_lambda_hex": selected["lambda_hex"],
            "selected_family_equal_exact_kl": selected[
                "family_equal_exact_kl"
            ],
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


def build_soft_polarity_simplex_shrinkage_selection_receipt(
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    vertex_score_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Select among exact lambda-zero, proposed-vertex, and lambda-one scores."""

    result = _build_selection_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    validate_soft_polarity_simplex_shrinkage_selection_receipt(
        result,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    return result


def validate_soft_polarity_simplex_shrinkage_selection_receipt(
    value: Mapping[str, object],
    *,
    anchor_receipt: Mapping[str, object],
    proposal_receipt: Mapping[str, object],
    vertex_score_receipt: Mapping[str, object],
) -> None:
    """Replay final selection from the authenticated two-stage receipt chain."""

    receipt = _mapping(value, "simplex-shrinkage selection receipt")
    expected = _build_selection_receipt(
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_score_receipt,
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-shrinkage selection receipt key set drifted")
    _sha(
        receipt.get("artifact_sha256"),
        "simplex-shrinkage selection artifact",
    )
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-shrinkage selection receipt content drifted")


def build_soft_polarity_simplex_shrinkage_fit_receipt(
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> dict[str, object]:
    """Run the pure two-stage receipt protocol without model or held execution."""

    anchors = build_soft_polarity_simplex_shrinkage_anchor_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
    )
    proposal = build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        anchor_receipt=anchors
    )
    vertex = build_soft_polarity_simplex_shrinkage_vertex_score_receipt(
        anchor_receipt=anchors,
        proposal_receipt=proposal,
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    return build_soft_polarity_simplex_shrinkage_selection_receipt(
        anchor_receipt=anchors,
        proposal_receipt=proposal,
        vertex_score_receipt=vertex,
    )


def validate_soft_polarity_simplex_shrinkage_fit_receipt(
    value: Mapping[str, object],
    *,
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_anchor_objectives_by_family_and_anchor: Mapping[
        str, Mapping[str, float]
    ],
    exact_vertex_objectives_by_family: Mapping[str, float],
) -> None:
    """Replay the complete V20n fit from authoritative exact objectives."""

    receipt = _mapping(value, "simplex-shrinkage fit receipt")
    expected = build_soft_polarity_simplex_shrinkage_fit_receipt(
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_anchor_objectives_by_family_and_anchor=(
            exact_anchor_objectives_by_family_and_anchor
        ),
        exact_vertex_objectives_by_family=exact_vertex_objectives_by_family,
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-shrinkage fit receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-shrinkage fit artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-shrinkage fit receipt content drifted")
