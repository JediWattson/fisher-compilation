"""Pure V20m conditional-inner-LOFO simplex-response selection protocol.

The scalar response is

``q(z) = (1-u*z^2)*tanh(r*z) + v*z^2``.

The coefficients obey ``r >= 0``, ``0 <= u <= 1/2``, and ``|v| <= u``.
This module freezes one zero control and eighteen ``(r, u, v)`` candidates
before it accepts any objective.  It then selects on the family-equal mean of
seven token-mean exact-float64 full-vocabulary
``KL(teacher || candidate)`` values.  The seven conditional inner
leave-one-family-out scores share one endpoint fitted on those seven families;
consequently this is calibrator selection on a fixed endpoint, not fully nested
endpoint fitting.  The outer-held family is absent from endpoint, direction,
reflection, and calibrator selection.

The module performs no model execution and exposes no outer-held score,
Calibration-B, validation, test, compression, speed, or serving authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

__all__ = [
    "SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS",
    "SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER",
    "SOFT_POLARITY_SIMPLEX_RESPONSE_TRIPLES",
    "build_soft_polarity_simplex_response_fit_receipt",
    "build_soft_polarity_simplex_response_inner_oof_selection_receipt",
    "build_soft_polarity_simplex_response_ladder_receipt",
    "soft_polarity_simplex_response_q",
    "validate_soft_polarity_simplex_response_fit_receipt",
    "validate_soft_polarity_simplex_response_inner_oof_selection_receipt",
    "validate_soft_polarity_simplex_response_ladder_receipt",
]


_R_VALUES = (1.0 / 8.0, 1.0 / 4.0)
_UV_VALUES = (
    (0.0, 0.0),
    (1.0 / 8.0, -1.0 / 8.0),
    (1.0 / 8.0, -1.0 / 16.0),
    (1.0 / 8.0, 0.0),
    (1.0 / 8.0, 1.0 / 16.0),
    (1.0 / 8.0, 1.0 / 8.0),
    (1.0 / 4.0, -1.0 / 8.0),
    (1.0 / 4.0, 0.0),
    (1.0 / 4.0, 1.0 / 8.0),
)
SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER = (
    (0.0, 0.0, 0.0),
    *((r, u, v) for r in _R_VALUES for u, v in _UV_VALUES),
)
SOFT_POLARITY_SIMPLEX_RESPONSE_TRIPLES = (
    SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER
)
SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS = tuple(
    f"simplex_response_{index:02d}"
    for index in range(len(SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER))
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DEVELOPMENT_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_FORMULA = "q(z)=(1-u*z^2)*tanh(r*z)+v*z^2"
_OBJECTIVE_KIND = (
    "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
)
_AGGREGATE_OBJECTIVE_KIND = (
    "family_equal_token_mean_exact_float64_full_vocabulary_"
    "kl_teacher_to_candidate"
)
_SELECTION_KEY = (
    "family_equal_token_mean_exact_float64_full_vocabulary_"
    "kl_teacher_to_candidate_then_smaller_u_then_smaller_abs_v_then_"
    "smaller_r_then_fixed_ladder_index_then_candidate_artifact_sha256"
)

_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:protocol:v20m\0"
)
_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:candidate:v20m\0"
)
_LADDER_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:ladder:v20m\0"
)
_OBJECTIVE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:objective:v20m\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:family:v20m\0"
)
_AGGREGATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:aggregate:v20m\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-simplex-response-fit:selection:v20m\0"
)

_DATA_BOUNDARY = {
    "role": "v20m_development_conditional_inner_lofo_simplex_response_selection_only",
    "development_family_count": _DEVELOPMENT_FAMILY_COUNT,
    "inner_oof_family_count": _INNER_FAMILY_COUNT,
    "conditional_inner_leave_one_family_out": True,
    "fixed_seven_family_endpoint_shared_across_inner_scores": True,
    "inner_endpoint_refit_per_inner_held_family": False,
    "inner_held_family_used_for_fixed_endpoint_fit": True,
    "fully_nested_endpoint_selection_claimed": False,
    "outer_held_family_absent_from_endpoint_direction_reflection_and_selection": True,
    "outer_held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "logits_or_h4_consumed": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_shadow_consumed": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
    "serving_authorized": False,
}
_PROTOCOL = {
    "protocol": "v20m_conditional_inner_lofo_soft_polarity_simplex_response_selection",
    "scientific_status": "development_only_after_v20l",
    "formula": _FORMULA,
    "coefficient_constraints": {
        "r": "r>=0",
        "u": "0<=u<=1/2",
        "v": "abs(v)<=u",
    },
    "candidate_order": SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
    "candidate_triples_r_u_v": SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER,
    "candidate_count": len(SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER),
    "zero_control_is_unique": True,
    "ladder_freeze": "before_any_conditional_inner_lofo_exact_kl_objective",
    "objective": _AGGREGATE_OBJECTIVE_KIND,
    "selection_key": _SELECTION_KEY,
    "inner_boundary": (
        "conditional_leave_one_family_out_calibrator_scoring_on_one_fixed_"
        "seven_family_endpoint_not_fully_nested_endpoint_fitting"
    ),
    "outer_boundary": (
        "one_of_eight_development_families_absent_from_endpoint_direction_"
        "reflection_and_calibrator_selection"
    ),
    "authorization": (
        "selection_receipt_only_no_outer_score_compression_speed_or_serving_authority"
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


SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256 = _hash(
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


def soft_polarity_simplex_response_q(
    z: float, *, r: float, u: float, v: float
) -> float:
    """Evaluate the finite scalar V20m simplex-response map."""

    selected_z = _number(z, "simplex-response input z")
    selected_r = _number(r, "simplex-response coefficient r")
    selected_u = _number(u, "simplex-response coefficient u")
    selected_v = _number(v, "simplex-response coefficient v")
    if selected_r < 0.0:
        raise ValueError("simplex-response coefficient r must be nonnegative")
    if not 0.0 <= selected_u <= 0.5:
        raise ValueError("simplex-response coefficient u must be inside [0,1/2]")
    if abs(selected_v) > selected_u:
        raise ValueError("simplex-response coefficient must satisfy abs(v)<=u")
    z_squared = selected_z * selected_z
    argument = selected_r * selected_z
    if not math.isfinite(z_squared) or not math.isfinite(argument):
        raise ValueError("simplex-response calibrator intermediate must be finite")
    try:
        result = math.fsum(
            (
                (1.0 - selected_u * z_squared) * math.tanh(argument),
                selected_v * z_squared,
            )
        )
    except OverflowError as error:
        raise ValueError(
            "simplex-response calibrator intermediate must be finite"
        ) from error
    if not math.isfinite(result):
        raise ValueError("simplex-response calibrator output must be finite")
    return 0.0 if result == 0.0 else result


def _candidate_receipt(index: int) -> dict[str, object]:
    candidate_id = SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS[index]
    r, u, v = SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER[index]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_response_candidate.v20m",
        _CANDIDATE_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256,
            "candidate_id": candidate_id,
            "ladder_index": index,
            "r": r,
            "u": u,
            "v": v,
            "formula": _FORMULA,
            "coefficient_constraints_satisfied": (
                r >= 0.0 and 0.0 <= u <= 0.5 and abs(v) <= u
            ),
            "zero_maps_to_zero": (
                soft_polarity_simplex_response_q(0.0, r=r, u=u, v=v) == 0.0
            ),
            "zero_control": index == 0,
            "candidate_frozen_before_any_conditional_inner_lofo_objective": True,
            "outer_held_objectives_consumed": False,
            "raw_model_tensors_serialized": False,
        },
    )


def _expected_ladder() -> dict[str, object]:
    candidates = tuple(
        _candidate_receipt(index)
        for index in range(len(SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER))
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_response_ladder.v20m",
        _LADDER_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256,
            "formula": _FORMULA,
            "coefficient_constraints": dict(_PROTOCOL["coefficient_constraints"]),
            "candidate_count": len(candidates),
            "candidate_order": SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
            "candidate_triples_r_u_v": SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER,
            "candidate_artifact_sha256s": {
                candidate_id: candidate["artifact_sha256"]
                for candidate_id, candidate in zip(
                    SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
                    candidates,
                    strict=True,
                )
            },
            "candidate_receipts": candidates,
            "zero_control_count": sum(
                candidate["zero_control"] for candidate in candidates
            ),
            "ladder_frozen_before_any_conditional_inner_lofo_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
            "raw_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_simplex_response_ladder_receipt() -> dict[str, object]:
    """Return the authenticated, immutable nineteen-candidate V20m ladder."""

    result = _expected_ladder()
    validate_soft_polarity_simplex_response_ladder_receipt(result)
    return result


def validate_soft_polarity_simplex_response_ladder_receipt(
    value: Mapping[str, object],
) -> None:
    """Authenticate every candidate, its order, fields, and ladder hash."""

    receipt = _mapping(value, "simplex-response ladder receipt")
    expected = _expected_ladder()
    if set(receipt) != set(expected):
        raise ValueError("simplex-response ladder receipt key set drifted")
    raw_candidates = _sequence(
        receipt.get("candidate_receipts"), "simplex-response ladder candidates"
    )
    if len(raw_candidates) != len(SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER):
        raise ValueError("simplex-response ladder candidate geometry drifted")
    for index, raw in enumerate(raw_candidates):
        candidate = _mapping(raw, "simplex-response candidate receipt")
        expected_candidate = _candidate_receipt(index)
        if set(candidate) != set(expected_candidate):
            raise ValueError("simplex-response candidate receipt key set drifted")
        _sha(candidate.get("artifact_sha256"), "simplex-response candidate artifact")
        if _canonical(candidate) != _canonical(expected_candidate):
            raise ValueError("simplex-response candidate receipt content drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-response ladder artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-response ladder receipt content drifted")


def _objective_rows(
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
    *,
    inner_families: tuple[str, ...],
    ladder: Mapping[str, object],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, str]],
    dict[str, dict[str, object]],
]:
    raw_families = _mapping(
        exact_objectives_by_family_and_candidate,
        "conditional inner-LOFO exact KL objectives",
    )
    if set(raw_families) != set(inner_families):
        raise ValueError("conditional inner-LOFO objective family geometry drifted")
    candidate_artifacts = _mapping(
        ladder.get("candidate_artifact_sha256s"),
        "simplex-response candidate artifacts",
    )
    objectives: dict[str, dict[str, float]] = {}
    objective_hashes: dict[str, dict[str, str]] = {}
    family_receipts: dict[str, dict[str, object]] = {}
    for family in inner_families:
        raw_candidates = _mapping(
            raw_families[family], f"{family} simplex-response exact KL objectives"
        )
        if set(raw_candidates) != set(SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS):
            raise ValueError(
                "conditional inner-LOFO objective candidate geometry drifted"
            )
        objectives[family] = {}
        objective_hashes[family] = {}
        for index, candidate_id in enumerate(
            SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS
        ):
            objective = _number(
                raw_candidates[candidate_id],
                f"{family} {candidate_id} exact KL objective",
            )
            objectives[family][candidate_id] = objective
            objective_hashes[family][candidate_id] = _hash(
                _OBJECTIVE_DOMAIN,
                {
                    "protocol_sha256": (
                        SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
                    ),
                    "family_id": family,
                    "candidate_id": candidate_id,
                    "ladder_index": index,
                    "candidate_artifact_sha256": _sha(
                        candidate_artifacts[candidate_id],
                        "simplex-response candidate artifact",
                    ),
                    "objective": objective,
                    "objective_kind": _OBJECTIVE_KIND,
                    "teacher_distribution_is_first_kl_argument": True,
                    "candidate_distribution_is_second_kl_argument": True,
                    "token_equal_within_family": True,
                    "exact_float64_execution": True,
                    "full_vocabulary_evaluated": True,
                },
            )
        family_receipts[family] = _finish(
            "fisher_graph.complete_h4_soft_polarity_simplex_response_family_oof.v20m",
            _FAMILY_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
                ),
                "family_id": family,
                "candidate_order": SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
                "objective_kind": _OBJECTIVE_KIND,
                "exact_kl_objective_by_candidate": objectives[family],
                "exact_kl_objective_sha256_by_candidate": objective_hashes[family],
                "candidate_ladder_artifact_sha256": ladder["artifact_sha256"],
                "all_candidates_frozen_before_family_score": True,
                "conditional_inner_leave_one_family_out": True,
                "fixed_seven_family_endpoint_shared_across_inner_scores": True,
                "inner_held_family_used_for_fixed_endpoint_fit": True,
                "outer_held_family_used": False,
                "exact_float64_execution": True,
                "full_vocabulary_evaluated": True,
                "exact_scalar_objectives_serialized": True,
                "raw_model_tensors_serialized": False,
            },
        )
    return objectives, objective_hashes, family_receipts


def _build_selection_receipt(
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Select one frozen response triple on seven family-equal exact KL scores."""

    validate_soft_polarity_simplex_response_ladder_receipt(ladder_receipt)
    ladder = _mapping(ladder_receipt, "simplex-response ladder receipt")
    families, outer, inner = _families(
        all_development_family_ids, outer_held_family_id
    )
    objectives, objective_hashes, family_receipts = _objective_rows(
        exact_objectives_by_family_and_candidate,
        inner_families=inner,
        ladder=ladder,
    )
    family_equal: dict[str, float] = {}
    aggregate_hashes: dict[str, str] = {}
    candidate_artifacts = _mapping(
        ladder["candidate_artifact_sha256s"],
        "simplex-response candidate artifacts",
    )
    for index, candidate_id in enumerate(
        SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS
    ):
        family_equal[candidate_id] = math.fsum(
            objectives[family][candidate_id] for family in inner
        ) / len(inner)
        aggregate_hashes[candidate_id] = _hash(
            _AGGREGATE_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
                ),
                "outer_held_family_id": outer,
                "inner_family_order": inner,
                "candidate_id": candidate_id,
                "ladder_index": index,
                "candidate_artifact_sha256": candidate_artifacts[candidate_id],
                "family_receipt_sha256s": {
                    family: family_receipts[family]["artifact_sha256"]
                    for family in inner
                },
                "exact_kl_objective_by_family": {
                    family: objectives[family][candidate_id] for family in inner
                },
                "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
                "family_equal_exact_kl": family_equal[candidate_id],
            },
        )
    candidate_by_id = {
        _identifier(candidate["candidate_id"], "simplex-response candidate id"): candidate
        for candidate in _sequence(
            ladder["candidate_receipts"], "simplex-response candidate receipts"
        )
    }
    ranking = tuple(
        sorted(
            SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
            key=lambda candidate_id: (
                family_equal[candidate_id],
                float(candidate_by_id[candidate_id]["u"]),
                abs(float(candidate_by_id[candidate_id]["v"])),
                float(candidate_by_id[candidate_id]["r"]),
                int(candidate_by_id[candidate_id]["ladder_index"]),
                str(candidate_by_id[candidate_id]["artifact_sha256"]),
            ),
        )
    )
    selected_id = ranking[0]
    selected = candidate_by_id[selected_id]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_simplex_response_selection.v20m",
        _SELECTION_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256,
            "ladder_artifact_sha256": ladder["artifact_sha256"],
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "inner_oof_family_order": inner,
            "candidate_order": SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
            "candidate_artifact_sha256s": dict(candidate_artifacts),
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family_and_candidate": objectives,
            "exact_kl_objective_sha256_by_family_and_candidate": objective_hashes,
            "family_oof_receipts": family_receipts,
            "family_oof_receipt_sha256s": {
                family: family_receipts[family]["artifact_sha256"]
                for family in inner
            },
            "family_equal_exact_kl_by_candidate": family_equal,
            "aggregate_artifact_sha256_by_candidate": aggregate_hashes,
            "candidate_ranking": ranking,
            "selection_key": _SELECTION_KEY,
            "selected_candidate_id": selected_id,
            "selected_candidate_artifact_sha256": selected["artifact_sha256"],
            "selected_ladder_index": selected["ladder_index"],
            "selected_r": selected["r"],
            "selected_u": selected["u"],
            "selected_v": selected["v"],
            "selected_family_equal_exact_kl": family_equal[selected_id],
            "selected_aggregate_artifact_sha256": aggregate_hashes[selected_id],
            "candidate_ladder_frozen_before_any_conditional_inner_lofo_objective": True,
            "selection_frozen_before_outer_held_score": True,
            "conditional_inner_leave_one_family_out": True,
            "fixed_seven_family_endpoint_shared_across_inner_scores": True,
            "inner_endpoint_refit_per_inner_held_family": False,
            "inner_held_family_used_for_fixed_endpoint_fit": True,
            "fully_nested_endpoint_selection_claimed": False,
            "outer_held_family_used_for_fit_or_selection": False,
            "all_objectives_token_mean_exact_float64_full_vocabulary_kl": True,
            "exact_scalar_objectives_serialized": True,
            "raw_model_tensors_serialized": False,
            "compression_claim_authorized": False,
            "speed_claim_authorized": False,
            "serving_authorized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_simplex_response_inner_oof_selection_receipt(
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Select one V20m response triple from conditional-inner-LOFO KL scores."""

    result = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    validate_soft_polarity_simplex_response_inner_oof_selection_receipt(
        result,
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    return result


def validate_soft_polarity_simplex_response_inner_oof_selection_receipt(
    value: Mapping[str, object],
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    """Replay the full V20m selection from authoritative exact KL values."""

    receipt = _mapping(value, "simplex-response selection receipt")
    expected = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    if set(receipt) != set(expected):
        raise ValueError("simplex-response selection receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "simplex-response selection artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("simplex-response selection receipt content drifted")


build_soft_polarity_simplex_response_fit_receipt = (
    build_soft_polarity_simplex_response_inner_oof_selection_receipt
)
validate_soft_polarity_simplex_response_fit_receipt = (
    validate_soft_polarity_simplex_response_inner_oof_selection_receipt
)
