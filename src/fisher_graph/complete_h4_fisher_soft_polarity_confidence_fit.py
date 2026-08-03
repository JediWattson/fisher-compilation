"""Pure V20j inner-OOF confidence-calibrator selection protocol.

The scalar confidence map is ``q(z) = tanh(a*z + b*z**3)``.  This module
freezes the twelve ``(a, b)`` candidates below before it accepts any exact
inner-OOF objective, commits every family/candidate objective, and selects by
family-equal exact NLL.  It performs no model execution and has no API for an
outer-held score, Calibration-B, validation, test, or serving evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

__all__ = [
    "SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS",
    "SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_CONFIDENCE_LADDER",
    "SOFT_POLARITY_CONFIDENCE_PAIRS",
    "build_soft_polarity_confidence_fit_receipt",
    "build_soft_polarity_confidence_inner_oof_selection_receipt",
    "build_soft_polarity_confidence_ladder_receipt",
    "soft_polarity_confidence_q",
    "validate_soft_polarity_confidence_fit_receipt",
    "validate_soft_polarity_confidence_inner_oof_selection_receipt",
    "validate_soft_polarity_confidence_ladder_receipt",
]


SOFT_POLARITY_CONFIDENCE_LADDER = (
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
SOFT_POLARITY_CONFIDENCE_PAIRS = SOFT_POLARITY_CONFIDENCE_LADDER
SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS = tuple(
    f"confidence_{index:02d}" for index in range(len(SOFT_POLARITY_CONFIDENCE_LADDER))
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DEVELOPMENT_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_FORMULA = "q(z)=tanh(a*z+b*z^3)"

_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:protocol:v20j\0"
)
_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:candidate:v20j\0"
)
_LADDER_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:ladder:v20j\0"
)
_OBJECTIVE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:objective:v20j\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:family:v20j\0"
)
_AGGREGATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:aggregate:v20j\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-confidence-fit:selection:v20j\0"
)

_DATA_BOUNDARY = {
    "role": "v20j_development_inner_oof_confidence_calibrator_selection_only",
    "development_family_count": _DEVELOPMENT_FAMILY_COUNT,
    "inner_oof_family_count": _INNER_FAMILY_COUNT,
    "outer_held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "logits_or_h4_consumed": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_shadow_consumed": False,
    "serving_authorized": False,
}
_PROTOCOL = {
    "protocol": "v20j_inner_oof_soft_polarity_confidence_calibrator_selection",
    "scientific_status": "development_only_after_v20i",
    "formula": _FORMULA,
    "candidate_order": SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
    "candidate_pairs": SOFT_POLARITY_CONFIDENCE_LADDER,
    "candidate_count": len(SOFT_POLARITY_CONFIDENCE_LADDER),
    "ladder_freeze": "before_any_inner_oof_exact_objective",
    "objective": "family_equal_exact_negative_log_likelihood",
    "selection_key": (
        "minimum_family_equal_exact_nll_then_smaller_a_plus_b_then_smaller_b_"
        "then_fixed_ladder_index_then_candidate_artifact_sha256"
    ),
    "outer_boundary": "one_of_eight_development_families_held_from_selection",
    "authorization": "selection_receipt_only_no_outer_score_or_serving_authority",
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


SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256 = _hash(
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


def soft_polarity_confidence_q(
    z: float, *, a: float, b: float
) -> float:
    """Evaluate the finite scalar confidence map ``tanh(a*z+b*z**3)``."""

    selected_z = _number(z, "confidence input z")
    selected_a = _number(a, "confidence coefficient a")
    selected_b = _number(b, "confidence coefficient b")
    if selected_a < 0.0 or selected_b < 0.0:
        raise ValueError("confidence coefficients must be nonnegative")
    try:
        logit = math.fsum(
            (selected_a * selected_z, selected_b * selected_z**3)
        )
    except OverflowError as error:
        raise ValueError("confidence calibrator intermediate must be finite") from error
    if not math.isfinite(logit):
        raise ValueError("confidence calibrator intermediate must be finite")
    result = math.tanh(logit)
    if not math.isfinite(result):
        raise ValueError("confidence calibrator output must be finite")
    return 0.0 if result == 0.0 else result


def _candidate_receipt(index: int) -> dict[str, object]:
    candidate_id = SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS[index]
    a, b = SOFT_POLARITY_CONFIDENCE_LADDER[index]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_confidence_candidate.v20j",
        _CANDIDATE_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256,
            "candidate_id": candidate_id,
            "ladder_index": index,
            "a": a,
            "b": b,
            "a_plus_b": math.fsum((a, b)),
            "formula": _FORMULA,
            "zero_maps_to_zero": soft_polarity_confidence_q(0.0, a=a, b=b)
            == 0.0,
            "candidate_frozen_before_any_inner_oof_objective": True,
            "outer_held_objectives_consumed": False,
            "raw_objectives_or_model_tensors_serialized": False,
        },
    )


def _expected_ladder() -> dict[str, object]:
    candidates = tuple(
        _candidate_receipt(index)
        for index in range(len(SOFT_POLARITY_CONFIDENCE_LADDER))
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_confidence_ladder.v20j",
        _LADDER_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256,
            "formula": _FORMULA,
            "candidate_count": len(candidates),
            "candidate_order": SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
            "candidate_pairs": SOFT_POLARITY_CONFIDENCE_LADDER,
            "candidate_artifact_sha256s": {
                candidate_id: candidate["artifact_sha256"]
                for candidate_id, candidate in zip(
                    SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
                    candidates,
                    strict=True,
                )
            },
            "candidate_receipts": candidates,
            "ladder_frozen_before_any_inner_oof_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
            "raw_objectives_or_model_tensors_serialized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_confidence_ladder_receipt() -> dict[str, object]:
    """Return the authenticated, immutable twelve-pair V20j ladder."""

    result = _expected_ladder()
    validate_soft_polarity_confidence_ladder_receipt(result)
    return result


def validate_soft_polarity_confidence_ladder_receipt(
    value: Mapping[str, object],
) -> None:
    """Authenticate every candidate, its order, fields, and ladder hash."""

    receipt = _mapping(value, "confidence ladder receipt")
    expected = _expected_ladder()
    if set(receipt) != set(expected):
        raise ValueError("confidence ladder receipt key set drifted")
    raw_candidates = _sequence(
        receipt.get("candidate_receipts"), "confidence ladder candidates"
    )
    if len(raw_candidates) != len(SOFT_POLARITY_CONFIDENCE_LADDER):
        raise ValueError("confidence ladder candidate geometry drifted")
    for index, raw in enumerate(raw_candidates):
        candidate = _mapping(raw, "confidence candidate receipt")
        expected_candidate = _candidate_receipt(index)
        if set(candidate) != set(expected_candidate):
            raise ValueError("confidence candidate receipt key set drifted")
        _sha(candidate.get("artifact_sha256"), "confidence candidate artifact")
        if _canonical(candidate) != _canonical(expected_candidate):
            raise ValueError("confidence candidate receipt content drifted")
    _sha(receipt.get("artifact_sha256"), "confidence ladder artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("confidence ladder receipt content drifted")


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
        "inner-OOF exact objectives",
    )
    if set(raw_families) != set(inner_families):
        raise ValueError("inner-OOF objective family geometry drifted")
    candidate_artifacts = _mapping(
        ladder.get("candidate_artifact_sha256s"),
        "confidence candidate artifacts",
    )
    objectives: dict[str, dict[str, float]] = {}
    objective_hashes: dict[str, dict[str, str]] = {}
    family_receipts: dict[str, dict[str, object]] = {}
    for family in inner_families:
        raw_candidates = _mapping(
            raw_families[family], f"{family} confidence objectives"
        )
        if set(raw_candidates) != set(SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS):
            raise ValueError("inner-OOF objective candidate geometry drifted")
        objectives[family] = {}
        objective_hashes[family] = {}
        for index, candidate_id in enumerate(
            SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
        ):
            objective = _number(
                raw_candidates[candidate_id],
                f"{family} {candidate_id} exact objective",
            )
            objectives[family][candidate_id] = objective
            objective_hashes[family][candidate_id] = _hash(
                _OBJECTIVE_DOMAIN,
                {
                    "protocol_sha256": (
                        SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256
                    ),
                    "family_id": family,
                    "candidate_id": candidate_id,
                    "ladder_index": index,
                    "candidate_artifact_sha256": _sha(
                        candidate_artifacts[candidate_id],
                        "confidence candidate artifact",
                    ),
                    "exact_objective": objective,
                    "objective_kind": "exact_negative_log_likelihood",
                    "exact_execution": True,
                },
            )
        family_receipts[family] = _finish(
            "fisher_graph.complete_h4_soft_polarity_confidence_family_oof.v20j",
            _FAMILY_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256
                ),
                "family_id": family,
                "candidate_order": SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
                "exact_objective_by_candidate": objectives[family],
                "exact_objective_sha256_by_candidate": objective_hashes[family],
                "candidate_ladder_artifact_sha256": ladder["artifact_sha256"],
                "all_candidates_frozen_before_family_score": True,
                "exact_execution": True,
                "outer_held_family_used": False,
                "raw_objectives_or_model_tensors_serialized": False,
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
    """Select one frozen confidence pair on seven family-equal OOF scores."""

    validate_soft_polarity_confidence_ladder_receipt(ladder_receipt)
    ladder = _mapping(ladder_receipt, "confidence ladder receipt")
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
        "confidence candidate artifacts",
    )
    for index, candidate_id in enumerate(
        SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS
    ):
        family_equal[candidate_id] = math.fsum(
            objectives[family][candidate_id] for family in inner
        ) / len(inner)
        aggregate_hashes[candidate_id] = _hash(
            _AGGREGATE_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256
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
                "exact_objective_by_family": {
                    family: objectives[family][candidate_id] for family in inner
                },
                "family_equal_exact_nll": family_equal[candidate_id],
            },
        )
    candidate_by_id = {
        _identifier(candidate["candidate_id"], "confidence candidate id"): candidate
        for candidate in _sequence(
            ladder["candidate_receipts"], "confidence candidate receipts"
        )
    }
    ranking = tuple(
        sorted(
            SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
            key=lambda candidate_id: (
                family_equal[candidate_id],
                math.fsum(
                    (
                        float(candidate_by_id[candidate_id]["a"]),
                        float(candidate_by_id[candidate_id]["b"]),
                    )
                ),
                float(candidate_by_id[candidate_id]["b"]),
                int(candidate_by_id[candidate_id]["ladder_index"]),
                str(candidate_by_id[candidate_id]["artifact_sha256"]),
            ),
        )
    )
    selected_id = ranking[0]
    selected = candidate_by_id[selected_id]
    result = _finish(
        "fisher_graph.complete_h4_soft_polarity_confidence_selection.v20j",
        _SELECTION_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_CONFIDENCE_FIT_PROTOCOL_SHA256,
            "ladder_artifact_sha256": ladder["artifact_sha256"],
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "inner_oof_family_order": inner,
            "candidate_order": SOFT_POLARITY_CONFIDENCE_CANDIDATE_IDS,
            "candidate_artifact_sha256s": dict(candidate_artifacts),
            "exact_objective_by_family_and_candidate": objectives,
            "exact_objective_sha256_by_family_and_candidate": objective_hashes,
            "family_oof_receipts": family_receipts,
            "family_oof_receipt_sha256s": {
                family: family_receipts[family]["artifact_sha256"]
                for family in inner
            },
            "family_equal_exact_nll_by_candidate": family_equal,
            "aggregate_artifact_sha256_by_candidate": aggregate_hashes,
            "candidate_ranking": ranking,
            "selection_key": (
                "family_equal_exact_nll_then_a_plus_b_then_b_then_ladder_index_"
                "then_candidate_artifact_sha256"
            ),
            "selected_candidate_id": selected_id,
            "selected_candidate_artifact_sha256": selected["artifact_sha256"],
            "selected_ladder_index": selected["ladder_index"],
            "selected_a": selected["a"],
            "selected_b": selected["b"],
            "selected_family_equal_exact_nll": family_equal[selected_id],
            "selected_aggregate_artifact_sha256": aggregate_hashes[selected_id],
            "candidate_ladder_frozen_before_any_inner_oof_objective": True,
            "selection_frozen_before_outer_held_score": True,
            "outer_held_family_used_for_fit_or_selection": False,
            "all_objectives_exact": True,
            "raw_objectives_or_model_tensors_serialized": False,
            "serving_authorized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )
    return result


def build_soft_polarity_confidence_inner_oof_selection_receipt(
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Select one frozen confidence pair on seven family-equal OOF scores."""

    result = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    validate_soft_polarity_confidence_inner_oof_selection_receipt(
        result,
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    return result


def validate_soft_polarity_confidence_inner_oof_selection_receipt(
    value: Mapping[str, object],
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    """Replay the full V20j selection from authoritative exact objectives."""

    receipt = _mapping(value, "confidence selection receipt")
    expected = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    if set(receipt) != set(expected):
        raise ValueError("confidence selection receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "confidence selection artifact")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("confidence selection receipt content drifted")
build_soft_polarity_confidence_fit_receipt = (
    build_soft_polarity_confidence_inner_oof_selection_receipt
)
validate_soft_polarity_confidence_fit_receipt = (
    validate_soft_polarity_confidence_inner_oof_selection_receipt
)
