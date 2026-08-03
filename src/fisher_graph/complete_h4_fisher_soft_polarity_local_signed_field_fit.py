"""Pure V20p training-only local signed-field selection protocol.

V20p replaces the single global V20o scalar with a bounded scalar field

``s(c) = clamp(b + a * psi(c), -1, 1)``

where ``psi`` is one of ``c1``, ``c2``, ``c1*c2``, or the canonical
``source_z`` coordinate.  The field library is frozen before any objective is
accepted: four features times three intercepts ``(-1/2, 0, 1/2)`` times two
slopes ``(-1, 1)`` give 24 adaptive fields, followed by the three exact
nonadaptive anchors ``s=-1``, ``s=0``, and ``s=1``.  All anchors use
``source_z`` as their canonical (inactive) feature.

Selection consumes exactly seven inner-family rows, each containing all 27
token-mean, exact-float64, full-vocabulary ``KL(teacher || candidate)``
objectives.  Candidate objectives are averaged equally across families.  The
deterministic ranking key is, in order: lower aggregate objective; nonadaptive
anchor before adaptive field; smaller ``abs(a)``; smaller ``abs(b)``; fixed
candidate index; candidate artifact SHA-256.  The latter terms never override
a genuinely smaller objective.

This module has no model-execution surface and cannot consume prompt text,
raw logits, H4 tensors, outer-held scores, or Calibration B.  Its receipts
grant no held-evaluation, compression, speed, or serving authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re


__all__ = [
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY",
    "build_soft_polarity_local_signed_field_fit_receipt",
    "build_soft_polarity_local_signed_field_inner_oof_selection_receipt",
    "build_soft_polarity_local_signed_field_ladder_receipt",
    "build_soft_polarity_local_signed_field_library_receipt",
    "soft_polarity_local_signed_field_scalar",
    "validate_soft_polarity_local_signed_field_fit_receipt",
    "validate_soft_polarity_local_signed_field_inner_oof_selection_receipt",
    "validate_soft_polarity_local_signed_field_ladder_receipt",
    "validate_soft_polarity_local_signed_field_library_receipt",
]


SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS = (
    "c1",
    "c2",
    "c1_times_c2",
    "source_z",
)
SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES = (-0.5, 0.0, 0.5)
SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES = (-1.0, 1.0)
SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES = (-1.0, 0.0, 1.0)

_ADAPTIVE_LIBRARY = tuple(
    (feature_id, b, a)
    for feature_id in SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS
    for b in SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES
    for a in SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES
)
_ANCHOR_LIBRARY = tuple(
    ("source_z", b, 0.0)
    for b in SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES
)
SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY = (
    _ADAPTIVE_LIBRARY + _ANCHOR_LIBRARY
)
SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS = tuple(
    f"local_signed_field_{index:02d}"
    for index in range(len(SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY))
)

_DEVELOPMENT_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_ADAPTIVE_CANDIDATE_COUNT = 24
_CANDIDATE_COUNT = 27
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORMULA = "s(c)=clamp(b+a*psi(c),-1,1)"
_FEATURE_FORMULAS = {
    "c1": "psi(c)=c1",
    "c2": "psi(c)=c2",
    "c1_times_c2": "psi(c)=c1*c2",
    "source_z": "psi(c)=source_z",
}
_OBJECTIVE_KIND = (
    "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
)
_AGGREGATE_OBJECTIVE_KIND = (
    "family_equal_token_mean_exact_float64_full_vocabulary_"
    "kl_teacher_to_candidate"
)
_SELECTION_KEY = (
    "family_equal_exact_kl_then_nonadaptive_before_adaptive_then_smaller_"
    "abs_a_then_smaller_abs_b_then_fixed_candidate_index_then_candidate_"
    "artifact_sha256"
)

_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"protocol:v20p\0"
)
_CANDIDATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"candidate:v20p\0"
)
_LADDER_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"ladder:v20p\0"
)
_OBJECTIVE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"objective:v20p\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"family:v20p\0"
)
_AGGREGATE_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"aggregate:v20p\0"
)
_RANKING_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"ranking:v20p\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-local-signed-field-fit:"
    b"selection:v20p\0"
)

_DATA_BOUNDARY = {
    "role": "v20p_training_only_local_signed_field_selection",
    "development_family_count": _DEVELOPMENT_FAMILY_COUNT,
    "inner_oof_family_count": _INNER_FAMILY_COUNT,
    "candidate_library_frozen_before_any_objective": True,
    "all_candidates_scored_on_every_inner_family": True,
    "outer_held_family_absent_from_fit_and_selection": True,
    "outer_held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "raw_logits_consumed": False,
    "raw_h4_consumed": False,
    "raw_model_tensors_serialized": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_shadow_consumed": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
    "serving_authorized": False,
}
_PROTOCOL = {
    "protocol": "v20p_training_only_local_signed_field_selection",
    "scientific_status": "development_only_after_v20o",
    "formula": _FORMULA,
    "feature_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS,
    "feature_formulas": _FEATURE_FORMULAS,
    "adaptive_b_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES,
    "adaptive_a_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES,
    "anchor_b_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES,
    "anchor_feature": "source_z",
    "anchor_a": 0.0,
    "library_order_feature_b_a": SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY,
    "candidate_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
    "adaptive_candidate_count": _ADAPTIVE_CANDIDATE_COUNT,
    "exact_anchor_count": len(_ANCHOR_LIBRARY),
    "candidate_count": _CANDIDATE_COUNT,
    "library_freeze": "before_any_inner_family_exact_kl_objective",
    "objective": _AGGREGATE_OBJECTIVE_KIND,
    "selection_key": _SELECTION_KEY,
    "selection_key_order": (
        "family_equal_exact_kl",
        "adaptive_complexity_zero_for_nonadaptive_one_for_adaptive",
        "abs_a",
        "abs_b",
        "fixed_candidate_index",
        "candidate_artifact_sha256",
    ),
    "outer_boundary": (
        "one_of_eight_development_families_absent_from_field_selection"
    ),
    "authorization": (
        "selection_receipt_only_no_outer_score_compression_speed_or_"
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


SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256 = _hash(
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


def soft_polarity_local_signed_field_scalar(
    *,
    feature_id: str,
    c1: float,
    c2: float,
    source_z: float,
    b: float,
    a: float,
) -> float:
    """Evaluate ``clamp(b + a*psi(c), -1, 1)`` on finite scalars."""

    selected_feature = _identifier(feature_id, "local signed-field feature")
    if selected_feature not in SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS:
        raise ValueError("unknown local signed-field feature")
    selected_c1 = _number(c1, "local signed-field c1")
    selected_c2 = _number(c2, "local signed-field c2")
    selected_z = _number(source_z, "local signed-field source_z")
    selected_b = _number(b, "local signed-field b")
    selected_a = _number(a, "local signed-field a")

    if selected_feature == "c1":
        psi = selected_c1
    elif selected_feature == "c2":
        psi = selected_c2
    elif selected_feature == "c1_times_c2":
        psi = _finite_operation(
            selected_c1 * selected_c2, "local signed-field c1*c2"
        )
    else:
        psi = selected_z
    try:
        unclamped = math.fsum((selected_b, selected_a * psi))
    except OverflowError as error:
        raise ValueError("local signed-field affine value must be finite") from error
    unclamped = _finite_operation(
        unclamped, "local signed-field affine value"
    )
    result = min(1.0, max(-1.0, unclamped))
    return 0.0 if result == 0.0 else result


def _candidate_receipt(index: int) -> dict[str, object]:
    candidate_id = SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS[index]
    feature_id, b, a = SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY[index]
    adaptive = a != 0.0
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
        "candidate.v20p",
        _CANDIDATE_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
            "candidate_id": candidate_id,
            "candidate_index": index,
            "feature_id": feature_id,
            "feature_formula": _FEATURE_FORMULAS[feature_id],
            "b": b,
            "b_hex": _float_hex(b),
            "a": a,
            "a_hex": _float_hex(a),
            "formula": _FORMULA,
            "adaptive": adaptive,
            "adaptive_complexity": 1 if adaptive else 0,
            "exact_nonadaptive_anchor": not adaptive,
            "anchor_signed_scalar": b if not adaptive else None,
            "zero_coordinate_signed_scalar": (
                soft_polarity_local_signed_field_scalar(
                    feature_id=feature_id,
                    c1=0.0,
                    c2=0.0,
                    source_z=0.0,
                    b=b,
                    a=a,
                )
            ),
            "candidate_frozen_before_any_inner_family_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
            "prompt_text_consumed": False,
            "raw_logits_or_h4_consumed": False,
        },
    )


def _expected_ladder() -> dict[str, object]:
    candidates = tuple(
        _candidate_receipt(index) for index in range(_CANDIDATE_COUNT)
    )
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
        "ladder.v20p",
        _LADDER_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
            "formula": _FORMULA,
            "feature_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS,
            "adaptive_b_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_B_VALUES,
            "adaptive_a_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ADAPTIVE_A_VALUES,
            "anchor_b_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_ANCHOR_B_VALUES,
            "library_order_feature_b_a": SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY,
            "candidate_count": len(candidates),
            "adaptive_candidate_count": sum(
                bool(item["adaptive"]) for item in candidates
            ),
            "exact_nonadaptive_anchor_count": sum(
                bool(item["exact_nonadaptive_anchor"]) for item in candidates
            ),
            "candidate_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
            "candidate_artifact_sha256s": {
                candidate_id: candidate["artifact_sha256"]
                for candidate_id, candidate in zip(
                    SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
                    candidates,
                    strict=True,
                )
            },
            "candidate_receipts": candidates,
            "library_frozen_before_any_inner_family_objective": True,
            "outer_held_objectives_consumed_before_freeze": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_local_signed_field_ladder_receipt() -> dict[str, object]:
    """Return the authenticated, objective-free 27-candidate library."""

    result = _expected_ladder()
    validate_soft_polarity_local_signed_field_ladder_receipt(result)
    return result


def validate_soft_polarity_local_signed_field_ladder_receipt(
    value: Mapping[str, object],
) -> None:
    """Authenticate every candidate definition, order, and library hash."""

    receipt = _mapping(value, "local signed-field ladder receipt")
    expected = _expected_ladder()
    if set(receipt) != set(expected):
        raise ValueError("local signed-field ladder receipt key set drifted")
    raw_candidates = _sequence(
        receipt.get("candidate_receipts"), "local signed-field candidates"
    )
    if len(raw_candidates) != _CANDIDATE_COUNT:
        raise ValueError("local signed-field candidate geometry drifted")
    for index, raw in enumerate(raw_candidates):
        candidate = _mapping(raw, "local signed-field candidate receipt")
        expected_candidate = _candidate_receipt(index)
        if set(candidate) != set(expected_candidate):
            raise ValueError("local signed-field candidate key set drifted")
        _sha(candidate.get("artifact_sha256"), "local signed-field candidate")
        if _canonical(candidate) != _canonical(expected_candidate):
            raise ValueError("local signed-field candidate content drifted")
    _sha(receipt.get("artifact_sha256"), "local signed-field ladder")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("local signed-field ladder receipt content drifted")


build_soft_polarity_local_signed_field_library_receipt = (
    build_soft_polarity_local_signed_field_ladder_receipt
)
validate_soft_polarity_local_signed_field_library_receipt = (
    validate_soft_polarity_local_signed_field_ladder_receipt
)


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
        "local signed-field exact KL objectives",
    )
    if set(raw_families) != set(inner_families):
        raise ValueError("local signed-field objective family geometry drifted")
    candidate_artifacts = _mapping(
        ladder.get("candidate_artifact_sha256s"),
        "local signed-field candidate artifacts",
    )
    objectives: dict[str, dict[str, float]] = {}
    objective_hashes: dict[str, dict[str, str]] = {}
    family_receipts: dict[str, dict[str, object]] = {}
    for family in inner_families:
        raw_candidates = _mapping(
            raw_families[family],
            f"{family} local signed-field exact KL objectives",
        )
        if set(raw_candidates) != set(
            SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
        ):
            raise ValueError(
                "local signed-field objective candidate geometry drifted"
            )
        objectives[family] = {}
        objective_hashes[family] = {}
        for index, candidate_id in enumerate(
            SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
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
                        SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256
                    ),
                    "family_id": family,
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "candidate_artifact_sha256": _sha(
                        candidate_artifacts[candidate_id],
                        "local signed-field candidate artifact",
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
            "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
            "family_oof.v20p",
            _FAMILY_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256
                ),
                "family_id": family,
                "candidate_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
                "objective_kind": _OBJECTIVE_KIND,
                "exact_kl_objective_by_candidate": objectives[family],
                "exact_kl_objective_hex_by_candidate": {
                    candidate_id: _float_hex(objective)
                    for candidate_id, objective in objectives[family].items()
                },
                "exact_kl_objective_sha256_by_candidate": objective_hashes[family],
                "candidate_ladder_artifact_sha256": ladder["artifact_sha256"],
                "all_candidates_frozen_before_family_score": True,
                "outer_held_family_used": False,
                "exact_float64_execution": True,
                "full_vocabulary_evaluated": True,
                "exact_scalar_objectives_serialized": True,
                "prompt_text_or_raw_logits_or_h4_serialized": False,
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
    validate_soft_polarity_local_signed_field_ladder_receipt(ladder_receipt)
    ladder = _mapping(ladder_receipt, "local signed-field ladder receipt")
    families, outer, inner = _families(
        all_development_family_ids, outer_held_family_id
    )
    objectives, objective_hashes, family_receipts = _objective_rows(
        exact_objectives_by_family_and_candidate,
        inner_families=inner,
        ladder=ladder,
    )
    candidate_artifacts = _mapping(
        ladder["candidate_artifact_sha256s"],
        "local signed-field candidate artifacts",
    )
    candidate_by_id = {
        _identifier(candidate["candidate_id"], "local signed-field candidate id"):
        candidate
        for candidate in (
            _mapping(item, "local signed-field candidate")
            for item in _sequence(
                ladder["candidate_receipts"],
                "local signed-field candidate receipts",
            )
        )
    }
    family_receipt_hashes = {
        family: family_receipts[family]["artifact_sha256"] for family in inner
    }
    family_equal: dict[str, float] = {}
    aggregate_receipts: dict[str, dict[str, object]] = {}
    for index, candidate_id in enumerate(
        SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
    ):
        objective = _family_equal(
            tuple(objectives[family][candidate_id] for family in inner),
            f"{candidate_id} family-equal exact KL",
        )
        family_equal[candidate_id] = objective
        aggregate_receipts[candidate_id] = _finish(
            "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
            "aggregate.v20p",
            _AGGREGATE_DOMAIN,
            {
                "protocol_sha256": (
                    SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256
                ),
                "outer_held_family_id": outer,
                "inner_oof_family_order": inner,
                "candidate_id": candidate_id,
                "candidate_index": index,
                "candidate_artifact_sha256": candidate_artifacts[candidate_id],
                "family_oof_receipt_sha256s": family_receipt_hashes,
                "exact_kl_objective_by_family": {
                    family: objectives[family][candidate_id] for family in inner
                },
                "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
                "family_equal_exact_kl": objective,
                "family_equal_exact_kl_hex": _float_hex(objective),
            },
        )

    def ranking_key(candidate_id: str) -> tuple[object, ...]:
        candidate = candidate_by_id[candidate_id]
        return (
            family_equal[candidate_id],
            int(candidate["adaptive_complexity"]),
            abs(float(candidate["a"])),
            abs(float(candidate["b"])),
            int(candidate["candidate_index"]),
            str(candidate["artifact_sha256"]),
        )

    ranking = tuple(
        sorted(SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS, key=ranking_key)
    )
    ranking_receipt = _finish(
        "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
        "ranking.v20p",
        _RANKING_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
            "ladder_artifact_sha256": ladder["artifact_sha256"],
            "outer_held_family_id": outer,
            "aggregate_artifact_sha256_by_candidate": {
                candidate_id: aggregate_receipts[candidate_id]["artifact_sha256"]
                for candidate_id in SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
            },
            "selection_key": _SELECTION_KEY,
            "selection_key_order": _PROTOCOL["selection_key_order"],
            "candidate_ranking": ranking,
            "selected_candidate_id": ranking[0],
        },
    )
    selected_id = ranking[0]
    selected = candidate_by_id[selected_id]
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_local_signed_field_"
        "selection.v20p",
        _SELECTION_DOMAIN,
        {
            "protocol_sha256": SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
            "ladder_artifact_sha256": ladder["artifact_sha256"],
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "inner_oof_family_order": inner,
            "candidate_order": SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS,
            "candidate_artifact_sha256s": dict(candidate_artifacts),
            "objective_kind": _OBJECTIVE_KIND,
            "aggregate_objective_kind": _AGGREGATE_OBJECTIVE_KIND,
            "exact_kl_objective_by_family_and_candidate": objectives,
            "exact_kl_objective_sha256_by_family_and_candidate": objective_hashes,
            "family_oof_receipts": family_receipts,
            "family_oof_receipt_sha256s": family_receipt_hashes,
            "family_equal_exact_kl_by_candidate": family_equal,
            "family_equal_exact_kl_hex_by_candidate": {
                candidate_id: _float_hex(objective)
                for candidate_id, objective in family_equal.items()
            },
            "aggregate_receipts": aggregate_receipts,
            "aggregate_artifact_sha256_by_candidate": {
                candidate_id: aggregate_receipts[candidate_id]["artifact_sha256"]
                for candidate_id in SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
            },
            "ranking_receipt": ranking_receipt,
            "ranking_artifact_sha256": ranking_receipt["artifact_sha256"],
            "candidate_ranking": ranking,
            "selection_key": _SELECTION_KEY,
            "selected_candidate_id": selected_id,
            "selected_candidate_artifact_sha256": selected["artifact_sha256"],
            "selected_candidate_index": selected["candidate_index"],
            "selected_feature_id": selected["feature_id"],
            "selected_b": selected["b"],
            "selected_a": selected["a"],
            "selected_adaptive": selected["adaptive"],
            "selected_family_equal_exact_kl": family_equal[selected_id],
            "selected_aggregate_artifact_sha256": aggregate_receipts[selected_id][
                "artifact_sha256"
            ],
            "candidate_library_frozen_before_any_inner_family_objective": True,
            "selection_frozen_before_outer_held_score": True,
            "outer_held_family_used_for_fit_or_selection": False,
            "all_objectives_token_mean_exact_float64_full_vocabulary_kl": True,
            "exact_scalar_objectives_serialized": True,
            "prompt_text_or_raw_logits_or_h4_serialized": False,
            "compression_claim_authorized": False,
            "speed_claim_authorized": False,
            "serving_authorized": False,
            "data_boundary": dict(_DATA_BOUNDARY),
        },
    )


def build_soft_polarity_local_signed_field_inner_oof_selection_receipt(
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Select one frozen V20p field from seven exact family rows."""

    result = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    validate_soft_polarity_local_signed_field_inner_oof_selection_receipt(
        result,
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    return result


def validate_soft_polarity_local_signed_field_inner_oof_selection_receipt(
    value: Mapping[str, object],
    *,
    ladder_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    exact_objectives_by_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    """Replay and authenticate family rows, aggregates, ranking, and winner."""

    receipt = _mapping(value, "local signed-field selection receipt")
    expected = _build_selection_receipt(
        ladder_receipt=ladder_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        exact_objectives_by_family_and_candidate=(
            exact_objectives_by_family_and_candidate
        ),
    )
    if set(receipt) != set(expected):
        raise ValueError("local signed-field selection receipt key set drifted")
    _sha(receipt.get("artifact_sha256"), "local signed-field selection")
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("local signed-field selection receipt content drifted")


build_soft_polarity_local_signed_field_fit_receipt = (
    build_soft_polarity_local_signed_field_inner_oof_selection_receipt
)
validate_soft_polarity_local_signed_field_fit_receipt = (
    validate_soft_polarity_local_signed_field_inner_oof_selection_receipt
)
