"""Pure nested-family protocol for finite Fisher microstep validation.

V20b chooses a V20a path and scale without observing the outer held family.
Eight outer folds contain seven ordered inner roles.  The 56 roles share 28
physical six-family fits because excluding ``(a, b)`` and ``(b, a)`` yields
the same fit set.  This module owns only scalar/hash receipts, deterministic
selection, qualification, and work accounting; it never executes a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_l3_l4_complete_h4_finite_microstep_preflight import (
    MICROSTEP_PATHS,
    POSITIVE_ALPHAS,
    numerical_improvement_floor,
)

__all__ = [
    "NESTED_MICROSTEP_PATHS",
    "NESTED_MICROSTEP_POSITIVE_ALPHAS",
    "NESTED_MICROSTEP_CANDIDATE_KEYS",
    "NESTED_MICROSTEP_PROTOCOL_SHA256",
    "nested_microstep_candidate_key",
    "nested_microstep_fit_pair_key",
    "build_nested_microstep_panel_receipt",
    "validate_nested_microstep_panel_receipt",
    "build_nested_microstep_shared_fit_receipt",
    "build_nested_microstep_outer_fit_receipt",
    "validate_nested_microstep_fit_receipt",
    "build_nested_microstep_baseline_score",
    "validate_nested_microstep_baseline_score",
    "build_nested_microstep_candidate_score",
    "validate_nested_microstep_candidate_score",
    "build_nested_microstep_inner_role",
    "validate_nested_microstep_inner_role",
    "select_nested_microstep_inner_candidate",
    "build_nested_microstep_selection_receipt",
    "validate_nested_microstep_selection_receipt",
    "build_nested_microstep_outer_score",
    "build_nested_microstep_validation_receipt",
    "validate_nested_microstep_validation_receipt",
    "nested_microstep_work_accounting",
]

NESTED_MICROSTEP_PATHS = tuple(MICROSTEP_PATHS)
NESTED_MICROSTEP_POSITIVE_ALPHAS = tuple(POSITIVE_ALPHAS)
_PATH_ORDER = {value: index for index, value in enumerate(NESTED_MICROSTEP_PATHS)}
_ALPHA_ORDER = {
    value: index for index, value in enumerate(NESTED_MICROSTEP_POSITIVE_ALPHAS)
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_INNER_COUNT = 7
_MATERIALITY = 0.01
_WORST_REGRESSION = 0.02
_INNER_WINS = 6
_OUTER_WINS = 6

_PANEL_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-panel:v20b\0"
_FIT_KEY_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-fit-key:v20b\0"
_FIT_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-fit:v20b\0"
_BASELINE_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-baseline:v20b\0"
_CANDIDATE_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-candidate:v20b\0"
_ROLE_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-role:v20b\0"
_SELECTION_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-selection:v20b\0"
_SELECTION_SET_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-selection-set:v20b\0"
_OUTER_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-outer:v20b\0"
_REPORT_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-report:v20b\0"
_PROTOCOL_DOMAIN = b"fisher-graph:complete-h4-fisher-nested-protocol:v20b\0"


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


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _integer(value: object, *, label: str, positive: bool = True) -> int:
    if type(value) is not int or (value <= 0 if positive else value < 0):
        raise TypeError(f"{label} must be {'positive' if positive else 'nonnegative'} integer")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields differ")


def _finish(schema: str, domain: bytes, payload: Mapping[str, object]) -> dict[str, object]:
    result = {"schema": schema, **dict(payload)}
    result["artifact_sha256"] = _hash(domain, result)
    return result


def _same(left: object, right: object, *, label: str) -> None:
    if _canonical(left) != _canonical(right):
        raise ValueError(f"{label} receipt drifted")


def nested_microstep_candidate_key(path: str, alpha: float) -> str:
    selected_path = _identifier(path, label="microstep path")
    if selected_path not in _PATH_ORDER:
        raise ValueError("microstep path differs")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("microstep alpha must be numeric")
    signed_alpha = float(alpha)
    if not math.isfinite(signed_alpha):
        raise ValueError("microstep alpha must be finite")
    if signed_alpha == 0.0:
        raise ValueError("microstep alpha must be nonzero")
    selected_alpha = abs(signed_alpha)
    if selected_alpha not in NESTED_MICROSTEP_POSITIVE_ALPHAS:
        raise ValueError("microstep alpha is outside the fixed ladder")
    sign = "+" if signed_alpha > 0.0 else "-"
    return f"{selected_path}:{sign}{selected_alpha.hex()}"


NESTED_MICROSTEP_CANDIDATE_KEYS = tuple(
    nested_microstep_candidate_key(path, alpha)
    for path in NESTED_MICROSTEP_PATHS
    for alpha in NESTED_MICROSTEP_POSITIVE_ALPHAS
)

_FIXED_PROTOCOL = {
    "protocol": "nested_family_disjoint_finite_fisher_microstep_v20b",
    "outer_folds": 8,
    "inner_roles_per_outer": 7,
    "unordered_shared_fit_pairs": 28,
    "candidate_paths": NESTED_MICROSTEP_PATHS,
    "positive_alphas": NESTED_MICROSTEP_POSITIVE_ALPHAS,
    "rollback": "checkpoint_zero",
    "tie_order": "baseline_then_smaller_alpha_then_direction_pedal_joint",
    "aggregation": "family_equal",
    "inner_materiality": _MATERIALITY,
    "inner_wins": _INNER_WINS,
    "outer_materiality": _MATERIALITY,
    "outer_wins": _OUTER_WINS,
    "worst_regression_maximum": _WORST_REGRESSION,
    "numerical_floor": "V20a_fixed_float64_floor",
}
NESTED_MICROSTEP_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _FIXED_PROTOCOL)


def nested_microstep_fit_pair_key(left: str, right: str) -> str:
    pair = tuple(sorted((_identifier(left, label="fit family"), _identifier(right, label="fit family"))))
    if pair[0] == pair[1]:
        raise ValueError("shared fit pair requires distinct families")
    return _hash(_FIT_KEY_DOMAIN, {"kind": "shared_pair", "excluded_family_ids": pair})


def _outer_fit_key(family: str) -> str:
    return _hash(_FIT_KEY_DOMAIN, {"kind": "outer_full", "excluded_family_ids": (_identifier(family, label="outer family"),)})


def build_nested_microstep_panel_receipt(
    family_prompt_sha256s: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    source = _mapping(family_prompt_sha256s, label="family prompt hashes")
    if len(source) != _FAMILY_COUNT:
        raise ValueError("nested panel requires exactly eight families")
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_family, raw_hashes in source.items():
        family = _identifier(raw_family, label="family_id")
        hashes = tuple(sorted(_sha(value, label=f"{family} prompt") for value in _sequence(raw_hashes, label=f"{family} prompts")))
        if len(hashes) != _PROMPTS_PER_FAMILY or len(set(hashes)) != len(hashes):
            raise ValueError("each nested family requires two unique prompts")
        normalized[family] = hashes
    normalized = dict(sorted(normalized.items()))
    all_hashes = tuple(value for values in normalized.values() for value in values)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("nested panel prompts must be globally disjoint")
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_panel.v1",
        _PANEL_DOMAIN,
        {
            "family_ids": tuple(normalized),
            "family_prompt_sha256s": normalized,
            "prompt_sha256s": tuple(sorted(all_hashes)),
            "family_count": _FAMILY_COUNT,
            "prompt_count": _FAMILY_COUNT * _PROMPTS_PER_FAMILY,
            "prompts_per_family": _PROMPTS_PER_FAMILY,
            "family_and_prompt_disjoint": True,
        },
    )


def validate_nested_microstep_panel_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="panel receipt")
    _exact(selected, {"schema", "family_ids", "family_prompt_sha256s", "prompt_sha256s", "family_count", "prompt_count", "prompts_per_family", "family_and_prompt_disjoint", "artifact_sha256"}, label="panel")
    prompts = _mapping(selected["family_prompt_sha256s"], label="panel family prompts")
    rebuilt = build_nested_microstep_panel_receipt({str(key): tuple(_sequence(item, label="panel prompts")) for key, item in prompts.items()})
    _same(selected, rebuilt, label="panel")
    return rebuilt


def _build_fit_receipt(
    *, panel_receipt: Mapping[str, object], excluded_family_ids: Sequence[str], kind: str,
    base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str,
    fit_protocol_sha256: str, fit_evidence_sha256: str, rank: int,
    conditional_rank: int, finite: bool, pointwise_trust_passed: bool,
) -> dict[str, object]:
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    excluded = tuple(sorted(_identifier(item, label="excluded family") for item in _sequence(excluded_family_ids, label="excluded families")))
    expected_excluded = 2 if kind == "shared_pair" else 1
    if len(excluded) != expected_excluded or len(set(excluded)) != len(excluded) or not set(excluded) <= set(panel["family_ids"]):
        raise ValueError("nested fit exclusion geometry differs")
    training = tuple(family for family in panel["family_ids"] if family not in excluded)
    family_prompts = panel["family_prompt_sha256s"]
    assert isinstance(family_prompts, Mapping)
    training_prompts = tuple(hash_value for family in training for hash_value in family_prompts[family])
    fit_key = nested_microstep_fit_pair_key(*excluded) if kind == "shared_pair" else _outer_fit_key(excluded[0])
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_fit.v1", _FIT_DOMAIN,
        {"kind": kind, "panel_artifact_sha256": panel["artifact_sha256"], "fit_key": fit_key,
         "excluded_family_ids": excluded, "training_family_ids": training,
         "training_prompt_sha256s": training_prompts, "fit_family_count": len(training),
         "fit_prompt_count": len(training_prompts),
         "base_provider_artifact_sha256": _sha(base_provider_artifact_sha256, label="base provider"),
         "proposal_provider_artifact_sha256": _sha(proposal_provider_artifact_sha256, label="proposal provider"),
         "fit_protocol_sha256": _sha(fit_protocol_sha256, label="fit protocol"),
         "fit_evidence_sha256": _sha(fit_evidence_sha256, label="fit evidence"),
         "rank": _integer(rank, label="rank"), "conditional_rank": _integer(conditional_rank, label="conditional rank"),
         "finite": _boolean(finite, label="fit finite"),
         "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="fit trust"),
         "held_families_and_prompts_excluded": True, "raw_tensors_serialized": False},
    )


def build_nested_microstep_shared_fit_receipt(*, panel_receipt: Mapping[str, object], excluded_family_ids: Sequence[str], base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str, fit_protocol_sha256: str, fit_evidence_sha256: str, rank: int, conditional_rank: int, finite: bool, pointwise_trust_passed: bool) -> dict[str, object]:
    return _build_fit_receipt(panel_receipt=panel_receipt, excluded_family_ids=excluded_family_ids, kind="shared_pair", base_provider_artifact_sha256=base_provider_artifact_sha256, proposal_provider_artifact_sha256=proposal_provider_artifact_sha256, fit_protocol_sha256=fit_protocol_sha256, fit_evidence_sha256=fit_evidence_sha256, rank=rank, conditional_rank=conditional_rank, finite=finite, pointwise_trust_passed=pointwise_trust_passed)


def build_nested_microstep_outer_fit_receipt(*, panel_receipt: Mapping[str, object], outer_held_family_id: str, base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str, fit_protocol_sha256: str, fit_evidence_sha256: str, rank: int, conditional_rank: int, finite: bool, pointwise_trust_passed: bool) -> dict[str, object]:
    return _build_fit_receipt(panel_receipt=panel_receipt, excluded_family_ids=(outer_held_family_id,), kind="outer_full", base_provider_artifact_sha256=base_provider_artifact_sha256, proposal_provider_artifact_sha256=proposal_provider_artifact_sha256, fit_protocol_sha256=fit_protocol_sha256, fit_evidence_sha256=fit_evidence_sha256, rank=rank, conditional_rank=conditional_rank, finite=finite, pointwise_trust_passed=pointwise_trust_passed)


def validate_nested_microstep_fit_receipt(value: Mapping[str, object], *, panel_receipt: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="fit receipt")
    keys = {"schema", "kind", "panel_artifact_sha256", "fit_key", "excluded_family_ids", "training_family_ids", "training_prompt_sha256s", "fit_family_count", "fit_prompt_count", "base_provider_artifact_sha256", "proposal_provider_artifact_sha256", "fit_protocol_sha256", "fit_evidence_sha256", "rank", "conditional_rank", "finite", "pointwise_trust_passed", "held_families_and_prompts_excluded", "raw_tensors_serialized", "artifact_sha256"}
    _exact(selected, keys, label="fit")
    kind = selected["kind"]
    if kind not in {"shared_pair", "outer_full"}:
        raise ValueError("fit kind differs")
    kwargs = dict(panel_receipt=panel_receipt, base_provider_artifact_sha256=selected["base_provider_artifact_sha256"], proposal_provider_artifact_sha256=selected["proposal_provider_artifact_sha256"], fit_protocol_sha256=selected["fit_protocol_sha256"], fit_evidence_sha256=selected["fit_evidence_sha256"], rank=selected["rank"], conditional_rank=selected["conditional_rank"], finite=selected["finite"], pointwise_trust_passed=selected["pointwise_trust_passed"])
    if kind == "shared_pair":
        rebuilt = build_nested_microstep_shared_fit_receipt(excluded_family_ids=tuple(_sequence(selected["excluded_family_ids"], label="excluded families")), **kwargs)
    else:
        excluded = _sequence(selected["excluded_family_ids"], label="excluded families")
        rebuilt = build_nested_microstep_outer_fit_receipt(outer_held_family_id=excluded[0], **kwargs)
    _same(selected, rebuilt, label="fit")
    return rebuilt


def build_nested_microstep_baseline_score(*, objective: float, fit_receipt_sha256: str, provider_artifact_sha256: str, execution_receipt_sha256: str, finite: bool, pointwise_trust_passed: bool, rank_is_16: bool) -> dict[str, object]:
    return _finish("fisher_graph.complete_h4_fisher_nested_baseline.v1", _BASELINE_DOMAIN,
        {"objective": _number(objective, label="baseline objective", positive=True),
         "fit_receipt_sha256": _sha(fit_receipt_sha256, label="fit receipt"),
         "provider_artifact_sha256": _sha(provider_artifact_sha256, label="baseline provider"),
         "execution_receipt_sha256": _sha(execution_receipt_sha256, label="baseline execution"),
         "finite": _boolean(finite, label="baseline finite"),
         "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="baseline trust"),
         "rank_is_16": _boolean(rank_is_16, label="baseline rank")})


def validate_nested_microstep_baseline_score(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="baseline score")
    _exact(selected, {"schema", "objective", "fit_receipt_sha256", "provider_artifact_sha256", "execution_receipt_sha256", "finite", "pointwise_trust_passed", "rank_is_16", "artifact_sha256"}, label="baseline")
    rebuilt = build_nested_microstep_baseline_score(objective=selected["objective"], fit_receipt_sha256=selected["fit_receipt_sha256"], provider_artifact_sha256=selected["provider_artifact_sha256"], execution_receipt_sha256=selected["execution_receipt_sha256"], finite=selected["finite"], pointwise_trust_passed=selected["pointwise_trust_passed"], rank_is_16=selected["rank_is_16"])
    _same(selected, rebuilt, label="baseline")
    return rebuilt


def build_nested_microstep_candidate_score(*, path: str, alpha: float, objective: float, fit_receipt_sha256: str, provider_artifact_sha256: str, microstep_receipt_sha256: str, execution_change_receipt_sha256: str, execution_changed: bool, finite: bool, pointwise_trust_passed: bool, rank_is_16: bool) -> dict[str, object]:
    path_value = _identifier(path, label="candidate path")
    if path_value not in _PATH_ORDER:
        raise ValueError("candidate path differs")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) == 0.0 or abs(float(alpha)) not in NESTED_MICROSTEP_POSITIVE_ALPHAS:
        raise ValueError("candidate alpha is outside the signed fixed ladder")
    alpha_value = float(alpha)
    return _finish("fisher_graph.complete_h4_fisher_nested_candidate.v1", _CANDIDATE_DOMAIN,
        {"key": nested_microstep_candidate_key(path_value, alpha_value), "path": path_value, "alpha": alpha_value,
         "objective": _number(objective, label="candidate objective"),
         "fit_receipt_sha256": _sha(fit_receipt_sha256, label="candidate fit receipt"),
         "provider_artifact_sha256": _sha(provider_artifact_sha256, label="candidate provider"),
         "microstep_receipt_sha256": _sha(microstep_receipt_sha256, label="microstep receipt"),
         "execution_change_receipt_sha256": _sha(execution_change_receipt_sha256, label="execution receipt"),
         "execution_changed": _boolean(execution_changed, label="candidate execution change"),
         "finite": _boolean(finite, label="candidate finite"),
         "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="candidate trust"),
         "rank_is_16": _boolean(rank_is_16, label="candidate rank")})


def validate_nested_microstep_candidate_score(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="candidate score")
    _exact(selected, {"schema", "key", "path", "alpha", "objective", "fit_receipt_sha256", "provider_artifact_sha256", "microstep_receipt_sha256", "execution_change_receipt_sha256", "execution_changed", "finite", "pointwise_trust_passed", "rank_is_16", "artifact_sha256"}, label="candidate")
    rebuilt = build_nested_microstep_candidate_score(path=selected["path"], alpha=selected["alpha"], objective=selected["objective"], fit_receipt_sha256=selected["fit_receipt_sha256"], provider_artifact_sha256=selected["provider_artifact_sha256"], microstep_receipt_sha256=selected["microstep_receipt_sha256"], execution_change_receipt_sha256=selected["execution_change_receipt_sha256"], execution_changed=selected["execution_changed"], finite=selected["finite"], pointwise_trust_passed=selected["pointwise_trust_passed"], rank_is_16=selected["rank_is_16"])
    _same(selected, rebuilt, label="candidate")
    return rebuilt


def _positive_grid(values: Sequence[Mapping[str, object]], *, fit_sha: str) -> tuple[dict[str, object], ...]:
    candidates = tuple(validate_nested_microstep_candidate_score(_mapping(row, label="positive candidate")) for row in _sequence(values, label="positive candidates"))
    if any(float(row["alpha"]) <= 0.0 for row in candidates):
        raise ValueError("positive grid contains a negative candidate")
    keys = tuple(str(row["key"]) for row in candidates)
    if len(candidates) != len(NESTED_MICROSTEP_CANDIDATE_KEYS) or len(set(keys)) != len(keys) or set(keys) != set(NESTED_MICROSTEP_CANDIDATE_KEYS):
        raise ValueError("positive candidate grid is partial or duplicated")
    if any(row["fit_receipt_sha256"] != fit_sha for row in candidates):
        raise ValueError("positive candidate fit binding differs")
    return tuple(sorted(candidates, key=lambda row: (_PATH_ORDER[str(row["path"])], _ALPHA_ORDER[float(row["alpha"])])))


def build_nested_microstep_inner_role(*, panel_receipt: Mapping[str, object], shared_fit_receipt: Mapping[str, object], outer_held_family_id: str, inner_held_family_id: str, baseline: Mapping[str, object], positive_candidates: Sequence[Mapping[str, object]], matched_negative: Mapping[str, object] | None) -> dict[str, object]:
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    fit = validate_nested_microstep_fit_receipt(shared_fit_receipt, panel_receipt=panel)
    outer = _identifier(outer_held_family_id, label="outer family")
    inner = _identifier(inner_held_family_id, label="inner family")
    if outer == inner or fit["kind"] != "shared_pair" or tuple(fit["excluded_family_ids"]) != tuple(sorted((outer, inner))):
        raise ValueError("inner role does not match its shared fit pair")
    base = validate_nested_microstep_baseline_score(baseline)
    if base["fit_receipt_sha256"] != fit["artifact_sha256"]:
        raise ValueError("baseline fit binding differs")
    positives = _positive_grid(positive_candidates, fit_sha=str(fit["artifact_sha256"]))
    mirror = None if matched_negative is None else validate_nested_microstep_candidate_score(matched_negative)
    if mirror is not None and (float(mirror["alpha"]) >= 0.0 or mirror["fit_receipt_sha256"] != fit["artifact_sha256"]):
        raise ValueError("inner mirror binding differs")
    family_prompts = panel["family_prompt_sha256s"]
    assert isinstance(family_prompts, Mapping)
    return _finish("fisher_graph.complete_h4_fisher_nested_inner_role.v1", _ROLE_DOMAIN,
        {"panel_artifact_sha256": panel["artifact_sha256"], "outer_held_family_id": outer,
         "inner_held_family_id": inner, "fit_pair_key": fit["fit_key"],
         "shared_fit_receipt_sha256": fit["artifact_sha256"],
         "inner_score_prompt_sha256s": tuple(family_prompts[inner]),
         "outer_prompt_capability_accessed": False, "baseline": base,
         "positive_candidates": positives, "matched_negative": mirror,
         "raw_tensors_or_logits_serialized": False})


def validate_nested_microstep_inner_role(value: Mapping[str, object], *, panel_receipt: Mapping[str, object], shared_fit_receipt: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="inner role")
    _exact(selected, {"schema", "panel_artifact_sha256", "outer_held_family_id", "inner_held_family_id", "fit_pair_key", "shared_fit_receipt_sha256", "inner_score_prompt_sha256s", "outer_prompt_capability_accessed", "baseline", "positive_candidates", "matched_negative", "raw_tensors_or_logits_serialized", "artifact_sha256"}, label="inner role")
    rebuilt = build_nested_microstep_inner_role(panel_receipt=panel_receipt, shared_fit_receipt=shared_fit_receipt, outer_held_family_id=selected["outer_held_family_id"], inner_held_family_id=selected["inner_held_family_id"], baseline=_mapping(selected["baseline"], label="baseline"), positive_candidates=tuple(_mapping(row, label="candidate") for row in _sequence(selected["positive_candidates"], label="candidates")), matched_negative=None if selected["matched_negative"] is None else _mapping(selected["matched_negative"], label="mirror"))
    _same(selected, rebuilt, label="inner role")
    return rebuilt


def _shared_fit_map(
    panel: Mapping[str, object], values: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    fits = tuple(
        validate_nested_microstep_fit_receipt(
            _mapping(value, label="shared fit"), panel_receipt=panel
        )
        for value in _sequence(values, label="shared fits")
    )
    if any(value["kind"] != "shared_pair" for value in fits):
        raise ValueError("selection contains a non-pair fit")
    keyed = {str(value["fit_key"]): value for value in fits}
    families = tuple(panel["family_ids"])
    expected = {
        nested_microstep_fit_pair_key(families[left], families[right])
        for left in range(len(families))
        for right in range(left + 1, len(families))
    }
    if len(fits) != 28 or len(keyed) != 28 or set(keyed) != expected:
        raise ValueError("nested selection requires exactly 28 unordered shared fits")
    return keyed


def _selection_hash_valid(value: Mapping[str, object]) -> None:
    artifact = _sha(value.get("artifact_sha256"), label="inner selection")
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    if _hash(_SELECTION_DOMAIN, payload) != artifact:
        raise ValueError("inner selection artifact hash drifted")


def select_nested_microstep_inner_candidate(
    *,
    panel_receipt: Mapping[str, object],
    shared_fit_receipts: Sequence[Mapping[str, object]],
    outer_held_family_id: str,
    inner_roles: Sequence[Mapping[str, object]],
    require_mirrors: bool = False,
) -> dict[str, object]:
    """Choose one path/alpha using seven held-inner family scores only."""

    if type(require_mirrors) is not bool:
        raise TypeError("require_mirrors must be boolean")
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    fits = _shared_fit_map(panel, shared_fit_receipts)
    outer = _identifier(outer_held_family_id, label="outer held family")
    if outer not in panel["family_ids"]:
        raise ValueError("outer held family is outside the panel")
    supplied = tuple(_mapping(value, label="inner role") for value in _sequence(inner_roles, label="inner roles"))
    roles: list[dict[str, object]] = []
    for value in supplied:
        key = _sha(value.get("fit_pair_key"), label="inner role fit key")
        fit = fits.get(key)
        if fit is None:
            raise ValueError("inner role references an unknown shared fit")
        roles.append(
            validate_nested_microstep_inner_role(
                value, panel_receipt=panel, shared_fit_receipt=fit
            )
        )
    roles.sort(key=lambda value: str(value["inner_held_family_id"]))
    inner_ids = tuple(str(value["inner_held_family_id"]) for value in roles)
    expected_inner = tuple(value for value in panel["family_ids"] if value != outer)
    if (
        len(roles) != _INNER_COUNT
        or len(set(inner_ids)) != len(inner_ids)
        or inner_ids != expected_inner
        or any(value["outer_held_family_id"] != outer for value in roles)
    ):
        raise ValueError("outer fold requires seven exact ordered inner roles")

    baselines = tuple(
        validate_nested_microstep_baseline_score(
            _mapping(value["baseline"], label="role baseline")
        )
        for value in roles
    )
    baseline_macro = math.fsum(float(value["objective"]) for value in baselines) / _INNER_COUNT
    candidates_by_role = tuple(
        {str(row["key"]): row for row in value["positive_candidates"]}
        for value in roles
    )
    macro_by_key = {
        key: math.fsum(float(rows[key]["objective"]) for rows in candidates_by_role)
        / _INNER_COUNT
        for key in NESTED_MICROSTEP_CANDIDATE_KEYS
    }
    # Exact objective ties retain rollback.  Candidate ties prefer smaller
    # alpha, then direction-only, pedal-only, joint.
    choices: list[tuple[float, int, float, int, str | None]] = [
        (baseline_macro, 0, 0.0, -1, None)
    ]
    for path in NESTED_MICROSTEP_PATHS:
        for alpha in NESTED_MICROSTEP_POSITIVE_ALPHAS:
            key = nested_microstep_candidate_key(path, alpha)
            choices.append((macro_by_key[key], 1, alpha, _PATH_ORDER[path], key))
    selected_key = min(choices) [4]
    selected_rows: tuple[Mapping[str, object], ...] = ()
    selected: dict[str, object] | None = None
    selected_macro = baseline_macro
    if selected_key is not None:
        selected_rows = tuple(rows[selected_key] for rows in candidates_by_role)
        first = selected_rows[0]
        selected = {
            "key": selected_key,
            "path": first["path"],
            "alpha": first["alpha"],
        }
        selected_macro = macro_by_key[selected_key]

    floors = tuple(
        numerical_improvement_floor(float(value["objective"])) for value in baselines
    )
    macro_floor = numerical_improvement_floor(baseline_macro)
    relative_by_family = (
        tuple(
            (float(base["objective"]) - float(candidate["objective"]))
            / float(base["objective"])
            for base, candidate in zip(baselines, selected_rows)
        )
        if selected is not None
        else (0.0,) * _INNER_COUNT
    )
    positive_wins = (
        sum(
            float(base["objective"]) - float(candidate["objective"]) > floor
            for base, candidate, floor in zip(baselines, selected_rows, floors)
        )
        if selected is not None
        else 0
    )
    mirrors: tuple[dict[str, object], ...] = ()
    mirrors_complete = selected is None
    if selected is not None:
        raw_mirrors = tuple(value["matched_negative"] for value in roles)
        if all(isinstance(value, Mapping) for value in raw_mirrors):
            mirrors = tuple(
                validate_nested_microstep_candidate_score(value)  # type: ignore[arg-type]
                for value in raw_mirrors
            )
            mirrors_complete = all(
                value["path"] == selected["path"]
                and float(value["alpha"]) == -float(selected["alpha"])
                and value["fit_receipt_sha256"]
                == selected_rows[index]["fit_receipt_sha256"]
                for index, value in enumerate(mirrors)
            )
        if require_mirrors and not mirrors_complete:
            raise ValueError("selected inner candidate requires seven matched mirrors")
    elif require_mirrors and any(value["matched_negative"] is not None for value in roles):
        raise ValueError("baseline rollback cannot consume matched mirrors")

    mirror_macro = (
        math.fsum(float(value["objective"]) for value in mirrors) / _INNER_COUNT
        if mirrors_complete and mirrors
        else None
    )
    mirror_wins = (
        sum(
            float(mirror["objective"]) - float(candidate["objective"]) > floor
            for mirror, candidate, floor in zip(mirrors, selected_rows, floors)
        )
        if mirrors_complete and mirrors
        else 0
    )
    absolute = baseline_macro - selected_macro
    relative = absolute / baseline_macro
    finite_trust_rank_execution = bool(
        selected is not None
        and all(
            base["finite"] is True
            and base["pointwise_trust_passed"] is True
            and base["rank_is_16"] is True
            for base in baselines
        )
        and all(
            value["finite"] is True
            and value["pointwise_trust_passed"] is True
            and value["rank_is_16"] is True
            and value["execution_changed"] is True
            for value in (*selected_rows, *mirrors)
        )
        and mirrors_complete
    )
    beats_mirror_macro = bool(
        mirror_macro is not None and mirror_macro - selected_macro > macro_floor
    )
    worst = min(relative_by_family)
    passed = bool(
        selected is not None
        and mirrors_complete
        and relative >= _MATERIALITY
        and positive_wins >= _INNER_WINS
        and worst >= -_WORST_REGRESSION
        and beats_mirror_macro
        and mirror_wins >= _INNER_WINS
        and finite_trust_rank_execution
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_inner_selection.v1",
        _SELECTION_DOMAIN,
        {
            "outer_held_family_id": outer,
            "inner_held_family_ids": inner_ids,
            "inner_role_artifact_sha256s": tuple(value["artifact_sha256"] for value in roles),
            "selected": selected,
            "baseline_macro_objective": baseline_macro,
            "selected_macro_objective": selected_macro,
            "matched_negative_macro_objective": mirror_macro,
            "objective_numerical_improvement_floor": macro_floor,
            "family_numerical_improvement_floors": floors,
            "objective_absolute_improvement": absolute,
            "objective_relative_improvement": relative,
            "positive_win_count": positive_wins,
            "positive_required_win_count": _INNER_WINS,
            "worst_family_relative_improvement": worst,
            "worst_family_regression_maximum": _WORST_REGRESSION,
            "mirror_win_count": mirror_wins,
            "mirror_required_win_count": _INNER_WINS,
            "positive_beats_mirror_macro": beats_mirror_macro,
            "finite_trust_rank_execution_passed": finite_trust_rank_execution,
            "mirrors_complete": mirrors_complete,
            "passed": passed,
        },
    )


def build_nested_microstep_selection_receipt(
    *, panel_receipt: Mapping[str, object], shared_fit_receipts: Sequence[Mapping[str, object]],
    inner_roles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    fits = _shared_fit_map(panel, shared_fit_receipts)
    supplied = tuple(_mapping(value, label="inner role") for value in _sequence(inner_roles, label="inner roles"))
    if len(supplied) != 56 or len({value.get("artifact_sha256") for value in supplied}) != 56:
        raise ValueError("selection receipt requires 56 unique ordered inner roles")
    grouped: dict[str, list[Mapping[str, object]]] = {str(value): [] for value in panel["family_ids"]}
    for value in supplied:
        outer = _identifier(value.get("outer_held_family_id"), label="role outer family")
        if outer not in grouped:
            raise ValueError("inner role outer family differs from panel")
        grouped[outer].append(value)
    selections = tuple(
        select_nested_microstep_inner_candidate(
            panel_receipt=panel,
            shared_fit_receipts=tuple(fits.values()),
            outer_held_family_id=outer,
            inner_roles=grouped[outer],
            require_mirrors=True,
        )
        for outer in panel["family_ids"]
    )
    passed = all(value["passed"] is True for value in selections)
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_selection_receipt.v1",
        _SELECTION_SET_DOMAIN,
        {
            "protocol_sha256": NESTED_MICROSTEP_PROTOCOL_SHA256,
            "panel_artifact_sha256": panel["artifact_sha256"],
            "shared_fit_receipt_sha256s": tuple(
                fits[key]["artifact_sha256"] for key in sorted(fits)
            ),
            "outer_selections": selections,
            "physical_shared_fit_count": 28,
            "ordered_inner_role_count": 56,
            "passed_outer_selection_count": sum(value["passed"] is True for value in selections),
            "passed": passed,
            "outer_validation_authorized": passed,
            "raw_tensors_or_logits_serialized": False,
        },
    )


def validate_nested_microstep_selection_receipt(value: Mapping[str, object], *, panel_receipt: Mapping[str, object], shared_fit_receipts: Sequence[Mapping[str, object]], inner_roles: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = _mapping(value, label="selection receipt")
    _exact(selected, {"schema", "protocol_sha256", "panel_artifact_sha256", "shared_fit_receipt_sha256s", "outer_selections", "physical_shared_fit_count", "ordered_inner_role_count", "passed_outer_selection_count", "passed", "outer_validation_authorized", "raw_tensors_or_logits_serialized", "artifact_sha256"}, label="selection receipt")
    rebuilt = build_nested_microstep_selection_receipt(panel_receipt=panel_receipt, shared_fit_receipts=shared_fit_receipts, inner_roles=inner_roles)
    _same(selected, rebuilt, label="selection")
    return rebuilt


def _validate_outer_selection(value: Mapping[str, object]) -> Mapping[str, object]:
    selected = _mapping(value, label="outer selection")
    _selection_hash_valid(selected)
    chosen = selected.get("selected")
    if chosen is not None:
        chosen = _mapping(chosen, label="selected microstep")
        _exact(chosen, {"key", "path", "alpha"}, label="selected microstep")
        if chosen["key"] != nested_microstep_candidate_key(chosen["path"], chosen["alpha"]):
            raise ValueError("selected microstep key differs")
    return selected


def build_nested_microstep_outer_score(
    *, panel_receipt: Mapping[str, object], selection: Mapping[str, object],
    full_fit_receipt: Mapping[str, object], baseline: Mapping[str, object],
    selected_positive: Mapping[str, object], matched_negative: Mapping[str, object],
) -> dict[str, object]:
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    chosen_selection = _validate_outer_selection(selection)
    if chosen_selection.get("passed") is not True:
        raise ValueError("failed inner selection cannot access outer capability")
    chosen = _mapping(chosen_selection.get("selected"), label="selected microstep")
    outer = _identifier(chosen_selection.get("outer_held_family_id"), label="outer family")
    fit = validate_nested_microstep_fit_receipt(full_fit_receipt, panel_receipt=panel)
    if fit["kind"] != "outer_full" or tuple(fit["excluded_family_ids"]) != (outer,):
        raise ValueError("outer score fit geometry differs")
    base = validate_nested_microstep_baseline_score(baseline)
    positive = validate_nested_microstep_candidate_score(selected_positive)
    mirror = validate_nested_microstep_candidate_score(matched_negative)
    fit_sha = fit["artifact_sha256"]
    if any(value["fit_receipt_sha256"] != fit_sha for value in (base, positive, mirror)):
        raise ValueError("outer score fit binding differs")
    if (
        positive["key"] != chosen["key"]
        or positive["path"] != chosen["path"]
        or positive["alpha"] != chosen["alpha"]
        or mirror["path"] != chosen["path"]
        or float(mirror["alpha"]) != -float(chosen["alpha"])
    ):
        raise ValueError("outer score differs from frozen inner selection")
    baseline_objective = float(base["objective"])
    positive_objective = float(positive["objective"])
    mirror_objective = float(mirror["objective"])
    floor = numerical_improvement_floor(baseline_objective)
    relative = (baseline_objective - positive_objective) / baseline_objective
    finite_trust_rank_execution = bool(
        base["finite"] is True and base["pointwise_trust_passed"] is True and base["rank_is_16"] is True
        and all(value["finite"] is True and value["pointwise_trust_passed"] is True and value["rank_is_16"] is True and value["execution_changed"] is True for value in (positive, mirror))
    )
    family_prompts = panel["family_prompt_sha256s"]
    assert isinstance(family_prompts, Mapping)
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_outer_score.v1", _OUTER_DOMAIN,
        {"panel_artifact_sha256": panel["artifact_sha256"], "selection_artifact_sha256": chosen_selection["artifact_sha256"],
         "outer_held_family_id": outer, "outer_score_prompt_sha256s": tuple(family_prompts[outer]),
         "full_fit_receipt_sha256": fit_sha, "selected": dict(chosen),
         "baseline": base, "selected_positive": positive, "matched_negative": mirror,
         "objective_numerical_improvement_floor": floor,
         "objective_absolute_improvement": baseline_objective - positive_objective,
         "objective_relative_improvement": relative,
         "positive_beats_baseline_beyond_floor": baseline_objective - positive_objective > floor,
         "positive_beats_mirror_beyond_floor": mirror_objective - positive_objective > floor,
         "finite_trust_rank_execution_passed": finite_trust_rank_execution,
         "outer_family_used_once_after_selection_freeze": True,
         "raw_tensors_or_logits_serialized": False},
    )


def _validate_outer_score_hash(value: Mapping[str, object]) -> Mapping[str, object]:
    artifact = _sha(value.get("artifact_sha256"), label="outer score")
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    if _hash(_OUTER_DOMAIN, payload) != artifact:
        raise ValueError("outer score artifact hash drifted")
    for key in ("objective_numerical_improvement_floor", "objective_absolute_improvement", "objective_relative_improvement"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"outer {key} must be finite")
    return value


def nested_microstep_work_accounting(
    *, outer_scored: bool, positive_outer_selection_count: int = 8
) -> dict[str, object]:
    if type(outer_scored) is not bool:
        raise TypeError("outer_scored must be boolean")
    selected_count = _integer(
        positive_outer_selection_count,
        label="positive outer selection count",
        positive=False,
    )
    if selected_count > 8:
        raise ValueError("positive outer selection count exceeds eight")
    if outer_scored and selected_count != 8:
        raise ValueError("outer scoring requires all eight positive selections")
    pair_fits = 28
    ordered_roles = 56
    outer_fits = 8 if outer_scored else 0
    inner_positive = ordered_roles * len(NESTED_MICROSTEP_CANDIDATE_KEYS)
    inner_mirrors = selected_count * _INNER_COUNT
    inner_baselines = ordered_roles
    outer_baselines = 8 if outer_scored else 0
    outer_positives = 8 if outer_scored else 0
    outer_mirrors = 8 if outer_scored else 0
    collection_forwards = 32
    fit_training_forwards = pair_fits * 12 + outer_fits * 14
    scored_forwards = 2 * (
        inner_baselines + inner_positive + inner_mirrors
        + outer_baselines + outer_positives + outer_mirrors
    )
    forwards = collection_forwards + fit_training_forwards + scored_forwards
    suffix_backwards = 16 + fit_training_forwards
    local_contractions = fit_training_forwards
    capability_accesses = fit_training_forwards + scored_forwards
    return {
        "outer_scored": outer_scored,
        "physical_shared_pair_fit_count": pair_fits,
        "ordered_inner_role_count": ordered_roles,
        "ordered_role_fit_count_without_reuse": ordered_roles,
        "shared_fit_reuse_saved_physical_fit_count": ordered_roles - pair_fits,
        "physical_outer_full_fit_count": outer_fits,
        "physical_total_fit_count": pair_fits + outer_fits,
        "inner_baseline_score_count": inner_baselines,
        "inner_positive_candidate_score_count": inner_positive,
        "inner_matched_negative_score_count": inner_mirrors,
        "outer_baseline_score_count": outer_baselines,
        "outer_selected_positive_score_count": outer_positives,
        "outer_matched_negative_score_count": outer_mirrors,
        "full_model_forward_count": forwards,
        "full_suffix_backward_traversal_count": suffix_backwards,
        "local_head_autograd_contraction_count": local_contractions,
        "total_autograd_grad_call_count": suffix_backwards + local_contractions,
        "teacher_capability_access_count": capability_accesses,
        "post_cast_h4_hash_check_count": capability_accesses,
        "supervised_full_vocab_logits_hash_check_count": capability_accesses,
        "breakdown": {
            "collection_source_and_vjp_forwards": collection_forwards,
            "physical_fit_training_forwards": fit_training_forwards,
            "held_score_forwards": scored_forwards,
        },
    }


def build_nested_microstep_validation_receipt(
    *, panel_receipt: Mapping[str, object], shared_fit_receipts: Sequence[Mapping[str, object]],
    selection_receipt: Mapping[str, object], outer_scores: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    panel = validate_nested_microstep_panel_receipt(panel_receipt)
    fits = _shared_fit_map(panel, shared_fit_receipts)
    selection = _mapping(selection_receipt, label="selection receipt")
    artifact = _sha(selection.get("artifact_sha256"), label="selection receipt")
    selection_payload = dict(selection)
    selection_payload.pop("artifact_sha256", None)
    if _hash(_SELECTION_SET_DOMAIN, selection_payload) != artifact:
        raise ValueError("selection receipt artifact hash drifted")
    raw_scores = tuple(_mapping(value, label="outer score") for value in _sequence(outer_scores, label="outer scores"))
    scores = tuple(_validate_outer_score_hash(value) for value in raw_scores)
    authorized = selection.get("outer_validation_authorized") is True and selection.get("passed") is True
    if authorized:
        if len(scores) != 8:
            raise ValueError("authorized outer validation requires eight scores")
    elif scores:
        raise ValueError("failed inner selection cannot contain outer scores")
    outer_ids = tuple(sorted(_identifier(value.get("outer_held_family_id"), label="outer score family") for value in scores))
    if scores and (len(set(outer_ids)) != 8 or outer_ids != tuple(panel["family_ids"])):
        raise ValueError("outer score family geometry differs")
    selection_by_outer = {
        str(value["outer_held_family_id"]): value
        for value in _sequence(selection.get("outer_selections"), label="outer selections")
        if isinstance(value, Mapping)
    }
    for score in scores:
        outer = str(score["outer_held_family_id"])
        selected = selection_by_outer.get(outer)
        if selected is None or score.get("selection_artifact_sha256") != selected.get("artifact_sha256"):
            raise ValueError("outer score differs from frozen selection receipt")
    if scores:
        baselines = tuple(float(_mapping(value["baseline"], label="outer baseline")["objective"]) for value in scores)
        positives = tuple(float(_mapping(value["selected_positive"], label="outer positive")["objective"]) for value in scores)
        mirrors = tuple(float(_mapping(value["matched_negative"], label="outer mirror")["objective"]) for value in scores)
        baseline_macro = math.fsum(baselines) / 8
        positive_macro = math.fsum(positives) / 8
        mirror_macro = math.fsum(mirrors) / 8
        floors = tuple(numerical_improvement_floor(value) for value in baselines)
        macro_floor = numerical_improvement_floor(baseline_macro)
        relative_values = tuple((base - positive) / base for base, positive in zip(baselines, positives))
        positive_wins = sum(base - positive > floor for base, positive, floor in zip(baselines, positives, floors))
        mirror_wins = sum(mirror - positive > floor for mirror, positive, floor in zip(mirrors, positives, floors))
        relative = (baseline_macro - positive_macro) / baseline_macro
        worst = min(relative_values)
        finite_gate = all(value.get("finite_trust_rank_execution_passed") is True for value in scores)
        beats_mirror = mirror_macro - positive_macro > macro_floor
        passed = bool(relative >= _MATERIALITY and positive_wins >= _OUTER_WINS and worst >= -_WORST_REGRESSION and beats_mirror and mirror_wins >= _OUTER_WINS and finite_gate)
    else:
        baseline_macro = positive_macro = mirror_macro = None
        macro_floor = None
        relative = 0.0
        worst = 0.0
        positive_wins = mirror_wins = 0
        finite_gate = beats_mirror = passed = False
    work = nested_microstep_work_accounting(
        outer_scored=bool(scores),
        positive_outer_selection_count=sum(
            isinstance(value, Mapping) and value.get("selected") is not None
            for value in _sequence(selection.get("outer_selections"), label="outer selections")
        ),
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_nested_validation.v1", _REPORT_DOMAIN,
        {"protocol_sha256": NESTED_MICROSTEP_PROTOCOL_SHA256,
         "panel_artifact_sha256": panel["artifact_sha256"],
         "selection_receipt_sha256": artifact,
         "shared_fit_receipt_sha256s": tuple(fits[key]["artifact_sha256"] for key in sorted(fits)),
         "outer_score_artifact_sha256s": tuple(value["artifact_sha256"] for value in sorted(scores, key=lambda item: str(item["outer_held_family_id"]))),
         "baseline_macro_objective": baseline_macro, "selected_macro_objective": positive_macro,
         "matched_negative_macro_objective": mirror_macro,
         "objective_numerical_improvement_floor": macro_floor,
         "objective_relative_improvement": relative,
         "positive_win_count": positive_wins, "positive_required_win_count": _OUTER_WINS,
         "worst_family_relative_improvement": worst,
         "worst_family_regression_maximum": _WORST_REGRESSION,
         "mirror_win_count": mirror_wins, "mirror_required_win_count": _OUTER_WINS,
         "positive_beats_mirror_macro": beats_mirror,
         "finite_trust_rank_execution_passed": finite_gate,
         "passed": passed,
         "classification": "nested_family_disjoint_validation_passed" if passed else ("nested_inner_selection_failed" if not authorized else "nested_outer_validation_failed"),
         "held_fidelity_claim": passed, "serving_authorized": False,
         "compression_claim": False, "speed_or_latency_claim": False,
         "work_accounting": work, "raw_tensors_or_logits_serialized": False},
    )


def validate_nested_microstep_validation_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="validation receipt")
    _exact(
        selected,
        {
            "schema",
            "protocol_sha256",
            "panel_artifact_sha256",
            "selection_receipt_sha256",
            "shared_fit_receipt_sha256s",
            "outer_score_artifact_sha256s",
            "baseline_macro_objective",
            "selected_macro_objective",
            "matched_negative_macro_objective",
            "objective_numerical_improvement_floor",
            "objective_relative_improvement",
            "positive_win_count",
            "positive_required_win_count",
            "worst_family_relative_improvement",
            "worst_family_regression_maximum",
            "mirror_win_count",
            "mirror_required_win_count",
            "positive_beats_mirror_macro",
            "finite_trust_rank_execution_passed",
            "passed",
            "classification",
            "held_fidelity_claim",
            "serving_authorized",
            "compression_claim",
            "speed_or_latency_claim",
            "work_accounting",
            "raw_tensors_or_logits_serialized",
            "artifact_sha256",
        },
        label="validation receipt",
    )
    if selected.get("schema") != "fisher_graph.complete_h4_fisher_nested_validation.v1":
        raise ValueError("nested validation schema differs")
    protocol = _sha(selected.get("protocol_sha256"), label="validation protocol")
    if protocol != NESTED_MICROSTEP_PROTOCOL_SHA256:
        raise ValueError("nested validation protocol differs")
    _sha(selected.get("panel_artifact_sha256"), label="validation panel")
    _sha(selected.get("selection_receipt_sha256"), label="validation selection")
    artifact = _sha(selected.get("artifact_sha256"), label="validation receipt")
    payload = dict(selected)
    payload.pop("artifact_sha256", None)
    if _hash(_REPORT_DOMAIN, payload) != artifact:
        raise ValueError("validation receipt artifact hash drifted")
    for key in ("objective_relative_improvement", "worst_family_relative_improvement"):
        raw = selected.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"validation {key} must be finite")
    for key in (
        "positive_beats_mirror_macro",
        "finite_trust_rank_execution_passed",
        "passed",
        "held_fidelity_claim",
        "serving_authorized",
        "compression_claim",
        "speed_or_latency_claim",
        "raw_tensors_or_logits_serialized",
    ):
        _boolean(selected.get(key), label=f"validation {key}")
    if (
        selected.get("serving_authorized") is not False
        or selected.get("compression_claim") is not False
        or selected.get("speed_or_latency_claim") is not False
        or selected.get("raw_tensors_or_logits_serialized") is not False
    ):
        raise ValueError("nested validation overclaims authority")
    for key, expected in (
        ("positive_required_win_count", _OUTER_WINS),
        ("mirror_required_win_count", _OUTER_WINS),
    ):
        if type(selected.get(key)) is not int or selected.get(key) != expected:
            raise ValueError(f"validation {key} differs")
    for key in ("positive_win_count", "mirror_win_count"):
        count = selected.get(key)
        if type(count) is not int or not 0 <= count <= _FAMILY_COUNT:
            raise ValueError(f"validation {key} differs")
    if selected.get("worst_family_regression_maximum") != _WORST_REGRESSION:
        raise ValueError("validation worst-family boundary differs")

    normalized = dict(selected)
    shared_fits = tuple(
        _sha(item, label="shared fit receipt")
        for item in _sequence(
            selected.get("shared_fit_receipt_sha256s"),
            label="shared fit receipts",
        )
    )
    if len(shared_fits) != 28 or len(set(shared_fits)) != 28:
        raise ValueError("validation shared-fit geometry differs")
    outer_scores = tuple(
        _sha(item, label="outer score receipt")
        for item in _sequence(
            selected.get("outer_score_artifact_sha256s"),
            label="outer score receipts",
        )
    )
    if len(outer_scores) not in (0, _FAMILY_COUNT) or len(set(outer_scores)) != len(
        outer_scores
    ):
        raise ValueError("validation outer-score geometry differs")
    outer_scored = len(outer_scores) == _FAMILY_COUNT

    work = _mapping(selected.get("work_accounting"), label="validation work")
    if outer_scored:
        positive_selection_count = _FAMILY_COUNT
    else:
        mirror_count = work.get("inner_matched_negative_score_count")
        if (
            type(mirror_count) is not int
            or mirror_count < 0
            or mirror_count > _FAMILY_COUNT * _INNER_COUNT
            or mirror_count % _INNER_COUNT != 0
        ):
            raise ValueError("validation work selection count differs")
        positive_selection_count = mirror_count // _INNER_COUNT
    expected_work = nested_microstep_work_accounting(
        outer_scored=outer_scored,
        positive_outer_selection_count=positive_selection_count,
    )
    if _canonical(work) != _canonical(expected_work):
        raise ValueError("validation work accounting drifted")

    passed = bool(selected["passed"])
    relative = float(selected["objective_relative_improvement"])
    worst = float(selected["worst_family_relative_improvement"])
    positive_wins = int(selected["positive_win_count"])
    mirror_wins = int(selected["mirror_win_count"])
    if outer_scored:
        baseline_macro = _number(
            selected.get("baseline_macro_objective"),
            label="validation baseline macro",
            positive=True,
        )
        selected_macro = _number(
            selected.get("selected_macro_objective"),
            label="validation selected macro",
        )
        matched_negative_macro = _number(
            selected.get("matched_negative_macro_objective"),
            label="validation matched-negative macro",
        )
        floor = _number(
            selected.get("objective_numerical_improvement_floor"),
            label="validation numerical floor",
            positive=True,
        )
        if relative != (baseline_macro - selected_macro) / baseline_macro:
            raise ValueError("validation relative improvement drifted")
        if floor != numerical_improvement_floor(baseline_macro):
            raise ValueError("validation numerical floor drifted")
        if selected.get("positive_beats_mirror_macro") is not (
            matched_negative_macro - selected_macro > floor
        ):
            raise ValueError("validation matched-negative comparison drifted")
        expected_passed = bool(
            relative >= _MATERIALITY
            and positive_wins >= _OUTER_WINS
            and worst >= -_WORST_REGRESSION
            and selected.get("positive_beats_mirror_macro") is True
            and mirror_wins >= _OUTER_WINS
            and selected.get("finite_trust_rank_execution_passed") is True
        )
    else:
        if any(
            selected.get(key) is not None
            for key in (
                "baseline_macro_objective",
                "selected_macro_objective",
                "matched_negative_macro_objective",
                "objective_numerical_improvement_floor",
            )
        ) or any(
            (
                relative != 0.0,
                worst != 0.0,
                positive_wins != 0,
                mirror_wins != 0,
                selected.get("positive_beats_mirror_macro") is not False,
                selected.get("finite_trust_rank_execution_passed") is not False,
            )
        ):
            raise ValueError("failed inner validation metrics differ")
        expected_passed = False
    if passed is not expected_passed:
        raise ValueError("nested validation pass decision drifted")
    expected_classification = (
        "nested_family_disjoint_validation_passed"
        if passed
        else (
            "nested_outer_validation_failed"
            if outer_scored
            else "nested_inner_selection_failed"
        )
    )
    if selected.get("classification") != expected_classification:
        raise ValueError("nested validation classification differs")
    if selected.get("held_fidelity_claim") is not passed:
        raise ValueError("nested validation held-fidelity claim differs")

    normalized["shared_fit_receipt_sha256s"] = shared_fits
    normalized["outer_score_artifact_sha256s"] = outer_scores
    normalized["work_accounting"] = dict(work)
    return normalized
