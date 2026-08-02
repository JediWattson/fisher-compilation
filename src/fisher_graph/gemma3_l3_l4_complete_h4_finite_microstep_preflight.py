"""V20a signed finite-microstep fit-only preflight.

V19 selected checkpoint zero in every Fisher and PCA outer fold even though
its first Adam proposal changed the provider substantially.  This preflight
asks the narrower question that must be answered before a nested held-out
experiment is justified: does any *fraction* of the actual first V19 Fisher
Adam proposal produce a finite, execution-changing descent on the fit rows?

The lexically first Fisher LOFO fold is a joint-path sentinel.  Only if that
sentinel has a positive descent which also beats its signed mirror does the
run expand to all eight Fisher folds and the direction-only, pedal-only, and
joint paths.  Held rows may exist in the authenticated A16 cache, but each
fold's teacher capability excludes them and this module never evaluates them.

This is deliberately not a candidate-producing rung.  It writes one 0600,
scalar/hash-only JSON report and never emits a provider sidecar or makes a
serving, compression, speed, parameter, FLOP, or held-fidelity claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat

import torch
from torch import Tensor

from .complete_h4_fisher_finite_microstep import (
    FisherFiniteMicrostepReceipt,
    build_autonomous_complete_h4_fisher_finite_microstep,
    fisher_finite_microstep_selected_tensor_sha256s,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_fisher_pedal_development as _v18
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    gemma3_l3_l4_shadow_model_inputs_sha256,
)


__all__ = [
    "ALPHA_LADDER",
    "POSITIVE_ALPHAS",
    "MICROSTEP_PATHS",
    "DEFAULT_OUTPUT",
    "detect_execution_change",
    "numerical_improvement_floor",
    "choose_positive_candidate",
    "select_best_positive_microstep",
    "evaluate_sentinel_decision",
    "evaluate_fold_qualification",
    "classify_preflight",
    "build_finite_microstep_preflight_report",
    "run_gemma3_l3_l4_complete_h4_finite_microstep_preflight",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V19_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-finite-joint-pedal-r16-k256-"
    "outer-lofo-a-fit16-dev-v19.json"
)
_V19_LOGICAL_SHA256 = (
    "4f0439858b7e636ae648aa12d3cdb6837350510f10b520ab1c09e69074417d46"
)
_V19_FILE_SHA256 = (
    "b29e45590c3085c18ba9ad516a3bf508d34a83c57a622f8069035d3e457a9a1e"
)
_V19_CLASSIFICATION = "finite_joint_pedal_outer_fidelity_insufficient"

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-finite-microstep-preflight-"
    "r16-k256-a-fit16-dev-v20a.json"
)
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_finite_microstep_"
    "fit_only_preflight.v20a"
)
_FORMAT_VERSION = 20
_REPORT_DOMAIN = b"fisher-graph:complete-h4-finite-microstep-preflight:v20a\0"
_PROTOCOL_DOMAIN = b"fisher-graph:finite-microstep-preflight-protocol:v20a\0"
_EVIDENCE_DOMAIN = b"fisher-graph:finite-microstep-preflight-evidence:v20a\0"
_ENDPOINT_DOMAIN = b"fisher-graph:finite-microstep-preflight-endpoints:v20a\0"
_EXECUTION_DOMAIN = b"fisher-graph:finite-microstep-preflight-execution:v20a\0"
_REPORT_READY_CHECKPOINT_DOMAIN = (
    b"fisher-graph:finite-microstep-preflight-report-ready:v20a\0"
)
_REPORT_READY_CHECKPOINT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_finite_microstep_"
    "report_ready_checkpoint.v20a"
)
_REPORT_INPUT_KEYS = frozenset(
    {
        "panel",
        "bridge_binding_sha256",
        "prerequisite",
        "fit_collection",
        "sentinel",
        "folds",
        "work",
        "integrity",
    }
)
_REPORT_READY_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "microstep_protocol_sha256",
        "v19_report_sha256",
        "v19_file_sha256",
        "candidate",
        "provider_sidecar",
        "report_inputs",
        "checkpoint_sha256",
    }
)

_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_FIT_PROMPTS = 14
_EXPECTED_HELD_PROMPTS = 2
_EXPECTED_VOCABULARY = 262_144
_PARENT_RANK = 256
_CONDITIONAL_RANK = 16
_MATERIALITY_THRESHOLD = 0.01
_ABSOLUTE_IMPROVEMENT_FLOOR = 1.0e-12
_ROUNDOFF_MULTIPLIER = 128.0

ALPHA_LADDER = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
POSITIVE_ALPHAS = ALPHA_LADDER[1:]
MICROSTEP_PATHS = ("direction_only", "pedal_only", "joint")
_PATH_ORDER = {path: index for index, path in enumerate(MICROSTEP_PATHS)}

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "signed_fit_only_first_v19_adam_factor_space_microstep",
    "prerequisite": "pinned_V19_checkpoint_zero_and_first_Adam_proposal",
    "panel": "opened_A16_eight_families_two_prompts_each",
    "split": "eight_family_LOFO_fit14_with_held2_capability_excluded",
    "sentinel": "lexically_first_held_family_joint_path_only",
    "sentinel_short_circuit": True,
    "conditional_expansion": "all_eight_folds_all_three_paths",
    "coordinate_objective": "reverse_vjp_fisher",
    "microstep_paths": MICROSTEP_PATHS,
    "alpha_ladder": ALPHA_LADDER,
    "interpolation": "actual_V19_factor_and_pedal_parameter_endpoints",
    "dense_gradient_normalization": False,
    "svd_or_rank_retraction_after_checkpoint_zero": False,
    "checkpoint_zero_gradient_reused_across_paths": True,
    "objective": (
        "family_equal_example_equal_token_mean_exact_float64_full_vocab_"
        "KL_source_to_candidate_through_full_suffix_fit_rows_only"
    ),
    "selection": (
        "minimum_objective_exact_tie_baseline_then_smaller_alpha_then_"
        "direction_only_pedal_only_joint"
    ),
    "execution_change": (
        "at_least_one_fit_prompt_post_cast_H4_hash_and_at_least_one_fit_"
        "prompt_supervised_full_vocab_logits_hash_differ_from_checkpoint_zero"
    ),
    "mirror": "one_exact_negative_alpha_at_selected_positive_path_and_magnitude",
    "materiality_relative_improvement_min": _MATERIALITY_THRESHOLD,
    "candidate_or_sidecar": False,
}
_MICROSTEP_PROTOCOL_SHA256 = _v14._sha256(
    _FIXED_PROTOCOL, domain=_PROTOCOL_DOMAIN
)


def _is_under_local_runs(path: Path) -> bool:
    return ".local-runs" in path.resolve(strict=False).parts


def _same_destination(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or not _is_under_local_runs(output):
        raise ValueError("V20a output must be JSON under .local-runs")
    if _same_destination(output, _V19_OUTPUT):
        raise ValueError("V20a must preserve the write-once V19 artifact")
    return output


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _sha_mapping(value: Mapping[str, object], *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} hashes must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        name = _v14._identifier(key, label=f"{label} key")
        result[name] = _v19._sha256_identifier(item, label=f"{label} {name}")
    return result


def detect_execution_change(
    *,
    base_parameter_sha256s: Mapping[str, object],
    candidate_parameter_sha256s: Mapping[str, object],
    base_h4_sha256s: Mapping[str, object],
    candidate_h4_sha256s: Mapping[str, object],
    base_logits_sha256s: Mapping[str, object],
    candidate_logits_sha256s: Mapping[str, object],
) -> dict[str, object]:
    """Compare authenticated parameter and per-prompt execution hashes."""

    base_parameters = _sha_mapping(base_parameter_sha256s, label="base parameter")
    candidate_parameters = _sha_mapping(
        candidate_parameter_sha256s, label="candidate parameter"
    )
    base_h4 = _sha_mapping(base_h4_sha256s, label="base H4")
    candidate_h4 = _sha_mapping(candidate_h4_sha256s, label="candidate H4")
    base_logits = _sha_mapping(base_logits_sha256s, label="base logits")
    candidate_logits = _sha_mapping(candidate_logits_sha256s, label="candidate logits")
    if set(base_parameters) != set(candidate_parameters):
        raise ValueError("parameter hash geometry differs")
    if set(base_h4) != set(candidate_h4) or set(base_logits) != set(candidate_logits):
        raise ValueError("execution prompt hash geometry differs")
    if set(base_h4) != set(base_logits):
        raise ValueError("H4/logits prompt ownership differs")

    changed_parameters = tuple(
        key for key in sorted(base_parameters) if base_parameters[key] != candidate_parameters[key]
    )
    changed_h4 = tuple(
        key for key in sorted(base_h4) if base_h4[key] != candidate_h4[key]
    )
    changed_logits = tuple(
        key for key in sorted(base_logits) if base_logits[key] != candidate_logits[key]
    )
    payload: dict[str, object] = {
        "parameter_changed": bool(changed_parameters),
        "parameter_changed_tensor_count": len(changed_parameters),
        "parameter_changed_tensor_names": changed_parameters,
        "h4_changed_prompt_count": len(changed_h4),
        "h4_changed_example_ids": changed_h4,
        "logits_changed_prompt_count": len(changed_logits),
        "logits_changed_example_ids": changed_logits,
        "execution_changed": bool(changed_h4) and bool(changed_logits),
        "fit_prompt_count": len(base_h4),
    }
    payload["receipt_sha256"] = _v14._sha256(
        payload, domain=_EXECUTION_DOMAIN
    )
    return payload


def _candidate_path(value: Mapping[str, object]) -> str:
    path = value.get("path", value.get("microstep_path"))
    if not isinstance(path, str) or path not in _PATH_ORDER:
        raise ValueError("finite-microstep candidate path differs")
    return path


def _candidate_alpha(value: Mapping[str, object]) -> float:
    alpha = _finite_float(value.get("alpha"), label="candidate alpha")
    if alpha not in ALPHA_LADDER or alpha <= 0.0:
        raise ValueError("positive candidate alpha is outside the fixed ladder")
    return alpha


def _candidate_objective(value: Mapping[str, object]) -> float:
    return _finite_float(
        value.get("objective", value.get("fit_teacher_kl")),
        label="candidate objective",
    )


def _candidate_execution_changed(value: Mapping[str, object]) -> bool:
    direct = value.get("execution_changed")
    if type(direct) is bool:
        return direct
    change = value.get("execution_change")
    if isinstance(change, Mapping) and type(change.get("execution_changed")) is bool:
        return bool(change["execution_changed"])
    raise TypeError("candidate execution-change flag is missing")


_CANDIDATE_SHA_FIELDS = (
    "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256",
    "provider_artifact_sha256",
    "microstep_artifact_sha256",
    "microstep_receipt_sha256",
    "microstep_evidence_sha256",
)


def _validate_candidate_authentication(
    value: Mapping[str, object],
    *,
    baseline: Mapping[str, object],
    signed: bool,
) -> dict[str, object]:
    """Recompute one candidate's copied scalar/hash execution evidence."""

    path = _candidate_path(value)
    alpha = _finite_float(value.get("alpha"), label="candidate alpha")
    if signed:
        if alpha >= 0.0 or -alpha not in POSITIVE_ALPHAS:
            raise ValueError("matched negative alpha is outside the mirrored ladder")
    elif alpha not in POSITIVE_ALPHAS:
        raise ValueError("positive candidate alpha is outside the fixed ladder")
    _candidate_objective(value)
    for key in _CANDIDATE_SHA_FIELDS:
        _v19._sha256_identifier(value.get(key), label=f"candidate {key}")
    receipt_metadata = _mapping_field(
        value, "microstep_receipt", label="candidate microstep receipt"
    )
    receipt = FisherFiniteMicrostepReceipt.from_metadata(receipt_metadata)
    parameter_hashes = _sha_mapping(
        _mapping_field(value, "parameter_sha256s", label="candidate parameters"),
        label="candidate parameter",
    )
    if (
        receipt.base_provider_artifact_sha256
        != value.get("base_provider_artifact_sha256")
        or receipt.proposal_provider_artifact_sha256
        != value.get("proposal_provider_artifact_sha256")
        or receipt.selected_provider_artifact_sha256
        != value.get("provider_artifact_sha256")
        or receipt.microstep_path != path
        or receipt.alpha != alpha
        or receipt.microstep_protocol_sha256 != _MICROSTEP_PROTOCOL_SHA256
        or receipt.microstep_evidence_sha256
        != value.get("microstep_evidence_sha256")
        or receipt.artifact_sha256 != value.get("microstep_receipt_sha256")
        or receipt.artifact_sha256 != value.get("microstep_artifact_sha256")
        or dict(receipt.selected_tensor_sha256s) != parameter_hashes
        or receipt.rank != _PARENT_RANK
        or receipt.conditional_rank != _CONDITIONAL_RANK
    ):
        raise ValueError("candidate core microstep receipt binding differs")
    for key in ("finite", "pointwise_trust_passed", "rank_is_16"):
        if type(value.get(key)) is not bool:
            raise TypeError(f"candidate {key} must be boolean")
    change = value.get("execution_change")
    if not isinstance(change, Mapping):
        raise TypeError("candidate execution-change receipt is missing")
    recomputed = detect_execution_change(
        base_parameter_sha256s=_mapping_field(
            baseline, "parameter_sha256s", label="baseline parameters"
        ),
        candidate_parameter_sha256s=parameter_hashes,
        base_h4_sha256s=_mapping_field(
            baseline, "post_cast_h4_sha256s", label="baseline H4"
        ),
        candidate_h4_sha256s=_mapping_field(
            value, "post_cast_h4_sha256s", label="candidate H4"
        ),
        base_logits_sha256s=_mapping_field(
            baseline,
            "supervised_full_vocab_logits_sha256s",
            label="baseline logits",
        ),
        candidate_logits_sha256s=_mapping_field(
            value,
            "supervised_full_vocab_logits_sha256s",
            label="candidate logits",
        ),
    )
    if _v14._canonical_json_bytes(dict(change)) != _v14._canonical_json_bytes(
        recomputed
    ):
        raise ValueError("candidate execution-change receipt drifted")
    if value.get("execution_changed") is not recomputed["execution_changed"]:
        raise ValueError("candidate execution-change summary drifted")
    families = _mapping_field(
        value, "family_objectives", label="candidate family objectives"
    )
    if len(families) != _EXPECTED_FAMILIES - 1:
        raise ValueError("candidate family objective geometry differs")
    family_values = tuple(
        _finite_float(item, label="candidate family objective")
        for _, item in sorted(families.items())
    )
    if math.fsum(family_values) / len(family_values) != _candidate_objective(value):
        raise ValueError("candidate family-equal objective drifted")
    return {
        "path": path,
        "alpha": alpha,
        "execution_change": recomputed,
    }


def _mapping_field(
    value: Mapping[str, object], key: str, *, label: str
) -> Mapping[str, object]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise TypeError(f"{label} mapping is missing")
    return selected


def _validate_exact_positive_grid(
    candidates: Sequence[Mapping[str, object]],
    *,
    expected_paths: Sequence[str] | None = None,
) -> tuple[Mapping[str, object], ...]:
    selected = tuple(candidates)
    if not selected or any(not isinstance(row, Mapping) for row in selected):
        raise TypeError("positive candidate grid must contain mappings")
    paths = {_candidate_path(row) for row in selected}
    if expected_paths is None:
        if paths == {"joint"}:
            required_paths = ("joint",)
        elif paths == set(MICROSTEP_PATHS):
            required_paths = MICROSTEP_PATHS
        else:
            raise ValueError("positive candidate grid path geometry differs")
    else:
        required_paths = tuple(expected_paths)
        if (
            not required_paths
            or len(required_paths) != len(set(required_paths))
            or any(path not in MICROSTEP_PATHS for path in required_paths)
            or paths != set(required_paths)
        ):
            raise ValueError("positive candidate grid path geometry differs")
    expected = {
        (path, alpha) for path in required_paths for alpha in POSITIVE_ALPHAS
    }
    observed = {
        (_candidate_path(row), _candidate_alpha(row)) for row in selected
    }
    if len(selected) != len(expected) or observed != expected:
        raise ValueError("positive candidate grid is not exhaustive")
    return selected


def select_best_positive_microstep(
    *,
    baseline_objective: float,
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Select the best nonzero execution-changing positive ladder point.

    The baseline participates in the eventual decision, but an execution-
    changing positive is still returned when it loses to baseline so its
    matched negative control can be evaluated without outcome-dependent work.
    """

    _finite_float(baseline_objective, label="baseline objective")
    selected: list[tuple[float, float, int, Mapping[str, object]]] = []
    for row in _validate_exact_positive_grid(candidates):
        path = _candidate_path(row)
        alpha = _candidate_alpha(row)
        objective = _candidate_objective(row)
        if _candidate_execution_changed(row):
            selected.append((objective, alpha, _PATH_ORDER[path], row))
    if not selected:
        return None
    return min(selected, key=lambda value: value[:3])[3]


def _numerical_floor(baseline_objective: float) -> float:
    baseline = _finite_float(baseline_objective, label="baseline objective")
    return max(
        _ABSOLUTE_IMPROVEMENT_FLOOR,
        _ROUNDOFF_MULTIPLIER
        * torch.finfo(torch.float64).eps
        * abs(baseline),
    )


def numerical_improvement_floor(baseline_objective: float) -> float:
    """Public fixed float64 qualification floor."""

    return _numerical_floor(baseline_objective)


def choose_positive_candidate(
    *,
    baseline_objective: float,
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Compatibility name for the frozen positive-candidate selector."""

    return select_best_positive_microstep(
        baseline_objective=baseline_objective,
        candidates=candidates,
    )


def _decision(
    *,
    baseline_objective: float,
    selected_positive: Mapping[str, object] | None,
    matched_negative: Mapping[str, object] | None,
    numerical_floor: float | None,
    scope: str,
) -> dict[str, object]:
    baseline = _finite_float(baseline_objective, label=f"{scope} baseline")
    floor = _numerical_floor(baseline) if numerical_floor is None else _finite_float(
        numerical_floor, label=f"{scope} numerical floor"
    )
    if floor != _numerical_floor(baseline):
        raise ValueError(f"{scope} numerical floor differs from the fixed floor")
    if selected_positive is None:
        if matched_negative is not None:
            raise ValueError(f"{scope} cannot score a mirror without a positive")
        return {
            "scope": scope,
            "baseline_objective": baseline,
            "selected_positive": None,
            "matched_negative": None,
            "objective_numerical_improvement_floor": floor,
            "positive_execution_changed": False,
            "matched_negative_execution_changed": False,
            "positive_beats_baseline_beyond_floor": False,
            "positive_beats_mirror_beyond_floor": False,
            "objective_absolute_improvement": 0.0,
            "objective_relative_improvement": 0.0,
            "finite_trust_rank_passed": False,
            "passed": False,
        }
    path = _candidate_path(selected_positive)
    alpha = _candidate_alpha(selected_positive)
    objective = _candidate_objective(selected_positive)
    if not isinstance(matched_negative, Mapping):
        raise TypeError(f"{scope} matched negative candidate is missing")
    mirror_path = _candidate_path(matched_negative)
    mirror_alpha = _finite_float(
        matched_negative.get("alpha"), label=f"{scope} mirror alpha"
    )
    mirror = _candidate_objective(matched_negative)
    if mirror_path != path or mirror_alpha != -alpha:
        raise ValueError(f"{scope} matched negative path/alpha differs")
    for key in ("base_provider_artifact_sha256", "proposal_provider_artifact_sha256"):
        positive_endpoint = _v19._sha256_identifier(
            selected_positive.get(key), label=f"{scope} positive {key}"
        )
        negative_endpoint = _v19._sha256_identifier(
            matched_negative.get(key), label=f"{scope} negative {key}"
        )
        if positive_endpoint != negative_endpoint:
            raise ValueError(f"{scope} matched negative endpoint differs")
    for row, label in (
        (selected_positive, "positive"),
        (matched_negative, "matched negative"),
    ):
        for key in _CANDIDATE_SHA_FIELDS:
            _v19._sha256_identifier(row.get(key), label=f"{scope} {label} {key}")
        change = row.get("execution_change")
        if not isinstance(change, Mapping):
            raise TypeError(f"{scope} {label} execution receipt is missing")
        _v19._sha256_identifier(
            change.get("receipt_sha256"),
            label=f"{scope} {label} execution receipt",
        )
        for key in (
            "parameter_sha256s",
            "post_cast_h4_sha256s",
            "supervised_full_vocab_logits_sha256s",
        ):
            _sha_mapping(_mapping_field(row, key, label=f"{scope} {label} {key}"), label=key)
        receipt = FisherFiniteMicrostepReceipt.from_metadata(
            _mapping_field(
                row,
                "microstep_receipt",
                label=f"{scope} {label} microstep receipt",
            )
        )
        if (
            receipt.microstep_path != _candidate_path(row)
            or receipt.alpha
            != _finite_float(row.get("alpha"), label=f"{scope} {label} alpha")
            or receipt.base_provider_artifact_sha256
            != row.get("base_provider_artifact_sha256")
            or receipt.proposal_provider_artifact_sha256
            != row.get("proposal_provider_artifact_sha256")
            or receipt.selected_provider_artifact_sha256
            != row.get("provider_artifact_sha256")
            or receipt.artifact_sha256 != row.get("microstep_receipt_sha256")
            or receipt.artifact_sha256 != row.get("microstep_artifact_sha256")
            or receipt.microstep_protocol_sha256 != _MICROSTEP_PROTOCOL_SHA256
            or receipt.microstep_evidence_sha256
            != row.get("microstep_evidence_sha256")
            or receipt.rank != _PARENT_RANK
            or receipt.conditional_rank != _CONDITIONAL_RANK
            or dict(receipt.selected_tensor_sha256s)
            != dict(_mapping_field(row, "parameter_sha256s", label="parameters"))
        ):
            raise ValueError(f"{scope} {label} core receipt differs")
    finite = (
        selected_positive.get("finite") is True
        and matched_negative.get("finite") is True
    )
    trust = (
        selected_positive.get("pointwise_trust_passed") is True
        and matched_negative.get("pointwise_trust_passed") is True
    )
    rank = (
        selected_positive.get("rank_is_16") is True
        and matched_negative.get("rank_is_16") is True
    )
    execution_changed = _candidate_execution_changed(selected_positive)
    mirror_execution_changed = _candidate_execution_changed(matched_negative)
    absolute = baseline - objective
    relative = absolute / baseline if baseline > 0.0 else 0.0
    beats_baseline = absolute > floor
    beats_mirror = mirror - objective > floor
    passed = bool(
        execution_changed
        and mirror_execution_changed
        and finite
        and trust
        and rank
        and beats_baseline
        and beats_mirror
    )
    return {
        "scope": scope,
        "baseline_objective": baseline,
        "selected_positive": {
            "path": path,
            "alpha": alpha,
            "objective": objective,
            "provider_artifact_sha256": selected_positive.get(
                "provider_artifact_sha256"
            ),
            "microstep_artifact_sha256": selected_positive.get(
                "microstep_artifact_sha256"
            ),
            "execution_change": selected_positive.get("execution_change"),
        },
        "matched_negative": {
            "path": mirror_path,
            "alpha": mirror_alpha,
            "objective": mirror,
            "provider_artifact_sha256": matched_negative.get(
                "provider_artifact_sha256"
            ),
            "microstep_artifact_sha256": matched_negative.get(
                "microstep_artifact_sha256"
            ),
            "microstep_evidence_sha256": matched_negative.get(
                "microstep_evidence_sha256"
            ),
            "execution_change": matched_negative.get("execution_change"),
        },
        "objective_numerical_improvement_floor": floor,
        "positive_execution_changed": execution_changed,
        "matched_negative_execution_changed": mirror_execution_changed,
        "positive_beats_baseline_beyond_floor": beats_baseline,
        "positive_beats_mirror_beyond_floor": beats_mirror,
        "objective_absolute_improvement": absolute,
        "objective_relative_improvement": relative,
        "finite_trust_rank_passed": finite and trust and rank,
        "passed": passed,
    }


def evaluate_sentinel_decision(
    *,
    baseline_objective: float,
    selected_positive: Mapping[str, object] | None,
    matched_negative: Mapping[str, object] | None,
    numerical_floor: float | None = None,
) -> dict[str, object]:
    """Qualify the lexically first fold's joint-only sentinel."""

    if selected_positive is not None and _candidate_path(selected_positive) != "joint":
        raise ValueError("sentinel candidate must use the joint path")
    return _decision(
        baseline_objective=baseline_objective,
        selected_positive=selected_positive,
        matched_negative=matched_negative,
        numerical_floor=numerical_floor,
        scope="lexically_first_fisher_fold_joint_only_sentinel",
    )


def evaluate_fold_qualification(
    *,
    held_family_id: str,
    baseline_objective: float,
    selected_positive: Mapping[str, object] | None,
    matched_negative: Mapping[str, object] | None,
    numerical_floor: float | None = None,
) -> dict[str, object]:
    """Qualify one expanded fit-only Fisher fold."""

    held = _v14._identifier(held_family_id, label="V20a held family")
    return {
        "held_family_id": held,
        **_decision(
            baseline_objective=baseline_objective,
            selected_positive=selected_positive,
            matched_negative=matched_negative,
            numerical_floor=numerical_floor,
            scope="expanded_fisher_fold_all_paths",
        ),
    }


def _work_accounting(
    *,
    tested_fold_count: int,
    positive_candidate_count: int,
    mirror_candidate_count: int,
) -> dict[str, object]:
    """Return exact nominal work for either legal V20a execution shape."""

    for label, value in (
        ("tested fold", tested_fold_count),
        ("positive candidate", positive_candidate_count),
        ("mirror candidate", mirror_candidate_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{label} count must be a nonnegative integer")
    legal = (
        tested_fold_count == 1
        and positive_candidate_count == 7
        and mirror_candidate_count in {0, 1}
    ) or (
        tested_fold_count == 8
        and positive_candidate_count == 168
        and 1 <= mirror_candidate_count <= 9
    )
    if not legal:
        raise ValueError("V20a work geometry differs from the frozen protocol")

    collection_forwards = 2 * _EXPECTED_PROMPTS
    checkpoint_zero_forwards = tested_fold_count * _EXPECTED_FIT_PROMPTS
    positive_forwards = positive_candidate_count * _EXPECTED_FIT_PROMPTS
    mirror_forwards = mirror_candidate_count * _EXPECTED_FIT_PROMPTS
    forwards = (
        collection_forwards
        + checkpoint_zero_forwards
        + positive_forwards
        + mirror_forwards
    )
    suffix_backwards = _EXPECTED_PROMPTS + checkpoint_zero_forwards
    local_contractions = checkpoint_zero_forwards
    capability_accesses = checkpoint_zero_forwards + positive_forwards + mirror_forwards
    expected_forwards = (
        144 + 14 * mirror_candidate_count
        if tested_fold_count == 1
        else 2_496 + 14 * mirror_candidate_count
    )
    if forwards != expected_forwards:
        raise RuntimeError("V20a forward accounting drifted")
    return {
        "tested_fold_count": tested_fold_count,
        "parent_fit_count": tested_fold_count,
        "fisher_start_fit_count": tested_fold_count,
        "shared_checkpoint_zero_gradient_count": tested_fold_count,
        "positive_nonzero_candidate_count": positive_candidate_count,
        "mirror_candidate_execution_count": mirror_candidate_count,
        "full_model_forward_count": forwards,
        "full_suffix_backward_traversal_count": suffix_backwards,
        "local_head_autograd_contraction_count": local_contractions,
        "total_autograd_grad_call_count": suffix_backwards + local_contractions,
        "teacher_capability_access_count": capability_accesses,
        "post_cast_h4_hash_check_count": capability_accesses,
        "supervised_full_vocab_logits_hash_check_count": capability_accesses,
        "breakdown": {
            "collection_native_source_forwards": _EXPECTED_PROMPTS,
            "collection_base_vjp_forwards": _EXPECTED_PROMPTS,
            "collection_base_vjp_backwards": _EXPECTED_PROMPTS,
            "checkpoint_zero_gradient_forwards": checkpoint_zero_forwards,
            "checkpoint_zero_gradient_backwards": checkpoint_zero_forwards,
            "positive_finite_execution_forwards": positive_forwards,
            "matched_negative_finite_execution_forwards": mirror_forwards,
        },
    }


def _classification(
    *,
    sentinel: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    macro_relative_improvement: float,
) -> str:
    sentinel_decision = sentinel.get("qualification", sentinel)
    if not isinstance(sentinel_decision, Mapping):
        raise TypeError("sentinel qualification is missing")
    fold_decisions = tuple(
        fold.get("qualification", fold) for fold in folds
    )
    if any(not isinstance(value, Mapping) for value in fold_decisions):
        raise TypeError("fold qualification is missing")
    if sentinel_decision.get("passed") is not True:
        if (
            sentinel_decision.get("positive_execution_changed") is True
            and sentinel_decision.get("positive_beats_baseline_beyond_floor") is True
            and sentinel_decision.get("positive_beats_mirror_beyond_floor") is not True
        ):
            return "finite_microstep_direction_ambiguous"
        return "finite_microstep_no_descent_interval_sentinel"
    if any(
        fold.get("positive_execution_changed") is True
        and fold.get("positive_beats_baseline_beyond_floor") is True
        and fold.get("positive_beats_mirror_beyond_floor") is not True
        for fold in fold_decisions  # type: ignore[union-attr]
    ):
        return "finite_microstep_direction_ambiguous"
    if len(folds) != _EXPECTED_FAMILIES or any(
        fold.get("passed") is not True for fold in fold_decisions  # type: ignore[union-attr]
    ):
        return "finite_microstep_descent_not_uniform"
    if macro_relative_improvement < _MATERIALITY_THRESHOLD:
        return "finite_microstep_descent_below_materiality"
    return "finite_microstep_preflight_passed_for_nested_validation"


def classify_preflight(
    *,
    sentinel: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    macro_relative_improvement: float,
) -> str:
    """Public deterministic V20a classification rule."""

    return _classification(
        sentinel=sentinel,
        folds=folds,
        macro_relative_improvement=_finite_float(
            macro_relative_improvement,
            label="macro relative improvement",
        ),
    )


def _load_authenticated_v19_artifact(
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Read the pinned V19 report and derive its per-fold Fisher authority."""

    _v18._validate_prerequisite_report(
        _V19_OUTPUT,
        logical_sha256=_V19_LOGICAL_SHA256,
        file_sha256=_V19_FILE_SHA256,
        classification=_V19_CLASSIFICATION,
        format_version=19,
    )
    try:
        payload = json.loads(_V19_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("pinned V19 report is unreadable") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != 19
        or payload.get("report_sha256") != _V19_LOGICAL_SHA256
        or payload.get("classification") != _V19_CLASSIFICATION
        or payload.get("passed") is not False
        or payload.get("candidate") is not None
        or payload.get("full_refit_qualification") is not None
        or _V19_OUTPUT.with_suffix(".provider.pt").exists()
    ):
        raise RuntimeError("pinned V19 report semantics drifted")
    arms = _mapping_field(payload, "arms", label="pinned V19 arms")
    fisher = _mapping_field(
        _mapping_field(
            arms, _v19.FISHER_CONDITIONAL_ID, label="pinned V19 Fisher arm"
        ),
        "optimization_receipts",
        label="pinned V19 Fisher optimization receipts",
    )
    fisher_ownership = _mapping_field(
        _mapping_field(
            arms, _v19.FISHER_CONDITIONAL_ID, label="pinned V19 Fisher arm"
        ),
        "fold_ownership_receipts",
        label="pinned V19 Fisher ownership receipts",
    )
    parent_hashes = _mapping_field(
        _mapping_field(arms, _v19.PARENT_ID, label="pinned V19 parent arm"),
        "fold_provider_artifact_sha256s",
        label="pinned V19 parent hashes",
    )
    start_hashes = _mapping_field(
        _mapping_field(arms, _v19.FISHER_START_ID, label="pinned V19 start arm"),
        "fold_provider_artifact_sha256s",
        label="pinned V19 start hashes",
    )
    held_ids = tuple(sorted(fisher))
    if (
        len(held_ids) != _EXPECTED_FAMILIES
        or set(parent_hashes) != set(held_ids)
        or set(start_hashes) != set(held_ids)
        or set(fisher_ownership) != set(held_ids)
    ):
        raise RuntimeError("pinned V19 Fisher fold geometry differs")
    bundles: dict[str, dict[str, object]] = {}
    receipt_hashes: dict[str, str] = {}
    for held in held_ids:
        _v14._identifier(held, label="pinned V19 held family")
        receipt = fisher[held]
        if not isinstance(receipt, Mapping) or receipt.get("held_family_id") != held:
            raise RuntimeError("pinned V19 Fisher receipt ownership differs")
        receipt_hash = _v14._sha256(dict(receipt), domain=_EVIDENCE_DOMAIN)
        receipt_hashes[held] = receipt_hash
        bundles[held] = {
            "optimization_receipt": dict(receipt),
            "optimization_receipt_sha256": receipt_hash,
            "parent_provider_artifact_sha256": _v19._sha256_identifier(
                parent_hashes[held], label="pinned V19 parent provider"
            ),
            "start_provider_artifact_sha256": _v19._sha256_identifier(
                start_hashes[held], label="pinned V19 start provider"
            ),
            "capability_receipt": dict(
                _mapping_field(
                    receipt,
                    "capability_receipt",
                    label="pinned V19 Fisher capability receipt",
                )
            ),
            "ownership_receipt": dict(
                _mapping_field(
                    fisher_ownership,
                    held,
                    label="pinned V19 Fisher ownership receipt",
                )
            ),
        }
    authenticated_panel = dict(
        _mapping_field(payload, "panel", label="pinned V19 panel")
    )
    authenticated_fit_collection = dict(
        _mapping_field(
            payload, "fit_collection", label="pinned V19 fit collection"
        )
    )
    prerequisite = {
        "path": _V19_OUTPUT.as_posix(),
        "format_version": 19,
        "report_sha256": _V19_LOGICAL_SHA256,
        "file_sha256": _V19_FILE_SHA256,
        "classification": _V19_CLASSIFICATION,
        "passed": False,
        "candidate": None,
        "full_refit_qualification": None,
        "provider_sidecar_absent": True,
        "fisher_optimization_receipt_sha256s": receipt_hashes,
        "authenticated_panel": authenticated_panel,
        "authenticated_bridge_binding_sha256": _v19._sha256_identifier(
            payload.get("bridge_binding_sha256"),
            label="pinned V19 bridge binding",
        ),
        "authenticated_fit_collection": authenticated_fit_collection,
    }
    return prerequisite, bundles


def _validate_prerequisite_receipt(
    value: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    actual, bundles = _load_authenticated_v19_artifact()
    if _v14._canonical_json_bytes(dict(value)) != _v14._canonical_json_bytes(actual):
        raise ValueError("V20a prerequisite differs from authenticated V19")
    return actual, bundles


def _validate_fit_collection(
    value: Mapping[str, object],
    *,
    authenticated_v19: Mapping[str, object],
) -> dict[str, object]:
    if (
        value.get("prompt_count") != _EXPECTED_PROMPTS
        or value.get("family_count") != _EXPECTED_FAMILIES
        or value.get("held_teacher_rows_cached") is not True
        or value.get("held_teacher_rows_scored") is not False
    ):
        raise ValueError("V20a fit collection geometry or held authority differs")
    records = _sha_mapping(
        _mapping_field(value, "record_receipt_sha256s", label="record receipts"),
        label="record receipt",
    )
    if len(records) != _EXPECTED_PROMPTS:
        raise ValueError("V20a record receipt count differs")
    vault = _mapping_field(
        value, "teacher_vault_receipt", label="teacher vault receipt"
    )
    if vault.get("example_count") != _EXPECTED_PROMPTS or vault.get(
        "family_count"
    ) != _EXPECTED_FAMILIES:
        raise ValueError("V20a teacher vault geometry differs")
    _v19._sha256_identifier(
        vault.get("artifact_sha256"), label="V20a teacher vault"
    )
    authenticated_vault = _mapping_field(
        authenticated_v19,
        "teacher_vault",
        label="authenticated V19 teacher vault",
    )
    authenticated_traces = authenticated_v19.get("trace_receipt_sha256s")
    if (
        dict(vault) != dict(authenticated_vault)
        or not isinstance(authenticated_traces, Sequence)
        or isinstance(authenticated_traces, (str, bytes))
        or tuple(sorted(records.values()))
        != tuple(sorted(str(item) for item in authenticated_traces))
        or authenticated_v19.get("prompt_count") != _EXPECTED_PROMPTS
        or authenticated_v19.get("family_count") != _EXPECTED_FAMILIES
        or authenticated_v19.get(
            "held_rows_cached_but_capability_excluded_and_not_consumed_by_fold_fit"
        )
        is not True
    ):
        raise ValueError("V20a fit collection differs from authenticated V19")
    return dict(value)


def _validate_panel_and_bridge_authority(
    *,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    validated_prerequisite: Mapping[str, object],
) -> None:
    authenticated_panel = _mapping_field(
        validated_prerequisite,
        "authenticated_panel",
        label="authenticated V19 panel",
    )
    if _v14._canonical_json_bytes(dict(panel)) != _v14._canonical_json_bytes(
        dict(authenticated_panel)
    ):
        raise ValueError("V20a panel differs from authenticated V19")
    if bridge_binding_sha256 != validated_prerequisite.get(
        "authenticated_bridge_binding_sha256"
    ):
        raise ValueError("V20a bridge differs from authenticated V19")


_STATE_HASH_KEYS = {
    "direction_left_sha256",
    "direction_right_sha256",
    "pedal_weight_sha256",
    "pedal_bias_sha256",
}


def _state_hash_receipt(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _STATE_HASH_KEYS:
        raise ValueError(f"{label} state hash geometry differs")
    return {
        key: _v19._sha256_identifier(item, label=f"{label} {key}")
        for key, item in value.items()
    }


def _validate_endpoint_binding(
    value: Mapping[str, object],
    *,
    held_family_id: str,
    baseline: Mapping[str, object],
    pinned_v19_bundle: Mapping[str, object],
) -> dict[str, object]:
    held = _v14._identifier(held_family_id, label="endpoint held family")
    if value.get("held_family_id") != held:
        raise ValueError("endpoint held-family binding differs")
    receipt_pin = _v19._sha256_identifier(
        value.get("pinned_v19_optimization_receipt_sha256"),
        label="pinned V19 optimization receipt",
    )
    if receipt_pin != pinned_v19_bundle.get("optimization_receipt_sha256"):
        raise ValueError("endpoint V19 optimization receipt pin differs")
    pinned_receipt = _mapping_field(
        pinned_v19_bundle,
        "optimization_receipt",
        label="authenticated V19 optimization receipt",
    )
    pinned_provider_artifacts = pinned_receipt.get(
        "checkpoint_provider_artifact_sha256s"
    )
    pinned_state_receipts = pinned_receipt.get("checkpoint_state_receipts")
    pinned_scores = pinned_receipt.get("checkpoint_scores")
    authenticated_family_scores = pinned_receipt.get("checkpoint_family_scores")
    if any(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2
        for item in (
            pinned_provider_artifacts,
            pinned_state_receipts,
            pinned_scores,
            authenticated_family_scores,
        )
    ):
        raise ValueError("authenticated V19 checkpoint receipt geometry differs")
    for current, pinned, label in (
        ("parent_provider_artifact_sha256", "pinned_parent_provider_artifact_sha256", "parent"),
        ("start_provider_artifact_sha256", "pinned_start_provider_artifact_sha256", "start"),
        (
            "base_provider_artifact_sha256",
            "pinned_base_provider_artifact_sha256",
            "checkpoint zero",
        ),
        (
            "proposal_provider_artifact_sha256",
            "pinned_proposal_provider_artifact_sha256",
            "first Adam",
        ),
    ):
        actual = _v19._sha256_identifier(value.get(current), label=f"endpoint {label}")
        expected = _v19._sha256_identifier(
            value.get(pinned), label=f"pinned endpoint {label}"
        )
        if actual != expected:
            raise ValueError(f"reconstructed {label} endpoint differs from V19")
    direct_expected = {
        "parent_provider_artifact_sha256": pinned_v19_bundle.get(
            "parent_provider_artifact_sha256"
        ),
        "start_provider_artifact_sha256": pinned_receipt.get(
            "start_provider_artifact_sha256"
        ),
        "base_provider_artifact_sha256": pinned_provider_artifacts[0],  # type: ignore[index]
        "proposal_provider_artifact_sha256": pinned_provider_artifacts[1],  # type: ignore[index]
    }
    if pinned_v19_bundle.get("start_provider_artifact_sha256") != direct_expected[
        "start_provider_artifact_sha256"
    ] or any(value.get(key) != expected for key, expected in direct_expected.items()):
        raise ValueError("endpoint lineage differs from authenticated V19 data")
    for current, pinned, label in (
        (
            "checkpoint_zero_state_sha256s",
            "pinned_checkpoint_zero_state_sha256s",
            "checkpoint zero",
        ),
        ("first_adam_state_sha256s", "pinned_first_adam_state_sha256s", "first Adam"),
    ):
        if _state_hash_receipt(value.get(current), label=label) != _state_hash_receipt(
            value.get(pinned), label=f"pinned {label}"
        ):
            raise ValueError(f"reconstructed {label} state differs from V19")
    if _state_hash_receipt(
        value.get("checkpoint_zero_state_sha256s"), label="checkpoint zero"
    ) != _state_hash_receipt(
        pinned_state_receipts[0], label="authenticated checkpoint zero"  # type: ignore[index]
    ) or _state_hash_receipt(
        value.get("first_adam_state_sha256s"), label="first Adam"
    ) != _state_hash_receipt(
        pinned_state_receipts[1], label="authenticated first Adam"  # type: ignore[index]
    ):
        raise ValueError("endpoint states differ from authenticated V19 data")
    objective = _finite_float(
        value.get("checkpoint_zero_objective"), label="endpoint checkpoint zero objective"
    )
    pinned_objective = _finite_float(
        value.get("pinned_checkpoint_zero_objective"),
        label="pinned checkpoint zero objective",
    )
    authenticated_objective = _finite_float(
        pinned_scores[0], label="authenticated checkpoint zero objective"  # type: ignore[index]
    )
    if (
        objective != pinned_objective
        or objective != baseline.get("objective")
        or objective != authenticated_objective
    ):
        raise ValueError("checkpoint-zero scalar objective differs from V19")
    family_scores = _mapping_field(
        value,
        "checkpoint_zero_family_objectives",
        label="checkpoint-zero family objectives",
    )
    copied_pinned_family_scores = _mapping_field(
        value,
        "pinned_checkpoint_zero_family_objectives",
        label="pinned checkpoint-zero family objectives",
    )
    if (
        len(family_scores) != _EXPECTED_FAMILIES - 1
        or dict(family_scores) != dict(copied_pinned_family_scores)
        or dict(family_scores)
        != dict(
            _mapping_field(
                baseline,
                "family_objectives",
                label="baseline family objectives",
            )
        )
        or dict(family_scores)
        != dict(authenticated_family_scores[0])  # type: ignore[arg-type,index]
    ):
        raise ValueError("checkpoint-zero family objectives differ from V19")
    if (
        tuple(value.get("training_family_ids", ()))
        != tuple(pinned_receipt.get("training_family_ids", ()))
        or tuple(value.get("training_sequence_sha256s", ()))
        != tuple(pinned_receipt.get("training_sequence_sha256s", ()))
        or tuple(value.get("training_record_receipt_sha256s", ()))
        != tuple(pinned_receipt.get("training_record_receipt_sha256s", ()))
    ):
        raise ValueError("endpoint fit ownership differs from authenticated V19")
    checks = _mapping_field(value, "checks", label="endpoint checks")
    if not checks or any(item is not True for item in checks.values()):
        raise ValueError("endpoint reconstruction checks did not all pass")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    receipt = _v19._sha256_identifier(
        value.get("receipt_sha256"), label="endpoint binding receipt"
    )
    if _v14._sha256(payload, domain=_ENDPOINT_DOMAIN) != receipt:
        raise ValueError("endpoint binding receipt drifted")
    return dict(value)


def _validate_baseline(value: Mapping[str, object]) -> dict[str, object]:
    objective = _finite_float(value.get("objective"), label="baseline objective")
    families = _mapping_field(
        value, "family_objectives", label="baseline family objectives"
    )
    family_values = tuple(
        _finite_float(item, label="baseline family objective")
        for _, item in sorted(families.items())
    )
    if (
        len(family_values) != _EXPECTED_FAMILIES - 1
        or math.fsum(family_values) / len(family_values) != objective
    ):
        raise ValueError("baseline family-equal objective drifted")
    _v19._sha256_identifier(
        value.get("provider_artifact_sha256"), label="baseline provider"
    )
    parameters = _sha_mapping(
        _mapping_field(value, "parameter_sha256s", label="baseline parameters"),
        label="baseline parameter",
    )
    h4 = _sha_mapping(
        _mapping_field(value, "post_cast_h4_sha256s", label="baseline H4"),
        label="baseline H4",
    )
    logits = _sha_mapping(
        _mapping_field(
            value,
            "supervised_full_vocab_logits_sha256s",
            label="baseline logits",
        ),
        label="baseline logits",
    )
    if (
        len(parameters) != 4
        or len(h4) != _EXPECTED_FIT_PROMPTS
        or set(h4) != set(logits)
    ):
        raise ValueError("baseline parameter or execution hash geometry differs")
    for key in ("finite", "pointwise_trust_passed", "rank_is_16"):
        if value.get(key) is not True:
            raise ValueError(f"baseline {key} did not pass")
    return dict(value)


def _validate_capability_receipt(
    value: Mapping[str, object],
    *,
    held_family_id: str,
    executions_per_prompt: int,
    pinned_v19_capability: Mapping[str, object],
) -> dict[str, object]:
    held = _v14._identifier(held_family_id, label="capability held family")
    if (
        value.get("held_family_id") != held
        or value.get("authorized_example_count") != _EXPECTED_FIT_PROMPTS
        or value.get("authorized_family_count") != _EXPECTED_FAMILIES - 1
        or value.get("held_family_capability_excluded") is not True
        or value.get("teacher_rows_consumed_only_through_capability") is not True
        or value.get("access_count") != _EXPECTED_FIT_PROMPTS * executions_per_prompt
    ):
        raise ValueError("fold teacher capability authority or access count differs")
    _v19._sha256_identifier(
        value.get("artifact_sha256"), label="teacher capability"
    )
    counts = _mapping_field(
        value, "per_example_access_counts", label="capability access counts"
    )
    if len(counts) != _EXPECTED_FIT_PROMPTS or any(
        item != executions_per_prompt for item in counts.values()
    ):
        raise ValueError("teacher capability per-example accesses differ")
    pinned_counts = _mapping_field(
        pinned_v19_capability,
        "per_example_access_counts",
        label="authenticated V19 capability access ids",
    )
    if (
        value.get("artifact_sha256")
        != pinned_v19_capability.get("artifact_sha256")
        or set(counts) != set(pinned_counts)
        or pinned_v19_capability.get("held_family_id") != held
        or pinned_v19_capability.get("authorized_example_count")
        != _EXPECTED_FIT_PROMPTS
        or pinned_v19_capability.get("authorized_family_count")
        != _EXPECTED_FAMILIES - 1
        or pinned_v19_capability.get("held_family_capability_excluded") is not True
    ):
        raise ValueError("fold capability differs from authenticated V19 authority")
    return dict(value)


def _ownership_receipt_sha256(
    value: Mapping[str, object],
    *,
    label: str,
) -> str:
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    receipt = _v19._sha256_identifier(
        value.get("receipt_sha256"), label=f"{label} receipt"
    )
    if _v14._sha256(payload, domain=_v19._OWNERSHIP_DOMAIN) != receipt:
        raise ValueError(f"{label} receipt hash drifted")
    return receipt


def _validate_ownership_receipt(
    value: Mapping[str, object],
    *,
    held_family_id: str,
    pinned_v19_ownership: Mapping[str, object],
) -> dict[str, object]:
    held = _v14._identifier(held_family_id, label="ownership held family")
    held_sequences = value.get("held_sequence_sha256s")
    fit_families = value.get("fit_family_ids")
    fit_sequences = value.get("fit_sequence_sha256s")
    if (
        value.get("held_family_id") != held
        or not isinstance(held_sequences, Sequence)
        or isinstance(held_sequences, (str, bytes))
        or len(held_sequences) != _EXPECTED_HELD_PROMPTS
        or len(set(held_sequences)) != _EXPECTED_HELD_PROMPTS
        or not isinstance(fit_families, Sequence)
        or isinstance(fit_families, (str, bytes))
        or len(fit_families) != _EXPECTED_FAMILIES - 1
        or len(set(fit_families)) != _EXPECTED_FAMILIES - 1
        or held in set(fit_families)
        or not isinstance(fit_sequences, Sequence)
        or isinstance(fit_sequences, (str, bytes))
        or len(fit_sequences) != _EXPECTED_FIT_PROMPTS
        or len(set(fit_sequences)) != _EXPECTED_FIT_PROMPTS
        or not set(held_sequences).isdisjoint(set(fit_sequences))
        or value.get("held_family_absent_from_fit_family_ids") is not True
        or value.get("held_sequences_disjoint_from_fit_sequences") is not True
    ):
        raise ValueError("fold ownership receipt differs")
    for item in (*held_sequences, *fit_sequences):
        _v19._sha256_identifier(item, label="ownership sequence")
    _ownership_receipt_sha256(value, label="fold ownership")
    if _v14._canonical_json_bytes(dict(value)) != _v14._canonical_json_bytes(
        dict(pinned_v19_ownership)
    ):
        raise ValueError("fold ownership differs from authenticated V19")
    return dict(value)


def _recompute_fold(
    value: Mapping[str, object],
    *,
    expanded: bool,
    extra_prior_mirror_count: int,
    pinned_v19_bundle: Mapping[str, object],
) -> tuple[dict[str, object], int, int]:
    held = _v14._identifier(value.get("held_family_id"), label="V20a held family")
    if value.get("held_scoring_performed") is not False:
        raise ValueError("V20a fold consumed held score rows")
    baseline = _validate_baseline(
        _mapping_field(value, "baseline", label="fold baseline")
    )
    endpoint = _validate_endpoint_binding(
        _mapping_field(value, "endpoint_binding", label="endpoint binding"),
        held_family_id=held,
        baseline=baseline,
        pinned_v19_bundle=pinned_v19_bundle,
    )
    if baseline["provider_artifact_sha256"] != endpoint[
        "base_provider_artifact_sha256"
    ]:
        raise ValueError("baseline provider differs from checkpoint-zero endpoint")
    candidate_rows = value.get("positive_candidates")
    if not isinstance(candidate_rows, Sequence) or isinstance(
        candidate_rows, (str, bytes)
    ):
        raise TypeError("fold positive candidate grid is missing")
    candidates = _validate_exact_positive_grid(candidate_rows)  # type: ignore[arg-type]
    if expanded and len(candidates) != len(MICROSTEP_PATHS) * len(POSITIVE_ALPHAS):
        raise ValueError("expanded fold omitted a path/alpha candidate")
    if not expanded and (
        len(candidates) != len(POSITIVE_ALPHAS)
        or {_candidate_path(row) for row in candidates} != {"joint"}
    ):
        raise ValueError("sentinel fold grid differs from joint-only protocol")
    for row in candidates:
        _validate_candidate_authentication(row, baseline=baseline, signed=False)
        if (
            row.get("base_provider_artifact_sha256")
            != endpoint["base_provider_artifact_sha256"]
            or row.get("proposal_provider_artifact_sha256")
            != endpoint["proposal_provider_artifact_sha256"]
        ):
            raise ValueError("positive candidate endpoint binding differs")
    selected = select_best_positive_microstep(
        baseline_objective=float(baseline["objective"]),
        candidates=candidates,
    )
    negative_value = value.get("matched_negative")
    negative: Mapping[str, object] | None
    if selected is None:
        if negative_value is not None:
            raise ValueError("fold emitted a mirror without an execution-changing positive")
        negative = None
    else:
        if not isinstance(negative_value, Mapping):
            raise TypeError("fold matched negative is missing")
        negative = negative_value
        _validate_candidate_authentication(negative, baseline=baseline, signed=True)
    qualification = (
        evaluate_fold_qualification(
            held_family_id=held,
            baseline_objective=float(baseline["objective"]),
            selected_positive=selected,
            matched_negative=negative,
        )
        if expanded
        else evaluate_sentinel_decision(
            baseline_objective=float(baseline["objective"]),
            selected_positive=selected,
            matched_negative=negative,
        )
    )
    supplied = _mapping_field(value, "qualification", label="fold qualification")
    if dict(supplied) != qualification:
        raise ValueError("fold qualification decision arithmetic drifted")
    mirror_count = int(negative is not None) + extra_prior_mirror_count
    capability = _validate_capability_receipt(
        _mapping_field(value, "capability_receipt", label="capability receipt"),
        held_family_id=held,
        executions_per_prompt=1 + len(candidates) + mirror_count,
        pinned_v19_capability=_mapping_field(
            pinned_v19_bundle,
            "capability_receipt",
            label="authenticated V19 capability receipt",
        ),
    )
    ownership = _validate_ownership_receipt(
        _mapping_field(value, "ownership_receipt", label="ownership receipt"),
        held_family_id=held,
        pinned_v19_ownership=_mapping_field(
            pinned_v19_bundle,
            "ownership_receipt",
            label="authenticated V19 ownership receipt",
        ),
    )
    authorized_example_ids = set(
        _mapping_field(
            capability,
            "per_example_access_counts",
            label="authorized fit example ids",
        )
    )
    baseline_example_ids = set(
        _mapping_field(
            baseline,
            "post_cast_h4_sha256s",
            label="baseline scored H4 ids",
        )
    )
    fit_family_ids = set(ownership["fit_family_ids"])  # type: ignore[arg-type]
    baseline_family_ids = set(
        _mapping_field(
            baseline,
            "family_objectives",
            label="baseline scored family ids",
        )
    )
    if (
        baseline_example_ids != authorized_example_ids
        or set(
            _mapping_field(
                baseline,
                "supervised_full_vocab_logits_sha256s",
                label="baseline scored logits ids",
            )
        )
        != authorized_example_ids
        or baseline_family_ids != fit_family_ids
        or held in fit_family_ids
    ):
        raise ValueError("scored baseline rows differ from fold fit authority")
    for row in (*candidates, *((negative,) if negative is not None else ())):
        if (
            set(
                _mapping_field(
                    row,
                    "post_cast_h4_sha256s",
                    label="candidate scored H4 ids",
                )
            )
            != authorized_example_ids
            or set(
                _mapping_field(
                    row,
                    "supervised_full_vocab_logits_sha256s",
                    label="candidate scored logits ids",
                )
            )
            != authorized_example_ids
            or set(
                _mapping_field(
                    row,
                    "family_objectives",
                    label="candidate scored family ids",
                )
            )
            != fit_family_ids
        ):
            raise ValueError("candidate scored rows differ from fold fit authority")
    return (
        {
            **dict(value),
            "baseline": baseline,
            "positive_candidates": candidates,
            "matched_negative": negative,
            "qualification": qualification,
            "capability_receipt": capability,
            "ownership_receipt": ownership,
            "endpoint_binding": endpoint,
        },
        len(candidates),
        mirror_count,
    )


def build_finite_microstep_preflight_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    prerequisite: Mapping[str, object],
    fit_collection: Mapping[str, object],
    sentinel: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    work: Mapping[str, object],
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build and independently recompute the scalar/hash-only V20a report."""

    destination = _validate_output(artifact_path)
    validated_prerequisite, authenticated_v19_bundles = (
        _validate_prerequisite_receipt(prerequisite)
    )
    authenticated_collection = _mapping_field(
        validated_prerequisite,
        "authenticated_fit_collection",
        label="authenticated V19 fit collection",
    )
    validated_collection = _validate_fit_collection(
        fit_collection,
        authenticated_v19=authenticated_collection,
    )
    _validate_panel_and_bridge_authority(
        panel=panel,
        bridge_binding_sha256=bridge_binding_sha256,
        validated_prerequisite=validated_prerequisite,
    )
    selected_folds = tuple(folds)
    if not selected_folds:
        raise ValueError("V20a report has no tested fold")

    sentinel_held = _v14._identifier(
        sentinel.get("held_family_id"), label="V20a sentinel held family"
    )
    sentinel_fold, sentinel_positive_count, sentinel_mirror_count = _recompute_fold(
        sentinel,
        expanded=False,
        extra_prior_mirror_count=0,
        pinned_v19_bundle=authenticated_v19_bundles.get(sentinel_held, {}),
    )
    sentinel_decision = sentinel_fold["qualification"]
    if not isinstance(sentinel_decision, Mapping):
        raise TypeError("sentinel qualification is missing")

    validated_folds: list[dict[str, object]] = []
    positive_count = 0
    mirror_count = 0
    if sentinel_decision.get("passed") is True:
        if len(selected_folds) != _EXPECTED_FAMILIES:
            raise ValueError("passing sentinel must expand to all eight folds")
        held_ids = tuple(
            _v14._identifier(fold.get("held_family_id"), label="expanded held family")
            for fold in selected_folds
        )
        if len(set(held_ids)) != _EXPECTED_FAMILIES or sentinel_held != min(held_ids):
            raise ValueError("sentinel is not the lexically first expanded fold")
        first = next(
            fold
            for fold in selected_folds
            if fold.get("held_family_id") == sentinel_held
        )
        first_candidates = first.get("positive_candidates")
        if not isinstance(first_candidates, Sequence) or isinstance(
            first_candidates, (str, bytes)
        ):
            raise TypeError("first expanded fold candidate grid is missing")
        first_joint = tuple(
            row
            for row in first_candidates
            if isinstance(row, Mapping) and _candidate_path(row) == "joint"
        )
        if (
            tuple(sentinel_fold["positive_candidates"]) != first_joint
            or sentinel_fold["baseline"] != first.get("baseline")
            or sentinel_fold["endpoint_binding"] != first.get("endpoint_binding")
        ):
            raise ValueError("sentinel does not bind the first expanded fold")
        preview_best = select_best_positive_microstep(
            baseline_objective=float(
                _mapping_field(first, "baseline", label="first fold baseline").get(
                    "objective"
                )
            ),
            candidates=first_candidates,  # type: ignore[arg-type]
        )
        sentinel_best = sentinel_decision.get("selected_positive")
        same_winner = bool(
            isinstance(preview_best, Mapping)
            and isinstance(sentinel_best, Mapping)
            and _candidate_path(preview_best) == sentinel_best.get("path")
            and _candidate_alpha(preview_best) == sentinel_best.get("alpha")
        )
        if sentinel.get("expanded_winner_reused_sentinel_mirror") is not same_winner:
            raise ValueError("conditional sentinel mirror reuse flag drifted")
        for fold in selected_folds:
            extra = int(fold.get("held_family_id") == sentinel_held and not same_winner)
            validated, positives, mirrors = _recompute_fold(
                fold,
                expanded=True,
                extra_prior_mirror_count=extra,
                pinned_v19_bundle=authenticated_v19_bundles.get(
                    str(fold.get("held_family_id")), {}
                ),
            )
            if fold.get("held_family_id") == sentinel_held and same_winner:
                if validated["matched_negative"] != sentinel_fold["matched_negative"]:
                    raise ValueError("reused sentinel mirror artifact differs")
            validated_folds.append(validated)
            positive_count += positives
            mirror_count += mirrors
    else:
        if len(selected_folds) != 1 or selected_folds[0].get(
            "held_family_id"
        ) != sentinel_held:
            raise ValueError("failed sentinel did not short-circuit after its fold")
        if dict(selected_folds[0]) != dict(sentinel):
            raise ValueError("failed sentinel fold/report payload differs")
        validated_folds.append(sentinel_fold)
        positive_count = sentinel_positive_count
        mirror_count = sentinel_mirror_count

    expected_work = _work_accounting(
        tested_fold_count=len(validated_folds),
        positive_candidate_count=positive_count,
        mirror_candidate_count=mirror_count,
    )
    if dict(work) != expected_work:
        raise ValueError("V20a supplied work ledger differs from recomputation")
    expected_integrity = {
        "immutable_inputs_validated": True,
        "v19_prerequisite_exact": True,
        "all_tested_fold_endpoint_bindings_exact": True,
        "held_teacher_rows_capability_excluded": True,
        "held_score_row_count": 0,
        "held_scoring_performed": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "provider_sidecar_written": False,
    }
    if dict(integrity) != expected_integrity:
        raise ValueError("V20a integrity receipt differs from recomputation")
    pinned_receipts = _mapping_field(
        validated_prerequisite,
        "fisher_optimization_receipt_sha256s",
        label="validated V19 Fisher receipts",
    )
    for fold in validated_folds:
        held = str(fold["held_family_id"])
        endpoint = fold["endpoint_binding"]
        if (
            not isinstance(endpoint, Mapping)
            or endpoint.get("pinned_v19_optimization_receipt_sha256")
            != pinned_receipts.get(held)
        ):
            raise ValueError("fold endpoint is not bound to the pinned V19 receipt")

    baseline_values = tuple(
        float(fold["baseline"]["objective"])  # type: ignore[index]
        for fold in validated_folds
    )
    selected_values = tuple(
        (
            float(fold["qualification"]["selected_positive"]["objective"])  # type: ignore[index]
            if isinstance(
                fold["qualification"]["selected_positive"], Mapping  # type: ignore[index]
            )
            else float(fold["baseline"]["objective"])  # type: ignore[index]
        )
        for fold in validated_folds
    )
    macro_baseline = math.fsum(baseline_values) / len(baseline_values)
    macro_selected = math.fsum(selected_values) / len(selected_values)
    macro_relative = (
        (macro_baseline - macro_selected) / macro_baseline
        if macro_baseline > 0.0
        else 0.0
    )
    classification = _classification(
        sentinel=sentinel_fold,
        folds=validated_folds,
        macro_relative_improvement=macro_relative,
    )
    passed = classification == "finite_microstep_preflight_passed_for_nested_validation"
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "experiment_stage": "v20a",
        "scientific_status": "opened_A16_signed_fit_only_finite_microstep_preflight",
        "artifact": {
            "path": destination.as_posix(),
            "write_once": True,
            "file_mode": "0600",
            "scalar_and_hash_only": True,
            "provider_tensor_sidecar": False,
        },
        "panel": dict(panel),
        "bridge_binding_sha256": _v19._sha256_identifier(
            bridge_binding_sha256, label="V20a bridge binding"
        ),
        "prerequisite": validated_prerequisite,
        "fixed_protocol": {
            **_FIXED_PROTOCOL,
            "microstep_protocol_sha256": _MICROSTEP_PROTOCOL_SHA256,
        },
        "fit_collection": validated_collection,
        "sentinel": sentinel_fold,
        "folds": tuple(validated_folds),
        "fit_macro": {
            "checkpoint_zero_teacher_kl": macro_baseline,
            "selected_positive_teacher_kl": macro_selected,
            "relative_improvement": macro_relative,
            "relative_improvement_min": _MATERIALITY_THRESHOLD,
            "relative_improvement_at_least_threshold": macro_relative >= _MATERIALITY_THRESHOLD,
            "family_equal_across_tested_folds": True,
        },
        "work_accounting": expected_work,
        "integrity": expected_integrity,
        "classification": classification,
        "passed": passed,
        "nested_v20b_authorized": passed,
        "candidate": None,
        "provider_sidecar": None,
        "fresh_guard_authorized": False,
        "calibration_b_authorized": False,
        "held_fidelity_claim": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "success_authorizes": (
            "nested_family_disjoint_V20b_validation_only"
            if passed
            else "no_nested_validation"
        ),
    }
    _v14._scalar_report(report)
    return report


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V20a report")
    if report.get("candidate") is not None or report.get("provider_sidecar") is not None:
        raise ValueError("V20a cannot publish a candidate or provider sidecar")
    reservation = _v14._reserve_outputs((destination,))
    stage: Path | None = None
    try:
        _v14._scalar_report(report)
        report["report_sha256"] = _v14._sha256(report, domain=_REPORT_DOMAIN)
        stage = _v14._stage_json(report, destination)
        if stat.S_IMODE(stage.stat().st_mode) != 0o600:
            raise RuntimeError("staged V20a report mode differs from 0600")
        reservation.publish((stage,))
        if stat.S_IMODE(destination.stat().st_mode) != 0o600:
            raise RuntimeError("published V20a report mode differs from 0600")
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _v14._file_sha256(destination),
                "file_bytes": destination.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def _report_ready_checkpoint_path(output: Path | str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    return destination.with_name(f"{destination.stem}.report-ready.json")


def _report_ready_checkpoint_lock_path(output: Path | str) -> Path:
    checkpoint = _report_ready_checkpoint_path(output)
    return checkpoint.with_name(f".{checkpoint.name}.publish.lock")


def _validate_report_inputs(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _REPORT_INPUT_KEYS:
        raise ValueError("V20a report-ready input fields differ")
    if any(not isinstance(value.get(key), Mapping) for key in (
        "panel",
        "prerequisite",
        "fit_collection",
        "sentinel",
        "work",
        "integrity",
    )):
        raise TypeError("V20a report-ready mapping input is missing")
    folds = value.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise TypeError("V20a report-ready folds are missing")
    _v19._sha256_identifier(
        value.get("bridge_binding_sha256"),
        label="V20a report-ready bridge binding",
    )
    return dict(value)


def _publish_report_ready_checkpoint(
    *,
    output: Path | str,
    report_inputs: Mapping[str, object],
) -> Path:
    """Publish immutable scalar/hash-only finalizer inputs before final build."""

    destination = _validate_output(output)
    checkpoint = _report_ready_checkpoint_path(destination)
    _validate_output(checkpoint)
    if destination.exists():
        raise FileExistsError("V20a final report already exists")
    if checkpoint.exists():
        raise FileExistsError("refusing to overwrite V20a report-ready checkpoint")
    inputs = _validate_report_inputs(report_inputs)
    payload: dict[str, object] = {
        "schema": _REPORT_READY_CHECKPOINT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": destination.as_posix(),
        "microstep_protocol_sha256": _MICROSTEP_PROTOCOL_SHA256,
        "v19_report_sha256": _V19_LOGICAL_SHA256,
        "v19_file_sha256": _V19_FILE_SHA256,
        "candidate": None,
        "provider_sidecar": None,
        "report_inputs": inputs,
    }
    _v14._scalar_report(payload)
    payload["checkpoint_sha256"] = _v14._sha256(
        payload,
        domain=_REPORT_READY_CHECKPOINT_DOMAIN,
    )
    reservation = _v14._reserve_outputs((checkpoint,))
    stage: Path | None = None
    try:
        stage = _v14._stage_json(payload, checkpoint)
        stage_stat = stage.lstat()
        if (
            not stat.S_ISREG(stage_stat.st_mode)
            or stat.S_IMODE(stage_stat.st_mode) != 0o600
            or stage_stat.st_nlink != 1
            or stage_stat.st_uid != os.getuid()
        ):
            raise RuntimeError("staged V20a report-ready checkpoint is unsafe")
        reservation.publish((stage,))
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)
    published = checkpoint.lstat()
    if (
        not stat.S_ISREG(published.st_mode)
        or checkpoint.is_symlink()
        or stat.S_IMODE(published.st_mode) != 0o600
        or published.st_nlink != 1
        or published.st_uid != os.getuid()
    ):
        raise RuntimeError("published V20a report-ready checkpoint is unsafe")
    return checkpoint


def _load_report_ready_checkpoint(
    *,
    output: Path | str,
) -> dict[str, object]:
    """Strict-load one report-ready checkpoint for no-model finalization."""

    destination = _validate_output(output)
    checkpoint = _report_ready_checkpoint_path(destination)
    if not checkpoint.exists():
        raise FileNotFoundError("V20a report-ready checkpoint is absent")
    selected = checkpoint.lstat()
    if (
        not stat.S_ISREG(selected.st_mode)
        or checkpoint.is_symlink()
        or stat.S_IMODE(selected.st_mode) != 0o600
        or selected.st_nlink != 1
        or selected.st_uid != os.getuid()
    ):
        raise RuntimeError("V20a report-ready checkpoint is unsafe")
    try:
        value = json.loads(checkpoint.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("V20a report-ready checkpoint JSON is invalid") from error
    if not isinstance(value, Mapping) or set(value) != _REPORT_READY_CHECKPOINT_KEYS:
        raise ValueError("V20a report-ready checkpoint fields differ")
    payload = {
        key: item for key, item in value.items() if key != "checkpoint_sha256"
    }
    checkpoint_sha256 = _v19._sha256_identifier(
        value.get("checkpoint_sha256"),
        label="V20a report-ready checkpoint",
    )
    if _v14._sha256(
        payload,
        domain=_REPORT_READY_CHECKPOINT_DOMAIN,
    ) != checkpoint_sha256:
        raise ValueError("V20a report-ready checkpoint hash drifted")
    if (
        value.get("schema") != _REPORT_READY_CHECKPOINT_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("microstep_protocol_sha256")
        != _MICROSTEP_PROTOCOL_SHA256
        or value.get("v19_report_sha256") != _V19_LOGICAL_SHA256
        or value.get("v19_file_sha256") != _V19_FILE_SHA256
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
        or value.get("target_output") != destination.as_posix()
    ):
        raise ValueError("V20a report-ready checkpoint authority differs")
    _v14._scalar_report(payload)
    return _validate_report_inputs(value.get("report_inputs"))


def _build_report_from_inputs(
    *,
    output: Path | str,
    report_inputs: Mapping[str, object],
) -> dict[str, object]:
    inputs = _validate_report_inputs(report_inputs)
    return build_finite_microstep_preflight_report(
        artifact_path=_validate_output(output),
        **inputs,  # type: ignore[arg-type]
    )


# Live execution helpers and runner are defined below.  Keeping the report and
# decision contract above independent makes the scientific gates unit-testable
# without loading Gemma.


@dataclass(slots=True)
class _FoldWorkspace:
    held_family_id: str
    training_records: tuple[object, ...]
    held_records: tuple[object, ...]
    capability: object
    base_provider: object
    proposal_provider: object
    baseline: dict[str, object]
    endpoint_binding: dict[str, object]
    ownership_receipt: dict[str, object]


def _parameter_sha256s(provider: object) -> dict[str, str]:
    # Keep checkpoint-zero and candidate hashes in the exact same domain as
    # the authenticated core microstep receipt.  V14 runtime hashes use a
    # different domain and therefore cannot be compared to receipt hashes.
    result = fisher_finite_microstep_selected_tensor_sha256s(provider)
    if len(result) != 4:
        raise RuntimeError("V20a provider parameter hash geometry differs")
    return result


def _runtime_flags(provider: object, records: Sequence[object]) -> dict[str, object]:
    parameters = tuple(
        getattr(provider, name)
        for name in (
            "direction_left",
            "direction_right",
            "pedal_weight",
            "pedal_bias",
        )
    )
    finite = all(
        isinstance(value, Tensor) and bool(torch.isfinite(value).all())
        for value in parameters
    )
    runtime = _v19._held_runtime_diagnostics(
        provider,
        tuple(getattr(record, "sequence") for record in records),
    )
    product = parameters[0].detach().to(device="cpu", dtype=torch.float64) @ (
        parameters[1].detach().to(device="cpu", dtype=torch.float64)
    )
    numerical_rank = int(torch.linalg.matrix_rank(product))
    return {
        "finite": finite,
        "pointwise_trust_passed": runtime.get("pointwise_trust_passed") is True,
        "rank_is_16": numerical_rank == _CONDITIONAL_RANK,
        "numerical_conditional_rank": numerical_rank,
        "runtime_receipt_sha256": runtime["receipt_sha256"],
        "max_bounded_direction_to_parent_norm_ratio": runtime[
            "max_bounded_direction_to_parent_norm_ratio"
        ],
        "max_emitted_delta_to_parent_norm_ratio": runtime[
            "max_emitted_delta_to_parent_norm_ratio"
        ],
    }


def _verified_model_inputs(
    context: object,
    record: object,
) -> tuple[object, Tensor, Tensor]:
    model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
        getattr(context, "tokenize"),
        getattr(record, "example"),
    )
    if (
        gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        != getattr(record, "model_inputs_sha256")
        or _v14._tensor_sha256(supervised_indices)
        != getattr(record, "supervised_indices_sha256")
        or _v14._tensor_sha256(supervised_targets)
        != getattr(record, "supervised_targets_sha256")
    ):
        raise RuntimeError("V20a fit retokenization drifted")
    return model_inputs, supervised_indices, supervised_targets


def _execution_hashes_and_score(
    *,
    execution: object,
    record: object,
    teacher: Tensor,
    supervised_indices: Tensor,
    provider_artifact_sha256: str,
) -> tuple[float, str, str]:
    candidate_h4 = getattr(execution, "candidate_h4", None)
    logits = getattr(execution, "logits", None)
    if (
        not isinstance(candidate_h4, Tensor)
        or candidate_h4.ndim != 3
        or candidate_h4.shape[0] != 1
        or not isinstance(logits, Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not bool(torch.isfinite(candidate_h4).all())
        or getattr(execution, "h4_head_sha256", None)
        != provider_artifact_sha256
    ):
        raise RuntimeError("V20a finite execution authority differs")
    sequence = getattr(record, "sequence")
    support = sequence.support_mask
    candidate_cpu = candidate_h4[0].detach().to(device="cpu")
    # Training sequences canonicalize captured H4 rows to float64, while the
    # live Gemma boundary remains in its native dtype.  Promote the live rows
    # before a bitwise value check so dtype normalization alone cannot look
    # like an off-support write.
    cached_base = sequence.base_h4.to(dtype=torch.float64)
    if not _v14._bitwise_equal(
        candidate_cpu.to(dtype=torch.float64)[~support],
        cached_base[~support],
    ):
        raise RuntimeError("V20a provider escaped complete-H4 causal support")
    selected_logits = _v14._select_sequence_rows(logits, supervised_indices)
    if selected_logits.ndim != 2 or int(selected_logits.shape[1]) != _EXPECTED_VOCABULARY:
        raise RuntimeError("V20a did not score the full Gemma vocabulary")
    score = float(_v19.exact_float64_teacher_kl(teacher, selected_logits))
    return (
        score,
        _v14._tensor_sha256(candidate_h4),
        _v14._tensor_sha256(selected_logits),
    )


def _checkpoint_zero_and_first_adam(
    context: object,
    records: Sequence[object],
    capability: object,
    start_provider: object,
    *,
    held_family_id: str,
    pinned_bundle: Mapping[str, object],
) -> tuple[object, object, dict[str, object], dict[str, object]]:
    """Reconstruct and authenticate V19 checkpoints zero and one."""

    ordered = tuple(
        sorted(
            records,
            key=lambda value: (
                value.sequence.family_id,
                value.sequence.example_id,
            ),
        )
    )
    if len(ordered) != _EXPECTED_FIT_PROMPTS:
        raise RuntimeError("V20a checkpoint fit-row geometry differs")
    state0 = _v19._initial_joint_state(start_provider)
    base_provider = _v19._provisional_provider(
        start_provider,
        state0,
        held_family_id=held_family_id,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=0,
    )
    prompt_scores: dict[str, float] = {}
    prompt_gradients: list[object] = []
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    for record in ordered:
        model_inputs, supervised_indices, _targets = _verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id,
            family_id=record.sequence.family_id,
        )
        objective, captured = _v19._teacher_kl_objective(
            teacher, supervised_indices
        )
        execution, h4_gradient = context.bridge.execute_h4_vjp(
            context.adapter,
            model_inputs,
            objective=objective,
            h4_head=base_provider,
        )
        if len(captured) != 1:
            raise RuntimeError("V20a checkpoint-zero objective capture differs")
        score, h4_hash, logits_hash = _execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=base_provider.artifact_sha256,
        )
        if score != captured[0]:
            raise RuntimeError("V20a checkpoint-zero finite score replay drifted")
        prompt_scores[record.sequence.example_id] = score
        h4_hashes[record.sequence.example_id] = h4_hash
        logits_hashes[record.sequence.example_id] = logits_hash
        prompt_gradients.append(
            _v19._local_ste_parameter_gradients(
                base_provider,
                state0,
                record.sequence,
                h4_gradient,
            )
        )
        del model_inputs, teacher, execution, h4_gradient
    baseline_objective, family_objectives = _v19._family_equal_mean(
        prompt_scores,
        ordered,
    )
    zero = _v19._zero_state(state0)
    state1, _moments = _v19._adam_step(
        state0,
        _v19._mean_gradient(prompt_gradients),
        _v19._AdamMoments(first=zero, second=zero, step=0),
    )
    proposal_provider = _v19._provisional_provider(
        start_provider,
        state1,
        held_family_id=held_family_id,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=1,
    )
    _v19._validate_joint_provider(
        base_provider,
        start_provider=start_provider,
        pedal_mode="conditional",
        expected_family_count=_EXPECTED_FAMILIES - 1,
    )
    _v19._validate_joint_provider(
        proposal_provider,
        start_provider=start_provider,
        pedal_mode="conditional",
        expected_family_count=_EXPECTED_FAMILIES - 1,
    )

    pinned_receipt = _mapping_field(
        pinned_bundle,
        "optimization_receipt",
        label="pinned V19 optimization receipt",
    )
    pinned_provider_artifacts = pinned_receipt[
        "checkpoint_provider_artifact_sha256s"
    ]
    pinned_states = pinned_receipt["checkpoint_state_receipts"]
    pinned_scores = pinned_receipt["checkpoint_scores"]
    pinned_family_scores = pinned_receipt["checkpoint_family_scores"]
    current_training_records = tuple(record.receipt_sha256 for record in ordered)
    current_training_sequences = tuple(start_provider.fit_sequence_sha256s)
    current_training_families = tuple(start_provider.fit_family_ids)
    checks = {
        "parent_provider_exactly_reconstructed": (
            base_provider.parent_provider.artifact_sha256
            == pinned_bundle["parent_provider_artifact_sha256"]
        ),
        "start_provider_exactly_reconstructed": (
            start_provider.artifact_sha256
            == pinned_bundle["start_provider_artifact_sha256"]
            == pinned_receipt["start_provider_artifact_sha256"]
        ),
        "checkpoint_zero_provider_exactly_reconstructed": (
            base_provider.artifact_sha256 == pinned_provider_artifacts[0]
        ),
        "first_adam_provider_exactly_reconstructed": (
            proposal_provider.artifact_sha256 == pinned_provider_artifacts[1]
        ),
        "checkpoint_zero_state_exactly_reconstructed": (
            state0.receipt() == pinned_states[0]
        ),
        "first_adam_state_exactly_reconstructed": (
            state1.receipt() == pinned_states[1]
        ),
        "checkpoint_zero_scalar_exactly_reconstructed": (
            baseline_objective == pinned_scores[0]
        ),
        "checkpoint_zero_family_scores_exactly_reconstructed": (
            family_objectives == pinned_family_scores[0]
        ),
        "training_record_receipts_exactly_reconstructed": (
            current_training_records
            == tuple(pinned_receipt["training_record_receipt_sha256s"])
        ),
        "training_sequence_receipts_exactly_reconstructed": (
            current_training_sequences
            == tuple(pinned_receipt["training_sequence_sha256s"])
        ),
        "training_family_ids_exactly_reconstructed": (
            current_training_families
            == tuple(pinned_receipt["training_family_ids"])
        ),
        "teacher_capability_artifact_exactly_reconstructed": (
            capability.artifact_sha256
            == pinned_receipt["capability_receipt"]["artifact_sha256"]
        ),
    }
    if any(value is not True for value in checks.values()):
        failed = tuple(key for key, value in checks.items() if value is not True)
        raise RuntimeError(f"V20a V19 endpoint reconstruction failed: {failed}")
    baseline = {
        "objective": baseline_objective,
        "family_objectives": family_objectives,
        "provider_artifact_sha256": base_provider.artifact_sha256,
        "parameter_sha256s": _parameter_sha256s(base_provider),
        "post_cast_h4_sha256s": h4_hashes,
        "supervised_full_vocab_logits_sha256s": logits_hashes,
        **_runtime_flags(base_provider, ordered),
    }
    endpoint_payload: dict[str, object] = {
        "held_family_id": held_family_id,
        "pinned_v19_optimization_receipt_sha256": pinned_bundle[
            "optimization_receipt_sha256"
        ],
        "parent_provider_artifact_sha256": (
            base_provider.parent_provider.artifact_sha256
        ),
        "pinned_parent_provider_artifact_sha256": pinned_bundle[
            "parent_provider_artifact_sha256"
        ],
        "start_provider_artifact_sha256": start_provider.artifact_sha256,
        "pinned_start_provider_artifact_sha256": pinned_bundle[
            "start_provider_artifact_sha256"
        ],
        "base_provider_artifact_sha256": base_provider.artifact_sha256,
        "pinned_base_provider_artifact_sha256": pinned_provider_artifacts[0],
        "proposal_provider_artifact_sha256": proposal_provider.artifact_sha256,
        "pinned_proposal_provider_artifact_sha256": pinned_provider_artifacts[1],
        "checkpoint_zero_state_sha256s": state0.receipt(),
        "pinned_checkpoint_zero_state_sha256s": pinned_states[0],
        "first_adam_state_sha256s": state1.receipt(),
        "pinned_first_adam_state_sha256s": pinned_states[1],
        "checkpoint_zero_objective": baseline_objective,
        "pinned_checkpoint_zero_objective": pinned_scores[0],
        "checkpoint_zero_family_objectives": family_objectives,
        "pinned_checkpoint_zero_family_objectives": pinned_family_scores[0],
        "training_family_ids": current_training_families,
        "training_sequence_sha256s": current_training_sequences,
        "training_record_receipt_sha256s": current_training_records,
        "checks": checks,
        "endpoint_binding_verified_before_alpha_evaluation": True,
    }
    endpoint_payload["receipt_sha256"] = _v14._sha256(
        endpoint_payload,
        domain=_ENDPOINT_DOMAIN,
    )
    # No caller can receive the endpoints until every exact V19 binding above
    # has passed, which makes the phase ordering structural rather than a flag.
    return base_provider, proposal_provider, baseline, endpoint_payload


def _prepare_fold_workspace(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    held_family_id: str,
    pinned_bundle: Mapping[str, object],
) -> _FoldWorkspace:
    held = _v14._identifier(held_family_id, label="V20a held family")
    training_records = tuple(
        record for record in records if record.sequence.family_id != held
    )
    held_records = tuple(
        record for record in records if record.sequence.family_id == held
    )
    if (
        len(training_records) != _EXPECTED_FIT_PROMPTS
        or len(held_records) != _EXPECTED_HELD_PROMPTS
        or any(record.sequence.family_id == held for record in training_records)
    ):
        raise RuntimeError("V20a fold ownership geometry differs")
    sequences = tuple(record.sequence for record in training_records)
    parent = _v19._fit_parent(
        sequences,
        bridge_binding_sha256=context.bridge.bridge_binding_sha256,
    )
    _v18._validate_parent(parent, expected_fit_family_count=_EXPECTED_FAMILIES - 1)
    start = _v19._fit_v18_start(
        sequences,
        parent=parent,
        coordinate_objective="reverse_vjp_fisher",
    )
    _v18._validate_child(
        start,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode="conditional",
        expected_parent_artifact_sha256=parent.artifact_sha256,
        expected_fit_family_count=_EXPECTED_FAMILIES - 1,
    )
    authorized = tuple(record.sequence.example_id for record in training_records)
    capability = teacher_vault.capability(authorized, held_family_id=held)
    base, proposal, baseline, endpoint = _checkpoint_zero_and_first_adam(
        context,
        training_records,
        capability,
        start,
        held_family_id=held,
        pinned_bundle=pinned_bundle,
    )
    pinned_optimization = _mapping_field(
        pinned_bundle,
        "optimization_receipt",
        label="pinned V19 optimization receipt",
    )
    pinned_selected = _v19.refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=base.direction_left,
        direction_right=base.direction_right,
        pedal_weight=base.pedal_weight,
        pedal_bias=base.pedal_bias,
        fit_protocol_sha256=str(pinned_optimization["fit_protocol_sha256"]),
        fit_evidence_sha256=str(pinned_optimization["fit_evidence_sha256"]),
        pedal_mode="conditional",
    )
    if (
        pinned_selected.artifact_sha256
        != pinned_optimization["selected_provider_artifact_sha256"]
    ):
        raise RuntimeError("V20a pinned V19 selected provider did not reconstruct")
    ownership = _v19._fold_ownership_receipt(
        pinned_selected,
        held_family_id=held,
        held_sequences=tuple(record.sequence for record in held_records),
    )
    pinned_ownership = _mapping_field(
        pinned_bundle,
        "ownership_receipt",
        label="pinned V19 ownership receipt",
    )
    # JSON reloads V19's tuple-valued receipt fields as lists.  The receipt
    # hash is over canonical JSON, so it is the representation-independent
    # exact comparison between the freshly reconstructed tuple payload and
    # the authenticated on-disk payload.
    ownership_sha256 = _ownership_receipt_sha256(
        ownership,
        label="reconstructed V19 fold ownership",
    )
    pinned_ownership_sha256 = _ownership_receipt_sha256(
        pinned_ownership,
        label="pinned V19 fold ownership",
    )
    if ownership_sha256 != pinned_ownership_sha256:
        raise RuntimeError("V20a pinned V19 fold ownership did not reconstruct")
    ownership = dict(pinned_ownership)
    return _FoldWorkspace(
        held_family_id=held,
        training_records=training_records,
        held_records=held_records,
        capability=capability,
        base_provider=base,
        proposal_provider=proposal,
        baseline=baseline,
        endpoint_binding=endpoint,
        ownership_receipt=ownership,
    )


def _candidate_evidence_sha256(
    workspace: _FoldWorkspace,
    *,
    path: str,
    alpha: float,
) -> str:
    return _v14._sha256(
        {
            "held_family_id": workspace.held_family_id,
            "path": path,
            "alpha": alpha,
            "base_provider_artifact_sha256": (
                workspace.base_provider.artifact_sha256
            ),
            "proposal_provider_artifact_sha256": (
                workspace.proposal_provider.artifact_sha256
            ),
            "endpoint_binding_receipt_sha256": workspace.endpoint_binding[
                "receipt_sha256"
            ],
            "training_record_receipt_sha256s": tuple(
                record.receipt_sha256 for record in workspace.training_records
            ),
            "held_rows_scored": False,
        },
        domain=_EVIDENCE_DOMAIN,
    )


def _evaluate_microstep_candidate(
    context: object,
    workspace: _FoldWorkspace,
    *,
    path: str,
    alpha: float,
) -> dict[str, object]:
    evidence = _candidate_evidence_sha256(
        workspace,
        path=path,
        alpha=alpha,
    )
    result = build_autonomous_complete_h4_fisher_finite_microstep(
        workspace.base_provider,
        workspace.proposal_provider,
        microstep_path=path,
        alpha=alpha,
        microstep_protocol_sha256=_MICROSTEP_PROTOCOL_SHA256,
        microstep_evidence_sha256=evidence,
    )
    prompt_scores: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    for record in sorted(
        workspace.training_records,
        key=lambda value: (
            value.sequence.family_id,
            value.sequence.example_id,
        ),
    ):
        model_inputs, supervised_indices, _targets = _verified_model_inputs(
            context, record
        )
        teacher = workspace.capability.get(
            record.sequence.example_id,
            family_id=record.sequence.family_id,
        )
        execution = context.bridge.execute(
            context.adapter,
            model_inputs,
            h4_head=result.provider,
        )
        score, h4_hash, logits_hash = _execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=result.provider.artifact_sha256,
        )
        prompt_scores[record.sequence.example_id] = score
        h4_hashes[record.sequence.example_id] = h4_hash
        logits_hashes[record.sequence.example_id] = logits_hash
        del model_inputs, teacher, execution
    objective, family_objectives = _v19._family_equal_mean(
        prompt_scores,
        workspace.training_records,
    )
    parameter_hashes = _parameter_sha256s(result.provider)
    execution_change = detect_execution_change(
        base_parameter_sha256s=workspace.baseline["parameter_sha256s"],  # type: ignore[arg-type]
        candidate_parameter_sha256s=parameter_hashes,
        base_h4_sha256s=workspace.baseline["post_cast_h4_sha256s"],  # type: ignore[arg-type]
        candidate_h4_sha256s=h4_hashes,
        base_logits_sha256s=workspace.baseline[
            "supervised_full_vocab_logits_sha256s"
        ],  # type: ignore[arg-type]
        candidate_logits_sha256s=logits_hashes,
    )
    receipt = result.receipt.metadata()
    row = {
        "path": path,
        "alpha": alpha,
        "objective": objective,
        "family_objectives": family_objectives,
        "execution_changed": execution_change["execution_changed"],
        "execution_change": execution_change,
        "base_provider_artifact_sha256": (
            workspace.base_provider.artifact_sha256
        ),
        "proposal_provider_artifact_sha256": (
            workspace.proposal_provider.artifact_sha256
        ),
        "provider_artifact_sha256": result.provider.artifact_sha256,
        "microstep_artifact_sha256": result.artifact_sha256,
        "microstep_receipt_sha256": result.receipt.artifact_sha256,
        "microstep_evidence_sha256": evidence,
        "microstep_receipt": receipt,
        "parameter_sha256s": parameter_hashes,
        "post_cast_h4_sha256s": h4_hashes,
        "supervised_full_vocab_logits_sha256s": logits_hashes,
        **_runtime_flags(result.provider, workspace.training_records),
    }
    _validate_candidate_authentication(
        row,
        baseline=workspace.baseline,
        signed=alpha < 0.0,
    )
    return row


def _evaluate_positive_grid(
    context: object,
    workspace: _FoldWorkspace,
    *,
    paths: Sequence[str],
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        _evaluate_microstep_candidate(
            context,
            workspace,
            path=path,
            alpha=alpha,
        )
        for path in paths
        for alpha in POSITIVE_ALPHAS
    )
    _validate_exact_positive_grid(rows, expected_paths=paths)
    return rows


def _evaluate_matched_negative(
    context: object,
    workspace: _FoldWorkspace,
    selected_positive: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if selected_positive is None:
        return None
    return _evaluate_microstep_candidate(
        context,
        workspace,
        path=_candidate_path(selected_positive),
        alpha=-_candidate_alpha(selected_positive),
    )


def _fold_payload(
    workspace: _FoldWorkspace,
    *,
    positives: Sequence[Mapping[str, object]],
    matched_negative: Mapping[str, object] | None,
    qualification: Mapping[str, object],
    expanded_winner_reused_sentinel_mirror: bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "held_family_id": workspace.held_family_id,
        "baseline": workspace.baseline,
        "positive_candidates": tuple(positives),
        "matched_negative": (
            None if matched_negative is None else dict(matched_negative)
        ),
        "qualification": dict(qualification),
        "capability_receipt": workspace.capability.receipt(),
        "ownership_receipt": workspace.ownership_receipt,
        "endpoint_binding": workspace.endpoint_binding,
        "held_scoring_performed": False,
    }
    if expanded_winner_reused_sentinel_mirror is not None:
        payload["expanded_winner_reused_sentinel_mirror"] = (
            expanded_winner_reused_sentinel_mirror
        )
    return payload


def run_gemma3_l3_l4_complete_h4_finite_microstep_preflight(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the fixed signed fit-only V20a preflight."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V20a report")
    prerequisite, pinned_bundles = _load_authenticated_v19_artifact()
    checkpoint = _report_ready_checkpoint_path(destination)
    if checkpoint.exists():
        resumed_inputs = _load_report_ready_checkpoint(output=destination)
        resumed_report = _build_report_from_inputs(
            output=destination,
            report_inputs=resumed_inputs,
        )
        return _publish(resumed_report, output=destination)
    checkpoint_lock = _report_ready_checkpoint_lock_path(destination)
    if checkpoint_lock.exists():
        raise FileExistsError("V20a report-ready checkpoint is reserved")
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records, teacher_vault = _v19._collect_fit_records_and_teacher_vault(
            context
        )
        families = tuple(sorted({record.sequence.family_id for record in records}))
        if (
            len(records) != _EXPECTED_PROMPTS
            or len(families) != _EXPECTED_FAMILIES
            or set(families) != set(pinned_bundles)
        ):
            raise RuntimeError("V20a authenticated A16 family geometry differs")
        fit_collection = {
            "prompt_count": len(records),
            "family_count": len(families),
            "record_receipt_sha256s": {
                record.sequence.example_id: record.receipt_sha256
                for record in records
            },
            "teacher_vault_receipt": teacher_vault.receipt(),
            "held_teacher_rows_cached": True,
            "held_teacher_rows_scored": False,
            "full_native_logits_transiently_materialized_then_discarded": True,
            "only_supervised_source_rows_cached_in_native_dtype": True,
            "raw_fit_trace_or_teacher_tensor_serialization": False,
        }
        _validate_fit_collection(
            fit_collection,
            authenticated_v19=_mapping_field(
                prerequisite,
                "authenticated_fit_collection",
                label="authenticated V19 fit collection",
            ),
        )
        _validate_panel_and_bridge_authority(
            panel=context.panel_receipt,
            bridge_binding_sha256=context.bridge.bridge_binding_sha256,
            validated_prerequisite=prerequisite,
        )

        sentinel_held = families[0]
        sentinel_workspace = _prepare_fold_workspace(
            context,
            records,
            teacher_vault,
            held_family_id=sentinel_held,
            pinned_bundle=pinned_bundles[sentinel_held],
        )
        sentinel_positives = _evaluate_positive_grid(
            context,
            sentinel_workspace,
            paths=("joint",),
        )
        sentinel_selected = select_best_positive_microstep(
            baseline_objective=float(sentinel_workspace.baseline["objective"]),
            candidates=sentinel_positives,
        )
        sentinel_negative = _evaluate_matched_negative(
            context,
            sentinel_workspace,
            sentinel_selected,
        )
        sentinel_qualification = evaluate_sentinel_decision(
            baseline_objective=float(sentinel_workspace.baseline["objective"]),
            selected_positive=sentinel_selected,
            matched_negative=sentinel_negative,
        )
        sentinel_payload = _fold_payload(
            sentinel_workspace,
            positives=sentinel_positives,
            matched_negative=sentinel_negative,
            qualification=sentinel_qualification,
        )
        positive_candidate_count = len(sentinel_positives)
        mirror_candidate_count = int(sentinel_negative is not None)

        if sentinel_qualification["passed"] is not True:
            fold_payloads = (sentinel_payload,)
            work = _work_accounting(
                tested_fold_count=1,
                positive_candidate_count=positive_candidate_count,
                mirror_candidate_count=mirror_candidate_count,
            )
        else:
            additional = _evaluate_positive_grid(
                context,
                sentinel_workspace,
                paths=("direction_only", "pedal_only"),
            )
            all_first_candidates = tuple(
                sorted(
                    (*additional, *sentinel_positives),
                    key=lambda row: (
                        _PATH_ORDER[_candidate_path(row)],
                        _candidate_alpha(row),
                    ),
                )
            )
            _validate_exact_positive_grid(all_first_candidates)
            first_selected = select_best_positive_microstep(
                baseline_objective=float(
                    sentinel_workspace.baseline["objective"]
                ),
                candidates=all_first_candidates,
            )
            same_winner = bool(
                first_selected is not None
                and sentinel_selected is not None
                and _candidate_path(first_selected)
                == _candidate_path(sentinel_selected)
                and _candidate_alpha(first_selected)
                == _candidate_alpha(sentinel_selected)
            )
            if same_winner:
                first_negative = sentinel_negative
            else:
                first_negative = _evaluate_matched_negative(
                    context,
                    sentinel_workspace,
                    first_selected,
                )
                mirror_candidate_count += int(first_negative is not None)
            first_qualification = evaluate_fold_qualification(
                held_family_id=sentinel_held,
                baseline_objective=float(
                    sentinel_workspace.baseline["objective"]
                ),
                selected_positive=first_selected,
                matched_negative=first_negative,
            )
            sentinel_payload = {
                **sentinel_payload,
                "expanded_winner_reused_sentinel_mirror": same_winner,
            }
            fold_rows: list[dict[str, object]] = [
                _fold_payload(
                    sentinel_workspace,
                    positives=all_first_candidates,
                    matched_negative=first_negative,
                    qualification=first_qualification,
                )
            ]
            positive_candidate_count += len(additional)

            for held in families[1:]:
                workspace = _prepare_fold_workspace(
                    context,
                    records,
                    teacher_vault,
                    held_family_id=held,
                    pinned_bundle=pinned_bundles[held],
                )
                positives = _evaluate_positive_grid(
                    context,
                    workspace,
                    paths=MICROSTEP_PATHS,
                )
                selected = select_best_positive_microstep(
                    baseline_objective=float(workspace.baseline["objective"]),
                    candidates=positives,
                )
                negative = _evaluate_matched_negative(
                    context,
                    workspace,
                    selected,
                )
                qualification = evaluate_fold_qualification(
                    held_family_id=held,
                    baseline_objective=float(workspace.baseline["objective"]),
                    selected_positive=selected,
                    matched_negative=negative,
                )
                fold_rows.append(
                    _fold_payload(
                        workspace,
                        positives=positives,
                        matched_negative=negative,
                        qualification=qualification,
                    )
                )
                positive_candidate_count += len(positives)
                mirror_candidate_count += int(negative is not None)
            fold_payloads = tuple(fold_rows)
            work = _work_accounting(
                tested_fold_count=_EXPECTED_FAMILIES,
                positive_candidate_count=positive_candidate_count,
                mirror_candidate_count=mirror_candidate_count,
            )

        context.validate_immutable_inputs()
        teacher_vault.validate_integrity()
        integrity = {
            "immutable_inputs_validated": True,
            "v19_prerequisite_exact": True,
            "all_tested_fold_endpoint_bindings_exact": True,
            "held_teacher_rows_capability_excluded": True,
            "held_score_row_count": 0,
            "held_scoring_performed": False,
            "guard_opened": False,
            "calibration_b_opened": False,
            "provider_sidecar_written": False,
        }
        report_inputs = {
            "panel": context.panel_receipt,
            "bridge_binding_sha256": context.bridge.bridge_binding_sha256,
            "prerequisite": prerequisite,
            "fit_collection": fit_collection,
            "sentinel": sentinel_payload,
            "folds": fold_payloads,
            "work": work,
            "integrity": integrity,
        }
        _publish_report_ready_checkpoint(
            output=destination,
            report_inputs=report_inputs,
        )
        serialized_inputs = _load_report_ready_checkpoint(output=destination)
        report = _build_report_from_inputs(
            output=destination,
            report_inputs=serialized_inputs,
        )
        return _publish(report, output=destination)
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_finite_microstep_preflight(
        output=arguments.output,
        cache_dir=arguments.cache_dir,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
