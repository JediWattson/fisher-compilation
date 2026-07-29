"""Fused, fail-closed Calibration-B one-shot assessment transaction.

This is the only supported Calibration-B execution path.  It authenticates
the static protocol, runtime, adapter, and locally loaded tokenizer; consumes
the frozen manifest claim; then lazily loads each prompt exactly once while
privately streaming five-pass observations into the evaluator.  Only a scalar
report and immutable terminal receipt escape.

The claim is stored under one fixed per-user host state directory independent
of a checkout or installation.  This prevents another local worktree from
minting a fresh slot, but it is not a cross-machine or cryptographic global
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Literal

from .adapters import Gemma3CausalLMAdapter
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation,
    _load_and_validate_frozen_local_tokenizer,
    _validate_live_adapter,
    _validate_protocol_runtime,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    _evaluate_gemma3_l3_l4_graph_organized_svd_shadow,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
)


TerminalOutcome = Literal["success", "failure"]

__all__ = [
    "Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError",
    "Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError",
    "Gemma3L3L4GraphOrganizedSVDOneShotForeignClaimError",
    "Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError",
    "Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt",
    "Gemma3L3L4GraphOrganizedSVDOneShotTransactionResult",
    "TerminalOutcome",
    "run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_organized_svd_one_shot"
_FORMAT_VERSION = 1
_ROLE = "calibration_b_one_shot"
_CLAIM_STATE = "claimed_before_prompt_materialization"
_LEDGER_NAMESPACE = (
    "gemma3-l3-l4-graph-organized-svd-calibration-b-one-shot"
)
_FROZEN_LEDGER_ROOT = (
    Path.home()
    / ".local"
    / "state"
    / "fisher-graph"
    / _LEDGER_NAMESPACE
)
_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-one-shot-ledger-claim:v1\0"
)
_TERMINAL_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-one-shot-terminal-receipt:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_LEDGER_FILE_BYTES = 1 << 20
_RAW_PROMPT_FIELDS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "raw_prompt",
        "raw_prompts",
        "prompt_bytes",
    }
)
_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "role",
        "state",
        "protocol_sha256",
        "assessment_claim_sha256",
        "manifest_sha256",
        "manifest_example_count",
        "manifest_family_count",
        "manifest_families",
        "runtime_binding_sha256",
        "global_lock_identity",
        "claim_payload_sha256",
    }
)
_TERMINAL_COMMON_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "role",
        "state",
        "outcome",
        "manifest_sha256",
        "claim_file_sha256",
        "terminal_receipt_sha256",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "protocol_sha256",
        "assessment_claim_sha256",
        "assessment_claim_identity",
        "manifest",
        "scope",
        "calibration_a_development_evidence",
        "all_on",
        "routed",
        "authorization",
    }
)
_ALL_ON_FIELDS = frozenset(
    {
        "arm",
        "observation_count",
        "behavioral",
        "behavioral_scope",
        "boundary",
        "projection_capacity",
        "carrier_completeness",
        "passed",
    }
)
_BEHAVIORAL_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "semantics",
        "manifest",
        "thresholds",
        "aggregate",
        "per_prompt",
        "family_summary",
        "gates",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "target_modal_width",
        "valid_target_rows",
        "affected_target_rows",
        "valid_target_coverage",
        "pooled_target_modal_relative_error",
        "pooled_target_modal_cosine",
        "worst_family_target_modal_relative_error",
        "worst_family_target_modal_cosine",
        "minimum_family_source_modal_signal_l2_norm",
        "thresholds",
        "gates",
        "family_metrics",
        "per_example",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "target_modal_width",
        "target_full_width",
        "pooled_full_width_delta_relative_error",
        "pooled_full_width_delta_cosine",
        "worst_family_full_width_delta_relative_error",
        "worst_family_full_width_delta_cosine",
        "minimum_family_source_full_width_signal_l2_norm",
        "thresholds",
        "behavioral",
        "gates",
    }
)


class Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(RuntimeError):
    """A persisted ledger record is malformed, noncanonical, or changed."""


class Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError(
    FileExistsError
):
    """The frozen manifest role has already been consumed."""


class Gemma3L3L4GraphOrganizedSVDOneShotForeignClaimError(
    Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError
):
    """The role was consumed by a different protocol/runtime identity."""


class Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError(
    FileExistsError
):
    """The one-shot claim already has an immutable terminal outcome."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contains_raw_prompt_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key.lower() in _RAW_PROMPT_FIELDS
            or _contains_raw_prompt_field(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_raw_prompt_field(nested) for nested in value)
    return False


def _canonical_prompt_free_report(
    report: object,
) -> tuple[dict[str, object], bytes]:
    if not isinstance(report, Mapping):
        raise TypeError("one-shot evaluation report must be a mapping")
    try:
        encoded = _canonical_json_bytes(report)
        canonical = json.loads(encoded.decode("ascii"))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError(
            "one-shot evaluation report must be canonical JSON data"
        ) from error
    if not isinstance(canonical, dict):
        raise TypeError("one-shot evaluation report must be a JSON object")
    if _contains_raw_prompt_field(canonical):
        raise ValueError(
            "one-shot evaluation report must not contain raw prompt fields"
        )
    return canonical, encoded


def _sanitized_exception_sha256(error: Exception) -> str:
    return _file_sha256(
        _canonical_json_bytes(
            {
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _frozen_ledger_paths(manifest_sha256: str) -> tuple[Path, Path]:
    manifest = _require_sha256(
        manifest_sha256,
        label="ledger manifest",
    )
    return (
        _FROZEN_LEDGER_ROOT / f"{manifest}.claim.json",
        _FROZEN_LEDGER_ROOT / f"{manifest}.terminal.json",
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _expected_assessment_claim_identity(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
) -> dict[str, object]:
    metadata = protocol.metadata()
    model = _mapping(metadata.get("model"), label="protocol model")
    tokenizer = _mapping(
        metadata.get("tokenizer"),
        label="protocol tokenizer",
    )
    candidate = _mapping(
        metadata.get("graph_candidate"),
        label="protocol graph candidate",
    )
    basis = _mapping(
        metadata.get("prompt_blind_basis"),
        label="protocol basis",
    )
    lineage = _mapping(
        metadata.get("upstream_generator_lineage"),
        label="protocol generator lineage",
    )
    corpus = _mapping(metadata.get("corpus"), label="protocol corpus")
    manifest = _mapping(
        corpus.get("calibration_b_manifest"),
        label="protocol Calibration-B manifest",
    )
    runtime = _mapping(
        metadata.get("runtime_binding_contract"),
        label="protocol runtime binding",
    )
    return {
        "schema": (
            "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
            "shadow_protocol.calibration_b_assessment_claim"
        ),
        "format_version": 1,
        "role": _ROLE,
        "protocol_sha256": protocol.artifact_sha256,
        "manifest_sha256": manifest["artifact_sha256"],
        "example_count": manifest["example_count"],
        "family_count": manifest["family_count"],
        "source_model_sha256": model["source_model_sha256"],
        "tokenizer": dict(tokenizer),
        "graph_candidate_tensor_file_sha256": (
            candidate["tensor_file_sha256"]
        ),
        "graph_candidate_artifact_sha256": (
            candidate["logical_artifact_sha256"]
        ),
        "signed_plan_sha256": candidate["deployment_plan_sha256"],
        "basis_tensor_file_sha256": basis["tensor_file_sha256"],
        "basis_payload_sha256": basis["logical_payload_sha256"],
        "graph_basis_artifact_sha256": (
            candidate["graph_basis_artifact_sha256"]
        ),
        "refit_scientific_payload_sha256": (
            lineage["refit_scientific_payload_sha256"]
        ),
        "factorized_live_execution_sha256": (
            candidate["factorized_live_execution_sha256"]
        ),
        "factorized_refit_execution_sha256": (
            candidate["factorized_refit_execution_sha256"]
        ),
        "runtime_binding_sha256": runtime["artifact_sha256"],
        "candidate_independent_manifest": True,
        "subset_independent": True,
    }


def _gate_passed(value: object, *, label: str) -> bool:
    section = _mapping(value, label=label)
    gates = _mapping(section.get("gates"), label=f"{label} gates")
    passed = gates.get("passed")
    if type(passed) is not bool:
        raise ValueError(f"{label} passed gate must be boolean")
    return passed


def _validate_shadow_evaluation_report(
    report: Mapping[str, object],
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    session: "_Gemma3L3L4GraphOrganizedSVDOneShotSession",
) -> None:
    if set(report) != _EVALUATION_FIELDS:
        raise ValueError("one-shot report fields differ from evaluation schema")
    if (
        report["schema"]
        != (
            "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
            "shadow_protocol.evaluation"
        )
        or report["format_version"] != 1
        or report["protocol_sha256"] != session.protocol_sha256
        or (
            report["assessment_claim_sha256"]
            != session.assessment_claim_sha256
        )
    ):
        raise ValueError(
            "one-shot report protocol or assessment identity differs"
        )
    assessment_identity = _mapping(
        report["assessment_claim_identity"],
        label="assessment claim identity",
    )
    if _canonical_json_bytes(assessment_identity) != _canonical_json_bytes(
        _expected_assessment_claim_identity(protocol)
    ):
        raise ValueError(
            "one-shot report assessment claim identity differs"
        )
    manifest = _mapping(report["manifest"], label="report manifest")
    if (
        set(manifest)
        != {
            "role",
            "example_identity",
            "artifact_sha256",
            "example_count",
            "family_count",
            "complete",
            "matches_frozen_role",
            "derivation",
            "prompt_file_opened_by_evaluator",
        }
        or manifest["role"] != _ROLE
        or manifest["example_identity"] != "prompt_sha256_only"
        or manifest["artifact_sha256"] != session.manifest_sha256
        or manifest["example_count"] != 96
        or manifest["family_count"] != 8
        or manifest["complete"] is not True
        or manifest["matches_frozen_role"] is not True
        or manifest["derivation"]
        != (
            "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
            "calibration_b_and_family_file_calibration_b"
        )
        or manifest["prompt_file_opened_by_evaluator"] is not False
    ):
        raise ValueError(
            "one-shot report does not cover the exact frozen manifest"
        )
    all_on = _mapping(report["all_on"], label="all_on evaluation")
    if (
        set(all_on) != _ALL_ON_FIELDS
        or all_on["arm"] != "all_on"
        or all_on["observation_count"] != 96
        or type(all_on["passed"]) is not bool
    ):
        raise ValueError(
            "one-shot report lacks the complete all_on evaluation"
        )
    behavioral = _mapping(
        all_on["behavioral"],
        label="all_on behavioral evaluation",
    )
    behavioral_manifest = _mapping(
        behavioral.get("manifest"),
        label="all_on behavioral manifest",
    )
    behavioral_semantics = _mapping(
        behavioral.get("semantics"),
        label="all_on behavioral semantics",
    )
    if (
        set(behavioral) != _BEHAVIORAL_FIELDS
        or behavioral.get("schema")
        != "fisher_graph.source_authoritative_shadow_fidelity"
        or behavioral.get("format_version") != 1
        or set(behavioral_manifest)
        != {
            "strict_example_membership",
            "strict_family_membership",
            "expected_examples",
            "observed_examples",
            "complete",
            "family_count",
        }
        or (
            behavioral_manifest.get("strict_example_membership")
            is not True
        )
        or (
            behavioral_manifest.get("strict_family_membership")
            is not True
        )
        or behavioral_manifest.get("expected_examples") != 96
        or behavioral_manifest.get("observed_examples") != 96
        or behavioral_manifest.get("family_count") != 8
        or behavioral_manifest.get("complete") is not True
        or set(behavioral_semantics)
        != {
            "execution_mode",
            "authoritative_path",
            "source_outputs_authoritative",
            "candidate_outputs_authoritative",
            "candidate_logits_used_for_metrics_only",
            "candidate_outputs_must_not_be_served",
        }
        or behavioral_semantics.get("execution_mode") != "shadow"
        or behavioral_semantics.get("authoritative_path") != "source"
        or (
            behavioral_semantics.get("source_outputs_authoritative")
            is not True
        )
        or (
            behavioral_semantics.get("candidate_outputs_authoritative")
            is not False
        )
        or (
            behavioral_semantics.get(
                "candidate_outputs_must_not_be_served"
            )
            is not True
        )
    ):
        raise ValueError(
            "one-shot report behavioral panel is not complete or authoritative"
        )
    behavioral_scope = _mapping(
        all_on["behavioral_scope"],
        label="all_on behavioral scope",
    )
    if (
        set(behavioral_scope)
        != {
            "token_scope",
            "total_supervised_tokens",
            "affected_supervised_tokens",
            "affected_supervised_coverage",
            "unaffected_prefix_tokens_excluded",
        }
        or behavioral_scope["token_scope"]
        != "causally_affected_supervised_tokens_only"
        or type(behavioral_scope["total_supervised_tokens"]) is not int
        or behavioral_scope["total_supervised_tokens"] <= 0
        or type(behavioral_scope["affected_supervised_tokens"]) is not int
        or behavioral_scope["affected_supervised_tokens"] <= 0
        or (
            behavioral_scope["affected_supervised_tokens"]
            > behavioral_scope["total_supervised_tokens"]
        )
        or behavioral_scope["unaffected_prefix_tokens_excluded"] is not True
    ):
        raise ValueError("one-shot report behavioral scope differs")
    boundary = _mapping(all_on["boundary"], label="boundary")
    projection = _mapping(
        all_on["projection_capacity"],
        label="projection capacity",
    )
    carrier = _mapping(
        all_on["carrier_completeness"],
        label="carrier completeness",
    )
    if (
        set(boundary) != _BOUNDARY_FIELDS
        or boundary["target_modal_width"] != 64
        or set(projection) != _PROJECTION_FIELDS
        or projection["target_modal_width"] != 64
        or projection["target_full_width"] != 640
        or set(carrier)
        != {
            "carrier",
            "boundary",
            "interpretation",
            "behavioral",
            "gates",
        }
        or carrier["carrier"] != "clamped_y3_source_model_reference"
        or carrier["boundary"] != "exact_full_width_x4"
    ):
        raise ValueError("one-shot report modal evaluation schema differs")
    gate_values = (
        _gate_passed(behavioral, label="behavioral"),
        _gate_passed(boundary, label="boundary"),
        _gate_passed(
            projection,
            label="projection capacity",
        ),
        _gate_passed(
            carrier,
            label="carrier completeness",
        ),
    )
    if all_on["passed"] is not all(gate_values):
        raise ValueError("one-shot report all_on gate aggregate differs")
    routed = _mapping(report["routed"], label="routed evaluation")
    if dict(routed) != {
        "allowed": False,
        "evaluated": False,
        "reason": "locked_protocol_all_on_only",
    }:
        raise ValueError("one-shot report routed arm is not locked disabled")
    scope = _mapping(report["scope"], label="report scope")
    if dict(scope) != {
        "source_path": "sequential_refit_authoritative",
        "candidate_path": (
            "reference_carrier_incomplete_replacement_metrics_only"
        ),
        "reference_provider": "clamped_y3_source_model_oracle",
        "reference_pass_oracle_fallback_required": True,
        "candidate_outputs_must_not_be_served": True,
        "candidate_logits_interpretation": (
            "incomplete_replacement_not_isolated_boundary_fidelity"
        ),
        "behavioral_token_scope": (
            "causally_affected_supervised_tokens_only"
        ),
        "prompt_text_loaded": False,
        "tokenizer_loaded": False,
        "parameter_reduction_claim": False,
        "latency_or_speed_claim": False,
        "full_model_claim": False,
    }:
        raise ValueError("one-shot report scope exceeds the locked claim")
    calibration_a = _mapping(
        report["calibration_a_development_evidence"],
        label="Calibration-A evidence",
    )
    if dict(calibration_a) != {
        "selection_or_assessment_eligible": False,
        "deployment_authorized": False,
        "routing_authorized": False,
        "corrected_all_on_passed": False,
        "projection_capacity_passed": False,
        "carrier_completeness_passed": False,
    }:
        raise ValueError("one-shot report reuses Calibration-A authority")
    authorization = _mapping(
        report["authorization"],
        label="report authorization",
    )
    expected_passed = all_on["passed"]
    if dict(authorization) != {
        "partial_shadow_qualified": expected_passed,
        "partial_shadow_scope": (
            "partial_edge_reference_oracle_shadow"
            if expected_passed
            else "none"
        ),
        "all_on_passed": expected_passed,
        "deployment_authorized": False,
        "deployment_scope": "none",
        "routing_authorized": False,
        "routing_qualification_available": False,
        "non_authorization_reason": (
            "reference_oracle_required_and_candidate_outputs_metrics_only"
        ),
        "standalone_deployment_authorized": False,
        "full_model_deployment_authorized": False,
    }:
        raise ValueError(
            "one-shot report authorization differs from the locked scope"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_ledger_root() -> None:
    """Create or authenticate the fixed final host-state namespace."""

    try:
        _FROZEN_LEDGER_ROOT.mkdir(
            mode=0o700,
            parents=True,
            exist_ok=True,
        )
        details = os.lstat(_FROZEN_LEDGER_ROOT)
    except OSError as error:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "host-global ledger namespace cannot be authenticated"
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "host-global ledger namespace must be owner-only directory 0700"
        )


def _exclusive_private_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("ledger write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_private_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size <= 0
            or details.st_size > _MAX_LEDGER_FILE_BYTES
        ):
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "ledger record must be a nonempty private regular file"
            )
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                    "ledger record ended before its declared file size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "ledger record changed while it was read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_canonical_mapping(
    encoded: bytes,
    *,
    label: str,
) -> dict[str, object]:
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            f"{label} is not canonical ASCII JSON"
        ) from error
    if not isinstance(raw, dict) or _canonical_json_bytes(raw) != encoded:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            f"{label} is not a canonical JSON object"
        )
    return raw


def _claim_without_digest(
    *,
    protocol_sha256: str,
    assessment_claim_sha256: str,
    manifest_sha256: str,
    manifest_example_count: int,
    manifest_families: tuple[str, ...],
    runtime_binding_sha256: str,
) -> dict[str, object]:
    return {
        "schema": f"{_SCHEMA}.claim",
        "format_version": _FORMAT_VERSION,
        "role": _ROLE,
        "state": _CLAIM_STATE,
        "protocol_sha256": protocol_sha256,
        "assessment_claim_sha256": assessment_claim_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_example_count": manifest_example_count,
        "manifest_family_count": len(manifest_families),
        "manifest_families": list(manifest_families),
        "runtime_binding_sha256": runtime_binding_sha256,
        "global_lock_identity": "frozen_manifest_sha256",
    }


def _claim_payload(**values: object) -> dict[str, object]:
    payload = _claim_without_digest(**values)  # type: ignore[arg-type]
    payload["claim_payload_sha256"] = _domain_sha256(
        payload,
        domain=_CLAIM_DOMAIN,
    )
    return payload


def _validate_claim_payload(raw: Mapping[str, object]) -> None:
    if set(raw) != _CLAIM_FIELDS:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "claim fields differ from the frozen ledger schema"
        )
    families = raw["manifest_families"]
    if (
        raw["schema"] != f"{_SCHEMA}.claim"
        or raw["format_version"] != _FORMAT_VERSION
        or raw["role"] != _ROLE
        or raw["state"] != _CLAIM_STATE
        or raw["global_lock_identity"] != "frozen_manifest_sha256"
        or type(raw["manifest_example_count"]) is not int
        or raw["manifest_example_count"] <= 0
        or type(raw["manifest_family_count"]) is not int
        or raw["manifest_family_count"] <= 0
        or not isinstance(families, list)
        or not all(isinstance(value, str) for value in families)
    ):
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "claim metadata is malformed"
        )
    normalized_families = tuple(
        _require_identifier(value, label="claim manifest family")
        for value in families
    )
    if (
        normalized_families != tuple(sorted(set(normalized_families)))
        or len(normalized_families) != raw["manifest_family_count"]
    ):
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "claim manifest families are not exact sorted identities"
        )
    for field in (
        "protocol_sha256",
        "assessment_claim_sha256",
        "manifest_sha256",
        "runtime_binding_sha256",
        "claim_payload_sha256",
    ):
        try:
            _require_sha256(raw[field], label=f"claim {field}")
        except ValueError as error:
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                str(error)
            ) from error
    unsigned = dict(raw)
    recorded = unsigned.pop("claim_payload_sha256")
    if _domain_sha256(unsigned, domain=_CLAIM_DOMAIN) != recorded:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "claim payload hash mismatch"
        )


def _terminal_payload(
    *,
    outcome: TerminalOutcome,
    manifest_sha256: str,
    claim_file_sha256: str,
    evidence_sha256: str,
) -> dict[str, object]:
    evidence_field = (
        "report_sha256" if outcome == "success" else "error_sha256"
    )
    payload: dict[str, object] = {
        "schema": f"{_SCHEMA}.terminal_receipt",
        "format_version": _FORMAT_VERSION,
        "role": _ROLE,
        "state": f"terminal_{outcome}",
        "outcome": outcome,
        "manifest_sha256": manifest_sha256,
        "claim_file_sha256": claim_file_sha256,
        evidence_field: evidence_sha256,
    }
    payload["terminal_receipt_sha256"] = _domain_sha256(
        payload,
        domain=_TERMINAL_DOMAIN,
    )
    return payload


def _validate_terminal_payload(raw: Mapping[str, object]) -> None:
    outcome = raw.get("outcome")
    if outcome not in ("success", "failure"):
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "terminal receipt outcome is malformed"
        )
    evidence_field = (
        "report_sha256" if outcome == "success" else "error_sha256"
    )
    if set(raw) != _TERMINAL_COMMON_FIELDS | {evidence_field}:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "terminal receipt fields differ from its outcome schema"
        )
    if (
        raw["schema"] != f"{_SCHEMA}.terminal_receipt"
        or raw["format_version"] != _FORMAT_VERSION
        or raw["role"] != _ROLE
        or raw["state"] != f"terminal_{outcome}"
    ):
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "terminal receipt metadata is malformed"
        )
    for field in (
        "manifest_sha256",
        "claim_file_sha256",
        evidence_field,
        "terminal_receipt_sha256",
    ):
        try:
            _require_sha256(raw[field], label=f"terminal {field}")
        except ValueError as error:
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                str(error)
            ) from error
    unsigned = dict(raw)
    recorded = unsigned.pop("terminal_receipt_sha256")
    if _domain_sha256(unsigned, domain=_TERMINAL_DOMAIN) != recorded:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "terminal receipt payload hash mismatch"
        )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt:
    """Authenticated immutable terminal ledger record."""

    path: Path
    outcome: TerminalOutcome
    manifest_sha256: str
    claim_file_sha256: str
    evidence_sha256: str
    terminal_receipt_sha256: str
    terminal_file_sha256: str

    def metadata(self) -> dict[str, object]:
        evidence_field = (
            "report_sha256"
            if self.outcome == "success"
            else "error_sha256"
        )
        return {
            "path": str(self.path),
            "outcome": self.outcome,
            "manifest_sha256": self.manifest_sha256,
            "claim_file_sha256": self.claim_file_sha256,
            evidence_field: self.evidence_sha256,
            "terminal_receipt_sha256": self.terminal_receipt_sha256,
            "terminal_file_sha256": self.terminal_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class _Gemma3L3L4GraphOrganizedSVDOneShotSession:
    """A consumed Calibration-B claim that may be finalized exactly once."""

    claim_path: Path
    terminal_path: Path
    protocol_sha256: str
    assessment_claim_sha256: str
    manifest_sha256: str
    runtime_binding_sha256: str
    claim_payload_sha256: str
    claim_file_sha256: str
    _claim_bytes: bytes

    def _validate_claim(self) -> None:
        expected_claim_path, expected_terminal_path = (
            _frozen_ledger_paths(self.manifest_sha256)
        )
        if (
            self.claim_path != expected_claim_path
            or self.terminal_path != expected_terminal_path
        ):
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "session paths differ from the frozen manifest ledger"
            )
        encoded = _read_private_file(self.claim_path)
        raw = _decode_canonical_mapping(encoded, label="claim")
        _validate_claim_payload(raw)
        if (
            encoded != self._claim_bytes
            or _file_sha256(encoded) != self.claim_file_sha256
            or raw["claim_payload_sha256"] != self.claim_payload_sha256
        ):
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "claim differs from the consumed session identity"
            )

    def _complete(
        self,
        *,
        outcome: TerminalOutcome,
        evidence_sha256: str,
    ) -> Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt:
        evidence = _require_sha256(
            evidence_sha256,
            label=f"{outcome} evidence",
        )
        self._validate_claim()
        payload = _terminal_payload(
            outcome=outcome,
            manifest_sha256=self.manifest_sha256,
            claim_file_sha256=self.claim_file_sha256,
            evidence_sha256=evidence,
        )
        encoded = _canonical_json_bytes(payload)
        try:
            _exclusive_private_write(self.terminal_path, encoded)
        except FileExistsError as error:
            existing = _read_private_file(self.terminal_path)
            raw = _decode_canonical_mapping(
                existing,
                label="terminal receipt",
            )
            _validate_terminal_payload(raw)
            if (
                raw["manifest_sha256"] != self.manifest_sha256
                or raw["claim_file_sha256"] != self.claim_file_sha256
            ):
                raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                    "terminal receipt belongs to a foreign claim"
                ) from error
            raise (
                Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError(
                    "Calibration B already has a terminal receipt"
                )
            ) from error
        persisted = _read_private_file(self.terminal_path)
        if persisted != encoded:
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "terminal receipt changed immediately after creation"
            )
        return Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt(
            path=self.terminal_path,
            outcome=outcome,
            manifest_sha256=self.manifest_sha256,
            claim_file_sha256=self.claim_file_sha256,
            evidence_sha256=evidence,
            terminal_receipt_sha256=str(
                payload["terminal_receipt_sha256"]
            ),
            terminal_file_sha256=_file_sha256(persisted),
        )

    def _complete_success(
        self,
        report_sha256: str,
    ) -> Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt:
        """Persist the report hash as the only successful terminal outcome."""

        return self._complete(
            outcome="success",
            evidence_sha256=report_sha256,
        )

    def _complete_failure(
        self,
        error_sha256: str,
    ) -> Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt:
        """Persist only a sanitized error hash as the terminal outcome."""

        return self._complete(
            outcome="failure",
            evidence_sha256=error_sha256,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "claim_path": str(self.claim_path),
            "terminal_path": str(self.terminal_path),
            "protocol_sha256": self.protocol_sha256,
            "assessment_claim_sha256": self.assessment_claim_sha256,
            "manifest_sha256": self.manifest_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "claim_payload_sha256": self.claim_payload_sha256,
            "claim_file_sha256": self.claim_file_sha256,
            "state": _CLAIM_STATE,
        }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDOneShotTransactionResult:
    """Successful prompt-free report and its immutable ledger receipt."""

    report: Mapping[str, object]
    report_sha256: str
    terminal_receipt: (
        Gemma3L3L4GraphOrganizedSVDOneShotTerminalReceipt
    )


def _claim_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime_binding_sha256: str,
) -> _Gemma3L3L4GraphOrganizedSVDOneShotSession:
    """Consume the frozen Calibration-B role before prompt materialization."""

    if not isinstance(
        protocol,
        Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    ):
        raise TypeError("protocol must be the strict frozen shadow protocol")
    protocol.validate_integrity()
    supplied_runtime_binding = _require_sha256(
        runtime_binding_sha256,
        label="runtime binding",
    )
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    metadata = protocol.metadata()
    try:
        manifest_metadata = metadata["corpus"]["calibration_b_manifest"]
        manifest_sha256 = manifest_metadata["artifact_sha256"]
        expected_example_count = manifest_metadata["example_count"]
        expected_family_count = manifest_metadata["family_count"]
        runtime_metadata = metadata["runtime_binding_contract"]
        frozen_runtime_binding = runtime_metadata["artifact_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "protocol lacks a frozen Calibration-B or runtime identity"
        ) from error
    expected_runtime_binding = _require_sha256(
        frozen_runtime_binding,
        label="protocol runtime binding",
    )
    if supplied_runtime_binding != expected_runtime_binding:
        raise ValueError(
            "runtime binding differs from the frozen protocol"
        )
    manifest_digest = _require_sha256(
        manifest_sha256,
        label="Calibration-B manifest",
    )
    families = tuple(sorted(set(manifest.values())))
    if (
        expected_example_count != len(manifest)
        or expected_family_count != len(families)
        or not all(
            isinstance(example_id, str)
            and _SHA256.fullmatch(example_id) is not None
            and isinstance(family, str)
            and family in families
            for example_id, family in manifest.items()
        )
    ):
        raise ValueError(
            "protocol and frozen Calibration-B manifest identities differ"
        )
    protocol_sha256 = _require_sha256(
        protocol.artifact_sha256,
        label="protocol",
    )
    assessment_claim_sha256 = _require_sha256(
        protocol.calibration_b_assessment_claim_sha256(),
        label="derived assessment claim",
    )
    payload = _claim_payload(
        protocol_sha256=protocol_sha256,
        assessment_claim_sha256=assessment_claim_sha256,
        manifest_sha256=manifest_digest,
        manifest_example_count=len(manifest),
        manifest_families=families,
        runtime_binding_sha256=expected_runtime_binding,
    )
    encoded = _canonical_json_bytes(payload)
    _ensure_private_ledger_root()
    claim_path, terminal_path = _frozen_ledger_paths(manifest_digest)
    if terminal_path.exists():
        terminal_bytes = _read_private_file(terminal_path)
        terminal = _decode_canonical_mapping(
            terminal_bytes,
            label="terminal receipt",
        )
        _validate_terminal_payload(terminal)
        if terminal["manifest_sha256"] != manifest_digest:
            raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
                "terminal receipt belongs to a foreign manifest"
            )
        raise Gemma3L3L4GraphOrganizedSVDOneShotAlreadyFinalizedError(
            "Calibration B already has a terminal receipt"
        )
    try:
        _exclusive_private_write(claim_path, encoded)
    except FileExistsError as error:
        existing = _read_private_file(claim_path)
        raw = _decode_canonical_mapping(existing, label="claim")
        _validate_claim_payload(raw)
        if existing != encoded:
            raise Gemma3L3L4GraphOrganizedSVDOneShotForeignClaimError(
                "Calibration B was consumed by another protocol or runtime"
            ) from error
        raise Gemma3L3L4GraphOrganizedSVDOneShotAlreadyClaimedError(
            "Calibration B is already claimed for this frozen manifest"
        ) from error
    persisted = _read_private_file(claim_path)
    if persisted != encoded:
        raise Gemma3L3L4GraphOrganizedSVDOneShotIntegrityError(
            "claim changed immediately after exclusive creation"
        )
    return _Gemma3L3L4GraphOrganizedSVDOneShotSession(
        claim_path=claim_path,
        terminal_path=terminal_path,
        protocol_sha256=protocol_sha256,
        assessment_claim_sha256=assessment_claim_sha256,
        manifest_sha256=manifest_digest,
        runtime_binding_sha256=expected_runtime_binding,
        claim_payload_sha256=str(payload["claim_payload_sha256"]),
        claim_file_sha256=_file_sha256(persisted),
        _claim_bytes=persisted,
    )


def _preflight_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    adapter: Gemma3CausalLMAdapter,
    prompt_loader: Callable[[str], bytes],
) -> tuple[
    str,
    object,
    Mapping[str, object],
    tuple[tuple[str, str], ...],
]:
    """Authenticate every static input before consuming the one-shot claim."""

    if not callable(prompt_loader):
        raise TypeError("prompt_loader must be a one-argument callable")
    runtime_binding_sha256 = _validate_protocol_runtime(protocol, runtime)
    _validate_live_adapter(runtime, adapter)
    tokenizer, tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    canonical_items = tuple(sorted(manifest.items()))
    metadata = protocol.metadata()
    try:
        manifest_metadata = metadata["corpus"]["calibration_b_manifest"]
        expected_examples = manifest_metadata["example_count"]
        expected_families = manifest_metadata["family_count"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "protocol lacks the frozen Calibration-B manifest contract"
        ) from error
    if (
        expected_examples != 96
        or expected_examples != len(canonical_items)
        or len({example_id for example_id, _ in canonical_items}) != 96
        or expected_families != 8
        or len({family_id for _, family_id in canonical_items}) != 8
        or any(
            _SHA256.fullmatch(example_id) is None
            or not family_id
            or family_id != family_id.strip()
            for example_id, family_id in canonical_items
        )
    ):
        raise ValueError("frozen Calibration-B manifest preflight failed")
    return (
        runtime_binding_sha256,
        tokenizer,
        tokenizer_contract,
        canonical_items,
    )


def run_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    adapter: Gemma3CausalLMAdapter,
    prompt_loader: Callable[[str], bytes],
) -> Gemma3L3L4GraphOrganizedSVDOneShotTransactionResult:
    """Run the sole claim-first, prompt-owned Calibration-B transaction."""

    (
        runtime_binding_sha256,
        tokenizer,
        tokenizer_contract,
        canonical_items,
    ) = _preflight_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
        protocol=protocol,
        runtime=runtime,
        adapter=adapter,
        prompt_loader=prompt_loader,
    )
    session = (
        _claim_gemma3_l3_l4_graph_organized_svd_calibration_b_one_shot(
            protocol=protocol,
            runtime_binding_sha256=runtime_binding_sha256,
        )
    )

    canonical_manifest = dict(canonical_items)

    def observations():
        for example_id, family_id in canonical_items:
            prompt_utf8 = prompt_loader(example_id)
            if type(prompt_utf8) is not bytes:
                raise TypeError(
                    "prompt_loader must return exact strict UTF-8 bytes"
                )
            try:
                observation = (
                    _execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation(
                        protocol=protocol,
                        runtime=runtime,
                        adapter=adapter,
                        tokenizer=tokenizer,
                        prompt_utf8=prompt_utf8,
                        example_id=example_id,
                        family_id=family_id,
                        _validated_tokenizer_contract=tokenizer_contract,
                    )
                )
            finally:
                del prompt_utf8
            yield observation

    try:
        evaluated = _evaluate_gemma3_l3_l4_graph_organized_svd_shadow(
            protocol,
            observations(),
            assessment_claim_sha256=(
                protocol.calibration_b_assessment_claim_sha256()
            ),
            expected_family_by_example=canonical_manifest,
        )
        report, encoded = _canonical_prompt_free_report(evaluated)
        _validate_shadow_evaluation_report(
            report,
            protocol=protocol,
            session=session,
        )
    except Exception as error:
        session._complete_failure(_sanitized_exception_sha256(error))
        raise
    report_sha256 = _file_sha256(encoded)
    terminal = session._complete_success(report_sha256)
    return Gemma3L3L4GraphOrganizedSVDOneShotTransactionResult(
        report=report,
        report_sha256=report_sha256,
        terminal_receipt=terminal,
    )
