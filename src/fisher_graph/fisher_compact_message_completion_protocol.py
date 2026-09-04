"""Mechanical protocol primitives for a future Fisher D2 campaign.

This module implements the K-series row identity, authorization, sharding,
physical row binding, failure classification, support, and durable-ledger
mechanics for ``fisher_compact_message_completion_probe_v2``.  It is
deliberately model-free and standard-library only: importing it cannot load a
model, tokenize protected data, consume a panel, or create a ledger.

The current corpus remains closed at P0 with
``development_complete_independent_data_unavailable``.  These primitives do
not reopen it; callers must use them only for a separately initialized future
D2 campaign.

The canonical JSON encoder accepts a deliberately narrow RFC 8785-compatible
subset (objects with ASCII protocol keys, arrays, strings, booleans, null, and
bounded integers; never floats).  Keeping hashes inside this subset avoids
cross-runtime floating-point and object-key-order ambiguity.

Physical tensor rows are bound inside each finalized ``.safetensors`` shard as
``capture_row_id_raw32`` with dtype ``U8`` and shape ``[n, 32]``.  A strict
standard-library parser extracts those bytes from the shard itself; row IDs
are never accepted from caller metadata or a detached sidecar.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3
import stat
import string
import struct
import threading
from typing import Any, BinaryIO, Iterator, TypeAlias


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "CURRENT_P0_CAMPAIGN_ID",
    "CURRENT_P0_TERMINAL_STATE",
    "FUTURE_D2_CAMPAIGN_SCOPE",
    "SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE",
    "PRIMARY_FAILURE_RECEIPT_SCHEMA",
    "SUPPORT_FAILURE_RECEIPT_SCHEMA",
    "TERMINAL_STATE_SCHEMA",
    "FINAL_EXPERIMENT_MANIFEST_SCHEMA",
    "AuthorizationMode",
    "AuthorizationSpec",
    "ArtifactClosure",
    "ArtifactInventoryEntry",
    "CampaignPrefix",
    "CampaignTerminalResult",
    "CaptureState",
    "CaptureResumeResult",
    "CaptureVerification",
    "ChartHeadObservation",
    "ChartHeadSupportDecision",
    "EvaluationRow",
    "ExpectedChartHeadRow",
    "FailureClass",
    "FailureDisposition",
    "IntegrityViolation",
    "MeasurementContractViolation",
    "PanelSnapshot",
    "PartialShardInventory",
    "PhysicalShardEvidence",
    "PrefixClosure",
    "ProtocolLedger",
    "ProtocolSchemaError",
    "ProtocolStateError",
    "ProtocolViolation",
    "ResourceLimitViolation",
    "RowIndexPayload",
    "ShardPlan",
    "ShardPlanEntry",
    "StoredArtifact",
    "SupportStatus",
    "SupportFailureReason",
    "SupportFailureSpec",
    "TerminalState",
    "TransactionResult",
    "VerifiedShardRecord",
    "artifact_sha256",
    "build_row_index_payload",
    "build_shard_plan",
    "canonical_json_bytes",
    "canonicalize_rows",
    "classify_chart_head_support",
    "classify_failure",
    "derive_row_id",
    "ordered_row_sequence_sha256",
    "row_index_payload_sha256",
    "row_set_sha256",
    "sha256_file",
    "validate_sanitized_failure_details",
    "validate_reached_prefix_closure",
    "verify_capture_shards",
    "verify_capture_shard_files",
    "verify_partial_capture_shards",
    "verify_partial_capture_shard_files",
    "verify_physical_shard",
    "verify_physical_shard_file",
]


CURRENT_P0_TERMINAL_STATE = (
    "development_complete_independent_data_unavailable"
)
CURRENT_P0_CAMPAIGN_ID = "fisher-compact-message-completion-v2-current-abcd-p0"
FUTURE_D2_CAMPAIGN_SCOPE = "future_separately_initialized_D2_campaign"
SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE = "synthetic_qualification_only"
_CAMPAIGN_SCOPES = frozenset(
    {
        FUTURE_D2_CAMPAIGN_SCOPE,
        SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE,
    }
)

AUTHORIZATION_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.authorization.v1"
)
ROW_INDEX_SCHEMA = (
    "fisher-compact-message-completion/evaluation-row-index-payload/v1"
)
SHARD_PLAN_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.shard_plan.v1"
)
CAPTURE_VERIFICATION_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.capture_verification.v1"
)
PARTIAL_SHARD_INVENTORY_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.partial_shards.v1"
)
PRIMARY_FAILURE_RECEIPT_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.primary_failure_receipt.v1"
)
SUPPORT_FAILURE_RECEIPT_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.support_failure_receipt.v1"
)
TERMINAL_STATE_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.terminal_state.v1"
)
FINAL_EXPERIMENT_MANIFEST_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.final_experiment_manifest.v1"
)
LEDGER_SCHEMA = (
    "fisher_graph.fisher_compact_message_completion.protocol_ledger.v1"
)

_FORMAT_VERSION = 1
_PARTITION_ROLES = frozenset({"fit", "tune", "development_confirm"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TX_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_CANONICAL_NONLOWER_KEYS = frozenset({"attempted_locator_K_values"})
_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1
_MAX_SAFE_JSON_ITEMS = 1_000_000
_MAX_SAFE_JSON_DEPTH = 16
_MAXIMUM_SHARD_ROWS = 16_384
_MAX_PERSISTED_EXTERNAL_ARTIFACT_BYTES = 16 * 1024 * 1024
_EXACT_SHARD_PATH_TEMPLATE = (
    "capture_shards/{partition}/shard-{shard_index:06d}.safetensors"
)

_ROW_ID_DOMAIN = (
    b"fisher-compact-message-completion/evaluation-row/v1\x00"
)
_ROW_SET_DOMAIN = (
    b"fisher-compact-message-completion/evaluation-row-set/v1\x00"
)
_ROW_SEQUENCE_DOMAIN = (
    b"fisher-compact-message-completion/evaluation-row-sequence/v1\x00"
)
_ROW_INDEX_PAYLOAD_DOMAIN = (
    b"fisher-compact-message-completion/evaluation-row-index-payload/v1\x00"
)
_CONTENT_SET_DOMAIN = b"fisher-compact-message-completion/content-set/v1"
_ANALYSIS_UNIT_SET_DOMAIN = (
    b"fisher-compact-message-completion/analysis-unit-set/v1"
)
_PARTITION_COMMITMENT_DOMAIN = (
    b"fisher-compact-message-completion/partition/v1"
)
_AUTHORIZATION_DOMAIN = (
    b"fisher-compact-message-completion/capture-authorization/v1\x00"
)
_SHARD_SEQUENCE_DOMAIN = (
    b"fisher-compact-message-completion/capture-shard-row-sequence/v1\x00"
)
_SHARD_PLAN_DOMAIN = (
    b"fisher-compact-message-completion/capture-shard-plan/v1\x00"
)
_CAPTURE_VERIFICATION_DOMAIN = (
    b"fisher-compact-message-completion/capture-verification/v1\x00"
)
_PARTIAL_INVENTORY_DOMAIN = (
    b"fisher-compact-message-completion/partial-shard-inventory/v1\x00"
)
_CAPTURE_START_DOMAIN = (
    b"fisher-compact-message-completion/tensor-capture-start/v1\x00"
)
_CAPTURE_ROOT_DOMAIN = (
    b"fisher-compact-message-completion/capture-root-identity/v1\x00"
)
_RESUME_CHECKPOINT_DOMAIN = (
    b"fisher-compact-message-completion/resume-checkpoint/v1\x00"
)
_TERMINAL_SUCCESS_DOMAIN = (
    b"fisher-compact-message-completion/capture-success/v1\x00"
)
_TERMINAL_FAILURE_DOMAIN = (
    b"fisher-compact-message-completion/capture-failure/v1\x00"
)
_DEVELOPMENT_OUTCOME_DOMAIN = (
    b"fisher-compact-message-completion/development-outcome/v1\x00"
)
_ROW_INDEX_FAILURE_DOMAIN = (
    b"fisher-compact-message-completion/row-index-failure/v1\x00"
)

_SAFETENSORS_HEADER_LENGTH = struct.Struct("<Q")
_ROW_ID_TENSOR_NAME = "capture_row_id_raw32"
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
_VERIFIED_CONSTRUCTION_TOKEN = object()

JsonScalar: TypeAlias = None | bool | int | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ProtocolSchemaError(ValueError):
    """A caller supplied a malformed or noncanonical protocol value."""


class ProtocolStateError(RuntimeError):
    """A requested operation is forbidden from the durable current state."""


class ProtocolViolation(RuntimeError):
    """A classified campaign failure with an exact reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        self.disposition = classify_failure(reason_code)
        super().__init__(message)


class IntegrityViolation(ProtocolViolation):
    """A classified integrity failure."""


class MeasurementContractViolation(ProtocolViolation):
    """A classified measurement-contract failure."""


class ResourceLimitViolation(ProtocolViolation):
    """A classified resource-limit terminal."""


class AuthorizationMode(str, Enum):
    """The two permitted authorization row-set modes."""

    PRECOMPUTED = "precomputed"
    DERIVE_AFTER_AUTHORIZATION = "derive_after_authorization"


class CaptureState(str, Enum):
    """Sticky states in the K-series per-panel state machine."""

    UNOPENED = "unopened"
    AUTHORIZED = "authorized"
    ROW_INDEX_COMMITTED = "row_index_committed"
    ROW_INDEX_FAILED = "row_index_failed"
    CAPTURE_STARTED = "capture_started"
    CONSUMED_SUCCESS = "consumed_success"
    CONSUMED_FAILED = "consumed_failed"


class FailureClass(str, Enum):
    """The exclusive mechanical failure classes."""

    INTEGRITY = "integrity"
    MEASUREMENT_CONTRACT = "measurement_contract"
    RESOURCE_LIMIT = "resource_limit"


class TerminalState(str, Enum):
    """Terminal states produced by classified mechanical failures."""

    INVALID_CAMPAIGN_INTEGRITY = "invalid_campaign_integrity"
    INVALID_CAMPAIGN_MEASUREMENT_CONTRACT = (
        "invalid_campaign_measurement_contract"
    )
    DEVELOPMENT_COMPLETE_RESOURCE_LIMITED = (
        "development_complete_resource_limited"
    )
    DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT = (
        "development_complete_insufficient_support"
    )
    DEVELOPMENT_COMPLETE_REFIT_INSTABILITY = (
        "development_complete_refit_instability"
    )
    DEVELOPMENT_COMPLETE_NEGATIVE_J4 = "development_complete_negative_j4"
    DEVELOPMENT_COMPLETE_ATTENTION_OUTPUT_NOT_READY = (
        "development_complete_attention_output_not_ready"
    )
    DEVELOPMENT_COMPLETE_NO_ELIGIBLE_GUARD = (
        "development_complete_no_eligible_guard"
    )


_DEVELOPMENT_OUTCOME_REASON_CODES: dict[TerminalState, str] = {
    TerminalState.DEVELOPMENT_COMPLETE_REFIT_INSTABILITY: (
        "all_fit_tune_replay_exceeded_two_percent"
    ),
    TerminalState.DEVELOPMENT_COMPLETE_NEGATIVE_J4: (
        "development_screen_conjunction_failed"
    ),
    TerminalState.DEVELOPMENT_COMPLETE_ATTENTION_OUTPUT_NOT_READY: (
        "o2q_or_o7_fidelity_gate_failed"
    ),
}


class SupportStatus(str, Enum):
    """Result of chart/head completeness and support classification."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    INVALID_MEASUREMENT_CONTRACT = "invalid_measurement_contract"


class CampaignPrefix(IntEnum):
    """The exact monotonic G/H/I/J/K campaign prefixes."""

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6
    P7 = 7
    P8 = 8
    P9 = 9
    P10 = 10
    P11 = 11
    P12 = 12


_CAPTURE_PREFIX_BY_PARTITION = {
    "fit": CampaignPrefix.P3,
    "tune": CampaignPrefix.P5,
    "development_confirm": CampaignPrefix.P10,
}
_AUTHORIZATION_PREFIX_BY_PARTITION = {
    "fit": CampaignPrefix.P3,
    "tune": CampaignPrefix.P5,
    "development_confirm": CampaignPrefix.P9,
}


class SupportFailureReason(str, Enum):
    """Permitted insufficient-support reasons after chart/head cleanup."""

    ZERO_FIT_MONITOR_CHART_SUPPORT = "zero_fit_monitor_chart_support"
    NO_POSITIVE_OPERATOR_REFERENCE = "no_positive_operator_reference"
    LOWER_K_FALLBACK_EXHAUSTED = "lower_K_fallback_exhausted"
    INSUFFICIENT_INDEPENDENT_CHART_SUPPORT = (
        "insufficient_independent_chart_support"
    )
    EXCESSIVE_OUT_OF_SUPPORT_RATE = "excessive_out_of_support_rate"


_SUPPORT_STAGE_CONTRACT: dict[
    str, tuple[str, CampaignPrefix, CampaignPrefix]
] = {
    "fit_monitor_chart_support": ("fit", CampaignPrefix.P3, CampaignPrefix.P4),
    "fit_monitor_chart_validation": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "full_rank_coefficient_validation": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "lower_K_representation_fallback": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "operator_reference_construction": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "per_codec_operator_reference_construction": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "mandatory_chart_head_support": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "mandatory_chart_head_support_checks": (
        "fit",
        CampaignPrefix.P3,
        CampaignPrefix.P4,
    ),
    "development_confirm_support": (
        "development_confirm",
        CampaignPrefix.P10,
        CampaignPrefix.P11,
    ),
    "development_confirm_chart_head_support": (
        "development_confirm",
        CampaignPrefix.P10,
        CampaignPrefix.P11,
    ),
    "development_confirm_post_evaluation_support": (
        "development_confirm",
        CampaignPrefix.P11,
        CampaignPrefix.P12,
    ),
}


_PREFIX_MANDATORY_ARTIFACTS: dict[CampaignPrefix, frozenset[str]] = {
    CampaignPrefix.P0: frozenset(
        {
            "semantic_canonicalization_spec.json",
            "reserve_retired_exposed.json",
            "local_development_registry_status.json",
            "corpus_role_classification.json",
            "preflight_decision.json",
            "commands.md",
        }
    ),
    CampaignPrefix.P1: frozenset(
        {
            "independence_protocol.json",
            "split_manifest.json",
            "o1_calibration_assignment.json",
        }
    ),
    CampaignPrefix.P2: frozenset(
        {
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "execution_budget.json",
            "source_inventory.json",
            "row_derivation_protocol.json",
        }
    ),
    CampaignPrefix.P3: frozenset(
        {
            "fit_capture_authorization.json",
            "fit_capture_transition_authorized.json",
            "row_index_manifest_fit.json",
            "fit_row_index_committed.json",
            "capture_shard_plan_fit.json",
            "fit_tensor_capture_started.json",
            "fit_capture_transition_consumed_success.json",
            "capture_shard_manifest_fit.json",
            "capture_invariants.json",
        }
    ),
    CampaignPrefix.P4: frozenset(
        {
            "o1_tolerance_protocol.json",
            "o1_validation_results.json",
            "o1_complete_fit_results.json",
            "locator_protocol.json",
            "locator_candidates.json",
            "codec_protocol.json",
            "codec_candidate_manifests.json",
            "message_sketch_manifest_p32.json",
            "representation_numeric_constants.json",
            "selection_representation_manifest.json",
            "learning_protocol.json",
            "development_eligibility_protocol.json",
            "preregistration_root.json",
            "local_development_freshness_audit_receipt.json",
            "campaign_root.json",
            "full_rank_identity_smoke.json",
            "ceiling_gate_protocol.json",
        }
    ),
    CampaignPrefix.P5: frozenset(
        {
            "tune_capture_authorization.json",
            "tune_capture_transition_authorized.json",
            "row_index_manifest_tune.json",
            "tune_row_index_committed.json",
            "capture_shard_plan_tune.json",
            "tune_tensor_capture_started.json",
            "tune_capture_transition_consumed_success.json",
            "capture_shard_manifest_tune.json",
            "o2q_results.json",
            "o2v_results.json",
        }
    ),
    CampaignPrefix.P6: frozenset(
        {
            "selected_codec.json",
            "frozen_representation_manifest.json",
            "m2q_results.json",
            "m2v_results.json",
            "compact_ambient_oracle.json",
            "compact_chart_oracle.json",
        }
    ),
    CampaignPrefix.P7: frozenset(
        {"j_model_fit_results.json", "j_model_tune_results.json"}
    ),
    CampaignPrefix.P8: frozenset(
        {"all_fit_weight_refit.json", "nonselecting_tune_replay.json"}
    ),
    CampaignPrefix.P9: frozenset(
        {
            "frozen_B_star.json",
            "direct_full_rank_identity_null.json",
            "candidate_bundle.safetensors",
            "comparator_bundle.safetensors",
            "candidate_bundle_metadata.json",
            "development_confirm_capture_authorization.json",
            "development_confirm_capture_transition_authorized.json",
        }
    ),
    CampaignPrefix.P10: frozenset(
        {
            "row_index_manifest_development_confirm.json",
            "development_confirm_row_index_committed.json",
            "capture_shard_plan_development_confirm.json",
            "development_confirm_tensor_capture_started.json",
            "development_confirm_capture_transition_consumed_success.json",
            "capture_shard_manifest_development_confirm.json",
        }
    ),
    CampaignPrefix.P11: frozenset(
        {
            "development_confirm_results.json",
            "interaction_mean_ablation.json",
            "o7_attention_output_completion_results.json",
            "development_eligibility_result.json",
            "development_power_assessment.json",
        }
    ),
    CampaignPrefix.P12: frozenset({"development_candidate_manifest.json"}),
}
_TERMINAL_TAIL_ARTIFACTS = frozenset(
    {"terminal_state.json", "final_report.md", "final_experiment_manifest.json"}
)
_SAME_PREFIX_SUCCESS_DEPENDENCIES = {
    "capture_invariants.json": "fit",
    "o2q_results.json": "tune",
    "o2v_results.json": "tune",
}
_CONDITIONAL_PREFIX_ARTIFACTS: dict[CampaignPrefix, frozenset[str]] = {
    CampaignPrefix.P3: frozenset(
        {
            "row_index_failure_fit.json",
            "fit_row_index_failed.json",
            "fit_capture_transition_consumed_failed.json",
            "capture_failure_fit.json",
            "partial_shard_inventory_fit.json",
        }
    ),
    CampaignPrefix.P5: frozenset(
        {
            "row_index_failure_tune.json",
            "tune_row_index_failed.json",
            "tune_capture_transition_consumed_failed.json",
            "capture_failure_tune.json",
            "partial_shard_inventory_tune.json",
        }
    ),
    CampaignPrefix.P10: frozenset(
        {
            "row_index_failure_development_confirm.json",
            "development_confirm_row_index_failed.json",
            "development_confirm_capture_transition_consumed_failed.json",
            "capture_failure_development_confirm.json",
            "partial_shard_inventory_development_confirm.json",
        }
    ),
}
_ARTIFACT_PREFIX: dict[str, CampaignPrefix] = {
    name: prefix
    for prefix, names in _PREFIX_MANDATORY_ARTIFACTS.items()
    for name in names
}
for _prefix, _names in _CONDITIONAL_PREFIX_ARTIFACTS.items():
    for _name in _names:
        _ARTIFACT_PREFIX[_name] = _prefix

_GLOBAL_LEDGER_OWNED_ARTIFACTS = frozenset(
    {
        "invalid_campaign_receipt.json",
        "resource_limit_failure.json",
        "support_failure_receipt.json",
        "terminal_state.json",
        "final_report.md",
        "final_experiment_manifest.json",
    }
)
_PARTITION_LEDGER_OWNED_TEMPLATES = (
    "{partition}_capture_authorization.json",
    "{partition}_capture_transition_authorized.json",
    "{partition}_authorization_transaction_commit.json",
    "row_index_manifest_{partition}.json",
    "{partition}_row_index_committed.json",
    "{partition}_row_index_transaction_commit.json",
    "row_index_failure_{partition}.json",
    "{partition}_row_index_failed.json",
    "{partition}_row_index_failure_transaction_commit.json",
    "{partition}_row_index_failure_finalization_commit.json",
    "capture_shard_plan_{partition}.json",
    "{partition}_shard_plan_transaction_commit.json",
    "{partition}_tensor_capture_started.json",
    "{partition}_capture_start_transaction_commit.json",
    "capture_shard_manifest_{partition}.json",
    "{partition}_capture_transition_consumed_success.json",
    "{partition}_capture_success_transaction_commit.json",
    "capture_failure_{partition}.json",
    "{partition}_capture_transition_consumed_failed.json",
    "partial_shard_inventory_{partition}.json",
    "{partition}_capture_failure_transaction_commit.json",
    "{partition}_stage_failure_transaction_commit.json",
)
_LEDGER_OWNED_ARTIFACTS = _GLOBAL_LEDGER_OWNED_ARTIFACTS | frozenset(
    template.format(partition=partition)
    for partition in _PARTITION_ROLES
    for template in _PARTITION_LEDGER_OWNED_TEMPLATES
)
_LEDGER_OWNED_ARTIFACTS_CASEFOLD = frozenset(
    name.casefold() for name in _LEDGER_OWNED_ARTIFACTS
)
_FROZEN_ARTIFACT_NAMES_CASEFOLD = {
    name.casefold(): name for name in _ARTIFACT_PREFIX
}
_INTERRUPTION_ARTIFACT_RE = re.compile(
    r"^(?:capture_interruption_(?:fit|tune|development_confirm)_\d{6}|"
    r"(?:fit|tune|development_confirm)_interruption_\d{6}_"
    r"transaction_commit)\.json$"
)
_CAPTURE_SHARD_ARTIFACT_RE = re.compile(
    r"^capture_shards/(?:fit|tune|development_confirm)/"
    r"shard-\d{6}\.safetensors$"
)


def _is_ledger_owned_artifact_name(name: str) -> bool:
    return name in _LEDGER_OWNED_ARTIFACTS or bool(
        _INTERRUPTION_ARTIFACT_RE.fullmatch(name)
        or _CAPTURE_SHARD_ARTIFACT_RE.fullmatch(name)
    )


def _is_ledger_owned_artifact_casefold_alias(name: str) -> bool:
    """Return whether ``name`` aliases a reserved name on folded filesystems."""

    folded = name.casefold()
    return (
        folded in _LEDGER_OWNED_ARTIFACTS_CASEFOLD
        or bool(
            _INTERRUPTION_ARTIFACT_RE.fullmatch(folded)
            or _CAPTURE_SHARD_ARTIFACT_RE.fullmatch(folded)
        )
    )


def _coerce_prefix(value: CampaignPrefix | str, *, label: str) -> CampaignPrefix:
    if isinstance(value, CampaignPrefix):
        return value
    if isinstance(value, str):
        try:
            return CampaignPrefix[value]
        except KeyError as error:
            raise ProtocolSchemaError(f"{label} must be P0 through P12") from error
    raise ProtocolSchemaError(f"{label} must be a campaign prefix")


def _normalize_prefix_pair(
    last_completed_prefix: CampaignPrefix | str | None,
    active_prefix: CampaignPrefix | str,
) -> tuple[CampaignPrefix | None, CampaignPrefix]:
    active = _coerce_prefix(active_prefix, label="active_prefix")
    if last_completed_prefix is None:
        if active is not CampaignPrefix.P0:
            raise ProtocolSchemaError(
                "null last_completed_prefix is valid only with active P0"
            )
        return None, active
    completed = _coerce_prefix(
        last_completed_prefix, label="last_completed_prefix"
    )
    if completed.value + 1 != active.value:
        raise ProtocolSchemaError(
            "active_prefix must immediately follow last_completed_prefix"
        )
    return completed, active


def _validate_capture_prefix_for_partition(
    partition_role: str, active_prefix: CampaignPrefix
) -> None:
    expected = _CAPTURE_PREFIX_BY_PARTITION[_validate_partition_role(partition_role)]
    if active_prefix is not expected:
        raise ProtocolSchemaError(
            f"{partition_role} capture failure requires active prefix "
            f"{expected.name}"
        )


def _omission_records(
    omissions: Mapping[str, str] | Iterable[tuple[str, str]],
) -> list[JsonValue]:
    items = omissions.items() if isinstance(omissions, Mapping) else omissions
    return [
        {"name": name, "reason": reason}
        for name, reason in sorted(items)
    ]


@dataclass(frozen=True, slots=True)
class ArtifactInventoryEntry:
    """Durability-verified inventory record for one materialized artifact."""

    name: str
    artifact_byte_sha256: str
    byte_count: int
    prefix: CampaignPrefix
    materialization_ordinal: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.name, label="inventory artifact name")
        _validate_sha256(
            self.artifact_byte_sha256,
            label="artifact_byte_sha256",
        )
        if _validate_uint64(self.byte_count, label="byte_count") == 0:
            raise ProtocolSchemaError("inventory artifact must be nonempty")
        if not isinstance(self.prefix, CampaignPrefix):
            raise ProtocolSchemaError("inventory prefix must be CampaignPrefix")
        _validate_uint64(
            self.materialization_ordinal,
            label="materialization_ordinal",
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_byte_sha256": self.artifact_byte_sha256,
            "byte_count": self.byte_count,
            "materialization_ordinal": self.materialization_ordinal,
            "name": self.name,
            "prefix": self.prefix.name,
        }


@dataclass(frozen=True, slots=True)
class PrefixClosure:
    """Mechanical P0-P12 reached-prefix audit."""

    last_completed_prefix: CampaignPrefix | None
    active_prefix: CampaignPrefix
    inventory: tuple[ArtifactInventoryEntry, ...]
    required_completed_artifacts: frozenset[str]
    missing_completed_artifacts: frozenset[str]
    durably_reached_active_artifacts: frozenset[str]
    omitted_active_artifacts: tuple[tuple[str, str], ...]
    forbidden_future_artifacts_present: frozenset[str]
    prefix_mismatches: frozenset[str]
    omitted_future_prefixes: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_completed_artifacts
            or self.forbidden_future_artifacts_present
            or self.prefix_mismatches
        )


def validate_reached_prefix_closure(
    inventory: Sequence[ArtifactInventoryEntry],
    *,
    last_completed_prefix: CampaignPrefix | str | None,
    active_prefix: CampaignPrefix | str,
    omitted_active_artifacts: Mapping[str, str],
) -> PrefixClosure:
    """Audit exact predecessor, completed, active, and future prefix rules."""

    active = _coerce_prefix(active_prefix, label="active_prefix")
    if last_completed_prefix is None:
        completed = None
        if active is not CampaignPrefix.P0:
            raise ProtocolSchemaError(
                "null last_completed_prefix is valid only with active P0"
            )
    else:
        completed = _coerce_prefix(
            last_completed_prefix, label="last_completed_prefix"
        )
        final_prefix_complete = (
            completed is CampaignPrefix.P12 and active is CampaignPrefix.P12
        )
        if not final_prefix_complete and completed.value + 1 != active.value:
            raise ProtocolSchemaError(
                "active_prefix must immediately follow last_completed_prefix"
            )
    entries = tuple(inventory)
    if not all(isinstance(entry, ArtifactInventoryEntry) for entry in entries):
        raise ProtocolSchemaError("inventory contains an invalid entry")
    names = tuple(entry.name for entry in entries)
    ordinals = tuple(entry.materialization_ordinal for entry in entries)
    if len(names) != len(set(names)):
        raise ProtocolSchemaError("inventory artifact names must be unique")
    if len(ordinals) != len(set(ordinals)):
        raise ProtocolSchemaError("materialization ordinals must be unique")
    ordered_entries = tuple(
        sorted(entries, key=lambda entry: entry.materialization_ordinal)
    )
    expected_ordinals = tuple(range(1, len(ordered_entries) + 1))
    if tuple(entry.materialization_ordinal for entry in ordered_entries) != expected_ordinals:
        raise ProtocolSchemaError(
            "materialization ordinals must be contiguous and begin at one"
        )
    present = frozenset(names)
    required: set[str] = set()
    if completed is not None:
        for prefix in CampaignPrefix:
            if prefix.value <= completed.value:
                required.update(_PREFIX_MANDATORY_ARTIFACTS[prefix])
    missing = frozenset(required - present)
    reached_active = frozenset(
        entry.name for entry in entries if entry.prefix is active
    )
    prefix_mismatch_names = {
        entry.name
        for entry in entries
        if entry.name in _ARTIFACT_PREFIX
        and _ARTIFACT_PREFIX[entry.name] is not entry.prefix
    }
    for previous, current in zip(ordered_entries, ordered_entries[1:]):
        if current.prefix.value < previous.prefix.value:
            prefix_mismatch_names.add(current.name)
    prefix_mismatches = frozenset(prefix_mismatch_names)
    future_present = frozenset(
        entry.name for entry in entries if entry.prefix.value > active.value
    )
    if not isinstance(omitted_active_artifacts, Mapping):
        raise ProtocolSchemaError("omitted_active_artifacts must be an object")
    normalized_omitted: list[tuple[str, str]] = []
    for name, reason in omitted_active_artifacts.items():
        _validate_relative_path(name, label="omitted artifact name")
        normalized_omitted.append(
            (name, _validate_identifier(reason, label="omission reason"))
        )
    omitted_names = frozenset(name for name, _ in normalized_omitted)
    active_missing = _PREFIX_MANDATORY_ARTIFACTS[active] - present
    if omitted_names != active_missing:
        raise ProtocolSchemaError(
            "omitted_active_artifacts must name every and only missing active "
            "mandatory artifact"
        )
    future_prefixes = tuple(
        prefix.name for prefix in CampaignPrefix if prefix.value > active.value
    )
    return PrefixClosure(
        last_completed_prefix=completed,
        active_prefix=active,
        inventory=ordered_entries,
        required_completed_artifacts=frozenset(required),
        missing_completed_artifacts=missing,
        durably_reached_active_artifacts=reached_active,
        omitted_active_artifacts=tuple(sorted(normalized_omitted)),
        forbidden_future_artifacts_present=future_present,
        prefix_mismatches=prefix_mismatches,
        omitted_future_prefixes=future_prefixes,
    )


def _normalize_count_map(
    value: Mapping[str, int], *, label: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise ProtocolSchemaError(f"{label} must be an object")
    normalized: list[tuple[str, int]] = []
    for key, count in value.items():
        normalized.append(
            (
                _validate_identifier(key, label=f"{label} key"),
                _validate_uint64(count, label=f"{label}[{key!r}]"),
            )
        )
    if len({key for key, _ in normalized}) != len(normalized):
        raise ProtocolSchemaError(f"{label} keys must be unique")
    return tuple(sorted(normalized))


def _normalize_identifier_sequence(
    value: Iterable[str], *, label: str
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ProtocolSchemaError(f"{label} must be an array")
    normalized = tuple(
        _validate_identifier(item, label=f"{label} item") for item in value
    )
    if len(normalized) != len(set(normalized)):
        raise ProtocolSchemaError(f"{label} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class SupportFailureSpec:
    """Sanitized, fully bound J3/K5 insufficient-support evidence.

    For development-confirm stages, ``active_chart_ids`` is the frozen set of
    materially active source charts to which the four-unit floor applies.
    """

    campaign_id: str
    partition_role: str
    last_completed_prefix: CampaignPrefix | None
    active_prefix: CampaignPrefix
    support_stage: str
    reason_code: SupportFailureReason
    attempted_locator_K_values: tuple[int, ...]
    attempted_representation_or_codec_ids: tuple[str, ...]
    analysis_unit_counts: tuple[tuple[str, int], ...]
    sequence_counts: tuple[tuple[str, int], ...]
    chart_counts: tuple[tuple[str, int], ...]
    chart_head_validation_counts: tuple[tuple[str, int], ...]
    active_chart_ids: tuple[str, ...]
    expected_head_ids: tuple[str, ...]
    fallback_attempts: tuple[str, ...]
    out_of_support_token_count: int | None
    evaluated_token_count: int | None

    def __init__(
        self,
        *,
        campaign_id: str,
        partition_role: str,
        last_completed_prefix: CampaignPrefix | str | None,
        active_prefix: CampaignPrefix | str,
        support_stage: str,
        reason_code: SupportFailureReason | str,
        attempted_locator_K_values: Iterable[int] = (),
        attempted_representation_or_codec_ids: Iterable[str] = (),
        analysis_unit_counts: Mapping[str, int] | None = None,
        sequence_counts: Mapping[str, int] | None = None,
        chart_counts: Mapping[str, int] | None = None,
        chart_head_validation_counts: Mapping[str, int] | None = None,
        active_chart_ids: Iterable[str] = (),
        expected_head_ids: Iterable[str] = (),
        fallback_attempts: Iterable[str] = (),
        out_of_support_token_count: int | None = None,
        evaluated_token_count: int | None = None,
    ) -> None:
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        if campaign == CURRENT_P0_CAMPAIGN_ID:
            raise ProtocolSchemaError(
                "the finalized current-P0 campaign cannot be reopened"
            )
        role = _validate_partition_role(partition_role)
        active = _coerce_prefix(active_prefix, label="active_prefix")
        if last_completed_prefix is None:
            completed = None
            if active is not CampaignPrefix.P0:
                raise ProtocolSchemaError(
                    "null last_completed_prefix is valid only with active P0"
                )
        else:
            completed = _coerce_prefix(
                last_completed_prefix, label="last_completed_prefix"
            )
            if completed.value + 1 != active.value:
                raise ProtocolSchemaError(
                    "active_prefix must immediately follow last_completed_prefix"
                )
        try:
            reason = (
                reason_code
                if isinstance(reason_code, SupportFailureReason)
                else SupportFailureReason(reason_code)
            )
        except ValueError as error:
            raise ProtocolSchemaError(
                "reason_code must be a registered support-failure reason"
            ) from error
        locator_values = tuple(attempted_locator_K_values)
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > _UINT32_MAX
            for item in locator_values
        ):
            raise ProtocolSchemaError(
                "attempted_locator_K_values must contain positive uint32 values"
            )
        if len(locator_values) != len(set(locator_values)):
            raise ProtocolSchemaError(
                "attempted_locator_K_values must not contain duplicates"
            )
        object.__setattr__(self, "campaign_id", campaign)
        object.__setattr__(self, "partition_role", role)
        object.__setattr__(self, "last_completed_prefix", completed)
        object.__setattr__(self, "active_prefix", active)
        object.__setattr__(
            self,
            "support_stage",
            _validate_identifier(support_stage, label="support_stage"),
        )
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "attempted_locator_K_values", locator_values)
        object.__setattr__(
            self,
            "attempted_representation_or_codec_ids",
            _normalize_identifier_sequence(
                attempted_representation_or_codec_ids,
                label="attempted_representation_or_codec_ids",
            ),
        )
        object.__setattr__(
            self,
            "analysis_unit_counts",
            _normalize_count_map(
                {} if analysis_unit_counts is None else analysis_unit_counts,
                label="analysis_unit_counts",
            ),
        )
        object.__setattr__(
            self,
            "sequence_counts",
            _normalize_count_map(
                {} if sequence_counts is None else sequence_counts,
                label="sequence_counts",
            ),
        )
        object.__setattr__(
            self,
            "chart_counts",
            _normalize_count_map(
                {} if chart_counts is None else chart_counts,
                label="chart_counts",
            ),
        )
        object.__setattr__(
            self,
            "chart_head_validation_counts",
            _normalize_count_map(
                (
                    {}
                    if chart_head_validation_counts is None
                    else chart_head_validation_counts
                ),
                label="chart_head_validation_counts",
            ),
        )
        object.__setattr__(
            self,
            "active_chart_ids",
            _normalize_identifier_sequence(
                active_chart_ids, label="active_chart_ids"
            ),
        )
        object.__setattr__(
            self,
            "expected_head_ids",
            _normalize_identifier_sequence(
                expected_head_ids, label="expected_head_ids"
            ),
        )
        object.__setattr__(
            self,
            "fallback_attempts",
            _normalize_identifier_sequence(
                fallback_attempts, label="fallback_attempts"
            ),
        )
        for value, label in (
            (out_of_support_token_count, "out_of_support_token_count"),
            (evaluated_token_count, "evaluated_token_count"),
        ):
            if value is not None:
                _validate_uint64(value, label=label)
        object.__setattr__(
            self,
            "out_of_support_token_count",
            out_of_support_token_count,
        )
        object.__setattr__(self, "evaluated_token_count", evaluated_token_count)
        try:
            expected_role, expected_completed, expected_active = (
                _SUPPORT_STAGE_CONTRACT[self.support_stage]
            )
        except KeyError as error:
            raise ProtocolSchemaError(
                "support_stage is not a registered reached-prefix stage"
            ) from error
        if (
            self.partition_role != expected_role
            or self.last_completed_prefix is not expected_completed
            or self.active_prefix is not expected_active
        ):
            raise ProtocolSchemaError(
                "support stage, partition, and reached prefixes are inconsistent"
            )
        if (
            self.reason_code
            is not SupportFailureReason.EXCESSIVE_OUT_OF_SUPPORT_RATE
            and (
                self.out_of_support_token_count is not None
                or self.evaluated_token_count is not None
            )
        ):
            raise ProtocolSchemaError(
                "out-of-support token counts are reserved for the excessive-rate reason"
            )
        if self.reason_code is SupportFailureReason.ZERO_FIT_MONITOR_CHART_SUPPORT:
            chart_counts = dict(self.chart_counts)
            head_counts = dict(self.chart_head_validation_counts)
            expected_head_count_keys = {
                f"{chart_id}__{head_id}"
                for chart_id in self.active_chart_ids
                for head_id in self.expected_head_ids
            }
            zero_charts = {
                chart_id for chart_id, count in chart_counts.items() if count == 0
            }
            if (
                self.support_stage
                not in {
                    "fit_monitor_chart_support",
                    "fit_monitor_chart_validation",
                }
                or not self.active_chart_ids
                or not self.expected_head_ids
                or set(chart_counts) != set(self.active_chart_ids)
                or set(head_counts) != expected_head_count_keys
                or not zero_charts
                or any(
                    head_counts[f"{chart_id}__{head_id}"] != 0
                    for chart_id in zero_charts
                    for head_id in self.expected_head_ids
                )
            ):
                raise ProtocolSchemaError(
                    "zero chart support requires a frozen chart/head universe, "
                    "an exact per-chart count map, and an explicit empty chart"
                )
        elif self.reason_code is SupportFailureReason.NO_POSITIVE_OPERATOR_REFERENCE:
            if (
                self.support_stage
                not in {
                    "operator_reference_construction",
                    "per_codec_operator_reference_construction",
                }
                or not self.attempted_representation_or_codec_ids
            ):
                raise ProtocolSchemaError(
                    "no-positive-reference requires operator-reference evidence"
                )
        elif self.reason_code is SupportFailureReason.LOWER_K_FALLBACK_EXHAUSTED:
            if (
                self.support_stage != "lower_K_representation_fallback"
                or not self.attempted_locator_K_values
                or not self.fallback_attempts
            ):
                raise ProtocolSchemaError(
                    "fallback exhaustion requires attempted K values and fallback evidence"
                )
        elif self.reason_code is SupportFailureReason.INSUFFICIENT_INDEPENDENT_CHART_SUPPORT:
            chart_counts = dict(self.chart_counts)
            head_counts = dict(self.chart_head_validation_counts)
            expected_head_count_keys = {
                f"{chart_id}__{head_id}"
                for chart_id in self.active_chart_ids
                for head_id in self.expected_head_ids
            }
            development_stage = self.support_stage in {
                "development_confirm_support",
                "development_confirm_chart_head_support",
                "development_confirm_post_evaluation_support",
            }
            if development_stage:
                development_count = dict(self.analysis_unit_counts).get(
                    "development_confirm"
                )
                threshold_failed = (
                    development_count is not None
                    and (
                        development_count < 18
                        or any(count < 4 for count in chart_counts.values())
                    )
                )
            else:
                threshold_failed = any(
                    count < 2 for count in chart_counts.values()
                )
            if (
                self.support_stage
                not in {
                    "fit_monitor_chart_support",
                    "fit_monitor_chart_validation",
                    "mandatory_chart_head_support",
                    "mandatory_chart_head_support_checks",
                    "development_confirm_support",
                    "development_confirm_chart_head_support",
                    "development_confirm_post_evaluation_support",
                }
                or not self.active_chart_ids
                or not self.expected_head_ids
                or set(chart_counts) != set(self.active_chart_ids)
                or set(head_counts) != expected_head_count_keys
                or not threshold_failed
            ):
                raise ProtocolSchemaError(
                    "insufficient chart support requires an exact frozen "
                    "chart/head count matrix and a failed support threshold"
                )
        elif self.reason_code is SupportFailureReason.EXCESSIVE_OUT_OF_SUPPORT_RATE:
            if (
                self.support_stage
                not in {
                    "development_confirm_support",
                    "development_confirm_chart_head_support",
                    "development_confirm_post_evaluation_support",
                }
                or not self.analysis_unit_counts
                or not self.sequence_counts
                or self.out_of_support_token_count is None
                or self.evaluated_token_count is None
                or self.evaluated_token_count == 0
                or self.out_of_support_token_count > self.evaluated_token_count
                or self.out_of_support_token_count * 10
                <= self.evaluated_token_count
            ):
                raise ProtocolSchemaError(
                    "excessive out-of-support rate requires development-confirm "
                    "unit/sequence evidence and exact token counts above ten percent"
                )
    @staticmethod
    def _count_dict(value: tuple[tuple[str, int], ...]) -> dict[str, JsonValue]:
        return {key: count for key, count in value}

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "active_prefix": self.active_prefix.name,
            "active_chart_ids": list(self.active_chart_ids),
            "analysis_unit_counts": self._count_dict(self.analysis_unit_counts),
            "attempted_locator_K_values": list(self.attempted_locator_K_values),
            "attempted_representation_or_codec_ids": list(
                self.attempted_representation_or_codec_ids
            ),
            "campaign_id": self.campaign_id,
            "chart_counts": self._count_dict(self.chart_counts),
            "chart_head_validation_counts": self._count_dict(
                self.chart_head_validation_counts
            ),
            "fallback_attempts": list(self.fallback_attempts),
            "expected_head_ids": list(self.expected_head_ids),
            "last_completed_prefix": (
                None
                if self.last_completed_prefix is None
                else self.last_completed_prefix.name
            ),
            "partition_role": self.partition_role,
            "reason_code": self.reason_code.value,
            "out_of_support_token_count": self.out_of_support_token_count,
            "evaluated_token_count": self.evaluated_token_count,
            "sequence_counts": self._count_dict(self.sequence_counts),
            "support_stage": self.support_stage,
        }


_INTEGRITY_REASONS = frozenset(
    {
        "artifact_hash_mismatch",
        "authorization_mismatch",
        "capture_attempt_identity_mismatch",
        "capture_row_order_mismatch",
        "capture_root_identity_mismatch",
        "changed_row_derivation_identity",
        "duplicate_row_id",
        "malformed_sequence_identity",
        "panel_row_source_identity_mismatch",
        "precomputed_row_index_mismatch",
        "protocol_dimension_mismatch",
        "row_derivation_protocol_hash_mismatch",
        "row_index_payload_hash_mismatch",
        "row_set_mismatch",
        "safetensors_format_mismatch",
        "shard_file_missing",
        "shard_file_mutated_during_verification",
        "shard_hash_mismatch",
        "shard_not_regular_file",
        "shard_path_escape",
        "shard_plan_mismatch",
        "shard_range_mismatch",
        "shard_symlink_forbidden",
        "tokenizer_identity_mismatch",
    }
)
_MEASUREMENT_REASONS = frozenset(
    {
        "inconsistent_L1_target",
        "invalid_tensor_dtype",
        "invalid_tensor_shape",
        "invalid_token_position",
        "mask_contract_violation",
        "missing_head_in_validation_rows",
        "nonfinite_operator_value",
        "rmsnorm_execution_contract_violation",
        "row_derivation_execution_failure",
        "tensor_row_identity_alignment_mismatch",
        "unable_to_derive_expected_panel_rows",
        "verified_tokenizer_execution_failure",
        "wrong_head_shape",
    }
)
_RESOURCE_REASONS = frozenset(
    {
        "capture_resource_limit",
        "memory_resource_limit",
        "row_derivation_resource_limit",
        "storage_resource_limit",
        "time_resource_limit",
    }
)
_SAFE_FAILURE_DETAIL_KEYS = frozenset(
    {
        "artifact_name",
        "active_prefix",
        "attempted_count",
        "available",
        "chart_id",
        "distinct_unit_count",
        "end_row_ordinal_exclusive",
        "errno",
        "error_code",
        "exception_type",
        "failure_stage",
        "expected_count",
        "expected_dtype",
        "expected_hash",
        "expected_head_id",
        "expected_mode",
        "expected_sha256",
        "expected_shape",
        "expected_state",
        "head_id",
        "last_completed_prefix",
        "limit",
        "observed_count",
        "observed_dtype",
        "observed_hash",
        "observed_head_id",
        "observed_mode",
        "observed_sha256",
        "observed_shape",
        "observed_state",
        "partition_role",
        "protocol_field",
        "requested",
        "required_count",
        "resource_kind",
        "reason_code",
        "row_id",
        "row_ordinal",
        "shard_index",
        "source_token_position",
        "stage",
        "start_row_ordinal",
        "support_count",
        "target_token_position",
        "tensor_name",
    }
)
_SAFE_SEQUENCE_FAILURE_KEYS = frozenset({"expected_shape", "observed_shape"})
_SAFE_HASH_FAILURE_KEYS = frozenset(
    {"expected_hash", "observed_hash", "expected_sha256", "observed_sha256"}
)
_SAFE_INTEGER_FAILURE_KEYS = frozenset(
    {
        "attempted_count",
        "available",
        "distinct_unit_count",
        "end_row_ordinal_exclusive",
        "errno",
        "expected_count",
        "limit",
        "observed_count",
        "requested",
        "required_count",
        "row_ordinal",
        "shard_index",
        "source_token_position",
        "start_row_ordinal",
        "support_count",
        "target_token_position",
    }
)
_SAFE_IDENTIFIER_FAILURE_KEYS = (
    _SAFE_FAILURE_DETAIL_KEYS
    - _SAFE_SEQUENCE_FAILURE_KEYS
    - _SAFE_HASH_FAILURE_KEYS
    - _SAFE_INTEGER_FAILURE_KEYS
    - {"partition_role", "row_id", "active_prefix", "last_completed_prefix"}
)
_PROTECTED_BINDING_KEY_RE = re.compile(
    r"(?:^|_)(?:prompt|message|response|completion|raw_text|raw_content|token_ids?|tokens|"
    r"activation_values?|logits?|hidden_states?|embedding_values?|"
    r"tensor_values?|sequence_content)(?:_|$)"
)
_SAFE_PROTECTED_METADATA_SUFFIXES = (
    "_sha256",
    "_hash",
    "_count",
    "_counts",
    "_shape",
    "_dtype",
    "_protocol",
    "_protocol_id",
    "_identity",
    "_identity_id",
    "_ordinal",
    "_index",
    "_indexes",
)
_ALWAYS_RAW_PROTECTED_KEYS = frozenset(
    {
        "token_id",
        "token_ids",
        "source_token_id",
        "source_token_ids",
        "target_token_id",
        "target_token_ids",
    }
)
_SAFE_PUBLIC_PROTECTED_METADATA_KEYS = frozenset(
    {
        # Frozen scientific metadata fields.  Their schemas contain only
        # dimensions, ranks, rules, booleans, or file/hash identities; none is
        # a model-input payload despite containing a protected-word token.
        "completion_target_null_passed",
        # This is a derived scalar diagnostic, not captured completion data.
        "coefficient_completion_mse",
        "full_prompt_shadow",
        "message_dimension",
        "message_rank",
        "message_rank_per_head",
        "message_reference_scale",
        "message_roundtrip_passed",
        "prompt_tail_rule",
        "retained_message_dimension",
        "special_tokens_map_file",
        "transported_message_dimension",
        # Exact artifact names appear as keys inside dependency_hashes.
        "message_sketch_manifest_p32.json",
        "o7_attention_output_completion_results.json",
    }
)


def _is_protected_raw_value_key(key: str) -> bool:
    if not _PROTECTED_BINDING_KEY_RE.search(key):
        return False
    if key in _ALWAYS_RAW_PROTECTED_KEYS:
        return True
    if key in _SAFE_PUBLIC_PROTECTED_METADATA_KEYS:
        return False
    return not key.endswith(_SAFE_PROTECTED_METADATA_SUFFIXES)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    keys = frozenset(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise ProtocolSchemaError(
            f"{label} keys differ: missing={missing}, unknown={unknown}"
        )


def _validate_json_value(
    value: object,
    *,
    path: str = "$",
    depth: int = 0,
    item_counter: list[int] | None = None,
) -> None:
    if item_counter is None:
        item_counter = [0]
    if depth > _MAX_SAFE_JSON_DEPTH:
        raise ProtocolSchemaError(f"{path} exceeds maximum JSON depth")
    item_counter[0] += 1
    if item_counter[0] > _MAX_SAFE_JSON_ITEMS:
        raise ProtocolSchemaError("JSON value exceeds maximum item count")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if isinstance(value, bool) or not (-(1 << 63) <= value <= _UINT64_MAX):
            raise ProtocolSchemaError(f"{path} integer is outside protocol range")
        return
    if isinstance(value, float):
        raise ProtocolSchemaError(f"{path} floats are forbidden")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                item_counter=item_counter,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or (
                    not _KEY_RE.fullmatch(key)
                    and key not in _CANONICAL_NONLOWER_KEYS
                )
            ):
                raise ProtocolSchemaError(f"{path} contains invalid key {key!r}")
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                item_counter=item_counter,
            )
        return
    raise ProtocolSchemaError(
        f"{path} has forbidden JSON type {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode the protocol's deterministic, float-free JSON subset."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def artifact_sha256(content: bytes) -> str:
    """Return the ordinary exact-byte SHA-256 used by artifact inventories."""

    if not isinstance(content, bytes):
        raise ProtocolSchemaError("artifact content must be immutable bytes")
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Hash one file by exact bytes without loading it all into memory."""

    if isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ProtocolSchemaError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_partition_role(value: str) -> str:
    if value not in _PARTITION_ROLES:
        raise ProtocolSchemaError(
            f"partition_role must be one of {sorted(_PARTITION_ROLES)}"
        )
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProtocolSchemaError(f"{label} must be 64 lowercase hex characters")
    return value


def _validate_tx_id(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _TX_ID_RE.fullmatch(value):
        raise ProtocolSchemaError(f"{label} must be 32 lowercase hex characters")
    return value


def _validate_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ProtocolSchemaError(f"{label} must be a bounded ASCII identifier")
    return value


def _validate_uint64(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolSchemaError(f"{label} must be an integer")
    if not 0 <= value <= _UINT64_MAX:
        raise ProtocolSchemaError(f"{label} must fit uint64")
    return value


def _validate_uint32(value: int, *, label: str) -> int:
    _validate_uint64(value, label=label)
    if value > _UINT32_MAX:
        raise ProtocolSchemaError(f"{label} must fit uint32")
    return value


def _raw_digest(value: str, *, label: str) -> bytes:
    return bytes.fromhex(_validate_sha256(value, label=label))


def _normalize_row_id(value: str | bytes, *, label: str = "row_id") -> bytes:
    if isinstance(value, bytes):
        if len(value) != 32:
            raise ProtocolSchemaError(f"{label} bytes must have length 32")
        return value
    if isinstance(value, str):
        return _raw_digest(value, label=label)
    raise ProtocolSchemaError(f"{label} must be raw bytes or lowercase hex")


def _failure_exception(reason_code: str, message: str) -> ProtocolViolation:
    disposition = classify_failure(reason_code)
    if disposition.failure_class is FailureClass.INTEGRITY:
        return IntegrityViolation(reason_code, message)
    if disposition.failure_class is FailureClass.MEASUREMENT_CONTRACT:
        return MeasurementContractViolation(reason_code, message)
    return ResourceLimitViolation(reason_code, message)


@dataclass(frozen=True, slots=True)
class FailureDisposition:
    """Exclusive class and terminal state for one exact reason code."""

    reason_code: str
    failure_class: FailureClass
    terminal_state: TerminalState


def classify_failure(reason_code: str) -> FailureDisposition:
    """Classify a known failure reason; unknown reasons fail closed."""

    if reason_code in _INTEGRITY_REASONS:
        return FailureDisposition(
            reason_code,
            FailureClass.INTEGRITY,
            TerminalState.INVALID_CAMPAIGN_INTEGRITY,
        )
    if reason_code in _MEASUREMENT_REASONS:
        return FailureDisposition(
            reason_code,
            FailureClass.MEASUREMENT_CONTRACT,
            TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT,
        )
    if reason_code in _RESOURCE_REASONS:
        return FailureDisposition(
            reason_code,
            FailureClass.RESOURCE_LIMIT,
            TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED,
        )
    raise ProtocolSchemaError(f"unknown failure reason_code {reason_code!r}")


def _validate_safe_scalar(value: object, *, label: str) -> JsonScalar:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return _validate_uint64(value, label=label)
    if isinstance(value, str):
        return _validate_identifier(value, label=label)
    if isinstance(value, float):
        raise ProtocolSchemaError(f"{label} floats, including nonfinite values, are forbidden")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ProtocolSchemaError(f"{label} raw bytes are forbidden")
    raise ProtocolSchemaError(f"{label} must be sanitized scalar metadata")


def validate_sanitized_failure_details(
    details: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    """Validate a non-extensible metadata-only failure-detail schema.

    There is intentionally no free-form message field.  Keys are allowlisted,
    strings must be short identifier-like values, raw bytes and floats are
    rejected, and only tensor *shape* arrays are accepted.
    """

    if details is None:
        return {}
    if not isinstance(details, Mapping):
        raise ProtocolSchemaError("sanitized_failure_details must be an object")
    result: dict[str, JsonValue] = {}
    for key, value in details.items():
        if key not in _SAFE_FAILURE_DETAIL_KEYS:
            raise ProtocolSchemaError(
                f"sanitized_failure_details key {key!r} is not allowlisted"
            )
        if key in _SAFE_SEQUENCE_FAILURE_KEYS:
            if not isinstance(value, (list, tuple)):
                raise ProtocolSchemaError(f"{key} must be an integer array")
            shape: list[JsonValue] = []
            for index, dimension in enumerate(value):
                shape.append(
                    _validate_uint64(dimension, label=f"{key}[{index}]")
                )
            result[key] = shape
        elif key in _SAFE_HASH_FAILURE_KEYS:
            if not isinstance(value, str):
                raise ProtocolSchemaError(f"{key} must be a lowercase SHA-256")
            result[key] = _validate_sha256(value, label=key)
        elif key in _SAFE_INTEGER_FAILURE_KEYS:
            result[key] = _validate_uint64(value, label=key)
        elif key == "partition_role":
            if not isinstance(value, str):
                raise ProtocolSchemaError("partition_role must be a string")
            result[key] = _validate_partition_role(value)
        elif key == "row_id":
            if not isinstance(value, str):
                raise ProtocolSchemaError("row_id must be raw32 hexadecimal")
            result[key] = _normalize_row_id(value).hex()
        elif key in {"active_prefix", "last_completed_prefix"}:
            if not isinstance(value, (str, CampaignPrefix)):
                raise ProtocolSchemaError(f"{key} must be P0 through P12")
            result[key] = _coerce_prefix(value, label=key).name
        elif key in _SAFE_IDENTIFIER_FAILURE_KEYS:
            if not isinstance(value, str):
                raise ProtocolSchemaError(f"{key} must be an identifier string")
            result[key] = _validate_identifier(value, label=key)
        else:
            raise ProtocolSchemaError(f"{key} has no registered field type")
    canonical_json_bytes(result)
    return result


def _validate_failure_detail_context(
    details: Mapping[str, JsonValue],
    *,
    partition_role: str,
    reason_code: str,
    last_completed_prefix: str,
    active_prefix: str,
    failure_stage: str,
) -> None:
    expected: dict[str, str] = {
        "partition_role": partition_role,
        "reason_code": reason_code,
        "last_completed_prefix": last_completed_prefix,
        "active_prefix": active_prefix,
        "failure_stage": failure_stage,
        "stage": failure_stage,
    }
    for key, expected_value in expected.items():
        if key in details and details[key] != expected_value:
            raise ProtocolSchemaError(
                f"sanitized failure detail {key} differs from receipt context"
            )


def _normalize_safe_bindings(bindings: Mapping[str, object]) -> dict[str, JsonValue]:
    if not isinstance(bindings, Mapping):
        raise ProtocolSchemaError("frozen_bindings must be an object")

    def normalize(value: object, *, path: str, depth: int) -> JsonValue:
        if depth > 8:
            raise ProtocolSchemaError(f"{path} exceeds binding depth")
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return _validate_uint64(value, label=path)
        if isinstance(value, str):
            return _validate_identifier(value, label=path)
        if isinstance(value, float):
            raise ProtocolSchemaError(f"{path} floats are forbidden")
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise ProtocolSchemaError(f"{path} raw bytes are forbidden")
        if isinstance(value, (list, tuple)):
            return [
                normalize(item, path=f"{path}[{index}]", depth=depth + 1)
                for index, item in enumerate(value)
            ]
        if isinstance(value, Mapping):
            normalized: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
                    raise ProtocolSchemaError(f"{path} contains invalid key {key!r}")
                if _PROTECTED_BINDING_KEY_RE.search(key):
                    raise ProtocolSchemaError(
                        f"{path}.{key} is a protected-value field; bind a hash instead"
                    )
                normalized[key] = normalize(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )
            return normalized
        raise ProtocolSchemaError(f"{path} has forbidden type {type(value).__name__}")

    output = normalize(bindings, path="frozen_bindings", depth=0)
    assert isinstance(output, dict)
    canonical_json_bytes(output)
    return output


def derive_row_id(
    analysis_unit_commitment: str,
    canonical_sequence_hash: str,
    source_token_position: int,
    target_token_position: int,
) -> str:
    """Derive one causal L-1 evaluation-row ID exactly as specified."""

    source = _validate_uint64(
        source_token_position, label="source_token_position"
    )
    target = _validate_uint64(
        target_token_position, label="target_token_position"
    )
    if source == _UINT64_MAX or target != source + 1:
        raise _failure_exception(
            "inconsistent_L1_target",
            "target_token_position must equal source_token_position + 1",
        )
    payload = (
        _ROW_ID_DOMAIN
        + _raw_digest(
            analysis_unit_commitment, label="analysis_unit_commitment"
        )
        + _raw_digest(
            canonical_sequence_hash, label="canonical_sequence_hash"
        )
        + struct.pack(">Q", source)
        + struct.pack(">Q", target)
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """A protected-value-free row identity input."""

    analysis_unit_commitment: str
    canonical_sequence_hash: str
    source_token_position: int
    target_token_position: int

    def __post_init__(self) -> None:
        _validate_sha256(
            self.analysis_unit_commitment,
            label="analysis_unit_commitment",
        )
        _validate_sha256(
            self.canonical_sequence_hash,
            label="canonical_sequence_hash",
        )
        # Derivation validates both positions and the causal L-1 relation.
        derive_row_id(
            self.analysis_unit_commitment,
            self.canonical_sequence_hash,
            self.source_token_position,
            self.target_token_position,
        )

    @property
    def row_id(self) -> str:
        return derive_row_id(
            self.analysis_unit_commitment,
            self.canonical_sequence_hash,
            self.source_token_position,
            self.target_token_position,
        )

    @property
    def row_id_raw(self) -> bytes:
        return bytes.fromhex(self.row_id)

    def as_manifest_record(self, row_ordinal: int) -> dict[str, JsonValue]:
        _validate_uint64(row_ordinal, label="row_ordinal")
        return {
            "analysis_unit_commitment": self.analysis_unit_commitment,
            "canonical_sequence_hash": self.canonical_sequence_hash,
            "row_id": self.row_id,
            "row_ordinal": row_ordinal,
            "source_token_position": self.source_token_position,
            "target_token_position": self.target_token_position,
        }


def canonicalize_rows(rows: Iterable[EvaluationRow]) -> tuple[EvaluationRow, ...]:
    """Sort by raw row-ID bytes and reject duplicates."""

    materialized = tuple(rows)
    if not materialized:
        raise ProtocolSchemaError("a row index must contain at least one row")
    if not all(isinstance(row, EvaluationRow) for row in materialized):
        raise ProtocolSchemaError("rows must contain only EvaluationRow values")
    ordered = tuple(sorted(materialized, key=lambda row: row.row_id_raw))
    previous: bytes | None = None
    for row in ordered:
        raw = row.row_id_raw
        if previous == raw:
            raise _failure_exception(
                "duplicate_row_id", f"duplicate row ID {raw.hex()}"
            )
        previous = raw
    return ordered


def row_set_sha256(row_ids: Sequence[str | bytes]) -> str:
    """Commit the unique row-ID set in ascending raw-byte order."""

    raw_ids = sorted(_normalize_row_id(item) for item in row_ids)
    if not raw_ids:
        raise ProtocolSchemaError("row set must not be empty")
    for previous, current in zip(raw_ids, raw_ids[1:]):
        if previous == current:
            raise _failure_exception("duplicate_row_id", "duplicate row ID")
    digest = hashlib.sha256()
    digest.update(_ROW_SET_DOMAIN)
    digest.update(struct.pack(">Q", len(raw_ids)))
    for raw in raw_ids:
        digest.update(raw)
    return digest.hexdigest()


def ordered_row_sequence_sha256(row_ids: Sequence[str | bytes]) -> str:
    """Commit row IDs in their supplied physical order."""

    raw_ids = tuple(_normalize_row_id(item) for item in row_ids)
    if not raw_ids:
        raise ProtocolSchemaError("ordered row sequence must not be empty")
    digest = hashlib.sha256()
    digest.update(_ROW_SEQUENCE_DOMAIN)
    digest.update(struct.pack(">Q", len(raw_ids)))
    for raw in raw_ids:
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RowIndexPayload:
    """Canonical protected-value-free row-index payload."""

    partition_role: str
    row_derivation_protocol_hash: str
    rows: tuple[EvaluationRow, ...]

    def __post_init__(self) -> None:
        _validate_partition_role(self.partition_role)
        _validate_sha256(
            self.row_derivation_protocol_hash,
            label="row_derivation_protocol_hash",
        )
        canonical = canonicalize_rows(self.rows)
        if self.rows != canonical:
            raise ProtocolSchemaError(
                "RowIndexPayload.rows must already use ascending raw row-ID order"
            )

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.rows)

    @property
    def row_set_sha256(self) -> str:
        return row_set_sha256(self.row_ids)

    @property
    def ordered_row_sequence_sha256(self) -> str:
        return ordered_row_sequence_sha256(self.row_ids)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "format_version": _FORMAT_VERSION,
            "ordered_row_sequence_sha256": self.ordered_row_sequence_sha256,
            "partition_role": self.partition_role,
            "row_count": self.row_count,
            "row_derivation_protocol_hash": self.row_derivation_protocol_hash,
            "row_set_sha256": self.row_set_sha256,
            "rows": [
                row.as_manifest_record(index)
                for index, row in enumerate(self.rows)
            ],
            "schema": ROW_INDEX_SCHEMA,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def payload_sha256(self) -> str:
        return row_index_payload_sha256(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RowIndexPayload":
        expected = frozenset(
            {
                "format_version",
                "ordered_row_sequence_sha256",
                "partition_role",
                "row_count",
                "row_derivation_protocol_hash",
                "row_set_sha256",
                "rows",
                "schema",
            }
        )
        _require_exact_keys(value, expected, label="row-index payload")
        if value["schema"] != ROW_INDEX_SCHEMA or value["format_version"] != 1:
            raise ProtocolSchemaError("unsupported row-index schema/version")
        raw_rows = value["rows"]
        if not isinstance(raw_rows, list):
            raise ProtocolSchemaError("row-index rows must be an array")
        rows: list[EvaluationRow] = []
        row_expected = frozenset(
            {
                "analysis_unit_commitment",
                "canonical_sequence_hash",
                "row_id",
                "row_ordinal",
                "source_token_position",
                "target_token_position",
            }
        )
        for ordinal, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                raise ProtocolSchemaError("row-index row must be an object")
            _require_exact_keys(raw_row, row_expected, label="row-index row")
            if raw_row["row_ordinal"] != ordinal:
                raise ProtocolSchemaError("row_ordinal is not contiguous")
            row = EvaluationRow(
                analysis_unit_commitment=str(raw_row["analysis_unit_commitment"]),
                canonical_sequence_hash=str(raw_row["canonical_sequence_hash"]),
                source_token_position=_coerce_exact_int(
                    raw_row["source_token_position"], "source_token_position"
                ),
                target_token_position=_coerce_exact_int(
                    raw_row["target_token_position"], "target_token_position"
                ),
            )
            if raw_row["row_id"] != row.row_id:
                raise _failure_exception(
                    "changed_row_derivation_identity",
                    "stored row_id does not match its derivation inputs",
                )
            rows.append(row)
        payload = cls(
            partition_role=str(value["partition_role"]),
            row_derivation_protocol_hash=str(
                value["row_derivation_protocol_hash"]
            ),
            rows=tuple(rows),
        )
        if value["row_count"] != payload.row_count:
            raise ProtocolSchemaError("row_count does not match rows")
        if value["row_set_sha256"] != payload.row_set_sha256:
            raise _failure_exception("row_set_mismatch", "row-set hash mismatch")
        if (
            value["ordered_row_sequence_sha256"]
            != payload.ordered_row_sequence_sha256
        ):
            raise _failure_exception(
                "capture_row_order_mismatch", "ordered row hash mismatch"
            )
        return payload

    @classmethod
    def from_canonical_bytes(cls, content: bytes) -> "RowIndexPayload":
        try:
            value = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolSchemaError("invalid row-index JSON bytes") from error
        if not isinstance(value, dict):
            raise ProtocolSchemaError("row-index payload must be an object")
        payload = cls.from_mapping(value)
        if content != payload.canonical_bytes:
            raise ProtocolSchemaError("row-index bytes are not canonical")
        return payload


def _coerce_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolSchemaError(f"{label} must be an integer")
    return value


def build_row_index_payload(
    partition_role: str,
    row_derivation_protocol_hash: str,
    rows: Iterable[EvaluationRow],
) -> RowIndexPayload:
    """Sort, deduplicate, and build a canonical row-index payload."""

    return RowIndexPayload(
        partition_role=partition_role,
        row_derivation_protocol_hash=row_derivation_protocol_hash,
        rows=canonicalize_rows(rows),
    )


def row_index_payload_sha256(payload: RowIndexPayload) -> str:
    """Hash a canonical row-index payload with its protocol domain."""

    if not isinstance(payload, RowIndexPayload):
        raise ProtocolSchemaError("payload must be RowIndexPayload")
    return hashlib.sha256(
        _ROW_INDEX_PAYLOAD_DOMAIN + payload.canonical_bytes
    ).hexdigest()


_COMMON_AUTHORIZATION_BINDING_KEYS = frozenset(
    {
        "semantic_canonicalization_spec_sha256",
        "corpus_role_classification_sha256",
        "independence_protocol_sha256",
        "split_manifest_sha256",
        "source_inventory_sha256",
        "model_identity_sha256",
        "model_immutable_revision",
        "model_weight_file_byte_sha256s",
        "model_config_byte_sha256",
        "tokenizer_identity_sha256",
        "tokenizer_file_byte_sha256s",
        "tokenizer_config_byte_sha256",
        "special_vocabulary_map_byte_sha256",
        "carrier_identity_sha256",
        "h4_carrier_byte_sha256",
        "runtime_identity_sha256",
        "execution_budget_sha256",
        "row_derivation_protocol_artifact_sha256",
        "row_derivation_protocol_semantic_sha256",
        "canonical_content_sha256s",
        "content_set_sha256",
        "analysis_unit_commitments",
        "analysis_unit_set_sha256",
        "expected_sequence_count",
        "expected_analysis_unit_count",
        "row_count",
        "row_index_payload_sha256",
        "partition_commitment",
        "maximum_rows_per_shard",
        "capture_schema_hash",
        "capture_implementation_hash",
    }
)
_TUNE_AUTHORIZATION_BINDING_KEYS = frozenset(
    {
        "campaign_root_sha256",
        "selection_representation_manifest_sha256",
        "codec_protocol_sha256",
        "ceiling_gate_protocol_sha256",
    }
)
_DEVELOPMENT_AUTHORIZATION_BINDING_KEYS = frozenset(
    {
        "campaign_root_sha256",
        "frozen_representation_manifest_sha256",
        "selected_codec_artifact_sha256",
        "selected_codec_id",
        "candidate_bundle_byte_sha256",
        "comparator_bundle_byte_sha256",
        "candidate_bundle_metadata_sha256",
        "candidate_bundle_metadata_canonical_payload_sha256",
        "frozen_b_star_model_id",
        "frozen_b_star_artifact_sha256",
        "nonselecting_tune_replay_sha256",
        "nonselecting_tune_replay_result_sha256",
        "development_eligibility_protocol_sha256",
    }
)
_COMMON_AUTHORIZATION_SHA_KEYS = frozenset(
    {
        key
        for key in _COMMON_AUTHORIZATION_BINDING_KEYS
        if key.endswith("_sha256") or key.endswith("_hash")
    }
    - {"model_weight_file_byte_sha256s", "tokenizer_file_byte_sha256s"}
)


def _f19_set_sha256(values: Sequence[str], *, domain: bytes, label: str) -> str:
    digests = [
        _raw_digest(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if not digests:
        raise ProtocolSchemaError(f"{label} must not be empty")
    if len(set(digests)) != len(digests):
        raise ProtocolSchemaError(f"{label} must not contain duplicates")
    ordered = sorted(digests)
    return hashlib.sha256(
        domain + b"\0" + struct.pack(">Q", len(ordered)) + b"".join(ordered)
    ).hexdigest()


def _f19_partition_commitment(
    partition_role: str,
    *,
    content_set_sha256: str,
    analysis_unit_set_sha256: str,
) -> str:
    role = _validate_partition_role(partition_role)
    return hashlib.sha256(
        _PARTITION_COMMITMENT_DOMAIN
        + b"\0"
        + role.encode("utf-8")
        + b"\0"
        + _raw_digest(content_set_sha256, label="content_set_sha256")
        + _raw_digest(analysis_unit_set_sha256, label="analysis_unit_set_sha256")
    ).hexdigest()


def _validate_authorization_bindings(
    partition_role: str,
    bindings: Mapping[str, JsonValue],
    *,
    row_derivation_protocol_hash: str,
    expected_row_count: int | None,
    expected_row_index_payload_sha256: str | None,
) -> None:
    required = set(_COMMON_AUTHORIZATION_BINDING_KEYS)
    if partition_role == "tune":
        required.update(_TUNE_AUTHORIZATION_BINDING_KEYS)
    elif partition_role == "development_confirm":
        required.update(_DEVELOPMENT_AUTHORIZATION_BINDING_KEYS)
    missing = sorted(required - set(bindings))
    if missing:
        raise ProtocolSchemaError(
            f"frozen_bindings missing required {partition_role} keys: {missing!r}"
        )

    sha_keys = set(_COMMON_AUTHORIZATION_SHA_KEYS)
    if partition_role == "tune":
        sha_keys.update(_TUNE_AUTHORIZATION_BINDING_KEYS)
    elif partition_role == "development_confirm":
        sha_keys.update(
            key
            for key in _DEVELOPMENT_AUTHORIZATION_BINDING_KEYS
            if key.endswith("_sha256")
        )
    for key in sorted(sha_keys):
        _validate_sha256(bindings[key], label=f"frozen_bindings.{key}")
    _validate_sha256(
        bindings["model_immutable_revision"],
        label="frozen_bindings.model_immutable_revision",
    )

    for key in ("model_weight_file_byte_sha256s", "tokenizer_file_byte_sha256s"):
        values = bindings[key]
        if not isinstance(values, list) or not values:
            raise ProtocolSchemaError(f"frozen_bindings.{key} must be nonempty")
        for index, value in enumerate(values):
            _validate_sha256(value, label=f"frozen_bindings.{key}[{index}]")

    content_values = bindings["canonical_content_sha256s"]
    unit_values = bindings["analysis_unit_commitments"]
    if not isinstance(content_values, list) or not isinstance(unit_values, list):
        raise ProtocolSchemaError(
            "frozen content and analysis-unit identities must be arrays"
        )
    for label, values in (
        ("canonical_content_sha256s", content_values),
        ("analysis_unit_commitments", unit_values),
    ):
        for index, value in enumerate(values):
            _validate_sha256(value, label=f"frozen_bindings.{label}[{index}]")
        if values != sorted(set(values)):
            raise ProtocolSchemaError(
                f"frozen_bindings.{label} must be unique and raw-digest ordered"
            )

    sequence_count = _validate_uint64(
        bindings["expected_sequence_count"],
        label="frozen_bindings.expected_sequence_count",
    )
    unit_count = _validate_uint64(
        bindings["expected_analysis_unit_count"],
        label="frozen_bindings.expected_analysis_unit_count",
    )
    row_count = _validate_uint64(
        bindings["row_count"], label="frozen_bindings.row_count"
    )
    maximum_rows = _validate_uint64(
        bindings["maximum_rows_per_shard"],
        label="frozen_bindings.maximum_rows_per_shard",
    )
    if (
        sequence_count == 0
        or unit_count == 0
        or row_count == 0
        or maximum_rows == 0
        or maximum_rows > _MAXIMUM_SHARD_ROWS
    ):
        raise ProtocolSchemaError(
            "frozen authorization counts and shard limit must be positive and bounded"
        )
    if sequence_count != len(content_values) or unit_count != len(unit_values):
        raise ProtocolSchemaError(
            "frozen sequence/unit counts contradict their individual identities"
        )

    content_set = _f19_set_sha256(
        content_values,
        domain=_CONTENT_SET_DOMAIN,
        label="frozen_bindings.canonical_content_sha256s",
    )
    unit_set = _f19_set_sha256(
        unit_values,
        domain=_ANALYSIS_UNIT_SET_DOMAIN,
        label="frozen_bindings.analysis_unit_commitments",
    )
    if bindings["content_set_sha256"] != content_set:
        raise ProtocolSchemaError("frozen content-set commitment mismatch")
    if bindings["analysis_unit_set_sha256"] != unit_set:
        raise ProtocolSchemaError("frozen analysis-unit-set commitment mismatch")
    if bindings["partition_commitment"] != _f19_partition_commitment(
        partition_role,
        content_set_sha256=content_set,
        analysis_unit_set_sha256=unit_set,
    ):
        raise ProtocolSchemaError("frozen partition commitment mismatch")
    if bindings["row_derivation_protocol_semantic_sha256"] != (
        row_derivation_protocol_hash
    ):
        raise ProtocolSchemaError(
            "row-derivation semantic hash differs from the authorization"
        )
    if expected_row_count is not None and bindings["row_count"] != expected_row_count:
        raise ProtocolSchemaError(
            "precomputed row count differs from the frozen binding"
        )
    if (
        expected_row_index_payload_sha256 is not None
        and bindings["row_index_payload_sha256"]
        != expected_row_index_payload_sha256
    ):
        raise ProtocolSchemaError(
            "precomputed row-index payload differs from the frozen binding"
        )
    if partition_role == "development_confirm":
        _validate_identifier(
            bindings["selected_codec_id"],
            label="frozen_bindings.selected_codec_id",
        )
        _validate_identifier(
            bindings["frozen_b_star_model_id"],
            label="frozen_bindings.frozen_b_star_model_id",
        )
        if (
            bindings["candidate_bundle_metadata_sha256"]
            != bindings["candidate_bundle_metadata_canonical_payload_sha256"]
            or bindings["nonselecting_tune_replay_sha256"]
            != bindings["nonselecting_tune_replay_result_sha256"]
        ):
            raise ProtocolSchemaError(
                "development authorization duplicate identity bindings disagree"
            )


@dataclass(frozen=True, slots=True, init=False)
class AuthorizationSpec:
    """Strict semantic authorization, including Mode-A ordered binding."""

    partition_role: str
    campaign_id: str
    campaign_scope: str
    expected_row_set_mode: AuthorizationMode
    row_derivation_protocol_hash: str
    authorization_nonce: str
    expected_row_set_sha256: str | None
    expected_ordered_row_sequence_sha256: str | None
    expected_row_count: int | None
    expected_row_index_payload_sha256: str | None
    _frozen_bindings_bytes: bytes

    def __init__(
        self,
        *,
        partition_role: str,
        campaign_id: str,
        expected_row_set_mode: AuthorizationMode | str,
        row_derivation_protocol_hash: str,
        authorization_nonce: str,
        expected_row_set_sha256: str | None,
        expected_ordered_row_sequence_sha256: str | None,
        expected_row_count: int | None,
        expected_row_index_payload_sha256: str | None,
        frozen_bindings: Mapping[str, object],
        campaign_scope: str = FUTURE_D2_CAMPAIGN_SCOPE,
    ) -> None:
        role = _validate_partition_role(partition_role)
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        if campaign == CURRENT_P0_CAMPAIGN_ID:
            raise ProtocolSchemaError(
                "the current P0 campaign is reserved and cannot be authorized; "
                "initialize a distinct future D2 campaign"
            )
        scope = _validate_identifier(campaign_scope, label="campaign_scope")
        if scope not in _CAMPAIGN_SCOPES:
            raise ProtocolSchemaError(
                "campaign_scope must be the future-D2 or synthetic-qualification scope"
            )
        try:
            mode = AuthorizationMode(expected_row_set_mode)
        except (TypeError, ValueError) as error:
            raise ProtocolSchemaError("invalid expected_row_set_mode") from error
        protocol_hash = _validate_sha256(
            row_derivation_protocol_hash,
            label="row_derivation_protocol_hash",
        )
        nonce = _validate_identifier(authorization_nonce, label="authorization_nonce")
        normalized_bindings = _normalize_safe_bindings(frozen_bindings)
        if mode is AuthorizationMode.PRECOMPUTED:
            if expected_row_set_sha256 is None:
                raise ProtocolSchemaError(
                    "precomputed mode requires expected_row_set_sha256"
                )
            if expected_ordered_row_sequence_sha256 is None:
                raise ProtocolSchemaError(
                    "precomputed mode requires expected_ordered_row_sequence_sha256"
                )
            if expected_row_count is None:
                raise ProtocolSchemaError(
                    "precomputed mode requires expected_row_count"
                )
            if expected_row_index_payload_sha256 is None:
                raise ProtocolSchemaError(
                    "precomputed mode requires expected_row_index_payload_sha256"
                )
            set_hash = _validate_sha256(
                expected_row_set_sha256,
                label="expected_row_set_sha256",
            )
            order_hash = _validate_sha256(
                expected_ordered_row_sequence_sha256,
                label="expected_ordered_row_sequence_sha256",
            )
            count = _validate_uint64(
                expected_row_count, label="expected_row_count"
            )
            payload_hash = _validate_sha256(
                expected_row_index_payload_sha256,
                label="expected_row_index_payload_sha256",
            )
            if count == 0:
                raise ProtocolSchemaError("expected_row_count must be positive")
        else:
            if any(
                value is not None
                for value in (
                    expected_row_set_sha256,
                    expected_ordered_row_sequence_sha256,
                    expected_row_count,
                    expected_row_index_payload_sha256,
                )
            ):
                raise ProtocolSchemaError(
                    "derive_after_authorization mode requires null expected "
                    "row set, ordered sequence, and count"
                )
            set_hash = None
            order_hash = None
            count = None
            payload_hash = None

        _validate_authorization_bindings(
            role,
            normalized_bindings,
            row_derivation_protocol_hash=protocol_hash,
            expected_row_count=count,
            expected_row_index_payload_sha256=payload_hash,
        )
        bindings_bytes = canonical_json_bytes(normalized_bindings)

        object.__setattr__(self, "partition_role", role)
        object.__setattr__(self, "campaign_id", campaign)
        object.__setattr__(self, "campaign_scope", scope)
        object.__setattr__(self, "expected_row_set_mode", mode)
        object.__setattr__(self, "row_derivation_protocol_hash", protocol_hash)
        object.__setattr__(self, "authorization_nonce", nonce)
        object.__setattr__(self, "expected_row_set_sha256", set_hash)
        object.__setattr__(
            self, "expected_ordered_row_sequence_sha256", order_hash
        )
        object.__setattr__(self, "expected_row_count", count)
        object.__setattr__(
            self, "expected_row_index_payload_sha256", payload_hash
        )
        object.__setattr__(self, "_frozen_bindings_bytes", bindings_bytes)

    @property
    def frozen_bindings(self) -> dict[str, JsonValue]:
        value = json.loads(self._frozen_bindings_bytes.decode("utf-8"))
        assert isinstance(value, dict)
        return value

    def semantic_payload(self) -> dict[str, JsonValue]:
        return {
            "authorization_nonce": self.authorization_nonce,
            "campaign_id": self.campaign_id,
            "campaign_scope": self.campaign_scope,
            "expected_ordered_row_sequence_sha256": (
                self.expected_ordered_row_sequence_sha256
            ),
            "expected_row_index_payload_sha256": (
                self.expected_row_index_payload_sha256
            ),
            "expected_row_count": self.expected_row_count,
            "expected_row_set_mode": self.expected_row_set_mode.value,
            "expected_row_set_sha256": self.expected_row_set_sha256,
            "format_version": _FORMAT_VERSION,
            "frozen_bindings": self.frozen_bindings,
            "partition_role": self.partition_role,
            "row_derivation_protocol_hash": self.row_derivation_protocol_hash,
            "schema": AUTHORIZATION_SCHEMA,
        }

    @property
    def semantic_payload_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def authorization_hash(self) -> str:
        return hashlib.sha256(
            _AUTHORIZATION_DOMAIN + self.semantic_payload_bytes
        ).hexdigest()


def _shard_sequence_sha256(
    partition_role: str,
    shard_index: int,
    row_ids: Sequence[str | bytes],
) -> str:
    role_bytes = _validate_partition_role(partition_role).encode("utf-8")
    index = _validate_uint32(shard_index, label="shard_index")
    raw_ids = tuple(_normalize_row_id(item) for item in row_ids)
    if not raw_ids:
        raise ProtocolSchemaError("a shard must contain at least one row")
    digest = hashlib.sha256()
    digest.update(_SHARD_SEQUENCE_DOMAIN)
    digest.update(struct.pack(">H", len(role_bytes)))
    digest.update(role_bytes)
    digest.update(struct.pack(">I", index))
    digest.update(struct.pack(">Q", len(raw_ids)))
    for raw in raw_ids:
        digest.update(raw)
    return digest.hexdigest()


def _validate_relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProtocolSchemaError(f"{label} must be a nonempty POSIX path")
    if any(not 0x20 <= ord(character) <= 0x7E for character in value):
        raise ProtocolSchemaError(f"{label} must use printable ASCII")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or value != path.as_posix()
    ):
        raise ProtocolSchemaError(f"{label} must be a safe relative path")
    return value


@dataclass(frozen=True, slots=True)
class ShardPlanEntry:
    """One deterministic contiguous slice of the global row order."""

    partition_role: str
    shard_index: int
    start_row_ordinal: int
    end_row_ordinal_exclusive: int
    path: str
    row_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_partition_role(self.partition_role)
        _validate_uint32(self.shard_index, label="shard_index")
        start = _validate_uint64(
            self.start_row_ordinal, label="start_row_ordinal"
        )
        end = _validate_uint64(
            self.end_row_ordinal_exclusive,
            label="end_row_ordinal_exclusive",
        )
        if end <= start:
            raise ProtocolSchemaError("shard end must be greater than start")
        _validate_relative_path(self.path, label="shard path")
        if len(self.row_ids) != end - start:
            raise ProtocolSchemaError("shard row count does not match its range")
        for row_id in self.row_ids:
            _validate_sha256(row_id, label="row_id")

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def first_row_id(self) -> str:
        return self.row_ids[0]

    @property
    def last_row_id(self) -> str:
        return self.row_ids[-1]

    @property
    def shard_ordered_row_sequence_sha256(self) -> str:
        return _shard_sequence_sha256(
            self.partition_role, self.shard_index, self.row_ids
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "end_row_ordinal_exclusive": self.end_row_ordinal_exclusive,
            "first_row_id": self.first_row_id,
            "last_row_id": self.last_row_id,
            "path": self.path,
            "row_count": self.row_count,
            "shard_index": self.shard_index,
            "shard_ordered_row_sequence_sha256": (
                self.shard_ordered_row_sequence_sha256
            ),
            "start_row_ordinal": self.start_row_ordinal,
        }


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """A deterministic complete shard plan for one committed row index."""

    partition_role: str
    maximum_rows_per_shard: int
    row_set_sha256: str
    ordered_row_sequence_sha256: str
    row_index_payload_sha256: str
    row_count: int
    entries: tuple[ShardPlanEntry, ...]

    def __post_init__(self) -> None:
        _validate_partition_role(self.partition_role)
        maximum = _validate_uint64(
            self.maximum_rows_per_shard,
            label="maximum_rows_per_shard",
        )
        if maximum == 0:
            raise ProtocolSchemaError("maximum_rows_per_shard must be positive")
        if maximum > _MAXIMUM_SHARD_ROWS:
            raise ProtocolSchemaError(
                "maximum_rows_per_shard must not exceed 16384"
            )
        count = _validate_uint64(self.row_count, label="row_count")
        if count == 0:
            raise ProtocolSchemaError("row_count must be positive")
        _validate_sha256(self.row_set_sha256, label="row_set_sha256")
        _validate_sha256(
            self.ordered_row_sequence_sha256,
            label="ordered_row_sequence_sha256",
        )
        _validate_sha256(
            self.row_index_payload_sha256,
            label="row_index_payload_sha256",
        )
        if not self.entries:
            raise ProtocolSchemaError("shard plan must contain entries")
        expected_start = 0
        all_rows: list[str] = []
        seen_paths: set[str] = set()
        for expected_index, entry in enumerate(self.entries):
            if entry.partition_role != self.partition_role:
                raise ProtocolSchemaError("shard partition differs from plan")
            if entry.shard_index != expected_index:
                raise ProtocolSchemaError("shard indexes must be contiguous from zero")
            if entry.start_row_ordinal != expected_start:
                raise ProtocolSchemaError("shard ranges must be contiguous")
            if entry.row_count > maximum:
                raise ProtocolSchemaError("shard exceeds maximum row limit")
            if expected_index < len(self.entries) - 1 and entry.row_count != maximum:
                raise ProtocolSchemaError("every non-final shard must be full")
            if entry.path in seen_paths:
                raise ProtocolSchemaError("shard paths must be unique")
            expected_path = _format_shard_path(
                _EXACT_SHARD_PATH_TEMPLATE,
                self.partition_role,
                expected_index,
            )
            if entry.path != expected_path:
                raise ProtocolSchemaError(
                    "shard path differs from the frozen exact template"
                )
            seen_paths.add(entry.path)
            expected_start = entry.end_row_ordinal_exclusive
            all_rows.extend(entry.row_ids)
        if expected_start != count:
            raise ProtocolSchemaError("shard plan does not cover row_count")
        if row_set_sha256(all_rows) != self.row_set_sha256:
            raise ProtocolSchemaError("shard-plan row-set hash mismatch")
        if (
            ordered_row_sequence_sha256(all_rows)
            != self.ordered_row_sequence_sha256
        ):
            raise ProtocolSchemaError("shard-plan ordered hash mismatch")

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row_id for entry in self.entries for row_id in entry.row_ids)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "format_version": _FORMAT_VERSION,
            "maximum_rows_per_shard": self.maximum_rows_per_shard,
            "ordered_row_sequence_sha256": self.ordered_row_sequence_sha256,
            "partition_role": self.partition_role,
            "row_count": self.row_count,
            "row_index_payload_sha256": self.row_index_payload_sha256,
            "row_set_sha256": self.row_set_sha256,
            "schema": SHARD_PLAN_SCHEMA,
            "shard_path_template": _EXACT_SHARD_PATH_TEMPLATE,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def plan_sha256(self) -> str:
        return hashlib.sha256(_SHARD_PLAN_DOMAIN + self.canonical_bytes).hexdigest()


def _format_shard_path(
    path_template: str, partition_role: str, shard_index: int
) -> str:
    formatter = string.Formatter()
    fields: set[str] = set()
    try:
        parsed = tuple(formatter.parse(path_template))
    except ValueError as error:
        raise ProtocolSchemaError("invalid shard path template") from error
    for _, field_name, _, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in {"partition", "shard_index"}:
            raise ProtocolSchemaError(
                f"unsupported shard path field {field_name!r}"
            )
        if conversion is not None:
            raise ProtocolSchemaError("shard path conversions are forbidden")
        fields.add(field_name)
    if "shard_index" not in fields:
        raise ProtocolSchemaError("shard path template must include shard_index")
    try:
        rendered = path_template.format(
            partition=partition_role,
            shard_index=shard_index,
        )
    except (IndexError, KeyError, ValueError) as error:
        raise ProtocolSchemaError("invalid shard path format specifier") from error
    return _validate_relative_path(rendered, label="rendered shard path")


def build_shard_plan(
    row_index: RowIndexPayload,
    maximum_rows_per_shard: int,
) -> ShardPlan:
    """Build the unique deterministic contiguous shard plan."""

    if not isinstance(row_index, RowIndexPayload):
        raise ProtocolSchemaError("row_index must be RowIndexPayload")
    maximum = _validate_uint64(
        maximum_rows_per_shard,
        label="maximum_rows_per_shard",
    )
    if maximum == 0:
        raise ProtocolSchemaError("maximum_rows_per_shard must be positive")
    if maximum > _MAXIMUM_SHARD_ROWS:
        raise ProtocolSchemaError(
            "maximum_rows_per_shard must not exceed 16384"
        )
    entries: list[ShardPlanEntry] = []
    for shard_index, start in enumerate(range(0, row_index.row_count, maximum)):
        end = min(row_index.row_count, start + maximum)
        entries.append(
            ShardPlanEntry(
                partition_role=row_index.partition_role,
                shard_index=shard_index,
                start_row_ordinal=start,
                end_row_ordinal_exclusive=end,
                path=_format_shard_path(
                    _EXACT_SHARD_PATH_TEMPLATE,
                    row_index.partition_role,
                    shard_index,
                ),
                row_ids=row_index.row_ids[start:end],
            )
        )
    return ShardPlan(
        partition_role=row_index.partition_role,
        maximum_rows_per_shard=maximum,
        row_set_sha256=row_index.row_set_sha256,
        ordered_row_sequence_sha256=row_index.ordered_row_sequence_sha256,
        row_index_payload_sha256=row_index.payload_sha256,
        row_count=row_index.row_count,
        entries=tuple(entries),
    )


_SAFETENSORS_DTYPE_WIDTHS = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True, slots=True)
class _SafetensorsTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


@dataclass(frozen=True, slots=True)
class _ParsedSafetensors:
    data_offset: int
    tensors: tuple[_SafetensorsTensor, ...]

    def tensor(self, name: str) -> _SafetensorsTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolSchemaError(
                f"safetensors header repeats object key {key!r}"
            )
        value[key] = item
    return value


def _shape_element_count(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _parse_safetensors(content: bytes) -> _ParsedSafetensors:
    if not isinstance(content, bytes):
        raise ProtocolSchemaError("shard_bytes must be immutable bytes")
    if len(content) < _SAFETENSORS_HEADER_LENGTH.size:
        raise _failure_exception(
            "safetensors_format_mismatch",
            "safetensors shard is shorter than its length prefix",
        )
    header_length = _SAFETENSORS_HEADER_LENGTH.unpack_from(content, 0)[0]
    if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise _failure_exception(
            "safetensors_format_mismatch",
            "safetensors header length is outside the strict bound",
        )
    data_offset = _SAFETENSORS_HEADER_LENGTH.size + header_length
    if data_offset > len(content):
        raise _failure_exception(
            "safetensors_format_mismatch", "safetensors header is truncated"
        )
    header_bytes = content[_SAFETENSORS_HEADER_LENGTH.size:data_offset]
    if not header_bytes.startswith(b"{"):
        raise _failure_exception(
            "safetensors_format_mismatch",
            "safetensors header must begin with a JSON object",
        )
    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolSchemaError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ProtocolSchemaError) as error:
        raise _failure_exception(
            "safetensors_format_mismatch", "invalid safetensors JSON header"
        ) from error
    if not isinstance(header, dict) or not header:
        raise _failure_exception(
            "safetensors_format_mismatch",
            "safetensors header must be a nonempty object",
        )
    if "__metadata__" in header:
        raise _failure_exception(
            "safetensors_format_mismatch",
            "capture shards forbid free-form safetensors metadata",
        )
    data_length = len(content) - data_offset
    tensors: list[_SafetensorsTensor] = []
    intervals: list[tuple[int, int, str]] = []
    tensor_keys = frozenset({"data_offsets", "dtype", "shape"})
    for name, raw_descriptor in header.items():
        if not isinstance(name, str) or not name:
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor name is invalid"
            )
        if not isinstance(raw_descriptor, Mapping):
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor descriptor is not an object"
            )
        try:
            _require_exact_keys(
                raw_descriptor, tensor_keys, label=f"tensor {name!r}"
            )
        except ProtocolSchemaError as error:
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor descriptor keys differ"
            ) from error
        dtype = raw_descriptor["dtype"]
        shape_value = raw_descriptor["shape"]
        offsets_value = raw_descriptor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_WIDTHS:
            raise _failure_exception(
                "safetensors_format_mismatch", "unsupported tensor dtype"
            )
        if not isinstance(shape_value, list):
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor shape is not an array"
            )
        shape: list[int] = []
        for dimension in shape_value:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise _failure_exception(
                    "safetensors_format_mismatch", "tensor shape is not integral"
                )
            if not 0 <= dimension <= _UINT64_MAX:
                raise _failure_exception(
                    "safetensors_format_mismatch", "tensor dimension is invalid"
                )
            shape.append(dimension)
        if not isinstance(offsets_value, list) or len(offsets_value) != 2:
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor offsets are invalid"
            )
        start, end = offsets_value
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= data_length
        ):
            raise _failure_exception(
                "safetensors_format_mismatch", "tensor offsets are out of bounds"
            )
        expected_bytes = (
            _shape_element_count(shape) * _SAFETENSORS_DTYPE_WIDTHS[dtype]
        )
        if end - start != expected_bytes:
            raise _failure_exception(
                "safetensors_format_mismatch",
                "tensor byte range differs from dtype and shape",
            )
        tensors.append(
            _SafetensorsTensor(name, dtype, tuple(shape), start, end)
        )
        intervals.append((start, end, name))
    intervals.sort()
    cursor = 0
    for start, end, _ in intervals:
        if start != cursor:
            raise _failure_exception(
                "safetensors_format_mismatch",
                "tensor byte ranges overlap or leave unbound data",
            )
        cursor = end
    if cursor != data_length:
        raise _failure_exception(
            "safetensors_format_mismatch", "safetensors has trailing unbound data"
        )
    return _ParsedSafetensors(
        data_offset=data_offset,
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
    )


@dataclass(frozen=True, slots=True, init=False)
class PhysicalShardEvidence:
    """Actual safetensors bytes plus optional independent leading dimensions."""

    shard_bytes: bytes
    _observed_leading_rows: tuple[tuple[str, int], ...]

    def __init__(
        self,
        shard_bytes: bytes,
        observed_tensor_leading_row_counts: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(shard_bytes, bytes):
            raise ProtocolSchemaError("shard_bytes must be immutable bytes")
        normalized: list[tuple[str, int]] = []
        if observed_tensor_leading_row_counts is not None:
            if not isinstance(observed_tensor_leading_row_counts, Mapping):
                raise ProtocolSchemaError(
                    "observed_tensor_leading_row_counts must be an object"
                )
            for name, count in observed_tensor_leading_row_counts.items():
                if not isinstance(name, str) or not name:
                    raise ProtocolSchemaError("observed tensor name is invalid")
                normalized.append(
                    (
                        name,
                        _validate_uint64(
                            count,
                            label=f"observed leading rows for {name}",
                        ),
                    )
                )
        object.__setattr__(self, "shard_bytes", shard_bytes)
        object.__setattr__(
            self,
            "_observed_leading_rows",
            tuple(sorted(normalized)),
        )

    @property
    def observed_tensor_leading_row_counts(self) -> dict[str, int]:
        return dict(self._observed_leading_rows)


@dataclass(frozen=True, slots=True)
class VerifiedShardRecord:
    """Verified shard inventory; row IDs came from the shard tensor bytes."""

    partition_role: str
    shard_index: int
    start_row_ordinal: int
    end_row_ordinal_exclusive: int
    path: str
    row_ids: tuple[str, ...]
    shard_byte_sha256: str
    shard_plan_sha256: str
    _construction_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _VERIFIED_CONSTRUCTION_TOKEN:
            raise ProtocolSchemaError(
                "VerifiedShardRecord may only be created by a physical verifier"
            )
        _validate_partition_role(self.partition_role)
        _validate_uint32(self.shard_index, label="shard_index")
        start = _validate_uint64(
            self.start_row_ordinal, label="start_row_ordinal"
        )
        end = _validate_uint64(
            self.end_row_ordinal_exclusive,
            label="end_row_ordinal_exclusive",
        )
        if end <= start or end - start != len(self.row_ids):
            raise ProtocolSchemaError("verified shard range is inconsistent")
        _validate_relative_path(self.path, label="verified shard path")
        _validate_sha256(self.shard_byte_sha256, label="shard_byte_sha256")
        _validate_sha256(self.shard_plan_sha256, label="shard_plan_sha256")
        for row_id in self.row_ids:
            _validate_sha256(row_id, label="row_id")

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def first_row_id(self) -> str:
        return self.row_ids[0]

    @property
    def last_row_id(self) -> str:
        return self.row_ids[-1]

    @property
    def shard_ordered_row_sequence_sha256(self) -> str:
        return _shard_sequence_sha256(
            self.partition_role, self.shard_index, self.row_ids
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "end_row_ordinal_exclusive": self.end_row_ordinal_exclusive,
            "first_row_id": self.first_row_id,
            "last_row_id": self.last_row_id,
            "path": self.path,
            "row_count": self.row_count,
            "shard_byte_sha256": self.shard_byte_sha256,
            "shard_index": self.shard_index,
            "shard_plan_sha256": self.shard_plan_sha256,
            "shard_ordered_row_sequence_sha256": (
                self.shard_ordered_row_sequence_sha256
            ),
            "start_row_ordinal": self.start_row_ordinal,
        }


def verify_physical_shard(
    entry: ShardPlanEntry,
    evidence: PhysicalShardEvidence,
    *,
    shard_plan_sha256: str,
    _enforce_expected_rows: bool = True,
) -> VerifiedShardRecord:
    """Reopen shard bytes and verify co-serialized row IDs and row dimensions."""

    if not isinstance(entry, ShardPlanEntry):
        raise ProtocolSchemaError("entry must be ShardPlanEntry")
    if not isinstance(evidence, PhysicalShardEvidence):
        raise ProtocolSchemaError("evidence must be PhysicalShardEvidence")
    plan_hash = _validate_sha256(
        shard_plan_sha256, label="shard_plan_sha256"
    )
    parsed = _parse_safetensors(evidence.shard_bytes)
    try:
        row_tensor = parsed.tensor(_ROW_ID_TENSOR_NAME)
    except KeyError as error:
        raise _failure_exception(
            "tensor_row_identity_alignment_mismatch",
            "capture_row_id_raw32 is missing from the physical shard",
        ) from error
    if row_tensor.dtype != "U8":
        raise _failure_exception(
            "invalid_tensor_dtype", "capture_row_id_raw32 must have dtype U8"
        )
    if row_tensor.shape != (entry.row_count, 32):
        raise _failure_exception(
            "invalid_tensor_shape",
            "capture_row_id_raw32 must have shape [planned_rows, 32]",
        )
    for tensor in parsed.tensors:
        if not tensor.shape or tensor.shape[0] != entry.row_count:
            raise _failure_exception(
                "tensor_row_identity_alignment_mismatch",
                "every captured tensor must have the planned leading row count",
            )
    observed = evidence.observed_tensor_leading_row_counts
    if observed:
        expected_names = {tensor.name for tensor in parsed.tensors}
        if set(observed) != expected_names:
            raise _failure_exception(
                "tensor_row_identity_alignment_mismatch",
                "independently observed tensor names differ from the shard",
            )
        for tensor in parsed.tensors:
            if observed[tensor.name] != tensor.shape[0]:
                raise _failure_exception(
                    "tensor_row_identity_alignment_mismatch",
                    "independently observed leading rows differ from the shard",
                )
    start = parsed.data_offset + row_tensor.data_start
    end = parsed.data_offset + row_tensor.data_end
    raw_rows = evidence.shard_bytes[start:end]
    physical_row_ids = tuple(
        raw_rows[offset:offset + 32].hex()
        for offset in range(0, len(raw_rows), 32)
    )
    if len(set(physical_row_ids)) != len(physical_row_ids):
        raise _failure_exception(
            "duplicate_row_id",
            "physical capture_row_id_raw32 contains duplicate row IDs",
        )
    if _enforce_expected_rows and physical_row_ids != entry.row_ids:
        if (
            len(physical_row_ids) == len(entry.row_ids)
            and sorted(physical_row_ids) == sorted(entry.row_ids)
        ):
            raise _failure_exception(
                "capture_row_order_mismatch",
                "physical tensor has the correct row set in a different order",
            )
        raise _failure_exception(
            "row_set_mismatch",
            "physical row-ID tensor differs from the committed slice",
        )
    return VerifiedShardRecord(
        partition_role=entry.partition_role,
        shard_index=entry.shard_index,
        start_row_ordinal=entry.start_row_ordinal,
        end_row_ordinal_exclusive=entry.end_row_ordinal_exclusive,
        path=entry.path,
        row_ids=physical_row_ids,
        shard_byte_sha256=artifact_sha256(evidence.shard_bytes),
        shard_plan_sha256=plan_hash,
        _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class CaptureVerification:
    """Complete verification from co-serialized physical row-ID tensors."""

    partition_role: str
    shard_plan_sha256: str
    committed_row_set_sha256: str
    committed_ordered_row_sequence_sha256: str
    captured_row_set_sha256: str
    captured_ordered_row_sequence_sha256: str
    row_count: int
    shards: tuple[VerifiedShardRecord, ...]
    verification_source: str
    _construction_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _VERIFIED_CONSTRUCTION_TOKEN:
            raise ProtocolSchemaError(
                "CaptureVerification may only be created by a physical verifier"
            )
        if self.verification_source not in {
            "in_memory_bytes",
            "finalized_files_reopened",
        }:
            raise ProtocolSchemaError("invalid capture verification source")
        _validate_partition_role(self.partition_role)
        for label in (
            "shard_plan_sha256",
            "committed_row_set_sha256",
            "committed_ordered_row_sequence_sha256",
            "captured_row_set_sha256",
            "captured_ordered_row_sequence_sha256",
        ):
            _validate_sha256(getattr(self, label), label=label)
        _validate_uint64(self.row_count, label="row_count")
        if self.row_count == 0 or not self.shards:
            raise ProtocolSchemaError("capture verification must not be empty")
        expected_start = 0
        physical_rows: list[str] = []
        for expected_index, shard in enumerate(self.shards):
            if (
                shard.partition_role != self.partition_role
                or shard.shard_plan_sha256 != self.shard_plan_sha256
                or shard.shard_index != expected_index
                or shard.start_row_ordinal != expected_start
            ):
                raise ProtocolSchemaError(
                    "capture verification shard identity/range is inconsistent"
                )
            expected_start = shard.end_row_ordinal_exclusive
            physical_rows.extend(shard.row_ids)
        if expected_start != self.row_count:
            raise ProtocolSchemaError("capture verification row count is inconsistent")
        if row_set_sha256(physical_rows) != self.captured_row_set_sha256:
            raise ProtocolSchemaError("capture verification captured set is inconsistent")
        if (
            ordered_row_sequence_sha256(physical_rows)
            != self.captured_ordered_row_sequence_sha256
        ):
            raise ProtocolSchemaError(
                "capture verification captured order is inconsistent"
            )
        if self.committed_row_set_sha256 != self.captured_row_set_sha256:
            raise _failure_exception("row_set_mismatch", "captured row-set mismatch")
        if (
            self.committed_ordered_row_sequence_sha256
            != self.captured_ordered_row_sequence_sha256
        ):
            raise _failure_exception(
                "capture_row_order_mismatch", "captured row-order mismatch"
            )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "captured_ordered_row_sequence_sha256": (
                self.captured_ordered_row_sequence_sha256
            ),
            "captured_row_set_sha256": self.captured_row_set_sha256,
            "committed_ordered_row_sequence_sha256": (
                self.committed_ordered_row_sequence_sha256
            ),
            "committed_row_set_sha256": self.committed_row_set_sha256,
            "format_version": _FORMAT_VERSION,
            "ordered_row_sequence_match": True,
            "partition_role": self.partition_role,
            "row_count": self.row_count,
            "row_set_match": True,
            "schema": CAPTURE_VERIFICATION_SCHEMA,
            "shard_plan_sha256": self.shard_plan_sha256,
            "shards": [shard.as_dict() for shard in self.shards],
            "verification_source": self.verification_source,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def verification_sha256(self) -> str:
        return hashlib.sha256(
            _CAPTURE_VERIFICATION_DOMAIN + self.canonical_bytes
        ).hexdigest()


def _validate_evidence_keys(
    plan: ShardPlan,
    evidence_by_shard: Mapping[int, PhysicalShardEvidence],
    *,
    complete: bool,
) -> tuple[int, ...]:
    if not isinstance(evidence_by_shard, Mapping):
        raise ProtocolSchemaError("evidence_by_shard must be a mapping")
    indexes: list[int] = []
    for index, evidence in evidence_by_shard.items():
        _validate_uint32(index, label="evidence shard index")
        if not isinstance(evidence, PhysicalShardEvidence):
            raise ProtocolSchemaError("invalid PhysicalShardEvidence value")
        if index >= len(plan.entries):
            raise ProtocolSchemaError("evidence contains an unknown shard index")
        indexes.append(index)
    normalized = tuple(sorted(indexes))
    if len(normalized) != len(set(normalized)):
        raise ProtocolSchemaError("duplicate evidence shard index")
    expected = tuple(range(len(plan.entries)))
    if complete and normalized != expected:
        raise _failure_exception(
            "row_set_mismatch", "complete capture evidence omits or adds shards"
        )
    if not complete and not normalized:
        raise ProtocolSchemaError("partial shard evidence must not be empty")
    return normalized


def verify_capture_shards(
    plan: ShardPlan,
    evidence_by_shard: Mapping[int, PhysicalShardEvidence],
) -> CaptureVerification:
    """Verify every planned shard and recompute global captured commitments."""

    if not isinstance(plan, ShardPlan):
        raise ProtocolSchemaError("plan must be ShardPlan")
    indexes = _validate_evidence_keys(plan, evidence_by_shard, complete=True)
    shards = tuple(
        verify_physical_shard(
            plan.entries[index],
            evidence_by_shard[index],
            shard_plan_sha256=plan.plan_sha256,
            _enforce_expected_rows=False,
        )
        for index in indexes
    )
    captured_rows = tuple(
        row_id for shard in shards for row_id in shard.row_ids
    )
    captured_set = row_set_sha256(captured_rows)
    captured_order = ordered_row_sequence_sha256(captured_rows)
    if captured_set != plan.row_set_sha256:
        raise _failure_exception("row_set_mismatch", "captured row-set mismatch")
    if captured_order != plan.ordered_row_sequence_sha256:
        raise _failure_exception(
            "capture_row_order_mismatch", "captured physical order mismatch"
        )
    return CaptureVerification(
        partition_role=plan.partition_role,
        shard_plan_sha256=plan.plan_sha256,
        committed_row_set_sha256=plan.row_set_sha256,
        committed_ordered_row_sequence_sha256=(
            plan.ordered_row_sequence_sha256
        ),
        captured_row_set_sha256=captured_set,
        captured_ordered_row_sequence_sha256=captured_order,
        row_count=len(captured_rows),
        shards=shards,
        verification_source="in_memory_bytes",
        _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class PartialShardInventory:
    """Verified complete immutable shards retained after a failed capture."""

    partition_role: str
    shard_plan_sha256: str
    shards: tuple[VerifiedShardRecord, ...]
    verification_source: str
    _construction_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _VERIFIED_CONSTRUCTION_TOKEN:
            raise ProtocolSchemaError(
                "PartialShardInventory may only be created by a physical verifier"
            )
        if self.verification_source not in {
            "in_memory_bytes",
            "finalized_files_reopened",
        }:
            raise ProtocolSchemaError("invalid partial verification source")
        _validate_partition_role(self.partition_role)
        _validate_sha256(self.shard_plan_sha256, label="shard_plan_sha256")
        if not self.shards:
            raise ProtocolSchemaError("partial inventory must contain a shard")
        indexes = tuple(shard.shard_index for shard in self.shards)
        if indexes != tuple(sorted(set(indexes))):
            raise ProtocolSchemaError(
                "partial inventory indexes must be unique and ascending"
            )
        for shard in self.shards:
            if (
                shard.partition_role != self.partition_role
                or shard.shard_plan_sha256 != self.shard_plan_sha256
            ):
                raise ProtocolSchemaError(
                    "partial inventory shard identity differs from its plan"
                )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "finalized_shard_count": len(self.shards),
            "format_version": _FORMAT_VERSION,
            "partition_role": self.partition_role,
            "schema": PARTIAL_SHARD_INVENTORY_SCHEMA,
            "shard_plan_sha256": self.shard_plan_sha256,
            "shards": [shard.as_dict() for shard in self.shards],
            "verification_source": self.verification_source,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def inventory_sha256(self) -> str:
        return hashlib.sha256(
            _PARTIAL_INVENTORY_DOMAIN + self.canonical_bytes
        ).hexdigest()


def verify_partial_capture_shards(
    plan: ShardPlan,
    evidence_by_shard: Mapping[int, PhysicalShardEvidence],
) -> PartialShardInventory:
    """Verify any nonempty set of durable, complete shard slices."""

    if not isinstance(plan, ShardPlan):
        raise ProtocolSchemaError("plan must be ShardPlan")
    indexes = _validate_evidence_keys(plan, evidence_by_shard, complete=False)
    return PartialShardInventory(
        partition_role=plan.partition_role,
        shard_plan_sha256=plan.plan_sha256,
        shards=tuple(
            verify_physical_shard(
                plan.entries[index],
                evidence_by_shard[index],
                shard_plan_sha256=plan.plan_sha256,
            )
            for index in indexes
        ),
        verification_source="in_memory_bytes",
        _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )


def _read_finalized_shard_bytes(
    capture_root: str | os.PathLike[str],
    relative_path: str,
) -> bytes:
    """Open one frozen relative path without following symlinks or races."""

    root = Path(capture_root)
    _validate_relative_path(relative_path, label="finalized shard path")
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise _failure_exception(
            "shard_file_missing", "capture root is unavailable"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise _failure_exception(
            "shard_symlink_forbidden", "capture root must not be a symlink"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _failure_exception(
            "shard_not_regular_file", "capture root must be a directory"
        )
    root_resolved = root.resolve(strict=True)
    candidate = root
    parts = PurePosixPath(relative_path).parts
    try:
        for index, part in enumerate(parts):
            candidate = candidate / part
            component_stat = os.lstat(candidate)
            if stat.S_ISLNK(component_stat.st_mode):
                raise _failure_exception(
                    "shard_symlink_forbidden",
                    "frozen shard path contains a symlink",
                )
            if index < len(parts) - 1 and not stat.S_ISDIR(
                component_stat.st_mode
            ):
                raise _failure_exception(
                    "shard_not_regular_file",
                    "frozen shard parent is not a directory",
                )
    except FileNotFoundError as error:
        raise _failure_exception(
            "shard_file_missing", "a frozen shard path is missing"
        ) from error
    try:
        candidate_resolved = candidate.resolve(strict=True)
        if os.path.commonpath((root_resolved, candidate_resolved)) != os.fspath(
            root_resolved
        ):
            raise _failure_exception(
                "shard_path_escape", "frozen shard path escapes capture root"
            )
    except OSError as error:
        raise _failure_exception(
            "shard_file_missing", "cannot resolve frozen shard path"
        ) from error
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode):
        raise _failure_exception(
            "shard_not_regular_file", "frozen shard is not a regular file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise _failure_exception(
            "shard_symlink_forbidden",
            "cannot open frozen shard without following links",
        ) from error
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_dev != before.st_dev
            or opened_before.st_ino != before.st_ino
        ):
            raise _failure_exception(
                "shard_file_mutated_during_verification",
                "frozen shard identity changed before read",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(candidate)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    opened_before_identity = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
        opened_before.st_ctime_ns,
    )
    opened_after_identity = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
        opened_after.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not (
        before_identity
        == opened_before_identity
        == opened_after_identity
        == after_identity
    ) or len(content) != before.st_size:
        raise _failure_exception(
            "shard_file_mutated_during_verification",
            "frozen shard changed during verification",
        )
    return content


def _inventory_finalized_shard_indexes(
    plan: ShardPlan,
    capture_root: str | os.PathLike[str],
    *,
    require_complete: bool,
) -> tuple[int, ...]:
    """Inventory exact finalized filenames and reject every extra shard."""

    root = Path(capture_root)
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise _failure_exception(
            "shard_file_missing", "capture root is unavailable"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise _failure_exception(
            "shard_symlink_forbidden", "capture root must not be a symlink"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _failure_exception(
            "shard_not_regular_file", "capture root must be a directory"
        )
    directory_relative = PurePosixPath("capture_shards") / plan.partition_role
    directory = root.joinpath(*directory_relative.parts)
    if not os.path.lexists(directory):
        if require_complete:
            raise _failure_exception(
                "shard_file_missing", "finalized shard directory is missing"
            )
        return ()
    cursor = root
    for part in directory_relative.parts:
        cursor = cursor / part
        component = os.lstat(cursor)
        if stat.S_ISLNK(component.st_mode):
            raise _failure_exception(
                "shard_symlink_forbidden",
                "finalized shard directory contains a symlink",
            )
        if not stat.S_ISDIR(component.st_mode):
            raise _failure_exception(
                "shard_not_regular_file",
                "finalized shard directory component is not a directory",
            )
    expected_by_name = {
        PurePosixPath(entry.path).name: entry.shard_index
        for entry in plan.entries
    }
    found: list[int] = []
    with os.scandir(directory) as iterator:
        for directory_entry in iterator:
            if not directory_entry.name.endswith(".safetensors"):
                continue
            if directory_entry.name not in expected_by_name:
                raise _failure_exception(
                    "shard_plan_mismatch",
                    "finalized shard directory contains an extra shard",
                )
            found.append(expected_by_name[directory_entry.name])
    indexes = tuple(sorted(found))
    if len(indexes) != len(set(indexes)):
        raise _failure_exception(
            "shard_plan_mismatch", "finalized shard inventory is ambiguous"
        )
    expected_indexes = tuple(range(len(plan.entries)))
    if require_complete and indexes != expected_indexes:
        raise _failure_exception(
            "shard_file_missing", "complete capture omits a planned shard"
        )
    return indexes


def _capture_root_identity_sha256(
    capture_root: str | os.PathLike[str],
) -> str:
    root = Path(capture_root)
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise _failure_exception(
            "shard_file_missing", "capture root is unavailable"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise _failure_exception(
            "shard_symlink_forbidden", "capture root must not be a symlink"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _failure_exception(
            "shard_not_regular_file", "capture root must be a directory"
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise _failure_exception(
            "shard_file_missing", "capture root cannot be resolved"
        ) from error
    return hashlib.sha256(
        _CAPTURE_ROOT_DOMAIN
        + canonical_json_bytes(
            {
                "device": root_stat.st_dev,
                "inode": root_stat.st_ino,
                "resolved_path_sha256": hashlib.sha256(
                    os.fsencode(os.fspath(resolved))
                ).hexdigest(),
            }
        )
    ).hexdigest()


def verify_physical_shard_file(
    plan: ShardPlan,
    shard_index: int,
    capture_root: str | os.PathLike[str],
    *,
    observed_tensor_leading_row_counts: Mapping[str, int] | None = None,
    _enforce_expected_rows: bool = True,
) -> VerifiedShardRecord:
    """Reopen and verify one exact frozen plan path on disk."""

    if not isinstance(plan, ShardPlan):
        raise ProtocolSchemaError("plan must be ShardPlan")
    index = _validate_uint32(shard_index, label="shard_index")
    if index >= len(plan.entries):
        raise ProtocolSchemaError("shard_index is outside the plan")
    entry = plan.entries[index]
    content = _read_finalized_shard_bytes(capture_root, entry.path)
    return verify_physical_shard(
        entry,
        PhysicalShardEvidence(content, observed_tensor_leading_row_counts),
        shard_plan_sha256=plan.plan_sha256,
        _enforce_expected_rows=_enforce_expected_rows,
    )


def verify_capture_shard_files(
    plan: ShardPlan,
    capture_root: str | os.PathLike[str],
    *,
    observed_tensor_leading_row_counts_by_shard: (
        Mapping[int, Mapping[str, int]] | None
    ) = None,
) -> CaptureVerification:
    """Reopen every exact plan path and build finalization-grade evidence."""

    if not isinstance(plan, ShardPlan):
        raise ProtocolSchemaError("plan must be ShardPlan")
    _inventory_finalized_shard_indexes(
        plan, capture_root, require_complete=True
    )
    observed = observed_tensor_leading_row_counts_by_shard or {}
    unknown = set(observed) - set(range(len(plan.entries)))
    if unknown:
        raise ProtocolSchemaError("observed leading rows include unknown shards")
    shards = tuple(
        verify_physical_shard_file(
            plan,
            index,
            capture_root,
            observed_tensor_leading_row_counts=observed.get(index),
            _enforce_expected_rows=False,
        )
        for index in range(len(plan.entries))
    )
    physical_rows = tuple(row for shard in shards for row in shard.row_ids)
    captured_set = row_set_sha256(physical_rows)
    captured_order = ordered_row_sequence_sha256(physical_rows)
    if captured_set != plan.row_set_sha256:
        raise _failure_exception("row_set_mismatch", "captured row-set mismatch")
    if captured_order != plan.ordered_row_sequence_sha256:
        raise _failure_exception(
            "capture_row_order_mismatch", "captured physical order mismatch"
        )
    return CaptureVerification(
        partition_role=plan.partition_role,
        shard_plan_sha256=plan.plan_sha256,
        committed_row_set_sha256=plan.row_set_sha256,
        committed_ordered_row_sequence_sha256=(
            plan.ordered_row_sequence_sha256
        ),
        captured_row_set_sha256=captured_set,
        captured_ordered_row_sequence_sha256=captured_order,
        row_count=len(physical_rows),
        shards=shards,
        verification_source="finalized_files_reopened",
        _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )


def verify_partial_capture_shard_files(
    plan: ShardPlan,
    capture_root: str | os.PathLike[str],
    shard_indexes: Iterable[int],
    *,
    observed_tensor_leading_row_counts_by_shard: (
        Mapping[int, Mapping[str, int]] | None
    ) = None,
) -> PartialShardInventory:
    """Reopen a nonempty set of durable complete shard files for recovery."""

    if not isinstance(plan, ShardPlan):
        raise ProtocolSchemaError("plan must be ShardPlan")
    indexes = tuple(sorted(shard_indexes))
    if not indexes or len(indexes) != len(set(indexes)):
        raise ProtocolSchemaError("partial shard indexes must be nonempty and unique")
    observed = observed_tensor_leading_row_counts_by_shard or {}
    if set(observed) - set(indexes):
        raise ProtocolSchemaError("observed leading rows include unrequested shards")
    actual_indexes = _inventory_finalized_shard_indexes(
        plan, capture_root, require_complete=False
    )
    if actual_indexes != indexes:
        raise _failure_exception(
            "shard_plan_mismatch",
            "partial inventory must name every and only finalized shard file",
        )
    shards = tuple(
        verify_physical_shard_file(
            plan,
            index,
            capture_root,
            observed_tensor_leading_row_counts=observed.get(index),
        )
        for index in indexes
    )
    return PartialShardInventory(
        partition_role=plan.partition_role,
        shard_plan_sha256=plan.plan_sha256,
        shards=shards,
        verification_source="finalized_files_reopened",
        _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ChartHeadObservation:
    """One protected-value-free chart/head support observation."""

    analysis_unit_id: str
    chart_id: str | None
    head_id: str | None
    otherwise_valid: bool = True
    row_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.analysis_unit_id, label="analysis_unit_id")
        if self.chart_id is not None:
            _validate_identifier(self.chart_id, label="chart_id")
        if self.head_id is not None:
            _validate_identifier(self.head_id, label="head_id")
        if not isinstance(self.otherwise_valid, bool):
            raise ProtocolSchemaError("otherwise_valid must be boolean")
        if self.row_id is not None:
            object.__setattr__(
                self,
                "row_id",
                _normalize_row_id(self.row_id, label="row_id").hex(),
            )


@dataclass(frozen=True, slots=True)
class ExpectedChartHeadRow:
    """One frozen expected row/chart/head/unit identity quadruple."""

    row_id: str
    analysis_unit_id: str
    chart_id: str
    head_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_id", _normalize_row_id(self.row_id, label="row_id").hex()
        )
        _validate_identifier(self.analysis_unit_id, label="analysis_unit_id")
        _validate_identifier(self.chart_id, label="chart_id")
        _validate_identifier(self.head_id, label="head_id")


@dataclass(frozen=True, slots=True)
class ChartHeadSupportDecision:
    """Deterministic support result after completeness-first evaluation."""

    status: SupportStatus
    terminal_state: TerminalState | None
    reason_code: str | None
    otherwise_valid_row_count: int
    chart_valid_row_count: int
    missing_head_row_count: int
    required_distinct_units_per_pair: int
    qualifying_pair_count: int
    pair_support: tuple[tuple[str, str, int], ...]

    @property
    def is_sufficient(self) -> bool:
        return self.status is SupportStatus.SUFFICIENT


def classify_chart_head_support(
    observations: Iterable[ChartHeadObservation],
    *,
    active_chart_ids: Iterable[str] | None = None,
    expected_head_ids: Iterable[str] | None = None,
    expected_rows: Iterable[ExpectedChartHeadRow] | None = None,
    required_distinct_units_per_pair: int = 2,
) -> ChartHeadSupportDecision:
    """Classify chart/head support without the obsolete zero-head reason.

    Missing-head completeness is checked across otherwise-valid chart rows
    before any pair filtering or support counts are computed.
    """

    if active_chart_ids is None or expected_head_ids is None or expected_rows is None:
        raise ProtocolSchemaError(
            "frozen active_chart_ids, expected_head_ids, and expected_rows are required"
        )
    active_charts = _normalize_identifier_sequence(
        active_chart_ids, label="active_chart_ids"
    )
    expected_heads = _normalize_identifier_sequence(
        expected_head_ids, label="expected_head_ids"
    )
    if not active_charts or not expected_heads:
        raise ProtocolSchemaError(
            "active chart and expected head universes must be nonempty"
        )
    frozen_rows = tuple(expected_rows)
    if not all(isinstance(item, ExpectedChartHeadRow) for item in frozen_rows):
        raise ProtocolSchemaError(
            "expected_rows must contain ExpectedChartHeadRow values"
        )
    frozen_keys = tuple(
        (item.row_id, item.chart_id, item.head_id) for item in frozen_rows
    )
    if len(frozen_keys) != len(set(frozen_keys)):
        raise _failure_exception(
            "duplicate_row_id", "frozen chart/head row keys contain duplicates"
        )
    active_set = frozenset(active_charts)
    head_set = frozenset(expected_heads)
    if any(
        item.chart_id not in active_set or item.head_id not in head_set
        for item in frozen_rows
    ):
        raise ProtocolSchemaError(
            "expected rows fall outside the frozen chart/head universe"
        )
    unit_by_row: dict[str, str] = {}
    for item in frozen_rows:
        prior = unit_by_row.setdefault(item.row_id, item.analysis_unit_id)
        if prior != item.analysis_unit_id:
            raise _failure_exception(
                "malformed_sequence_identity",
                "one frozen row ID maps to conflicting analysis units",
            )
    required = _validate_uint64(
        required_distinct_units_per_pair,
        label="required_distinct_units_per_pair",
    )
    if required == 0:
        raise ProtocolSchemaError("pair support threshold must be positive")
    values = tuple(observations)
    if not all(isinstance(item, ChartHeadObservation) for item in values):
        raise ProtocolSchemaError(
            "observations must contain only ChartHeadObservation values"
        )
    if any(item.row_id is None for item in values):
        raise ProtocolSchemaError("every chart/head observation requires row_id")
    otherwise_valid = tuple(item for item in values if item.otherwise_valid)
    expected_by_chart = {
        chart_id: tuple(item for item in frozen_rows if item.chart_id == chart_id)
        for chart_id in active_charts
    }
    incomplete_expected_pairs = tuple(
        (row_id, chart_id)
        for row_id, chart_id in sorted(
            {(item.row_id, item.chart_id) for item in frozen_rows}
        )
        if {
            item.head_id
            for item in frozen_rows
            if item.row_id == row_id and item.chart_id == chart_id
        }
        != head_set
    )
    if incomplete_expected_pairs:
        return ChartHeadSupportDecision(
            status=SupportStatus.INVALID_MEASUREMENT_CONTRACT,
            terminal_state=TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT,
            reason_code="missing_head_in_validation_rows",
            otherwise_valid_row_count=len(otherwise_valid),
            chart_valid_row_count=len(otherwise_valid),
            missing_head_row_count=len(incomplete_expected_pairs),
            required_distinct_units_per_pair=required,
            qualifying_pair_count=0,
            pair_support=(),
        )
    observed_keys: list[tuple[str, str, str]] = []
    observed_units: dict[tuple[str, str, str], str] = {}
    for item in values:
        if item.chart_id is None or item.head_id is None:
            return ChartHeadSupportDecision(
                status=SupportStatus.INVALID_MEASUREMENT_CONTRACT,
                terminal_state=TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT,
                reason_code="missing_head_in_validation_rows",
                otherwise_valid_row_count=len(otherwise_valid),
                chart_valid_row_count=sum(
                    value.chart_id is not None for value in values
                ),
                missing_head_row_count=1,
                required_distinct_units_per_pair=required,
                qualifying_pair_count=0,
                pair_support=(),
            )
        assert item.row_id is not None
        key = (item.row_id, item.chart_id, item.head_id)
        if key in observed_units:
            raise _failure_exception(
                "duplicate_row_id", "observed chart/head row key is duplicated"
            )
        observed_keys.append(key)
        observed_units[key] = item.analysis_unit_id
    expected_by_key = {
        (item.row_id, item.chart_id, item.head_id): item.analysis_unit_id
        for item in frozen_rows
    }
    if set(observed_keys) - set(expected_by_key):
        raise _failure_exception(
            "row_set_mismatch", "observations contain an unfrozen chart/head row"
        )
    if any(
        observed_units[key] != expected_by_key[key] for key in observed_keys
    ):
        raise _failure_exception(
            "malformed_sequence_identity",
            "observed row-to-analysis-unit binding changed",
        )
    missing_keys = set(expected_by_key) - set(observed_keys)
    if missing_keys:
        return ChartHeadSupportDecision(
            status=SupportStatus.INVALID_MEASUREMENT_CONTRACT,
            terminal_state=TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT,
            reason_code="missing_head_in_validation_rows",
            otherwise_valid_row_count=len(otherwise_valid),
            chart_valid_row_count=len(otherwise_valid),
            missing_head_row_count=len(missing_keys),
            required_distinct_units_per_pair=required,
            qualifying_pair_count=0,
            pair_support=(),
        )
    if any(not rows for rows in expected_by_chart.values()):
        return ChartHeadSupportDecision(
            status=SupportStatus.INSUFFICIENT_SUPPORT,
            terminal_state=TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT,
            reason_code="zero_fit_monitor_chart_support",
            otherwise_valid_row_count=len(otherwise_valid),
            chart_valid_row_count=len(otherwise_valid),
            missing_head_row_count=0,
            required_distinct_units_per_pair=required,
            qualifying_pair_count=0,
            pair_support=(),
        )
    chart_valid = otherwise_valid
    support: dict[tuple[str, str], set[str]] = {}
    for item in chart_valid:
        assert item.chart_id is not None and item.head_id is not None
        support.setdefault((item.chart_id, item.head_id), set()).add(
            item.analysis_unit_id
        )
    pair_support = tuple(
        (chart_id, head_id, len(support.get((chart_id, head_id), set())))
        for chart_id in active_charts
        for head_id in expected_heads
    )
    qualifying = sum(count >= required for _, _, count in pair_support)
    if qualifying != len(pair_support):
        return ChartHeadSupportDecision(
            status=SupportStatus.INSUFFICIENT_SUPPORT,
            terminal_state=TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT,
            reason_code="insufficient_independent_chart_support",
            otherwise_valid_row_count=len(otherwise_valid),
            chart_valid_row_count=len(chart_valid),
            missing_head_row_count=0,
            required_distinct_units_per_pair=required,
            qualifying_pair_count=qualifying,
            pair_support=pair_support,
        )
    return ChartHeadSupportDecision(
        status=SupportStatus.SUFFICIENT,
        terminal_state=None,
        reason_code=None,
        otherwise_valid_row_count=len(otherwise_valid),
        chart_valid_row_count=len(chart_valid),
        missing_head_row_count=0,
        required_distinct_units_per_pair=required,
        qualifying_pair_count=qualifying,
        pair_support=pair_support,
    )


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """One exact canonical artifact committed inside a ledger transaction."""

    name: str
    kind: str
    content: bytes
    byte_sha256: str
    transaction_id: str

    def json_value(self) -> dict[str, Any]:
        value = json.loads(self.content.decode("utf-8"))
        if not isinstance(value, dict):
            raise ProtocolSchemaError("stored artifact is not a JSON object")
        return value


@dataclass(frozen=True, slots=True)
class TransactionResult:
    """Committed logical transaction and all artifacts it created."""

    transaction_id: str
    transaction_kind: str
    partition_role: str
    state_before: CaptureState
    state_after: CaptureState
    identity_sha256: str
    artifacts: tuple[StoredArtifact, ...]
    idempotent_recovery: bool = False

    def artifact(self, name: str) -> StoredArtifact:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class CampaignTerminalResult:
    """Atomic terminal-tail result whose final manifest is public-last."""

    transaction: TransactionResult
    terminal_state: TerminalState
    prefix_closure: PrefixClosure
    final_experiment_manifest_sha256: str

    @property
    def artifacts(self) -> tuple[StoredArtifact, ...]:
        return self.transaction.artifacts

    @property
    def idempotent_recovery(self) -> bool:
        return self.transaction.idempotent_recovery


@dataclass(frozen=True, slots=True)
class PanelSnapshot:
    """Read-only state and bound identities for one panel."""

    partition_role: str
    campaign_id: str
    state: CaptureState
    authorization_hash: str
    authorization_transaction_id: str
    expected_row_set_mode: AuthorizationMode
    row_derivation_protocol_hash: str
    expected_row_set_sha256: str | None
    expected_ordered_row_sequence_sha256: str | None
    expected_row_count: int | None
    expected_row_index_payload_sha256: str | None
    row_index_transaction_id: str | None
    row_index_payload_sha256: str | None
    committed_row_set_sha256: str | None
    committed_ordered_row_sequence_sha256: str | None
    committed_row_count: int | None
    row_index_failure_transaction_id: str | None
    capture_start_transaction_id: str | None
    capture_attempt_id: str | None
    capture_root_identity_sha256: str | None
    shard_plan_transaction_id: str | None
    shard_plan_sha256: str | None
    capture_schema_hash: str | None
    capture_implementation_hash: str | None
    resume_checkpoint_count: int
    resume_checkpoint_sha256: str | None
    terminal_transaction_id: str | None
    terminal_state: str | None
    failure_reason_code: str | None


@dataclass(frozen=True, slots=True)
class CaptureResumeResult:
    """Same-attempt recovery plus only reopened, verified durable shards."""

    start_transaction: TransactionResult
    verified_existing_shards: PartialShardInventory | None
    missing_shard_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArtifactClosure:
    """State-dependent named-artifact closure audit."""

    partition_role: str
    state: CaptureState
    required_names: frozenset[str]
    forbidden_names: frozenset[str]
    present_names: frozenset[str]

    @property
    def missing_names(self) -> frozenset[str]:
        return self.required_names - self.present_names

    @property
    def forbidden_present_names(self) -> frozenset[str]:
        return self.forbidden_names & self.present_names

    @property
    def valid(self) -> bool:
        return not self.missing_names and not self.forbidden_present_names


def _stored_authorization_bindings(panel: sqlite3.Row) -> dict[str, JsonValue]:
    try:
        payload = json.loads(bytes(panel["authorization_semantic_payload"]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise _failure_exception(
            "authorization_mismatch", "stored authorization payload is malformed"
        ) from error
    if not isinstance(payload, dict) or not isinstance(
        payload.get("frozen_bindings"), dict
    ):
        raise _failure_exception(
            "authorization_mismatch", "stored authorization bindings are malformed"
        )
    bindings = payload["frozen_bindings"]
    assert isinstance(bindings, dict)
    return bindings


def _validate_row_index_authorization_bindings(
    payload: RowIndexPayload, bindings: Mapping[str, JsonValue]
) -> None:
    sequence_units: dict[str, str] = {}
    for row in payload.rows:
        existing = sequence_units.setdefault(
            row.canonical_sequence_hash, row.analysis_unit_commitment
        )
        if existing != row.analysis_unit_commitment:
            raise _failure_exception(
                "malformed_sequence_identity",
                "one canonical sequence maps to multiple analysis units",
            )
    content_values = sorted(sequence_units)
    unit_values = sorted(set(sequence_units.values()))
    content_set = _f19_set_sha256(
        content_values,
        domain=_CONTENT_SET_DOMAIN,
        label="committed canonical content hashes",
    )
    unit_set = _f19_set_sha256(
        unit_values,
        domain=_ANALYSIS_UNIT_SET_DOMAIN,
        label="committed analysis-unit commitments",
    )
    observed = (
        content_values,
        content_set,
        unit_values,
        unit_set,
        len(content_values),
        len(unit_values),
        payload.row_count,
        payload.payload_sha256,
        _f19_partition_commitment(
            payload.partition_role,
            content_set_sha256=content_set,
            analysis_unit_set_sha256=unit_set,
        ),
    )
    expected = (
        bindings["canonical_content_sha256s"],
        bindings["content_set_sha256"],
        bindings["analysis_unit_commitments"],
        bindings["analysis_unit_set_sha256"],
        bindings["expected_sequence_count"],
        bindings["expected_analysis_unit_count"],
        bindings["row_count"],
        bindings["row_index_payload_sha256"],
        bindings["partition_commitment"],
    )
    if observed != expected:
        raise _failure_exception(
            "authorization_mismatch",
            "committed row index differs from frozen content/unit authorization",
        )


_COMMON_AUTHORIZATION_ARTIFACT_BINDINGS = {
    "semantic_canonicalization_spec_sha256": "semantic_canonicalization_spec.json",
    "corpus_role_classification_sha256": "corpus_role_classification.json",
    "independence_protocol_sha256": "independence_protocol.json",
    "split_manifest_sha256": "split_manifest.json",
    "source_inventory_sha256": "source_inventory.json",
    "model_identity_sha256": "model_identity.json",
    "carrier_identity_sha256": "fisher_carrier_identity.json",
    "runtime_identity_sha256": "runtime_identity.json",
    "execution_budget_sha256": "execution_budget.json",
    "row_derivation_protocol_artifact_sha256": "row_derivation_protocol.json",
}
_TUNE_AUTHORIZATION_ARTIFACT_BINDINGS = {
    "campaign_root_sha256": "campaign_root.json",
    "selection_representation_manifest_sha256": (
        "selection_representation_manifest.json"
    ),
    "codec_protocol_sha256": "codec_protocol.json",
    "ceiling_gate_protocol_sha256": "ceiling_gate_protocol.json",
}
_DEVELOPMENT_AUTHORIZATION_ARTIFACT_BINDINGS = {
    "campaign_root_sha256": "campaign_root.json",
    "frozen_representation_manifest_sha256": (
        "frozen_representation_manifest.json"
    ),
    "selected_codec_artifact_sha256": "selected_codec.json",
    "candidate_bundle_byte_sha256": "candidate_bundle.safetensors",
    "comparator_bundle_byte_sha256": "comparator_bundle.safetensors",
    "candidate_bundle_metadata_sha256": "candidate_bundle_metadata.json",
    "candidate_bundle_metadata_canonical_payload_sha256": (
        "candidate_bundle_metadata.json"
    ),
    "frozen_b_star_artifact_sha256": "frozen_B_star.json",
    "nonselecting_tune_replay_sha256": "nonselecting_tune_replay.json",
    "nonselecting_tune_replay_result_sha256": "nonselecting_tune_replay.json",
    "development_eligibility_protocol_sha256": (
        "development_eligibility_protocol.json"
    ),
}


def _campaign_inventory_content(
    connection: sqlite3.Connection, name: str
) -> tuple[bytes, str]:
    row = connection.execute(
        "SELECT * FROM campaign_artifact_inventory WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise _failure_exception(
            "authorization_mismatch",
            f"authorization dependency {name!r} is not registered",
        )
    source_kind = row["source_kind"]
    if source_kind == "external_reached_artifact_bytes":
        if row["persisted_content"] is None:
            raise _failure_exception(
                "artifact_hash_mismatch", f"artifact {name!r} lost persisted bytes"
            )
        content = bytes(row["persisted_content"])
    elif source_kind in {"external_reached_artifact_file", "capture_shard_file"}:
        if row["source_path"] is None:
            raise _failure_exception(
                "artifact_hash_mismatch", f"artifact {name!r} lost its source path"
            )
        try:
            content = Path(row["source_path"]).read_bytes()
        except OSError as error:
            raise _failure_exception(
                "artifact_hash_mismatch", f"artifact {name!r} cannot be reopened"
            ) from error
    elif source_kind in {"ledger_artifact", "campaign_terminal_artifact"}:
        artifact_row = connection.execute(
            "SELECT content FROM artifacts WHERE name = ?", (name,)
        ).fetchone()
        if artifact_row is None:
            raise _failure_exception(
                "artifact_hash_mismatch", f"ledger artifact {name!r} is missing"
            )
        content = bytes(artifact_row["content"])
    else:
        raise _failure_exception(
            "artifact_hash_mismatch", f"artifact {name!r} has unknown source kind"
        )
    digest = artifact_sha256(content)
    if digest != row["byte_sha256"]:
        raise _failure_exception(
            "artifact_hash_mismatch", f"artifact {name!r} changed after registration"
        )
    return content, digest


def _scientific_artifact_body(content: bytes, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _failure_exception(
            "authorization_mismatch", f"authorization dependency {name!r} is not JSON"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_name") != name
        or not isinstance(payload.get("body"), Mapping)
    ):
        raise _failure_exception(
            "authorization_mismatch",
            f"authorization dependency {name!r} has no bound body",
        )
    body = payload["body"]
    assert isinstance(body, Mapping)
    return body


def _validate_registered_authorization_bindings(
    connection: sqlite3.Connection, specification: AuthorizationSpec
) -> None:
    bindings = specification.frozen_bindings
    artifact_bindings = dict(_COMMON_AUTHORIZATION_ARTIFACT_BINDINGS)
    if specification.partition_role == "tune":
        artifact_bindings.update(_TUNE_AUTHORIZATION_ARTIFACT_BINDINGS)
    elif specification.partition_role == "development_confirm":
        artifact_bindings.update(_DEVELOPMENT_AUTHORIZATION_ARTIFACT_BINDINGS)

    contents: dict[str, bytes] = {}
    for binding_key, artifact_name in artifact_bindings.items():
        if artifact_name not in contents:
            contents[artifact_name], observed_hash = _campaign_inventory_content(
                connection, artifact_name
            )
        else:
            observed_hash = artifact_sha256(contents[artifact_name])
        if bindings[binding_key] != observed_hash:
            raise _failure_exception(
                "authorization_mismatch",
                f"frozen binding {binding_key!r} differs from {artifact_name!r}",
            )

    model = _scientific_artifact_body(
        contents["model_identity.json"], name="model_identity.json"
    )
    carrier = _scientific_artifact_body(
        contents["fisher_carrier_identity.json"],
        name="fisher_carrier_identity.json",
    )
    row_protocol = _scientific_artifact_body(
        contents["row_derivation_protocol.json"],
        name="row_derivation_protocol.json",
    )
    try:
        nested_observed = {
            "model_immutable_revision": model["immutable_revision"],
            "model_weight_file_byte_sha256s": [
                item["artifact_byte_sha256"] for item in model["weight_files"]
            ],
            "model_config_byte_sha256": model["config_file"][
                "artifact_byte_sha256"
            ],
            "tokenizer_identity_sha256": row_protocol["tokenizer_identity_sha256"],
            "tokenizer_file_byte_sha256s": [
                item["artifact_byte_sha256"] for item in model["tokenizer_files"]
            ],
            "tokenizer_config_byte_sha256": model["tokenizer_config_file"][
                "artifact_byte_sha256"
            ],
            "special_vocabulary_map_byte_sha256": model[
                "special_tokens_map_file"
            ]["artifact_byte_sha256"],
            "h4_carrier_byte_sha256": carrier["carrier_byte_sha256"],
        }
    except (KeyError, TypeError) as error:
        raise _failure_exception(
            "authorization_mismatch",
            "model/tokenizer/carrier authorization dependency is malformed",
        ) from error
    for key, observed in nested_observed.items():
        if bindings[key] != observed:
            raise _failure_exception(
                "authorization_mismatch",
                f"frozen nested identity {key!r} differs from its artifact",
            )

    if specification.partition_role == "development_confirm":
        selected = _scientific_artifact_body(
            contents["selected_codec.json"], name="selected_codec.json"
        )
        frozen_b = _scientific_artifact_body(
            contents["frozen_B_star.json"], name="frozen_B_star.json"
        )
        if (
            bindings["selected_codec_id"] != selected.get("codec_id")
            or bindings["frozen_b_star_model_id"] != frozen_b.get("model_id")
        ):
            raise _failure_exception(
                "authorization_mismatch",
                "selected codec or frozen B* identity differs from its artifact",
            )


class ProtocolLedger:
    """Durable SQLite implementation of the K-series panel state machine.

    The caller owns this object and must call :meth:`close` (or use it as a
    context manager).  ``synchronous=FULL`` and ``BEGIN IMMEDIATE`` make every
    protocol mutation a single durable SQLite transaction.  A ``:memory:``
    database is supported for tests but is, by definition, not durable.
    """

    _SCHEMA_VERSION = "4"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        if not self.path:
            raise ProtocolSchemaError("ledger path must not be empty")
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = FULL")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def __enter__(self) -> "ProtocolLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS protocol_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS panels (
                    partition_role TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    authorization_hash TEXT NOT NULL,
                    authorization_transaction_id TEXT NOT NULL,
                    authorization_semantic_payload BLOB NOT NULL,
                    expected_row_set_mode TEXT NOT NULL,
                    row_derivation_protocol_hash TEXT NOT NULL,
                    expected_row_set_sha256 TEXT,
                    expected_ordered_row_sequence_sha256 TEXT,
                    expected_row_count INTEGER,
                    expected_row_index_payload_sha256 TEXT,
                    row_index_transaction_id TEXT,
                    row_index_payload_sha256 TEXT,
                    committed_row_set_sha256 TEXT,
                    committed_ordered_row_sequence_sha256 TEXT,
                    committed_row_count INTEGER,
                    row_index_failure_transaction_id TEXT,
                    capture_start_transaction_id TEXT,
                    capture_start_identity_sha256 TEXT,
                    capture_attempt_id TEXT,
                    capture_root_identity_sha256 TEXT,
                    shard_plan_transaction_id TEXT,
                    shard_plan_sha256 TEXT,
                    capture_schema_hash TEXT,
                    capture_implementation_hash TEXT,
                    resume_checkpoint_count INTEGER NOT NULL DEFAULT 0,
                    resume_checkpoint_sha256 TEXT,
                    terminal_transaction_id TEXT,
                    terminal_identity_sha256 TEXT,
                    terminal_state TEXT,
                    failure_reason_code TEXT,
                    revision INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS ledger_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    transaction_kind TEXT NOT NULL,
                    partition_role TEXT NOT NULL,
                    state_before TEXT NOT NULL,
                    state_after TEXT NOT NULL,
                    identity_sha256 TEXT NOT NULL,
                    commit_receipt BLOB NOT NULL,
                    commit_receipt_sha256 TEXT NOT NULL,
                    FOREIGN KEY(partition_role) REFERENCES panels(partition_role)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    name TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    partition_role TEXT NOT NULL,
                    content BLOB NOT NULL,
                    byte_sha256 TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    FOREIGN KEY(partition_role) REFERENCES panels(partition_role),
                    FOREIGN KEY(transaction_id)
                        REFERENCES ledger_transactions(transaction_id)
                );

                CREATE INDEX IF NOT EXISTS artifacts_by_transaction
                    ON artifacts(transaction_id, name);

                CREATE TABLE IF NOT EXISTS campaign_artifact_inventory (
                    name TEXT PRIMARY KEY,
                    prefix TEXT NOT NULL,
                    byte_sha256 TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    materialization_ordinal INTEGER NOT NULL UNIQUE,
                    source_kind TEXT NOT NULL,
                    transaction_id TEXT,
                    persisted_content BLOB,
                    source_path TEXT,
                    source_identity_sha256 TEXT
                );

                CREATE INDEX IF NOT EXISTS campaign_inventory_by_ordinal
                    ON campaign_artifact_inventory(materialization_ordinal);

                CREATE TABLE IF NOT EXISTS capture_resume_shards (
                    partition_role TEXT NOT NULL,
                    capture_attempt_id TEXT NOT NULL,
                    shard_index INTEGER NOT NULL,
                    shard_plan_sha256 TEXT NOT NULL,
                    capture_root_identity_sha256 TEXT NOT NULL,
                    verified_record BLOB NOT NULL,
                    verified_record_sha256 TEXT NOT NULL,
                    PRIMARY KEY(
                        partition_role, capture_attempt_id, shard_index
                    ),
                    FOREIGN KEY(partition_role)
                        REFERENCES panels(partition_role)
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM protocol_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                    ("schema_version", self._SCHEMA_VERSION),
                )
            elif row["value"] != self._SCHEMA_VERSION:
                raise ProtocolStateError("unsupported protocol ledger schema")
            ordinal_row = self._connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'next_materialization_ordinal'"
            ).fetchone()
            if ordinal_row is None:
                self._connection.execute(
                    "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                    ("next_materialization_ordinal", "1"),
                )

    @contextmanager
    def _immediate_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _new_transaction_id() -> str:
        value = secrets.token_hex(16)
        return _validate_tx_id(value, label="generated transaction ID")

    @staticmethod
    def _new_capture_attempt_id() -> str:
        value = secrets.token_hex(16)
        return _validate_tx_id(value, label="generated capture attempt ID")

    @staticmethod
    def _artifact(
        *,
        name: str,
        kind: str,
        payload: Mapping[str, object],
        transaction_id: str,
    ) -> StoredArtifact:
        _validate_relative_path(name, label="artifact name")
        _validate_identifier(kind, label="artifact kind")
        _validate_tx_id(transaction_id, label="transaction_id")
        content = canonical_json_bytes(dict(payload))
        return StoredArtifact(
            name=name,
            kind=kind,
            content=content,
            byte_sha256=artifact_sha256(content),
            transaction_id=transaction_id,
        )

    @staticmethod
    def _default_prefix_for_partition(partition_role: str) -> CampaignPrefix:
        return {
            "fit": CampaignPrefix.P3,
            "tune": CampaignPrefix.P5,
            "development_confirm": CampaignPrefix.P10,
        }[_validate_partition_role(partition_role)]

    @staticmethod
    def _allocate_materialization_ordinal(
        connection: sqlite3.Connection,
    ) -> int:
        row = connection.execute(
            "SELECT value FROM protocol_metadata "
            "WHERE key = 'next_materialization_ordinal'"
        ).fetchone()
        if row is None:
            raise ProtocolStateError("materialization counter is missing")
        try:
            ordinal = int(row["value"])
        except (TypeError, ValueError) as error:
            raise ProtocolStateError(
                "materialization counter is malformed"
            ) from error
        _validate_uint64(ordinal, label="materialization ordinal")
        connection.execute(
            "UPDATE protocol_metadata SET value = ? "
            "WHERE key = 'next_materialization_ordinal'",
            (str(ordinal + 1),),
        )
        return ordinal

    @classmethod
    def _register_inventory_record(
        cls,
        connection: sqlite3.Connection,
        *,
        name: str,
        prefix: CampaignPrefix,
        byte_sha256: str,
        byte_count: int,
        source_kind: str,
        transaction_id: str | None,
        persisted_content: bytes | None = None,
        source_path: str | None = None,
        source_identity_sha256: str | None = None,
    ) -> ArtifactInventoryEntry:
        _validate_relative_path(name, label="inventory artifact name")
        _validate_sha256(byte_sha256, label="inventory artifact hash")
        if _validate_uint64(byte_count, label="inventory byte count") == 0:
            raise ProtocolSchemaError("inventory artifact must be nonempty")
        if not isinstance(prefix, CampaignPrefix):
            raise ProtocolSchemaError("inventory prefix must be CampaignPrefix")
        _validate_identifier(source_kind, label="inventory source kind")
        if transaction_id is not None:
            _validate_tx_id(transaction_id, label="inventory transaction ID")
        if persisted_content is not None and not isinstance(
            persisted_content, bytes
        ):
            raise ProtocolSchemaError("persisted content must be exact bytes")
        if source_path is not None and not isinstance(source_path, str):
            raise ProtocolSchemaError("source path must be a string")
        if source_identity_sha256 is not None:
            _validate_sha256(
                source_identity_sha256, label="source identity hash"
            )
        if source_kind == "external_reached_artifact_bytes":
            if (
                persisted_content is None
                or source_path is not None
                or source_identity_sha256 is not None
            ):
                raise ProtocolSchemaError(
                    "byte-backed external artifact evidence is malformed"
                )
            if (
                len(persisted_content) != byte_count
                or artifact_sha256(persisted_content) != byte_sha256
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "persisted external artifact bytes differ from inventory",
                )
        elif source_kind in {
            "external_reached_artifact_file",
            "capture_shard_file",
        }:
            if (
                persisted_content is not None
                or source_path is None
                or source_identity_sha256 is None
            ):
                raise ProtocolSchemaError(
                    "file-backed external artifact evidence is malformed"
                )
            if (
                source_kind == "capture_shard_file"
                and not _CAPTURE_SHARD_ARTIFACT_RE.fullmatch(name)
            ):
                raise ProtocolSchemaError(
                    "capture-shard evidence name is outside the exact namespace"
                )
        elif (
            persisted_content is not None
            or source_path is not None
            or source_identity_sha256 is not None
        ):
            raise ProtocolSchemaError(
                "internal artifact inventory cannot carry external evidence"
            )
        frozen_prefix = _ARTIFACT_PREFIX.get(name)
        if frozen_prefix is not None and frozen_prefix is not prefix:
            raise _failure_exception(
                "artifact_hash_mismatch",
                f"artifact {name!r} is assigned to the wrong prefix",
            )
        existing = connection.execute(
            "SELECT * FROM campaign_artifact_inventory WHERE name = ?",
            (name,),
        ).fetchone()
        if existing is not None:
            observed = (
                existing["prefix"],
                existing["byte_sha256"],
                existing["byte_count"],
                existing["source_kind"],
                existing["transaction_id"],
                (
                    None
                    if existing["persisted_content"] is None
                    else bytes(existing["persisted_content"])
                ),
                existing["source_path"],
                existing["source_identity_sha256"],
            )
            expected = (
                prefix.name,
                byte_sha256,
                byte_count,
                source_kind,
                transaction_id,
                persisted_content,
                source_path,
                source_identity_sha256,
            )
            if observed != expected:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"artifact {name!r} was already registered differently",
                )
            return ArtifactInventoryEntry(
                name=name,
                artifact_byte_sha256=byte_sha256,
                byte_count=byte_count,
                prefix=prefix,
                materialization_ordinal=existing["materialization_ordinal"],
            )
        folded_name = name.casefold()
        canonical_frozen_name = _FROZEN_ARTIFACT_NAMES_CASEFOLD.get(folded_name)
        if canonical_frozen_name is not None and canonical_frozen_name != name:
            raise ProtocolSchemaError(
                f"artifact {name!r} is a noncanonical case alias of "
                f"{canonical_frozen_name!r}"
            )
        case_alias = connection.execute(
            "SELECT name FROM campaign_artifact_inventory "
            "WHERE lower(name) = lower(?) AND name <> ? LIMIT 1",
            (name, name),
        ).fetchone()
        if case_alias is not None:
            raise ProtocolSchemaError(
                f"artifact {name!r} casefold-collides with "
                f"{case_alias['name']!r}"
            )
        if (
            source_kind.startswith("external_reached_artifact_")
            and (
                _is_ledger_owned_artifact_name(name)
                or _is_ledger_owned_artifact_casefold_alias(name)
            )
        ):
            raise ProtocolSchemaError(
                f"artifact {name!r} is reserved for the protocol ledger"
            )
        required_success_role = _SAME_PREFIX_SUCCESS_DEPENDENCIES.get(name)
        if required_success_role is not None:
            dependency = connection.execute(
                "SELECT state FROM panels WHERE partition_role = ?",
                (required_success_role,),
            ).fetchone()
            if (
                dependency is None
                or CaptureState(dependency["state"])
                is not CaptureState.CONSUMED_SUCCESS
            ):
                raise ProtocolStateError(
                    f"artifact {name!r} requires successful "
                    f"{required_success_role} capture"
                )
        last = connection.execute(
            "SELECT name, prefix FROM campaign_artifact_inventory "
            "ORDER BY materialization_ordinal DESC LIMIT 1"
        ).fetchone()
        if last is not None:
            last_prefix = _coerce_prefix(
                last["prefix"], label="last materialized prefix"
            )
            if prefix.value < last_prefix.value:
                raise ProtocolStateError(
                    f"artifact prefix regression from {last_prefix.name} "
                    f"to {prefix.name} is forbidden"
                )
        present_names = frozenset(
            row["name"]
            for row in connection.execute(
                "SELECT name FROM campaign_artifact_inventory"
            ).fetchall()
        )
        required_predecessors: set[str] = set()
        for predecessor in CampaignPrefix:
            if predecessor.value < prefix.value:
                required_predecessors.update(
                    _PREFIX_MANDATORY_ARTIFACTS[predecessor]
                )
        missing_predecessors = required_predecessors - present_names
        if missing_predecessors:
            raise ProtocolStateError(
                f"cannot materialize {prefix.name} before all mandatory "
                "predecessor artifacts"
            )
        ordinal = cls._allocate_materialization_ordinal(connection)
        connection.execute(
            """
            INSERT INTO campaign_artifact_inventory(
                name, prefix, byte_sha256, byte_count,
                materialization_ordinal, source_kind, transaction_id,
                persisted_content, source_path, source_identity_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                prefix.name,
                byte_sha256,
                byte_count,
                ordinal,
                source_kind,
                transaction_id,
                persisted_content,
                source_path,
                source_identity_sha256,
            ),
        )
        return ArtifactInventoryEntry(
            name=name,
            artifact_byte_sha256=byte_sha256,
            byte_count=byte_count,
            prefix=prefix,
            materialization_ordinal=ordinal,
        )

    def _commit_transaction_record(
        self,
        connection: sqlite3.Connection,
        *,
        transaction_id: str,
        transaction_kind: str,
        partition_role: str,
        state_before: CaptureState,
        state_after: CaptureState,
        identity_sha256: str,
        artifacts: Sequence[StoredArtifact],
        receipt_name: str,
        campaign_prefix: CampaignPrefix | None = None,
        allow_terminal_pending_transaction_id: str | None = None,
    ) -> tuple[StoredArtifact, ...]:
        self._ensure_no_global_terminal(
            connection,
            allow_pending_transaction_id=(
                allow_terminal_pending_transaction_id
            ),
        )
        inventory = [
            {"byte_sha256": artifact.byte_sha256, "name": artifact.name}
            for artifact in sorted(artifacts, key=lambda item: item.name)
        ]
        receipt = self._artifact(
            name=receipt_name,
            kind="transaction_commit_receipt",
            transaction_id=transaction_id,
            payload={
                "artifacts": inventory,
                "format_version": _FORMAT_VERSION,
                "identity_sha256": identity_sha256,
                "partition_role": partition_role,
                "schema": LEDGER_SCHEMA,
                "state_after": state_after.value,
                "state_before": state_before.value,
                "transaction_committed": True,
                "transaction_id": transaction_id,
                "transaction_kind": transaction_kind,
            },
        )
        all_artifacts = tuple(artifacts) + (receipt,)
        connection.execute(
            """
            INSERT INTO ledger_transactions(
                transaction_id, transaction_kind, partition_role,
                state_before, state_after, identity_sha256,
                commit_receipt, commit_receipt_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                transaction_kind,
                partition_role,
                state_before.value,
                state_after.value,
                identity_sha256,
                receipt.content,
                receipt.byte_sha256,
            ),
        )
        for artifact in all_artifacts:
            connection.execute(
                """
                INSERT INTO artifacts(
                    name, kind, partition_role, content,
                    byte_sha256, transaction_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.name,
                    artifact.kind,
                    partition_role,
                    artifact.content,
                    artifact.byte_sha256,
                    artifact.transaction_id,
                ),
            )
            self._register_inventory_record(
                connection,
                name=artifact.name,
                prefix=(
                    campaign_prefix
                    if campaign_prefix is not None
                    else self._default_prefix_for_partition(partition_role)
                ),
                byte_sha256=artifact.byte_sha256,
                byte_count=len(artifact.content),
                source_kind="ledger_artifact",
                transaction_id=transaction_id,
            )
        return all_artifacts

    @staticmethod
    def _ensure_no_global_terminal(
        connection: sqlite3.Connection,
        *,
        allow_pending_transaction_id: str | None = None,
    ) -> None:
        row = connection.execute(
            "SELECT value FROM protocol_metadata "
            "WHERE key = 'global_terminal_state'"
        ).fetchone()
        if row is not None:
            if allow_pending_transaction_id is not None:
                _validate_tx_id(
                    allow_pending_transaction_id,
                    label="allowed pending transaction ID",
                )
                phase = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_phase'"
                ).fetchone()
                origin = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_origin_transaction_id'"
                ).fetchone()
                if (
                    phase is not None
                    and phase["value"] == "failure_recorded"
                    and origin is not None
                    and origin["value"] == allow_pending_transaction_id
                ):
                    return
            raise ProtocolStateError(
                f"campaign is globally terminal: {row['value']}"
            )

    @staticmethod
    def _set_global_terminal_pending(
        connection: sqlite3.Connection,
        *,
        terminal_state: str,
        terminal_identity_sha256: str,
        origin_transaction_id: str,
        partition_role: str,
    ) -> None:
        ProtocolLedger._ensure_no_global_terminal(connection)
        _validate_sha256(
            terminal_identity_sha256,
            label="global pending terminal identity",
        )
        _validate_tx_id(
            origin_transaction_id,
            label="global pending origin transaction ID",
        )
        _validate_partition_role(partition_role)
        for key, value in (
            ("global_terminal_state", terminal_state),
            ("global_terminal_identity_sha256", terminal_identity_sha256),
            ("global_terminal_origin_transaction_id", origin_transaction_id),
            ("global_terminal_transaction_id", origin_transaction_id),
            ("global_terminal_partition_role", partition_role),
            ("global_terminal_phase", "failure_recorded"),
        ):
            connection.execute(
                "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                (key, value),
            )

    @staticmethod
    def _set_global_terminal(
        connection: sqlite3.Connection,
        *,
        terminal_state: str,
        terminal_identity_sha256: str,
        terminal_transaction_id: str,
        partition_role: str,
        pending_origin_transaction_id: str | None = None,
    ) -> None:
        ProtocolLedger._ensure_no_global_terminal(
            connection,
            allow_pending_transaction_id=pending_origin_transaction_id,
        )
        _validate_sha256(
            terminal_identity_sha256,
            label="global terminal identity",
        )
        _validate_tx_id(
            terminal_transaction_id,
            label="global terminal transaction ID",
        )
        _validate_partition_role(partition_role)
        values = (
            ("global_terminal_state", terminal_state),
            ("global_terminal_identity_sha256", terminal_identity_sha256),
            ("global_terminal_transaction_id", terminal_transaction_id),
            ("global_terminal_partition_role", partition_role),
            ("global_terminal_phase", "primary_committed"),
        )
        for key, value in values:
            connection.execute(
                "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    @staticmethod
    def _strict_validate_public_artifact_bytes(
        name: str, content: bytes
    ) -> None:
        if not content:
            raise ProtocolSchemaError("campaign artifact must be nonempty")
        if name.endswith(".json"):
            try:
                value = json.loads(
                    content.decode("utf-8"),
                    object_pairs_hook=_json_object_without_duplicates,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ProtocolSchemaError(
                            f"nonfinite JSON constant {token!r} is forbidden"
                        )
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProtocolSchemaError(
                    "campaign JSON artifact is not strict UTF-8 JSON"
                ) from error
            if not isinstance(value, dict):
                raise ProtocolSchemaError(
                    "campaign JSON artifact must contain one object"
                )
            stack: list[object] = [value]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    for key, child in current.items():
                        if _is_protected_raw_value_key(key):
                            raise ProtocolSchemaError(
                                f"campaign artifact contains protected field {key!r}"
                            )
                        stack.append(child)
                elif isinstance(current, list):
                    stack.extend(current)
        elif name.endswith(".md"):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ProtocolSchemaError(
                    "campaign Markdown artifact must be UTF-8"
                ) from error
            if "\x00" in text:
                raise ProtocolSchemaError(
                    "campaign Markdown artifact must not contain NUL"
                )

    @classmethod
    def _inspect_external_artifact_file(
        cls,
        name: str,
        path: str | os.PathLike[str],
    ) -> tuple[str, int, str, str]:
        candidate = Path(path)
        try:
            before = os.lstat(candidate)
        except OSError as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                f"external artifact {name!r} is unavailable",
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise _failure_exception(
                "artifact_hash_mismatch",
                "external artifact path must not be a symlink",
            )
        if not stat.S_ISREG(before.st_mode):
            raise _failure_exception(
                "artifact_hash_mismatch",
                "external artifact path must be a regular file",
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "external artifact path cannot be resolved",
            ) from error
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(candidate, flags)
        except OSError as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "external artifact cannot be opened without following links",
            ) from error
        digest = hashlib.sha256()
        textual = name.endswith(".json") or name.endswith(".md")
        text_buffer = bytearray()
        try:
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_before.st_mode)
                or opened_before.st_dev != before.st_dev
                or opened_before.st_ino != before.st_ino
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "external artifact identity changed before read",
                )
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                if textual:
                    text_buffer.extend(chunk)
                    if (
                        len(text_buffer)
                        > _MAX_PERSISTED_EXTERNAL_ARTIFACT_BYTES
                    ):
                        raise ProtocolSchemaError(
                            "external JSON/Markdown artifact exceeds size bound"
                        )
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(candidate)
        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        if not (
            identity(before)
            == identity(opened_before)
            == identity(opened_after)
            == identity(after)
        ):
            raise _failure_exception(
                "artifact_hash_mismatch",
                "external artifact changed during inspection",
            )
        if before.st_size == 0:
            raise ProtocolSchemaError("campaign artifact must be nonempty")
        if textual:
            cls._strict_validate_public_artifact_bytes(
                name, bytes(text_buffer)
            )
        resolved_text = os.fspath(resolved)
        identity_sha256 = hashlib.sha256(
            b"fisher-compact-message-completion/external-artifact-file/v1\x00"
            + canonical_json_bytes(
                {
                    "byte_count": before.st_size,
                    "device": before.st_dev,
                    "inode": before.st_ino,
                    "modified_time_ns": before.st_mtime_ns,
                    "path_sha256": hashlib.sha256(
                        os.fsencode(resolved_text)
                    ).hexdigest(),
                    "status_change_time_ns": before.st_ctime_ns,
                }
            )
        ).hexdigest()
        return digest.hexdigest(), before.st_size, resolved_text, identity_sha256

    def register_reached_artifact(
        self,
        *,
        campaign_id: str,
        prefix: CampaignPrefix | str,
        name: str,
        content: bytes,
    ) -> ArtifactInventoryEntry:
        """Persist one bounded, schema-public artifact as exact byte evidence.

        The exact sanitized bytes, byte hash, byte count, prefix, and
        materialization order are committed in the ledger.  This API is
        deliberately size-bounded so it cannot become a second store for large
        scientific tensors; use :meth:`register_reached_artifact_file` for
        immutable file-backed evidence.  JSON/Markdown framing is checked here;
        the artifact family's domain schema must be validated by its producer.
        """

        campaign = _validate_identifier(campaign_id, label="campaign_id")
        if campaign == CURRENT_P0_CAMPAIGN_ID:
            raise ProtocolSchemaError(
                "the finalized current-P0 campaign cannot be reopened"
            )
        artifact_name = _validate_relative_path(
            name, label="campaign artifact name"
        )
        artifact_prefix = _coerce_prefix(prefix, label="prefix")
        if not isinstance(content, bytes):
            raise ProtocolSchemaError("content must be exact bytes")
        if len(content) > _MAX_PERSISTED_EXTERNAL_ARTIFACT_BYTES:
            raise ProtocolSchemaError(
                "external artifact bytes exceed the persisted size bound; "
                "use register_reached_artifact_file"
            )
        self._strict_validate_public_artifact_bytes(artifact_name, content)
        digest = artifact_sha256(content)
        with self._immediate_transaction() as connection:
            self._ensure_no_global_terminal(connection)
            campaign_row = connection.execute(
                "SELECT value FROM protocol_metadata WHERE key = 'campaign_id'"
            ).fetchone()
            if campaign_row is None:
                connection.execute(
                    "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                    ("campaign_id", campaign),
                )
            elif campaign_row["value"] != campaign:
                raise _failure_exception(
                    "authorization_mismatch",
                    "ledger already belongs to a different campaign",
                )
            entry = self._register_inventory_record(
                connection,
                name=artifact_name,
                prefix=artifact_prefix,
                byte_sha256=digest,
                byte_count=len(content),
                source_kind="external_reached_artifact_bytes",
                transaction_id=None,
                persisted_content=content,
            )
        return entry

    def register_reached_artifact_file(
        self,
        *,
        campaign_id: str,
        prefix: CampaignPrefix | str,
        name: str,
        path: str | os.PathLike[str],
    ) -> ArtifactInventoryEntry:
        """Register immutable file-backed evidence and bind its exact identity."""

        campaign = _validate_identifier(campaign_id, label="campaign_id")
        if campaign == CURRENT_P0_CAMPAIGN_ID:
            raise ProtocolSchemaError(
                "the finalized current-P0 campaign cannot be reopened"
            )
        artifact_name = _validate_relative_path(
            name, label="campaign artifact name"
        )
        artifact_prefix = _coerce_prefix(prefix, label="prefix")
        digest, byte_count, resolved_path, identity_hash = (
            self._inspect_external_artifact_file(artifact_name, path)
        )
        with self._immediate_transaction() as connection:
            self._ensure_no_global_terminal(connection)
            campaign_row = connection.execute(
                "SELECT value FROM protocol_metadata WHERE key = 'campaign_id'"
            ).fetchone()
            if campaign_row is None:
                connection.execute(
                    "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                    ("campaign_id", campaign),
                )
            elif campaign_row["value"] != campaign:
                raise _failure_exception(
                    "authorization_mismatch",
                    "ledger already belongs to a different campaign",
                )
            entry = self._register_inventory_record(
                connection,
                name=artifact_name,
                prefix=artifact_prefix,
                byte_sha256=digest,
                byte_count=byte_count,
                source_kind="external_reached_artifact_file",
                transaction_id=None,
                source_path=resolved_path,
                source_identity_sha256=identity_hash,
            )
        return entry

    @staticmethod
    def _campaign_inventory_entries(
        connection: sqlite3.Connection,
    ) -> tuple[ArtifactInventoryEntry, ...]:
        rows = connection.execute(
            "SELECT * FROM campaign_artifact_inventory "
            "ORDER BY materialization_ordinal"
        ).fetchall()
        entries: list[ArtifactInventoryEntry] = []
        for row in rows:
            entry = ArtifactInventoryEntry(
                name=row["name"],
                artifact_byte_sha256=row["byte_sha256"],
                byte_count=row["byte_count"],
                prefix=_coerce_prefix(row["prefix"], label="stored prefix"),
                materialization_ordinal=row["materialization_ordinal"],
            )
            if (
                row["source_kind"].startswith("external_reached_artifact_")
                and _is_ledger_owned_artifact_name(entry.name)
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"ledger-owned name {entry.name!r} has external evidence",
                )
            if bool(_CAPTURE_SHARD_ARTIFACT_RE.fullmatch(entry.name)) != (
                row["source_kind"] == "capture_shard_file"
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "capture shard inventory source/name binding changed",
                )
            if row["source_kind"] in {
                "ledger_artifact",
                "campaign_terminal_artifact",
            }:
                artifact_row = connection.execute(
                    "SELECT content, byte_sha256, transaction_id "
                    "FROM artifacts WHERE name = ?",
                    (entry.name,),
                ).fetchone()
                if artifact_row is None:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"inventoried ledger artifact {entry.name!r} is missing",
                    )
                exact_content = bytes(artifact_row["content"])
                if (
                    artifact_sha256(exact_content)
                    != entry.artifact_byte_sha256
                    or artifact_row["byte_sha256"]
                    != entry.artifact_byte_sha256
                    or len(exact_content) != entry.byte_count
                    or artifact_row["transaction_id"] != row["transaction_id"]
                ):
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"inventoried ledger artifact {entry.name!r} changed",
                    )
            elif row["source_kind"] == "external_reached_artifact_bytes":
                if row["persisted_content"] is None:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"external artifact {entry.name!r} lost persisted bytes",
                    )
                content = bytes(row["persisted_content"])
                if (
                    len(content) != entry.byte_count
                    or artifact_sha256(content)
                    != entry.artifact_byte_sha256
                ):
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"persisted external artifact {entry.name!r} changed",
                    )
                ProtocolLedger._strict_validate_public_artifact_bytes(
                    entry.name, content
                )
            elif row["source_kind"] in {
                "external_reached_artifact_file",
                "capture_shard_file",
            }:
                if (
                    row["source_path"] is None
                    or row["source_identity_sha256"] is None
                ):
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"file-backed artifact {entry.name!r} lost source evidence",
                    )
                digest, count, resolved_path, identity_hash = (
                    ProtocolLedger._inspect_external_artifact_file(
                        entry.name, row["source_path"]
                    )
                )
                if (
                    digest != entry.artifact_byte_sha256
                    or count != entry.byte_count
                    or resolved_path != row["source_path"]
                    or identity_hash != row["source_identity_sha256"]
                ):
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"file-backed external artifact {entry.name!r} changed",
                    )
            else:
                raise ProtocolStateError("unknown campaign inventory source")
            entries.append(entry)
        seen_names: set[str] = set()
        seen_casefold_names: dict[str, str] = {}
        previous_entry: ArtifactInventoryEntry | None = None
        for expected_ordinal, entry in enumerate(entries, start=1):
            if entry.materialization_ordinal != expected_ordinal:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "campaign inventory ordinals are not contiguous",
                )
            previous_spelling = seen_casefold_names.get(entry.name.casefold())
            if previous_spelling is not None and previous_spelling != entry.name:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "campaign inventory contains casefold-duplicate names",
                )
            seen_casefold_names[entry.name.casefold()] = entry.name
            canonical_frozen_name = _FROZEN_ARTIFACT_NAMES_CASEFOLD.get(
                entry.name.casefold()
            )
            if (
                canonical_frozen_name is not None
                and canonical_frozen_name != entry.name
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"campaign artifact {entry.name!r} has noncanonical case",
                )
            frozen_prefix = _ARTIFACT_PREFIX.get(entry.name)
            if frozen_prefix is not None and frozen_prefix is not entry.prefix:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"campaign artifact {entry.name!r} has the wrong prefix",
                )
            if (
                previous_entry is not None
                and entry.prefix.value < previous_entry.prefix.value
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "campaign inventory prefix order regressed",
                )
            required_predecessors: set[str] = set()
            for predecessor in CampaignPrefix:
                if predecessor.value < entry.prefix.value:
                    required_predecessors.update(
                        _PREFIX_MANDATORY_ARTIFACTS[predecessor]
                    )
            if not required_predecessors <= seen_names:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "campaign inventory advanced before completing predecessors",
                )
            seen_names.add(entry.name)
            previous_entry = entry
        return tuple(entries)

    @classmethod
    def _register_capture_shard_files(
        cls,
        connection: sqlite3.Connection,
        *,
        plan: ShardPlan,
        capture_root: str | os.PathLike[str],
        shard_indexes: Iterable[int],
        prefix: CampaignPrefix,
        transaction_id: str,
        expected_hashes: Mapping[int, str] | None = None,
    ) -> tuple[ArtifactInventoryEntry, ...]:
        """Bind exact finalized shard files into the public campaign inventory."""

        indexes = tuple(sorted(shard_indexes))
        if len(indexes) != len(set(indexes)):
            raise ProtocolSchemaError("capture shard inventory indexes repeat")
        expected = {} if expected_hashes is None else dict(expected_hashes)
        if set(expected) - set(indexes):
            raise ProtocolSchemaError("capture shard hash map has unknown indexes")
        registered: list[ArtifactInventoryEntry] = []
        root = Path(capture_root)
        for index in indexes:
            _validate_uint32(index, label="capture shard inventory index")
            if index >= len(plan.entries):
                raise ProtocolSchemaError("capture shard inventory index is unknown")
            plan_entry = plan.entries[index]
            digest, byte_count, resolved_path, identity_hash = (
                cls._inspect_external_artifact_file(
                    plan_entry.path,
                    root.joinpath(*PurePosixPath(plan_entry.path).parts),
                )
            )
            if index in expected and expected[index] != digest:
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "finalized shard changed before durable inventory commit",
                )
            registered.append(
                cls._register_inventory_record(
                    connection,
                    name=plan_entry.path,
                    prefix=prefix,
                    byte_sha256=digest,
                    byte_count=byte_count,
                    source_kind="capture_shard_file",
                    transaction_id=transaction_id,
                    source_path=resolved_path,
                    source_identity_sha256=identity_hash,
                )
            )
        return tuple(registered)

    def campaign_inventory(self) -> tuple[ArtifactInventoryEntry, ...]:
        """Return the durability-verified inventory in materialization order."""

        with self._lock:
            return self._campaign_inventory_entries(self._connection)

    @staticmethod
    def _preview_inventory_entry(
        artifact: StoredArtifact,
        *,
        prefix: CampaignPrefix,
        materialization_ordinal: int,
    ) -> ArtifactInventoryEntry:
        return ArtifactInventoryEntry(
            name=artifact.name,
            artifact_byte_sha256=artifact.byte_sha256,
            byte_count=len(artifact.content),
            prefix=prefix,
            materialization_ordinal=materialization_ordinal,
        )

    def _preview_reached_inventory(
        self,
        connection: sqlite3.Connection,
        artifacts: Sequence[StoredArtifact],
        *,
        prefix: CampaignPrefix,
    ) -> tuple[ArtifactInventoryEntry, ...]:
        existing = self._campaign_inventory_entries(connection)
        row = connection.execute(
            "SELECT value FROM protocol_metadata "
            "WHERE key = 'next_materialization_ordinal'"
        ).fetchone()
        if row is None:
            raise ProtocolStateError("materialization counter is missing")
        ordinal = int(row["value"])
        previews: list[ArtifactInventoryEntry] = []
        for artifact in artifacts:
            previews.append(
                self._preview_inventory_entry(
                    artifact,
                    prefix=prefix,
                    materialization_ordinal=ordinal,
                )
            )
            ordinal += 1
        return existing + tuple(previews)

    def _validate_all_ledger_transactions(self) -> None:
        """Strictly recover every prior transaction and its commit receipt."""

        missing_inventory = self._connection.execute(
            "SELECT a.name FROM artifacts AS a "
            "LEFT JOIN campaign_artifact_inventory AS i ON i.name = a.name "
            "WHERE i.name IS NULL ORDER BY a.name LIMIT 1"
        ).fetchone()
        if missing_inventory is not None:
            raise _failure_exception(
                "artifact_hash_mismatch",
                f"ledger artifact {missing_inventory['name']!r} is absent from campaign inventory",
            )
        transaction_ids = tuple(
            row["transaction_id"]
            for row in self._connection.execute(
                "SELECT transaction_id FROM ledger_transactions ORDER BY rowid"
            ).fetchall()
        )
        for transaction_id in transaction_ids:
            self._load_transaction(transaction_id, idempotent=True)
        checkpoint_panels = self._connection.execute(
            "SELECT partition_role, capture_attempt_id, shard_plan_sha256, "
            "capture_root_identity_sha256 FROM panels "
            "WHERE resume_checkpoint_count > 0"
        ).fetchall()
        for panel in checkpoint_panels:
            if any(
                panel[key] is None
                for key in (
                    "capture_attempt_id",
                    "shard_plan_sha256",
                    "capture_root_identity_sha256",
                )
            ):
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "resume checkpoint lost its capture identity",
                )
            self._validate_resume_checkpoint(
                self._connection,
                partition_role=panel["partition_role"],
                capture_attempt_id=panel["capture_attempt_id"],
                shard_plan_sha256=panel["shard_plan_sha256"],
                capture_root_identity_sha256=panel[
                    "capture_root_identity_sha256"
                ],
            )

    @staticmethod
    def _validate_campaign_branch_exclusivity(
        connection: sqlite3.Connection,
        inventory: Sequence[ArtifactInventoryEntry],
    ) -> None:
        names = frozenset(entry.name for entry in inventory)
        primary = names & {
            "invalid_campaign_receipt.json",
            "resource_limit_failure.json",
            "support_failure_receipt.json",
        }
        if len(primary) > 1:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "mutually exclusive primary terminal receipts coexist",
            )
        for role in _PARTITION_ROLES:
            panel = connection.execute(
                "SELECT state, row_index_failure_transaction_id "
                "FROM panels WHERE partition_role = ?",
                (role,),
            ).fetchone()
            if panel is None:
                continue
            authorization = {
                f"{role}_capture_authorization.json",
                f"{role}_capture_transition_authorized.json",
                f"{role}_authorization_transaction_commit.json",
            }
            row_success = {
                f"row_index_manifest_{role}.json",
                f"{role}_row_index_committed.json",
                f"{role}_row_index_transaction_commit.json",
            }
            row_failure = {
                f"row_index_failure_{role}.json",
                f"{role}_row_index_failed.json",
                f"{role}_row_index_failure_transaction_commit.json",
            }
            shard_plan = {
                f"capture_shard_plan_{role}.json",
                f"{role}_shard_plan_transaction_commit.json",
            }
            capture_start = {
                f"{role}_tensor_capture_started.json",
                f"{role}_capture_start_transaction_commit.json",
            }
            success = {
                f"capture_shard_manifest_{role}.json",
                f"{role}_capture_transition_consumed_success.json",
                f"{role}_capture_success_transaction_commit.json",
            }
            capture_failure = {
                f"capture_failure_{role}.json",
                f"{role}_capture_failure_transaction_commit.json",
            }
            consumed_failed = {f"{role}_capture_transition_consumed_failed.json"}
            row_failure_finalization = {
                f"{role}_row_index_failure_finalization_commit.json"
            }
            partial = {f"partial_shard_inventory_{role}.json"}
            stage_failure_commit = {f"{role}_stage_failure_transaction_commit.json"}
            capture_shard_rows = connection.execute(
                "SELECT name, source_kind FROM campaign_artifact_inventory "
                "WHERE name LIKE ? ORDER BY name",
                (f"capture_shards/{role}/shard-%.safetensors",),
            ).fetchall()
            capture_shard_names = {
                item["name"] for item in capture_shard_rows
            }
            if any(
                item["source_kind"] != "capture_shard_file"
                for item in capture_shard_rows
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"{role} shard inventory is not protocol file-backed",
                )
            planned_shard_names: set[str] = set()
            plan_row = connection.execute(
                "SELECT content, byte_sha256 FROM artifacts WHERE name = ?",
                (f"capture_shard_plan_{role}.json",),
            ).fetchone()
            if plan_row is not None:
                plan_content = bytes(plan_row["content"])
                if artifact_sha256(plan_content) != plan_row["byte_sha256"]:
                    raise _failure_exception(
                        "artifact_hash_mismatch", "stored shard plan changed"
                    )
                try:
                    plan_payload = json.loads(plan_content.decode("utf-8"))
                    entries = plan_payload["shard_plan"]["entries"]
                    if not isinstance(entries, list):
                        raise TypeError
                    planned_shard_names = {
                        str(entry["path"])
                        for entry in entries
                        if isinstance(entry, dict)
                    }
                    if len(planned_shard_names) != len(entries):
                        raise TypeError
                except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        "stored shard plan cannot define branch closure",
                    ) from error
            if names & success and names & (
                capture_failure | row_failure | partial
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    f"{role} success and failure branch artifacts coexist",
                )
            state = CaptureState(panel["state"])
            if state in {
                CaptureState.AUTHORIZED,
                CaptureState.ROW_INDEX_COMMITTED,
                CaptureState.ROW_INDEX_FAILED,
                CaptureState.CAPTURE_STARTED,
            }:
                raise ProtocolStateError(
                    f"campaign terminal cannot strand live {role} state {state.value}"
                )
            if state is CaptureState.CONSUMED_SUCCESS:
                required = authorization | row_success | shard_plan | capture_start | success
                forbidden = (
                    row_failure
                    | capture_failure
                    | consumed_failed
                    | row_failure_finalization
                    | partial
                )
                if not required <= names or names & forbidden:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"{role} consumed-success closure is inconsistent",
                    )
                if capture_shard_names != planned_shard_names:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"{role} success lacks its exact finalized shard inventory",
                    )
                if names & stage_failure_commit:
                    stage_row = connection.execute(
                        "SELECT transaction_kind FROM ledger_transactions "
                        "WHERE transaction_id = (SELECT transaction_id FROM artifacts "
                        "WHERE name = ?)",
                        (f"{role}_stage_failure_transaction_commit.json",),
                    ).fetchone()
                    if (
                        stage_row is None
                        or stage_row["transaction_kind"]
                        != "post_capture_stage_failure"
                    ):
                        raise _failure_exception(
                            "artifact_hash_mismatch",
                            f"{role} stage-failure receipt is not authentic",
                        )
            elif state is CaptureState.CONSUMED_FAILED:
                if names & success:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"{role} consumed-failed closure contains success artifacts",
                    )
                if panel["row_index_failure_transaction_id"] is not None:
                    required = (
                        authorization
                        | row_failure
                        | consumed_failed
                        | row_failure_finalization
                    )
                    forbidden = (
                        row_success
                        | shard_plan
                        | capture_start
                        | capture_failure
                        | partial
                        | stage_failure_commit
                    )
                    if capture_shard_names:
                        raise _failure_exception(
                            "artifact_hash_mismatch",
                            f"{role} row-index failure contains capture shards",
                        )
                else:
                    required = (
                        authorization
                        | row_success
                        | shard_plan
                        | capture_start
                        | capture_failure
                        | consumed_failed
                    )
                    forbidden = (
                        row_failure
                        | row_failure_finalization
                        | stage_failure_commit
                    )
                    failure_row = connection.execute(
                        "SELECT content FROM artifacts WHERE name = ?",
                        (f"capture_failure_{role}.json",),
                    ).fetchone()
                    if failure_row is None:
                        raise _failure_exception(
                            "artifact_hash_mismatch",
                            f"{role} capture failure payload is missing",
                        )
                    try:
                        failure_payload = json.loads(
                            bytes(failure_row["content"]).decode("utf-8")
                        )
                        invalid_rows = failure_payload.get(
                            "unverified_finalized_shards", []
                        )
                        if not isinstance(invalid_rows, list):
                            raise TypeError
                        invalid_paths = {
                            str(item["path"])
                            for item in invalid_rows
                            if isinstance(item, dict)
                        }
                        file_backed_invalid_paths = {
                            str(item["path"])
                            for item in invalid_rows
                            if isinstance(item, dict)
                            and isinstance(item.get("artifact_byte_sha256"), str)
                        }
                        if len(invalid_paths) != len(invalid_rows):
                            raise TypeError
                        partial_row = connection.execute(
                            "SELECT content FROM artifacts WHERE name = ?",
                            (f"partial_shard_inventory_{role}.json",),
                        ).fetchone()
                        verified_paths: set[str] = set()
                        if partial_row is not None:
                            partial_payload = json.loads(
                                bytes(partial_row["content"]).decode("utf-8")
                            )
                            partial_shards = partial_payload["shards"]
                            if not isinstance(partial_shards, list):
                                raise TypeError
                            verified_paths = {
                                str(item["path"])
                                for item in partial_shards
                                if isinstance(item, dict)
                            }
                            if len(verified_paths) != len(partial_shards):
                                raise TypeError
                        if invalid_paths & verified_paths:
                            raise TypeError
                    except (
                        KeyError,
                        TypeError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as error:
                        raise _failure_exception(
                            "artifact_hash_mismatch",
                            f"{role} failure shard inventory is malformed",
                        ) from error
                    if (
                        capture_shard_names
                        != file_backed_invalid_paths | verified_paths
                        or not (capture_shard_names | invalid_paths)
                        <= planned_shard_names
                    ):
                        raise _failure_exception(
                            "artifact_hash_mismatch",
                            f"{role} failure shard evidence is not exact",
                        )
                if not required <= names or names & forbidden:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        f"{role} consumed-failed closure is inconsistent",
                    )
                if role == "fit" and "capture_invariants.json" in names:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        "fit capture invariants cannot coexist with fit failure",
                    )
                if role == "tune" and names & {
                    "o2q_results.json",
                    "o2v_results.json",
                }:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        "O2 results cannot coexist with tune capture failure",
                    )
            else:
                raise ProtocolStateError(
                    f"campaign terminal cannot use unopened {role} state"
                )

    @staticmethod
    def _terminal_report_bytes(
        *,
        campaign_id: str,
        terminal_state: TerminalState,
        last_completed_prefix: CampaignPrefix | None,
        active_prefix: CampaignPrefix,
        reason_code: str,
    ) -> bytes:
        completed = (
            "none"
            if last_completed_prefix is None
            else last_completed_prefix.name
        )
        if terminal_state is TerminalState.DEVELOPMENT_COMPLETE_NO_ELIGIBLE_GUARD:
            qualification = (
                "The ledger records a development-confirm success disposition "
                "only within the frozen metric. Its scientific evidentiary "
                "meaning remains contingent on separately bound campaign "
                "provenance. No compatible evaluator-controlled private guard "
                "exists, so this is a development candidate and not a guard claim."
            )
        elif terminal_state is TerminalState.DEVELOPMENT_COMPLETE_REFIT_INSTABILITY:
            qualification = (
                "The non-selecting tune replay exceeded the frozen two-percent "
                "stability limit. Development-confirm was not authorized and no "
                "development candidate or guard claim exists."
            )
        elif terminal_state is TerminalState.DEVELOPMENT_COMPLETE_NEGATIVE_J4:
            qualification = (
                "The frozen development comparison conjunction did not establish "
                "J4. No development candidate or guard claim exists."
            )
        elif (
            terminal_state
            is TerminalState.DEVELOPMENT_COMPLETE_ATTENTION_OUTPUT_NOT_READY
        ):
            qualification = (
                "The frozen attention-output fidelity conjunction did not pass. "
                "No development candidate or guard claim exists."
            )
        else:
            qualification = (
                "This is a protocol disposition. It is not a scientific veto "
                "and does not establish evidence for or against the compact-message "
                "completion hypothesis."
            )
        report = (
            "# Fisher compact-message completion campaign terminal\n\n"
            f"Campaign: `{campaign_id}`\n\n"
            f"Terminal state: `{terminal_state.value}`\n\n"
            f"Last completed prefix: `{completed}`\n\n"
            f"Active prefix: `{active_prefix.name}`\n\n"
            f"Reason: `{reason_code}`\n\n"
            f"{qualification} No raw prompt, response, token, or "
            "protected panel content is included. This terminal record does "
            "not itself establish a scientific result or guard claim; that "
            "status is contingent on separately bound campaign provenance.\n"
        )
        return report.encode("utf-8")

    def _commit_campaign_terminal_tail(
        self,
        connection: sqlite3.Connection,
        *,
        campaign_id: str,
        partition_role: str,
        terminal_state: TerminalState,
        reason_code: str,
        last_completed_prefix: CampaignPrefix | None,
        active_prefix: CampaignPrefix,
        omitted_active_artifacts: Mapping[str, str],
        transaction_kind: str,
        transaction_identity_sha256: str,
        support_spec: SupportFailureSpec | None,
        require_mechanical_primary: bool = True,
    ) -> tuple[str, PrefixClosure]:
        """Commit the public terminal tail, with the final manifest last."""

        campaign_row = connection.execute(
            "SELECT value FROM protocol_metadata WHERE key = 'campaign_id'"
        ).fetchone()
        if campaign_row is None or campaign_row["value"] != campaign_id:
            raise _failure_exception(
                "authorization_mismatch",
                "terminal campaign differs from the ledger campaign",
            )
        panel = self._required_panel(connection, partition_role)
        if panel["campaign_id"] != campaign_id:
            raise _failure_exception(
                "authorization_mismatch",
                "terminal partition belongs to a different campaign",
            )
        existing_inventory = self._campaign_inventory_entries(connection)
        self._validate_all_ledger_transactions()
        self._validate_campaign_branch_exclusivity(
            connection, existing_inventory
        )
        existing_names = frozenset(entry.name for entry in existing_inventory)
        if existing_names & _TERMINAL_TAIL_ARTIFACTS:
            raise ProtocolStateError("campaign terminal tail already exists")
        if "support_failure_receipt.json" in existing_names:
            raise ProtocolStateError("support receipt already exists")
        closure = validate_reached_prefix_closure(
            existing_inventory,
            last_completed_prefix=last_completed_prefix,
            active_prefix=active_prefix,
            omitted_active_artifacts=omitted_active_artifacts,
        )
        if not closure.valid:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "campaign prefix closure is incomplete or contains future artifacts",
            )
        if support_spec is not None:
            forbidden_primary = {
                "invalid_campaign_receipt.json",
                "resource_limit_failure.json",
            }
            if existing_names & forbidden_primary:
                raise ProtocolStateError(
                    "support terminal cannot coexist with a mechanical primary receipt"
                )
        elif require_mechanical_primary:
            primary_names = existing_names & {
                "invalid_campaign_receipt.json",
                "resource_limit_failure.json",
            }
            if len(primary_names) != 1:
                raise ProtocolStateError(
                    "mechanical terminal requires exactly one primary receipt"
                )
        elif existing_names & {
            "invalid_campaign_receipt.json",
            "resource_limit_failure.json",
            "support_failure_receipt.json",
        }:
            raise ProtocolStateError(
                "successful terminal cannot coexist with a failure receipt"
            )

        transaction_id = self._new_transaction_id()
        next_row = connection.execute(
            "SELECT value FROM protocol_metadata "
            "WHERE key = 'next_materialization_ordinal'"
        ).fetchone()
        if next_row is None:
            raise ProtocolStateError("materialization counter is missing")
        next_ordinal = int(next_row["value"])
        tail: list[StoredArtifact] = []
        preview_entries: list[ArtifactInventoryEntry] = []

        if support_spec is not None:
            support_payload: dict[str, JsonValue] = {
                **support_spec.as_dict(),
                "durably_reached_artifacts": [
                    entry.as_dict() for entry in existing_inventory
                ],
                "format_version": _FORMAT_VERSION,
                "omitted_active_artifacts": _omission_records(
                    closure.omitted_active_artifacts
                ),
                "omitted_future_prefixes": list(
                    closure.omitted_future_prefixes
                ),
                "schema": SUPPORT_FAILURE_RECEIPT_SCHEMA,
                "terminal_state": terminal_state.value,
                "terminal_transaction_id": transaction_id,
            }
            support_artifact = self._artifact(
                name="support_failure_receipt.json",
                kind="support_failure_receipt",
                transaction_id=transaction_id,
                payload=support_payload,
            )
            tail.append(support_artifact)
            preview_entries.append(
                self._preview_inventory_entry(
                    support_artifact,
                    prefix=active_prefix,
                    materialization_ordinal=next_ordinal,
                )
            )
            next_ordinal += 1

        reached_before_terminal = existing_inventory + tuple(preview_entries)
        terminal_artifact = self._artifact(
            name="terminal_state.json",
            kind="campaign_terminal_state",
            transaction_id=transaction_id,
            payload={
                "active_prefix": active_prefix.name,
                "campaign_id": campaign_id,
                "durably_reached_artifacts": [
                    entry.as_dict() for entry in reached_before_terminal
                ],
                "format_version": _FORMAT_VERSION,
                "last_completed_prefix": (
                    None
                    if last_completed_prefix is None
                    else last_completed_prefix.name
                ),
                "omitted_active_artifacts": _omission_records(
                    closure.omitted_active_artifacts
                ),
                "omitted_future_prefixes": list(
                    closure.omitted_future_prefixes
                ),
                "partition_role": partition_role,
                "reason_code": reason_code,
                "schema": TERMINAL_STATE_SCHEMA,
                "terminal_state": terminal_state.value,
                "terminal_transaction_id": transaction_id,
            },
        )
        tail.append(terminal_artifact)
        preview_entries.append(
            self._preview_inventory_entry(
                terminal_artifact,
                prefix=active_prefix,
                materialization_ordinal=next_ordinal,
            )
        )
        next_ordinal += 1

        report_content = self._terminal_report_bytes(
            campaign_id=campaign_id,
            terminal_state=terminal_state,
            last_completed_prefix=last_completed_prefix,
            active_prefix=active_prefix,
            reason_code=reason_code,
        )
        final_report = StoredArtifact(
            name="final_report.md",
            kind="campaign_final_report",
            content=report_content,
            byte_sha256=artifact_sha256(report_content),
            transaction_id=transaction_id,
        )
        tail.append(final_report)
        preview_entries.append(
            self._preview_inventory_entry(
                final_report,
                prefix=active_prefix,
                materialization_ordinal=next_ordinal,
            )
        )
        next_ordinal += 1

        manifest_inventory = existing_inventory + tuple(preview_entries)
        final_manifest = self._artifact(
            name="final_experiment_manifest.json",
            kind="final_experiment_manifest",
            transaction_id=transaction_id,
            payload={
                "active_prefix": active_prefix.name,
                "artifacts": [entry.as_dict() for entry in manifest_inventory],
                "campaign_finalized": True,
                "campaign_id": campaign_id,
                "excludes_only_self": True,
                "format_version": _FORMAT_VERSION,
                "last_completed_prefix": (
                    None
                    if last_completed_prefix is None
                    else last_completed_prefix.name
                ),
                "materialized_last": True,
                "schema": FINAL_EXPERIMENT_MANIFEST_SCHEMA,
                "terminal_state": terminal_state.value,
                "terminal_transaction_id": transaction_id,
            },
        )
        tail.append(final_manifest)

        receipt_payload: dict[str, JsonValue] = {
            "artifacts": [
                {"byte_sha256": artifact.byte_sha256, "name": artifact.name}
                for artifact in sorted(tail, key=lambda item: item.name)
            ],
            "format_version": _FORMAT_VERSION,
            "identity_sha256": transaction_identity_sha256,
            "partition_role": partition_role,
            "schema": LEDGER_SCHEMA,
            "state_after": panel["state"],
            "state_before": panel["state"],
            "transaction_committed": True,
            "transaction_id": transaction_id,
            "transaction_kind": transaction_kind,
        }
        receipt_content = canonical_json_bytes(receipt_payload)
        connection.execute(
            """
            INSERT INTO ledger_transactions(
                transaction_id, transaction_kind, partition_role,
                state_before, state_after, identity_sha256,
                commit_receipt, commit_receipt_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                transaction_kind,
                partition_role,
                panel["state"],
                panel["state"],
                transaction_identity_sha256,
                receipt_content,
                artifact_sha256(receipt_content),
            ),
        )
        expected_ordinals = {
            entry.name: entry.materialization_ordinal
            for entry in preview_entries
        }
        expected_ordinals[final_manifest.name] = next_ordinal
        for artifact in tail:
            connection.execute(
                """
                INSERT INTO artifacts(
                    name, kind, partition_role, content,
                    byte_sha256, transaction_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.name,
                    artifact.kind,
                    partition_role,
                    artifact.content,
                    artifact.byte_sha256,
                    transaction_id,
                ),
            )
            registered = self._register_inventory_record(
                connection,
                name=artifact.name,
                prefix=active_prefix,
                byte_sha256=artifact.byte_sha256,
                byte_count=len(artifact.content),
                source_kind="campaign_terminal_artifact",
                transaction_id=transaction_id,
            )
            if registered.materialization_ordinal != expected_ordinals[artifact.name]:
                raise ProtocolStateError(
                    "terminal artifact materialization order changed"
                )
        final_inventory = self._campaign_inventory_entries(connection)
        if not final_inventory or final_inventory[-1].name != final_manifest.name:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "final experiment manifest was not materialized last",
            )
        connection.execute(
            "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("global_terminal_phase", "finalized"),
        )
        connection.execute(
            "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("global_terminal_closure_transaction_id", transaction_id),
        )
        connection.execute(
            "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                "global_terminal_final_manifest_sha256",
                final_manifest.byte_sha256,
            ),
        )
        return transaction_id, closure

    def _load_campaign_terminal_result(
        self,
        transaction_id: str,
        *,
        campaign_id: str,
        terminal_state: TerminalState,
        last_completed_prefix: CampaignPrefix | None,
        active_prefix: CampaignPrefix,
        omitted_active_artifacts: Mapping[str, str],
        idempotent: bool,
    ) -> CampaignTerminalResult:
        transaction = self._load_transaction(
            transaction_id, idempotent=idempotent
        )
        manifest = transaction.artifact("final_experiment_manifest.json")
        terminal_artifact = transaction.artifact("terminal_state.json")
        final_report = transaction.artifact("final_report.md")
        with self._lock:
            inventory = self._campaign_inventory_entries(self._connection)
            self._validate_all_ledger_transactions()
            self._validate_campaign_branch_exclusivity(
                self._connection, inventory
            )
            phase = self._connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            recorded_hash = self._connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_final_manifest_sha256'"
            ).fetchone()
        if phase is None or phase["value"] != "finalized":
            raise ProtocolStateError("campaign terminal closure is not finalized")
        if (
            recorded_hash is None
            or recorded_hash["value"] != manifest.byte_sha256
        ):
            raise _failure_exception(
                "artifact_hash_mismatch",
                "final manifest metadata differs from the stored artifact",
            )
        if not inventory or inventory[-1].name != manifest.name:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "final experiment manifest is not materialized last",
            )
        try:
            payload = manifest.json_value()
            _require_exact_keys(
                payload,
                frozenset(
                    {
                        "active_prefix",
                        "artifacts",
                        "campaign_finalized",
                        "campaign_id",
                        "excludes_only_self",
                        "format_version",
                        "last_completed_prefix",
                        "materialized_last",
                        "schema",
                        "terminal_state",
                        "terminal_transaction_id",
                    }
                ),
                label="final experiment manifest",
            )
            expected_inventory: list[JsonValue] = [
                entry.as_dict() for entry in inventory[:-1]
            ]
            if payload["artifacts"] != expected_inventory:
                raise ProtocolSchemaError(
                    "final experiment manifest inventory is not exact"
                )
            expected_fields: dict[str, JsonValue] = {
                "active_prefix": active_prefix.name,
                "campaign_finalized": True,
                "campaign_id": campaign_id,
                "excludes_only_self": True,
                "format_version": _FORMAT_VERSION,
                "last_completed_prefix": (
                    None
                    if last_completed_prefix is None
                    else last_completed_prefix.name
                ),
                "materialized_last": True,
                "schema": FINAL_EXPERIMENT_MANIFEST_SCHEMA,
                "terminal_state": terminal_state.value,
                "terminal_transaction_id": transaction_id,
            }
            for key, expected in expected_fields.items():
                if payload[key] != expected:
                    raise ProtocolSchemaError(
                        f"final experiment manifest {key} differs"
                    )
        except (KeyError, TypeError, ValueError, ProtocolSchemaError) as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "final experiment manifest fails strict verification",
            ) from error
        terminal_index = next(
            (
                index
                for index, entry in enumerate(inventory)
                if entry.name == "terminal_state.json"
            ),
            -1,
        )
        if terminal_index < 0:
            raise _failure_exception(
                "artifact_hash_mismatch", "terminal-state inventory is missing"
            )
        if [entry.name for entry in inventory[-3:]] != [
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "terminal state, report, and manifest are not public-last in order",
            )
        try:
            terminal_payload = terminal_artifact.json_value()
            _require_exact_keys(
                terminal_payload,
                frozenset(
                    {
                        "active_prefix",
                        "campaign_id",
                        "durably_reached_artifacts",
                        "format_version",
                        "last_completed_prefix",
                        "omitted_active_artifacts",
                        "omitted_future_prefixes",
                        "partition_role",
                        "reason_code",
                        "schema",
                        "terminal_state",
                        "terminal_transaction_id",
                    }
                ),
                label="terminal state artifact",
            )
            terminal_expected: dict[str, JsonValue] = {
                "active_prefix": active_prefix.name,
                "campaign_id": campaign_id,
                "durably_reached_artifacts": [
                    entry.as_dict() for entry in inventory[:terminal_index]
                ],
                "format_version": _FORMAT_VERSION,
                "last_completed_prefix": (
                    None
                    if last_completed_prefix is None
                    else last_completed_prefix.name
                ),
                "omitted_active_artifacts": _omission_records(
                    omitted_active_artifacts
                ),
                "omitted_future_prefixes": list(
                    prefix.name
                    for prefix in CampaignPrefix
                    if prefix.value > active_prefix.value
                ),
                "partition_role": transaction.partition_role,
                "schema": TERMINAL_STATE_SCHEMA,
                "terminal_state": terminal_state.value,
                "terminal_transaction_id": transaction_id,
            }
            for key, expected in terminal_expected.items():
                if terminal_payload[key] != expected:
                    raise ProtocolSchemaError(
                        f"terminal state {key} differs"
                    )
            reason_code = terminal_payload["reason_code"]
            if not isinstance(reason_code, str):
                raise ProtocolSchemaError("terminal reason code is not a string")
            expected_report = self._terminal_report_bytes(
                campaign_id=campaign_id,
                terminal_state=terminal_state,
                last_completed_prefix=last_completed_prefix,
                active_prefix=active_prefix,
                reason_code=reason_code,
            )
            if final_report.content != expected_report:
                raise ProtocolSchemaError("final report bytes differ")
        except (KeyError, TypeError, ValueError, ProtocolSchemaError) as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "terminal state or final report fails strict verification",
            ) from error
        closure = validate_reached_prefix_closure(
            inventory,
            last_completed_prefix=last_completed_prefix,
            active_prefix=active_prefix,
            omitted_active_artifacts=omitted_active_artifacts,
        )
        if not closure.valid:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "finalized campaign no longer satisfies prefix closure",
            )
        return CampaignTerminalResult(
            transaction=transaction,
            terminal_state=terminal_state,
            prefix_closure=closure,
            final_experiment_manifest_sha256=manifest.byte_sha256,
        )

    def finalize_support_failure(
        self,
        specification: SupportFailureSpec,
        *,
        omitted_active_artifacts: Mapping[str, str],
    ) -> CampaignTerminalResult:
        """Atomically emit support receipt, terminal, report, and manifest."""

        if not isinstance(specification, SupportFailureSpec):
            raise ProtocolSchemaError(
                "specification must be SupportFailureSpec"
            )
        omitted = {
            _validate_relative_path(name, label="omitted artifact name"):
            _validate_identifier(reason, label="omission reason")
            for name, reason in omitted_active_artifacts.items()
        }
        identity_payload: dict[str, JsonValue] = {
            "omitted_active_artifacts": _omission_records(omitted),
            "specification": specification.as_dict(),
        }
        identity_hash = hashlib.sha256(
            _TERMINAL_FAILURE_DOMAIN + canonical_json_bytes(identity_payload)
        ).hexdigest()
        with self._immediate_transaction() as connection:
            phase = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            if phase is not None:
                if phase["value"] != "finalized":
                    raise ProtocolStateError(
                        "a different campaign terminal is pending"
                    )
                state_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_state'"
                ).fetchone()
                identity_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_identity_sha256'"
                ).fetchone()
                transaction_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_closure_transaction_id'"
                ).fetchone()
                if (
                    state_row is None
                    or state_row["value"]
                    != TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT.value
                    or identity_row is None
                    or identity_row["value"] != identity_hash
                    or transaction_row is None
                ):
                    raise ProtocolStateError(
                        "campaign already has a different finalized terminal"
                    )
                transaction_id = transaction_row["value"]
                existing = True
            else:
                self._ensure_no_global_terminal(connection)
                live_rows = connection.execute(
                    "SELECT partition_role, state FROM panels WHERE state IN "
                    "(?, ?, ?, ?)",
                    (
                        CaptureState.AUTHORIZED.value,
                        CaptureState.ROW_INDEX_COMMITTED.value,
                        CaptureState.ROW_INDEX_FAILED.value,
                        CaptureState.CAPTURE_STARTED.value,
                    ),
                ).fetchall()
                if live_rows:
                    raise ProtocolStateError(
                        "support terminal cannot strand a live capture state"
                    )
                anchor = self._required_panel(
                    connection, specification.partition_role
                )
                if CaptureState(anchor["state"]) is not CaptureState.CONSUMED_SUCCESS:
                    raise ProtocolStateError(
                        "support failure requires a successfully consumed capture"
                    )
                transaction_id, _ = self._commit_campaign_terminal_tail(
                    connection,
                    campaign_id=specification.campaign_id,
                    partition_role=specification.partition_role,
                    terminal_state=(
                        TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT
                    ),
                    reason_code=specification.reason_code.value,
                    last_completed_prefix=(
                        specification.last_completed_prefix
                    ),
                    active_prefix=specification.active_prefix,
                    omitted_active_artifacts=omitted,
                    transaction_kind="support_terminal_finalization",
                    transaction_identity_sha256=identity_hash,
                    support_spec=specification,
                )
                for key, value in (
                    (
                        "global_terminal_state",
                        TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT.value,
                    ),
                    ("global_terminal_identity_sha256", identity_hash),
                    (
                        "global_terminal_transaction_id",
                        transaction_id,
                    ),
                    (
                        "global_terminal_origin_transaction_id",
                        transaction_id,
                    ),
                    (
                        "global_terminal_partition_role",
                        specification.partition_role,
                    ),
                ):
                    connection.execute(
                        "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
                existing = False
        return self._load_campaign_terminal_result(
            transaction_id,
            campaign_id=specification.campaign_id,
            terminal_state=(
                TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT
            ),
            last_completed_prefix=specification.last_completed_prefix,
            active_prefix=specification.active_prefix,
            omitted_active_artifacts=omitted,
            idempotent=existing,
        )

    def record_stage_failure(
        self,
        partition_role: str,
        *,
        campaign_id: str,
        reason_code: str,
        last_completed_prefix: CampaignPrefix | str,
        active_prefix: CampaignPrefix | str,
        failure_stage: str,
        sanitized_failure_details: Mapping[str, object] | None = None,
    ) -> TransactionResult:
        """Commit one K5 primary after a successfully finalized capture.

        This is the post-capture counterpart to row-index and capture failure
        finalization.  It permits P4 and later measurement/integrity/resource
        terminals without pretending the already successful capture failed.
        """

        role = _validate_partition_role(partition_role)
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        completed, active = _normalize_prefix_pair(
            last_completed_prefix, active_prefix
        )
        if completed is None:
            raise ProtocolSchemaError(
                "post-capture failure requires a completed predecessor prefix"
            )
        if active.value < _CAPTURE_PREFIX_BY_PARTITION[role].value:
            raise ProtocolSchemaError(
                "post-capture failure cannot precede the partition capture prefix"
            )
        stage = _validate_identifier(failure_stage, label="failure_stage")
        disposition = classify_failure(reason_code)
        details = validate_sanitized_failure_details(sanitized_failure_details)
        _validate_failure_detail_context(
            details,
            partition_role=role,
            reason_code=reason_code,
            last_completed_prefix=completed.name,
            active_prefix=active.name,
            failure_stage=stage,
        )
        identity_payload: dict[str, JsonValue] = {
            "active_prefix": active.name,
            "campaign_id": campaign,
            "failure_class": disposition.failure_class.value,
            "failure_reason_code": reason_code,
            "failure_stage": stage,
            "last_completed_prefix": completed.name,
            "partition_role": role,
            "sanitized_failure_details": details,
            "terminal_state": disposition.terminal_state.value,
        }
        identity_hash = hashlib.sha256(
            _TERMINAL_FAILURE_DOMAIN + canonical_json_bytes(identity_payload)
        ).hexdigest()
        with self._immediate_transaction() as connection:
            phase = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            if phase is not None:
                identity_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_identity_sha256'"
                ).fetchone()
                transaction_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_transaction_id'"
                ).fetchone()
                if (
                    phase["value"] != "primary_committed"
                    or identity_row is None
                    or identity_row["value"] != identity_hash
                    or transaction_row is None
                ):
                    raise ProtocolStateError(
                        "campaign already has a different global terminal"
                    )
                transaction_id = transaction_row["value"]
                existing = True
            else:
                self._ensure_no_global_terminal(connection)
                panel = self._required_panel(connection, role)
                if panel["campaign_id"] != campaign:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "stage failure campaign differs from partition campaign",
                    )
                if CaptureState(panel["state"]) is not CaptureState.CONSUMED_SUCCESS:
                    raise ProtocolStateError(
                        "post-capture stage failure requires consumed_success"
                    )
                live = connection.execute(
                    "SELECT partition_role FROM panels WHERE state IN (?, ?, ?, ?)",
                    (
                        CaptureState.AUTHORIZED.value,
                        CaptureState.ROW_INDEX_COMMITTED.value,
                        CaptureState.ROW_INDEX_FAILED.value,
                        CaptureState.CAPTURE_STARTED.value,
                    ),
                ).fetchone()
                if live is not None:
                    raise ProtocolStateError(
                        "stage failure cannot strand another live capture"
                    )
                existing_inventory = self._campaign_inventory_entries(connection)
                validate = validate_reached_prefix_closure(
                    existing_inventory,
                    last_completed_prefix=completed,
                    active_prefix=active,
                    omitted_active_artifacts={
                        name: "not_reached_stage_failure"
                        for name in (
                            _PREFIX_MANDATORY_ARTIFACTS[active]
                            - {entry.name for entry in existing_inventory}
                        )
                    },
                )
                if not validate.valid:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        "stage failure lacks completed predecessor artifacts",
                    )
                transaction_id = self._new_transaction_id()
                if disposition.failure_class is FailureClass.RESOURCE_LIMIT:
                    primary_name = "resource_limit_failure.json"
                    primary_kind = "resource_limit_failure"
                else:
                    primary_name = "invalid_campaign_receipt.json"
                    primary_kind = "invalid_campaign_receipt"
                reached_inventory = self._campaign_inventory_entries(connection)
                primary = self._artifact(
                    name=primary_name,
                    kind=primary_kind,
                    transaction_id=transaction_id,
                    payload={
                        **identity_payload,
                        "authorization_hash": panel["authorization_hash"],
                        "authorization_transaction_id": panel[
                            "authorization_transaction_id"
                        ],
                        "capture_attempt_id": panel["capture_attempt_id"],
                        "capture_finalization_transaction_id": panel[
                            "terminal_transaction_id"
                        ],
                        "capture_implementation_hash": panel[
                            "capture_implementation_hash"
                        ],
                        "capture_root_identity_sha256": panel[
                            "capture_root_identity_sha256"
                        ],
                        "capture_schema_hash": panel["capture_schema_hash"],
                        "capture_start_transaction_id": panel[
                            "capture_start_transaction_id"
                        ],
                        "committed_ordered_row_sequence_sha256": panel[
                            "committed_ordered_row_sequence_sha256"
                        ],
                        "committed_row_count": panel["committed_row_count"],
                        "committed_row_set_sha256": panel[
                            "committed_row_set_sha256"
                        ],
                        "durably_reached_artifact_exclusions": [
                            "primary_receipt_self",
                            "transaction_commit_receipt",
                        ],
                        "durably_reached_artifacts": [
                            entry.as_dict() for entry in reached_inventory
                        ],
                        "format_version": _FORMAT_VERSION,
                        "primary_receipt_kind": primary_kind,
                        "row_index_payload_sha256": panel[
                            "row_index_payload_sha256"
                        ],
                        "row_index_transaction_id": panel[
                            "row_index_transaction_id"
                        ],
                        "schema": PRIMARY_FAILURE_RECEIPT_SCHEMA,
                        "shard_plan_sha256": panel["shard_plan_sha256"],
                        "shard_plan_transaction_id": panel[
                            "shard_plan_transaction_id"
                        ],
                        "terminal_transaction_id": transaction_id,
                    },
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="post_capture_stage_failure",
                    partition_role=role,
                    state_before=CaptureState.CONSUMED_SUCCESS,
                    state_after=CaptureState.CONSUMED_SUCCESS,
                    identity_sha256=identity_hash,
                    artifacts=(primary,),
                    receipt_name=f"{role}_stage_failure_transaction_commit.json",
                    campaign_prefix=active,
                )
                self._set_global_terminal(
                    connection,
                    terminal_state=disposition.terminal_state.value,
                    terminal_identity_sha256=identity_hash,
                    terminal_transaction_id=transaction_id,
                    partition_role=role,
                )
                existing = False
        return self._load_transaction(transaction_id, idempotent=existing)

    def finalize_development_outcome(
        self,
        *,
        campaign_id: str,
        terminal_state: TerminalState | str,
        reason_code: str,
    ) -> CampaignTerminalResult:
        """Finalize a frozen non-success development outcome at its exact prefix.

        Refit instability closes after the complete P8 prefix and before any P9
        development-confirm freeze.  Negative-J4 and attention-output outcomes
        close after the complete P11 prefix and omit only P12's successful
        candidate manifest.  These scientific-screen dispositions deliberately
        use neither a mechanical failure receipt nor a support-failure receipt;
        insufficient support must instead use :meth:`finalize_support_failure`.
        """

        campaign = _validate_identifier(campaign_id, label="campaign_id")
        reason = _validate_identifier(reason_code, label="reason_code")
        try:
            outcome = (
                terminal_state
                if isinstance(terminal_state, TerminalState)
                else TerminalState(terminal_state)
            )
        except ValueError as error:
            raise ProtocolSchemaError(
                "terminal_state is not a registered development outcome"
            ) from error
        contexts = {
            TerminalState.DEVELOPMENT_COMPLETE_REFIT_INSTABILITY: (
                "tune",
                CampaignPrefix.P8,
                CampaignPrefix.P9,
                "refit_instability",
            ),
            TerminalState.DEVELOPMENT_COMPLETE_NEGATIVE_J4: (
                "development_confirm",
                CampaignPrefix.P11,
                CampaignPrefix.P12,
                "negative_j4",
            ),
            TerminalState.DEVELOPMENT_COMPLETE_ATTENTION_OUTPUT_NOT_READY: (
                "development_confirm",
                CampaignPrefix.P11,
                CampaignPrefix.P12,
                "attention_output_not_ready",
            ),
        }
        if outcome not in contexts:
            raise ProtocolSchemaError(
                "terminal_state is not a non-success development outcome"
            )
        partition_role, completed, active, outcome_kind = contexts[outcome]
        expected_reason = _DEVELOPMENT_OUTCOME_REASON_CODES[outcome]
        if reason != expected_reason:
            raise ProtocolSchemaError(
                "reason_code does not match the development terminal state"
            )
        omitted = {
            name: outcome_kind
            for name in sorted(_PREFIX_MANDATORY_ARTIFACTS[active])
        }
        identity_hash = hashlib.sha256(
            _DEVELOPMENT_OUTCOME_DOMAIN
            + canonical_json_bytes(
                {
                    "active_prefix": active.name,
                    "campaign_id": campaign,
                    "last_completed_prefix": completed.name,
                    "omitted_active_artifacts": _omission_records(omitted),
                    "partition_role": partition_role,
                    "reason_code": reason,
                    "terminal_state": outcome.value,
                }
            )
        ).hexdigest()
        with self._immediate_transaction() as connection:
            phase = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            if phase is not None:
                state_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_state'"
                ).fetchone()
                identity_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_identity_sha256'"
                ).fetchone()
                transaction_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_closure_transaction_id'"
                ).fetchone()
                role_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_partition_role'"
                ).fetchone()
                if (
                    phase["value"] != "finalized"
                    or state_row is None
                    or state_row["value"] != outcome.value
                    or identity_row is None
                    or identity_row["value"] != identity_hash
                    or transaction_row is None
                    or role_row is None
                    or role_row["value"] != partition_role
                ):
                    raise ProtocolStateError(
                        "campaign already has a different global terminal"
                    )
                transaction_id = transaction_row["value"]
                existing = True
            else:
                self._ensure_no_global_terminal(connection)
                panel = self._required_panel(connection, partition_role)
                if (
                    panel["campaign_id"] != campaign
                    or CaptureState(panel["state"])
                    is not CaptureState.CONSUMED_SUCCESS
                ):
                    raise ProtocolStateError(
                        "development outcome requires its consumed source panel"
                    )
                live = connection.execute(
                    "SELECT partition_role FROM panels WHERE state IN (?, ?, ?, ?)",
                    (
                        CaptureState.AUTHORIZED.value,
                        CaptureState.ROW_INDEX_COMMITTED.value,
                        CaptureState.ROW_INDEX_FAILED.value,
                        CaptureState.CAPTURE_STARTED.value,
                    ),
                ).fetchone()
                if live is not None:
                    raise ProtocolStateError(
                        "development outcome cannot strand a live capture"
                    )
                transaction_id, _ = self._commit_campaign_terminal_tail(
                    connection,
                    campaign_id=campaign,
                    partition_role=partition_role,
                    terminal_state=outcome,
                    reason_code=reason,
                    last_completed_prefix=completed,
                    active_prefix=active,
                    omitted_active_artifacts=omitted,
                    transaction_kind="development_outcome_terminal",
                    transaction_identity_sha256=identity_hash,
                    support_spec=None,
                    require_mechanical_primary=False,
                )
                for key, value in (
                    ("global_terminal_state", outcome.value),
                    ("global_terminal_identity_sha256", identity_hash),
                    ("global_terminal_transaction_id", transaction_id),
                    ("global_terminal_origin_transaction_id", transaction_id),
                    ("global_terminal_partition_role", partition_role),
                ):
                    connection.execute(
                        "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
                existing = False
        return self._load_campaign_terminal_result(
            transaction_id,
            campaign_id=campaign,
            terminal_state=outcome,
            last_completed_prefix=completed,
            active_prefix=active,
            omitted_active_artifacts=omitted,
            idempotent=existing,
        )

    def finalize_successful_campaign(
        self,
        *,
        campaign_id: str,
    ) -> CampaignTerminalResult:
        """Finalize a fully reached P12 campaign without making a guard claim."""

        campaign = _validate_identifier(campaign_id, label="campaign_id")
        terminal_state = TerminalState.DEVELOPMENT_COMPLETE_NO_ELIGIBLE_GUARD
        identity_hash = hashlib.sha256(
            _TERMINAL_SUCCESS_DOMAIN
            + canonical_json_bytes(
                {
                    "active_prefix": CampaignPrefix.P12.name,
                    "campaign_id": campaign,
                    "last_completed_prefix": CampaignPrefix.P12.name,
                    "terminal_state": terminal_state.value,
                }
            )
        ).hexdigest()
        with self._immediate_transaction() as connection:
            phase = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            if phase is not None:
                state_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_state'"
                ).fetchone()
                identity_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_identity_sha256'"
                ).fetchone()
                transaction_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_closure_transaction_id'"
                ).fetchone()
                if (
                    phase["value"] != "finalized"
                    or state_row is None
                    or state_row["value"] != terminal_state.value
                    or identity_row is None
                    or identity_row["value"] != identity_hash
                    or transaction_row is None
                ):
                    raise ProtocolStateError(
                        "campaign already has a different global terminal"
                    )
                transaction_id = transaction_row["value"]
                existing = True
            else:
                self._ensure_no_global_terminal(connection)
                panel = self._required_panel(
                    connection, "development_confirm"
                )
                if (
                    panel["campaign_id"] != campaign
                    or CaptureState(panel["state"])
                    is not CaptureState.CONSUMED_SUCCESS
                ):
                    raise ProtocolStateError(
                        "successful P12 closure requires consumed development-confirm"
                    )
                live = connection.execute(
                    "SELECT partition_role FROM panels WHERE state IN (?, ?, ?, ?)",
                    (
                        CaptureState.AUTHORIZED.value,
                        CaptureState.ROW_INDEX_COMMITTED.value,
                        CaptureState.ROW_INDEX_FAILED.value,
                        CaptureState.CAPTURE_STARTED.value,
                    ),
                ).fetchone()
                if live is not None:
                    raise ProtocolStateError(
                        "successful closure cannot strand a live capture"
                    )
                inventory = self._campaign_inventory_entries(connection)
                closure = validate_reached_prefix_closure(
                    inventory,
                    last_completed_prefix=CampaignPrefix.P12,
                    active_prefix=CampaignPrefix.P12,
                    omitted_active_artifacts={},
                )
                if not closure.valid:
                    raise _failure_exception(
                        "artifact_hash_mismatch",
                        "successful P12 closure is incomplete",
                    )
                transaction_id, _ = self._commit_campaign_terminal_tail(
                    connection,
                    campaign_id=campaign,
                    partition_role="development_confirm",
                    terminal_state=terminal_state,
                    reason_code="no_eligible_private_guard",
                    last_completed_prefix=CampaignPrefix.P12,
                    active_prefix=CampaignPrefix.P12,
                    omitted_active_artifacts={},
                    transaction_kind="successful_campaign_terminal",
                    transaction_identity_sha256=identity_hash,
                    support_spec=None,
                    require_mechanical_primary=False,
                )
                for key, value in (
                    ("global_terminal_state", terminal_state.value),
                    ("global_terminal_identity_sha256", identity_hash),
                    ("global_terminal_transaction_id", transaction_id),
                    ("global_terminal_origin_transaction_id", transaction_id),
                    (
                        "global_terminal_partition_role",
                        "development_confirm",
                    ),
                ):
                    connection.execute(
                        "INSERT INTO protocol_metadata(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
                existing = False
        return self._load_campaign_terminal_result(
            transaction_id,
            campaign_id=campaign,
            terminal_state=terminal_state,
            last_completed_prefix=CampaignPrefix.P12,
            active_prefix=CampaignPrefix.P12,
            omitted_active_artifacts={},
            idempotent=existing,
        )

    def finalize_campaign_terminal(
        self,
        *,
        campaign_id: str,
        partition_role: str,
        last_completed_prefix: CampaignPrefix | str,
        active_prefix: CampaignPrefix | str,
        omitted_active_artifacts: Mapping[str, str],
    ) -> CampaignTerminalResult:
        """Close a committed K5 mechanical primary with manifest-last tail."""

        campaign = _validate_identifier(campaign_id, label="campaign_id")
        role = _validate_partition_role(partition_role)
        completed, active = _normalize_prefix_pair(
            last_completed_prefix, active_prefix
        )
        if completed is None:
            raise ProtocolSchemaError(
                "mechanical failure requires a completed predecessor prefix"
            )
        omitted = {
            _validate_relative_path(name, label="omitted artifact name"):
            _validate_identifier(reason, label="omission reason")
            for name, reason in omitted_active_artifacts.items()
        }
        with self._immediate_transaction() as connection:
            phase = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_phase'"
            ).fetchone()
            state_row = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_state'"
            ).fetchone()
            identity_row = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_identity_sha256'"
            ).fetchone()
            primary_transaction_row = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_transaction_id'"
            ).fetchone()
            role_row = connection.execute(
                "SELECT value FROM protocol_metadata "
                "WHERE key = 'global_terminal_partition_role'"
            ).fetchone()
            if (
                phase is None
                or state_row is None
                or identity_row is None
                or primary_transaction_row is None
                or role_row is None
            ):
                raise ProtocolStateError(
                    "no committed mechanical primary is available for closure"
                )
            try:
                terminal_state = TerminalState(state_row["value"])
            except ValueError as error:
                raise ProtocolStateError(
                    "stored global terminal state is invalid"
                ) from error
            if terminal_state not in {
                TerminalState.INVALID_CAMPAIGN_INTEGRITY,
                TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT,
                TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED,
            }:
                raise ProtocolStateError(
                    "global terminal is not a K5 mechanical failure"
                )
            if role_row["value"] != role:
                raise _failure_exception(
                    "authorization_mismatch",
                    "terminal closure uses the wrong partition",
                )
            closure_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {
                        "active_prefix": active.name,
                        "campaign_id": campaign,
                        "last_completed_prefix": completed.name,
                        "omitted_active_artifacts": _omission_records(omitted),
                        "primary_terminal_identity_sha256": identity_row[
                            "value"
                        ],
                        "primary_transaction_id": primary_transaction_row[
                            "value"
                        ],
                    }
                )
            ).hexdigest()
            if phase["value"] == "finalized":
                closure_transaction_row = connection.execute(
                    "SELECT value FROM protocol_metadata "
                    "WHERE key = 'global_terminal_closure_transaction_id'"
                ).fetchone()
                if closure_transaction_row is None:
                    raise ProtocolStateError(
                        "finalized terminal lacks its closure transaction"
                    )
                transaction_id = closure_transaction_row["value"]
                committed_closure = connection.execute(
                    "SELECT identity_sha256 FROM ledger_transactions "
                    "WHERE transaction_id = ?",
                    (transaction_id,),
                ).fetchone()
                if (
                    committed_closure is None
                    or committed_closure["identity_sha256"]
                    != closure_identity
                ):
                    raise ProtocolStateError(
                        "campaign terminal closure request differs from the sticky closure"
                    )
                existing = True
            elif phase["value"] == "primary_committed":
                primary_name = (
                    "resource_limit_failure.json"
                    if terminal_state
                    is TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED
                    else "invalid_campaign_receipt.json"
                )
                primary_row = connection.execute(
                    "SELECT * FROM artifacts WHERE name = ?",
                    (primary_name,),
                ).fetchone()
                if primary_row is None:
                    raise ProtocolStateError(
                        "global mechanical primary receipt is missing"
                    )
                primary_payload = self._stored_artifact_from_row(
                    primary_row
                ).json_value()
                expected_context = (
                    campaign,
                    completed.name,
                    active.name,
                    role,
                    terminal_state.value,
                )
                observed_context = (
                    primary_payload.get("campaign_id"),
                    primary_payload.get("last_completed_prefix"),
                    primary_payload.get("active_prefix"),
                    primary_payload.get("partition_role"),
                    primary_payload.get("terminal_state"),
                )
                if expected_context != observed_context:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "terminal closure context differs from primary receipt",
                    )
                reason_code = str(primary_payload["failure_reason_code"])
                transaction_id, _ = self._commit_campaign_terminal_tail(
                    connection,
                    campaign_id=campaign,
                    partition_role=role,
                    terminal_state=terminal_state,
                    reason_code=reason_code,
                    last_completed_prefix=completed,
                    active_prefix=active,
                    omitted_active_artifacts=omitted,
                    transaction_kind="campaign_terminal_closure",
                    transaction_identity_sha256=closure_identity,
                    support_spec=None,
                )
                existing = False
            else:
                raise ProtocolStateError(
                    f"mechanical terminal cannot close from phase {phase['value']!r}"
                )
        return self._load_campaign_terminal_result(
            transaction_id,
            campaign_id=campaign,
            terminal_state=terminal_state,
            last_completed_prefix=completed,
            active_prefix=active,
            omitted_active_artifacts=omitted,
            idempotent=existing,
        )

    def _strict_validate_transaction_contract(
        self,
        row: sqlite3.Row,
        artifacts: Sequence[StoredArtifact],
    ) -> None:
        """Reconstruct the semantic contract for one durable transaction.

        The commit receipt is an integrity envelope, not independent provenance:
        a coherent rewrite of an artifact and that envelope must still fail here
        when it disagrees with the frozen panel or the transaction identity.
        """

        transaction_id = str(row["transaction_id"])
        transaction_kind = str(row["transaction_kind"])
        role = str(row["partition_role"])
        state_before = str(row["state_before"])
        state_after = str(row["state_after"])
        identity_sha256 = str(row["identity_sha256"])
        expected_states: dict[str, tuple[str, str] | None] = {
            "authorization": ("unopened", "authorized"),
            "row_index_commit": ("authorized", "row_index_committed"),
            "row_index_failure": ("authorized", "row_index_failed"),
            "row_index_failure_finalization": (
                "row_index_failed",
                "consumed_failed",
            ),
            "shard_plan_commit": (
                "row_index_committed",
                "row_index_committed",
            ),
            "capture_start": ("row_index_committed", "capture_started"),
            "recoverable_interruption": (
                "capture_started",
                "capture_started",
            ),
            "capture_success_finalization": (
                "capture_started",
                "consumed_success",
            ),
            "capture_failure_finalization": (
                "capture_started",
                "consumed_failed",
            ),
            "post_capture_stage_failure": (
                "consumed_success",
                "consumed_success",
            ),
            "support_terminal_finalization": None,
            "campaign_terminal_closure": None,
            "successful_campaign_terminal": None,
            "development_outcome_terminal": None,
        }
        if transaction_kind not in expected_states:
            raise ProtocolSchemaError("unknown durable transaction kind")
        fixed_states = expected_states[transaction_kind]
        if fixed_states is not None and (state_before, state_after) != fixed_states:
            raise ProtocolSchemaError(
                "transaction state pair differs from its kind contract"
            )
        if transaction_kind in {
            "support_terminal_finalization",
            "campaign_terminal_closure",
            "successful_campaign_terminal",
            "development_outcome_terminal",
        } and state_before != state_after:
            raise ProtocolSchemaError(
                "campaign-terminal transaction must preserve panel state"
            )

        primary_names = {
            "invalid_campaign_receipt.json": "invalid_campaign_receipt",
            "resource_limit_failure.json": "resource_limit_failure",
        }
        receipt_names: dict[str, str] = {
            "authorization": f"{role}_authorization_transaction_commit.json",
            "row_index_commit": f"{role}_row_index_transaction_commit.json",
            "row_index_failure": (
                f"{role}_row_index_failure_transaction_commit.json"
            ),
            "row_index_failure_finalization": (
                f"{role}_row_index_failure_finalization_commit.json"
            ),
            "shard_plan_commit": f"{role}_shard_plan_transaction_commit.json",
            "capture_start": f"{role}_capture_start_transaction_commit.json",
            "capture_success_finalization": (
                f"{role}_capture_success_transaction_commit.json"
            ),
            "capture_failure_finalization": (
                f"{role}_capture_failure_transaction_commit.json"
            ),
            "post_capture_stage_failure": (
                f"{role}_stage_failure_transaction_commit.json"
            ),
        }
        expected: dict[str, str]
        if transaction_kind == "authorization":
            expected = {
                f"{role}_capture_authorization.json": "capture_authorization",
                f"{role}_capture_transition_authorized.json": (
                    "capture_transition_authorized"
                ),
            }
        elif transaction_kind == "row_index_commit":
            expected = {
                f"row_index_manifest_{role}.json": "row_index_manifest",
                f"{role}_row_index_committed.json": "row_index_committed",
            }
        elif transaction_kind == "row_index_failure":
            expected = {
                f"row_index_failure_{role}.json": "row_index_failure",
                f"{role}_row_index_failed.json": "row_index_failed",
            }
        elif transaction_kind == "row_index_failure_finalization":
            expected = {
                f"{role}_capture_transition_consumed_failed.json": (
                    "capture_transition_consumed_failed"
                )
            }
            present_primary = set(primary_names) & {item.name for item in artifacts}
            if len(present_primary) != 1:
                raise ProtocolSchemaError(
                    "row-index finalization requires exactly one primary receipt"
                )
            name = next(iter(present_primary))
            expected[name] = primary_names[name]
        elif transaction_kind == "shard_plan_commit":
            expected = {
                f"capture_shard_plan_{role}.json": "capture_shard_plan"
            }
        elif transaction_kind == "capture_start":
            expected = {
                f"{role}_tensor_capture_started.json": "tensor_capture_started"
            }
        elif transaction_kind == "recoverable_interruption":
            interruption = [
                item
                for item in artifacts
                if item.kind == "recoverable_interruption"
            ]
            if len(interruption) != 1:
                raise ProtocolSchemaError(
                    "interruption transaction requires one interruption artifact"
                )
            match = re.fullmatch(
                rf"capture_interruption_{re.escape(role)}_(\d{{6}})\.json",
                interruption[0].name,
            )
            if match is None:
                raise ProtocolSchemaError("interruption artifact name is invalid")
            sequence = match.group(1)
            receipt_names[transaction_kind] = (
                f"{role}_interruption_{sequence}_transaction_commit.json"
            )
            expected = {interruption[0].name: "recoverable_interruption"}
        elif transaction_kind == "capture_success_finalization":
            expected = {
                f"capture_shard_manifest_{role}.json": "capture_shard_manifest",
                f"{role}_capture_transition_consumed_success.json": (
                    "capture_transition_consumed_success"
                ),
            }
        elif transaction_kind == "capture_failure_finalization":
            expected = {
                f"capture_failure_{role}.json": "capture_failure",
                f"{role}_capture_transition_consumed_failed.json": (
                    "capture_transition_consumed_failed"
                ),
            }
            present_primary = set(primary_names) & {item.name for item in artifacts}
            if len(present_primary) != 1:
                raise ProtocolSchemaError(
                    "capture failure requires exactly one primary receipt"
                )
            name = next(iter(present_primary))
            expected[name] = primary_names[name]
            partial_name = f"partial_shard_inventory_{role}.json"
            if any(item.name == partial_name for item in artifacts):
                expected[partial_name] = "partial_shard_inventory"
        elif transaction_kind == "post_capture_stage_failure":
            present_primary = set(primary_names) & {item.name for item in artifacts}
            if len(present_primary) != 1:
                raise ProtocolSchemaError(
                    "stage failure requires exactly one primary receipt"
                )
            name = next(iter(present_primary))
            expected = {name: primary_names[name]}
        elif transaction_kind == "support_terminal_finalization":
            expected = {
                "support_failure_receipt.json": "support_failure_receipt",
                "terminal_state.json": "campaign_terminal_state",
                "final_report.md": "campaign_final_report",
                "final_experiment_manifest.json": "final_experiment_manifest",
            }
        else:
            expected = {
                "terminal_state.json": "campaign_terminal_state",
                "final_report.md": "campaign_final_report",
                "final_experiment_manifest.json": "final_experiment_manifest",
            }

        receipt_name = receipt_names.get(transaction_kind)
        if receipt_name is not None:
            expected[receipt_name] = "transaction_commit_receipt"
        observed = {item.name: item.kind for item in artifacts}
        if observed != expected:
            raise ProtocolSchemaError(
                "transaction artifact names/kinds differ from its exact contract"
            )

        payloads: dict[str, dict[str, JsonValue]] = {}
        for artifact in artifacts:
            if artifact.kind == "transaction_commit_receipt" or not artifact.name.endswith(
                ".json"
            ):
                continue
            value = artifact.json_value()
            if artifact.content != canonical_json_bytes(value):
                raise ProtocolSchemaError(
                    f"artifact {artifact.name!r} is not canonical JSON"
                )
            payloads[artifact.name] = value
            if "format_version" in value and value["format_version"] != _FORMAT_VERSION:
                raise ProtocolSchemaError("artifact format version changed")
            if "partition_role" in value and value["partition_role"] != role:
                raise ProtocolSchemaError("artifact partition differs from transaction")
            if "state_before" in value and value["state_before"] != state_before:
                raise ProtocolSchemaError("artifact state_before differs from transaction")
            if "state_after" in value and value["state_after"] != state_after:
                raise ProtocolSchemaError("artifact state_after differs from transaction")

        transaction_id_field = {
            "authorization": "authorization_transaction_id",
            "row_index_commit": "row_index_transaction_id",
            "row_index_failure": "row_index_failure_transaction_id",
            "row_index_failure_finalization": "terminal_transaction_id",
            "shard_plan_commit": "shard_plan_transaction_id",
            "capture_start": "capture_start_transaction_id",
            "recoverable_interruption": "transaction_id",
            "capture_success_finalization": "terminal_transaction_id",
            "capture_failure_finalization": "terminal_transaction_id",
            "post_capture_stage_failure": "terminal_transaction_id",
            "support_terminal_finalization": "terminal_transaction_id",
            "campaign_terminal_closure": "terminal_transaction_id",
            "successful_campaign_terminal": "terminal_transaction_id",
            "development_outcome_terminal": "terminal_transaction_id",
        }[transaction_kind]
        for value in payloads.values():
            if transaction_id_field in value and value[transaction_id_field] != transaction_id:
                raise ProtocolSchemaError(
                    "artifact transaction identity differs from transaction row"
                )

        # Every repeated binding is byte-for-byte identical across siblings.
        repeated_binding_keys = {
            "active_prefix",
            "authorization_hash",
            "authorization_transaction_id",
            "campaign_id",
            "capture_attempt_id",
            "capture_implementation_hash",
            "capture_root_identity_sha256",
            "capture_schema_hash",
            "capture_start_transaction_id",
            "committed_ordered_row_sequence_sha256",
            "committed_row_count",
            "committed_row_set_sha256",
            "failure_class",
            "failure_reason_code",
            "failure_stage",
            "last_completed_prefix",
            "partition_role",
            "row_derivation_protocol_hash",
            "row_index_payload_sha256",
            "row_index_transaction_id",
            "shard_plan_sha256",
            "shard_plan_transaction_id",
            "terminal_state",
            "terminal_transaction_id",
        }
        for key in repeated_binding_keys:
            values = [value[key] for value in payloads.values() if key in value]
            if values and any(value != values[0] for value in values[1:]):
                raise ProtocolSchemaError(
                    f"transaction sibling artifacts disagree on {key}"
                )

        panel = self._connection.execute(
            "SELECT * FROM panels WHERE partition_role = ?", (role,)
        ).fetchone()
        if panel is None:
            raise ProtocolSchemaError("transaction partition panel is missing")
        panel_bindings = {
            "campaign_id": "campaign_id",
            "authorization_hash": "authorization_hash",
            "authorization_transaction_id": "authorization_transaction_id",
            "expected_row_set_mode": "expected_row_set_mode",
            "row_derivation_protocol_hash": "row_derivation_protocol_hash",
            "expected_row_set_sha256": "expected_row_set_sha256",
            "expected_ordered_row_sequence_sha256": (
                "expected_ordered_row_sequence_sha256"
            ),
            "expected_row_count": "expected_row_count",
            "expected_row_index_payload_sha256": (
                "expected_row_index_payload_sha256"
            ),
            "row_index_payload_sha256": "row_index_payload_sha256",
            "row_index_transaction_id": "row_index_transaction_id",
            "committed_row_set_sha256": "committed_row_set_sha256",
            "committed_ordered_row_sequence_sha256": (
                "committed_ordered_row_sequence_sha256"
            ),
            "committed_row_count": "committed_row_count",
            "derived_row_set_sha256": "committed_row_set_sha256",
            "derived_ordered_row_sequence_sha256": (
                "committed_ordered_row_sequence_sha256"
            ),
            "derived_row_count": "committed_row_count",
            "captured_row_set_sha256": "committed_row_set_sha256",
            "captured_ordered_row_sequence_sha256": (
                "committed_ordered_row_sequence_sha256"
            ),
            "row_count": "committed_row_count",
            "shard_plan_sha256": "shard_plan_sha256",
            "shard_plan_transaction_id": "shard_plan_transaction_id",
            "capture_start_transaction_id": "capture_start_transaction_id",
            "capture_attempt_id": "capture_attempt_id",
            "capture_root_identity_sha256": "capture_root_identity_sha256",
            "capture_schema_hash": "capture_schema_hash",
            "capture_implementation_hash": "capture_implementation_hash",
        }
        for value in payloads.values():
            for payload_key, panel_key in panel_bindings.items():
                if payload_key in value and value[payload_key] != panel[panel_key]:
                    raise ProtocolSchemaError(
                        f"artifact {payload_key} differs from frozen panel"
                    )

        panel_transaction_bindings = {
            "authorization": "authorization_transaction_id",
            "row_index_commit": "row_index_transaction_id",
            "row_index_failure": "row_index_failure_transaction_id",
            "row_index_failure_finalization": "terminal_transaction_id",
            "shard_plan_commit": "shard_plan_transaction_id",
            "capture_start": "capture_start_transaction_id",
            "capture_success_finalization": "terminal_transaction_id",
            "capture_failure_finalization": "terminal_transaction_id",
        }
        panel_transaction_column = panel_transaction_bindings.get(transaction_kind)
        if (
            panel_transaction_column is not None
            and panel[panel_transaction_column] != transaction_id
        ):
            raise ProtocolSchemaError(
                "transaction ID differs from its sticky panel binding"
            )
        if transaction_kind == "recoverable_interruption":
            interruption_value = next(iter(payloads.values()))
            if interruption_value.get("capture_attempt_id") != panel["capture_attempt_id"]:
                raise ProtocolSchemaError(
                    "interruption capture attempt differs from frozen panel"
                )

        recomputed_identity: str | None = None
        if transaction_kind == "authorization":
            value = payloads[f"{role}_capture_authorization.json"]
            semantic_value = dict(value)
            if semantic_value.pop("authorization_hash", None) != identity_sha256:
                raise ProtocolSchemaError("authorization hash field changed")
            if semantic_value.pop("authorization_transaction_id", None) != transaction_id:
                raise ProtocolSchemaError("authorization transaction field changed")
            semantic_bytes = canonical_json_bytes(semantic_value)
            if semantic_bytes != bytes(panel["authorization_semantic_payload"]):
                raise ProtocolSchemaError("authorization semantic payload changed")
            recomputed_identity = hashlib.sha256(
                _AUTHORIZATION_DOMAIN + semantic_bytes
            ).hexdigest()
        elif transaction_kind == "row_index_commit":
            value = payloads[f"row_index_manifest_{role}.json"]
            raw_index = value.get("row_index")
            if not isinstance(raw_index, Mapping):
                raise ProtocolSchemaError("row-index manifest lost row_index")
            parsed_index = RowIndexPayload.from_mapping(raw_index)
            recomputed_identity = parsed_index.payload_sha256
            if value.get("row_index_payload_sha256") != recomputed_identity:
                raise ProtocolSchemaError("row-index payload hash changed")
        elif transaction_kind == "row_index_failure":
            value = payloads[f"row_index_failure_{role}.json"]
            recomputed_identity = hashlib.sha256(
                _ROW_INDEX_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {
                        "details": value.get("sanitized_failure_details"),
                        "campaign_id": value.get("campaign_id"),
                        "active_prefix": value.get("active_prefix"),
                        "failure_stage": value.get("failure_stage"),
                        "last_completed_prefix": value.get(
                            "last_completed_prefix"
                        ),
                        "partition_role": role,
                        "reason_code": value.get("failure_reason_code"),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "row_index_failure_finalization":
            value = payloads[f"{role}_capture_transition_consumed_failed.json"]
            failure_transaction_id = value.get("row_index_failure_transaction_id")
            if not isinstance(failure_transaction_id, str):
                raise ProtocolSchemaError("row-index failure identity is missing")
            _validate_tx_id(
                failure_transaction_id,
                label="row-index failure transaction ID",
            )
            recomputed_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN + bytes.fromhex(failure_transaction_id)
            ).hexdigest()
        elif transaction_kind == "shard_plan_commit":
            value = payloads[f"capture_shard_plan_{role}.json"]
            shard_plan = value.get("shard_plan")
            if not isinstance(shard_plan, Mapping):
                raise ProtocolSchemaError("shard-plan payload is missing")
            recomputed_identity = hashlib.sha256(
                _SHARD_PLAN_DOMAIN + canonical_json_bytes(dict(shard_plan))
            ).hexdigest()
            if value.get("shard_plan_sha256") != recomputed_identity:
                raise ProtocolSchemaError("shard-plan hash changed")
        elif transaction_kind == "capture_start":
            value = payloads[f"{role}_tensor_capture_started.json"]
            recomputed_identity = hashlib.sha256(
                _CAPTURE_START_DOMAIN
                + canonical_json_bytes(
                    {
                        "capture_implementation_hash": value.get(
                            "capture_implementation_hash"
                        ),
                        "capture_schema_hash": value.get("capture_schema_hash"),
                        "capture_root_identity_sha256": value.get(
                            "capture_root_identity_sha256"
                        ),
                        "partition_role": role,
                        "row_index_payload_sha256": value.get(
                            "row_index_payload_sha256"
                        ),
                        "shard_plan_sha256": value.get("shard_plan_sha256"),
                        "shard_plan_transaction_id": value.get(
                            "shard_plan_transaction_id"
                        ),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "recoverable_interruption":
            value = next(iter(payloads.values()))
            recomputed_identity = hashlib.sha256(
                _CAPTURE_START_DOMAIN
                + canonical_json_bytes(
                    {
                        "capture_attempt_id": value.get("capture_attempt_id"),
                        "interruption_sequence": value.get(
                            "interruption_sequence"
                        ),
                        "partition_role": role,
                        "sanitized_failure_details": value.get(
                            "sanitized_failure_details"
                        ),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "capture_success_finalization":
            manifest = payloads[f"capture_shard_manifest_{role}.json"]
            verification_keys = {
                "captured_ordered_row_sequence_sha256",
                "captured_row_set_sha256",
                "committed_ordered_row_sequence_sha256",
                "committed_row_set_sha256",
                "format_version",
                "ordered_row_sequence_match",
                "partition_role",
                "row_count",
                "row_set_match",
                "schema",
                "shard_plan_sha256",
                "shards",
                "verification_source",
            }
            verification = {
                key: manifest[key] for key in verification_keys if key in manifest
            }
            if set(verification) != verification_keys:
                raise ProtocolSchemaError("capture verification fields are incomplete")
            if (
                verification["schema"] != CAPTURE_VERIFICATION_SCHEMA
                or verification["verification_source"]
                != "finalized_files_reopened"
                or verification["row_set_match"] is not True
                or verification["ordered_row_sequence_match"] is not True
            ):
                raise ProtocolSchemaError("capture verification contract changed")
            verification_sha256 = hashlib.sha256(
                _CAPTURE_VERIFICATION_DOMAIN + canonical_json_bytes(verification)
            ).hexdigest()
            transition = payloads[
                f"{role}_capture_transition_consumed_success.json"
            ]
            if transition.get("capture_verification_sha256") != verification_sha256:
                raise ProtocolSchemaError("capture verification digest changed")
            recomputed_identity = hashlib.sha256(
                _TERMINAL_SUCCESS_DOMAIN
                + canonical_json_bytes(
                    {
                        "capture_attempt_id": manifest.get("capture_attempt_id"),
                        "capture_root_identity_sha256": manifest.get(
                            "capture_root_identity_sha256"
                        ),
                        "verification": verification,
                    }
                )
            ).hexdigest()
        elif transaction_kind == "capture_failure_finalization":
            value = payloads[f"capture_failure_{role}.json"]
            partial_name = f"partial_shard_inventory_{role}.json"
            partial_value: JsonValue = None
            if partial_name in payloads:
                partial_payload = payloads[partial_name]
                partial_keys = {
                    "finalized_shard_count",
                    "format_version",
                    "partition_role",
                    "schema",
                    "shard_plan_sha256",
                    "shards",
                    "verification_source",
                }
                partial_value = {
                    key: partial_payload[key]
                    for key in partial_keys
                    if key in partial_payload
                }
                if set(partial_value) != partial_keys:
                    raise ProtocolSchemaError("partial inventory fields are incomplete")
            recomputed_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {
                        "capture_attempt_id": value.get("capture_attempt_id"),
                        "campaign_id": value.get("campaign_id"),
                        "active_prefix": value.get("active_prefix"),
                        "capture_root_identity_sha256": value.get(
                            "capture_root_identity_sha256"
                        ),
                        "failure_reason_code": value.get("failure_reason_code"),
                        "failure_stage": value.get("failure_stage"),
                        "last_completed_prefix": value.get(
                            "last_completed_prefix"
                        ),
                        "partial_shard_inventory": partial_value,
                        "partition_role": role,
                        "sanitized_failure_details": value.get(
                            "sanitized_failure_details"
                        ),
                        "unverified_finalized_shards": value.get(
                            "unverified_finalized_shards", []
                        ),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "post_capture_stage_failure":
            value = payloads[next(iter(set(primary_names) & set(payloads)))]
            stage_identity_keys = {
                "active_prefix",
                "campaign_id",
                "failure_class",
                "failure_reason_code",
                "failure_stage",
                "last_completed_prefix",
                "partition_role",
                "sanitized_failure_details",
                "terminal_state",
            }
            recomputed_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {key: value.get(key) for key in stage_identity_keys}
                )
            ).hexdigest()
        elif transaction_kind == "support_terminal_finalization":
            value = payloads["support_failure_receipt.json"]
            specification_keys = {
                "active_prefix",
                "active_chart_ids",
                "analysis_unit_counts",
                "attempted_locator_K_values",
                "attempted_representation_or_codec_ids",
                "campaign_id",
                "chart_counts",
                "chart_head_validation_counts",
                "fallback_attempts",
                "expected_head_ids",
                "evaluated_token_count",
                "last_completed_prefix",
                "out_of_support_token_count",
                "partition_role",
                "reason_code",
                "sequence_counts",
                "support_stage",
            }
            recomputed_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {
                        "omitted_active_artifacts": value.get(
                            "omitted_active_artifacts"
                        ),
                        "specification": {
                            key: value.get(key) for key in specification_keys
                        },
                    }
                )
            ).hexdigest()
        elif transaction_kind == "development_outcome_terminal":
            value = payloads["terminal_state.json"]
            try:
                durable_outcome = TerminalState(str(value.get("terminal_state")))
            except ValueError as error:
                raise ProtocolSchemaError(
                    "development outcome terminal state changed"
                ) from error
            expected_reason = _DEVELOPMENT_OUTCOME_REASON_CODES.get(
                durable_outcome
            )
            if (
                expected_reason is None
                or value.get("reason_code") != expected_reason
            ):
                raise ProtocolSchemaError(
                    "development outcome reason differs from terminal state"
                )
            recomputed_identity = hashlib.sha256(
                _DEVELOPMENT_OUTCOME_DOMAIN
                + canonical_json_bytes(
                    {
                        "active_prefix": value.get("active_prefix"),
                        "campaign_id": value.get("campaign_id"),
                        "last_completed_prefix": value.get(
                            "last_completed_prefix"
                        ),
                        "omitted_active_artifacts": value.get(
                            "omitted_active_artifacts"
                        ),
                        "partition_role": value.get("partition_role"),
                        "reason_code": value.get("reason_code"),
                        "terminal_state": value.get("terminal_state"),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "successful_campaign_terminal":
            value = payloads["terminal_state.json"]
            recomputed_identity = hashlib.sha256(
                _TERMINAL_SUCCESS_DOMAIN
                + canonical_json_bytes(
                    {
                        "active_prefix": value.get("active_prefix"),
                        "campaign_id": value.get("campaign_id"),
                        "last_completed_prefix": value.get(
                            "last_completed_prefix"
                        ),
                        "terminal_state": value.get("terminal_state"),
                    }
                )
            ).hexdigest()
        elif transaction_kind == "campaign_terminal_closure":
            value = payloads["terminal_state.json"]
            metadata = {
                item["key"]: item["value"]
                for item in self._connection.execute(
                    "SELECT key, value FROM protocol_metadata WHERE key IN "
                    "('global_terminal_identity_sha256', "
                    "'global_terminal_transaction_id')"
                ).fetchall()
            }
            recomputed_identity = hashlib.sha256(
                _TERMINAL_FAILURE_DOMAIN
                + canonical_json_bytes(
                    {
                        "active_prefix": value.get("active_prefix"),
                        "campaign_id": value.get("campaign_id"),
                        "last_completed_prefix": value.get(
                            "last_completed_prefix"
                        ),
                        "omitted_active_artifacts": value.get(
                            "omitted_active_artifacts"
                        ),
                        "primary_terminal_identity_sha256": metadata.get(
                            "global_terminal_identity_sha256"
                        ),
                        "primary_transaction_id": metadata.get(
                            "global_terminal_transaction_id"
                        ),
                    }
                )
            ).hexdigest()
        if recomputed_identity is not None and recomputed_identity != identity_sha256:
            raise ProtocolSchemaError(
                "transaction semantic identity does not match durable payloads"
            )
        if transaction_kind in {
            "row_index_failure_finalization",
            "capture_success_finalization",
            "capture_failure_finalization",
        } and panel["terminal_identity_sha256"] != identity_sha256:
            raise ProtocolSchemaError(
                "transaction semantic identity differs from sticky panel"
            )

    def _load_transaction(
        self, transaction_id: str, *, idempotent: bool
    ) -> TransactionResult:
        _validate_tx_id(transaction_id, label="transaction_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM ledger_transactions WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
            if row is None:
                raise ProtocolStateError("ledger transaction is missing")
            artifacts = tuple(
                self._stored_artifact_from_row(artifact_row)
                for artifact_row in self._connection.execute(
                    "SELECT * FROM artifacts WHERE transaction_id = ? ORDER BY name",
                    (transaction_id,),
                ).fetchall()
            )
        receipt_content = bytes(row["commit_receipt"])
        if artifact_sha256(receipt_content) != row["commit_receipt_sha256"]:
            raise _failure_exception(
                "artifact_hash_mismatch", "transaction receipt hash mismatch"
            )
        receipt_artifacts = tuple(
            artifact
            for artifact in artifacts
            if artifact.kind == "transaction_commit_receipt"
        )
        internal_receipt_only = row["transaction_kind"] in {
            "campaign_terminal_closure",
            "support_terminal_finalization",
            "successful_campaign_terminal",
            "development_outcome_terminal",
        }
        if internal_receipt_only:
            if receipt_artifacts:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "campaign terminal commit receipt must remain internal",
                )
            receipt_artifact: StoredArtifact | None = None
        else:
            if len(receipt_artifacts) != 1:
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "transaction must have exactly one commit-receipt artifact",
                )
            receipt_artifact = receipt_artifacts[0]
            if (
                receipt_artifact.content != receipt_content
                or receipt_artifact.byte_sha256 != row["commit_receipt_sha256"]
                or receipt_artifact.transaction_id != row["transaction_id"]
            ):
                raise _failure_exception(
                    "artifact_hash_mismatch",
                    "transaction row and commit-receipt artifact differ",
                )
        try:
            if receipt_artifact is None:
                receipt_value = json.loads(
                    receipt_content.decode("utf-8"),
                    object_pairs_hook=_json_object_without_duplicates,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ProtocolSchemaError(
                            f"nonfinite JSON constant {token!r} is forbidden"
                        )
                    ),
                )
                if not isinstance(receipt_value, dict):
                    raise ProtocolSchemaError(
                        "internal commit receipt is not an object"
                    )
            else:
                receipt_value = receipt_artifact.json_value()
            receipt_keys = frozenset(
                {
                    "artifacts",
                    "format_version",
                    "identity_sha256",
                    "partition_role",
                    "schema",
                    "state_after",
                    "state_before",
                    "transaction_committed",
                    "transaction_id",
                    "transaction_kind",
                }
            )
            _require_exact_keys(
                receipt_value,
                receipt_keys,
                label="transaction commit receipt",
            )
            if receipt_content != canonical_json_bytes(receipt_value):
                raise ProtocolSchemaError("commit receipt is not canonical")
            expected_fields = {
                "format_version": _FORMAT_VERSION,
                "identity_sha256": row["identity_sha256"],
                "partition_role": row["partition_role"],
                "schema": LEDGER_SCHEMA,
                "state_after": row["state_after"],
                "state_before": row["state_before"],
                "transaction_committed": True,
                "transaction_id": row["transaction_id"],
                "transaction_kind": row["transaction_kind"],
            }
            for key, expected_value in expected_fields.items():
                if receipt_value[key] != expected_value:
                    raise ProtocolSchemaError(
                        f"commit receipt {key} differs from transaction row"
                    )
            inventory_artifacts = (
                artifacts
                if receipt_artifact is None
                else tuple(
                    artifact
                    for artifact in artifacts
                    if artifact is not receipt_artifact
                )
            )
            expected_inventory = [
                {"byte_sha256": artifact.byte_sha256, "name": artifact.name}
                for artifact in sorted(
                    inventory_artifacts,
                    key=lambda item: item.name,
                )
            ]
            if receipt_value["artifacts"] != expected_inventory:
                raise ProtocolSchemaError(
                    "commit receipt inventory differs from committed artifacts"
                )
            _validate_tx_id(row["transaction_id"], label="transaction_id")
            _validate_sha256(row["identity_sha256"], label="identity_sha256")
            _validate_partition_role(row["partition_role"])
            CaptureState(row["state_before"])
            CaptureState(row["state_after"])
            self._strict_validate_transaction_contract(row, artifacts)
        except (
            KeyError,
            TypeError,
            ValueError,
            ProtocolSchemaError,
            ProtocolViolation,
        ) as error:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "transaction commit receipt fails strict recovery verification",
            ) from error
        return TransactionResult(
            transaction_id=row["transaction_id"],
            transaction_kind=row["transaction_kind"],
            partition_role=row["partition_role"],
            state_before=CaptureState(row["state_before"]),
            state_after=CaptureState(row["state_after"]),
            identity_sha256=row["identity_sha256"],
            artifacts=artifacts,
            idempotent_recovery=idempotent,
        )

    @staticmethod
    def _stored_artifact_from_row(row: sqlite3.Row) -> StoredArtifact:
        content = bytes(row["content"])
        if artifact_sha256(content) != row["byte_sha256"]:
            raise _failure_exception(
                "artifact_hash_mismatch",
                f"stored artifact {row['name']} hash mismatch",
            )
        return StoredArtifact(
            name=row["name"],
            kind=row["kind"],
            content=content,
            byte_sha256=row["byte_sha256"],
            transaction_id=row["transaction_id"],
        )

    def authorize(self, specification: AuthorizationSpec) -> TransactionResult:
        """Atomically persist semantic authorization and sticky transition."""

        if not isinstance(specification, AuthorizationSpec):
            raise ProtocolSchemaError("specification must be AuthorizationSpec")
        role = specification.partition_role
        authorization_hash = specification.authorization_hash
        with self._immediate_transaction() as connection:
            self._ensure_no_global_terminal(connection)
            authorization_prefix = _AUTHORIZATION_PREFIX_BY_PARTITION[role]
            present_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM campaign_artifact_inventory"
                ).fetchall()
            }
            required_predecessors = {
                name
                for prefix in CampaignPrefix
                if prefix.value < authorization_prefix.value
                for name in _PREFIX_MANDATORY_ARTIFACTS[prefix]
            }
            if not required_predecessors <= present_names:
                raise ProtocolStateError(
                    f"cannot materialize {authorization_prefix.name} before all "
                    "mandatory predecessor artifacts"
                )
            _validate_registered_authorization_bindings(connection, specification)
            existing = connection.execute(
                "SELECT * FROM panels WHERE partition_role = ?", (role,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["authorization_hash"] == authorization_hash
                    and bytes(existing["authorization_semantic_payload"])
                    == specification.semantic_payload_bytes
                ):
                    transaction_id = existing["authorization_transaction_id"]
                    # Read after this no-op transaction commits.
                    pass
                else:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "partition already has a different authorization",
                    )
            else:
                transaction_id = self._new_transaction_id()
                campaign_row = connection.execute(
                    "SELECT value FROM protocol_metadata WHERE key = 'campaign_id'"
                ).fetchone()
                if campaign_row is None:
                    connection.execute(
                        "INSERT INTO protocol_metadata(key, value) VALUES (?, ?)",
                        ("campaign_id", specification.campaign_id),
                    )
                elif campaign_row["value"] != specification.campaign_id:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "ledger already belongs to a different campaign",
                    )
                common: dict[str, JsonValue] = {
                    "authorization_hash": authorization_hash,
                    "authorization_transaction_id": transaction_id,
                    "partition_role": role,
                }
                authorization_payload = specification.semantic_payload()
                authorization_payload.update(common)
                authorization = self._artifact(
                    name=f"{role}_capture_authorization.json",
                    kind="capture_authorization",
                    transaction_id=transaction_id,
                    payload=authorization_payload,
                )
                transition = self._artifact(
                    name=f"{role}_capture_transition_authorized.json",
                    kind="capture_transition_authorized",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.AUTHORIZED.value,
                        "state_before": CaptureState.UNOPENED.value,
                    },
                )
                connection.execute(
                    """
                    INSERT INTO panels(
                        partition_role, campaign_id, state, authorization_hash,
                        authorization_transaction_id,
                        authorization_semantic_payload,
                        expected_row_set_mode,
                        row_derivation_protocol_hash,
                        expected_row_set_sha256,
                        expected_ordered_row_sequence_sha256,
                        expected_row_count,
                        expected_row_index_payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        role,
                        specification.campaign_id,
                        CaptureState.AUTHORIZED.value,
                        authorization_hash,
                        transaction_id,
                        specification.semantic_payload_bytes,
                        specification.expected_row_set_mode.value,
                        specification.row_derivation_protocol_hash,
                        specification.expected_row_set_sha256,
                        specification.expected_ordered_row_sequence_sha256,
                        specification.expected_row_count,
                        specification.expected_row_index_payload_sha256,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="authorization",
                    partition_role=role,
                    state_before=CaptureState.UNOPENED,
                    state_after=CaptureState.AUTHORIZED,
                    identity_sha256=authorization_hash,
                    artifacts=(authorization, transition),
                    receipt_name=f"{role}_authorization_transaction_commit.json",
                    campaign_prefix=_AUTHORIZATION_PREFIX_BY_PARTITION[role],
                )
        return self._load_transaction(
            transaction_id, idempotent=existing is not None
        )

    def commit_row_index(self, payload: RowIndexPayload) -> TransactionResult:
        """Atomically commit the canonical row index in either auth mode."""

        if not isinstance(payload, RowIndexPayload):
            raise ProtocolSchemaError("payload must be RowIndexPayload")
        role = payload.partition_role
        payload_hash = payload.payload_sha256
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if panel["row_index_transaction_id"] is not None:
                if panel["row_index_payload_sha256"] != payload_hash:
                    raise _failure_exception(
                        "row_index_payload_hash_mismatch",
                        "a different row index is already committed",
                    )
                transaction_id = panel["row_index_transaction_id"]
                existing = True
            else:
                existing = False
                state = CaptureState(panel["state"])
                if state is not CaptureState.AUTHORIZED:
                    raise ProtocolStateError(
                        f"cannot commit row index from {state.value}"
                    )
                if panel["row_index_failure_transaction_id"] is not None:
                    raise ProtocolStateError("row-index failure is sticky")
                if (
                    payload.row_derivation_protocol_hash
                    != panel["row_derivation_protocol_hash"]
                ):
                    raise _failure_exception(
                        "row_derivation_protocol_hash_mismatch",
                        "row index uses a different derivation protocol",
                    )
                bindings = _stored_authorization_bindings(panel)
                _validate_row_index_authorization_bindings(payload, bindings)
                mode = AuthorizationMode(panel["expected_row_set_mode"])
                if mode is AuthorizationMode.PRECOMPUTED:
                    expected = (
                        panel["expected_row_set_sha256"],
                        panel["expected_ordered_row_sequence_sha256"],
                        panel["expected_row_count"],
                        panel["expected_row_index_payload_sha256"],
                    )
                    observed = (
                        payload.row_set_sha256,
                        payload.ordered_row_sequence_sha256,
                        payload.row_count,
                        payload_hash,
                    )
                    if observed != expected:
                        raise _failure_exception(
                            "precomputed_row_index_mismatch",
                            "Mode-A row index differs from the authorized ordered manifest",
                        )
                transaction_id = self._new_transaction_id()
                common: dict[str, JsonValue] = {
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "derived_ordered_row_sequence_sha256": (
                        payload.ordered_row_sequence_sha256
                    ),
                    "derived_row_count": payload.row_count,
                    "derived_row_set_sha256": payload.row_set_sha256,
                    "partition_role": role,
                    "row_derivation_protocol_hash": (
                        payload.row_derivation_protocol_hash
                    ),
                    "row_index_payload_sha256": payload_hash,
                    "row_index_transaction_id": transaction_id,
                }
                manifest = self._artifact(
                    name=f"row_index_manifest_{role}.json",
                    kind="row_index_manifest",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "row_index": payload.as_dict(),
                        "schema": ROW_INDEX_SCHEMA,
                    },
                )
                transition = self._artifact(
                    name=f"{role}_row_index_committed.json",
                    kind="row_index_committed",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.ROW_INDEX_COMMITTED.value,
                        "state_before": CaptureState.AUTHORIZED.value,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET
                        state = ?, row_index_transaction_id = ?,
                        row_index_payload_sha256 = ?,
                        committed_row_set_sha256 = ?,
                        committed_ordered_row_sequence_sha256 = ?,
                        committed_row_count = ?, revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.ROW_INDEX_COMMITTED.value,
                        transaction_id,
                        payload_hash,
                        payload.row_set_sha256,
                        payload.ordered_row_sequence_sha256,
                        payload.row_count,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="row_index_commit",
                    partition_role=role,
                    state_before=CaptureState.AUTHORIZED,
                    state_after=CaptureState.ROW_INDEX_COMMITTED,
                    identity_sha256=payload_hash,
                    artifacts=(manifest, transition),
                    receipt_name=f"{role}_row_index_transaction_commit.json",
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    def record_row_index_failure(
        self,
        partition_role: str,
        *,
        campaign_id: str,
        reason_code: str,
        last_completed_prefix: CampaignPrefix | str,
        active_prefix: CampaignPrefix | str,
        failure_stage: str,
        sanitized_failure_details: Mapping[str, object] | None = None,
    ) -> TransactionResult:
        """Atomically make row-index failure sticky, without tensor start."""

        role = _validate_partition_role(partition_role)
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        completed, active = _normalize_prefix_pair(
            last_completed_prefix, active_prefix
        )
        _validate_capture_prefix_for_partition(role, active)
        if completed is None:
            raise ProtocolSchemaError(
                "row-index failure requires a completed predecessor prefix"
            )
        completed_prefix = completed.name
        current_prefix = active.name
        stage = _validate_identifier(failure_stage, label="failure_stage")
        disposition = classify_failure(reason_code)
        details = validate_sanitized_failure_details(sanitized_failure_details)
        _validate_failure_detail_context(
            details,
            partition_role=role,
            reason_code=reason_code,
            last_completed_prefix=completed_prefix,
            active_prefix=current_prefix,
            failure_stage=stage,
        )
        identity_payload = {
            "details": details,
            "campaign_id": campaign,
            "active_prefix": current_prefix,
            "failure_stage": stage,
            "last_completed_prefix": completed_prefix,
            "partition_role": role,
            "reason_code": reason_code,
        }
        identity_hash = hashlib.sha256(
            _ROW_INDEX_FAILURE_DOMAIN + canonical_json_bytes(identity_payload)
        ).hexdigest()
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if panel["row_index_failure_transaction_id"] is not None:
                transaction_id = panel["row_index_failure_transaction_id"]
                transaction = connection.execute(
                    "SELECT identity_sha256 FROM ledger_transactions "
                    "WHERE transaction_id = ?",
                    (transaction_id,),
                ).fetchone()
                if transaction is None or transaction["identity_sha256"] != identity_hash:
                    raise ProtocolStateError("a different row-index failure is sticky")
                existing = True
            else:
                existing = False
                state = CaptureState(panel["state"])
                if state is not CaptureState.AUTHORIZED:
                    raise ProtocolStateError(
                        f"cannot fail row derivation from {state.value}"
                    )
                if panel["campaign_id"] != campaign:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "row-index failure campaign differs from authorization",
                    )
                if panel["row_index_transaction_id"] is not None:
                    raise ProtocolStateError("committed row index cannot become failed")
                if panel["capture_start_transaction_id"] is not None:
                    raise ProtocolStateError(
                        "row-index failure cannot coexist with tensor capture start"
                    )
                transaction_id = self._new_transaction_id()
                common: dict[str, JsonValue] = {
                    "active_prefix": current_prefix,
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "campaign_id": campaign,
                    "failure_class": disposition.failure_class.value,
                    "failure_reason_code": reason_code,
                    "failure_stage": stage,
                    "last_completed_prefix": completed_prefix,
                    "partition_role": role,
                    "row_derivation_protocol_hash": panel[
                        "row_derivation_protocol_hash"
                    ],
                    "row_index_failure_transaction_id": transaction_id,
                    "sanitized_failure_details": details,
                    "tensor_capture_started": False,
                    "terminal_state": disposition.terminal_state.value,
                }
                failure = self._artifact(
                    name=f"row_index_failure_{role}.json",
                    kind="row_index_failure",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                    },
                )
                transition = self._artifact(
                    name=f"{role}_row_index_failed.json",
                    kind="row_index_failed",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.ROW_INDEX_FAILED.value,
                        "state_before": CaptureState.AUTHORIZED.value,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET
                        state = ?, row_index_failure_transaction_id = ?,
                        terminal_state = ?, failure_reason_code = ?,
                        revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.ROW_INDEX_FAILED.value,
                        transaction_id,
                        disposition.terminal_state.value,
                        reason_code,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="row_index_failure",
                    partition_role=role,
                    state_before=CaptureState.AUTHORIZED,
                    state_after=CaptureState.ROW_INDEX_FAILED,
                    identity_sha256=identity_hash,
                    artifacts=(failure, transition),
                    receipt_name=f"{role}_row_index_failure_transaction_commit.json",
                    campaign_prefix=_coerce_prefix(
                        current_prefix, label="active_prefix"
                    ),
                )
                self._set_global_terminal_pending(
                    connection,
                    terminal_state=disposition.terminal_state.value,
                    terminal_identity_sha256=identity_hash,
                    origin_transaction_id=transaction_id,
                    partition_role=role,
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    def finalize_row_index_failure(
        self,
        partition_role: str,
        *,
        campaign_id: str,
        last_completed_prefix: CampaignPrefix | str,
        active_prefix: CampaignPrefix | str,
        failure_stage: str,
    ) -> TransactionResult:
        """Recover/finalize only row_index_failed -> consumed_failed."""

        role = _validate_partition_role(partition_role)
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        completed, active = _normalize_prefix_pair(
            last_completed_prefix, active_prefix
        )
        _validate_capture_prefix_for_partition(role, active)
        if completed is None:
            raise ProtocolSchemaError(
                "row-index failure requires a completed predecessor prefix"
            )
        completed_prefix = completed.name
        current_prefix = active.name
        stage = _validate_identifier(failure_stage, label="failure_stage")
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            state = CaptureState(panel["state"])
            if (
                state is CaptureState.CONSUMED_FAILED
                and panel["row_index_failure_transaction_id"] is not None
                and panel["terminal_transaction_id"] is not None
            ):
                failure_artifact_row = connection.execute(
                    "SELECT * FROM artifacts WHERE name = ?",
                    (f"row_index_failure_{role}.json",),
                ).fetchone()
                if failure_artifact_row is None:
                    raise ProtocolStateError("row-index failure artifact is missing")
                failure_payload = self._stored_artifact_from_row(
                    failure_artifact_row
                ).json_value()
                if (
                    campaign,
                    completed_prefix,
                    current_prefix,
                    stage,
                ) != (
                    failure_payload.get("campaign_id"),
                    failure_payload.get("last_completed_prefix"),
                    failure_payload.get("active_prefix"),
                    failure_payload.get("failure_stage"),
                ):
                    raise _failure_exception(
                        "authorization_mismatch",
                        "row-index finalization context differs from sticky failure",
                    )
                transaction_id = panel["terminal_transaction_id"]
                existing = True
            else:
                existing = False
                if state is not CaptureState.ROW_INDEX_FAILED:
                    raise ProtocolStateError(
                        f"cannot finalize row-index failure from {state.value}"
                    )
                if panel["capture_start_transaction_id"] is not None:
                    raise ProtocolStateError(
                        "row-index-failed panel has a forbidden capture start"
                    )
                failure_tx = panel["row_index_failure_transaction_id"]
                if failure_tx is None:
                    raise ProtocolStateError("row-index failure transaction is missing")
                failure_artifact_row = connection.execute(
                    "SELECT * FROM artifacts WHERE name = ?",
                    (f"row_index_failure_{role}.json",),
                ).fetchone()
                if failure_artifact_row is None:
                    raise ProtocolStateError("row-index failure artifact is missing")
                failure_artifact = self._stored_artifact_from_row(
                    failure_artifact_row
                )
                failure_payload = failure_artifact.json_value()
                expected_context = (
                    campaign,
                    completed_prefix,
                    current_prefix,
                    stage,
                )
                observed_context = (
                    failure_payload.get("campaign_id"),
                    failure_payload.get("last_completed_prefix"),
                    failure_payload.get("active_prefix"),
                    failure_payload.get("failure_stage"),
                )
                if expected_context != observed_context:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "row-index finalization context differs from sticky failure",
                    )
                reason_code = str(failure_payload["failure_reason_code"])
                disposition = classify_failure(reason_code)
                identity_hash = hashlib.sha256(
                    _TERMINAL_FAILURE_DOMAIN + bytes.fromhex(failure_tx)
                ).hexdigest()
                transaction_id = self._new_transaction_id()
                common: dict[str, JsonValue] = {
                    "active_prefix": current_prefix,
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "campaign_id": campaign,
                    "failure_class": disposition.failure_class.value,
                    "failure_reason_code": reason_code,
                    "failure_stage": stage,
                    "last_completed_prefix": completed_prefix,
                    "partition_role": role,
                    "row_derivation_protocol_hash": panel[
                        "row_derivation_protocol_hash"
                    ],
                    "row_index_failure_transaction_id": failure_tx,
                    "sanitized_failure_details": failure_payload[
                        "sanitized_failure_details"
                    ],
                    "tensor_capture_started": False,
                    "terminal_state": disposition.terminal_state.value,
                    "terminal_transaction_id": transaction_id,
                }
                transition = self._artifact(
                    name=f"{role}_capture_transition_consumed_failed.json",
                    kind="capture_transition_consumed_failed",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.CONSUMED_FAILED.value,
                        "state_before": CaptureState.ROW_INDEX_FAILED.value,
                    },
                )
                reached_inventory = self._preview_reached_inventory(
                    connection,
                    (transition,),
                    prefix=active,
                )
                if disposition.failure_class is FailureClass.RESOURCE_LIMIT:
                    primary_name = "resource_limit_failure.json"
                    primary_kind = "resource_limit_failure"
                else:
                    primary_name = "invalid_campaign_receipt.json"
                    primary_kind = "invalid_campaign_receipt"
                primary_receipt = self._artifact(
                    name=primary_name,
                    kind=primary_kind,
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "capture_attempt_id": None,
                        "capture_start_transaction_id": None,
                        "committed_ordered_row_sequence_sha256": None,
                        "committed_row_count": None,
                        "committed_row_set_sha256": None,
                        "durably_reached_artifact_exclusions": [
                            "primary_receipt_self",
                            "transaction_commit_receipt",
                        ],
                        "durably_reached_artifacts": [
                            entry.as_dict() for entry in reached_inventory
                        ],
                        "format_version": _FORMAT_VERSION,
                        "primary_receipt_kind": primary_kind,
                        "row_index_payload_sha256": None,
                        "row_index_transaction_id": None,
                        "schema": PRIMARY_FAILURE_RECEIPT_SCHEMA,
                        "shard_plan_sha256": None,
                        "shard_plan_transaction_id": None,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET state = ?, terminal_transaction_id = ?,
                        terminal_identity_sha256 = ?, revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.CONSUMED_FAILED.value,
                        transaction_id,
                        identity_hash,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="row_index_failure_finalization",
                    partition_role=role,
                    state_before=CaptureState.ROW_INDEX_FAILED,
                    state_after=CaptureState.CONSUMED_FAILED,
                    identity_sha256=identity_hash,
                    artifacts=(transition, primary_receipt),
                    receipt_name=(
                        f"{role}_row_index_failure_finalization_commit.json"
                    ),
                    campaign_prefix=_coerce_prefix(
                        current_prefix, label="active_prefix"
                    ),
                    allow_terminal_pending_transaction_id=failure_tx,
                )
                self._set_global_terminal(
                    connection,
                    terminal_state=disposition.terminal_state.value,
                    terminal_identity_sha256=identity_hash,
                    terminal_transaction_id=transaction_id,
                    partition_role=role,
                    pending_origin_transaction_id=failure_tx,
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    def commit_shard_plan(self, plan: ShardPlan) -> TransactionResult:
        """Freeze the separately named deterministic plan before capture start."""

        if not isinstance(plan, ShardPlan):
            raise ProtocolSchemaError("plan must be ShardPlan")
        role = plan.partition_role
        identity_hash = plan.plan_sha256
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if panel["shard_plan_transaction_id"] is not None:
                if panel["shard_plan_sha256"] != identity_hash:
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "a different capture shard plan is already frozen",
                    )
                transaction_id = panel["shard_plan_transaction_id"]
                existing = True
            else:
                existing = False
                state = CaptureState(panel["state"])
                if state is not CaptureState.ROW_INDEX_COMMITTED:
                    raise ProtocolStateError(
                        f"cannot freeze shard plan from {state.value}"
                    )
                expected = (
                    panel["committed_row_set_sha256"],
                    panel["committed_ordered_row_sequence_sha256"],
                    panel["committed_row_count"],
                    panel["row_index_payload_sha256"],
                )
                observed = (
                    plan.row_set_sha256,
                    plan.ordered_row_sequence_sha256,
                    plan.row_count,
                    plan.row_index_payload_sha256,
                )
                if expected != observed:
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "shard plan differs from the committed row index",
                    )
                bindings = _stored_authorization_bindings(panel)
                if (
                    plan.maximum_rows_per_shard
                    != bindings["maximum_rows_per_shard"]
                ):
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "shard plan limit differs from the frozen authorization",
                    )
                transaction_id = self._new_transaction_id()
                plan_artifact = self._artifact(
                    name=f"capture_shard_plan_{role}.json",
                    kind="capture_shard_plan",
                    transaction_id=transaction_id,
                    payload={
                        "authorization_hash": panel["authorization_hash"],
                        "authorization_transaction_id": panel[
                            "authorization_transaction_id"
                        ],
                        "format_version": _FORMAT_VERSION,
                        "partition_role": role,
                        "row_index_payload_sha256": panel[
                            "row_index_payload_sha256"
                        ],
                        "row_index_transaction_id": panel[
                            "row_index_transaction_id"
                        ],
                        "schema": SHARD_PLAN_SCHEMA,
                        "shard_plan": plan.as_dict(),
                        "shard_plan_sha256": identity_hash,
                        "shard_plan_transaction_id": transaction_id,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET shard_plan_transaction_id = ?,
                        shard_plan_sha256 = ?, revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (transaction_id, identity_hash, role),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="shard_plan_commit",
                    partition_role=role,
                    state_before=CaptureState.ROW_INDEX_COMMITTED,
                    state_after=CaptureState.ROW_INDEX_COMMITTED,
                    identity_sha256=identity_hash,
                    artifacts=(plan_artifact,),
                    receipt_name=f"{role}_shard_plan_transaction_commit.json",
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    def start_capture(
        self,
        plan: ShardPlan,
        capture_root: str | os.PathLike[str],
        *,
        capture_schema_hash: str,
        capture_implementation_hash: str,
    ) -> TransactionResult:
        """Persist the start receipt before any tensor work may occur."""

        if not isinstance(plan, ShardPlan):
            raise ProtocolSchemaError("plan must be ShardPlan")
        schema_hash = _validate_sha256(
            capture_schema_hash, label="capture_schema_hash"
        )
        implementation_hash = _validate_sha256(
            capture_implementation_hash,
            label="capture_implementation_hash",
        )
        root_identity_hash = _capture_root_identity_sha256(capture_root)
        role = plan.partition_role
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            identity_payload: dict[str, JsonValue] = {
                "capture_implementation_hash": implementation_hash,
                "capture_schema_hash": schema_hash,
                "capture_root_identity_sha256": root_identity_hash,
                "partition_role": role,
                "row_index_payload_sha256": plan.row_index_payload_sha256,
                "shard_plan_sha256": plan.plan_sha256,
                "shard_plan_transaction_id": panel[
                    "shard_plan_transaction_id"
                ],
            }
            identity_hash = hashlib.sha256(
                _CAPTURE_START_DOMAIN + canonical_json_bytes(identity_payload)
            ).hexdigest()
            if panel["capture_start_transaction_id"] is not None:
                if panel["capture_start_identity_sha256"] != identity_hash:
                    raise _failure_exception(
                        "capture_attempt_identity_mismatch",
                        "capture already started with different identities",
                    )
                transaction_id = panel["capture_start_transaction_id"]
                existing = True
            else:
                existing = False
                if _inventory_finalized_shard_indexes(
                    plan, capture_root, require_complete=False
                ):
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "finalized shard exists before durable capture start",
                    )
                state = CaptureState(panel["state"])
                if state is not CaptureState.ROW_INDEX_COMMITTED:
                    raise ProtocolStateError(
                        f"tensor capture cannot start from {state.value}"
                    )
                if panel["shard_plan_transaction_id"] is None:
                    raise ProtocolStateError(
                        "capture shard plan must commit before tensor capture starts"
                    )
                expected = (
                    panel["committed_row_set_sha256"],
                    panel["committed_ordered_row_sequence_sha256"],
                    panel["committed_row_count"],
                    panel["row_index_payload_sha256"],
                )
                observed = (
                    plan.row_set_sha256,
                    plan.ordered_row_sequence_sha256,
                    plan.row_count,
                    plan.row_index_payload_sha256,
                )
                if expected != observed:
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "shard plan differs from committed row index",
                    )
                if panel["shard_plan_sha256"] != plan.plan_sha256:
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "started shard plan differs from the frozen plan artifact",
                    )
                bindings = _stored_authorization_bindings(panel)
                if (
                    schema_hash != bindings["capture_schema_hash"]
                    or implementation_hash
                    != bindings["capture_implementation_hash"]
                ):
                    raise _failure_exception(
                        "capture_attempt_identity_mismatch",
                        "capture implementation differs from frozen authorization",
                    )
                attempt_id = self._new_capture_attempt_id()
                transaction_id = self._new_transaction_id()
                common: dict[str, JsonValue] = {
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "capture_attempt_id": attempt_id,
                    "capture_implementation_hash": implementation_hash,
                    "capture_schema_hash": schema_hash,
                    "capture_root_identity_sha256": root_identity_hash,
                    "capture_start_transaction_id": transaction_id,
                    "committed_ordered_row_sequence_sha256": plan.ordered_row_sequence_sha256,
                    "committed_row_count": plan.row_count,
                    "committed_row_set_sha256": plan.row_set_sha256,
                    "partition_role": role,
                    "row_index_transaction_id": panel["row_index_transaction_id"],
                    "row_index_payload_sha256": panel[
                        "row_index_payload_sha256"
                    ],
                    "shard_plan_transaction_id": panel[
                        "shard_plan_transaction_id"
                    ],
                    "shard_plan_sha256": plan.plan_sha256,
                    "tensor_capture_started": True,
                }
                receipt = self._artifact(
                    name=f"{role}_tensor_capture_started.json",
                    kind="tensor_capture_started",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.CAPTURE_STARTED.value,
                        "state_before": CaptureState.ROW_INDEX_COMMITTED.value,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET state = ?,
                        capture_start_transaction_id = ?,
                        capture_start_identity_sha256 = ?,
                        capture_attempt_id = ?,
                        capture_root_identity_sha256 = ?,
                        capture_schema_hash = ?, capture_implementation_hash = ?,
                        revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.CAPTURE_STARTED.value,
                        transaction_id,
                        identity_hash,
                        attempt_id,
                        root_identity_hash,
                        schema_hash,
                        implementation_hash,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="capture_start",
                    partition_role=role,
                    state_before=CaptureState.ROW_INDEX_COMMITTED,
                    state_after=CaptureState.CAPTURE_STARTED,
                    identity_sha256=identity_hash,
                    artifacts=(receipt,),
                    receipt_name=f"{role}_capture_start_transaction_commit.json",
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    @staticmethod
    def _resume_checkpoint_sha256(
        *,
        partition_role: str,
        capture_attempt_id: str,
        shard_plan_sha256: str,
        capture_root_identity_sha256: str,
        rows: Sequence[sqlite3.Row],
    ) -> str:
        records: list[JsonValue] = []
        previous_index = -1
        for row in rows:
            index = _validate_uint32(
                row["shard_index"], label="resume checkpoint shard index"
            )
            if index <= previous_index:
                raise ProtocolSchemaError(
                    "resume checkpoint shard indexes are not strictly ordered"
                )
            previous_index = index
            payload = bytes(row["verified_record"])
            payload_hash = artifact_sha256(payload)
            if payload_hash != row["verified_record_sha256"]:
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "resume checkpoint record hash changed",
                )
            records.append(
                {
                    "shard_index": index,
                    "verified_record_sha256": payload_hash,
                }
            )
        return hashlib.sha256(
            _RESUME_CHECKPOINT_DOMAIN
            + canonical_json_bytes(
                {
                    "capture_attempt_id": capture_attempt_id,
                    "capture_root_identity_sha256": (
                        capture_root_identity_sha256
                    ),
                    "partition_role": partition_role,
                    "records": records,
                    "shard_plan_sha256": shard_plan_sha256,
                }
            )
        ).hexdigest()

    @classmethod
    def _validate_resume_checkpoint(
        cls,
        connection: sqlite3.Connection,
        *,
        partition_role: str,
        capture_attempt_id: str,
        shard_plan_sha256: str,
        capture_root_identity_sha256: str,
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                "SELECT * FROM capture_resume_shards "
                "WHERE partition_role = ? AND capture_attempt_id = ? "
                "ORDER BY shard_index",
                (partition_role, capture_attempt_id),
            ).fetchall()
        )
        for row in rows:
            if (
                row["shard_plan_sha256"] != shard_plan_sha256
                or row["capture_root_identity_sha256"]
                != capture_root_identity_sha256
            ):
                raise _failure_exception(
                    "capture_attempt_identity_mismatch",
                    "resume checkpoint uses different capture identities",
                )
        panel = connection.execute(
            "SELECT resume_checkpoint_count, resume_checkpoint_sha256 "
            "FROM panels WHERE partition_role = ?",
            (partition_role,),
        ).fetchone()
        if panel is None:
            raise ProtocolStateError("resume checkpoint panel is missing")
        expected_count = panel["resume_checkpoint_count"]
        expected_hash = panel["resume_checkpoint_sha256"]
        if expected_count == 0:
            if rows or expected_hash is not None:
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "empty resume checkpoint has unexpected durable evidence",
                )
            return rows
        if expected_count != len(rows) or expected_hash is None:
            raise _failure_exception(
                "shard_hash_mismatch",
                "resume checkpoint row count changed",
            )
        observed_hash = cls._resume_checkpoint_sha256(
            partition_role=partition_role,
            capture_attempt_id=capture_attempt_id,
            shard_plan_sha256=shard_plan_sha256,
            capture_root_identity_sha256=capture_root_identity_sha256,
            rows=rows,
        )
        if observed_hash != expected_hash:
            raise _failure_exception(
                "shard_hash_mismatch", "resume checkpoint root changed"
            )
        return rows

    @staticmethod
    def _validate_resume_shard_bindings(
        connection: sqlite3.Connection,
        *,
        partition_role: str,
        capture_attempt_id: str,
        shard_plan_sha256: str,
        capture_root_identity_sha256: str,
        records: Sequence[VerifiedShardRecord],
    ) -> None:
        current = {record.shard_index: record for record in records}
        rows = ProtocolLedger._validate_resume_checkpoint(
            connection,
            partition_role=partition_role,
            capture_attempt_id=capture_attempt_id,
            shard_plan_sha256=shard_plan_sha256,
            capture_root_identity_sha256=capture_root_identity_sha256,
        )
        for row in rows:
            if (
                row["shard_plan_sha256"] != shard_plan_sha256
                or row["capture_root_identity_sha256"]
                != capture_root_identity_sha256
            ):
                raise _failure_exception(
                    "capture_attempt_identity_mismatch",
                    "durable resume evidence uses different capture identities",
                )
            index = row["shard_index"]
            if index not in current:
                raise _failure_exception(
                    "shard_file_missing",
                    "a shard preserved by resume is no longer finalized",
                )
            payload = canonical_json_bytes(current[index].as_dict())
            stored_payload = bytes(row["verified_record"])
            if (
                artifact_sha256(stored_payload)
                != row["verified_record_sha256"]
                or stored_payload != payload
            ):
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "a shard changed after durable resume verification",
                )

    @classmethod
    def _persist_resume_shard_bindings(
        cls,
        connection: sqlite3.Connection,
        *,
        partition_role: str,
        capture_attempt_id: str,
        shard_plan_sha256: str,
        capture_root_identity_sha256: str,
        records: Sequence[VerifiedShardRecord],
    ) -> None:
        cls._validate_resume_shard_bindings(
            connection,
            partition_role=partition_role,
            capture_attempt_id=capture_attempt_id,
            shard_plan_sha256=shard_plan_sha256,
            capture_root_identity_sha256=capture_root_identity_sha256,
            records=records,
        )
        for record in records:
            payload = canonical_json_bytes(record.as_dict())
            existing = connection.execute(
                "SELECT verified_record, verified_record_sha256 "
                "FROM capture_resume_shards WHERE partition_role = ? "
                "AND capture_attempt_id = ? AND shard_index = ?",
                (partition_role, capture_attempt_id, record.shard_index),
            ).fetchone()
            if existing is not None:
                if (
                    bytes(existing["verified_record"]) != payload
                    or existing["verified_record_sha256"]
                    != artifact_sha256(payload)
                ):
                    raise _failure_exception(
                        "shard_hash_mismatch",
                        "resume attempted to replace preserved shard evidence",
                    )
                continue
            connection.execute(
                """
                INSERT INTO capture_resume_shards(
                    partition_role, capture_attempt_id, shard_index,
                    shard_plan_sha256, capture_root_identity_sha256,
                    verified_record, verified_record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    partition_role,
                    capture_attempt_id,
                    record.shard_index,
                    shard_plan_sha256,
                    capture_root_identity_sha256,
                    payload,
                    artifact_sha256(payload),
                ),
            )
        rows = tuple(
            connection.execute(
                "SELECT * FROM capture_resume_shards "
                "WHERE partition_role = ? AND capture_attempt_id = ? "
                "ORDER BY shard_index",
                (partition_role, capture_attempt_id),
            ).fetchall()
        )
        checkpoint_hash = (
            None
            if not rows
            else cls._resume_checkpoint_sha256(
                partition_role=partition_role,
                capture_attempt_id=capture_attempt_id,
                shard_plan_sha256=shard_plan_sha256,
                capture_root_identity_sha256=capture_root_identity_sha256,
                rows=rows,
            )
        )
        connection.execute(
            "UPDATE panels SET resume_checkpoint_count = ?, "
            "resume_checkpoint_sha256 = ?, revision = revision + 1 "
            "WHERE partition_role = ?",
            (len(rows), checkpoint_hash, partition_role),
        )

    def resume_capture(
        self,
        plan: ShardPlan,
        *,
        capture_attempt_id: str,
        capture_root: str | os.PathLike[str],
        capture_schema_hash: str,
        capture_implementation_hash: str,
        observed_tensor_leading_row_counts_by_shard: (
            Mapping[int, Mapping[str, int]] | None
        ) = None,
    ) -> CaptureResumeResult:
        """Recover the same attempt and reopen every existing frozen shard."""

        if not isinstance(plan, ShardPlan):
            raise ProtocolSchemaError("plan must be ShardPlan")
        role = plan.partition_role
        _validate_tx_id(capture_attempt_id, label="capture_attempt_id")
        root_identity_hash = _capture_root_identity_sha256(capture_root)
        expected = (
            plan.plan_sha256,
            _validate_sha256(capture_schema_hash, label="capture_schema_hash"),
            _validate_sha256(
                capture_implementation_hash,
                label="capture_implementation_hash",
            ),
            capture_attempt_id,
            root_identity_hash,
        )
        with self._lock:
            panel = self._required_panel(self._connection, role)
            if CaptureState(panel["state"]) is not CaptureState.CAPTURE_STARTED:
                raise ProtocolStateError("only capture_started may resume")
            observed = (
                panel["shard_plan_sha256"],
                panel["capture_schema_hash"],
                panel["capture_implementation_hash"],
                panel["capture_attempt_id"],
                panel["capture_root_identity_sha256"],
            )
            if observed != expected:
                raise _failure_exception(
                    "capture_attempt_identity_mismatch",
                    "resume identities differ from the started capture",
                )
            transaction_id = panel["capture_start_transaction_id"]
        assert transaction_id is not None
        root = Path(capture_root)
        existing_indexes = _inventory_finalized_shard_indexes(
            plan,
            root,
            require_complete=False,
        )
        verified = (
            verify_partial_capture_shard_files(
                plan,
                root,
                existing_indexes,
                observed_tensor_leading_row_counts_by_shard=(
                    observed_tensor_leading_row_counts_by_shard
                ),
            )
            if existing_indexes
            else None
        )
        missing = tuple(
            entry.shard_index
            for entry in plan.entries
            if entry.shard_index not in set(existing_indexes)
        )
        with self._immediate_transaction() as connection:
            self._ensure_no_global_terminal(connection)
            panel = self._required_panel(connection, role)
            observed = (
                panel["shard_plan_sha256"],
                panel["capture_schema_hash"],
                panel["capture_implementation_hash"],
                panel["capture_attempt_id"],
                panel["capture_root_identity_sha256"],
            )
            if CaptureState(panel["state"]) is not CaptureState.CAPTURE_STARTED:
                raise ProtocolStateError("capture state changed during resume")
            if observed != expected:
                raise _failure_exception(
                    "capture_attempt_identity_mismatch",
                    "resume identities changed during shard verification",
                )
            self._persist_resume_shard_bindings(
                connection,
                partition_role=role,
                capture_attempt_id=capture_attempt_id,
                shard_plan_sha256=plan.plan_sha256,
                capture_root_identity_sha256=root_identity_hash,
                records=() if verified is None else verified.shards,
            )
        return CaptureResumeResult(
            start_transaction=self._load_transaction(
                transaction_id, idempotent=True
            ),
            verified_existing_shards=verified,
            missing_shard_indexes=missing,
        )

    def record_recoverable_interruption(
        self,
        partition_role: str,
        *,
        capture_attempt_id: str,
        sanitized_failure_details: Mapping[str, object] | None = None,
    ) -> TransactionResult:
        """Record a crash/interruption while leaving state capture_started.

        This is diagnostic only.  It deliberately creates no terminal failure
        and does not allocate a new capture attempt on resume.
        """

        role = _validate_partition_role(partition_role)
        attempt = _validate_tx_id(capture_attempt_id, label="capture_attempt_id")
        details = validate_sanitized_failure_details(sanitized_failure_details)
        if "partition_role" in details and details["partition_role"] != role:
            raise ProtocolSchemaError(
                "interruption detail partition_role differs from attempt"
            )
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if CaptureState(panel["state"]) is not CaptureState.CAPTURE_STARTED:
                raise ProtocolStateError(
                    "recoverable interruption requires capture_started"
                )
            if panel["capture_attempt_id"] != attempt:
                raise _failure_exception(
                    "capture_attempt_identity_mismatch",
                    "interruption refers to a different capture attempt",
                )
            sequence_row = connection.execute(
                "SELECT COUNT(*) AS count FROM ledger_transactions "
                "WHERE partition_role = ? AND transaction_kind = ?",
                (role, "recoverable_interruption"),
            ).fetchone()
            sequence = int(sequence_row["count"])
            identity_payload: dict[str, JsonValue] = {
                "capture_attempt_id": attempt,
                "interruption_sequence": sequence,
                "partition_role": role,
                "sanitized_failure_details": details,
            }
            identity_hash = hashlib.sha256(
                _CAPTURE_START_DOMAIN + canonical_json_bytes(identity_payload)
            ).hexdigest()
            transaction_id = self._new_transaction_id()
            receipt = self._artifact(
                name=f"capture_interruption_{role}_{sequence:06d}.json",
                kind="recoverable_interruption",
                transaction_id=transaction_id,
                payload={
                    **identity_payload,
                    "format_version": _FORMAT_VERSION,
                    "recoverable": True,
                    "schema": LEDGER_SCHEMA,
                    "state_after": CaptureState.CAPTURE_STARTED.value,
                    "state_before": CaptureState.CAPTURE_STARTED.value,
                    "transaction_id": transaction_id,
                },
            )
            self._commit_transaction_record(
                connection,
                transaction_id=transaction_id,
                transaction_kind="recoverable_interruption",
                partition_role=role,
                state_before=CaptureState.CAPTURE_STARTED,
                state_after=CaptureState.CAPTURE_STARTED,
                identity_sha256=identity_hash,
                artifacts=(receipt,),
                receipt_name=(
                    f"{role}_interruption_{sequence:06d}_transaction_commit.json"
                ),
            )
        return self._load_transaction(transaction_id, idempotent=False)

    def finalize_capture_success(
        self,
        plan: ShardPlan,
        capture_root: str | os.PathLike[str],
        *,
        capture_attempt_id: str,
        expected_verification: CaptureVerification,
        observed_tensor_leading_row_counts_by_shard: (
            Mapping[int, Mapping[str, int]] | None
        ) = None,
    ) -> TransactionResult:
        """Reopen exact files, then atomically publish success and manifest."""

        if not isinstance(plan, ShardPlan):
            raise ProtocolSchemaError("plan must be ShardPlan")
        root_identity_hash = _capture_root_identity_sha256(capture_root)
        verification = verify_capture_shard_files(
            plan,
            capture_root,
            observed_tensor_leading_row_counts_by_shard=(
                observed_tensor_leading_row_counts_by_shard
            ),
        )
        if not isinstance(expected_verification, CaptureVerification):
            raise ProtocolSchemaError(
                "expected_verification must be CaptureVerification"
            )
        if (
            expected_verification.verification_source
            != "finalized_files_reopened"
            or expected_verification.canonical_bytes
            != verification.canonical_bytes
        ):
            raise _failure_exception(
                "shard_hash_mismatch",
                "finalization re-verification differs from prior evidence",
            )
        attempt = _validate_tx_id(capture_attempt_id, label="capture_attempt_id")
        role = verification.partition_role
        identity_payload: dict[str, JsonValue] = {
            "capture_attempt_id": attempt,
            "capture_root_identity_sha256": root_identity_hash,
            "verification": verification.as_dict(),
        }
        identity_hash = hashlib.sha256(
            _TERMINAL_SUCCESS_DOMAIN + canonical_json_bytes(identity_payload)
        ).hexdigest()
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if panel["terminal_transaction_id"] is not None:
                if (
                    CaptureState(panel["state"]) is CaptureState.CONSUMED_SUCCESS
                    and panel["terminal_identity_sha256"] == identity_hash
                ):
                    transaction_id = panel["terminal_transaction_id"]
                    existing = True
                else:
                    raise ProtocolStateError("panel already has a different terminal")
            else:
                existing = False
                if CaptureState(panel["state"]) is not CaptureState.CAPTURE_STARTED:
                    raise ProtocolStateError("success requires capture_started")
                self._validate_capture_terminal_identity(
                    panel,
                    attempt=attempt,
                    verification=verification,
                    capture_root_identity_sha256=root_identity_hash,
                )
                self._validate_resume_shard_bindings(
                    connection,
                    partition_role=role,
                    capture_attempt_id=attempt,
                    shard_plan_sha256=plan.plan_sha256,
                    capture_root_identity_sha256=root_identity_hash,
                    records=verification.shards,
                )
                transaction_id = self._new_transaction_id()
                self._register_capture_shard_files(
                    connection,
                    plan=plan,
                    capture_root=capture_root,
                    shard_indexes=(
                        record.shard_index for record in verification.shards
                    ),
                    prefix=self._default_prefix_for_partition(role),
                    transaction_id=transaction_id,
                    expected_hashes={
                        record.shard_index: record.shard_byte_sha256
                        for record in verification.shards
                    },
                )
                common: dict[str, JsonValue] = {
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "capture_attempt_id": attempt,
                    "capture_implementation_hash": panel[
                        "capture_implementation_hash"
                    ],
                    "capture_root_identity_sha256": root_identity_hash,
                    "capture_schema_hash": panel["capture_schema_hash"],
                    "capture_start_transaction_id": panel[
                        "capture_start_transaction_id"
                    ],
                    "committed_ordered_row_sequence_sha256": panel[
                        "committed_ordered_row_sequence_sha256"
                    ],
                    "committed_row_count": panel["committed_row_count"],
                    "committed_row_set_sha256": panel[
                        "committed_row_set_sha256"
                    ],
                    "partition_role": role,
                    "row_index_payload_sha256": panel[
                        "row_index_payload_sha256"
                    ],
                    "row_index_transaction_id": panel[
                        "row_index_transaction_id"
                    ],
                    "shard_plan_sha256": panel["shard_plan_sha256"],
                    "shard_plan_transaction_id": panel[
                        "shard_plan_transaction_id"
                    ],
                    "terminal_transaction_id": transaction_id,
                }
                manifest = self._artifact(
                    name=f"capture_shard_manifest_{role}.json",
                    kind="capture_shard_manifest",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        **verification.as_dict(),
                    },
                )
                transition = self._artifact(
                    name=f"{role}_capture_transition_consumed_success.json",
                    kind="capture_transition_consumed_success",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "capture_verification_sha256": (
                            verification.verification_sha256
                        ),
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.CONSUMED_SUCCESS.value,
                        "state_before": CaptureState.CAPTURE_STARTED.value,
                    },
                )
                connection.execute(
                    """
                    UPDATE panels SET state = ?, terminal_transaction_id = ?,
                        terminal_identity_sha256 = ?, terminal_state = NULL,
                        failure_reason_code = NULL, revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.CONSUMED_SUCCESS.value,
                        transaction_id,
                        identity_hash,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="capture_success_finalization",
                    partition_role=role,
                    state_before=CaptureState.CAPTURE_STARTED,
                    state_after=CaptureState.CONSUMED_SUCCESS,
                    identity_sha256=identity_hash,
                    artifacts=(manifest, transition),
                    receipt_name=f"{role}_capture_success_transaction_commit.json",
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    def finalize_capture_failure(
        self,
        partition_role: str,
        *,
        campaign_id: str,
        capture_attempt_id: str,
        reason_code: str,
        last_completed_prefix: CampaignPrefix | str,
        active_prefix: CampaignPrefix | str,
        failure_stage: str,
        plan: ShardPlan,
        capture_root: str | os.PathLike[str],
        finalized_partial_shard_indexes: Iterable[int] = (),
        expected_partial_shard_inventory: PartialShardInventory | None = None,
        observed_tensor_leading_row_counts_by_shard: (
            Mapping[int, Mapping[str, int]] | None
        ) = None,
        sanitized_failure_details: Mapping[str, object] | None = None,
    ) -> TransactionResult:
        """Atomically publish capture failure, optional partials, and terminal."""

        role = _validate_partition_role(partition_role)
        campaign = _validate_identifier(campaign_id, label="campaign_id")
        attempt = _validate_tx_id(capture_attempt_id, label="capture_attempt_id")
        completed, active = _normalize_prefix_pair(
            last_completed_prefix, active_prefix
        )
        _validate_capture_prefix_for_partition(role, active)
        if completed is None:
            raise ProtocolSchemaError(
                "capture failure requires a completed predecessor prefix"
            )
        completed_prefix = completed.name
        current_prefix = active.name
        stage = _validate_identifier(failure_stage, label="failure_stage")
        if not isinstance(plan, ShardPlan) or plan.partition_role != role:
            raise ProtocolSchemaError("plan must match failure partition")
        root_identity_hash = _capture_root_identity_sha256(capture_root)
        disposition = classify_failure(reason_code)
        details = validate_sanitized_failure_details(sanitized_failure_details)
        _validate_failure_detail_context(
            details,
            partition_role=role,
            reason_code=reason_code,
            last_completed_prefix=completed_prefix,
            active_prefix=current_prefix,
            failure_stage=stage,
        )
        discovered_indexes = tuple(sorted(finalized_partial_shard_indexes))
        if len(discovered_indexes) != len(set(discovered_indexes)):
            raise ProtocolSchemaError("finalized shard indexes must be unique")
        actual_partial_indexes = _inventory_finalized_shard_indexes(
            plan,
            capture_root,
            require_complete=False,
        )
        if actual_partial_indexes != discovered_indexes:
            raise _failure_exception(
                "shard_plan_mismatch",
                "failure finalization must inventory every finalized shard",
            )
        observed_dimensions = observed_tensor_leading_row_counts_by_shard or {}
        if set(observed_dimensions) - set(discovered_indexes):
            raise ProtocolSchemaError(
                "observed leading rows include an undiscovered shard"
            )
        verified_records: list[VerifiedShardRecord] = []
        unverified_records: list[JsonValue] = []
        durable_hashes: dict[int, str] = {}
        capture_root_path = Path(capture_root)
        for index in discovered_indexes:
            try:
                record = verify_physical_shard_file(
                    plan,
                    index,
                    capture_root,
                    observed_tensor_leading_row_counts=(
                        observed_dimensions.get(index)
                    ),
                )
            except ProtocolViolation as error:
                if error.reason_code != reason_code:
                    raise _failure_exception(
                        error.reason_code,
                        "discovered invalid shard takes precedence over the "
                        "requested failure classification",
                    ) from error
                entry = plan.entries[index]
                candidate = capture_root_path.joinpath(
                    *PurePosixPath(entry.path).parts
                )
                try:
                    digest, byte_count, _, _ = (
                        self._inspect_external_artifact_file(
                            entry.path,
                            candidate,
                        )
                    )
                except ProtocolViolation:
                    try:
                        path_stat = os.lstat(candidate)
                    except OSError:
                        path_kind = "missing"
                        path_size: int | None = None
                    else:
                        if stat.S_ISLNK(path_stat.st_mode):
                            path_kind = "symlink"
                        elif stat.S_ISREG(path_stat.st_mode):
                            path_kind = "regular_unreadable"
                        elif stat.S_ISDIR(path_stat.st_mode):
                            path_kind = "directory"
                        else:
                            path_kind = "nonregular"
                        path_size = path_stat.st_size
                    unverified_records.append(
                        {
                            "artifact_byte_sha256": None,
                            "byte_count": path_size,
                            "path": entry.path,
                            "path_entry_type": path_kind,
                            "reason_code": error.reason_code,
                            "shard_index": index,
                        }
                    )
                else:
                    durable_hashes[index] = digest
                    unverified_records.append(
                        {
                            "artifact_byte_sha256": digest,
                            "byte_count": byte_count,
                            "path": entry.path,
                            "path_entry_type": "regular",
                            "reason_code": error.reason_code,
                            "shard_index": index,
                        }
                    )
            else:
                verified_records.append(record)
                durable_hashes[index] = record.shard_byte_sha256
        partial_shard_inventory = (
            PartialShardInventory(
                partition_role=role,
                shard_plan_sha256=plan.plan_sha256,
                shards=tuple(verified_records),
                verification_source="finalized_files_reopened",
                _construction_token=_VERIFIED_CONSTRUCTION_TOKEN,
            )
            if verified_records
            else None
        )
        if verified_records:
            if not isinstance(
                expected_partial_shard_inventory, PartialShardInventory
            ):
                raise ProtocolSchemaError(
                    "expected_partial_shard_inventory is required for partials"
                )
            if (
                expected_partial_shard_inventory.verification_source
                != "finalized_files_reopened"
                or expected_partial_shard_inventory.canonical_bytes
                != partial_shard_inventory.canonical_bytes
            ):
                raise _failure_exception(
                    "shard_hash_mismatch",
                    "failure finalization re-verification differs from prior partial evidence",
                )
        elif expected_partial_shard_inventory is not None:
            raise ProtocolSchemaError(
                "expected partial inventory must be null when no shard finalized"
            )
        partial_value: JsonValue = (
            None
            if partial_shard_inventory is None
            else partial_shard_inventory.as_dict()
        )
        identity_payload: dict[str, JsonValue] = {
            "capture_attempt_id": attempt,
            "campaign_id": campaign,
            "active_prefix": current_prefix,
            "capture_root_identity_sha256": root_identity_hash,
            "failure_reason_code": reason_code,
            "failure_stage": stage,
            "last_completed_prefix": completed_prefix,
            "partial_shard_inventory": partial_value,
            "partition_role": role,
            "sanitized_failure_details": details,
            "unverified_finalized_shards": unverified_records,
        }
        identity_hash = hashlib.sha256(
            _TERMINAL_FAILURE_DOMAIN + canonical_json_bytes(identity_payload)
        ).hexdigest()
        with self._immediate_transaction() as connection:
            panel = self._required_panel(connection, role)
            if panel["terminal_transaction_id"] is not None:
                if (
                    CaptureState(panel["state"]) is CaptureState.CONSUMED_FAILED
                    and panel["row_index_failure_transaction_id"] is None
                    and panel["terminal_identity_sha256"] == identity_hash
                ):
                    transaction_id = panel["terminal_transaction_id"]
                    existing = True
                else:
                    raise ProtocolStateError("panel already has a different terminal")
            else:
                existing = False
                if CaptureState(panel["state"]) is not CaptureState.CAPTURE_STARTED:
                    raise ProtocolStateError("capture failure requires capture_started")
                if panel["campaign_id"] != campaign:
                    raise _failure_exception(
                        "authorization_mismatch",
                        "failure receipt campaign differs from authorization",
                    )
                if panel["capture_attempt_id"] != attempt:
                    raise _failure_exception(
                        "capture_attempt_identity_mismatch",
                        "failure refers to a different capture attempt",
                    )
                if (
                    panel["capture_root_identity_sha256"]
                    != root_identity_hash
                ):
                    raise _failure_exception(
                        "capture_root_identity_mismatch",
                        "failure finalization uses a different capture root",
                    )
                if panel["shard_plan_sha256"] != plan.plan_sha256:
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "failure finalization uses a different shard plan",
                    )
                if (
                    partial_shard_inventory is not None
                    and partial_shard_inventory.shard_plan_sha256
                    != panel["shard_plan_sha256"]
                ):
                    raise _failure_exception(
                        "shard_plan_mismatch",
                        "partial inventory uses a different shard plan",
                    )
                self._validate_resume_shard_bindings(
                    connection,
                    partition_role=role,
                    capture_attempt_id=attempt,
                    shard_plan_sha256=plan.plan_sha256,
                    capture_root_identity_sha256=root_identity_hash,
                    records=(
                        ()
                        if partial_shard_inventory is None
                        else partial_shard_inventory.shards
                    ),
                )
                transaction_id = self._new_transaction_id()
                self._register_capture_shard_files(
                    connection,
                    plan=plan,
                    capture_root=capture_root,
                    shard_indexes=durable_hashes,
                    prefix=active,
                    transaction_id=transaction_id,
                    expected_hashes=durable_hashes,
                )
                common: dict[str, JsonValue] = {
                    "active_prefix": current_prefix,
                    "authorization_hash": panel["authorization_hash"],
                    "authorization_transaction_id": panel[
                        "authorization_transaction_id"
                    ],
                    "capture_attempt_id": attempt,
                    "campaign_id": campaign,
                    "capture_implementation_hash": panel[
                        "capture_implementation_hash"
                    ],
                    "capture_root_identity_sha256": root_identity_hash,
                    "capture_schema_hash": panel["capture_schema_hash"],
                    "capture_start_transaction_id": panel[
                        "capture_start_transaction_id"
                    ],
                    "committed_ordered_row_sequence_sha256": panel[
                        "committed_ordered_row_sequence_sha256"
                    ],
                    "committed_row_count": panel["committed_row_count"],
                    "committed_row_set_sha256": panel[
                        "committed_row_set_sha256"
                    ],
                    "failure_class": disposition.failure_class.value,
                    "failure_reason_code": reason_code,
                    "failure_stage": stage,
                    "last_completed_prefix": completed_prefix,
                    "partition_role": role,
                    "row_index_payload_sha256": panel[
                        "row_index_payload_sha256"
                    ],
                    "row_index_transaction_id": panel[
                        "row_index_transaction_id"
                    ],
                    "sanitized_failure_details": details,
                    "shard_plan_transaction_id": panel[
                        "shard_plan_transaction_id"
                    ],
                    "shard_plan_sha256": panel["shard_plan_sha256"],
                    "terminal_state": disposition.terminal_state.value,
                    "terminal_transaction_id": transaction_id,
                    "unverified_finalized_shards": unverified_records,
                }
                failure = self._artifact(
                    name=f"capture_failure_{role}.json",
                    kind="capture_failure",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                    },
                )
                transition = self._artifact(
                    name=f"{role}_capture_transition_consumed_failed.json",
                    kind="capture_transition_consumed_failed",
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "format_version": _FORMAT_VERSION,
                        "schema": LEDGER_SCHEMA,
                        "state_after": CaptureState.CONSUMED_FAILED.value,
                        "state_before": CaptureState.CAPTURE_STARTED.value,
                        "tensor_capture_started": True,
                    },
                )
                artifacts: list[StoredArtifact] = [failure, transition]
                if partial_shard_inventory is not None:
                    artifacts.append(
                        self._artifact(
                            name=f"partial_shard_inventory_{role}.json",
                            kind="partial_shard_inventory",
                            transaction_id=transaction_id,
                            payload={
                                **common,
                                **partial_shard_inventory.as_dict(),
                            },
                        )
                    )
                reached_inventory = self._preview_reached_inventory(
                    connection,
                    artifacts,
                    prefix=active,
                )
                if disposition.failure_class is FailureClass.RESOURCE_LIMIT:
                    primary_name = "resource_limit_failure.json"
                    primary_kind = "resource_limit_failure"
                else:
                    primary_name = "invalid_campaign_receipt.json"
                    primary_kind = "invalid_campaign_receipt"
                primary_receipt = self._artifact(
                    name=primary_name,
                    kind=primary_kind,
                    transaction_id=transaction_id,
                    payload={
                        **common,
                        "capture_implementation_hash": panel[
                            "capture_implementation_hash"
                        ],
                        "capture_schema_hash": panel["capture_schema_hash"],
                        "committed_ordered_row_sequence_sha256": panel[
                            "committed_ordered_row_sequence_sha256"
                        ],
                        "committed_row_count": panel["committed_row_count"],
                        "committed_row_set_sha256": panel[
                            "committed_row_set_sha256"
                        ],
                        "durably_reached_artifact_exclusions": [
                            "primary_receipt_self",
                            "transaction_commit_receipt",
                        ],
                        "durably_reached_artifacts": [
                            entry.as_dict() for entry in reached_inventory
                        ],
                        "format_version": _FORMAT_VERSION,
                        "primary_receipt_kind": primary_kind,
                        "schema": PRIMARY_FAILURE_RECEIPT_SCHEMA,
                    },
                )
                artifacts.append(primary_receipt)
                connection.execute(
                    """
                    UPDATE panels SET state = ?, terminal_transaction_id = ?,
                        terminal_identity_sha256 = ?, terminal_state = ?,
                        failure_reason_code = ?, revision = revision + 1
                    WHERE partition_role = ?
                    """,
                    (
                        CaptureState.CONSUMED_FAILED.value,
                        transaction_id,
                        identity_hash,
                        disposition.terminal_state.value,
                        reason_code,
                        role,
                    ),
                )
                self._commit_transaction_record(
                    connection,
                    transaction_id=transaction_id,
                    transaction_kind="capture_failure_finalization",
                    partition_role=role,
                    state_before=CaptureState.CAPTURE_STARTED,
                    state_after=CaptureState.CONSUMED_FAILED,
                    identity_sha256=identity_hash,
                    artifacts=tuple(artifacts),
                    receipt_name=f"{role}_capture_failure_transaction_commit.json",
                    campaign_prefix=_coerce_prefix(
                        current_prefix, label="active_prefix"
                    ),
                )
                self._set_global_terminal(
                    connection,
                    terminal_state=disposition.terminal_state.value,
                    terminal_identity_sha256=identity_hash,
                    terminal_transaction_id=transaction_id,
                    partition_role=role,
                )
        return self._load_transaction(transaction_id, idempotent=existing)

    @staticmethod
    def _required_panel(
        connection: sqlite3.Connection, partition_role: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM panels WHERE partition_role = ?", (partition_role,)
        ).fetchone()
        if row is None:
            raise ProtocolStateError(f"partition {partition_role!r} is unopened")
        return row

    @staticmethod
    def _validate_capture_terminal_identity(
        panel: sqlite3.Row,
        *,
        attempt: str,
        verification: CaptureVerification,
        capture_root_identity_sha256: str,
    ) -> None:
        if panel["capture_attempt_id"] != attempt:
            raise _failure_exception(
                "capture_attempt_identity_mismatch",
                "verification refers to a different capture attempt",
            )
        if (
            panel["capture_root_identity_sha256"]
            != capture_root_identity_sha256
        ):
            raise _failure_exception(
                "capture_root_identity_mismatch",
                "verification uses a different capture root",
            )
        expected = (
            panel["shard_plan_sha256"],
            panel["committed_row_set_sha256"],
            panel["committed_ordered_row_sequence_sha256"],
            panel["committed_row_count"],
        )
        observed = (
            verification.shard_plan_sha256,
            verification.committed_row_set_sha256,
            verification.committed_ordered_row_sequence_sha256,
            verification.row_count,
        )
        if expected != observed:
            raise _failure_exception(
                "shard_plan_mismatch",
                "capture verification differs from committed plan/index",
            )

    def snapshot(self, partition_role: str) -> PanelSnapshot | None:
        """Return the current panel state, or ``None`` when unopened."""

        role = _validate_partition_role(partition_role)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM panels WHERE partition_role = ?", (role,)
            ).fetchone()
        if row is None:
            return None
        return PanelSnapshot(
            partition_role=role,
            campaign_id=row["campaign_id"],
            state=CaptureState(row["state"]),
            authorization_hash=row["authorization_hash"],
            authorization_transaction_id=row["authorization_transaction_id"],
            expected_row_set_mode=AuthorizationMode(
                row["expected_row_set_mode"]
            ),
            row_derivation_protocol_hash=row["row_derivation_protocol_hash"],
            expected_row_set_sha256=row["expected_row_set_sha256"],
            expected_ordered_row_sequence_sha256=row[
                "expected_ordered_row_sequence_sha256"
            ],
            expected_row_count=row["expected_row_count"],
            expected_row_index_payload_sha256=row[
                "expected_row_index_payload_sha256"
            ],
            row_index_transaction_id=row["row_index_transaction_id"],
            row_index_payload_sha256=row["row_index_payload_sha256"],
            committed_row_set_sha256=row["committed_row_set_sha256"],
            committed_ordered_row_sequence_sha256=row[
                "committed_ordered_row_sequence_sha256"
            ],
            committed_row_count=row["committed_row_count"],
            row_index_failure_transaction_id=row[
                "row_index_failure_transaction_id"
            ],
            capture_start_transaction_id=row["capture_start_transaction_id"],
            capture_attempt_id=row["capture_attempt_id"],
            capture_root_identity_sha256=row[
                "capture_root_identity_sha256"
            ],
            shard_plan_transaction_id=row["shard_plan_transaction_id"],
            shard_plan_sha256=row["shard_plan_sha256"],
            capture_schema_hash=row["capture_schema_hash"],
            capture_implementation_hash=row["capture_implementation_hash"],
            resume_checkpoint_count=row["resume_checkpoint_count"],
            resume_checkpoint_sha256=row["resume_checkpoint_sha256"],
            terminal_transaction_id=row["terminal_transaction_id"],
            terminal_state=row["terminal_state"],
            failure_reason_code=row["failure_reason_code"],
        )

    def get_artifact(self, name: str) -> StoredArtifact | None:
        """Read and exact-byte verify one committed artifact."""

        _validate_relative_path(name, label="artifact name")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE name = ?", (name,)
            ).fetchone()
        return None if row is None else self._stored_artifact_from_row(row)

    def list_artifacts(self, partition_role: str) -> tuple[StoredArtifact, ...]:
        """Read and exact-byte verify all artifacts for one panel."""

        role = _validate_partition_role(partition_role)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE partition_role = ? ORDER BY name",
                (role,),
            ).fetchall()
        return tuple(self._stored_artifact_from_row(row) for row in rows)

    def artifact_closure(self, partition_role: str) -> ArtifactClosure:
        """Audit state-dependent success/failure artifact conditionality."""

        role = _validate_partition_role(partition_role)
        snapshot = self.snapshot(role)
        if snapshot is None:
            return ArtifactClosure(
                partition_role=role,
                state=CaptureState.UNOPENED,
                required_names=frozenset(),
                forbidden_names=frozenset(),
                present_names=frozenset(),
            )
        present = frozenset(artifact.name for artifact in self.list_artifacts(role))
        authorization = {
            f"{role}_capture_authorization.json",
            f"{role}_capture_transition_authorized.json",
            f"{role}_authorization_transaction_commit.json",
        }
        row_success = {
            f"row_index_manifest_{role}.json",
            f"{role}_row_index_committed.json",
            f"{role}_row_index_transaction_commit.json",
        }
        row_failure = {
            f"row_index_failure_{role}.json",
            f"{role}_row_index_failed.json",
            f"{role}_row_index_failure_transaction_commit.json",
        }
        capture_start = {
            f"{role}_tensor_capture_started.json",
            f"{role}_capture_start_transaction_commit.json",
        }
        shard_plan = {
            f"capture_shard_plan_{role}.json",
            f"{role}_shard_plan_transaction_commit.json",
        }
        complete = {
            f"capture_shard_manifest_{role}.json",
            f"{role}_capture_transition_consumed_success.json",
            f"{role}_capture_success_transaction_commit.json",
        }
        capture_failure = {
            f"capture_failure_{role}.json",
            f"{role}_capture_transition_consumed_failed.json",
            f"{role}_capture_failure_transaction_commit.json",
        }
        partial = {f"partial_shard_inventory_{role}.json"}
        row_failure_finalization = {
            f"{role}_row_index_failure_finalization_commit.json",
        }
        consumed_failed = {f"{role}_capture_transition_consumed_failed.json"}
        stage_failure_commit = {
            f"{role}_stage_failure_transaction_commit.json"
        }
        primary_failure = {
            "invalid_campaign_receipt.json",
            "resource_limit_failure.json",
        }
        required = set(authorization)
        forbidden: set[str] = set()
        state = snapshot.state
        if snapshot.row_index_transaction_id is not None:
            required.update(row_success)
        if snapshot.row_index_failure_transaction_id is not None:
            required.update(row_failure)
        if snapshot.capture_start_transaction_id is not None:
            required.update(capture_start)
        if snapshot.shard_plan_transaction_id is not None:
            required.update(shard_plan)
        if state is CaptureState.CONSUMED_SUCCESS:
            required.update(complete)
            forbidden.update(row_failure | capture_failure | partial)
            if present & stage_failure_commit:
                required.update(stage_failure_commit)
                stage_primary = present & primary_failure
                if not stage_primary:
                    required.add("invalid_campaign_receipt.json")
                elif len(stage_primary) > 1:
                    forbidden.update(stage_primary)
            else:
                forbidden.update(primary_failure | stage_failure_commit)
        elif state is CaptureState.CONSUMED_FAILED:
            if snapshot.row_index_failure_transaction_id is not None:
                required.update(row_failure_finalization | consumed_failed)
                forbidden.update(
                    row_success
                    | capture_start
                    | complete
                    | {f"capture_failure_{role}.json"}
                    | {f"{role}_capture_failure_transaction_commit.json"}
                    | partial
                    | shard_plan
                    | stage_failure_commit
                )
                if (
                    snapshot.terminal_state
                    == TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED.value
                ):
                    required.add("resource_limit_failure.json")
                    forbidden.add("invalid_campaign_receipt.json")
                else:
                    required.add("invalid_campaign_receipt.json")
                    forbidden.add("resource_limit_failure.json")
            else:
                required.update(capture_failure)
                forbidden.update(
                    complete | row_failure | row_failure_finalization | stage_failure_commit
                )
                if (
                    snapshot.terminal_state
                    == TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED.value
                ):
                    required.add("resource_limit_failure.json")
                    forbidden.add("invalid_campaign_receipt.json")
                else:
                    required.add("invalid_campaign_receipt.json")
                    forbidden.add("resource_limit_failure.json")
                if f"partial_shard_inventory_{role}.json" in present:
                    required.update(partial)
        elif state is CaptureState.ROW_INDEX_FAILED:
            forbidden.update(
                row_success
                | capture_start
                | complete
                | capture_failure
                | partial
                | shard_plan
                | primary_failure
                | row_failure_finalization
                | stage_failure_commit
            )
        elif state is CaptureState.CAPTURE_STARTED:
            forbidden.update(
                row_failure
                | complete
                | capture_failure
                | partial
                | primary_failure
                | row_failure_finalization
                | stage_failure_commit
            )
        elif state is CaptureState.ROW_INDEX_COMMITTED:
            forbidden.update(
                row_failure
                | capture_start
                | complete
                | capture_failure
                | partial
                | primary_failure
                | row_failure_finalization
                | stage_failure_commit
            )
        elif state is CaptureState.AUTHORIZED:
            forbidden.update(
                row_success
                | row_failure
                | capture_start
                | complete
                | capture_failure
                | partial
                | shard_plan
                | primary_failure
                | row_failure_finalization
                | stage_failure_commit
            )
        return ArtifactClosure(
            partition_role=role,
            state=state,
            required_names=frozenset(required),
            forbidden_names=frozenset(forbidden),
            present_names=present,
        )

    def verify_artifact_closure(self, partition_role: str) -> ArtifactClosure:
        """Return closure or raise an integrity failure when it is invalid."""

        closure = self.artifact_closure(partition_role)
        if not closure.valid:
            raise _failure_exception(
                "artifact_hash_mismatch",
                "artifact closure has missing or forbidden named artifacts",
            )
        return closure
