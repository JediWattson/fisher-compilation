"""Strict synthetic artifact contracts for compact-message completion P1--P12.

This module is deliberately model-free.  It freezes the *shape* of the
scientific artifacts that a future implementation must produce, but it does
not compute a scientific result and it cannot turn fixture values into
evidence.  Every payload built here is permanently marked
``synthetic_qualification_only`` and is ineligible for a scientific or guard
claim.

The durable authorization, row-index, capture, failure, and terminal-tail
artifacts remain owned by :mod:`fisher_compact_message_completion_protocol`.
They appear in the stage/dependency catalogue so that the scientific producer
cannot skip them, but this module refuses to build those ledger-owned files.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence


PROTOCOL_ID: Final = "fisher_compact_message_completion_probe_v2"
EVIDENCE_SCOPE: Final = "synthetic_qualification_only"
SCHEMA_DOMAIN: Final = "fisher-compact-message-completion"
CANDIDATE_HASH_DOMAIN: Final = (
    b"fisher-compact-message-completion/"
    b"development-candidate-manifest/v1"
)
FULL_PROMPT_LABEL: Final = (
    "source-authoritative, teacher-forced full-prompt shadow evaluation"
)
MODE_B_LIMITATION: Final = (
    "Statistical inference is limited to new semantic instances within the "
    "preregistered template families. It does not establish generalization "
    "to unseen template lineages."
)
REQUIRED_BASELINES: Final = ("J1", "J2N", "J2+", "J3")
MANDATORY_LEARNED_MODELS: Final = ("J0", "J2N", "J2+", "J3", "J4")
OPTIONAL_DIAGNOSTIC_SKIP_ORDER: Final = (
    "true_downstream_jvp_pullback",
    "p16_oracle_curve",
    "p64_oracle_curve",
    "matched_message_diagnostic",
    "interaction_donor_diagnostic",
    "extended_per_chart_head_operator_spectra",
    "whole_z_shuffle",
    "trained_full_rank_diagnostic",
    "dedicated_J2_consistency_training",
)
O1_SEAMS: Final = (
    "attention_q_head_sum",
    "attention_q_total",
    "attention_h_A",
    "mlp_q_block_sum",
    "mlp_q_total",
    "mlp_h_plus",
)
SPLIT_PURPOSES: Final = (
    "A-fit-train-vs-fit-monitor",
    "D-tune-vs-development-confirm",
    "O1-calibration-vs-validation",
)
CODEC_IDS: Final = ("C0", "C1", "C2", "C3")

# Exact pre-tune controls and fitted representation artifacts whose byte hashes
# are repeated in preregistration_root.json.  The three receipt/manifest hashes
# stored in dedicated scalar fields are direct dependencies too, but are kept
# out of this ordered body inventory.
PREREGISTRATION_BOUND_ARTIFACTS: Final = (
    "semantic_canonicalization_spec.json",
    "corpus_role_classification.json",
    "independence_protocol.json",
    "split_manifest.json",
    "o1_calibration_assignment.json",
    "model_identity.json",
    "fisher_carrier_identity.json",
    "runtime_identity.json",
    "execution_budget.json",
    "source_inventory.json",
    "row_derivation_protocol.json",
    "capture_invariants.json",
    "o1_tolerance_protocol.json",
    "o1_validation_results.json",
    "o1_complete_fit_results.json",
    "locator_protocol.json",
    "locator_candidates.json",
    "codec_protocol.json",
    "message_sketch_manifest_p32.json",
    "codec_candidate_manifests.json",
    "representation_numeric_constants.json",
    "full_rank_identity_smoke.json",
    "selection_representation_manifest.json",
    "ceiling_gate_protocol.json",
    "learning_protocol.json",
    "development_eligibility_protocol.json",
)
_PREREGISTRATION_SCALAR_ARTIFACTS: Final = (
    "capture_shard_manifest_fit.json",
    "reserve_retired_exposed.json",
    "local_development_registry_status.json",
)


class ArtifactSchemaError(ValueError):
    """A synthetic scientific artifact violates its frozen contract."""


class Stage(str, Enum):
    """Monotonic future-D2 campaign stages."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    P7 = "P7"
    P8 = "P8"
    P9 = "P9"
    P10 = "P10"
    P11 = "P11"
    P12 = "P12"


class ArtifactFamily(str, Enum):
    """Builder families exposed by this schema layer."""

    INDEPENDENCE = "independence"
    SPLIT = "split"
    IDENTITY = "identity"
    CAPTURE_INVARIANT = "capture_invariant"
    O1 = "o1"
    REPRESENTATION = "representation"
    O2 = "o2"
    M2 = "m2"
    ORACLE = "oracle"
    J_MODEL = "j_model"
    REFIT = "refit"
    BUNDLE = "bundle"
    DEVELOPMENT = "development"
    O7 = "o7"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class Rule:
    """Small exact JSON-schema subset used by the artifact catalogue."""

    kind: str
    nullable: bool = False
    literal: Any = None
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_length: int = 0
    item: "Rule | None" = None
    fields: Mapping[str, "Rule"] | None = None


def _string(
    *,
    literal: str | None = None,
    choices: Sequence[str] = (),
    nullable: bool = False,
    min_length: int = 1,
) -> Rule:
    return Rule(
        "string",
        nullable=nullable,
        literal=literal,
        choices=tuple(choices),
        min_length=min_length,
    )


def _boolean(*, literal: bool | None = None, nullable: bool = False) -> Rule:
    return Rule("boolean", nullable=nullable, literal=literal)


def _integer(
    *,
    minimum: int = 0,
    maximum: int | None = None,
    literal: int | None = None,
    nullable: bool = False,
) -> Rule:
    return Rule(
        "integer",
        nullable=nullable,
        literal=literal,
        minimum=minimum,
        maximum=maximum,
    )


def _number(
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    literal: float | None = None,
    nullable: bool = False,
) -> Rule:
    return Rule(
        "number",
        nullable=nullable,
        literal=literal,
        minimum=minimum,
        maximum=maximum,
    )


def _sha256(*, nullable: bool = False) -> Rule:
    return Rule("sha256", nullable=nullable)


def _array(item: Rule, *, min_length: int = 0) -> Rule:
    return Rule("array", item=item, min_length=min_length)


def _record(fields: Mapping[str, Rule], *, nullable: bool = False) -> Rule:
    return Rule(
        "object",
        nullable=nullable,
        fields=MappingProxyType(dict(fields)),
    )


SHA = _sha256()
NULLABLE_SHA = _sha256(nullable=True)
STRING = _string()
NULLABLE_STRING = _string(nullable=True)
NONNEG_INT = _integer()
POS_INT = _integer(minimum=1)
NONNEG_NUMBER = _number(minimum=0.0)
UNIT_INTERVAL = _number(minimum=0.0, maximum=1.0)


FILE_IDENTITY = _record(
    {
        "path": STRING,
        "artifact_byte_sha256": SHA,
        "byte_size": NONNEG_INT,
    }
)
TENSOR_IDENTITY = _record(
    {
        "name": STRING,
        "shape": _array(POS_INT, min_length=1),
        "dtype": STRING,
    }
)
SOURCE_FILE_IDENTITY = _record(
    {
        "path": STRING,
        "artifact_byte_sha256": SHA,
        "byte_size": NONNEG_INT,
        "file_mode": _string(),
        "tracked_status": _string(choices=("tracked", "modified", "untracked")),
    }
)
PARTITION_IDENTITY = _record(
    {
        "partition_role": _string(
            choices=("fit_train", "fit_monitor", "tune", "development_confirm")
        ),
        "analysis_unit_count": NONNEG_INT,
        "sequence_count": NONNEG_INT,
        "content_set_sha256": SHA,
        "analysis_unit_set_sha256": SHA,
        "partition_commitment": SHA,
    }
)
O1_TOLERANCE = _record(
    {
        "seam": _string(choices=O1_SEAMS),
        "reference_scale": _number(minimum=1e-8),
        "calibration_max_abs_error": NONNEG_NUMBER,
        "calibration_weighted_relative_l2": NONNEG_NUMBER,
        "cap_abs": _number(minimum=1e-6),
        "cap_rel": _number(literal=1e-4),
        "tau_abs": _number(minimum=1e-12),
        "tau_rel": _number(minimum=1e-12),
        "calibration_passed": _boolean(),
    }
)
O1_RESULT = _record(
    {
        "seam": _string(choices=O1_SEAMS),
        "observed_max_abs_error": NONNEG_NUMBER,
        "observed_weighted_relative_l2": NONNEG_NUMBER,
        "tau_abs": _number(minimum=1e-12),
        "tau_rel": _number(minimum=1e-12),
        "passed": _boolean(),
    }
)
CAPTURE_SEAM_OBSERVATION = _record(
    {
        "seam": _string(choices=O1_SEAMS),
        "observed_max_abs_error": NONNEG_NUMBER,
        "observed_weighted_relative_l2": NONNEG_NUMBER,
        "finite": _boolean(),
        "shape_contract_passed": _boolean(),
    }
)
GATE_RESULT = _record(
    {
        "gate_id": _string(
            choices=(
                "absolute_global_delta_nll_per_token",
                "global_kl_per_token",
                "global_top1_agreement",
                "p90_absolute_prompt_delta_nll_per_token",
                "p10_prompt_top1_agreement",
            )
        ),
        "observed": NONNEG_NUMBER,
        "threshold": NONNEG_NUMBER,
        "comparison": _string(choices=("<=", ">=")),
        "passed": _boolean(),
    }
)
PANEL_GATES = _record(
    {
        "partition_role": _string(choices=("fit_monitor", "tune", "development_confirm")),
        "gate_results": _array(GATE_RESULT, min_length=5),
        "all_five_passed": _boolean(),
        "evaluation_label": _string(literal=FULL_PROMPT_LABEL),
    }
)
CODEC_CANDIDATE = _record(
    {
        "codec_id": _string(choices=CODEC_IDS),
        "q_rank": POS_INT,
        "v_rank": POS_INT,
        "lambda_ratio": _number(minimum=0.0),
    }
)
CODEC_NUMERIC_RESULT = _record(
    {
        "codec_id": _string(choices=CODEC_IDS),
        "locator_candidate_id": STRING,
        "q_rank": _integer(minimum=1),
        "v_rank": _integer(minimum=1),
        "lambda_ratio": _number(minimum=0.0),
        "orthogonality_error": NONNEG_NUMBER,
        "idempotence_error": NONNEG_NUMBER,
        "encoder_construction_error": NONNEG_NUMBER,
        "operator_reference_norm": _number(minimum=0.0, nullable=True),
        "tau_R": _number(minimum=0.0, nullable=True),
        "status": _string(
            choices=(
                "eligible",
                "invalid_numerical_identity",
                "invalid_no_positive_operator_reference",
                "invalid_support",
            )
        ),
        "tensor_manifest_sha256": SHA,
    }
)
MODEL_RESULT = _record(
    {
        "model_id": _string(choices=("J0", "J1", "J2N", "J2", "J2+", "J3", "J4")),
        "mandatory": _boolean(),
        "representation_sha256": SHA,
        "architecture_sha256": SHA,
        "selected_seed": _integer(minimum=0, nullable=True),
        "selected_optimizer_step": _integer(minimum=0, nullable=True),
        "coefficient_loss": _number(minimum=0.0, nullable=True),
        "L_bar_down": _number(minimum=0.0, nullable=True),
        "status": _string(choices=("analytic", "fit", "evaluated", "skipped_optional")),
    }
)
COMPARISON_RESULT = _record(
    {
        "baseline": _string(choices=REQUIRED_BASELINES),
        "L_bar_down_baseline": NONNEG_NUMBER,
        "L_bar_down_J4": NONNEG_NUMBER,
        "rho": _number(nullable=True),
        "rho_status": _string(
            choices=(
                "regular",
                "both_at_numerical_floor",
                "candidate_worse_than_near_zero_baseline",
            )
        ),
        "rho_point_screen_passed": _boolean(),
        "paired_ci_lower": _number(),
        "paired_ci_upper": _number(),
        "paired_positive_passed": _boolean(),
        "valid_rho_replicate_fraction": UNIT_INTERVAL,
        "comparison_passed": _boolean(),
    }
)


@dataclass(frozen=True)
class ArtifactSchema:
    """One exact artifact contract."""

    name: str
    stage: Stage
    family: ArtifactFamily
    schema_id: str
    dependencies: tuple[str, ...]
    body: Rule
    binary: bool = False


def _schema_id(name: str) -> str:
    stem = name.removesuffix(".json").replace("_", "-")
    return f"{SCHEMA_DOMAIN}/{stem}/synthetic-v1"


# This is the exact successful-development prefix order.  Mechanical entries
# are included for dependency planning but remain ledger-owned.
STAGE_ARTIFACT_ORDER: Final[Mapping[Stage, tuple[str, ...]]] = MappingProxyType(
    {
        Stage.P1: (
            "independence_protocol.json",
            "split_manifest.json",
            "o1_calibration_assignment.json",
        ),
        Stage.P2: (
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "execution_budget.json",
            "source_inventory.json",
            "row_derivation_protocol.json",
        ),
        Stage.P3: (
            "fit_capture_authorization.json",
            "fit_capture_transition_authorized.json",
            "row_index_manifest_fit.json",
            "fit_row_index_committed.json",
            "capture_shard_plan_fit.json",
            "fit_tensor_capture_started.json",
            "fit_capture_transition_consumed_success.json",
            "capture_shard_manifest_fit.json",
            "capture_invariants.json",
        ),
        Stage.P4: (
            "o1_tolerance_protocol.json",
            "o1_validation_results.json",
            "o1_complete_fit_results.json",
            "locator_protocol.json",
            "locator_candidates.json",
            "codec_protocol.json",
            "message_sketch_manifest_p32.json",
            "codec_candidate_manifests.json",
            "representation_numeric_constants.json",
            "full_rank_identity_smoke.json",
            "selection_representation_manifest.json",
            "ceiling_gate_protocol.json",
            "learning_protocol.json",
            "development_eligibility_protocol.json",
            "preregistration_root.json",
            "local_development_freshness_audit_receipt.json",
            "campaign_root.json",
        ),
        Stage.P5: (
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
        ),
        Stage.P6: (
            "selected_codec.json",
            "frozen_representation_manifest.json",
            "m2q_results.json",
            "m2v_results.json",
            "compact_ambient_oracle.json",
            "compact_chart_oracle.json",
        ),
        Stage.P7: (
            "j_model_fit_results.json",
            "j_model_tune_results.json",
        ),
        Stage.P8: (
            "all_fit_weight_refit.json",
            "nonselecting_tune_replay.json",
        ),
        Stage.P9: (
            "frozen_B_star.json",
            "direct_full_rank_identity_null.json",
            "candidate_bundle.safetensors",
            "comparator_bundle.safetensors",
            "candidate_bundle_metadata.json",
            "development_confirm_capture_authorization.json",
            "development_confirm_capture_transition_authorized.json",
        ),
        Stage.P10: (
            "row_index_manifest_development_confirm.json",
            "development_confirm_row_index_committed.json",
            "capture_shard_plan_development_confirm.json",
            "development_confirm_tensor_capture_started.json",
            "development_confirm_capture_transition_consumed_success.json",
            "capture_shard_manifest_development_confirm.json",
        ),
        Stage.P11: (
            "development_confirm_results.json",
            "interaction_mean_ablation.json",
            "o7_attention_output_completion_results.json",
            "development_eligibility_result.json",
            "development_power_assessment.json",
        ),
        Stage.P12: ("development_candidate_manifest.json",),
    }
)


LEDGER_OWNED_ARTIFACTS: Final = frozenset(
    {
        name
        for names in STAGE_ARTIFACT_ORDER.values()
        for name in names
        if any(
            token in name
            for token in (
                "capture_authorization",
                "capture_transition",
                "row_index_manifest_",
                "row_index_committed",
                "capture_shard_plan_",
                "tensor_capture_started",
                "capture_shard_manifest_",
            )
        )
    }
    | {"terminal_state.json", "final_report.md", "final_experiment_manifest.json"}
)


CONDITIONAL_CAPTURE_ARTIFACTS: Final[Mapping[Stage, tuple[str, ...]]] = (
    MappingProxyType(
        {
            Stage.P3: (
                "row_index_failure_fit.json",
                "fit_row_index_failed.json",
                "fit_capture_transition_consumed_failed.json",
                "capture_failure_fit.json",
                "partial_shard_inventory_fit.json",
            ),
            Stage.P5: (
                "row_index_failure_tune.json",
                "tune_row_index_failed.json",
                "tune_capture_transition_consumed_failed.json",
                "capture_failure_tune.json",
                "partial_shard_inventory_tune.json",
            ),
            Stage.P10: (
                "row_index_failure_development_confirm.json",
                "development_confirm_row_index_failed.json",
                "development_confirm_capture_transition_consumed_failed.json",
                "capture_failure_development_confirm.json",
                "partial_shard_inventory_development_confirm.json",
            ),
        }
    )
)


def _deps() -> dict[str, tuple[str, ...]]:
    """Return the frozen acyclic success-path dependency graph."""

    return {
        "independence_protocol.json": ("corpus_role_classification.json",),
        "split_manifest.json": ("independence_protocol.json",),
        "o1_calibration_assignment.json": ("split_manifest.json",),
        "model_identity.json": ("split_manifest.json",),
        "fisher_carrier_identity.json": ("model_identity.json",),
        "runtime_identity.json": ("model_identity.json",),
        "execution_budget.json": ("runtime_identity.json",),
        "source_inventory.json": ("execution_budget.json",),
        "row_derivation_protocol.json": (
            "model_identity.json",
            "source_inventory.json",
        ),
        "fit_capture_authorization.json": (
            "split_manifest.json",
            "o1_calibration_assignment.json",
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "execution_budget.json",
            "source_inventory.json",
            "row_derivation_protocol.json",
        ),
        "fit_capture_transition_authorized.json": (
            "fit_capture_authorization.json",
        ),
        "row_index_manifest_fit.json": ("fit_capture_transition_authorized.json",),
        "fit_row_index_committed.json": ("row_index_manifest_fit.json",),
        "capture_shard_plan_fit.json": ("fit_row_index_committed.json",),
        "fit_tensor_capture_started.json": ("capture_shard_plan_fit.json",),
        "fit_capture_transition_consumed_success.json": (
            "fit_tensor_capture_started.json",
        ),
        "capture_shard_manifest_fit.json": ("fit_tensor_capture_started.json",),
        "capture_invariants.json": (
            "fit_capture_transition_consumed_success.json",
            "capture_shard_manifest_fit.json",
        ),
        "o1_tolerance_protocol.json": (
            "o1_calibration_assignment.json",
            "capture_invariants.json",
        ),
        "o1_validation_results.json": ("o1_tolerance_protocol.json",),
        "o1_complete_fit_results.json": ("o1_validation_results.json",),
        "locator_protocol.json": (
            "capture_invariants.json",
            "split_manifest.json",
        ),
        "locator_candidates.json": ("locator_protocol.json",),
        "codec_protocol.json": (
            "locator_candidates.json",
            "fisher_carrier_identity.json",
        ),
        "message_sketch_manifest_p32.json": (
            "capture_shard_manifest_fit.json",
            "locator_candidates.json",
        ),
        "codec_candidate_manifests.json": (
            "codec_protocol.json",
            "o1_tolerance_protocol.json",
            "message_sketch_manifest_p32.json",
        ),
        "representation_numeric_constants.json": (
            "o1_tolerance_protocol.json",
            "codec_candidate_manifests.json",
            "message_sketch_manifest_p32.json",
        ),
        "full_rank_identity_smoke.json": (
            "representation_numeric_constants.json",
            "codec_candidate_manifests.json",
        ),
        "selection_representation_manifest.json": (
            "locator_candidates.json",
            "codec_candidate_manifests.json",
            "message_sketch_manifest_p32.json",
            "representation_numeric_constants.json",
            "full_rank_identity_smoke.json",
        ),
        "ceiling_gate_protocol.json": (
            "split_manifest.json",
            "row_derivation_protocol.json",
        ),
        "learning_protocol.json": (
            "selection_representation_manifest.json",
            "representation_numeric_constants.json",
        ),
        "development_eligibility_protocol.json": (
            "learning_protocol.json",
            "ceiling_gate_protocol.json",
        ),
        "preregistration_root.json": (
            PREREGISTRATION_BOUND_ARTIFACTS
            + _PREREGISTRATION_SCALAR_ARTIFACTS
        ),
        "local_development_freshness_audit_receipt.json": (
            "preregistration_root.json",
        ),
        "campaign_root.json": (
            "preregistration_root.json",
            "local_development_freshness_audit_receipt.json",
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "source_inventory.json",
            "capture_shard_manifest_fit.json",
            "local_development_registry_status.json",
        ),
        "tune_capture_authorization.json": (
            "campaign_root.json",
            "selection_representation_manifest.json",
            "codec_protocol.json",
            "ceiling_gate_protocol.json",
        ),
        "tune_capture_transition_authorized.json": (
            "tune_capture_authorization.json",
        ),
        "row_index_manifest_tune.json": ("tune_capture_transition_authorized.json",),
        "tune_row_index_committed.json": ("row_index_manifest_tune.json",),
        "capture_shard_plan_tune.json": ("tune_row_index_committed.json",),
        "tune_tensor_capture_started.json": ("capture_shard_plan_tune.json",),
        "tune_capture_transition_consumed_success.json": (
            "tune_tensor_capture_started.json",
        ),
        "capture_shard_manifest_tune.json": ("tune_tensor_capture_started.json",),
        "o2q_results.json": (
            "capture_shard_manifest_tune.json",
            "selection_representation_manifest.json",
            "ceiling_gate_protocol.json",
        ),
        "o2v_results.json": (
            "capture_shard_manifest_tune.json",
            "selection_representation_manifest.json",
            "ceiling_gate_protocol.json",
        ),
        "selected_codec.json": ("o2q_results.json", "o2v_results.json"),
        "frozen_representation_manifest.json": (
            "selected_codec.json",
            "selection_representation_manifest.json",
        ),
        "m2q_results.json": (
            "frozen_representation_manifest.json",
            "capture_shard_manifest_tune.json",
            "ceiling_gate_protocol.json",
        ),
        "m2v_results.json": (
            "frozen_representation_manifest.json",
            "capture_shard_manifest_tune.json",
            "ceiling_gate_protocol.json",
        ),
        "compact_ambient_oracle.json": (
            "frozen_representation_manifest.json",
            "capture_shard_manifest_tune.json",
            "ceiling_gate_protocol.json",
        ),
        "compact_chart_oracle.json": (
            "frozen_representation_manifest.json",
            "capture_shard_manifest_tune.json",
            "ceiling_gate_protocol.json",
        ),
        "j_model_fit_results.json": (
            "frozen_representation_manifest.json",
            "learning_protocol.json",
            "capture_shard_manifest_fit.json",
        ),
        "j_model_tune_results.json": (
            "j_model_fit_results.json",
            "capture_shard_manifest_tune.json",
        ),
        "all_fit_weight_refit.json": (
            "j_model_fit_results.json",
            "j_model_tune_results.json",
            "frozen_representation_manifest.json",
        ),
        "nonselecting_tune_replay.json": (
            "all_fit_weight_refit.json",
            "j_model_tune_results.json",
            "capture_shard_manifest_tune.json",
        ),
        "frozen_B_star.json": ("nonselecting_tune_replay.json",),
        "direct_full_rank_identity_null.json": (
            "all_fit_weight_refit.json",
            "frozen_representation_manifest.json",
            "full_rank_identity_smoke.json",
        ),
        "candidate_bundle.safetensors": (
            "all_fit_weight_refit.json",
            "frozen_B_star.json",
            "direct_full_rank_identity_null.json",
        ),
        "comparator_bundle.safetensors": (
            "all_fit_weight_refit.json",
            "frozen_B_star.json",
        ),
        "candidate_bundle_metadata.json": (
            "candidate_bundle.safetensors",
            "comparator_bundle.safetensors",
            "frozen_representation_manifest.json",
            "frozen_B_star.json",
            "source_inventory.json",
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "row_derivation_protocol.json",
            "campaign_root.json",
            "ceiling_gate_protocol.json",
            "learning_protocol.json",
            "development_eligibility_protocol.json",
            "full_rank_identity_smoke.json",
        ),
        "development_confirm_capture_authorization.json": (
            "candidate_bundle_metadata.json",
            "nonselecting_tune_replay.json",
            "development_eligibility_protocol.json",
            "campaign_root.json",
        ),
        "development_confirm_capture_transition_authorized.json": (
            "development_confirm_capture_authorization.json",
        ),
        "row_index_manifest_development_confirm.json": (
            "development_confirm_capture_transition_authorized.json",
        ),
        "development_confirm_row_index_committed.json": (
            "row_index_manifest_development_confirm.json",
        ),
        "capture_shard_plan_development_confirm.json": (
            "development_confirm_row_index_committed.json",
        ),
        "development_confirm_tensor_capture_started.json": (
            "capture_shard_plan_development_confirm.json",
        ),
        "development_confirm_capture_transition_consumed_success.json": (
            "development_confirm_tensor_capture_started.json",
        ),
        "capture_shard_manifest_development_confirm.json": (
            "development_confirm_tensor_capture_started.json",
        ),
        "development_confirm_results.json": (
            "capture_shard_manifest_development_confirm.json",
            "candidate_bundle_metadata.json",
            "frozen_B_star.json",
            "selected_codec.json",
            "representation_numeric_constants.json",
            "codec_candidate_manifests.json",
        ),
        "interaction_mean_ablation.json": (
            "development_confirm_results.json",
            "frozen_B_star.json",
        ),
        "o7_attention_output_completion_results.json": (
            "development_confirm_results.json",
            "interaction_mean_ablation.json",
            "ceiling_gate_protocol.json",
        ),
        "development_eligibility_result.json": (
            "development_confirm_results.json",
            "interaction_mean_ablation.json",
            "o7_attention_output_completion_results.json",
            "development_eligibility_protocol.json",
        ),
        "development_power_assessment.json": (
            "development_confirm_results.json",
            "development_eligibility_result.json",
        ),
        "development_candidate_manifest.json": (
            "source_inventory.json",
            "model_identity.json",
            "fisher_carrier_identity.json",
            "runtime_identity.json",
            "execution_budget.json",
            "split_manifest.json",
            "capture_shard_manifest_fit.json",
            "capture_shard_manifest_tune.json",
            "capture_shard_manifest_development_confirm.json",
            "local_development_freshness_audit_receipt.json",
            "campaign_root.json",
            "frozen_representation_manifest.json",
            "selected_codec.json",
            "candidate_bundle.safetensors",
            "comparator_bundle.safetensors",
            "candidate_bundle_metadata.json",
            "frozen_B_star.json",
            "development_confirm_results.json",
            "development_eligibility_result.json",
            "development_power_assessment.json",
            "o7_attention_output_completion_results.json",
        ),
    }


ARTIFACT_DEPENDENCIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    _deps()
)


# Some artifact bodies repeat the byte hash of an envelope dependency under a
# domain-specific field name. Keep this list deliberately conservative: it
# contains only scalar body fields whose value is unambiguously the exact byte
# SHA-256 of a direct dependency. Semantic identities and transitive artifact
# references are intentionally not inferred here.
BODY_DEPENDENCY_HASH_BINDINGS: Final[Mapping[str, Mapping[str, str]]] = (
    MappingProxyType(
        {
            "capture_invariants.json": MappingProxyType(
                {
                    "fit_capture_manifest_sha256": (
                        "capture_shard_manifest_fit.json"
                    )
                }
            ),
            "o1_tolerance_protocol.json": MappingProxyType(
                {
                    "calibration_assignment_sha256": (
                        "o1_calibration_assignment.json"
                    )
                }
            ),
            "o1_validation_results.json": MappingProxyType(
                {"tolerance_protocol_sha256": "o1_tolerance_protocol.json"}
            ),
            "locator_candidates.json": MappingProxyType(
                {"locator_protocol_sha256": "locator_protocol.json"}
            ),
            "codec_protocol.json": MappingProxyType(
                {"carrier_identity_sha256": "fisher_carrier_identity.json"}
            ),
            "codec_candidate_manifests.json": MappingProxyType(
                {"codec_protocol_sha256": "codec_protocol.json"}
            ),
            "preregistration_root.json": MappingProxyType(
                {
                    "fit_capture_manifest_sha256": (
                        "capture_shard_manifest_fit.json"
                    ),
                    "reserve_retirement_receipt_sha256": (
                        "reserve_retired_exposed.json"
                    ),
                    "local_registry_snapshot_sha256": (
                        "local_development_registry_status.json"
                    ),
                }
            ),
            "local_development_freshness_audit_receipt.json": MappingProxyType(
                {"preregistration_root_sha256": "preregistration_root.json"}
            ),
            "campaign_root.json": MappingProxyType(
                {
                    "preregistration_root_sha256": "preregistration_root.json",
                    "freshness_receipt_sha256": (
                        "local_development_freshness_audit_receipt.json"
                    ),
                    "model_identity_sha256": "model_identity.json",
                    "carrier_identity_sha256": "fisher_carrier_identity.json",
                    "runtime_identity_sha256": "runtime_identity.json",
                    "source_inventory_sha256": "source_inventory.json",
                    "fit_capture_manifest_sha256": (
                        "capture_shard_manifest_fit.json"
                    ),
                    "local_registry_snapshot_sha256": (
                        "local_development_registry_status.json"
                    ),
                }
            ),
            "selected_codec.json": MappingProxyType(
                {
                    "o2q_result_sha256": "o2q_results.json",
                    "o2v_result_sha256": "o2v_results.json",
                }
            ),
            "frozen_B_star.json": MappingProxyType(
                {"tune_replay_sha256": "nonselecting_tune_replay.json"}
            ),
            "candidate_bundle_metadata.json": MappingProxyType(
                {
                    "candidate_bundle_byte_sha256": "candidate_bundle.safetensors",
                    "comparator_bundle_byte_sha256": (
                        "comparator_bundle.safetensors"
                    ),
                    "source_inventory_sha256": "source_inventory.json",
                    "model_identity_sha256": "model_identity.json",
                    "carrier_identity_sha256": "fisher_carrier_identity.json",
                    "runtime_identity_sha256": "runtime_identity.json",
                    "row_derivation_protocol_sha256": (
                        "row_derivation_protocol.json"
                    ),
                    "campaign_root_sha256": "campaign_root.json",
                    "ceiling_gate_protocol_sha256": "ceiling_gate_protocol.json",
                    "learning_protocol_sha256": "learning_protocol.json",
                    "development_eligibility_protocol_sha256": (
                        "development_eligibility_protocol.json"
                    ),
                    "full_rank_identity_smoke_sha256": (
                        "full_rank_identity_smoke.json"
                    ),
                }
            ),
            "development_candidate_manifest.json": MappingProxyType(
                {
                    "source_inventory_sha256": "source_inventory.json",
                    "model_identity_sha256": "model_identity.json",
                    "carrier_identity_sha256": "fisher_carrier_identity.json",
                    "runtime_identity_sha256": "runtime_identity.json",
                    "execution_budget_sha256": "execution_budget.json",
                    "split_manifest_sha256": "split_manifest.json",
                    "freshness_receipt_sha256": (
                        "local_development_freshness_audit_receipt.json"
                    ),
                    "campaign_root_sha256": "campaign_root.json",
                    "frozen_representation_sha256": (
                        "frozen_representation_manifest.json"
                    ),
                    "selected_codec_sha256": "selected_codec.json",
                    "candidate_bundle_byte_sha256": (
                        "candidate_bundle.safetensors"
                    ),
                    "comparator_bundle_byte_sha256": (
                        "comparator_bundle.safetensors"
                    ),
                    "candidate_bundle_metadata_sha256": (
                        "candidate_bundle_metadata.json"
                    ),
                    "frozen_B_star_sha256": "frozen_B_star.json",
                    "development_confirm_result_sha256": (
                        "development_confirm_results.json"
                    ),
                    "development_eligibility_result_sha256": (
                        "development_eligibility_result.json"
                    ),
                    "o7_result_sha256": (
                        "o7_attention_output_completion_results.json"
                    ),
                }
            ),
        }
    )
)


def _body_schemas() -> dict[str, tuple[ArtifactFamily, Rule]]:
    """Define exact domain bodies for every externally produced JSON file."""

    model_file_list = _array(FILE_IDENTITY, min_length=1)
    sha_map_record = _record({"name": STRING, "artifact_sha256": SHA})
    representation_binding = _record(
        {"object_name": STRING, "artifact_sha256": SHA, "immutable": _boolean(literal=True)}
    )
    chart_support = _record(
        {
            "chart_id": NONNEG_INT,
            "analysis_unit_count": NONNEG_INT,
            "sequence_count": NONNEG_INT,
            "row_count": NONNEG_INT,
            "supported": _boolean(),
        }
    )
    trained_operator_pair = _record(
        {
            "chart_id": NONNEG_INT,
            "head_id": NONNEG_INT,
            "analysis_unit_count": NONNEG_INT,
            "status": _string(
                choices=("supported", "insufficient_pair_support")
            ),
            "pair_median_ambient_operator_norm": _number(
                minimum=0.0, nullable=True
            ),
            "pair_median_H4_surrogate_operator_norm": _number(
                minimum=0.0, nullable=True
            ),
            "pair_median_ambient_stable_rank": _number(
                minimum=0.0, nullable=True
            ),
        }
    )
    trained_operator_aggregate = _record(
        {
            "panel": _string(literal="development_confirm"),
            "model_id": _string(literal="J4"),
            "codec_id": _string(choices=CODEC_IDS),
            "tau_R": NONNEG_NUMBER,
            "selected_codec_artifact_sha256": SHA,
            "representation_numeric_constants_sha256": SHA,
            "codec_candidate_manifests_sha256": SHA,
            "aggregate_median_ambient_operator_norm": NONNEG_NUMBER,
            "aggregate_median_H4_surrogate_operator_norm": NONNEG_NUMBER,
            "aggregate_median_ambient_stable_rank": NONNEG_NUMBER,
            "supported_chart_head_pair_count": POS_INT,
            "excluded_chart_head_pair_count": NONNEG_INT,
            "zero_ambient_pair_count": NONNEG_INT,
            "positive_ambient_pair_count": NONNEG_INT,
            "analysis_unit_support_by_pair": _array(
                trained_operator_pair, min_length=1
            ),
        }
    )
    full_rank_head_tolerance = _record(
        {
            "head_id": NONNEG_INT,
            "fit_train_row_count": POS_INT,
            "fit_train_analysis_unit_count": POS_INT,
            "message_reference_scale": _number(minimum=1e-8),
            "q_reference_scale": _number(minimum=1e-8),
            "observed_max_error_m": NONNEG_NUMBER,
            "observed_max_error_q": NONNEG_NUMBER,
            "cap_m": _number(minimum=1e-12),
            "cap_q": _number(minimum=1e-12),
            "tau_full_m": _number(minimum=1e-12),
            "tau_full_q": _number(minimum=1e-12),
            "passed": _boolean(),
        }
    )
    full_rank_coefficient_tolerance = _record(
        {
            "codec_id": _string(choices=CODEC_IDS),
            "chart_id": NONNEG_INT,
            "head_id": NONNEG_INT,
            "fit_train_row_count": POS_INT,
            "fit_train_analysis_unit_count": POS_INT,
            "coefficient_reference_scale": _number(minimum=1e-8),
            "observed_max_error_c": NONNEG_NUMBER,
            "cap_c": _number(minimum=1e-12),
            "tau_full_c": _number(minimum=1e-12),
            "tau_full_advantage": _number(minimum=0.0),
            "passed": _boolean(),
        }
    )
    full_rank_validation_record = _record(
        {
            "codec_id": _string(choices=CODEC_IDS),
            "chart_id": NONNEG_INT,
            "head_id": NONNEG_INT,
            "assigned_row_count": NONNEG_INT,
            "distinct_analysis_unit_count": NONNEG_INT,
            "observed_max_error_m": _number(minimum=0.0, nullable=True),
            "observed_max_error_q": _number(minimum=0.0, nullable=True),
            "observed_max_error_c": _number(minimum=0.0, nullable=True),
            "coefficient_completion_mse": _number(minimum=0.0, nullable=True),
            "tau_full_m": _number(minimum=1e-12),
            "tau_full_q": _number(minimum=1e-12),
            "tau_full_c": _number(minimum=1e-12),
            "tau_full_advantage": _number(minimum=0.0),
            "validation_status": _string(
                choices=("passed", "failed", "invalid_zero_validation_support")
            ),
            "zero_row_reason": _string(nullable=True),
        }
    )
    oracle_local_metrics = _record(
        {
            "q_total_max_abs_error": NONNEG_NUMBER,
            "q_total_weighted_relative_l2": NONNEG_NUMBER,
            "h_A_max_abs_error": NONNEG_NUMBER,
            "h_A_weighted_relative_l2": NONNEG_NUMBER,
            "h_plus_max_abs_error": NONNEG_NUMBER,
            "h_plus_weighted_relative_l2": NONNEG_NUMBER,
        }
    )
    oracle_panel = _record(
        {
            "partition_role": _string(choices=("fit_monitor", "tune")),
            "local": oracle_local_metrics,
            "full_prompt_shadow": PANEL_GATES,
        }
    )
    schemas: dict[str, tuple[ArtifactFamily, Rule]] = {
        "independence_protocol.json": (
            ArtifactFamily.INDEPENDENCE,
            _record(
                {
                    "access_policy": _string(literal="strict_metadata_only_v1"),
                    "selected_mode": _string(
                        choices=("template_lineage", "semantic_instance")
                    ),
                    "analysis_unit_field": _string(
                        choices=("template_lineage_id", "semantic_instance_id")
                    ),
                    "mode_a_eligible": _boolean(),
                    "mode_b_eligible": _boolean(),
                    "fit_analysis_unit_count": POS_INT,
                    "tune_analysis_unit_count": POS_INT,
                    "development_confirm_analysis_unit_count": POS_INT,
                    "inference_limitation": NULLABLE_STRING,
                    "selection_reason": STRING,
                }
            ),
        ),
        "split_manifest.json": (
            ArtifactFamily.SPLIT,
            _record(
                {
                    "independence_mode": _string(
                        choices=("template_lineage", "semantic_instance")
                    ),
                    "split_key_domain": _string(literal="fisher-cmc/split-key/v1"),
                    "purpose_identifiers": _array(
                        _string(choices=SPLIT_PURPOSES), min_length=3
                    ),
                    "allocation_rule": _string(literal="integer_hamilton_v1"),
                    "tie_order": _string(
                        literal="remainder_desc_raw_tie_digest_asc_label_bytes_asc"
                    ),
                    "partitions": _array(PARTITION_IDENTITY, min_length=4),
                    "fit_train_count": _integer(literal=38),
                    "fit_monitor_count": _integer(literal=10),
                    "tune_count": _integer(literal=24),
                    "development_confirm_count": _integer(literal=24),
                    "content_disjoint": _boolean(literal=True),
                    "analysis_unit_disjoint": _boolean(literal=True),
                }
            ),
        ),
        "o1_calibration_assignment.json": (
            ArtifactFamily.SPLIT,
            _record(
                {
                    "purpose": _string(literal="O1-calibration-vs-validation"),
                    "allocation_rule": _string(literal="integer_hamilton_v1"),
                    "calibration_analysis_unit_count": _integer(literal=12),
                    "validation_analysis_unit_count": _integer(literal=36),
                    "calibration_unit_set_sha256": SHA,
                    "validation_unit_set_sha256": SHA,
                    "disjoint": _boolean(literal=True),
                    "stratum_scheme": _string(
                        choices=("family", "family_x_template_id", "global")
                    ),
                }
            ),
        ),
        "model_identity.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "repository_id": STRING,
                    "immutable_revision": STRING,
                    "weight_files": model_file_list,
                    "config_file": FILE_IDENTITY,
                    "generation_config_file": _record(
                        FILE_IDENTITY.fields or {}, nullable=True
                    ),
                    "tokenizer_files": model_file_list,
                    "tokenizer_config_file": FILE_IDENTITY,
                    "special_tokens_map_file": FILE_IDENTITY,
                    "processor_files": _array(FILE_IDENTITY),
                }
            ),
        ),
        "fisher_carrier_identity.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "carrier_path": STRING,
                    "carrier_byte_sha256": SHA,
                    "tensors": _array(TENSOR_IDENTITY, min_length=1),
                    "empirical_rank": POS_INT,
                    "producing_source_identity_sha256": NULLABLE_SHA,
                    "producing_partition_commitment": NULLABLE_SHA,
                }
            ),
        ),
        "runtime_identity.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "operating_system": STRING,
                    "architecture": STRING,
                    "python_version": STRING,
                    "torch_version": STRING,
                    "transformers_version": STRING,
                    "numpy_version": STRING,
                    "safetensors_version": STRING,
                    "accelerator_backend": STRING,
                    "accelerator_model": STRING,
                    "device_memory_bytes": _integer(nullable=True),
                    "blas_backend": STRING,
                    "deterministic_algorithms_enabled": _boolean(),
                    "thread_configuration": _array(
                        _record({"name": STRING, "value": STRING})
                    ),
                    "dependency_lockfiles": _array(FILE_IDENTITY),
                    "locale": STRING,
                    "timezone": STRING,
                    "model_execution_dtype": STRING,
                    "source_commit": STRING,
                }
            ),
        ),
        "execution_budget.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "maximum_optimizer_updates": _integer(literal=40000),
                    "declared_worst_case_optimizer_updates": _integer(literal=30250),
                    "maximum_development_wall_seconds": _integer(literal=43200),
                    "maximum_generated_storage_bytes": _integer(
                        literal=100 * 1024**3
                    ),
                    "maximum_capture_storage_bytes": _integer(
                        literal=64 * 1024**3
                    ),
                    "maximum_process_rss_bytes": POS_INT,
                    "maximum_device_memory_bytes": _integer(nullable=True),
                    "maximum_retained_checkpoints_per_restart": _integer(literal=2),
                    "optional_skip_order": _array(STRING, min_length=9),
                    "frozen_before_fit_authorization": _boolean(literal=True),
                }
            ),
        ),
        "source_inventory.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "source_commit": STRING,
                    "files": _array(SOURCE_FILE_IDENTITY, min_length=1),
                    "exclusions": _array(STRING),
                    "behavior_affecting_files_complete": _boolean(literal=True),
                }
            ),
        ),
        "row_derivation_protocol.json": (
            ArtifactFamily.IDENTITY,
            _record(
                {
                    "tokenizer_identity_sha256": SHA,
                    "causal_mask": _string(literal="source positions 0..L-2; target=source+1"),
                    "canonical_sequence_identity": STRING,
                    "analysis_unit_commitment_format": _string(literal="raw_sha256_32_bytes"),
                    "source_position_encoding": _string(literal="uint64_be"),
                    "target_position_encoding": _string(literal="uint64_be"),
                    "row_id_domain": _string(
                        literal=(
                            "fisher-compact-message-completion/evaluation-row/v1"
                        )
                    ),
                    "row_ordering": _string(literal="ascending_raw_row_id_bytes"),
                    "duplicate_policy": _string(literal="reject"),
                    "row_set_domain": _string(
                        literal=(
                            "fisher-compact-message-completion/evaluation-row-set/v1"
                        )
                    ),
                    "ordered_sequence_domain": _string(
                        literal=(
                            "fisher-compact-message-completion/evaluation-row-sequence/v1"
                        )
                    ),
                }
            ),
        ),
        "capture_invariants.json": (
            ArtifactFamily.CAPTURE_INVARIANT,
            _record(
                {
                    "fit_capture_manifest_sha256": SHA,
                    "transported_message_dtype": _string(literal="float32"),
                    "causal_mask": _string(literal="source positions 0..L-2; target=source+1"),
                    "query_head_count": POS_INT,
                    "transported_message_dimension": POS_INT,
                    "all_heads_retained_per_row": _boolean(literal=True),
                    "recomputed_attention_rmsnorm": _boolean(literal=True),
                    "recomputed_mlp_rmsnorm": _boolean(literal=True),
                    "primary_mlp_partition": _string(literal="contiguous_floor_four_block"),
                    "seams": _array(CAPTURE_SEAM_OBSERVATION, min_length=6),
                    "passed": _boolean(),
                }
            ),
        ),
        "o1_tolerance_protocol.json": (
            ArtifactFamily.O1,
            _record(
                {
                    "calibration_assignment_sha256": SHA,
                    "weighting": _string(literal="equal_analysis_unit_nested_rows"),
                    "weighted_median_quantile": _number(literal=0.5),
                    "hard_abs_scale": _number(literal=1e-4),
                    "hard_abs_offset": _number(literal=1e-6),
                    "hard_rel_cap": _number(literal=1e-4),
                    "tolerances": _array(O1_TOLERANCE, min_length=6),
                    "protocol_construction_passed": _boolean(),
                }
            ),
        ),
        "o1_validation_results.json": (
            ArtifactFamily.O1,
            _record(
                {
                    "tolerance_protocol_sha256": SHA,
                    "analysis_unit_count": _integer(literal=36),
                    "row_count": POS_INT,
                    "results": _array(O1_RESULT, min_length=6),
                    "all_seams_passed": _boolean(),
                }
            ),
        ),
        "o1_complete_fit_results.json": (
            ArtifactFamily.O1,
            _record(
                {
                    "tolerance_protocol_sha256": SHA,
                    "analysis_unit_count": _integer(literal=48),
                    "row_count": POS_INT,
                    "results": _array(O1_RESULT, min_length=6),
                    "all_seams_passed": _boolean(),
                }
            ),
        ),
        "locator_protocol.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "atlas_names": _array(
                        _string(choices=("A_in", "A_mid", "A_out")), min_length=3
                    ),
                    "membership_atlas": _string(literal="A_in"),
                    "global_pca_rank": _integer(literal=32),
                    "local_locator_rank": _integer(literal=32),
                    "candidate_K_values": _array(POS_INT, min_length=2),
                    "minimum_fit_train_units_per_chart": _integer(literal=8),
                    "kmeans_seed": _integer(literal=1701),
                    "maximum_center_updates": _integer(literal=100),
                    "centroid_shift_tolerance": _number(literal=1e-6),
                    "support_quantile": _number(literal=0.99),
                    "support_rcond": _number(literal=1e-8),
                    "activation_only_membership": _boolean(literal=True),
                    "teacher_fallback": _boolean(literal=False),
                }
            ),
        ),
        "locator_candidates.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "locator_protocol_sha256": SHA,
                    "candidates": _array(
                        _record(
                            {
                                "candidate_id": STRING,
                                "K": POS_INT,
                                "status": _string(
                                    choices=(
                                        "eligible",
                                        "invalid_support",
                                        "invalid_empty_cluster",
                                        "invalid_covariance",
                                    )
                                ),
                                "chart_support": _array(chart_support),
                                "tensor_manifest_sha256": SHA,
                            }
                        ),
                        min_length=1,
                    ),
                    "selected_locator_candidate_id": NULLABLE_STRING,
                }
            ),
        ),
        "codec_protocol.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "carrier_identity_sha256": SHA,
                    "codec_candidates": _array(CODEC_CANDIDATE, min_length=4),
                    "uncentered_second_moments": _boolean(literal=True),
                    "linear_encoders": _boolean(literal=True),
                    "shared_q_decoder_across_heads": _boolean(literal=True),
                    "orthogonality_threshold": _number(literal=1e-6),
                    "idempotence_threshold": _number(literal=1e-6),
                    "encoder_construction_threshold": _number(literal=1e-10),
                    "epsilon_codec": _number(literal=1e-12),
                    "positive_only_operator_reference": _boolean(literal=True),
                }
            ),
        ),
        "codec_candidate_manifests.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "codec_protocol_sha256": SHA,
                    "candidates": _array(CODEC_NUMERIC_RESULT, min_length=4),
                    "lower_K_fallback_exhausted": _boolean(),
                }
            ),
        ),
        "message_sketch_manifest_p32.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "message_rank_per_head": _integer(literal=32),
                    "fit_partition": _string(literal="fit_train"),
                    "centered_covariance": _boolean(literal=True),
                    "deterministic_svd": _boolean(literal=True),
                    "sign_rule": _string(
                        literal="largest_magnitude_loading_positive_smallest_index_tie"
                    ),
                    "heads": _array(
                        _record(
                            {
                                "head_id": NONNEG_INT,
                                "message_dimension": POS_INT,
                                "center_sha256": SHA,
                                "basis_sha256": SHA,
                                "normalization_sha256": SHA,
                            }
                        ),
                        min_length=1,
                    ),
                }
            ),
        ),
        "representation_numeric_constants.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "variance_floor": _number(literal=1e-6),
                    "support_rcond": _number(literal=1e-8),
                    "support_absolute_floor": _number(literal=1e-8),
                    "epsilon_codec": _number(literal=1e-12),
                    "epsilon_down": _number(minimum=1e-12),
                    "rho_denominator_floor": _number(literal=1e-12),
                    "codec_constant_manifests": _array(sha_map_record, min_length=4),
                    "matching_calipers_sha256": SHA,
                    "donor_boundaries_sha256": SHA,
                    "full_rank_tolerances_sha256": SHA,
                }
            ),
        ),
        "full_rank_identity_smoke.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "message_rank": _string(literal="full_message_dimension"),
                    "standardization_roundtrip": _boolean(literal=True),
                    "calibration_fit_train": _record(
                        {
                            "normalization_sha256": SHA,
                            "head_tolerances": _array(
                                full_rank_head_tolerance, min_length=1
                            ),
                            "coefficient_tolerances": _array(
                                full_rank_coefficient_tolerance, min_length=1
                            ),
                        }
                    ),
                    "validation_fit_monitor": _record(
                        {
                            "representation_candidate_status": _string(
                                choices=(
                                    "eligible",
                                    "failed",
                                    "invalid_zero_validation_support",
                                )
                            ),
                            "records": _array(
                                full_rank_validation_record, min_length=1
                            ),
                        }
                    ),
                    "frozen_tolerances_sha256": SHA,
                    "all_eligible_records_passed": _boolean(),
                }
            ),
        ),
        "selection_representation_manifest.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "namespace": _string(literal="selection_representation"),
                    "fit_partition": _string(literal="fit_train"),
                    "selected_locator_candidate_id": STRING,
                    "bindings": _array(representation_binding, min_length=1),
                    "chart_id_order": _array(NONNEG_INT, min_length=1),
                    "immutable_after_tune_access": _boolean(literal=True),
                }
            ),
        ),
        "ceiling_gate_protocol.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "causal_mask": _string(literal="source positions 0..L-2; target=source+1"),
                    "gate_thresholds": _array(GATE_RESULT, min_length=5),
                    "prompt_tail_rule": _string(literal="nearest_rank_adverse_tail"),
                    "conjunction_required": _boolean(literal=True),
                    "evaluation_label": _string(literal=FULL_PROMPT_LABEL),
                }
            ),
        ),
        "learning_protocol.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "mandatory_learned_models": _array(
                        _string(choices=MANDATORY_LEARNED_MODELS), min_length=5
                    ),
                    "dedicated_J2_optional": _boolean(literal=True),
                    "coefficient_only": _boolean(literal=True),
                    "activation_dimension": _integer(literal=32),
                    "retained_message_dimension": _integer(literal=32),
                    "interaction_rank": _integer(literal=8),
                    "output_bias_field": _string(literal="output_bias"),
                    "width_table": _array(
                        _record(
                            {
                                "q_rank": _integer(minimum=1),
                                "J4_parameters": POS_INT,
                                "J2plus_width_z": POS_INT,
                                "J2plus_width_r": POS_INT,
                                "J2plus_parameters": POS_INT,
                                "J3_width": POS_INT,
                                "J3_parameters": POS_INT,
                            }
                        ),
                        min_length=2,
                    ),
                    "optimizer": _string(literal="AdamW"),
                    "learning_rate": _number(literal=1e-3),
                    "betas": _array(_number(), min_length=2),
                    "optimizer_epsilon": _number(literal=1e-8),
                    "weight_decay": _number(literal=1e-4),
                    "gradient_clip_global_norm": _number(literal=1.0),
                    "warmup_updates": _integer(literal=100),
                    "batch_rows": _integer(literal=256),
                    "maximum_updates": _integer(literal=2000),
                    "minimum_updates": _integer(literal=200),
                    "evaluation_interval_updates": _integer(literal=25),
                    "patience_updates": _integer(literal=200),
                    "restart_seeds": _array(_integer(minimum=0), min_length=3),
                    "epsilon_down": _number(minimum=1e-12),
                }
            ),
        ),
        "development_eligibility_protocol.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "required_baselines": _array(
                        _string(choices=REQUIRED_BASELINES), min_length=4
                    ),
                    "rho_point_threshold": _number(literal=0.05),
                    "rho_denominator_floor": _number(literal=1e-12),
                    "interaction_fraction_threshold": _number(literal=0.25),
                    "minimum_valid_bootstrap_fraction": _number(literal=0.95),
                    "bootstrap_replicates": _integer(literal=10000),
                    "bootstrap_seed": _integer(literal=424242),
                    "interval_level": _number(literal=0.95),
                    "minimum_development_units": _integer(literal=18),
                    "minimum_units_per_active_chart": _integer(literal=4),
                    "maximum_out_of_support_rate": _number(literal=0.10),
                    "epsilon_down": _number(minimum=1e-12),
                }
            ),
        ),
        "preregistration_root.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "bound_artifacts": _array(sha_map_record, min_length=1),
                    "fit_capture_manifest_sha256": SHA,
                    "reserve_retirement_receipt_sha256": SHA,
                    "local_registry_snapshot_sha256": SHA,
                    "complete_before_tune_authorization": _boolean(literal=True),
                }
            ),
        ),
        "local_development_freshness_audit_receipt.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "authority_scope": _string(literal="local_development_only"),
                    "preregistration_root_sha256": SHA,
                    "audited_content_set_sha256s": _array(SHA, min_length=1),
                    "audited_analysis_unit_set_sha256s": _array(SHA, min_length=1),
                    "overlap_status": _string(choices=("fresh", "overlap_detected")),
                    "can_authorize_guard": _boolean(literal=False),
                }
            ),
        ),
        "campaign_root.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "authority_scope": _string(literal="local_development_only"),
                    "preregistration_root_sha256": SHA,
                    "freshness_receipt_sha256": SHA,
                    "model_identity_sha256": SHA,
                    "carrier_identity_sha256": SHA,
                    "runtime_identity_sha256": SHA,
                    "source_inventory_sha256": SHA,
                    "fit_capture_manifest_sha256": SHA,
                    "local_registry_snapshot_sha256": SHA,
                }
            ),
        ),
        "o2q_results.json": (
            ArtifactFamily.O2,
            _record(
                {
                    "ceiling": _string(literal="O2q"),
                    "exact_downstream_mlp": _boolean(literal=True),
                    "recomputed_attention_rmsnorm": _boolean(literal=True),
                    "candidate_panels": _array(
                        _record(
                            {
                                "codec_id": _string(choices=CODEC_IDS),
                                "panels": _array(PANEL_GATES, min_length=2),
                                "passes_fit_monitor_and_tune": _boolean(),
                            }
                        ),
                        min_length=4,
                    ),
                    "passing_codec_ids": _array(_string(choices=CODEC_IDS)),
                }
            ),
        ),
        "o2v_results.json": (
            ArtifactFamily.O2,
            _record(
                {
                    "ceiling": _string(literal="O2v"),
                    "exact_downstream_mlp": _boolean(literal=True),
                    "projected_post_normalization_flow": _boolean(literal=True),
                    "candidate_panels": _array(
                        _record(
                            {
                                "codec_id": _string(choices=CODEC_IDS),
                                "panels": _array(PANEL_GATES, min_length=2),
                                "passes_fit_monitor_and_tune": _boolean(),
                            }
                        ),
                        min_length=4,
                    ),
                    "passing_codec_ids": _array(_string(choices=CODEC_IDS)),
                }
            ),
        ),
        "selected_codec.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "codec_id": _string(choices=CODEC_IDS),
                    "q_rank": _integer(minimum=1),
                    "v_rank": _integer(minimum=1),
                    "lambda_ratio": _number(minimum=0.0),
                    "selection_order": _array(STRING, min_length=7),
                    "o2q_result_sha256": SHA,
                    "o2v_result_sha256": SHA,
                }
            ),
        ),
        "frozen_representation_manifest.json": (
            ArtifactFamily.REPRESENTATION,
            _record(
                {
                    "selection_representation_sha256": SHA,
                    "selected_codec_sha256": SHA,
                    "bindings": _array(representation_binding, min_length=1),
                    "representation_refit_permitted": _boolean(literal=False),
                    "model_weight_refit_permitted": _boolean(literal=True),
                }
            ),
        ),
        "m2q_results.json": (
            ArtifactFamily.M2,
            _record(
                {
                    "ceiling": _string(literal="M2q"),
                    "primary_mlp_partition": _string(literal="contiguous_floor_four_block"),
                    "exact_teacher_attention": _boolean(literal=True),
                    "exact_h_A": _boolean(literal=True),
                    "learned_mlp_replacement": _boolean(literal=False),
                    "diagnostic_only": _boolean(literal=True),
                    "panels": _array(PANEL_GATES, min_length=1),
                }
            ),
        ),
        "m2v_results.json": (
            ArtifactFamily.M2,
            _record(
                {
                    "ceiling": _string(literal="M2v"),
                    "primary_mlp_partition": _string(literal="contiguous_floor_four_block"),
                    "exact_teacher_attention": _boolean(literal=True),
                    "exact_h_A": _boolean(literal=True),
                    "learned_mlp_replacement": _boolean(literal=False),
                    "diagnostic_only": _boolean(literal=True),
                    "panels": _array(PANEL_GATES, min_length=1),
                }
            ),
        ),
        "compact_ambient_oracle.json": (
            ArtifactFamily.ORACLE,
            _record(
                {
                    "oracle": _string(literal="C-Ambient"),
                    "message_rank_per_head": _integer(literal=32),
                    "q_codec_projection": _boolean(literal=False),
                    "selection_eligible": _boolean(literal=False),
                    "panels": _array(oracle_panel, min_length=1),
                }
            ),
        ),
        "compact_chart_oracle.json": (
            ArtifactFamily.ORACLE,
            _record(
                {
                    "oracle": _string(literal="C-Chart"),
                    "message_rank_per_head": _integer(literal=32),
                    "q_codec_projection": _boolean(literal=True),
                    "selection_eligible": _boolean(literal=False),
                    "panels": _array(oracle_panel, min_length=1),
                }
            ),
        ),
        "j_model_fit_results.json": (
            ArtifactFamily.J_MODEL,
            _record(
                {
                    "training_partition": _string(literal="fit_train"),
                    "selection_partition": _string(literal="fit_monitor"),
                    "coefficient_only": _boolean(literal=True),
                    "shared_representation_sha256": SHA,
                    "models": _array(MODEL_RESULT, min_length=6),
                    "J2N_J4_common_initialization": _boolean(literal=True),
                    "J2N_J4_identical_batches": _boolean(literal=True),
                }
            ),
        ),
        "j_model_tune_results.json": (
            ArtifactFamily.J_MODEL,
            _record(
                {
                    "partition": _string(literal="tune"),
                    "shared_representation_sha256": SHA,
                    "fixed_checkpoints": _boolean(literal=True),
                    "tune_adaptation_performed": _boolean(literal=False),
                    "models": _array(MODEL_RESULT, min_length=5),
                }
            ),
        ),
        "all_fit_weight_refit.json": (
            ArtifactFamily.REFIT,
            _record(
                {
                    "gradient_partition": _string(literal="all_fit"),
                    "model_ids": _array(
                        _string(choices=("J2N", "J2+", "J3", "J4")), min_length=4
                    ),
                    "optimizer_updates": _integer(literal=1000),
                    "seed": _integer(literal=8675309),
                    "early_stopping": _boolean(literal=False),
                    "representation_before_sha256": SHA,
                    "representation_after_sha256": SHA,
                    "representation_modified": _boolean(literal=False),
                    "weights_manifest_sha256": SHA,
                }
            ),
        ),
        "nonselecting_tune_replay.json": (
            ArtifactFamily.REFIT,
            _record(
                {
                    "partition": _string(literal="tune"),
                    "tune_adaptation_performed": _boolean(literal=False),
                    "maximum_allowed_relative_degradation": _number(literal=0.02),
                    "models": _array(
                        _record(
                            {
                                "model_id": _string(choices=("J2N", "J2+", "J3", "J4")),
                                "pre_refit_L_bar_down": NONNEG_NUMBER,
                                "post_refit_L_bar_down": NONNEG_NUMBER,
                                "relative_degradation": _number(),
                                "stable": _boolean(),
                            }
                        ),
                        min_length=4,
                    ),
                    "all_required_models_stable": _boolean(),
                }
            ),
        ),
        "frozen_B_star.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "model_id": _string(choices=REQUIRED_BASELINES),
                    "selection_metric": _string(literal="L_bar_down"),
                    "selected_value": NONNEG_NUMBER,
                    "tie_order": _array(_string(choices=REQUIRED_BASELINES), min_length=4),
                    "tune_replay_sha256": SHA,
                    "frozen_before_development_confirm": _boolean(literal=True),
                }
            ),
        ),
        "direct_full_rank_identity_null.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "selected_codec_sha256": SHA,
                    "frozen_tolerances_sha256": SHA,
                    "representation_refit_performed": _boolean(literal=False),
                    "message_roundtrip_passed": _boolean(),
                    "q_roundtrip_passed": _boolean(),
                    "coefficient_roundtrip_passed": _boolean(),
                    "completion_target_null_passed": _boolean(),
                    "coefficient_advantage_null_passed": _boolean(),
                    "downstream_vanishing_required": _boolean(literal=False),
                }
            ),
        ),
        "candidate_bundle_metadata.json": (
            ArtifactFamily.BUNDLE,
            _record(
                {
                    "candidate_bundle_byte_sha256": SHA,
                    "candidate_bundle_byte_size": POS_INT,
                    "comparator_bundle_byte_sha256": SHA,
                    "comparator_bundle_byte_size": POS_INT,
                    "loader": _string(literal="safetensors_weights_only"),
                    "pickle_permitted": _boolean(literal=False),
                    "selected_representation_sha256": SHA,
                    "source_inventory_sha256": SHA,
                    "model_identity_sha256": SHA,
                    "carrier_identity_sha256": SHA,
                    "runtime_identity_sha256": SHA,
                    "row_derivation_protocol_sha256": SHA,
                    "campaign_root_sha256": SHA,
                    "ceiling_gate_protocol_sha256": SHA,
                    "learning_protocol_sha256": SHA,
                    "development_eligibility_protocol_sha256": SHA,
                    "full_rank_identity_smoke_sha256": SHA,
                    "message_rank_per_head": _integer(literal=32),
                    "interaction_rank": _integer(literal=8),
                    "tensors": _array(
                        _record(
                            {
                                "bundle": _string(choices=("candidate", "comparator")),
                                "name": STRING,
                                "shape": _array(POS_INT, min_length=1),
                                "dtype": STRING,
                            }
                        ),
                        min_length=1,
                    ),
                    "deterministic_self_test_sha256": SHA,
                    "deterministic_self_test_passed": _boolean(),
                }
            ),
        ),
        "development_confirm_results.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "partition": _string(literal="development_confirm"),
                    "analysis_unit_count": NONNEG_INT,
                    "evaluable_analysis_unit_count": NONNEG_INT,
                    "out_of_support_rate": UNIT_INTERVAL,
                    "active_chart_unit_counts": _array(chart_support),
                    "models": _array(MODEL_RESULT, min_length=5),
                    "comparisons": _array(COMPARISON_RESULT, min_length=4),
                    "trained_J4_operator_aggregate": trained_operator_aggregate,
                    "bootstrap_replicates": _integer(literal=10000),
                    "bootstrap_seed": _integer(literal=424242),
                    "interval_level": _number(literal=0.95),
                    "stratum_scheme": _string(
                        choices=("family", "family_x_template_id", "global")
                    ),
                    "primary_metric_stage": _string(literal="after_exact_teacher_mlp"),
                    "pre_mlp_metric_label": _string(
                        literal="H4-carrier-overlap surrogate measured before the exact MLP"
                    ),
                }
            ),
        ),
        "interaction_mean_ablation.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "baseline_model": _string(choices=REQUIRED_BASELINES),
                    "A_z_unchanged": _boolean(literal=True),
                    "only_G_z_z_replaced": _boolean(literal=True),
                    "replacement": _string(literal="all_fit_chart_head_mean"),
                    "advantage_J4": _number(),
                    "advantage_floor": _number(minimum=1e-12),
                    "gamma_interaction": _number(nullable=True),
                    "gamma_status": _string(
                        choices=("regular", "advantage_at_or_below_floor", "insufficient_valid_bootstrap")
                    ),
                    "gamma_point_screen_passed": _boolean(),
                    "paired_ci_lower": _number(),
                    "paired_ci_upper": _number(),
                    "paired_positive_passed": _boolean(),
                    "valid_gamma_replicate_fraction": UNIT_INTERVAL,
                    "interaction_criterion_passed": _boolean(),
                }
            ),
        ),
        "o7_attention_output_completion_results.json": (
            ArtifactFamily.O7,
            _record(
                {
                    "replacement_scope": _string(literal="attention_output_message_completion"),
                    "exact_teacher_qkv_and_routing": _boolean(literal=True),
                    "exact_transported_messages": _boolean(literal=True),
                    "recomputed_attention_rmsnorm": _boolean(literal=True),
                    "exact_teacher_mlp": _boolean(literal=True),
                    "exact_remaining_teacher": _boolean(literal=True),
                    "learned_mlp_replacement": _boolean(literal=False),
                    "evaluation_label": _string(literal=FULL_PROMPT_LABEL),
                    "panel": PANEL_GATES,
                    "all_five_passed": _boolean(),
                }
            ),
        ),
        "development_eligibility_result.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "o2q_replay_passed": _boolean(),
                    "all_J4_comparisons_passed": _boolean(),
                    "interaction_mean_passed": _boolean(),
                    "o7_all_five_passed": _boolean(),
                    "out_of_support_passed": _boolean(),
                    "independent_support_passed": _boolean(),
                    "development_eligible": _boolean(),
                    "terminal_if_finalized": _string(
                        choices=(
                            "development_complete_no_eligible_guard",
                            "development_complete_negative_j4",
                            "development_complete_attention_output_not_ready",
                            "development_complete_insufficient_support",
                        )
                    ),
                    "reason_code": STRING,
                }
            ),
        ),
        "development_power_assessment.json": (
            ArtifactFamily.DEVELOPMENT,
            _record(
                {
                    "guard_analysis_unit_count_assumed": _integer(literal=12),
                    "expected_paired_interval_width": NONNEG_NUMBER,
                    "probability_confirming_five_percent_effect": UNIT_INTERVAL,
                    "one_adverse_unit_sensitivity": _number(),
                    "two_adverse_unit_sensitivity": _number(),
                    "expected_chart_support": _array(chart_support),
                    "reporting_only": _boolean(literal=True),
                    "eligibility_affecting": _boolean(literal=False),
                }
            ),
        ),
        "development_candidate_manifest.json": (
            ArtifactFamily.CANDIDATE,
            _record(
                {
                    "source_inventory_sha256": SHA,
                    "model_identity_sha256": SHA,
                    "tokenizer_identity_sha256": SHA,
                    "carrier_identity_sha256": SHA,
                    "runtime_identity_sha256": SHA,
                    "execution_budget_sha256": SHA,
                    "independence_mode": _string(
                        choices=("template_lineage", "semantic_instance")
                    ),
                    "inference_limitation": NULLABLE_STRING,
                    "split_manifest_sha256": SHA,
                    "partition_commitments": _array(PARTITION_IDENTITY, min_length=4),
                    "capture_manifest_sha256s": _array(sha_map_record, min_length=3),
                    "freshness_receipt_sha256": SHA,
                    "campaign_root_sha256": SHA,
                    "frozen_representation_sha256": SHA,
                    "selected_codec_sha256": SHA,
                    "candidate_bundle_byte_sha256": SHA,
                    "comparator_bundle_byte_sha256": SHA,
                    "candidate_bundle_metadata_sha256": SHA,
                    "frozen_B_star_sha256": SHA,
                    "development_confirm_result_sha256": SHA,
                    "development_eligibility_result_sha256": SHA,
                    "o7_result_sha256": SHA,
                    "contains_guard_claim_or_result": _boolean(literal=False),
                }
            ),
        ),
    }
    return schemas


def _stage_for(name: str) -> Stage:
    for stage, names in STAGE_ARTIFACT_ORDER.items():
        if name in names:
            return stage
    raise ArtifactSchemaError(f"unknown P1-P12 artifact {name!r}")


def _make_schemas() -> dict[str, ArtifactSchema]:
    bodies = _body_schemas()
    result: dict[str, ArtifactSchema] = {}
    for name, (family, body) in bodies.items():
        result[name] = ArtifactSchema(
            name=name,
            stage=_stage_for(name),
            family=family,
            schema_id=_schema_id(name),
            dependencies=ARTIFACT_DEPENDENCIES[name],
            body=body,
        )
    for name in ("candidate_bundle.safetensors", "comparator_bundle.safetensors"):
        result[name] = ArtifactSchema(
            name=name,
            stage=Stage.P9,
            family=ArtifactFamily.BUNDLE,
            schema_id=f"{SCHEMA_DOMAIN}/{name.removesuffix('.safetensors')}/safetensors-v1",
            dependencies=ARTIFACT_DEPENDENCIES[name],
            body=_record({}),
            binary=True,
        )
    return result


ARTIFACT_SCHEMAS: Final[Mapping[str, ArtifactSchema]] = MappingProxyType(
    _make_schemas()
)


def _synthetic_value(rule: Rule) -> object:
    """Construct a deterministic nonclaim placeholder satisfying one rule."""

    if rule.nullable:
        return None
    if rule.kind == "string":
        if rule.literal is not None:
            return rule.literal
        if rule.choices:
            return rule.choices[0]
        return "synthetic"
    if rule.kind == "boolean":
        return False if rule.literal is None else rule.literal
    if rule.kind == "integer":
        if rule.literal is not None:
            return rule.literal
        return int(rule.minimum or 0)
    if rule.kind == "number":
        if rule.literal is not None:
            return rule.literal
        return float(rule.minimum or 0.0)
    if rule.kind == "sha256":
        return "00" * 32
    if rule.kind == "array":
        assert rule.item is not None
        return [_synthetic_value(rule.item) for _ in range(rule.min_length)]
    if rule.kind == "object":
        assert rule.fields is not None
        return {key: _synthetic_value(value) for key, value in rule.fields.items()}
    raise AssertionError(f"unknown rule kind {rule.kind!r}")


def _synthetic_gate_results() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for gate_id, threshold, comparison in (
        ("absolute_global_delta_nll_per_token", 0.05, "<="),
        ("global_kl_per_token", 0.05, "<="),
        ("global_top1_agreement", 0.95, ">="),
        ("p90_absolute_prompt_delta_nll_per_token", 0.10, "<="),
        ("p10_prompt_top1_agreement", 0.90, ">="),
    ):
        observed = 0.0
        records.append(
            {
                "gate_id": gate_id,
                "observed": observed,
                "threshold": threshold,
                "comparison": comparison,
                "passed": observed <= threshold if comparison == "<=" else observed >= threshold,
            }
        )
    return records


def _synthetic_panel(role: str) -> dict[str, object]:
    return {
        "partition_role": role,
        "gate_results": _synthetic_gate_results(),
        "all_five_passed": False,
        "evaluation_label": FULL_PROMPT_LABEL,
    }


def _synthetic_model(model_id: str) -> dict[str, object]:
    analytic = model_id == "J1"
    return {
        "model_id": model_id,
        "mandatory": model_id != "J2",
        "representation_sha256": "00" * 32,
        "architecture_sha256": "00" * 32,
        "selected_seed": None if analytic else 1701,
        "selected_optimizer_step": None if analytic else 200,
        "coefficient_loss": None if analytic else 0.0,
        "L_bar_down": 0.0,
        "status": "analytic" if analytic else "evaluated",
    }


def _synthetic_partition(role: str, count: int) -> dict[str, object]:
    return {
        "partition_role": role,
        "analysis_unit_count": count,
        "sequence_count": count,
        "content_set_sha256": "00" * 32,
        "analysis_unit_set_sha256": "00" * 32,
        "partition_commitment": "00" * 32,
    }


def synthetic_body_template(name: str) -> dict[str, object]:
    """Return a fresh validator-passing body for any external JSON artifact.

    The values are deterministic sentinels, not measurements.  The only
    supported use is synthetic schema/orchestration qualification; passing the
    result to :func:`build_artifact_payload` permanently adds the nonclaim
    evidence envelope.
    """

    schema = artifact_schema(name)
    if schema.binary:
        raise ArtifactSchemaError(
            f"{name!r} is binary and has no JSON body template"
        )
    body = _synthetic_value(schema.body)
    assert isinstance(body, dict)
    zero = "00" * 32

    if name == "independence_protocol.json":
        body.update(
            {
                "selected_mode": "template_lineage",
                "analysis_unit_field": "template_lineage_id",
                "mode_a_eligible": True,
                "mode_b_eligible": False,
                "fit_analysis_unit_count": 48,
                "tune_analysis_unit_count": 24,
                "development_confirm_analysis_unit_count": 24,
                "inference_limitation": None,
            }
        )
    elif name == "split_manifest.json":
        body.update(
            {
                "purpose_identifiers": list(SPLIT_PURPOSES),
                "partitions": [
                    _synthetic_partition("fit_train", 38),
                    _synthetic_partition("fit_monitor", 10),
                    _synthetic_partition("tune", 24),
                    _synthetic_partition("development_confirm", 24),
                ],
            }
        )
    elif name in {
        "capture_invariants.json",
        "o1_tolerance_protocol.json",
        "o1_validation_results.json",
        "o1_complete_fit_results.json",
    }:
        if name == "capture_invariants.json":
            body["seams"] = [
                {
                    "seam": seam,
                    "observed_max_abs_error": 0.0,
                    "observed_weighted_relative_l2": 0.0,
                    "finite": True,
                    "shape_contract_passed": True,
                }
                for seam in O1_SEAMS
            ]
            body["passed"] = True
        elif name == "o1_tolerance_protocol.json":
            body["tolerances"] = [
                {
                    "seam": seam,
                    "reference_scale": 1e-8,
                    "calibration_max_abs_error": 0.0,
                    "calibration_weighted_relative_l2": 0.0,
                    "cap_abs": 1.000001e-6,
                    "cap_rel": 1e-4,
                    "tau_abs": 1e-12,
                    "tau_rel": 1e-12,
                    "calibration_passed": True,
                }
                for seam in O1_SEAMS
            ]
            body["protocol_construction_passed"] = True
        else:
            body["results"] = [
                {
                    "seam": seam,
                    "observed_max_abs_error": 0.0,
                    "observed_weighted_relative_l2": 0.0,
                    "tau_abs": 1e-12,
                    "tau_rel": 1e-12,
                    "passed": True,
                }
                for seam in O1_SEAMS
            ]
            body["row_count"] = 1
            body["all_seams_passed"] = True
    elif name == "execution_budget.json":
        body["optional_skip_order"] = list(OPTIONAL_DIAGNOSTIC_SKIP_ORDER)
    elif name == "locator_protocol.json":
        body["atlas_names"] = ["A_in", "A_mid", "A_out"]
        body["candidate_K_values"] = [4, 2]
    elif name == "locator_candidates.json":
        body["candidates"] = [
            {
                "candidate_id": "K4",
                "K": 4,
                "status": "eligible",
                "chart_support": [],
                "tensor_manifest_sha256": zero,
            }
        ]
        body["selected_locator_candidate_id"] = "K4"
    elif name == "codec_protocol.json":
        body["codec_candidates"] = [
            {"codec_id": "C0", "q_rank": 32, "v_rank": 32, "lambda_ratio": 1e-3},
            {"codec_id": "C1", "q_rank": 64, "v_rank": 64, "lambda_ratio": 1e-3},
            {"codec_id": "C2", "q_rank": 32, "v_rank": 32, "lambda_ratio": 1e-2},
            {"codec_id": "C3", "q_rank": 64, "v_rank": 64, "lambda_ratio": 1e-2},
        ]
    elif name == "codec_candidate_manifests.json":
        body["candidates"] = [
            {
                "codec_id": codec_id,
                "locator_candidate_id": "K4",
                "q_rank": rank,
                "v_rank": rank,
                "lambda_ratio": ratio,
                "orthogonality_error": 0.0,
                "idempotence_error": 0.0,
                "encoder_construction_error": 0.0,
                "operator_reference_norm": 1.0,
                "tau_R": 1e-12,
                "status": "eligible",
                "tensor_manifest_sha256": zero,
            }
            for codec_id, rank, ratio in (
                ("C0", 32, 1e-3),
                ("C1", 64, 1e-3),
                ("C2", 32, 1e-2),
                ("C3", 64, 1e-2),
            )
        ]
    elif name == "message_sketch_manifest_p32.json":
        body["heads"] = [
            {
                "head_id": head_id,
                "message_dimension": 256,
                "center_sha256": zero,
                "basis_sha256": zero,
                "normalization_sha256": zero,
            }
            for head_id in range(4)
        ]
    elif name == "representation_numeric_constants.json":
        body["codec_constant_manifests"] = [
            {"name": codec_id, "artifact_sha256": zero}
            for codec_id in CODEC_IDS
        ]
    elif name == "full_rank_identity_smoke.json":
        body["message_rank"] = "full_message_dimension"
        body["calibration_fit_train"] = {
            "normalization_sha256": zero,
            "head_tolerances": [
                {
                    "head_id": 0,
                    "fit_train_row_count": 1,
                    "fit_train_analysis_unit_count": 1,
                    "message_reference_scale": 1e-8,
                    "q_reference_scale": 1e-8,
                    "observed_max_error_m": 0.0,
                    "observed_max_error_q": 0.0,
                    "cap_m": 1e-7,
                    "cap_q": 1e-7,
                    "tau_full_m": 1e-12,
                    "tau_full_q": 1e-12,
                    "passed": True,
                }
            ],
            "coefficient_tolerances": [
                {
                    "codec_id": "C0",
                    "chart_id": 0,
                    "head_id": 0,
                    "fit_train_row_count": 1,
                    "fit_train_analysis_unit_count": 1,
                    "coefficient_reference_scale": 1e-8,
                    "observed_max_error_c": 0.0,
                    "cap_c": 1e-7,
                    "tau_full_c": 1e-12,
                    "tau_full_advantage": 1e-23,
                    "passed": True,
                }
            ],
        }
        body["validation_fit_monitor"] = {
            "representation_candidate_status": "eligible",
            "records": [
                {
                    "codec_id": "C0",
                    "chart_id": 0,
                    "head_id": 0,
                    "assigned_row_count": 1,
                    "distinct_analysis_unit_count": 1,
                    "observed_max_error_m": 0.0,
                    "observed_max_error_q": 0.0,
                    "observed_max_error_c": 0.0,
                    "coefficient_completion_mse": 0.0,
                    "tau_full_m": 1e-12,
                    "tau_full_q": 1e-12,
                    "tau_full_c": 1e-12,
                    "tau_full_advantage": 1e-23,
                    "validation_status": "passed",
                    "zero_row_reason": None,
                }
            ],
        }
        body["frozen_tolerances_sha256"] = zero
        body["all_eligible_records_passed"] = True
    elif name in {
        "selection_representation_manifest.json",
        "frozen_representation_manifest.json",
    }:
        body["bindings"] = [
            {"object_name": "synthetic_object", "artifact_sha256": zero, "immutable": True}
        ]
        if name == "selection_representation_manifest.json":
            body["selected_locator_candidate_id"] = "K4"
            body["chart_id_order"] = [0]
    elif name == "ceiling_gate_protocol.json":
        body["gate_thresholds"] = _synthetic_gate_results()
    elif name == "learning_protocol.json":
        body.update(
            {
                "mandatory_learned_models": list(MANDATORY_LEARNED_MODELS),
                "width_table": [
                    {
                        "q_rank": 32,
                        "J4_parameters": 2848,
                        "J2plus_width_z": 21,
                        "J2plus_width_r": 22,
                        "J2plus_parameters": 2859,
                        "J3_width": 29,
                        "J3_parameters": 2845,
                    },
                    {
                        "q_rank": 64,
                        "J4_parameters": 5184,
                        "J2plus_width_z": 26,
                        "J2plus_width_r": 26,
                        "J2plus_parameters": 5172,
                        "J3_width": 40,
                        "J3_parameters": 5224,
                    },
                ],
                "betas": [0.9, 0.999],
                "restart_seeds": [1701, 2718, 3141],
            }
        )
    elif name == "development_eligibility_protocol.json":
        body["required_baselines"] = list(REQUIRED_BASELINES)
    elif name in {"preregistration_root.json", "campaign_root.json"}:
        if name == "preregistration_root.json":
            body["bound_artifacts"] = [
                {"name": artifact_name, "artifact_sha256": zero}
                for artifact_name in PREREGISTRATION_BOUND_ARTIFACTS
            ]
    elif name == "local_development_freshness_audit_receipt.json":
        body["audited_content_set_sha256s"] = [zero]
        body["audited_analysis_unit_set_sha256s"] = [zero]
    elif name in {"o2q_results.json", "o2v_results.json"}:
        body["candidate_panels"] = [
            {
                "codec_id": codec_id,
                "panels": [
                    _synthetic_panel("fit_monitor"),
                    _synthetic_panel("tune"),
                ],
                "passes_fit_monitor_and_tune": False,
            }
            for codec_id in CODEC_IDS
        ]
        body["passing_codec_ids"] = []
    elif name == "selected_codec.json":
        body.update(
            {
                "codec_id": "C0",
                "q_rank": 32,
                "v_rank": 32,
                "lambda_ratio": 1e-3,
                "selection_order": [
                    "tune_absolute_delta_nll",
                    "tune_kl",
                    "tune_top1_desc",
                    "q_rank",
                    "v_rank",
                    "regularization_desc",
                    "codec_id",
                ],
            }
        )
    elif name in {"m2q_results.json", "m2v_results.json"}:
        body["panels"] = [_synthetic_panel("tune")]
    elif name in {"compact_ambient_oracle.json", "compact_chart_oracle.json"}:
        body["panels"] = [
            {
                "partition_role": "tune",
                "local": {
                    "q_total_max_abs_error": 0.0,
                    "q_total_weighted_relative_l2": 0.0,
                    "h_A_max_abs_error": 0.0,
                    "h_A_weighted_relative_l2": 0.0,
                    "h_plus_max_abs_error": 0.0,
                    "h_plus_weighted_relative_l2": 0.0,
                },
                "full_prompt_shadow": _synthetic_panel("tune"),
            }
        ]
    elif name == "j_model_fit_results.json":
        body["models"] = [
            _synthetic_model(model_id)
            for model_id in ("J0", "J1", "J2N", "J2+", "J3", "J4")
        ]
    elif name == "j_model_tune_results.json":
        body["models"] = [
            _synthetic_model(model_id)
            for model_id in ("J0", "J1", "J2N", "J2+", "J3", "J4")
        ]
    elif name == "all_fit_weight_refit.json":
        body.update(
            {
                "model_ids": ["J2N", "J2+", "J3", "J4"],
                "representation_before_sha256": zero,
                "representation_after_sha256": zero,
            }
        )
    elif name == "nonselecting_tune_replay.json":
        body["models"] = [
            {
                "model_id": model_id,
                "pre_refit_L_bar_down": 0.0,
                "post_refit_L_bar_down": 0.0,
                "relative_degradation": 0.0,
                "stable": True,
            }
            for model_id in ("J2N", "J2+", "J3", "J4")
        ]
        body["all_required_models_stable"] = True
    elif name == "frozen_B_star.json":
        body["tie_order"] = list(REQUIRED_BASELINES)
    elif name == "candidate_bundle_metadata.json":
        body.update(
            {
                "candidate_bundle_byte_size": 1,
                "comparator_bundle_byte_size": 1,
                "tensors": [
                    {
                        "bundle": "candidate",
                        "name": "synthetic.weight",
                        "shape": [1],
                        "dtype": "F32",
                    }
                ],
                "deterministic_self_test_passed": True,
            }
        )
    elif name == "development_confirm_results.json":
        body.update(
            {
                "analysis_unit_count": 24,
                "evaluable_analysis_unit_count": 2,
                "active_chart_unit_counts": [
                    {
                        "chart_id": 0,
                        "analysis_unit_count": 2,
                        "sequence_count": 2,
                        "row_count": 2,
                        "supported": False,
                    }
                ],
                "models": [
                    _synthetic_model(model_id)
                    for model_id in ("J1", "J2N", "J2+", "J3", "J4")
                ],
                "trained_J4_operator_aggregate": {
                    "panel": "development_confirm",
                    "model_id": "J4",
                    "codec_id": "C1",
                    "tau_R": 1e-8,
                    "selected_codec_artifact_sha256": zero,
                    "representation_numeric_constants_sha256": zero,
                    "codec_candidate_manifests_sha256": zero,
                    "aggregate_median_ambient_operator_norm": 0.0,
                    "aggregate_median_H4_surrogate_operator_norm": 0.0,
                    "aggregate_median_ambient_stable_rank": 0.0,
                    "supported_chart_head_pair_count": 4,
                    "excluded_chart_head_pair_count": 0,
                    "zero_ambient_pair_count": 4,
                    "positive_ambient_pair_count": 0,
                    "analysis_unit_support_by_pair": [
                        {
                            "chart_id": 0,
                            "head_id": head_id,
                            "analysis_unit_count": 2,
                            "status": "supported",
                            "pair_median_ambient_operator_norm": 0.0,
                            "pair_median_H4_surrogate_operator_norm": 0.0,
                            "pair_median_ambient_stable_rank": 0.0,
                        }
                        for head_id in range(4)
                    ],
                },
                "comparisons": [
                    {
                        "baseline": baseline,
                        "L_bar_down_baseline": 0.0,
                        "L_bar_down_J4": 0.0,
                        "rho": None,
                        "rho_status": "both_at_numerical_floor",
                        "rho_point_screen_passed": False,
                        "paired_ci_lower": 0.0,
                        "paired_ci_upper": 0.0,
                        "paired_positive_passed": False,
                        "valid_rho_replicate_fraction": 0.0,
                        "comparison_passed": False,
                    }
                    for baseline in REQUIRED_BASELINES
                ],
            }
        )
    elif name == "interaction_mean_ablation.json":
        body.update(
            {
                "baseline_model": "J1",
                "gamma_interaction": None,
                "gamma_status": "advantage_at_or_below_floor",
            }
        )
    elif name == "o7_attention_output_completion_results.json":
        body["panel"] = _synthetic_panel("development_confirm")
    elif name == "development_eligibility_result.json":
        body.update(
            {
                "terminal_if_finalized": "development_complete_negative_j4",
                "reason_code": "synthetic_nonclaim_fixture",
            }
        )
    elif name == "development_power_assessment.json":
        body["expected_chart_support"] = []
    elif name == "development_candidate_manifest.json":
        body.update(
            {
                "partition_commitments": [
                    _synthetic_partition("fit_train", 38),
                    _synthetic_partition("fit_monitor", 10),
                    _synthetic_partition("tune", 24),
                    _synthetic_partition("development_confirm", 24),
                ],
                "capture_manifest_sha256s": [
                    {"name": role, "artifact_sha256": zero}
                    for role in ("fit", "tune", "development_confirm")
                ],
            }
        )

    # The public guarantee is stronger than merely returning a best effort:
    # build a complete envelope and run the same validator callers use.
    dependency_hashes = {
        dependency: zero for dependency in schema.dependencies
    }
    build_artifact_payload(
        name,
        campaign_id="synthetic-template-self-check",
        dependency_hashes=dependency_hashes,
        body=body,
    )
    return copy.deepcopy(body)


def _validate_exact_object(
    value: object, fields: Mapping[str, Rule], *, path: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactSchemaError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ArtifactSchemaError(f"{path} keys must be strings")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactSchemaError(
            f"{path} keys differ; missing={missing!r}, extra={extra!r}"
        )
    for key, rule in fields.items():
        _validate_rule(value[key], rule, path=f"{path}.{key}")
    return value


def _validate_rule(value: object, rule: Rule, *, path: str) -> None:
    if value is None:
        if rule.nullable:
            return
        raise ArtifactSchemaError(f"{path} must not be null")
    if rule.kind == "string":
        if not isinstance(value, str):
            raise ArtifactSchemaError(f"{path} must be a string")
        if len(value) < rule.min_length:
            raise ArtifactSchemaError(f"{path} is too short")
    elif rule.kind == "boolean":
        if type(value) is not bool:
            raise ArtifactSchemaError(f"{path} must be a JSON Boolean")
    elif rule.kind == "integer":
        if type(value) is not int:
            raise ArtifactSchemaError(f"{path} must be a JSON integer")
    elif rule.kind == "number":
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ArtifactSchemaError(f"{path} must be a finite JSON number")
    elif rule.kind == "sha256":
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ArtifactSchemaError(f"{path} must be a lowercase SHA-256")
    elif rule.kind == "array":
        if not isinstance(value, list):
            raise ArtifactSchemaError(f"{path} must be an array")
        if len(value) < rule.min_length:
            raise ArtifactSchemaError(f"{path} has too few items")
        assert rule.item is not None
        for index, item in enumerate(value):
            _validate_rule(item, rule.item, path=f"{path}[{index}]")
        return
    elif rule.kind == "object":
        assert rule.fields is not None
        _validate_exact_object(value, rule.fields, path=path)
        return
    else:  # pragma: no cover - construction bug
        raise AssertionError(f"unknown rule kind {rule.kind!r}")
    if rule.literal is not None and value != rule.literal:
        raise ArtifactSchemaError(f"{path} must equal {rule.literal!r}")
    if rule.choices and value not in rule.choices:
        raise ArtifactSchemaError(f"{path} is outside the frozen vocabulary")
    if rule.minimum is not None and float(value) < rule.minimum:
        raise ArtifactSchemaError(f"{path} is below its minimum")
    if rule.maximum is not None and float(value) > rule.maximum:
        raise ArtifactSchemaError(f"{path} exceeds its maximum")


def _canonical_json_bytes(value: object) -> bytes:
    """Return the frozen finite-integer RFC-8785-equivalent JSON subset."""

    def validate(candidate: object, *, path: str) -> None:
        if candidate is None or type(candidate) in (bool, str):
            return
        if type(candidate) is int:
            if abs(candidate) > 2**53 - 1:
                raise ArtifactSchemaError(f"{path} integer exceeds exact JSON range")
            return
        if type(candidate) is float:
            if not math.isfinite(candidate):
                raise ArtifactSchemaError(f"{path} contains NaN or infinity")
            return
        if isinstance(candidate, list):
            for index, item in enumerate(candidate):
                validate(item, path=f"{path}[{index}]")
            return
        if isinstance(candidate, Mapping):
            for key, item in candidate.items():
                if not isinstance(key, str):
                    raise ArtifactSchemaError(f"{path} has a non-string key")
                validate(item, path=f"{path}.{key}")
            return
        raise ArtifactSchemaError(f"{path} is not a JSON value")

    validate(value, path="$")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def artifact_schema(name: str) -> ArtifactSchema:
    """Return the frozen schema for one externally produced artifact."""

    try:
        return ARTIFACT_SCHEMAS[name]
    except KeyError as error:
        if name in LEDGER_OWNED_ARTIFACTS:
            raise ArtifactSchemaError(
                f"{name!r} is owned by the durable protocol ledger"
            ) from error
        raise ArtifactSchemaError(f"unknown scientific artifact {name!r}") from error


def artifact_dependencies(name: str) -> tuple[str, ...]:
    """Return exact direct dependencies for one P1--P12 success artifact."""

    try:
        return ARTIFACT_DEPENDENCIES[name]
    except KeyError as error:
        raise ArtifactSchemaError(f"unknown P1-P12 artifact {name!r}") from error


def _envelope_fields(schema: ArtifactSchema) -> dict[str, Rule]:
    fields = {
        "schema_id": _string(literal=schema.schema_id),
        "protocol_id": _string(literal=PROTOCOL_ID),
        "campaign_id": STRING,
        "artifact_name": _string(literal=schema.name),
        "stage": _string(literal=schema.stage.value),
        "artifact_family": _string(literal=schema.family.value),
        "evidence_scope": _string(literal=EVIDENCE_SCOPE),
        "synthetic_evidence": _boolean(literal=True),
        "scientific_claim_eligible": _boolean(literal=False),
        "target_model_execution_performed": _boolean(literal=False),
        "protected_panel_accessed": _boolean(literal=False),
        "dependency_hashes": _record(
            {dependency: SHA for dependency in schema.dependencies}
        ),
        "body": schema.body,
    }
    if schema.family is ArtifactFamily.CANDIDATE:
        fields["development_candidate_hash"] = SHA
    return fields


def development_candidate_hash(payload: Mapping[str, object]) -> str:
    """Return the exact G14 domain-separated scientific identity."""

    if "development_candidate_hash" not in payload:
        raise ArtifactSchemaError("development_candidate_hash field is missing")
    if "artifact_byte_sha256" in payload:
        raise ArtifactSchemaError(
            "development candidate must not contain its own artifact byte hash"
        )
    unhashed = dict(payload)
    del unhashed["development_candidate_hash"]
    return hashlib.sha256(
        CANDIDATE_HASH_DOMAIN + b"\0" + _canonical_json_bytes(unhashed)
    ).hexdigest()


def _equal_pair_linear_median(
    records: Sequence[Mapping[str, object]],
    field: str,
) -> float:
    """Mirror the frozen weighted-linear q=.5 rule for equal pair weights."""

    ordered = sorted(
        records,
        key=lambda row: (
            float(row[field]),
            int(row["chart_id"]),
            int(row["head_id"]),
        ),
    )
    if not ordered:
        raise ArtifactSchemaError("operator aggregate has no supported pair")
    weight = 1.0 / len(ordered)
    normalized_weight = weight / math.fsum(weight for _ in ordered)
    cumulative: list[float] = []
    running = 0.0
    for _ in ordered:
        running = math.fsum((running, normalized_weight))
        cumulative.append(running)
    cumulative[-1] = 1.0
    quantile = 0.5
    values = [float(row[field]) for row in ordered]
    if quantile <= cumulative[0]:
        return values[0]
    for index in range(len(ordered) - 1):
        left_mass = cumulative[index]
        right_mass = cumulative[index + 1]
        if left_mass < quantile <= right_mass:
            fraction = (quantile - left_mass) / (right_mass - left_mass)
            return values[index] + fraction * (values[index + 1] - values[index])
    return values[-1]


def validate_artifact_payload(
    name: str, payload: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate one strict synthetic JSON payload and cross-field invariants."""

    schema = artifact_schema(name)
    if schema.binary:
        raise ArtifactSchemaError(f"{name!r} is binary; use validate_bundle_bytes")
    _validate_exact_object(payload, _envelope_fields(schema), path="$")
    body = payload["body"]
    assert isinstance(body, Mapping)
    dependency_hashes = payload["dependency_hashes"]
    assert isinstance(dependency_hashes, Mapping)
    for body_field, dependency_name in BODY_DEPENDENCY_HASH_BINDINGS.get(
        name, {}
    ).items():
        if body[body_field] != dependency_hashes[dependency_name]:
            raise ArtifactSchemaError(
                "body dependency hash mismatch: "
                f"{name}.body.{body_field} must equal "
                f"dependency_hashes[{dependency_name!r}]"
            )

    if name == "preregistration_root.json":
        records = body["bound_artifacts"]
        _require_unique_complete(
            records,
            "name",
            PREREGISTRATION_BOUND_ARTIFACTS,
        )
        assert isinstance(records, list)
        for record in records:
            artifact_name = record["name"]
            if record["artifact_sha256"] != dependency_hashes[artifact_name]:
                raise ArtifactSchemaError(
                    "preregistration bound artifact hash mismatch: "
                    f"{artifact_name}"
                )

    if name == "independence_protocol.json":
        mode = body["selected_mode"]
        expected_field = (
            "template_lineage_id" if mode == "template_lineage" else "semantic_instance_id"
        )
        if body["analysis_unit_field"] != expected_field:
            raise ArtifactSchemaError("analysis-unit field contradicts selected mode")
        if mode == "template_lineage" and body["inference_limitation"] is not None:
            raise ArtifactSchemaError("Mode A inference limitation must be null")
        if mode == "semantic_instance" and body["inference_limitation"] != MODE_B_LIMITATION:
            raise ArtifactSchemaError("Mode B must use the frozen inference limitation")
        selected_eligible = (
            body["mode_a_eligible"]
            if mode == "template_lineage"
            else body["mode_b_eligible"]
        )
        if not selected_eligible:
            raise ArtifactSchemaError("selected independence mode is not eligible")
        observed_counts = (
            body["fit_analysis_unit_count"],
            body["tune_analysis_unit_count"],
            body["development_confirm_analysis_unit_count"],
        )
        if any(observed < required for observed, required in zip(observed_counts, (48, 24, 24))):
            raise ArtifactSchemaError("independence analysis-unit minimum is not met")
    elif name == "split_manifest.json":
        if tuple(body["purpose_identifiers"]) != SPLIT_PURPOSES:
            raise ArtifactSchemaError("split purpose identifiers or order differ")
        roles = [item["partition_role"] for item in body["partitions"]]
        if roles != ["fit_train", "fit_monitor", "tune", "development_confirm"]:
            raise ArtifactSchemaError("partition records must use the frozen order")
        expected_counts = [38, 10, 24, 24]
        if [item["analysis_unit_count"] for item in body["partitions"]] != expected_counts:
            raise ArtifactSchemaError("partition analysis-unit counts differ")
    elif name == "execution_budget.json":
        if tuple(body["optional_skip_order"]) != OPTIONAL_DIAGNOSTIC_SKIP_ORDER:
            raise ArtifactSchemaError("optional diagnostic skip order differs")
    elif name == "capture_invariants.json":
        _require_unique_complete(body["seams"], "seam", O1_SEAMS)
        observed_pass = all(
            record["finite"] and record["shape_contract_passed"]
            for record in body["seams"]
        )
        if body["passed"] != observed_pass:
            raise ArtifactSchemaError("capture invariant result contradicts seam records")
    elif name == "o1_tolerance_protocol.json":
        _require_unique_complete(body["tolerances"], "seam", O1_SEAMS)
        for record in body["tolerances"]:
            if record["tau_abs"] > record["cap_abs"]:
                raise ArtifactSchemaError("O1 absolute tolerance exceeds its cap")
            if record["tau_rel"] > record["cap_rel"]:
                raise ArtifactSchemaError("O1 relative tolerance exceeds its cap")
        if body["protocol_construction_passed"] != all(
            record["calibration_passed"] for record in body["tolerances"]
        ):
            raise ArtifactSchemaError("O1 protocol result contradicts seam records")
    elif name in {"o1_validation_results.json", "o1_complete_fit_results.json"}:
        _require_unique_complete(body["results"], "seam", O1_SEAMS)
        if body["all_seams_passed"] != all(
            record["passed"] for record in body["results"]
        ):
            raise ArtifactSchemaError("O1 aggregate result contradicts seam records")
    elif name == "full_rank_identity_smoke.json":
        calibration = body["calibration_fit_train"]
        validation = body["validation_fit_monitor"]
        head_rows = calibration["head_tolerances"]
        coefficient_rows = calibration["coefficient_tolerances"]
        validation_rows = validation["records"]
        head_ids = [record["head_id"] for record in head_rows]
        if head_ids != sorted(set(head_ids)):
            raise ArtifactSchemaError(
                "full-rank head tolerances must be unique and ordered"
            )
        coefficient_keys = [
            (record["codec_id"], record["chart_id"], record["head_id"])
            for record in coefficient_rows
        ]
        validation_keys = [
            (record["codec_id"], record["chart_id"], record["head_id"])
            for record in validation_rows
        ]
        if len(coefficient_keys) != len(set(coefficient_keys)):
            raise ArtifactSchemaError("full-rank coefficient tolerances repeat a cell")
        if len(validation_keys) != len(set(validation_keys)):
            raise ArtifactSchemaError("full-rank validation repeats a cell")
        if coefficient_keys != sorted(coefficient_keys):
            raise ArtifactSchemaError(
                "full-rank coefficient tolerances must use canonical cell order"
            )
        if validation_keys != sorted(validation_keys):
            raise ArtifactSchemaError(
                "full-rank validation must use canonical cell order"
            )
        if set(validation_keys) != set(coefficient_keys):
            raise ArtifactSchemaError(
                "full-rank validation cells differ from calibrated cells"
            )
        for record in head_rows:
            if record["tau_full_m"] > record["cap_m"]:
                raise ArtifactSchemaError("full-rank message tolerance exceeds its cap")
            if record["tau_full_q"] > record["cap_q"]:
                raise ArtifactSchemaError("full-rank q tolerance exceeds its cap")
            observed_pass = (
                record["observed_max_error_m"] <= record["cap_m"]
                and record["observed_max_error_q"] <= record["cap_q"]
            )
            if record["passed"] != observed_pass:
                raise ArtifactSchemaError(
                    "full-rank head calibration result contradicts its caps"
                )
        for record in coefficient_rows:
            if record["tau_full_c"] > record["cap_c"]:
                raise ArtifactSchemaError(
                    "full-rank coefficient tolerance exceeds its cap"
                )
            if record["passed"] != (
                record["observed_max_error_c"] <= record["cap_c"]
            ):
                raise ArtifactSchemaError(
                    "full-rank coefficient result contradicts its cap"
                )
        for record in validation_rows:
            zero = record["assigned_row_count"] == 0
            if zero != (
                record["validation_status"] == "invalid_zero_validation_support"
            ):
                raise ArtifactSchemaError(
                    "full-rank zero-row status contradicts assigned row count"
                )
            numeric = (
                record["observed_max_error_m"],
                record["observed_max_error_q"],
                record["observed_max_error_c"],
                record["coefficient_completion_mse"],
            )
            if zero and (
                any(value is not None for value in numeric)
                or record["zero_row_reason"] is None
            ):
                raise ArtifactSchemaError(
                    "full-rank zero-row record must retain null observations and a reason"
                )
            if not zero and (
                any(value is None for value in numeric)
                or record["zero_row_reason"] is not None
            ):
                raise ArtifactSchemaError(
                    "full-rank observed row cannot use zero-row sentinels"
                )
        eligible = (
            validation["representation_candidate_status"] == "eligible"
            and all(record["validation_status"] == "passed" for record in validation_rows)
            and all(record["passed"] for record in head_rows)
            and all(record["passed"] for record in coefficient_rows)
        )
        if body["all_eligible_records_passed"] != eligible:
            raise ArtifactSchemaError(
                "full-rank aggregate contradicts calibration or validation records"
            )
    elif name == "codec_protocol.json":
        _require_codec_ladder(body["codec_candidates"])
    elif name == "codec_candidate_manifests.json":
        _require_unique_complete(body["candidates"], "codec_id", CODEC_IDS)
        _require_codec_ladder(body["candidates"])
        for record in body["candidates"]:
            invalid = record["status"] != "eligible"
            if invalid and record["status"] == "invalid_no_positive_operator_reference":
                if record["operator_reference_norm"] is not None or record["tau_R"] is not None:
                    raise ArtifactSchemaError(
                        "zero-positive-reference codec must not manufacture norm/tau_R"
                    )
    elif name == "ceiling_gate_protocol.json":
        _validate_five_gates(body["gate_thresholds"], require_observed_zero=True)
    elif name == "learning_protocol.json":
        if tuple(body["mandatory_learned_models"]) != MANDATORY_LEARNED_MODELS:
            raise ArtifactSchemaError("mandatory learned-model inventory differs")
        if tuple(body["restart_seeds"]) != (1701, 2718, 3141):
            raise ArtifactSchemaError("restart seed inventory differs")
        if tuple(body["betas"]) != (0.9, 0.999):
            raise ArtifactSchemaError("AdamW betas differ")
        _validate_width_table(body["width_table"])
    elif name == "development_eligibility_protocol.json":
        if tuple(body["required_baselines"]) != REQUIRED_BASELINES:
            raise ArtifactSchemaError("required baseline inventory differs")
    elif name in {"o2q_results.json", "o2v_results.json"}:
        _require_unique_complete(body["candidate_panels"], "codec_id", CODEC_IDS)
        observed_passing: list[str] = []
        for candidate in body["candidate_panels"]:
            roles = [panel["partition_role"] for panel in candidate["panels"]]
            if roles != ["fit_monitor", "tune"]:
                raise ArtifactSchemaError("O2 panels must be fit_monitor then tune")
            for panel in candidate["panels"]:
                _validate_five_gates(panel["gate_results"])
                _validate_panel_aggregate(panel)
            candidate_pass = all(
                panel["all_five_passed"] for panel in candidate["panels"]
            )
            if candidate["passes_fit_monitor_and_tune"] != candidate_pass:
                raise ArtifactSchemaError("O2 candidate result contradicts panel results")
            if candidate_pass:
                observed_passing.append(candidate["codec_id"])
        if list(body["passing_codec_ids"]) != observed_passing:
            raise ArtifactSchemaError("O2 passing-codec inventory contradicts candidates")
    elif name == "selected_codec.json":
        expected_codec = {
            "C0": (32, 32, 1e-3),
            "C1": (64, 64, 1e-3),
            "C2": (32, 32, 1e-2),
            "C3": (64, 64, 1e-2),
        }[body["codec_id"]]
        observed_codec = (body["q_rank"], body["v_rank"], body["lambda_ratio"])
        if observed_codec != expected_codec:
            raise ArtifactSchemaError("selected codec parameters contradict codec_id")
        expected_selection_order = (
            "tune_absolute_delta_nll",
            "tune_kl",
            "tune_top1_desc",
            "q_rank",
            "v_rank",
            "regularization_desc",
            "codec_id",
        )
        if tuple(body["selection_order"]) != expected_selection_order:
            raise ArtifactSchemaError("selected codec tie-break order differs")
    elif name in {"m2q_results.json", "m2v_results.json"}:
        for panel in body["panels"]:
            _validate_five_gates(panel["gate_results"])
            _validate_panel_aggregate(panel)
    elif name in {"compact_ambient_oracle.json", "compact_chart_oracle.json"}:
        for panel in body["panels"]:
            full_prompt = panel["full_prompt_shadow"]
            if full_prompt["partition_role"] != panel["partition_role"]:
                raise ArtifactSchemaError(
                    "compact-oracle local and full-prompt panel roles differ"
                )
            _validate_five_gates(full_prompt["gate_results"])
            _validate_panel_aggregate(full_prompt)
    elif name == "j_model_fit_results.json":
        _require_models(body["models"], include_analytic=True)
    elif name == "j_model_tune_results.json":
        _require_models(body["models"], include_analytic=True, allow_optional_absence=True)
    elif name == "all_fit_weight_refit.json":
        if tuple(body["model_ids"]) != ("J2N", "J2+", "J3", "J4"):
            raise ArtifactSchemaError("all-fit refit model inventory differs")
        if body["representation_before_sha256"] != body["representation_after_sha256"]:
            raise ArtifactSchemaError("all-fit model refit changed the representation")
    elif name == "nonselecting_tune_replay.json":
        _require_unique_complete(
            body["models"], "model_id", ("J2N", "J2+", "J3", "J4")
        )
        for record in body["models"]:
            before = float(record["pre_refit_L_bar_down"])
            after = float(record["post_refit_L_bar_down"])
            expected_relative_degradation = (after - before) / max(
                before, 1e-12
            )
            if record["relative_degradation"] != expected_relative_degradation:
                raise ArtifactSchemaError(
                    "tune replay relative_degradation contradicts pre/post losses"
                )
            expected_stable = after <= 1.02 * before
            if record["stable"] is not expected_stable:
                raise ArtifactSchemaError(
                    "tune replay stable flag contradicts the inclusive two-percent rule"
                )
        replay_pass = all(record["stable"] for record in body["models"])
        if body["all_required_models_stable"] != replay_pass:
            raise ArtifactSchemaError("tune replay aggregate contradicts model records")
    elif name == "frozen_B_star.json":
        if tuple(body["tie_order"]) != REQUIRED_BASELINES:
            raise ArtifactSchemaError("B* tie order differs")
    elif name == "candidate_bundle_metadata.json":
        if not body["deterministic_self_test_passed"]:
            raise ArtifactSchemaError("bundle self-test must pass before P9 completes")
        tensor_names = [(item["bundle"], item["name"]) for item in body["tensors"]]
        if len(tensor_names) != len(set(tensor_names)):
            raise ArtifactSchemaError("bundle metadata repeats a tensor name")
    elif name == "development_confirm_results.json":
        _require_unique_complete(body["comparisons"], "baseline", REQUIRED_BASELINES)
        for comparison in body["comparisons"]:
            _validate_rho_record(comparison)
        operator = body["trained_J4_operator_aggregate"]
        if (
            operator["selected_codec_artifact_sha256"]
            != dependency_hashes["selected_codec.json"]
            or operator["representation_numeric_constants_sha256"]
            != dependency_hashes["representation_numeric_constants.json"]
            or operator["codec_candidate_manifests_sha256"]
            != dependency_hashes["codec_candidate_manifests.json"]
        ):
            raise ArtifactSchemaError(
                "trained J4 operator identities differ from direct dependencies"
            )
        if operator["tau_R"] <= 0.0:
            raise ArtifactSchemaError(
                "trained J4 operator tau_R must be strictly positive"
            )
        chart_records = body["active_chart_unit_counts"]
        chart_ids = [row["chart_id"] for row in chart_records]
        if not chart_ids or chart_ids != sorted(set(chart_ids)):
            raise ArtifactSchemaError(
                "development active charts must be nonempty, unique, and ordered"
            )
        chart_unit_counts = {
            row["chart_id"]: row["analysis_unit_count"]
            for row in chart_records
        }
        pairs = operator["analysis_unit_support_by_pair"]
        pair_ids = [(row["chart_id"], row["head_id"]) for row in pairs]
        expected_pair_ids = [
            (chart_id, head_id)
            for chart_id in chart_ids
            for head_id in range(4)
        ]
        if pair_ids != expected_pair_ids:
            raise ArtifactSchemaError(
                "trained J4 operator pairs must be the complete ordered active-chart/head grid"
            )
        if any(
            row["analysis_unit_count"] != chart_unit_counts[row["chart_id"]]
            for row in pairs
        ):
            raise ArtifactSchemaError(
                "trained J4 operator pair unit counts differ from active-chart support"
            )
        supported = [row for row in pairs if row["status"] == "supported"]
        excluded = [
            row for row in pairs if row["status"] == "insufficient_pair_support"
        ]
        if any(row["analysis_unit_count"] < 2 for row in supported) or any(
            row["analysis_unit_count"] >= 2 for row in excluded
        ):
            raise ArtifactSchemaError(
                "trained J4 operator pair status contradicts the two-unit floor"
            )
        metric_fields = (
            "pair_median_ambient_operator_norm",
            "pair_median_H4_surrogate_operator_norm",
            "pair_median_ambient_stable_rank",
        )
        if any(
            any(row[field] is None for field in metric_fields)
            for row in supported
        ) or any(
            any(row[field] is not None for field in metric_fields)
            for row in excluded
        ):
            raise ArtifactSchemaError(
                "trained J4 operator pair metrics contradict support status"
            )
        if (
            operator["supported_chart_head_pair_count"] != len(supported)
            or operator["excluded_chart_head_pair_count"] != len(excluded)
            or operator["zero_ambient_pair_count"]
            != sum(
                row["pair_median_ambient_operator_norm"] == 0.0
                for row in supported
            )
            or operator["positive_ambient_pair_count"]
            != sum(
                row["pair_median_ambient_operator_norm"] > 0.0
                for row in supported
            )
        ):
            raise ArtifactSchemaError(
                "trained J4 operator aggregate counts contradict pair records"
            )
        aggregate_metrics = (
            (
                "aggregate_median_ambient_operator_norm",
                "pair_median_ambient_operator_norm",
            ),
            (
                "aggregate_median_H4_surrogate_operator_norm",
                "pair_median_H4_surrogate_operator_norm",
            ),
            (
                "aggregate_median_ambient_stable_rank",
                "pair_median_ambient_stable_rank",
            ),
        )
        for aggregate_field, pair_field in aggregate_metrics:
            expected = _equal_pair_linear_median(supported, pair_field)
            if not math.isclose(
                float(operator[aggregate_field]),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ArtifactSchemaError(
                    f"trained J4 {aggregate_field} contradicts pair medians"
                )
    elif name == "interaction_mean_ablation.json":
        gamma = body["gamma_interaction"]
        if body["gamma_status"] == "regular" and gamma is None:
            raise ArtifactSchemaError("regular gamma must be numeric")
        if body["gamma_status"] != "regular" and gamma is not None:
            raise ArtifactSchemaError("undefined gamma must be null")
    elif name == "o7_attention_output_completion_results.json":
        _validate_five_gates(body["panel"]["gate_results"])
        _validate_panel_aggregate(body["panel"])
        if body["all_five_passed"] != body["panel"]["all_five_passed"]:
            raise ArtifactSchemaError("O7 aggregate result contradicts its panel")
    elif name == "development_eligibility_result.json":
        conjunction = all(
            bool(body[key])
            for key in (
                "o2q_replay_passed",
                "all_J4_comparisons_passed",
                "interaction_mean_passed",
                "o7_all_five_passed",
                "out_of_support_passed",
                "independent_support_passed",
            )
        )
        if body["development_eligible"] != conjunction:
            raise ArtifactSchemaError("development eligibility is not the frozen conjunction")
        if conjunction and body["terminal_if_finalized"] != "development_complete_no_eligible_guard":
            raise ArtifactSchemaError("eligible local development must end no-eligible-guard")
    elif name == "development_candidate_manifest.json":
        mode = body["independence_mode"]
        if mode == "template_lineage" and body["inference_limitation"] is not None:
            raise ArtifactSchemaError("Mode A candidate limitation must be null")
        if mode == "semantic_instance" and body["inference_limitation"] != MODE_B_LIMITATION:
            raise ArtifactSchemaError("Mode B candidate must preserve the limitation")
        _require_unique_complete(
            body["partition_commitments"],
            "partition_role",
            ("fit_train", "fit_monitor", "tune", "development_confirm"),
        )
        _require_unique_complete(
            body["capture_manifest_sha256s"],
            "name",
            ("fit", "tune", "development_confirm"),
        )
        capture_dependencies = {
            "fit": "capture_shard_manifest_fit.json",
            "tune": "capture_shard_manifest_tune.json",
            "development_confirm": (
                "capture_shard_manifest_development_confirm.json"
            ),
        }
        for record in body["capture_manifest_sha256s"]:
            role = record["name"]
            dependency_name = capture_dependencies[role]
            if record["artifact_sha256"] != dependency_hashes[dependency_name]:
                raise ArtifactSchemaError(
                    "candidate capture manifest hash mismatch: "
                    f"{dependency_name}"
                )
        if not body.get("contains_guard_claim_or_result") is False:
            raise ArtifactSchemaError("development candidate contains guard state")
        forbidden = {
            "artifact_byte_sha256",
            "guard_claim",
            "guard_result",
            "guard_claim_sha256",
            "guard_result_sha256",
        }
        if forbidden & set(payload):
            raise ArtifactSchemaError("development candidate contains a forbidden field")
        expected = development_candidate_hash(payload)
        if payload["development_candidate_hash"] != expected:
            raise ArtifactSchemaError("development_candidate_hash mismatch")
    return payload


def _require_unique_complete(
    records: object, key: str, expected: Sequence[str]
) -> None:
    assert isinstance(records, list)
    observed = [record[key] for record in records]
    if tuple(observed) != tuple(expected):
        raise ArtifactSchemaError(
            f"{key} inventory/order differs: expected {tuple(expected)!r}"
        )


def _require_codec_ladder(records: object) -> None:
    assert isinstance(records, list)
    expected = (
        ("C0", 32, 32, 1e-3),
        ("C1", 64, 64, 1e-3),
        ("C2", 32, 32, 1e-2),
        ("C3", 64, 64, 1e-2),
    )
    observed = tuple(
        (item["codec_id"], item["q_rank"], item["v_rank"], item["lambda_ratio"])
        for item in records
    )
    if observed != expected:
        raise ArtifactSchemaError("codec ladder differs from C0-C3")


def _validate_five_gates(records: object, *, require_observed_zero: bool = False) -> None:
    assert isinstance(records, list)
    expected = (
        ("absolute_global_delta_nll_per_token", 0.05, "<="),
        ("global_kl_per_token", 0.05, "<="),
        ("global_top1_agreement", 0.95, ">="),
        ("p90_absolute_prompt_delta_nll_per_token", 0.10, "<="),
        ("p10_prompt_top1_agreement", 0.90, ">="),
    )
    observed = tuple(
        (item["gate_id"], item["threshold"], item["comparison"])
        for item in records
    )
    if observed != expected:
        raise ArtifactSchemaError("five-gate inventory/order/threshold differs")
    if require_observed_zero and any(item["observed"] != 0 for item in records):
        raise ArtifactSchemaError("gate protocol placeholders must use observed=0")
    if not require_observed_zero:
        for item in records:
            evaluated = (
                item["observed"] <= item["threshold"]
                if item["comparison"] == "<="
                else item["observed"] >= item["threshold"]
            )
            if item["passed"] != evaluated:
                raise ArtifactSchemaError("gate pass flag contradicts observed value")


def _validate_panel_aggregate(panel: Mapping[str, object]) -> None:
    records = panel["gate_results"]
    assert isinstance(records, list)
    if panel["all_five_passed"] != all(record["passed"] for record in records):
        raise ArtifactSchemaError("panel aggregate contradicts five gate results")


def _validate_width_table(records: object) -> None:
    assert isinstance(records, list)
    expected = (
        (32, 2848, 21, 22, 2859, 29, 2845),
        (64, 5184, 26, 26, 5172, 40, 5224),
    )
    observed = tuple(
        (
            item["q_rank"],
            item["J4_parameters"],
            item["J2plus_width_z"],
            item["J2plus_width_r"],
            item["J2plus_parameters"],
            item["J3_width"],
            item["J3_parameters"],
        )
        for item in records
    )
    if observed != expected:
        raise ArtifactSchemaError("parameter-matched width table differs")


def _require_models(
    records: object,
    *,
    include_analytic: bool,
    allow_optional_absence: bool = False,
) -> None:
    assert isinstance(records, list)
    ids = [item["model_id"] for item in records]
    required = ["J0", "J1", "J2N", "J2+", "J3", "J4"] if include_analytic else list(MANDATORY_LEARNED_MODELS)
    if any(model not in ids for model in required):
        raise ArtifactSchemaError("mandatory J-model result is absent")
    if len(ids) != len(set(ids)):
        raise ArtifactSchemaError("J-model inventory contains duplicates")
    if not allow_optional_absence and "J2" in ids:
        # Presence is permitted only as an explicit optional record.
        record = records[ids.index("J2")]
        if record["mandatory"]:
            raise ArtifactSchemaError("dedicated J2 must remain optional")


def _validate_rho_record(record: Mapping[str, object]) -> None:
    rho = record["rho"]
    status = record["rho_status"]
    if status == "regular" and rho is None:
        raise ArtifactSchemaError("regular rho must be numeric")
    if status != "regular" and rho is not None:
        raise ArtifactSchemaError("floor-case rho must be null")
    if float(record["valid_rho_replicate_fraction"]) < 0.95 and record["comparison_passed"]:
        raise ArtifactSchemaError("comparison cannot pass with <95% valid rho replicates")


def build_artifact_payload(
    name: str,
    *,
    campaign_id: str,
    dependency_hashes: Mapping[str, str],
    body: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate one synthetic-only scientific payload."""

    schema = artifact_schema(name)
    if schema.binary:
        raise ArtifactSchemaError(f"{name!r} is binary; use validate_bundle_bytes")
    payload: dict[str, object] = {
        "schema_id": schema.schema_id,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "artifact_name": name,
        "stage": schema.stage.value,
        "artifact_family": schema.family.value,
        "evidence_scope": EVIDENCE_SCOPE,
        "synthetic_evidence": True,
        "scientific_claim_eligible": False,
        "target_model_execution_performed": False,
        "protected_panel_accessed": False,
        "dependency_hashes": dict(dependency_hashes),
        "body": dict(body),
    }
    if schema.family is ArtifactFamily.CANDIDATE:
        payload["development_candidate_hash"] = "0" * 64
        payload["development_candidate_hash"] = development_candidate_hash(payload)
    validate_artifact_payload(name, payload)
    return payload


def _family_builder(family: ArtifactFamily, name: str, **kwargs: object) -> dict[str, object]:
    schema = artifact_schema(name)
    if schema.family is not family:
        raise ArtifactSchemaError(
            f"{name!r} belongs to {schema.family.value}, not {family.value}"
        )
    return build_artifact_payload(name, **kwargs)  # type: ignore[arg-type]


def build_independence_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.INDEPENDENCE, name, **kwargs)


def build_split_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.SPLIT, name, **kwargs)


def build_o1_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.O1, name, **kwargs)


def build_o2_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.O2, name, **kwargs)


def build_m2_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.M2, name, **kwargs)


def build_representation_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.REPRESENTATION, name, **kwargs)


def build_j_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.J_MODEL, name, **kwargs)


def build_development_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.DEVELOPMENT, name, **kwargs)


def build_o7_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.O7, name, **kwargs)


def build_bundle_metadata_artifact(name: str, **kwargs: object) -> dict[str, object]:
    return _family_builder(ArtifactFamily.BUNDLE, name, **kwargs)


def build_development_candidate_manifest(**kwargs: object) -> dict[str, object]:
    return _family_builder(
        ArtifactFamily.CANDIDATE,
        "development_candidate_manifest.json",
        **kwargs,
    )


def validate_bundle_bytes(name: str, content: bytes) -> str:
    """Strictly validate a nonempty safetensors container and return its SHA-256.

    This verifies container structure only.  Tensor names/shapes/dtypes and
    candidate semantics are bound by ``candidate_bundle_metadata.json``.
    """

    schema = artifact_schema(name)
    if not schema.binary:
        raise ArtifactSchemaError(f"{name!r} is not a binary bundle")
    if not isinstance(content, bytes) or len(content) < 10:
        raise ArtifactSchemaError("bundle must be nonempty safetensors bytes")
    header_length = struct.unpack("<Q", content[:8])[0]
    if header_length < 2 or header_length > len(content) - 8:
        raise ArtifactSchemaError("safetensors header length is invalid")
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactSchemaError(
                    f"safetensors header repeats object key {key!r}"
                )
            result[key] = value
        return result

    try:
        header = json.loads(
            content[8 : 8 + header_length].decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactSchemaError(
                    f"safetensors header contains nonfinite value {value!r}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactSchemaError("safetensors header is not strict UTF-8 JSON") from error
    if not isinstance(header, dict) or not header:
        raise ArtifactSchemaError("safetensors header must be a nonempty object")
    if "__metadata__" in header:
        raise ArtifactSchemaError("free-form safetensors metadata is forbidden")
    data_length = len(content) - 8 - header_length
    dtype_widths = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    intervals: list[tuple[int, int]] = []
    for tensor_name, descriptor in header.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactSchemaError("safetensors tensor name is invalid")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ArtifactSchemaError("safetensors descriptor keys differ")
        if descriptor["dtype"] not in dtype_widths:
            raise ArtifactSchemaError("safetensors dtype is invalid")
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ArtifactSchemaError("safetensors shape is invalid")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
            or offsets[1] > data_length
        ):
            raise ArtifactSchemaError("safetensors offsets are invalid")
        element_count = math.prod(shape)
        if offsets[1] - offsets[0] != element_count * dtype_widths[descriptor["dtype"]]:
            raise ArtifactSchemaError("safetensors tensor byte length is invalid")
        intervals.append((offsets[0], offsets[1]))
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != data_length:
        raise ArtifactSchemaError("safetensors data region is not fully bound")
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ArtifactSchemaError("safetensors data ranges overlap or contain gaps")
    return hashlib.sha256(content).hexdigest()


TERMINAL_STATES: Final = frozenset(
    {
        "development_complete_independence_unavailable",
        "development_complete_independent_data_unavailable",
        "development_complete_insufficient_support",
        "development_complete_refit_instability",
        "development_complete_resource_limited",
        "development_complete_no_eligible_guard",
        "stalled_q_chart_ceiling",
        "stalled_q_geometry_mismatch",
        "stalled_attention_chart_capacity",
        "development_complete_negative_j4",
        "development_complete_attention_output_not_ready",
        "invalid_campaign_source_changed",
        "invalid_campaign_freshness",
        "invalid_campaign_integrity",
        "invalid_campaign_measurement_contract",
    }
)


def build_terminal_decision(
    *,
    campaign_id: str,
    terminal_state: str,
    reason_code: str,
    last_completed_prefix: str | None,
    active_prefix: str,
    durably_reached_artifacts: Sequence[str],
) -> dict[str, object]:
    """Build a strict synthetic decision for the ledger terminal API.

    This is not ``terminal_state.json`` and must not be registered under that
    ledger-owned name.  It is a typed hand-off record for orchestration tests.
    """

    if terminal_state not in TERMINAL_STATES:
        raise ArtifactSchemaError("terminal state is outside the frozen vocabulary")
    if not campaign_id or not reason_code:
        raise ArtifactSchemaError("campaign_id and reason_code must be nonempty")
    stages = tuple(stage.value for stage in Stage)
    if active_prefix not in stages:
        raise ArtifactSchemaError("active prefix is invalid")
    if last_completed_prefix is not None and last_completed_prefix not in stages:
        raise ArtifactSchemaError("last completed prefix is invalid")
    if last_completed_prefix is not None and stages.index(last_completed_prefix) >= stages.index(active_prefix):
        raise ArtifactSchemaError("last completed prefix must precede active prefix")
    reached = list(durably_reached_artifacts)
    if len(reached) != len(set(reached)) or any(not item for item in reached):
        raise ArtifactSchemaError("durably reached artifact inventory is invalid")
    return {
        "schema_id": f"{SCHEMA_DOMAIN}/synthetic-terminal-decision/v1",
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "evidence_scope": EVIDENCE_SCOPE,
        "synthetic_evidence": True,
        "scientific_claim_eligible": False,
        "target_model_execution_performed": False,
        "protected_panel_accessed": False,
        "terminal_state": terminal_state,
        "reason_code": reason_code,
        "last_completed_prefix": last_completed_prefix,
        "active_prefix": active_prefix,
        "durably_reached_artifacts": reached,
        "scientific_result_exists": False,
        "guard_result_exists": False,
    }


__all__ = [
    "ARTIFACT_DEPENDENCIES",
    "ARTIFACT_SCHEMAS",
    "BODY_DEPENDENCY_HASH_BINDINGS",
    "CONDITIONAL_CAPTURE_ARTIFACTS",
    "EVIDENCE_SCOPE",
    "FULL_PROMPT_LABEL",
    "LEDGER_OWNED_ARTIFACTS",
    "MODE_B_LIMITATION",
    "PREREGISTRATION_BOUND_ARTIFACTS",
    "PROTOCOL_ID",
    "STAGE_ARTIFACT_ORDER",
    "TERMINAL_STATES",
    "ArtifactFamily",
    "ArtifactSchema",
    "ArtifactSchemaError",
    "Stage",
    "artifact_dependencies",
    "artifact_schema",
    "build_artifact_payload",
    "build_bundle_metadata_artifact",
    "build_development_artifact",
    "build_development_candidate_manifest",
    "build_independence_artifact",
    "build_j_artifact",
    "build_m2_artifact",
    "build_o1_artifact",
    "build_o2_artifact",
    "build_o7_artifact",
    "build_representation_artifact",
    "build_split_artifact",
    "build_terminal_decision",
    "development_candidate_hash",
    "synthetic_body_template",
    "validate_artifact_payload",
    "validate_bundle_bytes",
]
