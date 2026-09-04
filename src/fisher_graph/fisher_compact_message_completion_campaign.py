"""End-to-end synthetic qualification campaign for the P1--P12 protocol.

This orchestrator is deliberately sealed away from any corpus or local-run
directory.  It builds the deterministic public fixture in memory, executes
the row/capture ledger and qualification kernels, and publishes only
``synthetic_qualification_only`` artifacts below a caller-selected fresh
directory.  It can therefore qualify implementation mechanics, but it cannot
create a scientific result or eligible guard.

``quick`` uses shortened optimizer schedules for development feedback.
``exact`` selects the frozen initial, diagnostic, and all-fit-refit schedules.
Both profiles retain the same strict artifact schemas and nonclaim envelope;
the selected profile and actual optimizer counts are returned separately so a
quick run cannot be mistaken for an exact training execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import hashlib
import json
import locale
import os
import platform
from pathlib import Path
import stat
import subprocess
import sys
from typing import Final, Literal

import numpy as np
import torch

from .fisher_compact_message_completion_artifacts import (
    EVIDENCE_SCOPE,
    FULL_PROMPT_LABEL,
    LEDGER_OWNED_ARTIFACTS,
    MODE_B_LIMITATION,
    PREREGISTRATION_BOUND_ARTIFACTS,
    REQUIRED_BASELINES,
    STAGE_ARTIFACT_ORDER,
    Stage,
    artifact_dependencies,
    artifact_schema,
    build_artifact_payload,
    synthetic_body_template,
    validate_artifact_payload,
    validate_bundle_bytes,
)
from .fisher_compact_message_completion_capture import (
    encode_safetensors,
    publish_bytes_exclusive,
    write_capture_shard,
)
from .fisher_compact_message_completion_diagnostics import (
    build_full_rank_identity_smoke,
    evaluate_compact_oracles,
    evaluate_direct_full_rank_identity_null,
    evaluate_m2_diagnostics,
    fit_mlp_codec_candidates,
)
from .fisher_compact_message_completion_models import (
    FULL_DIAGNOSTIC_TRAINING_CONFIG,
    FULL_INITIAL_TRAINING_CONFIG,
    FULL_REFIT_CONFIG,
    ModelBankEvaluation,
    ModelBankFit,
    LoadedWeightsOnlyBundles,
    build_weights_only_bundles,
    evaluate_bank,
    evaluate_interaction_mean,
    fit_initial_model_banks,
    fit_refit_model_banks,
    freeze_epsilon_down_from_fit_train,
    load_weights_only_bundles,
    model_comparisons,
    self_test_weights_only_bundles,
    trained_j4_operator_aggregate,
)
from .fisher_compact_message_completion_protocol import (
    AuthorizationMode,
    AuthorizationSpec,
    CampaignPrefix,
    ProtocolLedger,
    ProtocolStateError,
    SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE,
    SupportFailureReason,
    SupportFailureSpec,
    TerminalState,
    artifact_sha256,
    build_shard_plan,
    validate_reached_prefix_closure,
    verify_capture_shard_files,
)
from .fisher_compact_message_completion_power import (
    SupportedChartMembership,
    project_development_power,
)
from .fisher_compact_message_completion_qualification import (
    DevelopmentSupportAssessment,
    O1Qualification,
    PreparedCompletionPanel,
    SelectionGeometry,
    assess_development_support,
    build_o1_qualification,
    evaluate_o2,
    evaluate_o2q_replay,
    evaluate_fidelity,
    fidelity_gate_records,
    fit_selection_geometry,
    prepare_completion_panel,
    subset_partition_by_units,
)
from .fisher_compact_message_completion_representation import (
    PreTuneRepresentationQualification,
    build_pre_tune_representation_qualification,
)
from .fisher_compact_message_completion_science import (
    CompletionTrainingConfig,
    FidelityMetrics,
    O2CodecEvaluation,
    OperatorAggregateSummary,
    bank_coefficient_loss,
    primary_downstream_metric,
)
from .fisher_compact_message_completion_splits import (
    SplitUnit,
    StratumDecision,
    StratumLabel,
    choose_strata,
)
from .fisher_compact_message_completion_synthetic import (
    SYNTHETIC_EVIDENCE_SCOPE,
    SyntheticCampaignConfig,
    SyntheticCampaignData,
    SyntheticPartition,
    build_synthetic_campaign,
    synthetic_row_derivation_protocol_body,
    synthetic_row_derivation_protocol_sha256,
)


__all__ = [
    "CampaignIntegrationError",
    "CampaignOutputError",
    "CampaignProfile",
    "SyntheticCampaignResult",
    "run_synthetic_campaign",
    "training_profile",
]


CampaignProfile = Literal["quick", "exact"]
_CAMPAIGN_ID: Final = "fisher-cmc-synthetic-qualification-v2"
_CAPTURE_SCHEMA_HASH: Final = hashlib.sha256(
    b"fisher-cmc/synthetic-capture-schema/v1"
).hexdigest()
_INITIAL_RESTART_SEEDS: Final = (1701, 2718, 3141)
_REFIT_SEED: Final = 8_675_309
_MAXIMUM_CAMPAIGN_OPTIMIZER_UPDATES: Final = 40_000
_FORBIDDEN_OUTPUT_COMPONENTS: Final = frozenset(
    {".local-runs", "corpus", "corpora"}
)
_MODEL_ORDER: Final = ("J0", "J1", "J2N", "J2+", "J3", "J4")
_EXACT_INITIAL_LITERAL: Final = CompletionTrainingConfig(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    optimizer_epsilon=1e-8,
    weight_decay=1e-4,
    gradient_clip_norm=1.0,
    warmup_steps=100,
    batch_size=256,
    maximum_steps=2_000,
    minimum_steps=200,
    evaluation_interval=25,
    early_stopping_patience=200,
    target_kind="completion_delta",
    early_stopping=True,
)
_EXACT_DIAGNOSTIC_LITERAL: Final = CompletionTrainingConfig(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    optimizer_epsilon=1e-8,
    weight_decay=1e-4,
    gradient_clip_norm=1.0,
    warmup_steps=100,
    batch_size=256,
    maximum_steps=1_000,
    minimum_steps=200,
    evaluation_interval=25,
    early_stopping_patience=200,
    target_kind="completion_delta",
    early_stopping=True,
)
_EXACT_REFIT_LITERAL: Final = CompletionTrainingConfig(
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    optimizer_epsilon=1e-8,
    weight_decay=1e-4,
    gradient_clip_norm=1.0,
    warmup_steps=100,
    batch_size=256,
    maximum_steps=1_000,
    minimum_steps=1_000,
    evaluation_interval=25,
    early_stopping_patience=1_000,
    target_kind="completion_delta",
    early_stopping=False,
)
class CampaignOutputError(ValueError):
    """The requested output root is unsafe or would overwrite prior work."""


class CampaignIntegrationError(RuntimeError):
    """Two frozen protocol layers cannot currently accept the same artifact."""


def _development_terminal_decision(
    *,
    local_comparisons_passed: bool,
    interaction_mean_passed: bool,
    o2q_replay_passed: bool,
    o7_passed: bool,
    support_passed: bool,
) -> tuple[str, str]:
    """Apply the frozen development-terminal precedence."""

    if not o2q_replay_passed or not o7_passed:
        return (
            "development_complete_attention_output_not_ready",
            "o2q_or_o7_fidelity_gate_failed",
        )
    if not local_comparisons_passed or not interaction_mean_passed:
        return (
            "development_complete_negative_j4",
            "development_screen_conjunction_failed",
        )
    if not support_passed:
        return (
            "development_complete_insufficient_support",
            "development_support_requirement_failed",
        )
    return (
        "development_complete_no_eligible_guard",
        "all_six_development_screens_passed",
    )


def _development_support_failure_spec(
    *,
    campaign_id: str,
    campaign: SyntheticCampaignData,
    representation: PreTuneRepresentationQualification,
    support: DevelopmentSupportAssessment,
) -> SupportFailureSpec:
    """Bind the measured development support failure to the J3 receipt."""

    if support.development_support_passed:
        raise ProtocolStateError(
            "cannot build a support-failure receipt for a passing panel"
        )
    materially_active_chart_ids = {
        chart_id
        for chart_id, fraction in support.chart_token_fractions
        if fraction > 0.10
    }
    chart_counts = {
        f"chart_{chart_id}": count
        for chart_id, count in support.chart_analysis_unit_counts
        if chart_id in materially_active_chart_ids
    }
    expected_head_ids = tuple(
        f"head_{head_id}" for head_id in range(campaign.config.head_count)
    )
    chart_head_counts = {
        f"chart_{chart_id}__head_{head_id}": count
        for chart_id, count in support.chart_analysis_unit_counts
        if chart_id in materially_active_chart_ids
        for head_id in range(campaign.config.head_count)
    }
    reason = (
        SupportFailureReason.INSUFFICIENT_INDEPENDENT_CHART_SUPPORT
        if not support.independent_support_passed
        else SupportFailureReason.EXCESSIVE_OUT_OF_SUPPORT_RATE
    )
    return SupportFailureSpec(
        campaign_id=campaign_id,
        partition_role="development_confirm",
        last_completed_prefix=CampaignPrefix.P11,
        active_prefix=CampaignPrefix.P12,
        support_stage="development_confirm_post_evaluation_support",
        reason_code=reason,
        attempted_locator_K_values=representation.attempted_locator_k_values,
        attempted_representation_or_codec_ids=tuple(
            codec.codec_id for codec in representation.codec_qualifications
        ),
        analysis_unit_counts={
            "development_confirm": support.evaluable_analysis_unit_count
        },
        sequence_counts={
            "development_confirm": len(
                {
                    item.canonical_sequence_hash
                    for item in campaign.development_confirm.metadata
                }
            )
        },
        chart_counts=chart_counts,
        chart_head_validation_counts=chart_head_counts,
        active_chart_ids=tuple(chart_counts),
        expected_head_ids=expected_head_ids,
        fallback_attempts=(
            ("lower_K_locator_selected",)
            if representation.lower_k_fallback_used
            else ()
        ),
        out_of_support_token_count=(
            support.out_of_support_token_count
            if reason is SupportFailureReason.EXCESSIVE_OUT_OF_SUPPORT_RATE
            else None
        ),
        evaluated_token_count=(
            campaign.development_confirm.row_index.row_count
            if reason is SupportFailureReason.EXCESSIVE_OUT_OF_SUPPORT_RATE
            else None
        ),
    )


def _development_operator_support_failure_spec(
    *,
    campaign_id: str,
    campaign: SyntheticCampaignData,
    representation: PreTuneRepresentationQualification,
    support: DevelopmentSupportAssessment,
    codec_id: str,
) -> SupportFailureSpec:
    """Close P10/P11 when I10 has no two-unit chart/head pair."""

    chart_counts = {
        f"chart_{chart_id}": count
        for chart_id, count in support.chart_analysis_unit_counts
    }
    if not chart_counts or any(count >= 2 for count in chart_counts.values()):
        raise ProtocolStateError(
            "operator support closure requires every chart below two units"
        )
    expected_head_ids = tuple(
        f"head_{head_id}" for head_id in range(campaign.config.head_count)
    )
    return SupportFailureSpec(
        campaign_id=campaign_id,
        partition_role="development_confirm",
        last_completed_prefix=CampaignPrefix.P10,
        active_prefix=CampaignPrefix.P11,
        support_stage="development_confirm_chart_head_support",
        reason_code=SupportFailureReason.INSUFFICIENT_INDEPENDENT_CHART_SUPPORT,
        attempted_locator_K_values=representation.attempted_locator_k_values,
        attempted_representation_or_codec_ids=(codec_id,),
        analysis_unit_counts={
            "development_confirm": support.evaluable_analysis_unit_count
        },
        sequence_counts={
            "development_confirm": len(
                {
                    item.canonical_sequence_hash
                    for item in campaign.development_confirm.metadata
                }
            )
        },
        chart_counts=chart_counts,
        chart_head_validation_counts={
            f"{chart_id}__{head_id}": count
            for chart_id, count in chart_counts.items()
            for head_id in expected_head_ids
        },
        active_chart_ids=tuple(chart_counts),
        expected_head_ids=expected_head_ids,
        fallback_attempts=(
            ("lower_K_locator_selected",)
            if representation.lower_k_fallback_used
            else ()
        ),
    )


def _trained_operator_aggregate_body(
    aggregate: OperatorAggregateSummary,
    *,
    codec_id: str,
    tau_r: float,
    selected_codec_sha256: str,
    numeric_constants_sha256: str,
    codec_manifests_sha256: str,
) -> dict[str, object]:
    """Serialize the exact two-stage I10 trained-J4 summary."""

    return {
        "panel": "development_confirm",
        "model_id": "J4",
        "codec_id": codec_id,
        "tau_R": tau_r,
        "selected_codec_artifact_sha256": selected_codec_sha256,
        "representation_numeric_constants_sha256": numeric_constants_sha256,
        "codec_candidate_manifests_sha256": codec_manifests_sha256,
        "aggregate_median_ambient_operator_norm": (
            aggregate.aggregate_median_ambient_operator_norm
        ),
        "aggregate_median_H4_surrogate_operator_norm": (
            aggregate.aggregate_median_h4_surrogate_operator_norm
        ),
        "aggregate_median_ambient_stable_rank": (
            aggregate.aggregate_median_ambient_stable_rank
        ),
        "supported_chart_head_pair_count": (
            aggregate.supported_chart_head_pair_count
        ),
        "excluded_chart_head_pair_count": (
            aggregate.excluded_chart_head_pair_count
        ),
        "zero_ambient_pair_count": aggregate.zero_ambient_pair_count,
        "positive_ambient_pair_count": aggregate.positive_ambient_pair_count,
        "analysis_unit_support_by_pair": [
            {
                "chart_id": row.chart_id,
                "head_id": row.head_id,
                "analysis_unit_count": row.analysis_unit_count,
                "status": row.status,
                "pair_median_ambient_operator_norm": (
                    row.median_ambient_operator_norm
                ),
                "pair_median_H4_surrogate_operator_norm": (
                    row.median_h4_surrogate_operator_norm
                ),
                "pair_median_ambient_stable_rank": (
                    row.median_ambient_operator_stable_rank
                ),
            }
            for row in aggregate.pair_summaries
        ],
    }


def _pre_tune_operator_support_failure_spec(
    *,
    campaign_id: str,
    campaign: SyntheticCampaignData,
    representation: PreTuneRepresentationQualification,
    attempted_codec_ids: Sequence[str],
) -> SupportFailureSpec:
    """Bind J4's exhausted positive-reference fallback to the P3/P4 receipt."""

    if not representation.operator_reference_fallback_exhausted:
        raise ProtocolStateError(
            "cannot build an operator support receipt before lower-K exhaustion"
        )
    pair_summaries = representation.codec_qualifications[0].operator_pair_summaries
    chart_head_counts = {
        f"chart_{summary.chart_id}__head_{summary.head_id}": (
            summary.analysis_unit_count
        )
        for summary in pair_summaries
    }
    chart_counts: dict[str, int] = {}
    for summary in pair_summaries:
        chart_id = f"chart_{summary.chart_id}"
        existing = chart_counts.setdefault(chart_id, summary.analysis_unit_count)
        if existing != summary.analysis_unit_count:
            raise ProtocolStateError(
                "operator support counts disagree across heads in one chart"
            )
    return SupportFailureSpec(
        campaign_id=campaign_id,
        partition_role="fit",
        last_completed_prefix=CampaignPrefix.P3,
        active_prefix=CampaignPrefix.P4,
        support_stage="per_codec_operator_reference_construction",
        reason_code=SupportFailureReason.NO_POSITIVE_OPERATOR_REFERENCE,
        attempted_locator_K_values=representation.attempted_locator_k_values,
        attempted_representation_or_codec_ids=attempted_codec_ids,
        analysis_unit_counts={
            "fit_train": len(campaign.fit_split.right_units),
        },
        sequence_counts={
            "fit_train": len(
                {
                    item.canonical_sequence_hash
                    for item in campaign.fit.metadata
                    if item.analysis_unit_commitment in campaign.fit_split.right_units
                }
            ),
        },
        chart_counts=chart_counts,
        chart_head_validation_counts=chart_head_counts,
        active_chart_ids=tuple(chart_counts),
        expected_head_ids=tuple(
            f"head_{head_id}" for head_id in range(campaign.config.head_count)
        ),
        fallback_attempts=(
            "lower_K_locator_selected",
            "lower_K_codecs_reconstructed",
            "lower_K_operator_references_checked",
        ),
    )


@dataclass(frozen=True, slots=True)
class _TrainingProfile:
    initial: CompletionTrainingConfig
    diagnostic: CompletionTrainingConfig
    refit: CompletionTrainingConfig
    exact_contract: bool


@dataclass(frozen=True, slots=True)
class SyntheticCampaignResult:
    """Compact, nonclaim summary of one synthetic qualification execution."""

    campaign_id: str
    profile: CampaignProfile
    exact_training_contract: bool
    output_directory: Path
    ledger_path: Path
    selected_codec_id: str | None
    frozen_baseline_id: str | None
    o1_passed: bool
    o2q_passing_codec_ids: tuple[str, ...]
    initial_optimizer_updates: int
    refit_optimizer_updates: int
    development_eligible: bool | None
    artifact_schema_traversal_complete: bool
    p12_schema_exercised: bool
    candidate_manifest_registered: bool
    last_reached_prefix: str
    development_stratum_initial_scheme: str | None
    development_stratum_initial_counts: tuple[tuple[str, int], ...]
    development_stratum_final_scheme: str | None
    development_stratum_final_counts: tuple[tuple[str, int], ...]
    development_stratum_coarsening_reason: str | None
    development_o2q_replay_passed: bool | None
    development_evaluable_analysis_unit_count: int | None
    development_chart_analysis_unit_counts: tuple[tuple[int, int], ...]
    development_chart_token_fractions: tuple[tuple[int, float], ...]
    development_out_of_support_rate: float | None
    development_out_of_support_passed: bool | None
    development_independent_support_passed: bool | None
    development_underpowered_chart_ids: tuple[int, ...]
    development_materially_underpowered_chart_ids: tuple[int, ...]
    scientific_readiness: bool
    readiness_blockers: tuple[str, ...]
    terminal_state: str | None
    final_experiment_manifest_sha256: str | None
    inventory_entry_count: int
    scientific_claim_eligible: bool = False
    protected_panel_accessed: bool = False


def training_profile(profile: CampaignProfile) -> _TrainingProfile:
    """Resolve the quick development or exact frozen optimizer schedules."""

    if profile == "exact":
        if (
            FULL_INITIAL_TRAINING_CONFIG != _EXACT_INITIAL_LITERAL
            or FULL_DIAGNOSTIC_TRAINING_CONFIG != _EXACT_DIAGNOSTIC_LITERAL
            or FULL_REFIT_CONFIG != _EXACT_REFIT_LITERAL
        ):
            raise CampaignIntegrationError(
                "model full-training constants drifted from the literal exact contract"
            )
        return _TrainingProfile(
            FULL_INITIAL_TRAINING_CONFIG,
            FULL_DIAGNOSTIC_TRAINING_CONFIG,
            FULL_REFIT_CONFIG,
            True,
        )
    if profile != "quick":
        raise ValueError("profile must be 'quick' or 'exact'")
    common = CompletionTrainingConfig(
        warmup_steps=1,
        batch_size=32,
        maximum_steps=2,
        minimum_steps=2,
        evaluation_interval=1,
        early_stopping_patience=2,
    )
    refit = CompletionTrainingConfig(
        warmup_steps=1,
        batch_size=32,
        maximum_steps=2,
        minimum_steps=2,
        evaluation_interval=1,
        early_stopping_patience=2,
        early_stopping=False,
    )
    return _TrainingProfile(common, common, refit, False)


def _prepare_output_root(output_directory: str | os.PathLike[str]) -> Path:
    root = Path(output_directory)
    if not os.fspath(root):
        raise CampaignOutputError("output directory must not be empty")
    absolute = root.expanduser().resolve(strict=False)
    if any(
        component.casefold() in _FORBIDDEN_OUTPUT_COMPONENTS
        for component in absolute.parts
    ):
        raise CampaignOutputError(
            "synthetic output cannot be placed in a corpus or .local-runs directory"
        )
    if absolute.exists() or absolute.is_symlink():
        raise CampaignOutputError(
            f"refusing to overwrite existing output directory {absolute}"
        )
    absolute.mkdir(mode=0o700, parents=True, exist_ok=False)
    return absolute


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_SOURCE_INVENTORY_RELATIVE_PATHS: Final = (
    "pyproject.toml",
    "src/fisher_graph/__init__.py",
    "src/fisher_graph/fisher_compact_message_completion_artifacts.py",
    "src/fisher_graph/fisher_compact_message_completion_campaign.py",
    "src/fisher_graph/fisher_compact_message_completion_capture.py",
    "src/fisher_graph/fisher_compact_message_completion_diagnostics.py",
    "src/fisher_graph/fisher_compact_message_completion_models.py",
    "src/fisher_graph/fisher_compact_message_completion_power.py",
    "src/fisher_graph/fisher_compact_message_completion_probe_v2.py",
    "src/fisher_graph/fisher_compact_message_completion_protocol.py",
    "src/fisher_graph/fisher_compact_message_completion_qualification.py",
    "src/fisher_graph/fisher_compact_message_completion_representation.py",
    "src/fisher_graph/fisher_compact_message_completion_science.py",
    "src/fisher_graph/fisher_compact_message_completion_splits.py",
    "src/fisher_graph/fisher_compact_message_completion_synthetic.py",
    "src/fisher_graph_compact_message_completion_synthetic_qualification.py",
    "tests/test_fisher_compact_message_completion_artifacts.py",
    "tests/test_fisher_compact_message_completion_campaign.py",
    "tests/test_fisher_compact_message_completion_capture.py",
    "tests/test_fisher_compact_message_completion_diagnostics.py",
    "tests/test_fisher_compact_message_completion_models.py",
    "tests/test_fisher_compact_message_completion_power.py",
    "tests/test_fisher_compact_message_completion_probe_v2.py",
    "tests/test_fisher_compact_message_completion_protocol.py",
    "tests/test_fisher_compact_message_completion_qualification.py",
    "tests/test_fisher_compact_message_completion_representation.py",
    "tests/test_fisher_compact_message_completion_science.py",
    "tests/test_fisher_compact_message_completion_splits.py",
    "tests/test_fisher_compact_message_completion_synthetic.py",
    "tests/test_fisher_compact_message_completion_synthetic_cli.py",
)


def _capture_implementation_hash() -> str:
    """Bind capture authorization to the exact behavior-affecting sources."""

    module_root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"fisher-cmc/synthetic-capture-implementation/v2\0")
    for name in (
        "fisher_compact_message_completion_campaign.py",
        "fisher_compact_message_completion_capture.py",
        "fisher_compact_message_completion_protocol.py",
        "fisher_compact_message_completion_synthetic.py",
    ):
        content = (module_root / name).read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_cross_layer_contracts(campaign_id: str) -> None:
    """Fail before creating output if schema-valid JSON is ledger-forbidden."""

    violations: list[str] = []
    for names in STAGE_ARTIFACT_ORDER.values():
        for name in names:
            if name in LEDGER_OWNED_ARTIFACTS or name.endswith(".safetensors"):
                continue
            body = synthetic_body_template(name)
            payload = build_artifact_payload(
                name,
                campaign_id=campaign_id,
                dependency_hashes={
                    dependency: "00" * 32
                    for dependency in artifact_dependencies(name)
                },
                body=body,
            )
            try:
                ProtocolLedger._strict_validate_public_artifact_bytes(  # noqa: SLF001
                    name, _json_bytes(payload)
                )
            except Exception as error:
                violations.append(f"{name}: {error}")
    if violations:
        raise CampaignIntegrationError(
            "artifact schema and ledger public-field policy disagree: "
            + "; ".join(violations)
        )


def _write_external_file(path: Path, content: bytes) -> None:
    """Publish exact bytes once, then make the registered source read-only."""

    publish_bytes_exclusive(path, content)
    os.chmod(path, stat.S_IRUSR)


def _sync_hashes(ledger: ProtocolLedger, hashes: dict[str, str]) -> None:
    for entry in ledger.campaign_inventory():
        hashes[entry.name] = entry.artifact_byte_sha256


def _assert_geometry_unchanged(
    geometry: SelectionGeometry,
    expected_sha256: str,
    *,
    phase: str,
) -> None:
    if geometry.identity_sha256 != expected_sha256:
        raise ProtocolStateError(
            f"frozen selection geometry changed during {phase}"
        )


def _register_external(
    ledger: ProtocolLedger,
    artifact_root: Path,
    *,
    campaign_id: str,
    prefix: CampaignPrefix,
    name: str,
    content: bytes,
    hashes: dict[str, str],
) -> None:
    path = artifact_root / name
    _write_external_file(path, content)
    entry = ledger.register_reached_artifact_file(
        campaign_id=campaign_id,
        prefix=prefix,
        name=name,
        path=path,
    )
    hashes[name] = entry.artifact_byte_sha256


def _register_p0(
    ledger: ProtocolLedger,
    artifact_root: Path,
    *,
    campaign_id: str,
    profile: CampaignProfile,
    hashes: dict[str, str],
) -> None:
    names = (
        "semantic_canonicalization_spec.json",
        "reserve_retired_exposed.json",
        "local_development_registry_status.json",
        "corpus_role_classification.json",
        "preflight_decision.json",
    )
    for name in names:
        payload = {
            "artifact_name": name,
            "campaign_id": campaign_id,
            "evidence_scope": EVIDENCE_SCOPE,
            "profile": profile,
            "protected_panel_accessed": False,
            "scientific_claim_eligible": False,
            "synthetic_evidence": True,
        }
        _register_external(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            prefix=CampaignPrefix.P0,
            name=name,
            content=_json_bytes(payload),
            hashes=hashes,
        )
    commands = (
        "# Synthetic qualification commands\n\n"
        f"Profile: `{profile}`. The campaign consumes only the deterministic "
        "in-memory fixture and never opens protected panels.\n"
    ).encode("utf-8")
    _register_external(
        ledger,
        artifact_root,
        campaign_id=campaign_id,
        prefix=CampaignPrefix.P0,
        name="commands.md",
        content=commands,
        hashes=hashes,
    )


_CONTENT_SET_DOMAIN: Final = b"fisher-compact-message-completion/content-set/v1"
_ANALYSIS_UNIT_SET_DOMAIN: Final = (
    b"fisher-compact-message-completion/analysis-unit-set/v1"
)
_PARTITION_DOMAIN: Final = b"fisher-compact-message-completion/partition/v1"


def _set_sha256(values: Sequence[str], domain: bytes) -> str:
    """Return F19's counted raw-digest set commitment without deduplication."""

    digests = [
        bytes.fromhex(_measured_sha256(value, label=f"set item {index}"))
        for index, value in enumerate(values)
    ]
    if len(set(digests)) != len(digests):
        raise CampaignIntegrationError("duplicate set commitments are forbidden")
    ordered = sorted(digests)
    return hashlib.sha256(
        domain
        + b"\0"
        + len(ordered).to_bytes(8, byteorder="big", signed=False)
        + b"".join(ordered)
    ).hexdigest()


def _partition_commitment(
    role: str, *, content_set_sha256: str, analysis_unit_set_sha256: str
) -> str:
    """Return F19's exact role/content/unit partition commitment."""

    if not isinstance(role, str) or not role:
        raise CampaignIntegrationError("partition role must be a nonempty string")
    content_hash = _measured_sha256(
        content_set_sha256, label="content_set_sha256"
    )
    unit_hash = _measured_sha256(
        analysis_unit_set_sha256, label="analysis_unit_set_sha256"
    )
    return hashlib.sha256(
        _PARTITION_DOMAIN
        + b"\0"
        + role.encode("utf-8")
        + b"\0"
        + bytes.fromhex(content_hash)
        + bytes.fromhex(unit_hash)
    ).hexdigest()


def _virtual_file_identity(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "artifact_byte_sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }


def _git_source_identity(path: Path, repository_root: Path) -> dict[str, object]:
    content = path.read_bytes()
    relative = path.relative_to(repository_root).as_posix()
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", relative],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = result.stdout[:2]
    tracked_status = (
        "untracked"
        if marker == "??"
        else "modified"
        if marker.strip()
        else "tracked"
    )
    return {
        **_virtual_file_identity(relative, content),
        "file_mode": oct(path.stat().st_mode & 0o777),
        "tracked_status": tracked_status,
    }


def _source_revision(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_inventory_body(repository_root: Path) -> dict[str, object]:
    paths = _SOURCE_INVENTORY_RELATIVE_PATHS
    if len(paths) != len(set(paths)) or tuple(sorted(paths)) != paths:
        raise CampaignIntegrationError(
            "source inventory paths must be unique and lexicographically ordered"
        )
    missing = [relative for relative in paths if not (repository_root / relative).is_file()]
    if missing:
        raise CampaignIntegrationError(
            f"source inventory is missing behavior-affecting files: {missing!r}"
        )
    return {
        "source_commit": _source_revision(repository_root),
        "files": [
            _git_source_identity(repository_root / relative, repository_root)
            for relative in paths
        ],
        "exclusions": [
            "protected_corpora",
            ".local-runs",
            "real_model_or_tokenizer_cache",
        ],
        "behavior_affecting_files_complete": True,
    }


def _assert_source_inventory_unchanged(
    expected: Mapping[str, object], *, phase: str
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    observed = _source_inventory_body(repository_root)
    if observed != expected:
        raise ProtocolStateError(
            f"behavior-affecting source changed after P2 during {phase}"
        )


def _p2_identity_bodies(
    campaign: SyntheticCampaignData,
    partition_records: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Build measured identities for the isolated in-memory fixture."""

    source_arrays = {
        descriptor.name: np.asarray(getattr(campaign.source, descriptor.name))
        for descriptor in fields(campaign.source)
    }
    source_bytes = encode_safetensors(source_arrays)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    config_bytes = json.dumps(
        asdict(campaign.config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    tokenizer_bytes = b"synthetic ordinal tokenizer; no text or vocabulary access\n"
    special_tokens_bytes = b"{}\n"
    model_identity = {
        "repository_id": "synthetic://fisher-cmc-public-fixture",
        "immutable_revision": source_sha256,
        "weight_files": [
            _virtual_file_identity(
                "in-memory://synthetic-source-parameters.safetensors",
                source_bytes,
            )
        ],
        "config_file": _virtual_file_identity(
            "in-memory://synthetic-campaign-config.json", config_bytes
        ),
        "generation_config_file": None,
        "tokenizer_files": [
            _virtual_file_identity(
                "in-memory://synthetic-ordinal-tokenizer.txt", tokenizer_bytes
            )
        ],
        "tokenizer_config_file": _virtual_file_identity(
            "in-memory://synthetic-tokenizer-config.json", b"{}\n"
        ),
        "special_tokens_map_file": _virtual_file_identity(
            "in-memory://synthetic-special-tokens-map.json", special_tokens_bytes
        ),
        "processor_files": [],
    }
    carrier_bytes = encode_safetensors(
        {
            "h4_fisher_diagonal": np.asarray(
                campaign.source.h4_fisher_diagonal, dtype=np.float64
            ),
            "h4_fisher_eigenvectors": np.asarray(
                campaign.source.q_carrier, dtype=np.float64
            ),
        }
    )
    fisher_identity = {
        "carrier_path": "in-memory://synthetic-h4-fisher-carrier.safetensors",
        "carrier_byte_sha256": hashlib.sha256(carrier_bytes).hexdigest(),
        "tensors": [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(
                {
                    "h4_fisher_diagonal": campaign.source.h4_fisher_diagonal,
                    "h4_fisher_eigenvectors": campaign.source.q_carrier,
                }.items()
            )
        ],
        "empirical_rank": int(np.linalg.matrix_rank(campaign.source.q_carrier)),
        "producing_source_identity_sha256": source_sha256,
        "producing_partition_commitment": partition_records[0][
            "partition_commitment"
        ],
    }
    repository_root = Path(__file__).resolve().parents[2]
    revision = _source_revision(repository_root)
    runtime_identity = {
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": "not_used_by_synthetic_fixture",
        "numpy_version": np.__version__,
        "safetensors_version": "builtin_metadata_free_codec_v1",
        "accelerator_backend": "cpu",
        "accelerator_model": platform.processor() or platform.machine(),
        "device_memory_bytes": None,
        "blas_backend": "numpy_runtime_default",
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "thread_configuration": [
            {"name": "torch_num_threads", "value": str(torch.get_num_threads())},
            {
                "name": "torch_num_interop_threads",
                "value": str(torch.get_num_interop_threads()),
            },
        ],
        "dependency_lockfiles": [],
        "locale": locale.setlocale(locale.LC_ALL, None),
        "timezone": str(datetime.now().astimezone().tzinfo),
        "model_execution_dtype": "float32",
        "source_commit": revision,
    }
    execution_budget = synthetic_body_template("execution_budget.json")
    execution_budget["maximum_process_rss_bytes"] = 16 * 1024**3
    source_inventory = _source_inventory_body(repository_root)
    return {
        "model_identity.json": model_identity,
        "fisher_carrier_identity.json": fisher_identity,
        "runtime_identity.json": runtime_identity,
        "execution_budget.json": execution_budget,
        "source_inventory.json": source_inventory,
        "row_derivation_protocol.json": (
            synthetic_row_derivation_protocol_body()
        ),
    }


def _measured_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignIntegrationError(f"{label} is not a measured SHA-256")
    return value


def _file_identity_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise CampaignIntegrationError(f"{label} is not a file identity")
    return _measured_sha256(
        value.get("artifact_byte_sha256"),
        label=f"{label}.artifact_byte_sha256",
    )


def _capture_identity_bindings(
    *,
    hashes: Mapping[str, str],
    p2_bodies: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Return E5/G3's exact frozen identities for every capture panel."""

    try:
        model_identity = p2_bodies["model_identity.json"]
        carrier_identity = p2_bodies["fisher_carrier_identity.json"]
        row_derivation = p2_bodies["row_derivation_protocol.json"]
    except KeyError as error:
        raise CampaignIntegrationError(
            f"capture identity body is missing {error.args[0]}"
        ) from error

    weight_files = model_identity.get("weight_files")
    tokenizer_files = model_identity.get("tokenizer_files")
    if not isinstance(weight_files, list) or not weight_files:
        raise CampaignIntegrationError("model identity has no weight-file identity")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise CampaignIntegrationError("model identity has no tokenizer-file identity")

    required_artifacts = (
        "semantic_canonicalization_spec.json",
        "corpus_role_classification.json",
        "independence_protocol.json",
        "split_manifest.json",
        "model_identity.json",
        "fisher_carrier_identity.json",
        "runtime_identity.json",
        "execution_budget.json",
        "source_inventory.json",
        "row_derivation_protocol.json",
    )
    missing = [name for name in required_artifacts if name not in hashes]
    if missing:
        raise CampaignIntegrationError(
            f"capture authorization identities are not frozen: {missing!r}"
        )

    return {
        "semantic_canonicalization_spec_sha256": hashes[
            "semantic_canonicalization_spec.json"
        ],
        "corpus_role_classification_sha256": hashes[
            "corpus_role_classification.json"
        ],
        "independence_protocol_sha256": hashes["independence_protocol.json"],
        "split_manifest_sha256": hashes["split_manifest.json"],
        "source_inventory_sha256": hashes["source_inventory.json"],
        "model_identity_sha256": hashes["model_identity.json"],
        "model_immutable_revision": _measured_sha256(
            model_identity.get("immutable_revision"),
            label="model_identity.immutable_revision",
        ),
        "model_weight_file_byte_sha256s": [
            _file_identity_sha256(
                record,
                label=f"model_identity.weight_files[{index}]",
            )
            for index, record in enumerate(weight_files)
        ],
        "model_config_byte_sha256": _file_identity_sha256(
            model_identity.get("config_file"),
            label="model_identity.config_file",
        ),
        "tokenizer_identity_sha256": _measured_sha256(
            row_derivation.get("tokenizer_identity_sha256"),
            label="row_derivation_protocol.tokenizer_identity_sha256",
        ),
        "tokenizer_file_byte_sha256s": [
            _file_identity_sha256(
                record,
                label=f"model_identity.tokenizer_files[{index}]",
            )
            for index, record in enumerate(tokenizer_files)
        ],
        "tokenizer_config_byte_sha256": _file_identity_sha256(
            model_identity.get("tokenizer_config_file"),
            label="model_identity.tokenizer_config_file",
        ),
        "special_vocabulary_map_byte_sha256": _file_identity_sha256(
            model_identity.get("special_tokens_map_file"),
            label="model_identity.special_tokens_map_file",
        ),
        "carrier_identity_sha256": hashes["fisher_carrier_identity.json"],
        "h4_carrier_byte_sha256": _measured_sha256(
            carrier_identity.get("carrier_byte_sha256"),
            label="fisher_carrier_identity.carrier_byte_sha256",
        ),
        "runtime_identity_sha256": hashes["runtime_identity.json"],
        "execution_budget_sha256": hashes["execution_budget.json"],
        "row_derivation_protocol_artifact_sha256": hashes[
            "row_derivation_protocol.json"
        ],
    }


def _partition_record(
    partition: SyntheticPartition,
    *,
    role: str,
) -> dict[str, object]:
    sequence_units: dict[str, str] = {}
    sequence_positions: dict[str, set[int]] = {}
    for item in partition.metadata:
        existing_unit = sequence_units.setdefault(
            item.canonical_sequence_hash, item.analysis_unit_commitment
        )
        if existing_unit != item.analysis_unit_commitment:
            raise CampaignIntegrationError(
                "one canonical sequence maps to multiple analysis units"
            )
        positions = sequence_positions.setdefault(item.canonical_sequence_hash, set())
        if item.token_position in positions:
            raise CampaignIntegrationError(
                "duplicate canonical sequence/token-position identity"
            )
        positions.add(item.token_position)
    sequences = sorted(sequence_units)
    units = sorted(set(sequence_units.values()))
    unit_hash = _set_sha256(units, _ANALYSIS_UNIT_SET_DOMAIN)
    content_hash = _set_sha256(sequences, _CONTENT_SET_DOMAIN)
    commitment = _partition_commitment(
        role,
        content_set_sha256=content_hash,
        analysis_unit_set_sha256=unit_hash,
    )
    return {
        "partition_role": role,
        "analysis_unit_count": len(units),
        "sequence_count": len(sequences),
        "content_set_sha256": content_hash,
        "analysis_unit_set_sha256": unit_hash,
        "partition_commitment": commitment,
    }


def _partition_records(campaign: SyntheticCampaignData) -> list[dict[str, object]]:
    # F22 assigns the lowest-key ten units to monitor.  The synthetic builder
    # therefore stores monitor on the left and fit-train on the right.
    fit_train = subset_partition_by_units(
        campaign.fit, campaign.fit_split.right_units, partition_role="fit"
    )
    fit_monitor = subset_partition_by_units(
        campaign.fit, campaign.fit_split.left_units, partition_role="fit"
    )
    return [
        _partition_record(fit_train, role="fit_train"),
        _partition_record(fit_monitor, role="fit_monitor"),
        _partition_record(campaign.tune, role="tune"),
        _partition_record(
            campaign.development_confirm, role="development_confirm"
        ),
    ]


def _register_json_stage(
    ledger: ProtocolLedger,
    artifact_root: Path,
    *,
    campaign_id: str,
    stage: Stage,
    hashes: dict[str, str],
    body_overrides: Mapping[str, Mapping[str, object]] | None = None,
    skip_names: Sequence[str] = (),
) -> None:
    overrides = body_overrides or {}
    skipped = frozenset(skip_names)
    prefix = CampaignPrefix[stage.value]
    for name in STAGE_ARTIFACT_ORDER[stage]:
        if (
            name in LEDGER_OWNED_ARTIFACTS
            or name.endswith(".safetensors")
            or name in skipped
        ):
            continue
        body = dict(overrides.get(name, synthetic_body_template(name)))
        dependencies = artifact_dependencies(name)
        missing = [dependency for dependency in dependencies if dependency not in hashes]
        if missing:
            raise ProtocolStateError(
                f"cannot materialize {name}; missing dependencies {missing!r}"
            )
        payload = build_artifact_payload(
            name,
            campaign_id=campaign_id,
            dependency_hashes={dependency: hashes[dependency] for dependency in dependencies},
            body=body,
        )
        validate_artifact_payload(name, payload)
        _register_external(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            prefix=prefix,
            name=name,
            content=_json_bytes(payload),
            hashes=hashes,
        )


def _register_json_artifact(
    ledger: ProtocolLedger,
    artifact_root: Path,
    *,
    campaign_id: str,
    name: str,
    hashes: dict[str, str],
    body: Mapping[str, object],
) -> None:
    """Validate and durably register one JSON artifact in frozen stage order."""

    dependencies = artifact_dependencies(name)
    missing = [dependency for dependency in dependencies if dependency not in hashes]
    if missing:
        raise ProtocolStateError(
            f"cannot register {name}; missing dependencies {missing!r}"
        )
    payload = build_artifact_payload(
        name,
        campaign_id=campaign_id,
        dependency_hashes={dependency: hashes[dependency] for dependency in dependencies},
        body=body,
    )
    validate_artifact_payload(name, payload)
    _register_external(
        ledger,
        artifact_root,
        campaign_id=campaign_id,
        prefix=CampaignPrefix[artifact_schema(name).stage.value],
        name=name,
        content=_json_bytes(payload),
        hashes=hashes,
    )


def _capture_partition(
    ledger: ProtocolLedger,
    capture_root: Path,
    partition: SyntheticPartition,
    *,
    campaign_id: str,
    profile: CampaignProfile,
    row_derivation_protocol_artifact_sha256: str,
    maximum_rows_per_shard: int,
    hashes: dict[str, str],
    frozen_source_inventory: Mapping[str, object],
    frozen_capture_implementation_hash: str,
    authorization_bindings: Mapping[str, object] | None = None,
) -> None:
    role = partition.partition_role
    if (
        partition.row_index.row_derivation_protocol_hash
        != synthetic_row_derivation_protocol_sha256()
    ):
        raise ProtocolStateError(
            "row index is not bound to the exported semantic derivation protocol"
        )
    _assert_source_inventory_unchanged(
        frozen_source_inventory, phase=f"{role} authorization"
    )
    identity = _partition_record(partition, role=role)
    implementation_sha256 = _capture_implementation_hash()
    if implementation_sha256 != frozen_capture_implementation_hash:
        raise ProtocolStateError(
            "capture implementation changed after its P2 freeze"
        )
    canonical_content_sha256s = sorted(
        {item.canonical_sequence_hash for item in partition.metadata}
    )
    analysis_unit_commitments = list(partition.analysis_unit_commitments)
    ledger.authorize(
        AuthorizationSpec(
            partition_role=role,
            campaign_id=campaign_id,
            campaign_scope=SYNTHETIC_QUALIFICATION_CAMPAIGN_SCOPE,
            expected_row_set_mode=AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            row_derivation_protocol_hash=(
                partition.row_index.row_derivation_protocol_hash
            ),
            authorization_nonce=f"synthetic-{role}-{profile}-v1",
            expected_row_set_sha256=None,
            expected_ordered_row_sequence_sha256=None,
            expected_row_count=None,
            expected_row_index_payload_sha256=None,
            frozen_bindings={
                **dict(authorization_bindings or {}),
                "evidence_scope": SYNTHETIC_EVIDENCE_SCOPE,
                "profile": profile,
                "protected_panel_accessed": False,
                "row_derivation_protocol_artifact_sha256": (
                    row_derivation_protocol_artifact_sha256
                ),
                "row_derivation_protocol_semantic_sha256": (
                    partition.row_index.row_derivation_protocol_hash
                ),
                "row_count": partition.row_index.row_count,
                "row_index_payload_sha256": partition.row_index.payload_sha256,
                "canonical_content_sha256s": canonical_content_sha256s,
                "content_set_sha256": identity["content_set_sha256"],
                "analysis_unit_commitments": analysis_unit_commitments,
                "analysis_unit_set_sha256": identity[
                    "analysis_unit_set_sha256"
                ],
                "expected_sequence_count": identity["sequence_count"],
                "expected_analysis_unit_count": identity[
                    "analysis_unit_count"
                ],
                "partition_commitment": identity["partition_commitment"],
                "maximum_rows_per_shard": maximum_rows_per_shard,
                "capture_schema_hash": _CAPTURE_SCHEMA_HASH,
                "capture_implementation_hash": implementation_sha256,
                "synthetic_only": True,
            },
        )
    )
    ledger.commit_row_index(partition.row_index)
    plan = build_shard_plan(partition.row_index, maximum_rows_per_shard)
    ledger.commit_shard_plan(plan)
    ledger.start_capture(
        plan,
        capture_root,
        capture_schema_hash=_CAPTURE_SCHEMA_HASH,
        capture_implementation_hash=implementation_sha256,
    )
    snapshot = ledger.snapshot(role)
    if snapshot is None or snapshot.capture_attempt_id is None:
        raise ProtocolStateError("capture start did not publish an attempt identity")
    for entry in plan.entries:
        write_capture_shard(
            capture_root,
            entry,
            partition.tensors_for_shard(entry),
        )
    verification = verify_capture_shard_files(plan, capture_root)
    _assert_source_inventory_unchanged(
        frozen_source_inventory, phase=f"{role} capture finalization"
    )
    if _capture_implementation_hash() != frozen_capture_implementation_hash:
        raise ProtocolStateError(
            "capture implementation changed during authoritative capture"
        )
    ledger.finalize_capture_success(
        plan,
        capture_root,
        capture_attempt_id=snapshot.capture_attempt_id,
        expected_verification=verification,
    )
    _sync_hashes(ledger, hashes)


def _register_pre_tune_representation(
    ledger: ProtocolLedger,
    artifact_root: Path,
    *,
    campaign_id: str,
    campaign: SyntheticCampaignData,
    qualification: PreTuneRepresentationQualification,
    o1_bodies: Mapping[str, Mapping[str, object]],
    partition_records: Sequence[Mapping[str, object]],
    hashes: dict[str, str],
) -> None:
    """Register the complete measured/static P4 freeze before tune access."""

    if not qualification.passed:
        raise ProtocolStateError("pre-tune representation qualification failed")

    def register(name: str, body: Mapping[str, object]) -> None:
        _register_json_artifact(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            name=name,
            hashes=hashes,
            body=body,
        )

    register("o1_tolerance_protocol.json", o1_bodies["o1_tolerance_protocol.json"])
    tolerance_protocol_sha256 = hashes["o1_tolerance_protocol.json"]
    for name in ("o1_validation_results.json", "o1_complete_fit_results.json"):
        body = dict(o1_bodies[name])
        body["tolerance_protocol_sha256"] = tolerance_protocol_sha256
        register(name, body)

    register("locator_protocol.json", qualification.locator_protocol_body())
    register(
        "locator_candidates.json",
        qualification.locator_candidates_body(
            locator_protocol_sha256=hashes["locator_protocol.json"]
        ),
    )
    register(
        "codec_protocol.json",
        qualification.codec_protocol_body(
            carrier_identity_sha256=hashes["fisher_carrier_identity.json"]
        ),
    )
    register(
        "message_sketch_manifest_p32.json",
        qualification.message_sketch_manifest_body(),
    )
    register(
        "codec_candidate_manifests.json",
        qualification.codec_candidate_manifests_body(
            codec_protocol_sha256=hashes["codec_protocol.json"]
        ),
    )
    register(
        "representation_numeric_constants.json",
        qualification.representation_numeric_constants_body(),
    )
    register(
        "full_rank_identity_smoke.json",
        qualification.full_rank_identity_smoke_body(),
    )
    register(
        "selection_representation_manifest.json",
        qualification.selection_representation_manifest_body(),
    )

    ceiling = synthetic_body_template("ceiling_gate_protocol.json")
    register("ceiling_gate_protocol.json", ceiling)
    learning = synthetic_body_template("learning_protocol.json")
    learning["epsilon_down"] = qualification.epsilon_down
    register("learning_protocol.json", learning)
    development = synthetic_body_template("development_eligibility_protocol.json")
    development["epsilon_down"] = qualification.epsilon_down
    register("development_eligibility_protocol.json", development)

    preregistration = synthetic_body_template("preregistration_root.json")
    preregistration.update(
        {
            "bound_artifacts": [
                {"name": name, "artifact_sha256": hashes[name]}
                for name in PREREGISTRATION_BOUND_ARTIFACTS
            ],
            "fit_capture_manifest_sha256": hashes[
                "capture_shard_manifest_fit.json"
            ],
            "reserve_retirement_receipt_sha256": hashes[
                "reserve_retired_exposed.json"
            ],
            "local_registry_snapshot_sha256": hashes[
                "local_development_registry_status.json"
            ],
            "complete_before_tune_authorization": True,
        }
    )
    register("preregistration_root.json", preregistration)

    parts = (
        subset_partition_by_units(
            campaign.fit, campaign.fit_split.right_units, partition_role="fit"
        ),
        subset_partition_by_units(
            campaign.fit, campaign.fit_split.left_units, partition_role="fit"
        ),
        campaign.tune,
        campaign.development_confirm,
    )
    unit_sets = [
        {item.analysis_unit_commitment for item in partition.metadata}
        for partition in parts
    ]
    content_sets = [
        {item.canonical_sequence_hash for item in partition.metadata}
        for partition in parts
    ]
    fresh = all(
        not unit_sets[left].intersection(unit_sets[right])
        and not content_sets[left].intersection(content_sets[right])
        for left in range(len(parts))
        for right in range(left + 1, len(parts))
    )
    freshness = synthetic_body_template(
        "local_development_freshness_audit_receipt.json"
    )
    freshness.update(
        {
            "preregistration_root_sha256": hashes["preregistration_root.json"],
            "audited_content_set_sha256s": [
                str(record["content_set_sha256"]) for record in partition_records
            ],
            "audited_analysis_unit_set_sha256s": [
                str(record["analysis_unit_set_sha256"])
                for record in partition_records
            ],
            "overlap_status": "fresh" if fresh else "overlap_detected",
            "can_authorize_guard": False,
        }
    )
    if not fresh:
        raise ProtocolStateError("synthetic partition freshness audit failed")
    register("local_development_freshness_audit_receipt.json", freshness)

    campaign_root = synthetic_body_template("campaign_root.json")
    campaign_root.update(
        {
            "preregistration_root_sha256": hashes["preregistration_root.json"],
            "freshness_receipt_sha256": hashes[
                "local_development_freshness_audit_receipt.json"
            ],
            "model_identity_sha256": hashes["model_identity.json"],
            "carrier_identity_sha256": hashes["fisher_carrier_identity.json"],
            "runtime_identity_sha256": hashes["runtime_identity.json"],
            "source_inventory_sha256": hashes["source_inventory.json"],
            "fit_capture_manifest_sha256": hashes[
                "capture_shard_manifest_fit.json"
            ],
            "local_registry_snapshot_sha256": hashes[
                "local_development_registry_status.json"
            ],
        }
    )
    register("campaign_root.json", campaign_root)


def _o1_bodies(
    qualification: O1Qualification,
    *,
    fit_manifest_sha256: str,
    assignment_sha256: str,
    fit_row_count: int,
    validation_row_count: int,
    head_count: int,
    message_dimension: int,
) -> dict[str, Mapping[str, object]]:
    capture = synthetic_body_template("capture_invariants.json")
    capture.update(
        {
            "fit_capture_manifest_sha256": fit_manifest_sha256,
            "query_head_count": head_count,
            "transported_message_dimension": message_dimension,
            "seams": [
                {
                    "seam": result.seam,
                    "observed_max_abs_error": result.observed_max_abs_error,
                    "observed_weighted_relative_l2": (
                        result.observed_weighted_relative_l2
                    ),
                    "finite": True,
                    "shape_contract_passed": True,
                }
                for result in qualification.complete_fit
            ],
            "passed": qualification.passed,
        }
    )
    tolerance = synthetic_body_template("o1_tolerance_protocol.json")
    tolerance.update(
        {
            "calibration_assignment_sha256": assignment_sha256,
            "tolerances": [item.as_dict() for item in qualification.tolerances],
            "protocol_construction_passed": all(
                item.calibration_passed for item in qualification.tolerances
            ),
        }
    )
    validation = synthetic_body_template("o1_validation_results.json")
    validation.update(
        {
            "row_count": validation_row_count,
            "results": [item.as_dict() for item in qualification.validation],
            "all_seams_passed": all(item.passed for item in qualification.validation),
        }
    )
    complete = synthetic_body_template("o1_complete_fit_results.json")
    complete.update(
        {
            "row_count": fit_row_count,
            "results": [item.as_dict() for item in qualification.complete_fit],
            "all_seams_passed": all(
                item.passed for item in qualification.complete_fit
            ),
        }
    )
    return {
        "capture_invariants.json": capture,
        "o1_tolerance_protocol.json": tolerance,
        "o1_validation_results.json": validation,
        "o1_complete_fit_results.json": complete,
    }


def _fidelity_panel(role: str, metrics: FidelityMetrics) -> dict[str, object]:
    gates = fidelity_gate_records(metrics)
    return {
        "partition_role": role,
        "gate_results": gates,
        "all_five_passed": all(bool(gate["passed"]) for gate in gates),
        "evaluation_label": FULL_PROMPT_LABEL,
    }


def _o2_bodies(
    evaluations: Sequence[O2CodecEvaluation],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for artifact_name, kind in (("o2q_results.json", "q"), ("o2v_results.json", "v")):
        body = synthetic_body_template(artifact_name)
        candidates = []
        passing = []
        for evaluation in evaluations:
            monitor = getattr(evaluation, f"fit_monitor_{kind}")
            tune = getattr(evaluation, f"tune_{kind}")
            panels = [
                _fidelity_panel("fit_monitor", monitor),
                _fidelity_panel("tune", tune),
            ]
            passed = all(bool(panel["all_five_passed"]) for panel in panels)
            candidates.append(
                {
                    "codec_id": evaluation.codec_id,
                    "panels": panels,
                    "passes_fit_monitor_and_tune": passed,
                }
            )
            if passed:
                passing.append(evaluation.codec_id)
        body["candidate_panels"] = candidates
        body["passing_codec_ids"] = passing
        result[artifact_name] = body
    return result


def _evaluate_fits(
    fits: Mapping[str, ModelBankFit],
    panel: PreparedCompletionPanel,
    geometry: SelectionGeometry,
    codec_id: str,
    campaign: SyntheticCampaignData,
    *,
    epsilon_down: float,
) -> dict[str, ModelBankEvaluation]:
    return {
        model_id: evaluate_bank(
            fit,
            panel,
            geometry.codec(codec_id),
            campaign,
            epsilon_down=epsilon_down,
        )
        for model_id, fit in fits.items()
    }


def _evaluate_loaded_bundle_models(
    loaded: LoadedWeightsOnlyBundles,
    model_ids: Sequence[str],
    partition: SyntheticPartition,
    panel: PreparedCompletionPanel,
    campaign: SyntheticCampaignData,
    *,
    epsilon_down: float,
) -> dict[str, ModelBankEvaluation]:
    """Evaluate later-panel models exclusively through decoded bundle state."""

    fisher_diagonal = torch.from_numpy(
        loaded.candidate_tensors["source.h4_fisher_diagonal"].copy()
    ).to(torch.float64)
    fisher = torch.diag(fisher_diagonal)
    evaluations: dict[str, ModelBankEvaluation] = {}
    for model_id in model_ids:
        execution = loaded.execute_raw(
            model_id,
            partition.tensors["h"],
            partition.tensors["message"],
        )
        if not torch.equal(execution.chart_ids, panel.chart_ids):
            raise CampaignIntegrationError(
                f"decoded bundle chart assignments differ for {model_id}"
            )
        if not torch.equal(execution.support_mask, panel.support_mask):
            raise CampaignIntegrationError(
                f"decoded bundle support decisions differ for {model_id}"
            )
        if not torch.equal(
            execution.c_base.to(torch.float32), panel.c_base.to(torch.float32)
        ):
            raise CampaignIntegrationError(
                f"decoded bundle compact base differs for {model_id}"
            )
        full = model_id == "J0"
        prediction = execution.coefficients
        if not full:
            prediction = prediction - execution.c_base
        target = panel.c_teacher if full else panel.delta_c
        epsilon = panel.epsilon_j0 if full else panel.epsilon_completion
        coefficient_loss = float(
            bank_coefficient_loss(
                prediction.to(torch.float64),
                target.to(torch.float64),
                epsilon.to(torch.float64),
                row_keys=panel.row_keys,
            )
        )
        downstream = primary_downstream_metric(
            execution.h_plus.to(torch.float64) - panel.h_plus,
            panel.h_plus - panel.h,
            fisher,
            panel.row_keys,
            epsilon_down=epsilon_down,
        )
        evaluations[model_id] = ModelBankEvaluation(
            model_id,
            coefficient_loss,
            downstream,
            evaluate_fidelity(partition, execution.h_plus, campaign),
        )
    return evaluations


def _model_records(
    fits: Mapping[str, ModelBankFit],
    evaluations: Mapping[str, ModelBankEvaluation],
    *,
    status: str,
    model_order: Sequence[str] = _MODEL_ORDER,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_id in model_order:
        fit = fits[model_id]
        analytic = model_id == "J1"
        records.append(
            {
                "model_id": model_id,
                "mandatory": True,
                "representation_sha256": "00" * 32,
                "architecture_sha256": fit.architecture_sha256,
                "selected_seed": None if analytic else fit.selected_seed,
                "selected_optimizer_step": None if analytic else fit.selected_step,
                "coefficient_loss": None if analytic else fit.coefficient_loss,
                "L_bar_down": evaluations[model_id].downstream.mean_loss,
                "status": "analytic" if analytic else status,
            }
        )
    return records


def _model_records_with_representation(
    fits: Mapping[str, ModelBankFit],
    evaluations: Mapping[str, ModelBankEvaluation],
    representation_sha256: str,
    *,
    status: str,
    model_order: Sequence[str] = _MODEL_ORDER,
) -> list[dict[str, object]]:
    records = _model_records(
        fits, evaluations, status=status, model_order=model_order
    )
    for record in records:
        record["representation_sha256"] = representation_sha256
    return records


def _stratum_label_key(label: StratumLabel) -> str:
    return json.dumps(label.as_json(), sort_keys=True, separators=(",", ":"))


def _stratum_scheme_for_artifact(scheme: str) -> str:
    return "family_x_template_id" if scheme == "family_template" else scheme


def _unit_strata(
    partition: SyntheticPartition,
    evaluation: ModelBankEvaluation,
) -> tuple[list[str], StratumDecision]:
    unit_by_commitment: dict[bytes, SplitUnit] = {}
    for item in partition.metadata:
        commitment = bytes.fromhex(item.analysis_unit_commitment)
        candidate = SplitUnit(
            item.analysis_unit_commitment,
            item.family_id,
            item.template_id,
        )
        previous = unit_by_commitment.setdefault(commitment, candidate)
        if previous != candidate:
            raise ProtocolStateError(
                "development analysis-unit metadata is internally inconsistent"
            )
    decision = choose_strata(
        tuple(
            unit_by_commitment[key]
            for key in sorted(unit_by_commitment)
        ),
        "mode_b_semantic_instance",
    )

    def label(unit: SplitUnit) -> StratumLabel:
        if decision.final_scheme == "family_template":
            return StratumLabel(
                "family_template", unit.family_id, unit.template_id
            )
        if decision.final_scheme == "family":
            return StratumLabel("family", unit.family_id, None)
        return StratumLabel("global", None, None)

    return (
        [
            _stratum_label_key(
                label(unit_by_commitment[item.analysis_unit_commitment])
            )
            for item in evaluation.downstream.units
        ],
        decision,
    )


def _chart_support_records(
    partition: SyntheticPartition,
    panel: PreparedCompletionPanel,
    assessment: DevelopmentSupportAssessment,
) -> list[dict[str, object]]:
    """Materialize per-assigned-chart unit, sequence, and token counts."""

    charts = tuple(int(value) for value in panel.chart_ids.tolist())
    support = tuple(bool(value) for value in panel.support_mask.tolist())
    if len(charts) != len(partition.metadata):
        raise ProtocolStateError("development chart assignments are row-misaligned")
    result: list[dict[str, object]] = []
    for chart, unit_count in assessment.chart_analysis_unit_counts:
        rows = tuple(
            item
            for item, assigned, supported in zip(
                partition.metadata, charts, support, strict=True
            )
            if assigned == chart and supported
        )
        result.append(
            {
                "chart_id": chart,
                "analysis_unit_count": unit_count,
                "sequence_count": len(
                    {item.canonical_sequence_hash for item in rows}
                ),
                "row_count": len(rows),
                "supported": unit_count >= 4,
            }
        )
    return result


def _power_chart_memberships(
    partition: SyntheticPartition,
    panel: PreparedCompletionPanel,
    evaluation: ModelBankEvaluation,
) -> tuple[tuple[SupportedChartMembership, ...], ...]:
    """Align supported chart/sequence/row counts to downstream unit order."""

    charts = tuple(int(value) for value in panel.chart_ids.tolist())
    support = tuple(bool(value) for value in panel.support_mask.tolist())
    if len(charts) != len(partition.metadata):
        raise ProtocolStateError("power chart assignments are row-misaligned")
    by_unit: dict[bytes, dict[int, list[object]]] = {}
    for metadata, chart, supported in zip(
        partition.metadata, charts, support, strict=True
    ):
        if not supported:
            continue
        unit = bytes.fromhex(metadata.analysis_unit_commitment)
        by_unit.setdefault(unit, {}).setdefault(chart, []).append(metadata)
    result: list[tuple[SupportedChartMembership, ...]] = []
    for unit_loss in evaluation.downstream.units:
        chart_rows = by_unit.get(unit_loss.analysis_unit_commitment, {})
        result.append(
            tuple(
                SupportedChartMembership(
                    chart_id=chart,
                    sequence_count=len(
                        {
                            str(item.canonical_sequence_hash)
                            for item in rows
                        }
                    ),
                    row_count=len(rows),
                )
                for chart, rows in sorted(chart_rows.items())
            )
        )
    return tuple(result)


def _bundle_tensor_records(content: bytes, bundle: str) -> list[dict[str, object]]:
    header_length = int.from_bytes(content[:8], "little")
    header = json.loads(content[8 : 8 + header_length].decode("utf-8"))
    return [
        {
            "bundle": bundle,
            "name": name,
            "shape": descriptor["shape"],
            "dtype": descriptor["dtype"],
        }
        for name, descriptor in sorted(header.items())
    ]


def _bundle_names(content: bytes) -> frozenset[str]:
    header_length = int.from_bytes(content[:8], "little")
    header = json.loads(content[8 : 8 + header_length].decode("utf-8"))
    return frozenset(header)


def _qualification_blockers(
    *,
    profile: _TrainingProfile,
    replay_records: Sequence[Mapping[str, object]],
    candidate_bundle: bytes,
    comparator_bundle: bytes,
    development_eligible: bool,
) -> tuple[str, ...]:
    """Return reasons this synthetic run cannot close its mechanical P12."""

    blockers: list[str] = []
    if not profile.exact_contract:
        blockers.append("quick optimizer schedules are not the frozen exact contract")
    if not all(bool(record["stable"]) for record in replay_records):
        blockers.append("all-fit refit failed the two-percent tune stability gate")
    if not development_eligible:
        blockers.append("the frozen six-part development eligibility conjunction failed")
    candidate_names = _bundle_names(candidate_bundle)
    candidate_prefixes = (
        "representation.",
        "support.",
        "message_sketch.",
        "codec.",
        "J4.",
    )
    if any(
        not any(name.startswith(prefix) for name in candidate_names)
        for prefix in candidate_prefixes
    ):
        blockers.append(
            "candidate bundle lacks representation, support, sketch, codec, or J4 state"
        )
    comparator_names = _bundle_names(comparator_bundle)
    if any(
        not any(name.startswith(f"{model_id}.") for name in comparator_names)
        for model_id in ("J2N", "J2+", "J3")
    ):
        blockers.append("comparator bundle does not contain J2N, J2+, and J3")
    return tuple(blockers)


def _export_ledger_artifacts(ledger: ProtocolLedger, artifact_root: Path) -> None:
    """Place readable copies of ledger-owned JSON/Markdown beside externals."""

    for entry in ledger.campaign_inventory():
        path = artifact_root / entry.name
        if path.exists():
            continue
        stored = ledger.get_artifact(entry.name)
        if stored is not None:
            _write_external_file(path, stored.content)


def run_synthetic_campaign(
    output_directory: str | os.PathLike[str],
    *,
    profile: CampaignProfile = "quick",
    campaign_id: str | None = None,
    synthetic_config: SyntheticCampaignConfig | None = None,
    maximum_rows_per_shard: int = 64,
) -> SyntheticCampaignResult:
    """Run one isolated, ledger-backed synthetic qualification campaign.

    The output directory must not already exist.  This no-clobber boundary is
    intentional: a rerun receives a new directory and therefore cannot mutate
    or silently combine with prior evidence.
    """

    if type(maximum_rows_per_shard) is not int or maximum_rows_per_shard <= 0:
        raise ValueError("maximum_rows_per_shard must be a positive integer")
    selected_profile = training_profile(profile)
    unresolved_root = Path(output_directory).expanduser().resolve(strict=False)
    if campaign_id is None:
        campaign_id = (
            _CAMPAIGN_ID
            + "-"
            + hashlib.sha256(os.fsencode(unresolved_root)).hexdigest()[:16]
        )
    _validate_cross_layer_contracts(campaign_id)
    root = _prepare_output_root(output_directory)
    artifact_root = root / "artifacts"
    capture_root = root / "capture"
    artifact_root.mkdir(mode=0o700)
    capture_root.mkdir(mode=0o700)
    ledger_path = root / "protocol-ledger.sqlite3"
    hashes: dict[str, str] = {}
    campaign = build_synthetic_campaign(synthetic_config or SyntheticCampaignConfig())
    epsilon_down = freeze_epsilon_down_from_fit_train(campaign)
    partition_records = _partition_records(campaign)

    with ProtocolLedger(ledger_path) as ledger:
        _register_p0(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            profile=profile,
            hashes=hashes,
        )

        p1_independence = synthetic_body_template("independence_protocol.json")
        p1_independence.update(
            {
                "selected_mode": "semantic_instance",
                "analysis_unit_field": "semantic_instance_id",
                "mode_a_eligible": False,
                "mode_b_eligible": True,
                "fit_analysis_unit_count": campaign.config.fit_analysis_units,
                "tune_analysis_unit_count": campaign.config.tune_analysis_units,
                "development_confirm_analysis_unit_count": (
                    campaign.config.d2_analysis_units
                    - campaign.config.tune_analysis_units
                ),
                "inference_limitation": MODE_B_LIMITATION,
                "selection_reason": (
                    "synthetic template lineages repeat; semantic instances "
                    "are the independent qualification units"
                ),
            }
        )
        split_body = synthetic_body_template("split_manifest.json")
        split_body["partitions"] = partition_records
        o1_assignment = synthetic_body_template("o1_calibration_assignment.json")
        o1_assignment.update(
            {
                "calibration_unit_set_sha256": _set_sha256(
                    list(campaign.o1_split.left_units),
                    _ANALYSIS_UNIT_SET_DOMAIN,
                ),
                "validation_unit_set_sha256": _set_sha256(
                    list(campaign.o1_split.right_units),
                    _ANALYSIS_UNIT_SET_DOMAIN,
                ),
            }
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P1,
            hashes=hashes,
            body_overrides={
                "independence_protocol.json": p1_independence,
                "split_manifest.json": split_body,
                "o1_calibration_assignment.json": o1_assignment,
            },
        )
        p2_identity_bodies = _p2_identity_bodies(campaign, partition_records)
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P2,
            hashes=hashes,
            body_overrides=p2_identity_bodies,
        )
        capture_identity_bindings = _capture_identity_bindings(
            hashes=hashes,
            p2_bodies=p2_identity_bodies,
        )
        frozen_source_inventory = p2_identity_bodies["source_inventory.json"]
        frozen_capture_implementation_hash = _capture_implementation_hash()

        _capture_partition(
            ledger,
            capture_root,
            campaign.fit,
            campaign_id=campaign_id,
            profile=profile,
            row_derivation_protocol_artifact_sha256=hashes[
                "row_derivation_protocol.json"
            ],
            maximum_rows_per_shard=maximum_rows_per_shard,
            hashes=hashes,
            frozen_source_inventory=frozen_source_inventory,
            frozen_capture_implementation_hash=(
                frozen_capture_implementation_hash
            ),
            authorization_bindings=capture_identity_bindings,
        )
        o1 = build_o1_qualification(campaign)
        if not o1.passed:
            raise ProtocolStateError("synthetic O1 seam qualification failed")
        validation_rows = sum(
            item.analysis_unit_commitment in campaign.o1_split.right_units
            for item in campaign.fit.metadata
        )
        o1_bodies = _o1_bodies(
            o1,
            fit_manifest_sha256=hashes["capture_shard_manifest_fit.json"],
            assignment_sha256=hashes["o1_calibration_assignment.json"],
            fit_row_count=campaign.fit.row_index.row_count,
            validation_row_count=validation_rows,
            head_count=campaign.config.head_count,
            message_dimension=campaign.config.message_dimension,
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P3,
            hashes=hashes,
            body_overrides={"capture_invariants.json": o1_bodies["capture_invariants.json"]},
        )
        geometry = fit_selection_geometry(campaign)
        full_rank_smoke = build_full_rank_identity_smoke(campaign, geometry)
        if not full_rank_smoke.passed:
            raise ProtocolStateError(
                "synthetic full-rank fit-monitor identity smoke failed"
            )
        representation = build_pre_tune_representation_qualification(
            campaign, geometry, o1, full_rank_smoke
        )
        attempted_operator_codec_ids = [
            f"K{representation.selected_locator_k}.{codec.codec_id}"
            for codec in representation.codec_qualifications
        ]
        if representation.requires_lower_k_operator_retry:
            # J4 requires a complete representation retry, not merely another
            # operator summary over the K=4 tensors.  No P4 artifact or tune
            # authorization exists yet, so every geometry-bound pre-tune
            # diagnostic can be rebuilt safely under the preregistered K=2.
            geometry = fit_selection_geometry(
                campaign,
                attempted_locator_k=(2,),
            )
            full_rank_smoke = build_full_rank_identity_smoke(campaign, geometry)
            if not full_rank_smoke.passed:
                raise ProtocolStateError(
                    "synthetic lower-K full-rank fit-monitor identity smoke failed"
                )
            representation = build_pre_tune_representation_qualification(
                campaign, geometry, o1, full_rank_smoke
            )
            attempted_operator_codec_ids.extend(
                f"K{representation.selected_locator_k}.{codec.codec_id}"
                for codec in representation.codec_qualifications
            )
        if representation.operator_reference_fallback_exhausted:
            support_spec = _pre_tune_operator_support_failure_spec(
                campaign_id=campaign_id,
                campaign=campaign,
                representation=representation,
                attempted_codec_ids=attempted_operator_codec_ids,
            )
            terminal = ledger.finalize_support_failure(
                support_spec,
                omitted_active_artifacts={
                    name: "no_positive_operator_reference"
                    for name in STAGE_ARTIFACT_ORDER[Stage.P4]
                },
            )
            _sync_hashes(ledger, hashes)
            _export_ledger_artifacts(ledger, artifact_root)
            return SyntheticCampaignResult(
                campaign_id=campaign_id,
                profile=profile,
                exact_training_contract=selected_profile.exact_contract,
                output_directory=root,
                ledger_path=ledger_path,
                selected_codec_id=None,
                frozen_baseline_id=None,
                o1_passed=o1.passed,
                o2q_passing_codec_ids=(),
                initial_optimizer_updates=0,
                refit_optimizer_updates=0,
                development_eligible=None,
                artifact_schema_traversal_complete=False,
                p12_schema_exercised=False,
                candidate_manifest_registered=False,
                last_reached_prefix=CampaignPrefix.P3.name,
                development_stratum_initial_scheme=None,
                development_stratum_initial_counts=(),
                development_stratum_final_scheme=None,
                development_stratum_final_counts=(),
                development_stratum_coarsening_reason=None,
                development_o2q_replay_passed=None,
                development_evaluable_analysis_unit_count=None,
                development_chart_analysis_unit_counts=(),
                development_chart_token_fractions=(),
                development_out_of_support_rate=None,
                development_out_of_support_passed=None,
                development_independent_support_passed=None,
                development_underpowered_chart_ids=(),
                development_materially_underpowered_chart_ids=(),
                scientific_readiness=False,
                readiness_blockers=(
                    "no positive operator reference after the preregistered lower-K fallback",
                    "synthetic evidence cannot establish scientific readiness",
                ),
                terminal_state=terminal.terminal_state.value,
                final_experiment_manifest_sha256=(
                    terminal.final_experiment_manifest_sha256
                ),
                inventory_entry_count=len(ledger.campaign_inventory()),
            )
        frozen_geometry_sha256 = geometry.identity_sha256
        mlp_codec_candidates = fit_mlp_codec_candidates(campaign, geometry)
        if representation.epsilon_down != epsilon_down:
            raise CampaignIntegrationError(
                "independent epsilon_down producers disagree"
            )
        _register_pre_tune_representation(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            campaign=campaign,
            qualification=representation,
            o1_bodies=o1_bodies,
            partition_records=partition_records,
            hashes=hashes,
        )

        _capture_partition(
            ledger,
            capture_root,
            campaign.tune,
            campaign_id=campaign_id,
            profile=profile,
            row_derivation_protocol_artifact_sha256=hashes[
                "row_derivation_protocol.json"
            ],
            maximum_rows_per_shard=maximum_rows_per_shard,
            hashes=hashes,
            frozen_source_inventory=frozen_source_inventory,
            frozen_capture_implementation_hash=(
                frozen_capture_implementation_hash
            ),
            authorization_bindings={
                **capture_identity_bindings,
                "campaign_root_sha256": hashes["campaign_root.json"],
                "selection_representation_manifest_sha256": hashes[
                    "selection_representation_manifest.json"
                ],
                "codec_protocol_sha256": hashes["codec_protocol.json"],
                "ceiling_gate_protocol_sha256": hashes[
                    "ceiling_gate_protocol.json"
                ],
            },
        )
        o2_evaluations, o2_decision = evaluate_o2(campaign, geometry)
        o2_bodies = _o2_bodies(o2_evaluations)
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P5,
            hashes=hashes,
            body_overrides=o2_bodies,
        )
        if not o2_decision.may_fit_j_models or o2_decision.selected_codec_id is None:
            raise ProtocolStateError("synthetic O2q qualification selected no codec")
        codec_id = o2_decision.selected_codec_id
        codec = geometry.codec(codec_id)
        selected_codec_body = synthetic_body_template("selected_codec.json")
        selected_codec_body.update(
            {
                "codec_id": codec.codec_id,
                "q_rank": codec.q_rank,
                "v_rank": codec.v_rank,
                "lambda_ratio": codec.regularization_ratio,
                "o2q_result_sha256": hashes["o2q_results.json"],
                "o2v_result_sha256": hashes["o2v_results.json"],
            }
        )
        frozen_representation = synthetic_body_template(
            "frozen_representation_manifest.json"
        )
        frozen_representation.update(
            {
                "selection_representation_sha256": geometry.identity_sha256,
                "selected_codec_sha256": codec.identity_sha256,
                "bindings": [
                    {
                        "object_name": "synthetic_selection_geometry",
                        "artifact_sha256": geometry.identity_sha256,
                        "immutable": True,
                    },
                    {
                        "object_name": f"synthetic_codec_{codec_id}",
                        "artifact_sha256": codec.identity_sha256,
                        "immutable": True,
                    },
                ],
            }
        )
        m2_diagnostics = evaluate_m2_diagnostics(
            campaign,
            geometry,
            mlp_codec_candidates,
            codec_id,
        )
        compact_oracles = evaluate_compact_oracles(
            campaign,
            geometry,
            codec_id,
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P6,
            hashes=hashes,
            body_overrides={
                "selected_codec.json": selected_codec_body,
                "frozen_representation_manifest.json": frozen_representation,
                "m2q_results.json": m2_diagnostics.artifact_body("M2q"),
                "m2v_results.json": m2_diagnostics.artifact_body("M2v"),
                "compact_ambient_oracle.json": compact_oracles.artifact_body(
                    "C-Ambient"
                ),
                "compact_chart_oracle.json": compact_oracles.artifact_body(
                    "C-Chart"
                ),
            },
        )
        _assert_geometry_unchanged(
            geometry, frozen_geometry_sha256, phase="tune/O2 qualification"
        )

        initial = fit_initial_model_banks(
            campaign,
            geometry,
            codec_id,
            config=selected_profile.initial,
            diagnostic_config=selected_profile.diagnostic,
            restart_seeds=_INITIAL_RESTART_SEEDS,
            include_optional_j2=False,
        )
        initial_monitor = _evaluate_fits(
            initial.fits,
            initial.monitor_panel,
            geometry,
            codec_id,
            campaign,
            epsilon_down=epsilon_down,
        )
        initial_tune = _evaluate_fits(
            initial.fits,
            initial.tune_panel,
            geometry,
            codec_id,
            campaign,
            epsilon_down=epsilon_down,
        )
        fit_body = synthetic_body_template("j_model_fit_results.json")
        fit_body.update(
            {
                "shared_representation_sha256": geometry.identity_sha256,
                "models": _model_records_with_representation(
                    initial.fits,
                    initial_monitor,
                    geometry.identity_sha256,
                    status="fit",
                ),
            }
        )
        tune_body = synthetic_body_template("j_model_tune_results.json")
        tune_body.update(
            {
                "shared_representation_sha256": geometry.identity_sha256,
                "models": _model_records_with_representation(
                    initial.fits,
                    initial_tune,
                    geometry.identity_sha256,
                    status="evaluated",
                ),
            }
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P7,
            hashes=hashes,
            body_overrides={
                "j_model_fit_results.json": fit_body,
                "j_model_tune_results.json": tune_body,
            },
        )
        _assert_geometry_unchanged(
            geometry, frozen_geometry_sha256, phase="initial model fitting"
        )

        refit = fit_refit_model_banks(
            campaign,
            geometry,
            codec_id,
            config=selected_profile.refit,
            seed=_REFIT_SEED,
        )
        optimizer_updates = initial.optimizer_updates + refit.optimizer_updates
        if optimizer_updates > _MAXIMUM_CAMPAIGN_OPTIMIZER_UPDATES:
            raise ProtocolStateError(
                "campaign exceeded the frozen 40,000 optimizer-update budget"
            )
        refit_tune = _evaluate_fits(
            refit.fits,
            refit.tune_panel,
            geometry,
            codec_id,
            campaign,
            epsilon_down=epsilon_down,
        )
        replay_records = []
        for model_id in ("J2N", "J2+", "J3", "J4"):
            before = initial_tune[model_id].downstream.mean_loss
            after = refit_tune[model_id].downstream.mean_loss
            relative = (after - before) / max(before, 1e-12)
            replay_records.append(
                {
                    "model_id": model_id,
                    "pre_refit_L_bar_down": before,
                    "post_refit_L_bar_down": after,
                    "relative_degradation": relative,
                    "stable": after <= 1.02 * before,
                }
            )
        refit_body = synthetic_body_template("all_fit_weight_refit.json")
        refit_body.update(
            {
                "representation_before_sha256": geometry.identity_sha256,
                "representation_after_sha256": geometry.identity_sha256,
                "weights_manifest_sha256": hashlib.sha256(
                    b"".join(
                        bytes.fromhex(refit.fits[name].weights_sha256)
                        for name in ("J2N", "J2+", "J3", "J4")
                    )
                ).hexdigest(),
            }
        )
        replay_body = synthetic_body_template("nonselecting_tune_replay.json")
        replay_body.update(
            {
                "models": replay_records,
                "all_required_models_stable": all(
                    bool(record["stable"]) for record in replay_records
                ),
            }
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P8,
            hashes=hashes,
            body_overrides={
                "all_fit_weight_refit.json": refit_body,
                "nonselecting_tune_replay.json": replay_body,
            },
        )
        _assert_geometry_unchanged(
            geometry, frozen_geometry_sha256, phase="all-fit refit"
        )
        replay_stable = all(bool(record["stable"]) for record in replay_records)
        if not replay_stable:
            terminal = ledger.finalize_development_outcome(
                campaign_id=campaign_id,
                terminal_state=(
                    TerminalState.DEVELOPMENT_COMPLETE_REFIT_INSTABILITY
                ),
                reason_code="all_fit_tune_replay_exceeded_two_percent",
            )
            _sync_hashes(ledger, hashes)
            _export_ledger_artifacts(ledger, artifact_root)
            inventory_count = len(ledger.campaign_inventory())
            return SyntheticCampaignResult(
                campaign_id=campaign_id,
                profile=profile,
                exact_training_contract=selected_profile.exact_contract,
                output_directory=root,
                ledger_path=ledger_path,
                selected_codec_id=codec_id,
                frozen_baseline_id=None,
                o1_passed=o1.passed,
                o2q_passing_codec_ids=tuple(
                    o2_decision.ranked_passing_codec_ids
                ),
                initial_optimizer_updates=initial.optimizer_updates,
                refit_optimizer_updates=refit.optimizer_updates,
                development_eligible=None,
                artifact_schema_traversal_complete=False,
                p12_schema_exercised=False,
                candidate_manifest_registered=False,
                last_reached_prefix=CampaignPrefix.P8.name,
                development_stratum_initial_scheme=None,
                development_stratum_initial_counts=(),
                development_stratum_final_scheme=None,
                development_stratum_final_counts=(),
                development_stratum_coarsening_reason=None,
                development_o2q_replay_passed=None,
                development_evaluable_analysis_unit_count=None,
                development_chart_analysis_unit_counts=(),
                development_chart_token_fractions=(),
                development_out_of_support_rate=None,
                development_out_of_support_passed=None,
                development_independent_support_passed=None,
                development_underpowered_chart_ids=(),
                development_materially_underpowered_chart_ids=(),
                scientific_readiness=False,
                readiness_blockers=(
                    "all-fit refit failed the two-percent tune stability gate",
                    "synthetic evidence cannot establish scientific readiness",
                ),
                terminal_state=terminal.terminal_state.value,
                final_experiment_manifest_sha256=(
                    terminal.final_experiment_manifest_sha256
                ),
                inventory_entry_count=inventory_count,
            )
        direct_full_rank_null = evaluate_direct_full_rank_identity_null(
            campaign,
            geometry,
            full_rank_smoke,
            codec_id,
        )
        if not direct_full_rank_null.passed:
            raise CampaignIntegrationError(
                "post-refit direct full-rank identity null failed"
            )

        baseline_evaluations = dict(initial_tune)
        baseline_evaluations.update(refit_tune)
        frozen_baseline_id = min(
            REQUIRED_BASELINES,
            key=lambda name: (
                baseline_evaluations[name].downstream.mean_loss,
                REQUIRED_BASELINES.index(name),
            ),
        )
        bundle_fits = dict(initial.fits)
        bundle_fits.update(refit.fits)
        candidate_bundle, comparator_bundle = build_weights_only_bundles(
            bundle_fits,
            baseline_id=frozen_baseline_id,
            geometry=geometry,
            codec_id=codec_id,
            interaction_means=refit.interaction_means,
            campaign=campaign,
        )
        validate_bundle_bytes("candidate_bundle.safetensors", candidate_bundle)
        validate_bundle_bytes("comparator_bundle.safetensors", comparator_bundle)
        bundle_self_test = self_test_weights_only_bundles(
            candidate_bundle, comparator_bundle
        )
        if not bundle_self_test.passed:
            raise CampaignIntegrationError(
                "deterministic execution differs after weights-only bundle reload"
            )
        loaded_bundles = load_weights_only_bundles(
            candidate_bundle, comparator_bundle
        )
        frozen_b = synthetic_body_template("frozen_B_star.json")
        frozen_b.update(
            {
                "model_id": frozen_baseline_id,
                "selected_value": baseline_evaluations[
                    frozen_baseline_id
                ].downstream.mean_loss,
                "tune_replay_sha256": hashes["nonselecting_tune_replay.json"],
            }
        )
        _register_json_artifact(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            name="frozen_B_star.json",
            hashes=hashes,
            body=frozen_b,
        )
        _register_json_artifact(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            name="direct_full_rank_identity_null.json",
            hashes=hashes,
            body=direct_full_rank_null.as_artifact_body(),
        )
        for name, content in (
            ("candidate_bundle.safetensors", candidate_bundle),
            ("comparator_bundle.safetensors", comparator_bundle),
        ):
            _register_external(
                ledger,
                artifact_root,
                campaign_id=campaign_id,
                prefix=CampaignPrefix.P9,
                name=name,
                content=content,
                hashes=hashes,
            )
        metadata_body = synthetic_body_template("candidate_bundle_metadata.json")
        metadata_body.update(
            {
                "candidate_bundle_byte_sha256": artifact_sha256(candidate_bundle),
                "candidate_bundle_byte_size": len(candidate_bundle),
                "comparator_bundle_byte_sha256": artifact_sha256(comparator_bundle),
                "comparator_bundle_byte_size": len(comparator_bundle),
                "selected_representation_sha256": geometry.identity_sha256,
                "source_inventory_sha256": hashes["source_inventory.json"],
                "model_identity_sha256": hashes["model_identity.json"],
                "carrier_identity_sha256": hashes[
                    "fisher_carrier_identity.json"
                ],
                "runtime_identity_sha256": hashes["runtime_identity.json"],
                "row_derivation_protocol_sha256": hashes[
                    "row_derivation_protocol.json"
                ],
                "campaign_root_sha256": hashes["campaign_root.json"],
                "ceiling_gate_protocol_sha256": hashes[
                    "ceiling_gate_protocol.json"
                ],
                "learning_protocol_sha256": hashes["learning_protocol.json"],
                "development_eligibility_protocol_sha256": hashes[
                    "development_eligibility_protocol.json"
                ],
                "full_rank_identity_smoke_sha256": hashes[
                    "full_rank_identity_smoke.json"
                ],
                "tensors": (
                    _bundle_tensor_records(candidate_bundle, "candidate")
                    + _bundle_tensor_records(comparator_bundle, "comparator")
                ),
                "deterministic_self_test_sha256": (
                    bundle_self_test.output_sha256
                ),
                "deterministic_self_test_passed": bundle_self_test.passed,
            }
        )
        _register_json_artifact(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            name="candidate_bundle_metadata.json",
            hashes=hashes,
            body=metadata_body,
        )
        _capture_partition(
            ledger,
            capture_root,
            campaign.development_confirm,
            campaign_id=campaign_id,
            profile=profile,
            row_derivation_protocol_artifact_sha256=hashes[
                "row_derivation_protocol.json"
            ],
            maximum_rows_per_shard=maximum_rows_per_shard,
            hashes=hashes,
            frozen_source_inventory=frozen_source_inventory,
            frozen_capture_implementation_hash=(
                frozen_capture_implementation_hash
            ),
            authorization_bindings={
                **capture_identity_bindings,
                "campaign_root_sha256": hashes["campaign_root.json"],
                "frozen_representation_manifest_sha256": hashes[
                    "frozen_representation_manifest.json"
                ],
                "selected_codec_artifact_sha256": hashes[
                    "selected_codec.json"
                ],
                "selected_codec_id": codec_id,
                "candidate_bundle_byte_sha256": hashes[
                    "candidate_bundle.safetensors"
                ],
                "comparator_bundle_byte_sha256": hashes[
                    "comparator_bundle.safetensors"
                ],
                "candidate_bundle_metadata_sha256": hashes[
                    "candidate_bundle_metadata.json"
                ],
                "candidate_bundle_metadata_canonical_payload_sha256": hashes[
                    "candidate_bundle_metadata.json"
                ],
                "frozen_b_star_model_id": frozen_baseline_id,
                "frozen_b_star_artifact_sha256": hashes[
                    "frozen_B_star.json"
                ],
                "nonselecting_tune_replay_sha256": hashes[
                    "nonselecting_tune_replay.json"
                ],
                "nonselecting_tune_replay_result_sha256": hashes[
                    "nonselecting_tune_replay.json"
                ],
                "development_eligibility_protocol_sha256": hashes[
                    "development_eligibility_protocol.json"
                ],
            },
        )

        development_panel = prepare_completion_panel(
            campaign.development_confirm, campaign, geometry, codec_id
        )
        development_support = assess_development_support(development_panel)
        if not any(
            count >= 2
            for _, count in development_support.chart_analysis_unit_counts
        ):
            support_spec = _development_operator_support_failure_spec(
                campaign_id=campaign_id,
                campaign=campaign,
                representation=representation,
                support=development_support,
                codec_id=codec_id,
            )
            terminal = ledger.finalize_support_failure(
                support_spec,
                omitted_active_artifacts={
                    name: "no_development_operator_pair_has_two_units"
                    for name in STAGE_ARTIFACT_ORDER[Stage.P11]
                },
            )
            _sync_hashes(ledger, hashes)
            _export_ledger_artifacts(ledger, artifact_root)
            return SyntheticCampaignResult(
                campaign_id=campaign_id,
                profile=profile,
                exact_training_contract=selected_profile.exact_contract,
                output_directory=root,
                ledger_path=ledger_path,
                selected_codec_id=codec_id,
                frozen_baseline_id=frozen_baseline_id,
                o1_passed=o1.passed,
                o2q_passing_codec_ids=tuple(
                    o2_decision.ranked_passing_codec_ids
                ),
                initial_optimizer_updates=initial.optimizer_updates,
                refit_optimizer_updates=refit.optimizer_updates,
                development_eligible=None,
                artifact_schema_traversal_complete=False,
                p12_schema_exercised=False,
                candidate_manifest_registered=False,
                last_reached_prefix=CampaignPrefix.P10.name,
                development_stratum_initial_scheme=None,
                development_stratum_initial_counts=(),
                development_stratum_final_scheme=None,
                development_stratum_final_counts=(),
                development_stratum_coarsening_reason=None,
                development_o2q_replay_passed=None,
                development_evaluable_analysis_unit_count=(
                    development_support.evaluable_analysis_unit_count
                ),
                development_chart_analysis_unit_counts=(
                    development_support.chart_analysis_unit_counts
                ),
                development_chart_token_fractions=(
                    development_support.chart_token_fractions
                ),
                development_out_of_support_rate=(
                    development_support.out_of_support_rate
                ),
                development_out_of_support_passed=(
                    development_support.out_of_support_passed
                ),
                development_independent_support_passed=(
                    development_support.independent_support_passed
                ),
                development_underpowered_chart_ids=(
                    development_support.underpowered_chart_ids
                ),
                development_materially_underpowered_chart_ids=(
                    development_support.materially_underpowered_chart_ids
                ),
                scientific_readiness=False,
                readiness_blockers=(
                    "no development chart/head operator pair has two independent units",
                    "synthetic evidence cannot establish scientific readiness",
                ),
                terminal_state=terminal.terminal_state.value,
                final_experiment_manifest_sha256=(
                    terminal.final_experiment_manifest_sha256
                ),
                inventory_entry_count=len(ledger.campaign_inventory()),
            )
        development_o2q = evaluate_o2q_replay(
            campaign.development_confirm,
            campaign,
            geometry,
            codec_id,
        )
        development_o2q_panel = _fidelity_panel(
            "development_confirm", development_o2q
        )
        development_fits = dict(initial.fits)
        development_fits.update(refit.fits)
        for name in ("J1", "J2N", "J2+", "J3", "J4"):
            development_fits[name] = replace(
                development_fits[name], bank=loaded_bundles.bank(name)
            )
        selected_representation_codec = next(
            (
                item
                for item in representation.codec_qualifications
                if item.codec_id == codec_id
            ),
            None,
        )
        if (
            selected_representation_codec is None
            or selected_representation_codec.operator_reference.tau_r is None
        ):
            raise ProtocolStateError(
                "selected codec has no frozen operator tolerance"
            )
        trained_operator_aggregate = trained_j4_operator_aggregate(
            development_fits["J4"],
            development_panel,
            geometry,
            campaign,
            codec_id=codec_id,
            tau_r=selected_representation_codec.operator_reference.tau_r,
        )
        development_evaluations = _evaluate_loaded_bundle_models(
            loaded_bundles,
            ("J1", "J2N", "J2+", "J3", "J4"),
            campaign.development_confirm,
            development_panel,
            campaign,
            epsilon_down=epsilon_down,
        )
        strata, stratum_decision = _unit_strata(
            campaign.development_confirm, development_evaluations["J4"]
        )
        comparisons = model_comparisons(
            development_evaluations,
            strata=strata,
            replicates=10_000,
        )
        interaction = evaluate_interaction_mean(
            development_fits["J4"],
            development_evaluations[frozen_baseline_id],
            development_evaluations["J4"],
            development_panel,
            refit.interaction_means,
            geometry.codec(codec_id),
            campaign,
            epsilon_down=epsilon_down,
            strata=strata,
            replicates=10_000,
        )
        comparison_records = []
        for comparison in comparisons:
            bootstrap = comparison.result
            baseline = development_evaluations[comparison.baseline]
            comparison_records.append(
                {
                    "baseline": comparison.baseline,
                    "L_bar_down_baseline": baseline.downstream.mean_loss,
                    "L_bar_down_J4": development_evaluations[
                        "J4"
                    ].downstream.mean_loss,
                    "rho": bootstrap.rho_point.rho,
                    "rho_status": bootstrap.rho_point.rho_status,
                    "rho_point_screen_passed": (
                        bootstrap.rho_point.relative_screen_passed
                    ),
                    "paired_ci_lower": bootstrap.paired_effect_interval[0],
                    "paired_ci_upper": bootstrap.paired_effect_interval[1],
                    "paired_positive_passed": (
                        bootstrap.paired_effect_interval[0] > 0.0
                    ),
                    "valid_rho_replicate_fraction": (
                        bootstrap.valid_rho_replicate_fraction
                    ),
                    "comparison_passed": comparison.passed,
                }
            )
        development_body = synthetic_body_template(
            "development_confirm_results.json"
        )
        development_body.update(
            {
                "analysis_unit_count": len(
                    campaign.development_confirm.analysis_unit_commitments
                ),
                "evaluable_analysis_unit_count": (
                    development_support.evaluable_analysis_unit_count
                ),
                "out_of_support_rate": development_support.out_of_support_rate,
                "active_chart_unit_counts": _chart_support_records(
                    campaign.development_confirm,
                    development_panel,
                    development_support,
                ),
                "models": _model_records_with_representation(
                    development_fits,
                    development_evaluations,
                    geometry.identity_sha256,
                    status="evaluated",
                    model_order=("J1", "J2N", "J2+", "J3", "J4"),
                ),
                "comparisons": comparison_records,
                "trained_J4_operator_aggregate": _trained_operator_aggregate_body(
                    trained_operator_aggregate,
                    codec_id=codec_id,
                    tau_r=(
                        selected_representation_codec.operator_reference.tau_r
                    ),
                    selected_codec_sha256=hashes["selected_codec.json"],
                    numeric_constants_sha256=hashes[
                        "representation_numeric_constants.json"
                    ],
                    codec_manifests_sha256=hashes[
                        "codec_candidate_manifests.json"
                    ],
                ),
                "stratum_scheme": _stratum_scheme_for_artifact(
                    stratum_decision.final_scheme
                ),
            }
        )
        interaction_body = synthetic_body_template("interaction_mean_ablation.json")
        interaction_body.update(
            {
                "baseline_model": frozen_baseline_id,
                "advantage_J4": interaction.advantage,
                "advantage_floor": interaction.advantage_floor,
                "gamma_interaction": interaction.gamma,
                "gamma_status": (
                    "regular"
                    if interaction.gamma is not None
                    else "advantage_at_or_below_floor"
                ),
                "gamma_point_screen_passed": (
                    interaction.gamma is not None and interaction.gamma >= 0.25
                ),
                "paired_ci_lower": interaction.paired_interval[0],
                "paired_ci_upper": interaction.paired_interval[1],
                "paired_positive_passed": interaction.paired_interval[0] > 0.0,
                "valid_gamma_replicate_fraction": interaction.valid_fraction,
                "interaction_criterion_passed": interaction.passed,
            }
        )
        o7_body = synthetic_body_template(
            "o7_attention_output_completion_results.json"
        )
        j4_fidelity = development_evaluations["J4"].fidelity
        if j4_fidelity is None:
            raise ProtocolStateError("J4 produced no O7 fidelity measurement")
        o7_body["panel"] = _fidelity_panel(
            "development_confirm", j4_fidelity
        )
        o7_body["all_five_passed"] = o7_body["panel"]["all_five_passed"]
        eligibility_body = synthetic_body_template(
            "development_eligibility_result.json"
        )
        eligibility_screens = {
            "o2q_replay_passed": bool(
                development_o2q_panel["all_five_passed"]
            ),
            "all_J4_comparisons_passed": all(
                comparison.passed for comparison in comparisons
            ),
            "interaction_mean_passed": interaction.passed,
            "o7_all_five_passed": bool(o7_body["all_five_passed"]),
            "out_of_support_passed": development_support.out_of_support_passed,
            "independent_support_passed": (
                development_support.independent_support_passed
            ),
        }
        development_eligible = all(eligibility_screens.values())
        terminal_if_finalized, reason_code = _development_terminal_decision(
            local_comparisons_passed=bool(
                eligibility_screens["all_J4_comparisons_passed"]
            ),
            interaction_mean_passed=bool(
                eligibility_screens["interaction_mean_passed"]
            ),
            o2q_replay_passed=bool(
                eligibility_screens["o2q_replay_passed"]
            ),
            o7_passed=bool(eligibility_screens["o7_all_five_passed"]),
            support_passed=(
                bool(eligibility_screens["out_of_support_passed"])
                and bool(eligibility_screens["independent_support_passed"])
            ),
        )
        eligibility_body.update(
            {
                **eligibility_screens,
                "development_eligible": development_eligible,
                "terminal_if_finalized": terminal_if_finalized,
                "reason_code": reason_code,
            }
        )
        j4_units = development_evaluations["J4"].downstream.units
        baseline_by_unit = {
            item.analysis_unit_commitment: item.loss
            for item in development_evaluations[
                frozen_baseline_id
            ].downstream.units
        }
        power = project_development_power(
            tuple(baseline_by_unit[item.analysis_unit_commitment] for item in j4_units),
            tuple(item.loss for item in j4_units),
            strata,
            _power_chart_memberships(
                campaign.development_confirm,
                development_panel,
                development_evaluations["J4"],
            ),
            unit_ids=tuple(item.analysis_unit_commitment for item in j4_units),
            active_chart_ids=development_panel.active_chart_ids,
        )
        _register_json_stage(
            ledger,
            artifact_root,
            campaign_id=campaign_id,
            stage=Stage.P11,
            hashes=hashes,
            body_overrides={
                "development_confirm_results.json": development_body,
                "interaction_mean_ablation.json": interaction_body,
                "o7_attention_output_completion_results.json": o7_body,
                "development_eligibility_result.json": eligibility_body,
                "development_power_assessment.json": power.as_artifact_body(),
            },
        )
        _assert_geometry_unchanged(
            geometry,
            frozen_geometry_sha256,
            phase="development-confirm evaluation",
        )

        candidate_body = synthetic_body_template(
            "development_candidate_manifest.json"
        )
        candidate_body.update(
            {
                "source_inventory_sha256": hashes["source_inventory.json"],
                "model_identity_sha256": hashes["model_identity.json"],
                "tokenizer_identity_sha256": str(
                    synthetic_row_derivation_protocol_body()[
                        "tokenizer_identity_sha256"
                    ]
                ),
                "carrier_identity_sha256": hashes["fisher_carrier_identity.json"],
                "runtime_identity_sha256": hashes["runtime_identity.json"],
                "execution_budget_sha256": hashes["execution_budget.json"],
                "split_manifest_sha256": hashes["split_manifest.json"],
                "partition_commitments": partition_records,
                "capture_manifest_sha256s": [
                    {
                        "name": role,
                        "artifact_sha256": hashes[
                            f"capture_shard_manifest_{role}.json"
                        ],
                    }
                    for role in ("fit", "tune", "development_confirm")
                ],
                "freshness_receipt_sha256": hashes[
                    "local_development_freshness_audit_receipt.json"
                ],
                "campaign_root_sha256": hashes["campaign_root.json"],
                "frozen_representation_sha256": hashes[
                    "frozen_representation_manifest.json"
                ],
                "selected_codec_sha256": hashes["selected_codec.json"],
                "candidate_bundle_byte_sha256": hashes[
                    "candidate_bundle.safetensors"
                ],
                "comparator_bundle_byte_sha256": hashes[
                    "comparator_bundle.safetensors"
                ],
                "candidate_bundle_metadata_sha256": hashes[
                    "candidate_bundle_metadata.json"
                ],
                "frozen_B_star_sha256": hashes["frozen_B_star.json"],
                "development_confirm_result_sha256": hashes[
                    "development_confirm_results.json"
                ],
                "development_eligibility_result_sha256": hashes[
                    "development_eligibility_result.json"
                ],
                "o7_result_sha256": hashes[
                    "o7_attention_output_completion_results.json"
                ],
            }
        )
        candidate_dependencies = artifact_dependencies(
            "development_candidate_manifest.json"
        )
        candidate_payload = build_artifact_payload(
            "development_candidate_manifest.json",
            campaign_id=campaign_id,
            dependency_hashes={
                name: hashes[name] for name in candidate_dependencies
            },
            body=candidate_body,
        )
        validate_artifact_payload(
            "development_candidate_manifest.json", candidate_payload
        )
        qualification_blockers = _qualification_blockers(
            profile=selected_profile,
            replay_records=replay_records,
            candidate_bundle=candidate_bundle,
            comparator_bundle=comparator_bundle,
            development_eligible=bool(eligibility_body["development_eligible"]),
        )
        candidate_manifest_registered = False
        terminal_state: str | None = None
        final_manifest_sha256: str | None = None
        if not qualification_blockers:
            _register_json_artifact(
                ledger,
                artifact_root,
                campaign_id=campaign_id,
                name="development_candidate_manifest.json",
                hashes=hashes,
                body=candidate_body,
            )
            terminal = ledger.finalize_successful_campaign(
                campaign_id=campaign_id
            )
            candidate_manifest_registered = True
            last_reached_prefix = CampaignPrefix.P12.name
            terminal_state = terminal.terminal_state.value
            final_manifest_sha256 = terminal.final_experiment_manifest_sha256
        elif selected_profile.exact_contract:
            if bool(eligibility_body["development_eligible"]):
                raise ProtocolStateError(
                    "exact development passed but a mechanical P12 blocker remains"
                )
            outcome = TerminalState(str(eligibility_body["terminal_if_finalized"]))
            if outcome is TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT:
                support_spec = _development_support_failure_spec(
                    campaign_id=campaign_id,
                    campaign=campaign,
                    representation=representation,
                    support=development_support,
                )
                terminal = ledger.finalize_support_failure(
                    support_spec,
                    omitted_active_artifacts={
                        "development_candidate_manifest.json": (
                            "development_confirm_insufficient_support"
                        )
                    },
                )
                last_reached_prefix = CampaignPrefix.P11.name
            else:
                terminal = ledger.finalize_development_outcome(
                    campaign_id=campaign_id,
                    terminal_state=outcome,
                    reason_code=str(eligibility_body["reason_code"]),
                )
                last_reached_prefix = CampaignPrefix.P11.name
            terminal_state = terminal.terminal_state.value
            final_manifest_sha256 = terminal.final_experiment_manifest_sha256
        else:
            prefix_closure = validate_reached_prefix_closure(
                ledger.campaign_inventory(),
                last_completed_prefix=CampaignPrefix.P11,
                active_prefix=CampaignPrefix.P12,
                omitted_active_artifacts={
                    "development_candidate_manifest.json": (
                        "quick_profile_not_candidate_eligible"
                    )
                },
            )
            if not prefix_closure.valid:
                raise ProtocolStateError(
                    "quick synthetic qualification prefix is incomplete"
                )
            last_reached_prefix = CampaignPrefix.P11.name
        _sync_hashes(ledger, hashes)
        _export_ledger_artifacts(ledger, artifact_root)
        inventory_count = len(ledger.campaign_inventory())

        readiness_blockers = tuple(qualification_blockers) + (
            "synthetic evidence cannot establish scientific readiness",
        )

    return SyntheticCampaignResult(
        campaign_id=campaign_id,
        profile=profile,
        exact_training_contract=selected_profile.exact_contract,
        output_directory=root,
        ledger_path=ledger_path,
        selected_codec_id=codec_id,
        frozen_baseline_id=frozen_baseline_id,
        o1_passed=o1.passed,
        o2q_passing_codec_ids=tuple(o2_decision.ranked_passing_codec_ids),
        initial_optimizer_updates=initial.optimizer_updates,
        refit_optimizer_updates=refit.optimizer_updates,
        development_eligible=bool(eligibility_body["development_eligible"]),
        artifact_schema_traversal_complete=True,
        p12_schema_exercised=True,
        candidate_manifest_registered=candidate_manifest_registered,
        last_reached_prefix=last_reached_prefix,
        development_stratum_initial_scheme=(
            stratum_decision.initial_scheme
        ),
        development_stratum_initial_counts=tuple(
            (_stratum_label_key(label), count)
            for label, count in stratum_decision.initial_counts
        ),
        development_stratum_final_scheme=stratum_decision.final_scheme,
        development_stratum_final_counts=tuple(
            (_stratum_label_key(label), count)
            for label, count in stratum_decision.final_counts
        ),
        development_stratum_coarsening_reason=(
            stratum_decision.coarsening_reason
        ),
        development_o2q_replay_passed=bool(
            development_o2q_panel["all_five_passed"]
        ),
        development_evaluable_analysis_unit_count=(
            development_support.evaluable_analysis_unit_count
        ),
        development_chart_analysis_unit_counts=(
            development_support.chart_analysis_unit_counts
        ),
        development_chart_token_fractions=(
            development_support.chart_token_fractions
        ),
        development_out_of_support_rate=development_support.out_of_support_rate,
        development_out_of_support_passed=(
            development_support.out_of_support_passed
        ),
        development_independent_support_passed=(
            development_support.independent_support_passed
        ),
        development_underpowered_chart_ids=(
            development_support.underpowered_chart_ids
        ),
        development_materially_underpowered_chart_ids=(
            development_support.materially_underpowered_chart_ids
        ),
        scientific_readiness=False,
        readiness_blockers=readiness_blockers,
        terminal_state=terminal_state,
        final_experiment_manifest_sha256=final_manifest_sha256,
        inventory_entry_count=inventory_count,
    )
