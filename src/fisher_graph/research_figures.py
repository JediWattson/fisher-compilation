"""Render the current research summary as deterministic, accessible SVGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_SUMMARY = Path("artifacts/research/current_research_summary_v1.json")
DEFAULT_LADDER_OUTPUT = Path("docs/images/research-ladder.svg")
DEFAULT_DIAGNOSTIC_OUTPUT = Path("docs/images/l3-l4-rank-diagnostic.svg")
DEFAULT_BILINEAR_OUTPUT = Path(
    "docs/images/bilinear-spectral-assessment.svg"
)
DEFAULT_ATTENUATION_OUTPUT = Path(
    "docs/images/reference-provider-collision-attenuation.svg"
)
DEFAULT_V3_ASSESSMENT_OUTPUT = Path(
    "docs/images/reference-provider-v3-assessment.svg"
)


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    path: str
    sha256: str
    sha256_kind: str
    source_status: str


@dataclass(frozen=True)
class ResearchStage:
    stage_id: str
    title: str
    status: str
    resource: str
    fidelity: str
    claim: str


@dataclass(frozen=True)
class RankDiagnostic:
    rank: int
    source_reconstruction_relative_l2: float
    target_reconstruction_relative_l2: float
    in_sample_jvp_relative_residual: float
    pair_output_cosine: float
    pair_output_relative_l2: float
    positive_lag_energy_fraction: float
    pair_parameter_fraction_of_flat: float
    whole_model_parameter_fraction_of_source: float


@dataclass(frozen=True)
class L3L4Diagnostic:
    fit_sequences: int
    probe_sequences: int
    jvp_probes_per_sequence: int
    logical_lags: tuple[int, ...]
    content_disjoint: bool
    family_disjoint: bool
    reference_provider_compiled: bool
    pair_control_scope: str
    reported_probe_statistic: str
    accounting_status: str
    accounting_correction: str
    rank_results: tuple[RankDiagnostic, ...]
    interpretation: str
    not_proven: str


@dataclass(frozen=True)
class BilinearDiagnostic:
    protocol_scope: str
    assessment_origin: int
    assessment_refit_performed: bool
    selected_plan_kind: str
    selected_source_rank: int
    selected_target_rank: int
    selected_plan_stored_coefficient_count: int
    direct_dense_branch_coefficient_count: int
    selected_fraction_of_direct_dense: float
    branch_coefficient_reduction_fraction: float
    base_plus_branch_stored_coefficient_count: int
    matched_dense_three_branch_coefficient_count: int
    combined_fraction_of_matched_dense: float
    combined_coefficient_reduction_fraction: float
    selection_base_relative_error: float
    selection_augmented_relative_error: float
    selection_error_reduction_fraction: float
    selection_augmented_cosine: float
    assessment_base_relative_error: float
    assessment_augmented_relative_error: float
    assessment_error_reduction_fraction: float
    assessment_augmented_cosine: float
    assessment_c11_relative_error: float
    assessment_c11_cosine: float
    decision: str
    prompt_conditioned_reference_provider_compiled: bool
    nll_measured: bool
    model_parameter_compression_claim: bool
    latency_measured: bool
    positive_pair_identity_generalization_claim: bool
    interpretation: str
    not_proven: str


@dataclass(frozen=True)
class ReferenceProviderDiagnostic:
    protocol_scope: str
    fit_probe_count: int
    selection_probe_count: int
    assessment_probe_count: int
    selected_candidate_id: str
    selected_source_rank: int
    selected_target_rank: int
    selected_stored_scalar_count: int
    dense_provider_stored_scalar_count: int
    selected_fraction_of_dense_provider: float
    provider_scalar_reduction_fraction: float
    selection_fisher_weighted_relative_error: float
    selection_reference_cosine: float
    selection_maximum_per_probe_p90_relative_error: float
    assessment_fisher_weighted_relative_error: float
    assessment_reference_cosine: float
    assessment_maximum_per_probe_p90_relative_error: float
    assessment_worst_family_relative_error: float
    assessment_error_reduction_vs_constant: float
    assessment_error_reduction_vs_position_only: float
    assessment_fidelity_and_structure_gates_passed: int
    assessment_fidelity_and_structure_gate_count: int
    assessment_collision_panel_gate_passed: bool
    assessment_collision_threshold: float
    assessment_minimum_collision_target_relative_difference: float
    assessment_formal_decision: str
    assessment_refit_performed: bool
    assessment_reselection_performed: bool
    assessment_claim_consumed: bool
    natural_prompt_transfer_tested: bool
    nll_measured: bool
    model_parameter_compression_claim: bool
    latency_measured: bool
    interpretation: str
    not_proven: str


@dataclass(frozen=True)
class CollisionAttenuationFamily:
    family_id: str
    pair_count: int
    gate_witness_count: int
    gate_witnesses_at_or_above_threshold: int
    target_relative_difference_minimum: float
    target_relative_difference_median: float
    target_relative_difference_maximum: float
    gate_witness_conclusion: str
    reference_baseline_relative_dilution_count: int
    pre_ff_norm_attenuation_count: int


@dataclass(frozen=True)
class ReferenceProviderCollisionAttenuationDiagnostic:
    protocol_scope: str
    source_report_sha256: str
    logical_artifact_sha256: str
    tensor_file_sha256: str
    source_assessment_artifact_sha256: str
    source_assessment_report_sha256: str
    collision_threshold: float
    collision_endpoint_count: int
    all_target_hashes_match_opened_assessment: bool
    collision_group_count: int
    unordered_pair_count: int
    gate_witness_count: int
    numerically_valid_gate_witness_count: int
    gate_witnesses_at_or_above_threshold: int
    gate_witnesses_below_threshold: int
    families: tuple[CollisionAttenuationFamily, ...]
    reference_baseline_relative_dilution_count: int
    pre_ff_norm_attenuation_count: int
    observations_shared_with_passing_controls: tuple[str, ...]
    observations_exclusive_to_failed_witnesses: tuple[str, ...]
    failed_witnesses_with_retained_fisher_subspace_miss: int
    minimum_failed_witness_retained_64_fisher_energy_fraction: float
    failed_witnesses_with_residual_attention_cancellation: int
    maximum_jvp_vjp_adjoint_relative_error: float
    maximum_pre_source_jvp_energy_fraction: float
    diagnostic_conclusion: str
    candidate_predictions_entered_collision_metric: bool
    assessment_panel_previously_opened: bool
    new_sealed_panel_opened: bool
    assessment_score_recomputed: bool
    candidate_refit_performed: bool
    candidate_reselection_performed: bool
    candidate_tracking_failure_can_be_assigned: bool
    formal_v2_decision_changed: bool
    target_derived_vjp_may_become_compiler_input: bool
    fresh_v3_assessment_required: bool
    interpretation: str
    not_proven: str


@dataclass(frozen=True)
class ReferenceProviderV3ContrastFamily:
    family_id: str
    intent: str
    planned_contrast_count: int
    required_eligible_count: int
    teacher_qualified_contrast_count: int
    candidate_scored_count: int
    candidate_pass_count: int
    decision_status: str
    qualified_rank_strata: tuple[str, ...]
    retained_and_discarded_covered: bool
    macro_rms_contrast_relative_error: float | None
    worst_contrast_relative_error: float | None
    minimum_direction_cosine: float | None
    minimum_projection_gain: float | None
    maximum_projection_gain: float | None
    maximum_orthogonal_leakage: float | None
    maximum_candidate_null_relative_effect_upper: float | None
    maximum_candidate_null_relative_error_upper: float | None


@dataclass(frozen=True)
class ReferenceProviderV3Assessment:
    protocol_scope: str
    source_report_sha256: str
    logical_artifact_sha256: str
    tensor_file_sha256: str
    protocol_sha256: str
    panel_spec_sha256: str
    measured_panel_sha256: str
    code_bundle_sha256: str
    claim_receipt_sha256: str
    claim_uniqueness_sha256: str
    assessment_probe_count: int
    ordinary_fidelity_probe_count: int
    contrast_probe_count: int
    contrast_group_count: int
    contrast_pair_count: int
    selected_candidate_id: str
    selected_stored_scalar_count: int
    assessment_refit_performed: bool
    assessment_reselection_performed: bool
    assessment_claim_consumed: bool
    ordinary_fidelity_passed: bool
    ordinary_fidelity_and_structure_gates_passed: int
    ordinary_fidelity_and_structure_gate_count: int
    ordinary_fidelity_fisher_weighted_relative_error: float
    ordinary_fidelity_reference_cosine: float
    ordinary_fidelity_maximum_per_probe_p90_relative_error: float
    ordinary_fidelity_worst_family_relative_error: float
    ordinary_fidelity_error_reduction_vs_constant: float
    ordinary_fidelity_error_reduction_vs_position_only: float
    ordinary_fidelity_in_support_fraction: float
    ordinary_fidelity_prepared_vs_analytic_relative_error: float
    ordinary_fidelity_causality_violation: float
    ordinary_fidelity_padding_violation: float
    ordinary_fidelity_repeat_relative_error: float
    contrast_overall_status: str
    formal_outcome: str
    provider_passed: bool
    contrast_families: tuple[ReferenceProviderV3ContrastFamily, ...]
    all_families_cover_retained_and_discarded_strata: bool
    weak_teacher_contrasts_entered_candidate_relative_metrics: bool
    intended_null_contrasts_entered_direction_metrics: bool
    candidate_parameters_changed: bool
    natural_prompt_transfer_tested: bool
    nll_measured: bool
    whole_model_replacement_tested: bool
    model_parameter_compression_claim: bool
    latency_measured: bool
    interpretation: str
    next_rung: str
    not_proven: str


@dataclass(frozen=True)
class ResearchFigureData:
    sources: tuple[ResearchSource, ...]
    stages: tuple[ResearchStage, ...]
    diagnostic: L3L4Diagnostic
    bilinear: BilinearDiagnostic
    reference_provider: ReferenceProviderDiagnostic
    collision_attenuation: ReferenceProviderCollisionAttenuationDiagnostic
    reference_provider_v3: ReferenceProviderV3Assessment
    claim_scope: str


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{path} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{path} must be at most {maximum}")
    return result


def _signed_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _integer(
    value: object,
    path: str,
    *,
    minimum: int = 0,
) -> int:
    number = _number(value, path, minimum=float(minimum))
    result = int(number)
    if result != number:
        raise ValueError(f"{path} must be an integer")
    return result


def _sha256(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def extract_research_figure_data(
    summary: Mapping[str, object],
) -> ResearchFigureData:
    """Validate and extract the fields represented by the documentation SVGs."""

    if summary.get("format_version") != 1:
        raise ValueError("summary.format_version must be 1")
    if summary.get("schema") != "fisher_graph.current_research_summary":
        raise ValueError(
            "summary.schema must be fisher_graph.current_research_summary"
        )

    sources: list[ResearchSource] = []
    seen_source_ids: set[str] = set()
    for index, source_value in enumerate(
        _array(summary.get("sources"), "summary.sources")
    ):
        source = _object(source_value, f"summary.sources[{index}]")
        source_id = _string(
            source.get("id"),
            f"summary.sources[{index}].id",
        )
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate research source id: {source_id}")
        seen_source_ids.add(source_id)
        sha256_kind = _string(
            source.get("sha256_kind"),
            f"summary.sources[{index}].sha256_kind",
        )
        if sha256_kind not in {"file_sha256", "report_payload_sha256"}:
            raise ValueError(
                f"unsupported research source SHA kind: {sha256_kind}"
            )
        sources.append(
            ResearchSource(
                source_id=source_id,
                path=_string(
                    source.get("path"),
                    f"summary.sources[{index}].path",
                ),
                sha256=_sha256(
                    source.get("sha256"),
                    f"summary.sources[{index}].sha256",
                ),
                sha256_kind=sha256_kind,
                source_status=_string(
                    source.get("source_status"),
                    f"summary.sources[{index}].source_status",
                ),
            )
        )
    if len(sources) < 2:
        raise ValueError("summary.sources must contain at least two records")
    expected_source_ids = (
        "toy_fused_executor",
        "gemma_structured_layer_v6",
        "gemma_flat_generator_refit",
        "gemma_l3_l4_rank_64",
        "gemma_l3_l4_rank_128",
        "gemma_conditional_assessment",
        "gemma_mixed_falsification",
        "gemma_bilinear_compile",
        "gemma_bilinear_assessment",
        "gemma_reference_provider_v2_compile",
        "gemma_reference_provider_v2_assessment",
        "gemma_reference_provider_v2_attenuation_localization",
        "gemma_reference_provider_v3_assessment",
    )
    if tuple(source.source_id for source in sources) != expected_source_ids:
        raise ValueError(
            "summary.sources must contain the fixed current-research provenance set"
        )
    if sources[0].source_status != "committed_authenticated_report":
        raise ValueError(
            "toy_fused_executor must remain a committed authenticated report"
        )
    if any(
        source.source_status
        != "ignored_local_report_summarized_without_prompts_or_tensors"
        for source in sources[1:]
    ):
        raise ValueError(
            "Gemma research sources must remain source-safe ignored-report summaries"
        )

    allowed_statuses = {
        "verified_reference",
        "fidelity_parent",
        "open_development",
        "analysis_only",
        "next_experiment",
        "frozen_assessment",
        "sealed_mixed_result",
    }
    stages: list[ResearchStage] = []
    seen_stage_ids: set[str] = set()
    for index, stage_value in enumerate(
        _array(summary.get("research_ladder"), "summary.research_ladder")
    ):
        stage = _object(stage_value, f"summary.research_ladder[{index}]")
        stage_id = _string(
            stage.get("id"),
            f"summary.research_ladder[{index}].id",
        )
        if stage_id in seen_stage_ids:
            raise ValueError(f"duplicate research stage id: {stage_id}")
        seen_stage_ids.add(stage_id)
        status = _string(
            stage.get("status"),
            f"summary.research_ladder[{index}].status",
        )
        if status not in allowed_statuses:
            raise ValueError(f"unsupported research stage status: {status}")
        stages.append(
            ResearchStage(
                stage_id=stage_id,
                title=_string(
                    stage.get("title"),
                    f"summary.research_ladder[{index}].title",
                ),
                status=status,
                resource=_string(
                    stage.get("resource"),
                    f"summary.research_ladder[{index}].resource",
                ),
                fidelity=_string(
                    stage.get("fidelity"),
                    f"summary.research_ladder[{index}].fidelity",
                ),
                claim=_string(
                    stage.get("claim"),
                    f"summary.research_ladder[{index}].claim",
                ),
            )
        )
    if len(stages) != 7:
        raise ValueError("summary.research_ladder must contain seven stages")

    diagnostic_value = _object(
        summary.get("l3_l4_diagnostic"),
        "summary.l3_l4_diagnostic",
    )
    logical_lags = tuple(
        _integer(
            lag,
            f"summary.l3_l4_diagnostic.logical_lags[{index}]",
        )
        for index, lag in enumerate(
            _array(
                diagnostic_value.get("logical_lags"),
                "summary.l3_l4_diagnostic.logical_lags",
            )
        )
    )
    if logical_lags != tuple(sorted(set(logical_lags))):
        raise ValueError(
            "summary.l3_l4_diagnostic.logical_lags must be unique and sorted"
        )

    rank_results: list[RankDiagnostic] = []
    for index, rank_value in enumerate(
        _array(
            diagnostic_value.get("rank_results"),
            "summary.l3_l4_diagnostic.rank_results",
        )
    ):
        rank = _object(
            rank_value,
            f"summary.l3_l4_diagnostic.rank_results[{index}]",
        )
        prefix = f"summary.l3_l4_diagnostic.rank_results[{index}]"
        rank_results.append(
            RankDiagnostic(
                rank=_integer(rank.get("rank"), f"{prefix}.rank", minimum=1),
                source_reconstruction_relative_l2=_number(
                    rank.get("source_reconstruction_relative_l2"),
                    f"{prefix}.source_reconstruction_relative_l2",
                ),
                target_reconstruction_relative_l2=_number(
                    rank.get("target_reconstruction_relative_l2"),
                    f"{prefix}.target_reconstruction_relative_l2",
                ),
                in_sample_jvp_relative_residual=_number(
                    rank.get("in_sample_jvp_relative_residual"),
                    f"{prefix}.in_sample_jvp_relative_residual",
                ),
                pair_output_cosine=_number(
                    rank.get("pair_output_cosine"),
                    f"{prefix}.pair_output_cosine",
                    minimum=-1.0,
                    maximum=1.0,
                ),
                pair_output_relative_l2=_number(
                    rank.get("pair_output_relative_l2"),
                    f"{prefix}.pair_output_relative_l2",
                ),
                positive_lag_energy_fraction=_number(
                    rank.get("positive_lag_energy_fraction"),
                    f"{prefix}.positive_lag_energy_fraction",
                    maximum=1.0,
                ),
                pair_parameter_fraction_of_flat=_number(
                    rank.get("pair_parameter_fraction_of_flat"),
                    f"{prefix}.pair_parameter_fraction_of_flat",
                    maximum=1.0,
                ),
                whole_model_parameter_fraction_of_source=_number(
                    rank.get("whole_model_parameter_fraction_of_source"),
                    f"{prefix}.whole_model_parameter_fraction_of_source",
                    maximum=1.0,
                ),
            )
        )
    if [result.rank for result in rank_results] != [64, 128]:
        raise ValueError(
            "summary.l3_l4_diagnostic.rank_results must contain ranks 64 and 128"
        )
    rank_64, rank_128 = rank_results
    improving_metrics = (
        (
            "source reconstruction",
            rank_64.source_reconstruction_relative_l2,
            rank_128.source_reconstruction_relative_l2,
        ),
        (
            "target reconstruction",
            rank_64.target_reconstruction_relative_l2,
            rank_128.target_reconstruction_relative_l2,
        ),
        (
            "in-sample JVP residual",
            rank_64.in_sample_jvp_relative_residual,
            rank_128.in_sample_jvp_relative_residual,
        ),
    )
    for label, value_64, value_128 in improving_metrics:
        if value_128 >= value_64:
            raise ValueError(
                f"rank diagnostic contract requires improving {label}"
            )
    if rank_128.pair_output_cosine >= rank_64.pair_output_cosine:
        raise ValueError(
            "rank diagnostic contract requires worsening pair-output cosine"
        )
    if rank_128.pair_output_relative_l2 <= rank_64.pair_output_relative_l2:
        raise ValueError(
            "rank diagnostic contract requires worsening pair-output relative L2"
        )
    if (
        rank_128.pair_parameter_fraction_of_flat
        <= rank_64.pair_parameter_fraction_of_flat
    ):
        raise ValueError(
            "rank diagnostic contract requires increasing pair parameter cost"
        )

    content_disjoint = _boolean(
        diagnostic_value.get("content_disjoint"),
        "summary.l3_l4_diagnostic.content_disjoint",
    )
    family_disjoint = _boolean(
        diagnostic_value.get("family_disjoint"),
        "summary.l3_l4_diagnostic.family_disjoint",
    )
    reference_provider_compiled = _boolean(
        diagnostic_value.get("reference_provider_compiled"),
        "summary.l3_l4_diagnostic.reference_provider_compiled",
    )
    if logical_lags != (0, 1, 2, 3, 4):
        raise ValueError(
            "summary.l3_l4_diagnostic.logical_lags must be contiguous 0 through 4"
        )
    if not content_disjoint or family_disjoint or reference_provider_compiled:
        raise ValueError(
            "rank diagnostic contract requires content-disjoint, "
            "not family-disjoint probes and an uncompiled reference provider"
        )
    pair_control_scope = _string(
        diagnostic_value.get("pair_control_scope"),
        "summary.l3_l4_diagnostic.pair_control_scope",
    )
    if pair_control_scope != "oracle_reference_local_factor_control":
        raise ValueError(
            "summary.l3_l4_diagnostic.pair_control_scope must identify the "
            "oracle-reference local factor control"
        )
    reported_probe_statistic = _string(
        diagnostic_value.get("reported_probe_statistic"),
        "summary.l3_l4_diagnostic.reported_probe_statistic",
    )
    if reported_probe_statistic != "mean_across_four_prompt_local_controls":
        raise ValueError(
            "summary.l3_l4_diagnostic.reported_probe_statistic must be "
            "mean_across_four_prompt_local_controls"
        )

    diagnostic = L3L4Diagnostic(
        fit_sequences=_integer(
            diagnostic_value.get("fit_sequences"),
            "summary.l3_l4_diagnostic.fit_sequences",
            minimum=1,
        ),
        probe_sequences=_integer(
            diagnostic_value.get("probe_sequences"),
            "summary.l3_l4_diagnostic.probe_sequences",
            minimum=1,
        ),
        jvp_probes_per_sequence=_integer(
            diagnostic_value.get("jvp_probes_per_sequence"),
            "summary.l3_l4_diagnostic.jvp_probes_per_sequence",
            minimum=1,
        ),
        logical_lags=logical_lags,
        content_disjoint=content_disjoint,
        family_disjoint=family_disjoint,
        reference_provider_compiled=reference_provider_compiled,
        pair_control_scope=pair_control_scope,
        reported_probe_statistic=reported_probe_statistic,
        accounting_status=_string(
            diagnostic_value.get("accounting_status"),
            "summary.l3_l4_diagnostic.accounting_status",
        ),
        accounting_correction=_string(
            diagnostic_value.get("accounting_correction"),
            "summary.l3_l4_diagnostic.accounting_correction",
        ),
        rank_results=tuple(rank_results),
        interpretation=_string(
            diagnostic_value.get("interpretation"),
            "summary.l3_l4_diagnostic.interpretation",
        ),
        not_proven=_string(
            diagnostic_value.get("not_proven"),
            "summary.l3_l4_diagnostic.not_proven",
        ),
    )

    bilinear_value = _object(
        summary.get("bilinear_diagnostic"),
        "summary.bilinear_diagnostic",
    )
    bilinear_prefix = "summary.bilinear_diagnostic"

    def bilinear_integer(field: str, *, minimum: int = 0) -> int:
        return _integer(
            bilinear_value.get(field),
            f"{bilinear_prefix}.{field}",
            minimum=minimum,
        )

    def bilinear_fraction(field: str) -> float:
        return _number(
            bilinear_value.get(field),
            f"{bilinear_prefix}.{field}",
            maximum=1.0,
        )

    def bilinear_cosine(field: str) -> float:
        return _number(
            bilinear_value.get(field),
            f"{bilinear_prefix}.{field}",
            minimum=-1.0,
            maximum=1.0,
        )

    bilinear = BilinearDiagnostic(
        protocol_scope=_string(
            bilinear_value.get("protocol_scope"),
            f"{bilinear_prefix}.protocol_scope",
        ),
        assessment_origin=bilinear_integer(
            "assessment_origin",
            minimum=1,
        ),
        assessment_refit_performed=_boolean(
            bilinear_value.get("assessment_refit_performed"),
            f"{bilinear_prefix}.assessment_refit_performed",
        ),
        selected_plan_kind=_string(
            bilinear_value.get("selected_plan_kind"),
            f"{bilinear_prefix}.selected_plan_kind",
        ),
        selected_source_rank=bilinear_integer(
            "selected_source_rank",
            minimum=1,
        ),
        selected_target_rank=bilinear_integer(
            "selected_target_rank",
            minimum=1,
        ),
        selected_plan_stored_coefficient_count=bilinear_integer(
            "selected_plan_stored_coefficient_count",
            minimum=1,
        ),
        direct_dense_branch_coefficient_count=bilinear_integer(
            "direct_dense_branch_coefficient_count",
            minimum=1,
        ),
        selected_fraction_of_direct_dense=bilinear_fraction(
            "selected_fraction_of_direct_dense"
        ),
        branch_coefficient_reduction_fraction=bilinear_fraction(
            "branch_coefficient_reduction_fraction"
        ),
        base_plus_branch_stored_coefficient_count=bilinear_integer(
            "base_plus_branch_stored_coefficient_count",
            minimum=1,
        ),
        matched_dense_three_branch_coefficient_count=bilinear_integer(
            "matched_dense_three_branch_coefficient_count",
            minimum=1,
        ),
        combined_fraction_of_matched_dense=bilinear_fraction(
            "combined_fraction_of_matched_dense"
        ),
        combined_coefficient_reduction_fraction=bilinear_fraction(
            "combined_coefficient_reduction_fraction"
        ),
        selection_base_relative_error=_number(
            bilinear_value.get("selection_base_relative_error"),
            f"{bilinear_prefix}.selection_base_relative_error",
        ),
        selection_augmented_relative_error=_number(
            bilinear_value.get("selection_augmented_relative_error"),
            f"{bilinear_prefix}.selection_augmented_relative_error",
        ),
        selection_error_reduction_fraction=bilinear_fraction(
            "selection_error_reduction_fraction"
        ),
        selection_augmented_cosine=bilinear_cosine(
            "selection_augmented_cosine"
        ),
        assessment_base_relative_error=_number(
            bilinear_value.get("assessment_base_relative_error"),
            f"{bilinear_prefix}.assessment_base_relative_error",
        ),
        assessment_augmented_relative_error=_number(
            bilinear_value.get("assessment_augmented_relative_error"),
            f"{bilinear_prefix}.assessment_augmented_relative_error",
        ),
        assessment_error_reduction_fraction=bilinear_fraction(
            "assessment_error_reduction_fraction"
        ),
        assessment_augmented_cosine=bilinear_cosine(
            "assessment_augmented_cosine"
        ),
        assessment_c11_relative_error=_number(
            bilinear_value.get("assessment_c11_relative_error"),
            f"{bilinear_prefix}.assessment_c11_relative_error",
        ),
        assessment_c11_cosine=bilinear_cosine(
            "assessment_c11_cosine"
        ),
        decision=_string(
            bilinear_value.get("decision"),
            f"{bilinear_prefix}.decision",
        ),
        prompt_conditioned_reference_provider_compiled=_boolean(
            bilinear_value.get(
                "prompt_conditioned_reference_provider_compiled"
            ),
            (
                f"{bilinear_prefix}."
                "prompt_conditioned_reference_provider_compiled"
            ),
        ),
        nll_measured=_boolean(
            bilinear_value.get("nll_measured"),
            f"{bilinear_prefix}.nll_measured",
        ),
        model_parameter_compression_claim=_boolean(
            bilinear_value.get("model_parameter_compression_claim"),
            f"{bilinear_prefix}.model_parameter_compression_claim",
        ),
        latency_measured=_boolean(
            bilinear_value.get("latency_measured"),
            f"{bilinear_prefix}.latency_measured",
        ),
        positive_pair_identity_generalization_claim=_boolean(
            bilinear_value.get(
                "positive_pair_identity_generalization_claim"
            ),
            (
                f"{bilinear_prefix}."
                "positive_pair_identity_generalization_claim"
            ),
        ),
        interpretation=_string(
            bilinear_value.get("interpretation"),
            f"{bilinear_prefix}.interpretation",
        ),
        not_proven=_string(
            bilinear_value.get("not_proven"),
            f"{bilinear_prefix}.not_proven",
        ),
    )
    if (
        bilinear.protocol_scope
        != "prompt_free_fixed_reference_modal_delta_component"
        or bilinear.assessment_origin != 20
        or bilinear.assessment_refit_performed
        or bilinear.selected_plan_kind != "spectral"
        or (bilinear.selected_source_rank, bilinear.selected_target_rank)
        != (8, 8)
        or bilinear.decision != "passes_frozen_assessment"
    ):
        raise ValueError(
            "bilinear diagnostic must identify the frozen rank-8 by rank-8 "
            "prompt-free assessment at origin 20"
        )
    forbidden_claims = (
        bilinear.prompt_conditioned_reference_provider_compiled,
        bilinear.nll_measured,
        bilinear.model_parameter_compression_claim,
        bilinear.latency_measured,
        bilinear.positive_pair_identity_generalization_claim,
    )
    if any(forbidden_claims):
        raise ValueError(
            "bilinear diagnostic must preserve provider, NLL, model "
            "compression, latency, and pair-generalization claim boundaries"
        )
    if (
        bilinear.selected_plan_stored_coefficient_count
        >= bilinear.direct_dense_branch_coefficient_count
        or bilinear.base_plus_branch_stored_coefficient_count
        >= bilinear.matched_dense_three_branch_coefficient_count
        or bilinear.selection_augmented_relative_error
        >= bilinear.selection_base_relative_error
        or bilinear.assessment_augmented_relative_error
        >= bilinear.assessment_base_relative_error
    ):
        raise ValueError(
            "bilinear diagnostic requires smaller selected accounting and "
            "lower augmented selection and assessment errors"
        )
    consistency_checks = (
        (
            bilinear.selected_fraction_of_direct_dense,
            bilinear.selected_plan_stored_coefficient_count
            / bilinear.direct_dense_branch_coefficient_count,
            "selected fraction of direct dense",
        ),
        (
            bilinear.branch_coefficient_reduction_fraction,
            1.0 - bilinear.selected_fraction_of_direct_dense,
            "branch coefficient reduction",
        ),
        (
            bilinear.combined_fraction_of_matched_dense,
            bilinear.base_plus_branch_stored_coefficient_count
            / bilinear.matched_dense_three_branch_coefficient_count,
            "combined fraction of matched dense",
        ),
        (
            bilinear.combined_coefficient_reduction_fraction,
            1.0 - bilinear.combined_fraction_of_matched_dense,
            "combined coefficient reduction",
        ),
        (
            bilinear.selection_error_reduction_fraction,
            1.0
            - bilinear.selection_augmented_relative_error
            / bilinear.selection_base_relative_error,
            "selection error reduction",
        ),
        (
            bilinear.assessment_error_reduction_fraction,
            1.0
            - bilinear.assessment_augmented_relative_error
            / bilinear.assessment_base_relative_error,
            "assessment error reduction",
        ),
    )
    for actual, expected, label in consistency_checks:
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"bilinear diagnostic has inconsistent {label}"
            )

    reference_value = _object(
        summary.get("reference_provider_diagnostic"),
        "summary.reference_provider_diagnostic",
    )
    reference_prefix = "summary.reference_provider_diagnostic"
    reference_fields = set(ReferenceProviderDiagnostic.__dataclass_fields__)
    if set(reference_value) != reference_fields:
        raise ValueError(
            "summary.reference_provider_diagnostic fields do not match "
            "the frozen format"
        )

    def reference_number(
        field_name: str,
        *,
        maximum: float | None = None,
    ) -> float:
        return _number(
            reference_value.get(field_name),
            f"{reference_prefix}.{field_name}",
            maximum=maximum,
        )

    def reference_integer(field_name: str) -> int:
        return _integer(
            reference_value.get(field_name),
            f"{reference_prefix}.{field_name}",
        )

    def reference_boolean(field_name: str) -> bool:
        return _boolean(
            reference_value.get(field_name),
            f"{reference_prefix}.{field_name}",
        )

    reference_provider = ReferenceProviderDiagnostic(
        protocol_scope=_string(
            reference_value.get("protocol_scope"),
            f"{reference_prefix}.protocol_scope",
        ),
        fit_probe_count=reference_integer("fit_probe_count"),
        selection_probe_count=reference_integer("selection_probe_count"),
        assessment_probe_count=reference_integer("assessment_probe_count"),
        selected_candidate_id=_string(
            reference_value.get("selected_candidate_id"),
            f"{reference_prefix}.selected_candidate_id",
        ),
        selected_source_rank=reference_integer("selected_source_rank"),
        selected_target_rank=reference_integer("selected_target_rank"),
        selected_stored_scalar_count=reference_integer(
            "selected_stored_scalar_count"
        ),
        dense_provider_stored_scalar_count=reference_integer(
            "dense_provider_stored_scalar_count"
        ),
        selected_fraction_of_dense_provider=reference_number(
            "selected_fraction_of_dense_provider",
            maximum=1.0,
        ),
        provider_scalar_reduction_fraction=reference_number(
            "provider_scalar_reduction_fraction",
            maximum=1.0,
        ),
        selection_fisher_weighted_relative_error=reference_number(
            "selection_fisher_weighted_relative_error"
        ),
        selection_reference_cosine=reference_number(
            "selection_reference_cosine",
            maximum=1.0,
        ),
        selection_maximum_per_probe_p90_relative_error=reference_number(
            "selection_maximum_per_probe_p90_relative_error"
        ),
        assessment_fisher_weighted_relative_error=reference_number(
            "assessment_fisher_weighted_relative_error"
        ),
        assessment_reference_cosine=reference_number(
            "assessment_reference_cosine",
            maximum=1.0,
        ),
        assessment_maximum_per_probe_p90_relative_error=reference_number(
            "assessment_maximum_per_probe_p90_relative_error"
        ),
        assessment_worst_family_relative_error=reference_number(
            "assessment_worst_family_relative_error"
        ),
        assessment_error_reduction_vs_constant=reference_number(
            "assessment_error_reduction_vs_constant",
            maximum=1.0,
        ),
        assessment_error_reduction_vs_position_only=reference_number(
            "assessment_error_reduction_vs_position_only",
            maximum=1.0,
        ),
        assessment_fidelity_and_structure_gates_passed=reference_integer(
            "assessment_fidelity_and_structure_gates_passed"
        ),
        assessment_fidelity_and_structure_gate_count=reference_integer(
            "assessment_fidelity_and_structure_gate_count"
        ),
        assessment_collision_panel_gate_passed=reference_boolean(
            "assessment_collision_panel_gate_passed"
        ),
        assessment_collision_threshold=reference_number(
            "assessment_collision_threshold"
        ),
        assessment_minimum_collision_target_relative_difference=(
            reference_number(
                "assessment_minimum_collision_target_relative_difference"
            )
        ),
        assessment_formal_decision=_string(
            reference_value.get("assessment_formal_decision"),
            f"{reference_prefix}.assessment_formal_decision",
        ),
        assessment_refit_performed=reference_boolean(
            "assessment_refit_performed"
        ),
        assessment_reselection_performed=reference_boolean(
            "assessment_reselection_performed"
        ),
        assessment_claim_consumed=reference_boolean(
            "assessment_claim_consumed"
        ),
        natural_prompt_transfer_tested=reference_boolean(
            "natural_prompt_transfer_tested"
        ),
        nll_measured=reference_boolean("nll_measured"),
        model_parameter_compression_claim=reference_boolean(
            "model_parameter_compression_claim"
        ),
        latency_measured=reference_boolean("latency_measured"),
        interpretation=_string(
            reference_value.get("interpretation"),
            f"{reference_prefix}.interpretation",
        ),
        not_proven=_string(
            reference_value.get("not_proven"),
            f"{reference_prefix}.not_proven",
        ),
    )
    if (
        reference_provider.protocol_scope
        != (
            "prompt_blind_after_frozen_upstream_prompt_conditioned_"
            "fisher_basis"
        )
        or (
            reference_provider.fit_probe_count,
            reference_provider.selection_probe_count,
            reference_provider.assessment_probe_count,
        )
        != (80, 32, 88)
        or reference_provider.selected_candidate_id != "spectral-r08-t08"
        or (
            reference_provider.selected_source_rank,
            reference_provider.selected_target_rank,
        )
        != (8, 8)
        or reference_provider.selected_stored_scalar_count != 910
        or reference_provider.dense_provider_stored_scalar_count != 15_046
        or reference_provider.assessment_formal_decision
        != "fails_collision_panel_identifiability_only"
    ):
        raise ValueError(
            "reference-provider diagnostic must identify the frozen v2 "
            "rank-8 prompt-blind result"
        )
    expected_fraction = (
        reference_provider.selected_stored_scalar_count
        / reference_provider.dense_provider_stored_scalar_count
    )
    if (
        not math.isclose(
            reference_provider.selected_fraction_of_dense_provider,
            expected_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reference_provider.provider_scalar_reduction_fraction,
            1.0 - expected_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "reference-provider diagnostic has inconsistent scalar accounting"
        )
    if (
        reference_provider.assessment_fidelity_and_structure_gate_count != 11
        or reference_provider.assessment_fidelity_and_structure_gates_passed
        != reference_provider.assessment_fidelity_and_structure_gate_count
        or reference_provider.assessment_collision_panel_gate_passed
        or reference_provider.assessment_minimum_collision_target_relative_difference
        >= reference_provider.assessment_collision_threshold
        or reference_provider.assessment_refit_performed
        or reference_provider.assessment_reselection_performed
        or not reference_provider.assessment_claim_consumed
    ):
        raise ValueError(
            "reference-provider diagnostic must preserve the sealed "
            "fidelity-pass and collision-control-fail decision"
        )
    if any(
        (
            reference_provider.natural_prompt_transfer_tested,
            reference_provider.nll_measured,
            reference_provider.model_parameter_compression_claim,
            reference_provider.latency_measured,
        )
    ):
        raise ValueError(
            "reference-provider diagnostic must preserve natural-prompt, "
            "NLL, model-compression, and latency claim boundaries"
        )

    attenuation_value = _object(
        summary.get("reference_provider_collision_attenuation_diagnostic"),
        "summary.reference_provider_collision_attenuation_diagnostic",
    )
    attenuation_prefix = (
        "summary.reference_provider_collision_attenuation_diagnostic"
    )
    attenuation_fields = set(
        ReferenceProviderCollisionAttenuationDiagnostic.__dataclass_fields__
    )
    if set(attenuation_value) != attenuation_fields:
        raise ValueError(
            "summary.reference_provider_collision_attenuation_diagnostic "
            "fields do not match the frozen format"
        )

    def attenuation_number(field_name: str) -> float:
        return _number(
            attenuation_value.get(field_name),
            f"{attenuation_prefix}.{field_name}",
        )

    def attenuation_integer(field_name: str) -> int:
        return _integer(
            attenuation_value.get(field_name),
            f"{attenuation_prefix}.{field_name}",
        )

    def attenuation_boolean(field_name: str) -> bool:
        return _boolean(
            attenuation_value.get(field_name),
            f"{attenuation_prefix}.{field_name}",
        )

    families: list[CollisionAttenuationFamily] = []
    family_fields = set(CollisionAttenuationFamily.__dataclass_fields__)
    for index, family_value in enumerate(
        _array(
            attenuation_value.get("families"),
            f"{attenuation_prefix}.families",
        )
    ):
        family = _object(
            family_value,
            f"{attenuation_prefix}.families[{index}]",
        )
        family_prefix = f"{attenuation_prefix}.families[{index}]"
        if set(family) != family_fields:
            raise ValueError(
                f"{family_prefix} fields do not match the frozen format"
            )
        row = CollisionAttenuationFamily(
            family_id=_string(
                family.get("family_id"),
                f"{family_prefix}.family_id",
            ),
            pair_count=_integer(
                family.get("pair_count"),
                f"{family_prefix}.pair_count",
            ),
            gate_witness_count=_integer(
                family.get("gate_witness_count"),
                f"{family_prefix}.gate_witness_count",
            ),
            gate_witnesses_at_or_above_threshold=_integer(
                family.get("gate_witnesses_at_or_above_threshold"),
                (
                    f"{family_prefix}."
                    "gate_witnesses_at_or_above_threshold"
                ),
            ),
            target_relative_difference_minimum=_number(
                family.get("target_relative_difference_minimum"),
                f"{family_prefix}.target_relative_difference_minimum",
            ),
            target_relative_difference_median=_number(
                family.get("target_relative_difference_median"),
                f"{family_prefix}.target_relative_difference_median",
            ),
            target_relative_difference_maximum=_number(
                family.get("target_relative_difference_maximum"),
                f"{family_prefix}.target_relative_difference_maximum",
            ),
            gate_witness_conclusion=_string(
                family.get("gate_witness_conclusion"),
                f"{family_prefix}.gate_witness_conclusion",
            ),
            reference_baseline_relative_dilution_count=_integer(
                family.get(
                    "reference_baseline_relative_dilution_count"
                ),
                (
                    f"{family_prefix}."
                    "reference_baseline_relative_dilution_count"
                ),
            ),
            pre_ff_norm_attenuation_count=_integer(
                family.get("pre_ff_norm_attenuation_count"),
                f"{family_prefix}.pre_ff_norm_attenuation_count",
            ),
        )
        if not (
            row.target_relative_difference_minimum
            <= row.target_relative_difference_median
            <= row.target_relative_difference_maximum
        ):
            raise ValueError(
                f"{family_prefix} target-relative range must be ordered"
            )
        if (
            row.gate_witnesses_at_or_above_threshold
            > row.gate_witness_count
            or row.gate_witness_count > row.pair_count
        ):
            raise ValueError(
                f"{family_prefix} has inconsistent witness accounting"
            )
        families.append(row)

    def attenuation_strings(field_name: str) -> tuple[str, ...]:
        return tuple(
            _string(
                value,
                f"{attenuation_prefix}.{field_name}[{index}]",
            )
            for index, value in enumerate(
                _array(
                    attenuation_value.get(field_name),
                    f"{attenuation_prefix}.{field_name}",
                )
            )
        )

    collision_attenuation = ReferenceProviderCollisionAttenuationDiagnostic(
        protocol_scope=_string(
            attenuation_value.get("protocol_scope"),
            f"{attenuation_prefix}.protocol_scope",
        ),
        source_report_sha256=_sha256(
            attenuation_value.get("source_report_sha256"),
            f"{attenuation_prefix}.source_report_sha256",
        ),
        logical_artifact_sha256=_sha256(
            attenuation_value.get("logical_artifact_sha256"),
            f"{attenuation_prefix}.logical_artifact_sha256",
        ),
        tensor_file_sha256=_sha256(
            attenuation_value.get("tensor_file_sha256"),
            f"{attenuation_prefix}.tensor_file_sha256",
        ),
        source_assessment_artifact_sha256=_sha256(
            attenuation_value.get("source_assessment_artifact_sha256"),
            f"{attenuation_prefix}.source_assessment_artifact_sha256",
        ),
        source_assessment_report_sha256=_sha256(
            attenuation_value.get("source_assessment_report_sha256"),
            f"{attenuation_prefix}.source_assessment_report_sha256",
        ),
        collision_threshold=attenuation_number("collision_threshold"),
        collision_endpoint_count=attenuation_integer(
            "collision_endpoint_count"
        ),
        all_target_hashes_match_opened_assessment=attenuation_boolean(
            "all_target_hashes_match_opened_assessment"
        ),
        collision_group_count=attenuation_integer("collision_group_count"),
        unordered_pair_count=attenuation_integer("unordered_pair_count"),
        gate_witness_count=attenuation_integer("gate_witness_count"),
        numerically_valid_gate_witness_count=attenuation_integer(
            "numerically_valid_gate_witness_count"
        ),
        gate_witnesses_at_or_above_threshold=attenuation_integer(
            "gate_witnesses_at_or_above_threshold"
        ),
        gate_witnesses_below_threshold=attenuation_integer(
            "gate_witnesses_below_threshold"
        ),
        families=tuple(families),
        reference_baseline_relative_dilution_count=attenuation_integer(
            "reference_baseline_relative_dilution_count"
        ),
        pre_ff_norm_attenuation_count=attenuation_integer(
            "pre_ff_norm_attenuation_count"
        ),
        observations_shared_with_passing_controls=attenuation_strings(
            "observations_shared_with_passing_controls"
        ),
        observations_exclusive_to_failed_witnesses=attenuation_strings(
            "observations_exclusive_to_failed_witnesses"
        ),
        failed_witnesses_with_retained_fisher_subspace_miss=(
            attenuation_integer(
                "failed_witnesses_with_retained_fisher_subspace_miss"
            )
        ),
        minimum_failed_witness_retained_64_fisher_energy_fraction=(
            attenuation_number(
                "minimum_failed_witness_retained_64_fisher_energy_fraction"
            )
        ),
        failed_witnesses_with_residual_attention_cancellation=(
            attenuation_integer(
                "failed_witnesses_with_residual_attention_cancellation"
            )
        ),
        maximum_jvp_vjp_adjoint_relative_error=attenuation_number(
            "maximum_jvp_vjp_adjoint_relative_error"
        ),
        maximum_pre_source_jvp_energy_fraction=attenuation_number(
            "maximum_pre_source_jvp_energy_fraction"
        ),
        diagnostic_conclusion=_string(
            attenuation_value.get("diagnostic_conclusion"),
            f"{attenuation_prefix}.diagnostic_conclusion",
        ),
        candidate_predictions_entered_collision_metric=attenuation_boolean(
            "candidate_predictions_entered_collision_metric"
        ),
        assessment_panel_previously_opened=attenuation_boolean(
            "assessment_panel_previously_opened"
        ),
        new_sealed_panel_opened=attenuation_boolean(
            "new_sealed_panel_opened"
        ),
        assessment_score_recomputed=attenuation_boolean(
            "assessment_score_recomputed"
        ),
        candidate_refit_performed=attenuation_boolean(
            "candidate_refit_performed"
        ),
        candidate_reselection_performed=attenuation_boolean(
            "candidate_reselection_performed"
        ),
        candidate_tracking_failure_can_be_assigned=attenuation_boolean(
            "candidate_tracking_failure_can_be_assigned"
        ),
        formal_v2_decision_changed=attenuation_boolean(
            "formal_v2_decision_changed"
        ),
        target_derived_vjp_may_become_compiler_input=attenuation_boolean(
            "target_derived_vjp_may_become_compiler_input"
        ),
        fresh_v3_assessment_required=attenuation_boolean(
            "fresh_v3_assessment_required"
        ),
        interpretation=_string(
            attenuation_value.get("interpretation"),
            f"{attenuation_prefix}.interpretation",
        ),
        not_proven=_string(
            attenuation_value.get("not_proven"),
            f"{attenuation_prefix}.not_proven",
        ),
    )

    attenuation_source = sources[-2]
    if (
        attenuation_source.source_id
        != "gemma_reference_provider_v2_attenuation_localization"
        or attenuation_source.sha256
        != collision_attenuation.source_report_sha256
        or collision_attenuation.source_assessment_report_sha256
        != sources[-3].sha256
        or collision_attenuation.protocol_scope
        != "retrospective_teacher_path_attenuation_on_consumed_v2_panel"
        or collision_attenuation.source_assessment_artifact_sha256
        != "21500080aed580e91b605a6fdd01984dcc41676c0dea96a7813ee0ec4a8cc57d"
        or collision_attenuation.collision_threshold
        != reference_provider.assessment_collision_threshold
        or collision_attenuation.collision_endpoint_count != 40
        or not collision_attenuation.all_target_hashes_match_opened_assessment
        or collision_attenuation.collision_group_count != 16
        or collision_attenuation.unordered_pair_count != 32
        or collision_attenuation.gate_witness_count != 16
        or collision_attenuation.numerically_valid_gate_witness_count != 16
        or collision_attenuation.gate_witnesses_at_or_above_threshold != 4
        or collision_attenuation.gate_witnesses_below_threshold != 12
        or tuple(family.family_id for family in families)
        != ("axis", "null_collision", "radial_collision")
    ):
        raise ValueError(
            "collision-attenuation diagnostic must identify the authenticated "
            "retrospective v2 teacher-path result"
        )
    if (
        sum(family.pair_count for family in families)
        != collision_attenuation.unordered_pair_count
        or sum(family.gate_witness_count for family in families)
        != collision_attenuation.gate_witness_count
        or sum(
            family.gate_witnesses_at_or_above_threshold
            for family in families
        )
        != collision_attenuation.gate_witnesses_at_or_above_threshold
        or collision_attenuation.gate_witnesses_at_or_above_threshold
        + collision_attenuation.gate_witnesses_below_threshold
        != collision_attenuation.gate_witness_count
        or sum(
            family.reference_baseline_relative_dilution_count
            for family in families
        )
        != collision_attenuation.reference_baseline_relative_dilution_count
        or sum(family.pre_ff_norm_attenuation_count for family in families)
        != collision_attenuation.pre_ff_norm_attenuation_count
    ):
        raise ValueError(
            "collision-attenuation diagnostic has inconsistent aggregate "
            "accounting"
        )
    axis, null_collision, radial_collision = families
    if (
        axis.gate_witness_conclusion != "teacher_contrast_ineligible"
        or null_collision.gate_witness_conclusion
        != "teacher_contrast_ineligible"
        or radial_collision.gate_witness_conclusion
        != "teacher_contrast_eligible"
        or axis.target_relative_difference_maximum
        >= collision_attenuation.collision_threshold
        or null_collision.target_relative_difference_maximum
        >= collision_attenuation.collision_threshold
        or radial_collision.target_relative_difference_minimum
        < collision_attenuation.collision_threshold
        or not math.isclose(
            null_collision.target_relative_difference_minimum,
            reference_provider.assessment_minimum_collision_target_relative_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or collision_attenuation.reference_baseline_relative_dilution_count
        != 10
        or collision_attenuation.pre_ff_norm_attenuation_count != 4
        or collision_attenuation.observations_shared_with_passing_controls
        != ("reference_baseline_relative_dilution",)
        or collision_attenuation.observations_exclusive_to_failed_witnesses
        != ("pre_ff_norm_attenuation",)
        or collision_attenuation.failed_witnesses_with_retained_fisher_subspace_miss
        != 0
        or not math.isclose(
            collision_attenuation.minimum_failed_witness_retained_64_fisher_energy_fraction,
            0.9999880706223239,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or collision_attenuation.failed_witnesses_with_residual_attention_cancellation
        != 0
        or collision_attenuation.maximum_pre_source_jvp_energy_fraction
        != 0.0
        or collision_attenuation.diagnostic_conclusion
        != (
            "collision_failure_precedes_candidate_tracking_and_is_a_"
            "teacher_contrast_eligibility_failure"
        )
    ):
        raise ValueError(
            "collision-attenuation diagnostic must preserve the reported "
            "family and mechanism result"
        )
    if (
        collision_attenuation.candidate_predictions_entered_collision_metric
        or not collision_attenuation.assessment_panel_previously_opened
        or collision_attenuation.new_sealed_panel_opened
        or collision_attenuation.assessment_score_recomputed
        or collision_attenuation.candidate_refit_performed
        or collision_attenuation.candidate_reselection_performed
        or collision_attenuation.candidate_tracking_failure_can_be_assigned
        or collision_attenuation.formal_v2_decision_changed
        or collision_attenuation.target_derived_vjp_may_become_compiler_input
        or not collision_attenuation.fresh_v3_assessment_required
    ):
        raise ValueError(
            "collision-attenuation diagnostic must preserve frozen v2 and "
            "require a fresh v3 assessment"
        )

    v3_value = _object(
        summary.get("reference_provider_v3_assessment"),
        "summary.reference_provider_v3_assessment",
    )
    v3_prefix = "summary.reference_provider_v3_assessment"
    if set(v3_value) != set(ReferenceProviderV3Assessment.__dataclass_fields__):
        raise ValueError(
            "summary.reference_provider_v3_assessment fields do not match "
            "the sealed format"
        )

    def v3_number(field_name: str) -> float:
        return _number(
            v3_value.get(field_name),
            f"{v3_prefix}.{field_name}",
        )

    def v3_integer(field_name: str) -> int:
        return _integer(
            v3_value.get(field_name),
            f"{v3_prefix}.{field_name}",
        )

    def v3_boolean(field_name: str) -> bool:
        return _boolean(
            v3_value.get(field_name),
            f"{v3_prefix}.{field_name}",
        )

    family_common_fields = {
        "family_id",
        "intent",
        "planned_contrast_count",
        "required_eligible_count",
        "teacher_qualified_contrast_count",
        "candidate_scored_count",
        "candidate_pass_count",
        "decision_status",
        "qualified_rank_strata",
        "retained_and_discarded_covered",
    }
    sensitivity_fields = family_common_fields | {
        "macro_rms_contrast_relative_error",
        "worst_contrast_relative_error",
        "minimum_direction_cosine",
        "minimum_projection_gain",
        "maximum_projection_gain",
        "maximum_orthogonal_leakage",
    }
    invariance_fields = family_common_fields | {
        "maximum_candidate_null_relative_effect_upper",
        "maximum_candidate_null_relative_error_upper",
    }
    v3_families: list[ReferenceProviderV3ContrastFamily] = []
    for index, family_value in enumerate(
        _array(
            v3_value.get("contrast_families"),
            f"{v3_prefix}.contrast_families",
        )
    ):
        family = _object(
            family_value,
            f"{v3_prefix}.contrast_families[{index}]",
        )
        family_prefix = f"{v3_prefix}.contrast_families[{index}]"
        intent = _string(family.get("intent"), f"{family_prefix}.intent")
        expected_fields = (
            sensitivity_fields if intent == "sensitivity" else invariance_fields
        )
        if intent not in {"sensitivity", "invariance"}:
            raise ValueError(f"{family_prefix}.intent is unsupported")
        if set(family) != expected_fields:
            raise ValueError(
                f"{family_prefix} fields do not match the sealed {intent} format"
            )
        qualified_rank_strata = tuple(
            _string(value, f"{family_prefix}.qualified_rank_strata[{item_index}]")
            for item_index, value in enumerate(
                _array(
                    family.get("qualified_rank_strata"),
                    f"{family_prefix}.qualified_rank_strata",
                )
            )
        )
        if (
            len(set(qualified_rank_strata)) != len(qualified_rank_strata)
            or any(
                stratum not in {"retained", "discarded"}
                for stratum in qualified_rank_strata
            )
        ):
            raise ValueError(
                f"{family_prefix}.qualified_rank_strata must be unique "
                "retained/discarded labels"
            )
        planned_count = _integer(
            family.get("planned_contrast_count"),
            f"{family_prefix}.planned_contrast_count",
        )
        teacher_count = _integer(
            family.get("teacher_qualified_contrast_count"),
            f"{family_prefix}.teacher_qualified_contrast_count",
        )
        candidate_scored_count = _integer(
            family.get("candidate_scored_count"),
            f"{family_prefix}.candidate_scored_count",
        )
        candidate_pass_count = _integer(
            family.get("candidate_pass_count"),
            f"{family_prefix}.candidate_pass_count",
        )
        if not (
            candidate_pass_count
            <= candidate_scored_count
            <= teacher_count
            <= planned_count
        ):
            raise ValueError(
                f"{family_prefix} has inconsistent contrast accounting"
            )
        if intent == "sensitivity":
            minimum_direction_cosine = _signed_number(
                family.get("minimum_direction_cosine"),
                f"{family_prefix}.minimum_direction_cosine",
            )
            minimum_projection_gain = _signed_number(
                family.get("minimum_projection_gain"),
                f"{family_prefix}.minimum_projection_gain",
            )
            maximum_projection_gain = _signed_number(
                family.get("maximum_projection_gain"),
                f"{family_prefix}.maximum_projection_gain",
            )
            if not -1.0 <= minimum_direction_cosine <= 1.0:
                raise ValueError(
                    f"{family_prefix}.minimum_direction_cosine must be "
                    "between -1 and 1"
                )
            if minimum_projection_gain > maximum_projection_gain:
                raise ValueError(
                    f"{family_prefix} projection-gain range must be ordered"
                )
            macro_rms_error = _number(
                family.get("macro_rms_contrast_relative_error"),
                f"{family_prefix}.macro_rms_contrast_relative_error",
            )
            worst_error = _number(
                family.get("worst_contrast_relative_error"),
                f"{family_prefix}.worst_contrast_relative_error",
            )
            maximum_orthogonal_leakage = _number(
                family.get("maximum_orthogonal_leakage"),
                f"{family_prefix}.maximum_orthogonal_leakage",
            )
            if macro_rms_error > worst_error:
                raise ValueError(
                    f"{family_prefix} RMS error cannot exceed worst error"
                )
            maximum_null_effect = None
            maximum_null_error = None
        else:
            macro_rms_error = None
            worst_error = None
            minimum_direction_cosine = None
            minimum_projection_gain = None
            maximum_projection_gain = None
            maximum_orthogonal_leakage = None
            maximum_null_effect = _number(
                family.get("maximum_candidate_null_relative_effect_upper"),
                (
                    f"{family_prefix}."
                    "maximum_candidate_null_relative_effect_upper"
                ),
            )
            maximum_null_error = _number(
                family.get("maximum_candidate_null_relative_error_upper"),
                (
                    f"{family_prefix}."
                    "maximum_candidate_null_relative_error_upper"
                ),
            )
        v3_families.append(
            ReferenceProviderV3ContrastFamily(
                family_id=_string(
                    family.get("family_id"),
                    f"{family_prefix}.family_id",
                ),
                intent=intent,
                planned_contrast_count=planned_count,
                required_eligible_count=_integer(
                    family.get("required_eligible_count"),
                    f"{family_prefix}.required_eligible_count",
                ),
                teacher_qualified_contrast_count=teacher_count,
                candidate_scored_count=candidate_scored_count,
                candidate_pass_count=candidate_pass_count,
                decision_status=_string(
                    family.get("decision_status"),
                    f"{family_prefix}.decision_status",
                ),
                qualified_rank_strata=qualified_rank_strata,
                retained_and_discarded_covered=_boolean(
                    family.get("retained_and_discarded_covered"),
                    f"{family_prefix}.retained_and_discarded_covered",
                ),
                macro_rms_contrast_relative_error=macro_rms_error,
                worst_contrast_relative_error=worst_error,
                minimum_direction_cosine=minimum_direction_cosine,
                minimum_projection_gain=minimum_projection_gain,
                maximum_projection_gain=maximum_projection_gain,
                maximum_orthogonal_leakage=maximum_orthogonal_leakage,
                maximum_candidate_null_relative_effect_upper=maximum_null_effect,
                maximum_candidate_null_relative_error_upper=maximum_null_error,
            )
        )

    reference_provider_v3 = ReferenceProviderV3Assessment(
        protocol_scope=_string(
            v3_value.get("protocol_scope"),
            f"{v3_prefix}.protocol_scope",
        ),
        source_report_sha256=_sha256(
            v3_value.get("source_report_sha256"),
            f"{v3_prefix}.source_report_sha256",
        ),
        logical_artifact_sha256=_sha256(
            v3_value.get("logical_artifact_sha256"),
            f"{v3_prefix}.logical_artifact_sha256",
        ),
        tensor_file_sha256=_sha256(
            v3_value.get("tensor_file_sha256"),
            f"{v3_prefix}.tensor_file_sha256",
        ),
        protocol_sha256=_sha256(
            v3_value.get("protocol_sha256"),
            f"{v3_prefix}.protocol_sha256",
        ),
        panel_spec_sha256=_sha256(
            v3_value.get("panel_spec_sha256"),
            f"{v3_prefix}.panel_spec_sha256",
        ),
        measured_panel_sha256=_sha256(
            v3_value.get("measured_panel_sha256"),
            f"{v3_prefix}.measured_panel_sha256",
        ),
        code_bundle_sha256=_sha256(
            v3_value.get("code_bundle_sha256"),
            f"{v3_prefix}.code_bundle_sha256",
        ),
        claim_receipt_sha256=_sha256(
            v3_value.get("claim_receipt_sha256"),
            f"{v3_prefix}.claim_receipt_sha256",
        ),
        claim_uniqueness_sha256=_sha256(
            v3_value.get("claim_uniqueness_sha256"),
            f"{v3_prefix}.claim_uniqueness_sha256",
        ),
        assessment_probe_count=v3_integer("assessment_probe_count"),
        ordinary_fidelity_probe_count=v3_integer(
            "ordinary_fidelity_probe_count"
        ),
        contrast_probe_count=v3_integer("contrast_probe_count"),
        contrast_group_count=v3_integer("contrast_group_count"),
        contrast_pair_count=v3_integer("contrast_pair_count"),
        selected_candidate_id=_string(
            v3_value.get("selected_candidate_id"),
            f"{v3_prefix}.selected_candidate_id",
        ),
        selected_stored_scalar_count=v3_integer(
            "selected_stored_scalar_count"
        ),
        assessment_refit_performed=v3_boolean(
            "assessment_refit_performed"
        ),
        assessment_reselection_performed=v3_boolean(
            "assessment_reselection_performed"
        ),
        assessment_claim_consumed=v3_boolean("assessment_claim_consumed"),
        ordinary_fidelity_passed=v3_boolean("ordinary_fidelity_passed"),
        ordinary_fidelity_and_structure_gates_passed=v3_integer(
            "ordinary_fidelity_and_structure_gates_passed"
        ),
        ordinary_fidelity_and_structure_gate_count=v3_integer(
            "ordinary_fidelity_and_structure_gate_count"
        ),
        ordinary_fidelity_fisher_weighted_relative_error=v3_number(
            "ordinary_fidelity_fisher_weighted_relative_error"
        ),
        ordinary_fidelity_reference_cosine=v3_number(
            "ordinary_fidelity_reference_cosine"
        ),
        ordinary_fidelity_maximum_per_probe_p90_relative_error=v3_number(
            "ordinary_fidelity_maximum_per_probe_p90_relative_error"
        ),
        ordinary_fidelity_worst_family_relative_error=v3_number(
            "ordinary_fidelity_worst_family_relative_error"
        ),
        ordinary_fidelity_error_reduction_vs_constant=v3_number(
            "ordinary_fidelity_error_reduction_vs_constant"
        ),
        ordinary_fidelity_error_reduction_vs_position_only=v3_number(
            "ordinary_fidelity_error_reduction_vs_position_only"
        ),
        ordinary_fidelity_in_support_fraction=v3_number(
            "ordinary_fidelity_in_support_fraction"
        ),
        ordinary_fidelity_prepared_vs_analytic_relative_error=v3_number(
            "ordinary_fidelity_prepared_vs_analytic_relative_error"
        ),
        ordinary_fidelity_causality_violation=v3_number(
            "ordinary_fidelity_causality_violation"
        ),
        ordinary_fidelity_padding_violation=v3_number(
            "ordinary_fidelity_padding_violation"
        ),
        ordinary_fidelity_repeat_relative_error=v3_number(
            "ordinary_fidelity_repeat_relative_error"
        ),
        contrast_overall_status=_string(
            v3_value.get("contrast_overall_status"),
            f"{v3_prefix}.contrast_overall_status",
        ),
        formal_outcome=_string(
            v3_value.get("formal_outcome"),
            f"{v3_prefix}.formal_outcome",
        ),
        provider_passed=v3_boolean("provider_passed"),
        contrast_families=tuple(v3_families),
        all_families_cover_retained_and_discarded_strata=v3_boolean(
            "all_families_cover_retained_and_discarded_strata"
        ),
        weak_teacher_contrasts_entered_candidate_relative_metrics=v3_boolean(
            "weak_teacher_contrasts_entered_candidate_relative_metrics"
        ),
        intended_null_contrasts_entered_direction_metrics=v3_boolean(
            "intended_null_contrasts_entered_direction_metrics"
        ),
        candidate_parameters_changed=v3_boolean("candidate_parameters_changed"),
        natural_prompt_transfer_tested=v3_boolean(
            "natural_prompt_transfer_tested"
        ),
        nll_measured=v3_boolean("nll_measured"),
        whole_model_replacement_tested=v3_boolean(
            "whole_model_replacement_tested"
        ),
        model_parameter_compression_claim=v3_boolean(
            "model_parameter_compression_claim"
        ),
        latency_measured=v3_boolean("latency_measured"),
        interpretation=_string(
            v3_value.get("interpretation"),
            f"{v3_prefix}.interpretation",
        ),
        next_rung=_string(
            v3_value.get("next_rung"),
            f"{v3_prefix}.next_rung",
        ),
        not_proven=_string(
            v3_value.get("not_proven"),
            f"{v3_prefix}.not_proven",
        ),
    )

    v3_source = sources[-1]
    radial, signed, intended_null = reference_provider_v3.contrast_families
    if (
        v3_source.source_id != "gemma_reference_provider_v3_assessment"
        or v3_source.sha256 != reference_provider_v3.source_report_sha256
        or reference_provider_v3.protocol_scope
        != "fresh_sealed_synthetic_contrast_assessment_of_exact_frozen_v2_rank8_provider"
        or reference_provider_v3.logical_artifact_sha256
        != "60e83fa843e4a2878f597f0f924e736d83b4165b2bdbb3bd40aab0ca24905594"
        or reference_provider_v3.tensor_file_sha256
        != "49ef76479663eefa66d67a8ae90f1f03cbaff266602368f147df1518adf472d3"
        or reference_provider_v3.protocol_sha256
        != "65959324d2815621a1d6420bdb4d41a9db74c4214205088da9545088bc19ce03"
        or reference_provider_v3.panel_spec_sha256
        != "919126906cc6f07074d76599843504ea81462485e8f93ee6d35c71732979249e"
        or reference_provider_v3.measured_panel_sha256
        != "4486367eb754ae197a25451ec86329cd6fb01d51c5c3bef32246f4ca0d30879a"
        or reference_provider_v3.code_bundle_sha256
        != "af06c779c18bf9bc860ca4683ed37c93a0954f090411c544b8062ddfa29086a0"
        or reference_provider_v3.claim_receipt_sha256
        != "0dec295146d80db94483d176d3dc0d93473e8a48957a9d41cee080d0744b8487"
        or reference_provider_v3.claim_uniqueness_sha256
        != "c7c361118fea271f8a7a968aa53a47edb366ed80c1089c5a8dbb57c423c83e95"
        or (
            reference_provider_v3.assessment_probe_count,
            reference_provider_v3.ordinary_fidelity_probe_count,
            reference_provider_v3.contrast_probe_count,
            reference_provider_v3.contrast_group_count,
            reference_provider_v3.contrast_pair_count,
        )
        != (48, 16, 32, 12, 24)
        or reference_provider_v3.selected_candidate_id != "spectral-r08-t08"
        or reference_provider_v3.selected_stored_scalar_count != 910
    ):
        raise ValueError(
            "reference-provider v3 assessment must identify the authenticated "
            "sealed panel and exact frozen rank-8 candidate"
        )
    if (
        reference_provider_v3.assessment_refit_performed
        or reference_provider_v3.assessment_reselection_performed
        or not reference_provider_v3.assessment_claim_consumed
        or not reference_provider_v3.ordinary_fidelity_passed
        or (
            reference_provider_v3.ordinary_fidelity_and_structure_gates_passed,
            reference_provider_v3.ordinary_fidelity_and_structure_gate_count,
        )
        != (12, 12)
        or reference_provider_v3.contrast_overall_status != "panel_inconclusive"
        or reference_provider_v3.formal_outcome
        != "panel_inconclusive_sensitivity"
        or reference_provider_v3.provider_passed
        or reference_provider_v3.all_families_cover_retained_and_discarded_strata
    ):
        raise ValueError(
            "reference-provider v3 assessment must preserve the ordinary-"
            "fidelity pass and panel-inconclusive provider result"
        )
    if (
        tuple(family.family_id for family in v3_families)
        != (
            "radial_block_sensitivity",
            "signed_block_sensitivity",
            "null_single_invariance",
        )
        or (
            radial.intent,
            radial.planned_contrast_count,
            radial.required_eligible_count,
            radial.teacher_qualified_contrast_count,
            radial.candidate_scored_count,
            radial.candidate_pass_count,
            radial.decision_status,
            radial.qualified_rank_strata,
            radial.retained_and_discarded_covered,
        )
        != (
            "sensitivity",
            8,
            6,
            8,
            8,
            0,
            "candidate_fail",
            ("discarded", "retained"),
            True,
        )
        or (
            signed.intent,
            signed.planned_contrast_count,
            signed.required_eligible_count,
            signed.teacher_qualified_contrast_count,
            signed.candidate_scored_count,
            signed.candidate_pass_count,
            signed.decision_status,
            signed.qualified_rank_strata,
            signed.retained_and_discarded_covered,
        )
        != (
            "sensitivity",
            4,
            4,
            1,
            1,
            0,
            "panel_inconclusive",
            ("retained",),
            False,
        )
        or (
            intended_null.intent,
            intended_null.planned_contrast_count,
            intended_null.required_eligible_count,
            intended_null.teacher_qualified_contrast_count,
            intended_null.candidate_scored_count,
            intended_null.candidate_pass_count,
            intended_null.decision_status,
            intended_null.qualified_rank_strata,
            intended_null.retained_and_discarded_covered,
        )
        != (
            "invariance",
            12,
            12,
            12,
            12,
            7,
            "candidate_fail",
            ("discarded", "retained"),
            True,
        )
    ):
        raise ValueError(
            "reference-provider v3 assessment must preserve sealed contrast-"
            "family accounting and decisions"
        )
    if (
        not math.isclose(
            reference_provider_v3.ordinary_fidelity_fisher_weighted_relative_error,
            0.06772962100197875,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reference_provider_v3.ordinary_fidelity_reference_cosine,
            0.9977221137523479,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            radial.worst_contrast_relative_error or 0.0,
            1.302609636349226,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            signed.minimum_direction_cosine or 0.0,
            -0.9215394885190904,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            signed.minimum_projection_gain or 0.0,
            -2.4427641222529104,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            intended_null.maximum_candidate_null_relative_effect_upper or 0.0,
            0.025328387634504894,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "reference-provider v3 assessment must preserve the sealed "
            "ordinary, radial, signed, and null measurements"
        )
    if (
        reference_provider_v3.weak_teacher_contrasts_entered_candidate_relative_metrics
        or reference_provider_v3.intended_null_contrasts_entered_direction_metrics
        or reference_provider_v3.candidate_parameters_changed
        or reference_provider_v3.natural_prompt_transfer_tested
        or reference_provider_v3.nll_measured
        or reference_provider_v3.whole_model_replacement_tested
        or reference_provider_v3.model_parameter_compression_claim
        or reference_provider_v3.latency_measured
    ):
        raise ValueError(
            "reference-provider v3 assessment must preserve metric-isolation "
            "and downstream claim boundaries"
        )
    return ResearchFigureData(
        sources=tuple(sources),
        stages=tuple(stages),
        diagnostic=diagnostic,
        bilinear=bilinear,
        reference_provider=reference_provider,
        collision_attenuation=collision_attenuation,
        reference_provider_v3=reference_provider_v3,
        claim_scope=_string(summary.get("claim_scope"), "summary.claim_scope"),
    )


def verify_available_source_digests(
    sources: Sequence[ResearchSource],
    *,
    source_root: Path,
) -> tuple[str, ...]:
    """Verify committed sources and every ignored upstream report if present."""

    verified: list[str] = []
    for source in sources:
        relative_path = Path(source.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"research source path must stay below source_root: {source.path}"
            )
        source_path = source_root / relative_path
        if not source_path.is_file():
            if source.source_status == "committed_authenticated_report":
                raise FileNotFoundError(
                    f"required committed research source is missing: {source.path}"
                )
            continue
        source_bytes = source_path.read_bytes()
        if source.sha256_kind == "file_sha256":
            actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        elif source.sha256_kind == "report_payload_sha256":
            report_value = json.loads(source_bytes)
            report = _object(report_value, f"source[{source.source_id}]")
            actual_sha256 = _sha256(
                report.get("report_sha256"),
                f"source[{source.source_id}].report_sha256",
            )
        else:  # Defensive: extraction rejects this state.
            raise ValueError(
                f"unsupported research source SHA kind: {source.sha256_kind}"
            )
        if actual_sha256 != source.sha256:
            raise ValueError(
                f"research source digest mismatch for {source.path}: "
                f"expected {source.sha256}, got {actual_sha256}"
            )
        verified.append(source.source_id)
    return tuple(verified)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    anchor: str | None = None,
) -> str:
    anchor_attribute = "" if anchor is None else f' text-anchor="{anchor}"'
    return (
        f'<text class="{css_class}" x="{x:.1f}" y="{y:.1f}"'
        f"{anchor_attribute}>{escape(value)}</text>"
    )


def _wrapped_text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str,
    width: int,
    line_height: float,
    max_lines: int,
) -> list[str]:
    lines = textwrap.wrap(
        value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    svg = [
        f'<text class="{css_class}" x="{x:.1f}" y="{y:.1f}">',
    ]
    for index, line in enumerate(lines):
        dy = 0.0 if index == 0 else line_height
        svg.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
        )
    svg.append("</text>")
    return svg


def _metadata(
    *,
    source_label: str,
    source_sha256: str,
    sources: Sequence[ResearchSource],
) -> str:
    upstream = ",".join(
        f"{source.source_id}:{source.path}:{source.sha256_kind}:{source.sha256}"
        for source in sources
    )
    return (
        f"source={source_label};sha256={source_sha256};"
        f"upstream={upstream}"
    )


def _svg_start(
    *,
    width: int,
    height: int,
    title: str,
    description: str,
    source_label: str,
    source_sha256: str,
    sources: Sequence[ResearchSource],
) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="figure-title figure-description">'
        ),
        f'<title id="figure-title">{escape(title)}</title>',
        f'<desc id="figure-description">{escape(description)}</desc>',
        (
            "<metadata>"
            + escape(
                _metadata(
                    source_label=source_label,
                    source_sha256=source_sha256,
                    sources=sources,
                )
            )
            + "</metadata>"
        ),
        """<style>
            .background { fill: #f1f5f9; }
            .panel { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.5; }
            .track { fill: #e2e8f0; }
            .callout { fill: #fff7ed; stroke: #fdba74; stroke-width: 1.5; }
            .divider { stroke: #cbd5e1; stroke-width: 1; }
            .connector { stroke: #94a3b8; stroke-width: 3;
                         stroke-linecap: round; }
            text { font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif; fill: #0f172a; }
            .figure-title { font-size: 34px; font-weight: 750; }
            .figure-subtitle { font-size: 18px; fill: #475569; }
            .panel-title { font-size: 22px; font-weight: 700; }
            .stage-number { font-size: 13px; font-weight: 800; fill: #64748b;
                            letter-spacing: 1.2px; }
            .stage-title { font-size: 22px; font-weight: 750; }
            .status { font-size: 12px; font-weight: 800; letter-spacing: .6px; }
            .section-label { font-size: 12px; font-weight: 800; fill: #64748b;
                             letter-spacing: .8px; }
            .body { font-size: 16px; font-weight: 550; }
            .claim { font-size: 15px; fill: #475569; }
            .metric-label { font-size: 16px; font-weight: 700; }
            .metric-scale { font-size: 12px; fill: #64748b; }
            .verdict-good { fill: #047857; }
            .verdict-bad { fill: #b91c1c; }
            .rank-label { font-size: 13px; font-weight: 750; }
            .metric-value { font-size: 14px; font-weight: 750; }
            .footer { font-size: 14px; fill: #475569; }
            .footer-strong { font-size: 15px; font-weight: 700; }
            @media (prefers-color-scheme: dark) {
                .background { fill: #0b1220; }
                .panel { fill: #111827; stroke: #334155; }
                .track { fill: #334155; }
                .callout { fill: #3b2616; stroke: #c2410c; }
                .divider { stroke: #334155; }
                .connector { stroke: #64748b; }
                text { fill: #f8fafc; }
                .figure-subtitle, .claim, .metric-scale, .footer {
                    fill: #cbd5e1;
                }
                .verdict-good { fill: #6ee7b7; }
                .verdict-bad { fill: #fca5a5; }
                .stage-number, .section-label { fill: #94a3b8; }
            }
        </style>""",
        (
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="5" orient="auto" markerUnits="strokeWidth">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>'
            "</marker></defs>"
        ),
        (
            f'<rect class="background" width="{width}" height="{height}" '
            'rx="24"/>'
        ),
    ]


_STATUS_LABELS = {
    "verified_reference": "VERIFIED REFERENCE",
    "fidelity_parent": "FIDELITY PARENT",
    "open_development": "OPEN DEVELOPMENT",
    "analysis_only": "ANALYSIS ONLY",
    "next_experiment": "NEXT EXPERIMENT",
    "frozen_assessment": "FROZEN ASSESSMENT",
    "sealed_mixed_result": "SEALED MIXED RESULT",
}
_STATUS_COLORS = {
    "verified_reference": ("#dcfce7", "#166534", "#22c55e"),
    "fidelity_parent": ("#dbeafe", "#1e40af", "#3b82f6"),
    "open_development": ("#fef3c7", "#92400e", "#f59e0b"),
    "analysis_only": ("#fee2e2", "#991b1b", "#ef4444"),
    "next_experiment": ("#ede9fe", "#5b21b6", "#8b5cf6"),
    "frozen_assessment": ("#ccfbf1", "#115e59", "#14b8a6"),
    "sealed_mixed_result": ("#ffedd5", "#9a3412", "#f97316"),
}


def render_research_ladder(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the empirical research progression and its claim boundaries."""

    width = 2240
    height = 820
    svg = _svg_start(
        width=width,
        height=height,
        title="Fisher graph compilation research ladder",
        description=(
            "Seven stages summarize the verified toy executor, Gemma "
            "single-layer fidelity parent, open-development flat generator "
            "stack, failed stationary L3-to-L4 transport diagnostic, "
            "conditional mixed-mode finding, and the frozen bilinear "
            "spectral component assessment, followed by the prompt-blind "
            "reference-provider fidelity result and failed collision control."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Fisher graph compilation — current research ladder",
                css_class="figure-title",
            ),
            _text(
                50,
                91,
                (
                    "Evidence becomes more model-realistic from left to right; "
                    "claim strength does not automatically increase."
                ),
                css_class="figure-subtitle",
            ),
        ]
    )

    card_width = 275.0
    card_height = 500.0
    card_y = 150.0
    gap = 32.0
    start_x = 39.0
    card_positions = [
        start_x + index * (card_width + gap)
        for index in range(len(data.stages))
    ]
    for index in range(len(card_positions) - 1):
        line_start = card_positions[index] + card_width + 7.0
        line_end = card_positions[index + 1] - 10.0
        svg.append(
            f'<line class="connector" x1="{line_start:.1f}" y1="400.0" '
            f'x2="{line_end:.1f}" y2="400.0" marker-end="url(#arrow)"/>'
        )

    for index, (stage, x) in enumerate(zip(data.stages, card_positions)):
        status_fill, status_text, accent = _STATUS_COLORS[stage.status]
        svg.extend(
            [
                (
                    f'<rect class="panel" x="{x:.1f}" y="{card_y:.1f}" '
                    f'width="{card_width:.1f}" height="{card_height:.1f}" '
                    'rx="18"/>'
                ),
                (
                    f'<rect x="{x:.1f}" y="{card_y:.1f}" '
                    f'width="{card_width:.1f}" height="8.0" rx="4" '
                    f'fill="{accent}"/>'
                ),
                _text(
                    x + 24.0,
                    card_y + 39.0,
                    f"RUNG {index + 1}",
                    css_class="stage-number",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 77.0,
                stage.title,
                css_class="stage-title",
                width=20,
                line_height=26.0,
                max_lines=2,
            )
        )
        pill_width = min(
            card_width - 48.0,
            max(150.0, 9.0 * len(_STATUS_LABELS[stage.status]) + 24.0),
        )
        svg.extend(
            [
                (
                    f'<rect x="{x + 24.0:.1f}" y="{card_y + 112.0:.1f}" '
                    f'width="{pill_width:.1f}" height="30.0" rx="15" '
                    f'fill="{status_fill}"/>'
                ),
                (
                    f'<text class="status" x="{x + 36.0:.1f}" '
                    f'y="{card_y + 132.5:.1f}" style="fill:{status_text}">'
                    f"{escape(_STATUS_LABELS[stage.status])}</text>"
                ),
                _text(
                    x + 24.0,
                    card_y + 180.0,
                    "RESOURCE",
                    css_class="section-label",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 207.0,
                stage.resource,
                css_class="body",
                width=27,
                line_height=21.0,
                max_lines=3,
            )
        )
        svg.extend(
            [
                (
                    f'<line class="divider" x1="{x + 24.0:.1f}" '
                    f'y1="{card_y + 284.0:.1f}" '
                    f'x2="{x + card_width - 24.0:.1f}" '
                    f'y2="{card_y + 284.0:.1f}"/>'
                ),
                _text(
                    x + 24.0,
                    card_y + 319.0,
                    "FIDELITY",
                    css_class="section-label",
                ),
            ]
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 346.0,
                stage.fidelity,
                css_class="body",
                width=27,
                line_height=21.0,
                max_lines=3,
            )
        )
        svg.extend(
            _wrapped_text(
                x + 24.0,
                card_y + 429.0,
                stage.claim,
                css_class="claim",
                width=29,
                line_height=20.0,
                max_lines=3,
            )
        )

    svg.extend(
        [
            _text(
                50,
                710,
                (
                    "Current finding: the rank-8 prompt-blind provider passed "
                    "all sealed fidelity gates; its collision-panel control failed."
                ),
                css_class="footer-strong",
            ),
            _text(
                50,
                742,
                (
                    "Gemma resource figures are logical accounting. No "
                    "replacement-model compression or latency result is claimed."
                ),
                css_class="footer",
            ),
            _text(
                50,
                775,
                f"Scope: {data.claim_scope}",
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def _metric_row(
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    scale_label: str,
    value_64: float,
    value_128: float,
    scale_max: float,
    formatter: Callable[[float], str],
    verdict: str,
    verdict_class: str,
    scale_min: float = 0.0,
) -> list[str]:
    label_x = x + 28.0
    track_x = x + 105.0
    track_width = width - 210.0
    value_x = x + width - 28.0
    if scale_max <= scale_min:
        raise ValueError("metric scale maximum must exceed its minimum")
    bar_64 = max(
        2.0,
        track_width
        * min(max((value_64 - scale_min) / (scale_max - scale_min), 0.0), 1.0),
    )
    bar_128 = max(
        2.0,
        track_width
        * min(
            max((value_128 - scale_min) / (scale_max - scale_min), 0.0),
            1.0,
        ),
    )
    return [
        _text(label_x, y, label, css_class="metric-label"),
        _text(
            x + width - 28.0,
            y,
            scale_label,
            css_class="metric-scale",
            anchor="end",
        ),
        _text(label_x, y + 35.0, "R64", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 21.0:.1f}" '
            f'width="{track_width:.1f}" height="20.0" rx="10"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 21.0:.1f}" '
            f'width="{bar_64:.1f}" height="20.0" rx="10" fill="#7c3aed"/>'
        ),
        _text(
            value_x,
            y + 36.0,
            formatter(value_64),
            css_class="metric-value",
            anchor="end",
        ),
        _text(label_x, y + 72.0, "R128", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 58.0:.1f}" '
            f'width="{track_width:.1f}" height="20.0" rx="10"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 58.0:.1f}" '
            f'width="{bar_128:.1f}" height="20.0" rx="10" fill="#0891b2"/>'
        ),
        _text(
            value_x,
            y + 73.0,
            formatter(value_128),
            css_class="metric-value",
            anchor="end",
        ),
        (
            f'<text class="metric-scale {verdict_class}" x="{label_x:.1f}" '
            f'y="{y + 103.0:.1f}">'
            f"{escape(verdict)}</text>"
        ),
    ]


def render_l3_l4_rank_diagnostic(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the rank-64/rank-128 L3-to-L4 transport diagnostic."""

    diagnostic = data.diagnostic
    rank_64, rank_128 = diagnostic.rank_results
    width = 1600
    height = 980
    svg = _svg_start(
        width=width,
        height=height,
        title="Gemma L3-to-L4 rank diagnostic",
        description=(
            "Two panels compare ranks 64 and 128. Reconstruction and the "
            "in-sample local JVP residual improve at rank 128, while finite "
            "oracle-backed local pair-control cosine, relative error, and "
            "logical parameter cost all move in the wrong direction. Values "
            "are means across four prompt-local controls."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    protocol = (
        f"{diagnostic.fit_sequences} fit sequences · "
        f"{diagnostic.probe_sequences} content-disjoint probes · "
        f"{diagnostic.jvp_probes_per_sequence} exact JVP directions each · "
        f"logical lags {diagnostic.logical_lags[0]}–"
        f"{diagnostic.logical_lags[-1]}"
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Gemma L3→L4 — rank is not the missing ingredient",
                css_class="figure-title",
            ),
            _text(50, 91, protocol, css_class="figure-subtitle"),
            (
                '<rect class="panel" x="50.0" y="135.0" width="735.0" '
                'height="660.0" rx="18"/>'
            ),
            (
                '<rect class="panel" x="815.0" y="135.0" width="735.0" '
                'height="660.0" rx="18"/>'
            ),
            _text(
                78,
                181,
                "What improves locally",
                css_class="panel-title",
            ),
            _text(
                843,
                181,
                "Oracle-backed local pair control",
                css_class="panel-title",
            ),
            _text(
                78,
                209,
                "Lower is better · same relative-L2 scale",
                css_class="figure-subtitle",
            ),
            _text(
                843,
                209,
                "Four-probe means · provider is not compiled",
                css_class="figure-subtitle",
            ),
        ]
    )
    left_metrics = (
        (
            "L3 reconstruction",
            rank_64.source_reconstruction_relative_l2,
            rank_128.source_reconstruction_relative_l2,
        ),
        (
            "L4 reconstruction",
            rank_64.target_reconstruction_relative_l2,
            rank_128.target_reconstruction_relative_l2,
        ),
        (
            "In-sample JVP residual",
            rank_64.in_sample_jvp_relative_residual,
            rank_128.in_sample_jvp_relative_residual,
        ),
    )
    for row_index, (label, value_64, value_128) in enumerate(left_metrics):
        svg.extend(
            _metric_row(
                x=50.0,
                y=255.0 + row_index * 170.0,
                width=735.0,
                label=label,
                scale_label="0–0.35 · lower is better",
                value_64=value_64,
                value_128=value_128,
                scale_max=0.35,
                formatter=lambda value: f"{value:.3f}",
                verdict="Improves with rank",
                verdict_class="verdict-good",
            )
        )
    right_metrics: tuple[
        tuple[
            str,
            float,
            float,
            float,
            Callable[[float], str],
            str,
        ],
        ...,
    ] = (
        (
            "Local pair cosine",
            rank_64.pair_output_cosine,
            rank_128.pair_output_cosine,
            1.0,
            lambda value: f"{value:.3f}",
            "Higher is better · worsens",
        ),
        (
            "Local pair relative L2",
            rank_64.pair_output_relative_l2,
            rank_128.pair_output_relative_l2,
            2.0,
            lambda value: f"{value:.3f}",
            "Lower is better · worsens",
        ),
        (
            "Flat-pair parameters",
            rank_64.pair_parameter_fraction_of_flat,
            rank_128.pair_parameter_fraction_of_flat,
            0.30,
            lambda value: f"{100.0 * value:.1f}%",
            "Lower is smaller · cost increases",
        ),
    )
    scale_labels = (
        "-1–1 · higher is better",
        "0–2 · lower is better",
        "0–30% · lower is smaller",
    )
    for row_index, (
        label,
        value_64,
        value_128,
        scale_max,
        formatter,
        verdict,
    ) in enumerate(right_metrics):
        svg.extend(
            _metric_row(
                x=815.0,
                y=255.0 + row_index * 170.0,
                width=735.0,
                label=label,
                scale_label=scale_labels[row_index],
                value_64=value_64,
                value_128=value_128,
                scale_max=scale_max,
                scale_min=-1.0 if row_index == 0 else 0.0,
                formatter=formatter,
                verdict=verdict,
                verdict_class="verdict-bad",
            )
        )

    lag_text = (
        "Nonlocal fan-out remains: positive-lag kernel energy is "
        f"({100.0 * rank_64.positive_lag_energy_fraction:.1f}% at rank 64; "
        f"{100.0 * rank_128.positive_lag_energy_fraction:.1f}% at rank 128), "
        "a topology signal—not a fidelity result."
    )
    svg.extend(
        [
            (
                '<rect class="callout" x="50.0" y="824.0" width="1500.0" '
                'height="94.0" rx="16"/>'
            ),
        ]
    )
    svg.extend(
        _wrapped_text(
            76.0,
            855.0,
            diagnostic.interpretation,
            css_class="footer-strong",
            width=150,
            line_height=22.0,
            max_lines=2,
        )
    )
    svg.extend(
        [
            _text(76, 901, lag_text, css_class="footer"),
            _text(
                50,
                936,
                f"Accounting: {diagnostic.accounting_correction}",
                css_class="footer",
            ),
            _text(
                50,
                963,
                (
                    "Development-only: probes are content-disjoint but not "
                    "family-disjoint; reference-base provider is not compiled."
                ),
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def _paired_error_bars(
    *,
    x: float,
    y: float,
    width: float,
    base_value: float,
    augmented_value: float,
) -> list[str]:
    track_x = x + 112.0
    track_width = width - 150.0
    scale_max = 0.25
    base_width = track_width * min(base_value / scale_max, 1.0)
    augmented_width = track_width * min(augmented_value / scale_max, 1.0)
    return [
        _text(x, y, "Relative error", css_class="metric-label"),
        _text(
            x + width,
            y,
            "0–0.25 · lower is better",
            css_class="metric-scale",
            anchor="end",
        ),
        _text(x, y + 40.0, "Base", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 25.0:.1f}" '
            f'width="{track_width:.1f}" height="22.0" rx="11"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 25.0:.1f}" '
            f'width="{base_width:.1f}" height="22.0" rx="11" '
            'fill="#64748b"/>'
        ),
        _text(
            x + width,
            y + 41.0,
            f"{base_value:.4f}",
            css_class="metric-value",
            anchor="end",
        ),
        _text(x, y + 90.0, "+ branch", css_class="rank-label"),
        (
            f'<rect class="track" x="{track_x:.1f}" y="{y + 75.0:.1f}" '
            f'width="{track_width:.1f}" height="22.0" rx="11"/>'
        ),
        (
            f'<rect x="{track_x:.1f}" y="{y + 75.0:.1f}" '
            f'width="{augmented_width:.1f}" height="22.0" rx="11" '
            'fill="#0d9488"/>'
        ),
        _text(
            x + width,
            y + 91.0,
            f"{augmented_value:.4f}",
            css_class="metric-value",
            anchor="end",
        ),
    ]


def render_bilinear_spectral_assessment(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the compact bilinear branch's rate and fidelity result."""

    result = data.bilinear
    width = 1600
    height = 900
    svg = _svg_start(
        width=width,
        height=height,
        title="Gemma L3-to-L4 bilinear spectral assessment",
        description=(
            "Three panels show compact branch coefficient accounting, "
            "selection-split fidelity, and response-unopened origin-20 "
            "assessment fidelity. The frozen rank-8 by rank-8 branch uses "
            "6,880 stored coefficients and passes its assessment gates, "
            "within fixed-reference modal-delta scope only."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    panel_width = 480.0
    panel_y = 135.0
    panel_height = 560.0
    panel_xs = (50.0, 560.0, 1070.0)
    svg.extend(
        [
            _text(
                50,
                58,
                "Bilinear spectral branch — compact correction holds out",
                css_class="figure-title",
            ),
            _text(
                50,
                91,
                (
                    "Explicit cross-mode products · position-conditioned "
                    "spectral kernel · frozen assessment at origin 20"
                ),
                css_class="figure-subtitle",
            ),
        ]
    )
    for panel_x in panel_xs:
        svg.append(
            f'<rect class="panel" x="{panel_x:.1f}" y="{panel_y:.1f}" '
            f'width="{panel_width:.1f}" height="{panel_height:.1f}" rx="18"/>'
        )

    left_x = panel_xs[0] + 28.0
    bar_width = panel_width - 56.0
    selected_bar_width = max(
        4.0,
        bar_width * result.selected_fraction_of_direct_dense,
    )
    combined_bar_width = max(
        4.0,
        bar_width * result.combined_fraction_of_matched_dense,
    )
    svg.extend(
        [
            _text(left_x, 181, "What was frozen", css_class="panel-title"),
            _text(
                left_x,
                211,
                (
                    f"Spectral rank {result.selected_source_rank} × "
                    f"{result.selected_target_rank}"
                ),
                css_class="figure-subtitle",
            ),
            _text(left_x, 263, "Bilinear branch", css_class="metric-label"),
            (
                f'<rect class="track" x="{left_x:.1f}" y="284.0" '
                f'width="{bar_width:.1f}" height="24.0" rx="12"/>'
            ),
            (
                f'<rect x="{left_x:.1f}" y="284.0" '
                f'width="{selected_bar_width:.1f}" height="24.0" rx="12" '
                'fill="#0d9488"/>'
            ),
            _text(
                left_x,
                337,
                (
                    f"{result.selected_plan_stored_coefficient_count:,} / "
                    f"{result.direct_dense_branch_coefficient_count:,} "
                    "coefficients"
                ),
                css_class="body",
            ),
            _text(
                left_x,
                371,
                (
                    f"{100.0 * result.branch_coefficient_reduction_fraction:.4f}% "
                    "fewer than direct dense"
                ),
                css_class="metric-scale verdict-good",
            ),
            _text(
                left_x,
                434,
                "Base + bilinear branch",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{left_x:.1f}" y="455.0" '
                f'width="{bar_width:.1f}" height="24.0" rx="12"/>'
            ),
            (
                f'<rect x="{left_x:.1f}" y="455.0" '
                f'width="{combined_bar_width:.1f}" height="24.0" rx="12" '
                'fill="#0d9488"/>'
            ),
            _text(
                left_x,
                508,
                (
                    f"{result.base_plus_branch_stored_coefficient_count:,} / "
                    f"{result.matched_dense_three_branch_coefficient_count:,} "
                    "coefficients"
                ),
                css_class="body",
            ),
            _text(
                left_x,
                542,
                (
                    f"{100.0 * result.combined_coefficient_reduction_fraction:.4f}% "
                    "fewer than matched dense"
                ),
                css_class="metric-scale verdict-good",
            ),
            _text(
                left_x,
                630,
                "Coefficient accounting only",
                css_class="section-label",
            ),
            _text(
                left_x,
                658,
                "No parameter or latency claim",
                css_class="claim",
            ),
        ]
    )

    selection_x = panel_xs[1] + 28.0
    svg.extend(
        [
            _text(
                selection_x,
                181,
                "Frozen selection",
                css_class="panel-title",
            ),
            _text(
                selection_x,
                211,
                "Origins 16 and 32 · pooled",
                css_class="figure-subtitle",
            ),
        ]
    )
    svg.extend(
        _paired_error_bars(
            x=selection_x,
            y=263.0,
            width=panel_width - 56.0,
            base_value=result.selection_base_relative_error,
            augmented_value=result.selection_augmented_relative_error,
        )
    )
    svg.extend(
        [
            _text(
                selection_x,
                409,
                (
                    f"{100.0 * result.selection_error_reduction_fraction:.3f}% "
                    "error reduction"
                ),
                css_class="metric-scale verdict-good",
            ),
            _text(
                selection_x,
                467,
                "Augmented cosine",
                css_class="metric-label",
            ),
            _text(
                selection_x + panel_width - 56.0,
                467,
                f"{result.selection_augmented_cosine:.6f}",
                css_class="metric-value",
                anchor="end",
            ),
            _text(
                selection_x,
                521,
                "Selected as minimal passing row",
                css_class="section-label",
            ),
            _text(
                selection_x,
                558,
                "Rate-first rule stopped at the",
                css_class="claim",
            ),
            _text(
                selection_x,
                591,
                "minimal passing rank 8 × 8 row",
                css_class="claim",
            ),
        ]
    )

    assessment_x = panel_xs[2] + 28.0
    svg.extend(
        [
            _text(
                assessment_x,
                181,
                "Fresh assessment",
                css_class="panel-title",
            ),
            _text(
                assessment_x,
                211,
                "Response-unopened origin 20",
                css_class="figure-subtitle",
            ),
        ]
    )
    svg.extend(
        _paired_error_bars(
            x=assessment_x,
            y=263.0,
            width=panel_width - 56.0,
            base_value=result.assessment_base_relative_error,
            augmented_value=result.assessment_augmented_relative_error,
        )
    )
    svg.extend(
        [
            _text(
                assessment_x,
                409,
                (
                    f"{100.0 * result.assessment_error_reduction_fraction:.3f}% "
                    "error reduction"
                ),
                css_class="metric-scale verdict-good",
            ),
            _text(
                assessment_x,
                467,
                "Augmented cosine",
                css_class="metric-label",
            ),
            _text(
                assessment_x + panel_width - 56.0,
                467,
                f"{result.assessment_augmented_cosine:.6f}",
                css_class="metric-value",
                anchor="end",
            ),
            _text(
                assessment_x,
                521,
                "Cross term C11",
                css_class="section-label",
            ),
            _text(
                assessment_x,
                558,
                (
                    f"relative error {result.assessment_c11_relative_error:.6f}"
                ),
                css_class="claim",
            ),
            _text(
                assessment_x,
                591,
                f"cosine {result.assessment_c11_cosine:.6f}",
                css_class="claim",
            ),
            _text(
                assessment_x,
                647,
                "PASSES FROZEN ASSESSMENT",
                css_class="metric-scale verdict-good",
            ),
        ]
    )

    svg.append(
        '<rect class="callout" x="50.0" y="728.0" width="1500.0" '
        'height="104.0" rx="16"/>'
    )
    svg.extend(
        _wrapped_text(
            76.0,
            762.0,
            result.interpretation,
            css_class="footer-strong",
            width=145,
            line_height=22.0,
            max_lines=2,
        )
    )
    svg.extend(
        [
            _text(
                76,
                813,
                (
                    "Boundary: fixed-reference modal-delta component only; "
                    "provider, NLL, model compression, and latency remain open."
                ),
                css_class="footer",
            ),
            _text(
                50,
                869,
                f"Scope: {data.claim_scope}",
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def render_reference_provider_collision_attenuation(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the retrospective teacher-path collision localization."""

    result = data.collision_attenuation
    width = 1600
    height = 930
    svg = _svg_start(
        width=width,
        height=height,
        title="Retrospective reference-provider collision attenuation",
        description=(
            "A post-assessment diagnostic preserves the frozen v2 failure while "
            "showing canonical collision-witness contrast ranges and two "
            "teacher-path observations. Twelve of sixteen witnesses remain "
            "below the one-percent teacher-separation threshold. A fresh v3 "
            "assessment remains required."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    panel_y = 135.0
    panel_height = 555.0
    panel_specs = (
        (50.0, 335.0),
        (415.0, 700.0),
        (1145.0, 405.0),
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Collision attenuation — what the opened V2 panel reveals",
                css_class="figure-title",
            ),
            _text(
                50,
                91,
                (
                    "Retrospective teacher-path localization · no candidate "
                    "refit, reselection, or score recomputation"
                ),
                css_class="figure-subtitle",
            ),
        ]
    )
    for x, panel_width in panel_specs:
        svg.append(
            f'<rect class="panel" x="{x:.1f}" y="{panel_y:.1f}" '
            f'width="{panel_width:.1f}" height="{panel_height:.1f}" rx="18"/>'
        )

    left_x = panel_specs[0][0] + 28.0
    left_width = panel_specs[0][1] - 56.0
    passed_width = (
        left_width
        * result.gate_witnesses_at_or_above_threshold
        / result.gate_witness_count
    )
    failed_width = left_width - passed_width
    svg.extend(
        [
            _text(left_x, 181, "Frozen result", css_class="panel-title"),
            _text(
                left_x,
                225,
                f"{result.gate_witness_count} canonical witnesses",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{left_x:.1f}" y="258.0" '
                f'width="{left_width:.1f}" height="28.0" rx="14"/>'
            ),
            (
                f'<rect x="{left_x:.1f}" y="258.0" '
                f'width="{failed_width:.1f}" height="28.0" rx="14" '
                'fill="#dc2626"/>'
            ),
            (
                f'<rect x="{left_x + failed_width:.1f}" y="258.0" '
                f'width="{passed_width:.1f}" height="28.0" rx="14" '
                'fill="#059669"/>'
            ),
            _text(
                left_x,
                323,
                f"{result.gate_witnesses_below_threshold} below 1%",
                css_class="metric-label verdict-bad",
            ),
            _text(
                left_x + left_width,
                323,
                f"{result.gate_witnesses_at_or_above_threshold} at/above",
                css_class="metric-label verdict-good",
                anchor="end",
            ),
            _text(left_x, 379, "AUTHENTICATED REPLAY", css_class="section-label"),
            _text(
                left_x,
                414,
                (
                    f"{result.collision_endpoint_count}/"
                    f"{result.collision_endpoint_count} target hashes match"
                ),
                css_class="body",
            ),
            _text(left_x, 470, "SAFETY BOUNDARY", css_class="section-label"),
            _text(left_x, 505, "Opened V2 panel only", css_class="body"),
            _text(left_x, 539, "Candidate predictions excluded", css_class="body"),
            _text(left_x, 573, "Target VJP stays diagnostic", css_class="body"),
            _text(
                left_x,
                642,
                "FORMAL V2 FAILURE UNCHANGED",
                css_class="metric-scale verdict-bad",
            ),
        ]
    )

    middle_x = panel_specs[1][0] + 28.0
    chart_x = middle_x + 145.0
    chart_width = panel_specs[1][1] - 205.0
    log_minimum = -6.0
    log_maximum = -1.0

    def contrast_x(value: float) -> float:
        fraction = (
            math.log10(value) - log_minimum
        ) / (log_maximum - log_minimum)
        return chart_x + chart_width * fraction

    svg.extend(
        [
            _text(middle_x, 181, "Teacher contrast by family", css_class="panel-title"),
            _text(
                middle_x,
                213,
                "Minimum — median — maximum across canonical gate witnesses",
                css_class="claim",
            ),
        ]
    )
    for exponent in range(-6, 0):
        x = chart_x + chart_width * (
            (float(exponent) - log_minimum) / (log_maximum - log_minimum)
        )
        svg.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="242.0" x2="{x:.1f}" y2="570.0" '
                    'stroke="#cbd5e1" stroke-width="1"/>'
                ),
                _text(
                    x,
                    597,
                    f"1e{exponent}",
                    css_class="metric-scale",
                    anchor="middle",
                ),
            ]
        )
    threshold_x = contrast_x(result.collision_threshold)
    svg.extend(
        [
            (
                f'<line x1="{threshold_x:.1f}" y1="231.0" '
                f'x2="{threshold_x:.1f}" y2="570.0" '
                'stroke="#f97316" stroke-width="3" stroke-dasharray="8 6"/>'
            ),
            _text(
                threshold_x,
                226,
                "1% gate",
                css_class="metric-label",
                anchor="middle",
            ),
        ]
    )
    family_labels = {
        "axis": "Axis sign",
        "null_collision": "Gain-null",
        "radial_collision": "Radial scale",
    }
    family_colors = {
        "axis": "#dc2626",
        "null_collision": "#ea580c",
        "radial_collision": "#059669",
    }
    for index, family in enumerate(result.families):
        y = 300.0 + index * 125.0
        minimum_x = contrast_x(family.target_relative_difference_minimum)
        median_x = contrast_x(family.target_relative_difference_median)
        maximum_x = contrast_x(family.target_relative_difference_maximum)
        color = family_colors[family.family_id]
        verdict_class = (
            "metric-scale verdict-good"
            if family.gate_witnesses_at_or_above_threshold
            == family.gate_witness_count
            else "metric-scale verdict-bad"
        )
        svg.extend(
            [
                _text(
                    middle_x,
                    y - 9.0,
                    family_labels[family.family_id],
                    css_class="metric-label",
                ),
                _text(
                    middle_x,
                    y + 18.0,
                    (
                        f"{family.gate_witnesses_at_or_above_threshold}/"
                        f"{family.gate_witness_count} clear"
                    ),
                    css_class=verdict_class,
                ),
                (
                    f'<line x1="{minimum_x:.1f}" y1="{y:.1f}" '
                    f'x2="{maximum_x:.1f}" y2="{y:.1f}" '
                    f'stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
                ),
                (
                    f'<circle cx="{median_x:.1f}" cy="{y:.1f}" r="9" '
                    f'fill="{color}" stroke="#ffffff" stroke-width="3"/>'
                ),
                _text(
                    chart_x + chart_width,
                    y + 30.0,
                    (
                        f"{family.target_relative_difference_minimum:.2e} — "
                        f"{family.target_relative_difference_maximum:.2e}"
                    ),
                    css_class="metric-value",
                    anchor="end",
                ),
            ]
        )

    right_x = panel_specs[2][0] + 28.0
    right_width = panel_specs[2][1] - 56.0
    baseline_width = (
        right_width
        * result.reference_baseline_relative_dilution_count
        / result.gate_witness_count
    )
    norm_width = (
        right_width
        * result.pre_ff_norm_attenuation_count
        / result.gate_witness_count
    )
    svg.extend(
        [
            _text(right_x, 181, "Localization observations", css_class="panel-title"),
            _text(
                right_x,
                224,
                "Reference-baseline dilution",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{right_x:.1f}" y="248.0" '
                f'width="{right_width:.1f}" height="22.0" rx="11"/>'
            ),
            (
                f'<rect x="{right_x:.1f}" y="248.0" '
                f'width="{baseline_width:.1f}" height="22.0" rx="11" '
                'fill="#64748b"/>'
            ),
            _text(
                right_x + right_width,
                240,
                f"{result.reference_baseline_relative_dilution_count}/16",
                css_class="metric-value",
                anchor="end",
            ),
            _text(
                right_x,
                303,
                "Shared with passing controls",
                css_class="claim",
            ),
            _text(
                right_x,
                363,
                "Pre-FF norm attenuation",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{right_x:.1f}" y="387.0" '
                f'width="{right_width:.1f}" height="22.0" rx="11"/>'
            ),
            (
                f'<rect x="{right_x:.1f}" y="387.0" '
                f'width="{norm_width:.1f}" height="22.0" rx="11" '
                'fill="#ea580c"/>'
            ),
            _text(
                right_x + right_width,
                379,
                f"{result.pre_ff_norm_attenuation_count}/16",
                css_class="metric-value",
                anchor="end",
            ),
            _text(
                right_x,
                442,
                "Exclusive to failing null witnesses",
                css_class="claim",
            ),
            _text(right_x, 502, "NEGATIVE CONTROLS", css_class="section-label"),
            _text(
                right_x,
                539,
                (
                    f"{result.failed_witnesses_with_retained_fisher_subspace_miss} "
                    "retained-Fisher misses"
                ),
                css_class="body",
            ),
            _text(
                right_x,
                575,
                (
                    f"{result.failed_witnesses_with_residual_attention_cancellation} "
                    "residual cancellations"
                ),
                css_class="body",
            ),
            _text(
                right_x,
                642,
                "LOCALIZATION, NOT A RESCUE",
                css_class="metric-scale verdict-bad",
            ),
        ]
    )

    svg.append(
        '<rect class="callout" x="50.0" y="720.0" width="1500.0" '
        'height="132.0" rx="16"/>'
    )
    svg.extend(
        _wrapped_text(
            76.0,
            758.0,
            result.interpretation,
            css_class="footer-strong",
            width=146,
            line_height=23.0,
            max_lines=3,
        )
    )
    svg.append(
        _text(
            76,
            832,
            (
                "Boundary: V2 remains a composite-control failure; any "
                "compiler change informed by this opened panel requires V3."
            ),
            css_class="footer",
        )
    )
    svg.extend(
        _wrapped_text(
            50.0,
            890.0,
            f"Scope: {data.claim_scope}",
            css_class="footer",
            width=185,
            line_height=18.0,
            max_lines=2,
        )
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def render_reference_provider_v3_assessment(
    data: ResearchFigureData,
    *,
    source_sha256: str,
    source_label: str,
) -> str:
    """Render the sealed V3 endpoint-fidelity and contrast result."""

    result = data.reference_provider_v3
    radial, signed, intended_null = result.contrast_families
    width = 1600
    height = 930
    svg = _svg_start(
        width=width,
        height=height,
        title="Fresh sealed V3 reference-provider assessment",
        description=(
            "The exact frozen rank-8 provider passes ordinary fidelity on "
            "sixteen fresh probes, but passes none of eight eligible radial "
            "contrasts, reverses the one eligible signed contrast, and "
            "hallucinates changes on five of twelve teacher-invariant null "
            "contrasts. The formal panel is inconclusive because the signed "
            "family lacks sufficient eligible coverage; the provider does "
            "not pass."
        ),
        source_label=source_label,
        source_sha256=source_sha256,
        sources=data.sources,
    )
    svg.extend(
        [
            _text(
                50,
                58,
                "Fresh V3 — average fidelity holds, conditional response does not",
                css_class="figure-title",
            ),
            _text(
                50,
                91,
                (
                    "48 sealed probes · exact frozen spectral-r08-t08 provider "
                    "· no refit, reselection, or threshold tuning"
                ),
                css_class="figure-subtitle",
            ),
            (
                '<rect x="1280.0" y="38.0" width="270.0" height="58.0" '
                'rx="29" fill="#ffedd5" stroke="#f97316" stroke-width="2"/>'
            ),
            (
                '<text class="status" x="1415.0" y="73.0" text-anchor="middle" '
                'style="fill:#9a3412">PANEL INCONCLUSIVE</text>'
            ),
        ]
    )

    panel_y = 145.0
    panel_height = 510.0
    panel_width = 355.0
    panel_xs = (50.0, 425.0, 800.0, 1175.0)
    accents = ("#059669", "#dc2626", "#f97316", "#dc2626")
    for panel_x, accent in zip(panel_xs, accents):
        svg.extend(
            [
                (
                    f'<rect class="panel" x="{panel_x:.1f}" y="{panel_y:.1f}" '
                    f'width="{panel_width:.1f}" height="{panel_height:.1f}" '
                    'rx="18"/>'
                ),
                (
                    f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" '
                    f'width="{panel_width:.1f}" height="8.0" rx="4" '
                    f'fill="{accent}"/>'
                ),
            ]
        )

    ordinary_x = panel_xs[0] + 26.0
    ordinary_track_width = panel_width - 52.0
    ordinary_error_width = ordinary_track_width * min(
        result.ordinary_fidelity_fisher_weighted_relative_error / 0.35,
        1.0,
    )
    svg.extend(
        [
            _text(
                ordinary_x,
                184,
                "ORDINARY FIDELITY",
                css_class="section-label",
            ),
            _text(
                ordinary_x,
                223,
                "Typical outputs hold",
                css_class="panel-title",
            ),
            _text(
                ordinary_x,
                266,
                f"{result.ordinary_fidelity_probe_count} fresh probes",
                css_class="body",
            ),
            _text(
                ordinary_x + ordinary_track_width,
                266,
                (
                    f"{result.ordinary_fidelity_and_structure_gates_passed}/"
                    f"{result.ordinary_fidelity_and_structure_gate_count} gates"
                ),
                css_class="metric-value",
                anchor="end",
            ),
            _text(
                ordinary_x,
                321,
                "Fisher-weighted error",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{ordinary_x:.1f}" y="342.0" '
                f'width="{ordinary_track_width:.1f}" height="24.0" rx="12"/>'
            ),
            (
                f'<rect x="{ordinary_x:.1f}" y="342.0" '
                f'width="{ordinary_error_width:.1f}" height="24.0" rx="12" '
                'fill="#059669"/>'
            ),
            _text(
                ordinary_x,
                402,
                (
                    f"{100.0 * result.ordinary_fidelity_fisher_weighted_relative_error:.2f}%"
                ),
                css_class="stage-title verdict-good",
            ),
            _text(
                ordinary_x,
                449,
                "Reference cosine",
                css_class="metric-label",
            ),
            _text(
                ordinary_x,
                493,
                f"{result.ordinary_fidelity_reference_cosine:.4f}",
                css_class="stage-title verdict-good",
            ),
            _text(
                ordinary_x,
                542,
                (
                    "Worst probe p90 "
                    f"{100.0 * result.ordinary_fidelity_maximum_per_probe_p90_relative_error:.2f}%"
                ),
                css_class="claim",
            ),
            _text(
                ordinary_x,
                617,
                "PASS",
                css_class="metric-scale verdict-good",
            ),
        ]
    )

    radial_x = panel_xs[1] + 26.0
    radial_track_width = panel_width - 52.0
    svg.extend(
        [
            _text(
                radial_x,
                184,
                "RADIAL SENSITIVITY",
                css_class="section-label",
            ),
            _text(
                radial_x,
                223,
                "Magnitude changes",
                css_class="panel-title",
            ),
            _text(
                radial_x,
                266,
                (
                    f"{radial.teacher_qualified_contrast_count}/"
                    f"{radial.planned_contrast_count} teacher eligible"
                ),
                css_class="body",
            ),
            _text(
                radial_x,
                321,
                "Candidate contrast passes",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{radial_x:.1f}" y="342.0" '
                f'width="{radial_track_width:.1f}" height="24.0" rx="12"/>'
            ),
            _text(
                radial_x,
                402,
                (
                    f"{radial.candidate_pass_count}/"
                    f"{radial.candidate_scored_count}"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                radial_x,
                449,
                "Contrast relative error",
                css_class="metric-label",
            ),
            _text(
                radial_x,
                493,
                (
                    "72.7–"
                    f"{100.0 * (radial.worst_contrast_relative_error or 0.0):.1f}%"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                radial_x,
                542,
                (
                    "Both retained + discarded strata; "
                    f"min cosine {radial.minimum_direction_cosine:.3f}"
                ),
                css_class="claim",
            ),
            _text(
                radial_x,
                617,
                "CANDIDATE FAIL",
                css_class="metric-scale verdict-bad",
            ),
        ]
    )

    signed_x = panel_xs[2] + 26.0
    signed_track_width = panel_width - 52.0
    signed_eligible_width = (
        signed_track_width
        * signed.teacher_qualified_contrast_count
        / signed.planned_contrast_count
    )
    svg.extend(
        [
            _text(
                signed_x,
                184,
                "SIGNED SENSITIVITY",
                css_class="section-label",
            ),
            _text(
                signed_x,
                223,
                "Direction changes",
                css_class="panel-title",
            ),
            _text(
                signed_x,
                266,
                (
                    f"{signed.teacher_qualified_contrast_count}/"
                    f"{signed.planned_contrast_count} teacher eligible"
                ),
                css_class="body",
            ),
            _text(
                signed_x + signed_track_width,
                266,
                "3 underpowered",
                css_class="metric-scale",
                anchor="end",
            ),
            _text(
                signed_x,
                321,
                "Teacher-qualified coverage",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{signed_x:.1f}" y="342.0" '
                f'width="{signed_track_width:.1f}" height="24.0" rx="12"/>'
            ),
            (
                f'<rect x="{signed_x:.1f}" y="342.0" '
                f'width="{signed_eligible_width:.1f}" height="24.0" rx="12" '
                'fill="#f97316"/>'
            ),
            _text(
                signed_x,
                402,
                (
                    f"{signed.candidate_pass_count}/"
                    f"{signed.candidate_scored_count} eligible pass"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                signed_x,
                449,
                (
                    f"cosine {signed.minimum_direction_cosine:.3f}"
                ),
                css_class="metric-label verdict-bad",
            ),
            _text(
                signed_x,
                493,
                (
                    f"gain {signed.minimum_projection_gain:.3f}"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                signed_x,
                542,
                "Eligible response reversed + amplified",
                css_class="claim",
            ),
            _text(
                signed_x,
                617,
                "FAMILY INCONCLUSIVE",
                css_class="metric-scale verdict-bad",
            ),
        ]
    )

    null_x = panel_xs[3] + 26.0
    null_track_width = panel_width - 52.0
    null_pass_width = (
        null_track_width
        * intended_null.candidate_pass_count
        / intended_null.candidate_scored_count
    )
    hallucinated_count = (
        intended_null.candidate_scored_count
        - intended_null.candidate_pass_count
    )
    svg.extend(
        [
            _text(
                null_x,
                184,
                "INTENDED NULL",
                css_class="section-label",
            ),
            _text(
                null_x,
                223,
                "Should not change",
                css_class="panel-title",
            ),
            _text(
                null_x,
                266,
                (
                    f"Teacher {intended_null.teacher_qualified_contrast_count}/"
                    f"{intended_null.planned_contrast_count} invariant"
                ),
                css_class="body",
            ),
            _text(
                null_x,
                321,
                "Provider null passes",
                css_class="metric-label",
            ),
            (
                f'<rect class="track" x="{null_x:.1f}" y="342.0" '
                f'width="{null_track_width:.1f}" height="24.0" rx="12"/>'
            ),
            (
                f'<rect x="{null_x:.1f}" y="342.0" '
                f'width="{null_pass_width:.1f}" height="24.0" rx="12" '
                'fill="#059669"/>'
            ),
            _text(
                null_x,
                402,
                (
                    f"{intended_null.candidate_pass_count}/"
                    f"{intended_null.candidate_scored_count}"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                null_x,
                449,
                f"{hallucinated_count} hallucinated changes",
                css_class="metric-label verdict-bad",
            ),
            _text(
                null_x,
                493,
                (
                    "max effect "
                    f"{100.0 * (intended_null.maximum_candidate_null_relative_effect_upper or 0.0):.2f}%"
                ),
                css_class="stage-title verdict-bad",
            ),
            _text(
                null_x,
                542,
                "Teacher invariant; provider moves",
                css_class="claim",
            ),
            _text(
                null_x,
                617,
                "CANDIDATE FAIL",
                css_class="metric-scale verdict-bad",
            ),
        ]
    )

    svg.append(
        '<rect class="callout" x="50.0" y="690.0" width="1500.0" '
        'height="148.0" rx="16"/>'
    )
    svg.extend(
        [
            _text(
                76,
                731,
                "FORMAL OUTCOME: PANEL INCONCLUSIVE · PROVIDER PASSED: FALSE",
                css_class="footer-strong verdict-bad",
            ),
            _text(
                76,
                770,
                (
                    "Why inconclusive: only 1/4 signed contrasts was strong "
                    "enough, leaving the discarded signed stratum uncovered."
                ),
                css_class="footer",
            ),
            _text(
                76,
                807,
                (
                    "Why actionable: radial tracking failed 0/8 and the provider "
                    "invented five changes where the teacher stayed invariant."
                ),
                css_class="footer",
            ),
            _text(
                50,
                887,
                (
                    "Boundary: endpoint fidelity does not authorize dynamic "
                    "graph composition, natural-prompt shadow, NLL, compression, "
                    "compute, or latency claims."
                ),
                css_class="footer",
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def render_summary_file(
    summary_path: Path,
    ladder_output_path: Path,
    diagnostic_output_path: Path,
    bilinear_output_path: Path,
    attenuation_output_path: Path,
    v3_assessment_output_path: Path,
    *,
    source_root: Path = Path("."),
) -> None:
    """Render all committed figures from one source-safe summary."""

    summary_bytes = summary_path.read_bytes()
    summary_value = json.loads(summary_bytes)
    summary = _object(summary_value, "summary")
    data = extract_research_figure_data(summary)
    verify_available_source_digests(data.sources, source_root=source_root)
    source_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    source_label = summary_path.as_posix()
    ladder_svg = render_research_ladder(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    diagnostic_svg = render_l3_l4_rank_diagnostic(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    bilinear_svg = render_bilinear_spectral_assessment(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    attenuation_svg = render_reference_provider_collision_attenuation(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    v3_assessment_svg = render_reference_provider_v3_assessment(
        data,
        source_sha256=source_sha256,
        source_label=source_label,
    )
    ladder_output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_output_path.parent.mkdir(parents=True, exist_ok=True)
    bilinear_output_path.parent.mkdir(parents=True, exist_ok=True)
    attenuation_output_path.parent.mkdir(parents=True, exist_ok=True)
    v3_assessment_output_path.parent.mkdir(parents=True, exist_ok=True)
    ladder_output_path.write_text(ladder_svg, encoding="utf-8", newline="\n")
    diagnostic_output_path.write_text(
        diagnostic_svg,
        encoding="utf-8",
        newline="\n",
    )
    bilinear_output_path.write_text(
        bilinear_svg,
        encoding="utf-8",
        newline="\n",
    )
    attenuation_output_path.write_text(
        attenuation_svg,
        encoding="utf-8",
        newline="\n",
    )
    v3_assessment_output_path.write_text(
        v3_assessment_svg,
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the current research ladder, Gemma L3/L4 rank "
            "diagnostic, bilinear assessment, and retrospective collision "
            "attenuation and sealed V3 assessment results as deterministic "
            "SVGs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"source-safe summary JSON (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--ladder-output",
        type=Path,
        default=DEFAULT_LADDER_OUTPUT,
        help=f"research-ladder SVG destination (default: {DEFAULT_LADDER_OUTPUT})",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_OUTPUT,
        help=(
            "L3/L4 rank-diagnostic SVG destination "
            f"(default: {DEFAULT_DIAGNOSTIC_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--bilinear-output",
        type=Path,
        default=DEFAULT_BILINEAR_OUTPUT,
        help=(
            "bilinear spectral assessment SVG destination "
            f"(default: {DEFAULT_BILINEAR_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--attenuation-output",
        type=Path,
        default=DEFAULT_ATTENUATION_OUTPUT,
        help=(
            "reference-provider collision-attenuation SVG destination "
            f"(default: {DEFAULT_ATTENUATION_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--v3-assessment-output",
        type=Path,
        default=DEFAULT_V3_ASSESSMENT_OUTPUT,
        help=(
            "reference-provider V3 assessment SVG destination "
            f"(default: {DEFAULT_V3_ASSESSMENT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help=(
            "repository root used to verify committed and available upstream "
            "reports (default: current directory)"
        ),
    )
    arguments = parser.parse_args(argv)
    render_summary_file(
        arguments.input,
        arguments.ladder_output,
        arguments.diagnostic_output,
        arguments.bilinear_output,
        arguments.attenuation_output,
        arguments.v3_assessment_output,
        source_root=arguments.source_root,
    )
    print(f"Wrote {arguments.ladder_output}")
    print(f"Wrote {arguments.diagnostic_output}")
    print(f"Wrote {arguments.bilinear_output}")
    print(f"Wrote {arguments.attenuation_output}")
    print(f"Wrote {arguments.v3_assessment_output}")


if __name__ == "__main__":
    main()
