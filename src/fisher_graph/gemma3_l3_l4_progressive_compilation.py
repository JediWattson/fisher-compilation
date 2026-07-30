"""Gemma L3/L4 bindings for the generic progressive compiler.

The existing Calibration-B executor is frozen to the legacy 64-mode candidate
and remains unchanged.  This module creates a prompt-blind Calibration-A
development protocol whose seed is that exact candidate and whose forbidden
assessment identity is the existing Calibration-B manifest.  A progressive
winner can be frozen as development evidence, but it cannot enter the legacy
one-shot executor until a new candidate-bound shadow protocol/runtime is
constructed and authenticated.
"""

from __future__ import annotations

from collections.abc import Mapping
import re

from .compiler.progressive import (
    DevelopmentCorpus,
    ProgressiveCandidate,
    ProgressiveBehavioralTargets,
    ProgressiveCompilationProtocol,
    ProgressiveFidelityTargets,
    ProgressiveResourceBudget,
    ProgressiveResourceFootprint,
    ProgressiveStagingTransition,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)


GEMMA3_L3_L4_PROGRESSIVE_PROTOCOL_ID = (
    "gemma3.l3-l4.graph-organized-svd.progressive.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_BEHAVIOR_HARD_GATE_AXES = (
    "candidate_behavior.absolute_delta_nll_per_token",
    "candidate_behavior.per_prompt_p10_top1_agreement_to_source",
    "candidate_behavior.per_prompt_p90_absolute_delta_nll_per_token",
    "candidate_behavior.source_to_candidate_kl_per_token",
    "candidate_behavior.top1_agreement_to_source",
)
_X4_FROZEN_DIAGNOSTIC_AXES = (
    "boundary.minimum_family_source_signal",
    "boundary.valid_target_coverage",
    "carrier_oracle_behavior.absolute_delta_nll_per_token",
    "carrier_oracle_behavior.per_prompt_p10_top1_agreement_to_source",
    "carrier_oracle_behavior.per_prompt_p90_absolute_delta_nll_per_token",
    "carrier_oracle_behavior.source_to_candidate_kl_per_token",
    "carrier_oracle_behavior.top1_agreement_to_source",
    "operator_nrmse",
    "projection.full_width_cosine",
    "projection.full_width_relative_error",
    "projection.minimum_family_source_signal",
    "projection.worst_family_cosine",
    "projection.worst_family_relative_error",
    "projection_oracle_behavior.absolute_delta_nll_per_token",
    "projection_oracle_behavior.per_prompt_p10_top1_agreement_to_source",
    "projection_oracle_behavior.per_prompt_p90_absolute_delta_nll_per_token",
    "projection_oracle_behavior.source_to_candidate_kl_per_token",
    "projection_oracle_behavior.top1_agreement_to_source",
)
_X4_PERMITTED_REGRESSION_AXES = (
    "boundary.cosine",
    "boundary.worst_family_cosine",
    *_CANDIDATE_BEHAVIOR_HARD_GATE_AXES,
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _legacy_binding() -> dict[str, object]:
    """Read only prompt-blind identities and gates from the frozen protocol."""

    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy.validate_integrity()
    metadata = legacy.metadata()
    model = _mapping(metadata.get("model"), label="legacy model")
    candidate = _mapping(
        metadata.get("graph_candidate"),
        label="legacy graph candidate",
    )
    corpus = _mapping(metadata.get("corpus"), label="legacy corpus")
    assessment = _mapping(
        corpus.get("calibration_b_manifest"),
        label="legacy Calibration-B manifest",
    )
    behavioral = _mapping(
        metadata.get("behavioral_gates"),
        label="legacy behavioral gates",
    )
    boundary = _mapping(
        metadata.get("boundary_gates"),
        label="legacy boundary gates",
    )
    runtime_binding = _mapping(
        metadata.get("runtime_binding_contract"),
        label="legacy runtime binding",
    )
    prompt_blind_basis = _mapping(
        metadata.get("prompt_blind_basis"),
        label="legacy prompt-blind basis",
    )
    projection = _mapping(
        metadata.get("projection_capacity_gates"),
        label="legacy projection-capacity gates",
    )
    projection_behavior = _mapping(
        projection.get("projection_oracle_behavioral_gates"),
        label="legacy projection-oracle behavioral gates",
    )
    carrier = _mapping(
        metadata.get("carrier_completeness_gates"),
        label="legacy carrier-completeness gates",
    )
    carrier_behavior = _mapping(
        carrier.get(
            "exact_full_width_x4_on_clamped_reference_carrier"
        ),
        label="legacy carrier-oracle behavioral gates",
    )
    runtime_expectations = {
        "candidate_logical_artifact_sha256": str(
            candidate["logical_artifact_sha256"]
        ),
        "source_model_sha256": str(model["source_model_sha256"]),
        "factorized_live_execution_sha256": str(
            candidate["factorized_live_execution_sha256"]
        ),
        "adapter_execution_fingerprint": str(
            candidate["factorized_refit_execution_sha256"]
        ),
    }
    for name, expected in runtime_expectations.items():
        if runtime_binding.get(name) != expected:
            raise ValueError(
                "legacy runtime binding does not match the frozen "
                f"candidate provenance: {name}"
            )
    values: dict[str, object] = {
        "legacy_shadow_protocol_sha256": legacy.artifact_sha256,
        "source_model_sha256": str(model["source_model_sha256"]),
        "seed_candidate_artifact_sha256": str(
            candidate["logical_artifact_sha256"]
        ),
        "seed_candidate_execution_sha256": str(
            candidate["factorized_refit_execution_sha256"]
        ),
        "calibration_b_manifest_sha256": str(
            assessment["artifact_sha256"]
        ),
        "seed_runtime_binding_sha256": str(
            runtime_binding["artifact_sha256"]
        ),
        "absolute_delta_nll_per_token_max": _float(
            behavioral["absolute_delta_nll_per_token_max"],
            label="absolute NLL gate",
        ),
        "source_to_candidate_kl_per_token_max": _float(
            behavioral["source_to_candidate_kl_per_token_max"],
            label="KL gate",
        ),
        "top1_agreement_to_source_min": _float(
            behavioral["top1_agreement_to_source_min"],
            label="top-1 gate",
        ),
        "per_prompt_p90_absolute_delta_nll_per_token_max": _float(
            behavioral[
                "per_prompt_p90_absolute_delta_nll_per_token_max"
            ],
            label="per-prompt NLL gate",
        ),
        "per_prompt_p10_top1_agreement_to_source_min": _float(
            behavioral[
                "per_prompt_p10_top1_agreement_to_source_min"
            ],
            label="per-prompt top-1 gate",
        ),
        "boundary_relative_error_max": _float(
            boundary["pooled_target_modal_relative_error_max"],
            label="boundary relative-error gate",
        ),
        "boundary_cosine_min": _float(
            boundary["pooled_target_modal_cosine_min"],
            label="boundary cosine gate",
        ),
        "valid_target_coverage_min": _float(
            boundary["valid_target_coverage_min"],
            label="valid target coverage gate",
        ),
        "worst_family_boundary_relative_error_max": _float(
            boundary[
                "worst_family_target_modal_relative_error_max"
            ],
            label="worst-family boundary relative-error gate",
        ),
        "worst_family_boundary_cosine_min": _float(
            boundary["worst_family_target_modal_cosine_min"],
            label="worst-family boundary cosine gate",
        ),
        "minimum_family_source_modal_signal_l2_norm": _float(
            boundary["minimum_family_source_modal_signal_l2_norm"],
            label="minimum modal signal gate",
        ),
        "projection_full_width_relative_error_max": _float(
            projection["pooled_full_width_delta_relative_error_max"],
            label="projection relative-error gate",
        ),
        "projection_full_width_cosine_min": _float(
            projection["pooled_full_width_delta_cosine_min"],
            label="projection cosine gate",
        ),
        "worst_family_projection_relative_error_max": _float(
            projection[
                "worst_family_full_width_delta_relative_error_max"
            ],
            label="worst-family projection relative-error gate",
        ),
        "worst_family_projection_cosine_min": _float(
            projection[
                "worst_family_full_width_delta_cosine_min"
            ],
            label="worst-family projection cosine gate",
        ),
        "minimum_family_source_full_width_signal_l2_norm": _float(
            projection[
                "minimum_family_source_full_width_signal_l2_norm"
            ],
            label="minimum full-width signal gate",
        ),
    }
    for prefix, gates in (
        ("projection_oracle", projection_behavior),
        ("carrier_oracle", carrier_behavior),
    ):
        for name in (
            "absolute_delta_nll_per_token_max",
            "source_to_candidate_kl_per_token_max",
            "top1_agreement_to_source_min",
            "per_prompt_p90_absolute_delta_nll_per_token_max",
            "per_prompt_p10_top1_agreement_to_source_min",
        ):
            values[f"{prefix}_{name}"] = _float(
                gates[name],
                label=f"{prefix} {name}",
            )
    for key in (
        "legacy_shadow_protocol_sha256",
        "source_model_sha256",
        "seed_candidate_artifact_sha256",
        "seed_candidate_execution_sha256",
        "calibration_b_manifest_sha256",
        "seed_runtime_binding_sha256",
    ):
        value = values[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"legacy {key} is not a lowercase SHA-256")
    seed_lineage = tuple(
        sorted(
            {
                str(candidate["tensor_file_sha256"]),
                str(candidate["global_svd_base_plan_sha256"]),
                str(candidate["graph_basis_artifact_sha256"]),
                str(candidate["deployment_plan_sha256"]),
                str(candidate["factorized_live_execution_sha256"]),
                str(prompt_blind_basis["tensor_file_sha256"]),
                str(prompt_blind_basis["logical_payload_sha256"]),
                str(values["legacy_shadow_protocol_sha256"]),
            }
        )
    )
    if any(
        len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
        for value in seed_lineage
    ):
        raise ValueError("legacy seed lineage contains a non-SHA identity")
    values["seed_lineage_sha256s"] = seed_lineage
    return values


def gemma3_l3_l4_progressive_fidelity_targets(
    *,
    operator_nrmse_max: float | None = None,
) -> ProgressiveFidelityTargets:
    """Use the frozen behavioral/boundary gates for progressive selection."""

    binding = _legacy_binding()
    candidate_behavior = ProgressiveBehavioralTargets(
        absolute_delta_nll_per_token_max=float(
            binding["absolute_delta_nll_per_token_max"]
        ),
        source_to_candidate_kl_per_token_max=float(
            binding["source_to_candidate_kl_per_token_max"]
        ),
        top1_agreement_to_source_min=float(
            binding["top1_agreement_to_source_min"]
        ),
        per_prompt_p90_absolute_delta_nll_per_token_max=float(
            binding[
                "per_prompt_p90_absolute_delta_nll_per_token_max"
            ]
        ),
        per_prompt_p10_top1_agreement_to_source_min=float(
            binding[
                "per_prompt_p10_top1_agreement_to_source_min"
            ]
        ),
    )
    projection_behavior = ProgressiveBehavioralTargets(
        absolute_delta_nll_per_token_max=float(
            binding[
                "projection_oracle_absolute_delta_nll_per_token_max"
            ]
        ),
        source_to_candidate_kl_per_token_max=float(
            binding[
                "projection_oracle_source_to_candidate_kl_per_token_max"
            ]
        ),
        top1_agreement_to_source_min=float(
            binding[
                "projection_oracle_top1_agreement_to_source_min"
            ]
        ),
        per_prompt_p90_absolute_delta_nll_per_token_max=float(
            binding[
                "projection_oracle_"
                "per_prompt_p90_absolute_delta_nll_per_token_max"
            ]
        ),
        per_prompt_p10_top1_agreement_to_source_min=float(
            binding[
                "projection_oracle_"
                "per_prompt_p10_top1_agreement_to_source_min"
            ]
        ),
    )
    carrier_behavior = ProgressiveBehavioralTargets(
        absolute_delta_nll_per_token_max=float(
            binding[
                "carrier_oracle_absolute_delta_nll_per_token_max"
            ]
        ),
        source_to_candidate_kl_per_token_max=float(
            binding[
                "carrier_oracle_source_to_candidate_kl_per_token_max"
            ]
        ),
        top1_agreement_to_source_min=float(
            binding[
                "carrier_oracle_top1_agreement_to_source_min"
            ]
        ),
        per_prompt_p90_absolute_delta_nll_per_token_max=float(
            binding[
                "carrier_oracle_"
                "per_prompt_p90_absolute_delta_nll_per_token_max"
            ]
        ),
        per_prompt_p10_top1_agreement_to_source_min=float(
            binding[
                "carrier_oracle_"
                "per_prompt_p10_top1_agreement_to_source_min"
            ]
        ),
    )
    return ProgressiveFidelityTargets(
        candidate_behavior=candidate_behavior,
        projection_oracle_behavior=projection_behavior,
        carrier_oracle_behavior=carrier_behavior,
        operator_nrmse_max=(
            float(binding["projection_full_width_relative_error_max"])
            if operator_nrmse_max is None
            else operator_nrmse_max
        ),
        boundary_relative_error_max=float(
            binding["boundary_relative_error_max"]
        ),
        boundary_cosine_min=float(binding["boundary_cosine_min"]),
        valid_target_coverage_min=float(
            binding["valid_target_coverage_min"]
        ),
        worst_family_boundary_relative_error_max=float(
            binding["worst_family_boundary_relative_error_max"]
        ),
        worst_family_boundary_cosine_min=float(
            binding["worst_family_boundary_cosine_min"]
        ),
        minimum_family_source_modal_signal_l2_norm=float(
            binding["minimum_family_source_modal_signal_l2_norm"]
        ),
        projection_full_width_relative_error_max=float(
            binding["projection_full_width_relative_error_max"]
        ),
        projection_full_width_cosine_min=float(
            binding["projection_full_width_cosine_min"]
        ),
        worst_family_projection_relative_error_max=float(
            binding["worst_family_projection_relative_error_max"]
        ),
        worst_family_projection_cosine_min=float(
            binding["worst_family_projection_cosine_min"]
        ),
        minimum_family_source_full_width_signal_l2_norm=float(
            binding[
                "minimum_family_source_full_width_signal_l2_norm"
            ]
        ),
        execution_fidelity_axis_names=(
            _CANDIDATE_BEHAVIOR_HARD_GATE_AXES
        ),
    )


def make_gemma3_l3_l4_progressive_protocol(
    *,
    corpus: DevelopmentCorpus,
    seed_runtime_binding_sha256: str,
    fit_panel_binding_sha256: str,
    selection_panel_binding_sha256: str,
    guard_preclaim_binding_sha256: str,
    resource_budget: ProgressiveResourceBudget,
    seed_resources: ProgressiveResourceFootprint,
    operator_nrmse_max: float | None = None,
    max_iterations: int = 16,
    max_proposals_per_iteration: int = 8,
    minimum_repair_relative_burden_reduction: float = 0.02,
    maximum_repair_axis_regression_fraction: float = 0.25,
    residual_head_rank: int = 8,
    compact_after_fidelity: bool = True,
) -> ProgressiveCompilationProtocol:
    """Bind the generic A-only loop to the exact current Gemma seed."""

    binding = _legacy_binding()
    if not isinstance(seed_resources, ProgressiveResourceFootprint):
        raise TypeError(
            "seed_resources must be ProgressiveResourceFootprint"
        )
    if seed_resources.candidate_execution_sha256 != str(
        binding["seed_candidate_execution_sha256"]
    ):
        raise ValueError(
            "seed resource accounting does not bind the legacy execution"
        )
    progressive_runtime = _require_sha256(
        seed_runtime_binding_sha256,
        label="progressive seed runtime binding",
    )
    return ProgressiveCompilationProtocol(
        protocol_id=GEMMA3_L3_L4_PROGRESSIVE_PROTOCOL_ID,
        source_model_sha256=str(binding["source_model_sha256"]),
        seed_candidate_artifact_sha256=str(
            binding["seed_candidate_artifact_sha256"]
        ),
        seed_candidate_execution_sha256=str(
            binding["seed_candidate_execution_sha256"]
        ),
        seed_runtime_binding_sha256=progressive_runtime,
        seed_resource_receipt_sha256=seed_resources.receipt_sha256,
        seed_lineage_sha256s=tuple(
            binding["seed_lineage_sha256s"]  # type: ignore[arg-type]
        ),
        corpus=corpus,
        development_role_binding_sha256s=(
            ("calibration_a_fit", fit_panel_binding_sha256),
            ("calibration_a_guard", guard_preclaim_binding_sha256),
            (
                "calibration_a_selection",
                selection_panel_binding_sha256,
            ),
        ),
        forbidden_assessment_manifest_sha256s=(
            str(binding["calibration_b_manifest_sha256"]),
        ),
        fidelity_targets=gemma3_l3_l4_progressive_fidelity_targets(
            operator_nrmse_max=operator_nrmse_max,
        ),
        resource_budget=resource_budget,
        max_iterations=max_iterations,
        max_proposals_per_iteration=max_proposals_per_iteration,
        minimum_repair_relative_burden_reduction=(
            minimum_repair_relative_burden_reduction
        ),
        maximum_repair_axis_regression_fraction=(
            maximum_repair_axis_regression_fraction
        ),
        staging_transitions=(
            ProgressiveStagingTransition(
                transition_id="gemma-l3-l4-x4-then-h4-rank-head",
                parent_iteration=0,
                mutation_kind="add_residual_edge",
                parent_mutation_kind="seed",
                required_successor_mutation_kind="widen_carrier",
                target_location="layer.4.mlp.normalized_input",
                target_rank_count=residual_head_rank,
                completion_target_location="layer.4.output",
                completion_target_rank_count=residual_head_rank,
                target_axis_names=(
                    "boundary.relative_error",
                    "boundary.worst_family_relative_error",
                ),
                invariant_axis_names=_X4_FROZEN_DIAGNOSTIC_AXES,
                permitted_regression_axis_names=(
                    _X4_PERMITTED_REGRESSION_AXES
                ),
                completion_axis_names=(
                    _CANDIDATE_BEHAVIOR_HARD_GATE_AXES
                ),
                minimum_relative_burden_reduction=0.02,
            ),
        ),
        compact_after_fidelity=compact_after_fidelity,
    )


def current_gemma3_l3_l4_progressive_seed(
    *,
    resources: ProgressiveResourceFootprint,
    runtime_binding_sha256: str,
) -> ProgressiveCandidate:
    """Wrap the authenticated legacy 64-mode candidate as iteration zero."""

    binding = _legacy_binding()
    return ProgressiveCandidate(
        candidate_id="gemma3-l3-l4-graph-organized-svd-rank64-seed",
        iteration=0,
        artifact_sha256=str(
            binding["seed_candidate_artifact_sha256"]
        ),
        execution_sha256=str(
            binding["seed_candidate_execution_sha256"]
        ),
        runtime_binding_sha256=_require_sha256(
            runtime_binding_sha256,
            label="progressive seed runtime binding",
        ),
        resources=resources,
        mutation_kind="seed",
    )


def gemma3_l3_l4_legacy_progressive_binding_metadata(
) -> dict[str, object]:
    """Expose prompt-blind lineage facts for reports and tests."""

    binding = _legacy_binding()
    return {
        **binding,
        "progressive_protocol_id": GEMMA3_L3_L4_PROGRESSIVE_PROTOCOL_ID,
        "development_roles": (
            "calibration_a_fit",
            "calibration_a_selection",
            "calibration_a_guard",
        ),
        "calibration_b_access": "forbidden_identity_only",
        "legacy_one_shot_accepts_new_progressive_winner": False,
        "required_next_boundary": (
            "candidate_bound_shadow_protocol_and_runtime_v2"
        ),
    }


__all__ = [
    "GEMMA3_L3_L4_PROGRESSIVE_PROTOCOL_ID",
    "current_gemma3_l3_l4_progressive_seed",
    "gemma3_l3_l4_legacy_progressive_binding_metadata",
    "gemma3_l3_l4_progressive_fidelity_targets",
    "make_gemma3_l3_l4_progressive_protocol",
]
