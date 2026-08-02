"""A-only three-basis shadow comparison for the signed-g8 research rung.

The runner reuses the already-consumed 16-prompt Calibration-A fit panel and
executes three rank-45 plans against one authenticated factorized Gemma model:
the signed eight-group local-SVD candidate, signed GFA, and global SVD.  The
source path remains authoritative and candidate values are metrics-only.

Calibration-B, validation, and test are never opened by this module.  The
result is a development localization diagnostic, not qualification.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .conditional_spectral_generator import ConditionalSpectralGeneratorPlan
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    INTERIOR_ORIGINS,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    evaluate_gemma3_l3_l4_conditional_spectral_development_shadow,
)
from .gemma3_l3_l4_conditional_spectral_shadow_runtime import (
    Gemma3L3L4ConditionalSpectralShadowRuntime,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    DEFAULT_FROZEN_ARTIFACT_SHA256,
    DEFAULT_FROZEN_REPORT_SHA256,
    DEFAULT_FROZEN_TENSOR_FILE_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE_ARTIFACT,
    _file_sha256,
    _reserve_outputs,
    _stage_json,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_confirmation_experiment import (
    _reference_plans,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PANEL,
    _EXPECTED_A_FIT_TOKENIZER_POST_SHA256,
    _EXPECTED_FACTORIZED_EXECUTION_SHA256,
    _EXPECTED_FACTORIZED_MODEL_SHA256,
    _EXPECTED_RAW_MODEL_SHA256,
    _frozen_tokenizer_integrity_check,
    _load_panel,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)
from .shadow_fidelity import ESTABLISHED_SHADOW_FIDELITY_GATES


__all__ = [
    "DEFAULT_OUTPUT",
    "run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "shadow-basis-comparison-a-fit16-dev-v2.json"
)
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_"
    "shadow_basis_comparison_development"
)
_FORMAT_VERSION = 2
_REPORT_DOMAIN = (
    b"fisher-graph:signed-g8-shadow-basis-comparison-report:v2\0"
)
_VARIANT_DOMAIN = (
    b"fisher-graph:signed-g8-shadow-basis-comparison-variant:v1\0"
)
_SOURCE_RECEIPT_DOMAIN = (
    b"fisher-graph:signed-g8-shadow-basis-source-summary-receipt:v2\0"
)
_FACTORIZED_SCOPE = "factorized_refit"
_MINIMUM_MAX_LENGTH = 10
_VARIANT_ORDER = (
    "signed_local_svd_g8",
    "signed_gfa_rank45",
    "global_svd_rank45",
)
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_compiled_plan_tensors": False,
    "contains_scalar_metrics": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _source_behavior_receipt(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    report = _mapping(value, label=label)
    aggregate = _mapping(report.get("aggregate"), label=f"{label}.aggregate")
    family_summary = _mapping(
        report.get("family_summary"),
        label=f"{label}.family_summary",
    )
    families = family_summary.get("families")
    if not isinstance(families, (tuple, list)):
        raise TypeError(f"{label}.families must be a sequence")
    family_rows = []
    for row in families:
        family = _mapping(row, label=f"{label}.family")
        family_rows.append(
            {
                "family_id": family["family_id"],
                "example_count": family["example_count"],
                "supervised_tokens": family["supervised_tokens"],
                "source_summed_nll": family["source_summed_nll"],
                "source_nll_per_token": family["source_nll_per_token"],
            }
        )
    return {
        "aggregate": {
            "example_count": aggregate["example_count"],
            "supervised_tokens": aggregate["supervised_tokens"],
            "source_summed_nll": aggregate["source_summed_nll"],
            "source_nll_per_token": aggregate["source_nll_per_token"],
        },
        "families": tuple(family_rows),
    }


def _source_execution_summary_receipt(evaluation: object) -> dict[str, object]:
    """Extract source inputs and scalar summaries that must match every arm."""

    report = _mapping(evaluation, label="evaluation")
    manifest = _mapping(report.get("manifest"), label="manifest")
    coverage = _mapping(report.get("coverage"), label="coverage")
    target_modal = _mapping(report.get("target_modal"), label="target_modal")
    modal_pooled = _mapping(target_modal.get("pooled"), label="target modal")
    full_width = _mapping(
        report.get("full_width_boundary"),
        label="full_width_boundary",
    )
    full_pooled = _mapping(full_width.get("pooled"), label="full width")
    receipts = report.get("receipts")
    if not isinstance(receipts, (tuple, list)) or not receipts:
        raise TypeError("evaluation receipts must be a nonempty sequence")
    prompt_rows = []
    for row in receipts:
        receipt = _mapping(row, label="prompt receipt")
        prompt_rows.append(
            {
                name: receipt[name]
                for name in (
                    "example_id",
                    "family_id",
                    "prompt_sha256",
                    "tokenized_tokens",
                    "supervised_tokens",
                    "affected_supervised_tokens",
                    "model_inputs_sha256",
                    "execution_grid_sha256",
                )
            }
        )
    result = {
        "manifest": {
            name: manifest[name]
            for name in (
                "manifest_sha256",
                "example_count",
                "family_count",
                "strict_example_membership",
                "strict_family_membership",
            )
        },
        "behavioral": _source_behavior_receipt(
            report.get("behavioral"),
            label="behavioral",
        ),
        "affected_behavioral": _source_behavior_receipt(
            report.get("affected_behavioral"),
            label="affected_behavioral",
        ),
        "coverage": {
            name: coverage[name]
            for name in (
                "example_count",
                "supervised_tokens",
                "affected_supervised_tokens",
                "valid_target_rows",
                "source_eligible_rows",
                "affected_target_rows",
            )
        },
        "target_modal_source": {
            "affected_rows": modal_pooled["affected_rows"],
            "scalar_elements": modal_pooled["scalar_elements"],
            "source_signal_l2_norm": modal_pooled["source_signal_l2_norm"],
        },
        "full_width_source": {
            "affected_rows": full_pooled["affected_rows"],
            "scalar_elements": full_pooled["scalar_elements"],
            "source_signal_l2_norm": full_pooled["source_signal_l2_norm"],
        },
        "prompts": tuple(prompt_rows),
    }
    _canonical_json_bytes(result)
    return result


def _variant_metrics(evaluation: object) -> dict[str, object]:
    report = _mapping(evaluation, label="evaluation")
    behavioral = _mapping(report.get("behavioral"), label="behavioral")
    affected = _mapping(
        report.get("affected_behavioral"),
        label="affected_behavioral",
    )
    behavior_aggregate = _mapping(
        behavioral.get("aggregate"),
        label="behavioral.aggregate",
    )
    affected_aggregate = _mapping(
        affected.get("aggregate"),
        label="affected.aggregate",
    )
    behavior_gates = _mapping(behavioral.get("gates"), label="behavioral.gates")
    affected_gates = _mapping(affected.get("gates"), label="affected.gates")
    behavior_per_prompt = _mapping(
        behavioral.get("per_prompt"),
        label="behavioral.per_prompt",
    )
    affected_per_prompt = _mapping(
        affected.get("per_prompt"),
        label="affected.per_prompt",
    )
    behavior_prompt_delta = _mapping(
        behavior_per_prompt.get("absolute_delta_nll_per_token"),
        label="behavioral per-prompt delta",
    )
    behavior_prompt_top1 = _mapping(
        behavior_per_prompt.get("top1_agreement_to_source"),
        label="behavioral per-prompt top1",
    )
    affected_prompt_delta = _mapping(
        affected_per_prompt.get("absolute_delta_nll_per_token"),
        label="affected per-prompt delta",
    )
    affected_prompt_top1 = _mapping(
        affected_per_prompt.get("top1_agreement_to_source"),
        label="affected per-prompt top1",
    )
    modal = _mapping(
        _mapping(report.get("target_modal"), label="target_modal").get("pooled"),
        label="target_modal.pooled",
    )
    full = _mapping(
        _mapping(
            report.get("full_width_boundary"),
            label="full_width_boundary",
        ).get("pooled"),
        label="full_width_boundary.pooled",
    )
    metrics = {
        "behavioral": {
            "delta_nll_per_token": behavior_aggregate["delta_nll_per_token"],
            "source_to_candidate_kl_per_token": behavior_aggregate[
                "source_to_candidate_kl_per_token"
            ],
            "top1_agreement_to_source": behavior_aggregate[
                "top1_agreement_to_source"
            ],
            "per_prompt_p90_absolute_delta_nll_per_token": (
                behavior_prompt_delta["p90"]
            ),
            "per_prompt_p10_top1_agreement_to_source": (
                behavior_prompt_top1["p10"]
            ),
            "gates_passed": behavior_gates["passed"],
        },
        "affected_behavioral": {
            "delta_nll_per_token": affected_aggregate["delta_nll_per_token"],
            "source_to_candidate_kl_per_token": affected_aggregate[
                "source_to_candidate_kl_per_token"
            ],
            "top1_agreement_to_source": affected_aggregate[
                "top1_agreement_to_source"
            ],
            "per_prompt_p90_absolute_delta_nll_per_token": (
                affected_prompt_delta["p90"]
            ),
            "per_prompt_p10_top1_agreement_to_source": (
                affected_prompt_top1["p10"]
            ),
            "gates_passed": affected_gates["passed"],
        },
        "target_modal": {
            "relative_l2_error": modal["relative_l2_error"],
            "cosine": modal["cosine"],
        },
        "full_width_boundary": {
            "relative_l2_error": full["relative_l2_error"],
            "cosine": full["cosine"],
        },
    }
    _canonical_json_bytes(metrics)
    return metrics


def _fraction_recovery(reference: float, candidate: float, *, square: bool) -> float:
    baseline = _finite(reference, label="reference metric")
    compared = _finite(candidate, label="candidate metric")
    if baseline <= 0.0 or compared < 0.0:
        raise ValueError("recovery metrics must be nonnegative with positive reference")
    ratio = compared / baseline
    return 1.0 - (ratio * ratio if square else ratio)


def _affected_burden(metrics: Mapping[str, object]) -> tuple[float, ...]:
    return (
        abs(
            _finite(
                metrics["delta_nll_per_token"],
                label="affected delta NLL",
            )
        ),
        _finite(
            metrics["source_to_candidate_kl_per_token"],
            label="affected KL",
        ),
        1.0
        - _finite(
            metrics["top1_agreement_to_source"],
            label="affected top1",
        ),
        _finite(
            metrics["per_prompt_p90_absolute_delta_nll_per_token"],
            label="affected prompt-p90 delta NLL",
        ),
        1.0
        - _finite(
            metrics["per_prompt_p10_top1_agreement_to_source"],
            label="affected prompt-p10 top1",
        ),
    )


def _axiswise_dominates(
    candidate: tuple[float, ...],
    reference: tuple[float, ...],
) -> bool:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("dominance burdens must have equal nonzero width")
    pairs = tuple(zip(candidate, reference, strict=True))
    return all(left <= right for left, right in pairs) and any(
        left < right for left, right in pairs
    )


def compare_shadow_basis_evaluations(
    evaluations: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on source drift, then reduce the three scalar reports."""

    if not isinstance(evaluations, Mapping) or tuple(evaluations) != _VARIANT_ORDER:
        raise ValueError("basis evaluations must follow the frozen variant order")
    source_receipts = {
        name: _source_execution_summary_receipt(evaluations[name])
        for name in _VARIANT_ORDER
    }
    reference_receipt = source_receipts[_VARIANT_ORDER[0]]
    if any(
        _canonical_json_bytes(source_receipts[name])
        != _canonical_json_bytes(reference_receipt)
        for name in _VARIANT_ORDER[1:]
    ):
        raise ValueError("basis comparison source execution differs")
    source_receipt_sha256 = _json_sha256(
        reference_receipt,
        domain=_SOURCE_RECEIPT_DOMAIN,
    )
    metrics = {
        name: _variant_metrics(evaluations[name])
        for name in _VARIANT_ORDER
    }
    baseline = metrics["signed_local_svd_g8"]
    comparisons: dict[str, object] = {}
    for name in _VARIANT_ORDER[1:]:
        row = metrics[name]
        baseline_modal = _mapping(
            baseline["target_modal"],
            label="baseline target modal",
        )
        row_modal = _mapping(row["target_modal"], label="target modal")
        baseline_full = _mapping(
            baseline["full_width_boundary"],
            label="baseline full width",
        )
        row_full = _mapping(row["full_width_boundary"], label="full width")
        baseline_behavior = _mapping(
            baseline["behavioral"],
            label="baseline behavioral",
        )
        row_behavior = _mapping(row["behavioral"], label="behavioral")
        baseline_affected = _mapping(
            baseline["affected_behavioral"],
            label="baseline affected",
        )
        row_affected = _mapping(
            row["affected_behavioral"],
            label="affected",
        )
        comparisons[name] = {
            "target_modal_sse_recovery_fraction": _fraction_recovery(
                _finite(
                    baseline_modal["relative_l2_error"],
                    label="baseline modal error",
                ),
                _finite(row_modal["relative_l2_error"], label="modal error"),
                square=True,
            ),
            "full_width_sse_recovery_fraction": _fraction_recovery(
                _finite(
                    baseline_full["relative_l2_error"],
                    label="baseline full-width error",
                ),
                _finite(
                    row_full["relative_l2_error"],
                    label="full-width error",
                ),
                square=True,
            ),
            "behavioral_kl_recovery_fraction": _fraction_recovery(
                _finite(
                    baseline_behavior["source_to_candidate_kl_per_token"],
                    label="baseline behavioral KL",
                ),
                _finite(
                    row_behavior["source_to_candidate_kl_per_token"],
                    label="behavioral KL",
                ),
                square=False,
            ),
            "affected_kl_recovery_fraction": _fraction_recovery(
                _finite(
                    baseline_affected["source_to_candidate_kl_per_token"],
                    label="baseline affected KL",
                ),
                _finite(
                    row_affected["source_to_candidate_kl_per_token"],
                    label="affected KL",
                ),
                square=False,
            ),
            "behavioral_top1_gain": _finite(
                row_behavior["top1_agreement_to_source"],
                label="behavioral top1",
            )
            - _finite(
                baseline_behavior["top1_agreement_to_source"],
                label="baseline behavioral top1",
            ),
            "affected_top1_gain": _finite(
                row_affected["top1_agreement_to_source"],
                label="affected top1",
            )
            - _finite(
                baseline_affected["top1_agreement_to_source"],
                label="baseline affected top1",
            ),
        }
    arm_passes = {}
    affected_burdens = {}
    for name in _VARIANT_ORDER:
        behavior = _mapping(
            metrics[name]["behavioral"],  # type: ignore[index]
            label=f"{name}.behavioral",
        )
        affected = _mapping(
            metrics[name]["affected_behavioral"],  # type: ignore[index]
            label=f"{name}.affected_behavioral",
        )
        arm_passes[name] = (
            behavior["gates_passed"] is True
            and affected["gates_passed"] is True
        )
        affected_burdens[name] = _affected_burden(affected)
    pass_pattern = "".join(
        "1" if arm_passes[name] else "0" for name in _VARIANT_ORDER
    )
    global_dominates = _axiswise_dominates(
        affected_burdens["global_svd_rank45"],
        affected_burdens["signed_local_svd_g8"],
    ) and _axiswise_dominates(
        affected_burdens["global_svd_rank45"],
        affected_burdens["signed_gfa_rank45"],
    )
    if pass_pattern == "111":
        classification = "rank45_linear_carrier_viable_across_all_three_bases"
    elif pass_pattern == "101":
        classification = "local_svd_and_global_svd_viable_signed_gfa_rejected"
    elif pass_pattern == "011":
        classification = (
            "signed_gfa_and_global_svd_viable_local_svd_grouping_rejected"
        )
    elif pass_pattern == "001":
        classification = "global_svd_only_viable_basis_construction_is_blocker"
    elif pass_pattern == "000" and global_dominates:
        classification = (
            "basis_contributes_but_rank45_fixed_reference_family_still_fails"
        )
    elif pass_pattern == "000":
        classification = "no_rank45_basis_viable_attribution_inconclusive"
    elif not arm_passes["global_svd_rank45"]:
        classification = "graph_specific_reversal_attribution_inconclusive"
    else:
        classification = "nonmonotonic_basis_outcome_inconclusive"
    return {
        "source_execution_summary_matched": True,
        "source_execution_summary_receipt_sha256": source_receipt_sha256,
        "source_execution_summary_receipt": reference_receipt,
        "source_hidden_tensor_equality_claim": False,
        "variant_metrics": metrics,
        "relative_to_signed_local_svd_g8": comparisons,
        "classification_protocol": {
            "arm_pass_requires_behavioral_and_affected_gates": True,
            "variant_bit_order": _VARIANT_ORDER,
            "global_dominance_scope": "affected_behavioral_five_axis_burden",
            "affected_burden_axes": (
                "absolute_delta_nll_per_token",
                "source_to_candidate_kl_per_token",
                "one_minus_top1_agreement",
                "per_prompt_p90_absolute_delta_nll_per_token",
                "one_minus_per_prompt_p10_top1_agreement",
            ),
            "dominance_requires_no_worse_every_axis_and_strictly_better_one": True,
        },
        "classification": classification,
        "arm_passes": arm_passes,
        "pass_pattern": pass_pattern,
        "affected_burdens": affected_burdens,
        "global_svd_axiswise_dominates_both_graph_arms": global_dominates,
    }


def _variant_receipt(
    *,
    role: str,
    plan: ConditionalSpectralGeneratorPlan,
    common_binding: Mapping[str, object],
) -> dict[str, object]:
    metadata = plan.metadata()
    tensor_hashes = _mapping(
        metadata.get("tensor_sha256s"),
        label=f"{role}.tensor_sha256s",
    )
    payload = {
        "schema": "fisher_graph.signed_g8_shadow_basis_arm_receipt",
        "format_version": 1,
        "role": role,
        "plan_artifact_sha256": plan.artifact_sha256,
        "source_basis_sha256": tensor_hashes["source_basis"],
        "target_basis_sha256": tensor_hashes["target_basis"],
        "source_scales_sha256": tensor_hashes["source_scales"],
        "target_singular_values_sha256": tensor_hashes[
            "target_singular_values"
        ],
        "response_binding_sha256": plan.response_binding_sha256,
        "fit_weighted_kernels_sha256": plan.fit_weighted_kernels_sha256,
        "fit_knot_origins": plan.fit_knot_origins,
        "source_modes": plan.source_modes,
        "source_rank": plan.source_rank,
        "target_modes": plan.target_modes,
        "target_rank": plan.target_rank,
        "lag_count": plan.lag_count,
        "fft_length": plan.fft_length,
        "rank_semantics": plan.rank_semantics,
        "stored_coefficient_count": plan.stored_coefficient_count,
        "prepared_storage_bytes": plan.accounting().prepared_storage_bytes,
        "common_binding": dict(common_binding),
        "development_only": True,
        "candidate_serving_authorized": False,
    }
    return {
        **payload,
        "artifact_sha256": _json_sha256(payload, domain=_VARIANT_DOMAIN),
    }


def _plan_comparison_invariants(
    plan: ConditionalSpectralGeneratorPlan,
    *,
    role: str,
) -> dict[str, object]:
    metadata = plan.metadata()
    tensor_hashes = _mapping(
        metadata.get("tensor_sha256s"),
        label=f"{role}.tensor_sha256s",
    )
    return {
        "response_binding_sha256": plan.response_binding_sha256,
        "fit_weighted_kernels_sha256": plan.fit_weighted_kernels_sha256,
        "source_scales_sha256": tensor_hashes["source_scales"],
        "target_basis_sha256": tensor_hashes["target_basis"],
        "target_singular_values_sha256": tensor_hashes[
            "target_singular_values"
        ],
        "fit_knot_origins": plan.fit_knot_origins,
        "source_modes": plan.source_modes,
        "source_rank": plan.source_rank,
        "target_modes": plan.target_modes,
        "target_rank": plan.target_rank,
        "lag_count": plan.lag_count,
        "fft_length": plan.fft_length,
        "input_transform": plan.input_transform,
        "input_transform_semantics": plan.input_transform_semantics,
        "square_transform_scope": plan.square_transform_scope,
        "interpolation_semantics": plan.interpolation_semantics,
        "factorization_semantics": plan.factorization_semantics,
        "core_semantics": plan.core_semantics,
        "fit_origin_scope": plan.fit_origin_scope,
        "heldout_origins_used_for_fit": plan.heldout_origins_used_for_fit,
        "cross_mode_terms_measured": plan.cross_mode_terms_measured,
        "stored_coefficient_count": plan.stored_coefficient_count,
        "prepared_storage_bytes": plan.accounting().prepared_storage_bytes,
    }


def _validate_rank45_basis_plans(
    plans: Mapping[str, ConditionalSpectralGeneratorPlan],
) -> dict[str, object]:
    if not isinstance(plans, Mapping) or tuple(plans) != _VARIANT_ORDER:
        raise ValueError("three-way plans do not follow the frozen order")
    plan_hashes = tuple(plans[name].artifact_sha256 for name in _VARIANT_ORDER)
    source_basis_hashes = tuple(
        _mapping(
            plans[name].metadata().get("tensor_sha256s"),
            label=f"{name}.tensor_sha256s",
        )["source_basis"]
        for name in _VARIANT_ORDER
    )
    if len(set(plan_hashes)) != len(plan_hashes):
        raise ValueError("three-way plans must have unique artifact identities")
    if len(set(source_basis_hashes)) != len(source_basis_hashes):
        raise ValueError("three-way plans must have unique source bases")
    invariants = {
        name: _plan_comparison_invariants(plans[name], role=name)
        for name in _VARIANT_ORDER
    }
    reference = invariants[_VARIANT_ORDER[0]]
    if any(
        _canonical_json_bytes(invariants[name])
        != _canonical_json_bytes(reference)
        for name in _VARIANT_ORDER[1:]
    ):
        raise ValueError("three-way plans are not rank, geometry, and size matched")
    if (
        reference["source_modes"] != 64
        or reference["source_rank"] != 45
        or reference["target_modes"] != 64
        or reference["target_rank"] != 64
    ):
        raise ValueError("three-way plans do not have the frozen rank-45 geometry")
    return {
        "shared_invariants": reference,
        "plan_artifact_sha256s": plan_hashes,
        "source_basis_sha256s": source_basis_hashes,
    }


def _live_factorized_identity(adapter: Gemma3CausalLMAdapter) -> tuple[str, str]:
    model_sha256 = adapter.model_fingerprint()
    execution_sha256 = adapter.execution_fingerprint()
    if (
        model_sha256 != _EXPECTED_FACTORIZED_MODEL_SHA256
        or execution_sha256 != _EXPECTED_FACTORIZED_EXECUTION_SHA256
    ):
        raise ValueError("live factorized Gemma differs")
    return model_sha256, execution_sha256


def _total_model_forward_count(evaluations: Mapping[str, object]) -> int:
    total = 0
    for name in _VARIANT_ORDER:
        execution = _mapping(
            _mapping(evaluations[name], label=name).get("execution"),
            label=f"{name}.execution",
        )
        count = execution.get("total_model_forward_count")
        if type(count) is not int or count != 48:
            raise ValueError(f"{name} must execute exactly 48 model forwards")
        total += count
    if total != 144:
        raise ValueError("three-way comparison did not execute exactly 144 forwards")
    return total


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("basis comparison output must be JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    reservation = _reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = _json_sha256(report, domain=_REPORT_DOMAIN)
        stage = _stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison(
    *,
    fit_source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    panel_path: Path | str = DEFAULT_PANEL,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Run all three frozen rank-45 bases on reused Calibration-A."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite basis comparison report")
    if (
        type(max_length) is not int
        or not _MINIMUM_MAX_LENGTH <= max_length <= 256
    ):
        raise ValueError("max_length must lie in [10, 256]")
    examples, panel_receipt = _load_panel(panel_path)
    fit_source = load_gemma3_spectral_source(
        fit_source_artifact_path,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_FROZEN_REPORT_SHA256,
    )
    signed_gfa_plan, global_svd_plan = _reference_plans(fit_source, parent)
    basis = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plans = {
        "signed_local_svd_g8": candidate.plan,
        "signed_gfa_rank45": signed_gfa_plan,
        "global_svd_rank45": global_svd_plan,
    }
    plan_comparison = _validate_rank45_basis_plans(plans)
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    (
        tokenizer,
        tokenizer_contract,
    ) = _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    tokenizer_integrity_check = _frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )
    common_arm_binding = {
        "signed_g8_candidate_artifact_sha256": candidate.artifact_sha256,
        "signed_g8_candidate_binding_sha256": _json_sha256(
            dict(candidate.binding),
            domain=_VARIANT_DOMAIN,
        ),
        "fit_response_tensor_file_sha256": fit_source.file_sha256,
        "fit_response_report_file_sha256": fit_source.report_file_sha256,
        "fit_response_report_payload_sha256": (
            fit_source.report_payload_sha256
        ),
        "fit_response_mapping_artifact_sha256": (
            fit_source.mapping.artifact_sha256
        ),
        "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
        "basis_package_payload_sha256": DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
        "panel_file_sha256": panel_receipt["file_sha256"],
        "panel_source_fit_prompt_index_sha256": panel_receipt[
            "source_fit_prompt_index_sha256"
        ],
        "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
        "factorized_live_model_sha256": _EXPECTED_FACTORIZED_MODEL_SHA256,
        "factorized_adapter_execution_sha256": (
            _EXPECTED_FACTORIZED_EXECUTION_SHA256
        ),
        "tokenizer_class": tokenizer_contract["tokenizer_class"],
        "tokenizer_configuration_sha256": tokenizer_contract[
            "configuration_sha256"
        ],
        "tokenizer_initial_backend_sha256": tokenizer_contract[
            "backend_serialized_sha256"
        ],
        "tokenizer_post_backend_sha256": (
            _EXPECTED_A_FIT_TOKENIZER_POST_SHA256
        ),
        "max_length": max_length,
        "shadow_fidelity_gates": ESTABLISHED_SHADOW_FIDELITY_GATES.metadata(),
    }
    variant_receipts = {
        name: _variant_receipt(
            role=name,
            plan=plan,
            common_binding=common_arm_binding,
        )
        for name, plan in plans.items()
    }
    if len(
        {
            str(receipt["artifact_sha256"])
            for receipt in variant_receipts.values()
        }
    ) != len(_VARIANT_ORDER):
        raise ValueError("three-way arm receipt identities must be unique")
    model_metadata = candidate.model
    if model_metadata.get("source_model_sha256") != _EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("candidate raw model lineage differs")
    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("live raw Gemma differs from the frozen candidate")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    evaluations: dict[str, object] = {}
    runtime_bindings: dict[str, object] = {}
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256, factorized_execution_sha256 = (
            _live_factorized_identity(adapter)
        )
        for name, plan in plans.items():
            _live_factorized_identity(adapter)
            runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
                plan,
                basis,
                candidate_artifact_sha256=str(
                    variant_receipts[name]["artifact_sha256"]
                ),
                candidate_method=name,
                candidate_binding=candidate.binding,
                candidate_model=candidate.model,
                expected_plan_artifact_sha256=plan.artifact_sha256,
                expected_basis_payload_sha256=(
                    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
                ),
                expected_live_model_sha256=factorized_model_sha256,
                expected_adapter_execution_sha256=(
                    factorized_execution_sha256
                ),
                analysis_device="cpu",
            )
            runtime_metadata = runtime.metadata()
            if (
                runtime_metadata.get("candidate_method") != name
                or runtime_metadata.get("plan_artifact_sha256")
                != plan.artifact_sha256
                or runtime_metadata.get("candidate_artifact_sha256")
                != variant_receipts[name]["artifact_sha256"]
            ):
                raise ValueError("three-way runtime binding differs from its arm")
            runtime_bindings[name] = runtime_metadata
            evaluations[name] = (
                evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
                    runtime=runtime,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    examples=examples,
                    max_length=max_length,
                    model_input_device=device,
                    tokenizer_integrity_check=tokenizer_integrity_check,
                )
            )
            runtime.validate_integrity()
            _live_factorized_identity(adapter)
        runtime_binding_hashes = tuple(
            _mapping(runtime_bindings[name], label=name)[
                "runtime_binding_sha256"
            ]
            for name in _VARIANT_ORDER
        )
        if len(set(runtime_binding_hashes)) != len(runtime_binding_hashes):
            raise ValueError("three-way runtime binding identities must be unique")
        tokenizer_integrity_check("after")
    finally:
        switcher.close()
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("basis comparison did not restore raw Gemma")
    comparison = compare_shadow_basis_evaluations(evaluations)
    total_model_forward_count = _total_model_forward_count(evaluations)
    variant_reports = {
        name: {
            "role": name,
            "variant_receipt": variant_receipts[name],
            "plan_artifact_sha256": plans[name].artifact_sha256,
            "rank_semantics": plans[name].rank_semantics,
            "source_rank": plans[name].source_rank,
            "target_rank": plans[name].target_rank,
            "stored_coefficient_count": plans[name].stored_coefficient_count,
            "prepared_storage_bytes": (
                plans[name].accounting().prepared_storage_bytes
            ),
            "runtime_binding": runtime_bindings[name],
            "evaluation": evaluations[name],
        }
        for name in _VARIANT_ORDER
    }
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "reused_calibration_a_fit_three_basis_localization",
        "lineage": {
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "basis_package_payload_sha256": (
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            "fit_response_tensor_file_sha256": fit_source.file_sha256,
            "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
            "raw_source_model_sha256": _EXPECTED_RAW_MODEL_SHA256,
            "factorized_live_model_sha256": (
                _EXPECTED_FACTORIZED_MODEL_SHA256
            ),
            "factorized_adapter_execution_sha256": (
                _EXPECTED_FACTORIZED_EXECUTION_SHA256
            ),
        },
        "panel": panel_receipt,
        "protocol": {
            "variant_order": _VARIANT_ORDER,
            "same_model_instance_for_all_variants": True,
            "same_tokenizer_instance_for_all_variants": True,
            "source_execution_summary_must_match_exactly": True,
            "source_hidden_tensor_equality_claim": False,
            "arm_receipts_are_domain_separated_metadata_only": True,
            "source_path_authoritative": True,
            "candidate_outputs_metrics_only": True,
            "max_length": max_length,
            "model_forwards_per_prompt_per_variant": 3,
            "expected_total_model_forward_count": 144,
            "tokenizer_integrity_checked_before_and_after_each_prompt": True,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "variants": variant_reports,
        "plan_comparison": plan_comparison,
        "comparison": comparison,
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "variant_count": 3,
            "total_model_forward_count": total_model_forward_count,
            "all_plans_size_matched": True,
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "development_localization_complete": True,
            "reused_calibration_a_fit_only": True,
            "source_execution_summary_matched": comparison[
                "source_execution_summary_matched"
            ],
            "formal_qualification": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "artifact": {
            "file": str(destination),
            "committable": False,
        },
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare signed local-SVD, signed GFA, and global SVD shadows",
    )
    parser.add_argument("--fit-source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--basis-package", default=DEFAULT_BASIS_PACKAGE)
    parser.add_argument("--base-artifact", default=DEFAULT_FULL_MLP_STACK_ARTIFACT)
    parser.add_argument(
        "--refit-artifact",
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "classification": report["comparison"][  # type: ignore[index]
                    "classification"
                ],
                "source_execution_summary_matched": report["comparison"][
                    "source_execution_summary_matched"
                ],  # type: ignore[index]
                "variant_metrics": report["comparison"][  # type: ignore[index]
                    "variant_metrics"
                ],
                "relative_to_signed_local_svd_g8": report["comparison"][
                    "relative_to_signed_local_svd_g8"
                ],  # type: ignore[index]
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
