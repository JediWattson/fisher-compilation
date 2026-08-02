"""A-only complete-H4 identity audit for the Gemma L3/L4 shadow.

The old exact-X4 suffix restored the normalized layer-4 MLP input but retained
the clamped residual carrier.  This runner asks the stronger noninvasive
question: after replaying that exact partial path, does injecting the exact
native ``layer.4.output`` restore the authoritative continuation?

The corrected rank64/X4 V2 report is strict-authenticated as the comparison
source.  Calibration-B, validation, and test are never opened.  All replay
and identity outputs are metrics-only; source logits remain authoritative.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from .adapters.gemma3 import Gemma3CausalLMAdapter
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
    _complete_h4_audit_receipt_sha256,
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
from .gemma3_l3_l4_graph_wavelet_signed_g8_rank64_oracle_ladder import (
    _EXPECTED_RANK64_PLAN_SHA256,
    _FACTORIZED_SCOPE,
    _REPORT_DOMAIN as _RANK64_V2_REPORT_DOMAIN,
    _behavior_metrics,
    _live_factorized_identity,
    _passed,
    build_rank64_global_svd_plan,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison import (
    _SOURCE_RECEIPT_DOMAIN,
    _canonical_json_bytes,
    _json_sha256,
    _mapping,
    _source_behavior_receipt,
    _source_execution_summary_receipt,
    _variant_metrics,
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
    "DEFAULT_RANK64_X4_BASELINE",
    "compare_complete_h4_identity_audit",
    "run_gemma3_l3_l4_complete_h4_identity_audit",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_RANK64_X4_BASELINE = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "rank64-oracle-ladder-a-fit16-dev-v2.json"
)
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "complete-h4-identity-a-fit16-dev-v1.json"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_complete_h4_identity_audit_development"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-complete-h4-audit-report:v1\0"
_MINIMUM_MAX_LENGTH = 10

_EXPECTED_BASELINE_FILE_SHA256 = (
    "a63b8c51e9364dfe057b7c05b4dc064fe5672d86aa0397159d16abb032f4a9d6"
)
_EXPECTED_BASELINE_REPORT_SHA256 = (
    "31fd41b1413e87d3e1fea3b51bd9eeb03bb02f0b69bfa6c22c4b58bb7e8bac40"
)
_EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "e1124a8b4ae14a217b80fe0bf6613e94168e23ff8102ed1bc2768829dee4914a"
)
_EXPECTED_RANK64_ARM_SHA256 = (
    "bd30b0283ab923c213afaf98aa42184c10598d01c87ffd32957719eb5bd27dd2"
)
_EXPECTED_RANK64_RUNTIME_SHA256 = (
    "2a9a722f89a79c717598628b9b8b93671900a94ab0d4c804e97f6575a97c4703"
)
_EXPECTED_PANEL_FILE_SHA256 = (
    "00e1f7bf07c918e3092b7b4cab5bbc2f7d0cac4df05a737061ce7383d8078809"
)
_EXPECTED_PANEL_INDEX_SHA256 = (
    "f16358f237a422ef9e66037c32fabf58edef0a976efeee34535b69458cba38ef"
)
_EXPECTED_MANIFEST_SHA256 = (
    "6823e88dd4c13e05f29bd252f0384ed806f4680e932851cec9fa14aa17013642"
)
_EXPECTED_TOKENIZER_CONFIGURATION_SHA256 = (
    "b02c42b40d0c95c70024c617c8774cde360991e2c949de1d35b51288ded31372"
)
_EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256 = (
    "c1a087240686a7d141101217051f76d5cd4cbe2b6093e3c3553fb26dcc4d0e9a"
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
    "truth_leaking_identity_control": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_rank64_x4_baseline(path: Path | str) -> dict[str, object]:
    source = Path(path)
    if _file_sha256(source) != _EXPECTED_BASELINE_FILE_SHA256:
        raise ValueError("rank64/X4 V2 baseline file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(_mapping(raw, label="rank64/X4 V2 baseline"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    if (
        claimed != _EXPECTED_BASELINE_REPORT_SHA256
        or _json_sha256(payload, domain=_RANK64_V2_REPORT_DOMAIN) != claimed
        or report.get("schema")
        != (
            "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_"
            "rank64_oracle_ladder_development"
        )
        or report.get("format_version") != 2
        or report.get("role")
        != "reused_calibration_a_fit_rank64_capacity_carrier_ladder"
    ):
        raise ValueError("rank64/X4 V2 baseline identity differs")
    safety = _mapping(report.get("safety"), label="baseline safety")
    status = _mapping(
        report.get("scientific_status"),
        label="baseline scientific status",
    )
    comparison = _mapping(report.get("comparison"), label="comparison")
    interpretation = _mapping(
        comparison.get("attribution_interpretation"),
        label="attribution interpretation",
    )
    if (
        safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_logits") is not False
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
        or status.get("reused_calibration_a_fit_only") is not True
        or status.get("boundary_audit_required") is not True
        or status.get("upstream_attribution_valid") is not False
        or status.get("formal_qualification") is not False
        or comparison.get("pass_pattern") != "000"
        or comparison.get("classification") != "exact_x4_continuation_invalid"
        or interpretation.get("exact_x4_site")
        != "layer.4.mlp.normalized_input"
        or interpretation.get("native_residual_stream_restored") is not False
    ):
        raise ValueError("rank64/X4 V2 premise or safety differs")
    lineage = _mapping(report.get("lineage"), label="baseline lineage")
    panel = _mapping(report.get("panel"), label="baseline panel")
    plan = _mapping(report.get("rank64_plan"), label="rank64 plan")
    arm = _mapping(report.get("rank64_arm_receipt"), label="rank64 arm")
    runtime = _mapping(report.get("runtime_binding"), label="runtime binding")
    source_receipt = _mapping(
        report.get("source_execution_summary_receipt"),
        label="source receipt",
    )
    source_receipt_sha256 = report.get(
        "source_execution_summary_receipt_sha256"
    )
    if (
        lineage.get("raw_source_model_sha256") != _EXPECTED_RAW_MODEL_SHA256
        or lineage.get("factorized_live_model_sha256")
        != _EXPECTED_FACTORIZED_MODEL_SHA256
        or lineage.get("factorized_adapter_execution_sha256")
        != _EXPECTED_FACTORIZED_EXECUTION_SHA256
        or panel.get("file_sha256") != _EXPECTED_PANEL_FILE_SHA256
        or panel.get("source_fit_prompt_index_sha256")
        != _EXPECTED_PANEL_INDEX_SHA256
        or panel.get("example_count") != 16
        or panel.get("family_count") != 8
        or plan.get("plan_artifact_sha256")
        != _EXPECTED_RANK64_PLAN_SHA256
        or arm.get("artifact_sha256") != _EXPECTED_RANK64_ARM_SHA256
        or runtime.get("runtime_binding_sha256")
        != _EXPECTED_RANK64_RUNTIME_SHA256
        or source_receipt_sha256 != _EXPECTED_SOURCE_RECEIPT_SHA256
        or _json_sha256(source_receipt, domain=_SOURCE_RECEIPT_DOMAIN)
        != source_receipt_sha256
    ):
        raise ValueError("rank64/X4 V2 execution lineage differs")
    common = _mapping(arm.get("common_binding"), label="arm common binding")
    if (
        common.get("tokenizer_configuration_sha256")
        != _EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        or common.get("tokenizer_initial_backend_sha256")
        != _EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
        or common.get("tokenizer_post_backend_sha256")
        != _EXPECTED_A_FIT_TOKENIZER_POST_SHA256
        or common.get("max_length") != DEFAULT_MAX_LENGTH
    ):
        raise ValueError("rank64/X4 V2 tokenizer or length differs")
    evaluation = _mapping(report.get("evaluation"), label="evaluation")
    manifest = _mapping(evaluation.get("manifest"), label="manifest")
    oracles = _mapping(
        evaluation.get("oracle_suffixes"),
        label="oracle suffixes",
    )
    oracle_receipts = oracles.get("receipts")
    if (
        manifest.get("manifest_sha256") != _EXPECTED_MANIFEST_SHA256
        or not isinstance(oracle_receipts, (tuple, list))
        or len(oracle_receipts) != 16
    ):
        raise ValueError("rank64/X4 V2 manifest or oracle receipts differ")
    metrics = _mapping(comparison.get("metrics"), label="baseline metrics")
    rank64_metrics = metrics.get("global_svd_rank64")
    exact_x4_metrics = metrics.get("exact_x4_carrier")
    if (
        _canonical_json_bytes(_variant_metrics(evaluation))
        != _canonical_json_bytes(rank64_metrics)
    ):
        raise ValueError("rank64/X4 V2 live rank64 metrics differ")
    exact_receipts: dict[str, object] = {}
    for raw_receipt in oracle_receipts:
        receipt = _mapping(raw_receipt, label="oracle receipt")
        example_id = receipt.get("example_id")
        exact = _mapping(
            receipt.get("exact_x4_carrier"),
            label="exact-X4 receipt",
        )
        if (
            not isinstance(example_id, str)
            or example_id in exact_receipts
            or exact.get("role") != "exact_x4_carrier"
            or exact.get("runtime_binding_sha256")
            != _EXPECTED_RANK64_RUNTIME_SHA256
            or exact.get("adapter_execution_sha256")
            != _EXPECTED_FACTORIZED_EXECUTION_SHA256
            or exact.get("metrics_only") is not True
            or exact.get("serving_authorized") is not False
        ):
            raise ValueError("rank64/X4 exact replay receipt differs")
        exact_receipts[example_id] = {
            "example_id": example_id,
            "family_id": receipt["family_id"],
            "prompt_sha256": receipt["prompt_sha256"],
            "model_inputs_sha256": receipt["model_inputs_sha256"],
            "execution_grid_sha256": receipt["execution_grid_sha256"],
            "shadow_result_artifact_sha256": receipt[
                "shadow_result_artifact_sha256"
            ],
            "injected_x4_sha256": exact["injected_x4_sha256"],
            "logits_sha256": exact["logits_sha256"],
            "oracle_artifact_sha256": exact["artifact_sha256"],
        }
    return {
        "file": str(source),
        "file_sha256": _EXPECTED_BASELINE_FILE_SHA256,
        "report_sha256": _EXPECTED_BASELINE_REPORT_SHA256,
        "panel": dict(panel),
        "manifest_sha256": _EXPECTED_MANIFEST_SHA256,
        "rank64_plan_artifact_sha256": _EXPECTED_RANK64_PLAN_SHA256,
        "rank64_plan": dict(plan),
        "rank64_arm_artifact_sha256": _EXPECTED_RANK64_ARM_SHA256,
        "rank64_arm_receipt": dict(arm),
        "runtime_binding_sha256": _EXPECTED_RANK64_RUNTIME_SHA256,
        "runtime_binding": dict(runtime),
        "source_execution_summary_receipt": dict(source_receipt),
        "source_execution_summary_receipt_sha256": source_receipt_sha256,
        "rank64_metrics": rank64_metrics,
        "exact_x4_metrics": exact_x4_metrics,
        "exact_x4_receipts": exact_receipts,
    }


def _validate_partial_replay_receipts(
    baseline: Mapping[str, object],
    receipts: object,
) -> tuple[dict[str, object], ...]:
    if not isinstance(receipts, (tuple, list)) or len(receipts) != 16:
        raise ValueError("complete-H4 audit requires exactly 16 receipts")
    expected = _mapping(
        baseline.get("exact_x4_receipts"),
        label="baseline exact-X4 receipts",
    )
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in receipts:
        receipt = _mapping(raw, label="complete-H4 prompt receipt")
        expected_receipt_fields = (
            "example_id",
            "family_id",
            "prompt_sha256",
            "model_inputs_sha256",
            "execution_grid_sha256",
            "shadow_result_artifact_sha256",
            "audit",
            "complete_h4_audit_receipt_sha256",
        )
        if tuple(receipt) != expected_receipt_fields:
            raise ValueError("complete-H4 prompt receipt ABI differs")
        example_id = receipt.get("example_id")
        audit = _mapping(
            receipt.get("audit"),
            label="complete-H4 audit metadata",
        )
        if (
            not isinstance(example_id, str)
            or example_id in seen
            or example_id not in expected
        ):
            raise ValueError("complete-H4 prompt identity differs")
        frozen = _mapping(expected[example_id], label="frozen exact-X4 receipt")
        for name in (
            "example_id",
            "family_id",
            "prompt_sha256",
            "model_inputs_sha256",
            "execution_grid_sha256",
            "shadow_result_artifact_sha256",
        ):
            expected_value = example_id if name == "example_id" else frozen[name]
            if receipt.get(name) != expected_value:
                raise ValueError(f"complete-H4 receipt {name} differs from V2")
        expected_audit_fields = (
            "execution_mode",
            "metrics_only",
            "serving_authorized",
            "model_forward_count",
            "native_h4_sha256",
            "incomplete_carrier_h4_sha256",
            "injected_h4_sha256",
            "shadow_result_artifact_sha256",
            "runtime_binding_sha256",
            "model_inputs_sha256",
            "execution_grid_sha256",
            "adapter_execution_sha256",
            "target_affected_rows",
            "incomplete_h4_difference_mask_sha256",
            "incomplete_h4_difference_rows",
            "incomplete_h4_difference_valid_rows",
            "incomplete_h4_difference_padding_rows",
            "incomplete_h4_difference_target_rows",
            "incomplete_h4_difference_outside_target_rows",
            "target_affected_h4_difference_observed",
            "incomplete_h4_difference_nonvacuous",
            "boundary_callbacks_exactly_once",
            "boundary_callback_order",
            "complete_h4_logits_bitwise_authoritative",
            "complete_h4_max_abs_logit_error",
            "partial_exact_x4_logits_sha256",
            "complete_h4_logits_sha256",
            "artifact_sha256",
        )
        if tuple(audit) != expected_audit_fields:
            raise ValueError("complete-H4 audit metadata ABI differs")
        identity = audit.get("complete_h4_logits_bitwise_authoritative")
        affected_rows = audit.get("target_affected_rows")
        if type(identity) is not bool:
            raise TypeError("complete-H4 identity outcome must be boolean")
        if type(affected_rows) is not int or affected_rows <= 0:
            raise ValueError("complete-H4 target affected rows must be positive")
        complete_error = _finite(
            audit.get("complete_h4_max_abs_logit_error"),
            label="complete H4 logits max abs error",
        )
        if complete_error < 0.0:
            raise ValueError("complete H4 logits max abs error is negative")
        hashes = (
            "native_h4_sha256",
            "incomplete_carrier_h4_sha256",
            "injected_h4_sha256",
            "shadow_result_artifact_sha256",
            "runtime_binding_sha256",
            "model_inputs_sha256",
            "execution_grid_sha256",
            "adapter_execution_sha256",
            "incomplete_h4_difference_mask_sha256",
            "partial_exact_x4_logits_sha256",
            "complete_h4_logits_sha256",
            "artifact_sha256",
        )
        if any(
            not isinstance(audit.get(name), str)
            or len(str(audit[name])) != 64
            or any(char not in "0123456789abcdef" for char in str(audit[name]))
            for name in hashes
        ):
            raise ValueError("complete-H4 audit hash metadata differs")
        count_names = (
            "incomplete_h4_difference_rows",
            "incomplete_h4_difference_valid_rows",
            "incomplete_h4_difference_padding_rows",
            "incomplete_h4_difference_target_rows",
            "incomplete_h4_difference_outside_target_rows",
        )
        counts = {name: audit.get(name) for name in count_names}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("incomplete-H4 difference counts must be integers")
        difference_rows = counts["incomplete_h4_difference_rows"]
        valid_rows = counts["incomplete_h4_difference_valid_rows"]
        padding_rows = counts["incomplete_h4_difference_padding_rows"]
        target_rows = counts["incomplete_h4_difference_target_rows"]
        outside_rows = counts[
            "incomplete_h4_difference_outside_target_rows"
        ]
        if (
            difference_rows != valid_rows + padding_rows
            or difference_rows != target_rows + outside_rows
            or difference_rows <= 0
            or target_rows <= 0
        ):
            raise ValueError("incomplete-H4 difference count partition differs")
        if (
            audit.get("execution_mode")
            != "authenticated_complete_h4_identity_audit"
            or audit.get("metrics_only") is not True
            or audit.get("serving_authorized") is not False
            or audit.get("model_forward_count") != 3
            or audit.get("runtime_binding_sha256")
            != _EXPECTED_RANK64_RUNTIME_SHA256
            or audit.get("adapter_execution_sha256")
            != _EXPECTED_FACTORIZED_EXECUTION_SHA256
            or audit.get("shadow_result_artifact_sha256")
            != receipt["shadow_result_artifact_sha256"]
            or audit.get("model_inputs_sha256")
            != receipt["model_inputs_sha256"]
            or audit.get("execution_grid_sha256")
            != receipt["execution_grid_sha256"]
            or audit.get("native_h4_sha256")
            != audit.get("injected_h4_sha256")
            or audit.get("native_h4_sha256")
            == audit.get("incomplete_carrier_h4_sha256")
            or audit.get("boundary_callbacks_exactly_once") is not True
            or tuple(audit.get("boundary_callback_order", ()))
            != (
                "partial_exact_x4.y3",
                "partial_exact_x4.x4",
                "complete_h4.y3",
                "complete_h4.x4",
                "complete_h4.h4",
            )
            or audit.get("target_affected_h4_difference_observed") is not True
            or audit.get("incomplete_h4_difference_nonvacuous") is not True
            or audit.get("partial_exact_x4_logits_sha256")
            != frozen["logits_sha256"]
        ):
            raise ValueError("partial exact-X4 replay or H4 audit receipt differs")
        receipt_payload = {
            name: receipt[name]
            for name in expected_receipt_fields
            if name != "complete_h4_audit_receipt_sha256"
        }
        if receipt.get("complete_h4_audit_receipt_sha256") != (
            _complete_h4_audit_receipt_sha256(receipt_payload)
        ):
            raise ValueError("complete-H4 prompt receipt hash differs")
        normalized.append(
            {
                "example_id": example_id,
                "family_id": receipt["family_id"],
                "prompt_sha256": receipt["prompt_sha256"],
                "model_inputs_sha256": receipt["model_inputs_sha256"],
                "execution_grid_sha256": receipt["execution_grid_sha256"],
                "shadow_result_artifact_sha256": receipt[
                    "shadow_result_artifact_sha256"
                ],
                "audit": dict(audit),
                "complete_h4_max_abs_logit_error": complete_error,
                "complete_h4_audit_receipt_sha256": receipt.get(
                    "complete_h4_audit_receipt_sha256"
                ),
            }
        )
        seen.add(example_id)
    if seen != set(expected):
        raise ValueError("complete-H4 receipt membership differs from V2")
    return tuple(sorted(normalized, key=lambda row: str(row["example_id"])))


def compare_complete_h4_identity_audit(
    baseline: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the replay, then classify exact H4 identity as EF."""

    if not isinstance(baseline, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("baseline and evaluation must be mappings")
    rank64_metrics = _variant_metrics(evaluation)
    if _canonical_json_bytes(rank64_metrics) != _canonical_json_bytes(
        baseline.get("rank64_metrics")
    ):
        raise ValueError("live rank64 metrics differ from corrected V2")
    source_receipt = _source_execution_summary_receipt(evaluation)
    if (
        _canonical_json_bytes(source_receipt)
        != _canonical_json_bytes(
            baseline.get("source_execution_summary_receipt")
        )
        or _json_sha256(source_receipt, domain=_SOURCE_RECEIPT_DOMAIN)
        != _EXPECTED_SOURCE_RECEIPT_SHA256
    ):
        raise ValueError("live source execution differs from corrected V2")
    audit = _mapping(
        evaluation.get("complete_h4_identity_audit"),
        label="complete_h4_identity_audit",
    )
    expected_fields = (
        "semantics",
        "partial_exact_x4_replay",
        "complete_h4_identity",
        "execution",
        "receipts",
    )
    if tuple(audit) != expected_fields:
        raise ValueError("complete-H4 audit report ABI differs")
    semantics = _mapping(audit.get("semantics"), label="audit semantics")
    execution = _mapping(audit.get("execution"), label="audit execution")
    expected_semantics_fields = (
        "execution_order",
        "truth_leaking_identity_control",
        "source_outputs_authoritative",
        "audit_outputs_must_not_be_served",
        "complete_boundary",
        "graph_target_affected_mask_semantics",
        "observed_h4_difference_mask_semantics",
        "graph_target_support_is_distinct_from_observed_h4_"
        "difference_support",
        "outside_graph_target_difference_is_not_integrity_failure",
    )
    if (
        tuple(semantics) != expected_semantics_fields
        or tuple(semantics.get("execution_order", ()))
        != (
            "native_h4_replay",
            "partial_exact_x4_replay",
            "complete_h4_identity",
        )
        or semantics.get("source_outputs_authoritative") is not True
        or semantics.get("audit_outputs_must_not_be_served") is not True
        or semantics.get("truth_leaking_identity_control") is not True
        or semantics.get("complete_boundary") != "layer.4.output"
        or semantics.get("graph_target_affected_mask_semantics")
        != "finite_lag_prediction_support"
        or semantics.get("observed_h4_difference_mask_semantics")
        != "bitwise_full_row_native_vs_incomplete_carrier_support"
        or semantics.get(
            "graph_target_support_is_distinct_from_observed_h4_"
            "difference_support"
        )
        is not True
        or semantics.get("outside_graph_target_difference_is_not_integrity_failure")
        is not True
        or execution.get("audit_forwards_per_prompt") != 3
        or execution.get("total_audit_model_forward_count") != 48
        or execution.get("total_fused_model_forward_count") != 96
    ):
        raise ValueError("complete-H4 execution or safety semantics differ")
    partial = _mapping(
        audit.get("partial_exact_x4_replay"),
        label="partial exact-X4 replay",
    )
    complete = _mapping(
        audit.get("complete_h4_identity"),
        label="complete-H4 identity",
    )
    partial_metrics = _behavior_metrics(partial, label="partial exact-X4")
    complete_metrics = _behavior_metrics(complete, label="complete H4")
    if (
        partial.get("role")
        != "exact_native_x4_on_incomplete_clamped_y3_carrier"
        or complete.get("role")
        != "exact_native_h4_at_complete_layer4_output"
    ):
        raise ValueError("complete-H4 audit arm role differs")
    if _canonical_json_bytes(partial_metrics) != _canonical_json_bytes(
        baseline.get("exact_x4_metrics")
    ):
        raise ValueError("partial exact-X4 replay metrics differ from V2")
    candidate_source = _source_behavior_receipt(
        evaluation.get("behavioral"),
        label="rank64 behavioral",
    )
    candidate_affected_source = _source_behavior_receipt(
        evaluation.get("affected_behavioral"),
        label="rank64 affected behavioral",
    )
    for label, row in (("partial", partial), ("complete", complete)):
        if (
            _canonical_json_bytes(
                _source_behavior_receipt(
                    row.get("behavioral"),
                    label=f"{label}.behavioral",
                )
            )
            != _canonical_json_bytes(candidate_source)
            or _canonical_json_bytes(
                _source_behavior_receipt(
                    row.get("affected_behavioral"),
                    label=f"{label}.affected_behavioral",
                )
            )
            != _canonical_json_bytes(candidate_affected_source)
        ):
            raise ValueError(f"{label} source behavioral summary differs")
    receipts = _validate_partial_replay_receipts(
        baseline,
        audit.get("receipts"),
    )
    receipt_exact_identity = all(
        _mapping(receipt["audit"], label="complete H4 receipt").get(
            "complete_h4_logits_bitwise_authoritative"
        )
        is True
        and receipt["complete_h4_max_abs_logit_error"] == 0.0
        for receipt in receipts
    )
    aggregate_identity = complete.get(
        "complete_h4_logits_bitwise_authoritative"
    )
    if type(aggregate_identity) is not bool:
        raise TypeError("aggregate complete-H4 identity must be boolean")
    aggregate_error = _finite(
        complete.get("complete_h4_max_abs_logit_error"),
        label="aggregate complete-H4 max abs logit error",
    )
    if aggregate_error < 0.0:
        raise ValueError("aggregate complete-H4 max abs logit error is negative")
    receipt_max_error = max(
        float(receipt["complete_h4_max_abs_logit_error"])
        for receipt in receipts
    )
    if (
        aggregate_identity is not receipt_exact_identity
        or aggregate_error != receipt_max_error
    ):
        raise ValueError("aggregate complete-H4 identity outcome differs")
    exact_identity = aggregate_identity and aggregate_error == 0.0
    support_summary = {
        "prompt_count": len(receipts),
        "incomplete_h4_difference_rows": sum(
            int(receipt["audit"]["incomplete_h4_difference_rows"])
            for receipt in receipts
        ),
        "incomplete_h4_difference_valid_rows": sum(
            int(receipt["audit"]["incomplete_h4_difference_valid_rows"])
            for receipt in receipts
        ),
        "incomplete_h4_difference_padding_rows": sum(
            int(receipt["audit"]["incomplete_h4_difference_padding_rows"])
            for receipt in receipts
        ),
        "incomplete_h4_difference_target_rows": sum(
            int(receipt["audit"]["incomplete_h4_difference_target_rows"])
            for receipt in receipts
        ),
        "incomplete_h4_difference_outside_target_rows": sum(
            int(
                receipt["audit"][
                    "incomplete_h4_difference_outside_target_rows"
                ]
            )
            for receipt in receipts
        ),
        "prompts_with_outside_target_h4_difference": sum(
            int(
                receipt["audit"][
                    "incomplete_h4_difference_outside_target_rows"
                ]
                > 0
            )
            for receipt in receipts
        ),
        "graph_target_support_distinct_from_observed_h4_difference_support": (
            True
        ),
        "outside_target_difference_is_descriptive_not_a_failure": True,
    }
    fidelity = _passed(complete_metrics)
    pattern = f"{int(exact_identity)}{int(fidelity)}"
    classifications = {
        "11": "complete_h4_identity_validated",
        "01": "fidelity_without_exact_identity_insufficient",
        "10": "exact_identity_fidelity_reducer_mismatch",
        "00": "complete_h4_identity_failed",
    }
    return {
        "classifier_axes": ("exact_full_logit_identity", "frozen_fidelity"),
        "pass_pattern": pattern,
        "pass_pattern_semantics": "exact_identity_fidelity",
        "arm_passes": {
            "exact_full_logit_identity": exact_identity,
            "frozen_fidelity": fidelity,
        },
        "classification_protocol": {
            "exact_identity_requires_all_16_full_logit_tensors_bitwise_equal": (
                True
            ),
            "exact_identity_requires_zero_max_abs_error": True,
            "fidelity_requires_ordinary_and_affected_gates": True,
            "exact_four_way_mapping": classifications,
            "no_posthoc_threshold_used": True,
        },
        "classification": classifications[pattern],
        "rank64_replay_metrics": rank64_metrics,
        "partial_exact_x4_replay_metrics": partial_metrics,
        "complete_h4_identity_metrics": complete_metrics,
        "observed_h4_difference_support": support_summary,
        "source_execution_summary_receipt": source_receipt,
        "source_execution_summary_receipt_sha256": (
            _EXPECTED_SOURCE_RECEIPT_SHA256
        ),
        "prompt_receipts": receipts,
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("complete-H4 output must be JSON under .local-runs")
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


def run_gemma3_l3_l4_complete_h4_identity_audit(
    *,
    fit_source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    panel_path: Path | str = DEFAULT_PANEL,
    rank64_x4_baseline_path: Path | str = DEFAULT_RANK64_X4_BASELINE,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Run the six-pass A-only complete-H4 identity audit."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite complete-H4 audit report")
    if type(max_length) is not int or max_length != DEFAULT_MAX_LENGTH:
        raise ValueError(
            "max_length must equal the corrected V2 baseline length of "
            f"{DEFAULT_MAX_LENGTH}"
        )
    baseline = _load_rank64_x4_baseline(rank64_x4_baseline_path)
    examples, panel_receipt = _load_panel(panel_path)
    if _canonical_json_bytes(panel_receipt) != _canonical_json_bytes(
        baseline["panel"]
    ):
        raise ValueError("live A-fit panel differs from corrected V2")
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
    basis = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plan, plan_receipt = build_rank64_global_svd_plan(fit_source, parent)
    if _canonical_json_bytes(plan_receipt) != _canonical_json_bytes(
        baseline["rank64_plan"]
    ):
        raise ValueError("rebuilt rank64 plan differs from corrected V2")
    arm_receipt = _mapping(
        baseline["rank64_arm_receipt"],
        label="rank64 arm receipt",
    )
    common_binding = _mapping(
        arm_receipt.get("common_binding"),
        label="rank64 common binding",
    )
    if (
        arm_receipt.get("artifact_sha256") != _EXPECTED_RANK64_ARM_SHA256
        or common_binding.get("signed_g8_candidate_artifact_sha256")
        != candidate.artifact_sha256
        or common_binding.get("fit_response_tensor_file_sha256")
        != fit_source.file_sha256
        or common_binding.get("parent_graph_wavelet_artifact_sha256")
        != parent.artifact_sha256
        or common_binding.get("basis_package_payload_sha256")
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or common_binding.get("panel_file_sha256")
        != panel_receipt["file_sha256"]
        or common_binding.get("panel_source_fit_prompt_index_sha256")
        != panel_receipt["source_fit_prompt_index_sha256"]
        or common_binding.get("max_length") != max_length
    ):
        raise ValueError("live rank64 arm inputs differ from corrected V2")
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    tokenizer, tokenizer_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=protocol
    )
    if (
        tokenizer_contract.get("tokenizer_class")
        != common_binding.get("tokenizer_class")
        or tokenizer_contract.get("configuration_sha256")
        != _EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        or tokenizer_contract.get("backend_serialized_sha256")
        != _EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
    ):
        raise ValueError("live tokenizer differs from corrected V2")
    tokenizer_integrity_check = _frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )
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
        raise ValueError("live raw Gemma differs from corrected V2")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256, factorized_execution_sha256 = (
            _live_factorized_identity(adapter)
        )
        runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis,
            candidate_artifact_sha256=_EXPECTED_RANK64_ARM_SHA256,
            candidate_method="global_svd_rank64_capacity_oracle",
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=_EXPECTED_RANK64_PLAN_SHA256,
            expected_basis_payload_sha256=(
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            expected_live_model_sha256=factorized_model_sha256,
            expected_adapter_execution_sha256=factorized_execution_sha256,
            analysis_device="cpu",
        )
        runtime_metadata = runtime.metadata()
        if (
            runtime_metadata.get("runtime_binding_sha256")
            != _EXPECTED_RANK64_RUNTIME_SHA256
            or _canonical_json_bytes(runtime_metadata)
            != _canonical_json_bytes(baseline["runtime_binding"])
        ):
            raise ValueError("live rank64 runtime differs from corrected V2")
        evaluation = (
            evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
                runtime=runtime,
                adapter=adapter,
                tokenizer=tokenizer,
                examples=examples,
                max_length=max_length,
                model_input_device=device,
                tokenizer_integrity_check=tokenizer_integrity_check,
                include_oracle_suffixes=False,
                include_complete_h4_identity_audit=True,
            )
        )
        runtime.validate_integrity()
        _live_factorized_identity(adapter)
        tokenizer_integrity_check("after")
    finally:
        switcher.close()
    if adapter.model_fingerprint() != _EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("complete-H4 audit did not restore raw Gemma")
    execution = _mapping(evaluation.get("execution"), label="execution")
    if (
        execution.get("model_forwards_per_prompt") != 6
        or execution.get("total_model_forward_count") != 96
    ):
        raise ValueError("complete-H4 audit must execute exactly 96 forwards")
    comparison = compare_complete_h4_identity_audit(baseline, evaluation)
    identity_pass = _mapping(
        comparison.get("arm_passes"),
        label="complete-H4 arm passes",
    )["exact_full_logit_identity"]
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": "reused_calibration_a_fit_complete_h4_identity_audit",
        "lineage": {
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "basis_package_payload_sha256": (
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            "fit_response_tensor_file_sha256": fit_source.file_sha256,
            "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
            "rank64_x4_baseline_file_sha256": (
                _EXPECTED_BASELINE_FILE_SHA256
            ),
            "rank64_x4_baseline_report_sha256": (
                _EXPECTED_BASELINE_REPORT_SHA256
            ),
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
            "execution_order": (
                "source_reference_candidate_shadow",
                "native_h4_replay",
                "partial_exact_x4_replay",
                "complete_h4_identity",
            ),
            "same_live_model_for_all_six_forwards": True,
            "same_live_tokenizer_for_all_six_forwards": True,
            "rank64_x4_baseline_replayed_live": True,
            "baseline_authenticated_by_file_and_payload_hash": True,
            "rank64_metrics_must_match_baseline_exactly": True,
            "source_execution_must_match_baseline_exactly": True,
            "partial_exact_x4_metrics_must_match_baseline_exactly": True,
            "partial_exact_x4_logits_hashes_must_match_baseline_exactly": (
                True
            ),
            "graph_target_support_is_distinct_from_observed_h4_"
            "difference_support": True,
            "outside_graph_target_difference_is_not_integrity_failure": True,
            "source_path_authoritative": True,
            "audit_outputs_metrics_only": True,
            "max_length": max_length,
            "model_forwards_per_prompt": 6,
            "expected_total_model_forward_count": 96,
            "tokenizer_integrity_checked_before_and_after_each_prompt": True,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "rank64_x4_baseline": {
            "file": baseline["file"],
            "file_sha256": baseline["file_sha256"],
            "report_sha256": baseline["report_sha256"],
            "manifest_sha256": baseline["manifest_sha256"],
            "rank64_plan_artifact_sha256": baseline[
                "rank64_plan_artifact_sha256"
            ],
            "rank64_arm_artifact_sha256": baseline[
                "rank64_arm_artifact_sha256"
            ],
            "runtime_binding_sha256": baseline[
                "runtime_binding_sha256"
            ],
            "source_execution_summary_receipt_sha256": baseline[
                "source_execution_summary_receipt_sha256"
            ],
        },
        "rank64_plan": plan_receipt,
        "rank64_arm_receipt": dict(arm_receipt),
        "runtime_binding": runtime_metadata,
        "evaluation": evaluation,
        "comparison": comparison,
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "shadow_model_forward_count": 48,
            "native_h4_replay_model_forward_count": 16,
            "partial_exact_x4_replay_model_forward_count": 16,
            "complete_h4_identity_model_forward_count": 16,
            "total_model_forward_count": 96,
            "rank64_is_capacity_oracle_not_compression": True,
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "development_complete_h4_audit_execution_complete": True,
            "exact_complete_h4_identity_validated": identity_pass,
            "classification": comparison["classification"],
            "reused_calibration_a_fit_only": True,
            "corrected_v2_rank64_replay_matched": True,
            "corrected_v2_partial_exact_x4_replay_matched": True,
            "formal_qualification": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "artifact": {"file": str(destination), "committable": False},
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the signed-g8 A-only complete-H4 identity audit",
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
    parser.add_argument(
        "--rank64-x4-baseline",
        default=DEFAULT_RANK64_X4_BASELINE,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_identity_audit(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        rank64_x4_baseline_path=arguments.rank64_x4_baseline,
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
                "arm_passes": report["comparison"][  # type: ignore[index]
                    "arm_passes"
                ],
                "pass_pattern": report["comparison"][  # type: ignore[index]
                    "pass_pattern"
                ],
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
