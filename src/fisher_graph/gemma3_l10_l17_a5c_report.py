"""Strict, tensor-free report contract for the A5c broader-row rung.

This module deliberately contains no model loading, activation capture,
fitting, or executor construction.  A runner supplies compact authenticated
receipt references, executable hashes, and scalar evaluation results.  The
contract binds those values into a distinct A5c artifact without changing or
reinterpreting the canonical A5b v1 report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from torch import Tensor

from .gemma3_a5_same_shape_locality import (
    validate_same_shape_off_row_locality_receipt,
)
from .gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION,
    GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA,
    _RECEIPT_DOMAIN as _COORDINATE_ROW_BANK_RECEIPT_DOMAIN,
)
from .gemma3_l10_l17_a5b_generator_microcanary import (
    _TARGET_RECEIPT_DOMAIN,
    _validate_error_summary as _validate_target_error_summary,
    _validate_summary as _validate_target_summary,
)
from .gemma3_l10_l17_a5c_breadth_split import (
    validate_a5c_breadth_split_receipt,
)
from .gemma3_l10_l17_a5c_family_ridge_cv import (
    A5C_RIDGE_GRID,
    validate_a5c_family_ridge_cv_receipt,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _MINIMUM_NUMERICAL_KL_PER_TOKEN,
    _validate_fold_evaluation as _validate_source_fold_evaluation,
)


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT",
    "GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5C_REPORT_SCHEMA",
    "a5c_outer_evaluation_sha256",
    "a5c_resource_accounting_sha256",
    "a5c_selection_freeze_sha256",
    "a5c_source_scorer_evaluation_sha256",
    "a5c_source_scorer_receipt_sha256",
    "a5c_target_token_locality_lineage_sha256",
    "build_gemma3_l10_l17_a5c_report",
    "derive_gemma3_l10_l17_a5c_conclusion",
    "load_gemma3_l10_l17_a5c_report",
    "save_gemma3_l10_l17_a5c_report",
    "validate_gemma3_l10_l17_a5c_report",
]


GEMMA3_L10_L17_A5C_REPORT_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5c_broader_selected_generator"
)
GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a5c-broader-selected-generator-v1.json"
)

_EXPECTED_MODEL_ID = "google/gemma-3-270m"
_EXPECTED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_EXPECTED_MODEL_FINGERPRINT = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_EXPECTED_A5B_FILE_SHA256 = (
    "c0ad54d30ac192c7ac8002aa8f9cf0fe78ba83edfed9c74502bc0b50ca1dd023"
)
_EXPECTED_A5B_REPORT_SHA256 = (
    "27397857c1707c774b5d78ead1ca0770e6a3d5eaa43a0b59e502c6b190bb2cd5"
)
_EXPECTED_A5B_SOURCE_BINDINGS = {
    "a5a_file_sha256": (
        "bace02fa1a290a5a076c6c6a723f9590513fa0290cccf5a7fd8b3a2117584390"
    ),
    "a5a_report_sha256": (
        "c46d7c587962c64fca48da5fb54525fc1630d04e14cf1b87830457732224948e"
    ),
    "a4_oracle_file_sha256": (
        "9669aca95cf81eb33e8c0ac941e31279e8f8e484fb0352f6a04e868cf1bc72a6"
    ),
    "a4_oracle_report_sha256": (
        "f38b55bcc65d76d6eba1daeeea2e04dbd57401e58d33659163cea2731d5546eb"
    ),
    "a4_report_file_sha256": (
        "78222a62eee08bc58a92aa70613018de2f5870e3df7ca24e5a187b60956ed80d"
    ),
    "a4_report_sha256": (
        "db0e5d938c9f71f457a8de5b535c659abd6a85a509b2a3aa4261c79a9de6f702"
    ),
    "composition_bundle_file_sha256": (
        "394906f8e84a50e18922de0dc8c114be1ea9889f0995ccca180b9f6a8d303d8d"
    ),
    "composition_payload_sha256": (
        "2f7c2179656fc16c614cd84b7a0b29d3250443a5d8c80db221b220e3d3f082bf"
    ),
    "fold_bundle_file_sha256": (
        "cd4d0621e5b4fce44430d5bfc2c680fd29373e53788c01437f61580829bda162"
    ),
    "fold_bundle_payload_sha256": (
        "6d0dd667ccf9a34c15fdd3deda35795d4ada56344e3e6408134262da687958d2"
    ),
    "protocol_sha256": (
        "9adefa7d75d11343d8ab103ac7c683aaea269f65b894b11039eaa508c08fa3dc"
    ),
    "source_runtime_catalog_sha256": (
        "84b80b3cbabc3b8ff8bcf9f63e1f97a620fb38e2f6940b196f30906d9dfcb1b7"
    ),
}
_EXPECTED_SOURCE_BINDINGS = {
    "a5b_file_sha256": _EXPECTED_A5B_FILE_SHA256,
    "a5b_report_sha256": _EXPECTED_A5B_REPORT_SHA256,
    **_EXPECTED_A5B_SOURCE_BINDINGS,
}
_EXPECTED_A5B_LEARNED_COMPOSITION = {
    "nll_per_token": 11.708115005493164,
    "delta_nll_per_token": 4.550132043021066,
    "native_to_candidate_kl_per_token": 3.4907681586316577,
    "top1_agreement_to_native": 0.4857142857142857,
}
_EXPECTED_LAYER10_GRAPH_SHA256 = (
    "67327f1ba3cff3bd9a49897245d0301d109ac1564eff2c4f70409d29a28a8b94"
)
_EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE = {
    "gemma3.layer-10.cluster-28.modal-generator.same-layer-0.graph-node": (
        "e07124c27cb4d61e5450c109a3a56d802da490ad04ca3056553ba34db822bc47"
    ),
    "gemma3.layer-10.cluster-0.modal-generator.same-layer-1.graph-node": (
        "69c7dbb56ae0700eb2796ffe85ee9096658afb365742a5ae72fc54e024efa473"
    ),
    "gemma3.layer-10.cluster-34.modal-generator.same-layer-2.graph-node": (
        "c2339759064eab6d6bbc0b79d22a76ba4d6aafc292d179585faa9aa8b4acf1f9"
    ),
    "gemma3.layer-10.cluster-63.modal-generator.same-layer-3.graph-node": (
        "e23c892eca14ff5493e2f19c9e4960016a133677db89f2daccb13b7d087ba3f4"
    ),
}
_EXPECTED_LAYER17_GRAPH_SHA256 = (
    "4b81283db0df73b3be06d67ed61be4733190824687b0d34a5d9b3662a26d1607"
)
_EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE = {
    "gemma3.layer-17.cluster-0.modal-generator.same-layer-0.graph-node": (
        "d69066c28e144b482787f2f66d7934e5a42b45b06dd80c66182c68df2eec5221"
    ),
    "gemma3.layer-17.cluster-28.modal-generator.same-layer-1.graph-node": (
        "27752e63c5b796c0c903649d1b5ca5cf7326cb11971f0c25e490f7c97bd53ff5"
    ),
    "gemma3.layer-17.cluster-34.modal-generator.same-layer-2.graph-node": (
        "501bf4f2faf924302db7322b56a91948bcfb158528e3657c718292984ccf7564"
    ),
    "gemma3.layer-17.cluster-54.modal-generator.same-layer-3.graph-node": (
        "60a081e187a2dc50bea9f60f3c5fdbd3eb83e5dacf9a9d5cac16fc8292fb0fd3"
    ),
}
_EXPECTED_PRIMARY_COMPOSITION_GRAPH_SHA256 = (
    "35d35f2318e0728bb649f2825601d2edbe06e13307a412c4d81c66e8e387c4ca"
)
_A5C_TOKEN_LOCALITY_POLICY = "same_shape_directed_off_row_v2"
_A5C_TOKEN_LOCALITY_METHOD = "same_shape_directed_off_row_native_and_a4"
_A5C_TOKEN_LOCALITY_ATOL = 1.0e-6
_A5C_TOKEN_LOCALITY_RTOL = 2.0e-6
_A5C_TOKEN_LOCALITY_PROBES = {
    "teacher": "native_teacher",
    "a4_baseline": "a4_euclidean_baseline",
}

_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-report:v1\0"
_FREEZE_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-freeze:v1\0"
_EVALUATION_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-evaluation:v1\0"
_SOURCE_SCORER_EVALUATION_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-source-scorer-evaluation:v1\0"
)
_SOURCE_SCORER_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-source-scorer-receipt:v1\0"
)
_RESOURCE_ACCOUNTING_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-resource-accounting:v1\0"
)
_TARGET_TOKEN_LOCALITY_LINEAGE_DOMAIN = (
    b"fisher-graph:a5c-target-token-locality-lineage:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_SCHEMA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_LEARNED = "learned_correction"
_FROZEN = "frozen_source_fallback"
_EXECUTABLE_KINDS = frozenset({_LEARNED, _FROZEN})

_SCIENTIFIC_ROLE = (
    "calibration_a_one_outer_fold_broader_row_nested_selected_generator"
)
_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_coordinate_tensors": False,
    "contains_source_model_weights": False,
    "contains_generator_weights": False,
    "source_safe": True,
}

_SOURCE_FIELDS = set(_EXPECTED_SOURCE_BINDINGS)
_RUNTIME_FIELDS = {
    "model_id",
    "requested_revision",
    "model_fingerprint",
    "device",
    "dtype",
    "local_files_only",
}
_CONFIG_FIELDS = {
    "outer_fold_index",
    "training_family_count",
    "training_examples_per_family",
    "row_selection_policy",
    "inner_audit_examples_per_family",
    "target_solver_steps",
    "target_batch_rows",
    "target_learning_rate_fraction",
    "target_ridge",
    "target_trust_radius",
    "generator_rank",
    "ridge_grid",
    "held_examples_scored",
}
_CAPTURE_FIELDS = {
    "capture_sha256",
    "capture_audit_sha256",
    "source_row_catalog_sha256",
    "training_family_count",
    "training_example_count",
    "captured_observation_count",
    "selected_target_row_count",
    "outer_held_family_rows_present",
    "all_required_capture_audits_pass",
}
_TARGET_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "row_count",
    "selected_coordinate_sha256",
    "contains_tensor_payloads",
}
_ROW_BANK_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "row_count",
    "example_count",
    "row_key_sha256",
    "compiled_inputs_sha256",
    "selected_coordinates_sha256",
    "outer_held_family_rows_present",
    "contains_tensor_payloads",
}
_BREADTH_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "all_row_count",
    "fit_row_count",
    "audit_row_count",
    "all_example_count",
    "fit_example_count",
    "audit_example_count",
    "fit_examples_fully_removed_for_signature_overlap",
    "removed_fit_rows_for_signature_overlap",
    "fit_audit_example_overlap_count",
    "post_purge_input_signature_overlap_count",
    "outer_held_family_rows_present",
    "contains_tensor_payloads",
}
_CV_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "candidate_count",
    "inner_fold_count",
    "selected_ridge",
    "use_frozen_fallback",
    "outer_held_family_accessed",
    "contains_tensor_payloads",
}
_DESCRIPTOR_FIELDS = {
    "layer17_graph_sha256",
    "layer17_lowering_sha256_by_node",
    "composition_graph_sha256",
    "layer17_parameter_count",
    "layer17_macs_per_token",
    "composition_parameter_count",
    "composition_macs_per_token",
    "layer17_post_feedforward_delta_layer_ordinals",
    "composition_post_feedforward_delta_layer_ordinals",
}
_LINEAGE_FIELDS = {
    "target_solve_receipt_sha256",
    "coordinate_row_bank_receipt_sha256",
    "breadth_split_receipt_sha256",
    "ridge_cv_receipt_sha256",
    "layer10_graph_sha256",
    "layer10_lowering_sha256_by_node",
    "matched_double_deletion_graph_sha256",
}
_SELECTED_FIELDS = {
    "kind",
    "selected_ridge",
    "lineage",
    "selected",
    "frozen_reference",
    "selection_freeze_sha256",
}
_CHRONOLOGY_FIELDS = {
    "ridge_cv_completed_event",
    "executable_frozen_event",
    "outer_held_batch_selected_event",
    "outer_held_model_evaluated_event",
    "outer_held_batch_selected_or_scored_before_freeze",
    "executable_frozen_before_outer_held_batch_selection",
    "executable_frozen_before_outer_held_model_evaluation",
    "ridge_cv_receipt_sha256",
    "selection_freeze_sha256",
    "outer_evaluation_sha256",
}
_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_CONDITION_FIELDS = {*_METRIC_FIELDS, "graph_sha256"}
_CONDITIONS = {
    "layer10_only",
    "selected_layer17_only",
    "frozen_uncorrected_composition",
    "selected_composition",
    "matched_double_deletion",
}
_SOURCE_SCORER_CONDITION_BY_A5C = {
    "layer10_only": "layer10_only",
    "selected_layer17_only": "trajectory_corrected_layer17_only",
    "frozen_uncorrected_composition": "frozen_uncorrected_composition",
    "selected_composition": "trajectory_corrected_composition",
    "matched_double_deletion": "matched_double_deletion",
}
_EVALUATION_FIELDS = {
    "assessment_role",
    "outer_fold_index",
    "logical_valid_tokens",
    "supervised_tokens",
    "native",
    "conditions",
    "source_scorer_evaluation",
    "source_scorer_receipt_sha256",
    "source_scorer_evaluation_sha256",
    "resource_accounting_sha256",
    "resource_accounting_reference",
    "full_model_logits_scored",
    "full_model_compiled",
    "heldout_confirmation",
}
_EVIDENCE_RECEIPT_FIELDS = {
    "target_solve",
    "coordinate_row_bank",
    "breadth_split",
    "ridge_cv",
}
_TARGET_EVIDENCE_FIELDS = {
    "schema",
    "objective",
    "scientific_method",
    "throughput_change_only",
    "teacher_boundary",
    "candidate_formula",
    "initialization",
    "canonical_target_dtype",
    "affine_arithmetic_dtype",
    "coordinate_layout",
    "runtime_correction_dtype",
    "runtime_correction_cast_count_per_materialization",
    "initial_correction_bit_identical_to_a4_float64_one_cast",
    "row_count",
    "row_chunk_size",
    "chunk_count",
    "batching",
    "solver",
    "initial_kl",
    "selected_kl",
    "absolute_mean_kl_improvement",
    "selected_not_worse_than_initial_for_every_token",
    "selected_step",
    "trust_projection_count",
    "initial_state_error",
    "selected_state_error",
    "hashes",
    "chunk_receipts",
    "frozen_affine_membership_by_construction",
    "basis_mean_or_decoder_changed",
    "deployable_generator_fitted",
    "contains_tensor_payloads",
    "receipt_sha256",
}
_DIRECTED_LOCALITY_FIELDS = {
    "schema",
    "scientific_role",
    "changes_solver_authorization",
    "probe_name",
    "method",
    "counterfactual_policy",
    "row_count",
    "hidden_width",
    "vocabulary_width",
    "projection_input_shape",
    "projection_output_shape",
    "projection_call_count",
    "baseline_call_count",
    "counterfactual_call_count",
    "directed_pair_order",
    "directed_pair_count",
    "mutation_scale",
    "absolute_tolerance",
    "relative_tolerance",
    "input_rows_sha256",
    "baseline_logits_sha256",
    "source_counterfactuals",
    "directed_pair_checks",
    "failing_directed_pair_count",
    "failing_directed_pairs",
    "worst_source_row_index",
    "worst_target_row_index",
    "worst_max_abs",
    "worst_rms",
    "worst_max_abs_over_allowed",
    "passed",
    "contains_tensor_payloads",
    "receipt_sha256",
}
_DIRECTED_LOCALITY_SOURCE_FIELDS = {
    "source_row_index",
    "source_row_sha256",
    "mutated_source_row_sha256",
    "counterfactual_input_sha256",
    "counterfactual_logits_sha256",
    "mutated_source_element_count",
    "preserved_target_row_count",
    "minimum_absolute_coordinate_change",
    "maximum_absolute_coordinate_change",
    "source_mutation_l2",
}
_DIRECTED_LOCALITY_PAIR_FIELDS = {
    "source_row_index",
    "target_row_index",
    "max_abs",
    "rms",
    "target_reference_max_abs",
    "allowed",
    "max_abs_over_allowed",
    "passed",
}
_A5C_TOKEN_LOCALITY_FIELDS = {
    "policy",
    "method",
    "probe_states",
    "row_count",
    "nontrivial_multirow_probe",
    "absolute_tolerance",
    "relative_tolerance",
    "teacher",
    "a4_baseline",
    "directed_receipt_sha256_by_probe",
    "native_teacher_baseline_logits_reused_by_solver",
    "changes_solver_authorization",
    "passed",
}
_COORDINATE_ROW_BANK_EVIDENCE_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "source_safe",
    "contains_tensors",
    "contains_prompt_text",
    "contains_prompt_identities",
    "contains_token_ids",
    "heldout_confirmation",
    "outer_split",
    "authentication",
    "frozen_affine_image",
    "joint_roundtrip_audit",
    "inner_split",
    "fisher_normalization",
    "row_accounting",
    "consumer_contract",
    "receipt_sha256",
}
_COMPARISON_FIELDS = {
    "a5b_report_sha256",
    "same_outer_fold",
    "same_held_example_policy",
    "a5b_learned_composition",
}
_TOP_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "source_bindings",
    "runtime",
    "configuration",
    "capture",
    "target_solve",
    "coordinate_row_bank",
    "breadth_split",
    "ridge_cv",
    "evidence_receipts",
    "selected_executable",
    "chronology",
    "outer_evaluation",
    "comparison_to_a5b",
    "conclusion",
    "full_model_forward_evaluated",
    "whole_model_compiled",
    "heldout_confirmation",
    "serving_authorized",
    "latency_or_kernel_speed_claim",
    "safety",
    "report_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _exact_mapping(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_receipt_schema(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _RECEIPT_SCHEMA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a source-safe schema identifier")
    return value


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside its finite range")
    return result


def _positive_int(value: object, *, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _assert_source_safe(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise TypeError(f"{path} contains a tensor payload")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _assert_source_safe(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _assert_source_safe(child, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _assert_no_sensitive_keys(value: object, *, path: str = "report") -> None:
    forbidden = {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "raw_prompt",
        "raw_prompts",
        "prompt_id",
        "prompt_ids",
        "prompt_identity",
        "prompt_identities",
        "token_ids",
        "input_ids",
        "target_ids",
        "labels",
        "logits",
        "activations",
        "activation_tensor",
        "activation_tensors",
        "coordinates",
        "coordinate_tensor",
        "coordinate_tensors",
        "source_model_weights",
        "generator_weights",
        "raw_rows",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            safety_key = path == "report.safety" and key in _SAFETY
            if not safety_key and key.casefold() in forbidden:
                raise ValueError(f"{path}.{key} contains prohibited source data")
            _assert_no_sensitive_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _assert_no_sensitive_keys(child, path=f"{path}[{index}]")


def _validate_receipt_ref(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    receipt = _exact_mapping(value, fields, label=label)
    _require_receipt_schema(receipt["receipt_schema"], label=f"{label} schema")
    _require_sha256(receipt["receipt_sha256"], label=f"{label} receipt")
    if receipt["contains_tensor_payloads"] is not False:
        raise ValueError(f"{label} must not contain tensor payloads")
    return receipt


def _validate_descriptor(value: object, *, label: str) -> dict[str, object]:
    descriptor = _exact_mapping(value, _DESCRIPTOR_FIELDS, label=label)
    for name in ("layer17_graph_sha256", "composition_graph_sha256"):
        _require_sha256(descriptor[name], label=f"{label} {name}")
    lowerings = descriptor["layer17_lowering_sha256_by_node"]
    if (
        not isinstance(lowerings, Mapping)
        or len(lowerings) != 4
        or any(not isinstance(name, str) or not name for name in lowerings)
    ):
        raise ValueError(f"{label} must bind exactly four Layer17 lowerings")
    for name, digest in lowerings.items():
        _require_sha256(digest, label=f"{label} lowering {name}")
    expected_resources = {
        "layer17_parameter_count": 163_094,
        "layer17_macs_per_token": 160_352,
        "composition_parameter_count": 295_129,
        "composition_macs_per_token": 289_600,
    }
    if any(
        descriptor[name] != expected
        for name, expected in expected_resources.items()
    ):
        raise ValueError(f"{label} resources differ from the canonical executor")
    for name in (
        "layer17_post_feedforward_delta_layer_ordinals",
        "composition_post_feedforward_delta_layer_ordinals",
    ):
        if descriptor[name] not in ([], [17]):
            raise ValueError(f"{label} {name} must be [] or [17]")
    return {
        **dict(descriptor),
        "layer17_lowering_sha256_by_node": dict(lowerings),
        "layer17_post_feedforward_delta_layer_ordinals": list(
            descriptor["layer17_post_feedforward_delta_layer_ordinals"]
        ),
        "composition_post_feedforward_delta_layer_ordinals": list(
            descriptor["composition_post_feedforward_delta_layer_ordinals"]
        ),
    }


def _validate_quality_metric(value: object, *, label: str) -> dict[str, float]:
    metric = _exact_mapping(value, _METRIC_FIELDS, label=label)
    nll = _finite(metric["nll_per_token"], label=f"{label} NLL", minimum=0.0)
    delta = _finite(metric["delta_nll_per_token"], label=f"{label} delta NLL")
    kl = _finite(
        metric["native_to_candidate_kl_per_token"],
        label=f"{label} KL",
        minimum=_MINIMUM_NUMERICAL_KL_PER_TOKEN,
    )
    top1 = _finite(metric["top1_agreement_to_native"], label=f"{label} top1")
    if top1 < 0.0 or top1 > 1.0:
        raise ValueError(f"{label} top1 is outside [0, 1]")
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _validate_descriptor_application_semantics(
    *,
    kind: str,
    selected: Mapping[str, object],
    frozen: Mapping[str, object],
) -> None:
    fields = (
        "layer17_post_feedforward_delta_layer_ordinals",
        "composition_post_feedforward_delta_layer_ordinals",
    )
    if any(frozen[name] != [] for name in fields):
        raise ValueError("A5c frozen source must use ordinary MLP-delta application")
    expected_selected = [] if kind == _FROZEN else [17]
    if any(selected[name] != expected_selected for name in fields):
        raise ValueError("A5c selected executor application semantics drifted")


def _validate_condition(value: object, *, label: str) -> dict[str, object]:
    condition = _exact_mapping(value, _CONDITION_FIELDS, label=label)
    graph = _require_sha256(condition["graph_sha256"], label=f"{label} graph")
    metric = _validate_quality_metric(
        {name: condition[name] for name in _METRIC_FIELDS}, label=label
    )
    return {"graph_sha256": graph, **metric}


def a5c_outer_evaluation_sha256(value: Mapping[str, object]) -> str:
    """Hash the exact tensor-free outer-evaluation payload for chronology."""

    if not isinstance(value, Mapping):
        raise TypeError("A5c outer evaluation must be a mapping")
    _assert_source_safe(value, path="outer_evaluation")
    _assert_no_sensitive_keys(value, path="outer_evaluation")
    return _sha256(_EVALUATION_DOMAIN, value)


def a5c_source_scorer_evaluation_sha256(
    value: Mapping[str, object],
) -> str:
    """Authenticate the complete tensor-free source scorer result."""

    _assert_source_safe(value, path="source_scorer_evaluation")
    _assert_no_sensitive_keys(value, path="source_scorer_evaluation")
    _validate_source_fold_evaluation(value, label="A5c source scorer")
    return _sha256(_SOURCE_SCORER_EVALUATION_DOMAIN, value)


def a5c_resource_accounting_sha256(value: Mapping[str, object]) -> str:
    """Hash the scorer's exact validated resource-accounting mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("A5c resource accounting must be a mapping")
    _assert_source_safe(value, path="resource_accounting")
    return _sha256(_RESOURCE_ACCOUNTING_DOMAIN, value)


def a5c_source_scorer_receipt_sha256(
    *,
    source_scorer_evaluation: Mapping[str, object],
    outer_fold_index: int,
    condition_graph_sha256_by_name: Mapping[str, object],
) -> str:
    """Bind one raw scorer result to the exact post-freeze graph panel."""

    _positive_int(outer_fold_index, label="A5c scorer outer fold", minimum=0)
    graph_hashes = _exact_mapping(
        condition_graph_sha256_by_name,
        _CONDITIONS,
        label="A5c scorer graph bindings",
    )
    for name, digest in graph_hashes.items():
        _require_sha256(digest, label=f"A5c scorer graph {name}")
    resources = source_scorer_evaluation.get("resource_accounting")
    if not isinstance(resources, Mapping):
        raise TypeError("A5c source scorer resource accounting is unavailable")
    return _sha256(
        _SOURCE_SCORER_RECEIPT_DOMAIN,
        {
            "outer_fold_index": outer_fold_index,
            "source_scorer_evaluation_sha256": (
                a5c_source_scorer_evaluation_sha256(
                    source_scorer_evaluation
                )
            ),
            "resource_accounting_sha256": (
                a5c_resource_accounting_sha256(resources)
            ),
            "resource_accounting_reference": (
                "score_trajectory_correction_fold.resource_accounting"
            ),
            "condition_graph_sha256_by_name": dict(graph_hashes),
        },
    )


def _validate_directed_locality_receipt(
    value: object,
    *,
    label: str,
    probe_name: str,
    row_count: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    receipt = _exact_mapping(value, _DIRECTED_LOCALITY_FIELDS, label=label)
    validate_same_shape_off_row_locality_receipt(receipt)
    hidden_width = receipt["hidden_width"]
    vocabulary_width = receipt["vocabulary_width"]
    pair_count = row_count * (row_count - 1)
    source_rows = receipt["source_counterfactuals"]
    pairs = receipt["directed_pair_checks"]
    if (
        receipt["schema"]
        != "fisher_graph.a5_same_shape_off_row_locality.v2"
        or receipt["scientific_role"] != "solver_authorization"
        or receipt["changes_solver_authorization"] is not True
        or receipt["probe_name"] != probe_name
        or receipt["row_count"] != row_count
        or receipt["projection_input_shape"]
        != [1, row_count, hidden_width]
        or receipt["projection_output_shape"]
        != [1, row_count, vocabulary_width]
        or receipt["projection_call_count"] != row_count + 1
        or receipt["baseline_call_count"] != 1
        or receipt["counterfactual_call_count"] != row_count
        or receipt["directed_pair_count"] != pair_count
        or not isinstance(source_rows, list)
        or len(source_rows) != row_count
        or not isinstance(pairs, list)
        or len(pairs) != pair_count
        or any(
            not isinstance(pair, Mapping) or pair.get("passed") is not True
            for pair in pairs
        )
        or receipt["failing_directed_pair_count"] != 0
        or receipt["failing_directed_pairs"] != []
        or receipt["absolute_tolerance"] != absolute_tolerance
        or receipt["relative_tolerance"] != relative_tolerance
        or receipt["passed"] is not True
        or receipt["contains_tensor_payloads"] is not False
    ):
        raise ValueError(f"{label} is not an exact passing A5c directed audit")
    mutation_scale = _finite(
        receipt["mutation_scale"], label=f"{label} mutation scale", minimum=1.0
    )
    validated_sources: list[dict[str, object]] = []
    for index, raw_source in enumerate(source_rows):
        source = _exact_mapping(
            raw_source,
            _DIRECTED_LOCALITY_SOURCE_FIELDS,
            label=f"{label} source row {index}",
        )
        source_sha = _require_sha256(
            source["source_row_sha256"], label=f"{label} source row {index}"
        )
        mutated_sha = _require_sha256(
            source["mutated_source_row_sha256"],
            label=f"{label} mutated source row {index}",
        )
        minimum_change = _finite(
            source["minimum_absolute_coordinate_change"],
            label=f"{label} minimum source change {index}",
            minimum=0.0,
        )
        if (
            source["source_row_index"] != index
            or source_sha == mutated_sha
            or minimum_change < 0.5 * mutation_scale
        ):
            raise ValueError(f"{label} source mutation evidence drifted")
        validated_sources.append(dict(source))
    validated_pairs: list[dict[str, object]] = []
    for index, raw_pair in enumerate(pairs):
        pair = _exact_mapping(
            raw_pair,
            _DIRECTED_LOCALITY_PAIR_FIELDS,
            label=f"{label} directed pair {index}",
        )
        for name in (
            "max_abs",
            "rms",
            "target_reference_max_abs",
            "allowed",
            "max_abs_over_allowed",
        ):
            _finite(
                pair[name],
                label=f"{label} directed pair {index} {name}",
                minimum=0.0,
            )
        if pair["passed"] is not True:
            raise ValueError(f"{label} contains a failing directed pair")
        validated_pairs.append(dict(pair))
    return {
        **dict(receipt),
        "source_counterfactuals": validated_sources,
        "directed_pair_checks": validated_pairs,
    }


def _validate_a5c_token_locality(
    value: object,
    *,
    label: str,
    row_count: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    locality = _exact_mapping(value, _A5C_TOKEN_LOCALITY_FIELDS, label=label)
    if (
        locality["policy"] != _A5C_TOKEN_LOCALITY_POLICY
        or locality["method"] != _A5C_TOKEN_LOCALITY_METHOD
        or locality["probe_states"]
        != ["native_teacher", "a4_euclidean_baseline"]
        or locality["row_count"] != row_count
        or locality["nontrivial_multirow_probe"] is not True
        or row_count < 2
        or locality["absolute_tolerance"] != absolute_tolerance
        or locality["relative_tolerance"] != relative_tolerance
        or locality["native_teacher_baseline_logits_reused_by_solver"]
        is not True
        or locality["changes_solver_authorization"] is not True
        or locality["passed"] is not True
    ):
        raise ValueError(f"{label} directed policy or row contract drifted")
    hashes = _exact_mapping(
        locality["directed_receipt_sha256_by_probe"],
        set(_A5C_TOKEN_LOCALITY_PROBES),
        label=f"{label} directed receipt hashes",
    )
    nested: dict[str, dict[str, object]] = {}
    for role, probe_name in _A5C_TOKEN_LOCALITY_PROBES.items():
        receipt = _validate_directed_locality_receipt(
            locality[role],
            label=f"{label} {role}",
            probe_name=probe_name,
            row_count=row_count,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        supplied = _require_sha256(
            hashes[role], label=f"{label} {role} directed receipt"
        )
        if supplied != receipt["receipt_sha256"]:
            raise ValueError(f"{label} directed receipt hash map drifted")
        nested[role] = receipt
    return {
        **dict(locality),
        **nested,
        "directed_receipt_sha256_by_probe": dict(hashes),
    }


def a5c_target_token_locality_lineage_sha256(
    target_receipt: Mapping[str, object],
) -> str:
    """Bind CV head scoring to every target-solver locality audit."""

    batching = target_receipt.get("batching")
    chunks = target_receipt.get("chunk_receipts")
    if not isinstance(batching, Mapping) or not isinstance(chunks, Sequence):
        raise TypeError("A5c target locality receipt is incomplete")
    if (
        batching.get("token_locality_policy") != _A5C_TOKEN_LOCALITY_POLICY
        or batching.get("token_locality_absolute_tolerance")
        != _A5C_TOKEN_LOCALITY_ATOL
        or batching.get("token_locality_relative_tolerance")
        != _A5C_TOKEN_LOCALITY_RTOL
    ):
        raise ValueError("A5c target locality policy or tolerance drifted")
    locality_catalog: list[dict[str, object]] = []
    expected_start = 0
    for index, raw_chunk in enumerate(chunks):
        if not isinstance(raw_chunk, Mapping):
            raise TypeError("A5c target locality chunk must be a mapping")
        raw_locality = raw_chunk.get("token_locality")
        start = raw_chunk.get("row_start")
        stop = raw_chunk.get("row_stop")
        count = raw_chunk.get("row_count")
        if (
            raw_chunk.get("chunk_index") != index
            or type(start) is not int
            or type(stop) is not int
            or type(count) is not int
            or start != expected_start
            or stop - start != count
            or count <= 0
        ):
            raise ValueError("A5c target locality catalog drifted")
        locality = _validate_a5c_token_locality(
            raw_locality,
            label=f"A5c target locality chunk {index}",
            row_count=count,
            absolute_tolerance=_A5C_TOKEN_LOCALITY_ATOL,
            relative_tolerance=_A5C_TOKEN_LOCALITY_RTOL,
        )
        locality_catalog.append(
            {
                "chunk_index": index,
                "row_start": start,
                "row_stop": stop,
                "row_count": count,
                "token_locality": dict(locality),
            }
        )
        expected_start = stop
    if (
        not locality_catalog
        or expected_start != target_receipt.get("row_count")
        or batching.get("token_locality_audited_on_native_and_a4_states")
        is not True
    ):
        raise ValueError("A5c target locality audits do not cover every row")
    return _sha256(
        _TARGET_TOKEN_LOCALITY_LINEAGE_DOMAIN,
        {
            "target_schema": target_receipt.get("schema"),
            "target_receipt_sha256": target_receipt.get("receipt_sha256"),
            "row_count": target_receipt.get("row_count"),
            "row_chunk_size": target_receipt.get("row_chunk_size"),
            "absolute_tolerance": batching.get(
                "token_locality_absolute_tolerance"
            ),
            "relative_tolerance": batching.get(
                "token_locality_relative_tolerance"
            ),
            "policy": batching.get("token_locality_policy"),
            "chunk_token_locality_catalog": locality_catalog,
        },
    )


def _validate_selection_lineage(
    value: Mapping[str, object], *, label: str
) -> dict[str, object]:
    lineage = _exact_mapping(value, _LINEAGE_FIELDS, label=label)
    lowerings = _exact_mapping(
        lineage["layer10_lowering_sha256_by_node"],
        set(_EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE),
        label=f"{label} Layer10 lowerings",
    )
    for name, digest in lowerings.items():
        _require_sha256(digest, label=f"{label} Layer10 lowering {name}")
    for name, digest in lineage.items():
        if name != "layer10_lowering_sha256_by_node":
            _require_sha256(digest, label=f"{label} {name}")
    return {
        **dict(lineage),
        "layer10_lowering_sha256_by_node": dict(lowerings),
    }


def a5c_selection_freeze_sha256(
    *,
    kind: str,
    selected_ridge: float | None,
    lineage: Mapping[str, object],
    selected: Mapping[str, object],
    frozen_reference: Mapping[str, object],
) -> str:
    """Hash the complete pre-held A5c executable selection."""

    if kind not in _EXECUTABLE_KINDS:
        raise ValueError("A5c executable kind is invalid")
    validated_lineage = _validate_selection_lineage(
        lineage, label="A5c selection lineage"
    )
    selected_descriptor = _validate_descriptor(selected, label="A5c selected")
    frozen_descriptor = _validate_descriptor(
        frozen_reference, label="A5c frozen reference"
    )
    _validate_descriptor_application_semantics(
        kind=kind,
        selected=selected_descriptor,
        frozen=frozen_descriptor,
    )
    if kind == _FROZEN:
        if selected_ridge is not None or selected_descriptor != frozen_descriptor:
            raise ValueError("A5c frozen fallback must exactly select the source graph")
    else:
        ridge = _finite(selected_ridge, label="A5c selected ridge", minimum=0.0)
        selected_ridge = ridge
    payload = {
        "kind": kind,
        "selected_ridge": selected_ridge,
        "selected_ridge_hex": (
            None if selected_ridge is None else float(selected_ridge).hex()
        ),
        "lineage": dict(validated_lineage),
        "selected": selected_descriptor,
        "frozen_reference": frozen_descriptor,
    }
    return _sha256(_FREEZE_DOMAIN, payload)


def _validate_selected_executable(
    value: object,
    *,
    target: Mapping[str, object],
    row_bank: Mapping[str, object],
    breadth: Mapping[str, object],
    ridge_cv: Mapping[str, object],
    ridge_grid: Sequence[float],
) -> dict[str, object]:
    executable = _exact_mapping(value, _SELECTED_FIELDS, label="A5c executable")
    kind = executable["kind"]
    if kind not in _EXECUTABLE_KINDS:
        raise ValueError("A5c executable kind is invalid")
    selected_ridge = executable["selected_ridge"]
    if kind == _FROZEN:
        if selected_ridge is not None:
            raise ValueError("A5c frozen fallback cannot select a ridge")
    else:
        selected_ridge = _finite(
            selected_ridge, label="A5c selected ridge", minimum=0.0
        )
        if selected_ridge not in ridge_grid:
            raise ValueError("A5c selected ridge is outside the declared grid")
    if (
        ridge_cv["use_frozen_fallback"] is not (kind == _FROZEN)
        or ridge_cv["selected_ridge"] != selected_ridge
    ):
        raise ValueError("A5c executable contradicts ridge selection")
    lineage = _validate_selection_lineage(
        executable["lineage"], label="A5c selection lineage"
    )
    expected_lineage = {
        "target_solve_receipt_sha256": target["receipt_sha256"],
        "coordinate_row_bank_receipt_sha256": row_bank["receipt_sha256"],
        "breadth_split_receipt_sha256": breadth["receipt_sha256"],
        "ridge_cv_receipt_sha256": ridge_cv["receipt_sha256"],
    }
    if any(lineage[name] != digest for name, digest in expected_lineage.items()):
        raise ValueError("A5c executable lineage contradicts its receipts")
    selected = _validate_descriptor(executable["selected"], label="A5c selected")
    frozen = _validate_descriptor(
        executable["frozen_reference"], label="A5c frozen reference"
    )
    _validate_descriptor_application_semantics(
        kind=str(kind),
        selected=selected,
        frozen=frozen,
    )
    if (
        lineage["layer10_graph_sha256"] != _EXPECTED_LAYER10_GRAPH_SHA256
        or lineage["layer10_lowering_sha256_by_node"]
        != _EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE
        or frozen["layer17_graph_sha256"] != _EXPECTED_LAYER17_GRAPH_SHA256
        or frozen["layer17_lowering_sha256_by_node"]
        != _EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
        or frozen["composition_graph_sha256"]
        != _EXPECTED_PRIMARY_COMPOSITION_GRAPH_SHA256
    ):
        raise ValueError(
            "A5c frozen executable differs from the canonical runtime catalog"
        )
    if kind == _FROZEN and selected != frozen:
        raise ValueError("A5c fallback executable hashes/resources must be exact")
    expected_freeze = a5c_selection_freeze_sha256(
        kind=str(kind),
        selected_ridge=selected_ridge,  # type: ignore[arg-type]
        lineage=lineage,
        selected=selected,
        frozen_reference=frozen,
    )
    supplied = _require_sha256(
        executable["selection_freeze_sha256"], label="A5c selection freeze"
    )
    if supplied != expected_freeze:
        raise ValueError("A5c selection freeze hash is contradictory")
    return {
        "kind": kind,
        "selected_ridge": selected_ridge,
        "lineage": dict(lineage),
        "selected": selected,
        "frozen_reference": frozen,
        "selection_freeze_sha256": supplied,
    }


def _validate_target_evidence(
    value: object,
    *,
    configuration: Mapping[str, object],
    expected_rows: int,
) -> dict[str, object]:
    target = _exact_mapping(
        value, _TARGET_EVIDENCE_FIELDS, label="A5c target evidence"
    )
    if (
        target["schema"]
        != "fisher_graph.gemma3_l10_l17_a5b_batched_capacity.v1"
        or target["objective"]
        != (
            "independent_per_token_exact_native_to_candidate_kl_through_"
            "adapter_project_logits"
        )
        or target["scientific_method"]
        != "a5a_frozen_affine_capacity_oracle"
        or target["throughput_change_only"] is not True
        or target["teacher_boundary"] != "captured_native_layer17_output"
        or target["candidate_formula"]
        != (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        )
        or target["initialization"]
        != "float64_affine_sum_svd_pseudoinverse_minimum_norm"
        or target["canonical_target_dtype"] != "torch.float64"
        or target["affine_arithmetic_dtype"] != "torch.float64"
        or target["coordinate_layout"]
        != "joint_concatenated_four_node_rank_182"
        or target["runtime_correction_dtype"] != "torch.float32"
        or target["runtime_correction_cast_count_per_materialization"] != 1
        or target[
            "initial_correction_bit_identical_to_a4_float64_one_cast"
        ]
        is not True
        or target["row_count"] != expected_rows
        or target["row_chunk_size"] != configuration["target_batch_rows"]
        or target["contains_tensor_payloads"] is not False
        or target["frozen_affine_membership_by_construction"] is not True
        or target["basis_mean_or_decoder_changed"] is not False
        or target["deployable_generator_fitted"] is not False
        or target["selected_not_worse_than_initial_for_every_token"] is not True
    ):
        raise ValueError("A5c target evidence contract drifted")

    batching = _exact_mapping(
        target["batching"],
        {
            "one_batched_head_callback_per_optimizer_evaluation",
            "independent_adam_parameter_group_per_token",
            "independent_kl_best_checkpoint_per_token",
            "token_locality_audited_on_native_and_a4_states",
            "token_locality_policy",
            "token_locality_absolute_tolerance",
            "token_locality_relative_tolerance",
        },
        label="A5c target batching",
    )
    if (
        any(
            batching[name] is not True
            for name in (
                "one_batched_head_callback_per_optimizer_evaluation",
                "independent_adam_parameter_group_per_token",
                "independent_kl_best_checkpoint_per_token",
                "token_locality_audited_on_native_and_a4_states",
            )
        )
        or batching["token_locality_policy"] != _A5C_TOKEN_LOCALITY_POLICY
        or batching["token_locality_absolute_tolerance"]
        != _A5C_TOKEN_LOCALITY_ATOL
        or batching["token_locality_relative_tolerance"]
        != _A5C_TOKEN_LOCALITY_RTOL
    ):
        raise ValueError("A5c target batching/locality contract drifted")
    solver = _exact_mapping(
        target["solver"],
        {
            "steps",
            "learning_rate_fraction_of_per_token_initial_coefficient_rms",
            "minimum_scale_for_zero_rms",
            "initial_coefficient_rms",
            "effective_learning_rate",
            "scale_is_independent_for_each_token",
            "ridge",
            "trust_radius",
            "initial_point_evaluated_as_safe_abstention",
        },
        label="A5c target solver",
    )
    if (
        solver["steps"] != configuration["target_solver_steps"]
        or solver[
            "learning_rate_fraction_of_per_token_initial_coefficient_rms"
        ]
        != configuration["target_learning_rate_fraction"]
        or _finite(
            solver["minimum_scale_for_zero_rms"],
            label="A5c target minimum scale",
            minimum=0.0,
        )
        <= 0.0
        or solver["scale_is_independent_for_each_token"] is not True
        or solver["ridge"] != configuration["target_ridge"]
        or solver["trust_radius"] != configuration["target_trust_radius"]
        or solver["initial_point_evaluated_as_safe_abstention"] is not True
    ):
        raise ValueError("A5c target solver contract drifted")
    _validate_target_summary(
        solver["initial_coefficient_rms"], label="A5c initial coefficient RMS"
    )
    _validate_target_summary(
        solver["effective_learning_rate"], label="A5c effective learning rate"
    )
    initial_kl = _validate_target_summary(
        target["initial_kl"], label="A5c initial KL"
    )
    selected_kl = _validate_target_summary(
        target["selected_kl"], label="A5c selected KL"
    )
    _validate_target_summary(target["selected_step"], label="A5c selected step")
    _validate_target_summary(
        target["trust_projection_count"], label="A5c trust projection count"
    )
    if not math.isclose(
        _finite(
            target["absolute_mean_kl_improvement"],
            label="A5c KL improvement",
        ),
        float(initial_kl["mean"]) - float(selected_kl["mean"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("A5c target KL summaries are contradictory")
    _validate_target_error_summary(
        target["initial_state_error"], label="A5c initial state"
    )
    _validate_target_error_summary(
        target["selected_state_error"], label="A5c selected state"
    )

    hashes = _exact_mapping(
        target["hashes"],
        {
            "initial_coefficient_sha256",
            "selected_coefficient_sha256",
            "initial_correction_sha256",
            "selected_correction_sha256",
            "initial_state_sha256",
            "selected_state_sha256",
        },
        label="A5c target hashes",
    )
    for name, digest in hashes.items():
        _require_sha256(digest, label=f"A5c target {name}")

    chunks = target["chunk_receipts"]
    expected_chunk_count = (
        expected_rows + int(configuration["target_batch_rows"]) - 1
    ) // int(configuration["target_batch_rows"])
    if (
        type(target["chunk_count"]) is not int
        or not isinstance(chunks, list)
        or target["chunk_count"] != expected_chunk_count
        or len(chunks) != expected_chunk_count
    ):
        raise ValueError("A5c target chunk catalog drifted")
    expected_start = 0
    for index, raw_chunk in enumerate(chunks):
        chunk = _exact_mapping(
            raw_chunk,
            {
                "chunk_index",
                "row_start",
                "row_stop",
                "row_count",
                "initial_kl",
                "selected_kl",
                "selected_step",
                "trust_projection_count",
                "token_locality",
                "full_solver_receipt_sha256",
            },
            label=f"A5c target chunk {index}",
        )
        start = chunk["row_start"]
        stop = chunk["row_stop"]
        count = chunk["row_count"]
        if (
            chunk["chunk_index"] != index
            or type(start) is not int
            or type(stop) is not int
            or type(count) is not int
            or start != expected_start
            or stop - start != count
            or count <= 0
            or count > configuration["target_batch_rows"]
        ):
            raise ValueError("A5c target chunk coverage drifted")
        for name in (
            "initial_kl",
            "selected_kl",
            "selected_step",
            "trust_projection_count",
        ):
            _validate_target_summary(
                chunk[name], label=f"A5c chunk {index} {name}"
            )
        _validate_a5c_token_locality(
            chunk["token_locality"],
            label=f"A5c chunk {index} locality",
            row_count=count,
            absolute_tolerance=_A5C_TOKEN_LOCALITY_ATOL,
            relative_tolerance=_A5C_TOKEN_LOCALITY_RTOL,
        )
        _require_sha256(
            chunk["full_solver_receipt_sha256"],
            label=f"A5c target chunk {index} solver",
        )
        expected_start = stop
    if expected_start != expected_rows:
        raise ValueError("A5c target chunks do not cover every selected row")
    supplied = _require_sha256(
        target["receipt_sha256"], label="A5c target evidence receipt"
    )
    payload = dict(target)
    payload.pop("receipt_sha256")
    if supplied != _sha256(_TARGET_RECEIPT_DOMAIN, payload):
        raise ValueError("A5c target evidence receipt hash mismatch")
    return {**dict(target), "hashes": dict(hashes)}


def _validate_coordinate_row_bank_evidence(
    value: object,
    *,
    configuration: Mapping[str, object],
    expected_rows: int,
    expected_examples: int,
) -> dict[str, object]:
    bridge = _exact_mapping(
        value,
        _COORDINATE_ROW_BANK_EVIDENCE_FIELDS,
        label="A5c coordinate row-bank evidence",
    )
    if (
        bridge["schema"]
        != GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA
        or bridge["format_version"]
        != GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION
        or bridge["source_safe"] is not True
        or bridge["contains_tensors"] is not False
        or bridge["contains_prompt_text"] is not False
        or bridge["contains_prompt_identities"] is not False
        or bridge["contains_token_ids"] is not False
        or bridge["heldout_confirmation"] is not False
        or bridge["scientific_role"]
        != "calibration_a_fit_outer_training_downstream_coordinate_targets"
    ):
        raise ValueError("A5c coordinate row-bank evidence boundary drifted")
    outer = _exact_mapping(
        bridge["outer_split"],
        {
            "training_family_aliases",
            "held_family_alias",
            "held_family_rows_accepted",
            "held_family_used_for_fit_or_audit",
        },
        label="A5c coordinate outer split",
    )
    aliases = tuple(outer["training_family_aliases"])  # type: ignore[arg-type]
    if (
        len(aliases) != configuration["training_family_count"]
        or len(set(aliases)) != len(aliases)
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or not isinstance(outer["held_family_alias"], str)
        or outer["held_family_alias"] in aliases
        or outer["held_family_rows_accepted"] is not False
        or outer["held_family_used_for_fit_or_audit"] is not False
    ):
        raise ValueError("A5c coordinate outer split drifted")
    authentication = _exact_mapping(
        bridge["authentication"],
        {
            "compiled_inputs_sha256",
            "selected_joint_coordinates_sha256",
            "joint_coordinate_width",
        },
        label="A5c coordinate authentication",
    )
    for name in ("compiled_inputs_sha256", "selected_joint_coordinates_sha256"):
        _require_sha256(authentication[name], label=f"A5c coordinate {name}")
    if authentication["joint_coordinate_width"] != 182:
        raise ValueError("A5c coordinate width drifted")
    affine = _exact_mapping(
        bridge["frozen_affine_image"],
        {
            "node_order",
            "rank_by_node",
            "coordinate_slices",
            "fragment_id_by_node",
            "basis_sha256_by_node",
            "mean_sha256_by_node",
            "encoder_sha256_by_node",
            "decoder_sha256_by_node",
            "basis_artifacts_unchanged_after_decode",
            "mean_tensors_byte_identical_after_decode",
            "encoder_tensors_byte_identical_after_decode",
            "decoder_tensors_byte_identical_after_decode",
        },
        label="A5c frozen affine image",
    )
    nodes = tuple(affine["node_order"])  # type: ignore[arg-type]
    ranks = tuple(affine["rank_by_node"])  # type: ignore[arg-type]
    fragments = affine["fragment_id_by_node"]
    slices = affine["coordinate_slices"]
    if (
        len(nodes) != 4
        or len(set(nodes)) != 4
        or len(ranks) != 4
        or any(type(rank) is not int or rank <= 0 for rank in ranks)
        or sum(ranks) != 182
        or not isinstance(fragments, Mapping)
        or set(fragments) != set(nodes)
        or len(set(fragments.values())) != 4
        or not isinstance(slices, Mapping)
        or set(slices) != set(nodes)
        or any(affine[name] is not True for name in (
            "basis_artifacts_unchanged_after_decode",
            "mean_tensors_byte_identical_after_decode",
            "encoder_tensors_byte_identical_after_decode",
            "decoder_tensors_byte_identical_after_decode",
        ))
    ):
        raise ValueError("A5c frozen affine catalog drifted")
    offset = 0
    for node, rank in zip(nodes, ranks, strict=True):
        if slices[node] != {"start": offset, "stop": offset + rank, "rank": rank}:
            raise ValueError("A5c coordinate slices drifted")
        offset += rank
    for catalog_name in (
        "basis_sha256_by_node",
        "mean_sha256_by_node",
        "encoder_sha256_by_node",
        "decoder_sha256_by_node",
    ):
        catalog = affine[catalog_name]
        if not isinstance(catalog, Mapping) or set(catalog) != set(nodes):
            raise ValueError("A5c affine hash catalog drifted")
        for digest in catalog.values():
            _require_sha256(digest, label=f"A5c {catalog_name}")
    roundtrip = _exact_mapping(
        bridge["joint_roundtrip_audit"],
        {
            "definition",
            "joint_roundtrip_sha256",
            "summed_decoded_contribution_sha256",
            "max_abs_difference",
            "rms_difference",
            "relative_tolerance",
            "absolute_tolerance",
            "passed",
        },
        label="A5c coordinate roundtrip",
    )
    if (
        roundtrip["definition"]
        != (
            "sum_node_decode(coordinate_slice)_equals_"
            "sum_node_mean_plus_joint_coordinate_times_"
            "concatenated_frozen_decoder"
        )
        or roundtrip["relative_tolerance"] != 1.0e-11
        or roundtrip["absolute_tolerance"] != 1.0e-11
        or roundtrip["passed"] is not True
    ):
        raise ValueError("A5c coordinate roundtrip contract drifted")
    for name in ("joint_roundtrip_sha256", "summed_decoded_contribution_sha256"):
        _require_sha256(roundtrip[name], label=f"A5c roundtrip {name}")
    roundtrip_max_abs = _finite(
        roundtrip["max_abs_difference"],
        label="A5c roundtrip max",
        minimum=0.0,
    )
    roundtrip_rms = _finite(
        roundtrip["rms_difference"],
        label="A5c roundtrip RMS",
        minimum=0.0,
    )
    # The canonical bridge applies torch.allclose elementwise, whose bound is
    # ``atol + rtol * abs(reference)``.  The receipt intentionally omits the
    # reference tensor/max, so an absolute-only threshold cannot be derived
    # here.  Preserve the authenticated canonical ``passed`` result and check
    # the scalar invariant that is derivable from the receipt.
    if roundtrip_rms > roundtrip_max_abs + 2.0 * math.ulp(roundtrip_max_abs):
        raise ValueError("A5c coordinate roundtrip pass is contradictory")
    inner = _exact_mapping(
        bridge["inner_split"],
        {
            "method",
            "inner_split_binding_sha256",
            "inner_audit_examples_per_family",
            "fit_example_count",
            "audit_example_count",
            "fit_example_membership_sha256",
            "audit_example_membership_sha256",
            "row_overlap_count",
            "example_overlap_count",
            "rows_exactly_partitioned",
            "examples_exactly_partitioned",
        },
        label="A5c coordinate inner split",
    )
    if (
        inner["method"]
        != (
            "domain_separated_hash_rank_per_training_family_"
            "then_preserve_source_row_order"
        )
        or inner["inner_audit_examples_per_family"]
        != configuration["inner_audit_examples_per_family"]
        or inner["fit_example_count"] + inner["audit_example_count"]
        != expected_examples
        or inner["audit_example_count"]
        != len(aliases) * configuration["inner_audit_examples_per_family"]
        or inner["row_overlap_count"] != 0
        or inner["example_overlap_count"] != 0
        or inner["rows_exactly_partitioned"] is not True
        or inner["examples_exactly_partitioned"] is not True
    ):
        raise ValueError("A5c coordinate inner split drifted")
    for name in (
        "inner_split_binding_sha256",
        "fit_example_membership_sha256",
        "audit_example_membership_sha256",
    ):
        _require_sha256(inner[name], label=f"A5c inner split {name}")

    fisher = _exact_mapping(
        bridge["fisher_normalization"],
        {
            "all_rows_preserve_raw_authenticated_fisher_weights",
            "fit_and_audit_normalized_independently",
            "audit_weights_influence_fit_normalization",
            "policy",
            "training_family_count",
            "target_total_mass_per_role_and_node",
            "target_mass_per_family",
            "fit_by_node",
            "audit_by_node",
        },
        label="A5c coordinate Fisher normalization",
    )
    if (
        fisher["all_rows_preserve_raw_authenticated_fisher_weights"] is not True
        or fisher["fit_and_audit_normalized_independently"] is not True
        or fisher["audit_weights_influence_fit_normalization"] is not False
        or fisher["policy"]
        != "equal_total_mass_per_outer_training_family_per_role_and_node"
        or fisher["training_family_count"] != len(aliases)
        or fisher["target_total_mass_per_role_and_node"] != 1.0
        or fisher["target_mass_per_family"] != 1.0 / len(aliases)
    ):
        raise ValueError("A5c coordinate Fisher policy drifted")
    for role in ("fit_by_node", "audit_by_node"):
        catalog = fisher[role]
        if not isinstance(catalog, Mapping) or set(catalog) != set(nodes):
            raise ValueError("A5c coordinate Fisher node catalog drifted")
        for node, raw_row in catalog.items():
            row = _exact_mapping(
                raw_row,
                {
                    "fragment_id",
                    "total_mass",
                    "family_mass_by_alias",
                    "unit_total_mass",
                    "equal_family_mass",
                },
                label=f"A5c Fisher {role} {node}",
            )
            masses = row["family_mass_by_alias"]
            if (
                row["fragment_id"] != fragments[node]
                or not isinstance(masses, Mapping)
                or set(masses) != set(aliases)
                or not math.isclose(float(row["total_mass"]), 1.0, abs_tol=2e-12)
                or any(
                    not math.isclose(float(mass), 1.0 / len(aliases), abs_tol=2e-12)
                    for mass in masses.values()
                )
                or row["unit_total_mass"] is not True
                or row["equal_family_mass"] is not True
            ):
                raise ValueError("A5c coordinate Fisher masses drifted")
    accounting = _exact_mapping(
        bridge["row_accounting"],
        {
            "all",
            "fit",
            "audit",
            "all_observations_equal_fit_plus_audit",
            "all_examples_equal_fit_plus_audit",
        },
        label="A5c coordinate row accounting",
    )
    expected_role_counts = {
        "all": (expected_rows, expected_examples),
        "fit": (None, inner["fit_example_count"]),
        "audit": (None, inner["audit_example_count"]),
    }
    parsed_rows: dict[str, Mapping[str, object]] = {}
    fragment_ids = set(fragments.values())
    for role, (rows_expected, examples_expected) in expected_role_counts.items():
        row = _exact_mapping(
            accounting[role],
            {"row_key_sha256", "observations", "sequences", "fragment_tensor_sha256s"},
            label=f"A5c coordinate {role} rows",
        )
        parsed_rows[role] = row
        _require_sha256(row["row_key_sha256"], label=f"A5c {role} row key")
        if (
            (rows_expected is not None and row["observations"] != rows_expected)
            or row["sequences"] != examples_expected
        ):
            raise ValueError("A5c coordinate row counts drifted")
        tensor_hashes = row["fragment_tensor_sha256s"]
        if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != fragment_ids:
            raise ValueError("A5c coordinate fragment tensor catalog drifted")
        for raw_hashes in tensor_hashes.values():
            item = _exact_mapping(
                raw_hashes,
                {"inputs_sha256", "contributions_sha256", "fisher_weights_sha256"},
                label=f"A5c coordinate {role} tensors",
            )
            for digest in item.values():
                _require_sha256(digest, label=f"A5c coordinate {role} tensor")
    if (
        parsed_rows["fit"]["observations"] + parsed_rows["audit"]["observations"]
        != expected_rows
        or accounting["all_observations_equal_fit_plus_audit"] is not True
        or accounting["all_examples_equal_fit_plus_audit"] is not True
    ):
        raise ValueError("A5c coordinate row partition drifted")
    consumer = _exact_mapping(
        bridge["consumer_contract"],
        {"compatible_with", "contribution_target", "generator_fit_performed"},
        label="A5c coordinate consumer",
    )
    if (
        consumer["compatible_with"] != "fit_frozen_basis_coordinate_generators"
        or consumer["contribution_target"]
        != "frozen_basis_decode_of_downstream_selected_coordinate_slice"
        or consumer["generator_fit_performed"] is not False
    ):
        raise ValueError("A5c coordinate consumer contract drifted")
    supplied = _require_sha256(
        bridge["receipt_sha256"], label="A5c coordinate row-bank receipt"
    )
    payload = dict(bridge)
    payload.pop("receipt_sha256")
    if supplied != _sha256(_COORDINATE_ROW_BANK_RECEIPT_DOMAIN, payload):
        raise ValueError("A5c coordinate row-bank receipt hash mismatch")
    return dict(bridge)


def _compact_target_evidence(value: Mapping[str, object]) -> dict[str, object]:
    hashes = value["hashes"]
    assert isinstance(hashes, Mapping)
    return {
        "receipt_schema": value["schema"],
        "receipt_sha256": value["receipt_sha256"],
        "row_count": value["row_count"],
        "selected_coordinate_sha256": hashes["selected_coefficient_sha256"],
        "contains_tensor_payloads": value["contains_tensor_payloads"],
    }


def _compact_coordinate_row_bank_evidence(
    value: Mapping[str, object],
) -> dict[str, object]:
    authentication = value["authentication"]
    accounting = value["row_accounting"]
    outer_split = value["outer_split"]
    assert all(
        isinstance(item, Mapping)
        for item in (authentication, accounting, outer_split)
    )
    all_rows = accounting["all"]
    assert isinstance(all_rows, Mapping)
    return {
        "receipt_schema": value["schema"],
        "receipt_sha256": value["receipt_sha256"],
        "row_count": all_rows["observations"],
        "example_count": all_rows["sequences"],
        "row_key_sha256": all_rows["row_key_sha256"],
        "compiled_inputs_sha256": authentication["compiled_inputs_sha256"],
        "selected_coordinates_sha256": authentication[
            "selected_joint_coordinates_sha256"
        ],
        "outer_held_family_rows_present": outer_split[
            "held_family_rows_accepted"
        ],
        "contains_tensor_payloads": value["contains_tensors"],
    }


def _compact_breadth_evidence(value: Mapping[str, object]) -> dict[str, object]:
    source = value["source"]
    final = value["final_split"]
    quarantine = value["collision_quarantine"]
    ownership = value["ownership"]
    safety = value["safety"]
    assert all(
        isinstance(item, Mapping)
        for item in (source, final, quarantine, ownership, safety)
    )
    fit = final["fit"]
    audit = final["audit"]
    assert isinstance(fit, Mapping) and isinstance(audit, Mapping)
    return {
        "receipt_schema": value["schema"],
        "receipt_sha256": value["receipt_sha256"],
        "all_row_count": source["observations"],
        "fit_row_count": fit["observations"],
        "audit_row_count": audit["observations"],
        "all_example_count": source["examples"],
        "fit_example_count": fit["examples"],
        "audit_example_count": audit["examples"],
        "fit_examples_fully_removed_for_signature_overlap": quarantine[
            "fit_examples_fully_removed"
        ],
        "removed_fit_rows_for_signature_overlap": quarantine[
            "fit_rows_removed"
        ],
        "fit_audit_example_overlap_count": final["example_overlap_count"],
        "post_purge_input_signature_overlap_count": final[
            "compiled_input_signature_overlap_count"
        ],
        "outer_held_family_rows_present": ownership[
            "outer_held_family_present"
        ],
        "contains_tensor_payloads": safety["contains_tensors"],
    }


def _compact_cv_evidence(value: Mapping[str, object]) -> dict[str, object]:
    configuration = value["configuration"]
    ownership = value["ownership"]
    selection = value["selection"]
    candidates = value["candidates"]
    safety = value["safety"]
    assert all(
        isinstance(item, Mapping)
        for item in (configuration, ownership, selection, safety)
    )
    assert isinstance(candidates, Sequence)
    return {
        "receipt_schema": value["schema"],
        "receipt_sha256": value["receipt_sha256"],
        "candidate_count": len(candidates),
        "inner_fold_count": configuration["inner_fold_count"],
        "selected_ridge": selection["selected_ridge"],
        "use_frozen_fallback": selection["use_frozen_fallback"],
        "outer_held_family_accessed": ownership[
            "outer_held_family_states_or_rows_accessed"
        ],
        "contains_tensor_payloads": safety["contains_tensors"],
    }


def _validate_evidence_receipts(
    value: object,
    *,
    configuration: Mapping[str, object],
    capture: Mapping[str, object],
    target_summary: Mapping[str, object],
    row_bank_summary: Mapping[str, object],
    breadth_summary: Mapping[str, object],
    cv_summary: Mapping[str, object],
    runtime: Mapping[str, object],
    executable: Mapping[str, object],
) -> dict[str, object]:
    evidence = _exact_mapping(
        value, _EVIDENCE_RECEIPT_FIELDS, label="A5c evidence receipts"
    )
    target = _validate_target_evidence(
        evidence["target_solve"],
        configuration=configuration,
        expected_rows=int(capture["selected_target_row_count"]),
    )
    bridge = _validate_coordinate_row_bank_evidence(
        evidence["coordinate_row_bank"],
        configuration=configuration,
        expected_rows=int(capture["selected_target_row_count"]),
        expected_examples=int(capture["training_example_count"]),
    )
    breadth = validate_a5c_breadth_split_receipt(evidence["breadth_split"])
    cv = validate_a5c_family_ridge_cv_receipt(evidence["ridge_cv"])
    reductions = (
        (target_summary, _compact_target_evidence(target), "target"),
        (
            row_bank_summary,
            _compact_coordinate_row_bank_evidence(bridge),
            "coordinate row bank",
        ),
        (breadth_summary, _compact_breadth_evidence(breadth), "breadth"),
        (cv_summary, _compact_cv_evidence(cv), "ridge CV"),
    )
    for supplied, expected, label in reductions:
        if _canonical_json_bytes(supplied) != _canonical_json_bytes(expected):
            raise ValueError(f"A5c compact {label} summary contradicts evidence")
    target_hashes = target["hashes"]
    bridge_auth = bridge["authentication"]
    breadth_source = breadth["source"]
    cv_source = cv["source"]
    bridge_outer = bridge["outer_split"]
    bridge_affine = bridge["frozen_affine_image"]
    bridge_rows = bridge["row_accounting"]
    breadth_ownership = breadth["ownership"]
    breadth_source_full = breadth["source"]
    cv_ownership = cv["ownership"]
    cv_final = cv["final_refit"]
    selected_descriptor = executable["selected"]
    frozen_descriptor = executable["frozen_reference"]
    assert all(
        isinstance(item, Mapping)
        for item in (
            target_hashes,
            bridge_auth,
            breadth_source,
            cv_source,
            bridge_outer,
            bridge_affine,
            bridge_rows,
            breadth_ownership,
            breadth_source_full,
            cv_ownership,
            cv_final,
            selected_descriptor,
            frozen_descriptor,
        )
    )
    bridge_all_rows = bridge_rows["all"]
    assert isinstance(bridge_all_rows, Mapping)
    if (
        target_hashes["selected_coefficient_sha256"]
        != bridge_auth["selected_joint_coordinates_sha256"]
        or breadth_source["source_bridge_receipt_sha256"]
        != bridge["receipt_sha256"]
        or cv_source["bridge_receipt_sha256"] != breadth["receipt_sha256"]
        or cv_source["final_head_token_locality_lineage_sha256"]
        != a5c_target_token_locality_lineage_sha256(target)
        or list(configuration["ridge_grid"])
        != list(cv["configuration"]["ridge_grid"])  # type: ignore[index]
        or cv_source["source_model_sha256"] != runtime["model_fingerprint"]
        or cv_source["source_graph_sha256"]
        != frozen_descriptor["layer17_graph_sha256"]
        or cv_source["source_lowering_sha256_by_node"]
        != frozen_descriptor["layer17_lowering_sha256_by_node"]
        or tuple(bridge_outer["training_family_aliases"])
        != tuple(breadth_ownership["training_family_aliases"])
        or tuple(bridge_outer["training_family_aliases"])
        != tuple(cv_ownership["outer_training_family_aliases"])
        or bridge_outer["held_family_alias"]
        != breadth_ownership["outer_held_family_alias"]
        or bridge_outer["held_family_alias"]
        != cv_ownership["outer_held_family_alias"]
        or cv_ownership["outer_training_example_count"]
        != bridge_all_rows["sequences"]
        or cv_ownership["outer_training_observation_count"]
        != bridge_all_rows["observations"]
        or breadth_source_full["examples"] != bridge_all_rows["sequences"]
        or breadth_source_full["observations"]
        != bridge_all_rows["observations"]
        or cv_source["all_rows_key_sha256"]
        != bridge_all_rows["row_key_sha256"]
        or breadth_source_full["row_key_sha256"]
        != bridge_all_rows["row_key_sha256"]
        or capture["source_row_catalog_sha256"]
        != bridge_all_rows["row_key_sha256"]
        or cv_source["bridge_compiled_inputs_sha256"]
        != bridge_auth["compiled_inputs_sha256"]
        or breadth_source_full["bridge_compiled_inputs_sha256"]
        != bridge_auth["compiled_inputs_sha256"]
        or tuple(breadth_source_full["node_order"])
        != tuple(bridge_affine["node_order"])
        or breadth_source_full["fragment_id_by_node"]
        != bridge_affine["fragment_id_by_node"]
    ):
        raise ValueError("A5c evidence receipt lineage is contradictory")
    if executable["kind"] == _LEARNED and (
        cv_final["performed"] is not True
        or cv_final["graph_sha256"]
        != selected_descriptor["layer17_graph_sha256"]
        or cv_final["lowering_sha256_by_node"]
        != selected_descriptor["layer17_lowering_sha256_by_node"]
    ):
        raise ValueError("A5c learned executable contradicts CV final refit")
    return {
        "target_solve": target,
        "coordinate_row_bank": bridge,
        "breadth_split": breadth,
        "ridge_cv": cv,
    }


def _validate_evaluation(
    value: object,
    *,
    outer_fold_index: int,
    executable: Mapping[str, object],
) -> dict[str, object]:
    evaluation = _exact_mapping(value, _EVALUATION_FIELDS, label="A5c evaluation")
    if (
        evaluation["assessment_role"]
        != "calibration_a_outer_family_bounded_development"
        or evaluation["outer_fold_index"] != outer_fold_index
        or evaluation["full_model_logits_scored"] is not True
        or evaluation["full_model_compiled"] is not False
        or evaluation["heldout_confirmation"] is not False
        or evaluation["resource_accounting_reference"]
        != "score_trajectory_correction_fold.resource_accounting"
    ):
        raise ValueError("A5c evaluation scope drifted")
    for name in (
        "source_scorer_receipt_sha256",
        "source_scorer_evaluation_sha256",
        "resource_accounting_sha256",
    ):
        _require_sha256(evaluation[name], label=f"A5c evaluation {name}")
    lineage = executable["lineage"]
    assert isinstance(lineage, Mapping)
    source_scorer = evaluation["source_scorer_evaluation"]
    if not isinstance(source_scorer, Mapping):
        raise TypeError("A5c source scorer evaluation must be a mapping")
    validated_source = _validate_source_fold_evaluation(
        source_scorer, label="A5c source scorer"
    )
    source_resources = validated_source["resource_accounting"]
    assert isinstance(source_resources, Mapping)
    expected_source_sha = a5c_source_scorer_evaluation_sha256(source_scorer)
    expected_resource_sha = a5c_resource_accounting_sha256(source_resources)
    if (
        evaluation["source_scorer_evaluation_sha256"] != expected_source_sha
        or evaluation["resource_accounting_sha256"] != expected_resource_sha
    ):
        raise ValueError("A5c source scorer hashes are contradictory")
    logical = _positive_int(
        evaluation["logical_valid_tokens"], label="A5c logical tokens"
    )
    supervised = _positive_int(
        evaluation["supervised_tokens"], label="A5c supervised tokens"
    )
    if supervised > logical:
        raise ValueError("A5c supervised token count exceeds logical rows")
    native = _exact_mapping(
        evaluation["native"], {"nll_per_token"}, label="A5c native metric"
    )
    native_nll = _finite(
        native["nll_per_token"], label="A5c native NLL", minimum=0.0
    )
    raw_conditions = _exact_mapping(
        evaluation["conditions"], _CONDITIONS, label="A5c conditions"
    )
    conditions = {
        name: _validate_condition(raw_conditions[name], label=f"A5c {name}")
        for name in sorted(_CONDITIONS)
    }
    if (
        logical != validated_source["logical_valid_tokens"]
        or supervised != validated_source["supervised_tokens"]
        or _canonical_json_bytes(evaluation["native"])
        != _canonical_json_bytes(validated_source["native"])
    ):
        raise ValueError("A5c compact scorer token/native projection drifted")
    source_conditions = validated_source["conditions"]
    assert isinstance(source_conditions, Mapping)
    for name, source_name in _SOURCE_SCORER_CONDITION_BY_A5C.items():
        compact_metric = {
            field: raw_conditions[name][field]  # type: ignore[index]
            for field in _METRIC_FIELDS
        }
        if _canonical_json_bytes(compact_metric) != _canonical_json_bytes(
            source_conditions[source_name]
        ):
            raise ValueError(f"A5c compact scorer projection drifted for {name}")
    for name, metric in conditions.items():
        if not math.isclose(
            float(metric["delta_nll_per_token"]),
            float(metric["nll_per_token"]) - native_nll,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"A5c {name} delta NLL is contradictory")
    selected = executable["selected"]
    frozen = executable["frozen_reference"]
    assert isinstance(selected, Mapping) and isinstance(frozen, Mapping)
    graph_expectations = {
        "layer10_only": lineage["layer10_graph_sha256"],
        "selected_layer17_only": selected["layer17_graph_sha256"],
        "selected_composition": selected["composition_graph_sha256"],
        "frozen_uncorrected_composition": frozen["composition_graph_sha256"],
        "matched_double_deletion": lineage[
            "matched_double_deletion_graph_sha256"
        ],
    }
    for name, digest in graph_expectations.items():
        if conditions[name]["graph_sha256"] != digest:
            raise ValueError(f"A5c {name} graph hash contradicts executable")
    expected_scorer_receipt = a5c_source_scorer_receipt_sha256(
        source_scorer_evaluation=source_scorer,
        outer_fold_index=outer_fold_index,
        condition_graph_sha256_by_name={
            name: conditions[name]["graph_sha256"] for name in _CONDITIONS
        },
    )
    if evaluation["source_scorer_receipt_sha256"] != expected_scorer_receipt:
        raise ValueError("A5c source scorer receipt is contradictory")
    if executable["kind"] == _FROZEN:
        if conditions["selected_composition"] != conditions[
            "frozen_uncorrected_composition"
        ]:
            raise ValueError("A5c fallback selected/frozen metrics must be exact")
    return {
        "assessment_role": evaluation["assessment_role"],
        "outer_fold_index": outer_fold_index,
        "logical_valid_tokens": logical,
        "supervised_tokens": supervised,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
        "source_scorer_evaluation": dict(source_scorer),
        "source_scorer_receipt_sha256": evaluation[
            "source_scorer_receipt_sha256"
        ],
        "source_scorer_evaluation_sha256": evaluation[
            "source_scorer_evaluation_sha256"
        ],
        "resource_accounting_sha256": evaluation[
            "resource_accounting_sha256"
        ],
        "resource_accounting_reference": evaluation[
            "resource_accounting_reference"
        ],
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
    }


def derive_gemma3_l10_l17_a5c_conclusion(
    *,
    selected_executable: Mapping[str, object],
    outer_evaluation: Mapping[str, object],
    comparison_to_a5b: Mapping[str, object],
) -> dict[str, object]:
    """Derive the bounded A5c conclusion from authenticated scalar metrics."""

    kind = selected_executable["kind"]
    conditions = outer_evaluation["conditions"]
    assert isinstance(conditions, Mapping)
    selected = conditions["selected_composition"]
    frozen = conditions["frozen_uncorrected_composition"]
    a5b = comparison_to_a5b["a5b_learned_composition"]
    assert all(isinstance(value, Mapping) for value in (selected, frozen, a5b))
    return {
        "selected_executable_kind": kind,
        "use_frozen_fallback": kind == _FROZEN,
        "selected_ridge": selected_executable["selected_ridge"],
        "outer_selected_kl": selected["native_to_candidate_kl_per_token"],
        "outer_frozen_kl": frozen["native_to_candidate_kl_per_token"],
        "outer_selected_delta_nll": selected["delta_nll_per_token"],
        "outer_frozen_delta_nll": frozen["delta_nll_per_token"],
        "outer_selected_top1": selected["top1_agreement_to_native"],
        "outer_frozen_top1": frozen["top1_agreement_to_native"],
        "selected_improves_frozen_kl": (
            selected["native_to_candidate_kl_per_token"]
            < frozen["native_to_candidate_kl_per_token"]
        ),
        "selected_improves_frozen_delta_nll": (
            selected["delta_nll_per_token"] < frozen["delta_nll_per_token"]
        ),
        "selected_improves_frozen_top1": (
            selected["top1_agreement_to_native"]
            > frozen["top1_agreement_to_native"]
        ),
        "selected_exactly_matches_frozen": selected == frozen,
        "selected_improves_a5b_learned_kl": (
            selected["native_to_candidate_kl_per_token"]
            < a5b["native_to_candidate_kl_per_token"]
        ),
        "selected_improves_a5b_learned_delta_nll": (
            selected["delta_nll_per_token"] < a5b["delta_nll_per_token"]
        ),
        "selected_improves_a5b_learned_top1": (
            selected["top1_agreement_to_native"]
            > a5b["top1_agreement_to_native"]
        ),
        "one_outer_fold_only": True,
        "does_not_establish_eight_fold_competitive_compilation": True,
    }


def validate_gemma3_l10_l17_a5c_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return one strict source-safe A5c report."""

    if not isinstance(value, Mapping):
        raise TypeError("A5c report must be a mapping")
    _assert_source_safe(value)
    raw = dict(value)
    _exact_mapping(raw, _TOP_FIELDS, label="A5c report")
    if (
        raw["schema"] != GEMMA3_L10_L17_A5C_REPORT_SCHEMA
        or raw["format_version"] != GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION
        or raw["scientific_role"] != _SCIENTIFIC_ROLE
        or raw["full_model_forward_evaluated"] is not True
        or raw["whole_model_compiled"] is not False
        or raw["heldout_confirmation"] is not False
        or raw["serving_authorized"] is not False
        or raw["latency_or_kernel_speed_claim"] is not False
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("A5c report header, scope, or safety drifted")
    _assert_no_sensitive_keys(raw)

    source = _exact_mapping(
        raw["source_bindings"], _SOURCE_FIELDS, label="A5c source bindings"
    )
    for name, digest in source.items():
        _require_sha256(digest, label=f"A5c source {name}")
    if dict(source) != _EXPECTED_SOURCE_BINDINGS:
        raise ValueError("A5c does not bind the canonical A5b v1 lineage")

    runtime = _exact_mapping(raw["runtime"], _RUNTIME_FIELDS, label="A5c runtime")
    if runtime != {
        "model_id": _EXPECTED_MODEL_ID,
        "requested_revision": _EXPECTED_MODEL_REVISION,
        "model_fingerprint": _EXPECTED_MODEL_FINGERPRINT,
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
    }:
        raise ValueError("A5c runtime differs from the canonical replay")

    config = _exact_mapping(
        raw["configuration"], _CONFIG_FIELDS, label="A5c configuration"
    )
    outer_fold = _positive_int(
        config["outer_fold_index"], label="A5c outer fold", minimum=0
    )
    training_examples_per_family = _positive_int(
        config["training_examples_per_family"],
        label="A5c training examples per family",
    )
    training_family_count = _positive_int(
        config["training_family_count"], label="A5c training family count"
    )
    inner_audit_examples_per_family = _positive_int(
        config["inner_audit_examples_per_family"],
        label="A5c audit examples per family",
    )
    target_solver_steps = _positive_int(
        config["target_solver_steps"], label="A5c solver steps"
    )
    target_batch_rows = _positive_int(
        config["target_batch_rows"], label="A5c target batch"
    )
    target_learning_rate = _finite(
        config["target_learning_rate_fraction"],
        label="A5c target learning rate",
        minimum=0.0,
    )
    target_ridge = _finite(
        config["target_ridge"], label="A5c target ridge", minimum=0.0
    )
    generator_rank = _positive_int(
        config["generator_rank"], label="A5c generator rank"
    )
    held_examples_scored = _positive_int(
        config["held_examples_scored"], label="A5c held examples scored"
    )
    if (
        outer_fold != 0
        or training_family_count != 7
        or training_examples_per_family != 4
        or config["row_selection_policy"]
        != "all_valid_captured_rows_per_example"
        or inner_audit_examples_per_family != 1
        or target_solver_steps != 64
        or target_batch_rows != 8
        or target_learning_rate != 1.0e-2
        or target_ridge != 0.0
        or config["target_trust_radius"] is not None
        or generator_rank != 16
        or held_examples_scored != 1
    ):
        raise ValueError("A5c fixed configuration drifted")
    ridge_grid_raw = config["ridge_grid"]
    if (
        not isinstance(ridge_grid_raw, list)
        or ridge_grid_raw != list(A5C_RIDGE_GRID)
    ):
        raise ValueError("A5c ridge grid differs from the fixed canonical grid")
    ridge_grid = tuple(
        _finite(item, label="A5c ridge candidate", minimum=0.0)
        for item in ridge_grid_raw
    )

    capture = _exact_mapping(raw["capture"], _CAPTURE_FIELDS, label="A5c capture")
    for name in (
        "capture_sha256",
        "capture_audit_sha256",
        "source_row_catalog_sha256",
    ):
        _require_sha256(capture[name], label=f"A5c capture {name}")
    training_examples = _positive_int(
        capture["training_example_count"], label="A5c training examples"
    )
    captured_rows = _positive_int(
        capture["captured_observation_count"], label="A5c captured rows"
    )
    target_rows = _positive_int(
        capture["selected_target_row_count"], label="A5c target rows"
    )
    if (
        capture["training_family_count"] != 7
        or training_examples
        != 7 * int(config["training_examples_per_family"])
        or target_rows != captured_rows
        or capture["outer_held_family_rows_present"] is not False
        or capture["all_required_capture_audits_pass"] is not True
    ):
        raise ValueError("A5c capture accounting or ownership drifted")

    target = _validate_receipt_ref(
        raw["target_solve"], _TARGET_FIELDS, label="A5c target solve"
    )
    if _positive_int(target["row_count"], label="A5c target solve rows") != target_rows:
        raise ValueError("A5c target solve row count drifted")
    _require_sha256(
        target["selected_coordinate_sha256"], label="A5c selected coordinates"
    )

    row_bank = _validate_receipt_ref(
        raw["coordinate_row_bank"], _ROW_BANK_FIELDS, label="A5c coordinate bank"
    )
    for name in (
        "row_key_sha256",
        "compiled_inputs_sha256",
        "selected_coordinates_sha256",
    ):
        _require_sha256(row_bank[name], label=f"A5c coordinate bank {name}")
    if (
        _positive_int(row_bank["row_count"], label="A5c row-bank rows")
        != target_rows
        or _positive_int(row_bank["example_count"], label="A5c row-bank examples")
        != training_examples
        or row_bank["selected_coordinates_sha256"]
        != target["selected_coordinate_sha256"]
        or row_bank["outer_held_family_rows_present"] is not False
    ):
        raise ValueError("A5c coordinate row-bank lineage drifted")

    breadth = _validate_receipt_ref(
        raw["breadth_split"], _BREADTH_FIELDS, label="A5c breadth split"
    )
    all_rows = _positive_int(breadth["all_row_count"], label="A5c breadth rows")
    fit_rows = _positive_int(breadth["fit_row_count"], label="A5c fit rows")
    audit_rows = _positive_int(breadth["audit_row_count"], label="A5c audit rows")
    removed = _positive_int(
        breadth["removed_fit_rows_for_signature_overlap"],
        label="A5c removed overlap rows",
        minimum=0,
    )
    all_examples = _positive_int(
        breadth["all_example_count"], label="A5c breadth examples"
    )
    fit_examples = _positive_int(
        breadth["fit_example_count"], label="A5c fit examples"
    )
    audit_examples = _positive_int(
        breadth["audit_example_count"], label="A5c audit examples"
    )
    fully_removed_examples = _positive_int(
        breadth["fit_examples_fully_removed_for_signature_overlap"],
        label="A5c fully removed fit examples",
        minimum=0,
    )
    if (
        all_rows != target_rows
        or fit_rows + audit_rows + removed != all_rows
        or all_examples != training_examples
        or fit_examples + audit_examples + fully_removed_examples
        != all_examples
        or breadth["fit_audit_example_overlap_count"] != 0
        or breadth["post_purge_input_signature_overlap_count"] != 0
        or breadth["outer_held_family_rows_present"] is not False
    ):
        raise ValueError("A5c breadth split is not disjoint or exhaustive")

    ridge_cv = _validate_receipt_ref(raw["ridge_cv"], _CV_FIELDS, label="A5c ridge CV")
    if (
        ridge_cv["candidate_count"] != len(ridge_grid)
        or ridge_cv["inner_fold_count"] != 7
        or ridge_cv["outer_held_family_accessed"] is not False
        or type(ridge_cv["use_frozen_fallback"]) is not bool
    ):
        raise ValueError("A5c ridge-CV scope drifted")
    cv_ridge = ridge_cv["selected_ridge"]
    if ridge_cv["use_frozen_fallback"]:
        if cv_ridge is not None:
            raise ValueError("A5c fallback CV cannot expose a selected ridge")
    elif _finite(cv_ridge, label="A5c CV ridge", minimum=0.0) not in ridge_grid:
        raise ValueError("A5c CV selected ridge is outside its grid")

    executable = _validate_selected_executable(
        raw["selected_executable"],
        target=target,
        row_bank=row_bank,
        breadth=breadth,
        ridge_cv=ridge_cv,
        ridge_grid=ridge_grid,
    )
    _validate_evidence_receipts(
        raw["evidence_receipts"],
        configuration=config,
        capture=capture,
        target_summary=target,
        row_bank_summary=row_bank,
        breadth_summary=breadth,
        cv_summary=ridge_cv,
        runtime=runtime,
        executable=executable,
    )
    evaluation = _validate_evaluation(
        raw["outer_evaluation"],
        outer_fold_index=outer_fold,
        executable=executable,
    )
    evaluation_sha = a5c_outer_evaluation_sha256(raw["outer_evaluation"])

    chronology = _exact_mapping(
        raw["chronology"], _CHRONOLOGY_FIELDS, label="A5c chronology"
    )
    events = tuple(
        _positive_int(chronology[name], label=f"A5c chronology {name}")
        for name in (
            "ridge_cv_completed_event",
            "executable_frozen_event",
            "outer_held_batch_selected_event",
            "outer_held_model_evaluated_event",
        )
    )
    if (
        events != tuple(sorted(events))
        or len(set(events)) != 4
        or chronology[
            "outer_held_batch_selected_or_scored_before_freeze"
        ]
        is not False
        or chronology["executable_frozen_before_outer_held_batch_selection"]
        is not True
        or chronology["executable_frozen_before_outer_held_model_evaluation"]
        is not True
        or chronology["ridge_cv_receipt_sha256"] != ridge_cv["receipt_sha256"]
        or chronology["selection_freeze_sha256"]
        != executable["selection_freeze_sha256"]
        or chronology["outer_evaluation_sha256"] != evaluation_sha
    ):
        raise ValueError("A5c freeze-before-held chronology is contradictory")

    comparison = _exact_mapping(
        raw["comparison_to_a5b"], _COMPARISON_FIELDS, label="A5c comparison"
    )
    if (
        comparison["a5b_report_sha256"] != _EXPECTED_A5B_REPORT_SHA256
        or comparison["same_outer_fold"] is not True
        or comparison["same_held_example_policy"] is not True
    ):
        raise ValueError("A5c comparison is not aligned to canonical A5b")
    a5b_metric = _validate_quality_metric(
        comparison["a5b_learned_composition"], label="A5b learned composition"
    )
    if a5b_metric != _EXPECTED_A5B_LEARNED_COMPOSITION:
        raise ValueError("A5c comparison metrics differ from canonical A5b")
    comparison_validated = {**dict(comparison), "a5b_learned_composition": a5b_metric}
    expected_conclusion = derive_gemma3_l10_l17_a5c_conclusion(
        selected_executable=executable,
        outer_evaluation=evaluation,
        comparison_to_a5b=comparison_validated,
    )
    if raw["conclusion"] != expected_conclusion:
        raise ValueError("A5c conclusion contradicts its evaluation")

    supplied = _require_sha256(raw["report_sha256"], label="A5c report")
    payload = dict(raw)
    payload.pop("report_sha256")
    if supplied != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A5c report hash mismatch")
    return json.loads(json.dumps(raw, allow_nan=False))


def build_gemma3_l10_l17_a5c_report(
    *,
    source_bindings: Mapping[str, object],
    runtime: Mapping[str, object],
    configuration: Mapping[str, object],
    capture: Mapping[str, object],
    target_solve: Mapping[str, object],
    coordinate_row_bank: Mapping[str, object],
    breadth_split: Mapping[str, object],
    ridge_cv: Mapping[str, object],
    evidence_receipts: Mapping[str, object],
    selected_executable: Mapping[str, object],
    chronology: Mapping[str, object],
    outer_evaluation: Mapping[str, object],
    comparison_to_a5b: Mapping[str, object],
) -> dict[str, object]:
    """Build one strict A5c report from compact tensor-free inputs."""

    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5C_REPORT_SCHEMA,
        "format_version": GEMMA3_L10_L17_A5C_REPORT_FORMAT_VERSION,
        "scientific_role": _SCIENTIFIC_ROLE,
        "source_bindings": dict(source_bindings),
        "runtime": dict(runtime),
        "configuration": dict(configuration),
        "capture": dict(capture),
        "target_solve": dict(target_solve),
        "coordinate_row_bank": dict(coordinate_row_bank),
        "breadth_split": dict(breadth_split),
        "ridge_cv": dict(ridge_cv),
        "evidence_receipts": dict(evidence_receipts),
        "selected_executable": dict(selected_executable),
        "chronology": dict(chronology),
        "outer_evaluation": dict(outer_evaluation),
        "comparison_to_a5b": dict(comparison_to_a5b),
        "full_model_forward_evaluated": True,
        "whole_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": dict(_SAFETY),
    }
    # Validate all caller-supplied material before deriving a conclusion.  A
    # temporary syntactically valid conclusion/report hash lets the strict
    # validator authenticate every independent section exactly once below.
    executable = dict(selected_executable)
    evaluation = dict(outer_evaluation)
    comparison = dict(comparison_to_a5b)
    payload["conclusion"] = derive_gemma3_l10_l17_a5c_conclusion(
        selected_executable=executable,
        outer_evaluation=evaluation,
        comparison_to_a5b=comparison,
    )
    payload["report_sha256"] = _sha256(_REPORT_DOMAIN, payload)
    return validate_gemma3_l10_l17_a5c_report(payload)


def save_gemma3_l10_l17_a5c_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Atomically publish a validated A5c report without overwriting."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5c report")
    validated = validate_gemma3_l10_l17_a5c_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        validated,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if destination.exists():
            raise FileExistsError("refusing to overwrite A5c report")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_a5c_report(path: Path | str) -> dict[str, object]:
    """Load one strict-JSON A5c report and authenticate every field."""

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("A5c report is not strict JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("A5c report must contain one object")
    return validate_gemma3_l10_l17_a5c_report(raw)
