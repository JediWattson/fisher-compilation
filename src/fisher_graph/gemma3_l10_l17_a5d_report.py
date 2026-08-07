"""Strict source-safe report contract for the A5d residual-generator rung.

The report is deliberately a publication boundary rather than an experiment
runner.  It accepts only authenticated, tensor-free receipts, one frozen
source-owning executable description, and scalar full-model measurements.
The Layer-17 source graph always remains the owning graph.  A learned A5d
candidate may add one zero-mean residual graph after the feed-forward RMSNorm;
it may never replace or mutate the source owner.
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

from .gemma3_l10_l17_a5d_family_residual_cv import (
    A5D_ALPHA_GRID,
    A5D_FIXED_GENERATOR_RANK,
    A5D_RIDGE_GRID,
    GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA,
    validate_a5d_family_residual_cv_receipt,
)
from .gemma3_l10_l17_a5d_source_anchored_residual import (
    GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA,
    validate_a5d_source_anchored_residual_receipt,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _MINIMUM_NUMERICAL_KL_PER_TOKEN,
)


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_A5D_REPORT_OUTPUT",
    "GEMMA3_L10_L17_A5D_REPORT_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5D_REPORT_SCHEMA",
    "a5d_outer_evaluation_sha256",
    "a5d_selection_freeze_sha256",
    "build_gemma3_l10_l17_a5d_report",
    "compact_a5d_family_residual_cv_receipt",
    "compact_a5d_source_anchored_residual_receipt",
    "derive_gemma3_l10_l17_a5d_conclusion",
    "load_gemma3_l10_l17_a5d_report",
    "save_gemma3_l10_l17_a5d_report",
    "validate_gemma3_l10_l17_a5d_report",
]


GEMMA3_L10_L17_A5D_REPORT_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5d_source_anchored_residual_generator"
)
GEMMA3_L10_L17_A5D_REPORT_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A5D_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a5d-source-anchored-residual-generator-v1.json"
)

_EXPECTED_MODEL_ID = "google/gemma-3-270m"
_EXPECTED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_EXPECTED_MODEL_FINGERPRINT = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_EXPECTED_A5C_FILE_SHA256 = (
    "10a0389f6c6a91893697fcb915edd1917ba4966a4e30b8f9870caed10f43a840"
)
_EXPECTED_A5C_REPORT_SHA256 = (
    "94238d934e9c5db4e6ddbb67eb9a4426c70cb1d12684c4ffc25777b62f437fe7"
)
_EXPECTED_A5C_CAPTURE = {
    "capture_sha256": (
        "668d570e98809ee5311a7d3f2378114603d377565d805c54c1b165c0b95d8bf6"
    ),
    "capture_audit_sha256": (
        "2497a4471d2a00c53f9649866e15d24a8fa29d9ebd4b37ac28b3163083c3b031"
    ),
    "source_row_catalog_sha256": (
        "274e0951614051af6ee1a4932c9fb308e68970a67e176eaad9122c46d8439398"
    ),
    "training_family_count": 7,
    "training_example_count": 28,
    "captured_observation_count": 1_798,
    "outer_held_family_rows_present": False,
    "all_required_capture_audits_pass": True,
}
_EXPECTED_TARGET_SOLVE_RECEIPT_SHA256 = (
    "0019bfa08625fcd63d5c796cb320a7d1fd0bcf6b262db965a903bad6a205fdeb"
)
_EXPECTED_COORDINATE_ROW_BANK_RECEIPT_SHA256 = (
    "74b4d6193c12145b197c51f6a3c0f61c6a17cf23ea0beeb8724280675a1e066c"
)
_EXPECTED_BREADTH_SPLIT_RECEIPT_SHA256 = (
    "a1832bde5704dcfde5c2d77aac93d309861584464424f876e635d9d63fe24230"
)
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
_EXPECTED_COMPOSITION_GRAPH_SHA256 = (
    "35d35f2318e0728bb649f2825601d2edbe06e13307a412c4d81c66e8e387c4ca"
)
_EXPECTED_A5C_COMPOSITION_METRIC = {
    "nll_per_token": 7.300289344787598,
    "delta_nll_per_token": 0.14230638231549975,
    "native_to_candidate_kl_per_token": 0.10553961873443381,
    "top1_agreement_to_native": 0.7714285714285715,
}

_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-report:v1\0"
_FREEZE_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-freeze:v1\0"
_EVALUATION_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-evaluation:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_FROZEN = "frozen_source_fallback"
_ADDITIVE = "additive_zero_mean_residual"
_EXECUTABLE_KINDS = {_FROZEN, _ADDITIVE}
_OUTPUT_BOUNDARY = "layer.17.mlp.delta"
_APPLICATION_ORDER = "post_feedforward_rmsnorm_then_scaled_additive_residual"
_SCIENTIFIC_ROLE = (
    "calibration_a_one_outer_fold_source_anchored_residual_generator"
)
_ASSESSMENT_ROLE = (
    "calibration_a_outer_family_bounded_source_anchored_residual"
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

_SOURCE_FIELDS = {"a5c_file_sha256", "a5c_report_sha256"}
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
    "generator_rank",
    "ridge_grid",
    "alpha_grid",
    "held_examples_scored",
    "output_boundary",
    "final_head_chunk_rows",
}
_CAPTURE_FIELDS = set(_EXPECTED_A5C_CAPTURE)
_TARGET_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "observation_count",
    "sequence_count",
    "residual_width",
    "joint_coordinate_width",
    "node_order",
    "source_mapping_preserved_at_alpha_zero",
    "projection_uses_affine_means",
    "source_affine_means_injected",
    "contains_tensor_payloads",
}
_CV_FIELDS = {
    "receipt_schema",
    "receipt_sha256",
    "ridge_candidate_count",
    "alpha_candidate_count",
    "inner_fold_count",
    "selected_alpha",
    "selected_ridge",
    "use_frozen_fallback",
    "final_residual_graph_sha256",
    "final_residual_lowering_sha256_by_node",
    "final_residual_parameter_count",
    "final_residual_macs_per_token",
    "removed_source_affine_mean_parameters",
    "outer_held_family_accessed",
    "contains_tensor_payloads",
}
_EVIDENCE_FIELDS = {"source_anchored_residual", "residual_cv"}
_RESOURCE_FIELDS = {
    "node_count",
    "interaction_count",
    "parameter_count",
    "macs_per_token",
    "additions_per_token",
    "peak_live_modal_width",
}
_SOURCE_OWNER_FIELDS = {
    "layer17_graph_sha256",
    "layer17_lowering_sha256_by_node",
    "composition_graph_sha256",
    "layer17_resources",
    "composition_resources",
}
_ADDITIVE_FIELDS = {
    "graph_sha256",
    "lowering_sha256_by_node",
    "application_layer_ordinal",
    "basis_means_exactly_zero",
    "source_decoders_reused",
    "source_affine_means_reinjected",
    "resources",
}
_TOTAL_RESOURCE_FIELDS = {"layer17_scope", "composition_scope"}
_LINEAGE_FIELDS = {
    "a5c_report_sha256",
    "capture_sha256",
    "target_solve_receipt_sha256",
    "coordinate_row_bank_receipt_sha256",
    "breadth_split_receipt_sha256",
    "source_anchored_residual_receipt_sha256",
    "residual_cv_receipt_sha256",
    "layer10_graph_sha256",
    "layer10_lowering_sha256_by_node",
    "matched_double_deletion_graph_sha256",
}
_EXECUTABLE_FIELDS = {
    "kind",
    "selected_alpha",
    "selected_alpha_hex",
    "selected_ridge",
    "selected_ridge_hex",
    "application_boundary",
    "application_order",
    "source_ownership_preserved",
    "source_affine_means_reinjected",
    "lineage",
    "source_owner",
    "additive_residual",
    "selected_resources",
    "selection_freeze_sha256",
}
_CHRONOLOGY_FIELDS = {
    "residual_cv_completed_event",
    "executable_frozen_event",
    "outer_held_batch_selected_event",
    "outer_held_model_evaluated_event",
    "outer_held_batch_selected_or_scored_before_freeze",
    "executable_frozen_before_outer_held_batch_selection",
    "executable_frozen_before_outer_held_model_evaluation",
    "residual_cv_receipt_sha256",
    "selection_freeze_sha256",
    "outer_evaluation_sha256",
}
_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_CONDITION_FIELDS = {
    *_METRIC_FIELDS,
    "owning_graph_sha256",
    "additive_graph_sha256",
}
_CONDITIONS = {
    "layer10_only",
    "selected_layer17_only",
    "frozen_uncorrected_composition",
    "selected_composition",
    "matched_double_deletion",
}
_EXECUTION_RESOURCE_FIELDS = {
    "replaced_layer_count",
    "owning_graph_node_count",
    "additive_graph_node_count",
    "total_graph_node_count",
    "owning_interaction_count",
    "additive_interaction_count",
    "total_interaction_count",
    "native_removed_parameters",
    "owning_graph_parameters",
    "additive_graph_parameters",
    "total_graph_parameters",
    "candidate_whole_model_learned_parameters",
    "net_parameter_savings",
    "owning_dense_graph_macs_per_token",
    "additive_dense_graph_macs_per_token",
    "total_dense_graph_macs_per_token",
    "executed_graph_macs_per_token",
    "net_executed_macs_saved_per_token",
    "owning_dense_graph_additions_per_token",
    "additive_dense_graph_additions_per_token",
    "total_dense_graph_additions_per_token",
    "executed_graph_additions_per_token",
    "executed_peak_live_modal_width",
}
_EVALUATION_FIELDS = {
    "assessment_role",
    "outer_fold_index",
    "logical_valid_tokens",
    "supervised_tokens",
    "source_whole_model_learned_parameters",
    "native",
    "conditions",
    "resource_accounting",
    "full_model_logits_scored",
    "full_model_compiled",
    "heldout_confirmation",
    "exact_resources_match_frozen_executable",
    "latency_or_kernel_speed_claim",
}
_COMPARISON_FIELDS = {
    "a5c_file_sha256",
    "a5c_report_sha256",
    "same_outer_fold",
    "same_held_example_policy",
    "a5c_frozen_composition",
}
_TOP_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "source_bindings",
    "runtime",
    "configuration",
    "capture",
    "source_anchored_residual",
    "residual_cv",
    "evidence_receipts",
    "selected_executable",
    "chronology",
    "outer_evaluation",
    "comparison_to_a5c",
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
    value: object, fields: set[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{label} is outside its finite range")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
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
        "hidden_states",
        "coordinates",
        "coordinate_tensor",
        "coordinate_tensors",
        "source_model_weights",
        "generator_weights",
        "parameter_tensors",
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


def _hash_catalog(
    value: object,
    *,
    label: str,
    expected_names: set[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a hash catalog")
    if expected_names is not None and set(value) != expected_names:
        raise ValueError(f"{label} node catalog is incomplete")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} contains an invalid node name")
        result[name] = _require_sha256(digest, label=f"{label} {name}")
    return result


def compact_a5d_source_anchored_residual_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and compact one A5d residual-target receipt."""

    receipt = validate_a5d_source_anchored_residual_receipt(value)
    construction = receipt["construction"]
    rows = receipt["rows"]
    assert isinstance(construction, Mapping) and isinstance(rows, Mapping)
    return {
        "receipt_schema": receipt["schema"],
        "receipt_sha256": receipt["receipt_sha256"],
        "observation_count": construction["observations"],
        "sequence_count": construction["sequences"],
        "residual_width": construction["residual_width"],
        "joint_coordinate_width": construction["joint_coordinate_width"],
        "node_order": list(construction["node_order"]),
        "source_mapping_preserved_at_alpha_zero": construction[
            "source_mapping_preserved_at_alpha_zero"
        ],
        "projection_uses_affine_means": construction[
            "projection_uses_affine_means"
        ],
        "source_affine_means_injected": rows["source_affine_means_injected"],
        "contains_tensor_payloads": False,
    }


def compact_a5d_family_residual_cv_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and compact one family-disjoint A5d selection receipt."""

    receipt = validate_a5d_family_residual_cv_receipt(value)
    config = receipt["configuration"]
    ownership = receipt["ownership"]
    selection = receipt["selection"]
    final_refit = receipt["final_refit"]
    assert all(
        isinstance(item, Mapping)
        for item in (config, ownership, selection, final_refit)
    )
    fit = final_refit["fit"]
    if fit is None:
        graph_sha256 = None
        lowerings: dict[str, str] = {}
        parameter_count = None
        macs_per_token = None
        removed_mean_parameters = None
    else:
        assert isinstance(fit, Mapping)
        graph_sha256 = fit["graph_sha256"]
        raw_lowerings = fit["lowering_sha256_by_node"]
        assert isinstance(raw_lowerings, Mapping)
        lowerings = dict(raw_lowerings)
        parameter_count = fit["parameter_count"]
        macs_per_token = fit["macs_per_token"]
        removed_mean_parameters = fit["removed_source_affine_mean_parameters"]
    return {
        "receipt_schema": receipt["schema"],
        "receipt_sha256": receipt["receipt_sha256"],
        "ridge_candidate_count": len(config["ridge_grid"]),
        "alpha_candidate_count": len(config["alpha_grid"]),
        "inner_fold_count": config["inner_fold_count"],
        "selected_alpha": selection["selected_alpha"],
        "selected_ridge": selection["selected_ridge"],
        "use_frozen_fallback": selection["use_frozen_fallback"],
        "final_residual_graph_sha256": graph_sha256,
        "final_residual_lowering_sha256_by_node": lowerings,
        "final_residual_parameter_count": parameter_count,
        "final_residual_macs_per_token": macs_per_token,
        "removed_source_affine_mean_parameters": removed_mean_parameters,
        "outer_held_family_accessed": ownership[
            "outer_held_family_states_or_rows_accessed"
        ],
        "contains_tensor_payloads": False,
    }


def _validate_target_summary(value: object) -> dict[str, object]:
    target = _exact_mapping(value, _TARGET_FIELDS, label="A5d residual target")
    if (
        target["receipt_schema"]
        != GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA
        or _integer(
            target["observation_count"], label="A5d target observations", minimum=1
        )
        != _EXPECTED_A5C_CAPTURE["captured_observation_count"]
        or _integer(target["sequence_count"], label="A5d target sequences", minimum=1)
        != _EXPECTED_A5C_CAPTURE["training_example_count"]
        or _integer(target["residual_width"], label="A5d residual width", minimum=1)
        <= 0
        or target["joint_coordinate_width"] != 182
        or target["source_mapping_preserved_at_alpha_zero"] is not True
        or target["projection_uses_affine_means"] is not False
        or target["source_affine_means_injected"] is not False
        or target["contains_tensor_payloads"] is not False
    ):
        raise ValueError("A5d residual-target summary drifted")
    _require_sha256(target["receipt_sha256"], label="A5d target receipt")
    names = target["node_order"]
    if (
        not isinstance(names, list)
        or len(names) != 4
        or len(set(names)) != 4
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("A5d residual-target node order is invalid")
    return dict(target)


def _validate_cv_summary(value: object) -> dict[str, object]:
    cv = _exact_mapping(value, _CV_FIELDS, label="A5d residual CV")
    if (
        cv["receipt_schema"] != GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA
        or cv["ridge_candidate_count"] != len(A5D_RIDGE_GRID)
        or cv["alpha_candidate_count"] != len(A5D_ALPHA_GRID)
        or cv["inner_fold_count"] != 7
        or type(cv["use_frozen_fallback"]) is not bool
        or cv["outer_held_family_accessed"] is not False
        or cv["contains_tensor_payloads"] is not False
    ):
        raise ValueError("A5d residual-CV summary drifted")
    _require_sha256(cv["receipt_sha256"], label="A5d CV receipt")
    alpha = _finite(
        cv["selected_alpha"], label="A5d selected alpha", minimum=0.0, maximum=1.0
    )
    if alpha not in A5D_ALPHA_GRID:
        raise ValueError("A5d selected alpha is outside the fixed grid")
    lowerings = _hash_catalog(
        cv["final_residual_lowering_sha256_by_node"],
        label="A5d final residual lowerings",
        allow_empty=True,
    )
    if cv["use_frozen_fallback"]:
        if (
            alpha != 0.0
            or cv["selected_ridge"] is not None
            or cv["final_residual_graph_sha256"] is not None
            or lowerings
            or cv["final_residual_parameter_count"] is not None
            or cv["final_residual_macs_per_token"] is not None
            or cv["removed_source_affine_mean_parameters"] is not None
        ):
            raise ValueError("A5d alpha-zero fallback retained a residual fit")
    else:
        ridge = _finite(cv["selected_ridge"], label="A5d selected ridge", minimum=0.0)
        if (
            alpha <= 0.0
            or ridge not in A5D_RIDGE_GRID
            or len(lowerings) != 4
            or cv["final_residual_parameter_count"] != 160_534
            or cv["final_residual_macs_per_token"] != 160_352
            or cv["removed_source_affine_mean_parameters"] != 2_560
        ):
            raise ValueError("A5d positive selection is incomplete")
        _require_sha256(
            cv["final_residual_graph_sha256"], label="A5d residual graph"
        )
    return {**dict(cv), "final_residual_lowering_sha256_by_node": lowerings}


def _validate_resource(
    value: object,
    *,
    label: str,
    expected: Mapping[str, int] | None = None,
) -> dict[str, int]:
    resource = _exact_mapping(value, _RESOURCE_FIELDS, label=label)
    validated = {
        name: _integer(resource[name], label=f"{label} {name}")
        for name in _RESOURCE_FIELDS
    }
    if validated["node_count"] <= 0 or validated["parameter_count"] <= 0:
        raise ValueError(f"{label} must describe a nonempty graph")
    if validated["macs_per_token"] <= 0:
        raise ValueError(f"{label} MAC count must be positive")
    if expected is not None and any(
        validated[name] != expected_value
        for name, expected_value in expected.items()
    ):
        raise ValueError(f"{label} differs from the canonical source graph")
    return validated


def _validate_source_owner(value: object) -> dict[str, object]:
    owner = _exact_mapping(value, _SOURCE_OWNER_FIELDS, label="A5d source owner")
    if (
        owner["layer17_graph_sha256"] != _EXPECTED_LAYER17_GRAPH_SHA256
        or owner["composition_graph_sha256"]
        != _EXPECTED_COMPOSITION_GRAPH_SHA256
    ):
        raise ValueError("A5d source owner graph identity drifted")
    lowerings = _hash_catalog(
        owner["layer17_lowering_sha256_by_node"],
        label="A5d source Layer17 lowerings",
        expected_names=set(_EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE),
    )
    if lowerings != _EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE:
        raise ValueError("A5d source owner lowerings drifted")
    layer17 = _validate_resource(
        owner["layer17_resources"],
        label="A5d source Layer17 resources",
        expected={
            "node_count": 4,
            "interaction_count": 0,
            "parameter_count": 163_094,
            "macs_per_token": 160_352,
        },
    )
    composition = _validate_resource(
        owner["composition_resources"],
        label="A5d source composition resources",
        expected={
            "node_count": 8,
            "interaction_count": 3,
            "parameter_count": 295_129,
            "macs_per_token": 289_600,
        },
    )
    return {
        **dict(owner),
        "layer17_lowering_sha256_by_node": lowerings,
        "layer17_resources": layer17,
        "composition_resources": composition,
    }


def _validate_additive(
    value: object, *, cv: Mapping[str, object]
) -> dict[str, object] | None:
    if value is None:
        return None
    additive = _exact_mapping(value, _ADDITIVE_FIELDS, label="A5d additive residual")
    graph = _require_sha256(additive["graph_sha256"], label="A5d additive graph")
    names = set(_EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE)
    lowerings = _hash_catalog(
        additive["lowering_sha256_by_node"],
        label="A5d additive lowerings",
        expected_names=names,
    )
    if (
        additive["application_layer_ordinal"] != 17
        or additive["basis_means_exactly_zero"] is not True
        or additive["source_decoders_reused"] is not True
        or additive["source_affine_means_reinjected"] is not False
        or graph != cv["final_residual_graph_sha256"]
        or lowerings != cv["final_residual_lowering_sha256_by_node"]
    ):
        raise ValueError("A5d additive residual semantics drifted")
    resources = _validate_resource(
        additive["resources"],
        label="A5d additive resources",
        expected={
            "node_count": 4,
            "interaction_count": 0,
            "parameter_count": 160_534,
            "macs_per_token": 160_352,
        },
    )
    if (
        resources["parameter_count"] != cv["final_residual_parameter_count"]
        or resources["macs_per_token"] != cv["final_residual_macs_per_token"]
        or cv["removed_source_affine_mean_parameters"] != 2_560
    ):
        raise ValueError("A5d additive resources contradict residual-CV fit")
    return {
        **dict(additive),
        "lowering_sha256_by_node": lowerings,
        "resources": resources,
    }


def _sum_resource(
    owner: Mapping[str, int], additive: Mapping[str, int] | None
) -> dict[str, int]:
    if additive is None:
        return dict(owner)
    return {
        name: int(owner[name]) + int(additive[name])
        for name in _RESOURCE_FIELDS
    }


def _validate_lineage(value: object) -> dict[str, object]:
    lineage = _exact_mapping(value, _LINEAGE_FIELDS, label="A5d lineage")
    lowerings = _hash_catalog(
        lineage["layer10_lowering_sha256_by_node"],
        label="A5d Layer10 lowerings",
        expected_names=set(_EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE),
    )
    for name in _LINEAGE_FIELDS - {"layer10_lowering_sha256_by_node"}:
        _require_sha256(lineage[name], label=f"A5d lineage {name}")
    expected = {
        "a5c_report_sha256": _EXPECTED_A5C_REPORT_SHA256,
        "capture_sha256": _EXPECTED_A5C_CAPTURE["capture_sha256"],
        "target_solve_receipt_sha256": _EXPECTED_TARGET_SOLVE_RECEIPT_SHA256,
        "coordinate_row_bank_receipt_sha256": (
            _EXPECTED_COORDINATE_ROW_BANK_RECEIPT_SHA256
        ),
        "breadth_split_receipt_sha256": _EXPECTED_BREADTH_SPLIT_RECEIPT_SHA256,
        "layer10_graph_sha256": _EXPECTED_LAYER10_GRAPH_SHA256,
        "matched_double_deletion_graph_sha256": (
            _EXPECTED_COMPOSITION_GRAPH_SHA256
        ),
    }
    if any(lineage[name] != digest for name, digest in expected.items()):
        raise ValueError("A5d lineage differs from canonical A5c source")
    if lowerings != _EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE:
        raise ValueError("A5d Layer10 lineage drifted")
    return {**dict(lineage), "layer10_lowering_sha256_by_node": lowerings}


def a5d_selection_freeze_sha256(
    *,
    kind: str,
    selected_alpha: float,
    selected_ridge: float | None,
    application_boundary: str,
    application_order: str,
    source_ownership_preserved: bool,
    source_affine_means_reinjected: bool,
    lineage: Mapping[str, object],
    source_owner: Mapping[str, object],
    additive_residual: Mapping[str, object] | None,
    selected_resources: Mapping[str, object],
) -> str:
    """Hash every pre-held field that can alter A5d execution."""

    if kind not in _EXECUTABLE_KINDS:
        raise ValueError("A5d executable kind is invalid")
    alpha = _finite(
        selected_alpha, label="A5d freeze alpha", minimum=0.0, maximum=1.0
    )
    if alpha not in A5D_ALPHA_GRID:
        raise ValueError("A5d freeze alpha is outside the fixed grid")
    ridge = None
    if selected_ridge is not None:
        ridge = _finite(selected_ridge, label="A5d freeze ridge", minimum=0.0)
        if ridge not in A5D_RIDGE_GRID:
            raise ValueError("A5d freeze ridge is outside the fixed grid")
    if (
        application_boundary != _OUTPUT_BOUNDARY
        or application_order != _APPLICATION_ORDER
        or source_ownership_preserved is not True
        or source_affine_means_reinjected is not False
    ):
        raise ValueError("A5d freeze application semantics drifted")
    validated_lineage = _validate_lineage(lineage)
    owner = _validate_source_owner(source_owner)
    cv_stub = {
        "final_residual_graph_sha256": (
            None if additive_residual is None else additive_residual.get("graph_sha256")
        ),
        "final_residual_lowering_sha256_by_node": (
            {}
            if additive_residual is None
            else additive_residual.get("lowering_sha256_by_node")
        ),
        "final_residual_parameter_count": (
            None
            if additive_residual is None
            else additive_residual.get("resources", {}).get("parameter_count")
        ),
        "final_residual_macs_per_token": (
            None
            if additive_residual is None
            else additive_residual.get("resources", {}).get("macs_per_token")
        ),
        "removed_source_affine_mean_parameters": (
            None if additive_residual is None else 2_560
        ),
    }
    additive = _validate_additive(additive_residual, cv=cv_stub)
    totals = _exact_mapping(
        selected_resources, _TOTAL_RESOURCE_FIELDS, label="A5d selected resources"
    )
    layer17_total = _validate_resource(
        totals["layer17_scope"], label="A5d selected Layer17 resources"
    )
    composition_total = _validate_resource(
        totals["composition_scope"], label="A5d selected composition resources"
    )
    additive_resources = None if additive is None else additive["resources"]
    assert additive_resources is None or isinstance(additive_resources, Mapping)
    if (
        layer17_total
        != _sum_resource(owner["layer17_resources"], additive_resources)
        or composition_total
        != _sum_resource(owner["composition_resources"], additive_resources)
    ):
        raise ValueError("A5d selected resources do not sum owner plus additive")
    if kind == _FROZEN:
        if alpha != 0.0 or ridge is not None or additive is not None:
            raise ValueError("A5d fallback must omit the additive residual exactly")
    elif alpha <= 0.0 or ridge is None or additive is None:
        raise ValueError("A5d positive residual executable is incomplete")
    payload = {
        "kind": kind,
        "selected_alpha": alpha,
        "selected_alpha_hex": alpha.hex(),
        "selected_ridge": ridge,
        "selected_ridge_hex": None if ridge is None else ridge.hex(),
        "application_boundary": application_boundary,
        "application_order": application_order,
        "source_ownership_preserved": True,
        "source_affine_means_reinjected": False,
        "lineage": validated_lineage,
        "source_owner": owner,
        "additive_residual": additive,
        "selected_resources": {
            "layer17_scope": layer17_total,
            "composition_scope": composition_total,
        },
    }
    return _sha256(_FREEZE_DOMAIN, payload)


def _validate_selected_executable(
    value: object,
    *,
    target: Mapping[str, object],
    cv: Mapping[str, object],
) -> dict[str, object]:
    executable = _exact_mapping(value, _EXECUTABLE_FIELDS, label="A5d executable")
    kind = executable["kind"]
    if kind not in _EXECUTABLE_KINDS:
        raise ValueError("A5d executable kind is invalid")
    alpha = _finite(
        executable["selected_alpha"],
        label="A5d executable alpha",
        minimum=0.0,
        maximum=1.0,
    )
    ridge = executable["selected_ridge"]
    if ridge is not None:
        ridge = _finite(ridge, label="A5d executable ridge", minimum=0.0)
    if (
        executable["selected_alpha_hex"] != alpha.hex()
        or executable["selected_ridge_hex"]
        != (None if ridge is None else ridge.hex())
        or cv["selected_alpha"] != alpha
        or cv["selected_ridge"] != ridge
        or cv["use_frozen_fallback"] is not (kind == _FROZEN)
    ):
        raise ValueError("A5d executable contradicts residual-CV selection")
    lineage = _validate_lineage(executable["lineage"])
    if (
        lineage["source_anchored_residual_receipt_sha256"]
        != target["receipt_sha256"]
        or lineage["residual_cv_receipt_sha256"] != cv["receipt_sha256"]
    ):
        raise ValueError("A5d executable lineage contradicts its receipts")
    owner = _validate_source_owner(executable["source_owner"])
    additive = _validate_additive(executable["additive_residual"], cv=cv)
    totals_raw = _exact_mapping(
        executable["selected_resources"],
        _TOTAL_RESOURCE_FIELDS,
        label="A5d selected resources",
    )
    totals = {
        "layer17_scope": _validate_resource(
            totals_raw["layer17_scope"], label="A5d selected Layer17 resources"
        ),
        "composition_scope": _validate_resource(
            totals_raw["composition_scope"],
            label="A5d selected composition resources",
        ),
    }
    additive_resources = None if additive is None else additive["resources"]
    assert additive_resources is None or isinstance(additive_resources, Mapping)
    if (
        totals["layer17_scope"]
        != _sum_resource(owner["layer17_resources"], additive_resources)
        or totals["composition_scope"]
        != _sum_resource(owner["composition_resources"], additive_resources)
    ):
        raise ValueError("A5d executable resource accounting is not additive")
    expected_freeze = a5d_selection_freeze_sha256(
        kind=str(kind),
        selected_alpha=alpha,
        selected_ridge=ridge,  # type: ignore[arg-type]
        application_boundary=str(executable["application_boundary"]),
        application_order=str(executable["application_order"]),
        source_ownership_preserved=executable[
            "source_ownership_preserved"
        ],  # type: ignore[arg-type]
        source_affine_means_reinjected=executable[
            "source_affine_means_reinjected"
        ],  # type: ignore[arg-type]
        lineage=lineage,
        source_owner=owner,
        additive_residual=additive,
        selected_resources=totals,
    )
    supplied = _require_sha256(
        executable["selection_freeze_sha256"], label="A5d selection freeze"
    )
    if supplied != expected_freeze:
        raise ValueError("A5d selection freeze hash is contradictory")
    return {
        **dict(executable),
        "selected_alpha": alpha,
        "selected_ridge": ridge,
        "lineage": lineage,
        "source_owner": owner,
        "additive_residual": additive,
        "selected_resources": totals,
        "selection_freeze_sha256": supplied,
    }


def _validate_metric(value: object, *, label: str) -> dict[str, float]:
    metric = _exact_mapping(value, _METRIC_FIELDS, label=label)
    nll = _finite(metric["nll_per_token"], label=f"{label} NLL", minimum=0.0)
    delta = _finite(metric["delta_nll_per_token"], label=f"{label} delta NLL")
    kl = _finite(
        metric["native_to_candidate_kl_per_token"],
        label=f"{label} KL",
        minimum=_MINIMUM_NUMERICAL_KL_PER_TOKEN,
    )
    top1 = _finite(
        metric["top1_agreement_to_native"],
        label=f"{label} top1",
        minimum=0.0,
        maximum=1.0,
    )
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _validate_condition(value: object, *, label: str) -> dict[str, object]:
    condition = _exact_mapping(value, _CONDITION_FIELDS, label=label)
    metric = _validate_metric(
        {name: condition[name] for name in _METRIC_FIELDS}, label=label
    )
    owner = _require_sha256(
        condition["owning_graph_sha256"], label=f"{label} owner graph"
    )
    additive = condition["additive_graph_sha256"]
    if additive is not None:
        additive = _require_sha256(additive, label=f"{label} additive graph")
    return {
        **metric,
        "owning_graph_sha256": owner,
        "additive_graph_sha256": additive,
    }


def _validate_execution_resource(
    value: object,
    *,
    label: str,
    source_parameters: int,
    expected_static: Mapping[str, int],
    deletion: bool,
) -> dict[str, int]:
    raw = _exact_mapping(value, _EXECUTION_RESOURCE_FIELDS, label=label)
    row = {
        name: _integer(raw[name], label=f"{label} {name}")
        for name in _EXECUTION_RESOURCE_FIELDS
    }
    identities = (
        row["total_graph_node_count"]
        == row["owning_graph_node_count"] + row["additive_graph_node_count"]
        and row["total_interaction_count"]
        == row["owning_interaction_count"] + row["additive_interaction_count"]
        and row["total_graph_parameters"]
        == row["owning_graph_parameters"] + row["additive_graph_parameters"]
        and row["total_dense_graph_macs_per_token"]
        == row["owning_dense_graph_macs_per_token"]
        + row["additive_dense_graph_macs_per_token"]
        and row["total_dense_graph_additions_per_token"]
        == row["owning_dense_graph_additions_per_token"]
        + row["additive_dense_graph_additions_per_token"]
        and row["candidate_whole_model_learned_parameters"]
        == source_parameters
        - row["native_removed_parameters"]
        + row["total_graph_parameters"]
        and row["net_parameter_savings"]
        == row["native_removed_parameters"] - row["total_graph_parameters"]
        and row["net_executed_macs_saved_per_token"]
        == row["native_removed_parameters"]
        - row["executed_graph_macs_per_token"]
        and row["executed_graph_macs_per_token"]
        <= row["total_dense_graph_macs_per_token"]
        and row["executed_graph_additions_per_token"]
        <= row["total_dense_graph_additions_per_token"]
    )
    if not identities or any(
        row[name] != expected
        for name, expected in expected_static.items()
    ):
        raise ValueError(f"{label} resource identities drifted")
    if deletion:
        if any(
            row[name] != 0
            for name in (
                "executed_graph_macs_per_token",
                "executed_graph_additions_per_token",
                "executed_peak_live_modal_width",
            )
        ):
            raise ValueError("A5d matched deletion executed graph work")
    elif row["executed_peak_live_modal_width"] <= 0:
        raise ValueError(f"{label} generated execution has no live modal width")
    return row


def a5d_outer_evaluation_sha256(value: Mapping[str, object]) -> str:
    """Hash the complete scalar A5d outer evaluation."""

    if not isinstance(value, Mapping):
        raise TypeError("A5d outer evaluation must be a mapping")
    _assert_source_safe(value, path="outer_evaluation")
    _assert_no_sensitive_keys(value, path="outer_evaluation")
    return _sha256(_EVALUATION_DOMAIN, value)


def _expected_execution_resources(
    executable: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    owner = executable["source_owner"]
    totals = executable["selected_resources"]
    additive = executable["additive_residual"]
    assert isinstance(owner, Mapping) and isinstance(totals, Mapping)
    layer17 = owner["layer17_resources"]
    composition = owner["composition_resources"]
    selected_layer17 = totals["layer17_scope"]
    selected_composition = totals["composition_scope"]
    assert all(
        isinstance(item, Mapping)
        for item in (layer17, composition, selected_layer17, selected_composition)
    )
    additive_resource = None
    if additive is not None:
        assert isinstance(additive, Mapping)
        additive_resource = additive["resources"]
        assert isinstance(additive_resource, Mapping)

    def static(
        resource: Mapping[str, object],
        *,
        owning: Mapping[str, object],
        additive_part: Mapping[str, object] | None,
        replaced: int,
        removed: int,
    ) -> dict[str, int]:
        return {
            "replaced_layer_count": replaced,
            "owning_graph_node_count": int(owning["node_count"]),
            "additive_graph_node_count": (
                0 if additive_part is None else int(additive_part["node_count"])
            ),
            "total_graph_node_count": int(resource["node_count"]),
            "owning_interaction_count": int(owning["interaction_count"]),
            "additive_interaction_count": (
                0
                if additive_part is None
                else int(additive_part["interaction_count"])
            ),
            "total_interaction_count": int(resource["interaction_count"]),
            "native_removed_parameters": removed,
            "owning_graph_parameters": int(owning["parameter_count"]),
            "additive_graph_parameters": (
                0 if additive_part is None else int(additive_part["parameter_count"])
            ),
            "total_graph_parameters": int(resource["parameter_count"]),
            "owning_dense_graph_macs_per_token": int(owning["macs_per_token"]),
            "additive_dense_graph_macs_per_token": (
                0 if additive_part is None else int(additive_part["macs_per_token"])
            ),
            "total_dense_graph_macs_per_token": int(resource["macs_per_token"]),
            "owning_dense_graph_additions_per_token": int(
                owning["additions_per_token"]
            ),
            "additive_dense_graph_additions_per_token": (
                0
                if additive_part is None
                else int(additive_part["additions_per_token"])
            ),
            "total_dense_graph_additions_per_token": int(
                resource["additions_per_token"]
            ),
        }

    return {
        "layer10_only": {
            "replaced_layer_count": 1,
            "owning_graph_node_count": 4,
            "additive_graph_node_count": 0,
            "total_graph_node_count": 4,
            "owning_interaction_count": 3,
            "additive_interaction_count": 0,
            "total_interaction_count": 3,
            "native_removed_parameters": 641_280,
            "owning_graph_parameters": 132_035,
            "additive_graph_parameters": 0,
            "total_graph_parameters": 132_035,
            "owning_dense_graph_macs_per_token": 129_248,
            "additive_dense_graph_macs_per_token": 0,
            "total_dense_graph_macs_per_token": 129_248,
            # Layer10 additions are authenticated by the evaluation hash and
            # their own sum identity; no A5d artifact recreates this graph.
        },
        "selected_layer17_only": static(
            selected_layer17,
            owning=layer17,
            additive_part=additive_resource,
            replaced=1,
            removed=441_600,
        ),
        "frozen_uncorrected_composition": static(
            composition,
            owning=composition,
            additive_part=None,
            replaced=2,
            removed=1_082_880,
        ),
        "selected_composition": static(
            selected_composition,
            owning=composition,
            additive_part=additive_resource,
            replaced=2,
            removed=1_082_880,
        ),
        "matched_double_deletion": static(
            selected_composition,
            owning=composition,
            additive_part=additive_resource,
            replaced=2,
            removed=1_082_880,
        ),
    }


def _validate_evaluation(
    value: object,
    *,
    outer_fold_index: int,
    executable: Mapping[str, object],
) -> dict[str, object]:
    evaluation = _exact_mapping(value, _EVALUATION_FIELDS, label="A5d evaluation")
    if (
        evaluation["assessment_role"] != _ASSESSMENT_ROLE
        or evaluation["outer_fold_index"] != outer_fold_index
        or evaluation["full_model_logits_scored"] is not True
        or evaluation["full_model_compiled"] is not False
        or evaluation["heldout_confirmation"] is not False
        or evaluation["exact_resources_match_frozen_executable"] is not True
        or evaluation["latency_or_kernel_speed_claim"] is not False
    ):
        raise ValueError("A5d evaluation boundary drifted")
    _integer(
        evaluation["logical_valid_tokens"], label="A5d logical tokens", minimum=1
    )
    _integer(evaluation["supervised_tokens"], label="A5d supervised tokens", minimum=1)
    source_parameters = _integer(
        evaluation["source_whole_model_learned_parameters"],
        label="A5d source parameter count",
        minimum=1,
    )
    native = _exact_mapping(evaluation["native"], {"nll_per_token"}, label="A5d native")
    native_nll = _finite(native["nll_per_token"], label="A5d native NLL", minimum=0.0)
    raw_conditions = _exact_mapping(
        evaluation["conditions"], _CONDITIONS, label="A5d conditions"
    )
    conditions = {
        name: _validate_condition(raw_conditions[name], label=f"A5d {name}")
        for name in sorted(_CONDITIONS)
    }
    for name, row in conditions.items():
        if not math.isclose(
            row["nll_per_token"],
            native_nll + row["delta_nll_per_token"],
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"A5d {name} NLL identity drifted")
    owner = executable["source_owner"]
    additive = executable["additive_residual"]
    assert isinstance(owner, Mapping)
    additive_graph = None
    if additive is not None:
        assert isinstance(additive, Mapping)
        additive_graph = additive["graph_sha256"]
    expected_graphs = {
        "layer10_only": (_EXPECTED_LAYER10_GRAPH_SHA256, None),
        "selected_layer17_only": (_EXPECTED_LAYER17_GRAPH_SHA256, additive_graph),
        "frozen_uncorrected_composition": (
            _EXPECTED_COMPOSITION_GRAPH_SHA256,
            None,
        ),
        "selected_composition": (_EXPECTED_COMPOSITION_GRAPH_SHA256, additive_graph),
        "matched_double_deletion": (
            _EXPECTED_COMPOSITION_GRAPH_SHA256,
            additive_graph,
        ),
    }
    if any(
        (
            conditions[name]["owning_graph_sha256"],
            conditions[name]["additive_graph_sha256"],
        )
        != expected
        for name, expected in expected_graphs.items()
    ):
        raise ValueError("A5d evaluation graph identity drifted")
    raw_resources = _exact_mapping(
        evaluation["resource_accounting"],
        _CONDITIONS,
        label="A5d resource accounting",
    )
    expected_static = _expected_execution_resources(executable)
    resources: dict[str, dict[str, int]] = {}
    for name in sorted(_CONDITIONS):
        static = dict(expected_static[name])
        # Layer10 additions are not part of the A5d freeze, so use the
        # reported owning/total value while retaining exact sum validation.
        if name == "layer10_only":
            raw_row = raw_resources[name]
            if not isinstance(raw_row, Mapping):
                raise ValueError("A5d Layer10 resource row is invalid")
            static["owning_dense_graph_additions_per_token"] = _integer(
                raw_row.get("owning_dense_graph_additions_per_token"),
                label="A5d Layer10 additions",
            )
            static["additive_dense_graph_additions_per_token"] = 0
            static["total_dense_graph_additions_per_token"] = static[
                "owning_dense_graph_additions_per_token"
            ]
        resources[name] = _validate_execution_resource(
            raw_resources[name],
            label=f"A5d {name}",
            source_parameters=source_parameters,
            expected_static=static,
            deletion=name == "matched_double_deletion",
        )
    if resources["matched_double_deletion"] != {
        **resources["selected_composition"],
        "executed_graph_macs_per_token": 0,
        "net_executed_macs_saved_per_token": resources[
            "matched_double_deletion"
        ]["native_removed_parameters"],
        "executed_graph_additions_per_token": 0,
        "executed_peak_live_modal_width": 0,
    }:
        raise ValueError("A5d selected/deletion static scopes differ")
    if executable["kind"] == _FROZEN:
        if (
            conditions["selected_composition"]
            != conditions["frozen_uncorrected_composition"]
            or resources["selected_composition"]
            != resources["frozen_uncorrected_composition"]
        ):
            raise ValueError("A5d fallback is not bit-identical to frozen source")
    return {
        **dict(evaluation),
        "source_whole_model_learned_parameters": source_parameters,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
        "resource_accounting": resources,
    }


def derive_gemma3_l10_l17_a5d_conclusion(
    *,
    selected_executable: Mapping[str, object],
    outer_evaluation: Mapping[str, object],
    comparison_to_a5c: Mapping[str, object],
) -> dict[str, object]:
    """Derive the bounded one-fold conclusion from scalar metrics only."""

    conditions = outer_evaluation["conditions"]
    assert isinstance(conditions, Mapping)
    selected = conditions["selected_composition"]
    frozen = conditions["frozen_uncorrected_composition"]
    a5c = comparison_to_a5c["a5c_frozen_composition"]
    assert all(isinstance(item, Mapping) for item in (selected, frozen, a5c))
    kind = selected_executable["kind"]
    return {
        "selected_executable_kind": kind,
        "use_frozen_fallback": kind == _FROZEN,
        "additive_residual_deployed": kind == _ADDITIVE,
        "source_owner_preserved": True,
        "selected_alpha": selected_executable["selected_alpha"],
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
        "selected_improves_canonical_a5c_kl": (
            selected["native_to_candidate_kl_per_token"]
            < a5c["native_to_candidate_kl_per_token"]
        ),
        "selected_improves_canonical_a5c_delta_nll": (
            selected["delta_nll_per_token"] < a5c["delta_nll_per_token"]
        ),
        "selected_improves_canonical_a5c_top1": (
            selected["top1_agreement_to_native"]
            > a5c["top1_agreement_to_native"]
        ),
        "one_outer_fold_only": True,
        "does_not_establish_eight_fold_competitive_compilation": True,
    }


def validate_gemma3_l10_l17_a5d_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return one strict tensor-free A5d report."""

    if not isinstance(value, Mapping):
        raise TypeError("A5d report must be a mapping")
    _assert_source_safe(value)
    raw = dict(value)
    _exact_mapping(raw, _TOP_FIELDS, label="A5d report")
    if (
        raw["schema"] != GEMMA3_L10_L17_A5D_REPORT_SCHEMA
        or raw["format_version"] != GEMMA3_L10_L17_A5D_REPORT_FORMAT_VERSION
        or raw["scientific_role"] != _SCIENTIFIC_ROLE
        or raw["full_model_forward_evaluated"] is not True
        or raw["whole_model_compiled"] is not False
        or raw["heldout_confirmation"] is not False
        or raw["serving_authorized"] is not False
        or raw["latency_or_kernel_speed_claim"] is not False
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("A5d report header, scope, or safety drifted")
    _assert_no_sensitive_keys(raw)

    source = _exact_mapping(raw["source_bindings"], _SOURCE_FIELDS, label="A5d source")
    if dict(source) != {
        "a5c_file_sha256": _EXPECTED_A5C_FILE_SHA256,
        "a5c_report_sha256": _EXPECTED_A5C_REPORT_SHA256,
    }:
        raise ValueError("A5d does not bind the canonical A5c report")

    runtime = _exact_mapping(raw["runtime"], _RUNTIME_FIELDS, label="A5d runtime")
    if dict(runtime) != {
        "model_id": _EXPECTED_MODEL_ID,
        "requested_revision": _EXPECTED_MODEL_REVISION,
        "model_fingerprint": _EXPECTED_MODEL_FINGERPRINT,
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
    }:
        raise ValueError("A5d runtime differs from canonical A5c replay")

    config = _exact_mapping(raw["configuration"], _CONFIG_FIELDS, label="A5d config")
    if (
        config["outer_fold_index"] != 0
        or config["training_family_count"] != 7
        or config["training_examples_per_family"] != 4
        or config["row_selection_policy"] != "all_valid_captured_rows_per_example"
        or config["generator_rank"] != A5D_FIXED_GENERATOR_RANK
        or config["ridge_grid"] != list(A5D_RIDGE_GRID)
        or config["alpha_grid"] != list(A5D_ALPHA_GRID)
        or config["held_examples_scored"] != 1
        or config["output_boundary"] != _OUTPUT_BOUNDARY
        or config["final_head_chunk_rows"] != 8
    ):
        raise ValueError("A5d fixed configuration drifted")
    capture = _exact_mapping(raw["capture"], _CAPTURE_FIELDS, label="A5d capture")
    if dict(capture) != _EXPECTED_A5C_CAPTURE:
        raise ValueError("A5d capture differs from canonical A5c source")

    target = _validate_target_summary(raw["source_anchored_residual"])
    cv = _validate_cv_summary(raw["residual_cv"])
    evidence = _exact_mapping(
        raw["evidence_receipts"], _EVIDENCE_FIELDS, label="A5d evidence receipts"
    )
    target_receipt = validate_a5d_source_anchored_residual_receipt(
        evidence["source_anchored_residual"]  # type: ignore[arg-type]
    )
    cv_receipt = validate_a5d_family_residual_cv_receipt(
        evidence["residual_cv"]  # type: ignore[arg-type]
    )
    if (
        compact_a5d_source_anchored_residual_receipt(target_receipt) != target
        or compact_a5d_family_residual_cv_receipt(cv_receipt) != cv
    ):
        raise ValueError("A5d compact receipts contradict full evidence")
    cv_source = cv_receipt["source"]
    assert isinstance(cv_source, Mapping)
    if (
        cv_source["residual_target_receipt_sha256"] != target["receipt_sha256"]
        or cv_source["source_graph_sha256"] != _EXPECTED_LAYER17_GRAPH_SHA256
    ):
        raise ValueError("A5d target/CV/source graph lineage drifted")

    executable = _validate_selected_executable(
        raw["selected_executable"], target=target, cv=cv
    )
    evaluation = _validate_evaluation(
        raw["outer_evaluation"], outer_fold_index=0, executable=executable
    )
    evaluation_sha = a5d_outer_evaluation_sha256(evaluation)
    chronology = _exact_mapping(
        raw["chronology"], _CHRONOLOGY_FIELDS, label="A5d chronology"
    )
    events = tuple(
        chronology[name]
        for name in (
            "residual_cv_completed_event",
            "executable_frozen_event",
            "outer_held_batch_selected_event",
            "outer_held_model_evaluated_event",
        )
    )
    if (
        events != (1, 2, 3, 4)
        or chronology["outer_held_batch_selected_or_scored_before_freeze"]
        is not False
        or chronology["executable_frozen_before_outer_held_batch_selection"]
        is not True
        or chronology["executable_frozen_before_outer_held_model_evaluation"]
        is not True
        or chronology["residual_cv_receipt_sha256"] != cv["receipt_sha256"]
        or chronology["selection_freeze_sha256"]
        != executable["selection_freeze_sha256"]
        or chronology["outer_evaluation_sha256"] != evaluation_sha
    ):
        raise ValueError("A5d freeze-before-held chronology is contradictory")

    comparison = _exact_mapping(
        raw["comparison_to_a5c"], _COMPARISON_FIELDS, label="A5d comparison"
    )
    if (
        comparison["a5c_file_sha256"] != _EXPECTED_A5C_FILE_SHA256
        or comparison["a5c_report_sha256"] != _EXPECTED_A5C_REPORT_SHA256
        or comparison["same_outer_fold"] is not True
        or comparison["same_held_example_policy"] is not True
        or _validate_metric(
            comparison["a5c_frozen_composition"], label="canonical A5c composition"
        )
        != _EXPECTED_A5C_COMPOSITION_METRIC
    ):
        raise ValueError("A5d comparison is not canonical A5c")
    expected_conclusion = derive_gemma3_l10_l17_a5d_conclusion(
        selected_executable=executable,
        outer_evaluation=evaluation,
        comparison_to_a5c=comparison,
    )
    if raw["conclusion"] != expected_conclusion:
        raise ValueError("A5d conclusion contradicts its evaluation")
    supplied = _require_sha256(raw["report_sha256"], label="A5d report")
    payload = dict(raw)
    payload.pop("report_sha256")
    if supplied != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A5d report hash mismatch")
    return json.loads(json.dumps(raw, allow_nan=False))


def build_gemma3_l10_l17_a5d_report(
    *,
    source_bindings: Mapping[str, object],
    runtime: Mapping[str, object],
    configuration: Mapping[str, object],
    capture: Mapping[str, object],
    source_anchored_residual: Mapping[str, object],
    residual_cv: Mapping[str, object],
    evidence_receipts: Mapping[str, object],
    selected_executable: Mapping[str, object],
    chronology: Mapping[str, object],
    outer_evaluation: Mapping[str, object],
    comparison_to_a5c: Mapping[str, object],
) -> dict[str, object]:
    """Build one strict A5d report from tensor-free inputs."""

    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5D_REPORT_SCHEMA,
        "format_version": GEMMA3_L10_L17_A5D_REPORT_FORMAT_VERSION,
        "scientific_role": _SCIENTIFIC_ROLE,
        "source_bindings": dict(source_bindings),
        "runtime": dict(runtime),
        "configuration": dict(configuration),
        "capture": dict(capture),
        "source_anchored_residual": dict(source_anchored_residual),
        "residual_cv": dict(residual_cv),
        "evidence_receipts": dict(evidence_receipts),
        "selected_executable": dict(selected_executable),
        "chronology": dict(chronology),
        "outer_evaluation": dict(outer_evaluation),
        "comparison_to_a5c": dict(comparison_to_a5c),
        "full_model_forward_evaluated": True,
        "whole_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(payload)
    _assert_no_sensitive_keys(payload)
    payload["conclusion"] = derive_gemma3_l10_l17_a5d_conclusion(
        selected_executable=selected_executable,
        outer_evaluation=outer_evaluation,
        comparison_to_a5c=comparison_to_a5c,
    )
    payload["report_sha256"] = _sha256(_REPORT_DOMAIN, payload)
    return validate_gemma3_l10_l17_a5d_report(payload)


def save_gemma3_l10_l17_a5d_report(
    path: Path | str, report: Mapping[str, object]
) -> dict[str, object]:
    """Atomically publish a validated A5d report without overwriting."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite A5d report")
    validated = validate_gemma3_l10_l17_a5d_report(report)
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
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("refusing to overwrite A5d report")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_a5d_report(path: Path | str) -> dict[str, object]:
    """Load strict JSON and authenticate every A5d report field."""

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("A5d report is not strict JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("A5d report must contain one object")
    return validate_gemma3_l10_l17_a5d_report(raw)
