"""Strict artifacts for sequential compiled-trajectory Gemma MLP refits.

This is an overlay artifact.  Layers 0 through 9 remain in the authenticated
frozen full-stack artifact; only refitted generator states for layers 10
through 17 are stored here.  Compact source-layer summaries bind the overlay
to the frozen source without retaining a second in-memory copy of its large
analysis curves.

The paired JSON report is tensor-free.  Prompt text, token IDs, raw rows,
source parameter values, and source model weights are forbidden from both
boundaries.  Digests detect accidental mutation and do not constitute an
external provenance attestation.
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
import unicodedata

import torch

from .full_mlp_stack_generators import FullMLPStackGeneratorFit


__all__ = [
    "GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION",
    "GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA",
    "build_gemma3_full_mlp_stack_refit_payload",
    "build_gemma3_full_mlp_stack_refit_report",
    "compiled_prefix_catalog_sha256",
    "frozen_baseline_conditions_sha256",
    "load_gemma3_full_mlp_stack_refit_artifact",
    "save_gemma3_full_mlp_stack_refit_artifact",
    "trajectory_breakpoint_row_sha256",
]


GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA = (
    "fisher_graph.gemma3_sequential_full_mlp_stack_refit_development"
)
GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION = 1

_LAYER_COUNT = 18
_REFIT_START = 10
_UNCHANGED_ORDINALS = tuple(range(_REFIT_START))
_REFIT_ORDINALS = tuple(range(_REFIT_START, _LAYER_COUNT))
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3.full_mlp_stack.refit.payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3.full_mlp_stack.refit.report.v1\0"
_PREFIX_DOMAIN = b"fisher_graph.gemma3.compiled_prefix.catalog.v1\0"
_BREAKPOINT_DOMAIN = b"fisher_graph.gemma3.trajectory.breakpoint.row.v1\0"
_BASELINE_DOMAIN = b"fisher_graph.gemma3.refit.frozen_baseline.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "development_only_sequential_compiled_trajectory_refit",
    "development_only": True,
    "compression_claim": False,
    "heldout_confirmation": False,
    "generator_refit_performed": True,
    "generator_rank_selection_performed": False,
    "jacobian_correction_performed": False,
    "latency_or_kernel_speed_claim": False,
    "scope": "full_native_mlp_stack_replacement_overlay",
    "refit_start_layer": _REFIT_START,
    "assessment_role": "open_development_assessment",
    "assessment_used_for_refit": False,
}
_SAFETY: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_source_generator_weights": False,
    "contains_refit_generator_weights": True,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_token_rows": False,
    "contains_tokenizer_state": False,
    "requires_authenticated_frozen_source_overlay": True,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "text",
        "token",
        "tokens",
        "token_ids",
        "input_ids",
        "target_ids",
        "targets",
        "score_gradients",
        "activations",
        "activation_rows",
        "gradient_rows",
        "raw_rows",
        "raw_prompt_rows",
        "raw_token_rows",
        "raw_fit_rows",
        "raw_eval_rows",
        "source_weights",
        "source_model_weights",
        "source_parameter_values",
        "source_generator_weights",
        "model_state_dict",
        "source_state_dict",
        "tokenizer_state",
    }
)

_PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "frozen_sources",
    "splits",
    "protocol",
    "source_layer_summaries",
    "unchanged_prefix_fit_sha256s",
    "refit_generator_fits",
    "layer_refits",
    "resource_accounting",
    "evaluation",
    "safety",
    "scientific_payload_sha256",
}
_MODEL_FIELDS = {
    "model_id",
    "requested_revision",
    "resolved_commit",
    "adapter_model_fingerprint",
    "source_whole_model_learned_parameters",
    "local_files_only",
}
_FROZEN_SOURCE_FIELDS = {"full_stack", "trajectory"}
_FULL_STACK_SOURCE_FIELDS = {
    "schema",
    "format_version",
    "artifact_file_sha256",
    "scientific_payload_sha256",
    "baseline_conditions_sha256",
}
_TRAJECTORY_SOURCE_FIELDS = {
    "schema",
    "format_version",
    "artifact_file_sha256",
    "scientific_payload_sha256",
    "breakpoint_direction",
    "breakpoint_depth",
    "breakpoint_row_sha256",
}
_SPLIT_FIELDS = {"fit", "selection", "assessment", "provenance"}
_SPLIT_ENTRY_FIELDS = {
    "role",
    "serialized_sha256",
    "content_sha256",
    "example_count",
    "logical_valid_tokens",
    "supervised_tokens",
}
_PROVENANCE_FIELDS = {
    "assurance",
    "externally_authenticated",
    "fit_used_for_generator_refit",
    "selection_used_for_generator_refit",
    "assessment_used_for_generator_refit",
    "assessment_used_for_generator_rank_selection",
    "fit_selection_assessment_disjoint",
    "assessment_evaluated_only_after_refit_freeze",
    "heldout_confirmation",
}
_PROTOCOL_FIELDS = {
    "scope",
    "transformer_layer_count",
    "refit_start_layer",
    "unchanged_layer_ordinals",
    "refit_layer_order",
    "refit_rule",
    "fisher_weighting",
    "jacobian_policy",
    "rank_policy",
    "resource_budget_policy",
    "execution_path",
    "source_model_weights_mutated",
    "assessment_role",
    "assessment_after_refit_freeze",
    "generator_rank_selection_performed",
    "heldout_confirmation",
    "compression_claim",
    "latency_or_kernel_speed_claim",
    "local_files_only",
}
_SOURCE_LAYER_FIELDS = {
    "layer_ordinal",
    "layer_id",
    "input_site",
    "output_site",
    "input_width",
    "intermediate_width",
    "residual_width",
    "source_fit_sha256",
    "superfragment_sha256",
    "superfragment_plan_sha256",
    "source_model_sha256",
    "parameter_catalog_sha256",
    "source_fisher_coupling_sha256",
    "source_fragment_plan_sha256",
    "source_cluster_plan_sha256",
    "dense_plan_sha256",
    "selected_mode_rank",
    "selected_generator_rank",
    "native_mlp_parameter_count",
    "dense_fused_parameter_count",
    "dense_fused_macs_per_token",
}
_LAYER_REFIT_FIELDS = {
    "layer_ordinal",
    "generated_prefix_ordinals",
    "generated_prefix_plan_sha256s",
    "generated_prefix_catalog_sha256",
    "source_fit_sha256",
    "refit_fit_sha256",
    "fit_row_key_sha256",
    "selection_row_key_sha256",
    "fit_observations",
    "fit_sequences",
    "selection_observations",
    "selection_sequences",
    "old_selected_mode_rank",
    "old_selected_generator_rank",
    "refit_selected_mode_rank",
    "refit_selected_generator_rank",
    "old_plan_fit_metrics",
    "old_plan_selection_metrics",
    "refit_plan_fit_metrics",
    "refit_plan_selection_metrics",
    "refit_resource_metadata",
}
_METRIC_FIELDS = {
    "observations",
    "mse",
    "nrmse",
    "weighted_mse",
    "weighted_nrmse",
    "cosine_similarity",
    "weighted_cosine_similarity",
    "max_abs_error",
    "target_rms",
    "weighted_target_rms",
}
_FIT_RESOURCE_FIELDS = {
    "native_mlp_parameter_count",
    "native_mlp_linear_macs_per_token",
    "selected_basis_stored_scalar_count",
    "coordinate_generator_parameter_count",
    "coordinate_generator_macs_per_token",
    "selected_unfused_stored_scalar_count",
    "dense_fused_parameter_count",
    "dense_fused_macs_per_token",
    "net_stored_parameter_savings",
    "net_linear_macs_saved_per_token",
    "dense_parameter_reduction_fraction",
    "dense_linear_mac_reduction_fraction",
    "dense_execution_stores_basis_separately",
}
_RESOURCE_FIELDS = {
    "scope",
    "source_whole_model_learned_parameters",
    "native_mlp_stack_learned_parameters",
    "retained_native_non_mlp_learned_parameters",
    "dense_generator_stack_learned_parameters",
    "logical_candidate_learned_parameters",
    "net_stored_parameter_savings",
    "removed_mode_count",
    "native_mlp_stack_linear_macs_per_token",
    "dense_generator_stack_macs_per_token",
    "dense_generator_stack_bias_additions_per_token",
    "net_linear_macs_saved_per_token",
    "unchanged_layer_count",
    "refit_layer_count",
    "per_layer_rank_and_resource_budget_unchanged",
    "logical_candidate_excludes_native_mlp_stack",
    "whole_transformer_replaced",
}
_EVALUATION_FIELDS = {
    "execution_path",
    "assessment_role",
    "heldout_confirmation",
    "assessment_membership_exact",
    "refit_frozen_before_assessment",
    "fit_and_selection_used_for_refit",
    "assessment_used_for_refit",
    "generator_rank_selection_performed",
    "latency_or_kernel_speed_claim",
    "supervised_tokens",
    "logical_valid_tokens",
    "assessment_split_sha256",
    "frozen_baseline_conditions_sha256",
    "conditions",
    "control_validation",
    "resource_accounting",
}
_CONDITION_FIELDS = {
    "native",
    "frozen_generated_full_stack",
    "sequential_refit_full_stack",
    "matched_deletion",
}
_LM_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_CONTROL_FIELDS = {
    "native_matches_frozen_full_stack_artifact",
    "frozen_generated_matches_frozen_full_stack_artifact",
    "matched_deletion_matches_frozen_full_stack_artifact",
    "frozen_generated_matches_trajectory_full_stack_endpoint",
    "physical_scope_identical",
    "refit_generator_compute_executed",
    "matched_deletion_compute_zero",
}


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_digest(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _update_payload_digest(digest: object, value: object) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a hashlib SHA-256 object")
    if value is None:
        digest.update(b"N;")
    elif isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
    elif type(value) is int:
        digest.update(f"I{value};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payload floats must be finite")
        digest.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    "scientific payload tensors must be finite"
                )
        digest.update(b"T")
        _update_payload_digest(digest, str(tensor.dtype))
        _update_payload_digest(digest, tuple(tensor.shape))
        raw = (
            tensor.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        digest.update(f"{len(raw)}:".encode("ascii"))
        digest.update(raw)
        digest.update(b";")
    elif isinstance(value, Mapping):
        keys = sorted(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("scientific payload mapping keys must be strings")
        digest.update(f"M{len(keys)}[".encode("ascii"))
        for key in keys:
            _update_payload_digest(digest, key)
            _update_payload_digest(digest, value[key])
        digest.update(b"];")
    elif isinstance(value, tuple):
        digest.update(f"U{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    elif isinstance(value, list):
        digest.update(f"L{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(
            "scientific payload contains unsupported "
            f"{type(value).__qualname__}"
        )


def _assert_source_safe(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact mapping keys must be strings")
            if key in _FORBIDDEN_KEYS:
                location = ".".join((*path, key))
                raise ValueError(f"artifact contains forbidden field {location}")
            _assert_source_safe(child, path=(*path, key))
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_source_safe(child, path=(*path, str(index)))
        return
    if isinstance(value, str):
        if any(
            character.isspace()
            or unicodedata.category(character).startswith("C")
            for character in value
        ):
            location = ".".join(path) or "<root>"
            raise ValueError(
                f"artifact contains non-machine string at {location}"
            )
        return
    if isinstance(value, torch.Tensor):
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError("artifact contains a non-source-safe scalar")


def _canonical_ordinals(
    value: object,
    *,
    label: str,
) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise ValueError(f"{label} must contain exact integers")
    return result  # type: ignore[return-value]


def _canonical_hashes(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise TypeError(f"{label} must be a sequence")
    return tuple(
        _require_sha256(item, label=label)
        for item in value
    )


def compiled_prefix_catalog_sha256(
    layer_ordinals: Sequence[int],
    plan_sha256s: Sequence[str],
) -> str:
    """Hash one exact ordered compiled-prefix plan catalog."""

    ordinals = _canonical_ordinals(
        layer_ordinals,
        label="compiled prefix layer ordinals",
    )
    hashes = _canonical_hashes(
        plan_sha256s,
        label="compiled prefix plan_sha256",
    )
    if (
        ordinals != tuple(range(len(ordinals)))
        or len(ordinals) != len(hashes)
    ):
        raise ValueError("compiled prefix must be exact layers zero onward")
    return _json_digest(
        {
            "layer_ordinals": ordinals,
            "plan_sha256s": hashes,
        },
        domain=_PREFIX_DOMAIN,
    )


def trajectory_breakpoint_row_sha256(row: Mapping[str, object]) -> str:
    """Hash the frozen trajectory prefix-depth-10 row exactly."""

    if not isinstance(row, Mapping):
        raise TypeError("trajectory breakpoint row must be a mapping")
    _assert_source_safe(row)
    return _json_digest(dict(row), domain=_BREAKPOINT_DOMAIN)


def frozen_baseline_conditions_sha256(
    conditions: Mapping[str, object],
) -> str:
    """Hash native/generated/deletion conditions from the frozen source."""

    if not isinstance(conditions, Mapping):
        raise TypeError("frozen baseline conditions must be a mapping")
    expected = {"native", "generated_full_stack", "matched_deletion"}
    _strict_fields(conditions, expected, label="frozen baseline conditions")
    _assert_source_safe(conditions)
    return _json_digest(dict(conditions), domain=_BASELINE_DOMAIN)


def _validate_model(model: Mapping[str, object]) -> None:
    _strict_fields(model, _MODEL_FIELDS, label="model metadata")
    _require_name(model["model_id"], label="model_id")
    requested = model["requested_revision"]
    if (
        not isinstance(requested, str)
        or _REVISION.fullmatch(requested) is None
        or requested != model["resolved_commit"]
    ):
        raise ValueError("model revisions must bind the same exact commit")
    _require_sha256(
        model["adapter_model_fingerprint"],
        label="adapter_model_fingerprint",
    )
    _require_int(
        model["source_whole_model_learned_parameters"],
        label="source_whole_model_learned_parameters",
        minimum=1,
    )
    if model["local_files_only"] is not True:
        raise ValueError("refit artifact must be local-only")


def _validate_frozen_sources(sources: Mapping[str, object]) -> None:
    _strict_fields(
        sources,
        _FROZEN_SOURCE_FIELDS,
        label="frozen sources",
    )
    full_stack = sources["full_stack"]
    trajectory = sources["trajectory"]
    if not isinstance(full_stack, Mapping) or not isinstance(
        trajectory,
        Mapping,
    ):
        raise TypeError("frozen source entries must be mappings")
    _strict_fields(
        full_stack,
        _FULL_STACK_SOURCE_FIELDS,
        label="frozen full-stack source",
    )
    _strict_fields(
        trajectory,
        _TRAJECTORY_SOURCE_FIELDS,
        label="frozen trajectory source",
    )
    if (
        full_stack["schema"]
        != "fisher_graph.gemma3_full_native_mlp_stack_development"
        or full_stack["format_version"] != 1
        or trajectory["schema"]
        != (
            "fisher_graph."
            "gemma3_frozen_full_mlp_stack_trajectory_development"
        )
        or trajectory["format_version"] != 1
        or trajectory["breakpoint_direction"] != "prefix"
        or trajectory["breakpoint_depth"] != _REFIT_START
    ):
        raise ValueError("frozen source schemas or breakpoint are invalid")
    for source, label in (
        (full_stack, "frozen full stack"),
        (trajectory, "frozen trajectory"),
    ):
        _require_sha256(
            source["artifact_file_sha256"],
            label=f"{label} artifact_file_sha256",
        )
        _require_sha256(
            source["scientific_payload_sha256"],
            label=f"{label} scientific_payload_sha256",
        )
    _require_sha256(
        full_stack["baseline_conditions_sha256"],
        label="frozen baseline_conditions_sha256",
    )
    _require_sha256(
        trajectory["breakpoint_row_sha256"],
        label="trajectory breakpoint_row_sha256",
    )


def _validate_split_entry(
    value: Mapping[str, object],
    *,
    name: str,
    expected_role: str,
    expected_count: int,
) -> tuple[str, ...]:
    _strict_fields(value, _SPLIT_ENTRY_FIELDS, label=f"{name} split")
    if value["role"] != expected_role:
        raise ValueError(f"{name} split role is invalid")
    _require_sha256(
        value["serialized_sha256"],
        label=f"{name} serialized_sha256",
    )
    hashes = _canonical_hashes(
        value["content_sha256"],
        label=f"{name} content_sha256",
    )
    if (
        value["example_count"] != expected_count
        or len(hashes) != expected_count
        or len(hashes) != len(set(hashes))
    ):
        raise ValueError(
            f"{name} split must contain exactly {expected_count} unique "
            "members"
        )
    _require_int(
        value["logical_valid_tokens"],
        label=f"{name} logical_valid_tokens",
        minimum=1,
    )
    _require_int(
        value["supervised_tokens"],
        label=f"{name} supervised_tokens",
        minimum=1,
    )
    return hashes


def _validate_splits(splits: Mapping[str, object]) -> None:
    _strict_fields(splits, _SPLIT_FIELDS, label="split metadata")
    entries: list[tuple[str, tuple[str, ...]]] = []
    for name, role, count in (
        ("fit", "generator_fit", 40),
        ("selection", "generator_selection", 20),
        ("assessment", "open_development_assessment", 20),
    ):
        value = splits[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} split must be a mapping")
        entries.append(
            (
                name,
                _validate_split_entry(
                    value,
                    name=name,
                    expected_role=role,
                    expected_count=count,
                ),
            )
        )
    all_hashes = tuple(
        value for _, hashes in entries for value in hashes
    )
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("fit, selection, and assessment must be disjoint")
    provenance = splits["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("split provenance must be a mapping")
    _strict_fields(
        provenance,
        _PROVENANCE_FIELDS,
        label="split provenance",
    )
    if provenance != {
        "assurance": "caller_declared_self_attested",
        "externally_authenticated": False,
        "fit_used_for_generator_refit": True,
        "selection_used_for_generator_refit": True,
        "assessment_used_for_generator_refit": False,
        "assessment_used_for_generator_rank_selection": False,
        "fit_selection_assessment_disjoint": True,
        "assessment_evaluated_only_after_refit_freeze": True,
        "heldout_confirmation": False,
    }:
        raise ValueError("split provenance overclaims or permits leakage")


def _validate_protocol(protocol: Mapping[str, object]) -> None:
    _strict_fields(protocol, _PROTOCOL_FIELDS, label="protocol")
    if (
        protocol["scope"]
        != "sequential_compiled_trajectory_full_mlp_stack_refit"
        or protocol["transformer_layer_count"] != _LAYER_COUNT
        or protocol["refit_start_layer"] != _REFIT_START
        or _canonical_ordinals(
            protocol["unchanged_layer_ordinals"],
            label="unchanged_layer_ordinals",
        )
        != _UNCHANGED_ORDINALS
        or _canonical_ordinals(
            protocol["refit_layer_order"],
            label="refit_layer_order",
        )
        != _REFIT_ORDINALS
        or protocol["refit_rule"]
        != "sequential_teacher_on_actual_compiled_prefix"
        or protocol["fisher_weighting"]
        != "current_compiled_prefix_rows"
        or protocol["jacobian_policy"]
        != "no_explicit_jacobian_correction_in_direct_refit_rung"
        or protocol["rank_policy"] != "fixed_from_frozen_full_stack"
        or protocol["resource_budget_policy"]
        != "exact_per_layer_equality_to_frozen_full_stack"
        or protocol["execution_path"]
        != "frozen_prefix_then_sequential_refit_dense_generators"
        or protocol["source_model_weights_mutated"] is not False
        or protocol["assessment_role"]
        != "open_development_assessment"
        or protocol["assessment_after_refit_freeze"] is not True
        or protocol["generator_rank_selection_performed"] is not False
        or protocol["heldout_confirmation"] is not False
        or protocol["compression_claim"] is not False
        or protocol["latency_or_kernel_speed_claim"] is not False
        or protocol["local_files_only"] is not True
    ):
        raise ValueError("protocol is not the fixed direct-refit rung")


def _validate_source_layers(
    value: object,
    *,
    model: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise TypeError("source_layer_summaries must be a sequence")
    rows = tuple(value)
    if len(rows) != _LAYER_COUNT:
        raise ValueError("source summaries must cover exactly 18 layers")
    plan_hash: str | None = None
    result: list[Mapping[str, object]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("source layer summaries must be mappings")
        _strict_fields(
            row,
            _SOURCE_LAYER_FIELDS,
            label=f"source layer {ordinal}",
        )
        if row["layer_ordinal"] != ordinal:
            raise ValueError("source layers must be ordered zero through 17")
        for name in ("layer_id", "input_site", "output_site"):
            _require_name(row[name], label=f"source layer {ordinal} {name}")
        for name in (
            "input_width",
            "intermediate_width",
            "residual_width",
            "selected_mode_rank",
            "selected_generator_rank",
            "native_mlp_parameter_count",
            "dense_fused_parameter_count",
            "dense_fused_macs_per_token",
        ):
            _require_int(
                row[name],
                label=f"source layer {ordinal} {name}",
                minimum=1,
            )
        for name in (
            "source_fit_sha256",
            "superfragment_sha256",
            "superfragment_plan_sha256",
            "source_model_sha256",
            "parameter_catalog_sha256",
            "source_fisher_coupling_sha256",
            "source_fragment_plan_sha256",
            "source_cluster_plan_sha256",
            "dense_plan_sha256",
        ):
            _require_sha256(
                row[name],
                label=f"source layer {ordinal} {name}",
            )
        if (
            row["source_model_sha256"]
            != model["adapter_model_fingerprint"]
            or row["selected_mode_rank"] > row["intermediate_width"]
            or row["selected_generator_rank"] > row["selected_mode_rank"]
            or row["dense_fused_parameter_count"]
            >= row["native_mlp_parameter_count"]
            or row["dense_fused_macs_per_token"]
            >= row["native_mlp_parameter_count"]
        ):
            raise ValueError(f"source layer {ordinal} metadata is inconsistent")
        if plan_hash is None:
            plan_hash = row["superfragment_plan_sha256"]  # type: ignore[assignment]
        elif row["superfragment_plan_sha256"] != plan_hash:
            raise ValueError("source layers bind different superfragment plans")
        result.append(row)
    return tuple(result)


def _authenticated_refit_fits(
    value: Sequence[FullMLPStackGeneratorFit],
) -> tuple[FullMLPStackGeneratorFit, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("refit_generator_fits must be a sequence")
    result = tuple(value)
    if any(
        not isinstance(item, FullMLPStackGeneratorFit)
        for item in result
    ):
        raise TypeError("refit_generator_fits contain an invalid item")
    fits = result  # type: ignore[assignment]
    for fit in fits:
        fit.validate_integrity()
    if tuple(
        fit.superfragment.layer_ordinal for fit in fits
    ) != _REFIT_ORDINALS:
        raise ValueError("refit fits must be exact ordered layers 10 through 17")
    return fits


def _validate_metrics(
    value: Mapping[str, object],
    *,
    label: str,
    observations: int,
) -> None:
    _strict_fields(value, _METRIC_FIELDS, label=label)
    if value["observations"] != observations:
        raise ValueError(f"{label} observations differ from captured rows")
    for name in (
        "mse",
        "nrmse",
        "weighted_mse",
        "weighted_nrmse",
        "max_abs_error",
        "target_rms",
        "weighted_target_rms",
    ):
        _require_finite(
            value[name],
            label=f"{label} {name}",
            minimum=0.0,
        )
    for name in ("cosine_similarity", "weighted_cosine_similarity"):
        _require_finite(
            value[name],
            label=f"{label} {name}",
            minimum=-1.0,
            maximum=1.0,
        )


def _validate_fit_resource_metadata(
    value: Mapping[str, object],
    *,
    label: str,
) -> None:
    _strict_fields(value, _FIT_RESOURCE_FIELDS, label=label)
    for name in (
        "native_mlp_parameter_count",
        "native_mlp_linear_macs_per_token",
        "selected_basis_stored_scalar_count",
        "coordinate_generator_parameter_count",
        "coordinate_generator_macs_per_token",
        "selected_unfused_stored_scalar_count",
        "dense_fused_parameter_count",
        "dense_fused_macs_per_token",
    ):
        _require_int(value[name], label=f"{label} {name}", minimum=1)
    for name in (
        "net_stored_parameter_savings",
        "net_linear_macs_saved_per_token",
    ):
        _require_int(value[name], label=f"{label} {name}", minimum=0)
    for name in (
        "dense_parameter_reduction_fraction",
        "dense_linear_mac_reduction_fraction",
    ):
        _require_finite(
            value[name],
            label=f"{label} {name}",
            minimum=0.0,
            maximum=1.0,
        )
    if value["dense_execution_stores_basis_separately"] is not False:
        raise ValueError(f"{label} must describe dense fused execution")


def _source_resource_subset(
    source: Mapping[str, object],
) -> tuple[int, int, int]:
    return (
        source["native_mlp_parameter_count"],  # type: ignore[return-value]
        source["dense_fused_parameter_count"],  # type: ignore[return-value]
        source["dense_fused_macs_per_token"],  # type: ignore[return-value]
    )


def _validate_refit_layers(
    layer_refits: object,
    *,
    source_layers: tuple[Mapping[str, object], ...],
    fits: tuple[FullMLPStackGeneratorFit, ...],
    splits: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if (
        isinstance(layer_refits, (str, bytes))
        or not isinstance(layer_refits, Sequence)
    ):
        raise TypeError("layer_refits must be a sequence")
    rows = tuple(layer_refits)
    if len(rows) != len(_REFIT_ORDINALS):
        raise ValueError("layer_refits must contain exactly eight rows")
    deployed_plan_hashes = [
        source["dense_plan_sha256"] for source in source_layers[:_REFIT_START]
    ]
    result: list[Mapping[str, object]] = []
    for expected_ordinal, row, fit in zip(
        _REFIT_ORDINALS,
        rows,
        fits,
        strict=True,
    ):
        if not isinstance(row, Mapping):
            raise TypeError("layer_refits entries must be mappings")
        _strict_fields(
            row,
            _LAYER_REFIT_FIELDS,
            label=f"layer {expected_ordinal} refit",
        )
        source = source_layers[expected_ordinal]
        if row["layer_ordinal"] != expected_ordinal:
            raise ValueError("layer refits must be ordered 10 through 17")
        prefix_ordinals = _canonical_ordinals(
            row["generated_prefix_ordinals"],
            label=f"layer {expected_ordinal} generated prefix ordinals",
        )
        prefix_hashes = _canonical_hashes(
            row["generated_prefix_plan_sha256s"],
            label=f"layer {expected_ordinal} generated prefix plans",
        )
        expected_prefix = tuple(range(expected_ordinal))
        if (
            prefix_ordinals != expected_prefix
            or prefix_hashes != tuple(deployed_plan_hashes)
            or row["generated_prefix_catalog_sha256"]
            != compiled_prefix_catalog_sha256(
                prefix_ordinals,
                prefix_hashes,
            )
        ):
            raise ValueError(
                f"layer {expected_ordinal} compiled-prefix lineage is invalid"
            )
        for name in (
            "source_fit_sha256",
            "refit_fit_sha256",
            "fit_row_key_sha256",
            "selection_row_key_sha256",
            "generated_prefix_catalog_sha256",
        ):
            _require_sha256(
                row[name],
                label=f"layer {expected_ordinal} {name}",
            )
        if (
            row["source_fit_sha256"] != source["source_fit_sha256"]
            or row["refit_fit_sha256"] != fit.artifact_sha256
            or row["refit_fit_sha256"] == row["source_fit_sha256"]
            or row["fit_row_key_sha256"] != fit.fit_row_key_sha256
            or row["selection_row_key_sha256"] != fit.eval_row_key_sha256
            or fit.superfragment.layer_ordinal != expected_ordinal
            or fit.superfragment.layer_id != source["layer_id"]
            or fit.superfragment.input_site != source["input_site"]
            or fit.superfragment.output_site != source["output_site"]
            or fit.superfragment.input_width != source["input_width"]
            or fit.superfragment.mode_count != source["intermediate_width"]
            or fit.superfragment.output_width != source["residual_width"]
            or fit.superfragment.artifact_sha256
            != source["superfragment_sha256"]
            or fit.superfragment_plan_sha256
            != source["superfragment_plan_sha256"]
            or fit.superfragment.source_model_sha256
            != source["source_model_sha256"]
            or fit.superfragment.parameter_catalog_sha256
            != source["parameter_catalog_sha256"]
            or fit.superfragment.source_fisher_coupling_sha256
            != source["source_fisher_coupling_sha256"]
            or fit.superfragment.source_fragment_plan_sha256
            != source["source_fragment_plan_sha256"]
            or fit.superfragment.source_cluster_plan_sha256
            != source["source_cluster_plan_sha256"]
        ):
            raise ValueError(
                f"layer {expected_ordinal} refit lineage differs from source"
            )
        fit_binding = fit.selected_basis.binding
        if (
            fit_binding.fit_split_sha256
            != splits["fit"]["serialized_sha256"]  # type: ignore[index]
            or fit_binding.eval_split_sha256
            != splits["selection"]["serialized_sha256"]  # type: ignore[index]
        ):
            raise ValueError(
                f"layer {expected_ordinal} split binding is invalid"
            )
        fit_observations = _require_int(
            row["fit_observations"],
            label=f"layer {expected_ordinal} fit_observations",
            minimum=1,
        )
        selection_observations = _require_int(
            row["selection_observations"],
            label=f"layer {expected_ordinal} selection_observations",
            minimum=1,
        )
        fit_sequences = _require_int(
            row["fit_sequences"],
            label=f"layer {expected_ordinal} fit_sequences",
            minimum=1,
        )
        selection_sequences = _require_int(
            row["selection_sequences"],
            label=f"layer {expected_ordinal} selection_sequences",
            minimum=1,
        )
        if (
            fit_sequences != 40
            or selection_sequences != 20
            or fit_sequences > fit_observations
            or selection_sequences > selection_observations
        ):
            raise ValueError(
                f"layer {expected_ordinal} row coverage is invalid"
            )
        for name, observations in (
            ("old_plan_fit_metrics", fit_observations),
            ("refit_plan_fit_metrics", fit_observations),
            ("old_plan_selection_metrics", selection_observations),
            ("refit_plan_selection_metrics", selection_observations),
        ):
            metrics = row[name]
            if not isinstance(metrics, Mapping):
                raise TypeError(
                    f"layer {expected_ordinal} {name} must be a mapping"
                )
            _validate_metrics(
                metrics,
                label=f"layer {expected_ordinal} {name}",
                observations=observations,
            )
        rank_fields = (
            "old_selected_mode_rank",
            "old_selected_generator_rank",
            "refit_selected_mode_rank",
            "refit_selected_generator_rank",
        )
        for name in rank_fields:
            _require_int(
                row[name],
                label=f"layer {expected_ordinal} {name}",
                minimum=1,
            )
        if (
            row["old_selected_mode_rank"] != source["selected_mode_rank"]
            or row["old_selected_generator_rank"]
            != source["selected_generator_rank"]
            or row["refit_selected_mode_rank"] != fit.selected_mode_rank
            or row["refit_selected_generator_rank"]
            != fit.selected_generator_rank
            or row["refit_selected_mode_rank"]
            != row["old_selected_mode_rank"]
            or row["refit_selected_generator_rank"]
            != row["old_selected_generator_rank"]
        ):
            raise ValueError(
                f"layer {expected_ordinal} changed a fixed rank"
            )
        resources = row["refit_resource_metadata"]
        if not isinstance(resources, Mapping):
            raise TypeError("refit_resource_metadata must be a mapping")
        _validate_fit_resource_metadata(
            resources,
            label=f"layer {expected_ordinal} refit resources",
        )
        if (
            resources != fit.resource_metadata
            or _source_resource_subset(source)
            != (
                resources["native_mlp_parameter_count"],
                resources["dense_fused_parameter_count"],
                resources["dense_fused_macs_per_token"],
            )
            or fit.executable_plan.parameter_count
            != source["dense_fused_parameter_count"]
            or fit.executable_plan.macs_per_token
            != source["dense_fused_macs_per_token"]
        ):
            raise ValueError(
                f"layer {expected_ordinal} changed its resource budget"
            )
        deployed_plan_hashes.append(fit.executable_plan.artifact_sha256)
        result.append(row)
    return tuple(result)


def _resource_accounting(
    *,
    model: Mapping[str, object],
    source_layers: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    source = model["source_whole_model_learned_parameters"]
    native = sum(
        row["native_mlp_parameter_count"] for row in source_layers
    )
    generated = sum(
        row["dense_fused_parameter_count"] for row in source_layers
    )
    native_macs = sum(
        row["native_mlp_parameter_count"] for row in source_layers
    )
    generated_macs = sum(
        row["dense_fused_macs_per_token"] for row in source_layers
    )
    additions = sum(row["residual_width"] for row in source_layers)
    removed_modes = sum(
        row["intermediate_width"] for row in source_layers
    )
    retained = source - native
    candidate = retained + generated
    if retained < 0 or generated >= native or generated_macs >= native_macs:
        raise ValueError("full-stack source resource totals are invalid")
    return {
        "scope": "full_native_mlp_stack_replacement_overlay",
        "source_whole_model_learned_parameters": source,
        "native_mlp_stack_learned_parameters": native,
        "retained_native_non_mlp_learned_parameters": retained,
        "dense_generator_stack_learned_parameters": generated,
        "logical_candidate_learned_parameters": candidate,
        "net_stored_parameter_savings": native - generated,
        "removed_mode_count": removed_modes,
        "native_mlp_stack_linear_macs_per_token": native_macs,
        "dense_generator_stack_macs_per_token": generated_macs,
        "dense_generator_stack_bias_additions_per_token": additions,
        "net_linear_macs_saved_per_token": native_macs - generated_macs,
        "unchanged_layer_count": len(_UNCHANGED_ORDINALS),
        "refit_layer_count": len(_REFIT_ORDINALS),
        "per_layer_rank_and_resource_budget_unchanged": True,
        "logical_candidate_excludes_native_mlp_stack": True,
        "whole_transformer_replaced": False,
    }


def _validate_lm_metrics(
    value: Mapping[str, object],
    *,
    label: str,
    native_nll: float | None,
) -> float:
    _strict_fields(value, _LM_METRIC_FIELDS, label=label)
    nll = _require_finite(
        value["nll_per_token"],
        label=f"{label} nll_per_token",
        minimum=0.0,
    )
    delta = _require_finite(
        value["delta_nll_per_token"],
        label=f"{label} delta_nll_per_token",
    )
    _require_finite(
        value["native_to_candidate_kl_per_token"],
        label=f"{label} native_to_candidate_kl_per_token",
        minimum=0.0,
    )
    _require_finite(
        value["top1_agreement_to_native"],
        label=f"{label} top1_agreement_to_native",
        minimum=0.0,
        maximum=1.0,
    )
    if native_nll is None:
        if (
            delta != 0.0
            or value["native_to_candidate_kl_per_token"] != 0.0
            or value["top1_agreement_to_native"] != 1.0
        ):
            raise ValueError("native metrics must be the exact baseline")
    elif not math.isclose(
        delta,
        nll - native_nll,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} delta NLL differs from native")
    return nll


def _validate_evaluation(
    evaluation: Mapping[str, object],
    *,
    splits: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    resources: Mapping[str, object],
) -> None:
    _strict_fields(evaluation, _EVALUATION_FIELDS, label="evaluation")
    assessment = splits["assessment"]
    if (
        evaluation["execution_path"]
        != "sequential_refit_full_mlp_stack_rung"
        or evaluation["assessment_role"]
        != "open_development_assessment"
        or evaluation["heldout_confirmation"] is not False
        or evaluation["assessment_membership_exact"] is not True
        or evaluation["refit_frozen_before_assessment"] is not True
        or evaluation["fit_and_selection_used_for_refit"] is not True
        or evaluation["assessment_used_for_refit"] is not False
        or evaluation["generator_rank_selection_performed"] is not False
        or evaluation["latency_or_kernel_speed_claim"] is not False
        or evaluation["assessment_split_sha256"]
        != assessment["serialized_sha256"]  # type: ignore[index]
        or evaluation["supervised_tokens"]
        != assessment["supervised_tokens"]  # type: ignore[index]
        or evaluation["logical_valid_tokens"]
        != assessment["logical_valid_tokens"]  # type: ignore[index]
        or evaluation["resource_accounting"] != resources
    ):
        raise ValueError("evaluation overclaims, leaks, or drifts in scope")
    _require_int(
        evaluation["supervised_tokens"],
        label="evaluation supervised_tokens",
        minimum=1,
    )
    _require_int(
        evaluation["logical_valid_tokens"],
        label="evaluation logical_valid_tokens",
        minimum=1,
    )
    conditions = evaluation["conditions"]
    if not isinstance(conditions, Mapping):
        raise TypeError("evaluation conditions must be a mapping")
    _strict_fields(conditions, _CONDITION_FIELDS, label="conditions")
    native = conditions["native"]
    if not isinstance(native, Mapping):
        raise TypeError("native condition must be a mapping")
    native_nll = _validate_lm_metrics(
        native,
        label="native condition",
        native_nll=None,
    )
    for name in (
        "frozen_generated_full_stack",
        "sequential_refit_full_stack",
        "matched_deletion",
    ):
        condition = conditions[name]
        if not isinstance(condition, Mapping):
            raise TypeError(f"{name} condition must be a mapping")
        _validate_lm_metrics(
            condition,
            label=f"{name} condition",
            native_nll=native_nll,
        )
    baseline = {
        "native": conditions["native"],
        "generated_full_stack": conditions[
            "frozen_generated_full_stack"
        ],
        "matched_deletion": conditions["matched_deletion"],
    }
    expected_baseline = frozen_baseline_conditions_sha256(baseline)
    declared_baseline = _require_sha256(
        evaluation["frozen_baseline_conditions_sha256"],
        label="evaluation frozen_baseline_conditions_sha256",
    )
    if (
        declared_baseline != expected_baseline
        or declared_baseline
        != frozen_sources["full_stack"]["baseline_conditions_sha256"]  # type: ignore[index]
    ):
        raise ValueError("evaluation differs from its frozen baseline")
    control = evaluation["control_validation"]
    if not isinstance(control, Mapping):
        raise TypeError("control_validation must be a mapping")
    _strict_fields(control, _CONTROL_FIELDS, label="control validation")
    if control != {
        "native_matches_frozen_full_stack_artifact": True,
        "frozen_generated_matches_frozen_full_stack_artifact": True,
        "matched_deletion_matches_frozen_full_stack_artifact": True,
        "frozen_generated_matches_trajectory_full_stack_endpoint": True,
        "physical_scope_identical": True,
        "refit_generator_compute_executed": True,
        "matched_deletion_compute_zero": True,
    }:
        raise ValueError("evaluation controls are incomplete")


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[FullMLPStackGeneratorFit, ...],
    tuple[Mapping[str, object], ...],
]:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="refit artifact")
    if (
        raw["schema"] != GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA
        or raw["format_version"] != GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("refit artifact header or safety is invalid")
    _assert_source_safe(raw)
    digest = _require_sha256(
        raw["scientific_payload_sha256"],
        label="scientific_payload_sha256",
    )
    without_digest = {
        key: value
        for key, value in raw.items()
        if key != "scientific_payload_sha256"
    }
    if _payload_sha256(without_digest) != digest:
        raise ValueError("refit scientific payload hash mismatch")
    model = raw["model"]
    frozen_sources = raw["frozen_sources"]
    splits = raw["splits"]
    protocol = raw["protocol"]
    evaluation = raw["evaluation"]
    resources = raw["resource_accounting"]
    for value, label in (
        (model, "model"),
        (frozen_sources, "frozen_sources"),
        (splits, "splits"),
        (protocol, "protocol"),
        (evaluation, "evaluation"),
        (resources, "resource_accounting"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_model(model)
    _validate_frozen_sources(frozen_sources)
    _validate_splits(splits)
    _validate_protocol(protocol)
    source_layers = _validate_source_layers(
        raw["source_layer_summaries"],
        model=model,
    )
    prefix_hashes = _canonical_hashes(
        raw["unchanged_prefix_fit_sha256s"],
        label="unchanged_prefix_fit_sha256s",
    )
    if prefix_hashes != tuple(
        row["source_fit_sha256"]
        for row in source_layers[:_REFIT_START]
    ):
        raise ValueError("unchanged prefix fit hashes differ from source")
    raw_fits = raw["refit_generator_fits"]
    if type(raw_fits) is not tuple:
        raise TypeError("serialized refit_generator_fits must be a tuple")
    fits = tuple(
        FullMLPStackGeneratorFit.from_state_dict(value)
        for value in raw_fits  # type: ignore[arg-type]
    )
    layer_refits = _validate_refit_layers(
        raw["layer_refits"],
        source_layers=source_layers,
        fits=fits,
        splits=splits,
    )
    expected_resources = _resource_accounting(
        model=model,
        source_layers=source_layers,
    )
    _strict_fields(
        resources,
        _RESOURCE_FIELDS,
        label="resource accounting",
    )
    if resources != expected_resources:
        raise ValueError("saved resource accounting is inconsistent")
    _validate_evaluation(
        evaluation,
        splits=splits,
        frozen_sources=frozen_sources,
        resources=resources,
    )
    return source_layers, fits, layer_refits


def build_gemma3_full_mlp_stack_refit_payload(
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    splits: Mapping[str, object],
    protocol: Mapping[str, object],
    source_layer_summaries: Sequence[Mapping[str, object]],
    refit_generator_fits: Sequence[FullMLPStackGeneratorFit],
    layer_refits: Sequence[Mapping[str, object]],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Build and fully validate one sequential refit overlay payload."""

    fits = _authenticated_refit_fits(refit_generator_fits)
    for value, label in (
        (model, "model"),
        (frozen_sources, "frozen_sources"),
        (splits, "splits"),
        (protocol, "protocol"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    source_rows = tuple(dict(value) for value in source_layer_summaries)
    refit_rows = tuple(dict(value) for value in layer_refits)
    validated_source = _validate_source_layers(source_rows, model=model)
    resources = _resource_accounting(
        model=model,
        source_layers=validated_source,
    )
    evaluation_copy = dict(evaluation)
    evaluation_copy["resource_accounting"] = resources
    without_digest: dict[str, object] = {
        "schema": GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA,
        "format_version": GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION,
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "frozen_sources": dict(frozen_sources),
        "splits": dict(splits),
        "protocol": dict(protocol),
        "source_layer_summaries": source_rows,
        "unchanged_prefix_fit_sha256s": tuple(
            row["source_fit_sha256"]
            for row in validated_source[:_REFIT_START]
        ),
        "refit_generator_fits": tuple(
            fit.state_dict() for fit in fits
        ),
        "layer_refits": refit_rows,
        "resource_accounting": resources,
        "evaluation": evaluation_copy,
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    payload = {
        **without_digest,
        "scientific_payload_sha256": _payload_sha256(without_digest),
    }
    _validate_and_restore_payload(payload)
    return payload


def build_gemma3_full_mlp_stack_refit_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Build a compact, tensor-free report for one strict overlay payload."""

    source_layers, fits, layer_refits = _validate_and_restore_payload(payload)
    _require_name(tensor_file, label="tensor_file")
    report_without_digest: dict[str, object] = {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "scientific_status": payload["scientific_status"],
        "model": payload["model"],
        "frozen_sources": payload["frozen_sources"],
        "splits": {
            name: (
                {
                    "role": payload["splits"][name]["role"],  # type: ignore[index]
                    "serialized_sha256": payload["splits"][name][  # type: ignore[index]
                        "serialized_sha256"
                    ],
                    "example_count": payload["splits"][name][  # type: ignore[index]
                        "example_count"
                    ],
                    "logical_valid_tokens": payload["splits"][name][  # type: ignore[index]
                        "logical_valid_tokens"
                    ],
                    "supervised_tokens": payload["splits"][name][  # type: ignore[index]
                        "supervised_tokens"
                    ],
                }
                if name != "provenance"
                else payload["splits"][name]  # type: ignore[index]
            )
            for name in ("fit", "selection", "assessment", "provenance")
        },
        "protocol": payload["protocol"],
        "unchanged_prefix": tuple(
            {
                "layer_ordinal": row["layer_ordinal"],
                "layer_id": row["layer_id"],
                "source_fit_sha256": row["source_fit_sha256"],
                "dense_plan_sha256": row["dense_plan_sha256"],
            }
            for row in source_layers[:_REFIT_START]
        ),
        "refit_layers": tuple(
            {
                **dict(row),
                "refit_dense_plan_sha256": fit.executable_plan.artifact_sha256,
            }
            for row, fit in zip(layer_refits, fits, strict=True)
        ),
        "resource_accounting": payload["resource_accounting"],
        "evaluation": payload["evaluation"],
        "artifact": {
            "tensor_file": tensor_file,
            "scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
            "safety": payload["safety"],
        },
    }
    _assert_source_safe(report_without_digest)
    return {
        **report_without_digest,
        "report_sha256": _json_digest(
            report_without_digest,
            domain=_REPORT_DOMAIN,
        ),
    }


def _write_tensor_temp(path: Path, payload: Mapping[str, object]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_temp(path: Path, report: Mapping[str, object]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _strict_json_report(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid non-finite JSON constant {value}")

    def strict_object(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=strict_object,
        )
    if not isinstance(value, Mapping):
        raise TypeError("refit report must be a mapping")
    _assert_source_safe(value)
    return value


def save_gemma3_full_mlp_stack_refit_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    splits: Mapping[str, object],
    protocol: Mapping[str, object],
    source_layer_summaries: Sequence[Mapping[str, object]],
    refit_generator_fits: Sequence[FullMLPStackGeneratorFit],
    layer_refits: Sequence[Mapping[str, object]],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Atomically save a strict tensor overlay and tensor-free JSON report."""

    output = Path(path)
    if output.suffix != ".pt":
        raise ValueError("refit artifact output must use .pt")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite refit artifact")
    payload = build_gemma3_full_mlp_stack_refit_payload(
        model=model,
        frozen_sources=frozen_sources,
        splits=splits,
        protocol=protocol,
        source_layer_summaries=source_layer_summaries,
        refit_generator_fits=refit_generator_fits,
        layer_refits=layer_refits,
        evaluation=evaluation,
    )
    report = build_gemma3_full_mlp_stack_refit_report(
        payload,
        tensor_file=output.name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_temp = _write_tensor_temp(output, payload)
    try:
        report_temp = _write_json_temp(report_path, report)
    except BaseException:
        tensor_temp.unlink(missing_ok=True)
        raise
    tensor_published = False
    report_published = False
    try:
        os.link(tensor_temp, output)
        tensor_published = True
        os.link(report_temp, report_path)
        report_published = True
        restored = load_gemma3_full_mlp_stack_refit_artifact(output)
        if restored["scientific_payload_sha256"] != (
            payload["scientific_payload_sha256"]
        ):
            raise ValueError("post-save refit payload verification failed")
        if _canonical_json_bytes(
            _strict_json_report(report_path)
        ) != _canonical_json_bytes(report):
            raise ValueError("post-save refit report verification failed")
    except FileExistsError as error:
        if report_published:
            report_path.unlink(missing_ok=True)
        if tensor_published:
            output.unlink(missing_ok=True)
        raise FileExistsError(
            "refusing to overwrite refit artifact"
        ) from error
    except BaseException:
        if report_published:
            report_path.unlink(missing_ok=True)
        if tensor_published:
            output.unlink(missing_ok=True)
        raise
    finally:
        tensor_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
    return report


def load_gemma3_full_mlp_stack_refit_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load every nested refit generator artifact."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("refit artifact must be a dict")
    _validate_and_restore_payload(raw)
    return raw
