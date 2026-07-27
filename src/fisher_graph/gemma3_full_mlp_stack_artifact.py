"""Strict source-safe artifacts for the exhaustive Gemma MLP-stack rung.

The artifact represented here replaces every native MLP in the 18-block Gemma
checkpoint with one dense fused residual generator.  It does *not* replace the
whole transformer: embeddings, attention, normalization, and the language
model head remain native.

Only authenticated compiler artifacts and compact machine metadata are saved.
Prompt text, token IDs, raw activation/gradient rows, and source weights are
forbidden.  The outer tensor-aware digest detects accidental mutation; it is
not a cryptographic signature or external provenance attestation.
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
from .parameter_layer_superfragments import (
    ParameterLayerSuperfragmentPlan,
)


__all__ = [
    "GEMMA3_FULL_MLP_STACK_FORMAT_VERSION",
    "GEMMA3_FULL_MLP_STACK_SCHEMA",
    "build_gemma3_full_mlp_stack_payload",
    "build_gemma3_full_mlp_stack_report",
    "load_gemma3_full_mlp_stack_artifact",
    "save_gemma3_full_mlp_stack_artifact",
]


GEMMA3_FULL_MLP_STACK_SCHEMA = (
    "fisher_graph.gemma3_full_native_mlp_stack_development"
)
GEMMA3_FULL_MLP_STACK_FORMAT_VERSION = 1

_EXPECTED_LAYER_COUNT = 18
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_full_mlp_stack.payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_full_mlp_stack.report.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_NATIVE_COMPONENTS = (
    "embeddings",
    "attention",
    "normalization",
    "language_model_head",
)
_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "development_only_full_native_mlp_stack_measurement",
    "development_only": True,
    "compression_claim": False,
    "heldout_confirmation": False,
    "scope": "full_native_mlp_stack_only",
    "full_native_mlp_stack_replaced": True,
    "whole_transformer_replaced": False,
    "native_components_retained": _NATIVE_COMPONENTS,
    "assessment_role": "open_development_assessment",
    "assessment_used_for_fitting": False,
    "ready_for_closed_heldout": False,
}
_SAFETY: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_token_rows": False,
    "contains_tokenizer_state": False,
    "contains_computational_mode_bases": True,
    "contains_generator_weights": True,
    "executable_dense_generator_stack": True,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "text",
        "token_ids",
        "input_ids",
        "targets",
        "score_gradients",
        "activation_rows",
        "gradient_rows",
        "raw_prompt_rows",
        "raw_token_rows",
        "raw_fit_rows",
        "raw_eval_rows",
        "source_model_weights",
        "source_parameter_values",
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
    "protocol",
    "splits",
    "upstream_metadata",
    "superfragment_plan",
    "generator_fits",
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
_PROTOCOL_FIELDS = {
    "scope",
    "transformer_layer_count",
    "source_fragment_count",
    "removed_mode_count",
    "mode_ranks",
    "selected_mode_rank",
    "generator_ranks",
    "selected_generator_rank",
    "generator_ridge",
    "fit_rule",
    "execution_path",
    "native_components_retained",
    "local_files_only",
}
_SPLIT_FIELDS = {
    "fit_export",
    "eval_export",
    "fit",
    "upstream_evaluation",
    "selection",
    "assessment",
    "partition",
    "provenance",
}
_UPSTREAM_FIELDS = {
    "source_schema",
    "source_format_version",
    "source_scientific_payload_sha256",
    "fit_prompt_trace_sha256",
    "parameter_catalog_sha256",
    "fisher_coupling_sha256",
    "parameter_clusters_sha256",
    "parameter_cluster_fragments_sha256",
}
_EVALUATION_FIELDS = {
    "execution_path",
    "assessment_role",
    "heldout_confirmation",
    "assessment_membership_exact",
    "assessment_used_for_fitting",
    "supervised_tokens",
    "logical_valid_tokens",
    "declared_scope",
    "conditions",
    "control_validation",
    "resource_accounting",
    "latency_or_kernel_speed_claim",
    "assessment_split_sha256",
}
_DECLARED_SCOPE_FIELDS = {
    "replacement_scope",
    "layer_count",
    "removed_mode_count",
    "mode_counts_by_layer",
    "all_declared_layers_and_modes_replaced",
}
_CONDITION_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_RESOURCE_FIELDS = {
    "scope",
    "source_whole_model_learned_parameters",
    "native_mlp_stack_learned_parameters",
    "retained_native_non_mlp_learned_parameters",
    "dense_generator_stack_learned_parameters",
    "logical_candidate_learned_parameters",
    "net_stored_parameter_savings",
    "native_mlp_stack_linear_macs_per_token",
    "dense_generator_stack_macs_per_token",
    "dense_generator_stack_bias_additions_per_token",
    "net_linear_macs_saved_per_token",
    "logical_candidate_excludes_native_mlp_stack",
    "whole_transformer_replaced",
    "experimental_artifact_includes_analysis_curves",
    "logical_deployment_requires_analysis_curves",
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _canonical_ranks(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value)
    if (
        not result
        or any(type(item) is not int or item <= 0 for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(f"{label} must be unique increasing positive integers")
    return result


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _update_payload_digest(digest: object, value: object) -> None:
    """Hash JSON-like values and exact logical tensor bytes deterministically."""

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
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("scientific payload tensors must be finite")
        digest.update(b"T")
        _update_payload_digest(digest, str(tensor.dtype))
        _update_payload_digest(digest, tuple(tensor.shape))
        raw = tensor.numpy().tobytes(order="C")
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


def _authenticated_plan(
    value: ParameterLayerSuperfragmentPlan,
) -> ParameterLayerSuperfragmentPlan:
    if not isinstance(value, ParameterLayerSuperfragmentPlan):
        raise TypeError(
            "superfragment_plan must be ParameterLayerSuperfragmentPlan"
        )
    return ParameterLayerSuperfragmentPlan.from_state_dict(
        value.state_dict()
    )


def _authenticated_fits(
    values: Sequence[FullMLPStackGeneratorFit],
) -> tuple[FullMLPStackGeneratorFit, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("generator_fits must be a sequence")
    result = tuple(
        FullMLPStackGeneratorFit.from_state_dict(value.state_dict())
        if isinstance(value, FullMLPStackGeneratorFit)
        else None
        for value in values
    )
    if any(value is None for value in result):
        raise TypeError("generator_fits contain an invalid value")
    return result  # type: ignore[return-value]


def _validate_model(
    model: Mapping[str, object],
    *,
    source_model_sha256: str,
) -> None:
    _strict_fields(model, _MODEL_FIELDS, label="model metadata")
    _require_name(model["model_id"], label="model_id")
    requested = model["requested_revision"]
    resolved = model["resolved_commit"]
    if (
        not isinstance(requested, str)
        or _REVISION.fullmatch(requested) is None
        or requested != resolved
    ):
        raise ValueError("model revisions must bind the same exact commit")
    if (
        _require_sha256(
            model["adapter_model_fingerprint"],
            label="adapter_model_fingerprint",
        )
        != source_model_sha256
    ):
        raise ValueError("model fingerprint differs from superfragment plan")
    _require_int(
        model["source_whole_model_learned_parameters"],
        label="source_whole_model_learned_parameters",
        minimum=1,
    )
    if model["local_files_only"] is not True:
        raise ValueError("full-stack development artifact must be local-only")


def _validate_protocol(
    protocol: Mapping[str, object],
    *,
    plan: ParameterLayerSuperfragmentPlan,
    fits: tuple[FullMLPStackGeneratorFit, ...],
) -> None:
    _strict_fields(protocol, _PROTOCOL_FIELDS, label="protocol")
    mode_ranks = _canonical_ranks(protocol["mode_ranks"], label="mode_ranks")
    generator_ranks = _canonical_ranks(
        protocol["generator_ranks"],
        label="generator_ranks",
    )
    selected_mode = protocol["selected_mode_rank"]
    selected_generator = protocol["selected_generator_rank"]
    ridge = protocol["generator_ridge"]
    if (
        protocol["scope"] != "full_native_mlp_stack_replacement"
        or protocol["transformer_layer_count"] != _EXPECTED_LAYER_COUNT
        or protocol["transformer_layer_count"] != plan.layer_count
        or protocol["source_fragment_count"] != plan.source_fragment_count
        or protocol["removed_mode_count"] != plan.assigned_group_count
        or type(selected_mode) is not int
        or selected_mode not in mode_ranks
        or type(selected_generator) is not int
        or selected_generator not in generator_ranks
        or generator_ranks[-1] > selected_mode
        or isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not math.isfinite(float(ridge))
        or float(ridge) < 0.0
        or protocol["fit_rule"]
        != "fisher_weighted_full_layer_before_modes"
        or protocol["execution_path"]
        != "edgeless_dense_fused_residual_generators"
        or protocol["native_components_retained"] != _NATIVE_COMPONENTS
        or protocol["local_files_only"] is not True
    ):
        raise ValueError(
            "protocol is not an exhaustive full-native-MLP-stack declaration"
        )
    for fit in fits:
        if (
            fit.computational_modes.config.ranks != mode_ranks
            or fit.selected_mode_rank != selected_mode
            or fit.coordinate_generators.config.ranks != generator_ranks
            or fit.selected_generator_rank != selected_generator
            or fit.selected_coordinate_plan.config.ridge != float(ridge)
        ):
            raise ValueError("generator fit differs from the saved protocol")


def _content_hashes(
    entry: Mapping[str, object],
    *,
    label: str,
) -> tuple[str, ...]:
    values = entry.get("content_sha256")
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{label} content_sha256 must be a sequence")
    result = tuple(
        _require_sha256(value, label=f"{label} content_sha256")
        for value in values
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} content hashes must be nonempty and unique")
    _require_sha256(
        entry.get("serialized_sha256"),
        label=f"{label} serialized_sha256",
    )
    return result


def _validate_splits(splits: Mapping[str, object]) -> None:
    _strict_fields(splits, _SPLIT_FIELDS, label="split metadata")
    for name in _SPLIT_FIELDS:
        if not isinstance(splits[name], Mapping):
            raise TypeError(f"split {name} must be a mapping")
    fit = splits["fit"]
    upstream = splits["upstream_evaluation"]
    selection = splits["selection"]
    assessment = splits["assessment"]
    if (
        fit.get("role") != "generator_fit"
        or upstream.get("role") != "development_partition_source"
        or selection.get("role") != "generator_selection"
        or assessment.get("role") != "open_development_assessment"
    ):
        raise ValueError("split roles are invalid")
    fit_hashes = _content_hashes(fit, label="fit")
    upstream_hashes = _content_hashes(
        upstream,
        label="upstream evaluation",
    )
    selection_hashes = _content_hashes(selection, label="selection")
    assessment_hashes = _content_hashes(assessment, label="assessment")
    if (
        set(fit_hashes) & (set(selection_hashes) | set(assessment_hashes))
        or set(selection_hashes) & set(assessment_hashes)
        or set(selection_hashes) | set(assessment_hashes)
        != set(upstream_hashes)
    ):
        raise ValueError(
            "fit/selection/assessment content partition is invalid"
        )
    provenance = splits["provenance"]
    if set(provenance) != {
        "assurance",
        "externally_authenticated",
        "selection_assessment_disjoint",
        "heldout_confirmation",
    } or (
        provenance["assurance"] != "caller_declared_self_attested"
        or provenance["externally_authenticated"] is not False
        or provenance["selection_assessment_disjoint"] is not True
        or provenance["heldout_confirmation"] is not False
    ):
        raise ValueError("split provenance overclaims its assurance")


def _validate_upstream(
    upstream: Mapping[str, object],
    *,
    plan: ParameterLayerSuperfragmentPlan,
) -> None:
    _strict_fields(upstream, _UPSTREAM_FIELDS, label="upstream metadata")
    _require_name(upstream["source_schema"], label="source_schema")
    _require_int(
        upstream["source_format_version"],
        label="source_format_version",
        minimum=1,
    )
    for name in _UPSTREAM_FIELDS - {
        "source_schema",
        "source_format_version",
    }:
        _require_sha256(upstream[name], label=name)
    if (
        upstream["parameter_catalog_sha256"]
        != plan.parameter_catalog_sha256
        or upstream["fisher_coupling_sha256"]
        != plan.source_fisher_coupling_sha256
        or upstream["parameter_clusters_sha256"]
        != plan.source_cluster_plan_sha256
        or upstream["parameter_cluster_fragments_sha256"]
        != plan.source_fragment_plan_sha256
    ):
        raise ValueError(
            "upstream analysis hashes differ from the superfragment plan"
        )


def _resource_accounting(
    *,
    model: Mapping[str, object],
    fits: tuple[FullMLPStackGeneratorFit, ...],
) -> dict[str, object]:
    source = int(model["source_whole_model_learned_parameters"])
    native = sum(
        fit.superfragment.native_parameter_count for fit in fits
    )
    generated = sum(
        fit.executable_plan.parameter_count for fit in fits
    )
    retained = source - native
    if retained < 0:
        raise ValueError("native MLP parameters exceed the source model")
    candidate = retained + generated
    native_macs = native
    generator_macs = sum(
        fit.executable_plan.macs_per_token for fit in fits
    )
    bias_additions = sum(
        fit.executable_plan.output_width
        if fit.executable_plan.factors.bias is not None
        else 0
        for fit in fits
    )
    return {
        "scope": "full_native_mlp_stack_only",
        "source_whole_model_learned_parameters": source,
        "native_mlp_stack_learned_parameters": native,
        "retained_native_non_mlp_learned_parameters": retained,
        "dense_generator_stack_learned_parameters": generated,
        "logical_candidate_learned_parameters": candidate,
        "net_stored_parameter_savings": source - candidate,
        "native_mlp_stack_linear_macs_per_token": native_macs,
        "dense_generator_stack_macs_per_token": generator_macs,
        "dense_generator_stack_bias_additions_per_token": bias_additions,
        "net_linear_macs_saved_per_token": native_macs - generator_macs,
        "logical_candidate_excludes_native_mlp_stack": True,
        "whole_transformer_replaced": False,
        "experimental_artifact_includes_analysis_curves": True,
        "logical_deployment_requires_analysis_curves": False,
    }


def _validate_evaluation(
    evaluation: Mapping[str, object],
    *,
    splits: Mapping[str, object],
    plan: ParameterLayerSuperfragmentPlan,
    resources: Mapping[str, object],
) -> None:
    _strict_fields(evaluation, _EVALUATION_FIELDS, label="evaluation")
    assessment_split = splits["assessment"]["serialized_sha256"]  # type: ignore[index]
    if (
        evaluation["execution_path"] != "edgeless_full_mlp_stack_rung"
        or evaluation["assessment_role"] != "open_development_assessment"
        or evaluation["heldout_confirmation"] is not False
        or evaluation["assessment_membership_exact"] is not True
        or evaluation["assessment_used_for_fitting"] is not False
        or evaluation["latency_or_kernel_speed_claim"] is not False
        or evaluation["assessment_split_sha256"] != assessment_split
    ):
        raise ValueError("evaluation overclaims or has invalid split lineage")
    _require_int(
        evaluation["supervised_tokens"],
        label="supervised_tokens",
        minimum=1,
    )
    valid_tokens = _require_int(
        evaluation["logical_valid_tokens"],
        label="logical_valid_tokens",
        minimum=1,
    )
    declared = evaluation["declared_scope"]
    if not isinstance(declared, Mapping):
        raise TypeError("evaluation declared_scope must be a mapping")
    _strict_fields(
        declared,
        _DECLARED_SCOPE_FIELDS,
        label="evaluation declared scope",
    )
    mode_counts = tuple(
        value.mode_count for value in plan.superfragments
    )
    if (
        declared["replacement_scope"]
        != "full_native_mlp_stack_replacement"
        or declared["layer_count"] != _EXPECTED_LAYER_COUNT
        or declared["removed_mode_count"] != sum(mode_counts)
        or tuple(declared["mode_counts_by_layer"]) != mode_counts
        or declared["all_declared_layers_and_modes_replaced"] is not True
    ):
        raise ValueError(
            "evaluation scope is not the complete native MLP stack"
        )
    conditions = evaluation["conditions"]
    controls = evaluation["control_validation"]
    execution_resources = evaluation["resource_accounting"]
    if (
        not isinstance(conditions, Mapping)
        or set(conditions)
        != {"native", "generated_full_stack", "matched_deletion"}
        or not isinstance(controls, Mapping)
        or controls
        != {
            "physical_scope_identical": True,
            "generated_compute_executed": True,
            "matched_deletion_compute_zero": True,
        }
        or not isinstance(execution_resources, Mapping)
        or set(execution_resources)
        != {"generated_full_stack", "matched_deletion"}
    ):
        raise ValueError("evaluation conditions or controls are invalid")
    metric_rows: dict[str, Mapping[str, object]] = {}
    for name in ("native", "generated_full_stack", "matched_deletion"):
        row = conditions[name]
        if not isinstance(row, Mapping):
            raise TypeError("evaluation condition metrics must be mappings")
        _strict_fields(
            row,
            _CONDITION_METRIC_FIELDS,
            label=f"evaluation condition {name}",
        )
        nll = row["nll_per_token"]
        delta = row["delta_nll_per_token"]
        divergence = row["native_to_candidate_kl_per_token"]
        agreement = row["top1_agreement_to_native"]
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0.0
            or isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or isinstance(divergence, bool)
            or not isinstance(divergence, (int, float))
            or not math.isfinite(float(divergence))
            or float(divergence) < 0.0
            or isinstance(agreement, bool)
            or not isinstance(agreement, (int, float))
            or not math.isfinite(float(agreement))
            or not 0.0 <= float(agreement) <= 1.0
        ):
            raise ValueError("evaluation condition metrics are invalid")
        metric_rows[name] = row
    native_metrics = metric_rows["native"]
    native_nll = float(native_metrics["nll_per_token"])
    if (
        float(native_metrics["delta_nll_per_token"]) != 0.0
        or float(
            native_metrics["native_to_candidate_kl_per_token"]
        )
        != 0.0
        or float(native_metrics["top1_agreement_to_native"]) != 1.0
    ):
        raise ValueError("native evaluation metrics must be exact baselines")
    for name in ("generated_full_stack", "matched_deletion"):
        row = metric_rows[name]
        expected_delta = float(row["nll_per_token"]) - native_nll
        if not math.isclose(
            float(row["delta_nll_per_token"]),
            expected_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("evaluation NLL delta does not match native")
    expected_static = {
        "source_whole_model_learned_parameters": resources[
            "source_whole_model_learned_parameters"
        ],
        "logical_native_mlp_stack_learned_parameters": resources[
            "native_mlp_stack_learned_parameters"
        ],
        "logical_retained_native_non_mlp_learned_parameters": resources[
            "retained_native_non_mlp_learned_parameters"
        ],
        "logical_generator_stack_learned_parameters": resources[
            "dense_generator_stack_learned_parameters"
        ],
        "logical_candidate_learned_parameters": resources[
            "logical_candidate_learned_parameters"
        ],
        "logical_net_stored_parameter_savings": resources[
            "net_stored_parameter_savings"
        ],
    }
    for name in ("generated_full_stack", "matched_deletion"):
        value = execution_resources[name]
        if not isinstance(value, Mapping):
            raise TypeError("evaluation resource entries must be mappings")
        if any(value.get(key) != expected for key, expected in expected_static.items()):
            raise ValueError("evaluation resource accounting differs from fits")
        if (
            value.get("replacement_scope")
            != "full_native_mlp_stack_replacement"
            or value.get("replaced_layer_count") != _EXPECTED_LAYER_COUNT
            or value.get("removed_mode_count") != plan.assigned_group_count
            or value.get("logical_linear_macs_native_mlp_stack")
            != valid_tokens
            * int(resources["native_mlp_stack_linear_macs_per_token"])
            or value.get("logical_generator_macs")
            != valid_tokens
            * int(resources["dense_generator_stack_macs_per_token"])
            or value.get("logical_generator_bias_additions")
            != valid_tokens
            * int(
                resources[
                    "dense_generator_stack_bias_additions_per_token"
                ]
            )
        ):
            raise ValueError("evaluation full-stack accounting is invalid")
    generated = execution_resources["generated_full_stack"]
    deletion = execution_resources["matched_deletion"]
    if (
        generated.get("logical_executed_generator_macs")
        != generated.get("logical_generator_macs")
        or generated.get("logical_executed_generator_bias_additions")
        != generated.get("logical_generator_bias_additions")
        or deletion.get("logical_executed_generator_macs") != 0
        or deletion.get("logical_executed_generator_bias_additions") != 0
    ):
        raise ValueError("evaluation generated/deletion work controls drifted")


def _validate_cross_lineage(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    plan: ParameterLayerSuperfragmentPlan,
    fits: tuple[FullMLPStackGeneratorFit, ...],
) -> None:
    expected_ordinals = tuple(range(_EXPECTED_LAYER_COUNT))
    if (
        plan.layer_count != _EXPECTED_LAYER_COUNT
        or tuple(
            value.layer_ordinal for value in plan.superfragments
        )
        != expected_ordinals
        or tuple(
            value.superfragment.layer_ordinal for value in fits
        )
        != expected_ordinals
    ):
        raise ValueError(
            "artifact requires ordered fits for exactly 18 native layers"
        )
    fit_split = splits["fit"]["serialized_sha256"]  # type: ignore[index]
    selection_split = splits["selection"]["serialized_sha256"]  # type: ignore[index]
    for expected_superfragment, fit in zip(
        plan.superfragments,
        fits,
        strict=True,
    ):
        if (
            fit.superfragment.artifact_sha256
            != expected_superfragment.artifact_sha256
            or fit.superfragment_plan_sha256 != plan.artifact_sha256
            or fit.selected_basis.binding.source_model_sha256
            != model["adapter_model_fingerprint"]
            or fit.selected_basis.binding.fit_split_sha256 != fit_split
            or fit.selected_basis.binding.eval_split_sha256
            != selection_split
            or fit.executable_plan.binding.parameter_cluster_fragment_sha256
            is not None
        ):
            raise ValueError(
                "ordered full-layer fit lineage differs from the artifact"
            )
    _validate_protocol(protocol, plan=plan, fits=fits)


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    ParameterLayerSuperfragmentPlan,
    tuple[FullMLPStackGeneratorFit, ...],
]:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="full-stack artifact")
    if (
        raw["schema"] != GEMMA3_FULL_MLP_STACK_SCHEMA
        or raw["format_version"] != GEMMA3_FULL_MLP_STACK_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("full-stack artifact header or safety is invalid")
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
        raise ValueError("full-stack scientific payload hash mismatch")
    if not isinstance(raw["superfragment_plan"], Mapping):
        raise TypeError("superfragment_plan state must be a mapping")
    plan = ParameterLayerSuperfragmentPlan.from_state_dict(
        raw["superfragment_plan"]
    )
    raw_fits = raw["generator_fits"]
    if type(raw_fits) is not tuple:
        raise TypeError("serialized generator_fits must be a tuple")
    fits = tuple(
        FullMLPStackGeneratorFit.from_state_dict(value)
        for value in raw_fits  # type: ignore[arg-type]
    )
    model = raw["model"]
    protocol = raw["protocol"]
    splits = raw["splits"]
    upstream = raw["upstream_metadata"]
    evaluation = raw["evaluation"]
    resources = raw["resource_accounting"]
    for value, label in (
        (model, "model"),
        (protocol, "protocol"),
        (splits, "splits"),
        (upstream, "upstream_metadata"),
        (evaluation, "evaluation"),
        (resources, "resource_accounting"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_model(model, source_model_sha256=plan.source_model_sha256)
    _validate_splits(splits)
    _validate_upstream(upstream, plan=plan)
    _validate_cross_lineage(
        model=model,
        protocol=protocol,
        splits=splits,
        plan=plan,
        fits=fits,
    )
    expected_resources = _resource_accounting(model=model, fits=fits)
    _strict_fields(
        resources,
        _RESOURCE_FIELDS,
        label="resource accounting",
    )
    if resources != expected_resources:
        raise ValueError("saved full-stack resource accounting is inconsistent")
    _validate_evaluation(
        evaluation,
        splits=splits,
        plan=plan,
        resources=resources,
    )
    return plan, fits


def build_gemma3_full_mlp_stack_payload(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    upstream_metadata: Mapping[str, object],
    superfragment_plan: ParameterLayerSuperfragmentPlan,
    generator_fits: Sequence[FullMLPStackGeneratorFit],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Build and fully validate one exhaustive development payload."""

    plan = _authenticated_plan(superfragment_plan)
    fits = _authenticated_fits(generator_fits)
    for value, label in (
        (model, "model"),
        (protocol, "protocol"),
        (splits, "splits"),
        (upstream_metadata, "upstream_metadata"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    resources = _resource_accounting(model=model, fits=fits)
    without_digest: dict[str, object] = {
        "schema": GEMMA3_FULL_MLP_STACK_SCHEMA,
        "format_version": GEMMA3_FULL_MLP_STACK_FORMAT_VERSION,
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "protocol": dict(protocol),
        "splits": dict(splits),
        "upstream_metadata": dict(upstream_metadata),
        "superfragment_plan": plan.state_dict(),
        "generator_fits": tuple(value.state_dict() for value in fits),
        "resource_accounting": resources,
        "evaluation": dict(evaluation),
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    payload = {
        **without_digest,
        "scientific_payload_sha256": _payload_sha256(without_digest),
    }
    _validate_and_restore_payload(payload)
    return payload


def _split_report(splits: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("fit", "upstream_evaluation", "selection", "assessment"):
        value = splits[name]
        result[name] = {
            "role": value["role"],
            "serialized_sha256": value["serialized_sha256"],
            "content_count": len(value["content_sha256"]),
        }
    result["provenance"] = dict(splits["provenance"])
    return result


def build_gemma3_full_mlp_stack_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Build a compact tensor-free JSON report from a strict payload."""

    plan, fits = _validate_and_restore_payload(payload)
    _require_name(tensor_file, label="tensor_file")
    report_without_digest: dict[str, object] = {
        "schema": GEMMA3_FULL_MLP_STACK_SCHEMA,
        "format_version": GEMMA3_FULL_MLP_STACK_FORMAT_VERSION,
        "scientific_status": payload["scientific_status"],
        "model": payload["model"],
        "protocol": payload["protocol"],
        "splits": _split_report(payload["splits"]),  # type: ignore[arg-type]
        "upstream_metadata": payload["upstream_metadata"],
        "superfragment_plan": {
            "artifact_sha256": plan.artifact_sha256,
            "layer_count": plan.layer_count,
            "source_fragment_count": plan.source_fragment_count,
            "source_group_count": plan.source_group_count,
            "assigned_native_parameter_count": (
                plan.assigned_native_parameter_count
            ),
        },
        "layers": tuple(
            {
                "layer_ordinal": fit.superfragment.layer_ordinal,
                "layer_id": fit.superfragment.layer_id,
                "fit_sha256": fit.artifact_sha256,
                "superfragment_sha256": (
                    fit.superfragment.artifact_sha256
                ),
                "selected_mode_rank": fit.selected_mode_rank,
                "selected_generator_rank": fit.selected_generator_rank,
                "dense_plan_sha256": fit.executable_plan.artifact_sha256,
                "resources": fit.resource_metadata,
            }
            for fit in fits
        ),
        "resource_accounting": payload["resource_accounting"],
        "evaluation": payload["evaluation"],
        "artifact": {
            "tensor_file": tensor_file,
            "scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
            "safety": dict(_SAFETY),
        },
    }
    _assert_source_safe(report_without_digest)
    return {
        **report_without_digest,
        "report_sha256": _json_sha256(
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


def save_gemma3_full_mlp_stack_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    upstream_metadata: Mapping[str, object],
    superfragment_plan: ParameterLayerSuperfragmentPlan,
    generator_fits: Sequence[FullMLPStackGeneratorFit],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Atomically save one strict tensor artifact and compact JSON report."""

    output = Path(path)
    if output.suffix != ".pt":
        raise ValueError("full-stack artifact output must use .pt")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite full-stack artifact")
    payload = build_gemma3_full_mlp_stack_payload(
        model=model,
        protocol=protocol,
        splits=splits,
        upstream_metadata=upstream_metadata,
        superfragment_plan=superfragment_plan,
        generator_fits=generator_fits,
        evaluation=evaluation,
    )
    report = build_gemma3_full_mlp_stack_report(
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
    try:
        os.link(tensor_temp, output)
        tensor_published = True
        os.link(report_temp, report_path)
    except FileExistsError as error:
        if tensor_published:
            output.unlink(missing_ok=True)
        raise FileExistsError(
            "refusing to overwrite full-stack artifact"
        ) from error
    except BaseException:
        if tensor_published:
            output.unlink(missing_ok=True)
        raise
    finally:
        tensor_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
    return report


def load_gemma3_full_mlp_stack_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load every nested full-MLP-stack compiler artifact."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("full-stack artifact must be a dict")
    _validate_and_restore_payload(raw)
    return raw
