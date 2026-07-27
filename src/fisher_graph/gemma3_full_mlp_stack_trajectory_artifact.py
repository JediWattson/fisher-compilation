"""Strict JSON artifacts for a frozen Gemma MLP-stack trajectory ladder.

This boundary records a diagnostic performed *after* the exhaustive Gemma
MLP-stack generator artifact has been frozen.  It contains only model and
artifact lineage, split hashes, the preregistered protocol, scalar metrics,
and exact logical resource accounting.  It deliberately contains no tensors,
generator parameters, source parameters, prompt text, token IDs, or raw rows.

Both trajectory directions contain depths 1 through 18.  Their depth-18 rows
must have the same canonical JSON representation: both directions execute the
same already-frozen full-stack endpoint.
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


__all__ = [
    "GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION",
    "GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA",
    "build_gemma3_full_mlp_stack_trajectory_payload",
    "load_gemma3_full_mlp_stack_trajectory_artifact",
    "save_gemma3_full_mlp_stack_trajectory_artifact",
]


GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA = (
    "fisher_graph.gemma3_frozen_full_mlp_stack_trajectory_development"
)
GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION = 1

_EXPECTED_LAYER_COUNT = 18
_EXPECTED_DEPTHS = tuple(range(1, _EXPECTED_LAYER_COUNT + 1))
_DIGEST_DOMAIN = b"fisher_graph.gemma3.full_mlp_stack.trajectory.json.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "development_only_frozen_trajectory_diagnostic",
    "development_only": True,
    "compression_claim": False,
    "heldout_confirmation": False,
    "generator_refit_performed": False,
    "generator_rank_selection_performed": False,
    "latency_or_kernel_speed_claim": False,
    "scope": "full_native_mlp_stack_trajectory_only",
}
_SAFETY: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_generator_weights": False,
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_token_rows": False,
    "contains_tokenizer_state": False,
    "contains_only_lineage_metrics_and_resource_counts": True,
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
        "generator_weights",
        "model_state_dict",
        "source_state_dict",
        "state_dict",
        "tokenizer_state",
    }
)

_PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "frozen_source_artifact",
    "splits",
    "protocol",
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
    "native_mlp_stack_learned_parameters",
    "native_mlp_stack_linear_macs_per_token",
    "local_files_only",
}
_SOURCE_FIELDS = {
    "source_schema",
    "source_format_version",
    "artifact_file_sha256",
    "scientific_payload_sha256",
    "source_scope",
    "frozen_before_trajectory",
}
_SPLIT_FIELDS = {"assessment", "provenance"}
_ASSESSMENT_FIELDS = {
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
    "heldout_confirmation",
    "assessment_used_for_generator_refit",
    "assessment_used_for_generator_rank_selection",
}
_PROTOCOL_FIELDS = {
    "scope",
    "transformer_layer_count",
    "removed_mode_count",
    "prefix_depths",
    "suffix_depths",
    "prefix_rule",
    "suffix_rule",
    "depth_18_endpoint_rule",
    "execution_path",
    "generators_frozen",
    "generator_refit_performed",
    "generator_rank_selection_performed",
    "source_model_weights_mutated",
    "assessment_role",
    "heldout_confirmation",
    "latency_or_kernel_speed_claim",
    "local_files_only",
}
_EVALUATION_FIELDS = {
    "execution_path",
    "assessment_role",
    "heldout_confirmation",
    "assessment_membership_exact",
    "frozen_before_assessment",
    "generator_refit_performed",
    "generator_rank_selection_performed",
    "latency_or_kernel_speed_claim",
    "supervised_tokens",
    "logical_valid_tokens",
    "assessment_split_sha256",
    "native",
    "prefix_ladder",
    "suffix_ladder",
}
_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_LADDER_ROW_FIELDS = {"depth", "metrics", "resources"}
_RESOURCE_FIELDS = {
    "replacement_scope",
    "replaced_layer_count",
    "replaced_layer_ordinals",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "native_replaced_mlp_learned_parameters",
    "generator_replacement_learned_parameters",
    "logical_candidate_learned_parameters",
    "net_stored_parameter_savings",
    "native_replaced_mlp_linear_macs_per_token",
    "generator_replacement_macs_per_token",
    "generator_replacement_bias_additions_per_token",
    "net_linear_macs_saved_per_token",
    "logical_candidate_excludes_replaced_native_mlps",
    "whole_transformer_replaced",
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


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
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


def _json_digest(value: object) -> str:
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


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
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(
        "artifact must contain only finite JSON scalars, arrays, and objects"
    )


def _json_clone(value: object) -> object:
    """Return the exact JSON representation used by the on-disk boundary."""

    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _validate_model(model: Mapping[str, object]) -> None:
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
    _require_sha256(
        model["adapter_model_fingerprint"],
        label="adapter_model_fingerprint",
    )
    source = _require_int(
        model["source_whole_model_learned_parameters"],
        label="source_whole_model_learned_parameters",
        minimum=1,
    )
    native = _require_int(
        model["native_mlp_stack_learned_parameters"],
        label="native_mlp_stack_learned_parameters",
        minimum=1,
    )
    native_macs = _require_int(
        model["native_mlp_stack_linear_macs_per_token"],
        label="native_mlp_stack_linear_macs_per_token",
        minimum=1,
    )
    if native > source or native_macs <= 0:
        raise ValueError("model MLP resource totals are inconsistent")
    if model["local_files_only"] is not True:
        raise ValueError("trajectory artifact must be local-only")


def _validate_source(source: Mapping[str, object]) -> None:
    _strict_fields(
        source,
        _SOURCE_FIELDS,
        label="frozen source artifact",
    )
    _require_name(source["source_schema"], label="source_schema")
    _require_int(
        source["source_format_version"],
        label="source_format_version",
        minimum=1,
    )
    _require_sha256(
        source["artifact_file_sha256"],
        label="artifact_file_sha256",
    )
    _require_sha256(
        source["scientific_payload_sha256"],
        label="source scientific_payload_sha256",
    )
    if (
        source["source_scope"] != "full_native_mlp_stack_replacement"
        or source["frozen_before_trajectory"] is not True
    ):
        raise ValueError("source artifact is not a frozen full-stack source")


def _validate_splits(splits: Mapping[str, object]) -> None:
    _strict_fields(splits, _SPLIT_FIELDS, label="split metadata")
    assessment = splits["assessment"]
    provenance = splits["provenance"]
    if not isinstance(assessment, Mapping) or not isinstance(
        provenance,
        Mapping,
    ):
        raise TypeError("split entries must be mappings")
    _strict_fields(
        assessment,
        _ASSESSMENT_FIELDS,
        label="assessment split",
    )
    if assessment["role"] != "open_development_assessment":
        raise ValueError("trajectory assessment must remain development-only")
    _require_sha256(
        assessment["serialized_sha256"],
        label="assessment serialized_sha256",
    )
    hashes = assessment["content_sha256"]
    if (
        isinstance(hashes, (str, bytes))
        or not isinstance(hashes, Sequence)
    ):
        raise TypeError("assessment content_sha256 must be a sequence")
    content_hashes = tuple(
        _require_sha256(value, label="assessment content_sha256")
        for value in hashes
    )
    count = _require_int(
        assessment["example_count"],
        label="assessment example_count",
        minimum=1,
    )
    _require_int(
        assessment["logical_valid_tokens"],
        label="assessment logical_valid_tokens",
        minimum=1,
    )
    _require_int(
        assessment["supervised_tokens"],
        label="assessment supervised_tokens",
        minimum=1,
    )
    if (
        len(content_hashes) != count
        or len(content_hashes) != len(set(content_hashes))
    ):
        raise ValueError(
            "assessment content hashes must be exact, unique membership"
        )
    _strict_fields(
        provenance,
        _PROVENANCE_FIELDS,
        label="split provenance",
    )
    if provenance != {
        "assurance": "caller_declared_self_attested",
        "externally_authenticated": False,
        "heldout_confirmation": False,
        "assessment_used_for_generator_refit": False,
        "assessment_used_for_generator_rank_selection": False,
    }:
        raise ValueError("split provenance overclaims or permits adaptation")


def _depths(value: object, *, label: str) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value)
    if result != _EXPECTED_DEPTHS:
        raise ValueError(f"{label} must be exactly depths 1 through 18")
    return result


def _validate_protocol(protocol: Mapping[str, object]) -> None:
    _strict_fields(protocol, _PROTOCOL_FIELDS, label="protocol")
    if (
        protocol["scope"]
        != "frozen_full_native_mlp_stack_trajectory_ladder"
        or protocol["transformer_layer_count"] != _EXPECTED_LAYER_COUNT
        or _require_int(
            protocol["removed_mode_count"],
            label="removed_mode_count",
            minimum=1,
        )
        <= 0
        or _depths(protocol["prefix_depths"], label="prefix_depths")
        != _EXPECTED_DEPTHS
        or _depths(protocol["suffix_depths"], label="suffix_depths")
        != _EXPECTED_DEPTHS
        or protocol["prefix_rule"]
        != "generated_layers_0_through_depth_minus_1"
        or protocol["suffix_rule"]
        != "generated_layers_18_minus_depth_through_17"
        or protocol["depth_18_endpoint_rule"]
        != "canonical_exact_prefix_suffix_equality"
        or protocol["execution_path"]
        != "frozen_mixed_native_generated_mlp_stack"
        or protocol["generators_frozen"] is not True
        or protocol["generator_refit_performed"] is not False
        or protocol["generator_rank_selection_performed"] is not False
        or protocol["source_model_weights_mutated"] is not False
        or protocol["assessment_role"]
        != "open_development_assessment"
        or protocol["heldout_confirmation"] is not False
        or protocol["latency_or_kernel_speed_claim"] is not False
        or protocol["local_files_only"] is not True
    ):
        raise ValueError("protocol is not the frozen development ladder")


def _validate_metrics(
    metrics: Mapping[str, object],
    *,
    label: str,
    native_nll: float | None,
) -> float:
    _strict_fields(metrics, _METRIC_FIELDS, label=label)
    nll = _require_finite(
        metrics["nll_per_token"],
        label=f"{label} nll_per_token",
        minimum=0.0,
    )
    delta = _require_finite(
        metrics["delta_nll_per_token"],
        label=f"{label} delta_nll_per_token",
    )
    divergence = _require_finite(
        metrics["native_to_candidate_kl_per_token"],
        label=f"{label} native_to_candidate_kl_per_token",
        minimum=0.0,
    )
    agreement = _require_finite(
        metrics["top1_agreement_to_native"],
        label=f"{label} top1_agreement_to_native",
        minimum=0.0,
        maximum=1.0,
    )
    if native_nll is None:
        if delta != 0.0 or divergence != 0.0 or agreement != 1.0:
            raise ValueError("native metrics must be exact baselines")
    elif not math.isclose(
        delta,
        nll - native_nll,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} delta NLL differs from native")
    return nll


def _validate_resources(
    resources: Mapping[str, object],
    *,
    direction: str,
    depth: int,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[int, ...]:
    _strict_fields(
        resources,
        _RESOURCE_FIELDS,
        label=f"{direction} depth {depth} resources",
    )
    expected_ordinals = (
        tuple(range(depth))
        if direction == "prefix"
        else tuple(range(_EXPECTED_LAYER_COUNT - depth, _EXPECTED_LAYER_COUNT))
    )
    ordinals = resources["replaced_layer_ordinals"]
    if (
        isinstance(ordinals, (str, bytes))
        or not isinstance(ordinals, Sequence)
        or tuple(ordinals) != expected_ordinals
    ):
        raise ValueError(
            f"{direction} depth {depth} layer scope is not exact"
        )
    source = _require_int(
        resources["source_whole_model_learned_parameters"],
        label="resource source parameters",
        minimum=1,
    )
    native = _require_int(
        resources["native_replaced_mlp_learned_parameters"],
        label="resource native replaced parameters",
        minimum=1,
    )
    generated = _require_int(
        resources["generator_replacement_learned_parameters"],
        label="resource generator parameters",
        minimum=1,
    )
    candidate = _require_int(
        resources["logical_candidate_learned_parameters"],
        label="resource candidate parameters",
        minimum=1,
    )
    savings = _require_int(
        resources["net_stored_parameter_savings"],
        label="resource parameter savings",
        minimum=1,
    )
    native_macs = _require_int(
        resources["native_replaced_mlp_linear_macs_per_token"],
        label="resource native MLP MACs",
        minimum=1,
    )
    generator_macs = _require_int(
        resources["generator_replacement_macs_per_token"],
        label="resource generator MACs",
        minimum=1,
    )
    additions = _require_int(
        resources["generator_replacement_bias_additions_per_token"],
        label="resource generator bias additions",
        minimum=0,
    )
    mac_savings = _require_int(
        resources["net_linear_macs_saved_per_token"],
        label="resource MAC savings",
        minimum=1,
    )
    removed_modes = _require_int(
        resources["removed_mode_count"],
        label="resource removed modes",
        minimum=1,
    )
    expected_scope = (
        "full_native_mlp_stack_replacement"
        if depth == _EXPECTED_LAYER_COUNT
        else "partial_native_mlp_stack_replacement"
    )
    if (
        resources["replacement_scope"] != expected_scope
        or resources["replaced_layer_count"] != depth
        or source
        != model["source_whole_model_learned_parameters"]
        or candidate != source - native + generated
        or savings != native - generated
        or mac_savings != native_macs - generator_macs
        or resources[
            "logical_candidate_excludes_replaced_native_mlps"
        ]
        is not True
        or resources["whole_transformer_replaced"] is not False
    ):
        raise ValueError(
            f"{direction} depth {depth} resource arithmetic is invalid"
        )
    if depth == _EXPECTED_LAYER_COUNT and (
        native != model["native_mlp_stack_learned_parameters"]
        or native_macs
        != model["native_mlp_stack_linear_macs_per_token"]
        or removed_modes != protocol["removed_mode_count"]
    ):
        raise ValueError("depth-18 resources do not cover the full MLP stack")
    return (
        removed_modes,
        native,
        generated,
        candidate,
        savings,
        native_macs,
        generator_macs,
        additions,
        mac_savings,
    )


def _strictly_increasing(values: Sequence[int], *, label: str) -> None:
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must increase at every depth")


def _validate_ladder(
    ladder: object,
    *,
    direction: str,
    native_nll: float,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    if (
        isinstance(ladder, (str, bytes))
        or not isinstance(ladder, Sequence)
    ):
        raise TypeError(f"{direction}_ladder must be a sequence")
    rows = tuple(ladder)
    if len(rows) != _EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"{direction}_ladder must contain exactly 18 depths"
        )
    resource_columns: list[tuple[int, ...]] = []
    validated: list[Mapping[str, object]] = []
    for expected_depth, row in zip(_EXPECTED_DEPTHS, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError(f"{direction} ladder rows must be mappings")
        _strict_fields(
            row,
            _LADDER_ROW_FIELDS,
            label=f"{direction} depth {expected_depth}",
        )
        if row["depth"] != expected_depth:
            raise ValueError(
                f"{direction}_ladder must be ordered depths 1 through 18"
            )
        metrics = row["metrics"]
        resources = row["resources"]
        if not isinstance(metrics, Mapping) or not isinstance(
            resources,
            Mapping,
        ):
            raise TypeError("ladder metrics and resources must be mappings")
        _validate_metrics(
            metrics,
            label=f"{direction} depth {expected_depth} metrics",
            native_nll=native_nll,
        )
        resource_columns.append(
            _validate_resources(
                resources,
                direction=direction,
                depth=expected_depth,
                model=model,
                protocol=protocol,
            )
        )
        validated.append(row)

    columns = tuple(zip(*resource_columns, strict=True))
    for index, label in (
        (0, "removed mode count"),
        (1, "native replaced parameters"),
        (2, "generator parameters"),
        (4, "stored parameter savings"),
        (5, "native replaced MACs"),
        (6, "generator MACs"),
        (8, "linear MAC savings"),
    ):
        _strictly_increasing(
            columns[index],
            label=f"{direction} {label}",
        )
    candidate_parameters = columns[3]
    if any(
        left <= right
        for left, right in zip(
            candidate_parameters,
            candidate_parameters[1:],
        )
    ):
        raise ValueError(
            f"{direction} logical candidate parameters must decrease"
        )
    bias_additions = columns[7]
    if any(
        left > right
        for left, right in zip(bias_additions, bias_additions[1:])
    ):
        raise ValueError(
            f"{direction} generator bias additions must be monotone"
        )
    return tuple(validated)


def _validate_evaluation(
    evaluation: Mapping[str, object],
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
) -> None:
    _strict_fields(evaluation, _EVALUATION_FIELDS, label="evaluation")
    if (
        evaluation["execution_path"]
        != "frozen_prefix_suffix_full_mlp_stack_ladder"
        or evaluation["assessment_role"]
        != "open_development_assessment"
        or evaluation["heldout_confirmation"] is not False
        or evaluation["assessment_membership_exact"] is not True
        or evaluation["frozen_before_assessment"] is not True
        or evaluation["generator_refit_performed"] is not False
        or evaluation["generator_rank_selection_performed"] is not False
        or evaluation["latency_or_kernel_speed_claim"] is not False
        or evaluation["assessment_split_sha256"]
        != splits["assessment"]["serialized_sha256"]  # type: ignore[index]
    ):
        raise ValueError("evaluation overclaims or permits adaptation")
    supervised_tokens = _require_int(
        evaluation["supervised_tokens"],
        label="supervised_tokens",
        minimum=1,
    )
    logical_valid_tokens = _require_int(
        evaluation["logical_valid_tokens"],
        label="logical_valid_tokens",
        minimum=1,
    )
    assessment = splits["assessment"]
    if (
        supervised_tokens != assessment["supervised_tokens"]  # type: ignore[index]
        or logical_valid_tokens
        != assessment["logical_valid_tokens"]  # type: ignore[index]
    ):
        raise ValueError(
            "evaluation token totals differ from the frozen assessment"
        )
    native = evaluation["native"]
    if not isinstance(native, Mapping):
        raise TypeError("native metrics must be a mapping")
    native_nll = _validate_metrics(
        native,
        label="native metrics",
        native_nll=None,
    )
    prefix = _validate_ladder(
        evaluation["prefix_ladder"],
        direction="prefix",
        native_nll=native_nll,
        model=model,
        protocol=protocol,
    )
    suffix = _validate_ladder(
        evaluation["suffix_ladder"],
        direction="suffix",
        native_nll=native_nll,
        model=model,
        protocol=protocol,
    )
    if _canonical_json_bytes(prefix[-1]) != _canonical_json_bytes(suffix[-1]):
        raise ValueError(
            "prefix and suffix depth-18 endpoints must be exactly equal"
        )
    additive_resource_fields = (
        "removed_mode_count",
        "native_replaced_mlp_learned_parameters",
        "generator_replacement_learned_parameters",
        "net_stored_parameter_savings",
        "native_replaced_mlp_linear_macs_per_token",
        "generator_replacement_macs_per_token",
        "generator_replacement_bias_additions_per_token",
        "net_linear_macs_saved_per_token",
    )
    for field in additive_resource_fields:
        prefix_values = tuple(
            row["resources"][field]  # type: ignore[index]
            for row in prefix
        )
        suffix_values = tuple(
            row["resources"][field]  # type: ignore[index]
            for row in suffix
        )
        prefix_increments = tuple(
            value - (0 if index == 0 else prefix_values[index - 1])
            for index, value in enumerate(prefix_values)
        )
        suffix_increments = tuple(
            value - (0 if index == 0 else suffix_values[index - 1])
            for index, value in enumerate(suffix_values)
        )
        if prefix_increments != tuple(reversed(suffix_increments)):
            raise ValueError(
                "prefix and suffix per-layer resource increments differ"
            )


def _validate_payload(raw: Mapping[str, object]) -> None:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="trajectory artifact")
    if (
        raw["schema"] != GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA
        or raw["format_version"]
        != GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("trajectory artifact header or safety is invalid")
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
    if _json_digest(without_digest) != digest:
        raise ValueError("trajectory scientific payload hash mismatch")
    model = raw["model"]
    source = raw["frozen_source_artifact"]
    splits = raw["splits"]
    protocol = raw["protocol"]
    evaluation = raw["evaluation"]
    for value, label in (
        (model, "model"),
        (source, "frozen_source_artifact"),
        (splits, "splits"),
        (protocol, "protocol"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_model(model)
    _validate_source(source)
    _validate_splits(splits)
    _validate_protocol(protocol)
    _validate_evaluation(
        evaluation,
        model=model,
        protocol=protocol,
        splits=splits,
    )


def build_gemma3_full_mlp_stack_trajectory_payload(
    *,
    model: Mapping[str, object],
    frozen_source_artifact: Mapping[str, object],
    splits: Mapping[str, object],
    protocol: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate one tensor-free frozen-trajectory payload."""

    for value, label in (
        (model, "model"),
        (frozen_source_artifact, "frozen_source_artifact"),
        (splits, "splits"),
        (protocol, "protocol"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    without_digest = {
        "schema": GEMMA3_FULL_MLP_STACK_TRAJECTORY_SCHEMA,
        "format_version": (
            GEMMA3_FULL_MLP_STACK_TRAJECTORY_FORMAT_VERSION
        ),
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "frozen_source_artifact": dict(frozen_source_artifact),
        "splits": dict(splits),
        "protocol": dict(protocol),
        "evaluation": dict(evaluation),
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    canonical = _json_clone(without_digest)
    if not isinstance(canonical, dict):
        raise AssertionError("canonical trajectory payload must be a dict")
    payload = {
        **canonical,
        "scientific_payload_sha256": _json_digest(canonical),
    }
    _validate_payload(payload)
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid non-finite JSON constant {value}")


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def save_gemma3_full_mlp_stack_trajectory_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    frozen_source_artifact: Mapping[str, object],
    splits: Mapping[str, object],
    protocol: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Exclusively and atomically save one strict JSON-only artifact."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("trajectory artifact output must use .json")
    if output.exists():
        raise FileExistsError("refusing to overwrite trajectory artifact")
    payload = build_gemma3_full_mlp_stack_trajectory_payload(
        model=model,
        frozen_source_artifact=frozen_source_artifact,
        splits=splits,
        protocol=protocol,
        evaluation=evaluation,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite trajectory artifact"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_gemma3_full_mlp_stack_trajectory_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load and authenticate one JSON-only trajectory artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(
            handle,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(raw, dict):
        raise TypeError("trajectory artifact must be a JSON object")
    _validate_payload(raw)
    return raw
