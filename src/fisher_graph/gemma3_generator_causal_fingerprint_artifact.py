"""Strict JSON boundary for Gemma generator causal fingerprints.

This artifact records an observational, adaptive open-development analysis of
the already-frozen 18-layer deployed generator stack.  A fixed temporary
zero-residual intervention is applied to one generator at a time and restored
before the next observation.  Only lineage hashes and aggregate causal
fingerprint statistics cross this boundary.

The artifact never contains prompt text, token IDs, logits, tensors, model or
generator weights, or raw per-prompt rows.  It does not authorize mutation,
refitting, pruning, merging, deployment, or any compression or latency claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata


__all__ = [
    "GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION",
    "GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA",
    "build_gemma3_generator_causal_fingerprint_payload",
    "gemma3_generator_prompt_fingerprint_sha256",
    "load_gemma3_generator_causal_fingerprint_artifact",
    "save_gemma3_generator_causal_fingerprint_artifact",
]


GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA = (
    "fisher_graph.gemma3_generator_causal_fingerprint_development"
)
GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION = 1

_LAYER_COUNT = 18
_REFIT_START_LAYER = 10
_PAIR_COUNT = _LAYER_COUNT * (_LAYER_COUNT - 1) // 2
_DIGEST_DOMAIN = (
    b"fisher_graph.gemma3.generator_causal_fingerprint.json.v1\0"
)
_FINGERPRINT_DOMAIN = (
    b"fisher_graph.gemma3.generator_prompt_fingerprint.json.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_FULL_STACK_SCHEMA = (
    "fisher_graph.gemma3_full_native_mlp_stack_development"
)
_REFIT_SCHEMA = (
    "fisher_graph.gemma3_sequential_full_mlp_stack_refit_development"
)
_HYPOTHESIS_LABELS = frozenset(
    {
        "aligned_observational_family_hypothesis",
        "mixed_observational_family_evidence",
        "distinct_observational_effect_hypothesis",
        "insufficient_causal_variation",
    }
)

_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "observational_adaptive_open_development_causal_fingerprint",
    "development_only": True,
    "adaptive_analysis": True,
    "observational_metrics_only": True,
    "heldout_confirmation": False,
    "compression_claim": False,
    "latency_or_kernel_speed_claim": False,
    "authorizes_model_mutation": False,
    "authorizes_generator_refit": False,
    "authorizes_intervention": False,
    "authorizes_pruning": False,
    "authorizes_merge": False,
    "authorizes_routing": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
    "authorizes_compression_deployment": False,
    "scope": "frozen_deployed_full_mlp_stack_generators",
}

_OBSERVATION_PROTOCOL: dict[str, object] = {
    "protocol_id": "temporary_muted_residual_prompt_fingerprint_v1",
    "transformer_layer_count": _LAYER_COUNT,
    "generator_order": "ascending_layer_ordinal_0_through_17",
    "prompt_order": "analysis_split_content_sha256_order",
    "baseline_execution_path": "frozen_sequential_refit_generator_stack",
    "intervention_unit": "one_deployed_generator",
    "intervention_operation": (
        "exactly_one_generator_muted_against_shared_baseline"
    ),
    "intervention_schedule": "one_generator_at_a_time",
    "restoration_rule": "restore_before_next_prompt_generator_observation",
    "prompt_signature_fields": [
        "muted_minus_baseline_nll_per_token",
        "baseline_to_muted_kl_per_token",
        "top1_agreement_to_baseline",
        "centered_anchor_logit_effect_rms",
    ],
    "shared_frame": (
        "per_supervised_token_target_then_stable_baseline_top_non_target_logits"
    ),
    "effect_centering": "per_supervised_token_anchor_mean",
    "shared_effect_gram_weighting": (
        "equal_prompt_mean_over_supervised_anchor_coordinates"
    ),
    "anchor_count": 8,
    "anchor_frame_width": 9,
    "pair_order": "lexicographic_unordered_layer_pairs",
    "top_importance_count": 5,
    "top_importance_rule": (
        "largest_absolute_prompt_nll_effect_stable_index_tiebreak"
    ),
    "top_importance_overlap_rule": (
        "intersection_count_divided_by_top_importance_count"
    ),
    "cosine_zero_norm_rule": (
        "zero_when_either_generator_gram_diagonal_is_zero"
    ),
    "constant_correlation_rule": (
        "zero_when_either_rank_vector_has_zero_centered_norm"
    ),
    "observational_family_policy": {
        "minimum_centered_effect_cosine": 0.9,
        "minimum_prompt_nll_spearman": 0.8,
        "minimum_top_importance_overlap": 0.6,
        "minimum_top_importance_sign_agreement": 0.8,
        "minimum_prompt_count": 3,
    },
    "deterministic_seed": 0,
    "stochastic_sampling": False,
    "source_model_weights_mutated": False,
    "generator_weights_mutated": False,
    "generator_refit_performed": False,
    "pruning_performed": False,
    "merge_performed": False,
}

_SAFETY: dict[str, bool] = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_generator_weights": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_logit_rows": False,
    "contains_prompt_conditioned_scalar_signatures": True,
    "contains_only_lineage_protocol_hashes_and_safe_metrics": True,
    "analysis_only": True,
    "authorizes_execution": False,
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
        "logit",
        "logits",
        "raw_logits",
        "score_gradients",
        "activations",
        "activation_rows",
        "gradient_rows",
        "raw_rows",
        "raw_prompt_rows",
        "raw_token_rows",
        "raw_logit_rows",
        "raw_fingerprint_rows",
        "fingerprint_rows",
        "source_weights",
        "source_model_weights",
        "source_parameter_values",
        "generator_weights",
        "weights",
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
    "frozen_sources",
    "analysis_split",
    "causal_analysis_lineage",
    "deployed_generator_plan_sha256s",
    "generator_fit_lineage",
    "observation_protocol",
    "generator_causal_summaries",
    "pairwise_similarities",
    "safety",
    "scientific_payload_sha256",
}
_MODEL_FIELDS = {
    "model_id",
    "requested_revision",
    "resolved_commit",
    "adapter_model_fingerprint",
    "local_files_only",
}
_FROZEN_SOURCE_FIELDS = {"base_full_stack", "sequential_refit"}
_SOURCE_FIELDS = {
    "schema",
    "format_version",
    "artifact_file_sha256",
    "scientific_payload_sha256",
    "frozen_before_analysis",
}
_SPLIT_FIELDS = {
    "role",
    "serialized_sha256",
    "content_sha256",
    "example_count",
    "logical_valid_tokens",
    "supervised_tokens",
    "membership_exact",
    "assurance",
    "externally_authenticated",
    "heldout_confirmation",
    "used_for_adaptive_analysis",
    "used_for_generator_fit",
    "used_for_generator_selection",
}
_CAUSAL_ANALYSIS_FIELDS = {
    "artifact_kind",
    "format_version",
    "artifact_sha256",
    "source_model_sha256",
    "generator_catalog_sha256",
    "evaluation_split_sha256",
    "objective_sha256",
    "intervention",
    "generator_count",
    "prompt_count",
    "anchor_count",
    "anchor_frame_width",
    "shared_frame",
    "effect_centering",
    "gram_weighting",
    "top_importance_count",
    "observational_family_policy",
    "tensor_sha256s",
}
_CORE_TENSOR_SHA256_FIELDS = {
    "supervised_token_counts",
    "prompt_nll_effects",
    "prompt_baseline_to_muted_kls",
    "prompt_top1_agreements",
    "prompt_centered_anchor_effect_rms",
    "centered_shared_effect_gram",
}
_LINEAGE_FIELDS = {
    "layer_ordinal",
    "layer_id",
    "deployment_source",
    "source_artifact_scientific_payload_sha256",
    "base_fit_sha256",
    "deployed_fit_sha256",
    "deployed_generator_plan_sha256",
}
_SUMMARY_FIELDS = {
    "layer_ordinal",
    "deployed_generator_plan_sha256",
    "deployed_fit_sha256",
    "analysis_split_sha256",
    "prompt_observation_count",
    "fingerprint_sha256",
    "prompt_signatures",
    "mean_muted_minus_baseline_nll_per_token",
    "rms_muted_minus_baseline_nll_per_token",
    "mean_absolute_muted_minus_baseline_nll_per_token",
    "maximum_absolute_muted_minus_baseline_nll_per_token",
    "mean_baseline_to_muted_kl_per_token",
    "mean_top1_agreement_to_baseline",
    "mean_centered_anchor_logit_effect_rms",
    "positive_delta_fraction",
}
_PROMPT_SIGNATURE_FIELDS = {
    "prompt_content_sha256",
    "muted_minus_baseline_nll_per_token",
    "baseline_to_muted_kl_per_token",
    "top1_agreement_to_baseline",
    "centered_anchor_logit_effect_rms",
}
_PAIR_FIELDS = {
    "left_layer_ordinal",
    "right_layer_ordinal",
    "left_generator_plan_sha256",
    "right_generator_plan_sha256",
    "left_fingerprint_sha256",
    "right_fingerprint_sha256",
    "analysis_split_sha256",
    "shared_prompt_count",
    "centered_shared_logit_effect_cosine",
    "prompt_nll_effect_spearman",
    "top_importance_overlap",
    "top_importance_sign_agreement",
    "top_importance_intersection_count",
    "sufficient_causal_variation",
    "observational_hypothesis",
    "observational_only",
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_mutation",
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
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _hash_sequence(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(
        _require_sha256(item, label=label)
        for item in _sequence(value, label=label)
    )


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
    if model["local_files_only"] is not True:
        raise ValueError("causal fingerprint analysis must be local-only")


def _validate_frozen_sources(sources: Mapping[str, object]) -> None:
    _strict_fields(sources, _FROZEN_SOURCE_FIELDS, label="frozen sources")
    expected = (
        ("base_full_stack", _FULL_STACK_SCHEMA),
        ("sequential_refit", _REFIT_SCHEMA),
    )
    file_hashes: list[str] = []
    scientific_hashes: list[str] = []
    for name, schema in expected:
        source = sources[name]
        if not isinstance(source, Mapping):
            raise TypeError(f"{name} source must be a mapping")
        _strict_fields(source, _SOURCE_FIELDS, label=f"{name} source")
        if (
            source["schema"] != schema
            or source["format_version"] != 1
            or source["frozen_before_analysis"] is not True
        ):
            raise ValueError(f"{name} source header or freeze is invalid")
        file_hashes.append(
            _require_sha256(
                source["artifact_file_sha256"],
                label=f"{name} artifact_file_sha256",
            )
        )
        scientific_hashes.append(
            _require_sha256(
                source["scientific_payload_sha256"],
                label=f"{name} scientific_payload_sha256",
            )
        )
    if len(set(file_hashes)) != 2 or len(set(scientific_hashes)) != 2:
        raise ValueError("base and refit sources must bind distinct artifacts")


def _validate_analysis_split(split: Mapping[str, object]) -> tuple[str, ...]:
    _strict_fields(split, _SPLIT_FIELDS, label="analysis split")
    role = _require_name(split["role"], label="analysis split role")
    if (
        "heldout" in role.lower()
        or role.lower() in {"test", "reserved_test", "closed_test"}
        or split["membership_exact"] is not True
        or split["assurance"] != "caller_declared_self_attested"
        or split["externally_authenticated"] is not False
        or split["heldout_confirmation"] is not False
        or split["used_for_adaptive_analysis"] is not True
        or split["used_for_generator_fit"] is not False
        or split["used_for_generator_selection"] is not False
    ):
        raise ValueError("analysis split role or provenance overclaims")
    _require_sha256(
        split["serialized_sha256"],
        label="analysis split serialized_sha256",
    )
    members = _hash_sequence(
        split["content_sha256"],
        label="analysis split content_sha256",
    )
    count = _require_int(
        split["example_count"],
        label="analysis split example_count",
        minimum=1,
    )
    top_count = _OBSERVATION_PROTOCOL["top_importance_count"]
    if type(top_count) is not int:
        raise AssertionError("fixed top importance count must be an integer")
    if len(members) != count or len(members) != len(set(members)):
        raise ValueError(
            "analysis split membership count must be exact and unique"
        )
    if count < top_count:
        raise ValueError(
            "analysis split must cover the fixed top-importance count"
        )
    _require_int(
        split["logical_valid_tokens"],
        label="analysis split logical_valid_tokens",
        minimum=1,
    )
    _require_int(
        split["supervised_tokens"],
        label="analysis split supervised_tokens",
        minimum=1,
    )
    return members


def _validate_causal_analysis_lineage(
    lineage: Mapping[str, object],
    *,
    model: Mapping[str, object],
    split: Mapping[str, object],
) -> None:
    _strict_fields(
        lineage,
        _CAUSAL_ANALYSIS_FIELDS,
        label="causal analysis lineage",
    )
    for field in (
        "artifact_sha256",
        "source_model_sha256",
        "generator_catalog_sha256",
        "evaluation_split_sha256",
        "objective_sha256",
    ):
        _require_sha256(
            lineage[field],
            label=f"causal analysis {field}",
        )
    tensor_hashes = lineage["tensor_sha256s"]
    if not isinstance(tensor_hashes, Mapping):
        raise TypeError("causal analysis tensor_sha256s must be a mapping")
    _strict_fields(
        tensor_hashes,
        _CORE_TENSOR_SHA256_FIELDS,
        label="causal analysis tensor hash catalog",
    )
    for field in sorted(_CORE_TENSOR_SHA256_FIELDS):
        _require_sha256(
            tensor_hashes[field],
            label=f"causal analysis tensor_sha256s.{field}",
        )
    policy = lineage["observational_family_policy"]
    if not isinstance(policy, Mapping):
        raise TypeError(
            "causal analysis observational_family_policy must be a mapping"
        )
    expected_policy = _OBSERVATION_PROTOCOL["observational_family_policy"]
    if not isinstance(expected_policy, Mapping):
        raise AssertionError("fixed observational policy must be a mapping")
    if (
        lineage["artifact_kind"]
        != "fisher_graph.modal_generator_causal_fingerprints"
        or lineage["format_version"] != 1
        or lineage["source_model_sha256"]
        != model["adapter_model_fingerprint"]
        or lineage["evaluation_split_sha256"]
        != split["serialized_sha256"]
        or lineage["intervention"]
        != _OBSERVATION_PROTOCOL["intervention_operation"]
        or lineage["generator_count"] != _LAYER_COUNT
        or lineage["prompt_count"] != split["example_count"]
        or lineage["anchor_count"]
        != _OBSERVATION_PROTOCOL["anchor_count"]
        or lineage["anchor_frame_width"]
        != _OBSERVATION_PROTOCOL["anchor_frame_width"]
        or lineage["shared_frame"] != _OBSERVATION_PROTOCOL["shared_frame"]
        or lineage["effect_centering"]
        != _OBSERVATION_PROTOCOL["effect_centering"]
        or lineage["gram_weighting"]
        != _OBSERVATION_PROTOCOL["shared_effect_gram_weighting"]
        or lineage["top_importance_count"]
        != _OBSERVATION_PROTOCOL["top_importance_count"]
        or policy != expected_policy
    ):
        raise ValueError(
            "causal analysis lineage differs from the fixed core protocol"
        )


def _validate_lineage(
    value: object,
    *,
    sources: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label="generator_fit_lineage")
    if len(rows) != _LAYER_COUNT:
        raise ValueError("generator_fit_lineage must contain exactly 18 rows")
    validated: list[Mapping[str, object]] = []
    deployed_fit_hashes: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("generator fit lineage rows must be mappings")
        _strict_fields(
            row,
            _LINEAGE_FIELDS,
            label=f"generator fit lineage layer {ordinal}",
        )
        expected_source = (
            "frozen_full_stack"
            if ordinal < _REFIT_START_LAYER
            else "sequential_refit_overlay"
        )
        source_key = (
            "base_full_stack"
            if ordinal < _REFIT_START_LAYER
            else "sequential_refit"
        )
        base_fit = _require_sha256(
            row["base_fit_sha256"],
            label=f"layer {ordinal} base_fit_sha256",
        )
        deployed_fit = _require_sha256(
            row["deployed_fit_sha256"],
            label=f"layer {ordinal} deployed_fit_sha256",
        )
        _require_sha256(
            row["deployed_generator_plan_sha256"],
            label=f"layer {ordinal} deployed_generator_plan_sha256",
        )
        if (
            row["layer_ordinal"] != ordinal
            or row["layer_id"] != f"layer.{ordinal}"
            or row["deployment_source"] != expected_source
            or row["source_artifact_scientific_payload_sha256"]
            != sources[source_key]["scientific_payload_sha256"]  # type: ignore[index]
            or (
                ordinal < _REFIT_START_LAYER
                and base_fit != deployed_fit
            )
            or (
                ordinal >= _REFIT_START_LAYER
                and base_fit == deployed_fit
            )
        ):
            raise ValueError(
                f"generator fit lineage layer {ordinal} is inconsistent"
            )
        if deployed_fit in deployed_fit_hashes:
            raise ValueError("deployed fit hashes must be layer-unique")
        deployed_fit_hashes.add(deployed_fit)
        validated.append(row)
    return tuple(validated)


def _validate_prompt_signature(
    row: Mapping[str, object],
    *,
    label: str,
) -> tuple[str, float, float, float, float]:
    _strict_fields(row, _PROMPT_SIGNATURE_FIELDS, label=label)
    prompt_sha256 = _require_sha256(
        row["prompt_content_sha256"],
        label=f"{label} prompt_content_sha256",
    )
    delta_nll = _require_finite(
        row["muted_minus_baseline_nll_per_token"],
        label=f"{label} muted_minus_baseline_nll_per_token",
    )
    kl = _require_finite(
        row["baseline_to_muted_kl_per_token"],
        label=f"{label} baseline_to_muted_kl_per_token",
        minimum=0.0,
    )
    agreement = _require_finite(
        row["top1_agreement_to_baseline"],
        label=f"{label} top1_agreement_to_baseline",
        minimum=0.0,
        maximum=1.0,
    )
    effect_rms = _require_finite(
        row["centered_anchor_logit_effect_rms"],
        label=f"{label} centered_anchor_logit_effect_rms",
        minimum=0.0,
    )
    return prompt_sha256, delta_nll, kl, agreement, effect_rms


def gemma3_generator_prompt_fingerprint_sha256(
    *,
    layer_ordinal: int,
    analysis_split_sha256: str,
    prompt_signatures: Sequence[Mapping[str, object]],
) -> str:
    """Hash one exact ordered, prompt-conditioned scalar signature."""

    ordinal = _require_int(
        layer_ordinal,
        label="layer_ordinal",
        minimum=0,
    )
    if ordinal >= _LAYER_COUNT:
        raise ValueError("layer_ordinal exceeds the 18-layer stack")
    split_sha256 = _require_sha256(
        analysis_split_sha256,
        label="analysis_split_sha256",
    )
    rows = _sequence(prompt_signatures, label="prompt_signatures")
    if not rows:
        raise ValueError("prompt_signatures must be nonempty")
    canonical_rows: list[dict[str, object]] = []
    prompt_hashes: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("prompt signature rows must be mappings")
        prompt_hash, *_ = _validate_prompt_signature(
            row,
            label=f"prompt signature {index}",
        )
        prompt_hashes.append(prompt_hash)
        canonical_rows.append(dict(row))
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("prompt signature hashes must be unique")
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "layer_ordinal": ordinal,
                "analysis_split_sha256": split_sha256,
                "prompt_signatures": canonical_rows,
            }
        )
    )
    return digest.hexdigest()


def _validate_summary(
    row: Mapping[str, object],
    *,
    ordinal: int,
    lineage: Mapping[str, object],
    split: Mapping[str, object],
) -> None:
    _strict_fields(
        row,
        _SUMMARY_FIELDS,
        label=f"causal summary layer {ordinal}",
    )
    count = _require_int(
        row["prompt_observation_count"],
        label=f"layer {ordinal} prompt_observation_count",
        minimum=1,
    )
    _require_sha256(
        row["fingerprint_sha256"],
        label=f"layer {ordinal} fingerprint_sha256",
    )
    signature_rows = _sequence(
        row["prompt_signatures"],
        label=f"layer {ordinal} prompt_signatures",
    )
    if len(signature_rows) != count:
        raise ValueError(
            f"layer {ordinal} prompt signature count is inconsistent"
        )
    metrics: list[tuple[str, float, float, float, float]] = []
    for index, signature in enumerate(signature_rows):
        if not isinstance(signature, Mapping):
            raise TypeError("prompt signature rows must be mappings")
        metrics.append(
            _validate_prompt_signature(
                signature,
                label=f"layer {ordinal} prompt signature {index}",
            )
        )
    prompt_hashes = tuple(metric[0] for metric in metrics)
    split_members = _hash_sequence(
        split["content_sha256"],
        label="analysis split content_sha256",
    )
    if prompt_hashes != split_members:
        raise ValueError(
            f"layer {ordinal} prompt signature membership or order differs"
        )
    _require_finite(
        row["mean_muted_minus_baseline_nll_per_token"],
        label=f"layer {ordinal} mean muted-minus-baseline NLL",
    )
    _require_finite(
        row["rms_muted_minus_baseline_nll_per_token"],
        label=f"layer {ordinal} rms_delta_nll_per_token",
        minimum=0.0,
    )
    _require_finite(
        row["mean_absolute_muted_minus_baseline_nll_per_token"],
        label=f"layer {ordinal} mean_absolute_delta_nll_per_token",
        minimum=0.0,
    )
    _require_finite(
        row["maximum_absolute_muted_minus_baseline_nll_per_token"],
        label=f"layer {ordinal} maximum_absolute_delta_nll_per_token",
        minimum=0.0,
    )
    _require_finite(
        row["mean_baseline_to_muted_kl_per_token"],
        label=f"layer {ordinal} mean KL",
        minimum=0.0,
    )
    _require_finite(
        row["mean_top1_agreement_to_baseline"],
        label=f"layer {ordinal} mean top1 agreement",
        minimum=0.0,
        maximum=1.0,
    )
    _require_finite(
        row["mean_centered_anchor_logit_effect_rms"],
        label=f"layer {ordinal} mean centered logit effect RMS",
        minimum=0.0,
    )
    _require_finite(
        row["positive_delta_fraction"],
        label=f"layer {ordinal} positive delta fraction",
        minimum=0.0,
        maximum=1.0,
    )
    deltas = tuple(metric[1] for metric in metrics)
    expected_values = {
        "mean_muted_minus_baseline_nll_per_token": (
            math.fsum(deltas) / count
        ),
        "rms_muted_minus_baseline_nll_per_token": math.sqrt(
            math.fsum(value * value for value in deltas) / count
        ),
        "mean_absolute_muted_minus_baseline_nll_per_token": (
            math.fsum(abs(value) for value in deltas) / count
        ),
        "maximum_absolute_muted_minus_baseline_nll_per_token": max(
            abs(value) for value in deltas
        ),
        "mean_baseline_to_muted_kl_per_token": (
            math.fsum(metric[2] for metric in metrics) / count
        ),
        "mean_top1_agreement_to_baseline": (
            math.fsum(metric[3] for metric in metrics) / count
        ),
        "mean_centered_anchor_logit_effect_rms": (
            math.fsum(metric[4] for metric in metrics) / count
        ),
        "positive_delta_fraction": (
            sum(value > 0.0 for value in deltas) / count
        ),
    }
    aggregates_match = all(
        math.isclose(
            float(row[field]),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field, expected in expected_values.items()
    )
    expected_fingerprint_sha256 = (
        gemma3_generator_prompt_fingerprint_sha256(
            layer_ordinal=ordinal,
            analysis_split_sha256=split["serialized_sha256"],  # type: ignore[arg-type]
            prompt_signatures=signature_rows,  # type: ignore[arg-type]
        )
    )
    if (
        row["layer_ordinal"] != ordinal
        or row["deployed_generator_plan_sha256"]
        != lineage["deployed_generator_plan_sha256"]
        or row["deployed_fit_sha256"] != lineage["deployed_fit_sha256"]
        or row["analysis_split_sha256"] != split["serialized_sha256"]
        or count != split["example_count"]
        or row["fingerprint_sha256"] != expected_fingerprint_sha256
        or not aggregates_match
    ):
        raise ValueError(f"causal summary layer {ordinal} is inconsistent")


def _validate_summaries(
    value: object,
    *,
    lineage: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label="generator_causal_summaries")
    if len(rows) != _LAYER_COUNT:
        raise ValueError(
            "generator_causal_summaries must contain exactly 18 rows"
        )
    validated: list[Mapping[str, object]] = []
    for ordinal, (row, lineage_row) in enumerate(
        zip(rows, lineage, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise TypeError("generator causal summaries must be mappings")
        _validate_summary(
            row,
            ordinal=ordinal,
            lineage=lineage_row,
            split=split,
        )
        validated.append(row)
    return tuple(validated)


def _validate_pair(
    row: Mapping[str, object],
    *,
    expected_pair: tuple[int, int],
    lineage: tuple[Mapping[str, object], ...],
    summaries: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
) -> None:
    left, right = expected_pair
    _strict_fields(
        row,
        _PAIR_FIELDS,
        label=f"pairwise similarity {left},{right}",
    )
    for field in (
        "centered_shared_logit_effect_cosine",
        "prompt_nll_effect_spearman",
    ):
        _require_finite(
            row[field],
            label=f"pair {left},{right} {field}",
            minimum=-1.0,
            maximum=1.0,
        )
    for field in (
        "top_importance_overlap",
        "top_importance_sign_agreement",
    ):
        _require_finite(
            row[field],
            label=f"pair {left},{right} {field}",
            minimum=0.0,
            maximum=1.0,
        )
    intersection = _require_int(
        row["top_importance_intersection_count"],
        label=f"pair {left},{right} top importance intersection count",
        minimum=0,
    )
    if type(row["sufficient_causal_variation"]) is not bool:
        raise TypeError("sufficient_causal_variation must be a bool")
    hypothesis = _require_name(
        row["observational_hypothesis"],
        label=f"pair {left},{right} observational hypothesis",
    )
    policy = _OBSERVATION_PROTOCOL["observational_family_policy"]
    if not isinstance(policy, Mapping):
        raise AssertionError("fixed observational policy must be a mapping")
    sufficient = row["sufficient_causal_variation"]
    if sufficient:
        passed = (
            row["centered_shared_logit_effect_cosine"]
            >= policy["minimum_centered_effect_cosine"],
            row["prompt_nll_effect_spearman"]
            >= policy["minimum_prompt_nll_spearman"],
            row["top_importance_overlap"]
            >= policy["minimum_top_importance_overlap"],
            row["top_importance_sign_agreement"]
            >= policy["minimum_top_importance_sign_agreement"],
        )
        expected_hypothesis = (
            "aligned_observational_family_hypothesis"
            if all(passed)
            else (
                "mixed_observational_family_evidence"
                if sum(passed) >= 2
                else "distinct_observational_effect_hypothesis"
            )
        )
    else:
        expected_hypothesis = "insufficient_causal_variation"
    top_count = _OBSERVATION_PROTOCOL["top_importance_count"]
    if type(top_count) is not int:
        raise AssertionError("fixed top importance count must be an integer")
    expected_overlap = intersection / top_count
    if (
        row["left_layer_ordinal"] != left
        or row["right_layer_ordinal"] != right
        or row["left_generator_plan_sha256"]
        != lineage[left]["deployed_generator_plan_sha256"]
        or row["right_generator_plan_sha256"]
        != lineage[right]["deployed_generator_plan_sha256"]
        or row["left_fingerprint_sha256"]
        != summaries[left]["fingerprint_sha256"]
        or row["right_fingerprint_sha256"]
        != summaries[right]["fingerprint_sha256"]
        or row["analysis_split_sha256"] != split["serialized_sha256"]
        or row["shared_prompt_count"] != split["example_count"]
        or intersection > top_count
        or not math.isclose(
            float(row["top_importance_overlap"]),
            expected_overlap,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or (
            intersection == 0
            and row["top_importance_sign_agreement"] != 0.0
        )
        or hypothesis not in _HYPOTHESIS_LABELS
        or hypothesis != expected_hypothesis
        or (
            sufficient
            and split["example_count"]
            < policy["minimum_prompt_count"]
        )
        or row["observational_only"] is not True
        or row["authorizes_merge"] is not False
        or row["authorizes_pruning"] is not False
        or row["authorizes_routing"] is not False
        or row["authorizes_mutation"] is not False
    ):
        raise ValueError(
            f"pairwise similarity {left},{right} lineage is inconsistent"
        )


def _validate_pairs(
    value: object,
    *,
    lineage: tuple[Mapping[str, object], ...],
    summaries: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
) -> None:
    rows = _sequence(value, label="pairwise_similarities")
    expected_pairs = tuple(combinations(range(_LAYER_COUNT), 2))
    if len(rows) != _PAIR_COUNT:
        raise ValueError(
            "pairwise_similarities must contain all 153 unordered pairs"
        )
    for expected_pair, row in zip(expected_pairs, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("pairwise similarity rows must be mappings")
        _validate_pair(
            row,
            expected_pair=expected_pair,
            lineage=lineage,
            summaries=summaries,
            split=split,
        )


def _validate_payload(raw: Mapping[str, object]) -> None:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="causal fingerprint artifact")
    if (
        raw["schema"] != GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA
        or raw["format_version"]
        != GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["observation_protocol"] != _OBSERVATION_PROTOCOL
        or raw["safety"] != _SAFETY
    ):
        raise ValueError(
            "causal fingerprint artifact header or authority is invalid"
        )
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
        raise ValueError("causal fingerprint scientific payload hash mismatch")

    model = raw["model"]
    sources = raw["frozen_sources"]
    split = raw["analysis_split"]
    causal_analysis = raw["causal_analysis_lineage"]
    for value, label in (
        (model, "model"),
        (sources, "frozen_sources"),
        (split, "analysis_split"),
        (causal_analysis, "causal_analysis_lineage"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_model(model)
    _validate_frozen_sources(sources)
    _validate_analysis_split(split)
    _validate_causal_analysis_lineage(
        causal_analysis,
        model=model,
        split=split,
    )
    lineage = _validate_lineage(
        raw["generator_fit_lineage"],
        sources=sources,
    )
    plan_hashes = _hash_sequence(
        raw["deployed_generator_plan_sha256s"],
        label="deployed_generator_plan_sha256s",
    )
    expected_plan_hashes = tuple(
        row["deployed_generator_plan_sha256"] for row in lineage
    )
    if (
        len(plan_hashes) != _LAYER_COUNT
        or plan_hashes != expected_plan_hashes
        or len(plan_hashes) != len(set(plan_hashes))
    ):
        raise ValueError(
            "deployed generator plan catalog must be 18 ordered unique hashes"
        )
    summaries = _validate_summaries(
        raw["generator_causal_summaries"],
        lineage=lineage,
        split=split,
    )
    _validate_pairs(
        raw["pairwise_similarities"],
        lineage=lineage,
        summaries=summaries,
        split=split,
    )


def build_gemma3_generator_causal_fingerprint_payload(
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    analysis_split: Mapping[str, object],
    causal_analysis_lineage: Mapping[str, object],
    generator_fit_lineage: Sequence[Mapping[str, object]],
    generator_causal_summaries: Sequence[Mapping[str, object]],
    pairwise_similarities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build and authenticate one tensor-free causal-fingerprint payload."""

    for value, label in (
        (model, "model"),
        (frozen_sources, "frozen_sources"),
        (analysis_split, "analysis_split"),
        (causal_analysis_lineage, "causal_analysis_lineage"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    lineage = tuple(dict(row) for row in generator_fit_lineage)
    summaries = tuple(dict(row) for row in generator_causal_summaries)
    pairs = tuple(dict(row) for row in pairwise_similarities)
    without_digest: dict[str, object] = {
        "schema": GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA,
        "format_version": (
            GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION
        ),
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "frozen_sources": dict(frozen_sources),
        "analysis_split": dict(analysis_split),
        "causal_analysis_lineage": dict(causal_analysis_lineage),
        "deployed_generator_plan_sha256s": tuple(
            row.get("deployed_generator_plan_sha256")
            for row in lineage
        ),
        "generator_fit_lineage": lineage,
        "observation_protocol": dict(_OBSERVATION_PROTOCOL),
        "generator_causal_summaries": summaries,
        "pairwise_similarities": pairs,
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    canonical = _json_clone(without_digest)
    if not isinstance(canonical, dict):
        raise AssertionError("canonical causal fingerprint payload must be a dict")
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


def save_gemma3_generator_causal_fingerprint_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    analysis_split: Mapping[str, object],
    causal_analysis_lineage: Mapping[str, object],
    generator_fit_lineage: Sequence[Mapping[str, object]],
    generator_causal_summaries: Sequence[Mapping[str, object]],
    pairwise_similarities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Exclusively and atomically publish one strict JSON-only artifact."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("causal fingerprint artifact output must use .json")
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite causal fingerprint artifact"
        )
    payload = build_gemma3_generator_causal_fingerprint_payload(
        model=model,
        frozen_sources=frozen_sources,
        analysis_split=analysis_split,
        causal_analysis_lineage=causal_analysis_lineage,
        generator_fit_lineage=generator_fit_lineage,
        generator_causal_summaries=generator_causal_summaries,
        pairwise_similarities=pairwise_similarities,
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
                "refusing to overwrite causal fingerprint artifact"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_gemma3_generator_causal_fingerprint_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load and authenticate one JSON-only causal fingerprint."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(
            handle,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(raw, dict):
        raise TypeError("causal fingerprint artifact must be a JSON object")
    _validate_payload(raw)
    return raw
