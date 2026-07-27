"""Strict JSON boundary for the Gemma modal-generator causal map.

The artifact binds a frozen deployed generator stack, its already-published
singleton causal fingerprint, exhaustive pair suppressions, forward local
responses, and hashed declared-family cohorts.  It contains only lineage,
prompt hashes, and finite scalar summaries.  It never contains prompt text,
token IDs, logits, tensors, weights, or raw activation rows.

Every relation is observational.  This boundary deliberately grants no
merge, prune, route, compile, execute, or mutation authority.
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
    "GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION",
    "GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA",
    "build_gemma3_generator_causal_map_payload",
    "gemma3_generator_causal_map_cohort_id_sha256",
    "gemma3_generator_causal_map_cohort_membership_sha256",
    "load_gemma3_generator_causal_map_artifact",
    "save_gemma3_generator_causal_map_artifact",
]


GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA = (
    "fisher_graph.gemma3_generator_causal_map_development"
)
GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION = 1

_LAYER_COUNT = 18
_REFIT_START_LAYER = 10
_PAIR_COUNT = _LAYER_COUNT * (_LAYER_COUNT - 1) // 2
_COHORT_COUNT = 8
_COHORT_SIZE_MULTISET = (1, 1, 2, 2, 3, 3, 4, 4)
_DIGEST_DOMAIN = b"fisher_graph.gemma3.generator_causal_map.json.v1\0"
_INTERACTION_DIGEST_DOMAIN = (
    b"fisher_graph.generator_interaction_map.artifact.v1\0"
)
_FAMILY_ID_DOMAIN = b"fisher_graph.gemma3.causal_map.family_id.v1\0"
_COHORT_MEMBERSHIP_DOMAIN = (
    b"fisher_graph.gemma3.causal_map.cohort_membership.v1\0"
)
_GENERATOR_CATALOG_DOMAIN = (
    b"fisher_graph.gemma3.generator_causal_fingerprint.catalog.v1\0"
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
_FINGERPRINT_SCHEMA = (
    "fisher_graph.gemma3_generator_causal_fingerprint_development"
)

_SCIENTIFIC_STATUS: dict[str, object] = {
    "outcome": "observational_adaptive_open_development_causal_map",
    "development_only": True,
    "adaptive_analysis": True,
    "observational_metrics_only": True,
    "heldout_confirmation": False,
    "compression_claim": False,
    "latency_or_kernel_speed_claim": False,
    "authorizes_merge": False,
    "authorizes_pruning": False,
    "authorizes_routing": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
    "authorizes_mutation": False,
    "scope": "frozen_deployed_full_mlp_stack_generators",
}

_OBSERVATION_PROTOCOL: dict[str, object] = {
    "protocol_id": "exhaustive_generator_interaction_map_v1",
    "transformer_layer_count": _LAYER_COUNT,
    "node_count": _LAYER_COUNT,
    "singleton_similarity_edge_count": _PAIR_COUNT,
    "joint_interaction_edge_count": _PAIR_COUNT,
    "forward_directed_response_edge_count": _PAIR_COUNT,
    "node_order": "ascending_layer_ordinal_0_through_17",
    "pair_order": "lexicographic_unordered_layer_pairs",
    "directed_order": "lexicographic_forward_layer_pairs",
    "prompt_order": "analysis_split_content_sha256_order",
    "joint_intervention": (
        "frozen_baseline_all_singletons_and_all_canonical_joint_pairs"
    ),
    "directed_response": (
        "singleton_upstream_suppression_to_downstream_generator_output"
    ),
    "cohort_source": "exact_declared_assessment_family_id",
    "cohort_id_storage": "domain_separated_sha256_only",
    "cohort_count": _COHORT_COUNT,
    "cohort_order": "lexicographic_cohort_id",
    "cohort_membership": "exact_disjoint_analysis_split_partition",
    "cohort_size_multiset": list(_COHORT_SIZE_MULTISET),
    "affinity_order": "layer_major_then_cohort_ordinal",
    "singleton_cohort_status": (
        "descriptive_singleton_insufficient_for_stability"
    ),
    "multi_prompt_cohort_status": (
        "descriptive_multi_prompt_open_development"
    ),
    "source_model_weights_mutated": False,
    "generator_weights_mutated": False,
    "pruning_performed": False,
    "merge_performed": False,
    "routing_performed": False,
    "compilation_performed": False,
}

_SAFETY: dict[str, bool] = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_generator_weights": False,
    "contains_raw_activation_rows": False,
    "contains_raw_generator_output_rows": False,
    "contains_hashed_prompt_membership": True,
    "contains_prompt_conditioned_scalar_summaries": True,
    "contains_only_lineage_protocol_hashes_and_safe_metrics": True,
    "analysis_only": True,
    "authorizes_merge": False,
    "authorizes_pruning": False,
    "authorizes_routing": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
    "authorizes_mutation": False,
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
        "activations",
        "activation_rows",
        "raw_activation_rows",
        "generator_outputs",
        "raw_generator_output_rows",
        "source_weights",
        "source_model_weights",
        "source_parameter_values",
        "generator_weights",
        "weights",
        "model_state_dict",
        "source_state_dict",
        "state_dict",
    }
)

_PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "frozen_sources",
    "analysis_split",
    "cohort_partition_lineage",
    "interaction_analysis_lineage",
    "deployed_generator_plan_sha256s",
    "generator_fit_lineage",
    "observation_protocol",
    "generator_nodes",
    "copied_pairwise_similarities",
    "joint_interactions",
    "directed_edges",
    "prompt_cohorts",
    "generator_cohort_affinities",
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
_FROZEN_SOURCE_FIELDS = {
    "base_full_stack",
    "sequential_refit",
    "singleton_causal_fingerprint",
}
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
_LINEAGE_FIELDS = {
    "layer_ordinal",
    "layer_id",
    "deployment_source",
    "source_artifact_scientific_payload_sha256",
    "base_fit_sha256",
    "deployed_fit_sha256",
    "deployed_generator_plan_sha256",
}
_COHORT_PARTITION_LINEAGE_FIELDS = {
    "partition_plan_sha256",
    "assessment_partition_sha256",
    "source_export_sha256",
    "source_fit_prompt_index_sha256",
    "role",
    "assessment_status",
    "membership_provenance",
    "membership_externally_authenticated",
    "serialized_contains_prompt_text",
    "prompt_count",
    "family_count",
    "family_id_storage",
    "exact_declared_family_membership",
}
_NODE_FIELDS = {
    "layer_ordinal",
    "layer_id",
    "generator_id",
    "deployment_source",
    "deployed_generator_plan_sha256",
    "deployed_fit_sha256",
    "singleton_fingerprint_sha256",
    "mean_muted_minus_baseline_nll_per_token",
    "rms_muted_minus_baseline_nll_per_token",
    "mean_absolute_muted_minus_baseline_nll_per_token",
    "maximum_absolute_muted_minus_baseline_nll_per_token",
    "mean_baseline_to_muted_kl_per_token",
    "mean_top1_agreement_to_baseline",
    "mean_centered_anchor_logit_effect_rms",
    "positive_delta_fraction",
    "observational_only",
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
}
_COPIED_PAIR_FIELDS = {
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
_PROMPT_INTERACTION_FIELDS = {
    "prompt_content_sha256",
    "nll_second_difference_per_token",
    "joint_baseline_to_condition_kl_per_token",
    "joint_top1_agreement_to_baseline",
    "centered_anchor_interaction_residual_rms",
    "relative_interaction_denominator_rms",
    "relative_interaction_ratio",
    "relative_interaction_defined",
}
_JOINT_FIELDS = {
    "left_layer_ordinal",
    "right_layer_ordinal",
    "left_generator_id",
    "right_generator_id",
    "left_generator_plan_sha256",
    "right_generator_plan_sha256",
    "left_singleton_fingerprint_sha256",
    "right_singleton_fingerprint_sha256",
    "analysis_split_sha256",
    "prompt_count",
    "prompt_interactions",
    "mean_nll_second_difference_per_token",
    "rms_nll_second_difference_per_token",
    "mean_absolute_nll_second_difference_per_token",
    "maximum_absolute_nll_second_difference_per_token",
    "mean_joint_baseline_to_condition_kl_per_token",
    "mean_joint_top1_agreement_to_baseline",
    "mean_centered_anchor_interaction_residual_rms",
    "mean_relative_interaction_denominator_rms",
    "relative_interaction_defined_fraction",
    "mean_relative_interaction_ratio_over_defined",
    "maximum_relative_interaction_ratio_over_defined",
    "observational_only",
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
}
_PROMPT_RESPONSE_FIELDS = {
    "prompt_content_sha256",
    "directed_response_rms",
    "baseline_generator_output_rms",
    "directed_response_cosine",
    "directed_response_cosine_defined",
    "directed_response_ratio",
    "directed_response_ratio_defined",
}
_DIRECTED_FIELDS = {
    "upstream_layer_ordinal",
    "downstream_layer_ordinal",
    "upstream_generator_id",
    "downstream_generator_id",
    "upstream_generator_plan_sha256",
    "downstream_generator_plan_sha256",
    "upstream_singleton_fingerprint_sha256",
    "downstream_singleton_fingerprint_sha256",
    "analysis_split_sha256",
    "prompt_count",
    "prompt_responses",
    "mean_directed_response_rms",
    "maximum_directed_response_rms",
    "mean_baseline_generator_output_rms",
    "directed_response_cosine_defined_fraction",
    "mean_directed_response_cosine_over_defined",
    "directed_response_ratio_defined_fraction",
    "mean_directed_response_ratio_over_defined",
    "maximum_directed_response_ratio_over_defined",
    "strict_upstream_invariance_confirmed",
    "observational_only",
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
}
_COHORT_FIELDS = {
    "cohort_ordinal",
    "cohort_id",
    "source_partition_sha256",
    "prompt_content_sha256s",
    "prompt_count",
    "membership_sha256",
    "membership_exact",
    "stability_status",
}
_AFFINITY_FIELDS = {
    "layer_ordinal",
    "cohort_ordinal",
    "cohort_id",
    "membership_sha256",
    "deployed_generator_plan_sha256",
    "singleton_fingerprint_sha256",
    "prompt_count",
    "mean_muted_minus_baseline_nll_per_token",
    "rms_muted_minus_baseline_nll_per_token",
    "mean_absolute_muted_minus_baseline_nll_per_token",
    "maximum_absolute_muted_minus_baseline_nll_per_token",
    "mean_baseline_to_muted_kl_per_token",
    "mean_top1_agreement_to_baseline",
    "mean_centered_anchor_logit_effect_rms",
    "positive_delta_count",
    "positive_delta_fraction",
    "mean_absolute_nll_importance_rank",
    "descriptive_only",
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
}
_AUTHORITY_FIELDS = (
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
)

_INTERACTION_SAFETY = {
    "contains_source_model_weights": False,
    "contains_generator_weights": False,
    "contains_prompt_text": False,
    "contains_raw_example_ids": False,
    "contains_token_ids": False,
    "contains_targets": False,
    "contains_raw_logits": False,
    "contains_local_activation_rows": False,
    "contains_local_generator_output_rows": False,
    "analysis_only": True,
    "observational_hypotheses_only": True,
    "strict_upstream_invariance": True,
    "mediation_measured": False,
    "authorizes_intervention": False,
    "authorizes_merge": False,
    "authorizes_pruning": False,
    "authorizes_routing": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
    "authorizes_mutation": False,
}
_INTERACTION_LINEAGE_FIELDS = {
    "artifact_kind",
    "format_version",
    *_INTERACTION_SAFETY,
    "provenance",
    "generator_ids",
    "pair_catalog",
    "directed_edge_catalog",
    "example_id_sha256s",
    "generator_count",
    "pair_count",
    "directed_edge_count",
    "prompt_count",
    "upstream_invariance_prompt_checks",
    "anchor_count",
    "anchor_frame_width",
    "shared_frame",
    "effect_centering",
    "interaction_normalization",
    "relative_interaction_numerator_field",
    "tensor_sha256s",
    "artifact_sha256",
}
_INTERACTION_PROVENANCE_FIELDS = {
    "source_model_sha256",
    "generator_catalog_sha256",
    "evaluation_split_sha256",
    "objective_sha256",
    "intervention",
    "local_response",
}
_INTERACTION_TENSOR_FIELDS = {
    "supervised_token_counts",
    "valid_token_counts",
    "prompt_nll_second_differences",
    "prompt_joint_baseline_to_condition_kls",
    "prompt_joint_top1_agreements",
    "prompt_centered_anchor_interaction_residual_rms",
    "prompt_relative_interaction_denominator_rms",
    "prompt_relative_interaction_ratios",
    "prompt_relative_interaction_defined",
    "prompt_directed_response_rms",
    "prompt_directed_baseline_output_rms",
    "prompt_directed_response_cosines",
    "prompt_directed_response_cosine_defined",
    "prompt_directed_response_ratios",
    "prompt_directed_response_ratio_defined",
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


def _finite(
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


def _domain_digest(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return _domain_digest(value, domain=_DIGEST_DOMAIN)


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


def _require_no_authority(
    row: Mapping[str, object],
    *,
    label: str,
) -> None:
    if any(row[field] is not False for field in _AUTHORITY_FIELDS):
        raise ValueError(f"{label} grants forbidden optimization authority")


def _close(actual: object, expected: float) -> bool:
    return math.isclose(
        float(actual),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def gemma3_generator_causal_map_cohort_id_sha256(
    source_family_id: str,
) -> str:
    """Hash one raw declared family ID without serializing it."""

    if not isinstance(source_family_id, str) or not source_family_id:
        raise ValueError("source_family_id must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(_FAMILY_ID_DOMAIN)
    digest.update(source_family_id.encode("utf-8"))
    return digest.hexdigest()


def gemma3_generator_causal_map_cohort_membership_sha256(
    *,
    cohort_id: str,
    prompt_content_sha256s: Sequence[str],
) -> str:
    """Hash one exact, ordered prompt-hash cohort membership."""

    cohort = _require_sha256(cohort_id, label="cohort_id")
    members = _hash_sequence(
        prompt_content_sha256s,
        label="prompt_content_sha256s",
    )
    if not members or len(members) != len(set(members)):
        raise ValueError("cohort membership must be nonempty and unique")
    return _domain_digest(
        {
            "cohort_id": cohort,
            "prompt_content_sha256s": members,
        },
        domain=_COHORT_MEMBERSHIP_DOMAIN,
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
        raise ValueError("causal-map analysis must be local-only")


def _validate_sources(sources: Mapping[str, object]) -> None:
    _strict_fields(sources, _FROZEN_SOURCE_FIELDS, label="frozen sources")
    expected = (
        ("base_full_stack", _FULL_STACK_SCHEMA),
        ("sequential_refit", _REFIT_SCHEMA),
        ("singleton_causal_fingerprint", _FINGERPRINT_SCHEMA),
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
    if (
        len(set(file_hashes)) != len(expected)
        or len(set(scientific_hashes)) != len(expected)
    ):
        raise ValueError("all frozen sources must bind distinct artifacts")


def _validate_split(split: Mapping[str, object]) -> tuple[str, ...]:
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
    if len(members) != count or len(set(members)) != count:
        raise ValueError("analysis split membership must be exact and unique")
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


def _validate_interaction_lineage(
    lineage: Mapping[str, object],
    *,
    model: Mapping[str, object],
    split: Mapping[str, object],
) -> None:
    _strict_fields(
        lineage,
        _INTERACTION_LINEAGE_FIELDS,
        label="interaction analysis lineage",
    )
    if (
        lineage["artifact_kind"]
        != "fisher_graph.modal_generator_interaction_map"
        or lineage["format_version"] != 1
        or any(
            lineage[field] is not expected
            for field, expected in _INTERACTION_SAFETY.items()
        )
    ):
        raise ValueError("interaction analysis header or safety is invalid")
    provenance = lineage["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("interaction analysis provenance must be a mapping")
    _strict_fields(
        provenance,
        _INTERACTION_PROVENANCE_FIELDS,
        label="interaction analysis provenance",
    )
    for field in (
        "source_model_sha256",
        "generator_catalog_sha256",
        "evaluation_split_sha256",
        "objective_sha256",
    ):
        _require_sha256(provenance[field], label=f"provenance {field}")
    if (
        provenance["source_model_sha256"]
        != model["adapter_model_fingerprint"]
        or provenance["evaluation_split_sha256"]
        != split["serialized_sha256"]
        or provenance["intervention"]
        != "frozen_baseline_all_singletons_and_all_canonical_joint_pairs"
        or provenance["local_response"]
        != "singleton_upstream_suppression_to_downstream_generator_output"
    ):
        raise ValueError("interaction analysis provenance is inconsistent")
    ids = tuple(
        _require_name(value, label="interaction generator id")
        for value in _sequence(
            lineage["generator_ids"],
            label="interaction generator ids",
        )
    )
    if len(ids) != _LAYER_COUNT or len(set(ids)) != _LAYER_COUNT:
        raise ValueError("interaction generator IDs must be 18 unique names")
    pairs = tuple(combinations(ids, 2))
    example_ids = _hash_sequence(
        lineage["example_id_sha256s"],
        label="interaction example_id_sha256s",
    )
    if (
        tuple(tuple(row) for row in lineage["pair_catalog"]) != pairs
        or tuple(
            tuple(row) for row in lineage["directed_edge_catalog"]
        )
        != pairs
        or len(example_ids) != split["example_count"]
        or len(set(example_ids)) != len(example_ids)
        or lineage["generator_count"] != _LAYER_COUNT
        or lineage["pair_count"] != _PAIR_COUNT
        or lineage["directed_edge_count"] != _PAIR_COUNT
        or lineage["prompt_count"] != split["example_count"]
        or lineage["upstream_invariance_prompt_checks"]
        != _PAIR_COUNT * split["example_count"]
        or lineage["anchor_count"] != 8
        or lineage["anchor_frame_width"] != 9
        or lineage["shared_frame"]
        != (
            "per_supervised_token_target_then_stable_baseline_"
            "top_non_target_logits"
        )
        or lineage["effect_centering"]
        != "per_supervised_token_anchor_mean"
        or lineage["interaction_normalization"]
        != "residual_rms_over_root_sum_singleton_anchor_mean_square"
        or lineage["relative_interaction_numerator_field"]
        != "prompt_centered_anchor_interaction_residual_rms"
    ):
        raise ValueError("interaction analysis protocol is inconsistent")
    tensor_hashes = lineage["tensor_sha256s"]
    if not isinstance(tensor_hashes, Mapping):
        raise TypeError("interaction tensor hash catalog must be a mapping")
    _strict_fields(
        tensor_hashes,
        _INTERACTION_TENSOR_FIELDS,
        label="interaction tensor hash catalog",
    )
    for field in sorted(_INTERACTION_TENSOR_FIELDS):
        _require_sha256(tensor_hashes[field], label=f"tensor hash {field}")
    artifact_sha256 = _require_sha256(
        lineage["artifact_sha256"],
        label="interaction artifact_sha256",
    )
    without_digest = {
        key: value
        for key, value in lineage.items()
        if key != "artifact_sha256"
    }
    expected_digest = _domain_digest(
        without_digest,
        domain=_INTERACTION_DIGEST_DOMAIN,
    )
    if artifact_sha256 != expected_digest:
        raise ValueError("interaction analysis artifact hash mismatch")


def _validate_cohort_partition_lineage(
    lineage: Mapping[str, object],
    *,
    split: Mapping[str, object],
) -> None:
    _strict_fields(
        lineage,
        _COHORT_PARTITION_LINEAGE_FIELDS,
        label="cohort partition lineage",
    )
    for field in (
        "partition_plan_sha256",
        "assessment_partition_sha256",
        "source_export_sha256",
        "source_fit_prompt_index_sha256",
    ):
        _require_sha256(lineage[field], label=f"cohort partition {field}")
    if (
        lineage["role"] != "open_development_assessment"
        or lineage["assessment_status"]
        != "open_development_not_closed_guard"
        or lineage["membership_provenance"]
        != "caller_declared_self_attested"
        or lineage["membership_externally_authenticated"] is not False
        or lineage["serialized_contains_prompt_text"] is not False
        or lineage["prompt_count"] != split["example_count"]
        or lineage["family_count"] != _COHORT_COUNT
        or lineage["family_id_storage"]
        != "domain_separated_sha256_only"
        or lineage["exact_declared_family_membership"] is not True
    ):
        raise ValueError("cohort partition lineage is inconsistent")


def _validate_lineage(
    value: object,
    *,
    sources: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label="generator_fit_lineage")
    if len(rows) != _LAYER_COUNT:
        raise ValueError("generator_fit_lineage must contain exactly 18 rows")
    result: list[Mapping[str, object]] = []
    plans: set[str] = set()
    fits: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("generator fit lineage rows must be mappings")
        _strict_fields(
            row,
            _LINEAGE_FIELDS,
            label=f"generator fit lineage layer {ordinal}",
        )
        source_key = (
            "base_full_stack"
            if ordinal < _REFIT_START_LAYER
            else "sequential_refit"
        )
        expected_deployment = (
            "frozen_full_stack"
            if ordinal < _REFIT_START_LAYER
            else "sequential_refit_overlay"
        )
        base_fit = _require_sha256(
            row["base_fit_sha256"],
            label=f"layer {ordinal} base fit",
        )
        deployed_fit = _require_sha256(
            row["deployed_fit_sha256"],
            label=f"layer {ordinal} deployed fit",
        )
        plan = _require_sha256(
            row["deployed_generator_plan_sha256"],
            label=f"layer {ordinal} deployed plan",
        )
        if (
            row["layer_ordinal"] != ordinal
            or row["layer_id"] != f"layer.{ordinal}"
            or row["deployment_source"] != expected_deployment
            or row["source_artifact_scientific_payload_sha256"]
            != sources[source_key]["scientific_payload_sha256"]  # type: ignore[index]
            or (ordinal < _REFIT_START_LAYER and base_fit != deployed_fit)
            or (ordinal >= _REFIT_START_LAYER and base_fit == deployed_fit)
            or plan in plans
            or deployed_fit in fits
        ):
            raise ValueError(f"generator lineage layer {ordinal} is invalid")
        plans.add(plan)
        fits.add(deployed_fit)
        result.append(row)
    return tuple(result)


def _generator_catalog_sha256(
    lineage: Sequence[Mapping[str, object]],
) -> str:
    rows = tuple(
        {
            "layer_ordinal": ordinal,
            "source_fit_sha256": row["base_fit_sha256"],
            "deployed_fit_sha256": row["deployed_fit_sha256"],
            "deployed_plan_sha256": row[
                "deployed_generator_plan_sha256"
            ],
        }
        for ordinal, row in enumerate(lineage)
    )
    return _domain_digest(rows, domain=_GENERATOR_CATALOG_DOMAIN)


def _validate_nodes(
    value: object,
    *,
    lineage: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label="generator_nodes")
    if len(rows) != _LAYER_COUNT:
        raise ValueError("generator_nodes must contain exactly 18 rows")
    result: list[Mapping[str, object]] = []
    fingerprints: set[str] = set()
    generator_ids: set[str] = set()
    for ordinal, (row, source) in enumerate(zip(rows, lineage, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError("generator node rows must be mappings")
        _strict_fields(row, _NODE_FIELDS, label=f"generator node {ordinal}")
        fingerprint = _require_sha256(
            row["singleton_fingerprint_sha256"],
            label=f"node {ordinal} singleton fingerprint",
        )
        generator_id = _require_name(
            row["generator_id"],
            label=f"node {ordinal} generator id",
        )
        for field in (
            "mean_muted_minus_baseline_nll_per_token",
            "rms_muted_minus_baseline_nll_per_token",
            "mean_absolute_muted_minus_baseline_nll_per_token",
            "maximum_absolute_muted_minus_baseline_nll_per_token",
            "mean_baseline_to_muted_kl_per_token",
            "mean_centered_anchor_logit_effect_rms",
        ):
            minimum = None if field.startswith("mean_muted") else 0.0
            _finite(row[field], label=f"node {ordinal} {field}", minimum=minimum)
        for field in (
            "mean_top1_agreement_to_baseline",
            "positive_delta_fraction",
        ):
            _finite(
                row[field],
                label=f"node {ordinal} {field}",
                minimum=0.0,
                maximum=1.0,
            )
        if (
            row["layer_ordinal"] != ordinal
            or row["layer_id"] != source["layer_id"]
            or row["deployment_source"] != source["deployment_source"]
            or row["deployed_generator_plan_sha256"]
            != source["deployed_generator_plan_sha256"]
            or row["deployed_fit_sha256"] != source["deployed_fit_sha256"]
            or fingerprint in fingerprints
            or generator_id in generator_ids
            or row["observational_only"] is not True
        ):
            raise ValueError(f"generator node {ordinal} is inconsistent")
        _require_no_authority(row, label=f"generator node {ordinal}")
        fingerprints.add(fingerprint)
        generator_ids.add(generator_id)
        result.append(row)
    return tuple(result)


def _validate_copied_pairs(
    value: object,
    *,
    nodes: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
) -> None:
    rows = _sequence(value, label="copied_pairwise_similarities")
    expected = tuple(combinations(range(_LAYER_COUNT), 2))
    if len(rows) != _PAIR_COUNT:
        raise ValueError("copied singleton similarities must contain 153 rows")
    for pair, row in zip(expected, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("copied singleton similarity rows must be mappings")
        left, right = pair
        _strict_fields(
            row,
            _COPIED_PAIR_FIELDS,
            label=f"copied singleton similarity {left},{right}",
        )
        for field in (
            "centered_shared_logit_effect_cosine",
            "prompt_nll_effect_spearman",
        ):
            _finite(
                row[field],
                label=f"copied pair {left},{right} {field}",
                minimum=-1.0,
                maximum=1.0,
            )
        for field in (
            "top_importance_overlap",
            "top_importance_sign_agreement",
        ):
            _finite(
                row[field],
                label=f"copied pair {left},{right} {field}",
                minimum=0.0,
                maximum=1.0,
            )
        _require_int(
            row["top_importance_intersection_count"],
            label="top importance intersection count",
        )
        intersection = int(row["top_importance_intersection_count"])
        sufficient = row["sufficient_causal_variation"]
        if sufficient:
            passed = (
                float(row["centered_shared_logit_effect_cosine"]) >= 0.9,
                float(row["prompt_nll_effect_spearman"]) >= 0.8,
                float(row["top_importance_overlap"]) >= 0.6,
                float(row["top_importance_sign_agreement"]) >= 0.8,
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
        if (
            row["left_layer_ordinal"] != left
            or row["right_layer_ordinal"] != right
            or row["left_generator_plan_sha256"]
            != nodes[left]["deployed_generator_plan_sha256"]
            or row["right_generator_plan_sha256"]
            != nodes[right]["deployed_generator_plan_sha256"]
            or row["left_fingerprint_sha256"]
            != nodes[left]["singleton_fingerprint_sha256"]
            or row["right_fingerprint_sha256"]
            != nodes[right]["singleton_fingerprint_sha256"]
            or row["analysis_split_sha256"] != split["serialized_sha256"]
            or row["shared_prompt_count"] != split["example_count"]
            or type(sufficient) is not bool
            or intersection < 0
            or intersection > 5
            or not _close(row["top_importance_overlap"], intersection / 5)
            or (
                intersection == 0
                and float(row["top_importance_sign_agreement"]) != 0.0
            )
            or row["observational_hypothesis"] != expected_hypothesis
            or (sufficient and int(split["example_count"]) < 3)
            or row["observational_only"] is not True
            or row["authorizes_merge"] is not False
            or row["authorizes_pruning"] is not False
            or row["authorizes_routing"] is not False
            or row["authorizes_mutation"] is not False
        ):
            raise ValueError(f"copied singleton pair {left},{right} is invalid")
        _require_name(
            row["observational_hypothesis"],
            label="observational hypothesis",
        )


def _prompt_interaction_metrics(
    rows: object,
    *,
    members: tuple[str, ...],
    label: str,
) -> tuple[list[float], ...]:
    values = _sequence(rows, label=f"{label} prompt interactions")
    if len(values) != len(members):
        raise ValueError(f"{label} prompt interaction count is invalid")
    columns: tuple[list[float], ...] = tuple([] for _ in range(6))
    defined_values: list[float] = []
    for index, (row, member) in enumerate(zip(values, members, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError("prompt interaction rows must be mappings")
        _strict_fields(
            row,
            _PROMPT_INTERACTION_FIELDS,
            label=f"{label} prompt interaction {index}",
        )
        metrics = (
            _finite(
                row["nll_second_difference_per_token"],
                label="NLL second difference",
            ),
            _finite(
                row["joint_baseline_to_condition_kl_per_token"],
                label="joint KL",
                minimum=0.0,
            ),
            _finite(
                row["joint_top1_agreement_to_baseline"],
                label="joint top1 agreement",
                minimum=0.0,
                maximum=1.0,
            ),
            _finite(
                row["centered_anchor_interaction_residual_rms"],
                label="interaction residual RMS",
                minimum=0.0,
            ),
            _finite(
                row["relative_interaction_denominator_rms"],
                label="interaction denominator RMS",
                minimum=0.0,
            ),
            _finite(
                row["relative_interaction_ratio"],
                label="relative interaction ratio",
                minimum=0.0,
            ),
        )
        defined = row["relative_interaction_defined"]
        if (
            row["prompt_content_sha256"] != member
            or type(defined) is not bool
            or defined != (metrics[4] > 0.0)
            or (not defined and metrics[5] != 0.0)
            or (
                defined
                and not _close(row["relative_interaction_ratio"], metrics[3] / metrics[4])
            )
        ):
            raise ValueError(f"{label} prompt interaction {index} is invalid")
        for column, metric in zip(columns, metrics, strict=True):
            column.append(metric)
        defined_values.append(1.0 if defined else 0.0)
    return (*columns, defined_values)


def _validate_joint_edges(
    value: object,
    *,
    nodes: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
    members: tuple[str, ...],
) -> None:
    rows = _sequence(value, label="joint_interactions")
    expected = tuple(combinations(range(_LAYER_COUNT), 2))
    if len(rows) != _PAIR_COUNT:
        raise ValueError("joint_interactions must contain all 153 pairs")
    for pair, row in zip(expected, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("joint interaction rows must be mappings")
        left, right = pair
        label = f"joint interaction {left},{right}"
        _strict_fields(row, _JOINT_FIELDS, label=label)
        (
            nll,
            kl,
            top1,
            residual,
            denominator,
            ratio,
            defined,
        ) = _prompt_interaction_metrics(
            row["prompt_interactions"],
            members=members,
            label=label,
        )
        defined_ratios = [
            metric
            for metric, flag in zip(ratio, defined, strict=True)
            if flag == 1.0
        ]
        expected_metrics = {
            "mean_nll_second_difference_per_token": _mean(nll),
            "rms_nll_second_difference_per_token": _rms(nll),
            "mean_absolute_nll_second_difference_per_token": _mean(
                [abs(item) for item in nll]
            ),
            "maximum_absolute_nll_second_difference_per_token": max(
                abs(item) for item in nll
            ),
            "mean_joint_baseline_to_condition_kl_per_token": _mean(kl),
            "mean_joint_top1_agreement_to_baseline": _mean(top1),
            "mean_centered_anchor_interaction_residual_rms": _mean(residual),
            "mean_relative_interaction_denominator_rms": _mean(denominator),
            "relative_interaction_defined_fraction": _mean(defined),
            "mean_relative_interaction_ratio_over_defined": (
                _mean(defined_ratios) if defined_ratios else 0.0
            ),
            "maximum_relative_interaction_ratio_over_defined": (
                max(defined_ratios, default=0.0)
            ),
        }
        if (
            row["left_layer_ordinal"] != left
            or row["right_layer_ordinal"] != right
            or row["left_generator_id"] != nodes[left]["generator_id"]
            or row["right_generator_id"] != nodes[right]["generator_id"]
            or row["left_generator_plan_sha256"]
            != nodes[left]["deployed_generator_plan_sha256"]
            or row["right_generator_plan_sha256"]
            != nodes[right]["deployed_generator_plan_sha256"]
            or row["left_singleton_fingerprint_sha256"]
            != nodes[left]["singleton_fingerprint_sha256"]
            or row["right_singleton_fingerprint_sha256"]
            != nodes[right]["singleton_fingerprint_sha256"]
            or row["analysis_split_sha256"] != split["serialized_sha256"]
            or row["prompt_count"] != len(members)
            or row["observational_only"] is not True
            or not all(_close(row[field], metric) for field, metric in expected_metrics.items())
        ):
            raise ValueError(f"{label} is inconsistent")
        _require_no_authority(row, label=label)


def _prompt_response_metrics(
    rows: object,
    *,
    members: tuple[str, ...],
    label: str,
) -> tuple[list[float], ...]:
    values = _sequence(rows, label=f"{label} prompt responses")
    if len(values) != len(members):
        raise ValueError(f"{label} prompt response count is invalid")
    response: list[float] = []
    baseline: list[float] = []
    cosine: list[float] = []
    cosine_defined: list[float] = []
    ratio: list[float] = []
    ratio_defined: list[float] = []
    for index, (row, member) in enumerate(zip(values, members, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError("prompt response rows must be mappings")
        _strict_fields(
            row,
            _PROMPT_RESPONSE_FIELDS,
            label=f"{label} prompt response {index}",
        )
        response_value = _finite(
            row["directed_response_rms"],
            label="directed response RMS",
            minimum=0.0,
        )
        baseline_value = _finite(
            row["baseline_generator_output_rms"],
            label="baseline generator output RMS",
            minimum=0.0,
        )
        cosine_value = _finite(
            row["directed_response_cosine"],
            label="directed response cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        ratio_value = _finite(
            row["directed_response_ratio"],
            label="directed response ratio",
            minimum=0.0,
        )
        cosine_flag = row["directed_response_cosine_defined"]
        ratio_flag = row["directed_response_ratio_defined"]
        if (
            row["prompt_content_sha256"] != member
            or type(cosine_flag) is not bool
            or type(ratio_flag) is not bool
            or cosine_flag != (baseline_value > 0.0 and response_value > 0.0)
            or ratio_flag != (baseline_value > 0.0)
            or (not cosine_flag and cosine_value != 0.0)
            or (not ratio_flag and ratio_value != 0.0)
            or (
                ratio_flag
                and not _close(
                    row["directed_response_ratio"],
                    response_value / baseline_value,
                )
            )
        ):
            raise ValueError(f"{label} prompt response {index} is invalid")
        response.append(response_value)
        baseline.append(baseline_value)
        cosine.append(cosine_value)
        cosine_defined.append(1.0 if cosine_flag else 0.0)
        ratio.append(ratio_value)
        ratio_defined.append(1.0 if ratio_flag else 0.0)
    return (
        response,
        baseline,
        cosine,
        cosine_defined,
        ratio,
        ratio_defined,
    )


def _validate_directed_edges(
    value: object,
    *,
    nodes: tuple[Mapping[str, object], ...],
    split: Mapping[str, object],
    members: tuple[str, ...],
) -> None:
    rows = _sequence(value, label="directed_edges")
    expected = tuple(combinations(range(_LAYER_COUNT), 2))
    if len(rows) != _PAIR_COUNT:
        raise ValueError("directed_edges must contain all 153 forward edges")
    downstream_baselines: dict[int, tuple[float, ...]] = {}
    for pair, row in zip(expected, rows, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("directed edge rows must be mappings")
        upstream, downstream = pair
        label = f"directed edge {upstream}->{downstream}"
        _strict_fields(row, _DIRECTED_FIELDS, label=label)
        (
            response,
            baseline,
            cosine,
            cosine_defined,
            ratio,
            ratio_defined,
        ) = _prompt_response_metrics(
            row["prompt_responses"],
            members=members,
            label=label,
        )
        baseline_tuple = tuple(baseline)
        prior_baseline = downstream_baselines.get(downstream)
        if prior_baseline is None:
            downstream_baselines[downstream] = baseline_tuple
        elif any(
            not _close(current, prior)
            for current, prior in zip(
                baseline_tuple,
                prior_baseline,
                strict=True,
            )
        ):
            raise ValueError(
                f"{label} baseline output differs across incoming edges"
            )
        defined_cosines = [
            metric
            for metric, flag in zip(cosine, cosine_defined, strict=True)
            if flag == 1.0
        ]
        defined_ratios = [
            metric
            for metric, flag in zip(ratio, ratio_defined, strict=True)
            if flag == 1.0
        ]
        expected_metrics = {
            "mean_directed_response_rms": _mean(response),
            "maximum_directed_response_rms": max(response),
            "mean_baseline_generator_output_rms": _mean(baseline),
            "directed_response_cosine_defined_fraction": _mean(
                cosine_defined
            ),
            "mean_directed_response_cosine_over_defined": (
                _mean(defined_cosines) if defined_cosines else 0.0
            ),
            "directed_response_ratio_defined_fraction": _mean(ratio_defined),
            "mean_directed_response_ratio_over_defined": (
                _mean(defined_ratios) if defined_ratios else 0.0
            ),
            "maximum_directed_response_ratio_over_defined": (
                max(defined_ratios, default=0.0)
            ),
        }
        if (
            row["upstream_layer_ordinal"] != upstream
            or row["downstream_layer_ordinal"] != downstream
            or row["upstream_generator_id"] != nodes[upstream]["generator_id"]
            or row["downstream_generator_id"]
            != nodes[downstream]["generator_id"]
            or row["upstream_generator_plan_sha256"]
            != nodes[upstream]["deployed_generator_plan_sha256"]
            or row["downstream_generator_plan_sha256"]
            != nodes[downstream]["deployed_generator_plan_sha256"]
            or row["upstream_singleton_fingerprint_sha256"]
            != nodes[upstream]["singleton_fingerprint_sha256"]
            or row["downstream_singleton_fingerprint_sha256"]
            != nodes[downstream]["singleton_fingerprint_sha256"]
            or row["analysis_split_sha256"] != split["serialized_sha256"]
            or row["prompt_count"] != len(members)
            or row["strict_upstream_invariance_confirmed"] is not True
            or row["observational_only"] is not True
            or not all(_close(row[field], metric) for field, metric in expected_metrics.items())
        ):
            raise ValueError(f"{label} is inconsistent")
        _require_no_authority(row, label=label)


def _validate_cohorts(
    value: object,
    *,
    split_members: tuple[str, ...],
    assessment_partition_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    rows = _sequence(value, label="prompt_cohorts")
    if len(rows) != _COHORT_COUNT:
        raise ValueError("prompt_cohorts must contain exactly 8 cohorts")
    result: list[Mapping[str, object]] = []
    ids: list[str] = []
    observed_members: list[str] = []
    split_order = {member: index for index, member in enumerate(split_members)}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("prompt cohort rows must be mappings")
        _strict_fields(row, _COHORT_FIELDS, label=f"prompt cohort {ordinal}")
        cohort_id = _require_sha256(
            row["cohort_id"],
            label=f"cohort {ordinal} id",
        )
        source_partition = _require_sha256(
            row["source_partition_sha256"],
            label=f"cohort {ordinal} source partition",
        )
        members = _hash_sequence(
            row["prompt_content_sha256s"],
            label=f"cohort {ordinal} members",
        )
        count = _require_int(
            row["prompt_count"],
            label=f"cohort {ordinal} prompt count",
            minimum=1,
        )
        membership = _require_sha256(
            row["membership_sha256"],
            label=f"cohort {ordinal} membership",
        )
        expected_status = (
            "descriptive_singleton_insufficient_for_stability"
            if count == 1
            else "descriptive_multi_prompt_open_development"
        )
        if (
            row["cohort_ordinal"] != ordinal
            or len(members) != count
            or len(set(members)) != count
            or any(member not in split_order for member in members)
            or tuple(sorted(members, key=split_order.__getitem__)) != members
            or membership
            != gemma3_generator_causal_map_cohort_membership_sha256(
                cohort_id=cohort_id,
                prompt_content_sha256s=members,
            )
            or row["membership_exact"] is not True
            or row["stability_status"] != expected_status
            or source_partition != assessment_partition_sha256
        ):
            raise ValueError(f"prompt cohort {ordinal} is inconsistent")
        ids.append(cohort_id)
        observed_members.extend(members)
        result.append(row)
    if (
        ids != sorted(ids)
        or len(set(ids)) != len(ids)
        or sorted(observed_members, key=split_order.__getitem__)
        != list(split_members)
        or len(set(observed_members)) != len(split_members)
        or tuple(sorted(len(row["prompt_content_sha256s"]) for row in rows))
        != _COHORT_SIZE_MULTISET
    ):
        raise ValueError("prompt cohorts are not the exact canonical partition")
    return tuple(result)


def _validate_affinities(
    value: object,
    *,
    nodes: tuple[Mapping[str, object], ...],
    cohorts: tuple[Mapping[str, object], ...],
) -> None:
    rows = _sequence(value, label="generator_cohort_affinities")
    if len(rows) != _LAYER_COUNT * len(cohorts):
        raise ValueError("generator affinities must contain 18 x cohort rows")
    index = 0
    for layer, node in enumerate(nodes):
        layer_rows: list[Mapping[str, object]] = []
        for cohort_ordinal, cohort in enumerate(cohorts):
            row = rows[index]
            index += 1
            if not isinstance(row, Mapping):
                raise TypeError("generator affinity rows must be mappings")
            label = f"generator affinity {layer},{cohort_ordinal}"
            _strict_fields(row, _AFFINITY_FIELDS, label=label)
            count = _require_int(
                row["prompt_count"],
                label=f"{label} prompt_count",
                minimum=1,
            )
            positive_count = _require_int(
                row["positive_delta_count"],
                label=f"{label} positive_delta_count",
            )
            for field in (
                "mean_muted_minus_baseline_nll_per_token",
                "rms_muted_minus_baseline_nll_per_token",
                "mean_absolute_muted_minus_baseline_nll_per_token",
                "maximum_absolute_muted_minus_baseline_nll_per_token",
                "mean_baseline_to_muted_kl_per_token",
                "mean_centered_anchor_logit_effect_rms",
            ):
                minimum = None if field.startswith("mean_muted") else 0.0
                _finite(row[field], label=f"{label} {field}", minimum=minimum)
            _finite(
                row["mean_top1_agreement_to_baseline"],
                label=f"{label} mean top1 agreement",
                minimum=0.0,
                maximum=1.0,
            )
            _finite(
                row["positive_delta_fraction"],
                label=f"{label} positive fraction",
                minimum=0.0,
                maximum=1.0,
            )
            _finite(
                row["mean_absolute_nll_importance_rank"],
                label=f"{label} mean importance rank",
                minimum=1.0,
                maximum=float(_LAYER_COUNT),
            )
            if (
                row["layer_ordinal"] != layer
                or row["cohort_ordinal"] != cohort_ordinal
                or row["cohort_id"] != cohort["cohort_id"]
                or row["membership_sha256"] != cohort["membership_sha256"]
                or row["deployed_generator_plan_sha256"]
                != node["deployed_generator_plan_sha256"]
                or row["singleton_fingerprint_sha256"]
                != node["singleton_fingerprint_sha256"]
                or count != cohort["prompt_count"]
                or positive_count > count
                or not _close(
                    row["positive_delta_fraction"],
                    positive_count / count,
                )
                or row["descriptive_only"] is not True
            ):
                raise ValueError(f"{label} is inconsistent")
            _require_no_authority(row, label=label)
            layer_rows.append(row)
        total = sum(int(row["prompt_count"]) for row in layer_rows)
        weighted_fields = {
            "mean_muted_minus_baseline_nll_per_token": (
                "mean_muted_minus_baseline_nll_per_token"
            ),
            "mean_absolute_muted_minus_baseline_nll_per_token": (
                "mean_absolute_muted_minus_baseline_nll_per_token"
            ),
            "mean_baseline_to_muted_kl_per_token": (
                "mean_baseline_to_muted_kl_per_token"
            ),
            "mean_top1_agreement_to_baseline": (
                "mean_top1_agreement_to_baseline"
            ),
            "mean_centered_anchor_logit_effect_rms": (
                "mean_centered_anchor_logit_effect_rms"
            ),
        }
        for node_field, affinity_field in weighted_fields.items():
            expected = math.fsum(
                int(row["prompt_count"]) * float(row[affinity_field])
                for row in layer_rows
            ) / total
            if not _close(node[node_field], expected):
                raise ValueError(
                    f"generator affinity layer {layer} does not pool to node"
                )
        pooled_rms = math.sqrt(
            math.fsum(
                int(row["prompt_count"])
                * float(row["rms_muted_minus_baseline_nll_per_token"]) ** 2
                for row in layer_rows
            )
            / total
        )
        pooled_maximum = max(
            float(
                row[
                    "maximum_absolute_muted_minus_baseline_nll_per_token"
                ]
            )
            for row in layer_rows
        )
        positive_fraction = (
            sum(int(row["positive_delta_count"]) for row in layer_rows)
            / total
        )
        if (
            not _close(
                node["rms_muted_minus_baseline_nll_per_token"],
                pooled_rms,
            )
            or not _close(
                node[
                    "maximum_absolute_muted_minus_baseline_nll_per_token"
                ],
                pooled_maximum,
            )
            or not _close(
                node["positive_delta_fraction"],
                positive_fraction,
            )
        ):
            raise ValueError(
                f"generator affinity layer {layer} does not pool to node"
            )


def _validate_payload(raw: Mapping[str, object]) -> None:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="causal-map artifact")
    if (
        raw["schema"] != GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA
        or raw["format_version"] != GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["observation_protocol"] != _OBSERVATION_PROTOCOL
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("causal-map artifact header or authority is invalid")
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
    if digest != _json_digest(without_digest):
        raise ValueError("causal-map scientific payload hash mismatch")

    model = raw["model"]
    sources = raw["frozen_sources"]
    split = raw["analysis_split"]
    cohort_partition = raw["cohort_partition_lineage"]
    interaction = raw["interaction_analysis_lineage"]
    for value, label in (
        (model, "model"),
        (sources, "frozen_sources"),
        (split, "analysis_split"),
        (cohort_partition, "cohort_partition_lineage"),
        (interaction, "interaction_analysis_lineage"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_model(model)
    _validate_sources(sources)
    members = _validate_split(split)
    _validate_cohort_partition_lineage(
        cohort_partition,
        split=split,
    )
    _validate_interaction_lineage(
        interaction,
        model=model,
        split=split,
    )
    lineage = _validate_lineage(
        raw["generator_fit_lineage"],
        sources=sources,
    )
    provenance = interaction["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance["generator_catalog_sha256"]
        != _generator_catalog_sha256(lineage)
    ):
        raise ValueError(
            "interaction generator catalog hash differs from fit lineage"
        )
    plans = _hash_sequence(
        raw["deployed_generator_plan_sha256s"],
        label="deployed_generator_plan_sha256s",
    )
    expected_plans = tuple(
        row["deployed_generator_plan_sha256"] for row in lineage
    )
    if plans != expected_plans or len(plans) != len(set(plans)):
        raise ValueError("deployed generator plan catalog is inconsistent")
    nodes = _validate_nodes(raw["generator_nodes"], lineage=lineage)
    interaction_ids = tuple(interaction["generator_ids"])
    node_ids = tuple(row["generator_id"] for row in nodes)
    if interaction_ids != node_ids:
        raise ValueError(
            "interaction generator catalog differs from generator nodes"
        )
    _validate_copied_pairs(
        raw["copied_pairwise_similarities"],
        nodes=nodes,
        split=split,
    )
    _validate_joint_edges(
        raw["joint_interactions"],
        nodes=nodes,
        split=split,
        members=members,
    )
    _validate_directed_edges(
        raw["directed_edges"],
        nodes=nodes,
        split=split,
        members=members,
    )
    cohorts = _validate_cohorts(
        raw["prompt_cohorts"],
        split_members=members,
        assessment_partition_sha256=cohort_partition[
            "assessment_partition_sha256"
        ],  # type: ignore[arg-type]
    )
    _validate_affinities(
        raw["generator_cohort_affinities"],
        nodes=nodes,
        cohorts=cohorts,
    )


def build_gemma3_generator_causal_map_payload(
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    analysis_split: Mapping[str, object],
    cohort_partition_lineage: Mapping[str, object],
    interaction_analysis_lineage: Mapping[str, object],
    generator_fit_lineage: Sequence[Mapping[str, object]],
    generator_nodes: Sequence[Mapping[str, object]],
    copied_pairwise_similarities: Sequence[Mapping[str, object]],
    joint_interactions: Sequence[Mapping[str, object]],
    directed_edges: Sequence[Mapping[str, object]],
    prompt_cohorts: Sequence[Mapping[str, object]],
    generator_cohort_affinities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build and authenticate one tensor-free causal-map payload."""

    for value, label in (
        (model, "model"),
        (frozen_sources, "frozen_sources"),
        (analysis_split, "analysis_split"),
        (cohort_partition_lineage, "cohort_partition_lineage"),
        (interaction_analysis_lineage, "interaction_analysis_lineage"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    lineage = tuple(dict(row) for row in generator_fit_lineage)
    without_digest: dict[str, object] = {
        "schema": GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA,
        "format_version": GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION,
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "frozen_sources": dict(frozen_sources),
        "analysis_split": dict(analysis_split),
        "cohort_partition_lineage": dict(cohort_partition_lineage),
        "interaction_analysis_lineage": dict(
            interaction_analysis_lineage
        ),
        "deployed_generator_plan_sha256s": tuple(
            row.get("deployed_generator_plan_sha256") for row in lineage
        ),
        "generator_fit_lineage": lineage,
        "observation_protocol": dict(_OBSERVATION_PROTOCOL),
        "generator_nodes": tuple(dict(row) for row in generator_nodes),
        "copied_pairwise_similarities": tuple(
            dict(row) for row in copied_pairwise_similarities
        ),
        "joint_interactions": tuple(
            dict(row) for row in joint_interactions
        ),
        "directed_edges": tuple(dict(row) for row in directed_edges),
        "prompt_cohorts": tuple(dict(row) for row in prompt_cohorts),
        "generator_cohort_affinities": tuple(
            dict(row) for row in generator_cohort_affinities
        ),
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    canonical = _json_clone(without_digest)
    if not isinstance(canonical, dict):
        raise AssertionError("canonical causal-map payload must be a dict")
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


def save_gemma3_generator_causal_map_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    frozen_sources: Mapping[str, object],
    analysis_split: Mapping[str, object],
    cohort_partition_lineage: Mapping[str, object],
    interaction_analysis_lineage: Mapping[str, object],
    generator_fit_lineage: Sequence[Mapping[str, object]],
    generator_nodes: Sequence[Mapping[str, object]],
    copied_pairwise_similarities: Sequence[Mapping[str, object]],
    joint_interactions: Sequence[Mapping[str, object]],
    directed_edges: Sequence[Mapping[str, object]],
    prompt_cohorts: Sequence[Mapping[str, object]],
    generator_cohort_affinities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Exclusively and atomically publish one strict JSON-only map."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("causal-map artifact output must use .json")
    if output.exists():
        raise FileExistsError("refusing to overwrite causal-map artifact")
    payload = build_gemma3_generator_causal_map_payload(
        model=model,
        frozen_sources=frozen_sources,
        analysis_split=analysis_split,
        cohort_partition_lineage=cohort_partition_lineage,
        interaction_analysis_lineage=interaction_analysis_lineage,
        generator_fit_lineage=generator_fit_lineage,
        generator_nodes=generator_nodes,
        copied_pairwise_similarities=copied_pairwise_similarities,
        joint_interactions=joint_interactions,
        directed_edges=directed_edges,
        prompt_cohorts=prompt_cohorts,
        generator_cohort_affinities=generator_cohort_affinities,
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
                "refusing to overwrite causal-map artifact"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_gemma3_generator_causal_map_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load and authenticate one JSON-only causal map."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(
            handle,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(raw, dict):
        raise TypeError("causal-map artifact must be a JSON object")
    _validate_payload(raw)
    return raw
