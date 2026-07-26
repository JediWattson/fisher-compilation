"""Build and gate the first structured MLP compression candidate on A only.

This runner deliberately stops before every heldout role.  It statically
authenticates the complete v6 corpus, but only calibration A may be tokenized
or executed.  A passing result is an unevaluated compression candidate, never
a scientific compression-success claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
)
from .gemma3_codimension_rotation_experiment import _file_sha256
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_width_single_layer_experiment import (
    DEFAULT_BLOCK_DELTA_COSINE_MIN,
    DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS,
    DEFAULT_MINIMUM_FISHER_ROWS,
    DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS,
    DEFAULT_MINIMUM_LENGTH_BUCKETS,
    DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS,
    DEFAULT_NLL_ATOL,
    DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    DEFAULT_TEACHER_KL_MAX,
    DEFAULT_TOP1_MIN,
    DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    FAMILY_STATUS,
    PROMPT_STATUS,
    _direct_gates,
    _require_complete_middle_layer_demand,
    _require_prompt_protocol,
    _tokenized_stream_contract,
    load_prompt_family_manifest,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_stability_experiment import (
    _library_versions,
    _tokenizer_provenance,
    load_gemma3_prompt_splits,
)
from .gemma3_structured_single_layer_experiment import (
    DEFAULT_BRANCH_DELTA_COSINE_MIN,
    DEFAULT_BRANCH_DELTA_NRMSE_MAX,
    DEFAULT_NATIVE_PARITY_TOLERANCE,
    _branch_gates,
    _corpus_audit_binding,
    _format4_family_binding,
    collect_structured_training_batches,
    evaluate_calibration_a_fidelity,
    load_gemma3_structured_single_layer_artifact,
)
from .structured_mlp_compression import (
    GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH,
    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
)
from .structured_mlp_compression_pipeline import (
    STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION,
    STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA,
    build_gemma_mlp_first_rung_candidate,
    collect_gemma_mlp_fisher_taylor_batches,
)
from .structured_operator_bootstrap import (
    structured_operator_coefficient_sha256,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_COMPRESSION_A_SCHEMA = (
    "fisher_graph.gemma3_structured_mlp_compression_a_candidate"
)
STRUCTURED_MLP_COMPRESSION_A_FORMAT_VERSION = 1
STRUCTURED_MLP_COMPRESSION_A_PREFLIGHT_SCHEMA = (
    "fisher_graph.gemma3_structured_mlp_compression_a_preflight"
)
DEFAULT_COMPRESSION_RIDGE = 1e-6
DEFAULT_LAYER_INDEX = 4
DEFAULT_MAX_LENGTH = 256
DEFAULT_TOKENIZATION_BATCH_SIZE = 4
REQUIRED_SOURCE_CORPUS_ID = "structured-strong-v6"

_PRIMARY = "structured_source_visibility"
_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_compression_a.payload.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_compression_a.report.v1\0"
)
_JSON_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_compression_a.json.v1\0"
)
_OUTER_FIELDS = {
    "schema",
    "format_version",
    "contains_source_model_weights",
    "contains_full_parent_executor_state",
    "contains_compressed_executor_weights",
    "contains_prompt_text",
    "contains_tokenizer_state",
    "contains_teacher_targets",
    "contains_fisher_taylor_scores",
    "scientific_status",
    "model",
    "protocol",
    "parent",
    "calibration_a",
    "pipeline",
    "resource_report",
    "executor",
    "scientific_payload_sha256",
    "report_sha256",
}


def _json_sha256(value: object, *, domain: bytes = _JSON_DOMAIN) -> str:
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


def _payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    return _json_sha256(report, domain=_REPORT_DOMAIN)


def _standard_thresholds() -> dict[str, float]:
    return {
        "nll_atol": DEFAULT_NLL_ATOL,
        "top1_min": DEFAULT_TOP1_MIN,
        "teacher_kl_max": DEFAULT_TEACHER_KL_MAX,
        "p90_abs_nll_max": DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
        "p10_top1_min": DEFAULT_PER_PROMPT_P10_TOP1_MIN,
        "block_delta_nrmse_max": DEFAULT_BLOCK_DELTA_NRMSE_MAX,
        "block_delta_cosine_min": DEFAULT_BLOCK_DELTA_COSINE_MIN,
        "branch_delta_nrmse_max": DEFAULT_BRANCH_DELTA_NRMSE_MAX,
        "branch_delta_cosine_min": DEFAULT_BRANCH_DELTA_COSINE_MIN,
        "native_parity_tolerance": DEFAULT_NATIVE_PARITY_TOLERANCE,
    }


def _standard_minima() -> dict[str, int]:
    return {
        "minimum_calibration_a_prompts": (
            DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        ),
        "minimum_heldout_prompts_per_role": (
            DEFAULT_MINIMUM_HELDOUT_PROMPTS
        ),
        "minimum_effective_fisher_rows": DEFAULT_MINIMUM_FISHER_ROWS,
        "minimum_train_supervised_tokens": (
            DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
        ),
        "minimum_heldout_supervised_tokens_per_role": (
            DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        ),
        "minimum_populated_length_buckets_per_tokenized_role": (
            DEFAULT_MINIMUM_LENGTH_BUCKETS
        ),
    }


@dataclass(frozen=True, slots=True)
class _StaticCorpus:
    prompts: object
    prompt_metadata: Mapping[str, object]
    family_metadata: Mapping[str, object]
    audit_binding: Mapping[str, object]
    source_corpus: Mapping[str, object]


def _load_structured_v6_corpus_preflight(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
) -> _StaticCorpus:
    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    audit_path = Path(corpus_audit_path)
    prompts = load_gemma3_prompt_splits(prompt_path)
    _require_prompt_protocol(
        prompts,
        minimum_calibration_a_prompts=(
            DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        ),
        minimum_heldout_prompts=DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    )
    families = load_prompt_family_manifest(
        family_path,
        prompts=prompts,
    )
    prompt_metadata = prompts.metadata()
    family_metadata = {
        **families.metadata(),
        **_format4_family_binding(prompts, families),
    }
    audit_binding = _corpus_audit_binding(
        audit_path,
        prompts=prompts,
        prompt_path=prompt_path,
        family_path=family_path,
    )
    if not isinstance(audit_binding, Mapping):
        raise ValueError("compression requires a bound v6 corpus audit")
    payload = audit_binding.get("payload")
    lexical = audit_binding.get("lexical_length_audit")
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] < 2
        or payload.get("corpus_id") != REQUIRED_SOURCE_CORPUS_ID
        or not isinstance(lexical, Mapping)
        or lexical.get("all_roles_cover_all_bands") is not True
    ):
        raise ValueError(
            "compression requires the breadth-validated structured-strong-v6 "
            "corpus"
        )
    prompt_hashes = prompt_metadata.get("per_prompt_sha256")
    family_hashes = family_metadata.get("per_prompt_family_sha256")
    if not isinstance(prompt_hashes, Mapping) or not isinstance(
        family_hashes,
        Mapping,
    ):
        raise ValueError("v6 corpus hash metadata is incomplete")
    roles = ("calibration_a", "calibration_b", "validation", "test")
    source_corpus = {
        "corpus_id": REQUIRED_SOURCE_CORPUS_ID,
        "prompt_status": PROMPT_STATUS,
        "family_status": FAMILY_STATUS,
        "counts": copy.deepcopy(prompt_metadata["counts"]),
        "prompt_sha256_by_role": {
            role: copy.deepcopy(prompt_hashes[role])
            for role in roles
        },
        "family_sha256_by_role": {
            role: copy.deepcopy(family_hashes[role])
            for role in roles
        },
        "ordered_prompt_sha256_by_role": copy.deepcopy(
            prompt_metadata["normalized_sha256"]
        ),
        "ordered_family_sha256_by_role": copy.deepcopy(
            family_metadata["ordered_hashed_family_sha256"]
        ),
        "corpus_audit_payload_sha256": audit_binding[
            "audit_payload_sha256"
        ],
        "prompt_fixture_file_sha256": audit_binding[
            "prompt_fixture_file_sha256"
        ],
        "family_manifest_file_sha256": audit_binding[
            "family_manifest_file_sha256"
        ],
    }
    return _StaticCorpus(
        prompts=prompts,
        prompt_metadata=prompt_metadata,
        family_metadata=family_metadata,
        audit_binding=audit_binding,
        source_corpus=source_corpus,
    )


def validate_structured_v6_corpus_preflight(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
) -> dict[str, object]:
    """Return hash-only v6 corpus bindings without loading a model."""

    corpus = _load_structured_v6_corpus_preflight(
        prompt_splits_path=prompt_splits_path,
        family_manifest_path=family_manifest_path,
        corpus_audit_path=corpus_audit_path,
    )
    return {
        "source_corpus": copy.deepcopy(dict(corpus.source_corpus)),
        "prompt_metadata": copy.deepcopy(dict(corpus.prompt_metadata)),
        "family_metadata": copy.deepcopy(dict(corpus.family_metadata)),
        "corpus_audit": copy.deepcopy(dict(corpus.audit_binding)),
    }


def _parent_binding_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(value)


def _authenticate_parent(
    loaded: Mapping[str, object],
    corpus: _StaticCorpus,
    *,
    model_id: str,
    revision: str,
    layer_index: int,
    max_length: int,
    tokenization_batch_size: int,
) -> tuple[
    StructuredTransformerLayerExecutor,
    Mapping[str, object],
    Mapping[str, object],
]:
    report = loaded.get("report")
    protocol = loaded.get("protocol")
    training = loaded.get("training")
    executors = loaded.get("executors")
    model = loaded.get("model")
    status = loaded.get("scientific_status")
    if (
        not isinstance(report, Mapping)
        or report.get("format_version") != 5
        or not isinstance(protocol, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(executors, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(status, Mapping)
        or protocol.get("fitting_method")
        != "activation_only_structured_operator_bootstrap"
        or protocol.get("operator_bootstrap_enabled") is not True
        or protocol.get("layer_index") != layer_index
        or protocol.get("maximum_tokenized_length") != max_length
        or protocol.get("tokenization_batch_size")
        != tokenization_batch_size
        or protocol.get("data_minima") != _standard_minima()
        or protocol.get("thresholds") != _standard_thresholds()
        or protocol.get("strong_data_minima_enforced") is not True
        or protocol.get("prompt_splits") != corpus.prompt_metadata
        or protocol.get("prompt_families") != corpus.family_metadata
        or protocol.get("corpus_audit") != corpus.audit_binding
        or model.get("model_id") != model_id
        or model.get("requested_revision") != revision
        or model.get("resolved_commit") != revision
        or status.get("calibration_b_passed") is not True
        or status.get("validation_passed") is not True
    ):
        raise ValueError(
            "format-5 parent does not authenticate the requested v6 A rung"
        )
    primary = executors.get(_PRIMARY)
    primary_training = training.get(_PRIMARY)
    if (
        not isinstance(primary, StructuredTransformerLayerExecutor)
        or not isinstance(primary_training, Mapping)
        or primary.config.transformer.feed_forward.intermediate_width
        != GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH
    ):
        raise ValueError("format-5 parent primary executor is invalid")
    return primary, primary_training, protocol


def _resource_report(
    construction: Mapping[str, object],
) -> dict[str, object]:
    return {
        "rung": copy.deepcopy(construction["rung"]),
        "parameters": copy.deepcopy(construction["parameters"]),
        "compute_per_valid_token": copy.deepcopy(
            construction["compute_per_valid_token"]
        ),
    }


def _safe_score_collection_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: copy.deepcopy(report[key])
        for key in (
            "schema",
            "format_version",
            "objective",
            "provenance",
            "accounting",
            "source_audit",
            "heldout_opened",
            "collection_sha256",
        )
    }


def _pipeline_binding(
    pipeline_report: Mapping[str, object],
    score_report: Mapping[str, object],
) -> dict[str, object]:
    selection = pipeline_report["selection"]
    final = pipeline_report["final_candidate"]
    terminal = pipeline_report["terminal_projection_refit"]
    parent_authentication = pipeline_report["parent_authentication"]
    assert isinstance(selection, Mapping)
    assert isinstance(final, Mapping)
    assert isinstance(terminal, Mapping)
    return {
        "schema": pipeline_report["schema"],
        "format_version": pipeline_report["format_version"],
        "pipeline_report_sha256": pipeline_report["report_sha256"],
        "parent_authentication": copy.deepcopy(parent_authentication),
        "score_collection": _safe_score_collection_report(score_report),
        "score_collection_sha256": score_report["collection_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "selection": {
            "algorithm": selection["algorithm"],
            "source_width": selection["source_width"],
            "retained_width": selection["retained_width"],
            "valid_rows": selection["valid_rows"],
            "input_batches_sha256": selection["input_batches_sha256"],
            "unit_scores_sha256": selection["unit_scores_sha256"],
            "retained_score_fraction": (
                selection["retained_score_fraction"]
            ),
        },
        "terminal_projection_refit": copy.deepcopy(dict(terminal)),
        "final_candidate": copy.deepcopy(dict(final)),
        "heldout_opened": False,
    }


def _build_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    scientific_payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema": STRUCTURED_MLP_COMPRESSION_A_SCHEMA,
        "format_version": STRUCTURED_MLP_COMPRESSION_A_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(payload["scientific_status"]),
        "model": copy.deepcopy(payload["model"]),
        "protocol": copy.deepcopy(payload["protocol"]),
        "parent": copy.deepcopy(payload["parent"]),
        "calibration_a": copy.deepcopy(payload["calibration_a"]),
        "pipeline": copy.deepcopy(payload["pipeline"]),
        "resource_report": copy.deepcopy(payload["resource_report"]),
        "artifact": {
            "tensor_file": tensor_file,
            "contains_compressed_executor_weights": True,
            "contains_source_model_weights": False,
            "contains_full_parent_executor_state": False,
            "contains_prompt_text": False,
            "contains_teacher_targets": False,
            "contains_fisher_taylor_scores": False,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }


def _write_preflight(
    path: Path,
    *,
    parent: Mapping[str, object],
    calibration_split_sha256: str,
    thresholds: Mapping[str, object],
    fidelity: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema": STRUCTURED_MLP_COMPRESSION_A_PREFLIGHT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "outcome": "rejected_on_calibration_a",
            "calibration_a_passed": False,
            "heldout_opened": False,
            "diagnostic_only": True,
            "scientific_compression_success": False,
        },
        "parent": copy.deepcopy(dict(parent)),
        "calibration_split_sha256": calibration_split_sha256,
        "thresholds": copy.deepcopy(dict(thresholds)),
        "fidelity": copy.deepcopy(dict(fidelity)),
        "main_artifact_written": False,
    }
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _validate_resource_report(
    value: object,
    executor: StructuredTransformerLayerExecutor,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "rung",
        "parameters",
        "compute_per_valid_token",
    }:
        raise ValueError("compression resource report fields are invalid")
    residual_width = executor.width
    source_width = GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH
    retained_width = GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
    removed_width = source_width - retained_width
    projection_bias = executor.config.transformer.feed_forward.projection_bias
    removed_parameters = removed_width * (
        3 * residual_width + 2 * int(projection_bias)
    )
    source_parameters = executor.learned_parameter_count + removed_parameters
    source_macs = 3 * residual_width * source_width
    compressed_macs = 3 * residual_width * retained_width
    source_bias_additions = (
        2 * source_width + residual_width if projection_bias else 0
    )
    compressed_bias_additions = (
        2 * retained_width + residual_width if projection_bias else 0
    )
    expected = {
        "rung": {
            "source_intermediate_width": source_width,
            "retained_intermediate_width": retained_width,
            "removed_intermediate_width": removed_width,
        },
        "parameters": {
            "source_full_layer": source_parameters,
            "compressed_full_layer": executor.learned_parameter_count,
            "removed_full_layer": removed_parameters,
            "expected_removed_from_mlp_slices": removed_parameters,
            "retained_ratio": (
                executor.learned_parameter_count / source_parameters
            ),
        },
        "compute_per_valid_token": {
            "scope": (
                "gate_up_down_linear_weight_matmuls_only; "
                "nonlinear_activation_flops_excluded_as "
                "implementation_dependent; attention_and_norm_unchanged"
            ),
            "macs": {
                "source": source_macs,
                "compressed": compressed_macs,
                "removed": source_macs - compressed_macs,
            },
            "flops_two_per_mac": {
                "source": 2 * source_macs,
                "compressed": 2 * compressed_macs,
                "removed": 2 * (source_macs - compressed_macs),
            },
            "bias_additions": {
                "source": source_bias_additions,
                "compressed": compressed_bias_additions,
                "removed": source_bias_additions - compressed_bias_additions,
            },
            "gate_up_multiplications": {
                "source": source_width,
                "compressed": retained_width,
                "removed": removed_width,
            },
            "nonlinear_activation_elements": {
                "source": source_width,
                "compressed": retained_width,
                "removed": removed_width,
            },
        },
    }
    if value != expected:
        raise ValueError("compression resource report does not recompute")


def _ordered_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _validate_source_corpus(value: object) -> None:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    fields = {
        "corpus_id",
        "prompt_status",
        "family_status",
        "counts",
        "prompt_sha256_by_role",
        "family_sha256_by_role",
        "ordered_prompt_sha256_by_role",
        "ordered_family_sha256_by_role",
        "corpus_audit_payload_sha256",
        "prompt_fixture_file_sha256",
        "family_manifest_file_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("compression source-corpus fields are invalid")
    counts = value["counts"]
    prompt_hashes = value["prompt_sha256_by_role"]
    family_hashes = value["family_sha256_by_role"]
    ordered_prompts = value["ordered_prompt_sha256_by_role"]
    ordered_families = value["ordered_family_sha256_by_role"]
    if (
        value["corpus_id"] != REQUIRED_SOURCE_CORPUS_ID
        or value["prompt_status"] != PROMPT_STATUS
        or value["family_status"] != FAMILY_STATUS
        or not isinstance(counts, Mapping)
        or set(counts) != set(roles)
        or not isinstance(prompt_hashes, Mapping)
        or set(prompt_hashes) != set(roles)
        or not isinstance(family_hashes, Mapping)
        or set(family_hashes) != set(roles)
        or not isinstance(ordered_prompts, Mapping)
        or set(ordered_prompts) != set(roles)
        or not isinstance(ordered_families, Mapping)
        or set(ordered_families) != set(roles)
        or any(
            not _is_sha256(value[field])
            for field in (
                "corpus_audit_payload_sha256",
                "prompt_fixture_file_sha256",
                "family_manifest_file_sha256",
            )
        )
    ):
        raise ValueError("compression source-corpus binding is invalid")
    prompt_sets = {}
    family_sets = {}
    for role in roles:
        expected_minimum = (
            DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
            if role == "calibration_a"
            else DEFAULT_MINIMUM_HELDOUT_PROMPTS
        )
        prompts = prompt_hashes[role]
        families = family_hashes[role]
        if (
            type(counts[role]) is not int
            or counts[role] < expected_minimum
            or not isinstance(prompts, list)
            or len(prompts) != counts[role]
            or len(set(prompts)) != len(prompts)
            or any(not _is_sha256(item) for item in prompts)
            or not isinstance(families, list)
            or len(families) != counts[role]
            or any(not _is_sha256(item) for item in families)
            or ordered_prompts[role] != _ordered_sha256(prompts)
            or ordered_families[role] != _ordered_sha256(families)
        ):
            raise ValueError("compression source-corpus hashes are invalid")
        prompt_sets[role] = set(prompts)
        family_sets[role] = set(families)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if (
                prompt_sets[left] & prompt_sets[right]
                or family_sets[left] & family_sets[right]
            ):
                raise ValueError(
                    "compression source-corpus roles are not disjoint"
                )


def _validate_fidelity(
    value: object,
    *,
    executor: StructuredTransformerLayerExecutor,
    thresholds: Mapping[str, object],
    expected_sequences: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("compression calibration-A fidelity is invalid")
    direct = value.get("direct")
    branches = value.get("branches")
    gates = value.get("gates")
    if (
        value.get("split") != "calibration_a"
        or value.get("source_layer_calls") != 0
        or value.get("primary_passed") is not True
        or value.get("executor_fingerprints")
        != {_PRIMARY: executor.execution_fingerprint()}
        or not isinstance(direct, Mapping)
        or not isinstance(branches, Mapping)
        or not isinstance(gates, Mapping)
        or set(direct) != {_PRIMARY}
        or set(branches) != {_PRIMARY}
        or set(gates) != {_PRIMARY}
    ):
        raise ValueError("compression calibration-A fidelity is invalid")
    direct_metrics = direct[_PRIMARY]
    branch_metrics = branches[_PRIMARY]
    if (
        not isinstance(direct_metrics, Mapping)
        or direct_metrics.get("sequences") != expected_sequences
        or not isinstance(branch_metrics, Mapping)
    ):
        raise ValueError("compression calibration-A metrics are invalid")
    expected_direct = _direct_gates(
        direct_metrics,
        block_delta_nrmse_max=float(
            thresholds["block_delta_nrmse_max"]
        ),
        block_delta_cosine_min=float(
            thresholds["block_delta_cosine_min"]
        ),
    )
    expected_branches = _branch_gates(
        branch_metrics,
        nrmse_max=float(thresholds["branch_delta_nrmse_max"]),
        cosine_min=float(thresholds["branch_delta_cosine_min"]),
    )
    expected_gates = {
        "direct": expected_direct,
        "branches": expected_branches,
        "passed": all(
            (*expected_direct.values(), *expected_branches.values())
        ),
    }
    if gates[_PRIMARY] != expected_gates or expected_gates["passed"] is not True:
        raise ValueError("compression calibration-A gates are invalid")


def _validate_tensor_locations(payload: Mapping[str, object]) -> None:
    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Tensor):
            if path[:2] != ("executor", "model_state_dict"):
                raise ValueError(
                    "compression artifact contains a tensor outside the "
                    "compressed executor state"
                )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, (*path, str(key)))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(payload, ())


def run_gemma3_structured_mlp_compression_a_experiment(
    *,
    parent_artifact_path: Path | str,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
    revision: str,
    output: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    ridge: float = DEFAULT_COMPRESSION_RIDGE,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Build a strict 2048 -> 1536 candidate using calibration A only."""

    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
    ):
        raise ValueError("revision must be an exact lowercase commit hash")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    if max_length != DEFAULT_MAX_LENGTH:
        raise ValueError("compression A rung requires max_length=256")
    if tokenization_batch_size != DEFAULT_TOKENIZATION_BATCH_SIZE:
        raise ValueError("compression A rung requires tokenization batch size 4")
    if (
        not isinstance(ridge, (int, float))
        or isinstance(ridge, bool)
        or not math.isfinite(float(ridge))
        or float(ridge) <= 0
    ):
        raise ValueError("ridge must be finite and positive")
    resolved_output = Path(output)
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report_path = resolved_output.with_suffix(".json")
    preflight_path = resolved_output.with_suffix(".calibration-a.json")
    if any(
        path.exists()
        for path in (resolved_output, report_path, preflight_path)
    ):
        raise FileExistsError(
            "refusing to overwrite a compression diagnostic artifact"
        )

    corpus = _load_structured_v6_corpus_preflight(
        prompt_splits_path=prompt_splits_path,
        family_manifest_path=family_manifest_path,
        corpus_audit_path=corpus_audit_path,
    )
    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "compression Fisher/Taylor collection requires CPU or CUDA"
        )
    loaded_parent = load_gemma3_structured_single_layer_artifact(
        parent_artifact_path,
        map_location=device,
    )
    parent_executor, parent_training, parent_protocol = (
        _authenticate_parent(
            loaded_parent,
            corpus,
            model_id=model_id,
            revision=revision,
            layer_index=layer_index,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
        )
    )
    parent_metadata = loaded_parent["metadata"]
    parent_model = loaded_parent["model"]
    assert isinstance(parent_metadata, Mapping)
    assert isinstance(parent_model, Mapping)
    parent_binding = {
        "artifact_tensor_file_sha256": parent_metadata[
            "tensor_file_sha256"
        ],
        "artifact_scientific_payload_sha256": parent_metadata[
            "scientific_payload_sha256"
        ],
        "artifact_report_sha256": parent_metadata["report_sha256"],
        "artifact_format_version": 5,
        "model_resolved_commit": parent_model["resolved_commit"],
        "layer_index": layer_index,
        "layer_id": parent_training["bootstrap"]["layer_id"],
        "calibration_a_split_sha256": parent_training["bootstrap"][
            "calibration_split_sha256"
        ],
        "primary_execution_fingerprint": (
            parent_executor.execution_fingerprint()
        ),
        "primary_coefficient_sha256": (
            structured_operator_coefficient_sha256(parent_executor)
        ),
        "primary_training_binding_sha256": (
            _parent_binding_sha256(parent_training)
        ),
    }

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    if (
        model_metadata.get("resolved_commit") != revision
        or model_metadata.get("config_sha256")
        != parent_model.get("config_sha256")
    ):
        raise ValueError("local pinned model does not match the parent")
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(layer_index, layer_index)
    layer_id = plan.layer_ids[0]
    if layer_id != parent_binding["layer_id"]:
        raise ValueError("local layer does not match the parent layer")

    prompts = corpus.prompts
    train_batches, train_stream = _materialize_split(
        tokenizer,
        prompts.calibration_a,  # type: ignore[attr-defined]
        split_name="calibration_a",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    train_contract = _tokenized_stream_contract(
        train_stream,
        split_name="calibration_a",
        minimum_supervised_tokens=(
            DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
        ),
        minimum_length_buckets=DEFAULT_MINIMUM_LENGTH_BUCKETS,
    )
    parent_streams = parent_protocol["tokenized_splits"]
    assert isinstance(parent_streams, Mapping)
    parent_a_stream = parent_streams.get("calibration_a")
    calibration_split_sha256 = train_stream.get("serialized_sha256")
    if (
        train_stream != parent_a_stream
        or calibration_split_sha256
        != parent_binding["calibration_a_split_sha256"]
        or not _is_sha256(calibration_split_sha256)
    ):
        raise ValueError(
            "compression calibration-A token stream does not exactly match "
            "the parent"
        )
    training = collect_structured_training_batches(
        adapter,
        train_batches,
        layer_id=layer_id,
        positions_per_sequence=(
            DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
        ),
    )
    _require_complete_middle_layer_demand(
        adapter,
        training,  # type: ignore[arg-type]
    )
    score_batches, score_report = (
        collect_gemma_mlp_fisher_taylor_batches(
            adapter,
            train_batches,
            layer_id=layer_id,
            calibration_split_sha256=calibration_split_sha256,
        )
    )
    if (
        score_report["accounting"]["valid_rows"]
        < DEFAULT_MINIMUM_FISHER_ROWS
    ):
        raise ValueError(
            "compression Fisher/Taylor rows are below the standard minimum"
        )
    guard.assert_unchanged()

    candidate = build_gemma_mlp_first_rung_candidate(
        parent_executor,
        tuple(item.targets for item in training),
        score_batches,
        calibration_split_sha256=calibration_split_sha256,
        parent_artifact_format_version=5,
        parent_training_binding=parent_training,
        ridge=float(ridge),
    )
    restored = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            candidate.artifact_state,
            map_location=device,
        )
    )
    if (
        restored.execution_fingerprint()
        != candidate.executor.execution_fingerprint()
    ):
        raise RuntimeError("compressed candidate strict roundtrip drifted")
    thresholds = _standard_thresholds()
    fidelity = evaluate_calibration_a_fidelity(
        training,
        candidates={_PRIMARY: restored},
        thresholds=thresholds,
    )
    if fidelity["primary_passed"] is not True:
        _write_preflight(
            preflight_path,
            parent=parent_binding,
            calibration_split_sha256=calibration_split_sha256,
            thresholds=thresholds,
            fidelity=fidelity,
        )
        raise RuntimeError(
            "compressed candidate failed calibration-A fidelity; no main "
            f"artifact was written; see {preflight_path}"
        )
    guard.assert_unchanged()

    construction = candidate.report["construction"]
    assert isinstance(construction, Mapping)
    resource_report = _resource_report(construction)
    pipeline_binding = _pipeline_binding(
        candidate.report,
        score_report,
    )
    protocol = {
        "selection_split": "calibration_a",
        "heldout_roles_statically_authenticated_only": (
            ("calibration_b", "validation", "test")
        ),
        "heldout_tokenized": False,
        "heldout_evaluated": False,
        "heldout_ledger_created": False,
        "source_corpus": copy.deepcopy(dict(corpus.source_corpus)),
        "prompt_metadata_sha256": _json_sha256(
            corpus.prompt_metadata
        ),
        "family_metadata_sha256": _json_sha256(
            corpus.family_metadata
        ),
        "corpus_audit_payload_sha256": corpus.audit_binding[
            "audit_payload_sha256"
        ],
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "layer_index": layer_index,
        "layer_id": layer_id,
        "data_minima": _standard_minima(),
        "thresholds": thresholds,
        "ridge": float(ridge),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "library_versions": _library_versions(),
    }
    calibration_a = {
        "tokenized_stream": copy.deepcopy(train_stream),
        "tokenized_stream_contract": copy.deepcopy(train_contract),
        "structured_training_batches": len(training),
        "score_collection_valid_rows": score_report["accounting"][
            "valid_rows"
        ],
        "fidelity": copy.deepcopy(fidelity),
    }
    scientific_status = {
        "outcome": "calibration_a_compression_candidate_built",
        "calibration_a_passed": True,
        "candidate_strict_roundtrip_verified": True,
        "heldout_opened": False,
        "heldout_passed": False,
        "diagnostic_only": True,
        "compression_candidate_only": True,
        "scientific_compression_success": False,
        "parameter_reduction_measured": True,
        "analytic_mlp_mac_reduction_measured": True,
        "latency_or_kernel_speed_claim": False,
    }
    payload: dict[str, object] = {
        "schema": STRUCTURED_MLP_COMPRESSION_A_SCHEMA,
        "format_version": STRUCTURED_MLP_COMPRESSION_A_FORMAT_VERSION,
        "contains_source_model_weights": False,
        "contains_full_parent_executor_state": False,
        "contains_compressed_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_fisher_taylor_scores": False,
        "scientific_status": scientific_status,
        "model": model_metadata,
        "protocol": protocol,
        "parent": parent_binding,
        "calibration_a": calibration_a,
        "pipeline": pipeline_binding,
        "resource_report": resource_report,
        "executor": candidate.artifact_state,
    }
    _validate_tensor_locations(payload)
    scientific_digest = _payload_sha256(payload)
    report = _build_report(
        payload,
        tensor_file=resolved_output.name,
        scientific_payload_sha256=scientific_digest,
    )
    report_digest = _report_sha256(report)
    artifact = {
        **payload,
        "scientific_payload_sha256": scientific_digest,
        "report_sha256": report_digest,
    }
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, resolved_output)
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_gemma3_structured_mlp_compression_a_artifact(
        resolved_output,
        map_location=device,
    )


def load_gemma3_structured_mlp_compression_a_artifact(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly restore the A-only candidate and its resource binding."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _OUTER_FIELDS
        or raw.get("schema") != STRUCTURED_MLP_COMPRESSION_A_SCHEMA
        or raw.get("format_version")
        != STRUCTURED_MLP_COMPRESSION_A_FORMAT_VERSION
        or raw.get("contains_source_model_weights") is not False
        or raw.get("contains_full_parent_executor_state") is not False
        or raw.get("contains_compressed_executor_weights") is not True
        or raw.get("contains_prompt_text") is not False
        or raw.get("contains_tokenizer_state") is not False
        or raw.get("contains_teacher_targets") is not False
        or raw.get("contains_fisher_taylor_scores") is not False
        or not _is_sha256(raw.get("scientific_payload_sha256"))
        or not _is_sha256(raw.get("report_sha256"))
    ):
        raise ValueError("compression A artifact header is invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    _validate_tensor_locations(payload)
    if _payload_sha256(payload) != raw["scientific_payload_sha256"]:
        raise ValueError("compression A scientific payload digest mismatch")
    executor = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            raw["executor"],  # type: ignore[arg-type]
            map_location=map_location,
        )
    )
    if (
        executor.config.transformer.feed_forward.intermediate_width
        != GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
        or executor.owns_source_model_weights
    ):
        raise ValueError("compressed executor schema is invalid")
    protocol = raw["protocol"]
    parent = raw["parent"]
    calibration_a = raw["calibration_a"]
    pipeline = raw["pipeline"]
    status = raw["scientific_status"]
    model_binding = raw["model"]
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(parent, Mapping)
        or not isinstance(calibration_a, Mapping)
        or not isinstance(pipeline, Mapping)
        or not isinstance(status, Mapping)
        or not isinstance(model_binding, Mapping)
    ):
        raise ValueError("compression A artifact bindings are invalid")
    tokenized_stream = calibration_a.get("tokenized_stream")
    if not isinstance(tokenized_stream, Mapping):
        raise ValueError("compression A tokenized stream is invalid")
    _validate_source_corpus(protocol.get("source_corpus"))
    source_corpus = protocol["source_corpus"]
    assert isinstance(source_corpus, Mapping)
    source_prompt_hashes = source_corpus["prompt_sha256_by_role"]
    assert isinstance(source_prompt_hashes, Mapping)
    if (
        protocol.get("corpus_audit_payload_sha256")
        != source_corpus["corpus_audit_payload_sha256"]
        or protocol.get("data_minima") != _standard_minima()
        or protocol.get("thresholds") != _standard_thresholds()
        or protocol.get("layer_index") != parent.get("layer_index")
        or protocol.get("layer_id") != parent.get("layer_id")
        or parent.get("model_resolved_commit")
        != model_binding.get("resolved_commit")
        or parent.get("calibration_a_split_sha256")
        != tokenized_stream.get("serialized_sha256")
        or tokenized_stream.get("source_prompt_sha256")
        != source_prompt_hashes["calibration_a"]
    ):
        raise ValueError("compression A parent/corpus binding is invalid")
    parent_authentication = pipeline.get("parent_authentication")
    final_candidate = pipeline.get("final_candidate")
    score_collection = pipeline.get("score_collection")
    selection_binding = pipeline.get("selection")
    protocol_fields = {
        "selection_split",
        "heldout_roles_statically_authenticated_only",
        "heldout_tokenized",
        "heldout_evaluated",
        "heldout_ledger_created",
        "source_corpus",
        "prompt_metadata_sha256",
        "family_metadata_sha256",
        "corpus_audit_payload_sha256",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "layer_index",
        "layer_id",
        "data_minima",
        "thresholds",
        "ridge",
        "tokenizer",
        "library_versions",
    }
    parent_fields = {
        "artifact_tensor_file_sha256",
        "artifact_scientific_payload_sha256",
        "artifact_report_sha256",
        "artifact_format_version",
        "model_resolved_commit",
        "layer_index",
        "layer_id",
        "calibration_a_split_sha256",
        "primary_execution_fingerprint",
        "primary_coefficient_sha256",
        "primary_training_binding_sha256",
    }
    pipeline_fields = {
        "schema",
        "format_version",
        "pipeline_report_sha256",
        "parent_authentication",
        "score_collection",
        "score_collection_sha256",
        "selection_sha256",
        "selection",
        "terminal_projection_refit",
        "final_candidate",
        "heldout_opened",
    }
    status_fields = {
        "outcome",
        "calibration_a_passed",
        "candidate_strict_roundtrip_verified",
        "heldout_opened",
        "heldout_passed",
        "diagnostic_only",
        "compression_candidate_only",
        "scientific_compression_success",
        "parameter_reduction_measured",
        "analytic_mlp_mac_reduction_measured",
        "latency_or_kernel_speed_claim",
    }
    if (
        not isinstance(parent_authentication, Mapping)
        or not isinstance(final_candidate, Mapping)
        or not isinstance(score_collection, Mapping)
        or not isinstance(selection_binding, Mapping)
        or set(protocol) != protocol_fields
        or set(parent) != parent_fields
        or set(pipeline) != pipeline_fields
        or set(status) != status_fields
        or protocol.get("selection_split") != "calibration_a"
        or protocol.get("heldout_roles_statically_authenticated_only")
        != ("calibration_b", "validation", "test")
        or protocol.get("heldout_tokenized") is not False
        or protocol.get("heldout_evaluated") is not False
        or protocol.get("heldout_ledger_created") is not False
        or parent.get("artifact_format_version") != 5
        or parent.get("primary_execution_fingerprint")
        != parent_authentication.get("execution_fingerprint")
        or tokenized_stream.get("split") != "calibration_a"
        or pipeline.get("schema")
        != STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA
        or pipeline.get("format_version")
        != STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION
        or pipeline.get("heldout_opened") is not False
        or not _is_sha256(pipeline.get("pipeline_report_sha256"))
        or not _is_sha256(pipeline.get("score_collection_sha256"))
        or not _is_sha256(pipeline.get("selection_sha256"))
        or score_collection.get("collection_sha256")
        != pipeline.get("score_collection_sha256")
        or score_collection.get("heldout_opened") is not False
        or selection_binding.get("source_width")
        != GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH
        or selection_binding.get("retained_width")
        != GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
        or final_candidate.get("execution_fingerprint")
        != executor.execution_fingerprint()
        or status.get("outcome")
        != "calibration_a_compression_candidate_built"
        or status.get("calibration_a_passed") is not True
        or status.get("candidate_strict_roundtrip_verified") is not True
        or status.get("heldout_opened") is not False
        or status.get("heldout_passed") is not False
        or status.get("diagnostic_only") is not True
        or status.get("compression_candidate_only") is not True
        or status.get("scientific_compression_success") is not False
        or status.get("parameter_reduction_measured") is not True
        or status.get("analytic_mlp_mac_reduction_measured") is not True
        or status.get("latency_or_kernel_speed_claim") is not False
    ):
        raise ValueError("compression A artifact bindings are invalid")
    thresholds = protocol.get("thresholds")
    stream = calibration_a.get("tokenized_stream")
    if not isinstance(thresholds, Mapping) or not isinstance(stream, Mapping):
        raise ValueError("compression A protocol is invalid")
    sequences = stream.get("sequences")
    if type(sequences) is not int or sequences <= 0:
        raise ValueError("compression A stream sequence count is invalid")
    _validate_fidelity(
        calibration_a.get("fidelity"),
        executor=executor,
        thresholds=thresholds,
        expected_sequences=sequences,
    )
    _validate_resource_report(raw["resource_report"], executor)
    report_path = source.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = json.loads(
        json.dumps(
            _build_report(
                payload,
                tensor_file=source.name,
                scientific_payload_sha256=(
                    raw["scientific_payload_sha256"]
                ),
            ),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if (
        report != expected_report
        or _report_sha256(report) != raw["report_sha256"]
    ):
        raise ValueError("compression A JSON report is invalid")
    return {
        "model": copy.deepcopy(raw["model"]),
        "protocol": copy.deepcopy(protocol),
        "parent": copy.deepcopy(parent),
        "executor": executor,
        "calibration_a": copy.deepcopy(calibration_a),
        "pipeline": copy.deepcopy(pipeline),
        "resource_report": copy.deepcopy(raw["resource_report"]),
        "scientific_status": copy.deepcopy(status),
        "metadata": {
            "scientific_payload_sha256": raw[
                "scientific_payload_sha256"
            ],
            "report_sha256": raw["report_sha256"],
            "tensor_file_sha256": _file_sha256(source),
        },
        "report": copy.deepcopy(report),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the strict 2048-to-1536 structured MLP candidate using "
            "calibration A only."
        )
    )
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--prompt-splits", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=DEFAULT_COMPRESSION_RIDGE,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = run_gemma3_structured_mlp_compression_a_experiment(
        parent_artifact_path=args.parent_artifact,
        prompt_splits_path=args.prompt_splits,
        family_manifest_path=args.family_manifest,
        corpus_audit_path=args.corpus_audit,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        layer_index=args.layer_index,
        max_length=args.max_length,
        tokenization_batch_size=args.tokenization_batch_size,
        ridge=args.ridge,
        device_name=args.device,
        dtype=args.dtype,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "scientific_status": loaded["scientific_status"],
                "resource_report": loaded["resource_report"],
                "metadata": loaded["metadata"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "STRUCTURED_MLP_COMPRESSION_A_FORMAT_VERSION",
    "STRUCTURED_MLP_COMPRESSION_A_SCHEMA",
    "build_parser",
    "load_gemma3_structured_mlp_compression_a_artifact",
    "main",
    "run_gemma3_structured_mlp_compression_a_experiment",
    "validate_structured_v6_corpus_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
