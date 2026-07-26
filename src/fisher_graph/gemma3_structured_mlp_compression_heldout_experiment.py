"""Fresh heldout evaluation for one structured Gemma MLP compression rung.

This module deliberately separates compression construction on calibration A
from every confirmatory observation.  It consumes a strict source-free
candidate artifact, performs all static corpus and artifact checks first,
claims a fresh order-insensitive calibration-B identity exactly once, and
only then tokenizes or evaluates calibration B.  Validation remains sealed
unless B passes; test remains sealed unconditionally.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_experiment import (
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_width_single_layer_experiment import (
    DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS,
    DEFAULT_MINIMUM_LENGTH_BUCKETS,
    _direct_gates,
    _require_prompt_protocol,
    _tokenized_stream_contract,
    load_prompt_family_manifest,
)
from .gemma3_gated_executor_experiment import (
    _materialize_split,
    _source_block_macs,
    _source_block_static,
)
from .gemma3_rotated_span_executor_experiment import (
    _behavior_gates,
)
from .gemma3_stability_experiment import (
    _library_versions,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .gemma3_structured_single_layer_experiment import (
    _assert_tokenized_content_disjointness,
    _branch_gates,
    _corpus_audit_binding,
    _format4_family_binding,
    evaluate_structured_candidates,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_COMPRESSION_HELDOUT_SCHEMA = (
    "fisher_graph.gemma3_structured_mlp_compression_heldout"
)
STRUCTURED_MLP_COMPRESSION_HELDOUT_FORMAT_VERSION = 1
STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE = (
    "structured-mlp-compression-calibration-b"
)
STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE = (
    "structured_mlp_compressed_1536"
)
STRUCTURED_MLP_COMPRESSION_HELDOUT_CORPUS_ID = "structured-strong-v7"

_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_compression_heldout.payload.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_compression_heldout.report.v1\0"
)
_CLAIM_DOMAIN = (
    b"fisher_graph.structured_mlp_compression.heldout_claim.v1\0"
)
_CLAIM_IDENTITY_DOMAIN = (
    b"fisher_graph.structured_mlp_compression.heldout_identity.v1\0"
)
_FAMILY_DOMAIN = (
    b"fisher_graph.structured_mlp_compression.heldout_family.v1\0"
)
_THRESHOLD_FIELDS = {
    "nll_atol",
    "top1_min",
    "teacher_kl_max",
    "p90_abs_nll_max",
    "p10_top1_min",
    "block_delta_nrmse_max",
    "block_delta_cosine_min",
    "branch_delta_nrmse_max",
    "branch_delta_cosine_min",
    "native_parity_tolerance",
}
_ROLES = ("calibration_a", "calibration_b", "validation", "test")
_OUTER_FIELDS = {
    "schema",
    "format_version",
    "contains_model_weights",
    "contains_executor_weights",
    "contains_prompt_text",
    "contains_tokenizer_state",
    "contains_teacher_targets",
    "contains_fisher_scores",
    "candidate",
    "model",
    "protocol",
    "calibration_b",
    "validation",
    "scientific_status",
    "scientific_payload_sha256",
    "report_sha256",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mapping(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _load_compression_a_candidate(
    path: Path | str,
    *,
    map_location: torch.device | str,
) -> Mapping[str, object]:
    # Kept behind a narrow adapter so tests can assert ordering and the
    # heldout runner does not replicate candidate-artifact validation.
    from .gemma3_structured_mlp_compression_a_experiment import (
        load_gemma3_structured_mlp_compression_a_artifact,
    )

    return load_gemma3_structured_mlp_compression_a_artifact(
        path,
        map_location=map_location,
    )


def _candidate_protocol_value(
    protocol: Mapping[str, object],
    *names: str,
) -> object:
    for name in names:
        if name in protocol:
            return protocol[name]
    raise ValueError(
        f"candidate protocol lacks required field {names[0]!r}"
    )


def _candidate_preflight(
    loaded: Mapping[str, object],
    *,
    candidate_artifact_path: Path,
) -> dict[str, object]:
    required = {
        "model",
        "protocol",
        "parent",
        "executor",
        "calibration_a",
        "pipeline",
        "resource_report",
        "scientific_status",
        "metadata",
        "report",
    }
    if set(loaded) != required:
        raise ValueError("compression-A loader result fields are invalid")
    executor = loaded["executor"]
    model = loaded["model"]
    protocol = loaded["protocol"]
    pipeline = loaded["pipeline"]
    resource = loaded["resource_report"]
    status = loaded["scientific_status"]
    metadata = loaded["metadata"]
    if (
        not isinstance(executor, StructuredTransformerLayerExecutor)
        or executor.owns_source_model_weights
        or not isinstance(model, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(pipeline, Mapping)
        or not isinstance(resource, Mapping)
        or not isinstance(status, Mapping)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("compression-A candidate loader result is invalid")
    if (
        status.get("calibration_a_passed") is not True
        or status.get("heldout_opened") is not False
        or status.get("scientific_compression_success") is not False
    ):
        raise ValueError(
            "compression-A artifact is not an unopened candidate"
        )
    source_corpus = protocol.get("source_corpus")
    if not isinstance(source_corpus, Mapping):
        raise ValueError(
            "compression-A candidate lacks source-corpus hash bindings"
        )
    prompt_hashes = source_corpus.get("prompt_sha256_by_role")
    family_hashes = source_corpus.get("family_sha256_by_role")
    if (
        not isinstance(prompt_hashes, Mapping)
        or set(prompt_hashes) != set(_ROLES)
        or not isinstance(family_hashes, Mapping)
        or set(family_hashes) != set(_ROLES)
    ):
        raise ValueError(
            "compression-A source-corpus role bindings are invalid"
        )
    forbidden_prompts: set[str] = set()
    forbidden_families: set[str] = set()
    for role in _ROLES:
        role_prompts = prompt_hashes[role]
        role_families = family_hashes[role]
        if (
            not isinstance(role_prompts, (tuple, list))
            or not role_prompts
            or any(not _is_sha256(item) for item in role_prompts)
            or not isinstance(role_families, (tuple, list))
            or not role_families
            or any(not _is_sha256(item) for item in role_families)
        ):
            raise ValueError(
                "compression-A source-corpus hashes are invalid"
            )
        forbidden_prompts.update(role_prompts)
        forbidden_families.update(role_families)
    thresholds = _candidate_protocol_value(
        protocol,
        "thresholds",
        "fidelity_thresholds",
    )
    if not isinstance(thresholds, Mapping):
        raise ValueError("candidate thresholds are invalid")
    normalized_thresholds = _validated_thresholds(thresholds)
    layer_index = _candidate_protocol_value(
        protocol,
        "layer_index",
    )
    max_length = _candidate_protocol_value(
        protocol,
        "maximum_tokenized_length",
        "max_length",
    )
    tokenization_batch_size = _candidate_protocol_value(
        protocol,
        "tokenization_batch_size",
    )
    if (
        type(layer_index) is not int
        or layer_index < 0
        or type(max_length) is not int
        or max_length <= 1
        or type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("candidate evaluation dimensions are invalid")
    resolved_commit = model.get("resolved_commit")
    model_id = model.get("model_id")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(resolved_commit, str)
        or not resolved_commit
    ):
        raise ValueError(
            "candidate model lacks an immutable resolved revision"
        )
    final_candidate = pipeline.get("final_candidate")
    if not isinstance(final_candidate, Mapping):
        raise ValueError("candidate pipeline final binding is invalid")
    final_fingerprint = final_candidate.get("execution_fingerprint")
    final_state_sha256 = final_candidate.get("artifact_state_sha256")
    pipeline_report_sha256 = pipeline.get("pipeline_report_sha256")
    if (
        final_fingerprint != executor.execution_fingerprint()
        or not _is_sha256(final_fingerprint)
        or not _is_sha256(final_state_sha256)
        or not _is_sha256(pipeline_report_sha256)
        or not _is_sha256(pipeline.get("selection_sha256"))
        or not _is_sha256(pipeline.get("score_collection_sha256"))
    ):
        raise ValueError("candidate pipeline digest binding is invalid")
    candidate_payload_sha256 = metadata.get(
        "scientific_payload_sha256"
    )
    candidate_report_sha256 = metadata.get("report_sha256")
    candidate_file_sha256 = _file_sha256(candidate_artifact_path)
    if (
        not _is_sha256(candidate_payload_sha256)
        or not _is_sha256(candidate_report_sha256)
    ):
        raise ValueError("candidate artifact metadata digests are invalid")
    return {
        "executor": executor,
        "model": copy.deepcopy(dict(model)),
        "protocol": protocol,
        "source_corpus": source_corpus,
        "forbidden_prompt_sha256": frozenset(forbidden_prompts),
        "forbidden_family_sha256": frozenset(forbidden_families),
        "thresholds": normalized_thresholds,
        "layer_index": layer_index,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "model_id": model_id,
        "resolved_commit": resolved_commit,
        "resource_report": copy.deepcopy(dict(resource)),
        "binding": {
            "artifact_file_sha256": candidate_file_sha256,
            "scientific_payload_sha256": candidate_payload_sha256,
            "report_sha256": candidate_report_sha256,
            "pipeline_report_sha256": pipeline_report_sha256,
            "selection_sha256": pipeline.get("selection_sha256"),
            "score_collection_sha256": pipeline.get(
                "score_collection_sha256"
            ),
            "execution_fingerprint": final_fingerprint,
            "state_sha256": final_state_sha256,
            "strict_candidate_loader_used": True,
            "source_free": True,
        },
    }


def _validated_thresholds(
    value: Mapping[str, object],
) -> dict[str, float]:
    if set(value) != _THRESHOLD_FIELDS:
        raise ValueError("heldout threshold fields are invalid")
    result = {}
    for name in sorted(_THRESHOLD_FIELDS):
        item = value[name]
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"heldout threshold {name!r} is invalid")
        result[name] = float(item)
    if (
        result["nll_atol"] < 0
        or not 0 <= result["top1_min"] <= 1
        or result["teacher_kl_max"] < 0
        or result["p90_abs_nll_max"] < 0
        or not 0 <= result["p10_top1_min"] <= 1
        or result["block_delta_nrmse_max"] < 0
        or not -1 <= result["block_delta_cosine_min"] <= 1
        or result["branch_delta_nrmse_max"] < 0
        or not -1 <= result["branch_delta_cosine_min"] <= 1
        or result["native_parity_tolerance"] < 0
    ):
        raise ValueError("heldout threshold bounds are invalid")
    return result


def _audit_prior_file_hashes(
    payload: Mapping[str, object],
    field: str,
) -> set[str]:
    values = payload.get(field)
    if not isinstance(values, list):
        raise ValueError(f"v7 audit {field!r} is invalid")
    hashes = set()
    for item in values:
        if (
            not isinstance(item, Mapping)
            or not _is_sha256(item.get("file_sha256"))
        ):
            raise ValueError(f"v7 audit {field!r} is invalid")
        hashes.add(item["file_sha256"])  # type: ignore[arg-type]
    return hashes


def _source_file_digest(
    candidate_protocol: Mapping[str, object],
    *names: str,
) -> str:
    for name in names:
        value = candidate_protocol.get(name)
        if _is_sha256(value):
            return value  # type: ignore[return-value]
    source_corpus = candidate_protocol.get("source_corpus")
    if isinstance(source_corpus, Mapping):
        for name in names:
            value = source_corpus.get(name)
            if _is_sha256(value):
                return value  # type: ignore[return-value]
    raise ValueError(
        f"candidate lacks source file digest {names[0]!r}"
    )


def _v7_corpus_preflight(
    *,
    prompt_splits_path: Path,
    family_manifest_path: Path,
    corpus_audit_path: Path,
    candidate: Mapping[str, object],
    minimum_calibration_a_prompts: int,
    minimum_heldout_prompts: int,
) -> dict[str, object]:
    prompts = load_gemma3_prompt_splits(prompt_splits_path)
    _require_prompt_protocol(
        prompts,
        minimum_calibration_a_prompts=minimum_calibration_a_prompts,
        minimum_heldout_prompts=minimum_heldout_prompts,
    )
    families = load_prompt_family_manifest(
        family_manifest_path,
        prompts=prompts,
    )
    audit = _corpus_audit_binding(
        corpus_audit_path,
        prompts=prompts,
        prompt_path=prompt_splits_path,
        family_path=family_manifest_path,
    )
    if audit is None:
        raise ValueError("compression heldout requires a corpus audit")
    payload = audit.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("v7 corpus audit payload is invalid")
    if (
        payload.get("format_version") != 3
        or payload.get("corpus_id")
        != STRUCTURED_MLP_COMPRESSION_HELDOUT_CORPUS_ID
        or payload.get("heldout_splits_evaluated") is not False
        or payload.get("heldout_splits_tokenized") is not False
        or payload.get("heldout_splits_unevaluated") is not True
        or payload.get("heldout_splits_untokenized") is not True
        or payload.get("calibration_b_model_evaluated") is not False
        or payload.get("validation_model_evaluated") is not False
        or payload.get("test_model_evaluated") is not False
        or payload.get("prior_local_exact_prompt_overlap_count") != 0
        or payload.get("prior_raw_prompt_overlap_count") != 0
        or payload.get("prior_normalized_prompt_overlap_count") != 0
        or payload.get("prior_domain_slug_overlap_count") != 0
        or payload.get("prior_template_marker_overlap_count") != 0
        or payload.get("prior_template_signature_overlap_count") != 0
        or payload.get("prior_5_6_7_8_word_ngram_overlap_count") != 0
        or payload.get("prior_generator_imported_or_reused") is not False
        or payload.get("corpus_frozen_before_model_load") is not True
        or payload.get("tokenizer_or_model_accessed") is not False
    ):
        raise ValueError(
            "v7 corpus is not a fresh unopened heldout fixture"
        )
    prompt_metadata = prompts.metadata()
    family_metadata = {
        **families.metadata(),
        **_format4_family_binding(prompts, families),
    }
    prompt_hashes = prompt_metadata["per_prompt_sha256"]
    family_hashes = family_metadata["per_prompt_family_sha256"]
    assert isinstance(prompt_hashes, Mapping)
    assert isinstance(family_hashes, Mapping)
    all_prompt_hashes = {
        value
        for role in _ROLES
        for value in prompt_hashes[role]  # type: ignore[index]
    }
    all_family_hashes = {
        value
        for role in _ROLES
        for value in family_hashes[role]  # type: ignore[index]
    }
    forbidden_prompts = candidate["forbidden_prompt_sha256"]
    forbidden_families = candidate["forbidden_family_sha256"]
    if (
        not isinstance(forbidden_prompts, frozenset)
        or not isinstance(forbidden_families, frozenset)
        or all_prompt_hashes & forbidden_prompts
        or all_family_hashes & forbidden_families
    ):
        raise ValueError(
            "v7 corpus reuses compression-A or parent corpus content"
        )
    candidate_protocol = candidate["protocol"]
    if not isinstance(candidate_protocol, Mapping):
        raise ValueError("candidate protocol is invalid")
    source_prompt_file_sha256 = _source_file_digest(
        candidate_protocol,
        "prompt_fixture_file_sha256",
        "prompt_file_sha256",
    )
    source_family_file_sha256 = _source_file_digest(
        candidate_protocol,
        "family_manifest_file_sha256",
        "family_file_sha256",
    )
    prior_prompt_files = _audit_prior_file_hashes(
        payload,
        "prior_prompt_files",
    )
    prior_family_files = _audit_prior_file_hashes(
        payload,
        "prior_family_files",
    )
    if (
        source_prompt_file_sha256 not in prior_prompt_files
        or source_family_file_sha256 not in prior_family_files
    ):
        raise ValueError(
            "v7 audit did not include the compression-A source corpus"
        )
    return {
        "prompts": prompts,
        "prompt_metadata": prompt_metadata,
        "family_metadata": family_metadata,
        "audit": audit,
        "binding": {
            "corpus_id": STRUCTURED_MLP_COMPRESSION_HELDOUT_CORPUS_ID,
            "prompt_fixture_file_sha256": _file_sha256(
                prompt_splits_path
            ),
            "family_manifest_file_sha256": _file_sha256(
                family_manifest_path
            ),
            "corpus_audit_file_sha256": _file_sha256(
                corpus_audit_path
            ),
            "corpus_audit_payload_sha256": audit[
                "audit_payload_sha256"
            ],
            "prompt_splits": prompt_metadata,
            "prompt_families": family_metadata,
            "source_forbidden_prompt_count": len(forbidden_prompts),
            "source_forbidden_family_count": len(forbidden_families),
            "prompt_overlap_with_source_corpus": 0,
            "family_overlap_with_source_corpus": 0,
            "source_prompt_file_in_prior_scan": True,
            "source_family_file_in_prior_scan": True,
            "freshness_verified_before_model_load_or_tokenization": True,
        },
    }


def _claim_identity(prompt_hashes: Sequence[str]) -> str:
    values = tuple(prompt_hashes)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not _is_sha256(value) for value in values)
    ):
        raise ValueError("heldout calibration-B prompt hashes are invalid")
    return _sha256(
        {
            "role": "calibration_b",
            "count": len(values),
            "sorted_prompt_sha256": sorted(values),
        },
        domain=_CLAIM_IDENTITY_DOMAIN,
    )


def _claim_path(
    ledger_dir: Path | str,
    prompt_hashes: Sequence[str],
) -> Path:
    return (
        Path(ledger_dir)
        / STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
        / f"{_claim_identity(prompt_hashes)}.json"
    )


def _exclusive_heldout_claim(
    path: Path,
    *,
    prompt_hashes: Sequence[str],
    family_hashes: Sequence[str],
    candidate_binding: Mapping[str, object],
    v7_binding: Mapping[str, object],
    model_resolved_commit: str,
    layer_id: str,
    thresholds: Mapping[str, float],
    token_contract: Mapping[str, object],
) -> dict[str, object]:
    values = tuple(prompt_hashes)
    family_values = tuple(family_hashes)
    if (
        len(values) != len(family_values)
        or any(not _is_sha256(value) for value in family_values)
    ):
        raise ValueError("heldout calibration-B family hashes are invalid")
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.structured_mlp_compression_heldout_claim"
        ),
        "format_version": 1,
        "namespace": (
            STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
        ),
        "state": "claimed_immediately_before_tokenization",
        "role": "calibration_b",
        "role_prompt_set_sha256": _claim_identity(values),
        "role_prompt_count": len(values),
        "ordered_prompt_sha256": _sha256(
            values,
            domain=_CLAIM_IDENTITY_DOMAIN,
        ),
        "ordered_family_sha256": _sha256(
            family_values,
            domain=_FAMILY_DOMAIN,
        ),
        "candidate": copy.deepcopy(dict(candidate_binding)),
        "v7_corpus": copy.deepcopy(dict(v7_binding)),
        "model_resolved_commit": model_resolved_commit,
        "layer_id": layer_id,
        "thresholds_sha256": _sha256(
            thresholds,
            domain=_CLAIM_DOMAIN,
        ),
        "token_contract": copy.deepcopy(dict(token_contract)),
    }
    payload["claim_payload_sha256"] = _sha256(
        payload,
        domain=_CLAIM_DOMAIN,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return payload


def _single_candidate_gates(
    result: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    name = STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE
    behavior = result.get("behavior")
    direct = result.get("direct")
    branches = result.get("branches")
    audits = result.get("execution_audits")
    native = result.get("ordinary_vs_segmented_native")
    replay = result.get("native_boundary_replay")
    if not all(
        isinstance(value, Mapping)
        for value in (behavior, direct, branches, audits, native, replay)
    ):
        raise ValueError("heldout structured evaluation is invalid")
    behavior_gate = _behavior_gates(
        behavior[name],  # type: ignore[index]
        nll_atol=thresholds["nll_atol"],
        top1_min=thresholds["top1_min"],
        teacher_kl_max=thresholds["teacher_kl_max"],
        p90_abs_nll_max=thresholds["p90_abs_nll_max"],
        p10_top1_min=thresholds["p10_top1_min"],
    )
    direct_gate = _direct_gates(
        direct[name],  # type: ignore[index]
        block_delta_nrmse_max=thresholds["block_delta_nrmse_max"],
        block_delta_cosine_min=thresholds[
            "block_delta_cosine_min"
        ],
    )
    branch_gate = _branch_gates(
        branches[name],  # type: ignore[index]
        nrmse_max=thresholds["branch_delta_nrmse_max"],
        cosine_min=thresholds["branch_delta_cosine_min"],
    )
    execution_passed = (
        isinstance(audits[name], Mapping)  # type: ignore[index]
        and audits[name].get("passed") is True  # type: ignore[index,union-attr]
    )
    native_passed = native.get("passed") is True
    replay_passed = replay.get("passed") is True
    passed = (
        all(behavior_gate.values())
        and all(direct_gate.values())
        and all(branch_gate.values())
        and execution_passed
        and native_passed
        and replay_passed
    )
    return {
        "behavior": behavior_gate,
        "direct": direct_gate,
        "branches": branch_gate,
        "execution": execution_passed,
        "ordinary_vs_segmented_native": native_passed,
        "native_boundary_replay": replay_passed,
        "passed": passed,
    }


def _normalize_tokenizer_for_causal_lm_(tokenizer: object) -> None:
    """Apply exactly the tokenizer normalization used by calibration."""

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("tokenizer must define a pad or EOS token")
        setattr(tokenizer, "pad_token", eos_token)
    if hasattr(tokenizer, "padding_side"):
        setattr(tokenizer, "padding_side", "right")


def _resource_accounting(
    *,
    executor: StructuredTransformerLayerExecutor,
    logical: Mapping[str, object],
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
    resource_report: Mapping[str, object],
) -> dict[str, object]:
    parameters = resource_report.get("parameters")
    compute = resource_report.get("compute_per_valid_token")
    if not isinstance(parameters, Mapping) or not isinstance(
        compute,
        Mapping,
    ):
        raise ValueError("candidate resource report is invalid")
    per_token_macs = compute.get("macs")
    if not isinstance(per_token_macs, Mapping):
        raise ValueError("candidate MAC resource report is invalid")
    source_parameters = int(source_static["parameter_count"])
    candidate_parameters = executor.learned_parameter_count
    removed_parameters = source_parameters - candidate_parameters
    valid_tokens = int(logical["valid_tokens"])
    source_total_macs = int(source_macs["total_macs"])
    candidate_total_macs = int(logical["logical_total_macs"])
    removed_macs = source_total_macs - candidate_total_macs
    expected_removed_per_token = int(per_token_macs["removed"])
    if (
        parameters.get("source_full_layer") != source_parameters
        or parameters.get("compressed_full_layer")
        != candidate_parameters
        or parameters.get("removed_full_layer") != removed_parameters
        or removed_macs != valid_tokens * expected_removed_per_token
        or candidate_total_macs >= source_total_macs
        or candidate_parameters >= source_parameters
    ):
        raise ValueError(
            "candidate parameter or MAC savings do not recompute"
        )
    return {
        "parameters": {
            "source_layer": source_parameters,
            "compressed_layer": candidate_parameters,
            "removed": removed_parameters,
            "retained_ratio": candidate_parameters / source_parameters,
            "reduction_fraction": removed_parameters / source_parameters,
        },
        "analytic_macs": {
            "scope": (
                "linear_weight_MACs_plus_QK_and_AV_dot_products_on_"
                "identical_valid_lengths"
            ),
            "valid_tokens": valid_tokens,
            "source_layer": source_total_macs,
            "compressed_layer": candidate_total_macs,
            "removed": removed_macs,
            "removed_per_valid_token": expected_removed_per_token,
            "retained_ratio": candidate_total_macs / source_total_macs,
            "reduction_fraction": removed_macs / source_total_macs,
        },
        "exact_savings_recomputed": True,
        "resource_values_used_as_fidelity_gates": False,
        "latency_or_kernel_speed_claim": False,
    }


def _evaluation_payload(
    result: Mapping[str, object],
    *,
    executor: StructuredTransformerLayerExecutor,
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
    resource_report: Mapping[str, object],
    thresholds: Mapping[str, float],
    tokenized_stream: Mapping[str, object],
    tokenized_stream_contract: Mapping[str, object],
) -> dict[str, object]:
    name = STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE
    sanitized = copy.deepcopy(dict(result))
    sanitized.pop("boundaries", None)
    logical = sanitized.get("logical_accounting")
    if not isinstance(logical, Mapping) or not isinstance(
        logical.get(name),
        Mapping,
    ):
        raise ValueError("heldout logical accounting is invalid")
    gates = _single_candidate_gates(
        sanitized,
        thresholds=thresholds,
    )
    resources = _resource_accounting(
        executor=executor,
        logical=logical[name],  # type: ignore[arg-type,index]
        source_static=source_static,
        source_macs=source_macs,
        resource_report=resource_report,
    )
    return {
        "evaluated": True,
        **sanitized,
        "candidate_execution_fingerprint": (
            executor.execution_fingerprint()
        ),
        "gates": gates,
        "resources": resources,
        "resource_gates_applied": False,
        "passed": gates["passed"],
        "tokenized_stream": copy.deepcopy(dict(tokenized_stream)),
        "tokenized_stream_contract": copy.deepcopy(
            dict(tokenized_stream_contract)
        ),
    }


def _unevaluated_validation() -> dict[str, object]:
    return {
        "evaluated": False,
        "reason": "calibration_b_failed_validation_not_tokenized",
        "behavior": None,
        "direct": None,
        "branches": None,
        "execution_audits": None,
        "ordinary_vs_segmented_native": None,
        "native_boundary_replay": None,
        "logical_accounting": None,
        "candidate_execution_fingerprint": None,
        "gates": None,
        "resources": None,
        "resource_gates_applied": False,
        "passed": False,
        "tokenized_stream": None,
        "tokenized_stream_contract": None,
    }


def _report_from_payload(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "candidate": copy.deepcopy(payload["candidate"]),
        "model": copy.deepcopy(payload["model"]),
        "protocol": copy.deepcopy(payload["protocol"]),
        "calibration_b": copy.deepcopy(payload["calibration_b"]),
        "validation": copy.deepcopy(payload["validation"]),
        "scientific_status": copy.deepcopy(
            payload["scientific_status"]
        ),
        "artifact": {
            "tensor_file": output.name,
            "contains_model_weights": False,
            "contains_executor_weights": False,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "contains_teacher_targets": False,
            "contains_fisher_scores": False,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }


def run_gemma3_structured_mlp_compression_heldout_experiment(
    *,
    candidate_artifact_path: Path | str,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
    output: Path | str,
    calibration_b_ledger_dir: Path | str,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    local_files_only: bool = True,
    minimum_calibration_a_prompts: int = (
        DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
    ),
    minimum_heldout_prompts: int = DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    minimum_heldout_supervised_tokens: int = (
        DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
    ),
    minimum_length_buckets: int = DEFAULT_MINIMUM_LENGTH_BUCKETS,
) -> dict[str, object]:
    """Evaluate exactly one unopened compression candidate on fresh v7."""

    candidate_path = Path(candidate_artifact_path)
    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    audit_path = Path(corpus_audit_path)
    output_path = Path(output)
    expected_minima = {
        "minimum_calibration_a_prompts": (
            DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        ),
        "minimum_heldout_prompts": DEFAULT_MINIMUM_HELDOUT_PROMPTS,
        "minimum_heldout_supervised_tokens": (
            DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        ),
        "minimum_length_buckets": DEFAULT_MINIMUM_LENGTH_BUCKETS,
    }
    observed_minima = {
        "minimum_calibration_a_prompts": (
            minimum_calibration_a_prompts
        ),
        "minimum_heldout_prompts": minimum_heldout_prompts,
        "minimum_heldout_supervised_tokens": (
            minimum_heldout_supervised_tokens
        ),
        "minimum_length_buckets": minimum_length_buckets,
    }
    if observed_minima != expected_minima:
        raise ValueError(
            "compression heldout requires the frozen standard data minima"
        )
    if device_name != "cpu" or dtype != "float32":
        raise ValueError(
            "compression heldout requires the frozen CPU float32 runtime"
        )
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(
            "compression heldout output already exists; refusing overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(
            "compression heldout output directory is not writable"
        )

    # Required ordering: strict candidate load precedes any v7 inspection.
    loaded_candidate = _load_compression_a_candidate(
        candidate_path,
        map_location="cpu",
    )
    candidate = _candidate_preflight(
        loaded_candidate,
        candidate_artifact_path=candidate_path,
    )
    corpus = _v7_corpus_preflight(
        prompt_splits_path=prompt_path,
        family_manifest_path=family_path,
        corpus_audit_path=audit_path,
        candidate=candidate,
        minimum_calibration_a_prompts=(
            minimum_calibration_a_prompts
        ),
        minimum_heldout_prompts=minimum_heldout_prompts,
    )
    prompts = corpus["prompts"]
    prompt_metadata = corpus["prompt_metadata"]
    family_metadata = corpus["family_metadata"]
    assert hasattr(prompts, "calibration_b")
    assert isinstance(prompt_metadata, Mapping)
    assert isinstance(family_metadata, Mapping)
    prompt_hashes_by_role = prompt_metadata["per_prompt_sha256"]
    family_hashes_by_role = family_metadata[
        "per_prompt_family_sha256"
    ]
    assert isinstance(prompt_hashes_by_role, Mapping)
    assert isinstance(family_hashes_by_role, Mapping)
    claim_path = _claim_path(
        calibration_b_ledger_dir,
        prompt_hashes_by_role["calibration_b"],  # type: ignore[arg-type,index]
    )
    if claim_path.exists():
        raise FileExistsError(
            "compression calibration B was already claimed; refusing "
            f"heldout reuse: {claim_path}"
        )

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=candidate["model_id"],
        revision=candidate["resolved_commit"],
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    layer_index = candidate["layer_index"]
    assert isinstance(layer_index, int)
    plan = adapter.plan_layer_block(layer_index, layer_index)
    layer_id = plan.layer_ids[0]
    if layer_id != candidate["protocol"].get("layer_id"):
        raise ValueError(
            "heldout native layer differs from candidate layer binding"
        )
    executor = candidate["executor"]
    assert isinstance(executor, StructuredTransformerLayerExecutor)
    executor = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        executor.artifact_state_dict(),
        map_location=device,
    )
    if (
        executor.dtype is not torch.float32
        or executor.execution_fingerprint()
        != candidate["binding"]["execution_fingerprint"]
    ):
        raise ValueError(
            "candidate FP32 fingerprint changed during strict reload"
        )
    model_metadata = _model_provenance(
        model,
        model_id=candidate["model_id"],  # type: ignore[arg-type]
        requested_revision=candidate["resolved_commit"],  # type: ignore[arg-type]
    )
    candidate_model = candidate["model"]
    assert isinstance(candidate_model, Mapping)
    model_identity_fields = (
        "model_id",
        "resolved_commit",
        "model_class",
        "config_sha256",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "maximum_context",
        "parameter_count",
        "dtype",
    )
    if any(
        model_metadata.get(field) != candidate_model.get(field)
        for field in model_identity_fields
    ):
        raise ValueError("heldout model revision differs from candidate")
    source_static = _source_block_static(adapter, plan)
    resource_report = candidate["resource_report"]
    assert isinstance(resource_report, Mapping)
    parameters = resource_report.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("source_full_layer")
        != source_static["parameter_count"]
        or parameters.get("compressed_full_layer")
        != executor.learned_parameter_count
    ):
        raise ValueError(
            "candidate static resource report does not match native layer"
        )
    thresholds = candidate["thresholds"]
    assert isinstance(thresholds, Mapping)
    _normalize_tokenizer_for_causal_lm_(tokenizer)
    normalized_tokenizer = _tokenizer_provenance(tokenizer)
    token_contract = {
        "maximum_tokenized_length": candidate[
            "maximum_tokenized_length"
        ],
        "tokenization_batch_size": candidate[
            "tokenization_batch_size"
        ],
        "minimum_heldout_supervised_tokens": (
            minimum_heldout_supervised_tokens
        ),
        "minimum_length_buckets": minimum_length_buckets,
        "runtime_device": "cpu",
        "runtime_dtype": "float32",
        "candidate_executor_dtype": str(executor.dtype),
        "tokenizer": normalized_tokenizer,
    }
    if token_contract["tokenizer"] != candidate["protocol"].get(
        "tokenizer"
    ):
        raise ValueError(
            "heldout tokenizer differs from candidate token contract"
        )

    # This call is intentionally adjacent to tokenization. O_EXCL consumes
    # the claim even if tokenization or evaluation raises.
    claim = _exclusive_heldout_claim(
        claim_path,
        prompt_hashes=prompt_hashes_by_role[
            "calibration_b"
        ],  # type: ignore[arg-type,index]
        family_hashes=family_hashes_by_role[
            "calibration_b"
        ],  # type: ignore[arg-type,index]
        candidate_binding=candidate["binding"],  # type: ignore[arg-type]
        v7_binding=corpus["binding"],  # type: ignore[arg-type]
        model_resolved_commit=candidate["resolved_commit"],  # type: ignore[arg-type]
        layer_id=layer_id,
        thresholds=thresholds,  # type: ignore[arg-type]
        token_contract=token_contract,
    )
    calibration_b_batches, calibration_b_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=candidate["maximum_tokenized_length"],  # type: ignore[arg-type]
        tokenization_batch_size=candidate[
            "tokenization_batch_size"
        ],  # type: ignore[arg-type]
        device=device,
    )
    calibration_b_contract = _tokenized_stream_contract(
        calibration_b_stream,
        split_name="calibration_b",
        minimum_supervised_tokens=minimum_heldout_supervised_tokens,
        minimum_length_buckets=minimum_length_buckets,
    )
    if _tokenizer_provenance(tokenizer) != normalized_tokenizer:
        raise RuntimeError(
            "calibration-B tokenization changed normalized tokenizer state"
        )
    if calibration_b_stream.get(
        "source_prompt_sha256"
    ) != prompt_hashes_by_role["calibration_b"]:
        raise ValueError(
            "calibration-B tokenized stream does not bind v7 prompts"
        )
    _assert_tokenized_content_disjointness(
        {"calibration_b": calibration_b_stream}
    )
    candidate_map = {
        STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE: executor
    }
    calibration_b_result = evaluate_structured_candidates(
        adapter,
        calibration_b_batches,
        plan=plan,
        layer_id=layer_id,
        candidates=candidate_map,
        native_parity_tolerance=thresholds[
            "native_parity_tolerance"
        ],
    )
    boundaries = calibration_b_result.get("boundaries")
    if not isinstance(boundaries, tuple):
        raise ValueError("calibration-B boundaries are missing")
    calibration_b_source_macs = _source_block_macs(
        adapter,
        plan,
        boundaries,
        static=source_static,
    )
    calibration_b_payload = _evaluation_payload(
        calibration_b_result,
        executor=executor,
        source_static=source_static,
        source_macs=calibration_b_source_macs,
        resource_report=resource_report,
        thresholds=thresholds,  # type: ignore[arg-type]
        tokenized_stream=calibration_b_stream,
        tokenized_stream_contract=calibration_b_contract,
    )
    calibration_b_payload["reason"] = (
        "fresh_v7_calibration_b_one_shot_evaluation"
    )
    calibration_b_passed = calibration_b_payload["passed"] is True
    guard.assert_unchanged()

    if calibration_b_passed:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=candidate[
                "maximum_tokenized_length"
            ],  # type: ignore[arg-type]
            tokenization_batch_size=candidate[
                "tokenization_batch_size"
            ],  # type: ignore[arg-type]
            device=device,
        )
        validation_contract = _tokenized_stream_contract(
            validation_stream,
            split_name="validation",
            minimum_supervised_tokens=(
                minimum_heldout_supervised_tokens
            ),
            minimum_length_buckets=minimum_length_buckets,
        )
        if _tokenizer_provenance(tokenizer) != normalized_tokenizer:
            raise RuntimeError(
                "validation tokenization changed normalized tokenizer state"
            )
        if validation_stream.get(
            "source_prompt_sha256"
        ) != prompt_hashes_by_role["validation"]:
            raise ValueError(
                "validation tokenized stream does not bind v7 prompts"
            )
        _assert_tokenized_content_disjointness(
            {
                "calibration_b": calibration_b_stream,
                "validation": validation_stream,
            }
        )
        validation_result = evaluate_structured_candidates(
            adapter,
            validation_batches,
            plan=plan,
            layer_id=layer_id,
            candidates=candidate_map,
            native_parity_tolerance=thresholds[
                "native_parity_tolerance"
            ],
        )
        validation_boundaries = validation_result.get("boundaries")
        if not isinstance(validation_boundaries, tuple):
            raise ValueError("validation boundaries are missing")
        validation_source_macs = _source_block_macs(
            adapter,
            plan,
            validation_boundaries,
            static=source_static,
        )
        validation_payload = _evaluation_payload(
            validation_result,
            executor=executor,
            source_static=source_static,
            source_macs=validation_source_macs,
            resource_report=resource_report,
            thresholds=thresholds,  # type: ignore[arg-type]
            tokenized_stream=validation_stream,
            tokenized_stream_contract=validation_contract,
        )
        validation_payload["reason"] = (
            "calibration_b_passed_fresh_v7_validation_evaluation"
        )
    else:
        validation_payload = _unevaluated_validation()
    validation_passed = validation_payload["passed"] is True
    guard.assert_unchanged()

    outcome = (
        "single_layer_structured_mlp_compression_passed"
        if calibration_b_passed and validation_passed
        else (
            "rejected_on_validation"
            if calibration_b_passed
            else "rejected_on_calibration_b"
        )
    )
    protocol = {
        "candidate_artifact": copy.deepcopy(candidate["binding"]),
        "v7_corpus": copy.deepcopy(corpus["binding"]),
        "source_corpus_forbidden_hash_sets_bound": True,
        "calibration_b_claim": claim,
        "calibration_b_claim_path_namespace": (
            STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
        ),
        "calibration_b_claim_order_insensitive": True,
        "calibration_b_claim_precedes_tokenization": True,
        "calibration_b_claim_consumed_on_failure": True,
        "never_reuse_parent_calibration_b": True,
        "thresholds": copy.deepcopy(dict(thresholds)),
        "token_contract": token_contract,
        "layer_index": layer_index,
        "layer_id": layer_id,
        "model_resolved_commit": candidate["resolved_commit"],
        "test_policy": "sealed_hash_only_never_tokenized_or_evaluated",
        "library_versions": _library_versions(),
    }
    scientific_status = {
        "scope": "single_layer_layer4_structured_mlp_width_compression",
        "outcome": outcome,
        "compression_passed": (
            outcome
            == "single_layer_structured_mlp_compression_passed"
        ),
        "calibration_b_evaluated": True,
        "calibration_b_passed": calibration_b_passed,
        "validation_evaluated": calibration_b_passed,
        "validation_passed": validation_passed,
        "test_tokenized": False,
        "test_evaluated": False,
        "candidate_strict_loaded_before_v7_preflight": True,
        "v7_static_preflight_before_model_load": True,
        "exclusive_calibration_b_claim": True,
        "source_layer_calls_in_candidate_path": 0,
        "source_layer_removed_from_candidate_path": True,
        "model_weights_changed": False,
        "model_weights_in_artifact": False,
        "candidate_weights_in_artifact": False,
        "parameter_reduction_measured": True,
        "analytic_mac_reduction_measured": True,
        "parameter_reduction_supported": (
            calibration_b_passed and validation_passed
        ),
        "analytic_mac_reduction_supported": (
            calibration_b_passed and validation_passed
        ),
        "resource_values_used_as_fidelity_gates": False,
        "latency_or_kernel_speed_claim": False,
        "general_method_viable": False,
        "model_level_promotion_authorized": False,
    }
    payload: dict[str, object] = {
        "schema": STRUCTURED_MLP_COMPRESSION_HELDOUT_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_COMPRESSION_HELDOUT_FORMAT_VERSION
        ),
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_fisher_scores": False,
        "candidate": {
            **copy.deepcopy(candidate["binding"]),
            "resource_report": copy.deepcopy(resource_report),
        },
        "model": model_metadata,
        "protocol": protocol,
        "calibration_b": calibration_b_payload,
        "validation": validation_payload,
        "scientific_status": scientific_status,
    }
    scientific_digest = _sha256(payload, domain=_PAYLOAD_DOMAIN)
    report = _report_from_payload(
        payload,
        output=output_path,
        scientific_payload_sha256=scientific_digest,
    )
    report_sha256 = _sha256(report, domain=_REPORT_DOMAIN)
    artifact = {
        **payload,
        "scientific_payload_sha256": scientific_digest,
        "report_sha256": report_sha256,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    output_path.with_suffix(".json").write_text(
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
    # Required strict replay before returning a scientific result.
    load_gemma3_structured_mlp_compression_heldout_artifact(output_path)
    return report


def _validate_evaluation_payload(
    value: object,
    *,
    evaluated: bool,
    thresholds: Mapping[str, float],
    candidate_fingerprint: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("heldout evaluation payload is invalid")
    if not evaluated:
        if value != _unevaluated_validation():
            raise ValueError(
                "heldout unevaluated validation payload is invalid"
            )
        return
    stream, _ = _validated_tokenized_stream(
        value.get("tokenized_stream"),
        split_name=(
            "calibration_b"
            if value.get("reason")
            == "fresh_v7_calibration_b_one_shot_evaluation"
            else "validation"
        ),
    )
    if stream != value.get("tokenized_stream"):
        raise ValueError("heldout tokenized stream is not canonical")
    result_fields = {
        key: item
        for key, item in value.items()
        if key
        in {
            "behavior",
            "direct",
            "branches",
            "execution_audits",
            "ordinary_vs_segmented_native",
            "native_boundary_replay",
            "logical_accounting",
        }
    }
    gates = _single_candidate_gates(
        result_fields,
        thresholds=thresholds,
    )
    if (
        value.get("evaluated") is not True
        or value.get("candidate_execution_fingerprint")
        != candidate_fingerprint
        or value.get("gates") != gates
        or value.get("passed") is not gates["passed"]
        or value.get("resource_gates_applied") is not False
        or not isinstance(value.get("resources"), Mapping)
        or value["resources"].get("exact_savings_recomputed") is not True
    ):
        raise ValueError("heldout evaluation binding is invalid")
    resources = value["resources"]
    parameters = resources.get("parameters")
    macs = resources.get("analytic_macs")
    if not isinstance(parameters, Mapping) or not isinstance(
        macs,
        Mapping,
    ):
        raise ValueError("heldout resource accounting is invalid")
    source_parameters = parameters.get("source_layer")
    compressed_parameters = parameters.get("compressed_layer")
    removed_parameters = parameters.get("removed")
    source_macs = macs.get("source_layer")
    compressed_macs = macs.get("compressed_layer")
    removed_macs = macs.get("removed")
    valid_tokens = macs.get("valid_tokens")
    removed_per_token = macs.get("removed_per_valid_token")
    if (
        any(
            type(item) is not int
            for item in (
                source_parameters,
                compressed_parameters,
                removed_parameters,
                source_macs,
                compressed_macs,
                removed_macs,
                valid_tokens,
                removed_per_token,
            )
        )
        or source_parameters - compressed_parameters
        != removed_parameters
        or source_macs - compressed_macs != removed_macs
        or removed_macs != valid_tokens * removed_per_token
        or not 0 < compressed_parameters < source_parameters
        or not 0 < compressed_macs < source_macs
    ):
        raise ValueError("heldout resource savings do not recompute")


def _validate_claim_binding(
    value: object,
    *,
    protocol: Mapping[str, object],
) -> None:
    fields = {
        "schema",
        "format_version",
        "namespace",
        "state",
        "role",
        "role_prompt_set_sha256",
        "role_prompt_count",
        "ordered_prompt_sha256",
        "ordered_family_sha256",
        "candidate",
        "v7_corpus",
        "model_resolved_commit",
        "layer_id",
        "thresholds_sha256",
        "token_contract",
        "claim_payload_sha256",
    }
    claim = _exact_mapping(
        value,
        fields,
        label="compression heldout claim",
    )
    payload = {
        key: item
        for key, item in claim.items()
        if key != "claim_payload_sha256"
    }
    thresholds = protocol.get("thresholds")
    token_contract = protocol.get("token_contract")
    v7_corpus = protocol.get("v7_corpus")
    candidate = protocol.get("candidate_artifact")
    if (
        claim["schema"]
        != "fisher_graph.structured_mlp_compression_heldout_claim"
        or claim["format_version"] != 1
        or claim["namespace"]
        != STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
        or claim["state"]
        != "claimed_immediately_before_tokenization"
        or claim["role"] != "calibration_b"
        or claim["candidate"] != candidate
        or claim["v7_corpus"] != v7_corpus
        or claim["model_resolved_commit"]
        != protocol.get("model_resolved_commit")
        or claim["layer_id"] != protocol.get("layer_id")
        or claim["token_contract"] != token_contract
        or not isinstance(thresholds, Mapping)
        or claim["thresholds_sha256"]
        != _sha256(thresholds, domain=_CLAIM_DOMAIN)
        or claim["claim_payload_sha256"]
        != _sha256(payload, domain=_CLAIM_DOMAIN)
    ):
        raise ValueError("compression heldout claim binding is invalid")
    if not isinstance(v7_corpus, Mapping):
        raise ValueError("compression heldout v7 binding is invalid")
    prompt_splits = v7_corpus.get("prompt_splits")
    families = v7_corpus.get("prompt_families")
    if not isinstance(prompt_splits, Mapping) or not isinstance(
        families,
        Mapping,
    ):
        raise ValueError("compression heldout v7 role hashes are invalid")
    prompt_hashes = prompt_splits.get("per_prompt_sha256")
    family_hashes = families.get("per_prompt_family_sha256")
    if not isinstance(prompt_hashes, Mapping) or not isinstance(
        family_hashes,
        Mapping,
    ):
        raise ValueError("compression heldout v7 role hashes are invalid")
    b_prompts = prompt_hashes.get("calibration_b")
    b_families = family_hashes.get("calibration_b")
    if (
        not isinstance(b_prompts, list)
        or not isinstance(b_families, list)
        or claim["role_prompt_count"] != len(b_prompts)
        or claim["role_prompt_set_sha256"]
        != _claim_identity(b_prompts)
        or claim["ordered_prompt_sha256"]
        != _sha256(
            tuple(b_prompts),
            domain=_CLAIM_IDENTITY_DOMAIN,
        )
        or claim["ordered_family_sha256"]
        != _sha256(tuple(b_families), domain=_FAMILY_DOMAIN)
    ):
        raise ValueError(
            "compression heldout claim role identity is invalid"
        )


def load_gemma3_structured_mlp_compression_heldout_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly validate the source-free heldout result artifact."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or set(raw) != _OUTER_FIELDS:
        raise ValueError("compression heldout artifact fields are invalid")
    if (
        raw["schema"] != STRUCTURED_MLP_COMPRESSION_HELDOUT_SCHEMA
        or raw["format_version"]
        != STRUCTURED_MLP_COMPRESSION_HELDOUT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_executor_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or raw["contains_teacher_targets"] is not False
        or raw["contains_fisher_scores"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("compression heldout artifact header is invalid")
    if any(
        isinstance(value, Tensor)
        for value in _walk_values(raw)
    ):
        raise ValueError(
            "compression heldout artifact must not contain tensors"
        )
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    if _sha256(payload, domain=_PAYLOAD_DOMAIN) != raw[
        "scientific_payload_sha256"
    ]:
        raise ValueError(
            "compression heldout scientific payload digest mismatch"
        )
    candidate = _exact_mapping(
        raw["candidate"],
        {
            "artifact_file_sha256",
            "scientific_payload_sha256",
            "report_sha256",
            "pipeline_report_sha256",
            "selection_sha256",
            "score_collection_sha256",
            "execution_fingerprint",
            "state_sha256",
            "strict_candidate_loader_used",
            "source_free",
            "resource_report",
        },
        label="heldout candidate binding",
    )
    for field in (
        "artifact_file_sha256",
        "scientific_payload_sha256",
        "report_sha256",
        "pipeline_report_sha256",
        "selection_sha256",
        "score_collection_sha256",
        "execution_fingerprint",
        "state_sha256",
    ):
        if not _is_sha256(candidate[field]):
            raise ValueError("heldout candidate digest is invalid")
    protocol = raw["protocol"]
    status = raw["scientific_status"]
    if not isinstance(protocol, Mapping) or not isinstance(
        status,
        Mapping,
    ):
        raise ValueError("heldout protocol or status is invalid")
    protocol_fields = {
        "candidate_artifact",
        "v7_corpus",
        "source_corpus_forbidden_hash_sets_bound",
        "calibration_b_claim",
        "calibration_b_claim_path_namespace",
        "calibration_b_claim_order_insensitive",
        "calibration_b_claim_precedes_tokenization",
        "calibration_b_claim_consumed_on_failure",
        "never_reuse_parent_calibration_b",
        "thresholds",
        "token_contract",
        "layer_index",
        "layer_id",
        "model_resolved_commit",
        "test_policy",
        "library_versions",
    }
    if (
        set(protocol) != protocol_fields
        or protocol.get("candidate_artifact")
        != {
            key: candidate[key]
            for key in candidate
            if key != "resource_report"
        }
        or protocol.get("calibration_b_claim_path_namespace")
        != STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
        or protocol.get("source_corpus_forbidden_hash_sets_bound")
        is not True
        or protocol.get("calibration_b_claim_order_insensitive")
        is not True
        or protocol.get("calibration_b_claim_precedes_tokenization")
        is not True
        or protocol.get("calibration_b_claim_consumed_on_failure")
        is not True
        or protocol.get("never_reuse_parent_calibration_b") is not True
        or protocol.get("test_policy")
        != "sealed_hash_only_never_tokenized_or_evaluated"
    ):
        raise ValueError("heldout protocol binding is invalid")
    thresholds = protocol.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("heldout thresholds are invalid")
    thresholds = _validated_thresholds(thresholds)
    token_contract = protocol.get("token_contract")
    if (
        not isinstance(token_contract, Mapping)
        or token_contract.get("runtime_device") != "cpu"
        or token_contract.get("runtime_dtype") != "float32"
        or token_contract.get("candidate_executor_dtype")
        != "torch.float32"
        or token_contract.get("minimum_heldout_supervised_tokens")
        != DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        or token_contract.get("minimum_length_buckets")
        != DEFAULT_MINIMUM_LENGTH_BUCKETS
    ):
        raise ValueError("heldout frozen token contract is invalid")
    _validate_claim_binding(
        protocol.get("calibration_b_claim"),
        protocol=protocol,
    )
    calibration_b_passed = status.get("calibration_b_passed") is True
    validation_evaluated = status.get("validation_evaluated") is True
    validation_passed = status.get("validation_passed") is True
    expected_outcome = (
        "single_layer_structured_mlp_compression_passed"
        if calibration_b_passed and validation_passed
        else (
            "rejected_on_validation"
            if calibration_b_passed
            else "rejected_on_calibration_b"
        )
    )
    if (
        status.get("outcome") != expected_outcome
        or status.get("compression_passed")
        is not (
            expected_outcome
            == "single_layer_structured_mlp_compression_passed"
        )
        or status.get("scope")
        != "single_layer_layer4_structured_mlp_width_compression"
        or validation_evaluated is not calibration_b_passed
        or status.get("test_tokenized") is not False
        or status.get("test_evaluated") is not False
        or status.get("model_weights_in_artifact") is not False
        or status.get("candidate_weights_in_artifact") is not False
        or status.get("parameter_reduction_measured") is not True
        or status.get("analytic_mac_reduction_measured") is not True
        or status.get("parameter_reduction_supported")
        is not (
            expected_outcome
            == "single_layer_structured_mlp_compression_passed"
        )
        or status.get("analytic_mac_reduction_supported")
        is not (
            expected_outcome
            == "single_layer_structured_mlp_compression_passed"
        )
        or status.get("general_method_viable") is not False
        or status.get("model_level_promotion_authorized") is not False
    ):
        raise ValueError("heldout scientific status is invalid")
    _validate_evaluation_payload(
        raw["calibration_b"],
        evaluated=True,
        thresholds=thresholds,
        candidate_fingerprint=candidate[
            "execution_fingerprint"
        ],  # type: ignore[arg-type]
    )
    _validate_evaluation_payload(
        raw["validation"],
        evaluated=validation_evaluated,
        thresholds=thresholds,
        candidate_fingerprint=candidate[
            "execution_fingerprint"
        ],  # type: ignore[arg-type]
    )
    report = _report_from_payload(
        payload,
        output=source,
        scientific_payload_sha256=raw[
            "scientific_payload_sha256"
        ],  # type: ignore[arg-type]
    )
    if _sha256(report, domain=_REPORT_DOMAIN) != raw["report_sha256"]:
        raise ValueError("compression heldout report digest mismatch")
    report_path = source.with_suffix(".json")
    if not report_path.is_file():
        raise ValueError("compression heldout JSON report is missing")
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    canonical_report = json.loads(
        json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if stored_report != canonical_report:
        raise ValueError("compression heldout JSON report is invalid")
    return {
        "candidate": copy.deepcopy(dict(candidate)),
        "model": copy.deepcopy(raw["model"]),
        "protocol": copy.deepcopy(dict(protocol)),
        "calibration_b": copy.deepcopy(raw["calibration_b"]),
        "validation": copy.deepcopy(raw["validation"]),
        "scientific_status": copy.deepcopy(dict(status)),
        "metadata": {
            "scientific_payload_sha256": raw[
                "scientific_payload_sha256"
            ],
            "report_sha256": raw["report_sha256"],
            "artifact_file_sha256": _file_sha256(source),
        },
        "report": report,
    }


def _walk_values(value: object) -> Sequence[object]:
    values: list[object] = []
    pending = [value]
    while pending:
        current = pending.pop()
        values.append(current)
        if isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, (tuple, list)):
            pending.extend(current)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one strict structured Gemma MLP compression candidate "
            "on exclusive fresh-v7 calibration B and gated validation."
        )
    )
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--prompt-splits", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-b-ledger-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model files absent from the local cache to be downloaded.",
    )
    parser.add_argument(
        "--minimum-calibration-a-prompts",
        type=int,
        default=DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS,
    )
    parser.add_argument(
        "--minimum-heldout-prompts",
        type=int,
        default=DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--minimum-heldout-supervised-tokens",
        type=int,
        default=DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS,
    )
    parser.add_argument(
        "--minimum-length-buckets",
        type=int,
        default=DEFAULT_MINIMUM_LENGTH_BUCKETS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_structured_mlp_compression_heldout_experiment(
        candidate_artifact_path=args.candidate_artifact,
        prompt_splits_path=args.prompt_splits,
        family_manifest_path=args.family_manifest,
        corpus_audit_path=args.corpus_audit,
        output=args.output,
        calibration_b_ledger_dir=args.calibration_b_ledger_dir,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        minimum_calibration_a_prompts=(
            args.minimum_calibration_a_prompts
        ),
        minimum_heldout_prompts=args.minimum_heldout_prompts,
        minimum_heldout_supervised_tokens=(
            args.minimum_heldout_supervised_tokens
        ),
        minimum_length_buckets=args.minimum_length_buckets,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE",
    "STRUCTURED_MLP_COMPRESSION_HELDOUT_CORPUS_ID",
    "STRUCTURED_MLP_COMPRESSION_HELDOUT_FORMAT_VERSION",
    "STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE",
    "STRUCTURED_MLP_COMPRESSION_HELDOUT_SCHEMA",
    "build_parser",
    "load_gemma3_structured_mlp_compression_heldout_artifact",
    "main",
    "run_gemma3_structured_mlp_compression_heldout_experiment",
]
