"""Fit and guard a true reduced-width Gemma MLP pseudo-unit candidate.

This experiment is intentionally separate from the deletion-and-refit
compression rung.  Calibration A is split by whole prompt families: only the
fit half may affect the candidate, while the guard half evaluates the frozen
executor before any held-out role can be opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from .gemma3_full_width_single_layer_experiment import PromptFamilyManifest
from .gemma3_full_width_single_layer_experiment import (
    DEFAULT_BLOCK_DELTA_COSINE_MIN,
    DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    DEFAULT_MINIMUM_FISHER_ROWS,
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
from .gemma3_rotated_span_executor_experiment import _behavior_gates
from .gemma3_stability_experiment import (
    Gemma3PromptSplits,
    _library_versions,
    _tokenizer_provenance,
    load_gemma3_prompt_splits,
)
from .gemma3_structured_mlp_compression_a_experiment import (
    _standard_minima as _parent_standard_minima,
    _standard_thresholds as _parent_standard_thresholds,
)
from .gemma3_structured_single_layer_experiment import (
    DEFAULT_BRANCH_DELTA_COSINE_MIN,
    DEFAULT_BRANCH_DELTA_NRMSE_MAX,
    DEFAULT_NATIVE_PARITY_TOLERANCE,
    _branch_gates,
    _corpus_audit_binding,
    _format4_family_binding,
    collect_structured_training_batches,
    evaluate_structured_candidates,
    load_gemma3_structured_single_layer_artifact,
)
from .structured_layer_distillation import StructuredLayerTargets
from .structured_mlp_compression import (
    StructuredMLPUnitSelection,
    build_width_compressed_structured_executor,
    select_fisher_taylor_mlp_units,
)
from .structured_mlp_compression_pipeline import (
    collect_gemma_mlp_fisher_taylor_batches,
    refit_structured_mlp_down_projection_from_targets_,
)
from .structured_operator_bootstrap import (
    structured_operator_coefficient_sha256,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_PSEUDO_UNIT_A_SCHEMA = (
    "fisher_graph.gemma3_structured_mlp_pseudo_unit_a_candidate"
)
STRUCTURED_MLP_PSEUDO_UNIT_A_FORMAT_VERSION = 1
STRUCTURED_MLP_PSEUDO_UNIT_CORPUS_ID = "structured-strong-v9"
GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH = 2_048
GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH = 1_920
GEMMA_MLP_PSEUDO_UNIT_PAIR_COUNT = 128
DEFAULT_LAYER_INDEX = 4
DEFAULT_MAX_LENGTH = 256
DEFAULT_TOKENIZATION_BATCH_SIZE = 4
DEFAULT_GENERATOR_STEPS = 400
DEFAULT_GENERATOR_LEARNING_RATE = 1e-3
DEFAULT_GENERATOR_MINIBATCH_ROWS = 512
DEFAULT_GENERATOR_GRADIENT_CLIP_NORM = 1.0
DEFAULT_JACOBIAN_FLOOR_FRACTION = 1e-6
DEFAULT_DOWN_RIDGE = 1e-6
DEFAULT_GUARD_BLOCK_DELTA_NRMSE_MAX = 0.015
DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS = 45_000
DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS = 3

_DIRECT = "pseudo_unit_direct"
_REFIT = "pseudo_unit_down_refit"
_DELETION = "fisher_deletion_down_refit"
_CANDIDATE_NAMES = (_DIRECT, _REFIT, _DELETION)
_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_pseudo_unit_a.payload.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_pseudo_unit_a.report.v1\0"
)
_JSON_DOMAIN = (
    b"fisher_graph.gemma3_structured_mlp_pseudo_unit_a.json.v1\0"
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
    "calibration_a_fit",
    "calibration_a_guard",
    "pipeline",
    "deletion_baseline",
    "resource_report",
    "executor",
    "scientific_payload_sha256",
    "report_sha256",
}

_EXPECTED_ROLE_COUNTS = {
    "calibration_a": 512,
    "calibration_b": 96,
    "validation": 96,
    "test": 96,
}
_EXPECTED_FAMILY_COUNTS = {
    "calibration_a": 16,
    "calibration_b": 8,
    "validation": 8,
    "test": 8,
}
_EXPECTED_PARTITION_BANDS = {
    "micro": 8,
    "compact": 8,
    "medium": 16,
    "long": 224,
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


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _report_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(value, domain=_REPORT_DOMAIN)


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
        "primary_guard_block_delta_nrmse_max": (
            DEFAULT_GUARD_BLOCK_DELTA_NRMSE_MAX
        ),
    }


def _frozen_generator_protocol() -> dict[str, object]:
    return {
        "steps": DEFAULT_GENERATOR_STEPS,
        "learning_rate": DEFAULT_GENERATOR_LEARNING_RATE,
        "minibatch_rows": DEFAULT_GENERATOR_MINIBATCH_ROWS,
        "gradient_clip_norm": DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
        "jacobian_floor_fraction": DEFAULT_JACOBIAN_FLOOR_FRACTION,
        "down_ridge_for_diagnostic_variants": DEFAULT_DOWN_RIDGE,
        "fixed_before_guard": True,
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
    }


def _frozen_partition_token_contract() -> dict[str, object]:
    return {
        "minimum_supervised_tokens": (
            DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS
        ),
        "minimum_length_buckets": (
            DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS
        ),
        "applies_equally_to_fit_and_guard": True,
        "frozen_before_v9_tokenizer_or_model_access": True,
        "origin": "preregistered_for_fresh_structured_strong_v9",
    }


def _indices_sha256(indices: tuple[int, ...]) -> str:
    encoded = json.dumps(
        indices,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationAFamilyPartition:
    """One audit-bound, whole-family subset of calibration A."""

    name: str
    prompts: tuple[str, ...]
    family_ids: tuple[str, ...]
    prompt_indices: tuple[int, ...]
    prompt_index_sha256: str
    band_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.name not in {"fit", "guard"}:
            raise ValueError("partition name must be 'fit' or 'guard'")
        if (
            len(self.prompts) != 256
            or len(self.family_ids) != 8
            or len(set(self.family_ids)) != 8
            or len(self.prompt_indices) != 256
            or tuple(sorted(self.prompt_indices)) != self.prompt_indices
            or len(set(self.prompt_indices)) != 256
            or self.prompt_index_sha256
            != _indices_sha256(self.prompt_indices)
            or dict(self.band_counts) != _EXPECTED_PARTITION_BANDS
        ):
            raise ValueError(
                f"calibration-A {self.name} partition is invalid"
            )


def validate_v9_calibration_a_partitions(
    prompts: Gemma3PromptSplits,
    families: PromptFamilyManifest,
    audit_payload: Mapping[str, object],
) -> tuple[CalibrationAFamilyPartition, CalibrationAFamilyPartition]:
    """Authenticate and materialize the v9 A-fit/A-guard partition."""

    if not isinstance(prompts, Gemma3PromptSplits):
        raise TypeError("prompts must be Gemma3PromptSplits")
    if not isinstance(families, PromptFamilyManifest):
        raise TypeError("families must be PromptFamilyManifest")
    if not isinstance(audit_payload, Mapping):
        raise TypeError("audit_payload must be a mapping")
    prompts_metadata = prompts.metadata()
    family_metadata = families.metadata()
    if (
        prompts_metadata.get("counts") != _EXPECTED_ROLE_COUNTS
        or family_metadata.get("counts") != _EXPECTED_ROLE_COUNTS
        or family_metadata.get("unique_family_counts")
        != _EXPECTED_FAMILY_COUNTS
        or audit_payload.get("format_version") != 4
        or audit_payload.get("corpus_id")
        != STRUCTURED_MLP_PSEUDO_UNIT_CORPUS_ID
        or audit_payload.get("purpose")
        != "mode_bundling_fit_guard_and_frozen_evaluation"
        or audit_payload.get("calibration_a_policy")
        != "family_disjoint_fit_guard_development_only"
        or audit_payload.get("calibration_a_fit_may_train_candidate")
        is not True
        or audit_payload.get("calibration_a_guard_may_change_candidate")
        is not False
        or audit_payload.get("calibration_b_reuse_allowed") is not False
        or audit_payload.get("heldout_splits_evaluated") is not False
        or audit_payload.get("heldout_splits_tokenized") is not False
        or audit_payload.get("heldout_splits_unevaluated") is not True
        or audit_payload.get("heldout_splits_untokenized") is not True
        or audit_payload.get("calibration_b_model_evaluated") is not False
        or audit_payload.get("validation_model_evaluated") is not False
        or audit_payload.get("test_model_evaluated") is not False
        or audit_payload.get("tokenizer_or_model_accessed") is not False
        or audit_payload.get("corpus_frozen_before_model_load") is not True
    ):
        raise ValueError("v9 pseudo-unit corpus policy is invalid")
    for field in (
        "prior_local_exact_prompt_overlap_count",
        "prior_raw_prompt_overlap_count",
        "prior_normalized_prompt_overlap_count",
        "prior_domain_slug_overlap_count",
        "prior_template_marker_overlap_count",
        "prior_template_signature_overlap_count",
        "prior_5_6_7_8_word_ngram_overlap_count",
    ):
        if audit_payload.get(field) != 0:
            raise ValueError(f"v9 corpus overlap field {field!r} is invalid")

    raw_partitions = audit_payload.get(
        "calibration_a_family_partitions"
    )
    if (
        not isinstance(raw_partitions, Mapping)
        or raw_partitions.get("family_disjoint") is not True
        or raw_partitions.get("union_covers_calibration_a") is not True
    ):
        raise ValueError("v9 calibration-A family partition is invalid")

    def materialize(name: str) -> CalibrationAFamilyPartition:
        raw = raw_partitions.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"v9 calibration-A {name} partition is missing")
        raw_indices = raw.get("prompt_indices")
        raw_family_ids = raw.get("family_ids")
        raw_bands = raw.get("band_counts")
        if (
            not isinstance(raw_indices, list)
            or any(type(index) is not int for index in raw_indices)
            or not isinstance(raw_family_ids, list)
            or any(
                not isinstance(family, str)
                for family in raw_family_ids
            )
            or not isinstance(raw_bands, Mapping)
            or raw.get("prompt_count") != 256
            or raw.get("family_count") != 8
        ):
            raise ValueError(
                f"v9 calibration-A {name} partition fields are invalid"
            )
        indices = tuple(raw_indices)
        family_ids = tuple(raw_family_ids)
        try:
            selected_prompts = tuple(
                prompts.calibration_a[index]
                for index in indices
            )
            selected_families = tuple(
                families.calibration_a[index]
                for index in indices
            )
        except IndexError as error:
            raise ValueError(
                f"v9 calibration-A {name} index is out of range"
            ) from error
        if (
            set(selected_families) != set(family_ids)
            or any(
                family not in set(family_ids)
                for family in selected_families
            )
        ):
            raise ValueError(
                f"v9 calibration-A {name} family binding is invalid"
            )
        return CalibrationAFamilyPartition(
            name=name,
            prompts=selected_prompts,
            family_ids=family_ids,
            prompt_indices=indices,
            prompt_index_sha256=str(raw.get("prompt_index_sha256")),
            band_counts={
                str(key): int(value)
                for key, value in raw_bands.items()
            },
        )

    fit = materialize("fit")
    guard = materialize("guard")
    if (
        set(fit.family_ids) & set(guard.family_ids)
        or set(fit.prompt_indices) & set(guard.prompt_indices)
        or set(fit.prompt_indices) | set(guard.prompt_indices)
        != set(range(len(prompts.calibration_a)))
    ):
        raise ValueError(
            "v9 calibration-A fit and guard are not disjoint and complete"
        )
    return fit, guard


@dataclass(frozen=True, slots=True)
class _StaticV9Corpus:
    prompts: Gemma3PromptSplits
    families: PromptFamilyManifest
    prompt_metadata: Mapping[str, object]
    family_metadata: Mapping[str, object]
    audit_binding: Mapping[str, object]
    source_corpus: Mapping[str, object]
    fit: CalibrationAFamilyPartition
    guard: CalibrationAFamilyPartition


def _partition_binding(
    partition: CalibrationAFamilyPartition,
) -> dict[str, object]:
    return {
        "name": partition.name,
        "prompt_count": len(partition.prompts),
        "family_count": len(partition.family_ids),
        "prompt_index_sha256": partition.prompt_index_sha256,
        "family_ids_sha256": _json_sha256(partition.family_ids),
        "band_counts": copy.deepcopy(dict(partition.band_counts)),
    }


def _load_structured_v9_corpus_preflight(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
) -> _StaticV9Corpus:
    """Authenticate every v9 role without tokenizing any role."""

    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    audit_path = Path(corpus_audit_path)
    prompts = load_gemma3_prompt_splits(prompt_path)
    _require_prompt_protocol(
        prompts,
        minimum_calibration_a_prompts=_EXPECTED_ROLE_COUNTS[
            "calibration_a"
        ],
        minimum_heldout_prompts=_EXPECTED_ROLE_COUNTS["calibration_b"],
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
        raise ValueError("pseudo-unit fitting requires a bound v9 audit")
    payload = audit_binding.get("payload")
    lexical = audit_binding.get("lexical_length_audit")
    if (
        prompt_metadata.get("scientific_status") != PROMPT_STATUS
        or family_metadata.get("scientific_status") != FAMILY_STATUS
        or prompt_metadata.get("counts") != _EXPECTED_ROLE_COUNTS
        or family_metadata.get("counts") != _EXPECTED_ROLE_COUNTS
        or family_metadata.get("unique_family_counts")
        != _EXPECTED_FAMILY_COUNTS
        or not isinstance(payload, Mapping)
        or not isinstance(lexical, Mapping)
        or lexical.get("all_roles_cover_all_bands") is not True
    ):
        raise ValueError(
            "pseudo-unit fitting requires the breadth-validated v9 corpus"
        )
    fit, guard = validate_v9_calibration_a_partitions(
        prompts,
        families,
        payload,
    )
    prompt_hashes = prompt_metadata.get("per_prompt_sha256")
    family_hashes = family_metadata.get("per_prompt_family_sha256")
    if not isinstance(prompt_hashes, Mapping) or not isinstance(
        family_hashes,
        Mapping,
    ):
        raise ValueError("v9 corpus hash metadata is incomplete")
    roles = ("calibration_a", "calibration_b", "validation", "test")
    source_corpus = {
        "corpus_id": STRUCTURED_MLP_PSEUDO_UNIT_CORPUS_ID,
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
        "calibration_a_partitions": {
            "fit": _partition_binding(fit),
            "guard": _partition_binding(guard),
            "family_disjoint": True,
            "prompt_disjoint": True,
            "union_covers_calibration_a": True,
        },
    }
    return _StaticV9Corpus(
        prompts=prompts,
        families=families,
        prompt_metadata=prompt_metadata,
        family_metadata=family_metadata,
        audit_binding=audit_binding,
        source_corpus=source_corpus,
        fit=fit,
        guard=guard,
    )


def validate_structured_v9_pseudo_unit_corpus_preflight(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
) -> dict[str, object]:
    """Return the complete hash-only v9 binding before model access."""

    corpus = _load_structured_v9_corpus_preflight(
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


def _authenticate_format5_parent(
    loaded: Mapping[str, object],
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
    """Authenticate the v6 parent without equating v6 A to fresh v9 fit."""

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
        or protocol.get("data_minima") != _parent_standard_minima()
        or protocol.get("thresholds") != _parent_standard_thresholds()
        or model.get("model_id") != model_id
        or model.get("requested_revision") != revision
        or model.get("resolved_commit") != revision
        or status.get("calibration_b_passed") is not True
        or status.get("validation_passed") is not True
    ):
        raise ValueError("format-5 parent authentication failed")
    primary = executors.get("structured_source_visibility")
    primary_training = training.get("structured_source_visibility")
    if (
        not isinstance(primary, StructuredTransformerLayerExecutor)
        or not isinstance(primary_training, Mapping)
        or primary.owns_source_model_weights
        or primary.config.transformer.feed_forward.intermediate_width
        != GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH
        or primary_training.get("fitting_method")
        != "activation_only_structured_operator_bootstrap"
        or primary_training.get("optimizer") != "none"
        or primary_training.get("optimizer_steps") != 0
        or primary_training.get("suffix_training_steps") != 0
        or primary_training.get("final_execution_fingerprint")
        != primary.execution_fingerprint()
    ):
        raise ValueError("format-5 parent primary executor is invalid")
    bootstrap = primary_training.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or not _is_sha256(bootstrap.get("calibration_split_sha256"))
    ):
        raise ValueError("format-5 parent bootstrap binding is invalid")
    return primary, primary_training, protocol


def _guard_gate_report(
    evaluation: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    behavior = evaluation.get("behavior")
    direct = evaluation.get("direct")
    branches = evaluation.get("branches")
    audits = evaluation.get("execution_audits")
    parity = evaluation.get("ordinary_vs_segmented_native")
    replay = evaluation.get("native_boundary_replay")
    if not all(
        isinstance(value, Mapping)
        for value in (behavior, direct, branches, audits, parity, replay)
    ):
        raise ValueError("guard evaluation is incomplete")
    parity_passed = parity.get("passed") is True  # type: ignore[union-attr]
    replay_passed = replay.get("passed") is True  # type: ignore[union-attr]
    global_controls_passed = parity_passed and replay_passed
    candidates: dict[str, object] = {}
    for name in _CANDIDATE_NAMES:
        behavior_row = behavior.get(name)  # type: ignore[union-attr]
        direct_row = direct.get(name)  # type: ignore[union-attr]
        branch_row = branches.get(name)  # type: ignore[union-attr]
        audit = audits.get(name)  # type: ignore[union-attr]
        if not all(
            isinstance(value, Mapping)
            for value in (
                behavior_row,
                direct_row,
                branch_row,
                audit,
            )
        ):
            raise ValueError(f"guard candidate {name!r} is incomplete")
        behavior_gates = _behavior_gates(
            behavior_row,  # type: ignore[arg-type]
            nll_atol=thresholds["nll_atol"],
            top1_min=thresholds["top1_min"],
            teacher_kl_max=thresholds["teacher_kl_max"],
            p90_abs_nll_max=thresholds["p90_abs_nll_max"],
            p10_top1_min=thresholds["p10_top1_min"],
        )
        direct_gates = _direct_gates(
            direct_row,  # type: ignore[arg-type]
            block_delta_nrmse_max=thresholds[
                "block_delta_nrmse_max"
            ],
            block_delta_cosine_min=thresholds[
                "block_delta_cosine_min"
            ],
        )
        branch_gates = _branch_gates(
            branch_row,  # type: ignore[arg-type]
            nrmse_max=thresholds["branch_delta_nrmse_max"],
            cosine_min=thresholds["branch_delta_cosine_min"],
        )
        standard_passed = (
            all(behavior_gates.values())
            and all(direct_gates.values())
            and all(branch_gates.values())
            and audit.get("passed") is True  # type: ignore[union-attr]
            and global_controls_passed
        )
        margin_gate = (
            float(direct_row["block_delta_nrmse"])  # type: ignore[index]
            <= thresholds["primary_guard_block_delta_nrmse_max"]
        )
        candidates[name] = {
            "behavior": behavior_gates,
            "direct": direct_gates,
            "branches": branch_gates,
            "execution": audit.get("passed") is True,  # type: ignore[union-attr]
            "standard_passed": standard_passed,
            "primary_direct_block_nrmse_margin": margin_gate,
            "authorizes_b": (
                name == _DIRECT and standard_passed and margin_gate
            ),
            "diagnostic_only": name != _DIRECT,
        }
    primary = candidates[_DIRECT]
    assert isinstance(primary, Mapping)
    return {
        "candidates": candidates,
        "ordinary_vs_segmented_native": parity_passed,
        "native_boundary_replay": replay_passed,
        "primary_policy": (
            "direct_bundle_only_standard_gates_plus_block_nrmse_0_015"
        ),
        "primary_passed": primary["authorizes_b"] is True,
        "refit_or_deletion_may_promote_primary": False,
    }


def _resource_report(
    parent: StructuredTransformerLayerExecutor,
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
    logical_accounting: Mapping[str, object],
) -> dict[str, object]:
    source_width = (
        parent.config.transformer.feed_forward.intermediate_width
    )
    residual_width = parent.width
    source_parameters = parent.learned_parameter_count
    source_mlp_macs = 3 * residual_width * source_width
    rows: dict[str, object] = {}
    for name, candidate in candidates.items():
        logical = logical_accounting.get(name)
        if not isinstance(logical, Mapping):
            raise ValueError("candidate logical accounting is invalid")
        retained_width = (
            candidate.config.transformer.feed_forward.intermediate_width
        )
        candidate_mlp_macs = 3 * residual_width * retained_width
        valid_tokens = int(logical["valid_tokens"])
        candidate_total = int(logical["logical_total_macs"])
        source_total = candidate_total + valid_tokens * (
            source_mlp_macs - candidate_mlp_macs
        )
        rows[name] = {
            "source_intermediate_width": source_width,
            "retained_intermediate_width": retained_width,
            "source_layer_parameters": source_parameters,
            "candidate_layer_parameters": candidate.learned_parameter_count,
            "removed_layer_parameters": (
                source_parameters - candidate.learned_parameter_count
            ),
            "retained_layer_parameter_ratio": (
                candidate.learned_parameter_count / source_parameters
            ),
            "source_mlp_linear_macs_per_valid_token": source_mlp_macs,
            "candidate_mlp_linear_macs_per_valid_token": (
                candidate_mlp_macs
            ),
            "removed_mlp_linear_macs_per_valid_token": (
                source_mlp_macs - candidate_mlp_macs
            ),
            "guard_source_layer_analytic_macs": source_total,
            "guard_candidate_layer_analytic_macs": candidate_total,
            "guard_analytic_mac_ratio": candidate_total / source_total,
            "resource_values_require_guard_fidelity": True,
            "latency_or_kernel_speed_claim": False,
        }
    return {
        "scope": "single_gemma_layer_4",
        "source_width": source_width,
        "retained_width": GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH,
        "candidates": rows,
    }


def _execute_fit_then_guard(
    *,
    materialize_fit: Callable[[], object],
    build_from_fit: Callable[[object], object],
    candidate_fingerprints: Callable[[object], Mapping[str, str]],
    materialize_guard: Callable[[], object],
    evaluate_guard: Callable[[object, object], object],
) -> tuple[
    object,
    object,
    object,
    object,
    Mapping[str, str],
]:
    """Enforce the only legal stateful ordering for the v9 A protocol."""

    fit = materialize_fit()
    candidates = build_from_fit(fit)
    frozen = dict(candidate_fingerprints(candidates))
    if set(frozen) != set(_CANDIDATE_NAMES) or any(
        not _is_sha256(value) for value in frozen.values()
    ):
        raise ValueError("fit candidate fingerprints are invalid")
    guard = materialize_guard()
    evaluation = evaluate_guard(candidates, guard)
    after = dict(candidate_fingerprints(candidates))
    if after != frozen:
        raise RuntimeError("A-guard evaluation mutated a frozen candidate")
    return fit, candidates, guard, evaluation, frozen


def _without_boundaries(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    result = dict(evaluation)
    boundaries = result.pop("boundaries", None)
    if not isinstance(boundaries, tuple):
        raise ValueError("structured guard boundaries are missing")
    return copy.deepcopy(result)


def _candidate_fingerprints(
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
) -> dict[str, str]:
    if set(candidates) != set(_CANDIDATE_NAMES):
        raise ValueError("pseudo-unit candidate set is invalid")
    return {
        name: candidate.execution_fingerprint()
        for name, candidate in candidates.items()
    }


def _validate_artifact_candidate_fingerprint_bindings(
    *,
    fit: Mapping[str, object],
    guard: Mapping[str, object],
    pipeline: Mapping[str, object],
    deletion: Mapping[str, object],
    direct_fingerprint: str,
) -> None:
    """Require one exact candidate identity map across fit, guard, and reports."""

    frozen = fit.get("candidate_fingerprints_frozen_before_guard")
    before = guard.get("candidate_fingerprints_before")
    after = guard.get("candidate_fingerprints_after")
    maps = (frozen, before, after)
    if (
        not _is_sha256(direct_fingerprint)
        or any(not isinstance(value, Mapping) for value in maps)
        or any(
            set(value) != set(_CANDIDATE_NAMES)  # type: ignore[arg-type]
            or any(
                not _is_sha256(fingerprint)
                for fingerprint in value.values()  # type: ignore[union-attr]
            )
            for value in maps
        )
        or frozen != before
        or before != after
    ):
        raise ValueError(
            "pseudo-unit fit/guard candidate fingerprints are invalid"
        )

    variants = pipeline.get("variants")
    direct_variant = (
        variants.get("direct") if isinstance(variants, Mapping) else None
    )
    refit_variant = (
        variants.get("global_down_refit")
        if isinstance(variants, Mapping)
        else None
    )
    deletion_refit = deletion.get("terminal_projection_refit")
    deletion_targets = deletion.get("refit_targets")
    if (
        not isinstance(frozen, Mapping)
        or not isinstance(direct_variant, Mapping)
        or not isinstance(refit_variant, Mapping)
        or not isinstance(deletion_refit, Mapping)
        or not isinstance(deletion_targets, Mapping)
        or direct_variant.get("execution_fingerprint")
        != direct_fingerprint
        or refit_variant.get("execution_fingerprint")
        != frozen[_REFIT]
        or deletion_refit.get("executor_fingerprint_after")
        != frozen[_DELETION]
        or deletion.get("execution_fingerprint") != frozen[_DELETION]
        or deletion_targets.get(
            "candidate_execution_fingerprint_before_refit"
        )
        != deletion_refit.get("executor_fingerprint_before")
        or deletion_targets.get(
            "actual_runtime_features_used_for_down_refit"
        )
        is not True
        or deletion_targets.get(
            "native_selected_projection_features_used_for_down_refit"
        )
        is not False
        or frozen[_DIRECT] != direct_fingerprint
    ):
        raise ValueError(
            "pseudo-unit candidate report fingerprints are inconsistent"
        )


def _strict_executor_roundtrip(
    executor: StructuredTransformerLayerExecutor,
) -> StructuredTransformerLayerExecutor:
    restored = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        executor.artifact_state_dict(),
        map_location=executor.device,
    )
    if (
        restored.execution_fingerprint()
        != executor.execution_fingerprint()
        or restored.owns_source_model_weights
    ):
        raise RuntimeError("source-free executor strict roundtrip drifted")
    restored.eval()
    return restored


def _prepare_actual_runtime_mlp_refit_targets(
    executor: StructuredTransformerLayerExecutor,
    targets: Sequence[object],
    selection: StructuredMLPUnitSelection,
) -> tuple[tuple[StructuredLayerTargets, ...], dict[str, object]]:
    """Bind a down-only refit to the candidate's executable MLP features."""

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError(
            "executor must be a StructuredTransformerLayerExecutor"
        )
    if not isinstance(selection, StructuredMLPUnitSelection):
        raise TypeError("selection must be a StructuredMLPUnitSelection")
    if not targets:
        raise ValueError("actual-runtime refit targets cannot be empty")
    retained_width = (
        executor.config.transformer.feed_forward.intermediate_width
    )
    if retained_width != selection.retained_width:
        raise ValueError(
            "actual-runtime refit executor and selection widths differ"
        )

    transformed: list[StructuredLayerTargets] = []
    feature_records: list[dict[str, object]] = []
    valid_rows = 0
    executor_fingerprint = executor.execution_fingerprint()
    for batch_index, target in enumerate(targets):
        if (
            not isinstance(target, StructuredLayerTargets)
            or target.provenance != selection.provenance
            or target.normalized_feed_forward_input.shape[-1]
            != executor.width
            or target.normalized_feed_forward_input.device
            != executor.device
        ):
            raise ValueError(
                "actual-runtime refit target does not match the candidate"
            )
        with torch.no_grad():
            actual = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            ).detach()
        valid = target.sequence.query_valid_mask
        actual_rows = actual[valid]
        if (
            actual.shape[:2]
            != target.normalized_feed_forward_input.shape[:2]
            or actual.shape[-1] != retained_width
            or not bool(torch.isfinite(actual_rows).all())
        ):
            raise RuntimeError(
                "candidate produced invalid actual-runtime MLP features"
            )
        transformed.append(
            replace(
                target,
                feed_forward_projection_input=actual.clone(),
            )
        )
        rows = int(valid.sum().item())
        valid_rows += rows
        feature_records.append(
            {
                "batch_index": batch_index,
                "valid_rows": rows,
                "actual_runtime_feature_sha256": _payload_sha256(
                    {
                        "actual_runtime_features": (
                            actual_rows.detach().to(device="cpu")
                        )
                    }
                ),
            }
        )
    return tuple(transformed), {
        "schema": (
            "fisher_graph.structured_mlp_actual_runtime_refit_inputs"
        ),
        "format_version": 1,
        "selection_sha256": selection.selection_sha256,
        "candidate_execution_fingerprint_before_refit": (
            executor_fingerprint
        ),
        "batches": len(transformed),
        "valid_rows": valid_rows,
        "source_intermediate_width": selection.source_width,
        "retained_intermediate_width": selection.retained_width,
        "feature_source": (
            "candidate.feed_forward_projection_features("
            "native_normalized_feed_forward_input)"
        ),
        "actual_runtime_features_used_for_down_refit": True,
        "native_selected_projection_features_used_for_down_refit": False,
        "padding_rows_excluded_from_digest": True,
        "ordered_valid_feature_sha256": _json_sha256(feature_records),
    }


def _build_deletion_baseline(
    parent: StructuredTransformerLayerExecutor,
    targets: Sequence[object],
    score_batches: Sequence[object],
    *,
    calibration_split_sha256: str,
    down_ridge: float,
) -> tuple[
    StructuredTransformerLayerExecutor,
    dict[str, object],
]:
    transformer = parent.config.transformer
    sites = transformer.operator_sites
    if sites is None:
        raise ValueError("parent executor lacks structured operator sites")
    selection = select_fisher_taylor_mlp_units(
        score_batches,  # type: ignore[arg-type]
        calibration_split_sha256=calibration_split_sha256,
        activation_site=sites.feed_forward_down_input,
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_width=GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH,
        expected_source_width=GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH,
    )
    baseline, construction = build_width_compressed_structured_executor(
        parent,
        selection,
    )
    refit_targets, target_report = (
        _prepare_actual_runtime_mlp_refit_targets(
            baseline,
            targets,
            selection,
        )
    )
    refit = refit_structured_mlp_down_projection_from_targets_(
        baseline,
        refit_targets,
        calibration_split_sha256=calibration_split_sha256,
        ridge=down_ridge,
    )
    baseline = _strict_executor_roundtrip(baseline)
    if (
        refit.get("executor_fingerprint_after")
        != baseline.execution_fingerprint()
    ):
        raise RuntimeError("deletion refit strict roundtrip drifted")
    return baseline, {
        "selection": selection.metadata(),
        "construction": construction,
        "refit_targets": target_report,
        "terminal_projection_refit": refit,
        "execution_fingerprint": baseline.execution_fingerprint(),
        "strict_state_roundtrip_verified": True,
        "diagnostic_only": True,
    }


@dataclass(frozen=True, slots=True)
class _MaterializedSplit:
    batches: tuple[object, ...]
    stream: Mapping[str, object]
    contract: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BuiltFitCandidates:
    executors: Mapping[str, StructuredTransformerLayerExecutor]
    direct_artifact_state: Mapping[str, object]
    pipeline_report: Mapping[str, object]
    deletion_report: Mapping[str, object]
    score_report: Mapping[str, object]
    plan_report: Mapping[str, object]
    structured_training_batches: int


def _safe_score_collection_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "schema",
        "format_version",
        "objective",
        "provenance",
        "accounting",
        "source_audit",
        "heldout_opened",
        "collection_sha256",
    )
    if any(field not in report for field in fields):
        raise ValueError("score collection report is incomplete")
    return {
        field: copy.deepcopy(report[field])
        for field in fields
    }


def _assert_no_tensor(value: object, *, label: str) -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{label} unexpectedly contains a Tensor")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_tensor(item, label=f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_tensor(item, label=f"{label}[{index}]")


def _validate_tensor_locations(payload: Mapping[str, object]) -> None:
    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Tensor):
            if len(path) < 3 or path[:2] != (
                "executor",
                "model_state_dict",
            ):
                raise ValueError(
                    "pseudo-unit artifact has a Tensor outside executor "
                    f"state at {'.'.join(path)}"
                )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("artifact mapping keys must be strings")
                walk(item, (*path, key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(payload, ())


def _build_json_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    scientific_payload_sha256: str | None,
    main_artifact_written: bool,
) -> dict[str, object]:
    fields = (
        "scientific_status",
        "model",
        "protocol",
        "parent",
        "calibration_a_fit",
        "calibration_a_guard",
        "pipeline",
        "deletion_baseline",
        "resource_report",
    )
    report = {
        "schema": STRUCTURED_MLP_PSEUDO_UNIT_A_SCHEMA,
        "format_version": STRUCTURED_MLP_PSEUDO_UNIT_A_FORMAT_VERSION,
        **{
            field: copy.deepcopy(payload[field])
            for field in fields
        },
        "artifact": {
            "tensor_file": tensor_file if main_artifact_written else None,
            "main_artifact_written": main_artifact_written,
            "contains_compressed_executor_weights": (
                main_artifact_written
            ),
            "contains_source_model_weights": False,
            "contains_full_parent_executor_state": False,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "contains_teacher_targets": False,
            "contains_fisher_taylor_scores": False,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }
    _assert_no_tensor(report, label="JSON report")
    return report


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _new_staging_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _unlink_if_same_file(final: Path, staged: Path) -> None:
    try:
        if os.path.samestat(final.stat(), staged.stat()):
            final.unlink()
    except FileNotFoundError:
        pass


def _publish_json_exclusive(
    path: Path,
    value: Mapping[str, object],
    *,
    must_remain_absent: Sequence[Path] = (),
) -> None:
    """Atomically expose one complete JSON file without overwriting."""

    staged = _new_staging_path(path)
    linked = False
    try:
        if path.exists() or any(item.exists() for item in must_remain_absent):
            raise FileExistsError(
                "refusing to overwrite a pseudo-unit diagnostic"
            )
        _write_json(staged, value)
        _fsync_file(staged)
        if any(item.exists() for item in must_remain_absent):
            raise FileExistsError(
                "pseudo-unit artifact appeared during diagnostic staging"
            )
        os.link(staged, path)
        linked = True
        if any(item.exists() for item in must_remain_absent):
            raise FileExistsError(
                "pseudo-unit artifact appeared during diagnostic publication"
            )
    except BaseException:
        if linked:
            _unlink_if_same_file(path, staged)
        raise
    finally:
        staged.unlink(missing_ok=True)


def _publish_artifact_pair(
    output: Path,
    artifact: Mapping[str, object],
    report: Mapping[str, object],
    *,
    load_published: Callable[[], object] | None = None,
) -> object | None:
    """Publish complete tensor/JSON siblings exclusively, with rollback."""

    report_path = output.with_suffix(".json")
    staged_output: Path | None = None
    staged_report: Path | None = None
    linked: list[tuple[Path, Path]] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or report_path.exists():
            raise FileExistsError(
                "refusing to overwrite a pseudo-unit A diagnostic"
            )
        staged_output = _new_staging_path(output)
        staged_report = _new_staging_path(report_path)
        torch.save(artifact, staged_output)
        _write_json(staged_report, report)
        _fsync_file(staged_output)
        _fsync_file(staged_report)

        os.link(staged_output, output)
        linked.append((output, staged_output))
        os.link(staged_report, report_path)
        linked.append((report_path, staged_report))
        return None if load_published is None else load_published()
    except BaseException:
        for final, staged in reversed(linked):
            _unlink_if_same_file(final, staged)
        raise
    finally:
        if staged_output is not None:
            staged_output.unlink(missing_ok=True)
        if staged_report is not None:
            staged_report.unlink(missing_ok=True)


def _write_failed_guard_diagnostic(
    output: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Write JSON only; this helper can never create the requested .pt."""

    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a pseudo-unit artifact")
    report = _build_json_report(
        payload,
        tensor_file=output.name,
        scientific_payload_sha256=None,
        main_artifact_written=False,
    )
    _publish_json_exclusive(
        report_path,
        report,
        must_remain_absent=(output,),
    )
    if output.exists():
        raise RuntimeError("failed guard unexpectedly wrote a tensor artifact")
    return report


def run_gemma3_structured_mlp_pseudo_unit_a_experiment(
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
    generator_steps: int = DEFAULT_GENERATOR_STEPS,
    generator_learning_rate: float = DEFAULT_GENERATOR_LEARNING_RATE,
    generator_minibatch_rows: int = DEFAULT_GENERATOR_MINIBATCH_ROWS,
    generator_gradient_clip_norm: float = (
        DEFAULT_GENERATOR_GRADIENT_CLIP_NORM
    ),
    jacobian_floor_fraction: float = DEFAULT_JACOBIAN_FLOOR_FRACTION,
    down_ridge: float = DEFAULT_DOWN_RIDGE,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Fit on v9 A-fit, freeze, then evaluate every candidate on A-guard."""

    from .structured_mlp_pseudo_unit_bundling import (
        build_fisher_pseudo_unit_bundling_plan,
    )
    from .structured_mlp_pseudo_unit_pipeline import (
        build_structured_mlp_pseudo_unit_candidate,
    )

    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
    ):
        raise ValueError("revision must be an exact lowercase commit hash")
    if type(layer_index) is not int or layer_index != DEFAULT_LAYER_INDEX:
        raise ValueError(
            f"pseudo-unit v9 requires frozen layer_index={DEFAULT_LAYER_INDEX}"
        )
    if (
        max_length != DEFAULT_MAX_LENGTH
        or tokenization_batch_size != DEFAULT_TOKENIZATION_BATCH_SIZE
    ):
        raise ValueError(
            "pseudo-unit v9 requires max_length=256 and batch size 4"
        )
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("pseudo-unit v9 requires CPU float32 execution")
    for label, value in (
        ("generator_steps", generator_steps),
        ("generator_minibatch_rows", generator_minibatch_rows),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    for label, value, allow_zero in (
        ("generator_learning_rate", generator_learning_rate, False),
        (
            "generator_gradient_clip_norm",
            generator_gradient_clip_norm,
            False,
        ),
        ("jacobian_floor_fraction", jacobian_floor_fraction, True),
        ("down_ridge", down_ridge, False),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or (float(value) < 0 if allow_zero else float(value) <= 0)
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{label} must be finite and {qualifier}")
    frozen_hyperparameters = (
        (
            "generator_steps",
            generator_steps,
            DEFAULT_GENERATOR_STEPS,
        ),
        (
            "generator_learning_rate",
            generator_learning_rate,
            DEFAULT_GENERATOR_LEARNING_RATE,
        ),
        (
            "generator_minibatch_rows",
            generator_minibatch_rows,
            DEFAULT_GENERATOR_MINIBATCH_ROWS,
        ),
        (
            "generator_gradient_clip_norm",
            generator_gradient_clip_norm,
            DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
        ),
        (
            "jacobian_floor_fraction",
            jacobian_floor_fraction,
            DEFAULT_JACOBIAN_FLOOR_FRACTION,
        ),
        ("down_ridge", down_ridge, DEFAULT_DOWN_RIDGE),
    )
    for label, value, expected in frozen_hyperparameters:
        if value != expected:
            raise ValueError(
                f"pseudo-unit v9 requires frozen {label}={expected}"
            )

    resolved_output = Path(output)
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report_path = resolved_output.with_suffix(".json")
    if resolved_output.exists() or report_path.exists():
        raise FileExistsError(
            "refusing to overwrite a pseudo-unit A diagnostic"
        )

    # This is deliberately the first external artifact operation.
    corpus = _load_structured_v9_corpus_preflight(
        prompt_splits_path=prompt_splits_path,
        family_manifest_path=family_manifest_path,
        corpus_audit_path=corpus_audit_path,
    )
    device = resolve_torch_device(device_name)
    loaded_parent = load_gemma3_structured_single_layer_artifact(
        parent_artifact_path,
        map_location=device,
    )
    parent_executor, parent_training, parent_protocol = (
        _authenticate_format5_parent(
            loaded_parent,
            model_id=model_id,
            revision=revision,
            layer_index=layer_index,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
        )
    )
    del parent_protocol
    parent_metadata = loaded_parent.get("metadata")
    parent_model = loaded_parent.get("model")
    if not isinstance(parent_metadata, Mapping) or not isinstance(
        parent_model,
        Mapping,
    ):
        raise ValueError("format-5 parent metadata is invalid")
    bootstrap = parent_training.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("format-5 parent bootstrap is invalid")
    parent_fingerprint = parent_executor.execution_fingerprint()
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
        "layer_id": bootstrap["layer_id"],
        "parent_calibration_a_split_sha256": bootstrap[
            "calibration_split_sha256"
        ],
        "fresh_v9_fit_is_independent_of_parent_calibration_a": True,
        "primary_execution_fingerprint": parent_fingerprint,
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
    source_guard = _FrozenModelTensorGuard(model)
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
    layer_plan = adapter.plan_layer_block(layer_index, layer_index)
    layer_id = layer_plan.layer_ids[0]
    if layer_id != parent_binding["layer_id"]:
        raise ValueError("local layer does not match the parent layer")

    def materialize_fit() -> _MaterializedSplit:
        batches, stream = _materialize_split(
            tokenizer,
            corpus.fit.prompts,
            split_name="calibration_a_fit",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        contract = _tokenized_stream_contract(
            stream,
            split_name="calibration_a_fit",
            minimum_supervised_tokens=(
                DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS
            ),
            minimum_length_buckets=(
                DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS
            ),
        )
        return _MaterializedSplit(
            batches=tuple(batches),
            stream=stream,
            contract=contract,
        )

    def build_from_fit(
        value: object,
    ) -> _BuiltFitCandidates:
        if not isinstance(value, _MaterializedSplit):
            raise TypeError("fit materialization is invalid")
        split_sha256 = value.stream.get("serialized_sha256")
        if not _is_sha256(split_sha256):
            raise ValueError("A-fit token stream digest is invalid")
        training = collect_structured_training_batches(
            adapter,
            value.batches,  # type: ignore[arg-type]
            layer_id=layer_id,
            positions_per_sequence=DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
        )
        _require_complete_middle_layer_demand(
            adapter,
            training,  # type: ignore[arg-type]
        )
        score_batches, score_report = (
            collect_gemma_mlp_fisher_taylor_batches(
                adapter,
                value.batches,  # type: ignore[arg-type]
                layer_id=layer_id,
                calibration_split_sha256=split_sha256,
            )
        )
        accounting = score_report.get("accounting")
        if (
            not isinstance(accounting, Mapping)
            or int(accounting.get("valid_rows", 0))
            < DEFAULT_MINIMUM_FISHER_ROWS
        ):
            raise ValueError("A-fit Fisher rows are below the minimum")
        sites = parent_executor.config.transformer.operator_sites
        if sites is None:
            raise ValueError("parent has no structured MLP activation site")
        bundle_plan = build_fisher_pseudo_unit_bundling_plan(
            score_batches,
            source_down_weight=(
                parent_executor.feed_forward.down_proj.weight
            ),
            calibration_split_sha256=split_sha256,
            activation_site=sites.feed_forward_down_input,
            parent_executor_fingerprint=parent_fingerprint,
            retained_width=GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH,
            expected_source_width=GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH,
        )
        if (
            bundle_plan.pair_count != GEMMA_MLP_PSEUDO_UNIT_PAIR_COUNT
        ):
            raise RuntimeError("pseudo-unit pair count drifted")
        targets = tuple(item.targets for item in training)
        pseudo = build_structured_mlp_pseudo_unit_candidate(
            parent_executor,
            bundle_plan,
            targets,
            score_batches,
            calibration_split_sha256=split_sha256,
            generator_steps=generator_steps,
            generator_learning_rate=float(generator_learning_rate),
            generator_minibatch_rows=generator_minibatch_rows,
            generator_gradient_clip_norm=float(
                generator_gradient_clip_norm
            ),
            jacobian_floor_fraction=float(jacobian_floor_fraction),
            down_ridge=float(down_ridge),
        )
        deletion, deletion_report = _build_deletion_baseline(
            parent_executor,
            targets,
            score_batches,
            calibration_split_sha256=split_sha256,
            down_ridge=float(down_ridge),
        )
        candidates = {
            _DIRECT: _strict_executor_roundtrip(
                pseudo.direct_executor
            ),
            _REFIT: _strict_executor_roundtrip(
                pseudo.refit_executor
            ),
            _DELETION: deletion,
        }
        if (
            parent_executor.execution_fingerprint()
            != parent_fingerprint
        ):
            raise RuntimeError("A-fit construction mutated the parent")
        source_guard.assert_unchanged()
        _assert_no_tensor(
            pseudo.report,
            label="pseudo-unit pipeline report",
        )
        _assert_no_tensor(
            deletion_report,
            label="deletion diagnostic report",
        )
        return _BuiltFitCandidates(
            executors=candidates,
            direct_artifact_state=(
                candidates[_DIRECT].artifact_state_dict()
            ),
            pipeline_report=copy.deepcopy(dict(pseudo.report)),
            deletion_report=copy.deepcopy(deletion_report),
            score_report=_safe_score_collection_report(score_report),
            plan_report=copy.deepcopy(bundle_plan.metadata()),
            structured_training_batches=len(training),
        )

    def fingerprints(value: object) -> Mapping[str, str]:
        if not isinstance(value, _BuiltFitCandidates):
            raise TypeError("fit candidates are invalid")
        return _candidate_fingerprints(value.executors)

    def materialize_guard() -> _MaterializedSplit:
        batches, stream = _materialize_split(
            tokenizer,
            corpus.guard.prompts,
            split_name="calibration_a_guard",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        contract = _tokenized_stream_contract(
            stream,
            split_name="calibration_a_guard",
            minimum_supervised_tokens=(
                DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS
            ),
            minimum_length_buckets=(
                DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS
            ),
        )
        return _MaterializedSplit(
            batches=tuple(batches),
            stream=stream,
            contract=contract,
        )

    def evaluate_guard(
        candidate_value: object,
        guard_value: object,
    ) -> dict[str, object]:
        if (
            not isinstance(candidate_value, _BuiltFitCandidates)
            or not isinstance(guard_value, _MaterializedSplit)
        ):
            raise TypeError("frozen guard inputs are invalid")
        return evaluate_structured_candidates(
            adapter,
            guard_value.batches,  # type: ignore[arg-type]
            plan=layer_plan,
            layer_id=layer_id,
            candidates=candidate_value.executors,
            native_parity_tolerance=DEFAULT_NATIVE_PARITY_TOLERANCE,
        )

    (
        fit_value,
        built_value,
        guard_value,
        raw_evaluation,
        frozen_fingerprints,
    ) = _execute_fit_then_guard(
        materialize_fit=materialize_fit,
        build_from_fit=build_from_fit,
        candidate_fingerprints=fingerprints,
        materialize_guard=materialize_guard,
        evaluate_guard=evaluate_guard,
    )
    if (
        not isinstance(fit_value, _MaterializedSplit)
        or not isinstance(built_value, _BuiltFitCandidates)
        or not isinstance(guard_value, _MaterializedSplit)
        or not isinstance(raw_evaluation, Mapping)
    ):
        raise RuntimeError("fit/guard orchestration returned invalid values")
    source_guard.assert_unchanged()
    if parent_executor.execution_fingerprint() != parent_fingerprint:
        raise RuntimeError("guard evaluation mutated the parent executor")

    evaluation = _without_boundaries(raw_evaluation)
    del raw_evaluation
    thresholds = _standard_thresholds()
    gates = _guard_gate_report(
        evaluation,
        thresholds=thresholds,
    )
    evaluation["gates"] = gates
    logical = evaluation.get("logical_accounting")
    if not isinstance(logical, Mapping):
        raise ValueError("guard logical accounting is invalid")
    resources = _resource_report(
        parent_executor,
        built_value.executors,
        logical,
    )
    fit_sha256 = fit_value.stream.get("serialized_sha256")
    guard_sha256 = guard_value.stream.get("serialized_sha256")
    if not _is_sha256(fit_sha256) or not _is_sha256(guard_sha256):
        raise ValueError("fit or guard stream digest is invalid")
    if fit_sha256 == guard_sha256:
        raise ValueError("fit and guard token streams must be distinct")

    protocol = {
        "corpus": copy.deepcopy(dict(corpus.source_corpus)),
        "calibration_a_fit_partition": _partition_binding(corpus.fit),
        "calibration_a_guard_partition": _partition_binding(corpus.guard),
        "fit_guard_family_disjoint": True,
        "fit_guard_prompt_disjoint": True,
        "fit_may_construct_or_train_candidate": True,
        "guard_may_update_candidate": False,
        "guard_may_choose_refit_or_deletion_variant": False,
        "primary_candidate": _DIRECT,
        "primary_authorization_policy": (
            "direct_bundle_only_standard_gates_plus_block_nrmse_0_015"
        ),
        "diagnostic_candidates": (_REFIT, _DELETION),
        "source_width": GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH,
        "retained_width": GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH,
        "pair_count": GEMMA_MLP_PSEUDO_UNIT_PAIR_COUNT,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "partition_token_contract": _frozen_partition_token_contract(),
        "layer_index": layer_index,
        "layer_id": layer_id,
        "thresholds": thresholds,
        "generator": _frozen_generator_protocol(),
        "calibration_b_tokenized": False,
        "calibration_b_evaluated": False,
        "validation_tokenized": False,
        "validation_evaluated": False,
        "test_tokenized": False,
        "test_evaluated": False,
        "heldout_ledger_created": False,
        "tokenizer": _tokenizer_provenance(tokenizer),
        "library_versions": _library_versions(),
    }
    fit_report = {
        "partition": _partition_binding(corpus.fit),
        "tokenized_stream": copy.deepcopy(dict(fit_value.stream)),
        "tokenized_stream_contract": copy.deepcopy(
            dict(fit_value.contract)
        ),
        "structured_training_batches": (
            built_value.structured_training_batches
        ),
        "score_collection": copy.deepcopy(
            dict(built_value.score_report)
        ),
        "plan_sha256": built_value.plan_report["plan_sha256"],
        "candidate_fingerprints_frozen_before_guard": copy.deepcopy(
            dict(frozen_fingerprints)
        ),
    }
    guard_report = {
        "partition": _partition_binding(corpus.guard),
        "tokenized_stream": copy.deepcopy(dict(guard_value.stream)),
        "tokenized_stream_contract": copy.deepcopy(
            dict(guard_value.contract)
        ),
        "candidate_fingerprints_before": copy.deepcopy(
            dict(frozen_fingerprints)
        ),
        "candidate_fingerprints_after": _candidate_fingerprints(
            built_value.executors
        ),
        "candidate_mutation_observed": False,
        "evaluation": evaluation,
        "primary_passed": gates["primary_passed"] is True,
    }
    primary_passed = gates["primary_passed"] is True
    scientific_status = {
        "outcome": (
            "calibration_a_fit_guard_candidate_built"
            if primary_passed
            else "rejected_on_calibration_a_guard"
        ),
        "calibration_a_fit_completed": True,
        "calibration_a_guard_opened_after_candidate_freeze": True,
        "calibration_a_guard_passed": primary_passed,
        "direct_bundle_is_preregistered_primary": True,
        "refit_and_deletion_are_diagnostic_only": True,
        "candidate_strict_roundtrip_verified": primary_passed,
        "ready_for_one_fresh_calibration_b": primary_passed,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "scientific_compression_success": False,
        "parameter_reduction_measured": True,
        "analytic_mac_reduction_measured": True,
        "latency_or_kernel_speed_claim": False,
    }
    metadata_payload: dict[str, object] = {
        "scientific_status": scientific_status,
        "model": model_metadata,
        "protocol": protocol,
        "parent": parent_binding,
        "calibration_a_fit": fit_report,
        "calibration_a_guard": guard_report,
        "pipeline": copy.deepcopy(dict(built_value.pipeline_report)),
        "deletion_baseline": copy.deepcopy(
            dict(built_value.deletion_report)
        ),
        "resource_report": resources,
    }
    _assert_no_tensor(metadata_payload, label="pseudo-unit metadata")
    if not primary_passed:
        _write_failed_guard_diagnostic(
            resolved_output,
            metadata_payload,
        )
        raise RuntimeError(
            "direct pseudo-unit candidate failed frozen A-guard; no .pt "
            f"artifact was written; see {report_path}"
        )

    payload = {
        "schema": STRUCTURED_MLP_PSEUDO_UNIT_A_SCHEMA,
        "format_version": STRUCTURED_MLP_PSEUDO_UNIT_A_FORMAT_VERSION,
        "contains_source_model_weights": False,
        "contains_full_parent_executor_state": False,
        "contains_compressed_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_fisher_taylor_scores": False,
        **metadata_payload,
        "executor": copy.deepcopy(
            dict(built_value.direct_artifact_state)
        ),
    }
    _validate_tensor_locations(payload)
    scientific_digest = _payload_sha256(payload)
    report = _build_json_report(
        payload,
        tensor_file=resolved_output.name,
        scientific_payload_sha256=scientific_digest,
        main_artifact_written=True,
    )
    report_digest = _report_sha256(report)
    artifact = {
        **payload,
        "scientific_payload_sha256": scientific_digest,
        "report_sha256": report_digest,
    }
    published = _publish_artifact_pair(
        resolved_output,
        artifact,
        report,
        load_published=lambda: (
            load_gemma3_structured_mlp_pseudo_unit_a_artifact(
                resolved_output,
                map_location=device,
            )
        ),
    )
    if not isinstance(published, dict):
        raise RuntimeError("published pseudo-unit artifact did not reload")
    return published


def load_gemma3_structured_mlp_pseudo_unit_a_artifact(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly restore a direct-bundle candidate that passed v9 A-guard."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _OUTER_FIELDS
        or raw.get("schema") != STRUCTURED_MLP_PSEUDO_UNIT_A_SCHEMA
        or raw.get("format_version")
        != STRUCTURED_MLP_PSEUDO_UNIT_A_FORMAT_VERSION
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
        raise ValueError("pseudo-unit A artifact header is invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    _validate_tensor_locations(payload)
    if _payload_sha256(payload) != raw["scientific_payload_sha256"]:
        raise ValueError("pseudo-unit scientific payload digest mismatch")
    executor_state = raw.get("executor")
    if not isinstance(executor_state, Mapping):
        raise ValueError("pseudo-unit executor state is missing")
    executor = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        executor_state,
        map_location=map_location,
    )
    if (
        executor.owns_source_model_weights
        or executor.config.transformer.feed_forward.intermediate_width
        != GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH
    ):
        raise ValueError("pseudo-unit executor schema is invalid")

    protocol = raw.get("protocol")
    parent = raw.get("parent")
    fit = raw.get("calibration_a_fit")
    guard = raw.get("calibration_a_guard")
    pipeline = raw.get("pipeline")
    deletion = raw.get("deletion_baseline")
    resources = raw.get("resource_report")
    status = raw.get("scientific_status")
    model = raw.get("model")
    if not all(
        isinstance(value, Mapping)
        for value in (
            protocol,
            parent,
            fit,
            guard,
            pipeline,
            deletion,
            resources,
            status,
            model,
        )
    ):
        raise ValueError("pseudo-unit artifact bindings are invalid")
    fit_stream = fit.get("tokenized_stream")  # type: ignore[union-attr]
    guard_stream = guard.get("tokenized_stream")  # type: ignore[union-attr]
    before = guard.get(  # type: ignore[union-attr]
        "candidate_fingerprints_before"
    )
    after = guard.get(  # type: ignore[union-attr]
        "candidate_fingerprints_after"
    )
    evaluation = guard.get("evaluation")  # type: ignore[union-attr]
    thresholds = protocol.get("thresholds")  # type: ignore[union-attr]
    source_corpus = protocol.get("corpus")  # type: ignore[union-attr]
    fit_partition = protocol.get(  # type: ignore[union-attr]
        "calibration_a_fit_partition"
    )
    guard_partition = protocol.get(  # type: ignore[union-attr]
        "calibration_a_guard_partition"
    )
    variants = pipeline.get("variants")  # type: ignore[union-attr]
    direct_variant = (
        variants.get("direct") if isinstance(variants, Mapping) else None
    )
    pipeline_status = pipeline.get("status")  # type: ignore[union-attr]
    direct_fingerprint = executor.execution_fingerprint()
    _validate_artifact_candidate_fingerprint_bindings(
        fit=fit,  # type: ignore[arg-type]
        guard=guard,  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        deletion=deletion,  # type: ignore[arg-type]
        direct_fingerprint=direct_fingerprint,
    )
    if (
        protocol.get("primary_candidate") != _DIRECT  # type: ignore[union-attr]
        or protocol.get("diagnostic_candidates")  # type: ignore[union-attr]
        != (_REFIT, _DELETION)
        or protocol.get("layer_index") != DEFAULT_LAYER_INDEX  # type: ignore[union-attr]
        or protocol.get("maximum_tokenized_length")  # type: ignore[union-attr]
        != DEFAULT_MAX_LENGTH
        or protocol.get("tokenization_batch_size")  # type: ignore[union-attr]
        != DEFAULT_TOKENIZATION_BATCH_SIZE
        or protocol.get("generator")  # type: ignore[union-attr]
        != _frozen_generator_protocol()
        or protocol.get("partition_token_contract")  # type: ignore[union-attr]
        != _frozen_partition_token_contract()
        or protocol.get("source_width")  # type: ignore[union-attr]
        != GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH
        or protocol.get("retained_width")  # type: ignore[union-attr]
        != GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH
        or protocol.get("pair_count")  # type: ignore[union-attr]
        != GEMMA_MLP_PSEUDO_UNIT_PAIR_COUNT
        or thresholds != _standard_thresholds()
        or not isinstance(source_corpus, Mapping)
        or source_corpus.get("corpus_id")
        != STRUCTURED_MLP_PSEUDO_UNIT_CORPUS_ID
        or source_corpus.get("counts") != _EXPECTED_ROLE_COUNTS
        or not isinstance(fit_partition, Mapping)
        or not isinstance(guard_partition, Mapping)
        or fit_partition.get("name") != "fit"
        or guard_partition.get("name") != "guard"
        or fit.get("partition") != fit_partition  # type: ignore[union-attr]
        or guard.get("partition") != guard_partition  # type: ignore[union-attr]
        or source_corpus.get("calibration_a_partitions")
        != {
            "fit": fit_partition,
            "guard": guard_partition,
            "family_disjoint": True,
            "prompt_disjoint": True,
            "union_covers_calibration_a": True,
        }
        or protocol.get(  # type: ignore[union-attr]
            "guard_may_update_candidate"
        )
        is not False
        or protocol.get(  # type: ignore[union-attr]
            "guard_may_choose_refit_or_deletion_variant"
        )
        is not False
        or protocol.get(  # type: ignore[union-attr]
            "calibration_b_tokenized"
        )
        is not False
        or protocol.get(  # type: ignore[union-attr]
            "calibration_b_evaluated"
        )
        is not False
        or protocol.get("validation_tokenized") is not False  # type: ignore[union-attr]
        or protocol.get("validation_evaluated") is not False  # type: ignore[union-attr]
        or protocol.get("test_tokenized") is not False  # type: ignore[union-attr]
        or protocol.get("test_evaluated") is not False  # type: ignore[union-attr]
        or parent.get("artifact_format_version") != 5  # type: ignore[union-attr]
        or parent.get("layer_index") != DEFAULT_LAYER_INDEX  # type: ignore[union-attr]
        or parent.get(  # type: ignore[union-attr]
            "fresh_v9_fit_is_independent_of_parent_calibration_a"
        )
        is not True
        or parent.get("model_resolved_commit")  # type: ignore[union-attr]
        != model.get("resolved_commit")  # type: ignore[union-attr]
        or not isinstance(fit_stream, Mapping)
        or fit_stream.get("split") != "calibration_a_fit"
        or not _is_sha256(fit_stream.get("serialized_sha256"))
        or not isinstance(guard_stream, Mapping)
        or guard_stream.get("split") != "calibration_a_guard"
        or not _is_sha256(guard_stream.get("serialized_sha256"))
        or fit_stream.get("serialized_sha256")
        == guard_stream.get("serialized_sha256")
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or before != after
        or before.get(_DIRECT) != direct_fingerprint
        or guard.get(  # type: ignore[union-attr]
            "candidate_mutation_observed"
        )
        is not False
        or guard.get("primary_passed") is not True  # type: ignore[union-attr]
        or not isinstance(evaluation, Mapping)
        or not isinstance(evaluation.get("gates"), Mapping)
        or evaluation["gates"].get("primary_passed") is not True
        or not isinstance(pipeline_status, Mapping)
        or pipeline_status.get("direct_bundle_is_primary") is not True
        or pipeline_status.get("global_down_refit_is_ablation") is not True
        or not isinstance(direct_variant, Mapping)
        or direct_variant.get("execution_fingerprint")
        != direct_fingerprint
        or status.get("outcome")  # type: ignore[union-attr]
        != "calibration_a_fit_guard_candidate_built"
        or status.get(  # type: ignore[union-attr]
            "calibration_a_guard_passed"
        )
        is not True
        or status.get(  # type: ignore[union-attr]
            "ready_for_one_fresh_calibration_b"
        )
        is not True
        or status.get(  # type: ignore[union-attr]
            "scientific_compression_success"
        )
        is not False
    ):
        raise ValueError("pseudo-unit fit/guard binding is invalid")
    assert isinstance(evaluation, Mapping)
    assert isinstance(thresholds, Mapping)
    expected_gates = _guard_gate_report(
        evaluation,
        thresholds=thresholds,  # type: ignore[arg-type]
    )
    if evaluation.get("gates") != expected_gates:
        raise ValueError("pseudo-unit guard gates do not recompute")
    resource_candidates = resources.get("candidates")  # type: ignore[union-attr]
    direct_resources = (
        resource_candidates.get(_DIRECT)
        if isinstance(resource_candidates, Mapping)
        else None
    )
    if (
        not isinstance(direct_resources, Mapping)
        or direct_resources.get("retained_intermediate_width")
        != GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH
        or int(direct_resources.get("removed_layer_parameters", 0)) <= 0
    ):
        raise ValueError("pseudo-unit resource binding is invalid")

    report_path = source.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = json.loads(
        json.dumps(
            _build_json_report(
                payload,
                tensor_file=source.name,
                scientific_payload_sha256=raw[
                    "scientific_payload_sha256"
                ],
                main_artifact_written=True,
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
        raise ValueError("pseudo-unit JSON report is invalid")
    return {
        "model": copy.deepcopy(model),
        "protocol": copy.deepcopy(protocol),
        "parent": copy.deepcopy(parent),
        "executor": executor,
        "calibration_a_fit": copy.deepcopy(fit),
        "calibration_a_guard": copy.deepcopy(guard),
        "pipeline": copy.deepcopy(pipeline),
        "deletion_baseline": copy.deepcopy(deletion),
        "resource_report": copy.deepcopy(resources),
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
            "Fit the 2048-to-1920 direct pseudo-unit bundle on v9 A-fit "
            "and evaluate its frozen v9 A-guard."
        )
    )
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--family-manifest", required=True)
    parser.add_argument("--corpus-audit", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir")
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--generator-steps",
        type=int,
        default=DEFAULT_GENERATOR_STEPS,
    )
    parser.add_argument(
        "--generator-learning-rate",
        type=float,
        default=DEFAULT_GENERATOR_LEARNING_RATE,
    )
    parser.add_argument(
        "--generator-minibatch-rows",
        type=int,
        default=DEFAULT_GENERATOR_MINIBATCH_ROWS,
    )
    parser.add_argument(
        "--generator-gradient-clip-norm",
        type=float,
        default=DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--jacobian-floor-fraction",
        type=float,
        default=DEFAULT_JACOBIAN_FLOOR_FRACTION,
    )
    parser.add_argument(
        "--down-ridge",
        type=float,
        default=DEFAULT_DOWN_RIDGE,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_gemma3_structured_mlp_pseudo_unit_a_experiment(
        parent_artifact_path=arguments.parent_artifact,
        prompt_splits_path=arguments.prompt_splits,
        family_manifest_path=arguments.family_manifest,
        corpus_audit_path=arguments.corpus_audit,
        revision=arguments.revision,
        output=arguments.output,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        layer_index=arguments.layer_index,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        generator_steps=arguments.generator_steps,
        generator_learning_rate=arguments.generator_learning_rate,
        generator_minibatch_rows=arguments.generator_minibatch_rows,
        generator_gradient_clip_norm=(
            arguments.generator_gradient_clip_norm
        ),
        jacobian_floor_fraction=arguments.jacobian_floor_fraction,
        down_ridge=arguments.down_ridge,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(
        json.dumps(
            {
                "scientific_status": result["scientific_status"],
                "metadata": result["metadata"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CalibrationAFamilyPartition",
    "GEMMA_MLP_PSEUDO_UNIT_PAIR_COUNT",
    "GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH",
    "GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH",
    "STRUCTURED_MLP_PSEUDO_UNIT_A_FORMAT_VERSION",
    "STRUCTURED_MLP_PSEUDO_UNIT_A_SCHEMA",
    "STRUCTURED_MLP_PSEUDO_UNIT_CORPUS_ID",
    "build_parser",
    "load_gemma3_structured_mlp_pseudo_unit_a_artifact",
    "main",
    "run_gemma3_structured_mlp_pseudo_unit_a_experiment",
    "validate_structured_v9_pseudo_unit_corpus_preflight",
    "validate_v9_calibration_a_partitions",
]


if __name__ == "__main__":
    raise SystemExit(main())
