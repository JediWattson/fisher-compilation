"""Train and gate a source-free, Gemma-shaped replacement for one layer.

This experiment is deliberately a fidelity rung.  The learned executor has
the source layer's native attention and feed-forward geometry, so parameter
and logical-MAC ratios are reported as diagnostics and never used as fidelity
gates.  A successful run establishes that the compiler can regenerate one
Gemma layer from calibration-A activations and suffix behavior without
retaining source tensors or calling the source layer at execution time.  It
does not establish compression, decode support, multi-layer stability, or
model-level viability.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, LayerSpec, ModelAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
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
    DEFAULT_FISHER_FLOOR,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_GROUND_TRUTH_WEIGHT,
    DEFAULT_LAYER_INDEX,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOCAL_WARMUP_STEPS,
    DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS,
    DEFAULT_MINIMUM_FISHER_ROWS,
    DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS,
    DEFAULT_MINIMUM_LENGTH_BUCKETS,
    DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS,
    DEFAULT_NLL_ATOL,
    DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    DEFAULT_RIDGE_SCALE_FLOOR,
    DEFAULT_TEACHER_KL_MAX,
    DEFAULT_TEACHER_KL_WEIGHT,
    DEFAULT_TOP1_MIN,
    DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_WEIGHT_DECAY,
    FAMILY_STATUS,
    PROMPT_STATUS,
    PromptFamilyManifest,
    _assert_source_independence,
    _assert_tokenized_content_disjointness,
    _direct_gates,
    _finite,
    _full_width_structural_probes,
    _require_complete_middle_layer_demand,
    _require_prompt_protocol,
    _source_accounting_manifest,
    _tokenized_stream_contract,
    _tracked_prompt_exclusion_audit,
    load_prompt_family_manifest,
)
from .gemma3_gated_executor_experiment import (
    _BoundaryBatch,
    _aggregate_direct_examples,
    _materialize_split,
    _source_block_macs,
    _source_block_static,
)
from .gemma3_rotated_span_executor_experiment import (
    _aggregate_behavior_with_kl,
    _behavior_examples_with_kl,
    _behavior_gates,
    _direct_rows,
    _run_native_stack,
    _run_replacement_with_call_audit,
    _run_suffix_from_boundary,
    _selected_training_positions,
    _tensor_sha256,
)
from .gemma3_stability_experiment import (
    Gemma3PromptSplits,
    _library_versions,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .structured_layer_distillation import (
    StructuredLayerDistillationScales,
    StructuredLayerDistillationWeights,
    StructuredLayerProvenance,
    StructuredLayerTargets,
    StructuredOutputFisherMetric,
    capture_structured_layer_targets,
    estimate_structured_layer_scales,
    initialize_structured_rmsnorms_from_targets_,
    structured_layer_distillation_loss,
    structured_layer_provenance,
)
from .structured_operator_bootstrap import (
    DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS,
    DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION,
    DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY,
    DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL,
    DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE,
    STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY,
    STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
    STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION,
    STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
    StructuredOperatorCaptureBatch,
    StructuredOperatorIdentityBatch,
    StructuredOperatorRowSelection,
    bootstrap_structured_operator_executor_,
    select_structured_operator_rows,
    structured_operator_coefficient_sha256,
    structured_operator_site_schema,
    structured_operator_site_schema_sha256,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


DEFAULT_STRUCTURED_LOSS_SCALE = 1.0
DEFAULT_OUTPUT_FISHER_WEIGHT = 0.10
DEFAULT_COORDINATE_LOSS_WEIGHT = 1.0
DEFAULT_ENERGY_LOSS_WEIGHT = 1.0
DEFAULT_RMSNORM_INITIALIZATION = (
    "calibration_a_activation_pair_coordinate_least_squares_v1"
)
DEFAULT_RELATIVE_MEDIAN_SCALE_FLOOR = 1.0
DEFAULT_OPTIMIZATION_SEED = 91_104
DEFAULT_BRANCH_DELTA_NRMSE_MAX = DEFAULT_BLOCK_DELTA_NRMSE_MAX
DEFAULT_BRANCH_DELTA_COSINE_MIN = DEFAULT_BLOCK_DELTA_COSINE_MIN
DEFAULT_NATIVE_PARITY_TOLERANCE = 1e-5

_ARTIFACT_SCHEMA = "fisher_graph.gemma3_structured_single_layer_executor"
_ARTIFACT_FORMAT_VERSION = 5
_RMS_ARTIFACT_FORMAT_VERSION = 4
_SUPPORTED_ARTIFACT_FORMAT_VERSIONS = {3, 4, 5}
_CORPUS_AUDIT_DOMAIN = b"fisher_graph.structured_corpus_audit.v1\0"
_CORPUS_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
_CORPUS_LEXICAL_LENGTH_SCHEMA = (
    "fisher_graph.structured_corpus_lexical_length_audit"
)
_CORPUS_LEXICAL_LENGTH_ALGORITHM = (
    "unicode_nfkc_casefold_unicode_word_apostrophe_v1"
)
_CORPUS_MINIMUM_PROMPTS_PER_LENGTH_BAND = 4
_CORPUS_LENGTH_BANDS = {
    "micro": (None, 15),
    "compact": (16, 32),
    "medium": (33, 96),
    "long": (97, None),
}
_FAMILY_ID_DOMAIN = b"fisher_graph.structured_family_id.v1\0"
_FAMILY_SET_DOMAIN = b"fisher_graph.structured_family_set.v1\0"
_PROMPT_FAMILY_PAIR_DOMAIN = (
    b"fisher_graph.structured_prompt_family_pairs.v1\0"
)
_CALIBRATION_B_IDENTITY_DOMAIN = (
    b"fisher_graph.structured_calibration_b_identity.v1\0"
)
_CALIBRATION_B_CLAIM_DOMAIN = (
    b"fisher_graph.structured_calibration_b_claim.v1\0"
)
_RECIPE_DOMAIN = b"fisher_graph.structured_training_recipe.v1\0"
_THRESHOLDS_DOMAIN = b"fisher_graph.structured_thresholds.v1\0"
_HELDOUT_CLAIM_SCHEMA = "fisher_graph.structured_heldout_open_claim"
_OPERATOR_BOOTSTRAP_FITTING_METHOD = (
    "activation_only_structured_operator_bootstrap"
)
_PRIMARY = "structured_source_visibility"
_CONTROL = "attention_output_disabled_control"
_CANDIDATES = (_PRIMARY, _CONTROL)
_SCALE_NAMES = (
    "normalized_attention_input",
    "attention_operator_output",
    "attention_delta",
    "post_attention",
    "normalized_feed_forward_input",
    "feed_forward_operator_output",
    "feed_forward_delta",
    "output",
)
_OUTER_FIELDS = {
    "schema",
    "format_version",
    "contains_model_weights",
    "contains_executor_weights",
    "contains_prompt_text",
    "contains_tokenizer_state",
    "contains_teacher_targets",
    "contains_source_derived_statistics",
    "scientific_status",
    "model",
    "protocol",
    "executors",
    "training",
    "selection",
    "validation",
    "scientific_payload_sha256",
    "report_sha256",
}


def default_gemma3_structured_single_layer_output(
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = DEFAULT_LAYER_INDEX,
) -> Path:
    """Return an ignored, model/layer-specific output path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layer-{layer_index}-structured-single-layer-executor.pt"
    )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    format_version = payload.get("format_version")
    if format_version not in _SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError("unsupported structured payload format version")
    digest = hashlib.sha256()
    digest.update(
        (
            "fisher_graph.gemma3_structured_single_layer_executor_"
            f"payload.v{format_version}\0"
        ).encode("ascii")
    )
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    format_version = report.get("format_version")
    if format_version not in _SUPPORTED_ARTIFACT_FORMAT_VERSIONS:
        raise ValueError("unsupported structured report format version")
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(
        (
            "fisher_graph.gemma3_structured_single_layer_executor_"
            f"report.v{format_version}\0"
        ).encode("ascii")
    )
    digest.update(encoded)
    return digest.hexdigest()


def _sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _ordered_hash_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _family_id_sha256(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("family id must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(_FAMILY_ID_DOMAIN)
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _format4_family_binding(
    prompts: Gemma3PromptSplits,
    families: PromptFamilyManifest,
) -> dict[str, object]:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    prompt_metadata = prompts.metadata()
    prompt_hashes = prompt_metadata["per_prompt_sha256"]
    assert isinstance(prompt_hashes, Mapping)
    family_hashes = {
        role: [
            _family_id_sha256(value)
            for value in getattr(families, role)
        ]
        for role in roles
    }
    pair_hashes = {
        role: _sha256(
            list(
                zip(
                    prompt_hashes[role],
                    family_hashes[role],
                    strict=True,
                )
            ),
            domain=_PROMPT_FAMILY_PAIR_DOMAIN,
        )
        for role in roles
    }
    return {
        "algorithm": "domain_hashed_family_ids_and_ordered_prompt_pairs_v1",
        "per_prompt_family_sha256": family_hashes,
        "ordered_hashed_family_sha256": {
            role: _ordered_hash_digest(family_hashes[role])
            for role in roles
        },
        "ordered_prompt_family_pairs_sha256": pair_hashes,
    }


def _calibration_b_identity(prompt_hashes: Sequence[str]) -> str:
    hashes = list(prompt_hashes)
    if (
        not hashes
        or len(set(hashes)) != len(hashes)
        or any(not _is_sha256(value) for value in hashes)
    ):
        raise ValueError("calibration-B prompt hashes are invalid")
    return _sha256(
        {
            "count": len(hashes),
            "sorted_prompt_sha256": sorted(hashes),
        },
        domain=_CALIBRATION_B_IDENTITY_DOMAIN,
    )


def _calibration_b_claim_path(
    ledger_dir: Path | str,
    prompt_hashes: Sequence[str],
) -> Path:
    return Path(ledger_dir) / f"{_calibration_b_identity(prompt_hashes)}.json"


def _claim_payload_sha256(value: Mapping[str, object]) -> str:
    return _sha256(value, domain=_CALIBRATION_B_CLAIM_DOMAIN)


def _exclusive_calibration_b_claim(
    path: Path,
    *,
    prompt_hashes: Sequence[str],
    family_hashes: Sequence[str],
    prompt_family_pair_sha256: str,
    corpus_audit: Mapping[str, object] | None,
    prompt_fixture_file_sha256: str,
    family_manifest_file_sha256: str,
    resolved_commit: str,
    layer_id: str,
    training_recipe: Mapping[str, object],
    thresholds: Mapping[str, object],
    executors: Mapping[str, StructuredTransformerLayerExecutor],
) -> dict[str, object]:
    if set(executors) != set(_CANDIDATES):
        raise ValueError("calibration-B claim requires both executors")
    hashes = list(prompt_hashes)
    hashed_families = list(family_hashes)
    if (
        len(hashes) != len(hashed_families)
        or any(not _is_sha256(value) for value in hashed_families)
        or not _is_sha256(prompt_family_pair_sha256)
    ):
        raise ValueError("calibration-B family binding is invalid")
    audit_payload_sha256 = (
        None
        if corpus_audit is None
        else corpus_audit.get("audit_payload_sha256")
    )
    if audit_payload_sha256 is not None and not _is_sha256(
        audit_payload_sha256
    ):
        raise ValueError("calibration-B audit binding is invalid")
    payload: dict[str, object] = {
        "schema": _HELDOUT_CLAIM_SCHEMA,
        "format_version": 1,
        "state": "claimed_before_tokenization",
        "role": "calibration_b",
        "role_prompt_set_sha256": _calibration_b_identity(hashes),
        "role_prompt_count": len(hashes),
        "ordered_prompt_sha256": _ordered_hash_digest(hashes),
        "family_set_sha256": _sha256(
            sorted(set(hashed_families)),
            domain=_FAMILY_SET_DOMAIN,
        ),
        "ordered_prompt_family_pairs_sha256": (
            prompt_family_pair_sha256
        ),
        "corpus_audit_payload_sha256": audit_payload_sha256,
        "prompt_fixture_file_sha256": prompt_fixture_file_sha256,
        "family_manifest_file_sha256": family_manifest_file_sha256,
        "model_resolved_commit": resolved_commit,
        "layer_id": layer_id,
        "training_recipe_sha256": _sha256(
            training_recipe,
            domain=_RECIPE_DOMAIN,
        ),
        "thresholds_sha256": _sha256(
            thresholds,
            domain=_THRESHOLDS_DOMAIN,
        ),
        "executor_fingerprints": {
            name: executors[name].execution_fingerprint()
            for name in _CANDIDATES
        },
    }
    claim = {
        **payload,
        "claim_payload_sha256": _claim_payload_sha256(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                claim,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            "calibration B was already claimed; refusing heldout reuse: "
            f"{path}"
        ) from error
    return claim


def _normalized_corpus_word_count(value: str) -> int:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return len(_CORPUS_WORD_PATTERN.findall(normalized))


def _corpus_lexical_length_audit(
    counts_by_role: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    if (
        set(counts_by_role) != set(roles)
        or any(
            not isinstance(counts_by_role[role], (list, tuple))
            or any(
                type(value) is not int or value <= 0
                for value in counts_by_role[role]
            )
            for role in roles
        )
    ):
        raise ValueError(
            "structured corpus approximate word counts are invalid"
        )
    band_counts = {
        role: {
            band: sum(
                (
                    minimum is None or value >= minimum
                )
                and (
                    maximum is None or value <= maximum
                )
                for value in counts_by_role[role]
            )
            for band, (minimum, maximum) in _CORPUS_LENGTH_BANDS.items()
        }
        for role in roles
    }
    passed = all(
        count >= _CORPUS_MINIMUM_PROMPTS_PER_LENGTH_BAND
        for role_counts in band_counts.values()
        for count in role_counts.values()
    )
    return {
        "schema": _CORPUS_LEXICAL_LENGTH_SCHEMA,
        "format_version": 1,
        "normalized_word_count_algorithm": (
            _CORPUS_LEXICAL_LENGTH_ALGORITHM
        ),
        "bands": {
            band: {
                "minimum_inclusive": minimum,
                "maximum_inclusive": maximum,
            }
            for band, (minimum, maximum) in _CORPUS_LENGTH_BANDS.items()
        },
        "minimum_prompts_per_band_per_role": (
            _CORPUS_MINIMUM_PROMPTS_PER_LENGTH_BAND
        ),
        "counts_by_role": band_counts,
        "all_roles_cover_all_bands": passed,
    }


def _corpus_audit_binding(
    path: Path | str | None,
    *,
    prompts: Gemma3PromptSplits,
    prompt_path: Path,
    family_path: Path,
) -> dict[str, object] | None:
    """Validate and bind a frozen corpus audit before model loading."""

    if path is None:
        return None
    audit_path = Path(path)
    raw = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("structured corpus audit must be a mapping")
    roles = ("calibration_a", "calibration_b", "validation", "test")
    counts = raw.get("counts")
    hashes = raw.get("prompt_sha256_by_role")
    actual_prompts = {
        role: tuple(getattr(prompts, role))
        for role in roles
    }
    actual_counts = {
        role: len(values)
        for role, values in actual_prompts.items()
    }
    actual_hashes = {
        role: [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in values
        ]
        for role, values in actual_prompts.items()
    }
    actual_word_counts = {
        role: [
            _normalized_corpus_word_count(prompt)
            for prompt in values
        ]
        for role, values in actual_prompts.items()
    }
    generator = raw.get("generator")
    generator_sha256 = raw.get("generator_sha256")
    if (
        not isinstance(raw.get("schema"), str)
        or not raw["schema"]
        or type(raw.get("format_version")) is not int
        or raw["format_version"] <= 0
        or not isinstance(counts, Mapping)
        or dict(counts) != actual_counts
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(roles)
        or any(hashes[role] != actual_hashes[role] for role in roles)
        or raw.get("unique_prompt_count") != sum(actual_counts.values())
        or raw.get("cross_role_family_overlap_count") != 0
        or raw.get("tokenizer_or_model_accessed") is not False
        or raw.get("test_model_evaluated") is not False
        or raw.get("corpus_frozen_before_model_load") is not True
        or raw.get("prior_local_exact_prompt_overlap_count") != 0
        or not isinstance(generator, str)
        or not generator
        or Path(generator).name != generator
        or not _is_sha256(generator_sha256)
    ):
        raise ValueError("structured corpus audit is invalid")
    lexical_length_audit = None
    if raw["format_version"] >= 2:
        recorded_word_counts = raw.get(
            "approximate_word_counts_by_role"
        )
        if (
            not isinstance(recorded_word_counts, Mapping)
            or set(recorded_word_counts) != set(roles)
            or any(
                not isinstance(recorded_word_counts[role], list)
                or any(
                    type(value) is not int
                    for value in recorded_word_counts[role]
                )
                or recorded_word_counts[role]
                != actual_word_counts[role]
                for role in roles
            )
        ):
            raise ValueError(
                "structured corpus approximate word counts do not match "
                "the normalized prompts"
            )
        lexical_length_audit = _corpus_lexical_length_audit(
            actual_word_counts
        )
        if lexical_length_audit[
            "all_roles_cover_all_bands"
        ] is not True:
            raise ValueError(
                "structured corpus lacks conservative lexical length "
                "breadth in every role"
            )
    generator_path = audit_path.parent / generator
    if (
        not generator_path.is_file()
        or _file_sha256(generator_path) != generator_sha256
    ):
        raise ValueError("structured corpus generator binding is invalid")
    payload = copy.deepcopy(dict(raw))
    binding = {
        "audit_file_sha256": _file_sha256(audit_path),
        "audit_payload_sha256": _sha256(
            payload,
            domain=_CORPUS_AUDIT_DOMAIN,
        ),
        "generator": generator,
        "generator_file_sha256": generator_sha256,
        "prompt_fixture_file_sha256": _file_sha256(prompt_path),
        "family_manifest_file_sha256": _file_sha256(family_path),
        "payload": payload,
    }
    if lexical_length_audit is not None:
        binding["lexical_length_audit"] = lexical_length_audit
    return binding


def _validate_corpus_audit_binding(
    value: object,
    *,
    protocol: Mapping[str, object],
) -> bool:
    if value is None:
        return False
    base_fields = {
        "audit_file_sha256",
        "audit_payload_sha256",
        "generator",
        "generator_file_sha256",
        "prompt_fixture_file_sha256",
        "family_manifest_file_sha256",
        "payload",
    }
    if not isinstance(value, Mapping):
        raise ValueError("structured corpus audit binding fields are invalid")
    payload = value.get("payload")
    payload_format_version = (
        payload.get("format_version")
        if isinstance(payload, Mapping)
        else None
    )
    fields = (
        base_fields | {"lexical_length_audit"}
        if (
            type(payload_format_version) is int
            and payload_format_version >= 2
        )
        else base_fields
    )
    if set(value) != fields:
        raise ValueError("structured corpus audit binding fields are invalid")
    prompt_metadata = protocol.get("prompt_splits")
    protocol_counts = (
        prompt_metadata.get("counts")
        if isinstance(prompt_metadata, Mapping)
        else None
    )
    roles = {"calibration_a", "calibration_b", "validation", "test"}
    counts_are_valid = (
        isinstance(protocol_counts, Mapping)
        and set(protocol_counts) == roles
        and all(
            type(protocol_counts[role]) is int
            and protocol_counts[role] > 0
            for role in roles
        )
    )
    protocol_total = (
        sum(protocol_counts.values()) if counts_are_valid else -1
    )
    if (
        not isinstance(payload, Mapping)
        or not isinstance(prompt_metadata, Mapping)
        or not counts_are_valid
        or not _is_sha256(value["audit_file_sha256"])
        or value["audit_payload_sha256"]
        != _sha256(payload, domain=_CORPUS_AUDIT_DOMAIN)
        or not isinstance(value["generator"], str)
        or not value["generator"]
        or Path(value["generator"]).name != value["generator"]
        or not _is_sha256(value["generator_file_sha256"])
        or payload.get("generator") != value["generator"]
        or payload.get("generator_sha256")
        != value["generator_file_sha256"]
        or value["prompt_fixture_file_sha256"]
        != protocol.get("prompt_fixture_file_sha256")
        or value["family_manifest_file_sha256"]
        != protocol.get("family_manifest_file_sha256")
        or payload.get("counts") != protocol_counts
        or payload.get("unique_prompt_count")
        != protocol_total
        or payload.get("cross_role_family_overlap_count") != 0
        or payload.get("tokenizer_or_model_accessed") is not False
        or payload.get("test_model_evaluated") is not False
        or payload.get("corpus_frozen_before_model_load") is not True
        or payload.get("prior_local_exact_prompt_overlap_count") != 0
    ):
        raise ValueError("structured corpus audit binding is invalid")
    hashes = payload.get("prompt_sha256_by_role")
    counts = protocol_counts
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != roles
    ):
        raise ValueError("structured corpus audit prompt hashes are invalid")
    for role in roles:
        role_hashes = hashes[role]
        if (
            not isinstance(role_hashes, list)
            or len(role_hashes) != counts[role]
            or any(not _is_sha256(item) for item in role_hashes)
        ):
            raise ValueError(
                "structured corpus audit prompt hashes are invalid"
            )
    if type(payload_format_version) is not int or payload_format_version <= 0:
        raise ValueError("structured corpus audit format is invalid")
    if payload_format_version >= 2:
        word_counts = payload.get("approximate_word_counts_by_role")
        if (
            not isinstance(word_counts, Mapping)
            or set(word_counts) != roles
            or any(
                not isinstance(word_counts[role], list)
                or len(word_counts[role]) != counts[role]
                or any(
                    type(item) is not int or item <= 0
                    for item in word_counts[role]
                )
                for role in roles
            )
        ):
            raise ValueError(
                "structured corpus approximate word counts are invalid"
            )
        lexical_length_audit = _corpus_lexical_length_audit(
            word_counts  # type: ignore[arg-type]
        )
        if (
            value["lexical_length_audit"] != lexical_length_audit
            or lexical_length_audit[
                "all_roles_cover_all_bands"
            ]
            is not True
        ):
            raise ValueError(
                "structured corpus lexical length audit is invalid"
            )
    return True


def _validate_format4_prompt_family_binding(
    protocol: Mapping[str, object],
    *,
    streams: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    prompt_metadata = _strict_mapping(
        protocol.get("prompt_splits"),
        label="structured prompt split provenance",
        fields={
            "scientific_status",
            "counts",
            "normalized_sha256",
            "per_prompt_sha256",
        },
    )
    counts = _strict_mapping(
        prompt_metadata["counts"],
        label="structured prompt counts",
        fields=set(roles),
    )
    normalized = _strict_mapping(
        prompt_metadata["normalized_sha256"],
        label="structured normalized prompt hashes",
        fields=set(roles),
    )
    raw_prompt_hashes = _strict_mapping(
        prompt_metadata["per_prompt_sha256"],
        label="structured per-prompt hashes",
        fields=set(roles),
    )
    prompt_hashes: dict[str, list[str]] = {}
    all_prompt_hashes: list[str] = []
    for role in roles:
        values = raw_prompt_hashes[role]
        if (
            type(counts[role]) is not int
            or counts[role] <= 0
            or not isinstance(values, list)
            or len(values) != counts[role]
            or any(not _is_sha256(value) for value in values)
            or normalized[role] != _ordered_hash_digest(values)
        ):
            raise ValueError("structured prompt provenance is invalid")
        prompt_hashes[role] = list(values)
        all_prompt_hashes.extend(values)
    if (
        prompt_metadata["scientific_status"] != PROMPT_STATUS
        or len(all_prompt_hashes) != len(set(all_prompt_hashes))
    ):
        raise ValueError("structured prompt disjointness is invalid")
    for role, stream in streams.items():
        if stream.get("source_prompt_sha256") != prompt_hashes[role]:
            raise ValueError(
                "structured tokenized stream prompt binding is invalid"
            )
    if any(
        value in set(prompt_hashes["test"])
        for stream in streams.values()
        for value in stream["source_prompt_sha256"]  # type: ignore[index]
    ):
        raise ValueError("reserved test prompts were tokenized")

    family_metadata = _strict_mapping(
        protocol.get("prompt_families"),
        label="structured prompt family provenance",
        fields={
            "scientific_status",
            "counts",
            "unique_family_counts",
            "ordered_family_sha256",
            "cross_role_overlap_count",
            "algorithm",
            "per_prompt_family_sha256",
            "ordered_hashed_family_sha256",
            "ordered_prompt_family_pairs_sha256",
        },
    )
    if (
        family_metadata["scientific_status"] != FAMILY_STATUS
        or family_metadata["counts"] != counts
        or family_metadata["cross_role_overlap_count"] != 0
        or family_metadata["algorithm"]
        != "domain_hashed_family_ids_and_ordered_prompt_pairs_v1"
    ):
        raise ValueError("structured prompt family provenance is invalid")
    unique_counts = _strict_mapping(
        family_metadata["unique_family_counts"],
        label="structured unique family counts",
        fields=set(roles),
    )
    raw_family_hashes = _strict_mapping(
        family_metadata["per_prompt_family_sha256"],
        label="structured per-prompt family hashes",
        fields=set(roles),
    )
    pair_hashes = _strict_mapping(
        family_metadata["ordered_prompt_family_pairs_sha256"],
        label="structured prompt-family pair hashes",
        fields=set(roles),
    )
    hashed_family_digests = _strict_mapping(
        family_metadata["ordered_hashed_family_sha256"],
        label="structured ordered hashed-family hashes",
        fields=set(roles),
    )
    ordered_family_hashes = _strict_mapping(
        family_metadata["ordered_family_sha256"],
        label="structured ordered family hashes",
        fields=set(roles),
    )
    family_hashes: dict[str, list[str]] = {}
    family_sets: dict[str, set[str]] = {}
    for role in roles:
        values = raw_family_hashes[role]
        if (
            not isinstance(values, list)
            or len(values) != counts[role]
            or any(not _is_sha256(value) for value in values)
            or type(unique_counts[role]) is not int
            or unique_counts[role] != len(set(values))
            or not _is_sha256(ordered_family_hashes[role])
            or hashed_family_digests[role]
            != _ordered_hash_digest(values)
            or pair_hashes[role]
            != _sha256(
                list(
                    zip(
                        prompt_hashes[role],
                        values,
                        strict=True,
                    )
                ),
                domain=_PROMPT_FAMILY_PAIR_DOMAIN,
            )
        ):
            raise ValueError("structured prompt family binding is invalid")
        family_hashes[role] = list(values)
        family_sets[role] = set(values)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if family_sets[left] & family_sets[right]:
                raise ValueError(
                    "structured prompt families overlap across roles"
                )
    return prompt_hashes, family_hashes


def _validate_calibration_b_claim(
    value: object,
    *,
    protocol: Mapping[str, object],
    prompt_hashes: Sequence[str],
    family_hashes: Sequence[str],
    prompt_family_pair_sha256: str,
    executors: Mapping[str, StructuredTransformerLayerExecutor],
    model: Mapping[str, object],
    layer_id: str,
) -> None:
    fields = {
        "schema",
        "format_version",
        "state",
        "role",
        "role_prompt_set_sha256",
        "role_prompt_count",
        "ordered_prompt_sha256",
        "family_set_sha256",
        "ordered_prompt_family_pairs_sha256",
        "corpus_audit_payload_sha256",
        "prompt_fixture_file_sha256",
        "family_manifest_file_sha256",
        "model_resolved_commit",
        "layer_id",
        "training_recipe_sha256",
        "thresholds_sha256",
        "executor_fingerprints",
        "claim_payload_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("structured calibration-B claim fields are invalid")
    payload = {
        key: item
        for key, item in value.items()
        if key != "claim_payload_sha256"
    }
    corpus_audit = protocol.get("corpus_audit")
    audit_digest = (
        None
        if corpus_audit is None
        else corpus_audit.get("audit_payload_sha256")
        if isinstance(corpus_audit, Mapping)
        else object()
    )
    fingerprints = {
        name: executors[name].execution_fingerprint()
        for name in _CANDIDATES
    }
    expected = {
        "schema": _HELDOUT_CLAIM_SCHEMA,
        "format_version": 1,
        "state": "claimed_before_tokenization",
        "role": "calibration_b",
        "role_prompt_set_sha256": _calibration_b_identity(prompt_hashes),
        "role_prompt_count": len(prompt_hashes),
        "ordered_prompt_sha256": _ordered_hash_digest(prompt_hashes),
        "family_set_sha256": _sha256(
            sorted(set(family_hashes)),
            domain=_FAMILY_SET_DOMAIN,
        ),
        "ordered_prompt_family_pairs_sha256": (
            prompt_family_pair_sha256
        ),
        "corpus_audit_payload_sha256": audit_digest,
        "prompt_fixture_file_sha256": protocol.get(
            "prompt_fixture_file_sha256"
        ),
        "family_manifest_file_sha256": protocol.get(
            "family_manifest_file_sha256"
        ),
        "model_resolved_commit": model.get("resolved_commit"),
        "layer_id": layer_id,
        "training_recipe_sha256": _sha256(
            protocol["training_recipe"],
            domain=_RECIPE_DOMAIN,
        ),
        "thresholds_sha256": _sha256(
            protocol["thresholds"],
            domain=_THRESHOLDS_DOMAIN,
        ),
        "executor_fingerprints": fingerprints,
    }
    if (
        payload != expected
        or value["claim_payload_sha256"]
        != _claim_payload_sha256(payload)
    ):
        raise ValueError("structured calibration-B claim binding is invalid")


def _example_ids(
    batch: CalibrationBatch,
    *,
    sequence_offset: int,
) -> tuple[str, ...]:
    if batch.example_ids is not None:
        return batch.example_ids
    return tuple(
        f"sequence-{sequence_offset + index}"
        for index in range(batch.batch_size)
    )


@dataclass(frozen=True, slots=True)
class StructuredTrainingBatch:
    """One calibration-A batch with all native structured teacher targets."""

    batch: CalibrationBatch
    targets: StructuredLayerTargets
    selected_positions: Tensor
    ground_truth_targets: Tensor
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        selected_count = int(self.selected_positions.sum().item())
        if (
            not isinstance(self.batch, CalibrationBatch)
            or not isinstance(self.targets, StructuredLayerTargets)
            or self.targets.block_input.shape[:2]
            != self.batch.valid_positions.shape
            or self.selected_positions.shape
            != self.batch.valid_positions.shape
            or self.selected_positions.dtype is not torch.bool
            or bool(
                (
                    self.selected_positions
                    & ~self.batch.valid_positions.to(
                        device=self.selected_positions.device
                    )
                ).any()
            )
            or selected_count <= 0
            or self.ground_truth_targets.shape != (selected_count,)
            or self.ground_truth_targets.dtype not in (torch.int32, torch.int64)
            or self.targets.teacher_logits is None
            or self.targets.teacher_logits.ndim != 2
            or self.targets.teacher_logits.shape[0] != selected_count
            or len(self.example_ids) != self.batch.batch_size
        ):
            raise ValueError("structured training batch tensors are inconsistent")

    @property
    def block_input(self) -> Tensor:
        return self.targets.block_input

    @property
    def block_output(self) -> Tensor:
        return self.targets.output

    @property
    def teacher_logits(self) -> Tensor:
        assert self.targets.teacher_logits is not None
        return self.targets.teacher_logits


def collect_structured_training_batches(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    layer_id: str,
    positions_per_sequence: int,
) -> tuple[StructuredTrainingBatch, ...]:
    """Capture all eight structured targets in one ordinary forward per batch."""

    if not batches:
        raise ValueError("structured calibration batches cannot be empty")
    provenance = structured_layer_provenance(adapter, layer_id)
    result = []
    sequence_offset = 0
    for batch in batches:
        selected = _selected_training_positions(
            batch,
            positions_per_sequence=positions_per_sequence,
        )
        targets = capture_structured_layer_targets(
            adapter,
            layer_id,
            batch.model_inputs,
            teacher_logit_positions=selected,
            provenance=provenance,
        )
        ids = _example_ids(batch, sequence_offset=sequence_offset)
        sequence_offset += batch.batch_size
        result.append(
            StructuredTrainingBatch(
                batch=batch,
                targets=targets,
                selected_positions=selected,
                ground_truth_targets=batch.targets[selected].detach().clone(),
                example_ids=ids,
            )
        )
    return tuple(result)


def _capture_compact_operator_bootstrap_rows(
    adapter: ModelAdapter,
    training: Sequence[StructuredTrainingBatch],
    *,
    layer: LayerSpec,
    calibration_split_sha256: str,
    requested_rows: int,
) -> tuple[
    tuple[StructuredOperatorCaptureBatch, ...],
    StructuredOperatorRowSelection,
    dict[str, object],
]:
    """Select identities first, then retain one compact activation replay."""

    if not training:
        raise ValueError("operator bootstrap training batches cannot be empty")
    semantics = layer.transformer
    if semantics is None or semantics.operator_sites is None:
        raise ValueError(
            "operator bootstrap requires Gemma operator activation sites"
        )
    identity_batches = tuple(
        StructuredOperatorIdentityBatch(
            valid_positions=item.batch.valid_positions,
            logical_positions=item.targets.sequence.logical_positions,
            example_ids=item.example_ids,
        )
        for item in training
    )
    selection = select_structured_operator_rows(
        identity_batches,
        calibration_split_sha256=calibration_split_sha256,
        layer_id=layer.id,
        requested_rows=requested_rows,
    )
    attention, feed_forward = semantics.stages
    residual_sites = (
        layer.input_site,
        attention.normalized_input_site,
        attention.operator_output_site,
        attention.delta_site,
        attention.output_site,
        feed_forward.normalized_input_site,
        feed_forward.operator_output_site,
        feed_forward.delta_site,
        layer.output_site,
    )
    capture_sites = tuple(
        dict.fromkeys(
            (
                *residual_sites,
                *semantics.operator_sites.values(),
            )
        )
    )
    chunks: dict[str, list[Tensor]] = {
        site: [] for site in capture_sites
    }
    selected_example_ids: list[str] = []
    selected_logical_positions: list[int] = []
    forward_calls = 0
    for item, identity in zip(training, identity_batches, strict=True):
        selected = selection.mask_for(identity)
        rows, columns = selected.nonzero(as_tuple=True)
        if not rows.numel():
            continue
        with torch.no_grad():
            run = adapter.forward(
                item.batch.model_inputs,
                capture_sites=capture_sites,
                retain_gradients=False,
            )
        forward_calls += 1
        if (
            not torch.equal(
                run.sequence.query_valid_mask,
                item.batch.valid_positions.to(
                    device=run.sequence.query_valid_mask.device
                ),
            )
            or not torch.equal(
                run.sequence.logical_positions,
                item.targets.sequence.logical_positions.to(
                    device=run.sequence.logical_positions.device
                ),
            )
        ):
            raise RuntimeError(
                "operator capture replay changed calibration-A identities"
            )
        for site in capture_sites:
            value = run.activations.get(site)
            if not isinstance(value, Tensor):
                raise RuntimeError(
                    f"operator activation site {site!r} was not captured"
                )
            chunks[site].append(
                value[
                    rows.to(device=value.device),
                    columns.to(device=value.device),
                ]
                .detach()
                .to(device="cpu")
                .clone()
            )
        cpu_rows = rows.detach().cpu().tolist()
        cpu_columns = columns.detach().cpu().tolist()
        positions = identity.logical_positions.detach().cpu()
        selected_example_ids.extend(
            identity.example_ids[row] for row in cpu_rows
        )
        selected_logical_positions.extend(
            int(positions[row, column].item())
            for row, column in zip(
                cpu_rows,
                cpu_columns,
                strict=True,
            )
        )
    if (
        not selected_example_ids
        or len(selected_example_ids) != selection.selected_rows
        or any(not site_chunks for site_chunks in chunks.values())
    ):
        raise RuntimeError(
            "operator capture did not retain every selected row"
        )
    compact = StructuredOperatorCaptureBatch(
        activations={
            site: torch.cat(site_chunks, dim=0).unsqueeze(1)
            for site, site_chunks in chunks.items()
        },
        valid_positions=torch.ones(
            len(selected_example_ids),
            1,
            dtype=torch.bool,
        ),
        logical_positions=torch.tensor(
            selected_logical_positions,
            dtype=torch.long,
        ).unsqueeze(1),
        example_ids=tuple(selected_example_ids),
    )
    return (
        (compact,),
        selection,
        {
            "source_model_executed_for_activation_capture": True,
            "source_activation_capture_stream_passes": 1,
            "source_activation_capture_forward_calls": forward_calls,
            "capture_site_count": len(capture_sites),
            "residual_capture_site_count": len(residual_sites),
            "operator_capture_site_count": len(
                semantics.operator_sites.values()
            ),
            "capture_contains_only_selected_rows": True,
            "captured_activation_rows_serialized": False,
            "sufficient_statistics_serialized": False,
            "compiler_source_parameter_tensor_read": False,
            "direct_source_tensor_copy": False,
        },
    )


def compute_structured_activation_fisher(
    adapter: ModelAdapter,
    training: Sequence[StructuredTrainingBatch],
    *,
    plan: object,
) -> tuple[Tensor, dict[str, object]]:
    """Estimate empirical ground-truth-CE activation-gradient second moments.

    The matrix pools per-position gradient outer products over valid boundary
    rows.  This is not an expected model Fisher and does not include
    cross-position blocks.
    """

    if not training:
        raise ValueError("structured Fisher training batches cannot be empty")
    width = int(training[0].block_output.shape[-1])
    matrix = torch.zeros(width, width, dtype=torch.float64)
    selected_target_rows = 0
    valid_boundary_rows = 0
    effective_gradient_rows = 0
    effective_epsilon = torch.finfo(torch.float64).tiny
    for item in training:
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        boundary = item.block_output.detach().clone().requires_grad_(True)
        _, _, logits = _run_suffix_from_boundary(
            adapter,
            item.batch,
            plan=plan,  # type: ignore[arg-type]
            sequence=sequence,
            boundary_output=boundary,
            selected_positions=item.selected_positions,
            full_logits=False,
        )
        if logits is None:
            raise RuntimeError("structured Fisher suffix logits are missing")
        targets = item.ground_truth_targets.to(device=logits.device)
        loss = F.cross_entropy(logits.float(), targets, reduction="sum")
        gradient = torch.autograd.grad(
            loss,
            boundary,
            retain_graph=False,
            create_graph=False,
        )[0]
        valid = item.batch.valid_positions.to(device=gradient.device)
        rows = gradient[valid].detach().to(device="cpu", dtype=torch.float64)
        matrix.add_(rows.T @ rows)
        selected_target_rows += int(item.selected_positions.sum().item())
        valid_boundary_rows += int(rows.shape[0])
        effective_gradient_rows += int(
            (rows.square().sum(dim=-1) > effective_epsilon).sum().item()
        )
    if valid_boundary_rows <= 0:
        raise ValueError("structured activation Fisher has no valid rows")
    if effective_gradient_rows <= 0:
        raise RuntimeError(
            "structured activation Fisher has no effective gradient rows"
        )
    matrix.div_(valid_boundary_rows)
    matrix = ((matrix + matrix.T) * 0.5).contiguous()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    positive = eigenvalues.clamp_min(0)
    trace = float(positive.sum().item())
    descending = positive.flip(0)
    cumulative = descending.cumsum(0)

    def capture_rank(fraction: float) -> int:
        if trace <= torch.finfo(torch.float64).tiny:
            return width
        return int(
            torch.searchsorted(cumulative, fraction * trace).item()
        ) + 1

    return matrix, {
        "estimator": "ground_truth_CE_activation_gradient_second_moment",
        "score_targets": "deterministically_selected_supervised_positions",
        "gradient_outer_product_rows": "all_valid_boundary_positions",
        "expected_model_fisher_claim": False,
        "cross_position_blocks_included": False,
        "selected_target_rows": selected_target_rows,
        "valid_boundary_rows": valid_boundary_rows,
        "effective_nonzero_gradient_rows": effective_gradient_rows,
        "width": width,
        "trace": trace,
        "minimum_eigenvalue": float(eigenvalues.min().item()),
        "maximum_eigenvalue": float(eigenvalues.max().item()),
        "rank_for_90_percent_trace": capture_rank(0.90),
        "rank_for_99_percent_trace": capture_rank(0.99),
        "rank_for_99_9_percent_trace": capture_rank(0.999),
        "matrix_sha256": _tensor_sha256(
            matrix,
            domain=b"fisher_graph.structured_layer.raw_fisher.v1\0",
        ),
    }


def _normalized_fisher_metric(
    matrix: Tensor,
    delta_scale: Tensor,
    *,
    eigenvalue_floor: float,
) -> tuple[Tensor, dict[str, object]]:
    """Return a PSD mean-eigenvalue-one metric in scaled-error coordinates."""

    floor_fraction = _finite(
        eigenvalue_floor,
        label="Fisher eigenvalue floor",
        minimum=0.0,
    )
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or delta_scale.shape != (matrix.shape[0],)
    ):
        raise ValueError("structured Fisher metric shapes are incompatible")
    scale = delta_scale.detach().to(device="cpu", dtype=torch.float64)
    scaled = (
        scale.unsqueeze(1)
        * matrix.detach().to(device="cpu", dtype=torch.float64)
        * scale.unsqueeze(0)
    )
    scaled = (scaled + scaled.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(scaled)
    maximum = max(
        float(eigenvalues.max().item()),
        torch.finfo(torch.float64).tiny,
    )
    absolute_floor = floor_fraction * maximum
    floored = eigenvalues.clamp_min(absolute_floor)
    metric = (eigenvectors * floored.unsqueeze(0)) @ eigenvectors.T
    metric = (metric + metric.T) * 0.5
    mean_eigenvalue = float(torch.diagonal(metric).mean().item())
    if mean_eigenvalue <= torch.finfo(torch.float64).tiny:
        raise RuntimeError("structured Fisher metric has zero mean eigenvalue")
    metric = (metric / mean_eigenvalue).contiguous()
    return metric, {
        "coordinate_system": "per_coordinate_output_delta_rms_standardized",
        "full_quadratic_form_used_in_training": True,
        "eigenvalue_floor_relative_to_maximum": floor_fraction,
        "pre_floor_minimum_eigenvalue": float(eigenvalues.min().item()),
        "pre_floor_maximum_eigenvalue": float(eigenvalues.max().item()),
        "post_normalization_trace": float(torch.diagonal(metric).sum().item()),
        "sha256": _tensor_sha256(
            metric,
            domain=b"fisher_graph.structured_layer.training_metric.v1\0",
        ),
    }


def make_structured_executor(
    adapter: ModelAdapter,
    *,
    layer_id: str,
    causal_edges_enabled: bool,
    seed: int,
    device: torch.device,
) -> StructuredTransformerLayerExecutor:
    """Create an independently initialized source-shaped executor."""

    layer = adapter.layer(layer_id)
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        layer,
        causal_edges_enabled=causal_edges_enabled,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        executor = StructuredTransformerLayerExecutor(
            config,
            dtype=torch.float32,
            device="cpu",
        )
    return executor.to(device=device)


def _scaled_weights(
    *,
    structured_loss_scale: float,
    output_fisher_weight: float,
) -> StructuredLayerDistillationWeights:
    scale = _finite(
        structured_loss_scale,
        label="structured loss scale",
        minimum=torch.finfo(torch.float64).tiny,
    )
    fisher = _finite(
        output_fisher_weight,
        label="output Fisher weight",
        minimum=0.0,
    )
    defaults = StructuredLayerDistillationWeights()
    return StructuredLayerDistillationWeights(
        **{
            name: (
                fisher
                if name == "output_fisher"
                else scale * float(getattr(defaults, name))
            )
            for name in defaults.__dataclass_fields__
        }
    )


def fit_structured_executor(
    adapter: ModelAdapter,
    executor: StructuredTransformerLayerExecutor,
    training: Sequence[StructuredTrainingBatch],
    *,
    plan: object,
    scales: StructuredLayerDistillationScales,
    output_fisher: StructuredOutputFisherMetric,
    weights: StructuredLayerDistillationWeights,
    coordinate_loss_weight: float,
    energy_loss_weight: float,
    local_warmup_steps: int,
    train_steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    ground_truth_weight: float,
    teacher_kl_weight: float,
    progress_label: str | None = None,
) -> dict[str, object]:
    """Fit one fixed-schedule executor using local and suffix supervision."""

    if not training:
        raise ValueError("structured executor training cannot be empty")
    if (
        scales.provenance != output_fisher.provenance
        or scales.calibration_split_sha256
        != output_fisher.calibration_split_sha256
        or any(item.targets.provenance != scales.provenance for item in training)
        or executor.width != scales.width
    ):
        raise ValueError("structured training provenance is inconsistent")
    if type(local_warmup_steps) is not int or local_warmup_steps < 0:
        raise ValueError("local_warmup_steps must be nonnegative")
    if type(train_steps) is not int or train_steps < 0:
        raise ValueError("train_steps must be nonnegative")
    if local_warmup_steps + train_steps <= 0:
        raise ValueError("structured training requires at least one update")
    learning_rate = _finite(
        learning_rate,
        label="learning rate",
        minimum=torch.finfo(torch.float64).tiny,
    )
    weight_decay = _finite(
        weight_decay,
        label="weight decay",
        minimum=0.0,
    )
    gradient_clip_norm = _finite(
        gradient_clip_norm,
        label="gradient clip norm",
        minimum=torch.finfo(torch.float64).tiny,
    )
    ground_truth_weight = _finite(
        ground_truth_weight,
        label="ground-truth weight",
        minimum=0.0,
    )
    teacher_kl_weight = _finite(
        teacher_kl_weight,
        label="teacher KL weight",
        minimum=0.0,
    )
    coordinate_loss_weight = _finite(
        coordinate_loss_weight,
        label="coordinate loss weight",
        minimum=0.0,
    )
    energy_loss_weight = _finite(
        energy_loss_weight,
        label="energy loss weight",
        minimum=0.0,
    )
    if coordinate_loss_weight == 0 and energy_loss_weight == 0:
        raise ValueError(
            "coordinate or energy loss weight must be positive"
        )
    source_parameter_ids = {
        id(parameter) for parameter in adapter.module.parameters()
    }
    parameters = tuple(executor.parameters())
    if not parameters or any(
        id(parameter) in source_parameter_ids for parameter in parameters
    ):
        raise RuntimeError("structured executor aliases source parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = local_warmup_steps + train_steps
    rows: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    executor.train()
    for step in range(total_steps):
        item = training[step % len(training)]
        if item.targets.sequence.device != executor.device:
            raise ValueError(
                "captured targets and structured executor must share a device"
            )
        optimizer.zero_grad(set_to_none=True)
        prediction = executor.forward_components(
            item.block_input.to(device=executor.device),
            item.targets.sequence,
            prefix=None,
        )
        local = structured_layer_distillation_loss(
            prediction,
            item.targets,
            item.batch.valid_positions.to(device=executor.device),
            weights=weights,
            scales=scales,
            coordinate_loss_weight=coordinate_loss_weight,
            energy_loss_weight=energy_loss_weight,
            fisher_positions=item.selected_positions.to(
                device=executor.device
            ),
            output_fisher=output_fisher,
        )
        ground_truth_ce = prediction.output.new_zeros(())
        teacher_kl = prediction.output.new_zeros(())
        if step >= local_warmup_steps:
            _, _, logits = _run_suffix_from_boundary(
                adapter,
                item.batch,
                plan=plan,  # type: ignore[arg-type]
                sequence=item.targets.sequence,
                boundary_output=prediction.output,
                selected_positions=item.selected_positions,
                full_logits=False,
            )
            if logits is None:
                raise RuntimeError("structured student suffix logits are missing")
            ground_truth_ce = F.cross_entropy(
                logits.float(),
                item.ground_truth_targets.to(device=logits.device),
                reduction="mean",
            )
            teacher_log = F.log_softmax(
                item.teacher_logits.to(
                    device=logits.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
            student_log = F.log_softmax(logits.float(), dim=-1)
            teacher_kl = F.kl_div(
                student_log,
                teacher_log,
                reduction="batchmean",
                log_target=True,
            )
            total = (
                local.total
                + ground_truth_weight * ground_truth_ce
                + teacher_kl_weight * teacher_kl
            )
            phase = "structured_local_plus_suffix_distillation"
        else:
            total = local.total
            phase = "structured_local_warmup"
        if not bool(torch.isfinite(total)):
            raise RuntimeError("structured executor loss is nonfinite")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if any(
            parameter.grad is not None
            for parameter in adapter.module.parameters()
        ):
            raise RuntimeError("source-model parameter received a gradient")
        optimizer.step()
        row = {
            "step": step + 1,
            "phase": phase,
            "batch_index": step % len(training),
            **{
                name: float(getattr(local, name).detach().item())
                for name in (*_SCALE_NAMES, "output_fisher")
            },
            "structured_local_total": float(local.total.detach().item()),
            "coordinate_local_total": float(
                local.coordinate_total.detach().item()
            ),
            "energy_local_total": float(
                local.energy_total.detach().item()
            ),
            "ground_truth_cross_entropy": float(
                ground_truth_ce.detach().item()
            ),
            "teacher_kl": float(teacher_kl.detach().item()),
            "total_loss": float(total.detach().item()),
            "gradient_norm_before_clip": float(gradient_norm.detach().item()),
        }
        rows.append(row)
        if (
            step == 0
            or step + 1 == total_steps
            or (step + 1) % max(1, total_steps // 8) == 0
        ):
            snapshots.append(copy.deepcopy(row))
        if progress_label is not None and (
            step == 0
            or step + 1 == total_steps
            or (step + 1) % max(1, total_steps // 32) == 0
        ):
            print(
                json.dumps(
                    {
                        "candidate": progress_label,
                        "step": step + 1,
                        "total_steps": total_steps,
                        "phase": phase,
                        "total_loss": row["total_loss"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    executor.eval()
    return {
        "local_warmup_steps": local_warmup_steps,
        "downstream_train_steps": train_steps,
        "total_steps": total_steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "structured_loss_weights": asdict(weights),
        "coordinate_loss_weight": coordinate_loss_weight,
        "energy_loss_weight": energy_loss_weight,
        "ground_truth_cross_entropy_weight": ground_truth_weight,
        "teacher_kl_weight": teacher_kl_weight,
        "fixed_update_schedule": True,
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
        "all_eight_structured_targets_supervised": True,
        "local_position_scope": "all_valid_rows",
        "fisher_position_scope": "selected_supervised_rows",
        "first_update": copy.deepcopy(rows[0]),
        "last_update": copy.deepcopy(rows[-1]),
        "minimum_observed_total_loss": min(
            float(row["total_loss"]) for row in rows
        ),
        "snapshots": snapshots,
        "source_parameter_gradients_observed": False,
    }


def _stage_rows(
    *,
    target: Tensor,
    prediction: Tensor,
    valid_positions: Tensor,
    logical_positions: Tensor,
    example_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    boundary = _BoundaryBatch(
        input_hidden=torch.zeros_like(target),
        output_hidden=target,
        valid_positions=valid_positions,
        logical_positions=logical_positions,
        example_ids=example_ids,
    )
    return _direct_rows(boundary, prediction)


def evaluate_calibration_a_fidelity(
    training: Sequence[StructuredTrainingBatch],
    *,
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Aggregate direct training-split fidelity before opening calibration B."""

    if (
        not training
        or _PRIMARY not in candidates
        or not set(candidates).issubset(_CANDIDATES)
    ):
        raise ValueError(
            "calibration-A fidelity requires training batches and the "
            "primary candidate"
        )
    width = candidates[_PRIMARY].width
    candidate_names = tuple(candidates)
    direct_rows = {name: [] for name in candidate_names}
    attention_rows = {name: [] for name in candidate_names}
    feed_forward_rows = {name: [] for name in candidate_names}
    with torch.no_grad():
        for item in training:
            targets = item.targets
            boundary = _BoundaryBatch(
                input_hidden=targets.block_input,
                output_hidden=targets.output,
                valid_positions=item.batch.valid_positions,
                logical_positions=targets.sequence.logical_positions,
                example_ids=item.example_ids,
            )
            for name, candidate in candidates.items():
                components = candidate.forward_components(
                    targets.block_input.to(device=candidate.device),
                    targets.sequence,
                    prefix=None,
                )
                direct_rows[name].extend(
                    _direct_rows(boundary, components.output)
                )
                attention_rows[name].extend(
                    _stage_rows(
                        target=targets.attention_delta,
                        prediction=components.attention_delta,
                        valid_positions=item.batch.valid_positions,
                        logical_positions=targets.sequence.logical_positions,
                        example_ids=item.example_ids,
                    )
                )
                feed_forward_rows[name].extend(
                    _stage_rows(
                        target=targets.feed_forward_delta,
                        prediction=components.feed_forward_delta,
                        valid_positions=item.batch.valid_positions,
                        logical_positions=targets.sequence.logical_positions,
                        example_ids=item.example_ids,
                    )
                )
    direct = {
        name: _aggregate_direct_examples(rows, width=width)
        for name, rows in direct_rows.items()
    }
    branches = {
        name: {
            "attention_delta": _aggregate_direct_examples(
                attention_rows[name],
                width=width,
            ),
            "feed_forward_delta": _aggregate_direct_examples(
                feed_forward_rows[name],
                width=width,
            ),
        }
        for name in candidate_names
    }
    gates = {}
    for name in candidate_names:
        direct_gate = _direct_gates(
            direct[name],  # type: ignore[arg-type]
            block_delta_nrmse_max=thresholds[
                "block_delta_nrmse_max"
            ],
            block_delta_cosine_min=thresholds[
                "block_delta_cosine_min"
            ],
        )
        branch_gate = _branch_gates(
            branches[name],
            nrmse_max=thresholds["branch_delta_nrmse_max"],
            cosine_min=thresholds["branch_delta_cosine_min"],
        )
        gates[name] = {
            "direct": direct_gate,
            "branches": branch_gate,
            "passed": all((*direct_gate.values(), *branch_gate.values())),
        }
    return {
        "split": "calibration_a",
        "evaluation_scope": "aggregate_direct_fidelity_no_suffix_behavior",
        "source_layer_calls": 0,
        "executor_fingerprints": {
            name: candidates[name].execution_fingerprint()
            for name in candidate_names
        },
        "direct": direct,
        "branches": branches,
        "gates": gates,
        "primary_passed": gates[_PRIMARY]["passed"],
    }


def _validate_calibration_a_fidelity(
    value: object,
    *,
    thresholds: Mapping[str, float],
    expected_sequences: int,
    executors: Mapping[str, StructuredTransformerLayerExecutor],
) -> bool:
    fields = {
        "split",
        "evaluation_scope",
        "source_layer_calls",
        "executor_fingerprints",
        "direct",
        "branches",
        "gates",
        "primary_passed",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value["split"] != "calibration_a"
        or value["evaluation_scope"]
        != "aggregate_direct_fidelity_no_suffix_behavior"
        or value["source_layer_calls"] != 0
        or value["executor_fingerprints"]
        != {
            name: executors[name].execution_fingerprint()
            for name in _CANDIDATES
        }
    ):
        raise ValueError("structured calibration-A fidelity fields are invalid")
    direct = _strict_mapping(
        value["direct"],
        label="structured calibration-A direct metrics",
        fields=set(_CANDIDATES),
    )
    branches = _strict_mapping(
        value["branches"],
        label="structured calibration-A branch metrics",
        fields=set(_CANDIDATES),
    )
    gates = _strict_mapping(
        value["gates"],
        label="structured calibration-A fidelity gates",
        fields=set(_CANDIDATES),
    )
    for name in _CANDIDATES:
        direct_metrics = _strict_mapping(
            direct[name],
            label=f"structured calibration-A {name} direct metrics",
        )
        branch_metrics = _strict_mapping(
            branches[name],
            label=f"structured calibration-A {name} branch metrics",
            fields={"attention_delta", "feed_forward_delta"},
        )
        if direct_metrics.get("sequences") != expected_sequences:
            raise ValueError(
                "structured calibration-A sequence count is invalid"
            )
        expected_direct = _direct_gates(
            direct_metrics,
            block_delta_nrmse_max=thresholds[
                "block_delta_nrmse_max"
            ],
            block_delta_cosine_min=thresholds[
                "block_delta_cosine_min"
            ],
        )
        expected_branches = _branch_gates(
            branch_metrics,
            nrmse_max=thresholds["branch_delta_nrmse_max"],
            cosine_min=thresholds["branch_delta_cosine_min"],
        )
        expected = {
            "direct": expected_direct,
            "branches": expected_branches,
            "passed": all(
                (*expected_direct.values(), *expected_branches.values())
            ),
        }
        if gates[name] != expected:
            raise ValueError(
                "structured calibration-A fidelity gates are invalid"
            )
    primary_passed = gates[_PRIMARY]["passed"] is True
    if value["primary_passed"] is not primary_passed or not primary_passed:
        raise ValueError(
            "structured artifact did not pass calibration-A preflight"
        )
    return True


def _write_calibration_a_preflight(
    path: Path,
    *,
    model: Mapping[str, object],
    layer_id: str,
    calibration_split_sha256: str,
    artifact_format_version: int,
    training_recipe: Mapping[str, object],
    rmsnorm_initialization: Mapping[str, object] | None,
    training_reports: Mapping[str, Mapping[str, object]],
    fidelity: Mapping[str, object],
    stopped_after_calibration_a: bool = False,
) -> dict[str, object]:
    fingerprints = {
        name: report["final_execution_fingerprint"]
        for name, report in training_reports.items()
    }
    primary_passed = fidelity.get("primary_passed") is True
    outcome = (
        "calibration_a_only_passed"
        if stopped_after_calibration_a and primary_passed
        else (
            "calibration_a_passed"
            if primary_passed
            else "rejected_on_calibration_a"
        )
    )
    payload = {
        "schema": (
            "fisher_graph.gemma3_structured_layer_"
            "calibration_a_preflight"
        ),
        "format_version": 2,
        "artifact_format_version": artifact_format_version,
        "scientific_status": {
            "outcome": outcome,
            "calibration_a_passed": primary_passed,
            "stopped_after_calibration_a": stopped_after_calibration_a,
            "calibration_b_opened": False,
            "diagnostic_only": True,
            "scientific_parent": False,
        },
        "model": model,
        "layer_id": layer_id,
        "calibration_split_sha256": calibration_split_sha256,
        "training_recipe": training_recipe,
        "rmsnorm_initialization": rmsnorm_initialization,
        "training": training_reports,
        "executor_fingerprints": fingerprints,
        "fidelity": fidelity,
        "calibration_b_tokenized": False,
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


def _empty_accounting() -> dict[str, int]:
    return {
        "valid_tokens": 0,
        "logical_causal_key_pairs": 0,
        "attention_projection_macs": 0,
        "attention_score_macs": 0,
        "attention_value_macs": 0,
        "feed_forward_macs": 0,
        "logical_total_macs": 0,
    }


def evaluate_structured_candidates(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: object,
    layer_id: str,
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
    native_parity_tolerance: float = DEFAULT_NATIVE_PARITY_TOLERANCE,
) -> dict[str, object]:
    """Evaluate strict replacements against ordinary native model execution."""

    if not batches or not candidates:
        raise ValueError("evaluation batches and candidates are required")
    tolerance = _finite(
        native_parity_tolerance,
        label="native parity tolerance",
        minimum=0.0,
    )
    layer = adapter.layer(layer_id)
    semantics = layer.transformer
    if semantics is None:
        raise ValueError("selected layer has no transformer semantics")
    attention_stage, feed_forward_stage = semantics.stages
    capture_sites = (
        layer.input_site,
        attention_stage.delta_site,
        attention_stage.output_site,
        feed_forward_stage.delta_site,
        layer.output_site,
    )
    behavior_rows = {name: [] for name in candidates}
    direct_rows = {name: [] for name in candidates}
    attention_rows = {name: [] for name in candidates}
    feed_forward_rows = {name: [] for name in candidates}
    execution_counts = {
        name: {"executor_calls": 0, "source_layer_calls": 0}
        for name in candidates
    }
    prefix_errors = {name: [] for name in candidates}
    accounting = {name: _empty_accounting() for name in candidates}
    parity_logits = []
    parity_inputs = []
    parity_outputs = []
    replay_errors = []
    boundaries = []
    sequence_offset = 0
    with torch.no_grad():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            ordinary = adapter.forward(
                batch.model_inputs,
                capture_sites=capture_sites,
                retain_gradients=False,
            )
            segmented = _run_native_stack(
                adapter,
                batch,
                plan=plan,  # type: ignore[arg-type]
                full_logits=True,
            )
            if segmented.logits is None:
                raise RuntimeError("segmented native logits are missing")
            ordinary_input = ordinary.activations[layer.input_site]
            ordinary_output = ordinary.activations[layer.output_site]
            ordinary_attention_delta = ordinary.activations[
                attention_stage.delta_site
            ]
            ordinary_feed_forward_delta = ordinary.activations[
                feed_forward_stage.delta_site
            ]
            parity_logits.append(
                float(
                    (
                        segmented.logits.float() - ordinary.logits.float()
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
            parity_inputs.append(
                float(
                    (
                        segmented.block_input.float()
                        - ordinary_input.float()
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
            parity_outputs.append(
                float(
                    (
                        segmented.block_output.float()
                        - ordinary_output.float()
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
            boundary = _BoundaryBatch(
                input_hidden=ordinary_input.detach(),
                output_hidden=ordinary_output.detach(),
                valid_positions=batch.valid_positions,
                logical_positions=ordinary.sequence.logical_positions,
                example_ids=ids,
            )
            boundaries.append(boundary)
            for name, executor in candidates.items():
                replacement, audit = _run_replacement_with_call_audit(
                    adapter,
                    batch,
                    plan=plan,  # type: ignore[arg-type]
                    executor=executor,  # type: ignore[arg-type]
                    full_logits=True,
                )
                if replacement.logits is None:
                    raise RuntimeError("structured replacement logits are missing")
                execution_counts[name]["executor_calls"] += int(
                    audit["executor_calls"]
                )
                execution_counts[name]["source_layer_calls"] += int(
                    audit["source_block_calls_total"]
                )
                prefix_errors[name].append(
                    float(
                        (
                            replacement.block_input.float()
                            - ordinary_input.float()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )
                behavior_rows[name].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=ordinary.logits,
                        predicted_logits=replacement.logits,
                    )
                )
                direct_rows[name].extend(
                    _direct_rows(boundary, replacement.block_output)
                )
                components = executor.forward_components(
                    ordinary_input.to(device=executor.device),
                    ordinary.sequence,
                    prefix=None,
                )
                attention_rows[name].extend(
                    _stage_rows(
                        target=ordinary_attention_delta,
                        prediction=components.attention_delta,
                        valid_positions=batch.valid_positions,
                        logical_positions=ordinary.sequence.logical_positions,
                        example_ids=ids,
                    )
                )
                feed_forward_rows[name].extend(
                    _stage_rows(
                        target=ordinary_feed_forward_delta,
                        prediction=components.feed_forward_delta,
                        valid_positions=batch.valid_positions,
                        logical_positions=ordinary.sequence.logical_positions,
                        example_ids=ids,
                    )
                )
                ledger = executor.logical_accounting(ordinary.sequence)
                for field in accounting[name]:
                    accounting[name][field] += int(getattr(ledger, field))

            _, replay_logits, _ = _run_suffix_from_boundary(
                adapter,
                batch,
                plan=plan,  # type: ignore[arg-type]
                sequence=ordinary.sequence,
                boundary_output=ordinary_output,
                full_logits=True,
            )
            if replay_logits is None:
                raise RuntimeError("native boundary replay logits are missing")
            replay_errors.append(
                float(
                    (replay_logits.float() - ordinary.logits.float())
                    .abs()
                    .max()
                    .item()
                )
            )
    width = next(iter(candidates.values())).width
    execution_audits = {}
    for name, counts in execution_counts.items():
        maximum_prefix_error = max(prefix_errors[name], default=0.0)
        execution_audits[name] = {
            "batches": len(batches),
            "executor_calls": counts["executor_calls"],
            "source_block_calls_total": counts["source_layer_calls"],
            "source_layer_calls": {
                current_layer: 0
                for current_layer in plan.layer_ids  # type: ignore[attr-defined]
            },
            "maximum_prefix_boundary_replay_error": maximum_prefix_error,
            "native_layers_skipped": plan.layer_ids,  # type: ignore[attr-defined]
            "passed": (
                counts["source_layer_calls"] == 0
                and counts["executor_calls"] == len(batches)
                and maximum_prefix_error <= tolerance
            ),
        }
    maximum_native_error = max(
        (*parity_logits, *parity_inputs, *parity_outputs),
        default=0.0,
    )
    maximum_replay_error = max(replay_errors, default=math.inf)
    return {
        "behavior": {
            name: _aggregate_behavior_with_kl(rows)
            for name, rows in behavior_rows.items()
        },
        "direct": {
            name: _aggregate_direct_examples(rows, width=width)
            for name, rows in direct_rows.items()
        },
        "branches": {
            name: {
                "attention_delta": _aggregate_direct_examples(
                    attention_rows[name],
                    width=width,
                ),
                "feed_forward_delta": _aggregate_direct_examples(
                    feed_forward_rows[name],
                    width=width,
                ),
            }
            for name in candidates
        },
        "execution_audits": execution_audits,
        "ordinary_vs_segmented_native": {
            "maximum_absolute_logit_error": max(parity_logits, default=0.0),
            "maximum_absolute_layer_input_error": max(
                parity_inputs,
                default=0.0,
            ),
            "maximum_absolute_layer_output_error": max(
                parity_outputs,
                default=0.0,
            ),
            "tolerance": tolerance,
            "passed": maximum_native_error <= tolerance,
        },
        "native_boundary_replay": {
            "evaluated": True,
            "maximum_absolute_logit_error": maximum_replay_error,
            "tolerance": tolerance,
            "passed": maximum_replay_error <= tolerance,
        },
        "logical_accounting": accounting,
        "boundaries": tuple(boundaries),
    }


def _branch_gates(
    branches: Mapping[str, object],
    *,
    nrmse_max: float,
    cosine_min: float,
) -> dict[str, bool]:
    attention = branches["attention_delta"]
    feed_forward = branches["feed_forward_delta"]
    if not isinstance(attention, Mapping) or not isinstance(
        feed_forward,
        Mapping,
    ):
        raise ValueError("structured branch aggregates are invalid")
    return {
        "attention_delta_nrmse": (
            float(attention["block_delta_nrmse"]) <= nrmse_max
        ),
        "attention_delta_cosine": (
            float(attention["block_delta_cosine"]) >= cosine_min
        ),
        "feed_forward_delta_nrmse": (
            float(feed_forward["block_delta_nrmse"]) <= nrmse_max
        ),
        "feed_forward_delta_cosine": (
            float(feed_forward["block_delta_cosine"]) >= cosine_min
        ),
    }


def _evaluate_gates(
    result: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    behavior = result["behavior"]
    direct = result["direct"]
    branches = result["branches"]
    audits = result["execution_audits"]
    if not all(isinstance(value, Mapping) for value in (
        behavior,
        direct,
        branches,
        audits,
    )):
        raise ValueError("structured evaluation result is invalid")
    behavior_gates = {
        name: _behavior_gates(
            behavior[name],  # type: ignore[index]
            nll_atol=thresholds["nll_atol"],
            top1_min=thresholds["top1_min"],
            teacher_kl_max=thresholds["teacher_kl_max"],
            p90_abs_nll_max=thresholds["p90_abs_nll_max"],
            p10_top1_min=thresholds["p10_top1_min"],
        )
        for name in _CANDIDATES
    }
    direct_gates = {
        name: _direct_gates(
            direct[name],  # type: ignore[index]
            block_delta_nrmse_max=thresholds["block_delta_nrmse_max"],
            block_delta_cosine_min=thresholds["block_delta_cosine_min"],
        )
        for name in _CANDIDATES
    }
    branch_gates = {
        name: _branch_gates(
            branches[name],  # type: ignore[index]
            nrmse_max=thresholds["branch_delta_nrmse_max"],
            cosine_min=thresholds["branch_delta_cosine_min"],
        )
        for name in _CANDIDATES
    }
    primary_passed = (
        all(behavior_gates[_PRIMARY].values())
        and all(direct_gates[_PRIMARY].values())
        and all(branch_gates[_PRIMARY].values())
        and audits[_PRIMARY]["passed"] is True  # type: ignore[index]
        and result["ordinary_vs_segmented_native"]["passed"] is True  # type: ignore[index]
        and result["native_boundary_replay"]["passed"] is True  # type: ignore[index]
    )
    control_passed = (
        all(behavior_gates[_CONTROL].values())
        and all(direct_gates[_CONTROL].values())
        and all(branch_gates[_CONTROL].values())
        and audits[_CONTROL]["passed"] is True  # type: ignore[index]
    )
    return {
        "behavior": behavior_gates,
        "direct": direct_gates,
        "branches": branch_gates,
        "primary_passed": primary_passed,
        "attention_output_disabled_control_passed": control_passed,
        "attention_value_separation_observed": (
            primary_passed and not control_passed
        ),
    }


def _candidate_accounting(
    executor: StructuredTransformerLayerExecutor,
    logical: Mapping[str, int],
    *,
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
) -> dict[str, object]:
    source_parameters = int(source_static["parameter_count"])
    source_total_macs = int(source_macs["total_macs"])
    stored = executor.total_runtime_coefficient_count
    logical_macs = int(logical["logical_total_macs"])
    return {
        "learned_parameter_count": executor.learned_parameter_count,
        "runtime_stored_coefficient_count": stored,
        "source_layer_parameter_count": source_parameters,
        "stored_coefficient_ratio_to_source": stored / source_parameters,
        "logical_analytic_mac_count": logical_macs,
        "source_layer_analytic_mac_count": source_total_macs,
        "analytic_mac_ratio_to_source": logical_macs / source_total_macs,
        "valid_tokens": int(logical["valid_tokens"]),
        "logical_causal_key_pairs": int(
            logical["logical_causal_key_pairs"]
        ),
        "attention_projection_macs": int(
            logical["attention_projection_macs"]
        ),
        "attention_score_macs": int(logical["attention_score_macs"]),
        "attention_value_macs": int(logical["attention_value_macs"]),
        "feed_forward_macs": int(logical["feed_forward_macs"]),
        "causal_edge_control": executor.causal_edge_control,
        "resource_values_are_diagnostic_not_fidelity_gates": True,
        "normalization_and_softmax_operations_excluded": True,
        "latency_or_kernel_speed_claim": False,
    }


def _provenance_payload(
    provenance: StructuredLayerProvenance,
) -> dict[str, object]:
    return asdict(provenance)


def _restore_provenance(value: object) -> StructuredLayerProvenance:
    if not isinstance(value, Mapping) or set(value) != {
        "layer_id",
        "output_site",
        "source_segment_fingerprint",
    }:
        raise ValueError("structured layer provenance fields are invalid")
    return StructuredLayerProvenance(
        layer_id=value["layer_id"],  # type: ignore[arg-type]
        output_site=value["output_site"],  # type: ignore[arg-type]
        source_segment_fingerprint=value[  # type: ignore[arg-type]
            "source_segment_fingerprint"
        ],
    )


def _scales_payload(
    scales: StructuredLayerDistillationScales,
    *,
    floor: float,
    relative_median_floor: float,
    rows: int,
) -> dict[str, object]:
    values = {
        name: getattr(scales, name).detach().cpu().clone()
        for name in _SCALE_NAMES
    }
    return {
        "provenance": _provenance_payload(scales.provenance),
        "calibration_split_sha256": scales.calibration_split_sha256,
        "estimator": "calibration_a_per_coordinate_stage_rms",
        "floor": float(floor),
        "relative_median_floor": float(relative_median_floor),
        "valid_rows": rows,
        "values": values,
        "sha256": {
            name: _tensor_sha256(
                value,
                domain=(
                    b"fisher_graph.structured_layer.scale.v1\0"
                    + name.encode("utf-8")
                    + b"\0"
                ),
            )
            for name, value in values.items()
        },
    }


def _restore_scales(value: object) -> StructuredLayerDistillationScales:
    if not isinstance(value, Mapping) or set(value) != {
        "provenance",
        "calibration_split_sha256",
        "estimator",
        "floor",
        "relative_median_floor",
        "valid_rows",
        "values",
        "sha256",
    }:
        raise ValueError("structured scale artifact fields are invalid")
    raw_values = value["values"]
    hashes = value["sha256"]
    if (
        not isinstance(raw_values, Mapping)
        or set(raw_values) != set(_SCALE_NAMES)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_SCALE_NAMES)
        or value["estimator"]
        != "calibration_a_per_coordinate_stage_rms"
        or not isinstance(value["floor"], (int, float))
        or isinstance(value["floor"], bool)
        or not math.isfinite(float(value["floor"]))
        or float(value["floor"]) <= 0
        or not isinstance(
            value["relative_median_floor"],
            (int, float),
        )
        or isinstance(value["relative_median_floor"], bool)
        or not math.isfinite(float(value["relative_median_floor"]))
        or not 0 <= float(value["relative_median_floor"]) <= 1
        or type(value["valid_rows"]) is not int
        or value["valid_rows"] <= 0
    ):
        raise ValueError("structured scale artifact is invalid")
    tensors: dict[str, Tensor] = {}
    for name in _SCALE_NAMES:
        tensor = raw_values[name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.ndim != 1
            or tensor.dtype is not torch.float64
            or tensor.device.type != "cpu"
            or not bool(torch.isfinite(tensor).all())
            or not bool((tensor > 0).all())
        ):
            raise ValueError(f"structured scale tensor {name!r} is invalid")
        expected = _tensor_sha256(
            tensor,
            domain=(
                b"fisher_graph.structured_layer.scale.v1\0"
                + name.encode("utf-8")
                + b"\0"
            ),
        )
        if hashes[name] != expected:
            raise ValueError(f"structured scale tensor {name!r} hash mismatch")
        tensors[name] = tensor
    return StructuredLayerDistillationScales(
        provenance=_restore_provenance(value["provenance"]),
        calibration_split_sha256=value[  # type: ignore[arg-type]
            "calibration_split_sha256"
        ],
        **tensors,
    )


def _logical_accounting_from_stream(
    stream: Mapping[str, object],
    executor: StructuredTransformerLayerExecutor,
) -> dict[str, int]:
    examples = stream.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("accounting stream examples are invalid")
    lengths = []
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("accounting stream example is invalid")
        value = example.get("valid_tokens")
        if type(value) is not int or value <= 0:
            raise ValueError("accounting valid-token count is invalid")
        lengths.append(value)
    valid_tokens = sum(lengths)
    attention = executor.config.attention
    window = attention.window_size
    pairs = 0
    for length in lengths:
        if window is None or length <= window:
            pairs += length * (length + 1) // 2
        else:
            pairs += window * (window + 1) // 2
            pairs += (length - window) * window
    residual = executor.width
    query_width = attention.query_heads * attention.head_dimension
    key_value_width = (
        attention.key_value_heads * attention.head_dimension
    )
    projections = valid_tokens * residual * (
        query_width + 2 * key_value_width
    )
    projections += valid_tokens * query_width * residual
    scores = pairs * attention.query_heads * attention.head_dimension
    intermediate = (
        executor.config.transformer.feed_forward.intermediate_width
    )
    feed_forward = valid_tokens * (
        2 * residual * intermediate + intermediate * residual
    )
    return {
        "valid_tokens": valid_tokens,
        "logical_causal_key_pairs": pairs,
        "attention_projection_macs": projections,
        "attention_score_macs": scores,
        "attention_value_macs": scores,
        "feed_forward_macs": feed_forward,
        "logical_total_macs": projections + 2 * scores + feed_forward,
    }


def _source_macs_from_stream(
    source_manifest: Mapping[str, object],
    stream: Mapping[str, object],
) -> dict[str, object]:
    layer_ids = source_manifest.get("layer_ids")
    linear_by_layer = source_manifest.get(
        "linear_weight_coefficients_by_layer"
    )
    attention_by_layer = source_manifest.get("attention_by_layer")
    examples = stream.get("examples")
    if (
        not isinstance(layer_ids, tuple)
        or len(layer_ids) != 1
        or not isinstance(linear_by_layer, Mapping)
        or not isinstance(attention_by_layer, Mapping)
        or not isinstance(examples, list)
    ):
        raise ValueError("source accounting manifest is invalid")
    layer_id = layer_ids[0]
    attention = attention_by_layer.get(layer_id)
    if not isinstance(attention, Mapping):
        raise ValueError("source attention accounting is invalid")
    lengths = []
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("source accounting stream example is invalid")
        length = example.get("valid_tokens")
        if type(length) is not int or length <= 0:
            raise ValueError("source accounting length is invalid")
        lengths.append(length)
    window = attention.get("window_size")
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError("source attention window is invalid")
    edges = 0
    for length in lengths:
        if window is None or length <= window:
            edges += length * (length + 1) // 2
        else:
            edges += window * (window + 1) // 2
            edges += (length - window) * window
    valid = sum(lengths)
    linear = valid * int(linear_by_layer[layer_id])
    attention_macs = (
        edges
        * 2
        * int(attention["query_heads"])
        * int(attention["head_dimension"])
    )
    return {
        "valid_positions": valid,
        "linear_projection_macs": linear,
        "qk_and_av_attention_macs": attention_macs,
        "causal_attention_edges": edges,
        "total_macs": linear + attention_macs,
        "by_layer": {
            layer_id: {
                "linear_projection_macs": linear,
                "qk_and_av_attention_macs": attention_macs,
                "causal_attention_edges": edges,
                "total_macs": linear + attention_macs,
            }
        },
        "semantics": (
            "linear_weight_MACs_plus_QK_and_AV_dot_products_on_same_valid_"
            "lengths"
        ),
        "excluded": (
            "normalization_elementwise_bias_activation_softmax_rope_"
            "masking_additions_and_memory_traffic"
        ),
    }


def _evaluation_payload(
    result: dict[str, object],
    *,
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
    thresholds: Mapping[str, float],
    tokenized_stream: Mapping[str, object],
    tokenized_stream_contract: Mapping[str, object],
) -> dict[str, object]:
    result = dict(result)
    result.pop("boundaries")
    logical = result["logical_accounting"]
    if not isinstance(logical, Mapping):
        raise ValueError("structured logical accounting is invalid")
    accounting = {
        name: _candidate_accounting(
            candidate,
            logical[name],  # type: ignore[arg-type,index]
            source_static=source_static,
            source_macs=source_macs,
        )
        for name, candidate in candidates.items()
    }
    gates = _evaluate_gates(result, thresholds=thresholds)
    return {
        "evaluated": True,
        **result,
        "executor_fingerprints": {
            name: candidate.execution_fingerprint()
            for name, candidate in candidates.items()
        },
        "gates": gates,
        "accounting": accounting,
        "resource_gates_applied": False,
        "resource_diagnostics_only": True,
        "passed": gates["primary_passed"],
        "locked_candidate": (
            _PRIMARY if gates["primary_passed"] is True else None
        ),
        "tokenized_stream": copy.deepcopy(dict(tokenized_stream)),
        "tokenized_stream_contract": copy.deepcopy(
            dict(tokenized_stream_contract)
        ),
    }


def _unevaluated_validation_payload(
    *,
    format_version: int = _ARTIFACT_FORMAT_VERSION,
) -> dict[str, object]:
    payload = {
        "evaluated": False,
        "reason": "calibration_b_failed_validation_not_tokenized",
        "behavior": None,
        "direct": None,
        "branches": None,
        "execution_audits": None,
        "ordinary_vs_segmented_native": None,
        "native_boundary_replay": None,
        "logical_accounting": None,
        "gates": None,
        "accounting": None,
        "resource_gates_applied": False,
        "resource_diagnostics_only": True,
        "passed": False,
        "locked_candidate": None,
        "tokenized_stream": None,
        "tokenized_stream_contract": None,
    }
    if format_version >= 4:
        payload["executor_fingerprints"] = None
    return payload


def _build_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    scientific_digest: str,
) -> dict[str, object]:
    training = payload["training"]
    executors = payload["executors"]
    assert isinstance(training, Mapping)
    assert isinstance(executors, Mapping)
    fisher = training["activation_fisher"]
    scales = training["structured_scales"]
    output_fisher = training["output_fisher"]
    assert isinstance(fisher, Mapping)
    assert isinstance(scales, Mapping)
    assert isinstance(output_fisher, Mapping)
    return {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "scientific_status": copy.deepcopy(payload["scientific_status"]),
        "model": copy.deepcopy(payload["model"]),
        "protocol": copy.deepcopy(payload["protocol"]),
        "training": {
            "activation_fisher": {
                key: copy.deepcopy(value)
                for key, value in fisher.items()
                if key != "matrix"
            },
            "structured_scales": {
                key: copy.deepcopy(value)
                for key, value in scales.items()
                if key != "values"
            },
            "output_fisher": {
                key: copy.deepcopy(value)
                for key, value in output_fisher.items()
                if key
                not in {
                    "delta_scale",
                    "standardized_coordinate_metric",
                }
            },
            _PRIMARY: copy.deepcopy(training[_PRIMARY]),
            _CONTROL: copy.deepcopy(training[_CONTROL]),
            **(
                {
                    "calibration_a_fidelity": copy.deepcopy(
                        training["calibration_a_fidelity"]
                    )
                }
                if "calibration_a_fidelity" in training
                else {}
            ),
            "structural_probes": copy.deepcopy(
                training["structural_probes"]
            ),
        },
        "selection": copy.deepcopy(payload["selection"]),
        "validation": copy.deepcopy(payload["validation"]),
        "executors": {
            name: {
                "execution_fingerprint": state["execution_fingerprint"],
                "causal_edges_enabled": state["config"][
                    "causal_edges_enabled"
                ],
            }
            for name, state in executors.items()  # type: ignore[union-attr]
        },
        "artifact": {
            "tensor_file": tensor_file,
            "tensor_file_ignored_by_git_policy": True,
            "contains_model_weights": False,
            "contains_executor_weights": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "contains_teacher_targets": False,
            "contains_source_derived_statistics": True,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def _strong_data_minima(
    *,
    minimum_calibration_a_prompts: int,
    minimum_heldout_prompts: int,
    minimum_fisher_rows: int,
    minimum_train_supervised_tokens: int,
    minimum_heldout_supervised_tokens: int,
    minimum_length_buckets: int,
) -> bool:
    return (
        minimum_calibration_a_prompts
        >= DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        and minimum_heldout_prompts >= DEFAULT_MINIMUM_HELDOUT_PROMPTS
        and minimum_fisher_rows >= DEFAULT_MINIMUM_FISHER_ROWS
        and minimum_train_supervised_tokens
        >= DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
        and minimum_heldout_supervised_tokens
        >= DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        and minimum_length_buckets >= DEFAULT_MINIMUM_LENGTH_BUCKETS
    )


def _training_recipe_payload(
    *,
    local_warmup_steps: int,
    train_steps: int,
    train_positions_per_sequence: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    weights: StructuredLayerDistillationWeights,
    coordinate_loss_weight: float,
    energy_loss_weight: float,
    ground_truth_weight: float,
    teacher_kl_weight: float,
) -> dict[str, object]:
    return {
        "local_warmup_steps": local_warmup_steps,
        "downstream_train_steps": train_steps,
        "train_positions_per_sequence": train_positions_per_sequence,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "structured_loss_weights": asdict(weights),
        "coordinate_loss_weight": coordinate_loss_weight,
        "energy_loss_weight": energy_loss_weight,
        "rmsnorm_initialization": DEFAULT_RMSNORM_INITIALIZATION,
        "ground_truth_cross_entropy_weight": ground_truth_weight,
        "teacher_kl_weight": teacher_kl_weight,
        "fixed_update_schedule": True,
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
    }


def _operator_bootstrap_training_recipe_payload(
    *,
    train_positions_per_sequence: int,
    requested_rows: int,
    ridge_relative: float,
    rank_relative_tolerance: float,
    maximum_condition_number: float,
    maximum_nullity: int,
) -> dict[str, object]:
    return {
        "fitting_method": _OPERATOR_BOOTSTRAP_FITTING_METHOD,
        "local_warmup_steps": 0,
        "downstream_train_steps": 0,
        "train_positions_per_sequence": train_positions_per_sequence,
        "optimizer": "none",
        "optimizer_steps": 0,
        "suffix_training_steps": 0,
        "checkpoint_selection": "deterministic_closed_form",
        "early_stopping": False,
        "operator_bootstrap": {
            "schema": STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
            "format_version": (
                STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION
            ),
            "algorithm": STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
            "row_selection": "lowest_sha256_valid_token_rows_v1",
            "requested_rows": requested_rows,
            "ridge_relative_to_mean_gram_diagonal": ridge_relative,
            "rank_relative_tolerance": rank_relative_tolerance,
            "maximum_condition_number": maximum_condition_number,
            "rank_policy": STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY,
            "maximum_nullity": maximum_nullity,
            "selection_applied_before_activation_capture": True,
            "source_activation_capture_stream_passes": 1,
        },
    }


def _operator_bootstrap_report_binding(
    report: Mapping[str, object],
    *,
    capture_audit: Mapping[str, object],
    final_execution_fingerprint: str,
) -> dict[str, object]:
    """Retain replay bindings and aggregate audit flags, never captures."""

    binding = {
        key: copy.deepcopy(report[key])
        for key in (
            "schema",
            "format_version",
            "algorithm",
            "layer_id",
            "calibration_split_sha256",
            "source_segment_fingerprint",
            "site_schema",
            "site_schema_sha256",
            "row_selection",
            "solver",
            "operators",
            "normalizations",
            "coefficient_sha256",
            "source_module_or_parameter_read",
            "direct_source_tensor_copy",
            "activation_targets_serialized",
            "sufficient_statistics_serialized",
            "destination_source_weight_contamination",
            "destination_executor_local_source_free",
        )
    }
    return {
        "fitting_method": _OPERATOR_BOOTSTRAP_FITTING_METHOD,
        "optimizer": "none",
        "optimizer_steps": 0,
        "suffix_training_steps": 0,
        "bootstrap": binding,
        "capture_audit": copy.deepcopy(dict(capture_audit)),
        "final_execution_fingerprint": final_execution_fingerprint,
    }


def _preregistered_training_recipe(
    value: Mapping[str, object],
    *,
    format_version: int = _ARTIFACT_FORMAT_VERSION,
) -> bool:
    if format_version >= 5:
        return value == _operator_bootstrap_training_recipe_payload(
            train_positions_per_sequence=(
                DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
            ),
            requested_rows=DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS,
            ridge_relative=DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE,
            rank_relative_tolerance=(
                DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL
            ),
            maximum_condition_number=(
                DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION
            ),
            maximum_nullity=(
                DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY
            ),
        )
    expected_weights = asdict(
        _scaled_weights(
            structured_loss_scale=DEFAULT_STRUCTURED_LOSS_SCALE,
            output_fisher_weight=DEFAULT_OUTPUT_FISHER_WEIGHT,
        )
    )
    common = (
        value.get("local_warmup_steps") == DEFAULT_LOCAL_WARMUP_STEPS
        and value.get("downstream_train_steps") == DEFAULT_TRAIN_STEPS
        and value.get("train_positions_per_sequence")
        == DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
        and value.get("learning_rate") == DEFAULT_LEARNING_RATE
        and value.get("weight_decay") == DEFAULT_WEIGHT_DECAY
        and value.get("gradient_clip_norm") == DEFAULT_GRADIENT_CLIP_NORM
        and value.get("structured_loss_weights") == expected_weights
        and value.get("ground_truth_cross_entropy_weight")
        == DEFAULT_GROUND_TRUTH_WEIGHT
        and value.get("teacher_kl_weight") == DEFAULT_TEACHER_KL_WEIGHT
        and value.get("fixed_update_schedule") is True
        and value.get("checkpoint_selection") == "final_fixed_step"
        and value.get("early_stopping") is False
    )
    if format_version == 3:
        return common
    return (
        common
        and value.get("coordinate_loss_weight")
        == DEFAULT_COORDINATE_LOSS_WEIGHT
        and value.get("energy_loss_weight")
        == DEFAULT_ENERGY_LOSS_WEIGHT
        and value.get("rmsnorm_initialization")
        == DEFAULT_RMSNORM_INITIALIZATION
    )


def _preregistered_representative_protocol(
    *,
    training_recipe: Mapping[str, object],
    thresholds: Mapping[str, object],
    seed: object,
    maximum_length: object,
    tokenization_batch_size: object,
    fisher_floor: object,
    delta_scale_floor: object,
    relative_median_scale_floor: object,
    format_version: int = _ARTIFACT_FORMAT_VERSION,
) -> bool:
    expected_thresholds = {
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
    return (
        _preregistered_training_recipe(
            training_recipe,
            format_version=format_version,
        )
        and thresholds == expected_thresholds
        and seed == DEFAULT_OPTIMIZATION_SEED
        and maximum_length == 256
        and tokenization_batch_size == 4
        and fisher_floor == DEFAULT_FISHER_FLOOR
        and delta_scale_floor == DEFAULT_RIDGE_SCALE_FLOOR
        and relative_median_scale_floor
        == DEFAULT_RELATIVE_MEDIAN_SCALE_FLOOR
    )


def _validate_training_recipe_binding(
    value: object,
    *,
    training: Mapping[str, object],
    format_version: int,
) -> bool:
    if format_version >= 5:
        fields = {
            "fitting_method",
            "local_warmup_steps",
            "downstream_train_steps",
            "train_positions_per_sequence",
            "optimizer",
            "optimizer_steps",
            "suffix_training_steps",
            "checkpoint_selection",
            "early_stopping",
            "operator_bootstrap",
        }
        bootstrap_fields = {
            "schema",
            "format_version",
            "algorithm",
            "row_selection",
            "requested_rows",
            "ridge_relative_to_mean_gram_diagonal",
            "rank_relative_tolerance",
            "maximum_condition_number",
            "rank_policy",
            "maximum_nullity",
            "selection_applied_before_activation_capture",
            "source_activation_capture_stream_passes",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["fitting_method"]
            != _OPERATOR_BOOTSTRAP_FITTING_METHOD
            or value["local_warmup_steps"] != 0
            or value["downstream_train_steps"] != 0
            or type(value["train_positions_per_sequence"]) is not int
            or value["train_positions_per_sequence"] <= 0
            or value["optimizer"] != "none"
            or value["optimizer_steps"] != 0
            or value["suffix_training_steps"] != 0
            or value["checkpoint_selection"]
            != "deterministic_closed_form"
            or value["early_stopping"] is not False
            or not isinstance(value["operator_bootstrap"], Mapping)
            or set(value["operator_bootstrap"]) != bootstrap_fields
        ):
            raise ValueError(
                "structured operator-bootstrap recipe fields are invalid"
            )
        bootstrap = value["operator_bootstrap"]
        if (
            bootstrap["schema"] != STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA
            or bootstrap["format_version"]
            != STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION
            or bootstrap["algorithm"]
            != STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM
            or bootstrap["row_selection"]
            != "lowest_sha256_valid_token_rows_v1"
            or type(bootstrap["requested_rows"]) is not int
            or bootstrap["requested_rows"] <= 0
            or bootstrap[
                "selection_applied_before_activation_capture"
            ]
            is not True
            or bootstrap["source_activation_capture_stream_passes"] != 1
            or bootstrap["rank_policy"]
            != STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY
            or type(bootstrap["maximum_nullity"]) is not int
            or bootstrap["maximum_nullity"] < 0
        ):
            raise ValueError(
                "structured operator-bootstrap recipe is invalid"
            )
        _finite(
            bootstrap["ridge_relative_to_mean_gram_diagonal"],
            label="structured bootstrap recipe ridge",
            minimum=torch.finfo(torch.float64).tiny,
        )
        rank_tolerance = _finite(
            bootstrap["rank_relative_tolerance"],
            label="structured bootstrap recipe rank tolerance",
            minimum=torch.finfo(torch.float64).tiny,
            maximum=1.0,
        )
        if rank_tolerance >= 1.0:
            raise ValueError(
                "structured bootstrap recipe rank tolerance must be less "
                "than 1"
            )
        maximum_condition = _finite(
            bootstrap["maximum_condition_number"],
            label="structured bootstrap recipe maximum condition",
            minimum=1.0,
        )
        if maximum_condition <= 1.0:
            raise ValueError(
                "structured bootstrap recipe maximum condition must exceed 1"
            )
        for name in _CANDIDATES:
            report = training.get(name)
            if (
                not isinstance(report, Mapping)
                or report.get("fitting_method")
                != value["fitting_method"]
                or report.get("optimizer") != "none"
                or report.get("optimizer_steps") != 0
                or report.get("suffix_training_steps") != 0
            ):
                raise ValueError(
                    "structured bootstrap report does not match its recipe"
                )
        return _preregistered_training_recipe(
            value,
            format_version=format_version,
        )
    fields = {
        "local_warmup_steps",
        "downstream_train_steps",
        "train_positions_per_sequence",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "structured_loss_weights",
        "ground_truth_cross_entropy_weight",
        "teacher_kl_weight",
        "fixed_update_schedule",
        "checkpoint_selection",
        "early_stopping",
    }
    if format_version >= 4:
        fields |= {
            "coordinate_loss_weight",
            "energy_loss_weight",
            "rmsnorm_initialization",
        }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("structured training recipe fields are invalid")
    for name in _CANDIDATES:
        report = training.get(name)
        if not isinstance(report, Mapping) or any(
            report.get(field) != value[field]
            for field in (
                "local_warmup_steps",
                "downstream_train_steps",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
                "structured_loss_weights",
                *(
                    (
                        "coordinate_loss_weight",
                        "energy_loss_weight",
                    )
                    if format_version >= 4
                    else ()
                ),
                "ground_truth_cross_entropy_weight",
                "teacher_kl_weight",
                "fixed_update_schedule",
                "checkpoint_selection",
                "early_stopping",
            )
        ):
            raise ValueError(
                "structured training report does not match its recipe"
            )
    positions = value["train_positions_per_sequence"]
    if type(positions) is not int or positions <= 0:
        raise ValueError(
            "structured training positions per sequence are invalid"
        )
    if format_version >= 4 and any(
        training[name].get("rmsnorm_initialization", {}).get("algorithm")
        != value["rmsnorm_initialization"]
        for name in _CANDIDATES
    ):
        raise ValueError(
            "structured RMSNorm initialization does not match its recipe"
        )
    return _preregistered_training_recipe(
        value,
        format_version=format_version,
    )


def _validate_rmsnorm_initialization_binding(
    value: object,
    *,
    provenance: StructuredLayerProvenance,
    calibration_split_sha256: str,
    valid_rows: int,
    width: int,
) -> None:
    fields = {
        "algorithm",
        "calibration_split_sha256",
        "provenance",
        "valid_rows",
        "source_module_or_parameter_read",
        "direct_source_tensor_copy",
        "normalizations",
    }
    names = {
        "attention_input_norm",
        "attention_output_norm",
        "feed_forward_input_norm",
        "feed_forward_output_norm",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value["algorithm"] != DEFAULT_RMSNORM_INITIALIZATION
        or value["calibration_split_sha256"]
        != calibration_split_sha256
        or value["provenance"] != _provenance_payload(provenance)
        or value["valid_rows"] != valid_rows
        or value["source_module_or_parameter_read"] is not False
        or value["direct_source_tensor_copy"] is not False
    ):
        raise ValueError(
            "structured RMSNorm initialization binding is invalid"
        )
    normalizations = value["normalizations"]
    if not isinstance(normalizations, Mapping) or set(normalizations) != names:
        raise ValueError(
            "structured RMSNorm initialization modules are invalid"
        )
    metric_fields = {
        "width",
        "identified_coordinates",
        "fit_nrmse",
        "weight_minimum",
        "weight_median",
        "weight_maximum",
        "weight_rms",
    }
    for name in names:
        metrics = normalizations[name]
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != metric_fields
            or metrics["width"] != width
            or type(metrics["identified_coordinates"]) is not int
            or not 0 <= metrics["identified_coordinates"] <= width
            or any(
                not isinstance(metrics[field], (float, int))
                or isinstance(metrics[field], bool)
                or not math.isfinite(float(metrics[field]))
                for field in (
                    "fit_nrmse",
                    "weight_minimum",
                    "weight_median",
                    "weight_maximum",
                    "weight_rms",
                )
            )
            or float(metrics["fit_nrmse"]) < 0
            or float(metrics["weight_rms"]) < 0
            or not (
                float(metrics["weight_minimum"])
                <= float(metrics["weight_median"])
                <= float(metrics["weight_maximum"])
            )
        ):
            raise ValueError(
                "structured RMSNorm initialization metrics are invalid"
            )


def _validate_operator_bootstrap_training_binding(
    *,
    training: Mapping[str, object],
    training_recipe: Mapping[str, object],
    executors: Mapping[str, StructuredTransformerLayerExecutor],
    provenance: StructuredLayerProvenance,
    layer_index: int,
) -> None:
    report_fields = {
        "fitting_method",
        "optimizer",
        "optimizer_steps",
        "suffix_training_steps",
        "bootstrap",
        "capture_audit",
        "final_execution_fingerprint",
    }
    bootstrap_fields = {
        "schema",
        "format_version",
        "algorithm",
        "layer_id",
        "calibration_split_sha256",
        "source_segment_fingerprint",
        "site_schema",
        "site_schema_sha256",
        "row_selection",
        "solver",
        "operators",
        "normalizations",
        "coefficient_sha256",
        "source_module_or_parameter_read",
        "direct_source_tensor_copy",
        "activation_targets_serialized",
        "sufficient_statistics_serialized",
        "destination_source_weight_contamination",
        "destination_executor_local_source_free",
    }
    row_fields = {
        "algorithm",
        "hash_domain",
        "requested_rows",
        "valid_rows",
        "selected_rows",
        "selected_rows_sha256",
        "selection_depends_on_activation_values",
        "selection_depends_on_teacher_targets",
        "selection_applied_before_activation_capture",
        "capture_contains_only_selected_rows",
    }
    solver_fields = {
        "accumulation_dtype",
        "solve_dtype",
        "ridge_relative_to_mean_gram_diagonal",
        "rank_relative_tolerance",
        "maximum_condition_number",
        "rank_policy",
        "maximum_nullity",
        "bias_regularized",
    }
    capture_fields = {
        "source_model_executed_for_activation_capture",
        "source_activation_capture_stream_passes",
        "source_activation_capture_forward_calls",
        "capture_site_count",
        "residual_capture_site_count",
        "operator_capture_site_count",
        "capture_contains_only_selected_rows",
        "captured_activation_rows_serialized",
        "sufficient_statistics_serialized",
        "compiler_source_parameter_tensor_read",
        "direct_source_tensor_copy",
    }
    recipe_bootstrap = _strict_mapping(
        training_recipe["operator_bootstrap"],
        label="structured operator-bootstrap recipe",
    )
    shared_bindings = []
    for name in _CANDIDATES:
        executor = executors[name]
        report = _strict_mapping(
            training[name],
            label=f"structured {name} bootstrap training report",
            fields=report_fields,
        )
        bootstrap = _strict_mapping(
            report["bootstrap"],
            label=f"structured {name} bootstrap binding",
            fields=bootstrap_fields,
        )
        row_selection = _strict_mapping(
            bootstrap["row_selection"],
            label=f"structured {name} bootstrap row selection",
            fields=row_fields,
        )
        solver = _strict_mapping(
            bootstrap["solver"],
            label=f"structured {name} bootstrap solver",
            fields=solver_fields,
        )
        capture = _strict_mapping(
            report["capture_audit"],
            label=f"structured {name} bootstrap capture audit",
            fields=capture_fields,
        )
        site_schema = _strict_mapping(
            bootstrap["site_schema"],
            label=f"structured {name} bootstrap site schema",
        )
        operators = _strict_mapping(
            bootstrap["operators"],
            label=f"structured {name} bootstrap operator reports",
        )
        normalizations = _strict_mapping(
            bootstrap["normalizations"],
            label=f"structured {name} bootstrap normalization reports",
        )
        transformer = executor.config.transformer
        restored_layer = LayerSpec(
            id=provenance.layer_id,
            ordinal=layer_index,
            input_site=transformer.stages[0].input_site,
            output_site=provenance.output_site,
            residual_width=executor.width,
            kind="restored_structured_executor",
            attention=executor.config.attention,
            transformer=transformer,
        )
        fingerprint = executor.execution_fingerprint()
        if (
            report["fitting_method"]
            != _OPERATOR_BOOTSTRAP_FITTING_METHOD
            or report["optimizer"] != "none"
            or report["optimizer_steps"] != 0
            or report["suffix_training_steps"] != 0
            or report["final_execution_fingerprint"] != fingerprint
            or bootstrap["schema"]
            != STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA
            or bootstrap["format_version"]
            != STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION
            or bootstrap["algorithm"]
            != STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM
            or bootstrap["layer_id"] != provenance.layer_id
            or bootstrap["calibration_split_sha256"]
            != training["calibration_split_sha256"]
            or bootstrap["source_segment_fingerprint"]
            != provenance.source_segment_fingerprint
            or site_schema
            != structured_operator_site_schema(restored_layer)
            or bootstrap["site_schema_sha256"]
            != structured_operator_site_schema_sha256(restored_layer)
            or bootstrap["coefficient_sha256"]
            != structured_operator_coefficient_sha256(executor)
            or bootstrap["source_module_or_parameter_read"] is not False
            or bootstrap["direct_source_tensor_copy"] is not False
            or bootstrap["activation_targets_serialized"] is not False
            or bootstrap["sufficient_statistics_serialized"] is not False
            or bootstrap[
                "destination_source_weight_contamination"
            ]
            is not False
            or bootstrap["destination_executor_local_source_free"]
            is not True
            or row_selection["algorithm"]
            != "lowest_sha256_valid_token_rows_v1"
            or not isinstance(row_selection["hash_domain"], str)
            or not row_selection["hash_domain"]
            or row_selection["requested_rows"]
            != recipe_bootstrap["requested_rows"]
            or type(row_selection["valid_rows"]) is not int
            or row_selection["valid_rows"] <= 0
            or row_selection["selected_rows"]
            != min(
                row_selection["requested_rows"],
                row_selection["valid_rows"],
            )
            or not _is_sha256(row_selection["selected_rows_sha256"])
            or row_selection[
                "selection_depends_on_activation_values"
            ]
            is not False
            or row_selection[
                "selection_depends_on_teacher_targets"
            ]
            is not False
            or row_selection[
                "selection_applied_before_activation_capture"
            ]
            is not True
            or row_selection["capture_contains_only_selected_rows"]
            is not True
            or solver["accumulation_dtype"] != "torch.float64"
            or solver["solve_dtype"] != "torch.float64"
            or solver[
                "ridge_relative_to_mean_gram_diagonal"
            ]
            != recipe_bootstrap[
                "ridge_relative_to_mean_gram_diagonal"
            ]
            or solver["rank_relative_tolerance"]
            != recipe_bootstrap["rank_relative_tolerance"]
            or solver["maximum_condition_number"]
            != recipe_bootstrap["maximum_condition_number"]
            or solver["rank_policy"]
            != recipe_bootstrap["rank_policy"]
            or solver["maximum_nullity"]
            != recipe_bootstrap["maximum_nullity"]
            or solver["bias_regularized"] is not False
            or capture[
                "source_model_executed_for_activation_capture"
            ]
            is not True
            or capture["source_activation_capture_stream_passes"] != 1
            or type(capture["source_activation_capture_forward_calls"])
            is not int
            or capture["source_activation_capture_forward_calls"] <= 0
            or capture["capture_site_count"] != 18
            or capture["residual_capture_site_count"] != 9
            or capture["operator_capture_site_count"] != 9
            or capture["capture_contains_only_selected_rows"] is not True
            or capture["captured_activation_rows_serialized"] is not False
            or capture["sufficient_statistics_serialized"] is not False
            or capture["compiler_source_parameter_tensor_read"] is not False
            or capture["direct_source_tensor_copy"] is not False
        ):
            raise ValueError(
                "structured operator-bootstrap report binding is invalid"
            )
        residual_width = int(site_schema["residual_width"])
        query_width = int(site_schema["query_width"])
        key_value_width = int(site_schema["key_value_width"])
        feed_forward_width = int(site_schema["feed_forward_width"])
        head_dimension = int(site_schema["head_dimension"])
        query_heads = int(site_schema["query_heads"])
        key_value_heads = int(site_schema["key_value_heads"])
        projection_bias = _strict_mapping(
            site_schema["projection_bias"],
            label="structured bootstrap projection bias",
            fields={"attention", "feed_forward"},
        )
        expected_operator_shapes = {
            "attention.q_proj": (
                residual_width,
                query_width,
                projection_bias["attention"],
            ),
            "attention.k_proj": (
                residual_width,
                key_value_width,
                projection_bias["attention"],
            ),
            "attention.v_proj": (
                residual_width,
                key_value_width,
                projection_bias["attention"],
            ),
            "attention.o_proj": (
                query_width,
                residual_width,
                projection_bias["attention"],
            ),
            "feed_forward.gate_proj": (
                residual_width,
                feed_forward_width,
                projection_bias["feed_forward"],
            ),
            "feed_forward.up_proj": (
                residual_width,
                feed_forward_width,
                projection_bias["feed_forward"],
            ),
            "feed_forward.down_proj": (
                feed_forward_width,
                residual_width,
                projection_bias["feed_forward"],
            ),
        }
        operator_metric_fields = {
            "rows",
            "input_width",
            "output_width",
            "bias",
            "dimension",
            "effective_rank",
            "nullity",
            "full_column_rank",
            "active_condition_number",
            "rank_policy",
            "maximum_nullity",
            "rank_relative_tolerance",
            "maximum_condition_number",
            "ridge_relative_to_mean_gram_diagonal",
            "ridge_absolute",
            "fit_rmse",
            "fit_nrmse",
        }
        if set(operators) != set(expected_operator_shapes):
            raise ValueError(
                "structured bootstrap operator report names are invalid"
            )
        for operator_name, (
            input_width,
            output_width,
            bias,
        ) in expected_operator_shapes.items():
            metrics = _strict_mapping(
                operators[operator_name],
                label=f"structured bootstrap {operator_name} metrics",
                fields=operator_metric_fields,
            )
            if (
                metrics["rows"] != row_selection["selected_rows"]
                or metrics["input_width"] != input_width
                or metrics["output_width"] != output_width
                or metrics["bias"] is not bias
                or type(metrics["dimension"]) is not int
                or metrics["dimension"] != input_width + int(bias)
                or type(metrics["effective_rank"]) is not int
                or type(metrics["nullity"]) is not int
                or metrics["nullity"] < 0
                or metrics["nullity"] > solver["maximum_nullity"]
                or metrics["effective_rank"]
                != metrics["dimension"] - metrics["nullity"]
                or metrics["full_column_rank"]
                is not (metrics["nullity"] == 0)
                or metrics["rank_policy"] != solver["rank_policy"]
                or metrics["maximum_nullity"]
                != solver["maximum_nullity"]
                or metrics["rank_relative_tolerance"]
                != solver["rank_relative_tolerance"]
                or metrics["maximum_condition_number"]
                != solver["maximum_condition_number"]
                or metrics[
                    "ridge_relative_to_mean_gram_diagonal"
                ]
                != solver[
                    "ridge_relative_to_mean_gram_diagonal"
                ]
                or any(
                    not isinstance(metrics[field], (float, int))
                    or isinstance(metrics[field], bool)
                    or not math.isfinite(float(metrics[field]))
                    for field in (
                        "active_condition_number",
                        "ridge_absolute",
                        "fit_rmse",
                        "fit_nrmse",
                    )
                )
                or float(metrics["active_condition_number"]) <= 0
                or float(metrics["active_condition_number"])
                > float(metrics["maximum_condition_number"])
                or float(metrics["ridge_absolute"]) < 0
                or float(metrics["fit_rmse"]) < 0
                or float(metrics["fit_nrmse"]) < 0
            ):
                raise ValueError(
                    "structured bootstrap operator metrics are invalid"
                )
        expected_norm_shapes = {
            "attention_input_norm": (
                residual_width,
                row_selection["selected_rows"],
            ),
            "attention_output_norm": (
                residual_width,
                row_selection["selected_rows"],
            ),
            "feed_forward_input_norm": (
                residual_width,
                row_selection["selected_rows"],
            ),
            "feed_forward_output_norm": (
                residual_width,
                row_selection["selected_rows"],
            ),
            "attention.q_norm": (
                head_dimension,
                row_selection["selected_rows"] * query_heads,
            ),
            "attention.k_norm": (
                head_dimension,
                row_selection["selected_rows"] * key_value_heads,
            ),
        }
        norm_metric_fields = {
            "rows",
            "width",
            "identified_coordinates",
            "fit_nrmse",
            "weight_minimum",
            "weight_median",
            "weight_maximum",
            "weight_rms",
        }
        if set(normalizations) != set(expected_norm_shapes):
            raise ValueError(
                "structured bootstrap normalization report names are invalid"
            )
        for norm_name, (width, expected_rows) in (
            expected_norm_shapes.items()
        ):
            metrics = _strict_mapping(
                normalizations[norm_name],
                label=f"structured bootstrap {norm_name} metrics",
                fields=norm_metric_fields,
            )
            if (
                metrics["rows"] != expected_rows
                or metrics["width"] != width
                or metrics["identified_coordinates"] != width
                or any(
                    not isinstance(metrics[field], (float, int))
                    or isinstance(metrics[field], bool)
                    or not math.isfinite(float(metrics[field]))
                    for field in (
                        "fit_nrmse",
                        "weight_minimum",
                        "weight_median",
                        "weight_maximum",
                        "weight_rms",
                    )
                )
                or float(metrics["fit_nrmse"]) < 0
                or float(metrics["weight_rms"]) < 0
                or not (
                    float(metrics["weight_minimum"])
                    <= float(metrics["weight_median"])
                    <= float(metrics["weight_maximum"])
                )
            ):
                raise ValueError(
                    "structured bootstrap normalization metrics are invalid"
                )
        shared_bindings.append(
            {
                "bootstrap": {
                    key: bootstrap[key]
                    for key in bootstrap
                    if key
                    not in {
                        "coefficient_sha256",
                    }
                },
                "capture_audit": capture,
            }
        )
    if (
        shared_bindings[0] != shared_bindings[1]
        or training[_PRIMARY]["bootstrap"]["coefficient_sha256"]
        != training[_CONTROL]["bootstrap"]["coefficient_sha256"]
    ):
        raise ValueError(
            "structured candidates do not share one bootstrap capture"
        )


def run_gemma3_structured_single_layer_experiment(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_length: int = 256,
    tokenization_batch_size: int = 1,
    fisher_floor: float = DEFAULT_FISHER_FLOOR,
    delta_scale_floor: float = DEFAULT_RIDGE_SCALE_FLOOR,
    relative_median_scale_floor: float = (
        DEFAULT_RELATIVE_MEDIAN_SCALE_FLOOR
    ),
    local_warmup_steps: int | None = None,
    train_steps: int | None = None,
    train_positions_per_sequence: int = (
        DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
    ),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    structured_loss_scale: float = DEFAULT_STRUCTURED_LOSS_SCALE,
    output_fisher_weight: float = DEFAULT_OUTPUT_FISHER_WEIGHT,
    coordinate_loss_weight: float = DEFAULT_COORDINATE_LOSS_WEIGHT,
    energy_loss_weight: float = DEFAULT_ENERGY_LOSS_WEIGHT,
    ground_truth_weight: float = DEFAULT_GROUND_TRUTH_WEIGHT,
    teacher_kl_weight: float = DEFAULT_TEACHER_KL_WEIGHT,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    selection_teacher_kl_max: float = DEFAULT_TEACHER_KL_MAX,
    selection_p90_abs_nll_max: float = (
        DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
    ),
    selection_p10_top1_min: float = (
        DEFAULT_PER_PROMPT_P10_TOP1_MIN
    ),
    block_delta_nrmse_max: float = DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    block_delta_cosine_min: float = DEFAULT_BLOCK_DELTA_COSINE_MIN,
    branch_delta_nrmse_max: float = DEFAULT_BRANCH_DELTA_NRMSE_MAX,
    branch_delta_cosine_min: float = DEFAULT_BRANCH_DELTA_COSINE_MIN,
    native_parity_tolerance: float = DEFAULT_NATIVE_PARITY_TOLERANCE,
    minimum_calibration_a_prompts: int = (
        DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
    ),
    minimum_heldout_prompts: int = DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    minimum_fisher_rows: int = DEFAULT_MINIMUM_FISHER_ROWS,
    minimum_train_supervised_tokens: int = (
        DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
    ),
    minimum_heldout_supervised_tokens: int = (
        DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
    ),
    minimum_length_buckets: int = DEFAULT_MINIMUM_LENGTH_BUCKETS,
    seed: int = DEFAULT_OPTIMIZATION_SEED,
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
    calibration_b_ledger_dir: Path | str | None = None,
    operator_bootstrap: bool = False,
    operator_bootstrap_rows: int = (
        DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS
    ),
    operator_bootstrap_ridge_relative: float = (
        DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE
    ),
    operator_bootstrap_rank_rtol: float = (
        DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL
    ),
    operator_bootstrap_max_condition: float = (
        DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION
    ),
    operator_bootstrap_maximum_nullity: int = (
        DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY
    ),
    stop_after_calibration_a: bool = False,
    progress: bool = False,
) -> dict[str, object]:
    """Fit on A, lock on B, and conditionally validate one Gemma layer."""

    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    prompts = load_gemma3_prompt_splits(prompt_path)
    _require_prompt_protocol(
        prompts,
        minimum_calibration_a_prompts=minimum_calibration_a_prompts,
        minimum_heldout_prompts=minimum_heldout_prompts,
    )
    families: PromptFamilyManifest = load_prompt_family_manifest(
        family_path,
        prompts=prompts,
    )
    prompt_exclusions = _tracked_prompt_exclusion_audit(
        prompts,
        prompt_path=prompt_path,
    )
    prompt_metadata = prompts.metadata()
    family_metadata = {
        **families.metadata(),
        **_format4_family_binding(prompts, families),
    }
    corpus_audit = _corpus_audit_binding(
        corpus_audit_path,
        prompts=prompts,
        prompt_path=prompt_path,
        family_path=family_path,
    )
    if type(operator_bootstrap) is not bool:
        raise TypeError("operator_bootstrap must be boolean")
    if type(stop_after_calibration_a) is not bool:
        raise TypeError("stop_after_calibration_a must be boolean")
    local_warmup_steps = (
        0
        if local_warmup_steps is None and operator_bootstrap
        else (
            DEFAULT_LOCAL_WARMUP_STEPS
            if local_warmup_steps is None
            else local_warmup_steps
        )
    )
    train_steps = (
        0
        if train_steps is None and operator_bootstrap
        else DEFAULT_TRAIN_STEPS
        if train_steps is None
        else train_steps
    )
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    for label, value, minimum in (
        ("maximum length", max_length, 2),
        ("tokenization batch size", tokenization_batch_size, 1),
        ("local warmup steps", local_warmup_steps, 0),
        ("downstream training steps", train_steps, 0),
        (
            "training positions per sequence",
            train_positions_per_sequence,
            1,
        ),
        ("optimization seed", seed, 0),
        (
            "minimum calibration-A prompts",
            minimum_calibration_a_prompts,
            1,
        ),
        ("minimum heldout prompts", minimum_heldout_prompts, 1),
        ("minimum Fisher rows", minimum_fisher_rows, 1),
        (
            "minimum train supervised tokens",
            minimum_train_supervised_tokens,
            1,
        ),
        (
            "minimum heldout supervised tokens",
            minimum_heldout_supervised_tokens,
            1,
        ),
        ("minimum length buckets", minimum_length_buckets, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if local_warmup_steps + train_steps <= 0:
        if not operator_bootstrap:
            raise ValueError(
                "structured training requires at least one update"
            )
    if operator_bootstrap and (
        local_warmup_steps != 0 or train_steps != 0
    ):
        raise ValueError(
            "operator bootstrap requires exactly zero Adam and suffix updates"
        )
    if (
        type(operator_bootstrap_rows) is not int
        or operator_bootstrap_rows <= 0
    ):
        raise ValueError("operator bootstrap rows must be positive")
    if (
        type(operator_bootstrap_maximum_nullity) is not int
        or operator_bootstrap_maximum_nullity < 0
    ):
        raise ValueError(
            "operator bootstrap maximum nullity must be nonnegative"
        )
    operator_bootstrap_ridge_relative = _finite(
        operator_bootstrap_ridge_relative,
        label="operator bootstrap ridge relative",
        minimum=torch.finfo(torch.float64).tiny,
    )
    operator_bootstrap_rank_rtol = _finite(
        operator_bootstrap_rank_rtol,
        label="operator bootstrap rank relative tolerance",
        minimum=torch.finfo(torch.float64).tiny,
        maximum=1.0,
    )
    if operator_bootstrap_rank_rtol >= 1.0:
        raise ValueError(
            "operator bootstrap rank relative tolerance must be less than 1"
        )
    operator_bootstrap_max_condition = _finite(
        operator_bootstrap_max_condition,
        label="operator bootstrap maximum condition",
        minimum=1.0,
    )
    if operator_bootstrap_max_condition <= 1.0:
        raise ValueError(
            "operator bootstrap maximum condition must exceed 1"
        )
    fisher_floor = _finite(
        fisher_floor,
        label="Fisher floor",
        minimum=torch.finfo(torch.float64).tiny,
        maximum=1.0,
    )
    delta_scale_floor = _finite(
        delta_scale_floor,
        label="delta scale floor",
        minimum=torch.finfo(torch.float64).tiny,
    )
    relative_median_scale_floor = _finite(
        relative_median_scale_floor,
        label="relative median scale floor",
        minimum=0.0,
        maximum=1.0,
    )
    learning_rate = _finite(
        learning_rate,
        label="learning rate",
        minimum=torch.finfo(torch.float64).tiny,
    )
    weight_decay = _finite(
        weight_decay,
        label="weight decay",
        minimum=0.0,
    )
    gradient_clip_norm = _finite(
        gradient_clip_norm,
        label="gradient clip norm",
        minimum=torch.finfo(torch.float64).tiny,
    )
    ground_truth_weight = _finite(
        ground_truth_weight,
        label="ground-truth weight",
        minimum=0.0,
    )
    teacher_kl_weight = _finite(
        teacher_kl_weight,
        label="teacher KL weight",
        minimum=0.0,
    )
    coordinate_loss_weight = _finite(
        coordinate_loss_weight,
        label="coordinate loss weight",
        minimum=0.0,
    )
    energy_loss_weight = _finite(
        energy_loss_weight,
        label="energy loss weight",
        minimum=0.0,
    )
    if (
        not operator_bootstrap
        and coordinate_loss_weight == 0
        and energy_loss_weight == 0
    ):
        raise ValueError(
            "coordinate or energy loss weight must be positive"
        )
    thresholds = {
        "nll_atol": _finite(
            selection_nll_atol,
            label="selection NLL tolerance",
            minimum=0.0,
        ),
        "top1_min": _finite(
            selection_top1_min,
            label="selection top-1 minimum",
            minimum=0.0,
            maximum=1.0,
        ),
        "teacher_kl_max": _finite(
            selection_teacher_kl_max,
            label="selection KL maximum",
            minimum=0.0,
        ),
        "p90_abs_nll_max": _finite(
            selection_p90_abs_nll_max,
            label="selection p90 NLL maximum",
            minimum=0.0,
        ),
        "p10_top1_min": _finite(
            selection_p10_top1_min,
            label="selection p10 top-1 minimum",
            minimum=0.0,
            maximum=1.0,
        ),
        "block_delta_nrmse_max": _finite(
            block_delta_nrmse_max,
            label="block delta NRMSE maximum",
            minimum=0.0,
        ),
        "block_delta_cosine_min": _finite(
            block_delta_cosine_min,
            label="block delta cosine minimum",
            minimum=-1.0,
            maximum=1.0,
        ),
        "branch_delta_nrmse_max": _finite(
            branch_delta_nrmse_max,
            label="branch delta NRMSE maximum",
            minimum=0.0,
        ),
        "branch_delta_cosine_min": _finite(
            branch_delta_cosine_min,
            label="branch delta cosine minimum",
            minimum=-1.0,
            maximum=1.0,
        ),
        "native_parity_tolerance": _finite(
            native_parity_tolerance,
            label="native parity tolerance",
            minimum=0.0,
        ),
    }
    weights = _scaled_weights(
        structured_loss_scale=structured_loss_scale,
        output_fisher_weight=output_fisher_weight,
    )
    artifact_format_version = (
        _ARTIFACT_FORMAT_VERSION
        if operator_bootstrap
        else _RMS_ARTIFACT_FORMAT_VERSION
    )
    training_recipe = (
        _operator_bootstrap_training_recipe_payload(
            train_positions_per_sequence=train_positions_per_sequence,
            requested_rows=operator_bootstrap_rows,
            ridge_relative=operator_bootstrap_ridge_relative,
            rank_relative_tolerance=operator_bootstrap_rank_rtol,
            maximum_condition_number=(
                operator_bootstrap_max_condition
            ),
            maximum_nullity=operator_bootstrap_maximum_nullity,
        )
        if operator_bootstrap
        else _training_recipe_payload(
            local_warmup_steps=local_warmup_steps,
            train_steps=train_steps,
            train_positions_per_sequence=train_positions_per_sequence,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            weights=weights,
            coordinate_loss_weight=coordinate_loss_weight,
            energy_loss_weight=energy_loss_weight,
            ground_truth_weight=ground_truth_weight,
            teacher_kl_weight=teacher_kl_weight,
        )
    )
    resolved_output = (
        default_gemma3_structured_single_layer_output(model_id, layer_index)
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report_path = resolved_output.with_suffix(".json")
    calibration_a_report_path = resolved_output.with_suffix(
        ".calibration-a.json"
    )
    if (
        resolved_output.exists()
        or report_path.exists()
        or calibration_a_report_path.exists()
    ):
        raise FileExistsError(
            "refusing to overwrite a structured-layer scientific artifact"
        )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    raw_prompt_hashes = prompt_metadata["per_prompt_sha256"]
    assert isinstance(raw_prompt_hashes, Mapping)
    calibration_b_prompt_hashes = raw_prompt_hashes["calibration_b"]
    assert isinstance(calibration_b_prompt_hashes, list)
    resolved_ledger_dir = (
        Path(".local-runs")
        / "heldout-ledger"
        / "structured-calibration-b"
        if calibration_b_ledger_dir is None
        else Path(calibration_b_ledger_dir)
    )
    calibration_b_claim_path = _calibration_b_claim_path(
        resolved_ledger_dir,
        calibration_b_prompt_hashes,
    )
    if not stop_after_calibration_a and calibration_b_claim_path.exists():
        raise FileExistsError(
            "calibration B was already claimed; refusing heldout reuse: "
            f"{calibration_b_claim_path}"
        )

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "structured activation-Fisher audit requires CPU or CUDA float64"
        )
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(layer_index, layer_index)
    layer_id = plan.layer_ids[0]
    layer = adapter.layer(layer_id)
    provenance = structured_layer_provenance(adapter, layer_id)
    primary_config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        layer,
        causal_edges_enabled=True,
    )
    source_manifest = _source_accounting_manifest(
        adapter,
        layer_ids=plan.layer_ids,
    )
    source_static = _source_block_static(adapter, plan)
    if (
        source_static["parameter_count"] != source_manifest["parameter_count"]
        or source_static["parameter_bytes"]
        != source_manifest["parameter_bytes"]
    ):
        raise RuntimeError("source accounting manifests disagree")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )

    train_batches, train_stream = _materialize_split(
        tokenizer,
        prompts.calibration_a,
        split_name="calibration_a",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    train_contract = _tokenized_stream_contract(
        train_stream,
        split_name="calibration_a",
        minimum_supervised_tokens=minimum_train_supervised_tokens,
        minimum_length_buckets=minimum_length_buckets,
    )
    training = collect_structured_training_batches(
        adapter,
        train_batches,
        layer_id=layer_id,
        positions_per_sequence=train_positions_per_sequence,
    )
    _require_complete_middle_layer_demand(
        adapter,
        training,  # type: ignore[arg-type]
    )
    calibration_split_sha256 = train_stream["serialized_sha256"]
    if not isinstance(calibration_split_sha256, str):
        raise ValueError("calibration-A stream digest is invalid")
    scales = estimate_structured_layer_scales(
        tuple(item.targets for item in training),
        calibration_split_sha256=calibration_split_sha256,
        floor=delta_scale_floor,
        relative_median_floor=relative_median_scale_floor,
    )
    fisher_matrix, fisher_report = compute_structured_activation_fisher(
        adapter,
        training,
        plan=plan,
    )
    if (
        int(fisher_report["effective_nonzero_gradient_rows"])
        < minimum_fisher_rows
    ):
        raise ValueError(
            "effective activation-Fisher row count is below the minimum"
        )
    fisher_metric_tensor, fisher_metric_report = (
        _normalized_fisher_metric(
            fisher_matrix,
            scales.output,
            eigenvalue_floor=fisher_floor,
        )
    )
    output_fisher = StructuredOutputFisherMetric(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        delta_scale=scales.output,
        standardized_coordinate_metric=fisher_metric_tensor,
    )
    bootstrap_captures: tuple[
        StructuredOperatorCaptureBatch,
        ...,
    ] | None = None
    bootstrap_row_selection: StructuredOperatorRowSelection | None = None
    bootstrap_capture_audit: dict[str, object] | None = None
    if operator_bootstrap:
        (
            bootstrap_captures,
            bootstrap_row_selection,
            bootstrap_capture_audit,
        ) = _capture_compact_operator_bootstrap_rows(
            adapter,
            training,
            layer=layer,
            calibration_split_sha256=calibration_split_sha256,
            requested_rows=operator_bootstrap_rows,
        )
        guard.assert_unchanged()

    candidates: dict[str, StructuredTransformerLayerExecutor] = {}
    training_reports: dict[str, dict[str, object]] = {}
    structural_probes: dict[str, dict[str, object]] = {}
    rmsnorm_initialization: dict[str, dict[str, object]] = {}

    def fit_candidate(
        name: str,
        *,
        causal_edges_enabled: bool,
    ) -> StructuredTransformerLayerExecutor:
        candidate = make_structured_executor(
            adapter,
            layer_id=layer_id,
            causal_edges_enabled=causal_edges_enabled,
            seed=seed,
            device=device,
        )
        if operator_bootstrap:
            assert bootstrap_captures is not None
            assert bootstrap_row_selection is not None
            assert bootstrap_capture_audit is not None
            raw_bootstrap_report = bootstrap_structured_operator_executor_(
                candidate,
                bootstrap_captures,
                layer=layer,
                calibration_split_sha256=calibration_split_sha256,
                source_segment_fingerprint=(
                    provenance.source_segment_fingerprint
                ),
                requested_rows=operator_bootstrap_rows,
                ridge_relative=operator_bootstrap_ridge_relative,
                rank_relative_tolerance=operator_bootstrap_rank_rtol,
                maximum_condition_number=(
                    operator_bootstrap_max_condition
                ),
                maximum_nullity=operator_bootstrap_maximum_nullity,
                row_selection=bootstrap_row_selection,
            )
            training_reports[name] = _operator_bootstrap_report_binding(
                raw_bootstrap_report,
                capture_audit=bootstrap_capture_audit,
                final_execution_fingerprint=(
                    candidate.execution_fingerprint()
                ),
            )
        else:
            initialization = initialize_structured_rmsnorms_from_targets_(
                candidate,
                tuple(item.targets for item in training),
                calibration_split_sha256=calibration_split_sha256,
            )
            rmsnorm_initialization[name] = initialization
            training_reports[name] = fit_structured_executor(
                adapter,
                candidate,
                training,
                plan=plan,
                scales=scales,
                output_fisher=output_fisher,
                weights=weights,
                coordinate_loss_weight=coordinate_loss_weight,
                energy_loss_weight=energy_loss_weight,
                local_warmup_steps=local_warmup_steps,
                train_steps=train_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                gradient_clip_norm=gradient_clip_norm,
                ground_truth_weight=ground_truth_weight,
                teacher_kl_weight=teacher_kl_weight,
                progress_label=name if progress else None,
            )
            training_reports[name]["rmsnorm_initialization"] = (
                initialization
            )
            training_reports[name]["final_execution_fingerprint"] = (
                candidate.execution_fingerprint()
            )
        structural_probes[name] = _full_width_structural_probes(
            adapter,
            candidate,  # type: ignore[arg-type]
            training,  # type: ignore[arg-type]
        )
        structural_probes[name]["executor_execution_fingerprint"] = (
            candidate.execution_fingerprint()
        )
        if structural_probes[name]["passed"] is not True:
            raise RuntimeError(
                f"{name} failed causal or padding structural probes"
            )
        candidates[name] = candidate
        return candidate

    primary = fit_candidate(
        _PRIMARY,
        causal_edges_enabled=True,
    )
    _assert_source_independence(model, {_PRIMARY: primary})
    primary_state = primary.artifact_state_dict()
    strict_primary = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            primary_state,
            map_location=device,
        )
    )
    primary_a_fidelity = evaluate_calibration_a_fidelity(
        training,
        candidates={_PRIMARY: strict_primary},
        thresholds=thresholds,
    )
    if primary_a_fidelity["primary_passed"] is not True:
        guard.assert_unchanged()
        _write_calibration_a_preflight(
            calibration_a_report_path,
            model=model_metadata,
            layer_id=layer_id,
            calibration_split_sha256=calibration_split_sha256,
            artifact_format_version=artifact_format_version,
            training_recipe=training_recipe,
            rmsnorm_initialization=(
                None
                if operator_bootstrap
                else {_PRIMARY: rmsnorm_initialization[_PRIMARY]}
            ),
            training_reports={
                _PRIMARY: training_reports[_PRIMARY]
            },
            fidelity=primary_a_fidelity,
        )
        raise RuntimeError(
            "aggregate calibration-A direct fidelity failed; "
            "calibration B and the diagnostic control were not run; "
            f"see {calibration_a_report_path}"
        )
    if stop_after_calibration_a:
        guard.assert_unchanged()
        return _write_calibration_a_preflight(
            calibration_a_report_path,
            model=model_metadata,
            layer_id=layer_id,
            calibration_split_sha256=calibration_split_sha256,
            artifact_format_version=artifact_format_version,
            training_recipe=training_recipe,
            rmsnorm_initialization=(
                None
                if operator_bootstrap
                else {_PRIMARY: rmsnorm_initialization[_PRIMARY]}
            ),
            training_reports={
                _PRIMARY: training_reports[_PRIMARY]
            },
            fidelity=primary_a_fidelity,
            stopped_after_calibration_a=True,
        )

    fit_candidate(
        _CONTROL,
        causal_edges_enabled=False,
    )
    if operator_bootstrap:
        bootstrap_captures = None
    source_independence = _assert_source_independence(model, candidates)
    guard.assert_unchanged()

    # Heldout evaluation always uses strict executor roundtrips, never the
    # mutable in-memory training objects.
    executor_states = {
        name: candidate.artifact_state_dict()
        for name, candidate in candidates.items()
    }
    candidates = {
        name: StructuredTransformerLayerExecutor.from_artifact_state_dict(
            state,
            map_location=device,
        )
        for name, state in executor_states.items()
    }
    source_independence_after_reload = _assert_source_independence(
        model,
        candidates,
    )
    calibration_a_fidelity = evaluate_calibration_a_fidelity(
        training,
        candidates=candidates,
        thresholds=thresholds,
    )
    _write_calibration_a_preflight(
        calibration_a_report_path,
        model=model_metadata,
        layer_id=layer_id,
        calibration_split_sha256=calibration_split_sha256,
        artifact_format_version=artifact_format_version,
        training_recipe=training_recipe,
        rmsnorm_initialization=(
            None if operator_bootstrap else rmsnorm_initialization
        ),
        training_reports=training_reports,
        fidelity=calibration_a_fidelity,
    )
    if calibration_a_fidelity["primary_passed"] is not True:
        raise RuntimeError(
            "aggregate calibration-A direct fidelity failed; calibration B "
            f"was not tokenized; see {calibration_a_report_path}"
        )

    per_prompt_family_sha256 = family_metadata[
        "per_prompt_family_sha256"
    ]
    pair_sha256 = family_metadata[
        "ordered_prompt_family_pairs_sha256"
    ]
    assert isinstance(per_prompt_family_sha256, Mapping)
    assert isinstance(pair_sha256, Mapping)
    resolved_commit = model_metadata.get("resolved_commit")
    if not isinstance(resolved_commit, str):
        raise ValueError(
            "calibration-B claim requires an exact resolved model commit"
        )
    calibration_b_claim = _exclusive_calibration_b_claim(
        calibration_b_claim_path,
        prompt_hashes=calibration_b_prompt_hashes,
        family_hashes=per_prompt_family_sha256["calibration_b"],
        prompt_family_pair_sha256=pair_sha256["calibration_b"],
        corpus_audit=corpus_audit,
        prompt_fixture_file_sha256=_file_sha256(prompt_path),
        family_manifest_file_sha256=_file_sha256(family_path),
        resolved_commit=resolved_commit,
        layer_id=layer_id,
        training_recipe=training_recipe,
        thresholds=thresholds,
        executors=candidates,
    )
    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_contract = _tokenized_stream_contract(
        selection_stream,
        split_name="calibration_b",
        minimum_supervised_tokens=minimum_heldout_supervised_tokens,
        minimum_length_buckets=minimum_length_buckets,
    )
    content_audit = _assert_tokenized_content_disjointness(
        {
            "calibration_a": train_stream,
            "calibration_b": selection_stream,
        }
    )
    selection_result = evaluate_structured_candidates(
        adapter,
        selection_batches,
        plan=plan,
        layer_id=layer_id,
        candidates=candidates,
        native_parity_tolerance=native_parity_tolerance,
    )
    selection_boundaries = selection_result["boundaries"]
    assert isinstance(selection_boundaries, tuple)
    selection_source_macs = _source_block_macs(
        adapter,
        plan,
        selection_boundaries,
        static=source_static,
    )
    selection_payload = _evaluation_payload(
        selection_result,
        candidates=candidates,
        source_static=source_static,
        source_macs=selection_source_macs,
        thresholds=thresholds,
        tokenized_stream=selection_stream,
        tokenized_stream_contract=selection_contract,
    )
    selection_payload["reason"] = "calibration_b_one_shot_locked_evaluation"
    selection_passed = selection_payload["passed"] is True
    guard.assert_unchanged()

    validation_stream: Mapping[str, object] | None = None
    validation_contract: Mapping[str, object] | None = None
    if selection_passed:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        validation_contract = _tokenized_stream_contract(
            validation_stream,
            split_name="validation",
            minimum_supervised_tokens=minimum_heldout_supervised_tokens,
            minimum_length_buckets=minimum_length_buckets,
        )
        content_audit = _assert_tokenized_content_disjointness(
            {
                "calibration_a": train_stream,
                "calibration_b": selection_stream,
                "validation": validation_stream,
            }
        )
        validation_result = evaluate_structured_candidates(
            adapter,
            validation_batches,
            plan=plan,
            layer_id=layer_id,
            candidates=candidates,
            native_parity_tolerance=native_parity_tolerance,
        )
        validation_boundaries = validation_result["boundaries"]
        assert isinstance(validation_boundaries, tuple)
        validation_source_macs = _source_block_macs(
            adapter,
            plan,
            validation_boundaries,
            static=source_static,
        )
        validation_payload = _evaluation_payload(
            validation_result,
            candidates=candidates,
            source_static=source_static,
            source_macs=validation_source_macs,
            thresholds=thresholds,
            tokenized_stream=validation_stream,
            tokenized_stream_contract=validation_contract,
        )
        validation_payload["reason"] = (
            "calibration_b_passed_locked_executors_evaluated"
        )
    else:
        validation_payload = _unevaluated_validation_payload()
    validation_passed = validation_payload["passed"] is True
    guard.assert_unchanged()

    data_minima = {
        "minimum_calibration_a_prompts": minimum_calibration_a_prompts,
        "minimum_heldout_prompts_per_role": minimum_heldout_prompts,
        "minimum_effective_fisher_rows": minimum_fisher_rows,
        "minimum_train_supervised_tokens": (
            minimum_train_supervised_tokens
        ),
        "minimum_heldout_supervised_tokens_per_role": (
            minimum_heldout_supervised_tokens
        ),
        "minimum_populated_length_buckets_per_tokenized_role": (
            minimum_length_buckets
        ),
    }
    strong_minima = _strong_data_minima(
        minimum_calibration_a_prompts=minimum_calibration_a_prompts,
        minimum_heldout_prompts=minimum_heldout_prompts,
        minimum_fisher_rows=minimum_fisher_rows,
        minimum_train_supervised_tokens=minimum_train_supervised_tokens,
        minimum_heldout_supervised_tokens=(
            minimum_heldout_supervised_tokens
        ),
        minimum_length_buckets=minimum_length_buckets,
    )
    resolved_commit = model_metadata.get("resolved_commit")
    immutable_revision = (
        isinstance(resolved_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit) is not None
    )
    strong_training_recipe = _preregistered_representative_protocol(
        training_recipe=training_recipe,
        thresholds=thresholds,
        seed=seed,
        maximum_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        fisher_floor=fisher_floor,
        delta_scale_floor=delta_scale_floor,
        relative_median_scale_floor=relative_median_scale_floor,
        format_version=artifact_format_version,
    )
    corpus_audit_bound = corpus_audit is not None
    structured_fidelity_passed = (
        selection_passed
        and validation_passed
        and strong_minima
        and immutable_revision
        and strong_training_recipe
        and corpus_audit_bound
    )
    outcome = (
        "rejected_on_calibration_b"
        if not selection_passed
        else (
            "rejected_on_validation"
            if not validation_passed
            else (
                "single_layer_structured_fidelity_passed"
                if structured_fidelity_passed
                else "relaxed_protocol_passed_without_scientific_promotion"
            )
        )
    )
    tokenized_splits: dict[str, object] = {
        "calibration_a": train_stream,
        "calibration_b": selection_stream,
    }
    stream_contracts: dict[str, object] = {
        "calibration_a": train_contract,
        "calibration_b": selection_contract,
    }
    if validation_stream is not None and validation_contract is not None:
        tokenized_splits["validation"] = validation_stream
        stream_contracts["validation"] = validation_contract
    scales_artifact = _scales_payload(
        scales,
        floor=delta_scale_floor,
        relative_median_floor=relative_median_scale_floor,
        rows=int(train_contract["valid_tokens"]),
    )
    payload: dict[str, object] = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": artifact_format_version,
        "contains_model_weights": False,
        "contains_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_source_derived_statistics": True,
        "scientific_status": {
            "scope": (
                "source_free_native_shape_single_gemma_layer_fidelity"
            ),
            "outcome": outcome,
            "activation_fisher_computed": True,
            "all_eight_structured_targets_supervised": True,
            "calibration_a_executors_fitted": True,
            "calibration_b_evaluated": True,
            "calibration_b_passed": selection_passed,
            "validation_evaluated": selection_passed,
            "validation_passed": validation_passed,
            "test_evaluated": False,
            "strict_executor_reload_before_heldout": True,
            "strict_artifact_reload_required_and_run_before_return": True,
            "source_layer_calls_in_student_path": 0,
            "source_layer_removed_from_student_path": True,
            "source_independent_executor": True,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "strong_data_minima_enforced": strong_minima,
            "strong_training_recipe_enforced": strong_training_recipe,
            "corpus_audit_bound": corpus_audit_bound,
            "preregistered_fixed_schedule_enforced": (
                strong_training_recipe
            ),
            "immutable_model_revision_recorded": immutable_revision,
            "single_layer_structured_fidelity_passed": (
                structured_fidelity_passed
            ),
            "general_method_viable": False,
            "model_level_promotion_authorized": False,
            "compression_attempted": False,
            "parameter_reduction_supported": False,
            "analytic_mac_reduction_supported": False,
            "resource_values_used_as_fidelity_gates": False,
            "attention_value_separation_observed": selection_payload[
                "gates"
            ]["attention_value_separation_observed"],  # type: ignore[index]
            "decode_supported": False,
            "latency_or_kernel_speed_claim": False,
        },
        "model": model_metadata,
        "protocol": {
            "layer_index": layer_index,
            "layer_ids": plan.layer_ids,
            "canonical_boundaries": plan.activation_sites,
            "residual_width": primary_config.residual_width,
            "source_layer_provenance": _provenance_payload(provenance),
            "executor_architecture": (
                candidates[_PRIMARY].architecture_manifest()
            ),
            "attention_output_disabled_control": True,
            "maximum_tokenized_length": max_length,
            "tokenization_batch_size": tokenization_batch_size,
            "optimization_seed": seed,
            "single_seed_protocol": True,
            **(
                {
                    "fitting_method": (
                        _OPERATOR_BOOTSTRAP_FITTING_METHOD
                    ),
                    "operator_bootstrap_enabled": True,
                }
                if operator_bootstrap
                else {}
            ),
            "training_recipe": training_recipe,
            "fisher_eigenvalue_floor": fisher_floor,
            "delta_scale_floor": delta_scale_floor,
            "relative_median_scale_floor": (
                relative_median_scale_floor
            ),
            "native_parity_tolerance": native_parity_tolerance,
            "numeric_policy": {
                "requested_device": device_name,
                "resolved_device": str(device),
                "requested_model_dtype": dtype,
                "resolved_model_dtype": model_metadata["dtype"],
                "executor_parameter_dtype": "torch.float32",
                "fisher_accumulation_dtype": "torch.float64",
                "structured_loss_dtype": "torch.float32",
            },
            "source_accounting_manifest": source_manifest,
            "training_split": "calibration_a_only",
            "calibration_a_split_sha256": calibration_split_sha256,
            "selection_policy": (
                "one_shot_calibration_b_all_fidelity_gates_no_resource_gate"
            ),
            "calibration_b_claim": calibration_b_claim,
            "validation_policy": (
                "tokenize_once_only_after_calibration_b_passes"
            ),
            "test_policy": "parse_validate_hash_only",
            "student_execution": (
                "native_prefix_structured_executor_native_suffix"
            ),
            "ordinary_native_forward_is_heldout_baseline": True,
            "segmented_native_forward_is_parity_audit_only": True,
            "native_layer_output_available_to_student": False,
            "thresholds": thresholds,
            "data_minima": data_minima,
            "strong_data_minima_enforced": strong_minima,
            "prompt_splits": prompt_metadata,
            "prompt_families": family_metadata,
            "prompt_exclusions": prompt_exclusions,
            "corpus_audit": corpus_audit,
            "prompt_fixture_file_sha256": _file_sha256(prompt_path),
            "family_manifest_file_sha256": _file_sha256(family_path),
            "tokenized_splits": tokenized_splits,
            "tokenized_stream_contracts": stream_contracts,
            "tokenized_content_disjointness": content_audit,
            "library_versions": _library_versions(),
            "tokenizer": _tokenizer_provenance(tokenizer),
            "model_state_guard": guard.metadata(),
            "source_independence": source_independence,
            "source_independence_after_reload": (
                source_independence_after_reload
            ),
            "resource_gates_applied": False,
            "compression_not_attempted": True,
        },
        "executors": executor_states,
        "training": {
            "source_layer_provenance": _provenance_payload(provenance),
            "calibration_split_sha256": calibration_split_sha256,
            "structured_scales": scales_artifact,
            "activation_fisher": {
                **fisher_report,
                "matrix": fisher_matrix,
            },
            "output_fisher": {
                "provenance": _provenance_payload(provenance),
                "calibration_split_sha256": calibration_split_sha256,
                "delta_scale": output_fisher.delta_scale,
                "standardized_coordinate_metric": (
                    output_fisher.standardized_coordinate_metric
                ),
                "delta_scale_sha256": _tensor_sha256(
                    output_fisher.delta_scale,
                    domain=(
                        b"fisher_graph.structured_layer.output_scale.v1\0"
                    ),
                ),
                "metric_report": fisher_metric_report,
            },
            _PRIMARY: training_reports[_PRIMARY],
            _CONTROL: training_reports[_CONTROL],
            "calibration_a_fidelity": calibration_a_fidelity,
            "structural_probes": structural_probes,
            "tokenized_stream": train_stream,
        },
        "selection": selection_payload,
        "validation": validation_payload,
    }
    digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        tensor_file=str(resolved_output),
        scientific_digest=digest,
    )
    report_digest = _report_sha256(report)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
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
    loaded = load_gemma3_structured_single_layer_artifact(
        resolved_output,
        map_location="cpu",
    )
    return loaded["report"]  # type: ignore[return-value]


def _validate_tensor_locations(value: object) -> None:
    allowed_exact = {
        ("training", "activation_fisher", "matrix"),
        ("training", "output_fisher", "delta_scale"),
        (
            "training",
            "output_fisher",
            "standardized_coordinate_metric",
        ),
        *{
            ("training", "structured_scales", "values", name)
            for name in _SCALE_NAMES
        },
    }

    def visit(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, Tensor):
            allowed_executor = (
                len(path) >= 4
                and path[0] == "executors"
                and path[1] in _CANDIDATES
                and path[2] == "model_state_dict"
            )
            if path not in allowed_exact and not allowed_executor:
                raise ValueError(
                    "structured artifact tensor appears at an invalid path: "
                    + ".".join(path)
                )
            return
        if isinstance(current, Mapping):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "structured artifact mappings require string keys"
                    )
                visit(child, (*path, key))
        elif isinstance(current, (tuple, list)):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(value, ())


def _strict_mapping(
    value: object,
    *,
    label: str,
    fields: set[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _validate_accounting_section(
    section: Mapping[str, object],
    *,
    executors: Mapping[str, StructuredTransformerLayerExecutor],
    source_manifest: Mapping[str, object],
    split_name: str,
) -> None:
    stream, _ = _validated_tokenized_stream(
        section["tokenized_stream"],
        split_name=split_name,
    )
    logical = _strict_mapping(
        section["logical_accounting"],
        label=f"{split_name} logical accounting",
    )
    accounting = _strict_mapping(
        section["accounting"],
        label=f"{split_name} accounting",
    )
    source_macs = _source_macs_from_stream(source_manifest, stream)
    for name, executor in executors.items():
        expected_logical = _logical_accounting_from_stream(stream, executor)
        if logical.get(name) != expected_logical:
            raise ValueError(
                f"{split_name} {name} logical accounting does not recompute"
            )
        expected_accounting = _candidate_accounting(
            executor,
            expected_logical,
            source_static={
                "parameter_count": source_manifest["parameter_count"],
            },
            source_macs=source_macs,
        )
        if accounting.get(name) != expected_accounting:
            raise ValueError(
                f"{split_name} {name} accounting does not recompute"
            )


def load_gemma3_structured_single_layer_artifact(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly restore an artifact and recompute its scientific decisions."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or set(raw) != _OUTER_FIELDS:
        raise ValueError("structured-layer artifact fields are invalid")
    format_version = raw.get("format_version")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or format_version not in _SUPPORTED_ARTIFACT_FORMAT_VERSIONS
        or raw["contains_model_weights"] is not False
        or raw["contains_executor_weights"] is not True
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or raw["contains_teacher_targets"] is not False
        or raw["contains_source_derived_statistics"] is not True
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("structured-layer artifact header is invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("structured scientific payload digest mismatch")
    _validate_tensor_locations(payload)

    protocol = _strict_mapping(
        raw["protocol"],
        label="structured protocol",
    )
    training = _strict_mapping(
        raw["training"],
        label="structured training",
    )
    selection = _strict_mapping(
        raw["selection"],
        label="structured selection",
    )
    validation = _strict_mapping(
        raw["validation"],
        label="structured validation",
    )
    status = _strict_mapping(
        raw["scientific_status"],
        label="structured scientific status",
    )
    training_recipe = _strict_mapping(
        protocol.get("training_recipe"),
        label="structured training recipe",
    )
    _validate_training_recipe_binding(
        training_recipe,
        training=training,
        format_version=format_version,  # type: ignore[arg-type]
    )
    corpus_audit_bound = _validate_corpus_audit_binding(
        protocol.get("corpus_audit"),
        protocol=protocol,
    )
    raw_executors = _strict_mapping(
        raw["executors"],
        label="structured executors",
        fields=set(_CANDIDATES),
    )
    executors = {
        name: StructuredTransformerLayerExecutor.from_artifact_state_dict(
            raw_executors[name],  # type: ignore[arg-type]
            map_location=map_location,
        )
        for name in _CANDIDATES
    }
    if (
        not executors[_PRIMARY].config.causal_edges_enabled
        or executors[_CONTROL].config.causal_edges_enabled
    ):
        raise ValueError("structured candidate causal controls are invalid")
    primary_config = executors[_PRIMARY].config.to_dict()
    control_config = executors[_CONTROL].config.to_dict()
    if primary_config != {
        **control_config,
        "causal_edges_enabled": True,
    }:
        raise ValueError("structured candidate architectures do not match")
    architecture = protocol.get("executor_architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture
        != executors[_PRIMARY].architecture_manifest()
        or protocol.get("residual_width") != executors[_PRIMARY].width
    ):
        raise ValueError("structured executor protocol binding is invalid")

    provenance = _restore_provenance(
        training["source_layer_provenance"]
    )
    if (
        provenance
        != _restore_provenance(protocol["source_layer_provenance"])
        or training["calibration_split_sha256"]
        != protocol["calibration_a_split_sha256"]
    ):
        raise ValueError("structured source provenance binding is invalid")
    tokenized_splits = _strict_mapping(
        protocol["tokenized_splits"],
        label="structured tokenized splits",
    )
    expected_split_names = {"calibration_a", "calibration_b"}
    if validation.get("evaluated") is True:
        expected_split_names.add("validation")
    if set(tokenized_splits) != expected_split_names or "test" in (
        tokenized_splits
    ):
        raise ValueError(
            "structured tokenized roles violate the reserved-test policy"
        )
    validated_streams = {
        split_name: _validated_tokenized_stream(
            tokenized_splits[split_name],
            split_name=split_name,
        )[0]
        for split_name in expected_split_names
    }
    if (
        training.get("tokenized_stream")
        != validated_streams["calibration_a"]
        or selection.get("tokenized_stream")
        != validated_streams["calibration_b"]
        or (
            "validation" in validated_streams
            and validation.get("tokenized_stream")
            != validated_streams["validation"]
        )
        or (
            "validation" not in validated_streams
            and validation.get("tokenized_stream") is not None
        )
        or training["calibration_split_sha256"]
        != validated_streams["calibration_a"]["serialized_sha256"]
    ):
        raise ValueError("structured tokenized-stream binding is invalid")
    format4_prompt_hashes: dict[str, list[str]] | None = None
    format4_family_hashes: dict[str, list[str]] | None = None
    if format_version >= 4:
        (
            format4_prompt_hashes,
            format4_family_hashes,
        ) = _validate_format4_prompt_family_binding(
            protocol,
            streams=validated_streams,
        )
    elif protocol.get("calibration_b_claim") is not None:
        raise ValueError(
            "format-3 structured artifact contains a format-4 heldout claim"
        )
    raw_scales = _strict_mapping(
        training["structured_scales"],
        label="structured scales",
    )
    scales = _restore_scales(raw_scales)
    protocol_delta_scale_floor = _finite(
        protocol.get("delta_scale_floor"),
        label="structured protocol delta scale floor",
        minimum=torch.finfo(torch.float64).tiny,
    )
    protocol_fisher_floor = _finite(
        protocol.get("fisher_eigenvalue_floor"),
        label="structured protocol Fisher floor",
        minimum=torch.finfo(torch.float64).tiny,
        maximum=1.0,
    )
    protocol_relative_median_floor = _finite(
        protocol.get("relative_median_scale_floor"),
        label="structured protocol relative median scale floor",
        minimum=0.0,
        maximum=1.0,
    )
    calibration_a_valid_rows = _logical_accounting_from_stream(
        validated_streams["calibration_a"],
        executors[_PRIMARY],
    )["valid_tokens"]
    if (
        scales.provenance != provenance
        or scales.calibration_split_sha256
        != training["calibration_split_sha256"]
        or float(raw_scales["floor"])
        != protocol_delta_scale_floor
        or float(raw_scales["relative_median_floor"])
        != protocol_relative_median_floor
    ):
        raise ValueError("structured scales provenance is invalid")
    if raw_scales["valid_rows"] != calibration_a_valid_rows:
        raise ValueError(
            "structured scale rows do not match calibration-A valid tokens"
        )
    if format_version == 4:
        initialization_reports = []
        for name in _CANDIDATES:
            candidate_training = _strict_mapping(
                training[name],
                label=f"structured {name} training report",
            )
            initialization = candidate_training.get(
                "rmsnorm_initialization"
            )
            _validate_rmsnorm_initialization_binding(
                initialization,
                provenance=provenance,
                calibration_split_sha256=(
                    training["calibration_split_sha256"]
                ),  # type: ignore[arg-type]
                valid_rows=calibration_a_valid_rows,
                width=scales.width,
            )
            initialization_reports.append(initialization)
        if initialization_reports[0] != initialization_reports[1]:
            raise ValueError(
                "structured candidate RMSNorm initializations disagree"
            )
    elif format_version >= 5:
        layer_index = protocol.get("layer_index")
        if (
            protocol.get("fitting_method")
            != _OPERATOR_BOOTSTRAP_FITTING_METHOD
            or protocol.get("operator_bootstrap_enabled") is not True
            or type(layer_index) is not int
            or layer_index < 0
        ):
            raise ValueError(
                "structured operator-bootstrap protocol is invalid"
            )
        _validate_operator_bootstrap_training_binding(
            training=training,
            training_recipe=training_recipe,
            executors=executors,
            provenance=provenance,
            layer_index=layer_index,
        )
    elif (
        "fitting_method" in protocol
        or "operator_bootstrap_enabled" in protocol
        or any("bootstrap" in training[name] for name in _CANDIDATES)
    ):
        raise ValueError(
            "legacy structured artifact contains bootstrap fields"
        )
    if format_version >= 4:
        structural_probes = _strict_mapping(
            training.get("structural_probes"),
            label="structured structural probes",
            fields=set(_CANDIDATES),
        )
        for name in _CANDIDATES:
            fingerprint = executors[name].execution_fingerprint()
            if training[name].get(
                "final_execution_fingerprint"
            ) != fingerprint:
                raise ValueError(
                    "structured training executor fingerprint is invalid"
                )
            probe = _strict_mapping(
                structural_probes[name],
                label=f"structured {name} structural probe",
            )
            if (
                probe.get("passed") is not True
                or probe.get("executor_execution_fingerprint")
                != fingerprint
            ):
                raise ValueError(
                    "structured probe executor fingerprint is invalid"
                )
    elif any(
        "final_execution_fingerprint" in training[name]
        for name in _CANDIDATES
    ) or (
        isinstance(training.get("structural_probes"), Mapping)
        and any(
            "executor_execution_fingerprint" in probe
            for probe in training["structural_probes"].values()
            if isinstance(probe, Mapping)
        )
    ):
        raise ValueError(
            "format-3 structured artifact contains format-4 fingerprints"
        )
    for name in _SCALE_NAMES:
        scale = getattr(scales, name)
        recorded_lower_bound = max(
            protocol_delta_scale_floor,
            protocol_relative_median_floor
            * float(scale.median().item()),
        )
        if float(scale.min().item()) < recorded_lower_bound:
            raise ValueError(
                f"structured scale {name!r} violates its recorded floor"
            )
    fisher = _strict_mapping(
        training["activation_fisher"],
        label="structured activation Fisher",
    )
    matrix = fisher.get("matrix")
    calibration_examples = validated_streams["calibration_a"].get(
        "examples"
    )
    positions_per_sequence = training_recipe[
        "train_positions_per_sequence"
    ]
    expected_selected_rows = (
        sum(
            min(example["supervised_positions"], positions_per_sequence)
            for example in calibration_examples
        )
        if isinstance(calibration_examples, list)
        and all(isinstance(example, Mapping) for example in calibration_examples)
        else -1
    )
    if (
        not isinstance(matrix, Tensor)
        or matrix.dtype is not torch.float64
        or matrix.device.type != "cpu"
        or matrix.shape != (scales.width, scales.width)
        or not bool(torch.isfinite(matrix).all())
        or fisher.get("matrix_sha256")
        != _tensor_sha256(
            matrix,
            domain=b"fisher_graph.structured_layer.raw_fisher.v1\0",
        )
        or fisher.get("selected_target_rows") != expected_selected_rows
        or fisher.get("valid_boundary_rows") != calibration_a_valid_rows
        or type(fisher.get("effective_nonzero_gradient_rows")) is not int
        or not 0
        < fisher["effective_nonzero_gradient_rows"]
        <= calibration_a_valid_rows
    ):
        raise ValueError("structured activation Fisher matrix is invalid")
    output_fisher_raw = _strict_mapping(
        training["output_fisher"],
        label="structured output Fisher",
        fields={
            "provenance",
            "calibration_split_sha256",
            "delta_scale",
            "standardized_coordinate_metric",
            "delta_scale_sha256",
            "metric_report",
        },
    )
    output_fisher = StructuredOutputFisherMetric(
        provenance=_restore_provenance(output_fisher_raw["provenance"]),
        calibration_split_sha256=output_fisher_raw[  # type: ignore[arg-type]
            "calibration_split_sha256"
        ],
        delta_scale=output_fisher_raw["delta_scale"],  # type: ignore[arg-type]
        standardized_coordinate_metric=output_fisher_raw[  # type: ignore[arg-type]
            "standardized_coordinate_metric"
        ],
    )
    if (
        output_fisher.provenance != provenance
        or output_fisher.calibration_split_sha256
        != scales.calibration_split_sha256
        or not torch.equal(output_fisher.delta_scale, scales.output)
        or output_fisher_raw["delta_scale_sha256"]
        != _tensor_sha256(
            output_fisher.delta_scale,
            domain=b"fisher_graph.structured_layer.output_scale.v1\0",
        )
    ):
        raise ValueError("structured output Fisher provenance is invalid")
    expected_metric, expected_metric_report = _normalized_fisher_metric(
        matrix,
        scales.output,
        eigenvalue_floor=protocol_fisher_floor,
    )
    if (
        not torch.allclose(
            output_fisher.standardized_coordinate_metric,
            expected_metric,
            rtol=1e-10,
            atol=1e-12,
        )
        or output_fisher_raw["metric_report"] != expected_metric_report
    ):
        raise ValueError("structured output Fisher metric does not recompute")

    thresholds = _strict_mapping(
        protocol["thresholds"],
        label="structured thresholds",
    )
    model = _strict_mapping(raw["model"], label="structured model")
    if format_version >= 4:
        assert format4_prompt_hashes is not None
        assert format4_family_hashes is not None
        prompt_family_metadata = _strict_mapping(
            protocol["prompt_families"],
            label="structured prompt family provenance",
        )
        prompt_family_pairs = _strict_mapping(
            prompt_family_metadata[
                "ordered_prompt_family_pairs_sha256"
            ],
            label="structured prompt-family pair hashes",
        )
        _validate_calibration_b_claim(
            protocol.get("calibration_b_claim"),
            protocol=protocol,
            prompt_hashes=format4_prompt_hashes["calibration_b"],
            family_hashes=format4_family_hashes["calibration_b"],
            prompt_family_pair_sha256=prompt_family_pairs[
                "calibration_b"
            ],  # type: ignore[arg-type]
            executors=executors,
            model=model,
            layer_id=provenance.layer_id,
        )
    if format_version >= 4:
        train_sequences = validated_streams["calibration_a"].get(
            "sequences"
        )
        if type(train_sequences) is not int or train_sequences <= 0:
            raise ValueError(
                "structured calibration-A stream sequence count is invalid"
            )
        _validate_calibration_a_fidelity(
            training.get("calibration_a_fidelity"),
            thresholds=thresholds,  # type: ignore[arg-type]
            expected_sequences=train_sequences,
            executors=executors,
        )
    elif "calibration_a_fidelity" in training:
        raise ValueError(
            "format-3 structured artifact contains format-4 A preflight"
        )
    strong_training_recipe = _preregistered_representative_protocol(
        training_recipe=training_recipe,
        thresholds=thresholds,
        seed=protocol.get("optimization_seed"),
        maximum_length=protocol.get("maximum_tokenized_length"),
        tokenization_batch_size=protocol.get(
            "tokenization_batch_size"
        ),
        fisher_floor=protocol_fisher_floor,
        delta_scale_floor=protocol_delta_scale_floor,
        relative_median_scale_floor=(
            protocol_relative_median_floor
        ),
        format_version=format_version,  # type: ignore[arg-type]
    )
    expected_selection_gates = _evaluate_gates(
        selection,
        thresholds=thresholds,  # type: ignore[arg-type]
    )
    selection_passed = expected_selection_gates["primary_passed"] is True
    expected_executor_fingerprints = {
        name: executors[name].execution_fingerprint()
        for name in _CANDIDATES
    }
    if (
        selection.get("evaluated") is not True
        or selection.get("gates") != expected_selection_gates
        or selection.get("passed") is not selection_passed
        or selection.get("locked_candidate")
        != (_PRIMARY if selection_passed else None)
        or selection.get("resource_gates_applied") is not False
        or selection.get("resource_diagnostics_only") is not True
        or (
            format_version >= 4
            and selection.get("executor_fingerprints")
            != expected_executor_fingerprints
        )
        or (
            format_version == 3
            and "executor_fingerprints" in selection
        )
    ):
        raise ValueError("structured calibration-B decision is invalid")
    source_manifest = _strict_mapping(
        protocol["source_accounting_manifest"],
        label="structured source accounting manifest",
    )
    _validate_accounting_section(
        selection,
        executors=executors,
        source_manifest=source_manifest,
        split_name="calibration_b",
    )
    if selection_passed:
        if (
            validation.get("evaluated") is not True
            or validation.get("reason")
            != "calibration_b_passed_locked_executors_evaluated"
        ):
            raise ValueError("passing calibration B did not open validation")
        expected_validation_gates = _evaluate_gates(
            validation,
            thresholds=thresholds,  # type: ignore[arg-type]
        )
        validation_passed = (
            expected_validation_gates["primary_passed"] is True
        )
        if (
            validation.get("gates") != expected_validation_gates
            or validation.get("passed") is not validation_passed
            or validation.get("locked_candidate")
            != (_PRIMARY if validation_passed else None)
            or validation.get("resource_gates_applied") is not False
            or validation.get("resource_diagnostics_only") is not True
            or (
                format_version >= 4
                and validation.get("executor_fingerprints")
                != expected_executor_fingerprints
            )
            or (
                format_version == 3
                and "executor_fingerprints" in validation
            )
        ):
            raise ValueError("structured validation decision is invalid")
        _validate_accounting_section(
            validation,
            executors=executors,
            source_manifest=source_manifest,
            split_name="validation",
        )
    else:
        validation_passed = False
        expected_unevaluated = _unevaluated_validation_payload(
            format_version=format_version,  # type: ignore[arg-type]
        )
        if validation != expected_unevaluated:
            raise ValueError("closed validation contains evaluation results")

    minima = _strict_mapping(
        protocol["data_minima"],
        label="structured data minima",
    )
    strong_minima = _strong_data_minima(
        minimum_calibration_a_prompts=int(
            minima["minimum_calibration_a_prompts"]
        ),
        minimum_heldout_prompts=int(
            minima["minimum_heldout_prompts_per_role"]
        ),
        minimum_fisher_rows=int(minima["minimum_effective_fisher_rows"]),
        minimum_train_supervised_tokens=int(
            minima["minimum_train_supervised_tokens"]
        ),
        minimum_heldout_supervised_tokens=int(
            minima["minimum_heldout_supervised_tokens_per_role"]
        ),
        minimum_length_buckets=int(
            minima[
                "minimum_populated_length_buckets_per_tokenized_role"
            ]
        ),
    )
    resolved_commit = model.get("resolved_commit")
    immutable_revision = (
        isinstance(resolved_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit) is not None
    )
    fidelity_passed = (
        selection_passed
        and validation_passed
        and strong_minima
        and immutable_revision
        and strong_training_recipe
        and corpus_audit_bound
    )
    expected_outcome = (
        "rejected_on_calibration_b"
        if not selection_passed
        else (
            "rejected_on_validation"
            if not validation_passed
            else (
                "single_layer_structured_fidelity_passed"
                if fidelity_passed
                else "relaxed_protocol_passed_without_scientific_promotion"
            )
        )
    )
    required_status = {
        "scope": "source_free_native_shape_single_gemma_layer_fidelity",
        "outcome": expected_outcome,
        "activation_fisher_computed": True,
        "all_eight_structured_targets_supervised": True,
        "calibration_a_executors_fitted": True,
        "calibration_b_evaluated": True,
        "calibration_b_passed": selection_passed,
        "validation_evaluated": selection_passed,
        "validation_passed": validation_passed,
        "test_evaluated": False,
        "strict_executor_reload_before_heldout": True,
        "strict_artifact_reload_required_and_run_before_return": True,
        "source_layer_calls_in_student_path": 0,
        "source_layer_removed_from_student_path": True,
        "source_independent_executor": True,
        "model_weights_changed": False,
        "model_weights_in_artifact": False,
        "strong_data_minima_enforced": strong_minima,
        "strong_training_recipe_enforced": strong_training_recipe,
        "corpus_audit_bound": corpus_audit_bound,
        "preregistered_fixed_schedule_enforced": (
            strong_training_recipe
        ),
        "immutable_model_revision_recorded": immutable_revision,
        "single_layer_structured_fidelity_passed": fidelity_passed,
        "general_method_viable": False,
        "model_level_promotion_authorized": False,
        "compression_attempted": False,
        "parameter_reduction_supported": False,
        "analytic_mac_reduction_supported": False,
        "resource_values_used_as_fidelity_gates": False,
        "attention_value_separation_observed": (
            expected_selection_gates[
                "attention_value_separation_observed"
            ]
        ),
        "decode_supported": False,
        "latency_or_kernel_speed_claim": False,
    }
    if status != required_status:
        raise ValueError("structured scientific status is invalid")

    report_path = source.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("structured JSON report is invalid")
    artifact = _strict_mapping(
        report.get("artifact"),
        label="structured report artifact",
    )
    tensor_file = artifact.get("tensor_file")
    if not isinstance(tensor_file, str) or not tensor_file:
        raise ValueError("structured report tensor file is invalid")
    expected_report = _build_report(
        payload,
        tensor_file=tensor_file,
        scientific_digest=digest,
    )
    canonical_expected_report = json.loads(
        json.dumps(
            expected_report,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if (
        report != canonical_expected_report
        or _report_sha256(report) != raw["report_sha256"]
    ):
        raise ValueError("structured JSON report is invalid")
    return {
        "model": copy.deepcopy(model),
        "protocol": copy.deepcopy(protocol),
        "executors": executors,
        "training": copy.deepcopy(training),
        "selection": copy.deepcopy(selection),
        "validation": copy.deepcopy(validation),
        "scientific_status": copy.deepcopy(status),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "tensor_file_sha256": _file_sha256(source),
        },
        "report": copy.deepcopy(report),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and gate a native-shaped source-free replacement for one "
            "Gemma 3 text-decoder layer."
        )
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        required=True,
    )
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tokenization-batch-size", type=int, default=1)
    parser.add_argument(
        "--fisher-floor",
        type=float,
        default=DEFAULT_FISHER_FLOOR,
    )
    parser.add_argument(
        "--delta-scale-floor",
        type=float,
        default=DEFAULT_RIDGE_SCALE_FLOOR,
    )
    parser.add_argument(
        "--relative-median-scale-floor",
        type=float,
        default=DEFAULT_RELATIVE_MEDIAN_SCALE_FLOOR,
    )
    parser.add_argument(
        "--local-warmup-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--train-positions-per-sequence",
        type=int,
        default=DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--structured-loss-scale",
        type=float,
        default=DEFAULT_STRUCTURED_LOSS_SCALE,
    )
    parser.add_argument(
        "--output-fisher-weight",
        type=float,
        default=DEFAULT_OUTPUT_FISHER_WEIGHT,
    )
    parser.add_argument(
        "--coordinate-loss-weight",
        type=float,
        default=DEFAULT_COORDINATE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--energy-loss-weight",
        type=float,
        default=DEFAULT_ENERGY_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--ground-truth-weight",
        type=float,
        default=DEFAULT_GROUND_TRUTH_WEIGHT,
    )
    parser.add_argument(
        "--teacher-kl-weight",
        type=float,
        default=DEFAULT_TEACHER_KL_WEIGHT,
    )
    parser.add_argument(
        "--selection-nll-atol",
        type=float,
        default=DEFAULT_NLL_ATOL,
    )
    parser.add_argument(
        "--selection-top1-min",
        type=float,
        default=DEFAULT_TOP1_MIN,
    )
    parser.add_argument(
        "--selection-teacher-kl-max",
        type=float,
        default=DEFAULT_TEACHER_KL_MAX,
    )
    parser.add_argument(
        "--selection-p90-abs-nll-max",
        type=float,
        default=DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    )
    parser.add_argument(
        "--selection-p10-top1-min",
        type=float,
        default=DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    )
    parser.add_argument(
        "--block-delta-nrmse-max",
        type=float,
        default=DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    )
    parser.add_argument(
        "--block-delta-cosine-min",
        type=float,
        default=DEFAULT_BLOCK_DELTA_COSINE_MIN,
    )
    parser.add_argument(
        "--branch-delta-nrmse-max",
        type=float,
        default=DEFAULT_BRANCH_DELTA_NRMSE_MAX,
    )
    parser.add_argument(
        "--branch-delta-cosine-min",
        type=float,
        default=DEFAULT_BRANCH_DELTA_COSINE_MIN,
    )
    parser.add_argument(
        "--native-parity-tolerance",
        type=float,
        default=DEFAULT_NATIVE_PARITY_TOLERANCE,
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
        "--minimum-fisher-rows",
        type=int,
        default=DEFAULT_MINIMUM_FISHER_ROWS,
    )
    parser.add_argument(
        "--minimum-train-supervised-tokens",
        type=int,
        default=DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_OPTIMIZATION_SEED,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-b-ledger-dir", type=Path)
    parser.add_argument("--operator-bootstrap", action="store_true")
    parser.add_argument(
        "--operator-bootstrap-rows",
        type=int,
        default=DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS,
    )
    parser.add_argument(
        "--operator-bootstrap-ridge-relative",
        type=float,
        default=DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE,
    )
    parser.add_argument(
        "--operator-bootstrap-rank-rtol",
        type=float,
        default=DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL,
    )
    parser.add_argument(
        "--operator-bootstrap-max-condition",
        type=float,
        default=DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION,
    )
    parser.add_argument(
        "--operator-bootstrap-maximum-nullity",
        type=int,
        default=DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY,
    )
    parser.add_argument(
        "--stop-after-calibration-a",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_structured_single_layer_experiment(
        prompt_splits_path=args.prompt_splits,
        family_manifest_path=args.family_manifest,
        corpus_audit_path=args.corpus_audit,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        layer_index=args.layer_index,
        max_length=args.max_length,
        tokenization_batch_size=args.tokenization_batch_size,
        fisher_floor=args.fisher_floor,
        delta_scale_floor=args.delta_scale_floor,
        relative_median_scale_floor=args.relative_median_scale_floor,
        local_warmup_steps=args.local_warmup_steps,
        train_steps=args.train_steps,
        train_positions_per_sequence=args.train_positions_per_sequence,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        structured_loss_scale=args.structured_loss_scale,
        output_fisher_weight=args.output_fisher_weight,
        coordinate_loss_weight=args.coordinate_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        ground_truth_weight=args.ground_truth_weight,
        teacher_kl_weight=args.teacher_kl_weight,
        selection_nll_atol=args.selection_nll_atol,
        selection_top1_min=args.selection_top1_min,
        selection_teacher_kl_max=args.selection_teacher_kl_max,
        selection_p90_abs_nll_max=args.selection_p90_abs_nll_max,
        selection_p10_top1_min=args.selection_p10_top1_min,
        block_delta_nrmse_max=args.block_delta_nrmse_max,
        block_delta_cosine_min=args.block_delta_cosine_min,
        branch_delta_nrmse_max=args.branch_delta_nrmse_max,
        branch_delta_cosine_min=args.branch_delta_cosine_min,
        native_parity_tolerance=args.native_parity_tolerance,
        minimum_calibration_a_prompts=args.minimum_calibration_a_prompts,
        minimum_heldout_prompts=args.minimum_heldout_prompts,
        minimum_fisher_rows=args.minimum_fisher_rows,
        minimum_train_supervised_tokens=(
            args.minimum_train_supervised_tokens
        ),
        minimum_heldout_supervised_tokens=(
            args.minimum_heldout_supervised_tokens
        ),
        minimum_length_buckets=args.minimum_length_buckets,
        seed=args.seed,
        device_name=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        output=args.output,
        calibration_b_ledger_dir=args.calibration_b_ledger_dir,
        operator_bootstrap=args.operator_bootstrap,
        operator_bootstrap_rows=args.operator_bootstrap_rows,
        operator_bootstrap_ridge_relative=(
            args.operator_bootstrap_ridge_relative
        ),
        operator_bootstrap_rank_rtol=args.operator_bootstrap_rank_rtol,
        operator_bootstrap_max_condition=(
            args.operator_bootstrap_max_condition
        ),
        operator_bootstrap_maximum_nullity=(
            args.operator_bootstrap_maximum_nullity
        ),
        stop_after_calibration_a=args.stop_after_calibration_a,
        progress=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "StructuredTrainingBatch",
    "collect_structured_training_batches",
    "compute_structured_activation_fisher",
    "default_gemma3_structured_single_layer_output",
    "evaluate_structured_candidates",
    "fit_structured_executor",
    "load_gemma3_structured_single_layer_artifact",
    "main",
    "make_structured_executor",
    "run_gemma3_structured_single_layer_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
