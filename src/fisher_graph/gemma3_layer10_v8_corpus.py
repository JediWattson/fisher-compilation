"""Prepare the unused structured-strong-v8 Calibration-A rows for layer 10.

This module is intentionally a data-boundary tool, not a model experiment.  It
authenticates the exact frozen v8 source files before decoding them, projects
only Calibration-A into three family-disjoint runtime roles, and writes a
prompt-free receipt.  Calibration-B, validation, and test are never copied into
the derived role files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpusArtifact,
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


__all__ = [
    "DEFAULT_V8_AUDIT_PATH",
    "DEFAULT_V8_FAMILY_PATH",
    "DEFAULT_V8_GENERATOR_PATH",
    "DEFAULT_V8_PROMPT_PATH",
    "DEFAULT_CORPUS_OUTPUT",
    "DEFAULT_FIT_OUTPUT",
    "DEFAULT_GUARD_OUTPUT",
    "DEFAULT_RECEIPT_OUTPUT",
    "DEFAULT_SELECTION_OUTPUT",
    "GEMMA3_LAYER10_V8_CORPUS_ID",
    "Gemma3Layer10V8CorpusIntegrityError",
    "build_parser",
    "load_gemma3_layer10_v8_corpus_receipt",
    "main",
    "prepare_gemma3_layer10_v8_corpus",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_V8_AUDIT_PATH = _LOCAL_ROOT / "structured-strong-v8-corpus-audit.json"
DEFAULT_V8_FAMILY_PATH = _LOCAL_ROOT / "structured-strong-v8-families.json"
DEFAULT_V8_PROMPT_PATH = _LOCAL_ROOT / "structured-strong-v8-prompts.json"
DEFAULT_V8_GENERATOR_PATH = _LOCAL_ROOT / "generate_structured_strong_v8.py"

DEFAULT_FIT_OUTPUT = _LOCAL_ROOT / "layer10-v8-shape-flow-v1.fit.json"
DEFAULT_SELECTION_OUTPUT = (
    _LOCAL_ROOT / "layer10-v8-shape-flow-v1.selection.json"
)
DEFAULT_GUARD_OUTPUT = _LOCAL_ROOT / "layer10-v8-shape-flow-v1.guard.json"
DEFAULT_CORPUS_OUTPUT = _LOCAL_ROOT / "layer10-v8-shape-flow-v1.corpus.json"
DEFAULT_RECEIPT_OUTPUT = _LOCAL_ROOT / "layer10-v8-shape-flow-v1.receipt.json"

GEMMA3_LAYER10_V8_CORPUS_ID = "gemma3-layer10-shape-flow-v8-v1"

_V8_AUDIT_FILE_SHA256 = (
    "f34735a53b0e9eb6b0471ec9c73613ef64c39ca7aeb7006f80845eb8bef70988"
)
_V8_FAMILY_FILE_SHA256 = (
    "01654b2ded4ca0cf713c2d334f41faacdd336d13776ecdb338c392d50dbdd703"
)
_V8_PROMPT_FILE_SHA256 = (
    "81a5431cc8b84f66686bdf46e7861dc431bcf2d7e9bc39781d76b00a5f60b66a"
)
_V8_GENERATOR_FILE_SHA256 = (
    "ed28b27528fc122a8103bdc6d9ecc101292914ddc4ead38993b0515e9b637c64"
)
_V8_FIT_INDEX_SHA256 = (
    "aa7aaa35557b5ff402280d7811fe735f558c999bdbd52659aff152c68e872593"
)
_V8_GUARD_INDEX_SHA256 = (
    "f78d4b48e69cfc14b1fd0e9f99de5ed25ac3baf5f05fc235bffa35417f6b50a0"
)

_ROLE_IDS = (
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
)
_HELDOUT_ROLE_IDS = ("calibration_b", "validation", "test")
_SOURCE_TOP_LEVEL_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
}
_EXPECTED_COUNTS = {
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
    "compact": 8,
    "long": 224,
    "medium": 16,
    "micro": 8,
}
_ZERO_AUDIT_FIELDS = (
    "cross_role_family_overlap_count",
    "prior_local_exact_prompt_overlap_count",
    "prior_raw_prompt_overlap_count",
    "prior_normalized_prompt_overlap_count",
    "prior_domain_slug_overlap_count",
    "prior_template_marker_overlap_count",
    "prior_template_signature_overlap_count",
    "prior_5_6_7_8_word_ngram_overlap_count",
    "prior_file_embedded_prompt_overlap_count",
)
_GUARD_FAMILY_RANK_DOMAIN = (
    b"fisher-graph:gemma3-layer10-v8-a-guard-family-rank:v1\0"
)
_RECEIPT_DOMAIN = b"fisher-graph:gemma3-layer10-v8-corpus-receipt:v1\0"


class Gemma3Layer10V8CorpusIntegrityError(RuntimeError):
    """The exact v8 sources or their declared partition failed validation."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256_bytes(domain + _canonical_json_bytes(value))


def _raw_text_sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _indices_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(tuple(indices), separators=(",", ":")).encode("ascii")
    return _sha256_bytes(encoded)


def _read_exact_json(
    path: Path | str,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = source.read_bytes()
    observed = _sha256_bytes(encoded)
    if observed != expected_sha256:
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} exact file SHA-256 mismatch"
        )
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(raw, dict):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} must contain one JSON object"
        )
    return raw


def _validate_generator(path: Path | str) -> None:
    source = Path(path)
    if not source.is_file():
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 generator must be a regular file"
        )
    if _sha256_bytes(source.read_bytes()) != _V8_GENERATOR_FILE_SHA256:
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 generator exact file SHA-256 mismatch"
        )


def _require_string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} must be a nonempty-string JSON array"
        )
    return tuple(value)


def _require_index_list(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"{label} must be a nonnegative-integer JSON array"
        )
    return tuple(value)


def _validate_source_headers(
    prompts: Mapping[str, object],
    families: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    if (
        set(prompts) != _SOURCE_TOP_LEVEL_FIELDS
        or prompts.get("schema") != "fisher_graph.gemma3_prompt_splits"
        or prompts.get("format_version") != 1
        or prompts.get("scientific_status")
        != "full_width_single_layer_fresh_a_b_validation_test_hash_only"
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 prompt source header is invalid"
        )
    if (
        set(families) != _SOURCE_TOP_LEVEL_FIELDS
        or families.get("schema")
        != "fisher_graph.gemma3_prompt_family_manifest"
        or families.get("format_version") != 1
        or families.get("scientific_status")
        != "full_width_single_layer_family_disjoint_roles"
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 family source header is invalid"
        )
    if (
        audit.get("schema")
        != "fisher_graph.structured_strong_corpus_audit"
        or audit.get("format_version") != 4
        or audit.get("corpus_id") != "structured-strong-v8"
        or audit.get("purpose")
        != "mode_bundling_fit_guard_and_frozen_evaluation"
        or audit.get("counts") != _EXPECTED_COUNTS
        or audit.get("families_per_role") != _EXPECTED_FAMILY_COUNTS
        or audit.get("family_file_sha256") != _V8_FAMILY_FILE_SHA256
        or audit.get("prompt_file_sha256") != _V8_PROMPT_FILE_SHA256
        or audit.get("generator_sha256") != _V8_GENERATOR_FILE_SHA256
        or audit.get("generator_bound_by_sha256") is not True
        or audit.get("calibration_a_policy")
        != "family_disjoint_fit_guard_development_only"
        or audit.get("calibration_a_fit_may_train_candidate") is not True
        or audit.get("calibration_a_guard_may_change_candidate") is not False
        or audit.get("calibration_b_policy")
        != "one_shot_frozen_candidate_selection"
        or audit.get("calibration_b_reuse_allowed") is not False
        or audit.get("heldout_splits_evaluated") is not False
        or audit.get("heldout_splits_tokenized") is not False
        or audit.get("heldout_splits_unevaluated") is not True
        or audit.get("heldout_splits_untokenized") is not True
        or audit.get("calibration_b_model_evaluated") is not False
        or audit.get("validation_model_evaluated") is not False
        or audit.get("test_model_evaluated") is not False
        or audit.get("tokenizer_or_model_accessed") is not False
        or audit.get("corpus_frozen_before_model_load") is not True
        or audit.get("unique_prompt_count") != 800
        or audit.get("unique_normalized_prompt_count") != 800
        or any(audit.get(field) != 0 for field in _ZERO_AUDIT_FIELDS)
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 corpus audit policy or source binding is invalid"
        )


def _materialize_a_partition(
    *,
    name: str,
    raw: object,
    prompts: tuple[str, ...],
    families: tuple[str, ...],
    expected_index_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"v8 calibration-A {name} partition is missing"
        )
    indices = _require_index_list(
        raw.get("prompt_indices"),
        label=f"v8 calibration-A {name} prompt indices",
    )
    declared_families = _require_string_list(
        raw.get("family_ids"),
        label=f"v8 calibration-A {name} family ids",
    )
    if (
        len(indices) != 256
        or len(set(indices)) != 256
        or tuple(sorted(indices)) != indices
        or _indices_sha256(indices) != expected_index_sha256
        or raw.get("prompt_index_sha256") != expected_index_sha256
        or len(declared_families) != 8
        or len(set(declared_families)) != 8
        or raw.get("prompt_count") != 256
        or raw.get("family_count") != 8
        or raw.get("band_counts") != _EXPECTED_PARTITION_BANDS
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"v8 calibration-A {name} partition metadata is invalid"
        )
    try:
        selected_prompts = tuple(prompts[index] for index in indices)
        selected_families = tuple(families[index] for index in indices)
    except IndexError as error:
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"v8 calibration-A {name} index is out of range"
        ) from error
    declared_set = set(declared_families)
    if set(selected_families) != declared_set or any(
        selected_families.count(family_id) != 32
        for family_id in declared_families
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            f"v8 calibration-A {name} family binding is invalid"
        )
    return selected_prompts, selected_families, indices, declared_families


def _rank_guard_families(family_ids: Sequence[str]) -> tuple[str, ...]:
    """Return an order-independent, domain-separated family ranking."""

    values = tuple(family_ids)
    if len(values) != 8 or len(set(values)) != 8:
        raise ValueError("guard family ranking requires eight unique ids")
    return tuple(
        sorted(
            values,
            key=lambda family_id: (
                _sha256_bytes(
                    _GUARD_FAMILY_RANK_DOMAIN + family_id.encode("utf-8")
                ),
                family_id,
            ),
        )
    )


def _derive_roles(
    prompts_payload: Mapping[str, object],
    families_payload: Mapping[str, object],
    audit_payload: Mapping[str, object],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Materialize only Calibration-A and derive the three runtime roles."""

    prompts = _require_string_list(
        prompts_payload.get("calibration_a"),
        label="v8 calibration-A prompts",
    )
    families = _require_string_list(
        families_payload.get("calibration_a"),
        label="v8 calibration-A families",
    )
    if (
        len(prompts) != 512
        or len(families) != 512
        or len(set(prompts)) != 512
        or len(set(families)) != 16
        or any(prompt != prompt.strip() for prompt in prompts)
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 calibration-A columns are invalid"
        )
    raw_partitions = audit_payload.get("calibration_a_family_partitions")
    if (
        not isinstance(raw_partitions, Mapping)
        or raw_partitions.get("family_disjoint") is not True
        or raw_partitions.get("union_covers_calibration_a") is not True
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 calibration-A partition declaration is invalid"
        )
    fit_prompts, fit_families, fit_indices, fit_unique = (
        _materialize_a_partition(
            name="fit",
            raw=raw_partitions.get("fit"),
            prompts=prompts,
            families=families,
            expected_index_sha256=_V8_FIT_INDEX_SHA256,
        )
    )
    guard_prompts, guard_families, guard_indices, guard_unique = (
        _materialize_a_partition(
            name="guard",
            raw=raw_partitions.get("guard"),
            prompts=prompts,
            families=families,
            expected_index_sha256=_V8_GUARD_INDEX_SHA256,
        )
    )
    if (
        set(fit_indices) & set(guard_indices)
        or set(fit_indices) | set(guard_indices) != set(range(512))
        or set(fit_unique) & set(guard_unique)
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 calibration-A fit and guard are not disjoint and complete"
        )

    ranked = _rank_guard_families(guard_unique)
    selection_family_set = set(ranked[:4])
    guard_family_set = set(ranked[4:])
    selection_pairs = tuple(
        (prompt, family_id)
        for prompt, family_id in zip(
            guard_prompts,
            guard_families,
            strict=True,
        )
        if family_id in selection_family_set
    )
    guard_pairs = tuple(
        (prompt, family_id)
        for prompt, family_id in zip(
            guard_prompts,
            guard_families,
            strict=True,
        )
        if family_id in guard_family_set
    )
    if (
        len(selection_pairs) != 128
        or len(guard_pairs) != 128
        or len({family for _, family in selection_pairs}) != 4
        or len({family for _, family in guard_pairs}) != 4
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "v8 calibration-A guard did not split into balanced 128-row roles"
        )

    roles = {
        "calibration_a_fit": (fit_prompts, fit_families),
        "calibration_a_selection": (
            tuple(prompt for prompt, _ in selection_pairs),
            tuple(family for _, family in selection_pairs),
        ),
        "calibration_a_guard": (
            tuple(prompt for prompt, _ in guard_pairs),
            tuple(family for _, family in guard_pairs),
        ),
    }
    prompt_hash_sets = tuple(
        {_raw_text_sha256(prompt) for prompt in role_prompts}
        for role_prompts, _ in roles.values()
    )
    family_sets = tuple(
        set(role_families) for _, role_families in roles.values()
    )
    if any(
        left & right
        for index, left in enumerate(prompt_hash_sets)
        for right in prompt_hash_sets[index + 1 :]
    ) or any(
        left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "derived layer-10 roles are not prompt- and family-disjoint"
        )
    if sum(len(values[0]) for values in roles.values()) != 512:
        raise Gemma3Layer10V8CorpusIntegrityError(
            "derived layer-10 roles do not cover Calibration-A exactly"
        )
    return roles


def _tokenizer_contract() -> dict[str, object]:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    metadata = protocol.metadata()
    tokenizer = metadata.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "frozen tokenizer contract is unavailable"
        )
    return dict(tokenizer)


def _validate_output_paths(paths: Sequence[Path]) -> None:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(paths):
        raise ValueError("layer-10 corpus outputs must use distinct paths")
    existing = tuple(path for path in paths if path.exists())
    if existing:
        raise FileExistsError("refusing to overwrite layer-10 corpus outputs")


def _role_receipts(
    artifact: Gemma3L3L4ProgressiveACorpusArtifact,
) -> dict[str, object]:
    return {
        role: {
            "role_id": role,
            "example_count": artifact.role_view(role).example_count,
            "family_count": len(artifact.role_view(role).family_ids),
            "manifest_sha256": artifact.role_view(role).manifest_sha256,
            "role_input_file_sha256": (
                artifact.role_view(role).role_input_file_sha256
            ),
        }
        for role in _ROLE_IDS
    }


def prepare_gemma3_layer10_v8_corpus(
    *,
    prompt_splits_path: Path | str = DEFAULT_V8_PROMPT_PATH,
    family_manifest_path: Path | str = DEFAULT_V8_FAMILY_PATH,
    corpus_audit_path: Path | str = DEFAULT_V8_AUDIT_PATH,
    generator_path: Path | str = DEFAULT_V8_GENERATOR_PATH,
    fit_output: Path | str = DEFAULT_FIT_OUTPUT,
    selection_output: Path | str = DEFAULT_SELECTION_OUTPUT,
    guard_output: Path | str = DEFAULT_GUARD_OUTPUT,
    corpus_output: Path | str = DEFAULT_CORPUS_OUTPUT,
    receipt_output: Path | str = DEFAULT_RECEIPT_OUTPUT,
) -> dict[str, object]:
    """Write the exact 256/128/128 v8-derived layer-10 corpus once."""

    fit_path = Path(fit_output)
    selection_path = Path(selection_output)
    guard_path = Path(guard_output)
    corpus_path = Path(corpus_output)
    receipt_path = Path(receipt_output)
    outputs = (
        fit_path,
        selection_path,
        guard_path,
        corpus_path,
        receipt_path,
    )
    _validate_output_paths(outputs)

    prompts = _read_exact_json(
        prompt_splits_path,
        expected_sha256=_V8_PROMPT_FILE_SHA256,
        label="v8 prompt source",
    )
    families = _read_exact_json(
        family_manifest_path,
        expected_sha256=_V8_FAMILY_FILE_SHA256,
        label="v8 family source",
    )
    audit = _read_exact_json(
        corpus_audit_path,
        expected_sha256=_V8_AUDIT_FILE_SHA256,
        label="v8 corpus audit",
    )
    _validate_generator(generator_path)
    _validate_source_headers(prompts, families, audit)
    roles = _derive_roles(prompts, families, audit)

    role_paths = {
        "calibration_a_fit": fit_path,
        "calibration_a_selection": selection_path,
        "calibration_a_guard": guard_path,
    }
    for role in _ROLE_IDS:
        role_prompts, role_families = roles[role]
        write_gemma3_l3_l4_progressive_a_role_input(
            role_paths[role],
            corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
            profile="full",
            role=role,  # type: ignore[arg-type]
            prompts=role_prompts,
            family_ids=role_families,
        )
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
        profile="full",
        tokenizer_contract=_tokenizer_contract(),
        role_input_paths=role_paths,  # type: ignore[arg-type]
    )
    corpus_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_path,
        artifact,
    )

    payload: dict[str, object] = {
        "schema": "fisher_graph.gemma3_layer10_v8_corpus_receipt",
        "format_version": 1,
        "corpus_id": GEMMA3_LAYER10_V8_CORPUS_ID,
        "profile": "full",
        "source": {
            "source_corpus_id": "structured-strong-v8",
            "audit_file_sha256": _V8_AUDIT_FILE_SHA256,
            "family_file_sha256": _V8_FAMILY_FILE_SHA256,
            "prompt_file_sha256": _V8_PROMPT_FILE_SHA256,
            "generator_file_sha256": _V8_GENERATOR_FILE_SHA256,
            "calibration_a_fit_index_sha256": _V8_FIT_INDEX_SHA256,
            "calibration_a_guard_index_sha256": _V8_GUARD_INDEX_SHA256,
            "source_model_or_tokenizer_accessed": False,
        },
        "derivation": {
            "source_role_id": "calibration_a",
            "guard_split_policy_id": (
                "domain_sha256_rank_first4_selection_last4_guard"
            ),
            "guard_split_domain_sha256": _sha256_bytes(
                _GUARD_FAMILY_RANK_DOMAIN
            ),
            "source_example_count": 512,
            "source_family_count": 16,
            "roles_cover_source_exactly": True,
            "role_family_overlap_count": 0,
            "role_prompt_identity_overlap_count": 0,
        },
        "roles": _role_receipts(artifact),
        "corpus": {
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": corpus_file_sha256,
            "tokenizer_contract_sha256": (
                artifact.tokenizer_contract_sha256
            ),
        },
        "heldout": {
            "role_ids": _HELDOUT_ROLE_IDS,
            "roles_materialized": False,
            "roles_exported": False,
            "roles_tokenized": False,
            "roles_model_evaluated": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_family_ids": False,
            "contains_token_ids": False,
            "contains_model_outputs": False,
            "source_safe": True,
        },
    }
    payload["receipt_sha256"] = _domain_sha256(_RECEIPT_DOMAIN, payload)
    encoded = _canonical_json_bytes(payload)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(encoded)
    return payload


def load_gemma3_layer10_v8_corpus_receipt(
    path: Path | str = DEFAULT_RECEIPT_OUTPUT,
) -> dict[str, object]:
    """Strict-load one source-safe v8 derivation receipt."""

    source = Path(path)
    if not source.is_file():
        raise Gemma3Layer10V8CorpusIntegrityError(
            "layer-10 v8 receipt must be a regular file"
        )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3Layer10V8CorpusIntegrityError(
            "layer-10 v8 receipt is not strict UTF-8 JSON"
        ) from error
    if not isinstance(raw, dict):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "layer-10 v8 receipt must contain one JSON object"
        )
    supplied = raw.get("receipt_sha256")
    payload = {key: value for key, value in raw.items() if key != "receipt_sha256"}
    expected_source = {
        "source_corpus_id": "structured-strong-v8",
        "audit_file_sha256": _V8_AUDIT_FILE_SHA256,
        "family_file_sha256": _V8_FAMILY_FILE_SHA256,
        "prompt_file_sha256": _V8_PROMPT_FILE_SHA256,
        "generator_file_sha256": _V8_GENERATOR_FILE_SHA256,
        "calibration_a_fit_index_sha256": _V8_FIT_INDEX_SHA256,
        "calibration_a_guard_index_sha256": _V8_GUARD_INDEX_SHA256,
        "source_model_or_tokenizer_accessed": False,
    }
    expected_derivation = {
        "source_role_id": "calibration_a",
        "guard_split_policy_id": (
            "domain_sha256_rank_first4_selection_last4_guard"
        ),
        "guard_split_domain_sha256": _sha256_bytes(
            _GUARD_FAMILY_RANK_DOMAIN
        ),
        "source_example_count": 512,
        "source_family_count": 16,
        "roles_cover_source_exactly": True,
        "role_family_overlap_count": 0,
        "role_prompt_identity_overlap_count": 0,
    }
    expected_heldout = {
        "role_ids": list(_HELDOUT_ROLE_IDS),
        "roles_materialized": False,
        "roles_exported": False,
        "roles_tokenized": False,
        "roles_model_evaluated": False,
    }
    expected_safety = {
        "contains_prompt_text": False,
        "contains_family_ids": False,
        "contains_token_ids": False,
        "contains_model_outputs": False,
        "source_safe": True,
    }
    roles = raw.get("roles")
    corpus = raw.get("corpus")
    role_counts = {
        "calibration_a_fit": (256, 8),
        "calibration_a_selection": (128, 4),
        "calibration_a_guard": (128, 4),
    }
    roles_valid = isinstance(roles, dict) and set(roles) == set(_ROLE_IDS)
    if roles_valid:
        for role, (examples, families) in role_counts.items():
            value = roles.get(role)
            roles_valid = (
                isinstance(value, dict)
                and set(value)
                == {
                    "example_count",
                    "family_count",
                    "manifest_sha256",
                    "role_id",
                    "role_input_file_sha256",
                }
                and value.get("role_id") == role
                and value.get("example_count") == examples
                and value.get("family_count") == families
                and all(
                    isinstance(value.get(field), str)
                    and len(value[field]) == 64
                    and set(value[field]) <= set("0123456789abcdef")
                    for field in (
                        "manifest_sha256",
                        "role_input_file_sha256",
                    )
                )
            )
            if not roles_valid:
                break
    corpus_valid = (
        isinstance(corpus, dict)
        and set(corpus)
        == {
            "artifact_sha256",
            "artifact_file_sha256",
            "tokenizer_contract_sha256",
        }
        and all(
            isinstance(corpus.get(field), str)
            and len(corpus[field]) == 64
            and set(corpus[field]) <= set("0123456789abcdef")
            for field in corpus
        )
    )
    if (
        set(raw)
        != {
            "schema",
            "format_version",
            "corpus_id",
            "profile",
            "source",
            "derivation",
            "roles",
            "corpus",
            "heldout",
            "safety",
            "receipt_sha256",
        }
        or raw.get("schema")
        != "fisher_graph.gemma3_layer10_v8_corpus_receipt"
        or raw.get("format_version") != 1
        or raw.get("corpus_id") != GEMMA3_LAYER10_V8_CORPUS_ID
        or raw.get("profile") != "full"
        or raw.get("source") != expected_source
        or raw.get("derivation") != expected_derivation
        or raw.get("heldout") != expected_heldout
        or raw.get("safety") != expected_safety
        or not roles_valid
        or not corpus_valid
        or supplied != _domain_sha256(_RECEIPT_DOMAIN, payload)
    ):
        raise Gemma3Layer10V8CorpusIntegrityError(
            "layer-10 v8 receipt integrity check failed"
        )
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="prepare the exact unused v8 Calibration-A layer-10 corpus"
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_V8_PROMPT_PATH,
    )
    parser.add_argument(
        "--family-manifest",
        type=Path,
        default=DEFAULT_V8_FAMILY_PATH,
    )
    parser.add_argument("--corpus-audit", type=Path, default=DEFAULT_V8_AUDIT_PATH)
    parser.add_argument("--generator", type=Path, default=DEFAULT_V8_GENERATOR_PATH)
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT
    )
    parser.add_argument("--guard-output", type=Path, default=DEFAULT_GUARD_OUTPUT)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=DEFAULT_RECEIPT_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = prepare_gemma3_layer10_v8_corpus(
        prompt_splits_path=arguments.prompt_splits,
        family_manifest_path=arguments.family_manifest,
        corpus_audit_path=arguments.corpus_audit,
        generator_path=arguments.generator,
        fit_output=arguments.fit_output,
        selection_output=arguments.selection_output,
        guard_output=arguments.guard_output,
        corpus_output=arguments.corpus_output,
        receipt_output=arguments.receipt_output,
    )
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
