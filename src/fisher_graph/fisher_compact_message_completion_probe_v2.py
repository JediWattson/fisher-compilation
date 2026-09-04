"""Fail-closed P0 runner for the compact-message completion protocol v2.

The currently frozen D provenance is not independent under
``strict_metadata_only_v1``.  Consequently this module implements only the
reached P0 prefix.  It deliberately has no model, tokenizer, split, capture,
or guard dependency.

The one raw-data operation in P0 is retirement of the already exposed reserve.
The semantic-canonicalization artifact is durably frozen before that reserve is
decoded.  Raw reserve content is never returned, logged, or written to an
artifact.  A host-local SQLite registry records only hashes and commitments;
it is bookkeeping and is not private-evaluator or guard authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import sqlite3
import stat
import tempfile
from typing import Final
import unicodedata

if __package__:
    from .gemma3_chart_native_neighbor_fresh_panels import (
        DEFAULT_RECEIPT,
        DEFAULT_ROLE_CORPORA,
        ORDERED_ROLES,
        PINNED_RECEIPT_FILE_SHA256,
        PINNED_RECEIPT_SHA256,
        PINNED_ROLE_FILE_SHA256S,
        load_receipt,
    )
else:
    # A top-level stdlib shim executes this file with runpy so package
    # ``__init__`` (and its eager tensor imports) never runs.  Load only the
    # standard-library, import-inert sibling receipt module in that mode.
    _receipt_namespace = runpy.run_path(
        str(
            Path(__file__).with_name(
                "gemma3_chart_native_neighbor_fresh_panels.py"
            )
        ),
        run_name="_fisher_compact_message_completion_fresh_receipt",
    )
    DEFAULT_RECEIPT = _receipt_namespace["DEFAULT_RECEIPT"]
    DEFAULT_ROLE_CORPORA = _receipt_namespace["DEFAULT_ROLE_CORPORA"]
    ORDERED_ROLES = _receipt_namespace["ORDERED_ROLES"]
    PINNED_RECEIPT_FILE_SHA256 = _receipt_namespace[
        "PINNED_RECEIPT_FILE_SHA256"
    ]
    PINNED_RECEIPT_SHA256 = _receipt_namespace["PINNED_RECEIPT_SHA256"]
    PINNED_ROLE_FILE_SHA256S = _receipt_namespace[
        "PINNED_ROLE_FILE_SHA256S"
    ]
    load_receipt = _receipt_namespace["load_receipt"]
    del _receipt_namespace


PROTOCOL_ID: Final = "fisher_compact_message_completion_probe_v2"
CAMPAIGN_ID: Final = "fisher-compact-message-completion-v2-current-abcd-p0"
FORMAT_VERSION: Final = 2
D_ACCESS_POLICY: Final = "strict_metadata_only_v1"
CANONICALIZATION_VERSION: Final = (
    "fisher-compact-message-completion-semantic-canonicalization-v1"
)
LEGACY_REGISTRY_SCHEMA_VERSION: Final = 1
REGISTRY_SCHEMA_VERSION: Final = 2
LEGACY_RETIREMENT_IMPLEMENTATION_SHA256: Final = (
    "d06a73933cc405e26f790673979501228e77f17102117aab0389b0fa2a773647"
)
EXHAUSTED_REAL_RESERVE_FILE_SHA256: Final = (
    "687824c48689a7a54698e3d1aa6d6d685e1bdfcddc703675b03ee5b68d429004"
)
LEGACY_CONTRACT_INVALIDATION_REASON: Final = (
    "legacy_v1_registry_commit_preceded_reserve_receipt_publication"
)
CURRENT_TERMINAL_STATE: Final = (
    "development_complete_independent_data_unavailable"
)
TERMINAL_REASON: Final = (
    "No eligible independent tune/development-confirm corpus"
)

DEFAULT_OUTPUT_DIR: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "fisher-compact-message-completion-probe-v2"
)
DEFAULT_REGISTRY_PATH: Final = Path(
    ".local-runs/fisher-compact-message-completion-"
    "local-development-registry-v1.sqlite3"
)

P0_ARTIFACT_ORDER: Final = (
    "semantic_canonicalization_spec.json",
    "reserve_retired_exposed.json",
    "local_development_registry_status.json",
    "corpus_role_classification.json",
    "preflight_decision.json",
    "commands.md",
    "terminal_state.json",
    "final_report.md",
    "final_experiment_manifest.json",
)
_FINAL_MANIFEST_NAME: Final = P0_ARTIFACT_ORDER[-1]
_INVENTORIED_ARTIFACTS: Final = P0_ARTIFACT_ORDER[:-1]

_CONTENT_HASH_PAYLOAD: Final = (
    "RFC8785(canonical semantic model-input object)"
)
_CONTENT_SET_DOMAIN: Final = (
    b"fisher-compact-message-completion/content-set/v1"
)
_CONTENT_SET_PAYLOAD: Final = (
    "ASCII(domain) || 0x00 || uint64_be(n) || sorted raw 32-byte digests"
)
_ALLOWED_ROLES: Final = (
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
)
_NONSEMANTIC_WRAPPER_FIELDS: Final = (
    "local_row_id",
    "source_filename",
    "capture_timestamp",
    "previously_computed_digest",
    "split_label",
    "ledger_state",
    "storage_path",
)
_COMMITMENT_DOMAINS: Final = {
    "analysis_unit": b"fisher-graph:local-development:analysis-unit:v1\0",
    "template_lineage": b"fisher-graph:local-development:template-lineage:v1\0",
    "template_id": b"fisher-graph:local-development:template-id:v1\0",
    "semantic_instance": b"fisher-graph:local-development:semantic-instance:v1\0",
    "family": b"fisher-graph:local-development:family:v1\0",
}
_HISTORICAL_USAGE_ROLES: Final = {
    "A_chart_transport_fit": "historical_fit",
    "B_neighbor_discovery": "historical_tune",
    "C_neighbor_confirmation": "historical_development_confirm",
    "D_response_validation": "preoutcome_executable_access",
    "reserve_unopened": "retired_exposed",
}
_EXPOSURE_REASON: Final = (
    "The reserve raw prompt JSON was previously printed during metadata "
    "inspection, so procedural blindness is lost."
)
_TEMPLATE_LINEAGE_COMMITMENT_BASIS: Final = (
    "explicit template_lineage_id when present; otherwise the frozen "
    "template_id supplied by this panel schema"
)


class CompactMessageCompletionProtocolError(RuntimeError):
    """The P0 protocol could not complete safely."""


class CompactMessageCompletionIntegrityError(
    CompactMessageCompletionProtocolError
):
    """An input, artifact, or registry identity failed validation."""


class CompactMessageCompletionFinalizedError(
    CompactMessageCompletionProtocolError
):
    """A finalized or partially materialized campaign cannot be mutated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_sha256_digest(value: object, *, label: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must be a lowercase SHA-256"
        )
    return bytes.fromhex(value)


def content_set_sha256(content_sha256s: Sequence[str]) -> str:
    """Return the exact F19 aggregate commitment for unique content hashes."""

    digests = [
        _raw_sha256_digest(value, label=f"content_sha256s[{index}]")
        for index, value in enumerate(content_sha256s)
    ]
    if len(set(digests)) != len(digests):
        raise CompactMessageCompletionIntegrityError(
            "duplicate canonical content hashes are forbidden"
        )
    ordered = sorted(digests)
    payload = (
        _CONTENT_SET_DOMAIN
        + b"\0"
        + len(ordered).to_bytes(8, byteorder="big", signed=False)
        + b"".join(ordered)
    )
    return _sha256_bytes(payload)


def _validate_jcs_subset(value: object, *, path: str = "$") -> None:
    """Validate the documented RFC-8785-equivalent subset used by P0.

    P0 artifacts and canonical semantic objects use ASCII field names and no
    floating-point numbers.  For this subset, ``json.dumps`` with sorted keys,
    compact separators, UTF-8, and no NaN is byte-equivalent to RFC 8785.  We
    reject values outside the subset instead of making a broader JCS claim.
    """

    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        if not -(2**53) + 1 <= value <= (2**53) - 1:
            raise CompactMessageCompletionIntegrityError(
                f"{path} integer is outside the exact JSON interoperability range"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_jcs_subset(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise CompactMessageCompletionIntegrityError(
                    f"{path} contains a non-ASCII JSON field name"
                )
            _validate_jcs_subset(item, path=f"{path}.{key}")
        return
    raise CompactMessageCompletionIntegrityError(
        f"{path} uses unsupported canonical JSON type {type(value).__name__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    _validate_jcs_subset(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "artifact_sha256" in value:
        raise CompactMessageCompletionIntegrityError(
            "artifact payload already contains its self-digest field"
        )
    result = dict(value)
    result["artifact_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def _verify_self_hash(value: Mapping[str, object], *, label: str) -> None:
    observed = value.get("artifact_sha256")
    if not isinstance(observed, str):
        raise CompactMessageCompletionIntegrityError(
            f"{label} has no artifact_sha256"
        )
    unhashed = dict(value)
    del unhashed["artifact_sha256"]
    expected = _sha256_bytes(_canonical_json_bytes(unhashed))
    if observed != expected:
        raise CompactMessageCompletionIntegrityError(
            f"{label} self-digest does not match its canonical payload"
        )


def _reject_json_constant(value: str) -> object:
    raise CompactMessageCompletionIntegrityError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CompactMessageCompletionIntegrityError(
                f"duplicate JSON field {key!r} is forbidden"
            )
        result[key] = value
    return result


def _decode_json_mapping(encoded: bytes, *, label: str) -> dict[str, object]:
    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompactMessageCompletionIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must contain a JSON object"
        )
    return value


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise CompactMessageCompletionIntegrityError(
            f"{label} is unavailable"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CompactMessageCompletionIntegrityError(
            f"{label} could not be opened safely"
        ) from error
    try:
        opened_details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_details.st_mode)
            or opened_details.st_dev != details.st_dev
            or opened_details.st_ino != details.st_ino
        ):
            raise CompactMessageCompletionIntegrityError(
                f"{label} changed identity while being opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_details = os.fstat(descriptor)
        if (
            final_details.st_size != opened_details.st_size
            or final_details.st_mtime_ns != opened_details.st_mtime_ns
            or final_details.st_ctime_ns != opened_details.st_ctime_ns
        ):
            raise CompactMessageCompletionIntegrityError(
                f"{label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path, *, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    details = os.lstat(path)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must be a non-symlink directory"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_exclusive_write(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing an existing artifact.

    A same-directory temporary file is fsynced and hard-linked to the final
    name.  ``link`` supplies atomic create-if-absent semantics.  An existing
    byte-identical artifact is accepted for crash-safe prefix resumption;
    differing bytes fail closed.
    """

    if not payload:
        raise CompactMessageCompletionIntegrityError(
            f"refusing to write empty artifact {path.name}"
        )
    _ensure_directory(path.parent, label="campaign output directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("artifact write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            observed = _read_regular_bytes(path, label=path.name)
            if observed != payload:
                raise CompactMessageCompletionFinalizedError(
                    f"existing artifact {path.name} differs from the required bytes"
                )
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        # The destination link and removal of the temporary directory entry
        # are separate metadata operations; persist both.
        _fsync_directory(path.parent)


def _write_json_artifact(path: Path, value: Mapping[str, object]) -> bytes:
    encoded = _canonical_json_bytes(_self_hashed(value)) + b"\n"
    _atomic_exclusive_write(path, encoded)
    return encoded


def _write_text_artifact(path: Path, value: str) -> bytes:
    encoded = value.encode("utf-8")
    _atomic_exclusive_write(path, encoded)
    return encoded


def _timestamp_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _implementation_sha256() -> str:
    return _sha256_bytes(
        _read_regular_bytes(Path(__file__), label="retirement implementation")
    )


def _semantic_spec_payload() -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "semantic_canonicalization_spec.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "text_normalization": {
            "input_encoding": "strict_utf8",
            "unicode_normalization": "NFC",
            "newline_rule": "CRLF_to_LF_then_CR_to_LF",
            "preserve_leading_whitespace": True,
            "preserve_trailing_whitespace": True,
            "preserve_repeated_internal_whitespace": True,
            "preserve_code_fences": True,
            "preserve_punctuation": True,
            "lowercase": False,
        },
        "role_and_message_order": {
            "allowed_roles": list(_ALLOWED_ROLES),
            "normalize_only_role_labels": True,
            "preserve_message_order": True,
            "sort_messages_by_role": False,
            "prompt_string_role": "user",
        },
        "structured_content": {
            "preserve_ordered_part_order": True,
            "include_only_fields_supplied_to_model": True,
        },
        "wrapper_fields": {
            "policy": "exclude_only_explicitly_nonsemantic_fields",
            "excluded_fields": list(_NONSEMANTIC_WRAPPER_FIELDS),
            "remove_model_input_affecting_fields": False,
        },
        "canonical_json": {
            "standard": "RFC 8785",
            "implementation": (
                "python_stdlib_ascii_field_names_no_floats_byte_equivalent_v1"
            ),
            "utf8": True,
            "deterministic_key_order": True,
            "non_finite_numbers_forbidden": True,
            "floating_point_values_supported": False,
            "field_name_constraint": "ASCII",
        },
        "canonical_content_hash": {
            "algorithm": "SHA-256",
            "payload": _CONTENT_HASH_PAYLOAD,
            "domain_prefix": None,
        },
        "aggregate_content_set_hash": {
            "algorithm": "SHA-256",
            "payload": _CONTENT_SET_PAYLOAD,
            "domain_ascii": _CONTENT_SET_DOMAIN.decode("ascii"),
            "domain_separator_hex": "00",
            "count_encoding": "uint64_big_endian",
            "digest_encoding": "raw_32_bytes",
            "digest_order": "lexicographic_by_raw_bytes",
            "duplicate_content_hashes_forbidden": True,
        },
        "local_identity_commitments": {
            "algorithm": "domain-separated SHA-256",
            "authority": "deterministic_local_bookkeeping_only",
            "private_guard_sufficiency": False,
            "domains_hex": {
                key: value.hex() for key, value in _COMMITMENT_DOMAINS.items()
            },
        },
        "freeze_rule": (
            "This artifact must exist and verify before any reserve JSON decode "
            "or semantic content hashing."
        ),
    }


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def canonical_semantic_prompt_bytes(prompt: str) -> bytes:
    """Canonicalize one prompt-string model input under the frozen E2 rule."""

    if not isinstance(prompt, str):
        raise CompactMessageCompletionIntegrityError(
            "semantic prompt must be a string"
        )
    return _canonical_json_bytes(
        {
            "messages": [
                {"role": "user", "content": _normalize_text(prompt)}
            ]
        }
    )


def canonical_content_sha256(prompt: str) -> str:
    """Hash one prompt-string model input using the frozen semantic bytes."""

    return _sha256_bytes(canonical_semantic_prompt_bytes(prompt))


def _local_commitment(kind: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CompactMessageCompletionIntegrityError(
            f"reserve {kind} identity must be a nonempty string when present"
        )
    return _sha256_bytes(_COMMITMENT_DOMAINS[kind] + value.encode("utf-8"))


def _canonicalize_reserve(
    repository_root: Path,
) -> tuple[str, list[dict[str, str | None]], str]:
    """Decode only the retired reserve and return sanitized hash records."""

    _require_reserve_decode_authorization_available()
    reserve_path = repository_root / DEFAULT_ROLE_CORPORA["reserve_unopened"]
    encoded = _read_regular_bytes(reserve_path, label="exposed reserve")
    file_sha256 = _sha256_bytes(encoded)
    if file_sha256 != PINNED_ROLE_FILE_SHA256S["reserve_unopened"]:
        raise CompactMessageCompletionIntegrityError(
            "exposed reserve file bytes differ from the frozen public pin"
        )
    reserve = _decode_json_mapping(encoded, label="exposed reserve")
    if reserve.get("role") != "reserve_unopened":
        raise CompactMessageCompletionIntegrityError(
            "exposed reserve role identity drifted"
        )
    examples = reserve.get("examples")
    if (
        not isinstance(examples, list)
        or not examples
        or reserve.get("example_count") != len(examples)
    ):
        raise CompactMessageCompletionIntegrityError(
            "exposed reserve example inventory is invalid"
        )

    records: list[dict[str, str | None]] = []
    seen_content_hashes: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise CompactMessageCompletionIntegrityError(
                f"exposed reserve item {index} is not an object"
            )
        prompt = example.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise CompactMessageCompletionIntegrityError(
                f"exposed reserve item {index} has no prompt string"
            )
        content_sha256 = canonical_content_sha256(prompt)
        if content_sha256 in seen_content_hashes:
            raise CompactMessageCompletionIntegrityError(
                "exposed reserve contains duplicate canonical content"
            )
        seen_content_hashes.add(content_sha256)
        semantic_commitment = _local_commitment(
            "semantic_instance", example.get("semantic_instance_id")
        )
        template_id = example.get("template_id")
        explicit_template_lineage = example.get("template_lineage_id")
        # The frozen panel schema supplies template_id rather than a separate
        # lineage field.  For this local bookkeeping registry, that frozen
        # template identity is the most conservative available lineage key.
        template_lineage_value = (
            explicit_template_lineage
            if explicit_template_lineage is not None
            else template_id
        )
        records.append(
            {
                "content_sha256": content_sha256,
                "analysis_unit_commitment": semantic_commitment,
                "template_lineage_commitment": _local_commitment(
                    "template_lineage", template_lineage_value
                ),
                "template_id_commitment": _local_commitment(
                    "template_id", template_id
                ),
                "semantic_instance_commitment": semantic_commitment,
                "family_commitment": _local_commitment(
                    "family", example.get("family_id")
                ),
            }
        )

    ordered_hashes = sorted(
        str(record["content_sha256"]) for record in records
    )
    aggregate_sha256 = content_set_sha256(ordered_hashes)
    return file_sha256, records, aggregate_sha256


def _require_reserve_decode_authorization_available() -> None:
    """Reject any further decode of the exhausted authoritative reserve.

    The historical one-time authorization was consumed before this corrected
    runner existed.  This immutable content-identity gate is deliberately
    independent of the caller-selected output and registry paths.  Synthetic
    qualification substitutes a distinct pinned reserve identity.
    """

    configured_reserve_sha256 = PINNED_ROLE_FILE_SHA256S.get(
        "reserve_unopened"
    )
    if configured_reserve_sha256 == EXHAUSTED_REAL_RESERVE_FILE_SHA256:
        raise CompactMessageCompletionFinalizedError(
            "the authoritative real-reserve decode authorization is exhausted; "
            "refusing any further JSON decode regardless of output or registry path"
        )


def _resolve_under_root(repository_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repository_root / value).resolve()


def _assert_registry_outside_output(
    registry_path: Path, output_dir: Path
) -> None:
    if registry_path == output_dir or output_dir in registry_path.parents:
        raise CompactMessageCompletionIntegrityError(
            "local development registry must be outside immutable campaign output"
        )


_REGISTRY_V1_TABLE_STATEMENTS: Final = (
        """
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """,
        """
        CREATE TABLE IF NOT EXISTS corpus_backfill (
            corpus_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            protocol_role TEXT NOT NULL,
            usage_role TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            file_sha256 TEXT NOT NULL,
            identity_state TEXT NOT NULL,
            first_recorded_at TEXT NOT NULL
        ) WITHOUT ROWID
        """,
        """
        CREATE TABLE IF NOT EXISTS content_items (
            content_sha256 TEXT PRIMARY KEY,
            usage_role TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            analysis_unit_mode_version TEXT NOT NULL,
            analysis_unit_commitment TEXT,
            template_lineage_commitment TEXT,
            template_id_commitment TEXT,
            semantic_instance_commitment TEXT,
            family_commitment TEXT,
            first_use_timestamp TEXT NOT NULL,
            eligibility_state TEXT NOT NULL,
            canonicalization_version TEXT NOT NULL,
            source_file_sha256 TEXT NOT NULL
        ) WITHOUT ROWID
        """,
        """
        CREATE TABLE IF NOT EXISTS content_sets (
            content_set_sha256 TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            canonicalization_version TEXT NOT NULL,
            source_file_sha256 TEXT NOT NULL,
            item_count INTEGER NOT NULL,
            first_transition_timestamp TEXT NOT NULL
        ) WITHOUT ROWID
        """,
        """
        CREATE TABLE IF NOT EXISTS reserve_retirements (
            reserve_path TEXT PRIMARY KEY,
            reserve_file_sha256 TEXT NOT NULL,
            content_set_sha256 TEXT NOT NULL UNIQUE,
            canonicalization_version TEXT NOT NULL,
            canonicalization_artifact_sha256 TEXT NOT NULL,
            item_count INTEGER NOT NULL,
            exposure_timestamp TEXT NOT NULL,
            exposure_timestamp_basis TEXT NOT NULL,
            retirement_implementation_sha256 TEXT NOT NULL,
            transition_from TEXT NOT NULL,
            transition_to TEXT NOT NULL
        ) WITHOUT ROWID
        """,
)

_REGISTRY_V2_TABLE_STATEMENTS: Final = (
    *_REGISTRY_V1_TABLE_STATEMENTS,
    """
    CREATE TABLE IF NOT EXISTS reserve_decode_claims (
        reserve_path TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL,
        claim_id TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL CHECK(
            state IN ('claimed', 'receipt_published', 'retired_exposed')
        ),
        claim_timestamp TEXT NOT NULL,
        expected_reserve_file_sha256 TEXT NOT NULL,
        canonicalization_artifact_sha256 TEXT NOT NULL,
        retirement_implementation_sha256 TEXT NOT NULL,
        receipt_artifact_sha256 TEXT
    ) WITHOUT ROWID
    """,
)

_REGISTRY_V2_TRIGGER_STATEMENTS: Final = (
    """
    CREATE TRIGGER IF NOT EXISTS registry_meta_no_update
    BEFORE UPDATE ON registry_meta
    BEGIN SELECT RAISE(ABORT, 'registry metadata is immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS registry_meta_no_delete
    BEFORE DELETE ON registry_meta
    BEGIN SELECT RAISE(ABORT, 'registry metadata is immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS corpus_backfill_no_update
    BEFORE UPDATE ON corpus_backfill
    BEGIN SELECT RAISE(ABORT, 'corpus backfill is append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS corpus_backfill_no_delete
    BEFORE DELETE ON corpus_backfill
    BEGIN SELECT RAISE(ABORT, 'corpus backfill is append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS content_items_irreversible_update
    BEFORE UPDATE ON content_items
    WHEN NOT (
        OLD.usage_role = 'unused'
        AND OLD.eligibility_state = 'unused'
        AND NEW.usage_role = 'retired_exposed'
        AND NEW.eligibility_state = 'retired_exposed'
        AND NEW.content_sha256 = OLD.content_sha256
    )
    BEGIN SELECT RAISE(ABORT, 'content item retirement is irreversible'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS content_items_no_delete
    BEFORE DELETE ON content_items
    BEGIN SELECT RAISE(ABORT, 'content item retirement cannot be deleted'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS content_sets_irreversible_update
    BEFORE UPDATE ON content_sets
    WHEN NOT (
        OLD.state = 'unopened'
        AND NEW.state = 'retired_exposed'
        AND NEW.content_set_sha256 = OLD.content_set_sha256
        AND NEW.canonicalization_version = OLD.canonicalization_version
        AND NEW.source_file_sha256 = OLD.source_file_sha256
        AND NEW.item_count = OLD.item_count
    )
    BEGIN SELECT RAISE(ABORT, 'content set retirement is irreversible'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS content_sets_no_delete
    BEFORE DELETE ON content_sets
    BEGIN SELECT RAISE(ABORT, 'content set retirement cannot be deleted'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reserve_retirements_no_update
    BEFORE UPDATE ON reserve_retirements
    BEGIN SELECT RAISE(ABORT, 'reserve retirement is immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reserve_retirements_no_delete
    BEFORE DELETE ON reserve_retirements
    BEGIN SELECT RAISE(ABORT, 'reserve retirement cannot be deleted'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reserve_decode_claims_transition_only
    BEFORE UPDATE ON reserve_decode_claims
    WHEN NOT (
        (
            OLD.state = 'claimed'
            AND NEW.state = 'receipt_published'
            AND OLD.receipt_artifact_sha256 IS NULL
            AND NEW.receipt_artifact_sha256 IS NOT NULL
        )
        OR (
            OLD.state = 'receipt_published'
            AND NEW.state = 'retired_exposed'
            AND OLD.receipt_artifact_sha256 = NEW.receipt_artifact_sha256
        )
    )
    OR NOT (
        NEW.reserve_path = OLD.reserve_path
        AND NEW.campaign_id = OLD.campaign_id
        AND NEW.claim_id = OLD.claim_id
        AND NEW.claim_timestamp = OLD.claim_timestamp
        AND NEW.expected_reserve_file_sha256 = OLD.expected_reserve_file_sha256
        AND NEW.canonicalization_artifact_sha256 = OLD.canonicalization_artifact_sha256
        AND NEW.retirement_implementation_sha256 = OLD.retirement_implementation_sha256
    )
    BEGIN SELECT RAISE(ABORT, 'reserve decode claim transition is invalid'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reserve_decode_claims_no_delete
    BEFORE DELETE ON reserve_decode_claims
    BEGIN SELECT RAISE(ABORT, 'reserve decode claim cannot be deleted'); END
    """,
)

def _create_registry_schema(
    connection: sqlite3.Connection, *, version: int = REGISTRY_SCHEMA_VERSION
) -> None:
    if version != REGISTRY_SCHEMA_VERSION:
        raise CompactMessageCompletionIntegrityError(
            "only the current registry schema may be created"
        )
    for statement in _REGISTRY_V2_TABLE_STATEMENTS:
        connection.execute(statement)
    for statement in _REGISTRY_V2_TRIGGER_STATEMENTS:
        connection.execute(statement)


def _validate_registry_schema(
    connection: sqlite3.Connection, *, version: int
) -> None:
    if version not in (LEGACY_REGISTRY_SCHEMA_VERSION, REGISTRY_SCHEMA_VERSION):
        raise CompactMessageCompletionIntegrityError(
            f"unsupported local registry schema version {version}"
        )
    expected_connection = sqlite3.connect(":memory:")
    try:
        statements = (
            _REGISTRY_V1_TABLE_STATEMENTS
            if version == LEGACY_REGISTRY_SCHEMA_VERSION
            else (
                *_REGISTRY_V2_TABLE_STATEMENTS,
                *_REGISTRY_V2_TRIGGER_STATEMENTS,
            )
        )
        for statement in statements:
            expected_connection.execute(statement)

        def inventory(database: sqlite3.Connection):
            return tuple(
                (
                    str(kind),
                    str(name),
                    str(table),
                    None if sql is None else " ".join(str(sql).split()),
                )
                for kind, name, table, sql in database.execute(
                    """
                    SELECT type, name, tbl_name, sql FROM sqlite_master
                    WHERE type IN ('table', 'index', 'trigger', 'view')
                    ORDER BY type, name
                    """
                )
            )

        expected_inventory = inventory(expected_connection)
        observed_inventory = inventory(connection)
    finally:
        expected_connection.close()
    if observed_inventory != expected_inventory:
        raise CompactMessageCompletionIntegrityError(
            "local registry exact schema, constraints, or guards drifted"
        )


def _registry_version(connection: sqlite3.Connection) -> int | None:
    meta_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'registry_meta'"
    ).fetchone()
    if meta_exists is None:
        return None
    row = connection.execute(
        "SELECT value FROM registry_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise CompactMessageCompletionIntegrityError(
            "local registry has no authenticated schema version"
        )
    try:
        return int(row[0])
    except (TypeError, ValueError) as error:
        raise CompactMessageCompletionIntegrityError(
            "local registry schema version is invalid"
        ) from error


def _open_registry_rw(registry_path: Path) -> sqlite3.Connection:
    _ensure_directory(registry_path.parent, label="local registry directory")
    prior_identity: tuple[int, int] | None = None
    try:
        details = os.lstat(registry_path)
    except FileNotFoundError:
        details = None
    if details is not None:
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise CompactMessageCompletionIntegrityError(
                "local registry must be a regular non-symlink file"
            )
        prior_identity = (details.st_dev, details.st_ino)
    try:
        connection = sqlite3.connect(
            str(registry_path), timeout=30.0, isolation_level=None
        )
    except sqlite3.Error as error:
        raise CompactMessageCompletionIntegrityError(
            "local development registry could not be opened"
        ) from error
    try:
        opened_path_details = os.lstat(registry_path)
        if (
            stat.S_ISLNK(opened_path_details.st_mode)
            or not stat.S_ISREG(opened_path_details.st_mode)
            or (
                prior_identity is not None
                and prior_identity
                != (opened_path_details.st_dev, opened_path_details.st_ino)
            )
        ):
            raise CompactMessageCompletionIntegrityError(
                "local registry changed identity while being opened"
            )
        os.chmod(registry_path, 0o600)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA trusted_schema = OFF")
    except BaseException:
        connection.close()
        raise
    return connection


def _initialize_current_registry(connection: sqlite3.Connection) -> None:
    version = _registry_version(connection)
    if version is None:
        _create_registry_schema(connection)
        _insert_or_verify_meta(
            connection, "schema_version", str(REGISTRY_SCHEMA_VERSION)
        )
        _insert_or_verify_meta(
            connection, "authority", "non_authoritative_host_local_development"
        )
        _insert_or_verify_meta(
            connection,
            f"canonicalization_version:{CANONICALIZATION_VERSION}",
            "frozen",
        )
        _insert_or_verify_meta(
            connection, "schema_contract", "v2_exact_irreversible_guards"
        )
        version = REGISTRY_SCHEMA_VERSION
    if version != REGISTRY_SCHEMA_VERSION:
        raise CompactMessageCompletionFinalizedError(
            "legacy registry cannot be mutated or resumed by the conformant runner"
        )
    _validate_registry_schema(connection, version=version)
    expected_meta = {
        "schema_version": str(REGISTRY_SCHEMA_VERSION),
        "authority": "non_authoritative_host_local_development",
        f"canonicalization_version:{CANONICALIZATION_VERSION}": "frozen",
        "schema_contract": "v2_exact_irreversible_guards",
    }
    observed_meta = dict(
        connection.execute("SELECT key, value FROM registry_meta").fetchall()
    )
    if observed_meta != expected_meta:
        raise CompactMessageCompletionIntegrityError(
            "current local registry metadata field set drifted"
        )


def _acquire_reserve_decode_claim(
    *,
    registry_path: Path,
    expected_reserve_file_sha256: str,
    canonicalization_artifact_sha256: str,
    implementation_sha256: str,
) -> dict[str, str]:
    """Durably consume the one permission to decode the exposed reserve.

    Once this transaction commits, a retry without a published sanitized
    receipt is deliberately unrecoverable: the process may have decoded the
    reserve before crashing, so opening it again would violate at-most-once.
    """

    _raw_sha256_digest(
        expected_reserve_file_sha256,
        label="expected_reserve_file_sha256",
    )

    connection = _open_registry_rw(registry_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _initialize_current_registry(connection)
        reserve_path = str(DEFAULT_ROLE_CORPORA["reserve_unopened"])
        existing = connection.execute(
            """
            SELECT campaign_id, claim_id, state, claim_timestamp,
                   expected_reserve_file_sha256,
                   canonicalization_artifact_sha256,
                   retirement_implementation_sha256,
                   receipt_artifact_sha256
            FROM reserve_decode_claims WHERE reserve_path = ?
            """,
            (reserve_path,),
        ).fetchone()
        if existing is not None:
            raise CompactMessageCompletionFinalizedError(
                "durable reserve decode claim already exists without a usable "
                "sanitized receipt; refusing to reopen the raw reserve"
            )
        claim = {
            "claim_id": secrets.token_hex(32),
            "claim_timestamp": _timestamp_utc(),
            "expected_reserve_file_sha256": expected_reserve_file_sha256,
        }
        connection.execute(
            """
            INSERT INTO reserve_decode_claims(
                reserve_path, campaign_id, claim_id, state, claim_timestamp,
                expected_reserve_file_sha256,
                canonicalization_artifact_sha256,
                retirement_implementation_sha256, receipt_artifact_sha256
            ) VALUES (?, ?, ?, 'claimed', ?, ?, ?, ?, NULL)
            """,
            (
                reserve_path,
                CAMPAIGN_ID,
                claim["claim_id"],
                claim["claim_timestamp"],
                claim["expected_reserve_file_sha256"],
                canonicalization_artifact_sha256,
                implementation_sha256,
            ),
        )
        connection.execute("COMMIT")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    _fsync_directory(registry_path.parent)
    return claim


def _bind_published_reserve_receipt(
    *,
    registry_path: Path,
    receipt_path: Path,
    reserve_receipt: Mapping[str, object],
) -> str:
    """Bind exact durable receipt bytes to the already-consumed decode claim."""

    receipt_bytes = _read_regular_bytes(
        receipt_path, label="reserve_retired_exposed.json"
    )
    if receipt_bytes != _canonical_json_bytes(reserve_receipt) + b"\n":
        raise CompactMessageCompletionIntegrityError(
            "published reserve receipt does not have exact canonical bytes"
        )
    receipt_file_sha256 = _sha256_bytes(receipt_bytes)
    reserve_path = str(reserve_receipt["reserve_path"])
    expected_identity = (
        CAMPAIGN_ID,
        reserve_receipt["pre_decode_claim_id"],
        reserve_receipt["pre_decode_claim_timestamp"],
        reserve_receipt["reserve_file_sha256"],
        reserve_receipt["canonicalization_artifact_sha256"],
        reserve_receipt["retirement_implementation_sha256"],
    )
    connection = _open_registry_rw(registry_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _initialize_current_registry(connection)
        row = connection.execute(
            """
            SELECT campaign_id, claim_id, state, claim_timestamp,
                   expected_reserve_file_sha256,
                   canonicalization_artifact_sha256,
                   retirement_implementation_sha256,
                   receipt_artifact_sha256
            FROM reserve_decode_claims WHERE reserve_path = ?
            """,
            (reserve_path,),
        ).fetchone()
        if row is None or (
            row[0], row[1], row[3], row[4], row[5], row[6]
        ) != expected_identity:
            raise CompactMessageCompletionIntegrityError(
                "published reserve receipt has no matching durable decode claim"
            )
        if row[2] == "claimed" and row[7] is None:
            changed = connection.execute(
                """
                UPDATE reserve_decode_claims
                SET state = 'receipt_published', receipt_artifact_sha256 = ?
                WHERE reserve_path = ? AND state = 'claimed'
                  AND expected_reserve_file_sha256 = ?
                  AND receipt_artifact_sha256 IS NULL
                """,
                (
                    receipt_file_sha256,
                    reserve_path,
                    reserve_receipt["reserve_file_sha256"],
                ),
            ).rowcount
            if changed != 1:
                raise CompactMessageCompletionIntegrityError(
                    "reserve receipt binding lost atomicity"
                )
        elif row[2] not in ("receipt_published", "retired_exposed") or row[7] != (
            receipt_file_sha256
        ):
            raise CompactMessageCompletionIntegrityError(
                "durable decode claim binds different reserve receipt bytes"
            )
        connection.execute("COMMIT")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    _fsync_directory(registry_path.parent)
    return receipt_file_sha256


def _insert_or_verify_meta(
    connection: sqlite3.Connection, key: str, value: str
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO registry_meta(key, value) VALUES (?, ?)",
        (key, value),
    )
    row = connection.execute(
        "SELECT value FROM registry_meta WHERE key = ?", (key,)
    ).fetchone()
    if row != (value,):
        raise CompactMessageCompletionIntegrityError(
            f"local registry metadata mismatch for {key}"
        )


def _insert_or_verify_corpus(
    connection: sqlite3.Connection,
    *,
    role: str,
    recorded_at: str,
    allow_insert: bool = True,
) -> None:
    relative_path = str(DEFAULT_ROLE_CORPORA[role])
    expected = (
        f"gemma3-chart-native-neighbor-fresh-panels-v1:{role}",
        CAMPAIGN_ID,
        role,
        _HISTORICAL_USAGE_ROLES[role],
        relative_path,
        PINNED_ROLE_FILE_SHA256S[role],
        (
            "retired_exposed"
            if role == "reserve_unopened"
            else "historical_identity_unverifiable_guard_ineligible"
        ),
        recorded_at,
    )
    if allow_insert:
        connection.execute(
            """
            INSERT OR IGNORE INTO corpus_backfill(
                corpus_id, campaign_id, protocol_role, usage_role, relative_path,
                file_sha256, identity_state, first_recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            expected,
        )
    observed = connection.execute(
        """
        SELECT corpus_id, campaign_id, protocol_role, usage_role, relative_path,
               file_sha256, identity_state, first_recorded_at
        FROM corpus_backfill WHERE corpus_id = ?
        """,
        (expected[0],),
    ).fetchone()
    if observed != expected:
        raise CompactMessageCompletionIntegrityError(
            f"local registry corpus backfill mismatch for {role}"
        )


def _retire_content_item(
    connection: sqlite3.Connection,
    *,
    record: Mapping[str, str | None],
    reserve_file_sha256: str,
    recorded_at: str,
    allow_transition: bool = True,
) -> None:
    content_sha256 = record["content_sha256"]
    if not isinstance(content_sha256, str):
        raise CompactMessageCompletionIntegrityError(
            "sanitized reserve record has no content hash"
        )
    existing = connection.execute(
        """
        SELECT usage_role, campaign_id, analysis_unit_mode_version,
               analysis_unit_commitment, template_lineage_commitment,
               template_id_commitment, semantic_instance_commitment,
               family_commitment, first_use_timestamp, eligibility_state,
               canonicalization_version, source_file_sha256
        FROM content_items WHERE content_sha256 = ?
        """,
        (content_sha256,),
    ).fetchone()
    expected = (
        "retired_exposed",
        CAMPAIGN_ID,
        "unselected_preflight_v1",
        record["analysis_unit_commitment"],
        record["template_lineage_commitment"],
        record["template_id_commitment"],
        record["semantic_instance_commitment"],
        record["family_commitment"],
        recorded_at,
        "retired_exposed",
        CANONICALIZATION_VERSION,
        reserve_file_sha256,
    )
    if existing is None:
        if not allow_transition:
            raise CompactMessageCompletionIntegrityError(
                "retired reserve content row disappeared after atomic commit"
            )
        connection.execute(
            """
            INSERT INTO content_items(
                content_sha256, usage_role, campaign_id,
                analysis_unit_mode_version, analysis_unit_commitment,
                template_lineage_commitment, template_id_commitment,
                semantic_instance_commitment, family_commitment,
                first_use_timestamp, eligibility_state,
                canonicalization_version, source_file_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (content_sha256, *expected),
        )
    elif (
        allow_transition
        and existing[9] == "unused"
        and existing[0] == "unused"
    ):
        changed = connection.execute(
            """
            UPDATE content_items
            SET usage_role = ?, campaign_id = ?, analysis_unit_mode_version = ?,
                analysis_unit_commitment = ?,
                template_lineage_commitment = ?, template_id_commitment = ?,
                semantic_instance_commitment = ?, family_commitment = ?,
                first_use_timestamp = ?, eligibility_state = ?,
                canonicalization_version = ?, source_file_sha256 = ?
            WHERE content_sha256 = ? AND eligibility_state = 'unused'
            """,
            (*expected, content_sha256),
        ).rowcount
        if changed != 1:
            raise CompactMessageCompletionIntegrityError(
                "reserve content retirement transition lost atomicity"
            )
    elif existing != expected:
        raise CompactMessageCompletionIntegrityError(
            "reserve content hash already has a different or incompatible registry state"
        )


def _registry_snapshot(
    connection: sqlite3.Connection,
    *,
    content_set_sha256: str,
) -> dict[str, object]:
    corpus_rows = [
        list(row)
        for row in connection.execute(
            """
            SELECT corpus_id, protocol_role, usage_role, relative_path,
                   file_sha256, identity_state, first_recorded_at
            FROM corpus_backfill WHERE campaign_id = ?
            ORDER BY corpus_id
            """,
            (CAMPAIGN_ID,),
        )
    ]
    content_rows = [
        list(row)
        for row in connection.execute(
            """
            SELECT content_sha256, usage_role, analysis_unit_mode_version,
                   analysis_unit_commitment, template_lineage_commitment,
                   template_id_commitment, semantic_instance_commitment,
                   family_commitment, first_use_timestamp, eligibility_state,
                   canonicalization_version, source_file_sha256
            FROM content_items
            WHERE campaign_id = ? AND usage_role = 'retired_exposed'
            ORDER BY content_sha256
            """,
            (CAMPAIGN_ID,),
        )
    ]
    set_row = connection.execute(
        """
        SELECT content_set_sha256, state, canonicalization_version,
               source_file_sha256, item_count, first_transition_timestamp
        FROM content_sets WHERE content_set_sha256 = ?
        """,
        (content_set_sha256,),
    ).fetchone()
    if set_row is None:
        raise CompactMessageCompletionIntegrityError(
            "retired reserve aggregate set is absent from local registry"
        )
    retirement_row = connection.execute(
        """
        SELECT reserve_path, reserve_file_sha256, content_set_sha256,
               canonicalization_version, canonicalization_artifact_sha256,
               item_count, exposure_timestamp, exposure_timestamp_basis,
               retirement_implementation_sha256, transition_from,
               transition_to
        FROM reserve_retirements WHERE content_set_sha256 = ?
        """,
        (content_set_sha256,),
    ).fetchone()
    if retirement_row is None:
        raise CompactMessageCompletionIntegrityError(
            "retired reserve event is absent from local registry"
        )
    version = _registry_version(connection)
    if version is None:
        raise CompactMessageCompletionIntegrityError(
            "local registry snapshot has no schema version"
        )
    logical_snapshot = {
        "registry_schema_version": version,
        "campaign_id": CAMPAIGN_ID,
        "corpus_rows": corpus_rows,
        "retired_content_rows": content_rows,
        "retired_content_set_row": list(set_row),
        "reserve_retirement_row": list(retirement_row),
    }
    if version == REGISTRY_SCHEMA_VERSION:
        claim_row = connection.execute(
            """
            SELECT reserve_path, campaign_id, claim_id, state,
                   claim_timestamp, expected_reserve_file_sha256,
                   canonicalization_artifact_sha256,
                   retirement_implementation_sha256,
                   receipt_artifact_sha256
            FROM reserve_decode_claims WHERE reserve_path = ?
            """,
            (retirement_row[0],),
        ).fetchone()
        if claim_row is None:
            raise CompactMessageCompletionIntegrityError(
                "retired reserve has no durable pre-decode claim"
            )
        logical_snapshot["reserve_decode_claim_row"] = list(claim_row)
    return {
        "logical_snapshot_sha256": _sha256_bytes(
            _canonical_json_bytes(logical_snapshot)
        ),
        "backfilled_corpus_count": len(corpus_rows),
        "retired_individual_content_count": len(content_rows),
        "retired_aggregate_content_set_count": 1,
        "retired_content_set_state": set_row[1],
    }


def _reserve_retirement_payload(
    *,
    reserve_file_sha256: str,
    records: Sequence[Mapping[str, str | None]],
    content_set_sha256: str,
    canonicalization_artifact_sha256: str,
    implementation_sha256: str,
    claim_id: str,
    claim_timestamp: str,
) -> dict[str, object]:
    return {
        "reserve_path": str(DEFAULT_ROLE_CORPORA["reserve_unopened"]),
        "reserve_file_sha256": reserve_file_sha256,
        "canonical_content_set_sha256": content_set_sha256,
        "canonical_per_item_sha256s": sorted(
            str(record["content_sha256"]) for record in records
        ),
        "canonical_per_item_commitments": sorted(
            (dict(record) for record in records),
            key=lambda record: str(record["content_sha256"]),
        ),
        "semantic_instance_commitments": sorted(
            str(record["semantic_instance_commitment"])
            for record in records
            if record["semantic_instance_commitment"] is not None
        ),
        "template_lineage_commitments": sorted(
            str(record["template_lineage_commitment"])
            for record in records
            if record["template_lineage_commitment"] is not None
        ),
        "template_id_commitments": sorted(
            str(record["template_id_commitment"])
            for record in records
            if record["template_id_commitment"] is not None
        ),
        "family_commitments": sorted(
            str(record["family_commitment"])
            for record in records
            if record["family_commitment"] is not None
        ),
        "template_lineage_commitment_basis": (
            _TEMPLATE_LINEAGE_COMMITMENT_BASIS
        ),
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalization_artifact_sha256": canonicalization_artifact_sha256,
        "exposure_reason": _EXPOSURE_REASON,
        "exposure_timestamp": claim_timestamp,
        "exposure_timestamp_basis": "durable_pre_decode_claim_commit",
        "retirement_implementation_sha256": implementation_sha256,
        "pre_decode_claim_id": claim_id,
        "pre_decode_claim_timestamp": claim_timestamp,
        "pre_decode_claim_state_at_publication": "claimed",
        "receipt_published_before_registry_retirement": True,
        "irreversible_transition": {
            "from": "unopened",
            "to": "retired_exposed",
            "future_allowed_states": [],
        },
        "item_count": len(records),
        "every_individual_content_permanently_guard_ineligible": True,
        "aggregate_content_set_permanently_guard_ineligible": True,
        "raw_content_in_receipt": False,
        "raw_identity_in_receipt": False,
    }


def _retire_reserve_in_local_registry(
    *,
    registry_path: Path,
    receipt_path: Path,
    reserve_receipt: Mapping[str, object],
    records: Sequence[Mapping[str, str | None]],
) -> dict[str, object]:
    """Commit retirement only after the sanitized receipt is durable."""

    receipt_bytes = _read_regular_bytes(
        receipt_path, label="reserve_retired_exposed.json"
    )
    if receipt_bytes != _canonical_json_bytes(reserve_receipt) + b"\n":
        raise CompactMessageCompletionIntegrityError(
            "reserve receipt was not durably published with exact canonical bytes"
        )
    receipt_file_sha256 = _sha256_bytes(receipt_bytes)
    reserve_path = str(reserve_receipt["reserve_path"])
    reserve_file_sha256 = str(reserve_receipt["reserve_file_sha256"])
    content_set_sha256 = str(
        reserve_receipt["canonical_content_set_sha256"]
    )
    recorded_at = str(reserve_receipt["exposure_timestamp"])
    canonicalization_artifact_sha256 = str(
        reserve_receipt["canonicalization_artifact_sha256"]
    )
    implementation_sha256 = str(
        reserve_receipt["retirement_implementation_sha256"]
    )
    claim_id = str(reserve_receipt["pre_decode_claim_id"])

    connection = _open_registry_rw(registry_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _initialize_current_registry(connection)
        claim_row = connection.execute(
            """
            SELECT campaign_id, claim_id, state, claim_timestamp,
                   expected_reserve_file_sha256,
                   canonicalization_artifact_sha256,
                   retirement_implementation_sha256,
                   receipt_artifact_sha256
            FROM reserve_decode_claims WHERE reserve_path = ?
            """,
            (reserve_path,),
        ).fetchone()
        expected_claim_prefix = (
            CAMPAIGN_ID,
            claim_id,
            recorded_at,
            reserve_file_sha256,
            canonicalization_artifact_sha256,
            implementation_sha256,
        )
        if claim_row is None:
            raise CompactMessageCompletionFinalizedError(
                "sanitized reserve receipt cannot recover a missing durable decode claim"
            )
        observed_claim_prefix = (
            claim_row[0],
            claim_row[1],
            claim_row[3],
            claim_row[4],
            claim_row[5],
            claim_row[6],
        )
        if observed_claim_prefix != expected_claim_prefix:
            raise CompactMessageCompletionIntegrityError(
                "durable reserve decode claim differs from the sanitized receipt"
            )
        claim_state = str(claim_row[2])
        if claim_state == "claimed":
            raise CompactMessageCompletionFinalizedError(
                "reserve receipt publication was not durably bound to its decode "
                "claim; refusing uncertain recovery"
            )
        if claim_state not in ("receipt_published", "retired_exposed") or (
            claim_row[7] != receipt_file_sha256
        ):
            raise CompactMessageCompletionIntegrityError(
                "durable reserve claim binds different receipt bytes"
            )

        expected_retirement = (
            reserve_file_sha256,
            content_set_sha256,
            CANONICALIZATION_VERSION,
            canonicalization_artifact_sha256,
            len(records),
            recorded_at,
            "durable_pre_decode_claim_commit",
            implementation_sha256,
            "unopened",
            "retired_exposed",
        )
        existing_retirement = connection.execute(
            """
            SELECT reserve_file_sha256, content_set_sha256,
                   canonicalization_version,
                   canonicalization_artifact_sha256, item_count,
                   exposure_timestamp, exposure_timestamp_basis,
                   retirement_implementation_sha256, transition_from,
                   transition_to
            FROM reserve_retirements WHERE reserve_path = ?
            """,
            (reserve_path,),
        ).fetchone()
        if existing_retirement is None and claim_state == "retired_exposed":
            raise CompactMessageCompletionIntegrityError(
                "reserve retirement row disappeared after atomic commit"
            )
        if existing_retirement is None:
            connection.execute(
                """
                INSERT INTO reserve_retirements(
                    reserve_path, reserve_file_sha256, content_set_sha256,
                    canonicalization_version,
                    canonicalization_artifact_sha256, item_count,
                    exposure_timestamp, exposure_timestamp_basis,
                    retirement_implementation_sha256, transition_from,
                    transition_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (reserve_path, *expected_retirement),
            )
        elif existing_retirement != expected_retirement:
            raise CompactMessageCompletionIntegrityError(
                "existing reserve retirement differs from the sanitized receipt"
            )

        for role in ORDERED_ROLES:
            _insert_or_verify_corpus(
                connection,
                role=role,
                recorded_at=recorded_at,
                allow_insert=claim_state != "retired_exposed",
            )

        set_row = connection.execute(
            """
            SELECT state, canonicalization_version, source_file_sha256,
                   item_count, first_transition_timestamp
            FROM content_sets WHERE content_set_sha256 = ?
            """,
            (content_set_sha256,),
        ).fetchone()
        expected_set = (
            "retired_exposed",
            CANONICALIZATION_VERSION,
            reserve_file_sha256,
            len(records),
            recorded_at,
        )
        if set_row is None and claim_state == "retired_exposed":
            raise CompactMessageCompletionIntegrityError(
                "reserve aggregate row disappeared after atomic commit"
            )
        if set_row is None:
            connection.execute(
                """
                INSERT INTO content_sets(
                    content_set_sha256, state, canonicalization_version,
                    source_file_sha256, item_count,
                    first_transition_timestamp
                ) VALUES (?, 'unopened', ?, ?, ?, ?)
                """,
                (
                    content_set_sha256,
                    CANONICALIZATION_VERSION,
                    reserve_file_sha256,
                    len(records),
                    recorded_at,
                ),
            )
            changed = connection.execute(
                """
                UPDATE content_sets SET state = 'retired_exposed'
                WHERE content_set_sha256 = ? AND state = 'unopened'
                """,
                (content_set_sha256,),
            ).rowcount
            if changed != 1:
                raise CompactMessageCompletionIntegrityError(
                    "reserve aggregate retirement transition was not exclusive"
                )
        elif set_row != expected_set:
            raise CompactMessageCompletionIntegrityError(
                "reserve aggregate set is not irreversibly retired_exposed"
            )

        for record in records:
            _retire_content_item(
                connection,
                record=record,
                reserve_file_sha256=reserve_file_sha256,
                recorded_at=recorded_at,
                allow_transition=claim_state != "retired_exposed",
            )

        if claim_state == "receipt_published":
            changed = connection.execute(
                """
                UPDATE reserve_decode_claims
                SET state = 'retired_exposed', receipt_artifact_sha256 = ?
                WHERE reserve_path = ? AND state = 'receipt_published'
                  AND expected_reserve_file_sha256 = ?
                  AND receipt_artifact_sha256 = ?
                """,
                (
                    receipt_file_sha256,
                    reserve_path,
                    reserve_file_sha256,
                    receipt_file_sha256,
                ),
            ).rowcount
            if changed != 1:
                raise CompactMessageCompletionIntegrityError(
                    "reserve claim retirement transition lost atomicity"
                )
        snapshot = _registry_snapshot(
            connection, content_set_sha256=content_set_sha256
        )
        connection.execute("COMMIT")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    _fsync_directory(registry_path.parent)
    return snapshot


def _require_exact_nested_bool(
    root: Mapping[str, object], path: Sequence[str]
) -> bool:
    current: object = root
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise CompactMessageCompletionIntegrityError(
                f"required frozen provenance field {'.'.join(path)} is absent"
            )
        current = current[component]
    if type(current) is not bool:
        raise CompactMessageCompletionIntegrityError(
            f"required frozen provenance field {'.'.join(path)} is not an exact Boolean"
        )
    return current


def classify_corpus_roles(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Classify A/B/C/D and the exposed reserve without opening raw corpora."""

    semantic_d = _require_exact_nested_bool(
        receipt,
        (
            "authoring_access_receipt",
            "semantic_parse_access",
            "D_response_validation",
        ),
    )
    tokenizer_d = _require_exact_nested_bool(
        receipt,
        (
            "authoring_access_receipt",
            "tokenizer_access",
            "D_response_validation",
        ),
    )
    if not semantic_d and not tokenizer_d:
        raise CompactMessageCompletionIntegrityError(
            "current pinned D provenance unexpectedly satisfies metadata-only access; "
            "a separately versioned non-P0 campaign is required"
        )
    disqualifying_flags = [
        name
        for name, observed in (
            ("semantic_parse_access", semantic_d),
            ("tokenizer_access", tokenizer_d),
        )
        if observed
    ]
    access_reason = " and ".join(disqualifying_flags)
    access_verb = "are" if len(disqualifying_flags) > 1 else "is"

    common_false = {
        "eligible_for_fit": False,
        "eligible_for_tune": False,
        "eligible_for_development_confirm": False,
        "eligible_for_guard": False,
    }
    return {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "corpus_role_classification.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "exposure_policy_version": D_ACCESS_POLICY,
        "frozen_provenance_receipt": {
            "path": str(DEFAULT_RECEIPT),
            "file_sha256": PINNED_RECEIPT_FILE_SHA256,
            "receipt_sha256": PINNED_RECEIPT_SHA256,
        },
        "corpora": {
            "A_chart_transport_fit": {
                "classification": "fit_only",
                "eligible_for_fit": True,
                "eligible_for_tune": False,
                "eligible_for_development_confirm": False,
                "eligible_for_guard": False,
                "decision_reason": "A is frozen for fit use only.",
            },
            "B_neighbor_discovery": {
                "classification": "ineligible_prior_scientific_use",
                **common_false,
                "decision_reason": "B is ineligible due to prior scientific use.",
            },
            "C_neighbor_confirmation": {
                "classification": "ineligible_prior_scientific_use",
                **common_false,
                "decision_reason": "C is ineligible due to prior scientific use.",
            },
            "D_response_validation": {
                "classification": "preoutcome_executable_access",
                **common_false,
                "frozen_access_flags": {
                    "semantic_parse_access": semantic_d,
                    "tokenizer_access": tokenizer_d,
                },
                "decision_reason": (
                    "D is ineligible under strict_metadata_only_v1 because "
                    f"{access_reason} {access_verb} true."
                ),
                "outcome_access_asserted": False,
            },
            "reserve_unopened": {
                "classification": "retired_exposed",
                **common_false,
                "decision_reason": (
                    "The formerly unopened reserve is permanently ineligible "
                    "after prior raw prompt exposure."
                ),
                "irreversible": True,
            },
        },
        "eligible_independent_tune_corpus_exists": False,
        "eligible_independent_development_confirm_corpus_exists": False,
        "raw_corpus_content_read_for_classification": False,
        "flattened_access_field_fallback_allowed": False,
    }


def _legacy_classify_corpus_roles(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Reproduce the already-published v1 classification for audit only."""

    classification = classify_corpus_roles(receipt)
    corpora = classification["corpora"]
    assert isinstance(corpora, dict)
    del corpora["reserve_unopened"]
    return classification


def build_preflight_decision(
    classification: Mapping[str, object],
) -> dict[str, object]:
    corpora = classification.get("corpora")
    if not isinstance(corpora, Mapping):
        raise CompactMessageCompletionIntegrityError(
            "corpus classification has no corpus decisions"
        )
    a = corpora.get("A_chart_transport_fit")
    d = corpora.get("D_response_validation")
    if (
        not isinstance(a, Mapping)
        or a.get("eligible_for_fit") is not True
        or not isinstance(d, Mapping)
        or d.get("classification") != "preoutcome_executable_access"
    ):
        raise CompactMessageCompletionIntegrityError(
            "corpus classification cannot support the frozen P0 decision"
        )
    return {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "preflight_decision.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "can_construct_fit": True,
        "can_construct_tune": False,
        "can_construct_development_confirm": False,
        "campaign_can_start": False,
        "reason": TERMINAL_REASON,
        "terminal_state": CURRENT_TERMINAL_STATE,
        "termination_before": [
            "independence_mode_selection",
            "split_construction",
            "model_identity_resolution",
            "source_inventory_sealing",
            "execution_budget_campaign_sealing",
            "authoritative_fit_capture",
            "O1",
            "representation_fitting",
        ],
        "successful_fail_closed_protocol_outcome": True,
        "scientific_result_present": False,
    }


def _registry_status_payload(
    *,
    registry_path: Path,
    snapshot: Mapping[str, object],
    registry_schema_version: int = REGISTRY_SCHEMA_VERSION,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "local_development_registry_status.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "registry_kind": "host_local_sqlite",
        "registry_schema_version": registry_schema_version,
        "registry_location": str(registry_path),
        "registry_outside_immutable_campaign_output": True,
        "transactional_writes": True,
        "unique_content_hash_constraints": True,
        "reserve_file_hash_registered": True,
        "every_reserve_content_hash_registered": True,
        "reserve_aggregate_content_set_hash_registered": True,
        "reserve_state": "retired_exposed",
        "historical_roles_backfilled": list(ORDERED_ROLES),
        "historical_identity_policy": (
            "Unavailable historical canonical identities are recorded as "
            "historical_identity_unverifiable and guard-ineligible."
        ),
        "logical_snapshot": dict(snapshot),
        "authority_disclaimer": (
            "This registry is host-local and does not establish private-evaluator "
            "freshness, cross-host exclusivity, or guard authority."
        ),
        "future_private_guard_authority": False,
    }
    if registry_schema_version == REGISTRY_SCHEMA_VERSION:
        payload.update(
            {
                "durable_pre_decode_claim": True,
                "reserve_receipt_published_before_retirement_commit": True,
                "registry_schema_exactly_verified": True,
                "irreversible_update_delete_guards": True,
                "contract_conformant": True,
            }
        )
    return payload


def _commands_markdown(
    *, repository_root: Path, output_dir: Path, registry_path: Path
) -> str:
    return f"""# Compact-message completion probe v2 commands

The current corpus set reaches only P0 and terminates fail-closed. The runner
does not load a model or tokenizer and does not construct a split, capture, or
guard claim.

```sh
cd {repository_root}
.venv/bin/python -m fisher_graph_compact_message_completion_probe_v2 \\
  --repository-root {repository_root} \\
  --output-dir {output_dir} \\
  --registry {registry_path}
```

The same command is the immutable verification command after finalization.
It verifies exact artifact bytes and the host-local retirement rows without
reopening the raw reserve.
"""


def _terminal_payload(
    *, classification: Mapping[str, object], decision: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.fisher_compact_message_completion.terminal_state.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "terminal_state": CURRENT_TERMINAL_STATE,
        "reason": TERMINAL_REASON,
        "reached_prefixes": ["P0"],
        "unreached_prefixes": [
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "P10",
            "P11",
            "P12",
        ],
        "corpus_role_classification_sha256": classification[
            "artifact_sha256"
        ],
        "preflight_decision_sha256": decision["artifact_sha256"],
        "successful_fail_closed_protocol_outcome": True,
        "scientific_result_present": False,
        "experiment_metrics_exist": False,
        "model_verdict_exists": False,
        "guard_verdict_exists": False,
        "authoritative_capture_performed": False,
        "campaign_identity_sealed": False,
    }


def _final_report_markdown(
    *,
    reserve: Mapping[str, object],
    registry_status: Mapping[str, object],
    classification: Mapping[str, object],
    decision: Mapping[str, object],
) -> str:
    return f"""# Fisher compact-message completion probe v2

## Outcome

The protocol completed its fail-closed corpus-provenance preflight.
Corpus D does not satisfy the frozen strict metadata-only eligibility policy because its authoritative provenance records semantic parsing and tokenization access.
With B and C already scientifically used, no eligible independent tune and development-confirm source remains.
The campaign therefore terminated before source sealing, capture, model execution, or scientific evaluation.

No raw D content was inspected by this campaign.

No experiment was run.

No model candidate was fitted.

No development or guard verdict exists.

The terminal state is
development_complete_independent_data_unavailable.

This is a successful fail-closed P0 protocol outcome. It is not evidence for
or against the compact-message completion hypothesis.

## Corpus disposition

- A is fit-only.
- B and C are ineligible due to prior scientific use.
- D is `preoutcome_executable_access` and is ineligible under
  `{D_ACCESS_POLICY}` because the frozen receipt records both semantic parse
  access and tokenizer access.
- No fallback to B or C was made.

No raw D content was opened for this classification. The already exposed
reserve was decoded only after semantic canonicalization was frozen; only its
sanitized hashes and local commitments were retained. Its aggregate state is
irreversibly `retired_exposed` in the host-local development registry.

## Boundary

`campaign_can_start = false`: {TERMINAL_REASON}.

The campaign stopped before independence selection, splits, model/carrier/
runtime/source sealing, execution-budget sealing, capture authorization,
capture, O1, representation fitting, model fitting, or guard activity.

The local registry is non-authoritative. It does not establish private-
evaluator freshness, cross-host exclusivity, or guard authority. A future D2
must begin as a new campaign and must not mutate this finalized output.

## Authenticated P0 references

- reserve retirement: `{reserve['artifact_sha256']}`
- local registry status: `{registry_status['artifact_sha256']}`
- corpus classification: `{classification['artifact_sha256']}`
- preflight decision: `{decision['artifact_sha256']}`
"""


def _artifact_inventory(
    output_dir: Path,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for name in _INVENTORIED_ARTIFACTS:
        encoded = _read_regular_bytes(output_dir / name, label=name)
        if not encoded:
            raise CompactMessageCompletionIntegrityError(
                f"reached P0 artifact {name} is empty"
            )
        inventory.append(
            {
                "path": name,
                "byte_count": len(encoded),
                "artifact_byte_sha256": _sha256_bytes(encoded),
            }
        )
    return inventory


def _final_manifest_payload(output_dir: Path) -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "final_experiment_manifest.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "terminal_state": CURRENT_TERMINAL_STATE,
        "terminal_reason": TERMINAL_REASON,
        "reached_prefixes": ["P0"],
        "artifact_write_order": list(P0_ARTIFACT_ORDER),
        "artifact_inventory": _artifact_inventory(output_dir),
        "self_included_in_artifact_inventory": False,
        "final_manifest_written_last": True,
        "successful_fail_closed_protocol_outcome": True,
        "scientific_result_present": False,
        "model_verdict_exists": False,
        "guard_verdict_exists": False,
        "future_d2_requires_new_campaign": True,
    }


def _validate_existing_prefix(
    output_dir: Path, *, create: bool = True
) -> tuple[str, ...]:
    if create:
        _ensure_directory(output_dir, label="campaign output directory")
    else:
        try:
            details = os.lstat(output_dir)
        except OSError as error:
            raise CompactMessageCompletionFinalizedError(
                "finalized campaign output directory is unavailable"
            ) from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise CompactMessageCompletionFinalizedError(
                "finalized campaign output must be a non-symlink directory"
            )
    removed_temporary = False
    for path in tuple(output_dir.iterdir()):
        if not (path.name.startswith(".") and ".tmp." in path.name):
            continue
        details = os.lstat(path)
        matching_artifacts = [
            name
            for name in P0_ARTIFACT_ORDER
            if path.name.startswith(f".{name}.tmp.")
            and len(path.name) > len(f".{name}.tmp.")
        ]
        if (
            len(matching_artifacts) != 1
            or stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
        ):
            raise CompactMessageCompletionFinalizedError(
                "campaign output contains an unsafe temporary artifact"
            )
        # mkstemp artifacts are private, same-directory, and not authoritative
        # until their hard-link publication succeeds.  A crashed writer either
        # left an uncommitted file or a redundant link beside the final file;
        # both are safe to unlink without replacing any reached artifact.
        if stat.S_IMODE(details.st_mode) & 0o077:
            raise CompactMessageCompletionFinalizedError(
                "campaign temporary artifact is not private"
            )
        try:
            path.unlink()
        except OSError as error:
            raise CompactMessageCompletionFinalizedError(
                "campaign temporary artifact could not be recovered safely"
            ) from error
        removed_temporary = True
    if removed_temporary:
        _fsync_directory(output_dir)

    names: list[str] = []
    for path in output_dir.iterdir():
        details = os.lstat(path)
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise CompactMessageCompletionFinalizedError(
                "campaign output contains a non-regular entry"
            )
        names.append(path.name)
    unknown = set(names) - set(P0_ARTIFACT_ORDER)
    if unknown:
        raise CompactMessageCompletionFinalizedError(
            "campaign output contains artifacts outside the reached P0 prefix"
        )
    present = tuple(name for name in P0_ARTIFACT_ORDER if name in names)
    if set(present) != set(P0_ARTIFACT_ORDER[: len(present)]):
        raise CompactMessageCompletionFinalizedError(
            "campaign output artifacts do not form a monotonic P0 prefix"
        )
    return present


def _load_hashed_artifact(path: Path) -> dict[str, object]:
    encoded = _read_regular_bytes(path, label=path.name)
    value = _decode_json_mapping(encoded, label=path.name)
    if encoded != _canonical_json_bytes(value) + b"\n":
        raise CompactMessageCompletionIntegrityError(
            f"{path.name} does not use exact canonical JSON bytes plus one newline"
        )
    _verify_self_hash(value, label=path.name)
    return value


def _validate_commitment_sequence(
    value: object, *, label: str
) -> list[str]:
    if not isinstance(value, list):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must be a sorted list"
        )
    result: list[str] = []
    for index, item in enumerate(value):
        _raw_sha256_digest(item, label=f"{label}[{index}]")
        assert isinstance(item, str)
        result.append(item)
    if result != sorted(result):
        raise CompactMessageCompletionIntegrityError(
            f"{label} must be sorted"
        )
    return result


def _validate_reserve_receipt(
    value: Mapping[str, object],
    *,
    canonicalization_artifact_sha256: str,
    legacy: bool | None = None,
) -> bool:
    observed_implementation = value.get("retirement_implementation_sha256")
    if legacy is None:
        legacy = observed_implementation == LEGACY_RETIREMENT_IMPLEMENTATION_SHA256
    expected_implementation = (
        LEGACY_RETIREMENT_IMPLEMENTATION_SHA256
        if legacy
        else _implementation_sha256()
    )
    expected_identity = {
        "schema": (
            "fisher_graph.fisher_compact_message_completion."
            "reserve_retired_exposed.v1"
        ),
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": CAMPAIGN_ID,
        "reserve_path": str(DEFAULT_ROLE_CORPORA["reserve_unopened"]),
        "reserve_file_sha256": PINNED_ROLE_FILE_SHA256S["reserve_unopened"],
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalization_artifact_sha256": canonicalization_artifact_sha256,
        "exposure_reason": _EXPOSURE_REASON,
        "exposure_timestamp_basis": (
            "first_local_retirement_record"
            if legacy
            else "durable_pre_decode_claim_commit"
        ),
        "retirement_implementation_sha256": expected_implementation,
        "template_lineage_commitment_basis": (
            _TEMPLATE_LINEAGE_COMMITMENT_BASIS
        ),
        "every_individual_content_permanently_guard_ineligible": True,
        "aggregate_content_set_permanently_guard_ineligible": True,
        "raw_content_in_receipt": False,
        "raw_identity_in_receipt": False,
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise CompactMessageCompletionIntegrityError(
                f"reserve retirement identity field {key} drifted"
            )
    expected_fields = {
        "schema",
        "format_version",
        "protocol_id",
        "campaign_id",
        "reserve_path",
        "reserve_file_sha256",
        "canonical_content_set_sha256",
        "canonical_per_item_sha256s",
        "canonical_per_item_commitments",
        "semantic_instance_commitments",
        "template_lineage_commitments",
        "template_id_commitments",
        "family_commitments",
        "template_lineage_commitment_basis",
        "canonicalization_version",
        "canonicalization_artifact_sha256",
        "exposure_reason",
        "exposure_timestamp",
        "exposure_timestamp_basis",
        "retirement_implementation_sha256",
        "irreversible_transition",
        "item_count",
        "every_individual_content_permanently_guard_ineligible",
        "aggregate_content_set_permanently_guard_ineligible",
        "raw_content_in_receipt",
        "raw_identity_in_receipt",
        "artifact_sha256",
    }
    if not legacy:
        expected_fields.update(
            {
                "pre_decode_claim_id",
                "pre_decode_claim_timestamp",
                "pre_decode_claim_state_at_publication",
                "receipt_published_before_registry_retirement",
            }
        )
    if set(value) != expected_fields:
        raise CompactMessageCompletionIntegrityError(
            "reserve retirement receipt field set drifted"
        )
    if value.get("irreversible_transition") != {
        "from": "unopened",
        "to": "retired_exposed",
        "future_allowed_states": [],
    }:
        raise CompactMessageCompletionIntegrityError(
            "reserve retirement transition is not irreversible"
        )
    exposure_timestamp = value.get("exposure_timestamp")
    if not isinstance(exposure_timestamp, str) or not exposure_timestamp.endswith("Z"):
        raise CompactMessageCompletionIntegrityError(
            "reserve exposure timestamp is not an authenticated UTC timestamp"
        )
    try:
        datetime.fromisoformat(exposure_timestamp.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise CompactMessageCompletionIntegrityError(
            "reserve exposure timestamp is invalid"
        ) from error
    if not legacy:
        claim_id = value.get("pre_decode_claim_id")
        _raw_sha256_digest(claim_id, label="pre_decode_claim_id")
        if value.get("pre_decode_claim_timestamp") != exposure_timestamp:
            raise CompactMessageCompletionIntegrityError(
                "reserve receipt claim timestamp differs from exposure timestamp"
            )
        if value.get("pre_decode_claim_state_at_publication") != "claimed":
            raise CompactMessageCompletionIntegrityError(
                "reserve receipt was not published from the claimed state"
            )
        if value.get("receipt_published_before_registry_retirement") is not True:
            raise CompactMessageCompletionIntegrityError(
                "reserve receipt does not commit to receipt-before-retirement order"
            )

    content_hashes = _validate_commitment_sequence(
        value.get("canonical_per_item_sha256s"),
        label="canonical_per_item_sha256s",
    )
    if len(content_hashes) != len(set(content_hashes)) or not content_hashes:
        raise CompactMessageCompletionIntegrityError(
            "reserve canonical content hashes must be nonempty and unique"
        )
    if value.get("item_count") != len(content_hashes):
        raise CompactMessageCompletionIntegrityError(
            "reserve item count differs from its canonical hash inventory"
        )
    if value.get("canonical_content_set_sha256") != content_set_sha256(
        content_hashes
    ):
        raise CompactMessageCompletionIntegrityError(
            "reserve aggregate does not satisfy the exact F19 commitment"
        )

    per_item = value.get("canonical_per_item_commitments")
    if not isinstance(per_item, list) or len(per_item) != len(content_hashes):
        raise CompactMessageCompletionIntegrityError(
            "reserve per-item commitment inventory is incomplete"
        )
    expected_fields = {
        "content_sha256",
        "analysis_unit_commitment",
        "template_lineage_commitment",
        "template_id_commitment",
        "semantic_instance_commitment",
        "family_commitment",
    }
    observed_content_hashes: list[str] = []
    derived_commitments = {
        "semantic_instance_commitments": [],
        "template_lineage_commitments": [],
        "template_id_commitments": [],
        "family_commitments": [],
    }
    for index, record in enumerate(per_item):
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise CompactMessageCompletionIntegrityError(
                f"reserve per-item commitment record {index} has schema drift"
            )
        content_sha256 = record.get("content_sha256")
        _raw_sha256_digest(
            content_sha256,
            label=f"canonical_per_item_commitments[{index}].content_sha256",
        )
        assert isinstance(content_sha256, str)
        observed_content_hashes.append(content_sha256)
        for field, destination in (
            ("semantic_instance_commitment", "semantic_instance_commitments"),
            ("template_lineage_commitment", "template_lineage_commitments"),
            ("template_id_commitment", "template_id_commitments"),
            ("family_commitment", "family_commitments"),
        ):
            commitment = record.get(field)
            if commitment is not None:
                _raw_sha256_digest(
                    commitment,
                    label=f"canonical_per_item_commitments[{index}].{field}",
                )
                assert isinstance(commitment, str)
                derived_commitments[destination].append(commitment)
        analysis = record.get("analysis_unit_commitment")
        if analysis != record.get("semantic_instance_commitment"):
            raise CompactMessageCompletionIntegrityError(
                "reserve analysis-unit commitment must bind its semantic instance"
            )
    if observed_content_hashes != content_hashes:
        raise CompactMessageCompletionIntegrityError(
            "reserve per-item commitment order differs from content-hash order"
        )
    for field, commitments in derived_commitments.items():
        if value.get(field) != sorted(commitments):
            raise CompactMessageCompletionIntegrityError(
                f"reserve aggregate commitment list {field} drifted"
            )
    return legacy


def _records_from_reserve_receipt(
    value: Mapping[str, object],
) -> list[dict[str, str | None]]:
    """Recover sanitized registry inputs; never opens or reconstructs raw text."""

    records = value.get("canonical_per_item_commitments")
    if not isinstance(records, list):
        raise CompactMessageCompletionIntegrityError(
            "reserve receipt has no sanitized per-item commitments"
        )
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _verify_registry_rows(
    *,
    registry_path: Path,
    reserve_receipt: Mapping[str, object],
    registry_status: Mapping[str, object],
) -> dict[str, object]:
    content_set_sha256 = reserve_receipt.get("canonical_content_set_sha256")
    item_hashes = reserve_receipt.get("canonical_per_item_sha256s")
    if (
        not isinstance(content_set_sha256, str)
        or len(content_set_sha256) != 64
        or not isinstance(item_hashes, list)
        or not item_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in item_hashes
        )
        or item_hashes != sorted(set(item_hashes))
    ):
        raise CompactMessageCompletionIntegrityError(
            "reserve retirement receipt has an invalid hash inventory"
        )
    try:
        details = os.lstat(registry_path)
    except OSError as error:
        raise CompactMessageCompletionIntegrityError(
            "local registry is unavailable"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise CompactMessageCompletionIntegrityError(
            "local registry is not a regular non-symlink file"
        )
    connection = sqlite3.connect(
        registry_path.as_uri() + "?mode=ro", uri=True, timeout=30.0
    )
    try:
        opened_details = os.lstat(registry_path)
        if (
            stat.S_ISLNK(opened_details.st_mode)
            or not stat.S_ISREG(opened_details.st_mode)
            or opened_details.st_dev != details.st_dev
            or opened_details.st_ino != details.st_ino
        ):
            raise CompactMessageCompletionIntegrityError(
                "local registry changed identity while being opened"
            )
        registry_version = _registry_version(connection)
        status_version = registry_status.get("registry_schema_version")
        if type(status_version) is not int or registry_version != status_version:
            raise CompactMessageCompletionIntegrityError(
                "local registry schema version differs from finalized status"
            )
        assert isinstance(registry_version, int)
        _validate_registry_schema(connection, version=registry_version)
        set_row = connection.execute(
            """
            SELECT state, item_count FROM content_sets
            WHERE content_set_sha256 = ?
            """,
            (content_set_sha256,),
        ).fetchone()
        if set_row != ("retired_exposed", len(item_hashes)):
            raise CompactMessageCompletionIntegrityError(
                "local registry aggregate retirement no longer matches the receipt"
            )
        retirement_row = connection.execute(
            """
            SELECT reserve_file_sha256, content_set_sha256,
                   canonicalization_version,
                   canonicalization_artifact_sha256, item_count,
                   exposure_timestamp, exposure_timestamp_basis,
                   retirement_implementation_sha256, transition_from,
                   transition_to
            FROM reserve_retirements WHERE reserve_path = ?
            """,
            (reserve_receipt.get("reserve_path"),),
        ).fetchone()
        expected_retirement = (
            reserve_receipt.get("reserve_file_sha256"),
            content_set_sha256,
            reserve_receipt.get("canonicalization_version"),
            reserve_receipt.get("canonicalization_artifact_sha256"),
            len(item_hashes),
            reserve_receipt.get("exposure_timestamp"),
            reserve_receipt.get("exposure_timestamp_basis"),
            reserve_receipt.get("retirement_implementation_sha256"),
            "unopened",
            "retired_exposed",
        )
        if retirement_row != expected_retirement:
            raise CompactMessageCompletionIntegrityError(
                "local registry reserve transition differs from the receipt"
            )
        if registry_version == REGISTRY_SCHEMA_VERSION:
            expected_receipt_file_sha256 = _sha256_bytes(
                _canonical_json_bytes(reserve_receipt) + b"\n"
            )
            claim_row = connection.execute(
                """
                SELECT campaign_id, claim_id, state, claim_timestamp,
                       expected_reserve_file_sha256,
                       canonicalization_artifact_sha256,
                       retirement_implementation_sha256,
                       receipt_artifact_sha256
                FROM reserve_decode_claims WHERE reserve_path = ?
                """,
                (reserve_receipt.get("reserve_path"),),
            ).fetchone()
            expected_claim = (
                CAMPAIGN_ID,
                reserve_receipt.get("pre_decode_claim_id"),
                "retired_exposed",
                reserve_receipt.get("pre_decode_claim_timestamp"),
                reserve_receipt.get("reserve_file_sha256"),
                reserve_receipt.get("canonicalization_artifact_sha256"),
                reserve_receipt.get("retirement_implementation_sha256"),
                expected_receipt_file_sha256,
            )
            if claim_row != expected_claim:
                raise CompactMessageCompletionIntegrityError(
                    "durable reserve decode claim differs from finalized receipt"
                )
        placeholders = ",".join("?" for _ in item_hashes)
        rows = connection.execute(
            f"""
            SELECT content_sha256, usage_role, campaign_id,
                   analysis_unit_mode_version, analysis_unit_commitment,
                   template_lineage_commitment, template_id_commitment,
                   semantic_instance_commitment, family_commitment,
                   first_use_timestamp, eligibility_state,
                   canonicalization_version, source_file_sha256
            FROM content_items
            WHERE content_sha256 IN ({placeholders})
            ORDER BY content_sha256
            """,
            tuple(item_hashes),
        ).fetchall()
        per_item = reserve_receipt.get("canonical_per_item_commitments")
        assert isinstance(per_item, list)
        expected_rows = []
        for record in per_item:
            assert isinstance(record, Mapping)
            expected_rows.append(
                (
                    record["content_sha256"],
                    "retired_exposed",
                    CAMPAIGN_ID,
                    "unselected_preflight_v1",
                    record["analysis_unit_commitment"],
                    record["template_lineage_commitment"],
                    record["template_id_commitment"],
                    record["semantic_instance_commitment"],
                    record["family_commitment"],
                    reserve_receipt["exposure_timestamp"],
                    "retired_exposed",
                    CANONICALIZATION_VERSION,
                    reserve_receipt["reserve_file_sha256"],
                )
            )
        if rows != expected_rows:
            raise CompactMessageCompletionIntegrityError(
                "not every reserve content hash remains retired_exposed"
            )
        corpus_rows = connection.execute(
            """
            SELECT corpus_id, campaign_id, protocol_role, usage_role,
                   relative_path, file_sha256, identity_state,
                   first_recorded_at
            FROM corpus_backfill WHERE campaign_id = ?
            ORDER BY protocol_role
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
        expected_corpus_rows = sorted(
            [
                (
                f"gemma3-chart-native-neighbor-fresh-panels-v1:{role}",
                CAMPAIGN_ID,
                role,
                _HISTORICAL_USAGE_ROLES[role],
                str(DEFAULT_ROLE_CORPORA[role]),
                PINNED_ROLE_FILE_SHA256S[role],
                (
                    "retired_exposed"
                    if role == "reserve_unopened"
                    else "historical_identity_unverifiable_guard_ineligible"
                ),
                reserve_receipt["exposure_timestamp"],
                )
                for role in ORDERED_ROLES
            ],
            key=lambda row: row[2],
        )
        if corpus_rows != expected_corpus_rows:
            raise CompactMessageCompletionIntegrityError(
                "local registry historical corpus backfill drifted"
            )
        expected_meta = {
            "schema_version": str(registry_version),
            "authority": "non_authoritative_host_local_development",
            f"canonicalization_version:{CANONICALIZATION_VERSION}": "frozen",
        }
        if registry_version == REGISTRY_SCHEMA_VERSION:
            expected_meta["schema_contract"] = "v2_exact_irreversible_guards"
        observed_meta = dict(
            connection.execute(
                "SELECT key, value FROM registry_meta"
            ).fetchall()
        )
        if (
            observed_meta != expected_meta
            if registry_version == REGISTRY_SCHEMA_VERSION
            else any(
                observed_meta.get(key) != value
                for key, value in expected_meta.items()
            )
        ):
            raise CompactMessageCompletionIntegrityError(
                "local registry metadata drifted"
            )
        snapshot = _registry_snapshot(
            connection, content_set_sha256=content_set_sha256
        )
    finally:
        connection.close()
    if registry_status.get("logical_snapshot") != snapshot:
        raise CompactMessageCompletionIntegrityError(
            "local registry logical snapshot differs from the finalized status"
        )
    return snapshot


def verify_finalized_p0(
    output_dir: Path | str,
    *,
    registry_path: Path | str,
    repository_root: Path | str = Path("."),
) -> dict[str, object]:
    """Verify an immutable finalized P0 without reopening any raw corpus."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise CompactMessageCompletionIntegrityError(
            "repository root must be an existing directory"
        )
    output = _resolve_under_root(root, Path(output_dir))
    registry = _resolve_under_root(root, Path(registry_path))
    _assert_registry_outside_output(registry, output)
    present = _validate_existing_prefix(output, create=False)
    if present != P0_ARTIFACT_ORDER:
        raise CompactMessageCompletionFinalizedError(
            "campaign output is not a complete finalized P0 prefix"
        )

    hashed_names = tuple(
        name for name in P0_ARTIFACT_ORDER if name.endswith(".json")
    )
    artifacts = {
        name: _load_hashed_artifact(output / name) for name in hashed_names
    }
    expected_semantic_spec = _self_hashed(_semantic_spec_payload())
    if artifacts["semantic_canonicalization_spec.json"] != expected_semantic_spec:
        raise CompactMessageCompletionIntegrityError(
            "finalized semantic canonicalization differs from the frozen P0 rule"
        )
    reserve_receipt = artifacts["reserve_retired_exposed.json"]
    legacy = _validate_reserve_receipt(
        reserve_receipt,
        canonicalization_artifact_sha256=str(
            expected_semantic_spec["artifact_sha256"]
        ),
    )
    registry_status = artifacts["local_development_registry_status.json"]
    expected_registry_version = (
        LEGACY_REGISTRY_SCHEMA_VERSION if legacy else REGISTRY_SCHEMA_VERSION
    )
    if registry_status.get("registry_schema_version") != expected_registry_version:
        raise CompactMessageCompletionIntegrityError(
            "reserve receipt implementation and registry schema generation disagree"
        )
    if registry_status.get("registry_location") != str(registry):
        raise CompactMessageCompletionIntegrityError(
            "finalized P0 is bound to a different local registry"
        )
    snapshot = _verify_registry_rows(
        registry_path=registry,
        reserve_receipt=reserve_receipt,
        registry_status=registry_status,
    )
    expected_registry_status = _self_hashed(
        _registry_status_payload(
            registry_path=registry,
            snapshot=snapshot,
            registry_schema_version=expected_registry_version,
        )
    )
    if registry_status != expected_registry_status:
        raise CompactMessageCompletionIntegrityError(
            "finalized local registry status differs from its authenticated rows"
        )

    receipt = load_receipt(root / DEFAULT_RECEIPT)
    classification_payload = (
        _legacy_classify_corpus_roles(receipt)
        if legacy
        else classify_corpus_roles(receipt)
    )
    expected_classification = _self_hashed(classification_payload)
    if artifacts["corpus_role_classification.json"] != expected_classification:
        raise CompactMessageCompletionIntegrityError(
            "finalized corpus classification differs from frozen receipt metadata"
        )
    expected_decision = _self_hashed(
        build_preflight_decision(expected_classification)
    )
    if artifacts["preflight_decision.json"] != expected_decision:
        raise CompactMessageCompletionIntegrityError(
            "finalized preflight decision differs from the frozen classification"
        )
    expected_commands = _commands_markdown(
        repository_root=root,
        output_dir=output,
        registry_path=registry,
    ).encode("utf-8")
    if _read_regular_bytes(output / "commands.md", label="commands.md") != (
        expected_commands
    ):
        raise CompactMessageCompletionIntegrityError(
            "finalized commands artifact differs from the exact P0 invocation"
        )
    expected_terminal = _self_hashed(
        _terminal_payload(
            classification=expected_classification,
            decision=expected_decision,
        )
    )
    if artifacts["terminal_state.json"] != expected_terminal:
        raise CompactMessageCompletionIntegrityError(
            "finalized terminal artifact differs from the frozen P0 outcome"
        )
    expected_report = _final_report_markdown(
        reserve=reserve_receipt,
        registry_status=expected_registry_status,
        classification=expected_classification,
        decision=expected_decision,
    ).encode("utf-8")
    if _read_regular_bytes(
        output / "final_report.md", label="final_report.md"
    ) != expected_report:
        raise CompactMessageCompletionIntegrityError(
            "finalized report differs from the mandated P0 report"
        )
    manifest = artifacts[_FINAL_MANIFEST_NAME]
    expected_manifest = _self_hashed(_final_manifest_payload(output))
    if manifest != expected_manifest:
        raise CompactMessageCompletionIntegrityError(
            "final manifest differs from exact reached P0 bytes or semantics"
        )
    return {
        "terminal_state": CURRENT_TERMINAL_STATE,
        "output_dir": str(output),
        "registry_path": str(registry),
        "final_experiment_manifest_sha256": manifest["artifact_sha256"],
        "verified": True,
        "raw_reserve_reopened": False,
        "contract_conformant": not legacy,
        "legacy_verification": legacy,
        "verification_mode": (
            "legacy_v1_byte_audit_nonconformant"
            if legacy
            else "contract_v2_conformant"
        ),
        "contract_invalidation_reason": (
            LEGACY_CONTRACT_INVALIDATION_REASON if legacy else None
        ),
    }


def run_current_p0(
    *,
    repository_root: Path | str = Path("."),
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    registry_path: Path | str = DEFAULT_REGISTRY_PATH,
) -> dict[str, object]:
    """Materialize or exactly verify the current nine-artifact P0 prefix."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise CompactMessageCompletionIntegrityError(
            "repository root must be an existing directory"
        )
    output = _resolve_under_root(root, Path(output_dir))
    registry = _resolve_under_root(root, Path(registry_path))
    _assert_registry_outside_output(registry, output)
    present = _validate_existing_prefix(output)
    if _FINAL_MANIFEST_NAME in present:
        # An immutable rerun verifies the exact finalized state and never
        # reopens the raw reserve.
        return verify_finalized_p0(
            output, registry_path=registry, repository_root=root
        )

    semantic_spec = _self_hashed(_semantic_spec_payload())
    _atomic_exclusive_write(
        output / P0_ARTIFACT_ORDER[0],
        _canonical_json_bytes(semantic_spec) + b"\n",
    )

    reserve_receipt_path = output / P0_ARTIFACT_ORDER[1]
    if P0_ARTIFACT_ORDER[1] in present:
        # Artifact #2 is the recovery boundary.  Every value required by the
        # registry is a sanitized commitment in this receipt; raw reserve bytes
        # are neither needed nor permitted on this path.
        reserve_receipt = _load_hashed_artifact(reserve_receipt_path)
        legacy = _validate_reserve_receipt(
            reserve_receipt,
            canonicalization_artifact_sha256=str(
                semantic_spec["artifact_sha256"]
            ),
        )
        if legacy:
            raise CompactMessageCompletionFinalizedError(
                "a partial legacy wrong-order P0 cannot be resumed; it is audit-only"
            )
        reserve_records = _records_from_reserve_receipt(reserve_receipt)
    else:
        # The real reserve's global one-time mechanical authorization is
        # already exhausted.  Check its immutable configured identity before
        # creating a caller-selected registry claim.  Synthetic qualification
        # uses a distinct pin and still exercises the full transaction.
        _require_reserve_decode_authorization_available()
        implementation_sha256 = _implementation_sha256()
        claim = _acquire_reserve_decode_claim(
            registry_path=registry,
            expected_reserve_file_sha256=(
                PINNED_ROLE_FILE_SHA256S["reserve_unopened"]
            ),
            canonicalization_artifact_sha256=str(
                semantic_spec["artifact_sha256"]
            ),
            implementation_sha256=implementation_sha256,
        )
        # This is the sole raw-data operation.  The canonicalization artifact
        # and durable at-most-once claim both precede it.  If execution becomes
        # uncertain before the sanitized receipt is published, retries fail
        # closed instead of decoding a second time.
        reserve_file_sha256, reserve_records, content_set_sha256 = (
            _canonicalize_reserve(root)
        )
        retirement_payload = _reserve_retirement_payload(
            reserve_file_sha256=reserve_file_sha256,
            records=reserve_records,
            content_set_sha256=content_set_sha256,
            canonicalization_artifact_sha256=str(
                semantic_spec["artifact_sha256"]
            ),
            implementation_sha256=implementation_sha256,
            claim_id=claim["claim_id"],
            claim_timestamp=claim["claim_timestamp"],
        )
        reserve_receipt = _self_hashed(
            {
                "schema": (
                    "fisher_graph.fisher_compact_message_completion."
                    "reserve_retired_exposed.v1"
                ),
                "format_version": FORMAT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "campaign_id": CAMPAIGN_ID,
                **retirement_payload,
            }
        )
        _atomic_exclusive_write(
            reserve_receipt_path,
            _canonical_json_bytes(reserve_receipt) + b"\n",
        )
        _bind_published_reserve_receipt(
            registry_path=registry,
            receipt_path=reserve_receipt_path,
            reserve_receipt=reserve_receipt,
        )

    registry_snapshot = _retire_reserve_in_local_registry(
        registry_path=registry,
        receipt_path=reserve_receipt_path,
        reserve_receipt=reserve_receipt,
        records=reserve_records,
    )

    registry_status = _self_hashed(
        _registry_status_payload(
            registry_path=registry,
            snapshot=registry_snapshot,
            registry_schema_version=REGISTRY_SCHEMA_VERSION,
        )
    )
    _atomic_exclusive_write(
        output / P0_ARTIFACT_ORDER[2],
        _canonical_json_bytes(registry_status) + b"\n",
    )

    # D eligibility reads exact nested Booleans from the frozen prompt-free
    # receipt.  No D role file, manifest, tokenizer, or model is opened.
    receipt = load_receipt(root / DEFAULT_RECEIPT)
    classification = _self_hashed(classify_corpus_roles(receipt))
    _atomic_exclusive_write(
        output / P0_ARTIFACT_ORDER[3],
        _canonical_json_bytes(classification) + b"\n",
    )

    decision = _self_hashed(build_preflight_decision(classification))
    _atomic_exclusive_write(
        output / P0_ARTIFACT_ORDER[4],
        _canonical_json_bytes(decision) + b"\n",
    )
    _write_text_artifact(
        output / P0_ARTIFACT_ORDER[5],
        _commands_markdown(
            repository_root=root,
            output_dir=output,
            registry_path=registry,
        ),
    )

    terminal = _self_hashed(
        _terminal_payload(classification=classification, decision=decision)
    )
    _atomic_exclusive_write(
        output / P0_ARTIFACT_ORDER[6],
        _canonical_json_bytes(terminal) + b"\n",
    )
    _write_text_artifact(
        output / P0_ARTIFACT_ORDER[7],
        _final_report_markdown(
            reserve=reserve_receipt,
            registry_status=registry_status,
            classification=classification,
            decision=decision,
        ),
    )

    final_manifest = _final_manifest_payload(output)
    _write_json_artifact(output / _FINAL_MANIFEST_NAME, final_manifest)

    return verify_finalized_p0(
        output, registry_path=registry, repository_root=root
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(".")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args(argv)
    result = run_current_p0(
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        registry_path=args.registry,
    )
    print(
        json.dumps(
            {
                "terminal_state": result["terminal_state"],
                "verified": result["verified"],
                "contract_conformant": result["contract_conformant"],
                "legacy_verification": result["legacy_verification"],
                "verification_mode": result["verification_mode"],
                "contract_invalidation_reason": result[
                    "contract_invalidation_reason"
                ],
                "output_dir": result["output_dir"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CAMPAIGN_ID",
    "CANONICALIZATION_VERSION",
    "CURRENT_TERMINAL_STATE",
    "CompactMessageCompletionFinalizedError",
    "CompactMessageCompletionIntegrityError",
    "CompactMessageCompletionProtocolError",
    "D_ACCESS_POLICY",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REGISTRY_PATH",
    "EXHAUSTED_REAL_RESERVE_FILE_SHA256",
    "P0_ARTIFACT_ORDER",
    "PROTOCOL_ID",
    "LEGACY_CONTRACT_INVALIDATION_REASON",
    "LEGACY_RETIREMENT_IMPLEMENTATION_SHA256",
    "LEGACY_REGISTRY_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "build_preflight_decision",
    "canonical_content_sha256",
    "canonical_semantic_prompt_bytes",
    "classify_corpus_roles",
    "content_set_sha256",
    "main",
    "run_current_p0",
    "verify_finalized_p0",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
