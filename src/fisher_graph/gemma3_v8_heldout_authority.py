"""Claim-first data airlock for the untouched structured-strong-v8 roles.

The v8 prompt source stores Calibration-B, validation, and test in one JSON
file.  Model workers must not receive that aggregate file.  This module first
builds a prompt-free manifest from the audited prompt hashes and family file,
durably claims the exact manifest for one frozen challenger, and only then
allows a data-only process to export the claimed role into a standalone file.

The per-user ledger prevents accidental reuse across local worktrees on one
host.  It is not a cross-machine consensus service or hostile-process
attestation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat


__all__ = [
    "GEMMA3_V8_HELDOUT_ROLES",
    "Gemma3V8HeldoutAlreadyClaimedError",
    "Gemma3V8HeldoutClaimReceipt",
    "Gemma3V8HeldoutForeignClaimError",
    "Gemma3V8HeldoutIntegrityError",
    "Gemma3V8HeldoutManifest",
    "claim_gemma3_v8_heldout_role",
    "export_claimed_gemma3_v8_role",
    "load_claimed_gemma3_v8_role",
    "load_gemma3_v8_heldout_claim",
    "load_gemma3_v8_heldout_manifest",
]


GEMMA3_V8_HELDOUT_ROLES = ("calibration_b", "validation", "test")

_MANIFEST_SCHEMA = "fisher_graph.gemma3_v8_heldout_manifest"
_CLAIM_SCHEMA = "fisher_graph.gemma3_v8_heldout_claim"
_EXPORT_SCHEMA = "fisher_graph.gemma3_v8_claimed_role_export"
_FORMAT_VERSION = 1
_CORPUS_ID = "structured-strong-v8"
_MANIFEST_DOMAIN = b"fisher-graph:gemma3-v8-heldout-manifest:v1\0"
_CLAIM_DOMAIN = b"fisher-graph:gemma3-v8-heldout-claim:v1\0"
_EXPORT_DOMAIN = b"fisher-graph:gemma3-v8-heldout-export:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_LEDGER_FILE_BYTES = 1 << 20
_FROZEN_LEDGER_ROOT = (
    Path.home()
    / ".local"
    / "state"
    / "fisher-graph"
    / "gemma3-v8-heldout-roles"
)

_V8_AUDIT_FILE_SHA256 = (
    "f34735a53b0e9eb6b0471ec9c73613ef64c39ca7aeb7006f80845eb8bef70988"
)
_V8_FAMILY_FILE_SHA256 = (
    "01654b2ded4ca0cf713c2d334f41faacdd336d13776ecdb338c392d50dbdd703"
)
_V8_PROMPT_FILE_SHA256 = (
    "81a5431cc8b84f66686bdf46e7861dc431bcf2d7e9bc39781d76b00a5f60b66a"
)
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
_SOURCE_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
}


class Gemma3V8HeldoutIntegrityError(RuntimeError):
    """A v8 heldout source, claim, or export failed authentication."""


class Gemma3V8HeldoutAlreadyClaimedError(FileExistsError):
    """The manifest-global heldout role has already been consumed."""


class Gemma3V8HeldoutForeignClaimError(
    Gemma3V8HeldoutAlreadyClaimedError
):
    """The role was claimed by a different protocol or challenger."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256_bytes(domain + _canonical_json_bytes(value))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_role(value: object) -> str:
    if value not in GEMMA3_V8_HELDOUT_ROLES:
        raise ValueError("role must be calibration_b, validation, or test")
    return str(value)


def _read_exact_bytes(
    path: Path | str,
    *,
    expected_sha256: str,
    label: str,
) -> bytes:
    source = Path(path)
    if not source.is_file():
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = source.read_bytes()
    if _sha256_bytes(encoded) != expected_sha256:
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} exact file SHA-256 mismatch"
        )
    return encoded


def _decode_object(encoded: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} must contain one JSON object"
        )
    return value


def _string_sequence(
    value: object,
    *,
    label: str,
    count: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in value
        )
    ):
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} must contain exactly {count} trimmed strings"
        )
    return tuple(value)


def _sha256_sequence(
    value: object,
    *,
    label: str,
    count: int,
) -> tuple[str, ...]:
    result = _string_sequence(value, label=label, count=count)
    if any(_SHA256.fullmatch(item) is None for item in result):
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} contains a malformed SHA-256"
        )
    if len(set(result)) != len(result):
        raise Gemma3V8HeldoutIntegrityError(
            f"{label} contains duplicate identities"
        )
    return result


@dataclass(frozen=True, slots=True)
class Gemma3V8HeldoutManifest:
    """Prompt-free exact identity for one untouched v8 role."""

    role: str
    prompt_sha256s: tuple[str, ...]
    normalized_prompt_sha256s: tuple[str, ...]
    family_ids: tuple[str, ...]
    length_bands: tuple[str, ...]
    approximate_word_counts: tuple[int, ...]
    declared_policy: str | None
    artifact_sha256: str = ""
    schema: str = _MANIFEST_SCHEMA
    format_version: int = _FORMAT_VERSION
    corpus_id: str = _CORPUS_ID
    prompt_source_file_sha256: str = _V8_PROMPT_FILE_SHA256
    family_source_file_sha256: str = _V8_FAMILY_FILE_SHA256
    audit_source_file_sha256: str = _V8_AUDIT_FILE_SHA256

    def __post_init__(self) -> None:
        role = _require_role(self.role)
        expected = _EXPECTED_COUNTS[role]
        if (
            self.schema != _MANIFEST_SCHEMA
            or self.format_version != _FORMAT_VERSION
            or self.corpus_id != _CORPUS_ID
        ):
            raise ValueError("v8 heldout manifest header is invalid")
        for name, value in (
            ("prompt source", self.prompt_source_file_sha256),
            ("family source", self.family_source_file_sha256),
            ("audit source", self.audit_source_file_sha256),
        ):
            _require_sha256(value, label=name)
        if (
            type(self.prompt_sha256s) is not tuple
            or len(self.prompt_sha256s) != expected
            or len(set(self.prompt_sha256s)) != expected
            or any(_SHA256.fullmatch(value) is None for value in self.prompt_sha256s)
            or type(self.normalized_prompt_sha256s) is not tuple
            or len(self.normalized_prompt_sha256s) != expected
            or len(set(self.normalized_prompt_sha256s)) != expected
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.normalized_prompt_sha256s
            )
            or type(self.family_ids) is not tuple
            or len(self.family_ids) != expected
            or len(set(self.family_ids)) != _EXPECTED_FAMILY_COUNTS[role]
            or any(not value for value in self.family_ids)
            or type(self.length_bands) is not tuple
            or len(self.length_bands) != expected
            or any(
                value not in {"micro", "compact", "medium", "long"}
                for value in self.length_bands
            )
            or type(self.approximate_word_counts) is not tuple
            or len(self.approximate_word_counts) != expected
            or any(
                type(value) is not int or value <= 0
                for value in self.approximate_word_counts
            )
            or (
                self.declared_policy is not None
                and (
                    not isinstance(self.declared_policy, str)
                    or not self.declared_policy
                )
            )
        ):
            raise ValueError("v8 heldout manifest columns are invalid")
        computed = _domain_sha256(_MANIFEST_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256,
            label="manifest artifact",
        ) != computed:
            raise ValueError("v8 heldout manifest hash mismatch")

    @property
    def example_count(self) -> int:
        return len(self.prompt_sha256s)

    @property
    def family_count(self) -> int:
        return len(set(self.family_ids))

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "corpus_id": self.corpus_id,
            "role": self.role,
            "prompt_source_file_sha256": self.prompt_source_file_sha256,
            "family_source_file_sha256": self.family_source_file_sha256,
            "audit_source_file_sha256": self.audit_source_file_sha256,
            "prompt_sha256s": self.prompt_sha256s,
            "normalized_prompt_sha256s": self.normalized_prompt_sha256s,
            "family_ids": self.family_ids,
            "length_bands": self.length_bands,
            "approximate_word_counts": self.approximate_word_counts,
            "declared_policy": self.declared_policy,
            "example_count": self.example_count,
            "family_count": self.family_count,
            "prompt_text_materialized": False,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def validate_integrity(self) -> None:
        if _domain_sha256(_MANIFEST_DOMAIN, self._payload()) != self.artifact_sha256:
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout manifest hash mismatch"
            )


def load_gemma3_v8_heldout_manifest(
    role: str,
    *,
    audit_path: Path | str,
    family_path: Path | str,
) -> Gemma3V8HeldoutManifest:
    """Build a role manifest without opening the aggregate prompt file."""

    selected = _require_role(role)
    audit = _decode_object(
        _read_exact_bytes(
            audit_path,
            expected_sha256=_V8_AUDIT_FILE_SHA256,
            label="v8 corpus audit",
        ),
        label="v8 corpus audit",
    )
    families = _decode_object(
        _read_exact_bytes(
            family_path,
            expected_sha256=_V8_FAMILY_FILE_SHA256,
            label="v8 family source",
        ),
        label="v8 family source",
    )
    if (
        audit.get("schema") != "fisher_graph.structured_strong_corpus_audit"
        or audit.get("format_version") != 4
        or audit.get("corpus_id") != _CORPUS_ID
        or audit.get("counts") != _EXPECTED_COUNTS
        or audit.get("families_per_role") != _EXPECTED_FAMILY_COUNTS
        or audit.get("prompt_file_sha256") != _V8_PROMPT_FILE_SHA256
        or audit.get("family_file_sha256") != _V8_FAMILY_FILE_SHA256
        or audit.get("heldout_splits_evaluated") is not False
        or audit.get("heldout_splits_tokenized") is not False
        or audit.get("heldout_splits_unevaluated") is not True
        or audit.get("heldout_splits_untokenized") is not True
        or audit.get("calibration_b_model_evaluated") is not False
        or audit.get("validation_model_evaluated") is not False
        or audit.get("test_model_evaluated") is not False
        or audit.get("tokenizer_or_model_accessed") is not False
        or audit.get("cross_role_family_overlap_count") != 0
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 corpus audit heldout contract is invalid"
        )
    if (
        set(families) != _SOURCE_FIELDS
        or families.get("schema")
        != "fisher_graph.gemma3_prompt_family_manifest"
        or families.get("format_version") != 1
        or families.get("scientific_status")
        != "full_width_single_layer_family_disjoint_roles"
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 family source header is invalid"
        )
    count = _EXPECTED_COUNTS[selected]
    prompt_hashes_by_role = audit.get("prompt_sha256_by_role")
    normalized_hashes_by_role = audit.get("normalized_prompt_sha256_by_role")
    length_bands_by_role = audit.get("length_band_by_role")
    word_counts_by_role = audit.get("approximate_word_counts_by_role")
    if not all(
        isinstance(value, Mapping)
        for value in (
            prompt_hashes_by_role,
            normalized_hashes_by_role,
            length_bands_by_role,
            word_counts_by_role,
        )
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 audit role columns are unavailable"
        )
    role_families = _string_sequence(
        families.get(selected),
        label=f"v8 {selected} families",
        count=count,
    )
    policy = (
        audit.get("calibration_b_policy")
        if selected == "calibration_b"
        else audit.get("test_policy") if selected == "test" else None
    )
    if policy is not None and not isinstance(policy, str):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout policy must be a string or null"
        )
    return Gemma3V8HeldoutManifest(
        role=selected,
        prompt_sha256s=_sha256_sequence(
            prompt_hashes_by_role.get(selected),
            label=f"v8 {selected} prompt hashes",
            count=count,
        ),
        normalized_prompt_sha256s=_sha256_sequence(
            normalized_hashes_by_role.get(selected),
            label=f"v8 {selected} normalized prompt hashes",
            count=count,
        ),
        family_ids=role_families,
        length_bands=_string_sequence(
            length_bands_by_role.get(selected),
            label=f"v8 {selected} length bands",
            count=count,
        ),
        approximate_word_counts=tuple(
            word_counts_by_role.get(selected)  # type: ignore[arg-type]
        ),
        declared_policy=policy,
        prompt_source_file_sha256=_V8_PROMPT_FILE_SHA256,
        family_source_file_sha256=_V8_FAMILY_FILE_SHA256,
        audit_source_file_sha256=_V8_AUDIT_FILE_SHA256,
    )


def _claim_path(manifest_sha256: str) -> Path:
    manifest = _require_sha256(manifest_sha256, label="heldout manifest")
    return _FROZEN_LEDGER_ROOT / f"{manifest}.claim.json"


def _ensure_private_root() -> None:
    try:
        _FROZEN_LEDGER_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = os.lstat(_FROZEN_LEDGER_ROOT)
    except OSError as error:
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout ledger namespace cannot be authenticated"
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout ledger must be an owner-only directory 0700"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_private_write(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("v8 heldout claim write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_private_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size <= 0
            or details.st_size > _MAX_LEDGER_FILE_BYTES
        ):
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout claim must be a private regular file 0600"
            )
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if remaining or os.read(descriptor, 1):
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout claim changed while it was read"
            )
        return encoded
    finally:
        os.close(descriptor)


def _claim_payload(
    *,
    manifest: Gemma3V8HeldoutManifest,
    protocol_sha256: str,
    challenger_receipt_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": _CLAIM_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "corpus_id": _CORPUS_ID,
        "role": manifest.role,
        "state": "claimed_before_aggregate_prompt_source_open",
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            label="heldout protocol",
        ),
        "manifest_sha256": manifest.artifact_sha256,
        "challenger_receipt_sha256": _require_sha256(
            challenger_receipt_sha256,
            label="heldout challenger receipt",
        ),
        "global_lock_identity": "manifest_sha256",
    }
    return {
        **payload,
        "claim_sha256": _domain_sha256(_CLAIM_DOMAIN, payload),
    }


def _decode_claim(encoded: bytes) -> dict[str, object]:
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout claim is not canonical ASCII JSON"
        ) from error
    expected = {
        "schema",
        "format_version",
        "corpus_id",
        "role",
        "state",
        "protocol_sha256",
        "manifest_sha256",
        "challenger_receipt_sha256",
        "global_lock_identity",
        "claim_sha256",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected
        or _canonical_json_bytes(raw) != encoded
        or raw.get("schema") != _CLAIM_SCHEMA
        or raw.get("format_version") != _FORMAT_VERSION
        or raw.get("corpus_id") != _CORPUS_ID
        or raw.get("state")
        != "claimed_before_aggregate_prompt_source_open"
        or raw.get("global_lock_identity") != "manifest_sha256"
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout claim fields are invalid"
        )
    _require_role(raw.get("role"))
    for name in (
        "protocol_sha256",
        "manifest_sha256",
        "challenger_receipt_sha256",
        "claim_sha256",
    ):
        try:
            _require_sha256(raw.get(name), label=f"heldout claim {name}")
        except ValueError as error:
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout claim contains a malformed SHA-256"
            ) from error
    payload = dict(raw)
    observed = payload.pop("claim_sha256")
    if observed != _domain_sha256(_CLAIM_DOMAIN, payload):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 heldout claim hash mismatch"
        )
    return raw


@dataclass(frozen=True, slots=True)
class Gemma3V8HeldoutClaimReceipt:
    path: Path
    role: str
    protocol_sha256: str
    manifest_sha256: str
    challenger_receipt_sha256: str
    claim_sha256: str
    claim_file_sha256: str
    _claim_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_role(self.role)
        for name in (
            "protocol_sha256",
            "manifest_sha256",
            "challenger_receipt_sha256",
            "claim_sha256",
            "claim_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.path, Path) or not self._claim_bytes:
            raise ValueError("v8 heldout claim receipt is incomplete")

    def validate_integrity(self) -> None:
        if self.path != _claim_path(self.manifest_sha256):
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout claim path differs from its manifest identity"
            )
        _ensure_private_root()
        encoded = _read_private_file(self.path)
        raw = _decode_claim(encoded)
        if (
            encoded != self._claim_bytes
            or _sha256_bytes(encoded) != self.claim_file_sha256
            or raw.get("role") != self.role
            or raw.get("protocol_sha256") != self.protocol_sha256
            or raw.get("manifest_sha256") != self.manifest_sha256
            or raw.get("challenger_receipt_sha256")
            != self.challenger_receipt_sha256
            or raw.get("claim_sha256") != self.claim_sha256
        ):
            raise Gemma3V8HeldoutIntegrityError(
                "v8 heldout claim differs from its immutable receipt"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "protocol_sha256": self.protocol_sha256,
            "manifest_sha256": self.manifest_sha256,
            "challenger_receipt_sha256": self.challenger_receipt_sha256,
            "claim_sha256": self.claim_sha256,
            "claim_file_sha256": self.claim_file_sha256,
            "state": "claimed_before_aggregate_prompt_source_open",
        }


def _receipt(path: Path, encoded: bytes) -> Gemma3V8HeldoutClaimReceipt:
    raw = _decode_claim(encoded)
    return Gemma3V8HeldoutClaimReceipt(
        path=path,
        role=str(raw["role"]),
        protocol_sha256=str(raw["protocol_sha256"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        challenger_receipt_sha256=str(raw["challenger_receipt_sha256"]),
        claim_sha256=str(raw["claim_sha256"]),
        claim_file_sha256=_sha256_bytes(encoded),
        _claim_bytes=encoded,
    )


def claim_gemma3_v8_heldout_role(
    manifest: Gemma3V8HeldoutManifest,
    *,
    protocol_sha256: str,
    challenger_receipt_sha256: str,
) -> Gemma3V8HeldoutClaimReceipt:
    manifest.validate_integrity()
    payload = _claim_payload(
        manifest=manifest,
        protocol_sha256=protocol_sha256,
        challenger_receipt_sha256=challenger_receipt_sha256,
    )
    encoded = _canonical_json_bytes(payload)
    _ensure_private_root()
    path = _claim_path(manifest.artifact_sha256)
    try:
        _exclusive_private_write(path, encoded)
    except FileExistsError as error:
        existing = _read_private_file(path)
        raw = _decode_claim(existing)
        if existing != encoded:
            raise Gemma3V8HeldoutForeignClaimError(
                "v8 heldout role was claimed by another challenger"
            ) from error
        raise Gemma3V8HeldoutAlreadyClaimedError(
            "v8 heldout role was already claimed"
        ) from error
    receipt = _receipt(path, encoded)
    receipt.validate_integrity()
    return receipt


def load_gemma3_v8_heldout_claim(
    manifest: Gemma3V8HeldoutManifest,
    *,
    protocol_sha256: str,
    challenger_receipt_sha256: str,
) -> Gemma3V8HeldoutClaimReceipt:
    manifest.validate_integrity()
    path = _claim_path(manifest.artifact_sha256)
    _ensure_private_root()
    receipt = _receipt(path, _read_private_file(path))
    expected = _claim_payload(
        manifest=manifest,
        protocol_sha256=protocol_sha256,
        challenger_receipt_sha256=challenger_receipt_sha256,
    )
    if _canonical_json_bytes(expected) != receipt._claim_bytes:
        raise Gemma3V8HeldoutForeignClaimError(
            "v8 heldout claim belongs to another protocol or challenger"
        )
    receipt.validate_integrity()
    return receipt


def _export_payload(
    *,
    manifest: Gemma3V8HeldoutManifest,
    claim: Gemma3V8HeldoutClaimReceipt,
    prompts: Sequence[str],
) -> dict[str, object]:
    examples = tuple(
        {
            "example_id": f"v8.{manifest.role}.{index:03d}",
            "prompt": prompt,
            "prompt_sha256": manifest.prompt_sha256s[index],
            "family_id": manifest.family_ids[index],
            "length_band": manifest.length_bands[index],
            "approximate_word_count": manifest.approximate_word_counts[index],
        }
        for index, prompt in enumerate(prompts)
    )
    payload: dict[str, object] = {
        "schema": _EXPORT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "corpus_id": _CORPUS_ID,
        "role": manifest.role,
        "manifest_sha256": manifest.artifact_sha256,
        "claim": claim.metadata(),
        "prompt_source_file_sha256": manifest.prompt_source_file_sha256,
        "family_source_file_sha256": manifest.family_source_file_sha256,
        "audit_source_file_sha256": manifest.audit_source_file_sha256,
        "example_count": manifest.example_count,
        "family_count": manifest.family_count,
        "examples": examples,
        "safety": {
            "claimed_before_aggregate_prompt_source_open": True,
            "model_accessed": False,
            "tokenizer_accessed": False,
            "other_heldout_roles_exported": False,
            "contains_exactly_one_claimed_role": True,
        },
    }
    return {
        **payload,
        "export_sha256": _domain_sha256(_EXPORT_DOMAIN, payload),
    }


def export_claimed_gemma3_v8_role(
    manifest: Gemma3V8HeldoutManifest,
    claim: Gemma3V8HeldoutClaimReceipt,
    *,
    prompt_path: Path | str,
    output: Path | str,
) -> dict[str, object]:
    """Export one role only after the durable manifest claim validates."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite claimed v8 role export")
    manifest.validate_integrity()
    claim.validate_integrity()
    if (
        claim.role != manifest.role
        or claim.manifest_sha256 != manifest.artifact_sha256
    ):
        raise Gemma3V8HeldoutForeignClaimError(
            "v8 heldout claim does not authorize this manifest"
        )

    # This is deliberately the first access to the aggregate prompt source.
    prompts_source = _decode_object(
        _read_exact_bytes(
            prompt_path,
            expected_sha256=manifest.prompt_source_file_sha256,
            label="v8 aggregate prompt source",
        ),
        label="v8 aggregate prompt source",
    )
    if (
        set(prompts_source) != _SOURCE_FIELDS
        or prompts_source.get("schema") != "fisher_graph.gemma3_prompt_splits"
        or prompts_source.get("format_version") != 1
        or prompts_source.get("scientific_status")
        != "full_width_single_layer_fresh_a_b_validation_test_hash_only"
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 aggregate prompt source header is invalid"
        )
    prompts = _string_sequence(
        prompts_source.get(manifest.role),
        label=f"v8 {manifest.role} prompts",
        count=manifest.example_count,
    )
    observed_hashes = tuple(
        _sha256_bytes(value.encode("utf-8")) for value in prompts
    )
    if observed_hashes != manifest.prompt_sha256s:
        raise Gemma3V8HeldoutIntegrityError(
            "v8 claimed role prompts differ from the prompt-free manifest"
        )
    payload = _export_payload(
        manifest=manifest,
        claim=claim,
        prompts=prompts,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    _exclusive_private_write(destination, encoded)
    load_claimed_gemma3_v8_role(
        destination,
        manifest=manifest,
        claim=claim,
    )
    return payload


def load_claimed_gemma3_v8_role(
    path: Path | str,
    *,
    manifest: Gemma3V8HeldoutManifest,
    claim: Gemma3V8HeldoutClaimReceipt,
) -> tuple[dict[str, object], ...]:
    manifest.validate_integrity()
    claim.validate_integrity()
    raw = _decode_object(Path(path).read_bytes(), label="v8 claimed role export")
    expected_fields = {
        "schema",
        "format_version",
        "corpus_id",
        "role",
        "manifest_sha256",
        "claim",
        "prompt_source_file_sha256",
        "family_source_file_sha256",
        "audit_source_file_sha256",
        "example_count",
        "family_count",
        "examples",
        "safety",
        "export_sha256",
    }
    digest_payload = {
        key: value for key, value in raw.items() if key != "export_sha256"
    }
    if (
        set(raw) != expected_fields
        or raw.get("schema") != _EXPORT_SCHEMA
        or raw.get("format_version") != _FORMAT_VERSION
        or raw.get("corpus_id") != _CORPUS_ID
        or raw.get("role") != manifest.role
        or raw.get("manifest_sha256") != manifest.artifact_sha256
        or raw.get("claim") != claim.metadata()
        or raw.get("prompt_source_file_sha256")
        != manifest.prompt_source_file_sha256
        or raw.get("family_source_file_sha256")
        != manifest.family_source_file_sha256
        or raw.get("audit_source_file_sha256")
        != manifest.audit_source_file_sha256
        or raw.get("example_count") != manifest.example_count
        or raw.get("family_count") != manifest.family_count
        or raw.get("safety")
        != {
            "claimed_before_aggregate_prompt_source_open": True,
            "model_accessed": False,
            "tokenizer_accessed": False,
            "other_heldout_roles_exported": False,
            "contains_exactly_one_claimed_role": True,
        }
        or raw.get("export_sha256")
        != _domain_sha256(_EXPORT_DOMAIN, digest_payload)
    ):
        raise Gemma3V8HeldoutIntegrityError(
            "v8 claimed role export lineage or hash is invalid"
        )
    examples = raw.get("examples")
    if not isinstance(examples, list) or len(examples) != manifest.example_count:
        raise Gemma3V8HeldoutIntegrityError(
            "v8 claimed role export example count is invalid"
        )
    restored: list[dict[str, object]] = []
    expected_example_fields = {
        "example_id",
        "prompt",
        "prompt_sha256",
        "family_id",
        "length_band",
        "approximate_word_count",
    }
    for index, value in enumerate(examples):
        if not isinstance(value, dict) or set(value) != expected_example_fields:
            raise Gemma3V8HeldoutIntegrityError(
                "v8 claimed role export example fields are invalid"
            )
        prompt = value.get("prompt")
        expected_hash = manifest.prompt_sha256s[index]
        if (
            value.get("example_id") != f"v8.{manifest.role}.{index:03d}"
            or not isinstance(prompt, str)
            or not prompt
            or _sha256_bytes(prompt.encode("utf-8")) != expected_hash
            or value.get("prompt_sha256") != expected_hash
            or value.get("family_id") != manifest.family_ids[index]
            or value.get("length_band") != manifest.length_bands[index]
            or value.get("approximate_word_count")
            != manifest.approximate_word_counts[index]
        ):
            raise Gemma3V8HeldoutIntegrityError(
                "v8 claimed role export example binding is invalid"
            )
        restored.append(dict(value))
    return tuple(restored)
