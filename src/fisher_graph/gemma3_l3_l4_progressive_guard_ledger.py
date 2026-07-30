"""Durable claim-first authority for the progressive Gemma A guard.

The final Calibration-A guard is a one-use development role.  This module
claims that role in one fixed per-user host namespace before a worker may
materialize or evaluate guard evidence.  Lock identity depends only on the
guard manifest, while the canonical claim payload binds the exact progressive
protocol and frozen challenger.

The filesystem receipt is an integrity and replay-prevention record for one
host.  It is not a cross-machine consensus service or hostile-process
attestation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping


__all__ = [
    "Gemma3L3L4ProgressiveGuardAlreadyClaimedError",
    "Gemma3L3L4ProgressiveGuardClaimAuthority",
    "Gemma3L3L4ProgressiveGuardClaimReceipt",
    "Gemma3L3L4ProgressiveGuardForeignClaimError",
    "Gemma3L3L4ProgressiveGuardIntegrityError",
    "claim_gemma3_l3_l4_progressive_guard",
    "load_gemma3_l3_l4_progressive_guard_claim",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_progressive_guard_claim"
_FORMAT_VERSION = 1
_ROLE = "calibration_a_guard"
_CLAIM_STATE = "claimed_before_guard_materialization"
_LEDGER_NAMESPACE = "gemma3-l3-l4-progressive-calibration-a-guard"
_FROZEN_LEDGER_ROOT = (
    Path.home()
    / ".local"
    / "state"
    / "fisher-graph"
    / _LEDGER_NAMESPACE
)
_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-guard-claim:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_LEDGER_FILE_BYTES = 1 << 20
_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "role",
        "state",
        "protocol_sha256",
        "guard_manifest_sha256",
        "challenger_receipt_sha256",
        "global_lock_identity",
        "claim_sha256",
    }
)


class Gemma3L3L4ProgressiveGuardIntegrityError(RuntimeError):
    """A ledger namespace or persisted claim is not trustworthy."""


class Gemma3L3L4ProgressiveGuardAlreadyClaimedError(FileExistsError):
    """The manifest-global A-guard role has already been consumed."""


class Gemma3L3L4ProgressiveGuardForeignClaimError(
    Gemma3L3L4ProgressiveGuardAlreadyClaimedError
):
    """The guard was consumed by another protocol or challenger."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_sha256(value: object) -> str:
    return hashlib.sha256(
        _CLAIM_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _claim_path(guard_manifest_sha256: str) -> Path:
    manifest = _require_sha256(
        guard_manifest_sha256,
        label="guard manifest",
    )
    return _FROZEN_LEDGER_ROOT / f"{manifest}.claim.json"


def _claim_without_digest(
    *,
    protocol_sha256: str,
    guard_manifest_sha256: str,
    challenger_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": _ROLE,
        "state": _CLAIM_STATE,
        "protocol_sha256": protocol_sha256,
        "guard_manifest_sha256": guard_manifest_sha256,
        "challenger_receipt_sha256": challenger_receipt_sha256,
        "global_lock_identity": "guard_manifest_sha256",
    }


def _claim_payload(
    *,
    protocol_sha256: str,
    guard_manifest_sha256: str,
    challenger_receipt_sha256: str,
) -> dict[str, object]:
    payload = _claim_without_digest(
        protocol_sha256=protocol_sha256,
        guard_manifest_sha256=guard_manifest_sha256,
        challenger_receipt_sha256=challenger_receipt_sha256,
    )
    payload["claim_sha256"] = _domain_sha256(payload)
    return payload


def _validate_claim_payload(raw: Mapping[str, object]) -> None:
    if set(raw) != _CLAIM_FIELDS:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim fields differ from the frozen schema"
        )
    if (
        raw["schema"] != _SCHEMA
        or raw["format_version"] != _FORMAT_VERSION
        or raw["role"] != _ROLE
        or raw["state"] != _CLAIM_STATE
        or raw["global_lock_identity"] != "guard_manifest_sha256"
    ):
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim metadata differs from the frozen contract"
        )
    try:
        for field in (
            "protocol_sha256",
            "guard_manifest_sha256",
            "challenger_receipt_sha256",
            "claim_sha256",
        ):
            _require_sha256(raw[field], label=f"guard claim {field}")
    except ValueError as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim contains a malformed SHA-256"
        ) from error
    payload = dict(raw)
    observed = payload.pop("claim_sha256")
    if observed != _domain_sha256(payload):
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim hash mismatch"
        )


def _ensure_private_ledger_root() -> None:
    try:
        _FROZEN_LEDGER_ROOT.mkdir(
            mode=0o700,
            parents=True,
            exist_ok=True,
        )
        details = os.lstat(_FROZEN_LEDGER_ROOT)
    except OSError as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard ledger namespace cannot be authenticated"
        ) from error
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard ledger namespace must be an owner-only directory 0700"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard ledger directory cannot be synchronized"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard ledger directory synchronization failed"
        ) from error
    finally:
        os.close(descriptor)


def _exclusive_private_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("guard claim write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_private_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim cannot be opened as a private regular file"
        ) from error
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size <= 0
            or details.st_size > _MAX_LEDGER_FILE_BYTES
        ):
            raise Gemma3L3L4ProgressiveGuardIntegrityError(
                "guard claim must be a nonempty owner-only regular file 0600"
            )
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise Gemma3L3L4ProgressiveGuardIntegrityError(
                    "guard claim ended before its declared file size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Gemma3L3L4ProgressiveGuardIntegrityError(
                "guard claim changed while it was read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_canonical_claim(encoded: bytes) -> dict[str, object]:
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim is not canonical ASCII JSON"
        ) from error
    if not isinstance(raw, dict) or _canonical_json_bytes(raw) != encoded:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim is not a canonical JSON object"
        )
    _validate_claim_payload(raw)
    return raw


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ProgressiveGuardClaimReceipt:
    """Immutable identity for one persisted manifest-global guard claim."""

    path: Path
    protocol_sha256: str
    guard_manifest_sha256: str
    challenger_receipt_sha256: str
    claim_sha256: str
    claim_file_sha256: str
    _claim_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("guard claim path must be a Path")
        for name in (
            "protocol_sha256",
            "guard_manifest_sha256",
            "challenger_receipt_sha256",
            "claim_sha256",
            "claim_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self._claim_bytes, bytes) or not self._claim_bytes:
            raise ValueError("guard claim receipt requires persisted bytes")

    def validate_integrity(self) -> None:
        expected_path = _claim_path(self.guard_manifest_sha256)
        if self.path != expected_path:
            raise Gemma3L3L4ProgressiveGuardIntegrityError(
                "guard claim path differs from the manifest-global ledger"
            )
        _ensure_private_ledger_root()
        encoded = _read_private_file(self.path)
        raw = _decode_canonical_claim(encoded)
        if (
            encoded != self._claim_bytes
            or _file_sha256(encoded) != self.claim_file_sha256
            or raw["protocol_sha256"] != self.protocol_sha256
            or raw["guard_manifest_sha256"]
            != self.guard_manifest_sha256
            or raw["challenger_receipt_sha256"]
            != self.challenger_receipt_sha256
            or raw["claim_sha256"] != self.claim_sha256
        ):
            raise Gemma3L3L4ProgressiveGuardIntegrityError(
                "guard claim differs from its immutable receipt"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "protocol_sha256": self.protocol_sha256,
            "guard_manifest_sha256": self.guard_manifest_sha256,
            "challenger_receipt_sha256": (
                self.challenger_receipt_sha256
            ),
            "claim_sha256": self.claim_sha256,
            "claim_file_sha256": self.claim_file_sha256,
            "state": _CLAIM_STATE,
            "global_lock_identity": "guard_manifest_sha256",
        }


def _receipt_from_persisted(
    *,
    path: Path,
    encoded: bytes,
    raw: Mapping[str, object],
) -> Gemma3L3L4ProgressiveGuardClaimReceipt:
    return Gemma3L3L4ProgressiveGuardClaimReceipt(
        path=path,
        protocol_sha256=str(raw["protocol_sha256"]),
        guard_manifest_sha256=str(raw["guard_manifest_sha256"]),
        challenger_receipt_sha256=str(
            raw["challenger_receipt_sha256"]
        ),
        claim_sha256=str(raw["claim_sha256"]),
        claim_file_sha256=_file_sha256(encoded),
        _claim_bytes=encoded,
    )


def claim_gemma3_l3_l4_progressive_guard(
    *,
    protocol_sha256: str,
    guard_manifest_sha256: str,
    challenger_receipt_sha256: str,
) -> Gemma3L3L4ProgressiveGuardClaimReceipt:
    """Atomically consume one guard manifest before guard materialization."""

    protocol = _require_sha256(
        protocol_sha256,
        label="progressive protocol",
    )
    manifest = _require_sha256(
        guard_manifest_sha256,
        label="guard manifest",
    )
    challenger = _require_sha256(
        challenger_receipt_sha256,
        label="frozen challenger receipt",
    )
    payload = _claim_payload(
        protocol_sha256=protocol,
        guard_manifest_sha256=manifest,
        challenger_receipt_sha256=challenger,
    )
    encoded = _canonical_json_bytes(payload)
    _ensure_private_ledger_root()
    path = _claim_path(manifest)
    try:
        _exclusive_private_write(path, encoded)
    except FileExistsError as error:
        existing = _read_private_file(path)
        raw = _decode_canonical_claim(existing)
        if existing != encoded:
            raise Gemma3L3L4ProgressiveGuardForeignClaimError(
                "Calibration-A guard was consumed by another protocol or "
                "challenger"
            ) from error
        raise Gemma3L3L4ProgressiveGuardAlreadyClaimedError(
            "Calibration-A guard is already claimed for this manifest"
        ) from error
    persisted = _read_private_file(path)
    if persisted != encoded:
        raise Gemma3L3L4ProgressiveGuardIntegrityError(
            "guard claim changed immediately after exclusive creation"
        )
    receipt = _receipt_from_persisted(
        path=path,
        encoded=persisted,
        raw=payload,
    )
    receipt.validate_integrity()
    return receipt


def load_gemma3_l3_l4_progressive_guard_claim(
    *,
    protocol_sha256: str,
    guard_manifest_sha256: str,
    challenger_receipt_sha256: str,
) -> Gemma3L3L4ProgressiveGuardClaimReceipt:
    """Load and authenticate the exact expected persisted guard claim."""

    protocol = _require_sha256(
        protocol_sha256,
        label="progressive protocol",
    )
    manifest = _require_sha256(
        guard_manifest_sha256,
        label="guard manifest",
    )
    challenger = _require_sha256(
        challenger_receipt_sha256,
        label="frozen challenger receipt",
    )
    _ensure_private_ledger_root()
    path = _claim_path(manifest)
    encoded = _read_private_file(path)
    raw = _decode_canonical_claim(encoded)
    if (
        raw["protocol_sha256"] != protocol
        or raw["guard_manifest_sha256"] != manifest
        or raw["challenger_receipt_sha256"] != challenger
    ):
        raise Gemma3L3L4ProgressiveGuardForeignClaimError(
            "persisted A-guard claim belongs to another protocol or "
            "challenger"
        )
    receipt = _receipt_from_persisted(
        path=path,
        encoded=encoded,
        raw=raw,
    )
    receipt.validate_integrity()
    return receipt


class Gemma3L3L4ProgressiveGuardClaimAuthority:
    """Worker-compatible adapter retaining the full durable claim receipt."""

    def __init__(self) -> None:
        self._receipt: Gemma3L3L4ProgressiveGuardClaimReceipt | None = None

    @property
    def receipt(self) -> Gemma3L3L4ProgressiveGuardClaimReceipt | None:
        return self._receipt

    def claim(
        self,
        *,
        protocol_sha256: str,
        guard_manifest_sha256: str,
        challenger_receipt_sha256: str,
    ) -> str:
        receipt = claim_gemma3_l3_l4_progressive_guard(
            protocol_sha256=protocol_sha256,
            guard_manifest_sha256=guard_manifest_sha256,
            challenger_receipt_sha256=challenger_receipt_sha256,
        )
        self._receipt = receipt
        return receipt.claim_sha256
