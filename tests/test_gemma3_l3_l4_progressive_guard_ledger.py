from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import stat
from threading import Barrier

import pytest

import fisher_graph.gemma3_l3_l4_progressive_guard_ledger as ledger
from fisher_graph.gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
    Gemma3L3L4ProgressiveGuardClaimAuthority,
    Gemma3L3L4ProgressiveGuardClaimReceipt,
    Gemma3L3L4ProgressiveGuardForeignClaimError,
    Gemma3L3L4ProgressiveGuardIntegrityError,
    claim_gemma3_l3_l4_progressive_guard,
    load_gemma3_l3_l4_progressive_guard_claim,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


@pytest.fixture(autouse=True)
def _private_test_ledger(tmp_path, monkeypatch):
    root = tmp_path / "fixed-host-ledger"
    monkeypatch.setattr(ledger, "_FROZEN_LEDGER_ROOT", root)
    return root


def _claim(
    *,
    protocol: str = _sha(1),
    manifest: str = _sha(2),
    challenger: str = _sha(3),
) -> Gemma3L3L4ProgressiveGuardClaimReceipt:
    return claim_gemma3_l3_l4_progressive_guard(
        protocol_sha256=protocol,
        guard_manifest_sha256=manifest,
        challenger_receipt_sha256=challenger,
    )


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key
            for nested in value.values()
            for key in _all_keys(nested)
        )
    if isinstance(value, list):
        return tuple(
            key for nested in value for key in _all_keys(nested)
        )
    return ()


def test_claim_is_canonical_private_manifest_global_and_loadable(
    _private_test_ledger,
) -> None:
    receipt = _claim()
    encoded = receipt.path.read_bytes()
    payload = json.loads(encoded)

    assert encoded == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert receipt.path.parent == _private_test_ledger
    assert receipt.path.name == f"{_sha(2)}.claim.json"
    assert stat.S_IMODE(_private_test_ledger.stat().st_mode) == 0o700
    assert stat.S_IMODE(receipt.path.stat().st_mode) == 0o600
    assert payload["state"] == "claimed_before_guard_materialization"
    assert payload["global_lock_identity"] == "guard_manifest_sha256"
    assert payload["protocol_sha256"] == _sha(1)
    assert payload["guard_manifest_sha256"] == _sha(2)
    assert payload["challenger_receipt_sha256"] == _sha(3)
    assert payload["claim_sha256"] == receipt.claim_sha256
    assert receipt.claim_file_sha256 == hashlib.sha256(encoded).hexdigest()
    assert not any("prompt" in key.lower() for key in _all_keys(payload))

    loaded = load_gemma3_l3_l4_progressive_guard_claim(
        protocol_sha256=_sha(1),
        guard_manifest_sha256=_sha(2),
        challenger_receipt_sha256=_sha(3),
    )
    assert loaded == receipt
    loaded.validate_integrity()


def test_authority_matches_worker_protocol_and_retains_receipt() -> None:
    authority = Gemma3L3L4ProgressiveGuardClaimAuthority()

    claim_sha256 = authority.claim(
        protocol_sha256=_sha(1),
        guard_manifest_sha256=_sha(2),
        challenger_receipt_sha256=_sha(3),
    )

    assert authority.receipt is not None
    assert claim_sha256 == authority.receipt.claim_sha256
    assert authority.receipt.protocol_sha256 == _sha(1)
    authority.receipt.validate_integrity()


def test_repeated_and_foreign_claims_are_manifest_globally_blocked() -> None:
    first = _claim()

    with pytest.raises(
        Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
        match="already claimed",
    ):
        _claim()
    with pytest.raises(
        Gemma3L3L4ProgressiveGuardForeignClaimError,
        match="another protocol or challenger",
    ):
        _claim(protocol=_sha(8))
    with pytest.raises(
        Gemma3L3L4ProgressiveGuardForeignClaimError,
        match="another protocol or challenger",
    ):
        _claim(challenger=_sha(9))

    assert tuple(first.path.parent.glob("*.claim.json")) == (first.path,)


def test_two_racers_produce_exactly_one_persisted_claim() -> None:
    barrier = Barrier(2)

    def race(_: int):
        barrier.wait()
        try:
            return _claim()
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(race, (0, 1)))

    receipts = tuple(
        value
        for value in outcomes
        if isinstance(
            value,
            Gemma3L3L4ProgressiveGuardClaimReceipt,
        )
    )
    failures = tuple(
        value
        for value in outcomes
        if isinstance(
            value,
            (
                Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
                Gemma3L3L4ProgressiveGuardIntegrityError,
            ),
        )
    )
    assert len(receipts) == 1
    assert len(failures) == 1
    receipts[0].validate_integrity()


def test_crash_after_claim_permanently_consumes_the_guard() -> None:
    def claim_then_crash() -> None:
        _claim()
        raise KeyboardInterrupt("synthetic post-claim crash")

    with pytest.raises(KeyboardInterrupt, match="post-claim"):
        claim_then_crash()

    claim_paths = tuple(
        ledger._FROZEN_LEDGER_ROOT.glob("*.claim.json")
    )
    assert len(claim_paths) == 1
    with pytest.raises(
        Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
    ):
        _claim()


@pytest.mark.parametrize("claim_damage", ("corrupt", "permissive"))
def test_corrupt_or_repermissioned_claim_fails_closed(
    claim_damage: str,
) -> None:
    receipt = _claim()
    if claim_damage == "corrupt":
        receipt.path.write_bytes(b"{not-canonical")
        receipt.path.chmod(0o600)
    else:
        receipt.path.chmod(0o644)

    with pytest.raises(Gemma3L3L4ProgressiveGuardIntegrityError):
        _claim()
    with pytest.raises(Gemma3L3L4ProgressiveGuardIntegrityError):
        receipt.validate_integrity()


@pytest.mark.parametrize("root_damage", ("symlink", "permissive"))
def test_unsafe_ledger_root_fails_before_claim(
    tmp_path,
    root_damage: str,
) -> None:
    root = ledger._FROZEN_LEDGER_ROOT
    if root_damage == "symlink":
        target = tmp_path / "symlink-target"
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir(mode=0o755)
        root.chmod(0o755)

    with pytest.raises(
        Gemma3L3L4ProgressiveGuardIntegrityError,
        match="owner-only directory 0700",
    ):
        _claim()
    assert not tuple(tmp_path.rglob("*.claim.json"))


def test_claim_path_symlink_and_wrong_expected_binding_fail_closed(
    tmp_path,
) -> None:
    root = ledger._FROZEN_LEDGER_ROOT
    root.mkdir(mode=0o700)
    target = tmp_path / "foreign-claim"
    target.write_text("{}", encoding="ascii")
    path = root / f"{_sha(2)}.claim.json"
    path.symlink_to(target)

    with pytest.raises(Gemma3L3L4ProgressiveGuardIntegrityError):
        _claim()
    assert target.read_text(encoding="ascii") == "{}"

    path.unlink()
    receipt = _claim()
    with pytest.raises(
        Gemma3L3L4ProgressiveGuardForeignClaimError,
        match="another protocol or challenger",
    ):
        load_gemma3_l3_l4_progressive_guard_claim(
            protocol_sha256=_sha(1),
            guard_manifest_sha256=_sha(2),
            challenger_receipt_sha256=_sha(99),
        )
    receipt.validate_integrity()
