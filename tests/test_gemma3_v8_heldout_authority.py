from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

import fisher_graph.gemma3_v8_heldout_authority as authority


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    counts = {
        "calibration_a": 512,
        "calibration_b": 96,
        "validation": 96,
        "test": 96,
    }
    family_counts = {
        "calibration_a": 16,
        "calibration_b": 8,
        "validation": 8,
        "test": 8,
    }
    prompts = {
        role: [f"{role} synthetic prompt {index:03d}" for index in range(count)]
        for role, count in counts.items()
    }
    families = {
        role: [
            f"{role}.family.{index % family_counts[role]}"
            for index in range(count)
        ]
        for role, count in counts.items()
    }
    prompt_payload = {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": (
            "full_width_single_layer_fresh_a_b_validation_test_hash_only"
        ),
        **prompts,
    }
    family_payload = {
        "schema": "fisher_graph.gemma3_prompt_family_manifest",
        "format_version": 1,
        "scientific_status": (
            "full_width_single_layer_family_disjoint_roles"
        ),
        **families,
    }
    prompt_bytes = _encoded(prompt_payload)
    family_bytes = _encoded(family_payload)
    audit_payload = {
        "schema": "fisher_graph.structured_strong_corpus_audit",
        "format_version": 4,
        "corpus_id": "structured-strong-v8",
        "counts": counts,
        "families_per_role": family_counts,
        "prompt_file_sha256": _sha(prompt_bytes),
        "family_file_sha256": _sha(family_bytes),
        "heldout_splits_evaluated": False,
        "heldout_splits_tokenized": False,
        "heldout_splits_unevaluated": True,
        "heldout_splits_untokenized": True,
        "calibration_b_model_evaluated": False,
        "validation_model_evaluated": False,
        "test_model_evaluated": False,
        "tokenizer_or_model_accessed": False,
        "cross_role_family_overlap_count": 0,
        "calibration_b_policy": "one_shot_frozen_candidate_selection",
        "test_policy": "generated_and_hash_audited_only",
        "prompt_sha256_by_role": {
            role: [_sha(prompt.encode("utf-8")) for prompt in values]
            for role, values in prompts.items()
        },
        "normalized_prompt_sha256_by_role": {
            role: [_sha(prompt.lower().encode("utf-8")) for prompt in values]
            for role, values in prompts.items()
        },
        "length_band_by_role": {
            role: [
                ("micro", "compact", "medium", "long")[index % 4]
                for index in range(count)
            ]
            for role, count in counts.items()
        },
        "approximate_word_counts_by_role": {
            role: [14 + index % 4 for index in range(count)]
            for role, count in counts.items()
        },
    }
    audit_bytes = _encoded(audit_payload)
    prompt_path = tmp_path / "prompts.json"
    family_path = tmp_path / "families.json"
    audit_path = tmp_path / "audit.json"
    prompt_path.write_bytes(prompt_bytes)
    family_path.write_bytes(family_bytes)
    audit_path.write_bytes(audit_bytes)
    monkeypatch.setattr(authority, "_V8_PROMPT_FILE_SHA256", _sha(prompt_bytes))
    monkeypatch.setattr(authority, "_V8_FAMILY_FILE_SHA256", _sha(family_bytes))
    monkeypatch.setattr(authority, "_V8_AUDIT_FILE_SHA256", _sha(audit_bytes))
    monkeypatch.setattr(authority, "_FROZEN_LEDGER_ROOT", tmp_path / "ledger")
    return audit_path, family_path, prompt_path


def test_manifest_is_prompt_free_and_does_not_open_prompt_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    prompt_path.unlink()

    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )

    assert manifest.example_count == 96
    assert manifest.family_count == 8
    assert manifest.declared_policy == "one_shot_frozen_candidate_selection"
    assert manifest.metadata()["prompt_text_materialized"] is False


def test_claim_replay_and_foreign_challenger_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, _ = _fixture_sources(tmp_path, monkeypatch)
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    protocol = "1" * 64
    challenger = "2" * 64
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256=protocol,
        challenger_receipt_sha256=challenger,
    )

    claim.validate_integrity()
    loaded = authority.load_gemma3_v8_heldout_claim(
        manifest,
        protocol_sha256=protocol,
        challenger_receipt_sha256=challenger,
    )
    assert loaded.claim_sha256 == claim.claim_sha256
    with pytest.raises(authority.Gemma3V8HeldoutAlreadyClaimedError):
        authority.claim_gemma3_v8_heldout_role(
            manifest,
            protocol_sha256=protocol,
            challenger_receipt_sha256=challenger,
        )
    with pytest.raises(authority.Gemma3V8HeldoutForeignClaimError):
        authority.load_gemma3_v8_heldout_claim(
            manifest,
            protocol_sha256=protocol,
            challenger_receipt_sha256="3" * 64,
        )


def test_export_opens_only_after_claim_and_roundtrips_exact_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256="4" * 64,
        challenger_receipt_sha256="5" * 64,
    )
    output = tmp_path / "claimed-calibration-b.json"

    payload = authority.export_claimed_gemma3_v8_role(
        manifest,
        claim,
        prompt_path=prompt_path,
        output=output,
    )
    rows = authority.load_claimed_gemma3_v8_role(
        output,
        manifest=manifest,
        claim=claim,
    )

    assert payload["role"] == "calibration_b"
    assert len(rows) == 96
    assert {row["family_id"] for row in rows} == set(manifest.family_ids)
    assert payload["safety"]["other_heldout_roles_exported"] is False
    assert "validation" not in output.read_text()
    assert "test synthetic prompt" not in output.read_text()
    details = output.stat()
    assert stat.S_ISREG(details.st_mode)
    assert stat.S_IMODE(details.st_mode) == 0o600
    assert details.st_uid == os.getuid()


def test_export_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256="a" * 64,
        challenger_receipt_sha256="b" * 64,
    )
    synced_modes: list[int] = []

    def record_fsync(descriptor: int) -> None:
        synced_modes.append(os.fstat(descriptor).st_mode)

    monkeypatch.setattr(authority.os, "fsync", record_fsync)
    output = tmp_path / "claimed-calibration-b.json"
    authority.export_claimed_gemma3_v8_role(
        manifest,
        claim,
        prompt_path=prompt_path,
        output=output,
    )

    assert any(stat.S_ISREG(mode) for mode in synced_modes)
    assert any(stat.S_ISDIR(mode) for mode in synced_modes)


def test_export_refuses_overwrite_without_changing_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256="c" * 64,
        challenger_receipt_sha256="d" * 64,
    )
    output = tmp_path / "claimed-calibration-b.json"
    original = b"do not replace\n"
    output.write_bytes(original)
    prompt_path.unlink()

    with pytest.raises(FileExistsError, match="overwrite"):
        authority.export_claimed_gemma3_v8_role(
            manifest,
            claim,
            prompt_path=prompt_path,
            output=output,
        )

    assert output.read_bytes() == original


def test_invalid_claim_is_rejected_before_prompt_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "calibration_b",
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256="6" * 64,
        challenger_receipt_sha256="7" * 64,
    )
    claim.path.chmod(0o644)
    prompt_path.unlink()

    with pytest.raises(
        authority.Gemma3V8HeldoutIntegrityError,
        match="private regular file",
    ):
        authority.export_claimed_gemma3_v8_role(
            manifest,
            claim,
            prompt_path=prompt_path,
            output=tmp_path / "must-not-exist.json",
        )


def test_export_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, family_path, prompt_path = _fixture_sources(
        tmp_path,
        monkeypatch,
    )
    manifest = authority.load_gemma3_v8_heldout_manifest(
        "validation",
        audit_path=audit_path,
        family_path=family_path,
    )
    claim = authority.claim_gemma3_v8_heldout_role(
        manifest,
        protocol_sha256="8" * 64,
        challenger_receipt_sha256="9" * 64,
    )
    output = tmp_path / "claimed-validation.json"
    payload = authority.export_claimed_gemma3_v8_role(
        manifest,
        claim,
        prompt_path=prompt_path,
        output=output,
    )
    tampered = copy.deepcopy(payload)
    tampered["examples"][0]["prompt"] += " changed"
    output.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(
        authority.Gemma3V8HeldoutIntegrityError,
        match="lineage or hash",
    ):
        authority.load_claimed_gemma3_v8_role(
            output,
            manifest=manifest,
            claim=claim,
        )
