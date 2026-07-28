from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_l3_l4_transport_protocol import (
    claim_transport_calibration_b,
    load_transport_corpus_preclaim_view,
    load_transport_corpus_plan,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    calibration_b = [
        f"assessment prompt {index:03d}"
        for index in range(96)
    ]
    calibration_b_families = [
        f"assessment-family-{index % 8}"
        for index in range(96)
    ]
    prompts = {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": (
            "full_width_single_layer_fresh_a_b_validation_test_hash_only"
        ),
        "calibration_a": ["fit zero", "guard zero", "fit one", "guard one"],
        "calibration_b": calibration_b,
        "validation": ["sealed validation zero", "sealed validation one"],
        "test": ["sealed test zero", "sealed test one"],
    }
    families = {
        "schema": "fisher_graph.gemma3_prompt_family_manifest",
        "format_version": 1,
        "scientific_status": "full_width_single_layer_family_disjoint_roles",
        "calibration_a": ["fit-a", "guard-a", "fit-b", "guard-b"],
        "calibration_b": calibration_b_families,
        "validation": ["validation-a", "validation-b"],
        "test": ["test-a", "test-b"],
    }
    prompt_path = tmp_path / "prompts.json"
    family_path = tmp_path / "families.json"
    audit_path = tmp_path / "audit.json"
    _write_json(prompt_path, prompts)
    _write_json(family_path, families)
    fit_indices = [0, 2]
    guard_indices = [1, 3]
    audit = {
        "schema": "fisher_graph.structured_strong_corpus_audit",
        "format_version": 4,
        "corpus_id": "structured-strong-v9",
        "prompt_file_sha256": _sha256(prompt_path),
        "family_file_sha256": _sha256(family_path),
        "calibration_a_policy": "family_disjoint_fit_guard_development_only",
        "calibration_a_fit_may_train_candidate": True,
        "calibration_a_guard_may_change_candidate": False,
        "calibration_a_family_partitions": {
            "family_disjoint": True,
            "union_covers_calibration_a": True,
            "fit": {
                "name": "fit",
                "prompt_count": 2,
                "family_count": 2,
                "family_ids": ["fit-a", "fit-b"],
                "prompt_indices": fit_indices,
                "prompt_index_sha256": _index_sha256(fit_indices),
            },
            "guard": {
                "name": "guard",
                "prompt_count": 2,
                "family_count": 2,
                "family_ids": ["guard-a", "guard-b"],
                "prompt_indices": guard_indices,
                "prompt_index_sha256": _index_sha256(guard_indices),
            },
        },
        "calibration_b_policy": "one_shot_frozen_candidate_selection",
        "calibration_b_model_evaluated": False,
        "calibration_b_reuse_allowed": False,
        "validation_model_evaluated": False,
        "test_model_evaluated": False,
        "heldout_splits_evaluated": False,
        "heldout_splits_tokenized": False,
        "heldout_splits_unevaluated": True,
        "heldout_splits_untokenized": True,
        "cross_role_family_overlap_count": 0,
        "corpus_frozen_before_model_load": True,
        "generator_adjacent_to_outputs": True,
        "generator_bound_by_sha256": True,
    }
    _write_json(audit_path, audit)
    return prompt_path, family_path, audit_path


def test_loads_only_family_disjoint_transport_roles(tmp_path: Path) -> None:
    prompt_path, family_path, audit_path = _fixtures(tmp_path)

    plan = load_transport_corpus_plan(
        prompt_path,
        family_path,
        audit_path,
        fit_limit=2,
        guard_limit=2,
        calibration_b_limit=2,
    )

    assert plan.fit.prompts == ("fit zero", "fit one")
    assert plan.guard.prompts == ("guard zero", "guard one")
    assert plan.calibration_b.prompts == (
        "assessment prompt 000",
        "assessment prompt 001",
    )
    assert set(plan.fit.family_ids).isdisjoint(plan.guard.family_ids)
    assert set(plan.fit.family_ids).isdisjoint(
        plan.calibration_b.family_ids
    )
    metadata = plan.metadata()
    assert metadata["cross_role_prompt_overlap_count"] == 0
    assert metadata["cross_role_family_overlap_count"] == 0
    assert metadata["validation_or_test_exposed"] is False
    assert "sealed validation zero" not in json.dumps(metadata)
    assert "assessment prompt 000" not in json.dumps(metadata)


def test_preclaim_view_is_hash_only_and_covers_full_role(
    tmp_path: Path,
) -> None:
    prompt_path, family_path, audit_path = _fixtures(tmp_path)

    preclaim = load_transport_corpus_preclaim_view(
        prompt_path,
        family_path,
        audit_path,
    )

    assert preclaim.prompt_count == 96
    assert len(preclaim.prompt_sha256s) == 96
    assert not hasattr(preclaim, "prompts")
    metadata = preclaim.metadata()
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_family_strings"] is False
    serialized = json.dumps(metadata)
    assert "assessment prompt 000" not in serialized
    assert "assessment prompt 095" not in serialized
    assert "assessment-family-0" not in serialized


def test_rejects_tampered_partition_index_digest(tmp_path: Path) -> None:
    prompt_path, family_path, audit_path = _fixtures(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["calibration_a_family_partitions"]["fit"][
        "prompt_index_sha256"
    ] = "0" * 64
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="counts drifted"):
        load_transport_corpus_plan(
            prompt_path,
            family_path,
            audit_path,
        )


def test_calibration_b_claim_is_exclusive_and_text_free(
    tmp_path: Path,
) -> None:
    prompt_path, family_path, audit_path = _fixtures(tmp_path)
    plan = load_transport_corpus_plan(
        prompt_path,
        family_path,
        audit_path,
        fit_limit=2,
        guard_limit=2,
    )

    claim = claim_transport_calibration_b(
        tmp_path / "ledger",
        candidate_artifact_sha256="a" * 64,
        corpus=plan.preclaim_view(),
    )

    payload = json.loads(claim.path.read_text(encoding="ascii"))
    assert payload["candidate_artifact_sha256"] == "a" * 64
    assert payload["calibration_b_prompt_count"] == 96
    assert payload["global_lock_identity_excludes_candidate"] is True
    assert payload["global_lock_identity_excludes_subset"] is True
    assert payload["source_authoritative_shadow"] is True
    assert payload["validation_or_test_accessed"] is False
    serialized = claim.path.read_text(encoding="ascii")
    assert "assessment prompt 000" not in serialized
    assert claim.claim_sha256 == _sha256(claim.path)
    assert claim.candidate_artifact_sha256 == "a" * 64
    with pytest.raises(FileExistsError, match="already claimed"):
        claim_transport_calibration_b(
            tmp_path / "ledger",
            candidate_artifact_sha256="b" * 64,
            corpus=plan,
        )
    assert len(tuple(claim.path.parent.glob("*.json"))) == 1


def test_calibration_b_claim_rejects_partial_role_before_lock(
    tmp_path: Path,
) -> None:
    prompt_path, family_path, audit_path = _fixtures(tmp_path)
    plan = load_transport_corpus_plan(
        prompt_path,
        family_path,
        audit_path,
        fit_limit=2,
        guard_limit=2,
        calibration_b_limit=8,
    )
    ledger = tmp_path / "ledger"

    with pytest.raises(ValueError, match="full fixed 96-prompt role"):
        claim_transport_calibration_b(
            ledger,
            candidate_artifact_sha256="a" * 64,
            corpus=plan,
        )

    assert not ledger.exists()
