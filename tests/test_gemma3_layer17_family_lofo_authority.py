from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)
from fisher_graph.gemma3_layer10_v8_corpus import (
    GEMMA3_LAYER10_V8_CORPUS_ID,
)
import fisher_graph.gemma3_layer17_family_lofo_authority as lofo


_TOKENIZER_CONTRACT: dict[str, object] = {
    "kind": "synthetic-tokenizer",
    "max_length": 8,
    "tokenization_batch_size": 32,
    "device": "cpu",
}


def _digest(seed: int) -> str:
    return hashlib.sha256(f"synthetic-{seed}".encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_files(
    root: Path,
) -> tuple[
    dict[str, Path],
    object,
    dict[str, object],
    tuple[str, ...],
    tuple[str, ...],
]:
    root.mkdir(parents=True)
    paths = {
        "calibration_a_fit": root / "fit.json",
        "calibration_a_selection": root / "selection.json",
        "calibration_a_guard": root / "guard.json",
    }
    fit_prompts = tuple(
        f"Synthetic fit prompt {family:02d}-{row:02d}."
        for family in range(8)
        for row in range(32)
    )
    fit_families = tuple(
        f"source_family_{family:02d}"
        for family in range(8)
        for _ in range(32)
    )
    write_gemma3_l3_l4_progressive_a_role_input(
        paths["calibration_a_fit"],
        corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
        profile="full",
        role="calibration_a_fit",
        prompts=fit_prompts,
        family_ids=fit_families,
    )
    write_gemma3_l3_l4_progressive_a_role_input(
        paths["calibration_a_selection"],
        corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
        profile="full",
        role="calibration_a_selection",
        prompts=("Synthetic selection prompt.",),
        family_ids=("selection_source_family",),
    )
    write_gemma3_l3_l4_progressive_a_role_input(
        paths["calibration_a_guard"],
        corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
        profile="full",
        role="calibration_a_guard",
        prompts=("Synthetic guard prompt.",),
        family_ids=("guard_source_family",),
    )
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id=GEMMA3_LAYER10_V8_CORPUS_ID,
        profile="full",
        tokenizer_contract=_TOKENIZER_CONTRACT,
        role_input_paths=paths,  # type: ignore[arg-type]
    )
    artifact_path = root / "corpus.json"
    artifact_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        artifact_path,
        artifact,
    )
    receipt_path = root / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    paths["corpus"] = artifact_path
    paths["receipt"] = receipt_path
    fit_view = artifact.role_view("calibration_a_fit")
    receipt: dict[str, object] = {
        "corpus_id": GEMMA3_LAYER10_V8_CORPUS_ID,
        "profile": "full",
        "receipt_sha256": _digest(1),
        "corpus": {
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": artifact_file_sha256,
            "tokenizer_contract_sha256": artifact.tokenizer_contract_sha256,
        },
        "roles": {
            "calibration_a_fit": {
                "role_id": "calibration_a_fit",
                "manifest_sha256": fit_view.manifest_sha256,
                "role_input_file_sha256": fit_view.role_input_file_sha256,
                "example_count": 256,
                "family_count": 8,
            }
        },
        "heldout": {
            "role_ids": ["calibration_b", "validation", "test"],
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
    return paths, artifact, receipt, fit_prompts, fit_families


def _install_synthetic_frozen_protocol(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact: object,
    prompts: tuple[str, ...],
    families: tuple[str, ...],
) -> dict[str, object]:
    fit_view = artifact.role_view("calibration_a_fit")
    slices = tuple(
        lofo._FamilySlice(
            opaque_label=f"family_{family_index:02d}",
            source_family=source_family,
            source_indices=tuple(
                index
                for index, family in enumerate(families)
                if family == source_family
            ),
            prompts=tuple(
                prompt
                for prompt, family in zip(prompts, families, strict=True)
                if family == source_family
            ),
            ordered_prompt_sha256s=tuple(
                identity
                for identity, family in zip(
                    fit_view.ordered_prompt_sha256s,
                    families,
                    strict=True,
                )
                if family == source_family
            ),
        )
        for family_index, source_family in enumerate(fit_view.family_ids)
    )
    derived = lofo._derived_private_protocol_binding(
        fit_manifest_sha256=fit_view.manifest_sha256,
        slices=slices,
    )
    protocol_sha256 = _digest(900)
    protocol: dict[str, object] = {
        "artifact_sha256": protocol_sha256,
        "corpus_authority": {
            "artifact_sha256": artifact.artifact_sha256,
            "tokenizer_contract_sha256": (
                artifact.tokenizer_contract_sha256
            ),
            "fit_membership_sha256": derived[
                "fit_membership_sha256"
            ],
            "family_alias_mapping_sha256": derived[
                "family_alias_mapping_sha256"
            ],
        },
        "role_bindings": {
            "fit": {
                "manifest_sha256": fit_view.manifest_sha256,
                "source_file_sha256": fit_view.role_input_file_sha256,
            }
        },
        "folds": list(derived["folds"]),
    }

    def authenticate(raw: object) -> dict[str, object]:
        assert raw == artifact.to_dict()
        return copy.deepcopy(protocol)

    monkeypatch.setattr(
        lofo,
        "FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256",
        protocol_sha256,
    )
    monkeypatch.setattr(
        lofo,
        "validate_v8_layer17_family_lofo_protocol",
        lambda raw: copy.deepcopy(raw),
    )
    monkeypatch.setattr(
        lofo,
        "build_default_v8_layer17_family_lofo_protocol",
        lambda: copy.deepcopy(protocol),
    )
    monkeypatch.setattr(
        lofo,
        "build_authenticated_v8_layer17_family_lofo_protocol",
        authenticate,
    )
    return protocol


def _load_synthetic_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    lofo.Gemma3Layer17FamilyLOFOAuthority,
    dict[str, Path],
    tuple[str, ...],
    tuple[str, ...],
]:
    paths, artifact, receipt, prompts, families = _synthetic_files(
        tmp_path / "synthetic"
    )
    _install_synthetic_frozen_protocol(
        monkeypatch,
        artifact=artifact,
        prompts=prompts,
        families=families,
    )
    observed_receipts: list[Path] = []

    def load_receipt(path: Path | str) -> dict[str, object]:
        observed_receipts.append(Path(path))
        return receipt

    monkeypatch.setattr(
        lofo,
        "load_gemma3_layer10_v8_corpus_receipt",
        load_receipt,
    )
    paths["calibration_a_selection"].unlink()
    paths["calibration_a_guard"].unlink()
    authority = lofo.load_gemma3_layer17_family_lofo_authority(
        corpus_receipt_path=paths["receipt"],
        corpus_artifact_path=paths["corpus"],
        fit_input_path=paths["calibration_a_fit"],
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )
    assert observed_receipts == [paths["receipt"].resolve()]
    return authority, paths, prompts, families


def _walk_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key
            for child in value.values()
            for key in _walk_keys(child)
        )
    if isinstance(value, (tuple, list)):
        return tuple(key for child in value for key in _walk_keys(child))
    return ()


def test_authority_opens_only_authenticated_fit_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, paths, prompts, families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )

    metadata = authority.metadata()
    lofo.validate_gemma3_layer17_family_lofo_authority_metadata(metadata)
    assert metadata["corpus"]["example_count"] == 256  # type: ignore[index]
    assert metadata["corpus"]["block_count"] == 8  # type: ignore[index]
    assert metadata["access"] == {
        "fit_opened": True,
        "fit_tokenized": False,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "model_loaded": False,
        "model_evaluated": False,
    }
    assert not paths["calibration_a_selection"].exists()
    assert not paths["calibration_a_guard"].exists()
    rendered = json.dumps(metadata, sort_keys=True)
    assert all(prompt not in rendered for prompt in prompts)
    assert all(family not in rendered for family in set(families))
    assert all(
        block.source_family not in repr(authority)
        for block in authority._slices
    )


def test_receipt_artifact_file_mismatch_stops_before_fit_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _artifact, receipt, _prompts, _families = _synthetic_files(
        tmp_path / "mismatch"
    )
    receipt = copy.deepcopy(receipt)
    receipt["corpus"]["artifact_file_sha256"] = _digest(404)  # type: ignore[index]
    monkeypatch.setattr(
        lofo,
        "load_gemma3_layer10_v8_corpus_receipt",
        lambda path: receipt,
    )
    opened_fit = False

    def forbidden_fit_open(*args: object, **kwargs: object) -> object:
        nonlocal opened_fit
        opened_fit = True
        raise AssertionError("fit loader must not run before artifact binding")

    monkeypatch.setattr(
        lofo,
        "load_gemma3_l3_l4_progressive_a_fit_role",
        forbidden_fit_open,
    )

    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="artifact file differs",
    ):
        lofo.load_gemma3_layer17_family_lofo_authority(
            corpus_receipt_path=paths["receipt"],
            corpus_artifact_path=paths["corpus"],
            fit_input_path=paths["calibration_a_fit"],
            tokenizer_contract=_TOKENIZER_CONTRACT,
        )
    assert opened_fit is False


def test_consistent_nonfrozen_corpus_stops_before_fit_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _artifact, receipt, _prompts, _families = _synthetic_files(
        tmp_path / "nonfrozen"
    )
    monkeypatch.setattr(
        lofo,
        "load_gemma3_layer10_v8_corpus_receipt",
        lambda path: receipt,
    )
    opened_fit = False

    def forbidden_fit_open(*args: object, **kwargs: object) -> object:
        nonlocal opened_fit
        opened_fit = True
        raise AssertionError("nonfrozen authority must fail before fit open")

    monkeypatch.setattr(
        lofo,
        "load_gemma3_l3_l4_progressive_a_fit_role",
        forbidden_fit_open,
    )

    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="differs from the frozen family-LOFO protocol",
    ):
        lofo.load_gemma3_layer17_family_lofo_authority(
            corpus_receipt_path=paths["receipt"],
            corpus_artifact_path=paths["corpus"],
            fit_input_path=paths["calibration_a_fit"],
            tokenizer_contract=_TOKENIZER_CONTRACT,
        )
    assert opened_fit is False


def test_authority_hash_binds_frozen_protocol_and_private_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _paths, prompts, families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )
    metadata = authority.metadata()
    protocol = metadata["protocol"]
    assert protocol["protocol_artifact_sha256"] == (
        lofo.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
    )
    assert protocol["fold_count"] == 8
    assert tuple(
        fold["held_family_alias"] for fold in protocol["folds"]
    ) == tuple(f"family_{index:02d}" for index in range(8))
    assert all(
        fold["held_example_count"] == 32
        and fold["training_example_count"] == 224
        for fold in protocol["folds"]
    )

    left, right, *remaining = authority._slices
    swapped = (
        replace(
            left,
            source_family=right.source_family,
            source_indices=right.source_indices,
            prompts=right.prompts,
            ordered_prompt_sha256s=right.ordered_prompt_sha256s,
        ),
        replace(
            right,
            source_family=left.source_family,
            source_indices=left.source_indices,
            prompts=left.prompts,
            ordered_prompt_sha256s=left.ordered_prompt_sha256s,
        ),
        *remaining,
    )
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="differs from the frozen family-LOFO protocol",
    ):
        replace(authority, _slices=swapped)

    with pytest.raises(ValueError, match="prompt text differs"):
        replace(
            left,
            prompts=("Mutated synthetic prompt.", *left.prompts[1:]),
        )

    rendered = json.dumps(metadata, sort_keys=True)
    assert all(prompt not in rendered for prompt in prompts)
    assert all(family not in rendered for family in set(families))


def test_materialization_revalidates_private_slices_before_tokenization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _paths, _prompts, _families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )
    left, right, *remaining = authority._slices
    swapped = (
        replace(
            left,
            source_family=right.source_family,
            source_indices=right.source_indices,
            prompts=right.prompts,
            ordered_prompt_sha256s=right.ordered_prompt_sha256s,
        ),
        replace(
            right,
            source_family=left.source_family,
            source_indices=left.source_indices,
            prompts=left.prompts,
            ordered_prompt_sha256s=left.ordered_prompt_sha256s,
        ),
        *remaining,
    )
    object.__setattr__(authority, "_slices", swapped)
    tokenization_called = False

    def forbidden_materialization(*args: object, **kwargs: object) -> object:
        nonlocal tokenization_called
        tokenization_called = True
        raise AssertionError("drifted slices must fail before tokenization")

    monkeypatch.setattr(lofo, "_materialize_role", forbidden_materialization)
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="differs from the frozen family-LOFO protocol",
    ):
        lofo.materialize_gemma3_layer17_family_lofo(authority, object())
    assert tokenization_called is False


def test_materialization_builds_eight_opaque_complete_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _paths, prompts, families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )
    prompt_hashes = tuple(
        identity
        for block in authority._slices
        for identity in block.ordered_prompt_sha256s
    )
    leaked_stream_hashes: list[str] = []
    observed_splits: list[str] = []

    def materialize(
        tokenizer: object,
        role: object,
        *,
        split_name: str,
        max_length: int,
        tokenization_batch_size: int,
        device: torch.device,
    ) -> tuple[tuple[CalibrationBatch, ...], dict[str, object]]:
        del tokenizer
        index = len(observed_splits)
        observed_splits.append(split_name)
        assert max_length == 8
        assert tokenization_batch_size == 32
        assert device.type == "cpu"
        example_ids = role.ordered_prompt_sha256s
        batch = CalibrationBatch(
            model_inputs={
                "input_ids": torch.full(
                    (32, 2),
                    index + 1000,
                    dtype=torch.long,
                )
            },
            targets=torch.zeros((32, 2), dtype=torch.long),
            valid_positions=torch.ones((32, 2), dtype=torch.bool),
            example_ids=example_ids,
        )
        stream_sha256 = _digest(index + 100)
        leaked_stream_hashes.append(stream_sha256)
        return (batch,), {
            "split": split_name,
            "batches": 1,
            "sequences": 32,
            "serialized_sha256": stream_sha256,
            "source_prompt_sha256": list(example_ids),
            "examples": [
                {"content_sha256": _digest(index + 200)}
                for _ in range(32)
            ],
        }

    monkeypatch.setattr(lofo, "_materialize_role", materialize)
    blocks, metadata = lofo.materialize_gemma3_layer17_family_lofo(
        authority,
        object(),
    )

    lofo.validate_gemma3_layer17_family_lofo_materialization_metadata(metadata)
    assert tuple(label for label, _ in blocks) == tuple(
        f"family_{index:02d}" for index in range(8)
    )
    assert observed_splits == [
        f"layer17_fit_lofo_family_{index:02d}" for index in range(8)
    ]
    tokenization = metadata["tokenization"]
    assert tokenization["example_count"] == 256  # type: ignore[index]
    assert tokenization["logical_valid_tokens"] == 512  # type: ignore[index]
    assert tokenization["supervised_tokens"] == 512  # type: ignore[index]
    assert set(tokenization["blocks"]) == {  # type: ignore[index]
        f"family_{index:02d}" for index in range(8)
    }
    rendered = json.dumps(metadata, sort_keys=True)
    assert all(prompt not in rendered for prompt in prompts)
    assert all(family not in rendered for family in set(families))
    assert all(identity not in rendered for identity in prompt_hashes)
    assert all(stream_hash not in rendered for stream_hash in leaked_stream_hashes)
    assert not {
        "prompts",
        "prompt_sha256s",
        "ordered_prompt_sha256s",
        "family_ids",
        "ordered_family_ids",
        "input_ids",
        "token_ids",
        "content_sha256",
    } & set(_walk_keys(metadata))

    bad_authority_sha = copy.deepcopy(metadata)
    bad_authority_sha["authority_sha256"] = int("1" * 64)
    bad_authority_payload = {
        key: value
        for key, value in bad_authority_sha.items()
        if key != "materialization_sha256"
    }
    bad_authority_sha["materialization_sha256"] = lofo._domain_sha256(
        lofo._MATERIALIZATION_DOMAIN,
        bad_authority_payload,
    )
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="must be a lowercase SHA-256",
    ):
        lofo.validate_gemma3_layer17_family_lofo_materialization_metadata(
            bad_authority_sha
        )

    bad_stream_sha = copy.deepcopy(metadata)
    bad_stream_sha["tokenization"]["stream_catalog_sha256"] = int("1" * 64)
    bad_stream_payload = {
        key: value
        for key, value in bad_stream_sha.items()
        if key != "materialization_sha256"
    }
    bad_stream_sha["materialization_sha256"] = lofo._domain_sha256(
        lofo._MATERIALIZATION_DOMAIN,
        bad_stream_payload,
    )
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="must be a lowercase SHA-256",
    ):
        lofo.validate_gemma3_layer17_family_lofo_materialization_metadata(
            bad_stream_sha
        )


def test_materialization_rejects_example_membership_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _paths, _prompts, _families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )

    def materialize(
        tokenizer: object,
        role: object,
        *,
        split_name: str,
        max_length: int,
        tokenization_batch_size: int,
        device: torch.device,
    ) -> tuple[tuple[CalibrationBatch, ...], dict[str, object]]:
        del tokenizer, max_length, tokenization_batch_size, device
        batch = CalibrationBatch(
            model_inputs={"input_ids": torch.zeros((32, 2), dtype=torch.long)},
            targets=torch.zeros((32, 2), dtype=torch.long),
            valid_positions=torch.ones((32, 2), dtype=torch.bool),
            example_ids=tuple(reversed(role.ordered_prompt_sha256s)),
        )
        return (batch,), {
            "split": split_name,
            "batches": 1,
            "sequences": 32,
            "serialized_sha256": _digest(500),
        }

    monkeypatch.setattr(lofo, "_materialize_role", materialize)
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="membership drifted",
    ):
        lofo.materialize_gemma3_layer17_family_lofo(authority, object())


def test_public_loader_has_no_protected_role_path_surface() -> None:
    parameters = set(
        inspect.signature(
            lofo.load_gemma3_layer17_family_lofo_authority
        ).parameters
    )
    assert parameters == {
        "corpus_receipt_path",
        "corpus_artifact_path",
        "fit_input_path",
        "tokenizer_contract",
    }
    joined = " ".join(parameters)
    for protected in (
        "selection",
        "guard",
        "calibration_b",
        "validation",
        "test",
    ):
        assert protected not in joined


def test_public_metadata_validator_rejects_identity_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _paths, _prompts, families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )
    metadata = authority.metadata()
    metadata["corpus"]["family_ids"] = sorted(set(families))  # type: ignore[index]
    with pytest.raises(
        lofo.Gemma3Layer17FamilyLOFOAuthorityError,
        match="fields differ|forbidden source field",
    ):
        lofo.validate_gemma3_layer17_family_lofo_authority_metadata(metadata)


def test_synthetic_receipt_file_hash_is_only_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, paths, _prompts, _families = _load_synthetic_authority(
        tmp_path,
        monkeypatch,
    )
    assert authority.receipt_file_sha256 == _file_sha256(paths["receipt"])
