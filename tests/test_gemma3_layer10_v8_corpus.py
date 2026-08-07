from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_layer10_v8_corpus as corpus_module
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    load_gemma3_l3_l4_progressive_a_corpus,
)
from fisher_graph.gemma3_layer10_v8_corpus import (
    Gemma3Layer10V8CorpusIntegrityError,
    load_gemma3_layer10_v8_corpus_receipt,
    prepare_gemma3_layer10_v8_corpus,
)


_ROLES = (
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
)
_EXPECTED_ROLE_SIZES = {
    "calibration_a_fit": (256, 8),
    "calibration_a_selection": (128, 4),
    "calibration_a_guard": (128, 4),
}


def _local_sources() -> dict[str, Path]:
    sources = {
        "prompt_splits_path": corpus_module.DEFAULT_V8_PROMPT_PATH,
        "family_manifest_path": corpus_module.DEFAULT_V8_FAMILY_PATH,
        "corpus_audit_path": corpus_module.DEFAULT_V8_AUDIT_PATH,
        "generator_path": corpus_module.DEFAULT_V8_GENERATOR_PATH,
    }
    missing = tuple(path for path in sources.values() if not path.is_file())
    if missing:
        pytest.skip("private structured-strong-v8 sources are unavailable")
    return sources


def _outputs(root: Path) -> dict[str, Path]:
    return {
        "fit_output": root / "layer10-v8.fit.json",
        "selection_output": root / "layer10-v8.selection.json",
        "guard_output": root / "layer10-v8.guard.json",
        "corpus_output": root / "layer10-v8.corpus.json",
        "receipt_output": root / "layer10-v8.receipt.json",
    }


def _role_paths(outputs: dict[str, Path]) -> dict[str, Path]:
    return {
        "calibration_a_fit": outputs["fit_output"],
        "calibration_a_selection": outputs["selection_output"],
        "calibration_a_guard": outputs["guard_output"],
    }


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for nested in value.values()
            for key in _all_mapping_keys(nested)
        }
    if isinstance(value, list):
        return {
            key
            for nested in value
            for key in _all_mapping_keys(nested)
        }
    return set()


def test_guard_family_ranking_is_order_independent() -> None:
    families = tuple(f"fresh-family-{index}" for index in range(8))

    forward = corpus_module._rank_guard_families(families)
    backward = corpus_module._rank_guard_families(tuple(reversed(families)))

    assert forward == backward
    assert len(forward) == len(set(forward)) == 8
    assert set(forward) == set(families)


def test_preparation_is_deterministic_family_disjoint_and_prompt_free(
    tmp_path: Path,
) -> None:
    sources = _local_sources()
    first_outputs = _outputs(tmp_path / "first")
    second_outputs = _outputs(tmp_path / "second")

    first_receipt = prepare_gemma3_layer10_v8_corpus(
        **sources,
        **first_outputs,
    )
    second_receipt = prepare_gemma3_layer10_v8_corpus(
        **sources,
        **second_outputs,
    )

    assert first_receipt == second_receipt
    assert all(
        first_outputs[name].read_bytes() == second_outputs[name].read_bytes()
        for name in first_outputs
    )
    assert first_receipt["derivation"] == {
        "source_role_id": "calibration_a",
        "guard_split_policy_id": (
            "domain_sha256_rank_first4_selection_last4_guard"
        ),
        "guard_split_domain_sha256": hashlib.sha256(
            corpus_module._GUARD_FAMILY_RANK_DOMAIN
        ).hexdigest(),
        "source_example_count": 512,
        "source_family_count": 16,
        "roles_cover_source_exactly": True,
        "role_family_overlap_count": 0,
        "role_prompt_identity_overlap_count": 0,
    }
    assert first_receipt["heldout"] == {
        "role_ids": ("calibration_b", "validation", "test"),
        "roles_materialized": False,
        "roles_exported": False,
        "roles_tokenized": False,
        "roles_model_evaluated": False,
    }
    assert first_receipt["safety"] == {
        "contains_prompt_text": False,
        "contains_family_ids": False,
        "contains_token_ids": False,
        "contains_model_outputs": False,
        "source_safe": True,
    }

    role_payloads = {
        role: json.loads(path.read_bytes())
        for role, path in _role_paths(first_outputs).items()
    }
    for role, payload in role_payloads.items():
        examples, families = _EXPECTED_ROLE_SIZES[role]
        assert payload["role"] == role
        assert len(payload["prompts"]) == examples
        assert len(payload["family_ids"]) == examples
        assert len(set(payload["family_ids"])) == families
        assert first_receipt["roles"][role]["example_count"] == examples
        assert first_receipt["roles"][role]["family_count"] == families

    prompt_sets = tuple(
        set(role_payloads[role]["prompts"]) for role in _ROLES
    )
    family_sets = tuple(
        set(role_payloads[role]["family_ids"]) for role in _ROLES
    )
    assert all(
        not left & right
        for index, left in enumerate(prompt_sets)
        for right in prompt_sets[index + 1 :]
    )
    assert all(
        not left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    )
    assert len(set().union(*prompt_sets)) == 512
    assert len(set().union(*family_sets)) == 16

    receipt_bytes = first_outputs["receipt_output"].read_bytes()
    receipt_payload = json.loads(receipt_bytes)
    assert "prompts" not in _all_mapping_keys(receipt_payload)
    assert "family_ids" not in _all_mapping_keys(receipt_payload)
    for payload in role_payloads.values():
        assert all(
            prompt.encode("utf-8") not in receipt_bytes
            for prompt in payload["prompts"]
        )
        assert all(
            family.encode("utf-8") not in receipt_bytes
            for family in payload["family_ids"]
        )
    supplied_receipt_sha256 = receipt_payload.pop("receipt_sha256")
    assert supplied_receipt_sha256 == corpus_module._domain_sha256(
        corpus_module._RECEIPT_DOMAIN,
        receipt_payload,
    )

    corpus_receipt = first_receipt["corpus"]
    loaded = load_gemma3_l3_l4_progressive_a_corpus(
        first_outputs["corpus_output"],
        role_input_paths=_role_paths(first_outputs),
        expected_artifact_sha256=corpus_receipt["artifact_sha256"],
        tokenizer_contract=corpus_module._tokenizer_contract(),
    )
    assert loaded.guard_opened is False
    assert loaded.guard_consumed is False
    for role in _ROLES:
        examples, families = _EXPECTED_ROLE_SIZES[role]
        preclaim = loaded.preclaim_view(role)
        assert preclaim.example_count == examples
        assert len(preclaim.family_ids) == families
    assert loaded.guard_opened is False
    assert loaded.guard_consumed is False


def test_exact_source_hash_tamper_fails_before_writing_outputs(
    tmp_path: Path,
) -> None:
    sources = _local_sources()
    tampered_prompt_source = tmp_path / "tampered-prompts.json"
    tampered_prompt_source.write_bytes(
        sources["prompt_splits_path"].read_bytes() + b"\n"
    )
    outputs = _outputs(tmp_path / "outputs")

    with pytest.raises(
        Gemma3Layer10V8CorpusIntegrityError,
        match="prompt source exact file SHA-256 mismatch",
    ):
        prepare_gemma3_layer10_v8_corpus(
            **{
                **sources,
                "prompt_splits_path": tampered_prompt_source,
            },
            **outputs,
        )

    assert all(not path.exists() for path in outputs.values())


def test_existing_output_refuses_overwrite_before_opening_sources(
    tmp_path: Path,
) -> None:
    outputs = _outputs(tmp_path / "outputs")
    sentinel = b"keep-the-existing-private-role"
    outputs["fit_output"].parent.mkdir(parents=True)
    outputs["fit_output"].write_bytes(sentinel)
    missing_sources = {
        "prompt_splits_path": tmp_path / "missing-prompts.json",
        "family_manifest_path": tmp_path / "missing-families.json",
        "corpus_audit_path": tmp_path / "missing-audit.json",
        "generator_path": tmp_path / "missing-generator.py",
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_gemma3_layer10_v8_corpus(
            **missing_sources,
            **outputs,
        )

    assert outputs["fit_output"].read_bytes() == sentinel
    assert all(
        not path.exists()
        for name, path in outputs.items()
        if name != "fit_output"
    )


def test_receipt_loader_rejects_rehashed_nested_authority_tamper(
    tmp_path: Path,
) -> None:
    sources = _local_sources()
    outputs = _outputs(tmp_path / "outputs")
    prepare_gemma3_layer10_v8_corpus(**sources, **outputs)

    loaded = load_gemma3_layer10_v8_corpus_receipt(
        outputs["receipt_output"]
    )
    assert loaded["source"]["source_corpus_id"] == "structured-strong-v8"

    loaded["source"]["source_model_or_tokenizer_accessed"] = True
    payload = {
        key: value for key, value in loaded.items() if key != "receipt_sha256"
    }
    loaded["receipt_sha256"] = corpus_module._domain_sha256(
        corpus_module._RECEIPT_DOMAIN,
        payload,
    )
    outputs["receipt_output"].write_bytes(
        corpus_module._canonical_json_bytes(loaded)
    )

    with pytest.raises(
        Gemma3Layer10V8CorpusIntegrityError,
        match="receipt integrity check failed",
    ):
        load_gemma3_layer10_v8_corpus_receipt(outputs["receipt_output"])
