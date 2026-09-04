from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from fisher_graph import fisher_compact_message_completion_probe_v2 as probe


ROOT = Path(__file__).resolve().parents[1]

_F19_VECTOR = (
    "ff" * 32,
    "00" * 32,
    "0123456789abcdef" * 4,
)
_F19_VECTOR_SHA256 = (
    "8f68ab4056f5e90f853843393432885ef90a22648976f66dd052beeba0ff286b"
)
_NORMALIZATION_INPUT = "  Cafe\u0301\r\nLine  2!\r"
_NORMALIZED_SEMANTIC_BYTES = (
    b'{"messages":[{"content":"  Caf\xc3\xa9\\nLine  2!\\n","role":"user"}]}'
)
_NORMALIZED_CONTENT_SHA256 = (
    "3f2cd3fe46935af2bb0acd9485ea629f07d6b84057f92ea36cadce221394fe4a"
)
_G18_PARAGRAPH = (
    "The protocol completed its fail-closed corpus-provenance preflight. "
    "Corpus D does not satisfy the frozen strict metadata-only eligibility "
    "policy because its authoritative provenance records semantic parsing and "
    "tokenization access. With B and C already scientifically used, no eligible "
    "independent tune and development-confirm source remains. The campaign "
    "therefore terminated before source sealing, capture, model execution, or "
    "scientific evaluation."
)


def _minimal_receipt(
    *, semantic_parse_access: object, tokenizer_access: object
) -> dict[str, object]:
    return {
        "authoring_access_receipt": {
            "semantic_parse_access": {
                "D_response_validation": semantic_parse_access,
            },
            "tokenizer_access": {
                "D_response_validation": tokenizer_access,
            },
        }
    }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_hashed_json(path: Path, value: dict[str, object]) -> None:
    probe._atomic_exclusive_write(
        path,
        probe._canonical_json_bytes(value) + b"\n",
    )


def _legacy_reserve_receipt(
    synthetic_p0: dict[str, object],
    *,
    semantic_spec_sha256: str,
    timestamp: str,
) -> dict[str, object]:
    records = synthetic_p0["records"]
    assert isinstance(records, list)
    payload = probe._reserve_retirement_payload(
        reserve_file_sha256=synthetic_p0["reserve_file_sha256"],
        records=records,
        content_set_sha256=synthetic_p0["content_set_sha256"],
        canonicalization_artifact_sha256=semantic_spec_sha256,
        implementation_sha256=probe.LEGACY_RETIREMENT_IMPLEMENTATION_SHA256,
        claim_id="ef" * 32,
        claim_timestamp=timestamp,
    )
    for field in (
        "pre_decode_claim_id",
        "pre_decode_claim_timestamp",
        "pre_decode_claim_state_at_publication",
        "receipt_published_before_registry_retirement",
    ):
        del payload[field]
    payload["exposure_timestamp_basis"] = "first_local_retirement_record"
    return probe._self_hashed(
        {
            "schema": (
                "fisher_graph.fisher_compact_message_completion."
                "reserve_retired_exposed.v1"
            ),
            "format_version": probe.FORMAT_VERSION,
            "protocol_id": probe.PROTOCOL_ID,
            "campaign_id": probe.CAMPAIGN_ID,
            **payload,
        }
    )


@pytest.fixture
def synthetic_p0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Install a no-I/O synthetic reserve boundary for every runner test."""

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    reserve_file_sha256 = hashlib.sha256(
        b"synthetic-reserve-file-identity-only"
    ).hexdigest()
    pins = dict(probe.PINNED_ROLE_FILE_SHA256S)
    pins["reserve_unopened"] = reserve_file_sha256
    monkeypatch.setattr(probe, "PINNED_ROLE_FILE_SHA256S", pins)
    monkeypatch.setattr(
        probe,
        "load_receipt",
        lambda _path: _minimal_receipt(
            semantic_parse_access=True,
            tokenizer_access=True,
        ),
    )
    records = [
        {
            "content_sha256": probe.canonical_content_sha256(prompt),
            "analysis_unit_commitment": f"{index + 1:064x}",
            "template_lineage_commitment": f"{index + 11:064x}",
            "template_id_commitment": f"{index + 21:064x}",
            "semantic_instance_commitment": f"{index + 1:064x}",
            "family_commitment": f"{index + 31:064x}",
        }
        for index, prompt in enumerate(("synthetic alpha", "synthetic beta"))
    ]
    records.sort(key=lambda record: record["content_sha256"])
    content_set_sha256 = probe.content_set_sha256(
        [record["content_sha256"] for record in records]
    )
    state = {"canonicalize_calls": 0}

    def canonicalize_without_raw_access(_repository_root: Path):
        state["canonicalize_calls"] += 1
        return reserve_file_sha256, [dict(record) for record in records], content_set_sha256

    monkeypatch.setattr(
        probe, "_canonicalize_reserve", canonicalize_without_raw_access
    )
    return {
        "repository_root": repository_root,
        "reserve_file_sha256": reserve_file_sha256,
        "records": records,
        "content_set_sha256": content_set_sha256,
        "state": state,
    }


@pytest.mark.parametrize(
    ("semantic_parse_access", "tokenizer_access"),
    ((True, False), (False, True), (True, True)),
)
def test_classification_uses_only_exact_nested_d_access_booleans(
    semantic_parse_access: bool,
    tokenizer_access: bool,
) -> None:
    receipt = _minimal_receipt(
        semantic_parse_access=semantic_parse_access,
        tokenizer_access=tokenizer_access,
    )
    # Contradictory flattened fields are irrelevant. H1 permits no fallback to
    # them and requires the two exact nested paths above.
    receipt["semantic_parse_access"] = False
    receipt["tokenizer_access"] = False
    receipt["model_access"] = object()

    classification = probe.classify_corpus_roles(receipt)
    d_role = classification["corpora"]["D_response_validation"]

    assert d_role["classification"] == "preoutcome_executable_access"
    assert d_role["frozen_access_flags"] == {
        "semantic_parse_access": semantic_parse_access,
        "tokenizer_access": tokenizer_access,
    }
    assert "model_access" not in d_role["frozen_access_flags"]
    assert d_role["eligible_for_fit"] is False
    assert d_role["eligible_for_tune"] is False
    assert d_role["eligible_for_development_confirm"] is False
    assert d_role["eligible_for_guard"] is False
    reserve_role = classification["corpora"]["reserve_unopened"]
    assert reserve_role["classification"] == "retired_exposed"
    assert all(
        reserve_role[field] is False
        for field in (
            "eligible_for_fit",
            "eligible_for_tune",
            "eligible_for_development_confirm",
            "eligible_for_guard",
        )
    )

    decision = probe.build_preflight_decision(classification)
    assert decision["can_construct_fit"] is True
    assert decision["can_construct_tune"] is False
    assert decision["can_construct_development_confirm"] is False
    assert decision["campaign_can_start"] is False
    assert decision["reason"] == (
        "No eligible independent tune/development-confirm corpus"
    )
    assert decision["terminal_state"] == probe.CURRENT_TERMINAL_STATE


@pytest.mark.parametrize(
    "receipt",
    (
        {
            "authoring_access_receipt": {
                "tokenizer_access": {"D_response_validation": True},
            },
        },
        {
            "authoring_access_receipt": {
                "semantic_parse_access": {"D_response_validation": True},
            },
        },
        {
            "semantic_parse_access": True,
            "tokenizer_access": True,
        },
    ),
)
def test_missing_or_flattened_d_access_fields_fail_closed(
    receipt: dict[str, object],
) -> None:
    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="required frozen provenance field.*is absent",
    ):
        probe.classify_corpus_roles(receipt)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("semantic_parse_access", 1),
        ("semantic_parse_access", "true"),
        ("semantic_parse_access", None),
        ("tokenizer_access", 0),
        ("tokenizer_access", []),
        ("tokenizer_access", {"value": True}),
    ),
)
def test_non_boolean_nested_d_access_fields_fail_closed(
    field: str,
    bad_value: object,
) -> None:
    receipt = _minimal_receipt(
        semantic_parse_access=True,
        tokenizer_access=True,
    )
    access = receipt["authoring_access_receipt"]
    assert isinstance(access, dict)
    field_values = access[field]
    assert isinstance(field_values, dict)
    field_values["D_response_validation"] = bad_value

    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="is not an exact Boolean",
    ):
        probe.classify_corpus_roles(receipt)


def test_metadata_only_d_requires_a_new_non_p0_campaign() -> None:
    receipt = _minimal_receipt(
        semantic_parse_access=False,
        tokenizer_access=False,
    )

    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="separately versioned non-P0 campaign",
    ):
        probe.classify_corpus_roles(receipt)


def test_content_set_sha256_matches_exact_f19_vector() -> None:
    assert probe.content_set_sha256(_F19_VECTOR) == _F19_VECTOR_SHA256
    assert probe.content_set_sha256(tuple(reversed(_F19_VECTOR))) == (
        _F19_VECTOR_SHA256
    )

    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="duplicate canonical content hashes",
    ):
        probe.content_set_sha256((_F19_VECTOR[0], _F19_VECTOR[0]))
    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="lowercase SHA-256",
    ):
        probe.content_set_sha256(("AA" * 32,))


def test_semantic_prompt_canonicalization_is_nfc_and_newline_only() -> None:
    assert probe.canonical_semantic_prompt_bytes(_NORMALIZATION_INPUT) == (
        _NORMALIZED_SEMANTIC_BYTES
    )
    assert probe.canonical_content_sha256(_NORMALIZATION_INPUT) == (
        _NORMALIZED_CONTENT_SHA256
    )

    # NFC-equivalent inputs collide, while case, punctuation, and repeated or
    # boundary whitespace remain semantic and therefore do not.
    composed = "  Caf\u00e9\nLine  2!\n"
    assert probe.canonical_content_sha256(composed) == (
        _NORMALIZED_CONTENT_SHA256
    )
    assert probe.canonical_content_sha256("  caf\u00e9\nLine  2!\n") != (
        _NORMALIZED_CONTENT_SHA256
    )
    assert probe.canonical_content_sha256("Caf\u00e9\nLine 2\n") != (
        _NORMALIZED_CONTENT_SHA256
    )


def test_import_is_model_and_tokenizer_free() -> None:
    runner_source = (
        ROOT
        / "src"
        / "fisher_graph"
        / "fisher_compact_message_completion_probe_v2.py"
    )
    tree = ast.parse(runner_source.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint(
        {
            "mlx",
            "numpy",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "torch",
            "transformers",
        }
    )

    script = textwrap.dedent(
        """
        import sys
        import fisher_graph_compact_message_completion_probe_v2 as shim

        forbidden_roots = {
            "numpy",
            "fisher_graph",
            "tokenizers",
            "torch",
            "transformers",
        }

        def assert_clean(stage):
            loaded = sorted(
                name for name in sys.modules
                if name.partition(".")[0] in forbidden_roots
            )
            if loaded:
                raise SystemExit(
                    stage + " forbidden imports: " + ",".join(loaded)
                )

        assert_clean("import")
        try:
            shim.main(["--help"])
        except SystemExit as exc:
            if exc.code != 0:
                raise
        else:
            raise SystemExit("--help did not terminate through argparse")
        assert_clean("--help")
        """
    )
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not prior_pythonpath
        else source_root + os.pathsep + prior_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _run_synthetic_p0(
    synthetic_p0: dict[str, object], tmp_path: Path
) -> tuple[Path, Path, dict[str, object]]:
    output_dir = tmp_path / "p0"
    registry_path = tmp_path / "local-development-registry.sqlite3"
    result = probe.run_current_p0(
        repository_root=synthetic_p0["repository_root"],
        output_dir=output_dir,
        registry_path=registry_path,
    )
    return output_dir, registry_path, result


def test_current_p0_is_safe_complete_immutable_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    original_retire = probe._retire_reserve_in_local_registry
    observed_order: list[str] = []

    def assert_receipt_precedes_registry(**kwargs):
        receipt_path = kwargs["receipt_path"]
        assert receipt_path.is_file()
        if kwargs["registry_path"].exists():
            with closing(sqlite3.connect(kwargs["registry_path"])) as connection, connection:
                assert not connection.execute(
                    "SELECT 1 FROM content_sets LIMIT 1"
                ).fetchone()
        observed_order.append("receipt_before_registry")
        return original_retire(**kwargs)

    monkeypatch.setattr(
        probe, "_retire_reserve_in_local_registry", assert_receipt_precedes_registry
    )
    output_dir, registry_path, result = _run_synthetic_p0(
        synthetic_p0, tmp_path
    )
    assert observed_order == ["receipt_before_registry"]
    assert result["verified"] is True
    assert result["contract_conformant"] is True
    assert result["legacy_verification"] is False
    assert result["verification_mode"] == "contract_v2_conformant"
    assert result["contract_invalidation_reason"] is None
    assert result["terminal_state"] == probe.CURRENT_TERMINAL_STATE
    assert result["raw_reserve_reopened"] is False
    assert synthetic_p0["state"]["canonicalize_calls"] == 1

    assert {path.name for path in output_dir.iterdir()} == set(
        probe.P0_ARTIFACT_ORDER
    )
    manifest = _load_json(output_dir / "final_experiment_manifest.json")
    assert manifest["artifact_write_order"] == list(probe.P0_ARTIFACT_ORDER)
    inventory = manifest["artifact_inventory"]
    assert [entry["path"] for entry in inventory] == list(
        probe.P0_ARTIFACT_ORDER[:-1]
    )
    for entry in inventory:
        encoded = (output_dir / entry["path"]).read_bytes()
        assert entry["byte_count"] == len(encoded)
        assert entry["artifact_byte_sha256"] == hashlib.sha256(encoded).hexdigest()

    terminal = _load_json(output_dir / "terminal_state.json")
    assert terminal["unreached_prefixes"] == [
        f"P{index}" for index in range(1, 13)
    ]
    classification = _load_json(output_dir / "corpus_role_classification.json")
    reserve_role = classification["corpora"]["reserve_unopened"]
    assert reserve_role["classification"] == "retired_exposed"
    assert reserve_role["irreversible"] is True
    assert all(
        reserve_role[field] is False
        for field in (
            "eligible_for_fit",
            "eligible_for_tune",
            "eligible_for_development_confirm",
            "eligible_for_guard",
        )
    )

    report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    normalized_report = " ".join(report.split())
    assert _G18_PARAGRAPH in normalized_report
    for exact_sentence in (
        "No raw D content was inspected by this campaign.",
        "No experiment was run.",
        "No model candidate was fitted.",
        "No development or guard verdict exists.",
    ):
        assert exact_sentence in report

    retirement = _load_json(output_dir / "reserve_retired_exposed.json")
    assert retirement["receipt_published_before_registry_retirement"] is True
    assert retirement["pre_decode_claim_state_at_publication"] == "claimed"
    assert retirement["raw_content_in_receipt"] is False
    assert retirement["canonical_content_set_sha256"] == (
        probe.content_set_sha256(retirement["canonical_per_item_sha256s"])
    )

    with closing(sqlite3.connect(registry_path)) as connection, connection:
        assert connection.execute(
            "SELECT value FROM registry_meta WHERE key = 'schema_version'"
        ).fetchone() == (str(probe.REGISTRY_SCHEMA_VERSION),)
        assert connection.execute(
            "SELECT state, expected_reserve_file_sha256, "
            "receipt_artifact_sha256 FROM reserve_decode_claims"
        ).fetchone()[:2] == (
            "retired_exposed",
            synthetic_p0["reserve_file_sha256"],
        )
        with pytest.raises(sqlite3.DatabaseError, match="transition is invalid"):
            connection.execute(
                "UPDATE reserve_decode_claims "
                "SET expected_reserve_file_sha256 = ?",
                ("ff" * 32,),
            )
        connection.rollback()
        with pytest.raises(sqlite3.DatabaseError, match="irreversible"):
            connection.execute("UPDATE content_sets SET state = 'unused'")
        connection.rollback()
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            connection.execute("DELETE FROM content_items")
        connection.rollback()

    def reserve_reopen_forbidden(_repository_root: Path):
        raise AssertionError("a finalized rerun reopened the raw reserve")

    monkeypatch.setattr(probe, "_canonicalize_reserve", reserve_reopen_forbidden)
    rerun = probe.run_current_p0(
        repository_root=synthetic_p0["repository_root"],
        output_dir=output_dir,
        registry_path=registry_path,
    )
    assert rerun == result

    # Exact schema is authenticated, not merely the row values.
    with closing(sqlite3.connect(registry_path)) as connection, connection:
        connection.execute("DROP TRIGGER content_sets_no_delete")
    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="exact schema, constraints, or guards drifted",
    ):
        probe.verify_finalized_p0(
            output_dir,
            registry_path=registry_path,
            repository_root=synthetic_p0["repository_root"],
        )


def test_artifact_two_recovers_without_reopening_reserve_after_registry_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    output_dir = tmp_path / "p0"
    registry_path = tmp_path / "registry.sqlite3"
    original_retire = probe._retire_reserve_in_local_registry

    def crash_before_registry_commit(**kwargs):
        assert kwargs["receipt_path"].is_file()
        raise RuntimeError("synthetic crash after receipt publication")

    monkeypatch.setattr(
        probe, "_retire_reserve_in_local_registry", crash_before_registry_commit
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        probe.run_current_p0(
            repository_root=synthetic_p0["repository_root"],
            output_dir=output_dir,
            registry_path=registry_path,
        )
    present = {path.name for path in output_dir.iterdir() if not path.name.startswith(".")}
    assert present == set(probe.P0_ARTIFACT_ORDER[:2])
    assert synthetic_p0["state"]["canonicalize_calls"] == 1

    monkeypatch.setattr(
        probe, "_retire_reserve_in_local_registry", original_retire
    )

    def forbidden(_repository_root: Path):
        raise AssertionError("artifact #2 recovery reopened raw reserve")

    monkeypatch.setattr(probe, "_canonicalize_reserve", forbidden)
    result = probe.run_current_p0(
        repository_root=synthetic_p0["repository_root"],
        output_dir=output_dir,
        registry_path=registry_path,
    )
    assert result["contract_conformant"] is True


@pytest.mark.parametrize("prefix_length", range(2, 9))
def test_every_post_receipt_prefix_recovers_without_raw_reserve(
    prefix_length: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    output_dir, registry_path, _result = _run_synthetic_p0(
        synthetic_p0, tmp_path
    )
    for name in probe.P0_ARTIFACT_ORDER[prefix_length:]:
        (output_dir / name).unlink()

    def forbidden(_repository_root: Path):
        raise AssertionError(f"prefix {prefix_length} reopened raw reserve")

    monkeypatch.setattr(probe, "_canonicalize_reserve", forbidden)
    recovered = probe.run_current_p0(
        repository_root=synthetic_p0["repository_root"],
        output_dir=output_dir,
        registry_path=registry_path,
    )
    assert recovered["contract_conformant"] is True
    assert set(path.name for path in output_dir.iterdir()) == set(
        probe.P0_ARTIFACT_ORDER
    )


def test_uncertain_crash_before_receipt_fails_closed_without_second_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    output_dir = tmp_path / "p0"
    registry_path = tmp_path / "registry.sqlite3"
    original_write = probe._atomic_exclusive_write

    def crash_on_receipt(path: Path, payload: bytes) -> None:
        if path.name == "reserve_retired_exposed.json":
            raise RuntimeError("synthetic uncertain publication crash")
        original_write(path, payload)

    monkeypatch.setattr(probe, "_atomic_exclusive_write", crash_on_receipt)
    with pytest.raises(RuntimeError, match="uncertain publication"):
        probe.run_current_p0(
            repository_root=synthetic_p0["repository_root"],
            output_dir=output_dir,
            registry_path=registry_path,
        )
    assert synthetic_p0["state"]["canonicalize_calls"] == 1
    monkeypatch.setattr(probe, "_atomic_exclusive_write", original_write)
    with pytest.raises(
        probe.CompactMessageCompletionFinalizedError,
        match="refusing to reopen the raw reserve",
    ):
        probe.run_current_p0(
            repository_root=synthetic_p0["repository_root"],
            output_dir=output_dir,
            registry_path=registry_path,
        )
    assert synthetic_p0["state"]["canonicalize_calls"] == 1


def test_unbound_published_receipt_fails_closed_without_second_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    output_dir = tmp_path / "p0"
    registry_path = tmp_path / "registry.sqlite3"
    original_bind = probe._bind_published_reserve_receipt

    def crash_before_receipt_binding(**_kwargs):
        raise RuntimeError("synthetic crash before receipt binding")

    monkeypatch.setattr(
        probe, "_bind_published_reserve_receipt", crash_before_receipt_binding
    )
    with pytest.raises(RuntimeError, match="before receipt binding"):
        probe.run_current_p0(
            repository_root=synthetic_p0["repository_root"],
            output_dir=output_dir,
            registry_path=registry_path,
        )
    assert (output_dir / "reserve_retired_exposed.json").is_file()
    with closing(sqlite3.connect(registry_path)) as connection, connection:
        assert connection.execute(
            "SELECT state, receipt_artifact_sha256 FROM reserve_decode_claims"
        ).fetchone() == ("claimed", None)
    monkeypatch.setattr(probe, "_bind_published_reserve_receipt", original_bind)

    def forbidden(_repository_root: Path):
        raise AssertionError("unbound receipt recovery reopened raw reserve")

    monkeypatch.setattr(probe, "_canonicalize_reserve", forbidden)
    with pytest.raises(
        probe.CompactMessageCompletionFinalizedError,
        match="not durably bound.*refusing uncertain recovery",
    ):
        probe.run_current_p0(
            repository_root=synthetic_p0["repository_root"],
            output_dir=output_dir,
            registry_path=registry_path,
        )
    assert synthetic_p0["state"]["canonicalize_calls"] == 1


def test_predecode_claim_is_exclusive_under_concurrency(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite3"
    canonicalization_hash = "ab" * 32
    implementation_hash = "cd" * 32

    def acquire():
        return probe._acquire_reserve_decode_claim(
            registry_path=registry_path,
            expected_reserve_file_sha256="ef" * 32,
            canonicalization_artifact_sha256=canonicalization_hash,
            implementation_sha256=implementation_hash,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [future.exception() or future.result() for future in (
            executor.submit(acquire), executor.submit(acquire)
        )]
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert sum(isinstance(value, BaseException) for value in outcomes) == 1
    with closing(sqlite3.connect(registry_path)) as connection, connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM reserve_decode_claims"
        ).fetchone() == (1,)


def test_predecode_claim_rejects_a_receipt_for_a_different_reserve_identity(
    tmp_path: Path,
    synthetic_p0: dict[str, object],
) -> None:
    registry_path = tmp_path / "registry.sqlite3"
    receipt_path = tmp_path / "reserve_retired_exposed.json"
    semantic_spec = probe._self_hashed(probe._semantic_spec_payload())
    implementation_sha256 = probe._implementation_sha256()
    claim = probe._acquire_reserve_decode_claim(
        registry_path=registry_path,
        expected_reserve_file_sha256="ab" * 32,
        canonicalization_artifact_sha256=semantic_spec["artifact_sha256"],
        implementation_sha256=implementation_sha256,
    )
    records = synthetic_p0["records"]
    assert isinstance(records, list)
    retirement_payload = probe._reserve_retirement_payload(
        reserve_file_sha256=synthetic_p0["reserve_file_sha256"],
        records=records,
        content_set_sha256=synthetic_p0["content_set_sha256"],
        canonicalization_artifact_sha256=semantic_spec["artifact_sha256"],
        implementation_sha256=implementation_sha256,
        claim_id=claim["claim_id"],
        claim_timestamp=claim["claim_timestamp"],
    )
    receipt = probe._self_hashed(
        {
            "schema": (
                "fisher_graph.fisher_compact_message_completion."
                "reserve_retired_exposed.v1"
            ),
            "format_version": probe.FORMAT_VERSION,
            "protocol_id": probe.PROTOCOL_ID,
            "campaign_id": probe.CAMPAIGN_ID,
            **retirement_payload,
        }
    )
    _write_hashed_json(receipt_path, receipt)

    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="no matching durable decode claim",
    ):
        probe._bind_published_reserve_receipt(
            registry_path=registry_path,
            receipt_path=receipt_path,
            reserve_receipt=receipt,
        )
    with closing(sqlite3.connect(registry_path)) as connection, connection:
        assert connection.execute(
            "SELECT expected_reserve_file_sha256, state, "
            "receipt_artifact_sha256 FROM reserve_decode_claims"
        ).fetchone() == ("ab" * 32, "claimed", None)


def test_exhausted_real_reserve_gate_is_global_across_fresh_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    canonicalize_calls = 0

    def forbidden(_repository_root: Path):
        nonlocal canonicalize_calls
        canonicalize_calls += 1
        raise AssertionError("exhausted real reserve reached JSON decode")

    monkeypatch.setattr(probe, "_canonicalize_reserve", forbidden)
    for index in range(2):
        output_dir = tmp_path / f"p0-{index}"
        registry_path = tmp_path / f"registry-{index}.sqlite3"
        with pytest.raises(
            probe.CompactMessageCompletionFinalizedError,
            match="real-reserve decode authorization is exhausted",
        ):
            probe.run_current_p0(
                repository_root=repository_root,
                output_dir=output_dir,
                registry_path=registry_path,
            )
        assert not registry_path.exists()
    assert canonicalize_calls == 0


def test_exhausted_real_reserve_gate_defends_canonicalizer_before_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_read(_path: Path, *, label: str):
        raise AssertionError(f"exhausted real reserve was read as {label}")

    monkeypatch.setattr(probe, "_read_regular_bytes", forbidden_read)
    with pytest.raises(
        probe.CompactMessageCompletionFinalizedError,
        match="real-reserve decode authorization is exhausted",
    ):
        probe._canonicalize_reserve(tmp_path)


def test_orphan_same_directory_temps_are_recovered_safely(tmp_path: Path) -> None:
    output_dir = tmp_path / "p0"
    output_dir.mkdir()
    orphan = output_dir / ".semantic_canonicalization_spec.json.tmp.orphan"
    orphan.write_bytes(b"uncommitted")
    orphan.chmod(0o600)
    assert probe._validate_existing_prefix(output_dir) == ()
    assert not orphan.exists()

    unsafe = output_dir / ".unknown.json.tmp.orphan"
    unsafe.write_bytes(b"unknown")
    unsafe.chmod(0o600)
    with pytest.raises(
        probe.CompactMessageCompletionFinalizedError,
        match="unsafe temporary artifact",
    ):
        probe._validate_existing_prefix(output_dir)


def test_hashed_artifact_requires_exact_canonical_json_bytes(tmp_path: Path) -> None:
    value = probe._self_hashed({"schema": "synthetic", "value": 1})
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="exact canonical JSON bytes plus one newline",
    ):
        probe._load_hashed_artifact(path)


def test_registry_exact_schema_rejects_an_unexpected_view(
    tmp_path: Path,
    synthetic_p0: dict[str, object],
) -> None:
    output_dir, registry_path, _result = _run_synthetic_p0(
        synthetic_p0, tmp_path
    )
    with closing(sqlite3.connect(registry_path)) as connection, connection:
        connection.execute(
            "CREATE VIEW unexpected_registry_view AS SELECT key FROM registry_meta"
        )
    with pytest.raises(
        probe.CompactMessageCompletionIntegrityError,
        match="exact schema, constraints, or guards drifted",
    ):
        probe.verify_finalized_p0(
            output_dir,
            registry_path=registry_path,
            repository_root=synthetic_p0["repository_root"],
        )


def test_synthetic_legacy_v1_finalized_output_is_audit_only_and_never_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_p0: dict[str, object],
) -> None:
    repository_root = synthetic_p0["repository_root"]
    assert isinstance(repository_root, Path)
    output_dir = tmp_path / "legacy-p0"
    registry_path = tmp_path / "legacy-registry.sqlite3"
    timestamp = "2026-09-03T00:00:00.000000Z"
    records = synthetic_p0["records"]
    assert isinstance(records, list)

    semantic_spec = probe._self_hashed(probe._semantic_spec_payload())
    reserve_receipt = _legacy_reserve_receipt(
        synthetic_p0,
        semantic_spec_sha256=semantic_spec["artifact_sha256"],
        timestamp=timestamp,
    )

    with closing(
        sqlite3.connect(registry_path, isolation_level=None)
    ) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        for statement in probe._REGISTRY_V1_TABLE_STATEMENTS:
            connection.execute(statement)
        probe._insert_or_verify_meta(
            connection,
            "schema_version",
            str(probe.LEGACY_REGISTRY_SCHEMA_VERSION),
        )
        probe._insert_or_verify_meta(
            connection,
            "authority",
            "non_authoritative_host_local_development",
        )
        probe._insert_or_verify_meta(
            connection,
            f"canonicalization_version:{probe.CANONICALIZATION_VERSION}",
            "frozen",
        )
        for role in probe.ORDERED_ROLES:
            probe._insert_or_verify_corpus(
                connection,
                role=role,
                recorded_at=timestamp,
            )
        connection.execute(
            """
            INSERT INTO content_sets(
                content_set_sha256, state, canonicalization_version,
                source_file_sha256, item_count, first_transition_timestamp
            ) VALUES (?, 'retired_exposed', ?, ?, ?, ?)
            """,
            (
                synthetic_p0["content_set_sha256"],
                probe.CANONICALIZATION_VERSION,
                synthetic_p0["reserve_file_sha256"],
                len(records),
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO reserve_retirements(
                reserve_path, reserve_file_sha256, content_set_sha256,
                canonicalization_version, canonicalization_artifact_sha256,
                item_count, exposure_timestamp, exposure_timestamp_basis,
                retirement_implementation_sha256, transition_from,
                transition_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unopened', 'retired_exposed')
            """,
            (
                str(probe.DEFAULT_ROLE_CORPORA["reserve_unopened"]),
                synthetic_p0["reserve_file_sha256"],
                synthetic_p0["content_set_sha256"],
                probe.CANONICALIZATION_VERSION,
                semantic_spec["artifact_sha256"],
                len(records),
                timestamp,
                "first_local_retirement_record",
                probe.LEGACY_RETIREMENT_IMPLEMENTATION_SHA256,
            ),
        )
        for record in records:
            probe._retire_content_item(
                connection,
                record=record,
                reserve_file_sha256=synthetic_p0["reserve_file_sha256"],
                recorded_at=timestamp,
            )
        snapshot = probe._registry_snapshot(
            connection,
            content_set_sha256=synthetic_p0["content_set_sha256"],
        )
        connection.execute("COMMIT")
    registry_path.chmod(0o600)

    registry_status = probe._self_hashed(
        probe._registry_status_payload(
            registry_path=registry_path,
            snapshot=snapshot,
            registry_schema_version=probe.LEGACY_REGISTRY_SCHEMA_VERSION,
        )
    )
    classification = probe._self_hashed(
        probe._legacy_classify_corpus_roles(
            _minimal_receipt(
                semantic_parse_access=True,
                tokenizer_access=True,
            )
        )
    )
    decision = probe._self_hashed(
        probe.build_preflight_decision(classification)
    )
    terminal = probe._self_hashed(
        probe._terminal_payload(
            classification=classification,
            decision=decision,
        )
    )

    for name, value in (
        (probe.P0_ARTIFACT_ORDER[0], semantic_spec),
        (probe.P0_ARTIFACT_ORDER[1], reserve_receipt),
        (probe.P0_ARTIFACT_ORDER[2], registry_status),
        (probe.P0_ARTIFACT_ORDER[3], classification),
        (probe.P0_ARTIFACT_ORDER[4], decision),
    ):
        _write_hashed_json(output_dir / name, value)
    probe._write_text_artifact(
        output_dir / probe.P0_ARTIFACT_ORDER[5],
        probe._commands_markdown(
            repository_root=repository_root,
            output_dir=output_dir,
            registry_path=registry_path,
        ),
    )
    _write_hashed_json(output_dir / probe.P0_ARTIFACT_ORDER[6], terminal)
    probe._write_text_artifact(
        output_dir / probe.P0_ARTIFACT_ORDER[7],
        probe._final_report_markdown(
            reserve=reserve_receipt,
            registry_status=registry_status,
            classification=classification,
            decision=decision,
        ),
    )
    probe._write_json_artifact(
        output_dir / probe.P0_ARTIFACT_ORDER[8],
        probe._final_manifest_payload(output_dir),
    )

    def forbidden(_repository_root: Path):
        raise AssertionError("legacy audit attempted to decode a reserve")

    monkeypatch.setattr(probe, "_canonicalize_reserve", forbidden)
    result = probe.verify_finalized_p0(
        output_dir,
        registry_path=registry_path,
        repository_root=repository_root,
    )
    assert result["verified"] is True
    assert result["raw_reserve_reopened"] is False
    assert result["contract_conformant"] is False
    assert result["legacy_verification"] is True
    assert result["verification_mode"] == "legacy_v1_byte_audit_nonconformant"
    assert result["contract_invalidation_reason"] == (
        probe.LEGACY_CONTRACT_INVALIDATION_REASON
    )
    assert probe.run_current_p0(
        repository_root=repository_root,
        output_dir=output_dir,
        registry_path=registry_path,
    ) == result


def test_legacy_implementation_pin_is_explicit(
    synthetic_p0: dict[str, object],
) -> None:
    assert probe.LEGACY_RETIREMENT_IMPLEMENTATION_SHA256 == (
        "d06a73933cc405e26f790673979501228e77f17102117aab0389b0fa2a773647"
    )
    assert probe.LEGACY_CONTRACT_INVALIDATION_REASON == (
        "legacy_v1_registry_commit_preceded_reserve_receipt_publication"
    )
    semantic_spec = probe._self_hashed(probe._semantic_spec_payload())
    receipt = _legacy_reserve_receipt(
        synthetic_p0,
        semantic_spec_sha256=semantic_spec["artifact_sha256"],
        timestamp="2026-09-03T00:00:00.000000Z",
    )
    assert probe._validate_reserve_receipt(
        receipt,
        canonicalization_artifact_sha256=semantic_spec["artifact_sha256"],
    ) is True
