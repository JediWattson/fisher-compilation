from __future__ import annotations

import ast
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from fisher_graph import gemma3_stable_mutual_knn_confirmation_panel as panel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPOSITORY_ROOT / panel.DEFAULT_CORPUS
MANIFEST = REPOSITORY_ROOT / panel.DEFAULT_PANEL
RECEIPT = REPOSITORY_ROOT / panel.DEFAULT_RECEIPT


def _loaded() -> panel.LoadedPanel:
    return panel.load_panel(CORPUS, MANIFEST)


def _canonical_write(path: Path, value: object) -> None:
    path.write_bytes(panel._canonical_json_bytes(value))


def test_frozen_panel_is_balanced_length_bucketed_and_one_shot() -> None:
    loaded = _loaded()
    records = panel.confirmation_prompt_records(loaded)

    assert len(records) == 60
    assert len(panel.confirmation_prompts(loaded)) == 60
    assert all(record.role == panel.CONFIRMATION_ROLE for record in records)
    assert Counter(record.semantic_family for record in records) == Counter(
        {family: 10 for family in panel.SEMANTIC_FAMILIES}
    )
    assert Counter(record.target_valid_tokens for record in records) == Counter(
        {length: 6 for length in panel.EXACT_TARGET_LENGTHS}
    )
    assert Counter(record.kind for record in records) == Counter(
        {
            "short_control": 18,
            "boundary_probe": 12,
            "transition_probe": 12,
            "beyond_window_probe": 18,
        }
    )
    assert tuple(record.target_valid_tokens for record in records) == tuple(
        length
        for length in panel.EXACT_TARGET_LENGTHS
        for _ in panel.SEMANTIC_FAMILIES
    )
    assert tuple(record.semantic_family for record in records) == (
        panel.SEMANTIC_FAMILIES * len(panel.EXACT_TARGET_LENGTHS)
    )
    assert Counter(record.correct_choice for record in records) == Counter(
        {0: 15, 1: 15, 2: 15, 3: 15}
    )

    for family, expected_coordinates in (
        panel.EXPECTED_CONFIRMATION_CHOICE_COORDINATES.items()
    ):
        family_records = tuple(
            record for record in records if record.semantic_family == family
        )
        assert tuple(
            record.correct_choice for record in family_records
        ) == expected_coordinates
        assert sorted(
            Counter(
                record.correct_choice for record in family_records
            ).values()
        ) == [2, 2, 3, 3]
        assert len(
            {record.correct_choice for record in family_records[:3]}
        ) == 3
        assert len(
            {record.correct_choice for record in family_records[3:6]}
        ) == 3
        assert len(
            {record.correct_choice for record in family_records[7:]}
        ) == 3

    context_records = {
        "short": tuple(
            record
            for record in records
            if record.target_valid_tokens in panel.SHORT_CONTROL_LENGTHS
        ),
        "boundary": tuple(
            record
            for record in records
            if record.target_valid_tokens
            in (*panel.BOUNDARY_LENGTHS, *panel.TRANSITION_LENGTHS)
        ),
        "long": tuple(
            record
            for record in records
            if record.target_valid_tokens in panel.BEYOND_WINDOW_LENGTHS
        ),
    }
    for context, expected_counts in (
        panel.EXPECTED_CHOICE_COORDINATE_CONTEXT_COUNTS.items()
    ):
        assert Counter(
            record.correct_choice for record in context_records[context]
        ) == Counter(expected_counts)

    for record in records:
        assert record.stratum_id is not None
        assert record.target_valid_tokens is not None
        assert record.input_ids_sha256 is not None
        assert len(record.choices) == 4
        assert len(set(record.choices)) == 4
        assert all(
            choice.startswith(" ")
            and not choice.startswith("  ")
            and choice == " " + choice.strip()
            for choice in record.choices
        )
        assert 0 <= record.correct_choice < 4
        assert record.supports_correct_choice_index == record.correct_choice
        assert record.answer_rationale.strip()
        assert len(record.answer_rationale_sha256) == 64
        assert record.choice_token_ids is not None
        assert len(set(record.choice_token_ids)) == 4
        assert (
            record.choices[record.correct_choice].strip().casefold()
            not in panel._WORD_RE.findall(
                panel._answer_specific_text(
                    record.semantic_family,
                    record.text,
                ).casefold()
            )
        )

    long_records = [
        record
        for record in records
        if record.kind == "beyond_window_probe"
    ]
    assert len(long_records) == 18
    visible_suffixes: dict[str, set[str]] = {
        family: set() for family in panel.SEMANTIC_FAMILIES
    }
    for record in long_records:
        evidence = record.answer_evidence
        assert evidence is not None
        assert record.target_valid_tokens in panel.BEYOND_WINDOW_LENGTHS
        assert evidence["char_start"] == 0
        assert evidence["token_start"] == 1
        assert evidence["token_end_inclusive"] > 1
        assert str(evidence["text"]).endswith(".")
        assert record.text.startswith(str(evidence["text"]))
        assert evidence["final_query_token_index"] == (
            record.target_valid_tokens - 1
        )
        assert evidence["outside_final_query_sliding_window"] is True
        assert (
            evidence["token_end_inclusive"]
            < evidence["final_query_token_index"] - panel.SLIDING_WINDOW + 1
        )
        assert evidence["supports_correct_choice_index"] == (
            record.correct_choice
        )
        assert evidence["rationale_sha256"] == (
            record.answer_rationale_sha256
        )
        assert str(evidence["text"]).casefold() not in record.text[
            evidence["char_end_exclusive"] :
        ].casefold()
        suffix_start = record.text.index(
            panel.HARD_LONG_VISIBLE_SUFFIX_PREFIX
        )
        visible_suffixes[record.semantic_family].add(
            record.text[suffix_start:]
        )
    assert all(len(values) == 1 for values in visible_suffixes.values())

    transition_514 = tuple(
        record
        for record in records
        if record.target_valid_tokens == 514
    )
    assert len(transition_514) == 6
    assert all(record.kind == "transition_probe" for record in transition_514)
    assert all(record.stratum_id == "transition_514" for record in transition_514)
    assert all(record.answer_evidence is None for record in transition_514)

    assert loaded.manifest["role_policy"] == {
        panel.CONFIRMATION_ROLE: (
            "one_frozen_candidate_graph_pass_no_fit_no_selection_no_tuning"
        ),
        panel.RESERVE_ROLE: (
            "parse_hash_overlap_audit_only_no_tokenizer_no_model_access"
        ),
    }
    assert loaded.manifest["blind_authoring"] == panel.BLIND_AUTHORING_RECEIPT


def test_reserve_is_structural_only_and_absent_from_execution_api() -> None:
    loaded = _loaded()
    reserve = tuple(
        record
        for record in loaded.records
        if record.role == panel.RESERVE_ROLE
    )
    executable = panel.confirmation_prompt_records(loaded)

    assert len(reserve) == 12
    assert Counter(record.semantic_family for record in reserve) == Counter(
        {family: 2 for family in panel.SEMANTIC_FAMILIES}
    )
    assert all(record.kind == "short_control" for record in reserve)
    assert all(record.stratum_id is None for record in reserve)
    assert all(record.target_valid_tokens is None for record in reserve)
    assert all(record.input_ids_sha256 is None for record in reserve)
    assert all(record.choice_token_ids is None for record in reserve)
    assert all(record.answer_evidence is None for record in reserve)
    formal_reserve = tuple(
        record
        for record in reserve
        if record.semantic_family == "formal_logic"
    )
    assert len(formal_reserve) == 2
    assert all(
        set(record.choices) == set(panel.FORMAL_LOGIC_CHOICES)
        and record.text.endswith(panel.FORMAL_LOGIC_QUERY_SUFFIX)
        for record in formal_reserve
    )
    assert {record.prompt_id for record in executable}.isdisjoint(
        record.prompt_id for record in reserve
    )
    assert {record.text for record in executable}.isdisjoint(
        record.text for record in reserve
    )


def test_each_family_asks_its_declared_task_not_marker_retrieval() -> None:
    records = panel.load_default_confirmation_records(REPOSITORY_ROOT)
    query_suffix = {
        "observational_diagnosis": (
            "The component most strongly identified by the symptoms is the"
        ),
        "quantitative_reasoning": "The computed count is",
        "technical_explanation": (
            "The mechanism that best explains the observation is"
        ),
        "procedural_planning": "The next safe action is to",
        "formal_logic": panel.FORMAL_LOGIC_QUERY_SUFFIX,
        "causal_evidence_synthesis": (
            "The strongest interpretation of this evidence is"
        ),
    }
    correct_by_family: dict[str, set[str]] = {
        family: set() for family in panel.SEMANTIC_FAMILIES
    }
    for record in records:
        assert record.text.endswith(query_suffix[record.semantic_family])
        assert "opening marker" not in record.text.casefold()
        correct_by_family[record.semantic_family].add(
            record.choices[record.correct_choice].strip()
        )
        if record.answer_evidence is not None:
            assert (
                str(record.answer_evidence["text"]).casefold()
                != record.choices[record.correct_choice].strip().casefold()
            )
        if record.semantic_family == "formal_logic":
            assert set(record.choices) == set(panel.FORMAL_LOGIC_CHOICES)
            assert panel.FORMAL_LOGIC_QUERY_SUFFIX.endswith("The claim is")
            assert all(
                panel.FORMAL_LOGIC_QUERY_SUFFIX.casefold().count(
                    choice.strip().casefold()
                )
                == 1
                for choice in panel.FORMAL_LOGIC_CHOICES
            )
            assert not {
                "possible",
                "necessary",
                "consistent",
                "invalid",
                "impossible",
                "unknown",
            }.intersection(choice.strip() for choice in record.choices)
    assert all(len(values) >= 4 for values in correct_by_family.values())


def test_length_batch_iterator_restarts_each_exact_bucket() -> None:
    records = panel.load_default_confirmation_records(REPOSITORY_ROOT)
    batches = tuple(
        panel.iter_confirmation_length_batches(records, batch_size=4)
    )

    assert len(batches) == 20
    assert [len(batch) for batch in batches] == [4, 2] * 10
    assert sum(len(batch) for batch in batches) == 60
    for batch in batches:
        assert len({record.stratum_id for record in batch}) == 1
        assert len({record.target_valid_tokens for record in batch}) == 1
        assert all(record.role == panel.CONFIRMATION_ROLE for record in batch)

    with pytest.raises(ValueError, match="canonical, C-only"):
        tuple(
            panel.iter_confirmation_length_batches(
                tuple(reversed(records)),
                batch_size=4,
            )
        )
    with pytest.raises(ValueError, match="positive integer"):
        tuple(panel.iter_confirmation_length_batches(records, batch_size=0))


def test_token_authenticator_visits_all_C_rows_and_never_reserve() -> None:
    loaded = _loaded()
    confirmation = panel.confirmation_prompt_records(loaded)
    reserve_text = {
        record.text
        for record in loaded.records
        if record.role == panel.RESERVE_ROLE
    }
    by_text = {record.text: record for record in confirmation}
    by_choices = {record.choices: record for record in confirmation}
    visited: list[str] = []

    def vector_side_effect(
        _tokenizer: object,
        text: str,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        assert text not in reserve_text
        record = by_text[text]
        visited.append(record.prompt_id)
        assert record.target_valid_tokens is not None
        input_ids = tuple(
            [record.corpus_index, 1]
            + [0] * (record.target_valid_tokens - 2)
        )
        offsets = [(len(text), len(text))] * record.target_valid_tokens
        offsets[0] = (0, 0)
        if record.answer_evidence is not None:
            for index in range(
                int(record.answer_evidence["token_start"]),
                int(record.answer_evidence["token_end_inclusive"]) + 1,
            ):
                offsets[index] = (
                    int(record.answer_evidence["char_start"]),
                    int(record.answer_evidence["char_end_exclusive"]),
                )
        return input_ids, tuple(offsets)

    def digest_side_effect(input_ids: tuple[int, ...]) -> str:
        record = confirmation[input_ids[0]]
        assert record.input_ids_sha256 is not None
        return record.input_ids_sha256

    def choices_side_effect(
        _tokenizer: object,
        choices: tuple[str, str, str, str],
    ) -> tuple[int, int, int, int]:
        record = by_choices[tuple(choices)]
        assert record.choice_token_ids is not None
        return record.choice_token_ids

    tokenizer = mock.Mock(
        name_or_path=panel.MODEL_ID,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    with (
        mock.patch.object(
            panel,
            "_tokenizer_vector_and_offsets",
            side_effect=vector_side_effect,
        ),
        mock.patch.object(
            panel,
            "_token_stream_sha256",
            side_effect=digest_side_effect,
        ),
        mock.patch.object(
            panel,
            "_choice_token_ids",
            side_effect=choices_side_effect,
        ),
    ):
        receipt = panel.authenticate_confirmation_tokenization(
            loaded,
            tokenizer,
        )

    assert receipt["row_count"] == 60
    assert receipt["reserve_tokenized"] is False
    assert len(receipt["rows"]) == 60
    assert visited == [record.prompt_id for record in confirmation]
    assert all("choice_token_ids" not in row for row in receipt["rows"])
    assert all(
        "choice_token_ids_sha256" in row for row in receipt["rows"]
    )
    assert len(receipt["hard_long_visible_suffixes"]) == 6
    assert all(
        entry["row_count"] == 3
        for entry in receipt["hard_long_visible_suffixes"]
    )


def test_frozen_overlap_receipt_covers_spent_prompt_banks() -> None:
    loaded = _loaded()
    receipt = panel.validate_frozen_receipt(
        REPOSITORY_ROOT,
        loaded,
        RECEIPT,
    )

    assert receipt["passed"] is True
    assert receipt["blind_authoring"] == panel.BLIND_AUTHORING_RECEIPT
    assert receipt["checks"]["panel_prompt_count"] == 72
    assert receipt["checks"]["reference_prompt_count"] == 892
    assert all(
        value == 0
        for key, value in receipt["checks"].items()
        if key.endswith("_count")
        and key not in {"panel_prompt_count", "reference_prompt_count"}
    )
    inventory = {
        entry["path"]: entry
        for entry in receipt["source_prompt_inventory"]
    }
    discovery = inventory["examples/gemma3_downstream_retention_v1.json"]
    assert discovery == {
        "path": "examples/gemma3_downstream_retention_v1.json",
        "sha256": (
            "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
        ),
        "prompt_count": 60,
        "ordered_raw_prompt_sha256": (
            "25ed988b345eeb6d738304f748ad798fb1a2209aa21e6fe39dfaaa873b966cf8"
        ),
        "ordered_canonical_prompt_sha256": (
            "de247e614f54d8537eac05fea0101dae9e3616228d3263108ff876d15463cc09"
        ),
    }
    qualification = inventory[
        "examples/gemma3_downstream_qualification_v2.json"
    ]
    assert qualification["prompt_count"] == 120
    assert qualification["ordered_raw_prompt_sha256"] == (
        "c6edcfce942684e87481a712bef17428908affe5759ae43f726e81a03d4f0879"
    )
    assert qualification["ordered_canonical_prompt_sha256"] == (
        "73c8ad6a4a0d1c01852feea71835a59bc0c9d609186b7145818241f003e91240"
    )


def test_panel_module_is_model_free_by_static_import_audit() -> None:
    source_path = Path(panel.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
    assert "torch" not in imported_roots
    assert "transformers" not in imported_roots


def test_loader_rejects_duplicate_keys_nonfinite_and_noncanonical_json() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / CORPUS.name
        manifest = root / MANIFEST.name
        shutil.copyfile(CORPUS, corpus)
        shutil.copyfile(MANIFEST, manifest)

        duplicate = manifest.read_text(encoding="utf-8").replace(
            "{\n",
            '{\n  "schema": "duplicate",\n',
            1,
        )
        manifest.write_text(duplicate, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate JSON key"):
            panel.load_panel(corpus, manifest)

        shutil.copyfile(MANIFEST, manifest)
        nonfinite = manifest.read_text(encoding="utf-8").replace(
            '"format_version": 1',
            '"format_version": NaN',
            1,
        )
        manifest.write_text(nonfinite, encoding="utf-8")
        with pytest.raises(ValueError, match="non-finite JSON number"):
            panel.load_panel(corpus, manifest)

        shutil.copyfile(MANIFEST, manifest)
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with pytest.raises(ValueError, match="canonical JSON bytes"):
            panel.load_panel(corpus, manifest)


def test_loader_rejects_symlinks_traversal_and_oversized_authority() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / CORPUS.name
        manifest = root / MANIFEST.name
        shutil.copyfile(CORPUS, corpus)
        shutil.copyfile(MANIFEST, manifest)
        link = root / "linked-panel.json"
        os.symlink(manifest, link)
        with pytest.raises(ValueError, match="non-symlink regular file"):
            panel.load_panel(corpus, link)

        raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw["prompt_corpus"] = "../" + CORPUS.name
        _canonical_write(manifest, raw)
        with pytest.raises(ValueError, match="wrong corpus|traversal"):
            panel.load_panel(corpus, manifest)

        shutil.copyfile(MANIFEST, manifest)
        manifest.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match="size must be between"):
            panel.load_panel(corpus, manifest)


def test_receipt_rejects_noncanonical_or_stale_content() -> None:
    loaded = _loaded()
    with TemporaryDirectory(dir=REPOSITORY_ROOT) as directory:
        path = Path(directory) / RECEIPT.name
        shutil.copyfile(RECEIPT, path)
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError, match="canonical JSON bytes"):
            panel.validate_frozen_receipt(REPOSITORY_ROOT, loaded, path)

        raw = json.loads(RECEIPT.read_text(encoding="utf-8"))
        raw["passed"] = False
        _canonical_write(path, raw)
        with pytest.raises(ValueError, match="stale"):
            panel.validate_frozen_receipt(REPOSITORY_ROOT, loaded, path)
