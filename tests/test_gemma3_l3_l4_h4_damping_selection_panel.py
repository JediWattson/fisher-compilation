from __future__ import annotations

from collections import Counter
import inspect
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_l3_l4_h4_damping_selection_panel import (
    FRESH_DAMPING_SELECTION_FAMILIES,
    FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
    GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
    Gemma3L3L4H4DampingSelectionPanelArtifact,
    Gemma3L3L4H4DampingSelectionPanelClosedError,
    Gemma3L3L4H4DampingSelectionPanelIntegrityError,
    Gemma3L3L4H4DampingSelectionPanelSource,
    expanded_fit_lineage_from_corpus_artifact,
    freeze_gemma3_l3_l4_h4_damping_selection_panel,
    load_gemma3_l3_l4_h4_damping_expanded_fit_lineage,
    load_gemma3_l3_l4_h4_damping_selection_panel_artifact,
    load_gemma3_l3_l4_h4_damping_selection_role_input,
    write_gemma3_l3_l4_h4_damping_selection_panel_artifact,
    write_gemma3_l3_l4_h4_damping_selection_role_input,
)
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


_TOKENIZER_CONTRACT = {
    "model_id": "google/gemma-3-270m",
    "tokenizer_revision": "fixture-revision",
    "add_special_tokens": True,
    "padding_side": "left",
}
_FIT_BINDING_SHA256 = "a" * 64


def _fresh_prompts() -> tuple[str, ...]:
    return tuple(
        (
            "Fresh damping finite-NLL selection example "
            f"{index}: analyze the supplied relation and justify the result."
        )
        for index in range(16)
    )


def _expanded_corpus(
    root: Path,
    *,
    fit_family_ids: tuple[str, ...] | None = None,
):
    fit_families = (
        tuple(f"fixture-expanded-fit-family-{index}" for index in range(8))
        if fit_family_ids is None
        else fit_family_ids
    )
    assert len(fit_families) == 8
    fit_prompts = tuple(
        f"Expanded fit fixture prompt {index} with unique source content."
        for index in range(16)
    )
    paths = {}
    members = {
        "calibration_a_fit": (
            fit_prompts,
            fit_families + fit_families,
        ),
        "calibration_a_selection": (
            ("Prompt-free metadata fixture for prior selection.",),
            ("fixture-prior-selection-family",),
        ),
        "calibration_a_guard": (
            ("Prompt-free metadata fixture for prior guard.",),
            ("fixture-prior-guard-family",),
        ),
    }
    for role, (prompts, families) in members.items():
        path = root / f"{role}.json"
        write_gemma3_l3_l4_progressive_a_role_input(
            path,
            corpus_id="damping-selection-fixture",
            profile="pilot",
            role=role,  # type: ignore[arg-type]
            prompts=prompts,
            family_ids=families,
        )
        paths[role] = path
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id="damping-selection-fixture",
        profile="pilot",
        tokenizer_contract=_TOKENIZER_CONTRACT,
        role_input_paths=paths,  # type: ignore[arg-type]
    )
    artifact_path = root / "expanded.corpus.json"
    write_gemma3_l3_l4_progressive_a_corpus_artifact(
        artifact_path,
        artifact,
    )
    return artifact, artifact_path, paths, fit_prompts


def test_fresh_family_schedule_is_exact_balanced_and_new() -> None:
    assert len(FRESH_DAMPING_SELECTION_FAMILIES) == 8
    assert tuple(sorted(FRESH_DAMPING_SELECTION_FAMILIES)) == (
        FRESH_DAMPING_SELECTION_FAMILIES
    )
    assert len(FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE) == 16
    assert Counter(FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE) == {
        family: 2 for family in FRESH_DAMPING_SELECTION_FAMILIES
    }


def test_role_input_is_private_canonical_and_fixed_geometry(
    tmp_path,
) -> None:
    prompts = _fresh_prompts()
    input_path = tmp_path / "fresh-selection.private.json"
    input_sha256 = (
        write_gemma3_l3_l4_h4_damping_selection_role_input(
            input_path,
            prompts=prompts,
        )
    )

    role_input = (
        load_gemma3_l3_l4_h4_damping_selection_role_input(
            input_path
        )
    )

    assert role_input.source_file_sha256 == input_sha256
    assert role_input.panel_id == (
        GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID
    )
    assert role_input.prompts == prompts
    assert role_input.family_ids == (
        FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE
    )
    assert len(set(role_input.ordered_prompt_sha256s)) == 16
    with pytest.raises(ValueError, match="exactly 16"):
        write_gemma3_l3_l4_h4_damping_selection_role_input(
            tmp_path / "too-short.json",
            prompts=prompts[:-1],
        )


def test_prompt_free_lineage_and_freeze_need_no_role_input_paths(
    tmp_path,
) -> None:
    (
        expanded,
        expanded_path,
        role_paths,
        _fit_prompts,
    ) = _expanded_corpus(tmp_path / "expanded")
    for path in role_paths.values():
        path.unlink()

    lineage = load_gemma3_l3_l4_h4_damping_expanded_fit_lineage(
        expanded_path,
        expected_expanded_corpus_artifact_sha256=(
            expanded.artifact_sha256
        ),
        fit_binding_sha256=_FIT_BINDING_SHA256,
    )
    selection_path = tmp_path / "fresh.private.json"
    prompts = _fresh_prompts()
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=prompts,
    )

    artifact = freeze_gemma3_l3_l4_h4_damping_selection_panel(
        expanded_fit_lineage=lineage,
        selection_input_path=selection_path,
    )

    assert artifact.expanded_fit_lineage == lineage
    assert artifact.family_ids == FRESH_DAMPING_SELECTION_FAMILIES
    assert len(artifact.family_by_example) == 16
    metadata = artifact.to_dict()
    assert metadata["policy"] == {
        "opening": "one_shot_selection",
        "maximum_panel_open_count": 1,
        "authorized_alpha_pair": (0.0, 0.5),
        "authorized_candidate_count": 2,
        "adaptive_candidate_changes_authorized": False,
        "guard_authorized": False,
        "assessment_authorized": False,
    }
    assert metadata["safety"]["old_selection_input_capability_present"] is False
    assert metadata["safety"]["guard_input_capability_present"] is False
    assert (
        metadata["safety"]["calibration_b_input_capability_present"]
        is False
    )
    original_artifact_sha256 = artifact.artifact_sha256
    metadata["policy"]["authorized_candidate_count"] = 99
    assert artifact.to_dict()["policy"]["authorized_candidate_count"] == 2
    assert artifact.artifact_sha256 == original_artifact_sha256


def test_artifact_is_hash_authenticated_and_contains_no_prompt_text(
    tmp_path,
) -> None:
    expanded, _expanded_path, _paths, _fit_prompts = _expanded_corpus(
        tmp_path / "expanded"
    )
    lineage = expanded_fit_lineage_from_corpus_artifact(
        expanded,
        fit_binding_sha256=_FIT_BINDING_SHA256,
    )
    prompts = _fresh_prompts()
    selection_path = tmp_path / "fresh.private.json"
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=prompts,
    )
    artifact = freeze_gemma3_l3_l4_h4_damping_selection_panel(
        expanded_fit_lineage=lineage,
        selection_input_path=selection_path,
    )
    artifact_path = tmp_path / "fresh.panel.json"
    artifact_file_sha256 = (
        write_gemma3_l3_l4_h4_damping_selection_panel_artifact(
            artifact_path,
            artifact,
        )
    )

    encoded = artifact_path.read_text()
    assert artifact_file_sha256
    assert '"prompts"' not in encoded
    assert all(prompt not in encoded for prompt in prompts)
    restored = (
        load_gemma3_l3_l4_h4_damping_selection_panel_artifact(
            artifact_path,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
    )
    assert restored == artifact

    tampered = json.loads(encoded)
    tampered["selection"]["ordered_members"][0][
        "prompt_sha256"
    ] = "f" * 64
    tampered_path = tmp_path / "tampered.panel.json"
    tampered_path.write_text(
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    with pytest.raises(
        Gemma3L3L4H4DampingSelectionPanelIntegrityError
    ):
        load_gemma3_l3_l4_h4_damping_selection_panel_artifact(
            tampered_path
        )


def test_freeze_rejects_prompt_overlap_with_all_expanded_metadata(
    tmp_path,
) -> None:
    expanded, _artifact_path, _paths, fit_prompts = _expanded_corpus(
        tmp_path / "expanded"
    )
    lineage = expanded_fit_lineage_from_corpus_artifact(
        expanded,
        fit_binding_sha256=_FIT_BINDING_SHA256,
    )
    prompts = list(_fresh_prompts())
    prompts[0] = fit_prompts[0]
    selection_path = tmp_path / "overlap.private.json"
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=prompts,
    )

    with pytest.raises(
        ValueError,
        match="overlap prompt-free expanded development metadata",
    ):
        freeze_gemma3_l3_l4_h4_damping_selection_panel(
            expanded_fit_lineage=lineage,
            selection_input_path=selection_path,
        )


def test_freeze_rejects_family_overlap_with_expanded_metadata(
    tmp_path,
) -> None:
    overlapping_fit_families = (
        FRESH_DAMPING_SELECTION_FAMILIES[0],
        *(f"fixture-overlap-fit-family-{index}" for index in range(1, 8)),
    )
    expanded, _artifact_path, _paths, _fit_prompts = _expanded_corpus(
        tmp_path / "expanded",
        fit_family_ids=overlapping_fit_families,
    )
    lineage = expanded_fit_lineage_from_corpus_artifact(
        expanded,
        fit_binding_sha256=_FIT_BINDING_SHA256,
    )
    selection_path = tmp_path / "fresh.private.json"
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=_fresh_prompts(),
    )

    with pytest.raises(
        ValueError,
        match="families overlap prompt-free expanded development metadata",
    ):
        freeze_gemma3_l3_l4_h4_damping_selection_panel(
            expanded_fit_lineage=lineage,
            selection_input_path=selection_path,
        )


def test_private_family_schedule_tamper_fails_closed(tmp_path) -> None:
    selection_path = tmp_path / "fresh.private.json"
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=_fresh_prompts(),
    )
    raw = json.loads(selection_path.read_text())
    raw["family_ids"][0] = "unauthorized-family"
    selection_path.write_text(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        Gemma3L3L4H4DampingSelectionPanelIntegrityError,
        match="invalid",
    ):
        load_gemma3_l3_l4_h4_damping_selection_role_input(
            selection_path
        )


def test_selection_source_opens_only_new_input_once(tmp_path) -> None:
    expanded, _artifact_path, _paths, _fit_prompts = _expanded_corpus(
        tmp_path / "expanded"
    )
    lineage = expanded_fit_lineage_from_corpus_artifact(
        expanded,
        fit_binding_sha256=_FIT_BINDING_SHA256,
    )
    prompts = _fresh_prompts()
    selection_path = tmp_path / "fresh.private.json"
    write_gemma3_l3_l4_h4_damping_selection_role_input(
        selection_path,
        prompts=prompts,
    )
    artifact = freeze_gemma3_l3_l4_h4_damping_selection_panel(
        expanded_fit_lineage=lineage,
        selection_input_path=selection_path,
    )
    source = Gemma3L3L4H4DampingSelectionPanelSource(
        artifact=artifact,
        selection_input_path=selection_path,
    )

    opened = source.open_once()

    assert opened.prompts == prompts
    assert source.consumed is True
    assert source.opened is True
    with pytest.raises(
        Gemma3L3L4H4DampingSelectionPanelClosedError
    ):
        source.open_once()


def test_public_capability_boundary_has_no_protected_role_paths() -> None:
    assert set(
        inspect.signature(
            freeze_gemma3_l3_l4_h4_damping_selection_panel
        ).parameters
    ) == {"expanded_fit_lineage", "selection_input_path"}
    assert set(
        inspect.signature(
            load_gemma3_l3_l4_h4_damping_expanded_fit_lineage
        ).parameters
    ) == {
        "expanded_corpus_artifact_path",
        "expected_expanded_corpus_artifact_sha256",
        "fit_binding_sha256",
    }
    source_parameters = set(
        inspect.signature(
            Gemma3L3L4H4DampingSelectionPanelSource
        ).parameters
    )
    assert source_parameters == {"artifact", "selection_input_path"}
    assert all(
        "path" not in name
        for name in (
            Gemma3L3L4H4DampingSelectionPanelArtifact.__dataclass_fields__
        )
    )
    assert all(
        "guard" not in name and "calibration_b" not in name
        for name in Gemma3L3L4H4DampingSelectionPanelSource.__slots__
    )
