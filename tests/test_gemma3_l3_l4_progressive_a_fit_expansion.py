from __future__ import annotations

from collections import Counter
from pathlib import Path

from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    load_gemma3_l3_l4_progressive_a_fit_role,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)
from fisher_graph.gemma3_l3_l4_progressive_a_fit_expansion import (
    EXPANDED_FIT_FAMILIES,
    EXPANDED_FIT_PROMPTS,
    build_parser,
    prepare_gemma3_l3_l4_progressive_a_expanded_fit,
)


_TOKENIZER_CONTRACT = {
    "model_id": "google/gemma-3-270m",
    "tokenizer_revision": "fixture-revision",
    "add_special_tokens": True,
    "padding_side": "left",
}


def _parent_corpus(root: Path):
    paths = {}
    members = {
        "calibration_a_fit": (
            ("Old fit one.", "Old fit two."),
            ("old-fit-one", "old-fit-two"),
        ),
        "calibration_a_selection": (
            ("Protected selection.",),
            ("protected-selection",),
        ),
        "calibration_a_guard": (
            ("Protected guard.",),
            ("protected-guard",),
        ),
    }
    for role, (prompts, families) in members.items():
        path = root / f"{role}.json"
        write_gemma3_l3_l4_progressive_a_role_input(
            path,
            corpus_id="expanded-fit-fixture",
            profile="pilot",
            role=role,  # type: ignore[arg-type]
            prompts=prompts,
            family_ids=families,
        )
        paths[role] = path
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id="expanded-fit-fixture",
        profile="pilot",
        tokenizer_contract=_TOKENIZER_CONTRACT,
        role_input_paths=paths,  # type: ignore[arg-type]
    )
    artifact_path = root / "parent-corpus.json"
    write_gemma3_l3_l4_progressive_a_corpus_artifact(
        artifact_path,
        artifact,
    )
    return artifact, artifact_path, paths


def test_expanded_fit_panel_is_balanced_unique_and_fit_only() -> None:
    assert len(EXPANDED_FIT_PROMPTS) == 16
    assert len(EXPANDED_FIT_FAMILIES) == 16
    assert Counter(EXPANDED_FIT_FAMILIES) == {
        family: 2 for family in set(EXPANDED_FIT_FAMILIES)
    }
    assert len(set(EXPANDED_FIT_FAMILIES)) == 8
    hashes = tuple(
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
        for prompt in EXPANDED_FIT_PROMPTS
    )
    assert len(set(hashes)) == 16

    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--fit-output" in options
    assert "--corpus-output" in options
    assert "--selection-input" not in options
    assert "--guard-input" not in options


def test_expanded_fit_freeze_needs_no_protected_role_files(
    tmp_path,
) -> None:
    parent, parent_path, paths = _parent_corpus(tmp_path / "parent")
    paths["calibration_a_selection"].unlink()
    paths["calibration_a_guard"].unlink()
    fit_output = tmp_path / "expanded.fit.json"
    corpus_output = tmp_path / "expanded.corpus.json"

    result = prepare_gemma3_l3_l4_progressive_a_expanded_fit(
        parent_corpus_path=parent_path,
        expected_parent_artifact_sha256=parent.artifact_sha256,
        fit_output=fit_output,
        corpus_output=corpus_output,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )

    assert result["example_count"] == 16
    assert result["family_count"] == 8
    assert result["selection_input_capability_present"] is False
    assert result["guard_input_capability_present"] is False
    assert result["selection_opened"] is False
    assert result["guard_opened"] is False
    replacement, fit = load_gemma3_l3_l4_progressive_a_fit_role(
        corpus_output,
        fit_input_path=fit_output,
        tokenizer_contract=_TOKENIZER_CONTRACT,
    )
    assert len(fit.prompts) == 16
    assert len(set(fit.family_ids)) == 8
    for role in ("calibration_a_selection", "calibration_a_guard"):
        assert replacement.role_view(role) == parent.role_view(role)
