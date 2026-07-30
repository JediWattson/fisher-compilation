from __future__ import annotations

from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    load_gemma3_l3_l4_progressive_a_corpus,
)
from fisher_graph.gemma3_l3_l4_progressive_a_pilot import (
    FIT_FAMILIES,
    FIT_PROMPTS,
    GUARD_FAMILIES,
    GUARD_PROMPTS,
    SELECTION_FAMILIES,
    SELECTION_PROMPTS,
    prepare_gemma3_l3_l4_progressive_a_pilot,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)


def test_pilot_is_new_three_way_family_disjoint_development_data(
    tmp_path,
) -> None:
    prompt_sets = (
        set(FIT_PROMPTS),
        set(SELECTION_PROMPTS),
        set(GUARD_PROMPTS),
    )
    family_sets = (
        set(FIT_FAMILIES),
        set(SELECTION_FAMILIES),
        set(GUARD_FAMILIES),
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
    fit = tmp_path / "fit.json"
    selection = tmp_path / "selection.json"
    guard = tmp_path / "guard.json"
    artifact = tmp_path / "corpus.json"

    report = prepare_gemma3_l3_l4_progressive_a_pilot(
        fit_output=fit,
        selection_output=selection,
        guard_output=guard,
        corpus_output=artifact,
    )

    assert report["structured_v9_reused"] is False
    assert report["calibration_b_opened"] is False
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    corpus = load_gemma3_l3_l4_progressive_a_corpus(
        artifact,
        role_input_paths={
            "calibration_a_fit": fit,
            "calibration_a_selection": selection,
            "calibration_a_guard": guard,
        },
        tokenizer_contract=legacy.metadata()["tokenizer"],
    )
    assert corpus.guard_opened is False
    assert corpus.preclaim_view(
        "calibration_a_fit"
    ).example_count == len(FIT_PROMPTS)
    assert corpus.preclaim_view(
        "calibration_a_selection"
    ).example_count == len(SELECTION_PROMPTS)
    assert corpus.preclaim_view(
        "calibration_a_guard"
    ).example_count == len(GUARD_PROMPTS)
