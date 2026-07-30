from __future__ import annotations

import random

from fisher_graph.gemma3_l3_l4_progressive_a_guard_rotation import (
    rotate_gemma3_l3_l4_progressive_a_guard,
)
from fisher_graph.gemma3_l3_l4_progressive_a_pilot import (
    prepare_gemma3_l3_l4_progressive_a_pilot,
)


def test_rotated_guard_report_exposes_only_commitments(tmp_path) -> None:
    fit = tmp_path / "fit.json"
    selection = tmp_path / "selection.json"
    old_guard = tmp_path / "old-guard.json"
    old_corpus = tmp_path / "old-corpus.json"
    prepare_gemma3_l3_l4_progressive_a_pilot(
        fit_output=fit,
        selection_output=selection,
        guard_output=old_guard,
        corpus_output=old_corpus,
    )
    new_guard = tmp_path / "new-guard.json"
    new_corpus = tmp_path / "new-corpus.json"

    report = rotate_gemma3_l3_l4_progressive_a_guard(
        fit_input=fit,
        selection_input=selection,
        guard_output=new_guard,
        corpus_output=new_corpus,
        chooser=random.Random(7),
    )

    assert report["guard_example_count"] == 4
    assert report["prompt_text_exposed"] is False
    assert report["token_ids_exposed"] is False
    assert report["calibration_b_opened"] is False
    assert "prompts" not in report
    assert new_guard.exists()
    assert new_corpus.exists()
