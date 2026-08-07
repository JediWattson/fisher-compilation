from __future__ import annotations

import json
import random

import pytest

from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpusArtifact,
)
from fisher_graph.gemma3_l3_l4_progressive_a_pilot import (
    prepare_gemma3_l3_l4_progressive_a_pilot,
)
from fisher_graph.gemma3_layer10_shape_flow_corpus import (
    prepare_gemma3_layer10_shape_flow_corpus,
)


def test_layer10_corpus_freezes_fresh_decision_roles_without_prompt_leak(
    tmp_path,
) -> None:
    fit = tmp_path / "prior.fit.json"
    prior_selection = tmp_path / "prior.selection.json"
    prior_guard = tmp_path / "prior.guard.json"
    prior_corpus = tmp_path / "prior.corpus.json"
    prepare_gemma3_l3_l4_progressive_a_pilot(
        fit_output=fit,
        selection_output=prior_selection,
        guard_output=prior_guard,
        corpus_output=prior_corpus,
    )
    selection = tmp_path / "layer10.selection.json"
    guard = tmp_path / "layer10.guard.json"
    corpus = tmp_path / "layer10.corpus.json"

    report = prepare_gemma3_layer10_shape_flow_corpus(
        fit_input=fit,
        prior_corpus_artifact=prior_corpus,
        selection_output=selection,
        guard_output=guard,
        corpus_output=corpus,
        chooser=random.Random(11),
    )

    assert report["layer_ordinal"] == 10
    assert report["selection_example_count"] == 8
    assert report["guard_example_count"] == 4
    assert report["prior_layer17_prompt_overlap_count"] == 0
    assert report["prior_layer17_family_overlap_count"] == 0
    assert report["upstream_fisher_fit_reused"] is True
    assert report["selection_reused"] is False
    assert report["guard_reused"] is False
    assert report["prompt_text_exposed"] is False
    assert report["token_ids_exposed"] is False
    assert "prompts" not in report
    assert selection.is_file()
    assert guard.is_file()
    raw = json.loads(corpus.read_text(encoding="utf-8"))
    artifact = Gemma3L3L4ProgressiveACorpusArtifact.from_dict(raw)
    assert artifact.artifact_sha256 == report["corpus_artifact_sha256"]
    assert artifact.role_view("calibration_a_fit").example_count == 8


def test_layer10_corpus_refuses_to_overwrite_private_roles(tmp_path) -> None:
    fit = tmp_path / "prior.fit.json"
    prior_selection = tmp_path / "prior.selection.json"
    prior_guard = tmp_path / "prior.guard.json"
    prior_corpus = tmp_path / "prior.corpus.json"
    prepare_gemma3_l3_l4_progressive_a_pilot(
        fit_output=fit,
        selection_output=prior_selection,
        guard_output=prior_guard,
        corpus_output=prior_corpus,
    )
    selection = tmp_path / "layer10.selection.json"
    guard = tmp_path / "layer10.guard.json"
    corpus = tmp_path / "layer10.corpus.json"
    arguments = {
        "fit_input": fit,
        "prior_corpus_artifact": prior_corpus,
        "selection_output": selection,
        "guard_output": guard,
        "corpus_output": corpus,
        "chooser": random.Random(11),
    }
    prepare_gemma3_layer10_shape_flow_corpus(**arguments)

    with pytest.raises(FileExistsError):
        prepare_gemma3_layer10_shape_flow_corpus(**arguments)
