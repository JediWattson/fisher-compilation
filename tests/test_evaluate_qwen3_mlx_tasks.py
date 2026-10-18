"""CPU checks for custom task labels, frozen prompts, scoring and cached execution."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "qwen3_task_evaluation", Path(__file__).parents[1] / "scripts" / "evaluate_qwen3_mlx_tasks.py")
tasks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tasks)


def test_source_labels_map_to_fixed_letters_without_reordering():
    question = tasks.normalize_question("boolq", {"question": "Is this true?", "passage": "Some text.", "answer": False}, 0)
    assert question["choices"] == ["Yes", "No"]
    assert question["target_label"] == "B"
    assert "Passage:\nSome text." in question["user_prompt"]
    piqa = tasks.normalize_question("piqa", {"goal": "Do something", "sol1": "First", "sol2": "Second", "label": 0}, 2)
    assert piqa["target_label"] == "A"
    assert piqa["source_index"] == 2
    arc = tasks.normalize_question("arc_easy", {"id": "science-1", "question": "Which?",
        "choices": {"label": ["1", "2", "3"], "text": ["x", "y", "z"]}, "answerKey": "2"}, 1)
    assert arc["target_label"] == "B"
    assert arc["source_choice_labels"] == ["1", "2", "3"]
    assert arc["choices"] == ["x", "y", "z"]
    with pytest.raises(ValueError, match="boolean"):
        tasks.normalize_question("boolq", {"question": "q", "passage": "p", "answer": 1}, 0)
    with pytest.raises(ValueError, match="Target"):
        tasks.normalize_question("piqa", {"goal": "q", "sol1": "x", "sol2": "y", "label": -1}, 0)
    with pytest.raises(ValueError, match="answerKey"):
        tasks.normalize_question("arc_easy", {"question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "D"}, 0)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert messages[0]["role"] == "user"
        return "<user>" + messages[0]["content"] + "<assistant><think></think>\n"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def _example():
    return tasks.tokenize_question(tasks.normalize_question("piqa", {"goal": "q", "sol1": "x", "sol2": "y", "label": 0}, 0), Tokenizer())


def test_tokenization_freezes_non_thinking_chat_and_rejects_truncation_or_merging():
    example = _example()
    assert example["label_token_ids"] == [65, 66]
    assert example["rendered_prompt"].endswith("<think></think>\n")
    assert example["sha256"] == tasks.digest({key: value for key, value in example.items() if key != "sha256"})
    with pytest.raises(ValueError, match="no filtering/truncation"):
        tasks.tokenize_question(example, Tokenizer(), max_prompt_tokens=2)

    class MergingTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            return result[:-1] if text.endswith("\nA") else result

    with pytest.raises(ValueError, match="context-stable"):
        tasks.tokenize_question(example, MergingTokenizer())


def test_conditional_label_probability_and_tie_rule():
    score = tasks.score_labels([np.log(3), 0], 0)
    assert score["prediction_index"] == 0
    assert score["label_probabilities"] == pytest.approx([0.75, 0.25])
    assert score["target_conditional_nll"] == pytest.approx(-np.log(0.75))
    assert tasks.score_labels([0, 0], 1)["prediction_index"] == 0
    assert not tasks.score_labels([0, 0], 1)["correct"]
    for logits, target in (([float("nan"), 0], 0), ([0, 1], -1), ([0], 0), ([0, 1], True)):
        with pytest.raises(ValueError):
            tasks.score_labels(logits, target)


def _prediction(index, target, prediction):
    logits = [0.0] * 3
    logits[prediction] = 1.0
    return {"example_sha256": str(index), "prediction_label": "ABC"[prediction], "target_label": "ABC"[target],
            **tasks.score_labels(logits, target)}


def test_pair_counts_distinguish_regressions_gains_and_changed_wrong_answers():
    # Cover every mutually exclusive pair class, including changed wrong answers.
    teacher = [_prediction(i, 0, t) for i, t in enumerate([0, 0, 1, 1, 1])]
    candidate = [_prediction(i, 0, c) for i, c in enumerate([0, 1, 0, 1, 2])]
    summary = tasks.summarize_predictions(candidate, teacher)
    assert summary["accuracy"] == summary["teacher_accuracy"] == 0.4
    assert summary["accuracy_change"] == 0
    assert summary["teacher_agreement"] == 0.4
    assert set(summary["paired_counts"].values()) == {1}
    assert sum(summary["paired_counts"].values()) == 5
    assert summary["mean_teacher_conditional_label_kl"] > 0
    assert tasks.summarize_predictions(teacher, teacher)["mean_teacher_conditional_label_kl"] == 0
    with pytest.raises(ValueError, match="in order"):
        tasks.summarize_predictions(candidate, list(reversed(teacher)))


def _panel(tmp_path):
    example = _example()
    corpus, manifest_path = tmp_path / "panel.jsonl", tmp_path / "manifest.json"
    corpus.write_text(json.dumps(example) + "\n")
    manifest = {"format_version": 1, "complete": True, "scoring": tasks.SCORING,
                "corpus": {"sha256": tasks.sha256_file(corpus), "examples": 1, "ordered_example_sha256": [example["sha256"]]}}
    tasks.write_json(manifest_path, manifest)
    return example, corpus, manifest_path


def test_frozen_corpus_and_reference_are_bound_to_prompt_and_order(tmp_path):
    example, corpus, manifest_path = _panel(tmp_path)
    assert tasks.load_panel(corpus, manifest_path)[0] == [example]
    reference = {"complete": True, "phase": "reference", "scoring": tasks.SCORING,
                 "manifest_sha256": tasks.sha256_file(manifest_path), "corpus_sha256": tasks.sha256_file(corpus),
                 "per_example": [{"example_sha256": example["sha256"]}]}
    tasks.validate_reference(reference, [example], manifest_path, corpus)
    changed = copy.deepcopy(reference)
    changed["per_example"][0]["example_sha256"] = "different"
    with pytest.raises(ValueError, match="identities"):
        tasks.validate_reference(changed, [example], manifest_path, corpus)
    corpus.write_text(corpus.read_text().replace('"q"', '"different"'))
    with pytest.raises(ValueError, match="checksum"):
        tasks.load_panel(corpus, manifest_path)
    with pytest.raises(ValueError, match="identity"):
        tasks.validate_reference(reference, [example], manifest_path, corpus)
    with pytest.raises(ValueError, match="replace frozen"):
        tasks.write_json(manifest_path, {}, frozen=True)


def test_task_execution_runs_no_head_on_prefix_and_only_final_prompt_token_on_cache():
    calls = []
    cache = SimpleNamespace(key_values=((np.zeros((1, 2)), np.zeros((1, 2))),),
                            attention_mask=np.ones((1, 2)), position_ids=np.arange(2)[None])

    def runtime(ids, **kwargs):
        calls.append((ids.tolist(), kwargs))
        if kwargs.get("return_logits") is False:
            return SimpleNamespace(cache=cache)
        assert kwargs["cache"] is cache
        return SimpleNamespace(logits=np.array([[[0.0, 2.0, 1.0, -1.0]]], dtype=np.float32))

    mx = SimpleNamespace(array=np.array, int32=np.int32, float32=np.float32, eval=lambda *args: None,
                         exp=np.exp, sum=np.sum, argmax=np.argmax,
                         logsumexp=lambda x: np.log(np.exp(x).sum()))
    logits, mass, top = tasks.final_choice_logits(runtime, {"input_ids": [10, 11, 12], "label_token_ids": [1, 2]}, mx)
    assert calls[0][0] == [[10, 11]]
    assert calls[0][1] == {"return_logits": False}
    assert calls[1][0] == [[12]]
    assert logits == [2.0, 1.0]
    assert mass == pytest.approx((np.exp(2) + np.exp(1)) / (1 + np.exp(2) + np.exp(1) + np.exp(-1)))
    assert top == 1
