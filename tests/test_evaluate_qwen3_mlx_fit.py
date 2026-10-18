"""CPU checks for frozen-corpus evaluation masks, metrics and disk references."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_qwen3_mlx_fit.py"
_SPEC = importlib.util.spec_from_file_location("qwen3_fit_evaluation", _SCRIPT)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def _write_corpus(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_next_token_positions_exclude_padding_gaps_and_ignored_targets(tmp_path):
    corpus = _write_corpus(tmp_path / "eval.jsonl", [{
        "input_ids": [[0, 3, 4, 5, 6, 0], [7, 8, 0, 9, 10, 11]],
        "attention_mask": [[0, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1]],
        "labels": [[0, 3, 4, -100, 6, 0], [7, 8, 0, 9, 10, 11]],
    }])
    rows = evaluation.prepare_examples(corpus)
    assert rows[0]["predictor_positions"] == [1, 3]
    assert rows[1]["predictor_positions"] == [0, 3, 4]
    assert rows[0]["batch"]["position_ids"] == [0, 0, 1, 2, 3, 3]
    assert all(5 not in row["predictor_positions"] for row in rows)
    bounded = evaluation.prepare_examples(corpus, max_examples=2, max_tokens=10)
    assert [len(row["batch"]["input_ids"]) for row in bounded] == [6, 4]
    assert bounded[1]["predictor_positions"] == [0]


def test_probability_metrics_match_analytic_kl_and_weight_by_tokens():
    teacher = np.log(np.array([[0.75, 0.25], [0.25, 0.75], [0.75, 0.25]], np.float32))
    student = np.log(np.array([[0.5, 0.5], [0.8, 0.2], [0.75, 0.25]], np.float32))
    rows = evaluation.token_metrics(teacher, student, np.array([0, 1, 1]))
    assert rows[0]["teacher_kl"] == pytest.approx(0.75 * np.log(1.5) + 0.25 * np.log(0.5), abs=1e-7)
    assert rows[1]["student_nll"] == pytest.approx(-np.log(0.2), abs=1e-7)
    assert [row["top1_agrees"] for row in rows] == [True, False, True]
    summary = evaluation.summarize(rows)
    assert summary["top1_agreement"] == pytest.approx(2 / 3)
    assert summary["teacher_ground_truth_top1_accuracy"] == pytest.approx(2 / 3)
    assert summary["student_ground_truth_top1_accuracy"] == pytest.approx(1 / 3)
    assert summary["teacher_nll"] == pytest.approx(-np.log(0.75 * 0.75 * 0.25) / 3, abs=1e-7)
    assert summary["nll_increase"] == pytest.approx(summary["student_nll"] - summary["teacher_nll"])
    assert evaluation.summarize(evaluation.token_metrics(teacher, teacher, [0, 1, 1]))["teacher_kl"] == 0


def test_compact_metrics_match_dense_scores_without_inventing_kl():
    teacher = np.log(np.array([[0.75, 0.25], [0.25, 0.75], [0.75, 0.25]], np.float32))
    student = np.log(np.array([[0.5, 0.5], [0.8, 0.2], [0.75, 0.25]], np.float32))
    targets = np.array([0, 1, 1])
    dense = evaluation.token_metrics(teacher, student, targets)
    compact = evaluation.compact_token_metrics(-teacher[np.arange(3), targets], teacher.argmax(-1),
                                               -student[np.arange(3), targets], student.argmax(-1), targets, 2)
    assert compact == [{name: value for name, value in row.items() if name != "teacher_kl"} for row in dense]
    summary = evaluation.summarize(compact)
    assert "teacher_kl" not in summary
    assert summary["teacher_kl_available"] is False
    assert "do not retain" in summary["teacher_kl_unavailable_reason"]
    for key in ("teacher_nll", "student_nll", "top1_agreement", "nll_increase"):
        assert summary[key] == evaluation.summarize(dense)[key]
    with pytest.raises(ValueError, match="Cannot mix"):
        evaluation.summarize([dense[0], compact[1]])


@pytest.mark.parametrize("field,value,error", [
    ("teacher_nll", [float("nan"), 1.0], "Nonfinite"),
    ("student_nll", [1.0], "align"),
    ("teacher_top1", [-1, 0], "vocabulary"),
    ("student_top1", [0, 2], "vocabulary"),
    ("targets", [0.0, 1.0], "vocabulary"),
])
def test_compact_metrics_validate_saved_scores(field, value, error):
    kwargs = dict(teacher_nll=[1.0, 2.0], teacher_top1=[0, 1], student_nll=[1.5, 2.5],
                  student_top1=[0, 1], targets=[0, 1], vocabulary_size=2)
    kwargs[field] = value
    with pytest.raises(ValueError, match=error):
        evaluation.compact_token_metrics(**kwargs)


class _NumpyCore:
    """Exercise scoring orchestration with CPU arrays, without importing Metal."""
    array, arange, argmax, tanh = staticmethod(np.array), staticmethod(np.arange), staticmethod(np.argmax), staticmethod(np.tanh)
    all, isfinite = staticmethod(np.all), staticmethod(np.isfinite)
    int32, float32 = np.int32, np.float32

    @staticmethod
    def eval(*args):
        pass

    @staticmethod
    def logsumexp(values, axis, keepdims=False):
        maximum = values.max(axis=axis, keepdims=True)
        result = np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)) + maximum
        return result if keepdims else np.squeeze(result, axis=axis)


@pytest.mark.parametrize("lexical_kind", ["tied", "untied", "factored"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("softcap", [None, 0.7])
def test_compact_head_chunks_match_dense_and_preserve_dtype(tmp_path, monkeypatch, lexical_kind, dtype, softcap):
    mx = _NumpyCore()
    rng = np.random.default_rng(31)
    weight = rng.normal(size=(5, 3)).astype(dtype)
    parameters = {"lexical.weight": weight}
    if lexical_kind == "untied":
        parameters["lexical.head_weight"] = (weight * -0.75).astype(dtype)
    elif lexical_kind == "factored":
        parameters = {"lexical.codebook": rng.normal(size=(5, 2)).astype(dtype),
                      "lexical.basis": rng.normal(size=(3, 2)).astype(dtype)}
    normalized = rng.normal(size=(1, 7, 3)).astype(dtype)

    class Runtime:
        def __init__(self):
            self.parameters, self.config = parameters, {"final_logit_softcap": softcap}
            self.require_logits_free, self.calls = False, 0

        def __call__(self, **batch):
            self.calls += 1
            if self.require_logits_free:
                assert batch["return_logits"] is False
                assert batch["capture"] == ["terminal.normalized"]
            logits = evaluation._project_logits(self, normalized, mx) if batch.get("return_logits", True) else None
            return types.SimpleNamespace(logits=logits, captures={"terminal.normalized": normalized})

    runtime = Runtime()
    corpus = _write_corpus(tmp_path / "eval.jsonl", [{
        "input_ids": [0, 1, 0, 2, 3, 4, 1], "attention_mask": [1, 1, 0, 1, 1, 1, 1],
    }])
    example = evaluation.prepare_examples(corpus)[0]
    dense = evaluation._log_probabilities(runtime, example, mx)
    runtime.require_logits_free = True
    projection_shapes = []
    original = evaluation._project_logits

    def record_projection(runtime, hidden, core):
        projection_shapes.append((hidden.shape, hidden.dtype))
        return original(runtime, hidden, core)

    monkeypatch.setattr(evaluation, "_project_logits", record_projection)
    compact = evaluation._compact_scores(runtime, example, mx, 3)
    targets = [example["batch"]["labels"][t + 1] for t in example["predictor_positions"]]
    np.testing.assert_allclose(compact["nll"], -dense[np.arange(len(targets)), targets], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(compact["top1"], dense.argmax(-1))
    assert projection_shapes == [((3, 3), np.dtype(dtype)), ((1, 3), np.dtype(dtype))]
    assert compact["nll"].dtype == np.float32
    assert compact["vocabulary_size"] == 5
    invalid = copy.deepcopy(example)
    invalid["batch"]["labels"][1] = 5
    calls = runtime.calls
    with pytest.raises(ValueError, match="outside the vocabulary"):
        evaluation._compact_scores(runtime, invalid, mx, 3)
    assert runtime.calls == calls
    normalized[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite compact"):
        evaluation._compact_scores(runtime, example, mx, 3)


def test_artifact_identity_binds_weights_as_well_as_metadata(tmp_path):
    (tmp_path / "mlx_qwen3.json").write_text('{"format_version":1}')
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"first")
    original = evaluation.artifact_identity(tmp_path)
    weights.write_bytes(b"other")
    changed = evaluation.artifact_identity(tmp_path)
    assert original["sha256"] != changed["sha256"]
    assert original["files"][0] == changed["files"][0]
    assert original["files"][1]["size"] == changed["files"][1]["size"]


def test_cached_hf_snapshot_symlinks_are_hashed_but_index_traversal_is_rejected(tmp_path):
    snapshot, blobs = tmp_path / "snapshots" / "revision", tmp_path / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "config").write_text('{"model_type":"qwen3"}')
    (blobs / "weights").write_bytes(b"checkpoint")
    (snapshot / "config.json").symlink_to(blobs / "config")
    (snapshot / "model.safetensors").symlink_to(blobs / "weights")
    identity = evaluation.artifact_identity(snapshot)
    assert identity["kind"] == "hf_qwen3"
    assert identity["files"][1]["sha256"] == evaluation._sha256(blobs / "weights")
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.weight": "../../blobs/weights"},
    }))
    with pytest.raises(ValueError, match="must stay within"):
        evaluation.artifact_identity(snapshot)


def test_frozen_reference_rejects_positions_or_limits_changes(tmp_path):
    path = _write_corpus(tmp_path / "eval.jsonl", [{"input_ids": [1, 2, 3]}])
    examples = evaluation.prepare_examples(path)
    limits = {"max_examples": 32, "max_length": 128, "max_tokens": 4096}
    receipt = evaluation.corpus_receipt(path, examples, limits)
    manifest = {"format_version": 1, "complete": True, "corpus": receipt,
                "references": [{"example_sha256": examples[0]["sha256"]}]}
    evaluation.validate_reference(manifest, receipt, examples)
    changed = copy.deepcopy(receipt)
    changed["limits"]["max_length"] = 127
    with pytest.raises(ValueError, match="limits"):
        evaluation.validate_reference(manifest, changed, examples)
    changed_examples = copy.deepcopy(examples)
    changed_examples[0]["batch"]["position_ids"] = [4, 6, 9]
    changed_receipt = evaluation.corpus_receipt(path, changed_examples, limits)
    with pytest.raises(ValueError, match="prepared_sha256"):
        evaluation.validate_reference(manifest, changed_receipt, changed_examples)
    # Legacy dense manifests omit mode/chunk fields and continue to validate.
    with pytest.raises(ValueError, match="mode mismatch"):
        evaluation.validate_reference(manifest, receipt, examples, reference_mode="compact")
    compact_manifest = {**manifest, "reference_mode": "compact", "head_chunk_size": 64}
    evaluation.validate_reference(compact_manifest, receipt, examples, reference_mode="compact", head_chunk_size=64)
    with pytest.raises(ValueError, match="chunk size mismatch"):
        evaluation.validate_reference(compact_manifest, receipt, examples, reference_mode="compact", head_chunk_size=128)


def test_calibration_overlap_ignores_padding_and_labels(tmp_path):
    path = _write_corpus(tmp_path / "eval.jsonl", [{"input_ids": [1, 2, 3]}])
    calibration = _write_corpus(tmp_path / "train.jsonl", [{
        "input_ids": [0, 1, 2, 3], "attention_mask": [0, 1, 1, 1], "labels": [0, -100, 2, 3],
    }])
    examples = evaluation.prepare_examples(path)
    limits = {"max_examples": 32, "max_length": 128, "max_tokens": 4096}
    with pytest.raises(ValueError, match="overlaps calibration"):
        evaluation.corpus_receipt(path, examples, limits, calibration_corpus=calibration)


@pytest.mark.parametrize("reference_mode", ["dense", "compact"])
def test_two_phase_disk_protocol_and_tamper_rejection_without_gpu(tmp_path, monkeypatch, reference_mode):
    # Test orchestration independently of the already-covered MLX numerical path.
    corpus = _write_corpus(tmp_path / "eval.jsonl", [{"input_ids": [0, 1, 0]}])
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "mlx_qwen3.json").write_text('{"format_version":1}')
    (artifact / "weights.safetensors").write_bytes(b"test-weight-identity")
    runtime = types.SimpleNamespace(parameters={"weight": np.ones((2, 2), np.float32)})
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.reset_peak_memory = lambda: None
    fake_mx.get_peak_memory = lambda: 16
    fake_mx.eval = lambda *args: None
    fake_package = types.ModuleType("mlx")
    fake_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    from fisher_graph.mlx_qwen3 import MLXQwen3
    monkeypatch.setattr(MLXQwen3, "load", lambda path: runtime)
    probs = np.log(np.array([[0.3, 0.7], [0.8, 0.2]], np.float32))
    monkeypatch.setattr(evaluation, "_log_probabilities", lambda *args: probs)
    monkeypatch.setattr(evaluation, "_compact_scores", lambda *args: {
        "nll": -probs[np.arange(2), [1, 0]], "top1": probs.argmax(-1), "vocabulary_size": 2,
    })
    args = argparse.Namespace(phase="reference", artifact=str(artifact), corpus=str(corpus),
                              reference_dir=str(tmp_path / "references"), report=None,
                              tokenizer=None, calibration_corpus=None,
                              max_examples=32, max_length=128, max_tokens=4096,
                              reference_mode=reference_mode, head_chunk_size=128)
    evaluation.run(args)
    manifest = json.loads((Path(args.reference_dir) / "manifest.json").read_text())
    assert manifest["complete"]
    assert manifest["parameter_storage"]["bytes"] == 16
    assert manifest["parameter_storage"]["scalar_count"] == 4
    assert manifest["reference_mode"] == reference_mode
    assert manifest["head_chunk_size"] == (128 if reference_mode == "compact" else None)
    if reference_mode == "compact":
        with np.load(Path(args.reference_dir) / manifest["references"][0]["file"]) as reference:
            assert "teacher_log_probs" not in reference.files
            assert reference["teacher_nll"].dtype == np.float32
            assert reference["teacher_nll"].shape == (2,)
    args.phase = "evaluate"
    report = evaluation.run(args)
    assert report["aggregate"]["prediction_tokens"] == 2
    if reference_mode == "dense":
        assert report["aggregate"]["teacher_kl"] == 0
    else:
        assert "teacher_kl" not in report["aggregate"]
        assert report["aggregate"]["teacher_kl_available"] is False
    assert report["aggregate"]["nll_increase"] == 0
    tokens = [json.loads(row) for row in Path(report["per_token_metrics"]).read_text().splitlines()]
    assert [row["target_position"] for row in tokens] == [1, 2]
    reference_file = Path(args.reference_dir) / manifest["references"][0]["file"]
    with reference_file.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="NPZ identity"):
        evaluation.run(args)
