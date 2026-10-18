"""Check saved task outcomes with a full-prompt forward, without KV caching.

This is a follow-up execution check of the same frozen examples and choices.
It neither changes the prompts nor replaces the initial recorded task scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Completed cached-prefix task report")
    parser.add_argument("--corpus", default=".local-runs/qwen3-mlx-expanded-eval/tasks/panel.jsonl")
    parser.add_argument("--manifest", default="artifacts/qwen3/expanded_tasks_manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from evaluate_qwen3_mlx_tasks import load_panel, SCORING, score_labels, sha256_file
    from evaluate_qwen3_mlx_fit import artifact_identity
    examples, manifest = load_panel(args.corpus, args.manifest)
    original = json.loads(Path(args.report).read_text())
    if (original.get("complete") is not True or original.get("phase") not in ("reference", "evaluate")
            or original.get("scoring") != SCORING
            or original["manifest_sha256"] != sha256_file(args.manifest)
            or original["corpus_sha256"] != sha256_file(args.corpus)
            or [r["example_sha256"] for r in original["per_example"]] != [r["sha256"] for r in examples]):
        raise ValueError("Completed task report and frozen examples must match")
    if original["precision"]["mlx_enable_tf32"] != os.environ["MLX_ENABLE_TF32"]:
        raise ValueError("Precision settings differ from original run")
    identity = artifact_identity(original["artifact"]["path"])
    if identity != original["artifact"]:
        raise ValueError("Model artifact changed since cached-prefix evaluation")
    if Path(args.output).exists():
        raise ValueError("Choose a fresh verification report path")
    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    runtime = MLXQwen3.load(identity["path"])
    records = []
    for example, prior in zip(examples, original["per_example"]):
        output = runtime(mx.array([example["input_ids"]], dtype=mx.int32))
        logits = output.logits[0, -1, :].astype(mx.float32)
        selected = logits[mx.array(example["label_token_ids"], dtype=mx.int32)]
        mx.eval(selected)
        scored = score_labels(selected.tolist(), example["target_index"])
        records.append({
            "id": example["id"], "task": example["task"], "example_sha256": example["sha256"],
            "target_index": example["target_index"], "prediction_index": scored["prediction_index"],
            "cached_prediction_index": prior["prediction_index"], "correct": scored["correct"],
            "cached_correct": prior["correct"], "label_logits": scored["label_logits"],
            "max_label_logit_difference": max(abs(a - b) for a, b in zip(scored["label_logits"], prior["label_logits"])),
        })
        del output, logits, selected
    by_task = {}
    for task in sorted({r["task"] for r in records}):
        rows = [r for r in records if r["task"] == task]
        by_task[task] = {"examples": len(rows), "full_prefill_correct": sum(r["correct"] for r in rows),
                         "cached_correct": sum(r["cached_correct"] for r in rows),
                         "prediction_matches": sum(r["prediction_index"] == r["cached_prediction_index"] for r in rows)}
    result = {
        "scope": "Secondary full-prompt execution check after the frozen task results; same data, weights, labels and precision.",
        "artifact": identity, "original_report_sha256": sha256_file(args.report),
        "manifest_sha256": sha256_file(args.manifest), "script_sha256": sha256_file(__file__),
        "examples": len(records), "full_prefill_correct": sum(r["correct"] for r in records),
        "cached_correct": sum(r["cached_correct"] for r in records),
        "prediction_matches": sum(r["prediction_index"] == r["cached_prediction_index"] for r in records),
        "max_label_logit_difference": max(r["max_label_logit_difference"] for r in records),
        "by_task": by_task, "per_example": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("artifact", "per_example")}))


if __name__ == "__main__":
    main()
