"""Summarize frozen text evaluations with paired, article-level resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def article_bootstrap(examples, *, seed=20261018, resamples=5000):
    """Keep all adjacent tokens of an article together when resampling."""
    counts = np.asarray([x["prediction_tokens"] for x in examples], dtype=np.float64)
    deltas = np.asarray([x["nll_increase"] for x in examples], dtype=np.float64)
    if len(counts) == 0 or np.any(counts <= 0) or not np.isfinite(deltas).all():
        raise ValueError("Expected nonempty finite per-article results")
    indices = np.random.default_rng(seed).integers(len(counts), size=(resamples, len(counts)))
    means = (counts[indices] * deltas[indices]).sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "unit": "article", "resamples": resamples, "seed": seed,
        "token_weighted_nll_delta": float(np.sum(counts * deltas) / counts.sum()),
        "percentile_95_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "article_delta_range": [float(deltas.min()), float(deltas.max())],
        "articles_with_positive_nll_delta": int(np.sum(deltas > 0)),
        "article_count": len(counts),
        "scope": "Descriptive stability within this fixed panel; selection was not a random population sample.",
    }


def position_quarters(rows, length):
    result = []
    for quarter in range(4):
        selected = [r for r in rows if quarter * length // 4 <= r["target_position"] < (quarter + 1) * length // 4]
        if not selected:
            continue
        result.append({
            "target_position_start": quarter * length // 4,
            "target_position_end_exclusive": (quarter + 1) * length // 4,
            "prediction_tokens": len(selected),
            "nll_increase": float(np.mean([r["student_nll"] - r["teacher_nll"] for r in selected])),
            "top1_agreement": float(np.mean([r["top1_agrees"] for r in selected])),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(".local-runs/qwen3-mlx-expanded-eval"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen3/expanded_text_summary.json"))
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/qwen3/expanded_evaluation_protocol.json"))
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    corpus_manifest_path = Path(protocol["text_corpus_manifest"]["path"])
    if file_hash(corpus_manifest_path) != protocol["text_corpus_manifest"]["sha256"]:
        raise ValueError("Corpus manifest differs from frozen protocol")
    corpus_manifest = json.loads(corpus_manifest_path.read_text())
    panels = {}
    for length in (512, 2048):
        panels[str(length)] = {}
        for variant in ("attention-only", "attention-and-mlp"):
            path = args.run / f"{variant}-{length}.json"
            report = json.loads(path.read_text())
            expected = corpus_manifest["panels"][f"assessment_{length}"]
            if (report["artifact"] != protocol["variants"][variant]
                    or report["teacher_artifact"] != protocol["variants"]["exact"]
                    or report["corpus"]["sha256"] != expected["sha256"]
                    or report["corpus"]["examples"] != expected["window_count"]
                    or report["corpus"]["prediction_tokens"] != expected["next_token_targets"]):
                raise ValueError("Report identity or counts differ from frozen protocol")
            token_path = Path(report["per_token_metrics"])
            rows = [json.loads(line) for line in token_path.read_text().splitlines()]
            aggregate = report["aggregate"]
            if len(rows) != aggregate["prediction_tokens"]:
                raise ValueError("Token count differs from report")
            actual_delta = float(np.mean([r["student_nll"] - r["teacher_nll"] for r in rows]))
            if not np.isclose(actual_delta, aggregate["nll_increase"], atol=1e-12, rtol=0):
                raise ValueError("Per-token NLL differs from aggregate")
            grouped = {}
            for row in rows:
                grouped.setdefault(row["example_index"], []).append(row)
            examples = report["per_example"]
            if set(grouped) != set(range(expected["window_count"])) or len(examples) != len(grouped):
                raise ValueError("Per-token article indices differ from frozen corpus")
            for index, example in enumerate(examples):
                selected = grouped[index]
                delta = float(np.mean([r["student_nll"] - r["teacher_nll"] for r in selected]))
                if (example["example_index"] != index or example["prediction_tokens"] != len(selected)
                        or not np.isclose(delta, example["nll_increase"], atol=1e-12, rtol=0)):
                    raise ValueError("Per-article count or NLL differs from per-token rows")
            panels[str(length)][variant] = {
                "report_path": str(path), "report_sha256": file_hash(path),
                "per_token_sha256": file_hash(token_path), "aggregate": aggregate,
                "article_bootstrap": article_bootstrap(report["per_example"]),
                "position_quarters": position_quarters(rows, length),
                "per_example": report["per_example"],
                "corpus": report["corpus"], "artifact": report["artifact"],
                "teacher_artifact": report["teacher_artifact"],
                "peak_mlx_memory_bytes": report["peak_mlx_memory_bytes"],
                "parameter_storage": report["parameter_storage"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"panels": panels, "protocol_sha256": file_hash(args.protocol),
                                      "script_sha256": file_hash(__file__)}, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
