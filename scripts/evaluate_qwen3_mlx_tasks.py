"""Freeze and evaluate a small, custom Qwen3 multiple-choice task panel.

Preparation uses only a local tokenizer and pinned public Parquet data. Install
duckdb for preparation; model evaluation needs MLX but not duckdb/Transformers.
Run one artifact per process, first the exact reference, then each candidate.
This is constrained answer-letter scoring, not a standard benchmark harness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
COUNT = 32
SCORING = "qwen3_non_thinking_chat_constrained_single_letter_v1"
INSTRUCTION = "Answer the multiple-choice question. Reply with exactly one option letter and no other text."
SOURCES = {
    "boolq": {"repo": "google/boolq", "config": "default",
              "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "file": "data/validation-00000-of-00001.parquet",
              "sha256": "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
              "license_from_card": "cc-by-sa-3.0"},
    "piqa": {"repo": "ybisk/piqa", "config": "plain_text",
             "revision": "142c51238b3ca2bc61e9a075913871b8b600e8e1",
             "card_revision": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
             "revision_note": "Pinned publisher repository refs/convert/parquet commit; card/loader is a separate pinned commit",
             "file": "plain_text/validation/0000.parquet",
             "sha256": "40e1677bba609c8801deacc171d76f3e24c083f365c95a1cbda7520c923dfea2",
             "license_from_card": "unknown"},
    "arc_easy": {"repo": "allenai/ai2_arc", "config": "ARC-Easy",
                 "revision": "210d026faf9955653af8916fad021475a3f00453",
                 "file": "ARC-Easy/validation-00000-of-00001.parquet",
                 "sha256": "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",
                 "license_from_card": "cc-by-sa-4.0"},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value, *, frozen=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if frozen and path.exists() and path.read_text() != payload:
        raise ValueError(f"Refusing to replace frozen data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def normalize_question(task, row, index):
    """Preserve source choice order and map source labels to A, B, ... ."""
    passage = None
    if task == "boolq":
        if type(row["answer"]) is not bool:
            raise ValueError("BoolQ answer must be boolean")
        question, passage = row["question"], row["passage"]
        choices, target, original_labels = ["Yes", "No"], 0 if row["answer"] else 1, [True, False]
    elif task == "piqa":
        question, choices = row["goal"], [row["sol1"], row["sol2"]]
        target, original_labels = row["label"], [0, 1]
    elif task == "arc_easy":
        question, choices = row["question"], row["choices"]["text"]
        original_labels = row["choices"]["label"]
        if len(original_labels) != len(choices) or len(set(original_labels)) != len(original_labels):
            raise ValueError("ARC choice labels must uniquely align with choice texts")
        try:
            target = original_labels.index(row["answerKey"])
        except ValueError as error:
            raise ValueError("ARC answerKey does not name an offered choice") from error
    else:
        raise ValueError(f"Unknown task: {task}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 8 or any(
        not isinstance(choice, str) or not choice.strip() for choice in choices
    ):
        raise ValueError("Each question requires 2..8 nonempty choices")
    if type(target) is not int or not 0 <= target < len(choices):
        raise ValueError("Target must name one of the offered choices")
    if passage is not None and (not isinstance(passage, str) or not passage.strip()):
        raise ValueError("BoolQ passage must be nonempty text")
    labels = list("ABCDEFGH"[:len(choices)])
    parts = [INSTRUCTION]
    if passage is not None:
        parts.append("Passage:\n" + passage)
    parts.extend(["Question:\n" + question, "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))])
    return {"id": f"{task}:validation:{index}", "task": task, "split": "validation", "source_index": index,
            "source_id": row.get("id"), "source_row_sha256": digest(row), "question": question,
            "passage": passage, "choices": choices, "source_choice_labels": original_labels,
            "labels": labels, "target_index": target, "target_label": labels[target],
            "user_prompt": "\n\n".join(parts)}


def tokenize_question(example, tokenizer, *, max_prompt_tokens=2048):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": example["user_prompt"]}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError(f"{example['id']}: prompt exceeds the fixed token bound; no filtering/truncation is allowed")
    label_ids = []
    for label in example["labels"]:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.encode(prompt + label, add_special_tokens=False) != ids + encoded:
            raise ValueError(f"Label {label} must be a single, context-stable continuation token")
        label_ids.append(encoded[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Answer labels must have distinct token IDs")
    result = {**example, "rendered_prompt": prompt, "input_ids": ids, "label_token_ids": label_ids}
    return {**result, "sha256": digest(result)}


def prepare(args):
    import duckdb
    from transformers import AutoTokenizer
    import tokenizers
    import transformers
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    if tokenizer_path.name != TOKENIZER_REVISION:
        raise ValueError("Use the pinned Qwen3-1.7B tokenizer snapshot")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    output = Path(args.output)
    source_dir, sources, examples = output / "source", {}, []
    for task, specification in SOURCES.items():
        folder = source_dir / task
        if args.download:
            subprocess.run(["hf", "download", specification["repo"], specification["file"], "--type", "dataset",
                            "--revision", specification["revision"], "--local-dir", str(folder)], check=True)
            subprocess.run(["hf", "download", specification["repo"], "README.md", "--type", "dataset", "--revision",
                            specification.get("card_revision", specification["revision"]), "--local-dir", str(folder)], check=True)
        source, card = folder / specification["file"], folder / "README.md"
        if sha256_file(source) != specification["sha256"]:
            raise ValueError(f"Pinned source hash mismatch: {source}")
        # Explicit physical row ordering avoids relying on SQL's unspecified order.
        query = "SELECT to_json(t) FROM (SELECT * EXCLUDE(file_row_number) FROM read_parquet(?, file_row_number=true) ORDER BY file_row_number LIMIT ?) t"
        rows = [json.loads(row[0]) for row in duckdb.sql(query, params=[str(source), COUNT]).fetchall()]
        if len(rows) != COUNT:
            raise ValueError(f"{task} requires exactly the first {COUNT} rows")
        examples.extend(tokenize_question(normalize_question(task, row, index), tokenizer,
                                          max_prompt_tokens=args.max_prompt_tokens) for index, row in enumerate(rows))
        sources[task] = {**specification, "split": "validation", "source_path": str(source),
                         "card_sha256": sha256_file(card), "card_url": f"https://huggingface.co/datasets/{specification['repo']}/blob/{specification.get('card_revision', specification['revision'])}/README.md",
                         "selected_row_indices": list(range(COUNT))}
    corpus = output / "panel.jsonl"
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in examples)
    if corpus.exists() and corpus.read_text() != payload:
        raise ValueError("Refusing to replace frozen task corpus")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(payload)
    tokenizer_hashes = {p.name: sha256_file(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file() and (
        p.name.startswith("tokenizer") or p.name in ("chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"))}
    manifest = {"format_version": 1, "complete": True, "frozen_before_model_outputs": True,
                "scoring": SCORING, "selection": "first 32 physical validation rows in source order per task; no filtering",
                "sources": sources, "tokenizer": {"model": "Qwen/Qwen3-1.7B", "revision": TOKENIZER_REVISION,
                    "files_sha256": tokenizer_hashes, "transformers_version": transformers.__version__,
                    "tokenizers_version": tokenizers.__version__, "chat_template_sha256": digest(tokenizer.chat_template)},
                "protocol": {"instruction": INSTRUCTION, "enable_thinking": False, "add_generation_prompt": True,
                    "choice_order": "source order; BoolQ A=Yes B=No", "selection_rule": "maximum logit among offered single-token letters; ties choose first",
                    "label_distribution": "softmax restricted to the offered letter logits; not full-vocabulary probabilities",
                    "truncation": False, "max_prompt_tokens": args.max_prompt_tokens, "batch_size": 1,
                    "execution": "cached prefix without LM head, then final prompt token with full LM head",
                    "pretrained_contamination": "unknown; held out from this compression fit, not necessarily from model pretraining",
                    "qualification": "custom small task diagnostic, not standard leaderboard scores; no pass/fail gate"},
                "corpus": {"path": str(corpus), "sha256": sha256_file(corpus), "examples": len(examples),
                    "examples_per_task": dict(Counter(x["task"] for x in examples)),
                    "input_tokens": sum(len(x["input_ids"]) for x in examples),
                    "min_prompt_tokens": min(len(x["input_ids"]) for x in examples),
                    "max_prompt_tokens": max(len(x["input_ids"]) for x in examples),
                    "ordered_example_sha256": [x["sha256"] for x in examples]},
                "preparation": {"duckdb_version": duckdb.__version__, "script_sha256": sha256_file(__file__)}}
    write_json(args.manifest, manifest, frozen=True)
    print(json.dumps({"manifest": str(args.manifest), "corpus": manifest["corpus"]}))
    return manifest


def load_panel(corpus, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True or manifest.get("scoring") != SCORING:
        raise ValueError("Incomplete or incompatible panel manifest")
    if sha256_file(corpus) != manifest["corpus"]["sha256"]:
        raise ValueError("Frozen corpus checksum mismatch")
    examples = [json.loads(line) for line in Path(corpus).read_text().splitlines() if line.strip()]
    if len(examples) != manifest["corpus"]["examples"] or not examples:
        raise ValueError("Frozen example count mismatch")
    for example in examples:
        fields = dict(example)
        expected = fields.pop("sha256")
        if digest(fields) != expected:
            raise ValueError("Frozen example checksum mismatch")
        for name in ("input_ids", "label_token_ids"):
            values = example[name]
            if not values or any(type(x) is not int or x < 0 for x in values):
                raise ValueError("Token IDs must be nonempty nonnegative integer arrays")
        labels, target = example["labels"], example["target_index"]
        if (len(labels) != len(example["label_token_ids"]) or len(labels) != len(example["choices"])
                or len(set(example["label_token_ids"])) != len(labels) or type(target) is not int or not 0 <= target < len(labels)):
            raise ValueError("Labels, choices, and target must align")
    if [x["sha256"] for x in examples] != manifest["corpus"]["ordered_example_sha256"]:
        raise ValueError("Frozen example order mismatch")
    return examples, manifest


def score_labels(logits, target):
    values = [float(value) for value in logits]
    if len(values) < 2 or not all(math.isfinite(x) for x in values) or type(target) is not int or not 0 <= target < len(values):
        raise ValueError("Finite choice logits and a valid target are required")
    maximum = max(values)
    normalizer = math.log(math.fsum(math.exp(x - maximum) for x in values)) + maximum
    log_probs = [x - normalizer for x in values]
    prediction = max(range(len(values)), key=values.__getitem__)
    return {"label_logits": values, "label_log_probabilities": log_probs,
            "label_probabilities": [math.exp(x) for x in log_probs], "prediction_index": prediction,
            "correct": prediction == target, "target_conditional_nll": -log_probs[target]}


def summarize_predictions(rows, teacher_rows=None):
    if not rows:
        raise ValueError("Cannot summarize an empty task panel")
    count = len(rows)
    result = {"examples": count, "correct": sum(r["correct"] for r in rows),
              "accuracy": sum(r["correct"] for r in rows) / count,
              "mean_target_conditional_nll": math.fsum(r["target_conditional_nll"] for r in rows) / count,
              "prediction_label_counts": dict(Counter(r["prediction_label"] for r in rows)),
              "target_label_counts": dict(Counter(r["target_label"] for r in rows))}
    if teacher_rows is not None:
        if len(teacher_rows) != count or any(t["example_sha256"] != r["example_sha256"] for t, r in zip(teacher_rows, rows)):
            raise ValueError("Teacher examples must match candidate examples in order")
        pairs = list(zip(teacher_rows, rows))
        counts = {"both_correct": 0, "teacher_correct_candidate_wrong": 0,
                  "teacher_wrong_candidate_correct": 0, "both_wrong_same_choice": 0, "both_wrong_different_choice": 0}
        kl = []
        for teacher, row in pairs:
            if len(teacher["label_log_probabilities"]) != len(row["label_log_probabilities"]):
                raise ValueError("Teacher/candidate offered label sets differ")
            tc, cc = teacher["correct"], row["correct"]
            key = ("both_correct" if tc and cc else "teacher_correct_candidate_wrong" if tc else
                   "teacher_wrong_candidate_correct" if cc else "both_wrong_same_choice" if
                   teacher["prediction_index"] == row["prediction_index"] else "both_wrong_different_choice")
            counts[key] += 1
            kl.append(math.fsum(math.exp(t) * (t - c) for t, c in zip(teacher["label_log_probabilities"], row["label_log_probabilities"])))
        result.update(teacher_accuracy=sum(t["correct"] for t in teacher_rows) / count,
                      teacher_agreement=sum(t["prediction_index"] == r["prediction_index"] for t, r in pairs) / count,
                      accuracy_change=(result["correct"] - sum(t["correct"] for t in teacher_rows)) / count,
                      mean_teacher_conditional_label_kl=math.fsum(kl) / count, paired_counts=counts)
    return result


def validate_reference(reference, examples, manifest_path, corpus_path):
    if reference.get("complete") is not True or reference.get("phase") != "reference" or reference.get("scoring") != SCORING:
        raise ValueError("Reference report must be complete and use matching scoring")
    if reference["manifest_sha256"] != sha256_file(manifest_path) or reference["corpus_sha256"] != sha256_file(corpus_path):
        raise ValueError("Reference panel identity differs")
    if [row["example_sha256"] for row in reference["per_example"]] != [row["sha256"] for row in examples]:
        raise ValueError("Reference example identities or ordering differ")


def final_choice_logits(runtime, example, mx):
    ids = example["input_ids"]
    if len(ids) > 1:
        prefix = runtime(mx.array([ids[:-1]], dtype=mx.int32), return_logits=False)
        cache = prefix.cache
        mx.eval(cache.key_values, cache.attention_mask, cache.position_ids)
        del prefix
        output = runtime(mx.array([ids[-1:]], dtype=mx.int32), cache=cache)
    else:
        output = runtime(mx.array([ids], dtype=mx.int32))
    logits = output.logits[0, -1].astype(mx.float32)
    selected = logits[mx.array(example["label_token_ids"], dtype=mx.int32)]
    full_normalizer = mx.logsumexp(logits)
    mass = mx.sum(mx.exp(selected - full_normalizer))
    top = mx.argmax(logits)
    mx.eval(selected, mass, top)
    return selected.tolist(), float(mass.item()), int(top.item())


def evaluate(args):
    started = time.perf_counter()
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    examples, manifest = load_panel(args.corpus, args.manifest)
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference is not None:
        validate_reference(reference, examples, args.manifest, args.corpus)
        if reference["precision"]["mlx_enable_tf32"] != os.environ["MLX_ENABLE_TF32"]:
            raise ValueError("Reference and candidate precision settings differ")
    if Path(args.output).exists():
        raise ValueError("Output already exists; choose a fresh report path")
    # Importing here keeps preparation and all score/identity tests CPU-only.
    from evaluate_qwen3_mlx_fit import artifact_identity, _parameter_receipt
    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    identity = artifact_identity(args.artifact)
    mx.reset_peak_memory()
    runtime = MLXQwen3.load(identity["path"]) if identity["kind"] == "mlx_qwen3" else MLXQwen3.from_pretrained(identity["path"])
    mx.eval(runtime.parameters)
    vocabulary = next(runtime.parameters[name].shape[0] for name in ("lexical.weight", "lexical.codebook") if name in runtime.parameters)
    if any(token >= vocabulary for row in examples for token in row["input_ids"] + row["label_token_ids"]):
        raise ValueError("Panel tokens fall outside the runtime vocabulary")
    inference_started = time.perf_counter()
    predictions = []
    for index, example in enumerate(examples):
        logits, mass, top = final_choice_logits(runtime, example, mx)
        scored = score_labels(logits, example["target_index"])
        predictions.append({"id": example["id"], "task": example["task"], "example_sha256": example["sha256"],
            "question": example["question"], "passage": example["passage"], "choices": example["choices"],
            "labels": example["labels"], "label_token_ids": example["label_token_ids"],
            "target_index": example["target_index"], "target_label": example["target_label"],
            "prompt_tokens": len(example["input_ids"]), **scored,
            "prediction_label": example["labels"][scored["prediction_index"]],
            "offered_labels_full_vocabulary_probability_mass": mass, "full_vocabulary_top_token_id": top})
        if (index + 1) % COUNT == 0:
            print(json.dumps({"completed": index + 1, "task": example["task"]}), flush=True)
    teacher_rows = reference["per_example"] if reference else None
    by_task = {task: summarize_predictions([r for r in predictions if r["task"] == task],
                   [r for r in teacher_rows if r["task"] == task] if teacher_rows is not None else None) for task in SOURCES}
    report = {"format_version": 1, "complete": True, "phase": "evaluate" if reference else "reference", "scoring": SCORING,
              "artifact": identity, "corpus_sha256": sha256_file(args.corpus), "manifest_sha256": sha256_file(args.manifest),
              "script_sha256": sha256_file(__file__), "protocol": manifest["protocol"],
              "precision": {"weights": "preserved deployment dtype", "label_logits": "float32 cast of deployment logits",
                            "conditional_probabilities_and_metrics": "Python float64", "mlx_enable_tf32": os.environ["MLX_ENABLE_TF32"]},
              "parameter_storage": _parameter_receipt(runtime), "peak_mlx_memory_bytes": mx.get_peak_memory(),
              "elapsed_seconds": time.perf_counter() - started, "inference_seconds": time.perf_counter() - inference_started,
              "aggregate": summarize_predictions(predictions, teacher_rows), "by_task": by_task, "per_example": predictions}
    if reference is not None:
        report.update(teacher_artifact=reference["artifact"], reference_report_sha256=sha256_file(args.reference))
    write_json(args.output, report)
    print(json.dumps({"report": str(args.output), "aggregate": report["aggregate"], "by_task": by_task}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Freeze first32 task examples and exact chat prompts; no model inference")
    preparation.add_argument("--tokenizer", required=True)
    preparation.add_argument("--output", default=".local-runs/qwen3-mlx-expanded-eval/tasks")
    preparation.add_argument("--manifest", default="artifacts/qwen3/expanded_tasks_manifest.json")
    preparation.add_argument("--download", action="store_true")
    preparation.add_argument("--max-prompt-tokens", type=int, default=2048)
    preparation.set_defaults(function=prepare)
    evaluation = commands.add_parser("evaluate", help="Run one model; omit --reference for the original teacher")
    evaluation.add_argument("--artifact", required=True)
    evaluation.add_argument("--corpus", default=".local-runs/qwen3-mlx-expanded-eval/tasks/panel.jsonl")
    evaluation.add_argument("--manifest", default="artifacts/qwen3/expanded_tasks_manifest.json")
    evaluation.add_argument("--reference", help="Completed exact-teacher JSON report from a separate process")
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(function=evaluate)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    main()
