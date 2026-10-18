"""Bounded, two-process held-out comparison of local MLX Qwen3 artifacts.

Run --phase reference on the exact artifact, let that process exit, then run
--phase evaluate on a candidate. Each phase loads only one runtime. Reference
log probabilities are float32, one sequence per NPZ. Optional compact references
retain only target NLL and top1, using a chunked vocabulary projection; KL is
unavailable in that mode. Metrics use only adjacent valid next-token positions.
No quality threshold or speed qualification is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact_identity(directory):
    directory = Path(directory).expanduser().resolve()
    if (directory / "mlx_qwen3.json").is_file():
        kind, names = "mlx_qwen3", ["mlx_qwen3.json", "weights.safetensors"]
    else:
        kind, names = "hf_qwen3", ["config.json"]
        index = directory / "model.safetensors.index.json"
        if index.exists():
            names += [index.name, *sorted(set(json.loads(index.read_text())["weight_map"].values()))]
        else:
            names += ["model.safetensors"]
    files = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact file name must stay within its directory: {name}")
        # HF cache snapshots legitimately symlink files into ../.. / blobs.
        # Validate the declared name, then hash the local file it resolves to.
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"Missing artifact file: {name}")
        files.append({"name": name, "size": path.stat().st_size, "sha256": _sha256(path)})
    identity = {"kind": kind, "files": files}
    return {"path": str(directory), **identity, "sha256": _digest(identity)}


def _integer_rows(value, length, name, *, labels=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must align with input_ids")
    if any(type(x) is not int or (x < 0 and not (labels and x == -100)) for x in value):
        raise ValueError(f"{name} must contain nonnegative integers" + (" or -100" if labels else ""))
    return value


def _sequences(path, tokenizer, max_length):
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str) or (isinstance(item, dict) and "text" in item):
                if tokenizer is None:
                    raise ValueError("Text corpus requires --tokenizer with local tokenizer files")
                text = item if isinstance(item, str) else item["text"]
                if not isinstance(text, str):
                    raise ValueError("Corpus text must be a string")
                item = {"input_ids": tokenizer.encode(text, add_special_tokens=True,
                                                      truncation=True, max_length=max_length)}
            if not isinstance(item, dict) or not isinstance(item.get("input_ids"), list):
                raise ValueError(f"Corpus line {line_number}: expected text or input_ids")
            ids = item["input_ids"]
            batched = bool(ids and isinstance(ids[0], list))
            rows = ids if batched else [ids]
            for row_number, raw_ids in enumerate(rows):
                _integer_rows(raw_ids, len(raw_ids), "input_ids")
                fields = {"input_ids": raw_ids}
                for name in ("attention_mask", "position_ids", "labels"):
                    if name not in item:
                        continue
                    values = item[name]
                    if batched:
                        if not isinstance(values, list) or len(values) != len(rows):
                            raise ValueError(f"{name} batch size must match input_ids")
                        values = values[row_number]
                    if name == "attention_mask":
                        if not isinstance(values, list) or len(values) != len(raw_ids) or any(
                            type(x) not in (int, bool) or x not in (0, 1) for x in values
                        ):
                            raise ValueError("attention_mask must be binary and align with input_ids")
                    else:
                        _integer_rows(values, len(raw_ids), name, labels=name == "labels")
                    fields[name] = values
                mask = fields.get("attention_mask", [1] * len(raw_ids))
                # Exact overlap detection is independent of padding and labels.
                overlap_hash = _digest([token for token, valid in zip(raw_ids[:max_length], mask[:max_length]) if valid])
                yield line_number, row_number, fields, overlap_hash


def prepare_examples(path, *, tokenizer=None, max_examples=32, max_length=128, max_tokens=4096):
    if max_examples < 1 or max_length < 2 or max_tokens < 2:
        raise ValueError("Evaluation limits require examples >= 1 and length/tokens >= 2")
    examples, physical_tokens = [], 0
    for line, row, raw, overlap_hash in _sequences(path, tokenizer, max_length):
        width = min(len(raw["input_ids"]), max_length, max_tokens - physical_tokens)
        if width < 2:
            continue
        ids = raw["input_ids"][:width]
        mask = [int(x) for x in raw.get("attention_mask", [1] * len(raw["input_ids"]))[:width]]
        positions = raw.get("position_ids")
        positions = positions[:width] if positions is not None else np.maximum(np.cumsum(mask) - 1, 0).tolist()
        labels = raw.get("labels", raw["input_ids"])[:width]
        predictors = [t for t in range(width - 1) if mask[t] and mask[t + 1] and labels[t + 1] != -100]
        if not predictors:
            continue
        batch = {"input_ids": ids, "attention_mask": mask, "position_ids": positions, "labels": labels}
        examples.append({"line": line, "row": row, "batch": batch, "predictor_positions": predictors,
                         "sha256": _digest(batch), "overlap_sha256": overlap_hash})
        physical_tokens += width
        if len(examples) >= max_examples or physical_tokens >= max_tokens - 1:
            break
    if not examples:
        raise ValueError("Corpus has no adjacent valid supervised next-token positions within limits")
    return examples


def corpus_receipt(path, examples, limits, *, calibration_corpus=None, tokenizer=None):
    disjointness = {"status": "unverified", "definition": "exact valid token sequences truncated at max_length"}
    if calibration_corpus is not None:
        calibration_hashes = {overlap for _, _, _, overlap in _sequences(calibration_corpus, tokenizer, limits["max_length"])}
        overlap = calibration_hashes & {example["overlap_sha256"] for example in examples}
        if overlap:
            raise ValueError("Evaluation corpus overlaps calibration token sequences")
        disjointness.update(status="verified_exact_sequence_disjoint", calibration_corpus_sha256=_sha256(calibration_corpus))
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path), "limits": limits,
            "prepared_sha256": _digest([example["batch"] for example in examples]),
            "examples": len(examples), "physical_tokens": sum(len(x["batch"]["input_ids"]) for x in examples),
            "prediction_tokens": sum(len(x["predictor_positions"]) for x in examples),
            "disjointness": disjointness}


def validate_reference(manifest, receipt, examples, *, reference_mode="dense", head_chunk_size=128):
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Reference manifest is incomplete or unsupported")
    expected = manifest["corpus"]
    for key in ("sha256", "limits", "prepared_sha256", "examples", "physical_tokens", "prediction_tokens"):
        if expected[key] != receipt[key]:
            raise ValueError(f"Reference corpus mismatch: {key}")
    if [entry["example_sha256"] for entry in manifest["references"]] != [x["sha256"] for x in examples]:
        raise ValueError("Reference example ordering or contents changed")
    # First-fit dense manifests predate these fields and retain their semantics.
    if manifest.get("reference_mode", "dense") != reference_mode:
        raise ValueError("Reference mode mismatch")
    if reference_mode == "compact" and manifest.get("head_chunk_size") != head_chunk_size:
        raise ValueError("Reference head chunk size mismatch")


def token_metrics(teacher, candidate, targets):
    if teacher.shape != candidate.shape or teacher.ndim != 2 or teacher.shape[0] != len(targets):
        raise ValueError("Reference and candidate log probabilities must align")
    if not np.isfinite(teacher).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite evaluation log probabilities")
    if any(not 0 <= int(target) < teacher.shape[1] for target in targets):
        raise ValueError("Target label lies outside the vocabulary")
    rows = []
    for reference, actual, target in zip(teacher, candidate, targets):
        # One vocabulary row at a time bounds host float64 temporary storage.
        reference, actual = reference.astype(np.float64), actual.astype(np.float64)
        top_teacher, top_candidate = int(reference.argmax()), int(actual.argmax())
        rows.append({"teacher_kl": float(np.sum(np.exp(reference) * (reference - actual))),
                     "teacher_nll": -float(reference[target]), "student_nll": -float(actual[target]),
                     "top1_agrees": top_teacher == top_candidate,
                     "teacher_top1": top_teacher, "student_top1": top_candidate, "target": int(target)})
    return rows


def compact_token_metrics(teacher_nll, teacher_top1, student_nll, student_top1, targets, vocabulary_size):
    arrays = [np.asarray(value) for value in (teacher_nll, teacher_top1, student_nll, student_top1, targets)]
    if any(value.shape != (len(targets),) for value in arrays):
        raise ValueError("Compact scores and targets must align")
    teacher_nll, teacher_top1, student_nll, student_top1, targets = arrays
    if not np.isfinite(teacher_nll).all() or not np.isfinite(student_nll).all():
        raise ValueError("Nonfinite compact target NLL")
    for values in (teacher_top1, student_top1, targets):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= vocabulary_size)):
            raise ValueError("Compact top1/target token lies outside the vocabulary")
    return [{"teacher_nll": float(tnll), "student_nll": float(snll),
             "top1_agrees": bool(ttop == stop), "teacher_top1": int(ttop),
             "student_top1": int(stop), "target": int(target)}
            for tnll, ttop, snll, stop, target in zip(*arrays)]


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize zero prediction tokens")
    kl_present = ["teacher_kl" in row for row in rows]
    if any(kl_present) and not all(kl_present):
        raise ValueError("Cannot mix dense and compact token metrics")
    names = ("teacher_nll", "student_nll") + (("teacher_kl",) if all(kl_present) else ())
    result = {name: math.fsum(row[name] for row in rows) / len(rows) for name in names}
    result["teacher_kl_available"] = all(kl_present)
    if not all(kl_present):
        result["teacher_kl_unavailable_reason"] = "compact references do not retain vocabulary distributions"
    result.update(prediction_tokens=len(rows), top1_agreement=sum(row["top1_agrees"] for row in rows) / len(rows))
    result["teacher_ground_truth_top1_accuracy"] = sum(row["teacher_top1"] == row["target"] for row in rows) / len(rows)
    result["student_ground_truth_top1_accuracy"] = sum(row["student_top1"] == row["target"] for row in rows) / len(rows)
    result["nll_increase"] = result["student_nll"] - result["teacher_nll"]
    return result


def _log_probabilities(runtime, example, mx):
    batch = {name: mx.array([values], dtype=mx.int32) for name, values in example["batch"].items() if name != "labels"}
    output = runtime(**batch)
    logits = output.logits[0, mx.array(example["predictor_positions"], dtype=mx.int32), :].astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(log_probs)
    return np.asarray(log_probs)


def _project_logits(runtime, normalized, mx):
    """Mirror MLXQwen3's lexical projection without changing its weight dtype."""
    parameters = runtime.parameters
    if "lexical.head_weight" in parameters:
        logits = normalized @ parameters["lexical.head_weight"].T
    elif "lexical.weight" in parameters:
        logits = normalized @ parameters["lexical.weight"].T
    else:
        logits = (normalized @ parameters["lexical.basis"]) @ parameters["lexical.codebook"].T
    cap = runtime.config.get("final_logit_softcap")
    return mx.tanh(logits / cap) * cap if cap is not None else logits


def _compact_scores(runtime, example, mx, head_chunk_size):
    """Retain O(sequence) scores, with at most chunk_size x vocabulary logits."""
    if type(head_chunk_size) is not int or head_chunk_size < 1:
        raise ValueError("head_chunk_size must be a positive integer")
    parameters = runtime.parameters
    lexical = next(parameters[name] for name in ("lexical.head_weight", "lexical.weight", "lexical.codebook") if name in parameters)
    vocabulary_size = lexical.shape[0]
    positions = example["predictor_positions"]
    targets = np.array([example["batch"]["labels"][t + 1] for t in positions], dtype=np.int64)
    if np.any((targets < 0) | (targets >= vocabulary_size)):
        raise ValueError("Target label lies outside the vocabulary")
    batch = {name: mx.array([values], dtype=mx.int32) for name, values in example["batch"].items() if name != "labels"}
    output = runtime(**batch, return_logits=False, capture=["terminal.normalized"])
    normalized = output.captures["terminal.normalized"][0]
    mx.eval(normalized)
    del output  # Release the unused KV cache before the vocabulary projection.
    nll, top1 = np.empty(len(positions), dtype=np.float32), np.empty(len(positions), dtype=np.int64)
    for start in range(0, len(positions), head_chunk_size):
        end = min(start + head_chunk_size, len(positions))
        hidden = normalized[mx.array(positions[start:end], dtype=mx.int32)]
        logits = _project_logits(runtime, hidden, mx).astype(mx.float32)
        log_normalizer = mx.logsumexp(logits, axis=-1)
        target_logits = logits[mx.arange(end - start), mx.array(targets[start:end], dtype=mx.int32)]
        chunk_nll, chunk_top1 = log_normalizer - target_logits, mx.argmax(logits, axis=-1)
        finite = mx.all(mx.isfinite(logits)) & mx.all(mx.isfinite(chunk_nll))
        mx.eval(chunk_nll, chunk_top1, finite)
        if not bool(finite.item()):
            raise ValueError("Nonfinite compact logits or target NLL")
        nll[start:end], top1[start:end] = np.asarray(chunk_nll), np.asarray(chunk_top1)
        del hidden, logits, log_normalizer, target_logits, chunk_nll, chunk_top1, finite
    return {"nll": nll, "top1": top1, "vocabulary_size": vocabulary_size}


def _parameter_receipt(runtime):
    unique = {id(value): value for value in runtime.parameters.values()}
    by_dtype = {}
    for value in unique.values():
        by_dtype[str(value.dtype)] = by_dtype.get(str(value.dtype), 0) + value.nbytes
    return {"bytes": sum(by_dtype.values()), "bytes_by_dtype": by_dtype,
            "scalar_count": sum(value.size for value in unique.values()),
            "unique_array_count": len(unique), "identity_definition": "distinct MLX array objects; views may share storage"}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    started = time.perf_counter()
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    reference_mode = getattr(args, "reference_mode", "dense")
    head_chunk_size = getattr(args, "head_chunk_size", 128)
    if reference_mode not in ("dense", "compact") or type(head_chunk_size) is not int or head_chunk_size < 1:
        raise ValueError("reference_mode must be dense/compact and head_chunk_size must be positive")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    limits = {name: getattr(args, name) for name in ("max_examples", "max_length", "max_tokens")}
    examples = prepare_examples(args.corpus, tokenizer=tokenizer, **limits)
    receipt = corpus_receipt(args.corpus, examples, limits, calibration_corpus=args.calibration_corpus, tokenizer=tokenizer)
    directory = Path(args.reference_dir).expanduser().resolve()
    if args.phase == "reference":
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Reference directory must be empty; choose a new directory")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        manifest = json.loads((directory / "manifest.json").read_text())
        validate_reference(manifest, receipt, examples, reference_mode=reference_mode, head_chunk_size=head_chunk_size)
        if manifest["precision"]["mlx_enable_tf32"] != os.environ["MLX_ENABLE_TF32"]:
            raise ValueError("Reference and candidate MLX_ENABLE_TF32 settings differ")
    identity = artifact_identity(args.artifact)
    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    mx.reset_peak_memory()
    runtime = (MLXQwen3.load(identity["path"]) if identity["kind"] == "mlx_qwen3"
               else MLXQwen3.from_pretrained(identity["path"]))
    mx.eval(runtime.parameters)
    inference_started = time.perf_counter()
    precision = {"weight_dtype": "preserved", "log_probabilities": "float32", "metric_accumulation": "float64 NumPy per token",
                 "mlx_enable_tf32": os.environ["MLX_ENABLE_TF32"]}
    if reference_mode == "compact":
        precision["log_probabilities"] = "not retained; target NLL computed in float32"
    references, all_rows, per_example = [], [], []
    report_path = Path(args.report) if args.report else directory / (args.phase + "_report.json")
    token_path = report_path.with_suffix(".tokens.jsonl")
    if args.phase == "evaluate":
        token_path.parent.mkdir(parents=True, exist_ok=True)
    token_stream = token_path.open("w") if args.phase == "evaluate" else None
    try:
        for index, example in enumerate(examples):
            actual = (_compact_scores(runtime, example, mx, head_chunk_size) if reference_mode == "compact"
                      else _log_probabilities(runtime, example, mx))
            vocabulary_size = actual["vocabulary_size"] if reference_mode == "compact" else actual.shape[1]
            batch = example["batch"]
            targets = np.array([batch["labels"][t + 1] for t in example["predictor_positions"]], dtype=np.int64)
            if any(target < 0 or target >= vocabulary_size for target in targets):
                raise ValueError("Target label lies outside the vocabulary")
            if args.phase == "reference":
                if reference_mode == "dense" and not np.isfinite(actual).all():
                    raise ValueError("Nonfinite reference log probabilities")
                filename = f"sequence_{index:05d}.npz"
                destination = directory / filename
                scores = ({"teacher_nll": actual["nll"], "teacher_top1": actual["top1"]} if reference_mode == "compact"
                          else {"teacher_log_probs": actual})
                np.savez(destination, **scores, predictor_positions=np.array(example["predictor_positions"]),
                         **{name: np.array(values, dtype=np.int64) for name, values in batch.items()})
                references.append({"file": filename, "sha256": _sha256(destination), "example_sha256": example["sha256"],
                                   "prediction_tokens": len(targets), "vocabulary_size": vocabulary_size})
            else:
                entry = manifest["references"][index]
                if entry["vocabulary_size"] != vocabulary_size:
                    raise ValueError("Reference and candidate vocabulary sizes differ")
                source = (directory / entry["file"]).resolve()
                if not source.is_relative_to(directory) or _sha256(source) != entry["sha256"]:
                    raise ValueError("Reference NPZ identity mismatch")
                with np.load(source, allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[name], values) for name, values in batch.items()):
                        raise ValueError("Reference token IDs, mask, positions or labels changed")
                    if not np.array_equal(saved["predictor_positions"], example["predictor_positions"]):
                        raise ValueError("Reference prediction positions changed")
                    if reference_mode == "compact":
                        rows = compact_token_metrics(saved["teacher_nll"], saved["teacher_top1"], actual["nll"],
                                                     actual["top1"], targets, vocabulary_size)
                    else:
                        rows = token_metrics(saved["teacher_log_probs"], actual, targets)
                for position, row in zip(example["predictor_positions"], rows):
                    row.update(example_index=index, corpus_line=example["line"], corpus_row=example["row"],
                               predictor_position=position, target_position=position + 1)
                    token_stream.write(json.dumps(row, allow_nan=False) + "\n")
                all_rows.extend(rows)
                per_example.append({"example_index": index, "example_sha256": example["sha256"], **summarize(rows)})
            del actual
    finally:
        if token_stream is not None:
            token_stream.close()
    common = {"phase": args.phase, "artifact": identity, "corpus": receipt, "precision": precision,
              "reference_mode": reference_mode, "head_chunk_size": head_chunk_size if reference_mode == "compact" else None,
              "teacher_kl_available": reference_mode == "dense",
              "parameter_storage": _parameter_receipt(runtime), "peak_mlx_memory_bytes": mx.get_peak_memory(),
              "elapsed_seconds": time.perf_counter() - started,
              "inference_and_reference_io_seconds": time.perf_counter() - inference_started,
              "batch_size": 1, "scope": "frozen-corpus teacher-forced comparison; no quality threshold or throughput claim"}
    if reference_mode == "compact":
        common["teacher_kl_unavailable_reason"] = "compact references do not retain vocabulary distributions"
    if args.phase == "reference":
        manifest = {"format_version": 1, "complete": True, **common, "references": references}
        _write_json(directory / "manifest.json", manifest)
        report = {**common, "reference_manifest": str(directory / "manifest.json")}
    else:
        report = {**common, "teacher_artifact": manifest["artifact"], "reference_manifest_sha256": _sha256(directory / "manifest.json"),
                  "aggregate": summarize(all_rows), "per_example": per_example, "per_token_metrics": str(token_path.resolve())}
    _write_json(report_path, report)
    print(json.dumps({"report": str(report_path.resolve()), "phase": args.phase,
                      "aggregate": report.get("aggregate"), "peak_mlx_memory_bytes": common["peak_mlx_memory_bytes"]}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("reference", "evaluate"))
    parser.add_argument("--artifact", required=True, help="Local MLX artifact or dense Qwen3 HF safetensors snapshot")
    parser.add_argument("--corpus", required=True, help="Frozen JSONL evaluation corpus; tokenized rows avoid a tokenizer")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-mode", choices=("dense", "compact"), default="dense",
                        help="Compact stores only target NLL/top1 and cannot measure KL; dense preserves first-fit behavior")
    parser.add_argument("--head-chunk-size", type=int, default=128,
                        help="Maximum vocabulary-projection rows in compact mode; must match the reference phase")
    parser.add_argument("--report")
    parser.add_argument("--tokenizer", help="Existing local tokenizer directory for text rows")
    parser.add_argument("--calibration-corpus", help="Reject exact token sequence overlap with this calibration JSONL")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
