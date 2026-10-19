"""Local-only conversion, bounded calibration, and generation for MLX Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _local_directory(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Local directory does not exist: {path}")
    return path


def _tokenizer(directory):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(_local_directory(directory)), local_files_only=True, trust_remote_code=False
    )


def _layers(value, count):
    if value == "all":
        return tuple(range(count))
    indices = set()
    try:
        for part in value.split(","):
            limits = part.split("-")
            if len(limits) == 1:
                indices.add(int(limits[0]))
            elif len(limits) == 2:
                start, end = map(int, limits)
                if end < start:
                    raise ValueError
                indices.update(range(start, end + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise ValueError("layers must be all or zero-based indices/ranges, e.g. 0,2,4-7") from exc
    if not indices or min(indices) < 0 or max(indices) >= count:
        raise ValueError(f"layer indices must lie in [0, {count - 1}]")
    return tuple(sorted(indices))


def _read_corpus(path, *, tokenizer=None, max_examples=16, max_length=128, max_tokens=2048, vocab_size=None):
    """Read bounded single-sequence batches; no corpus rows are retained in reports."""
    if max_examples <= 0 or max_length < 2 or max_tokens < 2:
        raise ValueError("Corpus limits require positive max_examples and at least two tokens")
    batches, tokens = [], 0
    with Path(path).expanduser().open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str) or (isinstance(item, dict) and "text" in item):
                if tokenizer is None:
                    raise ValueError("Text corpus requires --tokenizer pointing to local tokenizer files")
                text = item if isinstance(item, str) else item["text"]
                if not isinstance(text, str):
                    raise ValueError(f"Corpus line {line_number}: text must be a string")
                item = {"input_ids": tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)}
            if not isinstance(item, dict) or not isinstance(item.get("input_ids"), list):
                raise ValueError(f"Corpus line {line_number}: expected text or input_ids")
            ids = item["input_ids"]
            batched = bool(ids and isinstance(ids[0], list))
            sequences = ids if batched else [ids]
            for row, sequence in enumerate(sequences):
                if not isinstance(sequence, list) or any(type(token) is not int or token < 0 for token in sequence):
                    raise ValueError(f"Corpus line {line_number}: input_ids must contain nonnegative integers")
                width = min(len(sequence), max_length, max_tokens - tokens)
                if width < 2:
                    continue
                if vocab_size is not None and any(token >= vocab_size for token in sequence[:width]):
                    raise ValueError(f"Corpus line {line_number}: input_ids outside the runtime vocabulary")
                batch = {"input_ids": [sequence[:width]]}
                for key in ("attention_mask", "position_ids"):
                    if key not in item:
                        continue
                    rows = item[key]
                    if batched:
                        if not isinstance(rows, list) or len(rows) != len(sequences):
                            raise ValueError(f"Corpus line {line_number}: {key} batch size differs")
                        values = rows[row]
                    else:
                        values = rows
                    if not isinstance(values, list) or len(values) != len(sequence):
                        raise ValueError(f"Corpus line {line_number}: {key} must align with input_ids")
                    if key == "attention_mask" and any(value not in (0, 1, False, True) for value in values):
                        raise ValueError("attention_mask must be binary")
                    if key == "position_ids" and any(type(value) is not int or value < 0 for value in values):
                        raise ValueError("position_ids must contain nonnegative integers")
                    batch[key] = [values[:width]]
                valid = sum(batch.get("attention_mask", [[1] * width])[0])
                if valid < 2:
                    continue
                batches.append(batch)
                tokens += width
                if len(batches) >= max_examples or tokens >= max_tokens - 1:
                    return batches
    if not batches:
        raise ValueError("Corpus has no sequences with at least two valid tokens within the limits")
    return batches


def _destination(path, overwrite):
    target = Path(path).expanduser().resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())) and not overwrite:
        raise ValueError(f"Output is not empty: {target}; choose a new directory or use --overwrite")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(runtime):
    dtypes = {}
    for value in runtime.parameters.values():
        dtype = str(value.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + value.size
    return {"scalar_count": sum(dtypes.values()), "scalars_by_dtype": dtypes, "layer_count": len(runtime.layers)}


def _save(runtime, destination, report):
    runtime.save(destination)
    report_path = destination / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(destination), "report": str(report_path), "parameters": _inventory(runtime)}))


def _precision():
    return {"mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"), "runtime": "MLX", "heldout_confirmation": False}


def _convert(args):
    import mlx.core as mx
    from .mlx_qwen3 import MLXQwen3

    source = _local_directory(args.hf_snapshot)
    destination = _destination(args.output, args.overwrite)
    if source == destination:
        raise ValueError("Conversion output must differ from the HF snapshot")
    dtype = None if args.dtype == "preserve" else getattr(mx, args.dtype)
    runtime = MLXQwen3.from_pretrained(source, dtype=dtype)
    provenance = {
        "source_hf_snapshot": str(source), "source_config_sha256": _sha256(source / "config.json"),
        "tokenizer_path": str(_local_directory(args.tokenizer)) if args.tokenizer else None,
        "requested_dtype": args.dtype,
    }
    runtime.config["provenance"] = provenance
    _save(runtime, destination, {"operation": "convert", **_precision(), **provenance, "parameters": _inventory(runtime)})


def _fit(args):
    from .mlx_qwen3 import MLXQwen3
    from .mlx_qwen3_fit import capture_runtime_rows, fit_reduced_attention, fit_chart_mlp, optimize_trajectory

    source = _local_directory(args.artifact)
    destination = _destination(args.output, args.overwrite)
    if source == destination:
        raise ValueError("Fit output must differ from its source artifact")
    runtime = MLXQwen3.load(source)
    teacher = runtime if args.trajectory_steps else None
    tokenizer = _tokenizer(args.tokenizer) if args.tokenizer else None
    lexical = next(runtime.parameters[name] for name in ("lexical.weight", "lexical.codebook") if name in runtime.parameters)
    batches = _read_corpus(args.corpus, tokenizer=tokenizer, max_examples=args.max_examples,
                           max_length=args.max_length, max_tokens=args.max_tokens, vocab_size=lexical.shape[0])
    indices = _layers(args.layers, len(runtime.layers))
    reports, updated_names = [], set()
    before = _inventory(runtime)
    for index in indices:
        for part in ("attention", "mlp"):
            if args.components != "both" and args.components != part:
                continue
            print(json.dumps({"event": "fitting", "layer": index, "component": part}), file=sys.stderr, flush=True)
            prefix = f"{part}.{index}."
            sites = (prefix + "normalized", prefix + ("context" if part == "attention" else "output"))
            captures = capture_runtime_rows(runtime, batches, sites)
            if part == "attention":
                result = fit_reduced_attention(runtime, index, captures,
                    input_rank=args.attention_input_rank, output_rank=args.attention_output_rank,
                    relative_ridge=args.relative_ridge, solver=args.solver)
            else:
                result = fit_chart_mlp(runtime, index, captures,
                    chart_count=args.chart_count, chart_rank=args.chart_rank, generator_rank=args.generator_rank,
                    relative_ridge=args.relative_ridge, solver=args.solver)
            runtime = result.runtime
            updated_names.update(result.updates)
            reports.append({"layer": index, "component": part, **result.report})
            del captures, result
    trajectory = None
    if args.trajectory_steps:
        coefficient_names = ("q_weight", "k_weight", "v_weight", "output_encoder_weight", "output_decoder_weight",
                             "affine_weight", "affine_bias", "local_generator_encoders", "chart_mode_decoders")
        names = sorted(name for name in updated_names if name.rsplit(".", 1)[-1] in coefficient_names)
        result = optimize_trajectory(runtime, teacher, batches, parameter_names=names,
            steps=args.trajectory_steps, learning_rate=args.learning_rate, objective=args.trajectory_objective,
            anchor_weight=args.anchor_weight)
        runtime, trajectory = result.runtime, result.report
    report = {
        "operation": "fit", **_precision(), "source_artifact": str(source),
        "source_metadata_sha256": _sha256(source / "mlx_qwen3.json"),
        "corpus_sha256": _sha256(Path(args.corpus).expanduser()),
        "tokenizer_path": str(_local_directory(args.tokenizer)) if args.tokenizer else None,
        "calibration_examples": len(batches),
        "calibration_physical_tokens": sum(len(batch["input_ids"][0]) for batch in batches),
        "layers": indices, "solver": args.solver,
        "solver_precision": (
            "MLX float32 moments and CPU linear algebra"
            if args.solver == "mlx" else
            "NumPy float64 host factorization of MLX float32 moments; anchored ridge accumulates in float64"
        ),
        "parameters_before": before, "parameters_after": _inventory(runtime),
        "fits": reports, "trajectory": trajectory,
        "evaluation_scope": "calibration only; no held-out quality or speed claim",
    }
    runtime.config["fit_provenance"] = {key: report[key] for key in ("source_artifact", "source_metadata_sha256", "corpus_sha256", "layers", "solver")}
    _save(runtime, destination, report)


def _generate(args):
    import mlx.core as mx
    from .mlx_qwen3 import MLXQwen3

    runtime = MLXQwen3.load(_local_directory(args.artifact))
    tokenizer = _tokenizer(args.tokenizer)
    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=not args.chat)
    if not ids or len(ids) > args.max_prompt_tokens:
        raise ValueError(f"Prompt must contain 1..{args.max_prompt_tokens} tokens")
    lexical = next(runtime.parameters[name] for name in ("lexical.weight", "lexical.codebook") if name in runtime.parameters)
    if any(token < 0 or token >= lexical.shape[0] for token in ids):
        raise ValueError("Prompt token IDs are outside the runtime vocabulary; use the matching local tokenizer")
    start = time.perf_counter()
    output = runtime.prefill(mx.array([ids], dtype=mx.int32))
    generated = []
    for index in range(args.max_new_tokens):
        token = int(mx.argmax(output.logits[0, -1]).item())
        generated.append(token)
        if not args.ignore_eos and token == tokenizer.eos_token_id:
            break
        if index + 1 < args.max_new_tokens:
            output = runtime.decode(mx.array([[token]], dtype=mx.int32), output.cache)
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "operation": "generate", **_precision(), "artifact": str(Path(args.artifact).expanduser().resolve()),
            "tokenizer_path": str(_local_directory(args.tokenizer)), "prompt_tokens": len(ids),
            "generated_token_ids": generated, "completion": text, "elapsed_seconds": elapsed,
            "decoding": "greedy", "timing_scope": "single run including prefill; not a benchmark",
        }, indent=2, allow_nan=False) + "\n")


def _xray(args):
    import mlx.core as mx
    import numpy as np
    from .mlx_qwen3 import MLXQwen3

    runtime = MLXQwen3.load(_local_directory(args.artifact))
    tokenizer = _tokenizer(args.tokenizer)
    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    ids = tokenizer.encode(prompt, add_special_tokens=not args.chat)
    if not ids or len(ids) > args.max_prompt_tokens:
        raise ValueError(f"Prompt must contain 1..{args.max_prompt_tokens} tokens")
    lexical = next(
        runtime.parameters[name]
        for name in ("lexical.weight", "lexical.codebook")
        if name in runtime.parameters
    )
    if any(token < 0 or token >= lexical.shape[0] for token in ids):
        raise ValueError("Prompt token IDs are outside the runtime vocabulary")

    chart_layers = sorted(
        i for i in range(len(runtime.layers)) if f"mlp.{i}.chart_centers" in runtime.parameters
    )
    reduced_attn_layers = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.input_basis" in runtime.parameters
    )
    capture_sites = [f"mlp.{i}.normalized" for i in chart_layers] + [
        f"mlp.{i}.delta" for i in chart_layers
    ] + [f"attention.{i}.delta" for i in reduced_attn_layers]
    arr_ids = mx.array([ids], dtype=mx.int32)
    out = runtime(arr_ids, capture=capture_sites or False)
    logits = out.logits[0, -1].astype(mx.float32)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(logits, probs)
    logits_np = np.asarray(logits)
    probs_np = np.asarray(probs)
    top_idx = np.argsort(logits_np)[::-1][:5]
    win_id = int(top_idx[0])

    token_routing = {}
    for i in chart_layers:
        n_val = out.captures[f"mlp.{i}.normalized"][0].astype(mx.float32)
        centers = runtime.parameters[f"mlp.{i}.chart_centers"].astype(mx.float32)
        scales = runtime.parameters[f"mlp.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["mlp"].get("temperature", 1.0))
        disp = n_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(
            mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1)
        )
        token_routing[str(i)] = [
            {
                "position": pos,
                "token": tokenizer.decode([tok]),
                "dominant_chart": int(np.argmax(memb[pos])),
                "dominant_membership": float(np.max(memb[pos])),
            }
            for pos, tok in enumerate(ids)
        ]

    ablation_impacts = []
    for i in reduced_attn_layers:
        site = f"attention.{i}.delta"
        ablated = runtime(arr_ids, perturbations={site: -out.captures[site]}).logits[0, -1].astype(mx.float32)
        mx.eval(ablated)
        ab_np = np.asarray(ablated)
        ablation_impacts.append({
            "site": site,
            "winning_logit_push": float(logits_np[win_id] - ab_np[win_id]),
            "ablated_top1_token_id": int(np.argmax(ab_np)),
            "ablated_top1_token": tokenizer.decode([int(np.argmax(ab_np))]),
        })
    for i in chart_layers:
        site = f"mlp.{i}.delta"
        ablated = runtime(arr_ids, perturbations={site: -out.captures[site]}).logits[0, -1].astype(mx.float32)
        mx.eval(ablated)
        ab_np = np.asarray(ablated)
        ablation_impacts.append({
            "site": site,
            "winning_logit_push": float(logits_np[win_id] - ab_np[win_id]),
            "ablated_top1_token_id": int(np.argmax(ab_np)),
            "ablated_top1_token": tokenizer.decode([int(np.argmax(ab_np))]),
        })

    payload = {
        "operation": "xray",
        **_precision(),
        "artifact": str(Path(args.artifact).expanduser().resolve()),
        "prompt_tokens": len(ids),
        "reduced_attention_layers": reduced_attn_layers,
        "chart_mlp_layers": chart_layers,
        "top5_predictions": [
            {
                "token_id": int(idx),
                "token": tokenizer.decode([int(idx)]),
                "logit": float(logits_np[idx]),
                "probability": float(probs_np[idx]),
            }
            for idx in top_idx
        ],
        "chart_routing": token_routing,
        "ablation_impacts": ablation_impacts,
    }
    print(json.dumps(payload, indent=2, allow_nan=False))
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Convert an existing local HF safetensors snapshot without a torch model")
    convert.add_argument("--hf-snapshot", required=True)
    convert.add_argument("--tokenizer", help="Optional local tokenizer provenance; tokenizer files are not copied")
    convert.add_argument("--dtype", choices=("preserve", "bfloat16", "float16", "float32"), default="preserve")
    convert.set_defaults(run=_convert)
    fit = commands.add_parser("fit", help="Fit selected layers on bounded local JSONL calibration rows")
    fit.add_argument("--artifact", required=True)
    fit.add_argument("--corpus", required=True, help="JSONL strings, text objects, or input_ids objects")
    fit.add_argument("--tokenizer", help="Required for text JSONL; always loaded with local_files_only=True")
    fit.add_argument("--layers", required=True, help="Zero-based indices/ranges (0,2,4-7), or all")
    fit.add_argument("--components", choices=("both", "attention", "mlp"), default="both")
    fit.add_argument("--max-examples", type=_positive_int, default=16)
    fit.add_argument("--max-length", type=_positive_int, default=128)
    fit.add_argument("--max-tokens", type=_positive_int, default=2048)
    fit.add_argument("--attention-input-rank", type=_positive_int, default=256)
    fit.add_argument("--attention-output-rank", type=_positive_int, default=64)
    fit.add_argument("--chart-count", type=_positive_int, default=4)
    fit.add_argument("--chart-rank", type=_positive_int, default=16)
    fit.add_argument("--generator-rank", type=_positive_int, default=None)
    fit.add_argument("--relative-ridge", type=float, default=0.01)
    fit.add_argument("--solver", choices=("mlx", "numpy_float64"), default="mlx")
    fit.add_argument("--trajectory-steps", type=int, default=0)
    fit.add_argument("--trajectory-objective", choices=("terminal_mse", "teacher_kl"), default="terminal_mse")
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.add_argument("--anchor-weight", type=float, default=0.0)
    fit.set_defaults(run=_fit)
    for command in (convert, fit):
        command.add_argument("--output", required=True)
        command.add_argument("--overwrite", action="store_true")
    generate = commands.add_parser("generate", help="Greedy prefill and cached decode from a saved MLX artifact")
    generate.add_argument("--artifact", required=True)
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--chat", action="store_true", help="Apply local tokenizer chat template with thinking disabled")
    generate.add_argument("--max-prompt-tokens", type=_positive_int, default=512)
    generate.add_argument("--max-new-tokens", type=_positive_int, default=32)
    generate.add_argument("--ignore-eos", action="store_true")
    generate.add_argument("--report", help="Optional JSON completion/timing report path")
    generate.set_defaults(run=_generate)
    xray = commands.add_parser("xray", help="Inspect chart routing and counterfactual event ablation pushes")
    xray.add_argument("--artifact", required=True)
    xray.add_argument("--tokenizer", required=True)
    xray.add_argument("--prompt", required=True)
    xray.add_argument("--chat", action="store_true")
    xray.add_argument("--max-prompt-tokens", type=_positive_int, default=512)
    xray.add_argument("--report", help="Optional JSON X-Ray report path")
    xray.set_defaults(run=_xray)
    return parser


def main(argv=None):
    # Must be set before the first MLX import/compute on M5. An explicitly
    # configured user value is preserved and recorded in every report.
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "trajectory_steps", 0) < 0:
        parser.error("--trajectory-steps must be nonnegative")
    try:
        args.run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
