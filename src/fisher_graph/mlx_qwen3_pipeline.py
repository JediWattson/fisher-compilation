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


def _resolve_artifact(value):
    """Resolve a local artifact directory or a Hugging Face repo id to a path."""
    path = Path(str(value)).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(value))).resolve()


def _tokenizer(directory):
    from transformers import AutoTokenizer
    path = Path(str(directory)).expanduser()
    if path.is_dir():
        return AutoTokenizer.from_pretrained(
            str(path.resolve()), local_files_only=True, trust_remote_code=False
        )
    return AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)


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
    from .mlx_qwen3_fit import (
        capture_empirical_token_fisher,
        capture_runtime_rows,
        fit_chart_mlp,
        fit_deltanet_carrier_basis,
        fit_reduced_attention,
        fit_reduced_deltanet,
        optimize_trajectory,
    )

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
        is_deltanet = "deltanet" in runtime.layers[index]
        for part in ("attention", "mlp"):
            if args.components != "both" and args.components != part:
                continue
            comp_label = "deltanet" if (part == "attention" and is_deltanet) else part
            print(json.dumps({"event": "fitting", "layer": index, "component": comp_label}), file=sys.stderr, flush=True)
            prefix = f"{comp_label}.{index}."
            sites = (prefix + "normalized", prefix + ("context" if part == "attention" else "output"))
            captures = capture_runtime_rows(runtime, batches, sites)
            if part == "attention" and is_deltanet:
                result = fit_reduced_deltanet(runtime, index, captures,
                    input_rank=args.attention_input_rank, output_rank=args.attention_output_rank,
                    relative_ridge=args.relative_ridge, solver=args.solver)
            elif part == "attention":
                result = fit_reduced_attention(runtime, index, captures,
                    input_rank=args.attention_input_rank, output_rank=args.attention_output_rank,
                    relative_ridge=args.relative_ridge, solver=args.solver)
            else:
                result = fit_chart_mlp(runtime, index, captures,
                    chart_count=args.chart_count, chart_rank=args.chart_rank, generator_rank=args.generator_rank,
                    relative_ridge=args.relative_ridge, solver=args.solver)
            runtime = result.runtime
            updated_names.update(result.updates)
            reports.append({"layer": index, "component": comp_label, **result.report})
            del captures, result
            import mlx.core as _mx
            _mx.clear_cache()
            if part == "attention" and is_deltanet and getattr(args, "carrier_rank", None) is not None:
                qk_sites = (f"deltanet.{index}.q", f"deltanet.{index}.k")
                qk_rows = capture_runtime_rows(runtime, batches, qk_sites)
                fisher_qk = capture_empirical_token_fisher(runtime, batches, qk_sites, max_tokens=min(args.max_tokens, 32))
                c_res = fit_deltanet_carrier_basis(
                    runtime, index, qk_rows, carrier_rank=args.carrier_rank,
                    fisher_diagonal=fisher_qk["diagonal"], solver=args.solver,
                )
                runtime = c_res.runtime
                updated_names.update(c_res.updates)
                reports.append({"layer": index, "component": "deltanet_carrier", **c_res.report})
                del qk_rows, fisher_qk, c_res
    trajectory = None
    if args.trajectory_steps:
        coefficient_names = ("q_weight", "k_weight", "v_weight", "qkv_weight", "z_weight", "carrier_basis",
                             "output_encoder_weight", "output_decoder_weight",
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


_PRESET_PROMPTS = (
    "The capital of France is",
    "In a right triangle, the square of the hypotenuse equals",
    "Water boils at 100 degrees Celsius at standard",
    "The chemical symbol for gold is",
    "Isaac Newton formulated the laws of",
    "Photosynthesis converts light energy into",
)


def _sample_next_token(logits_np, context_ids, *, temperature, top_p, repetition_penalty):
    import numpy as np

    logits = logits_np.astype(np.float64).copy()
    if repetition_penalty != 1.0:
        seen = np.unique(context_ids)
        vals = logits[seen]
        logits[seen] = np.where(vals > 0.0, vals / repetition_penalty, vals * repetition_penalty)
    logits = logits / max(float(temperature), 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cum, top_p, side="left")) + 1
        keep = order[: max(cutoff, 1)]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
    return int(np.random.choice(len(probs), p=probs))


def _greedy_continuation(runtime, tokenizer, ids, *, max_new_tokens, ignore_eos=False,
                         muted_sites=(), muted_charts=(),
                         temperature=0.0, top_p=1.0, repetition_penalty=1.0):
    import mlx.core as mx
    import numpy as np

    start = time.perf_counter()
    output = runtime.prefill(
        mx.array([ids], dtype=mx.int32),
        muted_sites=muted_sites,
        muted_charts=muted_charts,
    )
    first_logits = output.logits[0, -1].astype(mx.float32)
    first_probs = mx.softmax(first_logits, axis=-1)
    mx.eval(first_logits, first_probs)
    logits_np = np.asarray(first_logits)
    probs_np = np.asarray(first_probs)
    top5_idx = np.argsort(logits_np)[::-1][:5]
    top5 = [
        (tokenizer.decode([int(i)]), float(probs_np[i]) * 100.0, float(logits_np[i]))
        for i in top5_idx
    ]
    generated = []
    for step in range(max_new_tokens):
        logits_np = np.asarray(output.logits[0, -1].astype(mx.float32))
        if temperature and temperature > 0.0:
            token = _sample_next_token(
                logits_np, ids + generated,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            token = int(mx.argmax(output.logits[0, -1]).item())
        generated.append(token)
        if not ignore_eos and token == tokenizer.eos_token_id:
            break
        if step + 1 < max_new_tokens:
            output = runtime.decode(
                mx.array([[token]], dtype=mx.int32),
                output.cache,
                muted_sites=muted_sites,
                muted_charts=muted_charts,
            )
    elapsed = time.perf_counter() - start
    return tokenizer.decode(generated, skip_special_tokens=True), generated, top5, elapsed


def _default_chat_artifact():
    local = Path(".local-runs/qwen3_5-9b-carrier-compile/waterfill_carrier_50pct")
    if local.is_dir():
        return str(local)
    local = Path(".local-runs/qwen3-ewc-refine/joint_waterfill_286mib")
    if local.is_dir():
        return str(local)
    return "jediwattzon/qwen3-1.7b-fisher-ewc-graph"


def _default_tokenizer(runtime):
    if runtime.config.get("source") == "hf_qwen3_5":
        return "mlx-community/Qwen3.5-9B-MLX-4bit"
    return "Qwen/Qwen3-1.7B"


def _chat(args):
    args.chat = True
    args.presets = False
    args.reference_artifact = None
    args.max_prompt_tokens = args.max_context_tokens
    _interactive(args)


def _interactive(args):
    from .mlx_qwen3 import MLXQwen3

    artifact_dir = _resolve_artifact(args.artifact)
    runtime = MLXQwen3.load(artifact_dir)
    reference = MLXQwen3.load(_resolve_artifact(args.reference_artifact)) if args.reference_artifact else None
    tokenizer = _tokenizer(args.tokenizer or _default_tokenizer(runtime))
    xray_on = bool(args.xray)
    debug_on = bool(args.debug)
    chat_on = bool(args.chat)
    max_new_tokens = int(args.max_new_tokens)
    temperature = float(args.temperature)
    top_p = float(args.top_p)
    rep_penalty = float(args.repetition_penalty)
    muted_sites = {f"attention.{l}.delta" for l in (args.mute_attn_layer or ())} | {
        f"mlp.{l}.delta" for l in (args.mute_mlp_layer or ())
    } | {f"deltanet.{l}.delta" for l in (getattr(args, "mute_deltanet_layer", None) or ())}
    muted_charts = set()
    for spec in args.mute_chart or ():
        parts = str(spec).split(":")
        if len(parts) != 2:
            raise ValueError("--mute-chart must have format LAYER:CHART")
        muted_charts.add((int(parts[0]), int(parts[1])))

    reduced_attn = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.input_basis" in runtime.parameters
    )
    chart_mlps = sorted(
        i for i in range(len(runtime.layers)) if f"mlp.{i}.chart_centers" in runtime.parameters
    )
    reduced_deltanet = sorted(
        i for i in range(len(runtime.layers))
        if f"deltanet.{i}.output_encoder_weight" in runtime.parameters or f"deltanet.{i}.input_basis" in runtime.parameters
    )
    deltanet_ranks = {
        i: runtime.layers[i]["deltanet"].get("carrier_rank", runtime.layers[i]["deltanet"]["key_head_dimension"])
        for i in range(len(runtime.layers))
        if "deltanet" in runtime.layers[i]
    }

    def encode_prompt(raw_text: str) -> list[int]:
        rendered = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if chat_on
            else raw_text
        )
        ids = tokenizer.encode(rendered, add_special_tokens=not chat_on)
        if not ids or len(ids) > args.max_prompt_tokens:
            raise ValueError(f"Prompt must contain 1..{args.max_prompt_tokens} tokens")
        return ids

    def run_one(raw_text: str):
        ids = encode_prompt(raw_text)
        text, gen_ids, top5, elapsed = _greedy_continuation(
            runtime,
            tokenizer,
            ids,
            max_new_tokens=max_new_tokens,
            ignore_eos=args.ignore_eos,
            muted_sites=muted_sites,
            muted_charts=muted_charts,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )
        tok_s = len(gen_ids) / max(elapsed, 1e-6)
        if debug_on:
            print(f"\n[prompt]   {raw_text}")
            if muted_sites or muted_charts:
                print(f"[mutes]    sites={sorted(muted_sites)} charts={sorted(muted_charts)}")
            mode = "greedy" if not temperature else f"T={temperature}/top-p={top_p}/pen={rep_penalty}"
            print(f"[compiled] {text!r}  ({len(gen_ids)} tok in {elapsed:.2f}s, {tok_s:.1f} tok/s) [{mode}]")
            top_str = ", ".join(f"{t!r}:{p:.1f}%" for t, p, _ in top5[:5])
            print(f"[top-5]    {top_str}")
        else:
            print(f"\n{text}")
        if reference is not None:
            ref_text, ref_ids, ref_top5, ref_elapsed = _greedy_continuation(
                reference,
                tokenizer,
                ids,
                max_new_tokens=max_new_tokens,
                ignore_eos=args.ignore_eos,
            )
            if debug_on:
                ref_top_str = ", ".join(f"{t!r}:{p:.1f}%" for t, p, _ in ref_top5[:5])
                print(f"[exact]    {ref_text!r}  ({len(ref_ids)} tok in {ref_elapsed:.2f}s)")
                print(f"[ex-top-5] {ref_top_str}")
        if xray_on:
            xray_ns = argparse.Namespace(
                artifact=str(artifact_dir),
                tokenizer=args.tokenizer,
                prompt=raw_text,
                chat=chat_on,
                max_prompt_tokens=args.max_prompt_tokens,
                report=None,
            )
            _xray(xray_ns)

    # ---- persistent chat state (compiled-model conversation) ----
    chat_history = []          # [{role, content}, ...] for apply_chat_template
    chat_ctx_tokens = 0        # total tokens the model has seen this conversation
    _system_prompt = args.system_prompt or "You are a helpful assistant."

    def run_chat_turn(user_text: str):
        nonlocal chat_ctx_tokens
        if not chat_history and _system_prompt:
            chat_history.append({"role": "system", "content": _system_prompt})
        chat_history.append({"role": "user", "content": user_text})
        rendered = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > args.max_context_tokens:
            raise ValueError(
                f"Conversation context is {len(ids)} tokens (> {args.max_context_tokens}); "
                "use /reset to clear chat history"
            )
        text_resp, gen_ids, top5, elapsed = _greedy_continuation(
            runtime,
            tokenizer,
            ids,
            max_new_tokens=max_new_tokens,
            ignore_eos=args.ignore_eos,
            muted_sites=muted_sites,
            muted_charts=muted_charts,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )
        chat_history.append({"role": "assistant", "content": text_resp})
        chat_ctx_tokens = len(ids) + len(gen_ids)
        tok_s = len(gen_ids) / max(elapsed, 1e-6)
        if debug_on:
            print(f"\n[user]     {user_text}")
            print(f"[ctx]      {chat_ctx_tokens} tokens | {len(chat_history)//2} turns | prompt={len(ids)} tok")
            if muted_sites or muted_charts:
                print(f"[mutes]    sites={sorted(muted_sites)} charts={sorted(muted_charts)}")
            print(f"[model]    {text_resp!r}  ({len(gen_ids)} tok in {elapsed:.2f}s, {tok_s:.1f} tok/s)"
                  + ("" if not temperature else f" [T={temperature}/top-p={top_p}/pen={rep_penalty}]"))
            top_str = ", ".join(f"{t!r}:{p:.1f}%" for t, p, _ in top5[:5])
            print(f"[top-5]    {top_str}")
        else:
            print(f"\n{text_resp}")
        if xray_on:
            xray_ns = argparse.Namespace(
                artifact=str(artifact_dir),
                tokenizer=args.tokenizer,
                prompt=rendered,
                chat=False,
                max_prompt_tokens=args.max_context_tokens,
                report=None,
            )
            _xray(xray_ns)

    if args.presets:
        for p in _PRESET_PROMPTS:
            run_one(p)
        return
    if args.prompt is not None:
        run_one(args.prompt)
        return

    print("==============================================================================")
    print("  Fisher-Graph Qwen3 MLX Interactive Harness (Freeform REPL + X-Ray + Mutes)")
    print("==============================================================================")
    print(f"  Artifact       : {artifact_dir}")
    print(f"  Reduced Attn   : {reduced_attn}")
    print(f"  Chart MLPs     : {chart_mlps}")
    if deltanet_ranks:
        print(f"  Reduced DNet   : {reduced_deltanet}")
        print(f"  DeltaNet Ranks : {deltanet_ranks}")
    print(f"  Chat Template  : {'ON' if chat_on else 'OFF'} (toggle with /chat)")
    print(f"  X-Ray Mode     : {'ON' if xray_on else 'OFF'} (toggle with /xray)")
    print("  Commands       : /xray | /chat | /debug | /context | /reset |")
    print("                   /tokens <N> | /temp <F> | /topp <F> | /penalty <F> |")
    print("                   /preset | /unmute | /quit")
    print("                   /mute attn <L> | /mute mlp <L> | /mute deltanet <L> |")
    print("                   /mute chart <L> <C>")
    print("==============================================================================")

    while True:
        try:
            prompt_label = "qwen3(chat)>" if chat_on else "qwen3>"
            line = input(f"\n{prompt_label} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "quit", "exit"):
            break
        if line == "/xray":
            xray_on = not xray_on
            print(f"[harness] X-Ray mode is now {'ON' if xray_on else 'OFF'}")
            continue
        if line == "/debug":
            debug_on = not debug_on
            print(f"[harness] Debug stats are now {'ON' if debug_on else 'OFF'}")
            continue
        if line == "/chat":
            chat_on = not chat_on
            if chat_on:
                chat_history.clear()
                chat_ctx_tokens = 0
                print("[harness] Chat mode ON (persistent conversation with this model). /reset clears history.")
            else:
                print("[harness] Chat template is now OFF (raw completion mode)")
            continue
        if line == "/context":
            print(f"[harness] chat_ctx_tokens={chat_ctx_tokens} | turns={len(chat_history)//2} | "
                  f"mutes={sorted(muted_sites) or 'none'} charts={sorted(muted_charts) or 'none'}")
            for msg in chat_history[-6:]:
                body = msg["content"] if len(msg["content"]) <= 72 else msg["content"][:69] + "..."
                print(f"           [{msg['role']:>9}] {body}")
            continue
        if line == "/reset":
            chat_history.clear()
            chat_ctx_tokens = 0
            print("[harness] Cleared conversation history")
            continue
        if line == "/unmute":
            muted_sites.clear()
            muted_charts.clear()
            print("[harness] Cleared all active layer and chart mutes")
            continue
        if line == "/preset":
            for p in _PRESET_PROMPTS:
                run_one(p)
            continue
        if line.startswith("/tokens "):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                max_new_tokens = int(parts[1])
                print(f"[harness] max_new_tokens = {max_new_tokens}")
            else:
                print("[harness] usage: /tokens <positive_int>")
            continue
        if line.startswith("/temp "):
            try:
                temperature = float(line.split()[1])
                assert temperature >= 0.0
                print(f"[harness] temperature = {temperature} ({'sampling' if temperature > 0 else 'greedy'})")
            except (ValueError, IndexError, AssertionError):
                print("[harness] usage: /temp <float>=0 for greedy, e.g. /temp 0.7")
            continue
        if line.startswith("/topp "):
            try:
                top_p = float(line.split()[1])
                assert 0.0 < top_p <= 1.0
                print(f"[harness] top_p = {top_p}")
            except (ValueError, IndexError, AssertionError):
                print("[harness] usage: /topp <float in (0,1]>")
            continue
        if line.startswith("/penalty "):
            try:
                rep_penalty = float(line.split()[1])
                assert rep_penalty >= 1.0
                print(f"[harness] repetition_penalty = {rep_penalty}")
            except (ValueError, IndexError, AssertionError):
                print("[harness] usage: /penalty <float>=1.0, e.g. /penalty 1.15")
            continue
        if line.startswith("/mute "):
            parts = line.split()
            if len(parts) == 3 and parts[1] == "attn" and parts[2].isdigit():
                muted_sites.add(f"attention.{int(parts[2])}.delta")
                print(f"[harness] Muted attention.{int(parts[2])}.delta")
            elif len(parts) == 3 and parts[1] == "mlp" and parts[2].isdigit():
                muted_sites.add(f"mlp.{int(parts[2])}.delta")
                print(f"[harness] Muted mlp.{int(parts[2])}.delta")
            elif len(parts) == 3 and parts[1] == "deltanet" and parts[2].isdigit():
                muted_sites.add(f"deltanet.{int(parts[2])}.delta")
                print(f"[harness] Muted deltanet.{int(parts[2])}.delta")
            elif len(parts) == 4 and parts[1] == "chart" and parts[2].isdigit() and parts[3].isdigit():
                muted_charts.add((int(parts[2]), int(parts[3])))
                print(f"[harness] Muted mlp.{int(parts[2])} chart {int(parts[3])}")
            else:
                print("[harness] usage: /mute attn <L> | /mute mlp <L> | /mute deltanet <L> | /mute chart <L> <C>")
            continue
        if chat_on:
            run_chat_turn(line)
        else:
            run_one(line)


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
    fit.add_argument("--carrier-rank", type=_positive_int, default=None, help="Optional GatedDeltaNet carrier rank along Dk (Play 2)")
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
    interactive = commands.add_parser(
        "interactive",
        help="Interactive freeform prompt REPL with X-Ray, layer/chart muting, and exact comparison",
    )
    interactive.add_argument("--artifact", required=True)
    interactive.add_argument("--reference-artifact", help="Optional exact MLX artifact for side-by-side comparison")
    interactive.add_argument("--tokenizer", help="Local tokenizer dir or HF repo id (default: inferred from artifact source)")
    interactive.add_argument("--prompt", help="Run a single freeform prompt non-interactively and exit")
    interactive.add_argument("--presets", action="store_true", help="Run the 6 preset prompts and exit")
    interactive.add_argument("--chat", action="store_true", help="Apply Qwen3 chat template (thinking disabled)")
    interactive.add_argument("--xray", action="store_true", help="Enable per-prompt X-Ray chart/ablation output")
    interactive.add_argument("--debug", action="store_true", help="Show [ctx]/[top-5]/timing stats (default: answer only)")
    interactive.add_argument("--max-prompt-tokens", type=_positive_int, default=512)
    interactive.add_argument("--max-new-tokens", type=_positive_int, default=24)
    interactive.add_argument("--max-context-tokens", type=_positive_int, default=4096)
    interactive.add_argument("--system-prompt", help="System prompt for persistent chat mode")
    interactive.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy argmax)")
    interactive.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff (1.0 = disabled)")
    interactive.add_argument("--repetition-penalty", type=float, default=1.0, help="HF-style repetition penalty (1.0 = disabled)")
    interactive.add_argument("--ignore-eos", action="store_true")
    interactive.add_argument("--mute-attn-layer", type=int, action="append", default=[])
    interactive.add_argument("--mute-mlp-layer", type=int, action="append", default=[])
    interactive.add_argument("--mute-deltanet-layer", type=int, action="append", default=[])
    interactive.add_argument("--mute-chart", action="append", default=[], help="Format LAYER:CHART (e.g. 13:2)")
    interactive.set_defaults(run=_interactive)
    chat = commands.add_parser(
        "chat",
        help="One-arg persistent chat demo: compiled artifact, chat template, and sampling defaults",
    )
    chat.add_argument("--artifact", default=_default_chat_artifact(),
                      help="Local artifact dir or HF repo id (default: local joint_waterfill or published repo)")
    chat.add_argument("--tokenizer",
                      help="Local tokenizer dir or HF repo id (default: inferred from artifact source)")
    chat.add_argument("--prompt", help="Run a single chat question and exit (default: persistent REPL)")
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--repetition-penalty", type=float, default=1.15)
    chat.add_argument("--max-new-tokens", type=_positive_int, default=96)
    chat.add_argument("--max-context-tokens", type=_positive_int, default=4096)
    chat.add_argument("--system-prompt")
    chat.add_argument("--debug", action="store_true", help="Show [ctx]/[top-5]/timing stats")
    chat.add_argument("--xray", action="store_true", help="Enable per-turn X-Ray output")
    chat.add_argument("--ignore-eos", action="store_true")
    chat.add_argument("--mute-attn-layer", type=int, action="append", default=[])
    chat.add_argument("--mute-mlp-layer", type=int, action="append", default=[])
    chat.add_argument("--mute-deltanet-layer", type=int, action="append", default=[])
    chat.add_argument("--mute-chart", action="append", default=[], help="Format LAYER:CHART (e.g. 13:2)")
    chat.set_defaults(run=_chat)
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
