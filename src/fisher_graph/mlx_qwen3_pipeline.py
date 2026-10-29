"""Local-only conversion, bounded calibration, and generation for MLX Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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
    chart_deltanet_layers = sorted(
        i for i in range(len(runtime.layers)) if f"deltanet.{i}.chart_centers" in runtime.parameters
    )
    chart_attn_layers = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.chart_centers" in runtime.parameters
    )
    reduced_attn_layers = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.input_basis" in runtime.parameters
    )
    value_attn_layers = sorted(
        i for i in range(len(runtime.layers))
        if f"attention.{i}.value_basis" in runtime.parameters and i not in chart_attn_layers
    )
    bottleneck_mlp_layers = sorted(
        i for i in range(len(runtime.layers))
        if f"mlp.{i}.gate_weight" in runtime.parameters
        and int(runtime.parameters[f"mlp.{i}.gate_weight"].shape[0])
        < int(runtime.layers[i].get("mlp", {}).get("full_intermediate_size", 0) or 0)
    )
    attn_delta_layers = sorted(set(reduced_attn_layers) | set(chart_attn_layers) | set(value_attn_layers))
    dnet_route_site = {
        i: (f"deltanet.{i}.context" if f"deltanet.{i}.qkv_weight" in runtime.parameters else f"deltanet.{i}.normalized")
        for i in chart_deltanet_layers
    }
    capture_sites = (
        [f"mlp.{i}.normalized" for i in chart_layers]
        + [f"mlp.{i}.delta" for i in chart_layers]
        + [dnet_route_site[i] for i in chart_deltanet_layers]
        + [f"deltanet.{i}.delta" for i in chart_deltanet_layers]
        + [f"attention.{i}.context" for i in chart_attn_layers]
        + [f"attention.{i}.delta" for i in attn_delta_layers]
        + [f"mlp.{i}.delta" for i in bottleneck_mlp_layers]
    )
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
    chart_counts = {}
    for i in chart_layers:
        n_val = out.captures[f"mlp.{i}.normalized"][0].astype(mx.float32)
        centers = runtime.parameters[f"mlp.{i}.chart_centers"].astype(mx.float32)
        chart_counts[str(i)] = int(centers.shape[0])
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

    for i in chart_deltanet_layers:
        c_val = out.captures[dnet_route_site[i]][0].astype(mx.float32)
        centers = runtime.parameters[f"deltanet.{i}.chart_centers"].astype(mx.float32)
        chart_counts[f"deltanet.{i}"] = int(centers.shape[0])
        scales = runtime.parameters[f"deltanet.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["deltanet"].get("temperature", 1.0))
        disp = c_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(
            mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1)
        )
        token_routing[f"deltanet.{i}"] = [
            {
                "position": pos,
                "token": tokenizer.decode([tok]),
                "dominant_chart": int(np.argmax(memb[pos])),
                "dominant_membership": float(np.max(memb[pos])),
            }
            for pos, tok in enumerate(ids)
        ]

    for i in chart_attn_layers:
        c_val = out.captures[f"attention.{i}.context"][0].astype(mx.float32)
        centers = runtime.parameters[f"attention.{i}.chart_centers"].astype(mx.float32)
        scales = runtime.parameters[f"attention.{i}.chart_distance_scales"].astype(mx.float32)
        chart_counts[f"attention.{i}"] = int(centers.shape[0])
        temp = float(runtime.layers[i]["attention"].get("temperature", 1.0))
        disp = c_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(
            mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1)
        )
        token_routing[f"attention.{i}"] = [
            {
                "position": pos,
                "token": tokenizer.decode([tok]),
                "dominant_chart": int(np.argmax(memb[pos])),
                "dominant_membership": float(np.max(memb[pos])),
            }
            for pos, tok in enumerate(ids)
        ]

    ablation_impacts = []
    ablation_sites = (
        [f"deltanet.{i}.delta" for i in chart_deltanet_layers]
        + [f"attention.{i}.delta" for i in attn_delta_layers]
        + [f"mlp.{i}.delta" for i in chart_layers]
        + [f"mlp.{i}.delta" for i in bottleneck_mlp_layers]
    )
    for site in ablation_sites:
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
        "chart_attention_layers": chart_attn_layers,
        "value_crushed_attention_layers": value_attn_layers,
        "bottleneck_mlp_layers": bottleneck_mlp_layers,
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
        "chart_counts": chart_counts,
        "ablation_impacts": ablation_impacts,
    }
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    A = _Ansi(_ansi_enabled(args) and not getattr(args, "json", False))
    if A.enabled:
        _print_xray_panel(payload, A)
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


_DENSITY = "░▒▓█"
_CHART_PALETTE = ("cyan", "magenta", "yellow", "green", "red", "b_cyan", "b_green")


def _trunc(text, width):
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_xray_panel(payload, A):
    """Render an xray payload as an ANSI panel: top-5 probability bars, a
    token x layer routing heatmap, and diverging ablation-push bars."""
    top5 = payload.get("top5_predictions") or []
    winner = top5[0]["token"] if top5 else ""

    if top5:
        print(_hrule(A, "top-5", style=A.magenta))
        bar_w = 26
        for rank, pred in enumerate(top5):
            p = pred["probability"]
            fill = int(round(p * bar_w))
            name = _trunc(repr(pred["token"]), 16).ljust(18)
            bar = "█" * fill + "░" * (bar_w - fill)
            styled = (A.b_green if rank == 0 else A.cyan)(bar)
            print(f"  {name}{styled} {100 * p:5.1f}%  " + A.gray(f"logit {pred['logit']:.2f}"))

    routing = payload.get("chart_routing") or {}
    counts = payload.get("chart_counts") or {}
    if routing:
        print(_hrule(A, "chart routing", style=A.magenta, tail="░▒▓█ membership lift over uniform"))
        cols = sorted(routing, key=lambda k: int(k.split(".")[-1]))
        max_cols = 12
        shown, hidden = cols[:max_cols], cols[max_cols:]
        header = "  " + " " * 18 + "".join(
            (("d" if c.startswith("deltanet.") else "a" if c.startswith("attention.") else "m")
             + c.split(".")[-1]).center(4)
            for c in shown
        )
        print(A.gray(header))
        n_rows = len(routing[shown[0]])
        cap = 40
        for pos in range(min(n_rows, cap)):
            label = _trunc(repr(routing[shown[0]][pos]["token"]), 15).ljust(16)
            cells = []
            for c in shown:
                entry = routing[c][pos]
                n_ch = max(counts.get(c, 1), 1)
                uniform = 1.0 / n_ch
                lift = (entry["dominant_membership"] - uniform) / (1.0 - uniform) if n_ch > 1 else 1.0
                glyph = _DENSITY[min(3, max(0, int(max(0.0, lift) * 4)))]
                color = getattr(A, _CHART_PALETTE[entry["dominant_chart"] % len(_CHART_PALETTE)])
                cells.append(color(f"{entry['dominant_chart']}{glyph}".center(4)))
            print("  " + A.gray(label) + "".join(cells))
        if n_rows > cap or hidden:
            note = f"  … {n_rows - cap} more tokens" if n_rows > cap else ""
            note += f"{' · ' if note and hidden else '  … '}{len(hidden)} more layers" if hidden else ""
            print(A.dim(note))

    impacts = payload.get("ablation_impacts") or []
    if impacts:
        print(_hrule(A, f"ablation push on {winner!r}", style=A.magenta))
        peak = max(abs(i["winning_logit_push"]) for i in impacts) or 1.0
        half = 14
        for i in impacts:
            push = i["winning_logit_push"]
            n = int(round(abs(push) / peak * half))
            site = _trunc(
                i["site"].replace("deltanet.", "dnet.").replace("attention.", "attn."), 15
            ).ljust(16)
            left = A.red(("█" * n).rjust(half)) if push < 0 else " " * half
            right = A.green(("█" * n).ljust(half)) if push > 0 else " " * half
            flip = ""
            if i["ablated_top1_token"] != winner:
                flip = A.yellow(" ▸ " + _trunc(repr(i["ablated_top1_token"]), 10))
            print(f"  {site}{push:+.2f}  {left}{A.gray('│')}{right}{flip}")

    if not top5 and not routing and not impacts:
        print(A.dim("  no fitted chart or reduced sites in this artifact"))


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
                         temperature=0.0, top_p=1.0, repetition_penalty=1.0, emit=None,
                         return_timing_breakdown=False):
    import mlx.core as mx
    import numpy as np

    if not getattr(runtime, "_warmed", False) and hasattr(runtime, "_fast_cache"):
        w_out = runtime.prefill(mx.array([[ids[0]]], dtype=mx.int32))
        mx.eval(w_out.logits, w_out.cache.key_values)
        del w_out
        mx.clear_cache()
        runtime._warmed = True
    if hasattr(runtime, "reset_routing_stats"):
        runtime.reset_routing_stats()

    start = time.perf_counter()
    output = runtime.prefill(
        mx.array([ids], dtype=mx.int32),
        muted_sites=muted_sites,
        muted_charts=muted_charts,
    )
    first_logits = output.logits[0, -1].astype(mx.float32)
    first_probs = mx.softmax(first_logits, axis=-1)
    mx.eval(first_logits, first_probs, output.cache.key_values)
    prefill_elapsed = time.perf_counter() - start
    logits_np = np.asarray(first_logits)
    probs_np = np.asarray(first_probs)
    top5_idx = np.argsort(logits_np)[::-1][:5]
    top5 = [
        (tokenizer.decode([int(i)]), float(probs_np[i]) * 100.0, float(logits_np[i]))
        for i in top5_idx
    ]
    decode_start = time.perf_counter()
    generated = []
    emitted = ""
    if not (temperature and temperature > 0.0):
        next_tok = mx.argmax(output.logits[:, -1:], axis=-1)
        mx.async_eval(next_tok)
        cache = output.cache
        for step in range(max_new_tokens):
            if step + 1 < max_new_tokens:
                next_out = runtime.decode(
                    next_tok,
                    cache,
                    muted_sites=muted_sites,
                    muted_charts=muted_charts,
                )
                cand_tok = mx.argmax(next_out.logits[:, -1:], axis=-1)
                mx.async_eval(cand_tok, next_out.cache.key_values)
            token = int(next_tok.item())
            generated.append(token)
            if emit is not None:
                text_now = tokenizer.decode(generated, skip_special_tokens=True)
                if len(text_now) > len(emitted):
                    emit(text_now[len(emitted):])
                emitted = text_now
            if not ignore_eos and token == tokenizer.eos_token_id:
                break
            if step + 1 < max_new_tokens:
                next_tok, cache = cand_tok, next_out.cache
    else:
        for step in range(max_new_tokens):
            logits_np = np.asarray(output.logits[0, -1].astype(mx.float32))
            token = _sample_next_token(
                logits_np, ids + generated,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            generated.append(token)
            if emit is not None:
                text_now = tokenizer.decode(generated, skip_special_tokens=True)
                if len(text_now) > len(emitted):
                    emit(text_now[len(emitted):])
                emitted = text_now
            if not ignore_eos and token == tokenizer.eos_token_id:
                break
            if step + 1 < max_new_tokens:
                output = runtime.decode(
                    mx.array([[token]], dtype=mx.int32),
                    output.cache,
                    muted_sites=muted_sites,
                    muted_charts=muted_charts,
                )
    decode_elapsed = time.perf_counter() - decode_start
    elapsed = prefill_elapsed + decode_elapsed
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if return_timing_breakdown:
        return text, generated, top5, prefill_elapsed, decode_elapsed
    return text, generated, top5, elapsed


def _mac_stats(runtime, context_len):
    """Analytic MACs per decoded token plus recurrent/KV state bytes at context_len.

    Counts matmul multiply-accumulates from parameter shapes (packed quantized
    weights are unpacked via the quantization config). DeltaNet recurrence cost
    uses the compiled carrier_rank, so Play 2 savings are reflected directly.
    """
    p, layers = runtime.parameters, runtime.layers
    quant = runtime.config.get("quantization") or {}
    bits = int(quant.get("bits", 4))

    def lin(key):
        if key not in p:
            return 0
        out_dim, cols = p[key].shape
        in_dim = cols * 32 // bits if key + ".scales" in p else cols
        return out_dim * in_dim

    def basis(key):
        return p[key].size if key in p else 0

    def out_pair(prefix):
        if prefix + "output_encoder_weight" in p:
            return lin(prefix + "output_encoder_weight") + lin(prefix + "output_decoder_weight")
        return lin(prefix + "output_decoder_weight")

    total = lin("lexical.head_weight")
    state_bytes = 0
    r_stats = getattr(runtime, "_routing_stats", {}) or {}
    for i, layer in enumerate(layers):
        m = f"mlp.{i}."
        mlp_macs = lin(m + "gate_weight") + lin(m + "up_weight") + lin(m + "down_weight")
        if "deltanet" in layer:
            d, dc = f"deltanet.{i}.", layer["deltanet"]
            super_macs = (basis(d + "transport_context_basis") + basis(d + "transport_input_basis")
                          + basis(d + "transport_decoder") + basis(d + "chart_bases")
                          + basis(d + "local_generator_encoders") + basis(d + "chart_mode_decoders"))
            if d + "qkv_weight" not in p:
                total += super_macs + mlp_macs
            else:
                kh, vh = dc["key_heads"], dc["value_heads"]
                kd, vd = dc["key_head_dimension"], dc["value_head_dimension"]
                rank = dc.get("carrier_rank", kd)
                conv_dim = 2 * kh * kd + vh * vd
                full_dnet_macs = (
                    lin(d + "qkv_weight") + lin(d + "z_weight") + lin(d + "a_weight")
                    + lin(d + "b_weight") + out_pair(d) + basis(d + "input_basis")
                    + basis(d + "value_basis") + conv_dim * dc["conv_kernel_size"] + 5 * vh * vd * rank
                )
                if d + "energy_threshold" not in p and d + "chart_centers" in p:
                    full_dnet_macs += (basis(d + "chart_bases") + basis(d + "local_generator_encoders")
                                       + basis(d + "chart_mode_decoders"))
                counts = r_stats.get(i)
                if d + "energy_threshold" in p and counts and (counts[0] + counts[1]) > 0:
                    fast_frac = counts[0] / (counts[0] + counts[1])
                    gate_probe_macs = basis(d + "chart_centers")
                    fast_path_macs = gate_probe_macs + super_macs + (0 if dc.get("coalesced_mlp", False) else mlp_macs)
                    full_path_macs = gate_probe_macs + full_dnet_macs + mlp_macs
                    total += int(round(fast_frac * fast_path_macs + (1.0 - fast_frac) * full_path_macs))
                else:
                    total += full_dnet_macs + mlp_macs
                state_bytes += ((dc["conv_kernel_size"] - 1) * conv_dim + vh * vd * rank) * 4
        else:
            total += mlp_macs
            a, ac = f"attention.{i}.", layer["attention"]
            kh, d_head = ac["key_value_heads"], ac["head_dimension"]
            total += (lin(a + "q_weight") + lin(a + "k_weight") + lin(a + "v_weight")
                      + out_pair(a) + basis(a + "input_basis") + basis(a + "value_basis")
                      + basis(a + "chart_bases") + basis(a + "local_generator_encoders")
                      + basis(a + "chart_mode_decoders"))
            total += 3 * ac["query_heads"] * d_head * context_len
            state_bytes += 2 * kh * d_head * context_len * 2
    return total, state_bytes


def _default_chat_artifact():
    for candidate in (
        ".local-runs/qwen3_5-9b-carrier-compile/energy_gated_9b",
        ".local-runs/qwen3_5-9b-carrier-compile/waterfill_carrier_50pct",
        ".local-runs/qwen3-ewc-refine/joint_waterfill_286mib",
    ):
        local = Path(candidate)
        if local.is_dir():
            return str(local)
    return "jediwattzon/qwen3.5-9b-fisher-chart-graph"


def _default_tokenizer(runtime):
    if runtime.config.get("source") == "hf_qwen3_5":
        return "mlx-community/Qwen3.5-9B-MLX-4bit"
    return "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Terminal styling and streaming markdown rendering (zero-dependency ANSI)
# ---------------------------------------------------------------------------

_RULE_W = 74


class _Ansi:
    """Minimal ANSI styler — every method is a no-op when disabled."""

    def __init__(self, enabled):
        self.enabled = bool(enabled)

    def _w(self, code, text):
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, text): return self._w("1", text)
    def dim(self, text): return self._w("2", text)
    def italic(self, text): return self._w("3", text)
    def underline(self, text): return self._w("4", text)
    def strike(self, text): return self._w("9", text)
    def red(self, text): return self._w("31", text)
    def green(self, text): return self._w("32", text)
    def yellow(self, text): return self._w("33", text)
    def magenta(self, text): return self._w("35", text)
    def cyan(self, text): return self._w("36", text)
    def gray(self, text): return self._w("90", text)
    def b_cyan(self, text): return self._w("1;36", text)
    def b_green(self, text): return self._w("1;32", text)


def _ansi_enabled(args):
    if getattr(args, "no_color", False) or os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _hrule(A, label="", *, style=None, tail=""):
    """`─ label ──────────────── tail ` spanning _RULE_W columns."""
    left = f" {label} " if label else ""
    right = f"─ {tail} " if tail else ""
    fill = "─" * max(0, _RULE_W - 1 - len(left) - len(right))
    parts = [A.gray("─")]
    if label:
        parts.append((style or A.bold)(left))
    parts.append(A.gray(fill))
    if tail:
        parts.append(A.gray(right))
    return "".join(parts)


_MD_FENCE = re.compile(r"^\s*```(.*)$")
_MD_HEAD = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_MD_HR = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_MD_ULIST = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_MD_OLIST = re.compile(r"^(\s*)(\d{1,3})([.)])\s+(.*)$")
_MD_INLINE = re.compile(
    r"`(?P<code>[^`\n]+)`"
    r"|~~(?P<strike>\S[^~\n]*?)~~"
    r"|\*\*(?P<bold>\S(?:[^*\n]*?\S)?)\*\*"
    r"|__(?P<boldu>\S(?:[^_\n]*?\S)?)__"
    r"|\[(?P<ltext>[^\]\n]+)\]\((?P<lurl>[^)\s\n]+)[^)\n]*\)"
    r"|\*(?P<ital>\S(?:[^*\n]*?\S)?)\*"
    r"|(?<!\w)_(?P<italu>\S(?:[^_\n]*?\S)?)_(?!\w)"
)


def _inline_md(A, line):
    def repl(m):
        if m.group("code") is not None:
            return A.yellow(m.group("code"))
        if m.group("strike") is not None:
            return A.strike(m.group("strike"))
        inner = m.group("bold") or m.group("boldu")
        if inner is not None:
            return A.bold(inner)
        if m.group("ltext") is not None:
            return A.underline(m.group("ltext")) + A.gray(f" ({m.group('lurl')})")
        inner = m.group("ital") or m.group("italu")
        if inner is not None:
            return A.italic(inner)
        return m.group(0)
    return _MD_INLINE.sub(repl, line) if A.enabled else line


def _render_markdown(text, A):
    """Best-effort markdown rendering for assistant output (fenced code,
    headers, lists, quotes, inline spans). Returns text unchanged when ANSI
    is disabled."""
    if not A.enabled:
        return text
    out = []
    in_code = False
    for raw in text.split("\n"):
        fence = _MD_FENCE.match(raw)
        if fence:
            in_code = not in_code
            if in_code:
                lang = fence.group(1).strip()
                label = f" {lang} " if lang else "──"
                out.append(A.gray("  ╭─" + label + "─" * max(0, _RULE_W - 4 - len(label))))
            else:
                out.append(A.gray("  ╰" + "─" * (_RULE_W - 3)))
            continue
        if in_code:
            out.append(A.gray("  │ ") + raw)
            continue
        head = _MD_HEAD.match(raw)
        if head:
            out.append("  " + A.b_cyan(head.group(1).strip()))
            continue
        if _MD_HR.match(raw):
            out.append(A.gray("  " + "─" * (_RULE_W - 2)))
            continue
        quote = _MD_QUOTE.match(raw)
        if quote:
            out.append(A.gray("  ▎ ") + _inline_md(A, quote.group(1)))
            continue
        ul = _MD_ULIST.match(raw)
        if ul:
            out.append("  " + ul.group(1) + A.cyan("•") + " " + _inline_md(A, ul.group(3)))
            continue
        ol = _MD_OLIST.match(raw)
        if ol:
            out.append("  " + ol.group(1) + A.cyan(ol.group(2) + ol.group(3)) + " " + _inline_md(A, ol.group(4)))
            continue
        out.append("  " + _inline_md(A, raw))
    return "\n".join(out)


class _TermRepaint:
    """emit callback: streams raw token text, then repaints that region as
    rendered markdown once generation finishes."""

    def __init__(self, A):
        self.A = A
        self._col = 0
        self._rows = 0

    def feed(self, piece):
        if self.A.enabled:
            w = max(20, shutil.get_terminal_size((80, 24)).columns)
            for ch in piece:
                if ch == "\n":
                    self._rows += max(1, -(-self._col // w))
                    self._col = 0
                else:
                    self._col += 1
        sys.stdout.write(piece)
        sys.stdout.flush()

    def repaint(self, rendered):
        if not self.A.enabled:
            return
        w = max(20, shutil.get_terminal_size((80, 24)).columns)
        total = self._rows + max(1, -(-self._col // w))
        seq = f"\r\x1b[{total - 1}A\x1b[J" if total > 1 else "\r\x1b[2K"
        sys.stdout.write(seq + rendered + "\n")
        sys.stdout.flush()


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
    stream_on = bool(getattr(args, "stream", False))
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
        if f"deltanet.{i}.output_encoder_weight" in runtime.parameters
        or f"deltanet.{i}.input_basis" in runtime.parameters
        or f"deltanet.{i}.value_basis" in runtime.parameters
    )
    chart_deltanets = sorted(
        i for i in range(len(runtime.layers)) if f"deltanet.{i}.chart_centers" in runtime.parameters
    )
    bypassed_deltanets = sorted(
        i for i in range(len(runtime.layers))
        if "deltanet" in runtime.layers[i] and f"deltanet.{i}.qkv_weight" not in runtime.parameters
    )
    coalesced_full_layers = sorted(
        i for i in bypassed_deltanets if runtime.layers[i]["deltanet"].get("coalesced_mlp", False)
    )
    deltanet_ranks = {
        i: runtime.layers[i]["deltanet"].get("carrier_rank", runtime.layers[i]["deltanet"]["key_head_dimension"])
        for i in range(len(runtime.layers))
        if "deltanet" in runtime.layers[i] and f"deltanet.{i}.qkv_weight" in runtime.parameters
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

    A = _Ansi(_ansi_enabled(args))
    gated_deltanets = sorted(
        i for i in range(len(runtime.layers))
        if f"deltanet.{i}.energy_threshold" in runtime.parameters
    )
    value_crushed = sorted(
        i for i in range(len(runtime.layers))
        if f"deltanet.{i}.value_basis" in runtime.parameters
    )

    def _stat(label, body):
        print("   " + A.gray(label.ljust(6)) + " " + body)

    def _harness(msg):
        print(A.green("[harness]") + " " + msg)

    def _mutes_str():
        bits = []
        if muted_sites:
            bits.append(f"sites={sorted(muted_sites)}")
        if muted_charts:
            bits.append(f"charts={sorted(muted_charts)}")
        return " ".join(bits)

    def _macs_str(ctx_tokens):
        mac_tok, state_b = _mac_stats(runtime, ctx_tokens)
        parts = [f"~{mac_tok / 1e9:.2f} G-MAC/tok"]
        r_stats = getattr(runtime, "_routing_stats", {}) or {}
        if r_stats:
            f_tok = sum(c[0] for c in r_stats.values())
            tot_tok = sum(c[0] + c[1] for c in r_stats.values())
            parts.append(f"chart-skip {100 * f_tok / max(tot_tok, 1):.0f}% ({len(r_stats)} gated)")
        parts.append(f"cache+state {state_b / 2**20:.1f} MiB @ {ctx_tokens} tok")
        return " · ".join(parts)

    def _finish_turn(label, *, plain_tag=None, text, gen_ids, top5, prefill_s, decode_s,
                     n_prompt, ctx_tok, stream, echo=None, ctx=None):
        tag = plain_tag or label
        elapsed = prefill_s + decode_s
        prompt_tps = n_prompt / max(prefill_s, 1e-6)
        decode_tps = max(len(gen_ids) - 1, 1) / max(decode_s, 1e-6)
        mode = "greedy" if not temperature else f"T={temperature}/top-p={top_p}/pen={rep_penalty}"
        top_str = " · ".join(f"{t!r}:{p:.1f}%" for t, p, _ in top5[:5])
        if not A.enabled:
            if debug_on:
                if echo is not None:
                    print(f"\n[{echo[0]}]    {echo[1]}")
                if ctx is not None:
                    print(f"[ctx]      {ctx}")
                if muted_sites or muted_charts:
                    print(f"[mutes]    {_mutes_str()}")
                if stream_on:
                    print(f"[{tag}]    ({len(gen_ids)} tok in {elapsed:.2f}s | prompt {prompt_tps:.1f} tok/s | decode {decode_tps:.1f} tok/s) [{mode}]")
                else:
                    print(f"[{tag}]    {text!r}  ({len(gen_ids)} tok in {elapsed:.2f}s | prompt {prompt_tps:.1f} tok/s | decode {decode_tps:.1f} tok/s) [{mode}]")
                print(f"[top-5]    {top_str}")
                print(f"[macs]     {_macs_str(ctx_tok)}")
            elif not stream_on:
                print(f"\n{text}")
            return
        if stream is not None:
            stream.repaint(_render_markdown(text, A))
        else:
            print(_render_markdown(text, A))
        footer = (f"{len(gen_ids)} tok · {elapsed:.2f}s · prompt {prompt_tps:.1f} tok/s "
                  f"· decode {decode_tps:.1f} tok/s")
        if mode != "greedy":
            footer += f" · {mode}"
        print(_hrule(A, footer, style=A.dim))
        if debug_on:
            if ctx is not None:
                _stat("ctx", ctx)
            if muted_sites or muted_charts:
                _stat("mutes", _mutes_str())
            _stat("top-5", top_str)
            _stat("macs", _macs_str(ctx_tok))

    def run_one(raw_text: str, *, show_prompt=False):
        ids = encode_prompt(raw_text)
        stream = _TermRepaint(A) if stream_on else None
        if A.enabled:
            print()
            if show_prompt:
                print(_hrule(A, "prompt", style=A.b_green))
                print(f"  {raw_text}")
            print(_hrule(A, "model", style=A.b_cyan))
        elif stream_on:
            print()
        text, gen_ids, top5, prefill_s, decode_s = _greedy_continuation(
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
            emit=stream.feed if stream is not None else None,
            return_timing_breakdown=True,
        )
        _finish_turn(
            "model", plain_tag="compiled", text=text, gen_ids=gen_ids, top5=top5,
            prefill_s=prefill_s, decode_s=decode_s, n_prompt=len(ids),
            ctx_tok=len(ids) + len(gen_ids), stream=stream, echo=("prompt", raw_text),
        )
        if reference is not None:
            ref_text, ref_ids, ref_top5, ref_elapsed = _greedy_continuation(
                reference,
                tokenizer,
                ids,
                max_new_tokens=max_new_tokens,
                ignore_eos=args.ignore_eos,
            )
            if debug_on:
                ref_top_str = " · ".join(f"{t!r}:{p:.1f}%" for t, p, _ in ref_top5[:5])
                if A.enabled:
                    print(_hrule(A, "exact", style=A.magenta))
                    print(_render_markdown(ref_text, A))
                    print(_hrule(A, f"{len(ref_ids)} tok · {ref_elapsed:.2f}s", style=A.dim))
                    _stat("top-5", ref_top_str)
                else:
                    print(f"[exact]    {ref_text!r}  ({len(ref_ids)} tok in {ref_elapsed:.2f}s)")
                    print(f"[ex-top-5] {ref_top_str}")
        if xray_on:
            xray_ns = argparse.Namespace(
                artifact=str(artifact_dir),
                tokenizer=args.tokenizer or _default_tokenizer(runtime),
                prompt=raw_text,
                chat=chat_on,
                max_prompt_tokens=args.max_prompt_tokens,
                report=None,
                json=False,
                no_color=getattr(args, "no_color", False),
            )
            if A.enabled:
                print(_hrule(A, "x-ray", style=A.magenta))
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
        stream = _TermRepaint(A) if stream_on else None
        if A.enabled:
            print()
            print(_hrule(A, "assistant", style=A.b_cyan))
        elif stream_on:
            print()
        text_resp, gen_ids, top5, prefill_s, decode_s = _greedy_continuation(
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
            emit=stream.feed if stream is not None else None,
            return_timing_breakdown=True,
        )
        chat_history.append({"role": "assistant", "content": text_resp})
        chat_ctx_tokens = len(ids) + len(gen_ids)
        _finish_turn(
            "assistant", plain_tag="model", text=text_resp, gen_ids=gen_ids, top5=top5,
            prefill_s=prefill_s, decode_s=decode_s, n_prompt=len(ids),
            ctx_tok=chat_ctx_tokens, stream=stream, echo=("user", user_text),
            ctx=f"{chat_ctx_tokens} tokens | {len(chat_history)//2} turns | prompt={len(ids)} tok",
        )
        if xray_on:
            xray_ns = argparse.Namespace(
                artifact=str(artifact_dir),
                tokenizer=args.tokenizer or _default_tokenizer(runtime),
                prompt=rendered,
                chat=False,
                max_prompt_tokens=args.max_context_tokens,
                report=None,
                json=False,
                no_color=getattr(args, "no_color", False),
            )
            if A.enabled:
                print(_hrule(A, "x-ray", style=A.magenta))
            _xray(xray_ns)

    if args.presets:
        for p in _PRESET_PROMPTS:
            run_one(p, show_prompt=True)
        return
    if args.prompt is not None:
        run_one(args.prompt, show_prompt=True)
        return

    def _kv(label, value):
        print("  " + A.cyan(label.ljust(9)) + A.gray("│ ") + str(value))

    def _flag(on):
        return A.green("on") if on else A.gray("off")

    print()
    print("  " + A.bold("fisher-graph · qwen3-mlx") + A.gray("  interactive harness"))
    print(A.gray("  " + "─" * (_RULE_W - 2)))
    _kv("artifact", artifact_dir)
    value_crushed_attn = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.value_basis" in runtime.parameters
    )
    chart_attns = sorted(
        i for i in range(len(runtime.layers)) if f"attention.{i}.chart_centers" in runtime.parameters
    )
    bottleneck_mlps = {
        i: int(runtime.parameters[f"mlp.{i}.gate_weight"].shape[0])
        for i in range(len(runtime.layers))
        if f"mlp.{i}.gate_weight" in runtime.parameters
        and int(runtime.parameters[f"mlp.{i}.gate_weight"].shape[0]) < int(runtime.layers[i]["mlp"].get("full_intermediate_size", 12288 if runtime.config.get("source") == "hf_qwen3_5" else 0))
    }
    if reduced_attn or value_crushed_attn or chart_attns:
        a_bits = []
        if reduced_attn:
            a_bits.append(f"input-basis {reduced_attn}")
        if value_crushed_attn:
            a_bits.append(f"value-crushed {value_crushed_attn}")
        if chart_attns:
            a_bits.append(f"charts {chart_attns}")
        _kv("attn", " · ".join(a_bits))
    if chart_mlps or bottleneck_mlps:
        m_bits = []
        if chart_mlps:
            m_bits.append(f"charts {chart_mlps}")
        if bottleneck_mlps:
            m_bits.append(f"bottleneck-swiglu {bottleneck_mlps}")
        _kv("mlp", " · ".join(m_bits))
    if deltanet_ranks or bypassed_deltanets or chart_deltanets:
        hist = {}
        for r in deltanet_ranks.values():
            hist[r] = hist.get(r, 0) + 1
        bits = []
        if hist:
            bits.append("carriers " + " ".join(f"r{r}×{n}" for r, n in sorted(hist.items(), reverse=True)))
        if value_crushed:
            bits.append(f"value-crushed {value_crushed}")
        if bypassed_deltanets:
            bits.append(f"bypassed {bypassed_deltanets}")
        if coalesced_full_layers:
            bits.append(f"coalesced-mlp {coalesced_full_layers}")
        _kv("deltanet", " · ".join(bits))
        if chart_deltanets:
            gated = f" · energy-gated {gated_deltanets}" if gated_deltanets else ""
            _kv("charts", f"deltanet {chart_deltanets}{gated}")
    _kv("mode", f"chat {_flag(chat_on)} · stream {_flag(stream_on)} · xray {_flag(xray_on)} · debug {_flag(debug_on)}")
    samp = "greedy" if not temperature else f"T={temperature} top-p={top_p} pen={rep_penalty}"
    _kv("sampling", f"{samp} · max-new {max_new_tokens}")
    if muted_sites or muted_charts:
        _kv("mutes", _mutes_str())
    _kv("commands", "/xray /chat /stream /debug /context /reset /tokens N /temp F /topp F")
    _kv("", "/penalty F /macs /preset /unmute /mute attn|mlp|deltanet L /mute chart L C /quit")
    print(A.gray("  " + "─" * (_RULE_W - 2)))

    while True:
        try:
            prompt_label = A.b_green("chat › ") if chat_on else A.b_cyan("text › ")
            line = input(f"\n{prompt_label}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "quit", "exit"):
            break
        if line == "/xray":
            xray_on = not xray_on
            _harness(f"X-Ray mode is now {'ON' if xray_on else 'OFF'}")
            continue
        if line == "/stream":
            stream_on = not stream_on
            _harness(f"Streaming is now {'ON' if stream_on else 'OFF'}")
            continue
        if line == "/macs":
            ctx_tok = max(chat_ctx_tokens, 1)
            mac_tok, _ = _mac_stats(runtime, ctx_tok)
            _harness(_macs_str(ctx_tok))
            if reference is not None:
                ref_tok, _ = _mac_stats(reference, ctx_tok)
                saved = (ref_tok - mac_tok) / max(ref_tok, 1) * 100
                _harness(f"reference ~{ref_tok/1e9:.2f} G-MAC/tok · compiled saves {saved:.1f}%")
            continue
        if line == "/debug":
            debug_on = not debug_on
            _harness(f"Debug stats are now {'ON' if debug_on else 'OFF'}")
            continue
        if line == "/chat":
            chat_on = not chat_on
            if chat_on:
                chat_history.clear()
                chat_ctx_tokens = 0
                _harness("Chat mode ON (persistent conversation with this model). /reset clears history.")
            else:
                _harness("Chat template is now OFF (raw completion mode)")
            continue
        if line == "/context":
            _harness(f"chat_ctx_tokens={chat_ctx_tokens} | turns={len(chat_history)//2} | "
                     f"mutes={sorted(muted_sites) or 'none'} charts={sorted(muted_charts) or 'none'}")
            for msg in chat_history[-6:]:
                body = msg["content"] if len(msg["content"]) <= 72 else msg["content"][:69] + "..."
                print("           " + A.gray(f"[{msg['role']:>9}]") + " " + body)
            continue
        if line == "/reset":
            chat_history.clear()
            chat_ctx_tokens = 0
            _harness("Cleared conversation history")
            continue
        if line == "/unmute":
            muted_sites.clear()
            muted_charts.clear()
            _harness("Cleared all active layer and chart mutes")
            continue
        if line == "/preset":
            for p in _PRESET_PROMPTS:
                run_one(p, show_prompt=True)
            continue
        if line.startswith("/tokens "):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                max_new_tokens = int(parts[1])
                _harness(f"max_new_tokens = {max_new_tokens}")
            else:
                _harness("usage: /tokens <positive_int>")
            continue
        if line.startswith("/temp "):
            try:
                temperature = float(line.split()[1])
                assert temperature >= 0.0
                _harness(f"temperature = {temperature} ({'sampling' if temperature > 0 else 'greedy'})")
            except (ValueError, IndexError, AssertionError):
                _harness("usage: /temp <float>=0 for greedy, e.g. /temp 0.7")
            continue
        if line.startswith("/topp "):
            try:
                top_p = float(line.split()[1])
                assert 0.0 < top_p <= 1.0
                _harness(f"top_p = {top_p}")
            except (ValueError, IndexError, AssertionError):
                _harness("usage: /topp <float in (0,1]>")
            continue
        if line.startswith("/penalty "):
            try:
                rep_penalty = float(line.split()[1])
                assert rep_penalty >= 1.0
                _harness(f"repetition_penalty = {rep_penalty}")
            except (ValueError, IndexError, AssertionError):
                _harness("usage: /penalty <float>=1.0, e.g. /penalty 1.15")
            continue
        if line.startswith("/mute "):
            parts = line.split()
            if len(parts) == 3 and parts[1] == "attn" and parts[2].isdigit():
                muted_sites.add(f"attention.{int(parts[2])}.delta")
                _harness(f"Muted attention.{int(parts[2])}.delta")
            elif len(parts) == 3 and parts[1] == "mlp" and parts[2].isdigit():
                muted_sites.add(f"mlp.{int(parts[2])}.delta")
                _harness(f"Muted mlp.{int(parts[2])}.delta")
            elif len(parts) == 3 and parts[1] == "deltanet" and parts[2].isdigit():
                muted_sites.add(f"deltanet.{int(parts[2])}.delta")
                _harness(f"Muted deltanet.{int(parts[2])}.delta")
            elif len(parts) == 4 and parts[1] == "chart" and parts[2].isdigit() and parts[3].isdigit():
                muted_charts.add((int(parts[2]), int(parts[3])))
                _harness(f"Muted chart {int(parts[2])}:{int(parts[3])}")
            else:
                _harness("usage: /mute attn <L> | /mute mlp <L> | /mute deltanet <L> | /mute chart <L> <C>")
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
    xray.add_argument("--json", action="store_true", help="Emit the raw JSON report even in a TTY")
    xray.add_argument("--no-color", action="store_true", help="Disable ANSI styling")
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
    interactive.add_argument("--stream", action="store_true", help="Print generated tokens as they decode")
    interactive.add_argument("--no-color", action="store_true", help="Disable ANSI styling and markdown rendering")
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
    chat.add_argument("--stream", action="store_true", help="Print generated tokens as they decode")
    chat.add_argument("--no-color", action="store_true", help="Disable ANSI styling and markdown rendering")
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
