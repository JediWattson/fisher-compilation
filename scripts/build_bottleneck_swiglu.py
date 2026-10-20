"""Bottleneck-SwiGLU surrogate build for Qwen3 MLX artifacts.

Replaces each selected MLP's ``silu(gate)*up -> down`` (intermediate 6144)
with a reduced intermediate of ``k`` neurons — preserving the multiplicative
nonlinearity the additive chart generator cannot express. Neurons are
selected by activation-energy importance (second moment of
``silu(gate_i)*up_i`` x ||down column i||) on the student trajectory, then
``down_weight`` is ridge-refit to the teacher MLP output so the truncation
error is corrected.

Parameter cost per layer: ``3*2048*k`` vs ``3*2048*6144`` — at k=512,
3.15M vs 37.75M (-91.7%), cheaper than the 4-chart affine+modal (4.5M).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _act_stats(runtime, batches, layer):
    """Second moment of each intermediate activation on the student trajectory."""
    from fisher_graph.mlx_qwen3_fit import _mx, capture_runtime_rows
    mx = _mx()
    p = runtime.parameters
    n = capture_runtime_rows(runtime, batches, [f"mlp.{layer}.normalized"])[
        f"mlp.{layer}.normalized"]
    x = mx.astype(n, mx.float32)
    g = x @ p[f"mlp.{layer}.gate_weight"].astype(mx.float32).T
    u = x @ p[f"mlp.{layer}.up_weight"].astype(mx.float32).T
    act = (g * mx.sigmoid(g)) * u
    return mx.mean(act**2, axis=0), mx.astype(
        p[f"mlp.{layer}.down_weight"], mx.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact", default=".local-runs/qwen3-mlx-first-fit/exact")
    ap.add_argument("--corpus", default=".local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--rel-ridge", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_fit import capture_runtime_rows
    from fisher_graph.mlx_qwen3_pipeline import (
        _layers, _read_corpus, _tokenizer,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_joint_waterfill_qwen3 import _chatmix_batches

    t0 = time.time()
    rt = MLXQwen3.load(args.exact)
    rt_exact = rt
    tokenizer = _tokenizer(args.tokenizer)
    fit_batches = _read_corpus(args.corpus, max_examples=32, max_length=128,
                               max_tokens=4096)
    batches = _chatmix_batches(
        fit_batches, tokenizer,
        "Answer the multiple-choice question. Reply with exactly one option letter and no other text.",
    )
    k = args.intermediate
    layers = (list(range(28)) if args.layers == "all"
              else sorted(set(_layers(args.layers, 28))))
    print(f"[plan] bottleneck-swiglu k={k} on layers {layers}", flush=True)

    teacher_sites = [f"mlp.{l}.output" for l in layers]
    teacher = capture_runtime_rows(rt_exact, batches, teacher_sites)

    for l in layers:
        p = rt.parameters
        stats, down = _act_stats(rt, batches, l)
        importance = np.asarray(stats) * np.asarray(
            (down**2).sum(axis=0))
        top = np.argsort(-importance)[:k]
        top_sorted = np.sort(top)
        gate = np.asarray(p[f"mlp.{l}.gate_weight"].astype(mx.float32))[
            top_sorted]
        up = np.asarray(p[f"mlp.{l}.up_weight"].astype(mx.float32))[top_sorted]
        # refit down: act @ W = teacher_output  (on the truncated activations)
        n = capture_runtime_rows(rt, batches, [f"mlp.{l}.normalized"])[
            f"mlp.{l}.normalized"]
        x = mx.astype(n, mx.float32)
        g = x @ mx.array(gate).T
        u = x @ mx.array(up).T
        act = np.asarray((g * mx.sigmoid(g)) * u, dtype=np.float64)
        y = np.asarray(mx.astype(teacher[f"mlp.{l}.output"], mx.float32),
                       dtype=np.float64)
        gram = act.T @ act
        lam = args.rel_ridge * np.trace(gram) / gram.shape[0] + 1e-9
        W = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), act.T @ y)
        y_pred = act @ W
        err = float(np.mean((y - y_pred) ** 2) / np.mean(y**2))
        updates = {
            f"mlp.{l}.gate_weight": mx.array(gate).astype(mx.bfloat16),
            f"mlp.{l}.up_weight": mx.array(up).astype(mx.bfloat16),
            f"mlp.{l}.down_weight": mx.array(W.T.astype(np.float32)).astype(
                mx.bfloat16),
        }
        rt = rt.with_parameters(updates)
        print(f"[fit ] L{l:2d} bottleneck k={k} rel_err={err:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    total = sum(int(v.size) for v in rt.parameters.values())
    removed = sum(int(v.size) for v in rt_exact.parameters.values()) - total
    print(f"[done] scalars {total:,} (removed {removed:,} = "
          f"{removed*2/1024**2:.2f} MiB) in {time.time()-t0:.1f}s", flush=True)
    rt.save(args.output)
    print(f"[save] {args.output}", flush=True)


if __name__ == "__main__":
    main()
