"""MAC-first Qwen3 MLX build: bottleneck-SwiGLU MLPs + lexical factor.

Prioritizes decode MACs over parameter count:

- ``lexical.rank``: tied embedding/lm_head ``W (vocab, 2048)`` is factored as
  ``codebook (vocab, r) @ basis (2048, r)^T`` via SVD of ``W^T W`` — the head
  drops from `vocab*2048` to `2048*r + r*vocab` MACs/token.
- ``--mlp-layers``: bottleneck SwiGLU (intermediate ``k``) on the selected
  layers — preserves the multiplicative nonlinearity the chart generator
  cannot express, at ``3*2048*k`` MACs/layer.
- ``--attention-reductions``: optional ``layer:rank`` input-basis reductions
  (e.g. ``27:981``) re-fitted via ``fit_reduced_attention``.

All fits are sequential along the student trajectory on the chat-mixed
calibration corpus.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _fit_lexical(runtime, rank):
    """Factor the tied lexical matrix: W ≈ codebook @ basis^T (SVD)."""
    from fisher_graph.mlx_qwen3_fit import _mx
    mx = _mx()
    W = np.asarray(runtime.parameters["lexical.weight"].astype(mx.float32),
                   dtype=np.float64)
    gram = W.T @ W  # (2048, 2048)
    eigvals, eigvecs = np.linalg.eigh(gram)
    top = np.argsort(-eigvals)[:rank]
    B = eigvecs[:, top]  # (2048, r), orthonormal columns
    C = W @ B  # (vocab, r)
    updates = {
        "lexical.codebook": mx.array(C.astype(np.float32)),
        "lexical.basis": mx.array(B.astype(np.float32)),
    }
    removed = ("lexical.weight",)
    recon_err = float(np.mean((W - C @ B.T) ** 2) / np.mean(W**2))
    return updates, removed, recon_err


def _bottleneck_layer(rt, rt_exact, batches, layer, k, rel_ridge):
    """Energy-select top-k intermediate neurons; ridge-refit down_weight."""
    from fisher_graph.mlx_qwen3_fit import _mx, capture_runtime_rows
    mx = _mx()
    p = rt.parameters
    n = capture_runtime_rows(rt, batches, [f"mlp.{layer}.normalized"])[
        f"mlp.{layer}.normalized"]
    x = mx.astype(n, mx.float32)
    g = x @ p[f"mlp.{layer}.gate_weight"].astype(mx.float32).T
    u = x @ p[f"mlp.{layer}.up_weight"].astype(mx.float32).T
    act = (g * mx.sigmoid(g)) * u
    stats = mx.mean(act**2, axis=0)
    down = np.asarray(p[f"mlp.{layer}.down_weight"].astype(mx.float32),
                      dtype=np.float64)
    importance = np.asarray(stats) * (down**2).sum(axis=0)
    top = np.sort(np.argsort(-importance)[:k])
    gate = np.asarray(p[f"mlp.{layer}.gate_weight"].astype(mx.float32))[top]
    up = np.asarray(p[f"mlp.{layer}.up_weight"].astype(mx.float32))[top]
    g2 = x @ mx.array(gate).T
    u2 = x @ mx.array(up).T
    a = np.asarray((g2 * mx.sigmoid(g2)) * u2, dtype=np.float64)
    y = np.asarray(
        mx.astype(capture_runtime_rows(
            rt_exact, batches, [f"mlp.{layer}.output"])[f"mlp.{layer}.output"],
                  mx.float32), dtype=np.float64)
    gram = a.T @ a
    lam = rel_ridge * np.trace(gram) / gram.shape[0] + 1e-9
    W = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), a.T @ y)
    err = float(np.mean((y - a @ W) ** 2) / np.mean(y**2))
    return {
        f"mlp.{layer}.gate_weight": mx.array(gate).astype(mx.bfloat16),
        f"mlp.{layer}.up_weight": mx.array(up).astype(mx.bfloat16),
        f"mlp.{layer}.down_weight": mx.array(W.T.astype(np.float32)).astype(
            mx.bfloat16),
    }, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exact", default=".local-runs/qwen3-mlx-first-fit/exact")
    ap.add_argument("--corpus",
                    default=".local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mlp-layers", default="8,11,13,15,18,21,23,25")
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--attention-reductions", default="27:981",
                    help="Comma list of layer:input_rank")
    ap.add_argument("--lexical-rank", type=int, default=512)
    ap.add_argument("--rel-ridge", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_fit import (
        capture_runtime_rows, fit_reduced_attention,
    )
    from fisher_graph.mlx_qwen3_pipeline import _layers, _read_corpus, _tokenizer
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

    if args.lexical_rank:
        upd, rm, err = _fit_lexical(rt, args.lexical_rank)
        rt = rt.with_parameters(upd, remove=rm)
        print(f"[fit ] lexical rank={args.lexical_rank} rel_err={err:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    attn = {}
    for spec in (args.attention_reductions or "").split(","):
        if spec.strip():
            l, r = spec.split(":")
            attn[int(l)] = int(r)

    mlp_layers = sorted(set(_layers(args.mlp_layers, 28)))
    for l in range(28):
        if l in attn:
            caps_a = capture_runtime_rows(
                rt, batches,
                [f"attention.{l}.normalized", f"attention.{l}.context"])
            res_a = fit_reduced_attention(
                rt, l, caps_a, input_rank=attn[l], output_rank=2048,
                relative_ridge=0.15, solver="numpy_float64")
            u = dict(res_a.updates)
            if f"attention.{l}.output_encoder_weight" in u:
                del u[f"attention.{l}.output_encoder_weight"]
                u[f"attention.{l}.output_decoder_weight"] = rt_exact.parameters[
                    f"attention.{l}.output_decoder_weight"]
            rt = rt.with_parameters(u)
            print(f"[fit ] L{l:2d} attn r_in={attn[l]} ({time.time()-t0:.1f}s)",
                  flush=True)
        if l in mlp_layers:
            upd, err = _bottleneck_layer(rt, rt_exact, batches, l,
                                         args.intermediate, args.rel_ridge)
            rt = rt.with_parameters(upd)
            print(f"[fit ] L{l:2d} bottleneck k={args.intermediate} "
                  f"rel_err={err:.4f} ({time.time()-t0:.1f}s)", flush=True)

    total = sum(int(v.size) for v in rt.parameters.values())
    removed = sum(int(v.size) for v in rt_exact.parameters.values()) - total
    print(f"[done] scalars {total:,} (removed {removed:,} = "
          f"{removed*2/1024**2:.2f} MiB) in {time.time()-t0:.1f}s", flush=True)
    rt.save(args.output)
    print(f"[save] {args.output}", flush=True)


if __name__ == "__main__":
    main()
