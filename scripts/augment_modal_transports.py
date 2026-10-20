"""Fit cross-block transports + terminal refit onto a charted-MLP MLXQwen3 artifact.

Gemma-parity mechanisms for the modal MLP stack:

- Transports: edges ``attention.l -> mlp.l`` (one per chart) that feed the
  attention block's output coordinates directly into the chart generator
  packets. Weights are ridge-fitted so that
  ``sum_c membership_c * (x_a @ T_c) @ mode_dec_c`` closes the local gap
  ``teacher_mlp_output - student_chart_output`` on the student trajectory —
  i.e. trajectory-target fitting toward the teacher's residual correction
  rather than the chart's own output.

- Terminal refit: the last MLP is re-solved so the final block output closes
  ``teacher_terminal - state_before_last_mlp``. Charted MLPs refit their
  affine path (modal terms kept); exact SwiGLU MLPs refit ``down_weight``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def _chart_memberships(runtime, layer, normalized):
    from fisher_graph.mlx_qwen3_fit import _mx
    mx = _mx()
    p = runtime.parameters
    mcfg = runtime.layers[layer]["mlp"]
    centers = p[f"mlp.{layer}.chart_centers"]
    scales = p[f"mlp.{layer}.chart_distance_scales"]
    d = normalized[..., None, :] - centers
    logits = -0.5 * mx.sum(d**2, axis=-1) / (
        scales**2 * mcfg.get("temperature", 1.0)
    )
    return mx.softmax(logits, axis=-1)


def _ridge(Z, R, rel_ridge=0.05):
    gram = Z.T @ Z
    lam = rel_ridge * np.trace(gram) / gram.shape[0] + 1e-9
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), Z.T @ R)


def fit_transports(rt_student, rt_exact, batches, layers, *, rel_ridge=0.05,
                   sequential=False):
    """Fit attention.l -> mlp.l chart edges against the per-layer local gap.

    With ``sequential=True`` each edge is fitted on the trajectory *with all
    previously fitted transports active*, so corrections only target the
    residual that survives upstream transports. Fitting all edges against the
    pre-transport trajectory double-counts accumulated drift and overshoots
    when every edge fires simultaneously.
    """
    from fisher_graph.mlx_qwen3_fit import _mx, capture_runtime_rows
    from fisher_graph.mlx_qwen3 import MLXQwen3

    mx = _mx()
    teacher_sites = [f"mlp.{l}.output" for l in layers]
    print("[cap ] teacher mlp outputs ...", flush=True)
    teacher_caps = capture_runtime_rows(rt_exact, batches, teacher_sites)
    if not sequential:
        student_sites = []
        for l in layers:
            student_sites += [
                f"attention.{l}.context",
                f"mlp.{l}.normalized",
                f"mlp.{l}.output",
            ]
        print("[cap ] student sites ...", flush=True)
        caps = capture_runtime_rows(rt_student, batches, student_sites)

    updates, edges = {}, []
    all_edges = list(rt_student.config.get("transports", []))
    for l in layers:
        p = rt_student.parameters
        if sequential:
            caps = capture_runtime_rows(
                rt_student, batches,
                [f"attention.{l}.context", f"mlp.{l}.normalized", f"mlp.{l}.output"])
        x_a = np.asarray(_rows(mx.astype(caps[f"attention.{l}.context"], mx.float32)),
                         dtype=np.float64)
        n = mx.astype(caps[f"mlp.{l}.normalized"], mx.float32)
        resid = (
            np.asarray(_rows(mx.astype(teacher_caps[f"mlp.{l}.output"], mx.float32)),
                       dtype=np.float64)
            - np.asarray(_rows(mx.astype(caps[f"mlp.{l}.output"], mx.float32)),
                         dtype=np.float64)
        )
        mem = _rows(_chart_memberships(rt_student, l, n))
        mem = np.asarray(mem.astype(mx.float32), dtype=np.float64)
        C, d_model = mem.shape[-1], x_a.shape[-1]
        Z = (mem[..., None] * x_a[:, None, :]).reshape(mem.shape[0], C * d_model)
        base_err = float(np.mean(resid**2))
        W = _ridge(Z, resid, rel_ridge)
        pred_err = float(np.mean((resid - Z @ W) ** 2))
        decoders = np.asarray(mx.astype(p[f"mlp.{l}.chart_mode_decoders"], mx.float32),
                              dtype=np.float64)  # (C, G, D)
        g = decoders.shape[1]
        for c in range(C):
            E_c = W[c * d_model:(c + 1) * d_model]  # (d_model, D)
            dec = decoders[c]  # (G, D)
            pinv = dec.T @ np.linalg.inv(dec @ dec.T + 1e-4 * np.eye(g))
            T_c = E_c @ pinv  # (d_model, G)
            key = f"transport.{l}.{c}.weight"
            updates[key] = mx.array(
                T_c.astype(np.float32)).astype(p[f"mlp.{l}.chart_mode_decoders"].dtype)
            edges.append({
                "source": f"attention.{l}",
                "target": f"mlp.{l}",
                "source_chart": 0,
                "target_chart": c,
                "weight": key,
            })
        print(f"[fit ] transport attn.{l}->mlp.{l}: local residual "
              f"{base_err:.4f} -> {pred_err:.4f} ({100*pred_err/base_err:.1f}%)",
              flush=True)
        if sequential:
            layer_edges = [e for e in edges if e["target"] == f"mlp.{l}"]
            all_edges += layer_edges
            rt_student = MLXQwen3(
                dict(rt_student.parameters,
                     **{e["weight"]: updates[e["weight"]] for e in layer_edges}),
                dict(rt_student.config, transports=all_edges))
    return updates, edges


def refit_terminal(runtime, rt_exact, batches, layer):
    """Re-solve the last MLP so its output closes ``teacher_terminal - pre_state``."""
    from fisher_graph.mlx_qwen3_fit import _mx, capture_runtime_rows

    mx = _mx()
    p = runtime.parameters
    sites = [f"mlp.{layer}.input", f"mlp.{layer}.normalized", f"mlp.{layer}.output"]
    caps = capture_runtime_rows(runtime, batches, sites)
    teacher = capture_runtime_rows(rt_exact, batches, ["terminal"])["terminal"]
    pre = np.asarray(_rows(mx.astype(caps[f"mlp.{layer}.input"], mx.float32)),
                     dtype=np.float64)
    target = np.asarray(_rows(mx.astype(teacher, mx.float32)),
                        dtype=np.float64) - pre
    n = np.asarray(_rows(mx.astype(caps[f"mlp.{layer}.normalized"], mx.float32)),
                   dtype=np.float64)
    kind = "chart" if f"mlp.{layer}.chart_centers" in p else "swiglu"
    if kind == "chart":
        out = np.asarray(_rows(mx.astype(caps[f"mlp.{layer}.output"], mx.float32)),
                         dtype=np.float64)
        W0 = np.asarray(mx.astype(p[f"mlp.{layer}.affine_weight"], mx.float32),
                        dtype=np.float64)
        b0 = np.asarray(mx.astype(p[f"mlp.{layer}.affine_bias"], mx.float32),
                        dtype=np.float64)
        modal_part = out - (n @ W0.T + b0)
        y = target - modal_part
        Xb = np.concatenate([n, np.ones((len(n), 1))], axis=1)
        sol = _ridge(Xb, y)
        return {
            f"mlp.{layer}.affine_weight": mx.array(sol[:-1].T.astype(np.float32)),
            f"mlp.{layer}.affine_bias": mx.array(sol[-1].astype(np.float32)),
        }, f"chart-affine (mlp.{layer})"
    gate = n @ np.asarray(mx.astype(p[f"mlp.{layer}.gate_weight"], mx.float32),
                          dtype=np.float64).T
    up = n @ np.asarray(mx.astype(p[f"mlp.{layer}.up_weight"], mx.float32),
                        dtype=np.float64).T
    act = (gate / (1.0 + np.exp(-gate))) * up
    sol = _ridge(act, target)
    return {
        f"mlp.{layer}.down_weight": mx.array(sol.T.astype(np.float32)),
    }, f"swiglu-down (mlp.{layer})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--exact", default=".local-runs/qwen3-mlx-first-fit/exact")
    ap.add_argument("--corpus", default=".local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rel-ridge", type=float, default=0.05)
    ap.add_argument("--sequential", action="store_true",
                    help="Fit each transport with all previous transports active")
    ap.add_argument("--skip-terminal-refit", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx  # noqa: F401
    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_pipeline import _read_corpus, _tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_joint_waterfill_qwen3 import _chatmix_batches

    t0 = time.time()
    rt = MLXQwen3.load(args.artifact)
    rt_exact = MLXQwen3.load(args.exact)
    tokenizer = _tokenizer(args.tokenizer)
    fit_batches = _read_corpus(args.corpus, max_examples=32, max_length=128,
                               max_tokens=4096)
    batches = _chatmix_batches(
        fit_batches, tokenizer,
        "Answer the multiple-choice question. Reply with exactly one option letter and no other text.",
    )
    charted = [l for l, layer in enumerate(rt.layers)
               if f"mlp.{l}.chart_centers" in rt.parameters]
    print(f"[plan] charted mlp layers: {charted} "
          f"({len(batches)} calibration batches)", flush=True)

    updates, edges = fit_transports(rt, rt_exact, batches, charted,
                                    rel_ridge=args.rel_ridge,
                                    sequential=args.sequential)
    config = dict(rt.config)
    config["transports"] = list(config.get("transports", [])) + edges
    rt_new = MLXQwen3(dict(rt.parameters, **updates), config)

    if not args.skip_terminal_refit:
        last = 27
        refit, label = refit_terminal(rt_new, rt_exact, batches, last)
        rt_new = MLXQwen3(dict(rt_new.parameters, **refit), rt_new.config)
        print(f"[fit ] terminal refit: {label} ({time.time()-t0:.1f}s)", flush=True)

    rt_new.save(args.out)
    total = sum(int(v.size) for v in rt_new.parameters.values())
    print(f"[save] {args.out}: {total:,} scalars, "
          f"{len(rt_new.config.get('transports', []))} transport edges "
          f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
