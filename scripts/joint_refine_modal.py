"""Joint gradient refinement of a charted-MLP + transports MLXQwen3 artifact.

Runs ``optimize_trajectory`` (MLX autodiff + Adam) over every fitted
coefficient — chart-MLP affine/encoder/decoder arrays and transport edges —
against the full-trajectory ``teacher_kl`` objective. This is the global
consistency step that per-layer closed-form fitting lacks: every parameter
moves to minimize the whole-model KL to the teacher, not a local residual.

Usage:
  MLX_ENABLE_TF32=0 .venv/bin/python scripts/joint_refine_modal.py \
    --student .local-runs/qwen3-ewc-refine/modal_all28_seq \
    --tokenizer <qwen3 snapshot> --steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_COEFF_SUFFIXES = (
    "affine_weight",
    "affine_bias",
    "local_generator_encoders",
    "chart_mode_decoders",
    "gate_weight",
    "up_weight",
    "down_weight",
)


def _selected(runtime):
    names = [
        n for n in runtime.parameters
        if n.startswith("transport.") or n.rsplit(".", 1)[-1] in _COEFF_SUFFIXES
    ]
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default=".local-runs/qwen3-mlx-first-fit/exact")
    ap.add_argument("--corpus", default=".local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=8,
                    help="Calibration batches per step (autodiff graph holds all)")
    ap.add_argument("--batch-offset", type=int, default=0,
                    help="Skip this many leading batches for coverage rotation")
    ap.add_argument("--anchor-weight", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args()

    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_fit import optimize_trajectory
    from fisher_graph.mlx_qwen3_pipeline import _read_corpus, _tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_joint_waterfill_qwen3 import _chatmix_batches

    t0 = time.time()
    rt = MLXQwen3.load(args.student)
    rt_exact = MLXQwen3.load(args.teacher)
    tokenizer = _tokenizer(args.tokenizer)
    fit_batches = _read_corpus(args.corpus, max_examples=32, max_length=128,
                               max_tokens=4096)
    batches = _chatmix_batches(
        fit_batches, tokenizer,
        "Answer the multiple-choice question. Reply with exactly one option letter and no other text.",
    )[args.batch_offset : args.batch_offset + args.max_batches]
    names = _selected(rt)
    n_scalars = sum(int(rt.parameters[n].size) for n in names)
    print(f"[plan] {len(names)} parameter arrays, {n_scalars:,} scalars under Adam "
          f"({len(batches)} batches/step, lr={args.learning_rate}, "
          f"objective=teacher_kl)", flush=True)

    result = optimize_trajectory(
        rt, rt_exact, batches,
        parameter_names=names,
        steps=args.steps,
        learning_rate=args.learning_rate,
        anchor_weight=args.anchor_weight,
        objective="teacher_kl",
        max_grad_norm=args.max_grad_norm,
    )
    print(json.dumps(result.report["training_loss_history"]), flush=True)
    out = args.out or f"{args.student}_refined"
    result.runtime.save(out)
    print(f"[save] {out}: initial={result.report['initial_training_loss']:.4f} "
          f"best={result.report['best_training_loss']:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
