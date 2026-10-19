"""EWC-anchored trajectory refinement for fitted Qwen3 MLX artifacts.

Measures the diagonal parameter Fisher of the fitted coefficient arrays on
the calibration corpus, then optimizes those coefficients toward the exact
teacher's behavior under an optional quadratic anchor.  The arms isolate
the EWC contribution: an unanchored control, an isotropic anchor, and a
Fisher-weighted anchor at the same steps/learning-rate/objective.

The calibration corpus must remain disjoint from every evaluation panel;
this script accepts only the fitting corpus and never touches assessment
rows.  Run with MLX_ENABLE_TF32=0 before importing MLX (the script sets it
if absent).

Example:

    python scripts/refine_qwen3_ewc.py \
      --student .local-runs/qwen3-mlx-first-fit/attention-and-mlp \
      --teacher .local-runs/qwen3-mlx-first-fit/exact \
      --corpus .local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl \
      --layer 13 --steps 40 --objective teacher_kl \
      --anchor-weights 0,10,100 --fisher-max-tokens 512 \
      --output-dir .local-runs/qwen3-ewc-refine
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")

# Fitted coefficient suffixes, matching mlx_qwen3_pipeline's trajectory set:
# reduced-attention weights plus chart-MLP linear/decoded arrays.  Geometry
# (input bases, chart bases/centers/scales) and norm weights stay fixed.
_COEFFICIENT_SUFFIXES = (
    "q_weight",
    "k_weight",
    "v_weight",
    "output_encoder_weight",
    "output_decoder_weight",
    "affine_weight",
    "affine_bias",
    "local_generator_encoders",
    "chart_mode_decoders",
)


def _mx():
    import mlx.core as mx

    return mx


def _coefficient_names(runtime, layer: int) -> list[str]:
    marker = f".{layer}."
    names = sorted(
        name
        for name in runtime.parameters
        if marker in name and name.rsplit(".", 1)[-1] in _COEFFICIENT_SUFFIXES
    )
    if not names:
        raise ValueError(f"no fitted coefficient arrays found for layer {layer}")
    return names


def _diagonal_stats(diagonal: dict) -> dict[str, float]:
    import numpy as np

    values = np.concatenate(
        [np.asarray(v, dtype=np.float32).reshape(-1) for v in diagonal.values()]
    )
    return {
        "element_count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Fitted MLX artifact")
    parser.add_argument("--teacher", required=True, help="Exact MLX artifact")
    parser.add_argument("--corpus", required=True, help="Fitting JSONL only")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--objective", choices=("terminal_mse", "teacher_kl"), default="teacher_kl"
    )
    parser.add_argument(
        "--anchor-weights",
        default="0,10",
        help="Comma list; each nonzero weight runs both Fisher and isotropic arms",
    )
    parser.add_argument("--fisher-max-tokens", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap calibration batches (0 = all)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_fit import (
        capture_parameter_fisher,
        optimize_trajectory,
    )
    from fisher_graph.mlx_qwen3_pipeline import _read_corpus

    student_dir = Path(args.student)
    teacher_dir = Path(args.teacher)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is nonempty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_weights = sorted(
        {float(value) for value in args.anchor_weights.split(",") if value.strip()}
    )
    if any(value < 0 for value in anchor_weights):
        raise SystemExit("anchor weights must be nonnegative")

    print(json.dumps({"event": "loading", "student": str(student_dir)}), file=sys.stderr)
    student = MLXQwen3.load(student_dir)
    teacher = MLXQwen3.load(teacher_dir)
    if student.config is teacher.config:
        raise SystemExit("student and teacher must be distinct artifacts")

    lexical = next(
        student.parameters[name]
        for name in ("lexical.weight", "lexical.codebook")
        if name in student.parameters
    )
    batches = _read_corpus(
        Path(args.corpus),
        tokenizer=None,
        max_examples=args.max_batches or 10_000,
        max_length=args.max_length,
        max_tokens=2**31 - 1,
        vocab_size=lexical.shape[0],
    )
    if not batches:
        raise SystemExit("corpus produced no batches")

    names = _coefficient_names(student, args.layer)
    print(json.dumps({"event": "fisher", "parameters": len(names),
                      "max_tokens": args.fisher_max_tokens}), file=sys.stderr)
    fisher = capture_parameter_fisher(
        student, batches, names, max_tokens=args.fisher_max_tokens
    )
    diagonal = fisher["diagonal"]
    stats = _diagonal_stats(diagonal)
    print(json.dumps({"event": "fisher_done", **stats}), file=sys.stderr)

    report = {
        "kind": "qwen3_ewc_trajectory_refinement",
        "student_artifact": str(student_dir),
        "teacher_artifact": str(teacher_dir),
        "corpus_sha256": __import__("hashlib").sha256(
            Path(args.corpus).read_bytes()
        ).hexdigest(),
        "corpus": str(Path(args.corpus)),
        "layer": args.layer,
        "objective": args.objective,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "calibration_batches": len(batches),
        "calibration_physical_tokens": sum(
            len(b["input_ids"][0]) for b in batches
        ),
        "selected_parameters": names,
        "fisher": {key: value for key, value in fisher.items() if key != "diagonal"}
        | {"diagonal_stats": stats},
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        "arms": [],
    }

    for anchor_weight in anchor_weights:
        for fisher_arm in ((False, True) if anchor_weight else (False,)):
            arm = (
                "unanchored"
                if not anchor_weight
                else ("fisher_ewc" if fisher_arm else "isotropic_anchor")
            )
            arm_id = f"{arm}_a{anchor_weight:g}"
            print(json.dumps({"event": "arm", "arm": arm_id}), file=sys.stderr)
            result = optimize_trajectory(
                student,
                teacher,
                batches,
                parameter_names=names,
                steps=args.steps,
                learning_rate=args.learning_rate,
                objective=args.objective,
                anchor_weight=anchor_weight,
                fisher_diagonal=diagonal if fisher_arm else None,
            )
            destination = out_dir / arm_id
            result.runtime.save(destination)
            report["arms"].append(
                {
                    "arm": arm_id,
                    "anchor_weight": anchor_weight,
                    "fisher_weighted": fisher_arm,
                    "artifact": str(destination),
                    **result.report,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "arm_done",
                        "arm": arm_id,
                        "initial_loss": result.report["initial_training_loss"],
                        "best_loss": result.report["best_training_loss"],
                    }
                ),
                file=sys.stderr,
            )
            del result

    (out_dir / "ewc_refine_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({"event": "done", "report": str(out_dir / "ewc_refine_report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
