"""Fit a joint KKT Pullback-Fisher water-filled Qwen3 MLX artifact.

Allocates a scalar-removal budget across all 28 attention + 28 MLP blocks via
``allocate_waterfill_joint_ranks`` (lowest distortion-per-scalar MLPs become
Euler-anchored chart MLPs; remaining budget water-fills attention input ranks
with exact W_O), then fits each selected block sequentially along the student
trajectory using the chat-mixed calibration corpus.

Usage:
  MLX_ENABLE_TF32=0 .venv/bin/python scripts/build_joint_waterfill_qwen3.py \
      --target-mib 512 \
      --output .local-runs/qwen3-ewc-refine/joint_waterfill_512mib
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _chatmix_batches(fit_batches, tokenizer, instruction):
    chat_batches = []
    for idx, b in enumerate(fit_batches):
        seq = b["input_ids"][0]
        text = tokenizer.decode(seq[:80], skip_special_tokens=True).strip()
        if idx % 2 == 0:
            user_msg = (
                f"{instruction}\n\nPassage:\n{text}\n\n"
                "Question:\nIs the passage above written in English?\n\nA. Yes\nB. No"
            )
        else:
            sents = [s.strip() for s in text.split(".") if s.strip()][:4]
            while len(sents) < 4:
                sents.append("None of the above")
            opts = "\n".join(f"{l}. {s[:40]}" for l, s in zip("ABCD", sents))
            user_msg = (
                f"{instruction}\n\nPassage:\n{text}\n\n"
                "Question:\nWhich statement appears in the passage?\n\n"
                f"{opts}"
            )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(prompt + "A", add_special_tokens=False)
        chat_batches.append({"input_ids": [ids]})
    return list(fit_batches) + chat_batches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", default=".local-runs/qwen3-mlx-first-fit/exact")
    parser.add_argument("--corpus", default=".local-runs/qwen3-mlx-first-fit/corpus/fit.jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-mib", type=int,
                        help="Scalar-removal budget in MiB (required unless --mlp-layers)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chart-count", type=int, default=4)
    parser.add_argument("--chart-rank", type=int, default=16)
    parser.add_argument("--generator-rank", type=int, default=17)
    parser.add_argument("--mlp-ridge", type=float, default=0.35)
    parser.add_argument("--attn-ridge", type=float, default=0.15)
    parser.add_argument("--min-mlp-layer-spacing", type=int, default=2)
    parser.add_argument("--mlp-expected-relative-error", type=float, default=0.50)
    parser.add_argument("--min-rank", type=int, default=384,
                        help="Floor on attention ranks — the quadratic Fisher model is not "
                             "trustworthy below ~1/8 of full rank")
    parser.add_argument("--mlp-layers",
                        help="Explicit MLP layers to chart (e.g. 'all' or '0,2,4-7'). "
                             "Overrides the allocator and disables attention fitting.")
    parser.add_argument("--schedule-report", help="Optional JSON schedule output path")
    args = parser.parse_args(argv)
    if not args.mlp_layers and args.target_mib is None:
        parser.error("either --target-mib or --mlp-layers is required")

    import mlx.core as mx
    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_fit import (
        allocate_waterfill_joint_ranks,
        capture_runtime_rows,
        fit_chart_mlp,
        fit_reduced_attention,
    )
    from fisher_graph.mlx_qwen3_pipeline import _read_corpus, _tokenizer

    t0 = time.time()
    tokenizer = _tokenizer(args.tokenizer)
    rt_exact = MLXQwen3.load(args.exact)
    fit_batches = _read_corpus(args.corpus, max_examples=32, max_length=128, max_tokens=4096)
    all_batches = _chatmix_batches(
        fit_batches, tokenizer,
        "Answer the multiple-choice question. Reply with exactly one option letter and no other text.",
    )

    if args.mlp_layers:
        from fisher_graph.mlx_qwen3_pipeline import _layers
        mlp_layers = set(_layers(args.mlp_layers, 28))
        attn_plan = {}
        sched = {
            "kind": "explicit_mlp_layer_schedule",
            "selected_mlp_layers": sorted(mlp_layers),
            "attention_schedule": {"layers": {}},
        }
    else:
        sched = allocate_waterfill_joint_ranks(
            rt_exact,
            fit_batches[:8],
            target_scalars_removed=args.target_mib * 1024 * 1024 // 2,
            chart_count=args.chart_count,
            chart_rank=args.chart_rank,
            generator_rank=args.generator_rank,
            mlp_expected_relative_error=args.mlp_expected_relative_error,
            min_mlp_layer_spacing=args.min_mlp_layer_spacing,
            min_rank=args.min_rank,
            fisher_max_tokens=32,
        )
        mlp_layers = set(sched["selected_mlp_layers"])
        attn_plan = {
            l: a for l, a in sched["attention_schedule"]["layers"].items()
            if a["scalars_removed"] > 0
        }
    print(f"[plan] MLP layers -> chart: {sorted(mlp_layers)}", flush=True)
    print(f"[plan] Attn layers -> reduced: "
          f"{ {l: (a['input_rank'], a['output_rank']) for l, a in attn_plan.items()} }", flush=True)
    if args.schedule_report:
        Path(args.schedule_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.schedule_report).write_text(json.dumps(sched, indent=2, default=str) + "\n")

    rt = rt_exact
    for l in range(28):
        if l in attn_plan:
            a = attn_plan[l]
            caps_a = capture_runtime_rows(
                rt, all_batches, [f"attention.{l}.normalized", f"attention.{l}.context"])
            res_a = fit_reduced_attention(
                rt, l, caps_a,
                input_rank=int(a["input_rank"]),
                output_rank=int(a["output_rank"]),
                relative_ridge=args.attn_ridge, solver="numpy_float64")
            u = dict(res_a.updates)
            if int(a["output_rank"]) >= 2048 and f"attention.{l}.output_encoder_weight" in u:
                del u[f"attention.{l}.output_encoder_weight"]
                u[f"attention.{l}.output_decoder_weight"] = rt_exact.parameters[
                    f"attention.{l}.output_decoder_weight"]
            rt = rt.with_parameters(u)
            print(f"[fit ] L{l:2d} attn r_in={a['input_rank']} r_out={a['output_rank']} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if l in mlp_layers:
            caps_m = capture_runtime_rows(
                rt, all_batches, [f"mlp.{l}.normalized", f"mlp.{l}.output"])
            res_m = fit_chart_mlp(
                rt, l, caps_m,
                chart_count=args.chart_count, chart_rank=args.chart_rank,
                generator_rank=args.generator_rank, relative_ridge=args.mlp_ridge)
            rt = res_m.runtime
            print(f"[fit ] L{l:2d} chart-mlp rel_err={res_m.report['training_relative_error']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    exact_scalars = sum(int(v.size) for v in rt_exact.parameters.values())
    fit_scalars = sum(int(v.size) for v in rt.parameters.values())
    removed = exact_scalars - fit_scalars
    print(f"[done] scalars {fit_scalars:,} (removed {removed:,} = {removed*2/1024**2:.2f} MiB) "
          f"in {time.time()-t0:.1f}s", flush=True)
    rt.save(args.output)
    print(f"[save] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
