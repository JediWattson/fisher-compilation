"""Causal layer-vitality scan for compiled Qwen3 MLX artifacts.

For each layer l, measures the next-token KL impact of muting
``attention.{l}.delta`` and ``mlp.{l}.delta`` across a prompt battery, plus the
token-0 massive-activation sink magnitude at ``attention.3.input``. A layer
already starved by compression shows near-zero mute impact (nothing left to
kill); a live layer shows large impact.

Usage:
  MLX_ENABLE_TF32=0 .venv/bin/python scripts/layer_vitality_scan.py \
      --artifacts exact=.local-runs/qwen3-mlx-first-fit/exact \
                  wf286=.local-runs/qwen3-ewc-refine/joint_waterfill_286mib \
                  wf768=.local-runs/qwen3-ewc-refine/joint_waterfill_768mib \
      --tokenizer <qwen3 snapshot> --prompts 8 --report out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


_PROMPTS = [
    "The capital of France is",
    "In a right triangle, the square of the hypotenuse equals",
    "Water boils at 100 degrees Celsius at standard",
    "The chemical symbol for gold is",
    "Isaac Newton formulated the laws of",
    "Photosynthesis converts light energy into",
    "The first president of the United States was",
    "To bake bread, you need flour, water, and",
    "The largest planet in the solar system is",
    "Shakespeare wrote the play Romeo and",
    "The mitochondria is known as the powerhouse of",
    "In economics, supply and demand determine",
]


def _kl(p_logits, q_logits):
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    p = p - p.max(); q = q - q.max()
    pe = np.exp(p); qe = np.exp(q)
    pe /= pe.sum(); qe /= qe.sum()
    return float(np.sum(pe * (np.log(pe) - np.log(qe))))


def scan(runtime, tokenizer, *, prompt_texts, max_context_tokens):
    import mlx.core as mx

    n_layers = len(runtime.layers)
    impact_attn = [[] for _ in range(n_layers)]
    impact_mlp = [[] for _ in range(n_layers)]
    sink_norms = []
    base_top1 = []

    for text in prompt_texts:
        ids = tokenizer.encode(text)
        if not ids or len(ids) > max_context_tokens:
            continue
        x = mx.array([ids], dtype=mx.int32)
        out = runtime.prefill(x, capture={"attention.3.input"})
        base_logits = np.asarray(out.logits[0, -1].astype(mx.float32))
        base_top1.append(int(mx.argmax(out.logits[0, -1]).item()))
        sink_norms.append(float(mx.max(mx.abs(out.captures["attention.3.input"])).item()))
        for l in range(n_layers):
            for site, acc in ((f"attention.{l}.delta", impact_attn[l]),
                              (f"mlp.{l}.delta", impact_mlp[l])):
                om = runtime.prefill(x, muted_sites={site})
                acc.append(_kl(base_logits, np.asarray(om.logits[0, -1].astype(mx.float32))))
    return {
        "impact_attention_kl": [float(np.mean(v)) if v else None for v in impact_attn],
        "impact_mlp_kl": [float(np.mean(v)) if v else None for v in impact_mlp],
        "token0_sink_max_abs": float(np.mean(sink_norms)) if sink_norms else None,
        "n_prompts": len(base_top1),
        "base_top1_ids": base_top1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", nargs="+", required=True,
                        help="name=path pairs")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    from fisher_graph.mlx_qwen3 import MLXQwen3
    from fisher_graph.mlx_qwen3_pipeline import _tokenizer

    tokenizer = _tokenizer(args.tokenizer)
    prompt_texts = _PROMPTS[: args.prompts]

    results = {}
    for spec in args.artifacts:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise ValueError(f"--artifacts entries must be name=path, got {spec!r}")
        runtime = MLXQwen3.load(path)
        print(f"[scan] {name}: {len(prompt_texts)} prompts x {len(runtime.layers)*2} mutes...",
              file=sys.stderr, flush=True)
        results[name] = scan(runtime, tokenizer, prompt_texts=prompt_texts,
                             max_context_tokens=args.max_context_tokens)
        print(f"[done] {name} sink={results[name]['token0_sink_max_abs']:.1f}",
              file=sys.stderr, flush=True)

    payload = {"kind": "layer_vitality_scan", "prompts": prompt_texts, "results": results}
    print(json.dumps(payload, indent=2))
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
