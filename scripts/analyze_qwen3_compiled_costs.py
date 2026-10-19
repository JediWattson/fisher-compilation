"""MAC / memory cost accounting for compiled MLX Qwen3 artifacts.

Computes per-token decode MACs (multiply-accumulates), parameter memory, and
KV-cache memory from the actual weight tensors in a saved MLXQwen3 artifact —
no source model required.

Usage:
  MLX_ENABLE_TF32=0 .venv/bin/python scripts/analyze_qwen3_compiled_costs.py \
      --artifact .local-runs/qwen3-ewc-refine/joint_waterfill_286mib \
      --reference .local-runs/qwen3-mlx-first-fit/exact \
      --context-length 512 --report artifacts/qwen3/compiled_costs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_params(directory):
    import mlx.core as mx
    return mx.load(str(Path(directory) / "weights.safetensors"))


def _layer_ids(params):
    ids = set()
    for key in params:
        if key.startswith("attention.") and key.endswith(".input_norm_weight"):
            ids.add(int(key.split(".")[1]))
    return sorted(ids)


def _attn_macs(params, layer_id, config, hidden):
    """Per-token MACs for one attention block's dense projections (decode step)."""
    p = f"attention.{layer_id}."
    qkv_width = (config["query_heads"] + 2 * config["key_value_heads"]) * config["head_dimension"]
    if p + "input_basis" in params:
        r_in = int(params[p + "input_basis"].shape[1])
        in_macs = hidden * r_in + r_in * qkv_width  # basis project + QKV from coords
    else:
        in_macs = hidden * qkv_width
    ctx = params[p + "output_decoder_weight"]
    if p + "output_encoder_weight" in params:
        r_out = int(params[p + "output_encoder_weight"].shape[1])
        out_macs = config["query_heads"] * config["head_dimension"] * r_out + r_out * hidden
    else:
        out_macs = int(ctx.shape[0]) * int(ctx.shape[1])
    return {"in_macs": in_macs, "out_macs": out_macs,
            "input_rank": int(params[p + "input_basis"].shape[1]) if p + "input_basis" in params else hidden,
            "output_rank": int(params[p + "output_encoder_weight"].shape[1]) if p + "output_encoder_weight" in params else hidden}


def _mlp_macs(params, layer_id):
    """Per-token MACs for one MLP block (exact SwiGLU or chart)."""
    p = f"mlp.{layer_id}."
    if p + "gate_weight" in params:
        gate = params[p + "gate_weight"]
        return {"kind": "exact_swiglu", "macs": 3 * int(gate.shape[0]) * int(gate.shape[1])}
    hidden = int(params[p + "affine_weight"].shape[0])
    C = int(params[p + "chart_centers"].shape[0])
    R = int(params[p + "chart_bases"].shape[2])
    G = int(params[p + "local_generator_encoders"].shape[2])
    M = int(params[p + "chart_mode_decoders"].shape[2])
    macs = (
        hidden * hidden          # affine_weight
        + C * hidden             # membership distance ||n - center||^2
        + C * hidden * R         # chart_bases coordinates
        + C * (R + 1) * G        # local generator encoders
        + C * G * M              # packet @ chart_mode_decoders
    )
    if p + "mode_decoder_weight" in params:
        macs += int(params[p + "mode_decoder_weight"].shape[0]) * int(params[p + "mode_decoder_weight"].shape[1])
    return {"kind": "chart_mlp", "macs": macs, "charts": C, "chart_rank": R, "generator_rank": G}


def analyze(artifact_dir, *, context_length=512):
    import mlx.core as mx
    artifact_dir = Path(artifact_dir)
    params = _load_params(artifact_dir)
    config = json.loads((artifact_dir / "mlx_qwen3.json").read_text())
    layers = _layer_ids(params)

    scalar_total = int(sum(int(v.size) for v in params.values()))
    bytes_total = int(sum(int(v.nbytes) for v in params.values()))

    vocab, hidden = params["lexical.weight"].shape
    layer_cfg = config["layers"][0]["attention"] if "layers" in config else None
    acfg = config if layer_cfg is None else layer_cfg
    # MLXQwen3 config stores per-layer attention spec inside layers list when present.
    attn_specs = {}
    for l in layers:
        spec = config.get("layers", [{}] * (l + 1))[l].get("attention") if config.get("layers") else None
        attn_specs[l] = spec or {"query_heads": 16, "key_value_heads": 8, "head_dimension": 128}
    head_macs_per_key = 0
    for l in layers:
        a = attn_specs[l]
        head_macs_per_key += 2 * a["query_heads"] * a["head_dimension"]  # QK^T + PV per cached key
    attn_cache_macs_per_token = head_macs_per_key * context_length

    per_layer = []
    total_dense = 0
    for l in layers:
        spec = attn_specs[l]
        am = _attn_macs(params, l, spec, hidden)
        mm = _mlp_macs(params, l)
        dense = am["in_macs"] + am["out_macs"] + mm["macs"]
        total_dense += dense
        per_layer.append({
            "layer": l,
            "attention": am,
            "mlp": mm,
            "dense_macs_per_token": dense,
        })

    lm_head_macs = hidden * vocab  # tied head over full vocab per generated token
    kv_cache_bytes_per_token = sum(
        2 * a["key_value_heads"] * a["head_dimension"] * 2 for a in attn_specs.values()
    )

    return {
        "artifact": str(artifact_dir),
        "scalars": scalar_total,
        "weight_bytes_bf16": bytes_total,
        "weight_gib_bf16": bytes_total / (1024 ** 3),
        "layers": per_layer,
        "decode_macs": {
            "dense_body_per_token": total_dense,
            "attention_cache_per_token_at_ctx": attn_cache_macs_per_token,
            "context_length": context_length,
            "lm_head_per_generated_token": lm_head_macs,
            "total_per_token_at_ctx": total_dense + attn_cache_macs_per_token + lm_head_macs,
        },
        "kv_cache": {
            "bytes_per_token": kv_cache_bytes_per_token,
            "mib_at_ctx": kv_cache_bytes_per_token * context_length / (1024 ** 2),
            "context_length": context_length,
        },
    }


def _mib(scalars):
    return scalars * 2 / (1024 ** 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--reference", help="Optional exact artifact for comparison")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    result = analyze(args.artifact, context_length=args.context_length)
    if args.reference:
        result["reference"] = analyze(args.reference, context_length=args.context_length)
        ref = result["reference"]
        result["delta"] = {
            "scalars_removed": ref["scalars"] - result["scalars"],
            "weight_mib_saved": (ref["weight_bytes_bf16"] - result["weight_bytes_bf16"]) / (1024 ** 2),
            "dense_body_macs_saved_per_token": (
                ref["decode_macs"]["dense_body_per_token"]
                - result["decode_macs"]["dense_body_per_token"]
            ),
            "dense_body_macs_ratio": (
                result["decode_macs"]["dense_body_per_token"]
                / ref["decode_macs"]["dense_body_per_token"]
            ),
        }

    print(json.dumps(result, indent=2))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"report -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
