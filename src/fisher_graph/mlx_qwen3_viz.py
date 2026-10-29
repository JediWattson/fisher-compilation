"""Interactive 3D WebGL (Three.js) visualizer for Fisher-Graph Qwen3.5 MLX artifacts.

Exports full 64-sublayer cluster memberships, chart routing, Boltzmann free-energy
gating, KKT carrier/MLP rank geometry, cross-layer supernode transport bridges,
and causal logit pushes into an interactive 3D scene.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def compute_3d_trace(
    runtime: Any,
    tokenizer: Any,
    prompt: str,
    *,
    chat: bool = False,
    max_prompt_tokens: int = 512,
    include_counterfactuals: bool = True,
) -> dict[str, Any]:
    """Compute a complete 3D trajectory trace for ``prompt`` across all 32 layers."""
    import mlx.core as mx
    import numpy as np

    rendered = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if chat
        else prompt
    )
    ids = tokenizer.encode(rendered, add_special_tokens=not chat)
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError(f"Prompt must contain 1..{max_prompt_tokens} tokens")

    p = runtime.parameters
    n_layers = len(runtime.layers)
    chart_layers = sorted(i for i in range(n_layers) if f"mlp.{i}.chart_centers" in p)
    chart_dnet_layers = sorted(i for i in range(n_layers) if f"deltanet.{i}.chart_centers" in p)
    chart_attn_layers = sorted(i for i in range(n_layers) if f"attention.{i}.chart_centers" in p)
    gated_dnet_layers = sorted(
        i for i in range(n_layers)
        if f"deltanet.{i}.energy_threshold" in p and f"deltanet.{i}.chart_centers" in p
    )
    head_dnet_layers = sorted(
        i for i in range(n_layers)
        if f"deltanet.{i}.head_cluster_assign" in p and f"deltanet.{i}.qkv_weight" in p
    )
    head_attn_layers = sorted(
        i for i in range(n_layers) if f"attention.{i}.head_cluster_assign" in p
    )
    clustered_mlp_layers = sorted(
        i for i in range(n_layers) if f"mlp.{i}.cluster_scales" in p
    )
    dnet_route_site = {
        i: (f"deltanet.{i}.context" if f"deltanet.{i}.qkv_weight" in p else f"deltanet.{i}.normalized")
        for i in chart_dnet_layers
    }

    all_delta_sites = [
        f"deltanet.{i}.delta" if "deltanet" in runtime.layers[i] else f"attention.{i}.delta"
        for i in range(n_layers)
    ] + [f"mlp.{i}.delta" for i in range(n_layers)]

    capture_sites = sorted({
        *(f"mlp.{i}.normalized" for i in chart_layers),
        *(dnet_route_site[i] for i in chart_dnet_layers),
        *(f"deltanet.{i}.normalized" for i in gated_dnet_layers),
        *(f"attention.{i}.context" for i in chart_attn_layers),
        *(f"mlp.{i}.act" for i in clustered_mlp_layers),
        *(f"deltanet.{i}.context" for i in head_dnet_layers),
        *(f"attention.{i}.context" for i in head_attn_layers),
        *all_delta_sites,
        "terminal",
    })

    arr_ids = mx.array([ids], dtype=mx.int32)
    out = runtime(arr_ids, capture=capture_sites)
    logits = out.logits[0, -1].astype(mx.float32)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(logits, probs)
    logits_np = np.asarray(logits)
    probs_np = np.asarray(probs)
    top_idx = np.argsort(logits_np)[::-1][:5]
    win_id = int(top_idx[0])

    tokens = [
        {"position": pos, "token_id": int(tid), "token": tokenizer.decode([int(tid)])}
        for pos, tid in enumerate(ids)
    ]

    architecture = []
    for i, layer in enumerate(runtime.layers):
        mc = layer.get("mlp", {})
        m_inter = (
            int(p[f"mlp.{i}.gate_weight"].shape[0])
            if f"mlp.{i}.gate_weight" in p
            else int(mc.get("intermediate_size", 12288))
        )
        m_full = int(mc.get("full_intermediate_size", max(m_inter, 12288)))
        if "deltanet" in layer:
            dc = layer["deltanet"]
            architecture.append({
                "layer": i,
                "mixer_type": "deltanet",
                "mixer_key": f"deltanet.{i}",
                "carrier_rank": int(dc.get("carrier_rank", dc.get("key_head_dimension", 128))),
                "full_carrier_dim": int(dc.get("key_head_dimension", 128)),
                "value_heads": int(dc.get("value_heads", 32)),
                "charted": i in chart_dnet_layers,
                "energy_gated": i in gated_dnet_layers,
                "physically_bypassed": f"deltanet.{i}.qkv_weight" not in p,
                "coalesced_mlp": bool(dc.get("coalesced_mlp", False)),
                "transport_source_layer": dc.get("transport_source_layer"),
                "head_cluster_count": int(dc.get("head_cluster_count", 4 if i in head_dnet_layers else 1)),
                "mlp_intermediate_size": m_inter,
                "mlp_full_intermediate_size": m_full,
                "mlp_cluster_count": int(mc.get("cluster_count", 4 if i in clustered_mlp_layers else 1)),
            })
        else:
            ac = layer["attention"]
            architecture.append({
                "layer": i,
                "mixer_type": "attention",
                "mixer_key": f"attention.{i}",
                "query_heads": int(ac.get("query_heads", 16)),
                "kv_heads": int(ac.get("key_value_heads", 4)),
                "head_dimension": int(ac.get("head_dimension", 256)),
                "value_rank": int(ac.get("value_rank", ac.get("head_dimension", 256))),
                "charted": i in chart_attn_layers,
                "energy_gated": False,
                "physically_bypassed": False,
                "coalesced_mlp": False,
                "transport_source_layer": None,
                "head_cluster_count": int(ac.get("head_cluster_count", 4 if i in head_attn_layers else 1)),
                "mlp_intermediate_size": m_inter,
                "mlp_full_intermediate_size": m_full,
                "mlp_cluster_count": int(mc.get("cluster_count", 4 if i in clustered_mlp_layers else 1)),
            })

    def _pack_routing(memb_np):
        return [
            {
                "position": pos,
                "dominant_chart": int(np.argmax(memb_np[pos])),
                "dominant_membership": round(float(np.max(memb_np[pos])), 4),
                "memberships": [round(float(v), 4) for v in memb_np[pos]],
            }
            for pos in range(len(ids))
        ]

    chart_routing = {}
    for i in chart_dnet_layers:
        c_val = out.captures[dnet_route_site[i]][0].astype(mx.float32)
        centers = p[f"deltanet.{i}.chart_centers"].astype(mx.float32)
        scales = p[f"deltanet.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["deltanet"].get("temperature", 1.0))
        disp = c_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1))
        chart_routing[f"deltanet.{i}"] = _pack_routing(memb)

    for i in chart_attn_layers:
        c_val = out.captures[f"attention.{i}.context"][0].astype(mx.float32)
        centers = p[f"attention.{i}.chart_centers"].astype(mx.float32)
        scales = p[f"attention.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["attention"].get("temperature", 1.0))
        disp = c_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1))
        chart_routing[f"attention.{i}"] = _pack_routing(memb)

    for i in chart_layers:
        n_val = out.captures[f"mlp.{i}.normalized"][0].astype(mx.float32)
        centers = p[f"mlp.{i}.chart_centers"].astype(mx.float32)
        scales = p[f"mlp.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["mlp"].get("temperature", 1.0))
        disp = n_val[:, None, :] - centers[None, :, :]
        memb = np.asarray(mx.softmax(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1))
        chart_routing[f"mlp.{i}"] = _pack_routing(memb)

    head_cluster_routing = {}
    for i in head_dnet_layers:
        dc = runtime.layers[i]["deltanet"]
        vh, vd = int(dc["value_heads"]), int(dc["value_head_dimension"])
        ctx_val = np.asarray(out.captures[f"deltanet.{i}.context"][0].astype(mx.float32)).reshape(len(ids), vh, vd)
        e_head = np.sum(ctx_val ** 2, axis=-1)
        assign = np.asarray(p[f"deltanet.{i}.head_cluster_assign"])
        scales = np.maximum(np.asarray(p[f"deltanet.{i}.head_cluster_scales"].astype(mx.float32)), 1e-12)
        c_cnt = int(scales.shape[0])
        e_clus = np.stack([np.sum(e_head[:, assign == c], axis=-1) for c in range(c_cnt)], axis=-1) / scales[None, :]
        memb = e_clus / np.maximum(np.sum(e_clus, axis=-1, keepdims=True), 1e-12)
        head_cluster_routing[f"deltanet.{i}"] = _pack_routing(memb)

    for i in head_attn_layers:
        ac = runtime.layers[i]["attention"]
        qh, dh = int(ac["query_heads"]), int(ac["head_dimension"])
        ctx_val = np.asarray(out.captures[f"attention.{i}.context"][0].astype(mx.float32)).reshape(len(ids), qh, dh)
        e_head = np.sum(ctx_val ** 2, axis=-1)
        assign = np.asarray(p[f"attention.{i}.head_cluster_assign"])
        scales = np.maximum(np.asarray(p[f"attention.{i}.head_cluster_scales"].astype(mx.float32)), 1e-12)
        c_cnt = int(scales.shape[0])
        e_clus = np.stack([np.sum(e_head[:, assign == c], axis=-1) for c in range(c_cnt)], axis=-1) / scales[None, :]
        memb = e_clus / np.maximum(np.sum(e_clus, axis=-1, keepdims=True), 1e-12)
        head_cluster_routing[f"attention.{i}"] = _pack_routing(memb)

    neuron_cluster_routing = {}
    for i in clustered_mlp_layers:
        act_val = np.asarray(out.captures[f"mlp.{i}.act"][0].astype(mx.float32))
        if f"mlp.{i}.cluster_indices" in p:
            c_idx_arr = np.asarray(p[f"mlp.{i}.cluster_indices"])
            act_val = act_val[:, c_idx_arr]
        scales = np.maximum(np.asarray(p[f"mlp.{i}.cluster_scales"].astype(mx.float32)), 1e-12)
        c_cnt = int(scales.shape[0])
        act_clus = act_val.reshape(len(ids), c_cnt, -1)
        e_clus = np.sum(act_clus ** 2, axis=-1) / scales[None, :]
        memb = e_clus / np.maximum(np.sum(e_clus, axis=-1, keepdims=True), 1e-12)
        neuron_cluster_routing[f"mlp.{i}"] = _pack_routing(memb)

    energy_gating = {}
    for i in gated_dnet_layers:
        n_val = out.captures[f"deltanet.{i}.normalized"][0].astype(mx.float32)
        centers = p[f"deltanet.{i}.chart_centers"].astype(mx.float32)
        scales = p[f"deltanet.{i}.chart_distance_scales"].astype(mx.float32)
        temp = float(runtime.layers[i]["deltanet"].get("temperature", 1.0))
        disp = n_val[:, None, :] - centers[None, :, :]
        fe = np.asarray(-mx.logsumexp(-0.5 * mx.sum(disp ** 2, axis=-1) / (scales[None, :] ** 2 * temp), axis=-1))
        thr = float(p[f"deltanet.{i}.energy_threshold"].astype(mx.float32).item())
        mean_fe = float(np.mean(fe))
        energy_gating[str(i)] = {
            "threshold": round(thr, 4),
            "mean_free_energy": round(mean_fe, 4),
            "take_supernode": bool(mean_fe <= thr),
            "inside_count": int(np.sum(fe <= thr)),
            "free_energy": [round(float(v), 4) for v in fe],
            "token_skip": [bool(float(v) <= thr) for v in fe],
        }

    # Direct terminal logit attribution for all 64 sublayers across all token positions
    head_key = "lexical.head_weight" if "lexical.head_weight" in p else "lexical.weight"
    quant = runtime.config.get("quantization") or {}
    if head_key + ".scales" in p:
        u_win = mx.dequantize(
            p[head_key][win_id : win_id + 1],
            scales=p[head_key + ".scales"][win_id : win_id + 1],
            biases=p[head_key + ".biases"][win_id : win_id + 1] if head_key + ".biases" in p else None,
            group_size=int(quant.get("group_size", 64)),
            bits=int(quant.get("bits", 4)),
            mode=str(quant.get("mode", "affine")),
        )[0].astype(mx.float32)
    else:
        u_win = p[head_key][win_id].astype(mx.float32)
    t_spec = runtime.config["terminal_norm"]
    t_gain = p["terminal.weight"].astype(mx.float32)
    if t_spec.get("scale_parameterization") == "unit_offset":
        t_gain = t_gain + 1.0
    term_all = out.captures["terminal"][0].astype(mx.float32)
    inv_rms_all = np.asarray(mx.rsqrt(mx.mean(term_all * term_all, axis=-1) + float(t_spec["epsilon"])))
    u_scaled = np.asarray(u_win * t_gain)
    g_dir_last = u_scaled * float(inv_rms_all[-1])

    direct_logit_push = {}
    bands: tuple[list, list, list] = ([], [], [])
    b1, b2 = max(1, n_layers // 3), max(2, (2 * n_layers) // 3)
    for s_name in all_delta_sites:
        if s_name in out.captures:
            d_mat = np.asarray(out.captures[s_name][0].astype(mx.float32))
            pushes = (d_mat @ u_scaled) * inv_rms_all
            direct_logit_push[s_name] = [round(float(v), 4) for v in pushes]
            d_last = d_mat[-1]
            d_norm = float(np.linalg.norm(d_last)) + 1e-6
            l_idx = int(s_name.split(".")[1])
            b_idx = 0 if l_idx < b1 else (1 if l_idx < b2 else 2)
            bands[b_idx].append((abs(float(np.dot(d_last, g_dir_last))) / (d_norm ** 0.5), s_name))

    ablation_impacts = []
    if include_counterfactuals:
        top_sites = {s for band in bands for _, s in sorted(band, reverse=True)[:6]}
        ablation_sites = sorted(
            top_sites,
            key=lambda s: (int(s.split(".")[1]), 0 if not s.startswith("mlp.") else 1),
        )
        for site in ablation_sites:
            ablated = runtime(arr_ids, perturbations={site: -out.captures[site]}).logits[0, -1].astype(mx.float32)
            mx.eval(ablated)
            ab_np = np.asarray(ablated)
            ablation_impacts.append({
                "site": site,
                "winning_logit_push": round(float(logits_np[win_id] - ab_np[win_id]), 4),
                "ablated_top1_token_id": int(np.argmax(ab_np)),
                "ablated_top1_token": tokenizer.decode([int(np.argmax(ab_np))]),
            })
    mx.clear_cache()

    mac_stats = None
    try:
        from .mlx_qwen3_pipeline import _mac_stats

        macs, state_bytes = _mac_stats(runtime, len(ids))
        mac_stats = {"macs_per_token": int(macs), "state_bytes": int(state_bytes)}
    except Exception:
        pass

    return {
        "prompt": prompt,
        "chat": bool(chat),
        "tokens": tokens,
        "architecture": architecture,
        "top5_predictions": [
            {
                "token_id": int(idx),
                "token": tokenizer.decode([int(idx)]),
                "logit": round(float(logits_np[idx]), 3),
                "probability": round(float(probs_np[idx]), 4),
            }
            for idx in top_idx
        ],
        "chart_routing": chart_routing,
        "head_cluster_routing": head_cluster_routing,
        "neuron_cluster_routing": neuron_cluster_routing,
        "energy_gating": energy_gating,
        "direct_logit_push": direct_logit_push,
        "ablation_impacts": ablation_impacts,
        "mac_stats": mac_stats,
        "meta": {"mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"), "n_layers": n_layers},
    }


def _viewer_template() -> str:
    try:
        from importlib import resources

        return resources.files("fisher_graph").joinpath("assets/chart_graph_viz.html").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        return (Path(__file__).resolve().parent / "assets" / "chart_graph_viz.html").read_text(encoding="utf-8")


def render_viz_html(trace: dict[str, Any] | None = None, *, default_prompt: str = "") -> str:
    """Render the standalone viewer. ``trace`` may be embedded directly (offline
    snapshot) or left ``None`` for server-backed pages that fetch ``/trace.json``."""
    payload = "null" if trace is None else json.dumps(trace, allow_nan=False).replace("</", "<\\/")
    inject = (
        "<script>window.__TRACE__ = " + payload + ";"
        "window.__DEFAULT_PROMPT__ = " + json.dumps(default_prompt) + ";</script>\n"
    )
    return _viewer_template().replace("</head>", inject + "</head>", 1)


def write_viz_html(trace: dict[str, Any], path: str | Path, *, default_prompt: str = "") -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_viz_html(trace, default_prompt=default_prompt or trace.get("prompt", "")), encoding="utf-8")
    return out


def serve_viz(
    runtime: Any,
    tokenizer: Any,
    *,
    host: str,
    port: int,
    default_prompt: str,
    default_chat: bool = False,
    max_prompt_tokens: int = 512,
) -> HTTPServer:
    """Serve the interactive 3D viewer; ``/trace.json`` recomputes live traces.

    Uses a single-threaded ``HTTPServer`` on purpose: MLX quantized matmul
    kernels require ``Stream(gpu, 0)``, which only exists on the thread that
    loaded the runtime — request handlers must run on the serving thread.
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, render_viz_html(None, default_prompt=default_prompt).encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/healthz":
                self._send(200, b"ok", "text/plain")
                return
            if parsed.path == "/trace.json":
                from urllib.parse import parse_qs

                q = parse_qs(parsed.query)
                prompt = (q.get("prompt") or [default_prompt])[0]
                chat = (q.get("chat") or ["1" if default_chat else "0"])[0] == "1"
                fast = (q.get("fast") or ["0"])[0] == "1"
                try:
                    trace = compute_3d_trace(
                        runtime, tokenizer, prompt,
                        chat=chat, max_prompt_tokens=max_prompt_tokens,
                        include_counterfactuals=not fast,
                    )
                    body = json.dumps(trace, allow_nan=False).encode("utf-8")
                except (ValueError, KeyError) as exc:
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, body, "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, fmt: str, *args: Any) -> None:
            import sys

            path = getattr(self, "path", "")
            if "/trace.json" in path or path == "/":
                print(f"viz {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

    return HTTPServer((host, port), Handler)
