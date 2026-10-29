"""3D chart-graph trace export, viewer embedding, and endpoint tests."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request

import numpy as np
import pytest


def test_import_does_not_load_mlx():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import fisher_graph.mlx_qwen3_viz; assert 'mlx.core' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def mx():
    from fisher_graph.mlx_executor import mlx_metal_is_usable

    if not mlx_metal_is_usable():
        pytest.skip("MLX/Metal is unavailable")
    return pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def qwen(mx):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fisher_graph.mlx_qwen3 import MLXQwen3

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(482)
        config = transformers.Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4,
            max_position_embeddings=64, tie_word_embeddings=True,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    return MLXQwen3.from_hf(model)


class _ToyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(c) % 27) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + ((i - 3) % 26)) for i in ids)


def test_trace_schema_covers_all_sublayers(qwen, mx):
    from fisher_graph.mlx_qwen3_viz import compute_3d_trace

    runtime = qwen
    trace = compute_3d_trace(runtime, _ToyTokenizer(), "hunter2 ok", include_counterfactuals=False)
    n = len(runtime.layers)
    assert trace["meta"]["n_layers"] == n
    assert len(trace["architecture"]) == n
    ids = [3 + (ord(c) % 27) for c in "hunter2 ok"]
    assert [t["token_id"] for t in trace["tokens"]] == ids
    # every sublayer gets a per-position direct logit push
    expected_sites = {f"attention.{i}.delta" for i in range(n)} | {f"mlp.{i}.delta" for i in range(n)}
    assert set(trace["direct_logit_push"]) == expected_sites
    assert all(len(v) == len(ids) for v in trace["direct_logit_push"].values())
    # architecture rows carry the geometry the 3D scene needs
    row = trace["architecture"][0]
    assert row["mixer_type"] == "attention" and row["mixer_key"] == "attention.0"
    assert row["query_heads"] == 4 and row["mlp_intermediate_size"] == 32
    assert row["charted"] is False and row["energy_gated"] is False
    # routing maps exist (empty on an uncompiled artifact) and predictions are sane
    assert trace["chart_routing"] == {} and trace["energy_gating"] == {}
    assert len(trace["top5_predictions"]) == 5
    assert all(abs(p["probability"]) <= 1.0 for p in trace["top5_predictions"])
    assert trace["mac_stats"]["macs_per_token"] > 0


def test_trace_counterfactuals_and_output_equivalence(qwen, mx):
    from fisher_graph.mlx_qwen3_viz import compute_3d_trace

    runtime = qwen
    ids = mx.array([[3 + (ord(c) % 27) for c in "hello"]], dtype=mx.int32)
    reference = np.asarray(runtime(ids).logits[0, -1])
    trace = compute_3d_trace(runtime, _ToyTokenizer(), "hello", include_counterfactuals=True)
    # capture path must not perturb logits
    win = trace["top5_predictions"][0]
    assert win["token_id"] == int(np.argmax(reference))
    np.testing.assert_allclose(
        [p["logit"] for p in trace["top5_predictions"]],
        np.sort(reference)[::-1][:5], rtol=1e-4, atol=1e-3,
    )
    # counterfactual ablation rows correspond to real sites and real logits
    sites = {row["site"] for row in trace["ablation_impacts"]}
    assert sites <= set(trace["direct_logit_push"])
    assert all("winning_logit_push" in row and "ablated_top1_token" in row for row in trace["ablation_impacts"])


def test_render_viz_html_embeds_trace_and_defaults(qwen, mx):
    from fisher_graph.mlx_qwen3_viz import compute_3d_trace, render_viz_html

    trace = compute_3d_trace(qwen, _ToyTokenizer(), "ab", include_counterfactuals=False)
    html = render_viz_html(trace, default_prompt="ab")
    assert "window.__TRACE__" in html and '"architecture"' in html
    assert 'window.__DEFAULT_PROMPT__ = "ab"' in html
    assert "chart_graph" in html or "three" in html
    served = render_viz_html(None, default_prompt="zz")
    assert "window.__TRACE__ = null" in served


def test_serve_viz_endpoints(qwen, mx):
    from fisher_graph.mlx_qwen3_viz import serve_viz

    server = serve_viz(
        qwen, _ToyTokenizer(), host="127.0.0.1", port=0, default_prompt="hey there",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert "window.__TRACE__ = null" in html
        body = urllib.request.urlopen(base + "/trace.json?prompt=yo&fast=1", timeout=60).read()
        trace = json.loads(body)
        assert trace["prompt"] == "yo" and len(trace["architecture"]) == 2
        assert trace["ablation_impacts"] == []
        req = urllib.request.Request(base + "/trace.json?prompt=" + "x" * 600)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=10)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
