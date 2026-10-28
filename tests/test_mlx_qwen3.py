"""Independent Metal parity checks against tiny native HF Qwen3 models."""

from __future__ import annotations

import copy
import os

# M5 defaults to reduced-precision float32 GEMMs. These are fidelity tests;
# also set this in the process environment when collecting alongside other
# MLX modules, whose imports can initialize the backend before this module.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import numpy as np
import pytest
import torch

from fisher_graph.mlx_executor import mlx_metal_is_usable

_METAL_USABLE = mlx_metal_is_usable()
pytestmark = pytest.mark.skipif(not _METAL_USABLE, reason="MLX Metal is unavailable")
mx = pytest.importorskip("mlx.core") if _METAL_USABLE else None
transformers = pytest.importorskip("transformers")

from fisher_graph.adapters import Qwen3CausalLMAdapter
from fisher_graph.mlx_qwen3 import MLXQwen3, _rms
from fisher_graph.qwen3_joint_compiler import compile_qwen3_joint_exact_control


def _numpy(value):
    return np.array(value.astype(mx.float32))


@pytest.fixture(scope="module")
def runtimes():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(810)
            config = transformers.Qwen3Config(
                vocab_size=41, hidden_size=24, intermediate_size=40,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=128, tie_word_embeddings=True,
                attention_dropout=0.0, attention_bias=True,
            )
            config._attn_implementation = "eager"
            model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
            for module in model.modules():
                if type(module).__name__ == "Qwen3RMSNorm":
                    module.weight.uniform_(0.7, 1.3)
        adapter = Qwen3CausalLMAdapter(model)
        executor = compile_qwen3_joint_exact_control(adapter)
        direct = MLXQwen3.from_hf(model)
        compiled = MLXQwen3.from_torch(executor)
        yield model, adapter, executor, direct, compiled
    finally:
        torch.set_num_threads(old_threads)


def _batch(layout="plain"):
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    positions = torch.arange(4).expand(2, -1).clone()
    if layout == "left":
        mask[0, :2] = 0
        ids[0, :2] = 0
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
    elif layout == "right":
        mask[0, 2:] = 0
        ids[0, 2:] = 0
    elif layout == "gapped":
        positions = torch.tensor([[5, 8, 13, 15], [3, 4, 7, 12]])
    elif layout == "repeated":
        positions = torch.tensor([[0, 1, 1, 2], [2, 2, 4, 4]])
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}


def _mlx_batch(batch):
    return {key: mx.array(value.numpy()) for key, value in batch.items()}


@pytest.mark.parametrize("layout", ["plain", "left", "right", "gapped", "repeated"])
def test_native_and_compiled_forward_match_on_valid_queries(runtimes, layout):
    model, _, _, direct, compiled = runtimes
    batch = _batch(layout)
    with torch.inference_mode():
        reference = model(**batch).logits.numpy()
    valid = batch["attention_mask"].bool().numpy()
    for runtime in (direct, compiled):
        output = runtime(**_mlx_batch(batch))
        actual = _numpy(output.logits)
        np.testing.assert_allclose(actual[valid], reference[valid], rtol=2e-5, atol=3e-6)
        assert np.array_equal(actual[valid].argmax(-1), reference[valid].argmax(-1))
        assert output.cache.key_values[0][0].shape == (2, 2, 4, 8)


def test_boundary_captures_and_no_source_calls_after_conversion(runtimes, monkeypatch):
    model, _, executor, direct, compiled = runtimes
    batch = _batch("left")
    expected, handles = {}, []

    def capture(name):
        def hook(module, inputs, output):
            expected[name] = output.detach().clone()
        return hook

    def capture_input(name):
        def hook(module, inputs):
            expected[name] = inputs[0].detach().clone()
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(capture("embedding")))
    for i, layer in enumerate(model.model.layers):
        for name, module in ((f"attention.{i}", layer.input_layernorm),
                             (f"mlp.{i}", layer.post_attention_layernorm)):
            handles.append(module.register_forward_pre_hook(capture_input(name + ".input")))
            handles.append(module.register_forward_hook(capture(name + ".normalized")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            capture_input(f"attention.{i}.context")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            capture(f"attention.{i}.output")))
        handles.append(layer.mlp.register_forward_hook(capture(f"mlp.{i}.output")))
    handles.append(model.model.norm.register_forward_pre_hook(capture_input("terminal")))
    handles.append(model.model.norm.register_forward_hook(capture("terminal.normalized")))
    try:
        with torch.inference_mode():
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    def forbidden(*args, **kwargs):
        raise AssertionError("MLX execution called a source/torch module")

    valid = batch["attention_mask"].bool().numpy()
    with monkeypatch.context() as patch:
        for module in (*model.modules(), *executor.modules()):
            patch.setattr(module, "forward", forbidden)
        for runtime in (direct, compiled):
            output = runtime(**_mlx_batch(batch), capture=True, return_logits=False)
            assert output.logits is None
            for name, reference in expected.items():
                np.testing.assert_allclose(
                    _numpy(output.captures[name])[valid], reference.numpy()[valid],
                    rtol=3e-5, atol=3e-6, err_msg=name,
                )


@pytest.mark.parametrize("layout", ["left", "right", "gapped"])
def test_cache_decode_preserves_row_positions_and_matches_full_native(runtimes, layout):
    model, _, _, direct, compiled = runtimes
    original = _batch(layout)
    for runtime in (direct, compiled):
        batch = {k: v.clone() for k, v in original.items()}
        output = runtime.prefill(**_mlx_batch(batch))
        for step in range(4):
            token = torch.tensor([[9 + step], [13 + step]])
            next_positions = batch["position_ids"].masked_fill(
                ~batch["attention_mask"].bool(), -1
            ).max(-1, keepdim=True).values + 1
            output = runtime.decode(mx.array(token.numpy()), output.cache)
            batch["input_ids"] = torch.cat((batch["input_ids"], token), dim=-1)
            batch["attention_mask"] = torch.cat(
                (batch["attention_mask"], torch.ones_like(token)), dim=-1
            )
            batch["position_ids"] = torch.cat((batch["position_ids"], next_positions), dim=-1)
            with torch.inference_mode():
                reference = model(**batch).logits[:, -1].numpy()
            np.testing.assert_allclose(_numpy(output.logits)[:, -1], reference,
                                       rtol=3e-5, atol=3e-6)
            np.testing.assert_array_equal(np.array(output.cache.position_ids), batch["position_ids"].numpy())
            assert output.cache.length == 5 + step


@pytest.mark.parametrize("site", ["embedding", "mlp.0.output"])
def test_perturbation_vjp_matches_torch_autograd(runtimes, site):
    model, _, _, direct, compiled = runtimes
    batch = _batch()
    perturbation = torch.zeros(2, 4, 24, requires_grad=True)
    module = model.model.embed_tokens if site == "embedding" else model.model.layers[0].mlp
    hook = module.register_forward_hook(lambda module, inputs, output: output + perturbation)
    cotangent = torch.linspace(-0.3, 0.5, 2 * 4 * 41).reshape(2, 4, 41)
    try:
        loss = (model(**batch).logits * cotangent).sum()
        expected = torch.autograd.grad(loss, perturbation)[0].numpy()
    finally:
        hook.remove()
    for runtime in (direct, compiled):
        def objective(delta):
            out = runtime(**_mlx_batch(batch), perturbations={site: delta})
            return mx.sum(out.logits * mx.array(cotangent.numpy()))
        actual = mx.grad(objective)(mx.zeros((2, 4, 24)))
        np.testing.assert_allclose(_numpy(actual), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("sharded", [False, True])
def test_native_safetensors_and_standalone_roundtrip(runtimes, tmp_path, monkeypatch, sharded):
    model, _, _, direct, _ = runtimes
    native_dir, exported_dir = tmp_path / "native", tmp_path / "standalone"
    model.save_pretrained(native_dir, safe_serialization=True,
                          max_shard_size="6KB" if sharded else "5GB")
    assert (native_dir / "model.safetensors.index.json").exists() == sharded
    batch = _mlx_batch(_batch("gapped"))
    reference = _numpy(direct(**batch).logits)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native MLX loading tried to allocate a torch model")

    with monkeypatch.context() as patch:
        patch.setattr(transformers.Qwen3ForCausalLM, "from_pretrained", forbidden)
        patch.setattr(torch, "load", forbidden)
        loaded = MLXQwen3.from_pretrained(native_dir)
        np.testing.assert_allclose(_numpy(loaded(**batch).logits), reference, rtol=0, atol=0)
        loaded.save(exported_dir)
        restored = MLXQwen3.load(exported_dir)
        np.testing.assert_allclose(_numpy(restored(**batch).logits), reference, rtol=0, atol=0)
        assert restored.config == loaded.config
        assert set(restored.parameters) == set(loaded.parameters)


def test_native_bfloat16_norm_rounds_before_nonunit_gain():
    generator = torch.Generator().manual_seed(11)
    x = (torch.randn(5, 24, generator=generator) * 3).bfloat16()
    gain = torch.linspace(0.37, 1.91, 24).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized.to(x.dtype) * gain).float().numpy()
    actual = _rms(mx.array(x.float().numpy()).astype(mx.bfloat16),
                  mx.array(gain.float().numpy()).astype(mx.bfloat16),
                  {"kind": "rms_norm", "epsilon": 1e-6,
                   "scale_parameterization": "direct", "cast_before_scale": True})
    np.testing.assert_array_equal(_numpy(actual), expected)


def test_live_hf_norm_epsilon_and_attention_scale_override_stale_config(runtimes):
    source = copy.deepcopy(runtimes[0])
    # These interventions intentionally do not update source.config.
    source.model.norm.variance_epsilon = 0.025
    for i, layer in enumerate(source.model.layers):
        layer.input_layernorm.variance_epsilon = 0.01 * (i + 1)
        layer.post_attention_layernorm.variance_epsilon = 0.035 * (i + 1)
        layer.self_attn.q_norm.variance_epsilon = 0.02 * (i + 1)
        layer.self_attn.k_norm.variance_epsilon = 0.02 * (i + 1)
        layer.self_attn.scaling *= 0.6 + i * 0.2
    assert source.config.rms_norm_eps == runtimes[0].config.rms_norm_eps
    converted = MLXQwen3.from_hf(source)
    batch = _batch("gapped")
    with torch.inference_mode():
        expected = source(**batch).logits.numpy()
        original = runtimes[0](**batch).logits.numpy()
    assert np.max(np.abs(expected - original)) > 1e-3
    np.testing.assert_allclose(_numpy(converted(**_mlx_batch(batch)).logits), expected,
                               rtol=2e-5, atol=3e-6)
    assert converted.config["terminal_norm"]["epsilon"] == 0.025
    for i, layer in enumerate(converted.layers):
        assert layer["attention"]["input_norm"]["epsilon"] == 0.01 * (i + 1)
        assert layer["attention"]["qk_norm"]["epsilon"] == 0.02 * (i + 1)
        assert layer["mlp"]["input_norm"]["epsilon"] == 0.035 * (i + 1)


def test_conversion_rejects_different_live_qk_norm_semantics(runtimes):
    source = copy.deepcopy(runtimes[0])
    source.model.layers[0].self_attn.q_norm.variance_epsilon = 0.05
    with pytest.raises(ValueError, match="matching Q/K normalization semantics"):
        MLXQwen3.from_hf(source)


def test_unknown_selective_capture_is_rejected(runtimes):
    direct = runtimes[3]
    with pytest.raises(ValueError, match="Unknown capture sites"):
        direct(**_mlx_batch(_batch()), capture=["attention.99.normalized"], return_logits=False)


@pytest.mark.parametrize("invalid_id", [-1, 41, 2**32 - 1])
def test_invalid_token_ids_do_not_wrap_or_gather_outside_vocabulary(runtimes, invalid_id):
    ids = mx.array([[invalid_id, 2]], dtype=mx.int64)
    for runtime in runtimes[3:]:
        with pytest.raises(ValueError, match="vocabulary"):
            runtime(ids)


def test_live_untied_head_overrides_stale_tied_config(runtimes):
    source = copy.deepcopy(runtimes[0])
    source.lm_head.weight = torch.nn.Parameter(source.lm_head.weight.detach().clone() * 1.7,
                                             requires_grad=False)
    assert source.config.tie_word_embeddings is True
    converted = MLXQwen3.from_hf(source)
    assert "lexical.head_weight" in converted.parameters
    batch = _batch()
    with torch.inference_mode():
        reference = source(**batch).logits.numpy()
    np.testing.assert_allclose(_numpy(converted(**_mlx_batch(batch)).logits), reference,
                               rtol=2e-5, atol=3e-6)


def test_compiled_identity_terminal_norm_needs_no_weight(runtimes):
    _, adapter, exact, _, _ = runtimes
    candidate = copy.deepcopy(exact)
    candidate.terminal_norm = torch.nn.Identity()
    converted = MLXQwen3.from_torch(candidate)
    assert "terminal.weight" not in converted.parameters
    batch = _batch()
    with torch.inference_mode():
        reference = candidate.execute(batch["input_ids"], adapter.prepare_sequence(batch)).logits.numpy()
    np.testing.assert_allclose(_numpy(converted(**_mlx_batch(batch)).logits), reference,
                               rtol=2e-5, atol=3e-6)


def test_chart_transport_conversion_and_cached_decode_match_torch(runtimes, tmp_path):
    from fisher_graph.gemma3_attention_generator import ReducedAttentionGeneratorCoefficients
    from fisher_graph.gemma3_cross_layer_chart_transport_executor import Gemma3FactorizedChartGeneratorLayer
    from fisher_graph.gemma3_joint_cross_block_chart_executor import (
        JointGraphCrossBlockTransportEdge, JointGraphTransportSpec,
    )
    from fisher_graph.qwen3_attention_generator import compile_qwen3_reduced_attention_generator
    from fisher_graph.qwen3_joint_compiler import assemble_qwen3_joint_executor
    from fisher_graph.qwen3_joint_executor import QWEN3_JOINT_CHART_ARM_ID

    model, adapter, exact, _, _ = runtimes
    attention, charts = [], []
    rng = torch.Generator().manual_seed(92)
    for i in range(2):
        g = exact.mutations[2 * i].compiled_operator
        input_basis = torch.eye(24)[:, :12]
        attention.append(compile_qwen3_reduced_attention_generator(
            g.config,
            ReducedAttentionGeneratorCoefficients(
                input_norm_weight=g.input_norm_weight, input_basis=input_basis,
                q_decoder_weight=g.q_weight @ input_basis,
                k_decoder_weight=g.k_weight @ input_basis,
                v_decoder_weight=g.v_weight @ input_basis,
                output_encoder_weight=torch.eye(32)[:10],
                output_decoder_weight=g.output_decoder_weight[:, :10],
                output_norm_weight=g.output_norm_weight,
                q_norm_weight=g.q_norm_weight, k_norm_weight=g.k_norm_weight,
                q_bias=g.q_bias, k_bias=g.k_bias, v_bias=g.v_bias, o_bias=g.o_bias,
            ),
        ))
        def random(*shape):
            return torch.randn(*shape, generator=rng) * 0.1
        charts.append(Gemma3FactorizedChartGeneratorLayer(
            affine_weight=random(16, 24), affine_bias=random(16),
            mode_decoder_weight=random(24, 16), chart_centers=random(2, 24),
            chart_bases=random(2, 24, 3), chart_distance_scales=torch.ones(2) * 2,
            local_generator_encoders=random(2, 4, 2), chart_mode_decoders=random(2, 2, 16),
            chart_temperature=0.8, model_fingerprint="a" * 64, layer_ordinal=i,
            fit_row_identity_sha256="b" * 64, compiled_prefix_catalog_sha256="c" * 64,
        ))
    edges = (
        JointGraphCrossBlockTransportEdge(
            spec=JointGraphTransportSpec("edge.a0.m0", "mutation.1", "mutation.2", 0, 1),
            transport_weight=torch.randn(10, 2, generator=rng) * 0.5,
        ),
        JointGraphCrossBlockTransportEdge(
            spec=JointGraphTransportSpec("edge.m0.a1", "mutation.2", "mutation.3", 1, 0),
            transport_weight=torch.randn(2, 12, generator=rng) * 0.5,
        ),
        JointGraphCrossBlockTransportEdge(
            spec=JointGraphTransportSpec("edge.a0.m1", "mutation.1", "mutation.4", 0, 0),
            transport_weight=torch.randn(10, 2, generator=rng) * 0.5,
        ),
    )
    candidate = assemble_qwen3_joint_executor(
        adapter, lexical=exact.lexical, attention_generators=attention,
        mlp_input_norms=[copy.deepcopy(x.post_attention_layernorm) for x in model.model.layers],
        mlp_chart_generators=charts, mlp_output_norms=[torch.nn.Identity(), torch.nn.Identity()],
        transports=edges, terminal_norm=exact.terminal_norm,
        arm_id=QWEN3_JOINT_CHART_ARM_ID, layer_count=2,
    )
    runtime = MLXQwen3.from_torch(candidate)
    runtime.save(tmp_path / "chart_graph")
    runtime = MLXQwen3.load(tmp_path / "chart_graph")
    batch = _batch("left")
    output = runtime.prefill(**_mlx_batch(batch))
    with torch.inference_mode():
        reference = candidate.execute(batch["input_ids"], adapter.prepare_sequence(batch)).logits
    valid = batch["attention_mask"].bool().numpy()
    np.testing.assert_allclose(_numpy(output.logits)[valid], reference.numpy()[valid], rtol=5e-5, atol=5e-6)
    ablated = runtime.with_parameters({f"transport.{i}.weight": mx.zeros_like(runtime.parameters[f"transport.{i}.weight"])
                                       for i in range(len(edges))})
    assert np.max(np.abs(_numpy(ablated(**_mlx_batch(batch)).logits)[valid] - reference.numpy()[valid])) > 1e-4
    for step in range(3):
        token = torch.tensor([[10 + step], [15 + step]])
        output = runtime.decode(mx.array(token.numpy()), output.cache)
        batch["input_ids"] = torch.cat((batch["input_ids"], token), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(token)), dim=1)
        batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            reference = candidate.execute(batch["input_ids"], adapter.prepare_sequence(batch)).logits
        np.testing.assert_allclose(_numpy(output.logits)[:, -1], reference[:, -1].numpy(), rtol=5e-5, atol=5e-6)


def test_mlx_to_torch_bridge_supports_partial_fit_decode_and_roundtrip(runtimes):
    from fisher_graph.gemma3_incremental_decode import JointGraphDecodeSession
    from fisher_graph.mlx_qwen3_fit import capture_runtime_rows, fit_chart_mlp, fit_reduced_attention

    _, adapter, _, direct, _ = runtimes
    batch = _batch("left")
    mlx_batch = _mlx_batch(batch)
    attn_rows = capture_runtime_rows(direct, [mlx_batch], ["attention.0.normalized", "attention.0.context"])
    partial = fit_reduced_attention(direct, 0, attn_rows, input_rank=12, output_rank=10).runtime
    mlp_rows = capture_runtime_rows(partial, [mlx_batch], ["mlp.0.normalized", "mlp.0.output"])
    partial = fit_chart_mlp(partial, 0, mlp_rows, chart_count=2, chart_rank=4, generator_rank=4).runtime

    torch_executor = partial.to_torch()
    seq = adapter.prepare_sequence(batch)
    with torch.inference_mode():
        torch_out = torch_executor.execute(batch["input_ids"], seq)
    mlx_out = partial(**mlx_batch)
    valid = batch["attention_mask"].bool().numpy()
    np.testing.assert_allclose(torch_out.logits.numpy()[valid], _numpy(mlx_out.logits)[valid], rtol=5e-5, atol=5e-6)

    # Verify JointGraphDecodeSession works on the converted partial-fit executor
    session, prefill_logits = JointGraphDecodeSession.prefill(
        torch_executor, batch["input_ids"], seq
    )
    np.testing.assert_allclose(prefill_logits[:, -1, :].numpy(), torch_out.logits[:, -1, :].numpy(), rtol=5e-5, atol=5e-6)
    next_tok = torch.tensor([[11], [14]], dtype=torch.long)
    step_logits = session.step(next_tok)
    mlx_decode = partial.decode(mx.array(next_tok.numpy()), mlx_out.cache)
    np.testing.assert_allclose(step_logits[:, 0, :].numpy(), _numpy(mlx_decode.logits)[:, 0, :], rtol=5e-4, atol=5e-5)

    # Round-trip back to MLXQwen3
    roundtripped = MLXQwen3.from_torch(torch_executor)
    np.testing.assert_allclose(
        _numpy(roundtripped(**mlx_batch).logits)[valid], _numpy(mlx_out.logits)[valid], rtol=5e-5, atol=5e-6
    )


@pytest.fixture(scope="module")
def qwen3_5_runtimes():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(915)
            config = transformers.Qwen3_5TextConfig(
                vocab_size=43, hidden_size=32, intermediate_size=64,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4, full_attention_interval=2,
                max_position_embeddings=128, tie_word_embeddings=True,
                attention_dropout=0.0,
            )
            config._attn_implementation = "eager"
            model = transformers.Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
            for module in model.modules():
                if "RMSNorm" in type(module).__name__:
                    module.weight.uniform_(-0.2, 0.35)
        direct = MLXQwen3.from_hf(model)
        yield model, direct
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("layout", ["plain", "left", "right", "gapped", "repeated"])
def test_qwen3_5_hybrid_forward_matches_hf_across_layouts(qwen3_5_runtimes, layout):
    model, direct = qwen3_5_runtimes
    batch = _batch(layout)
    with torch.inference_mode():
        reference = model(**batch).logits.numpy()
    valid = batch["attention_mask"].bool().numpy()
    output = direct(**_mlx_batch(batch))
    actual = _numpy(output.logits)
    np.testing.assert_allclose(actual[valid], reference[valid], rtol=2e-5, atol=3e-6)
    assert np.array_equal(actual[valid].argmax(-1), reference[valid].argmax(-1))
    # Layer 0 is deltanet (conv_state, rec_state); Layer 1 is full_attention (k, v)
    assert output.cache.key_values[0][0].shape == (2, 3, 2 * 16 + 32)
    assert output.cache.key_values[0][1].shape == (2, 4, 8, 8)
    assert output.cache.key_values[1][0].shape == (2, 2, 4, 8)


@pytest.mark.parametrize("layout", ["left", "right", "gapped"])
def test_qwen3_5_hybrid_cached_decode_matches_full_hf(qwen3_5_runtimes, layout):
    model, direct = qwen3_5_runtimes
    original = _batch(layout)
    batch = {k: v.clone() for k, v in original.items()}
    output = direct.prefill(**_mlx_batch(batch))
    for step in range(3):
        token = torch.tensor([[9 + step], [13 + step]])
        next_positions = batch["position_ids"].masked_fill(
            ~batch["attention_mask"].bool(), -1
        ).max(-1, keepdim=True).values + 1
        output = direct.decode(mx.array(token.numpy()), output.cache)
        batch["input_ids"] = torch.cat((batch["input_ids"], token), dim=-1)
        batch["attention_mask"] = torch.cat(
            (batch["attention_mask"], torch.ones_like(token)), dim=-1
        )
        batch["position_ids"] = torch.cat((batch["position_ids"], next_positions), dim=-1)
        with torch.inference_mode():
            reference = model(**batch).logits[:, -1].numpy()
        np.testing.assert_allclose(_numpy(output.logits)[:, -1], reference, rtol=3e-5, atol=3e-6)


def test_qwen3_5_safetensors_and_vjp_parity(qwen3_5_runtimes, tmp_path):
    model, direct = qwen3_5_runtimes
    native_dir, exported_dir = tmp_path / "qwen3_5_native", tmp_path / "qwen3_5_standalone"
    model.save_pretrained(native_dir, safe_serialization=True)
    loaded = MLXQwen3.from_pretrained(native_dir)
    batch = _batch()
    mlx_batch = _mlx_batch(batch)
    reference = _numpy(direct(**mlx_batch).logits)
    np.testing.assert_allclose(_numpy(loaded(**mlx_batch).logits), reference, rtol=0, atol=0)
    loaded.save(exported_dir)
    restored = MLXQwen3.load(exported_dir)
    np.testing.assert_allclose(_numpy(restored(**mlx_batch).logits), reference, rtol=0, atol=0)

    perturbation = torch.zeros(2, 4, 32, requires_grad=True)
    hook = model.model.layers[0].linear_attn.register_forward_hook(
        lambda module, inputs, output: output + perturbation
    )
    cotangent = torch.linspace(-0.2, 0.4, 2 * 4 * 43).reshape(2, 4, 43)
    try:
        loss = (model(**batch).logits * cotangent).sum()
        expected = torch.autograd.grad(loss, perturbation)[0].numpy()
    finally:
        hook.remove()

    def objective(delta):
        out = direct(**mlx_batch, perturbations={"deltanet.0.output": delta})
        return mx.sum(out.logits * mx.array(cotangent.numpy()))

    actual = mx.grad(objective)(mx.zeros((2, 4, 32)))
    np.testing.assert_allclose(_numpy(actual), expected, rtol=2e-4, atol=2e-5)

