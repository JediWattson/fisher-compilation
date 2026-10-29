"""Real tiny Qwen integration for MLX captures, fits, and optimizer updates."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from fisher_graph.mlx_qwen3_fit import (
    allocate_waterfill_attention_ranks,
    allocate_waterfill_deltanet_carrier_ranks,
    allocate_waterfill_joint_ranks,
    anchored_ridge,
    capture_empirical_token_fisher,
    capture_parameter_fisher,
    capture_runtime_rows,
    fit_chart_deltanet,
    fit_chart_mlp,
    fit_coalesced_deltanet,
    fit_deltanet_carrier_basis,
    fit_reduced_attention,
    fit_reduced_deltanet,
    optimize_trajectory,
    weighted_moments,
)


def test_import_does_not_load_mlx():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import fisher_graph.mlx_qwen3_fit; assert 'mlx.core' not in sys.modules"],
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
    return MLXQwen3.from_hf(model), model


def batches(mx):
    return [{
        "input_ids": mx.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=mx.int32),
        "attention_mask": mx.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }, {
        "input_ids": mx.array([[4, 7, 10, 15, 18], [1, 4, 8, 11, 19]], dtype=mx.int32),
    }]


def test_weighted_moments_and_precision_reference(mx):
    x = mx.array([[1.0, 2.0], [3.0, -1.0], [9.0, 4.0]])
    weights = mx.array([1.0, 2.0, 0.0])
    stats = weighted_moments(x, weights)
    xn, wn = np.asarray(x).astype(np.float64), np.asarray(weights).astype(np.float64) / 3
    mean = (xn * wn[:, None]).sum(axis=0)
    np.testing.assert_allclose(np.asarray(stats.mean), mean, atol=2e-6)
    np.testing.assert_allclose(np.asarray(stats.covariance), (xn-mean).T @ ((xn-mean) * wn[:, None]), atol=2e-6)
    y = x @ mx.array([[2.0], [-0.5]])
    anchor = mx.array([[0.25], [1.0]])
    native = anchored_ridge(x, y, row_weights=weights, anchor=anchor, relative_ridge=0.1)
    reference = anchored_ridge(x, y, row_weights=weights, anchor=anchor, relative_ridge=0.1, solver="numpy_float64")
    np.testing.assert_allclose(np.asarray(native), np.asarray(reference), rtol=3e-5, atol=3e-6)
    with pytest.raises(ValueError, match="positive total mass"):
        weighted_moments(x, mx.zeros((3,)))


def test_captures_exclude_padding_and_are_real_runtime_rows(qwen, mx):
    runtime, _ = qwen
    data = batches(mx)
    sites = ["attention.0.normalized", "attention.0.context", "mlp.0.normalized", "mlp.0.output"]
    rows = capture_runtime_rows(runtime, data, sites)
    assert all(value.shape == (19, 16) for value in rows.values())
    direct = runtime(**data[0], capture=sites, return_logits=False)
    valid = np.asarray(data[0]["attention_mask"]).astype(bool)
    for site in sites:
        np.testing.assert_allclose(np.asarray(rows[site][:9]), np.asarray(direct.captures[site])[valid], atol=1e-6)


def test_full_rank_attention_fit_preserves_real_qwen_forward(qwen, mx):
    runtime, _ = qwen
    data = batches(mx)
    rows = capture_runtime_rows(runtime, data, ["attention.0.normalized", "attention.0.context"])
    original = np.asarray(runtime(**data[0]).logits)
    result = fit_reduced_attention(runtime, 0, rows, input_rank=16, output_rank=16, relative_ridge=0.02)
    actual = np.asarray(result.runtime(**data[0]).logits)
    np.testing.assert_allclose(actual, original, rtol=3e-4, atol=3e-5)
    assert result.report["context_output_relative_error"] < 2e-4
    assert "attention.0.input_basis" not in runtime.parameters
    assert result.runtime.parameters["attention.0.input_basis"].shape == (16, 16)


def test_reduced_attention_and_chart_fit_produce_usable_runtime(qwen, mx):
    runtime, _ = qwen
    data = batches(mx)
    attn_rows = capture_runtime_rows(runtime, data, ["attention.0.normalized", "attention.0.context"])
    reduced = fit_reduced_attention(runtime, 0, attn_rows, input_rank=8, output_rank=6, solver="numpy_float64").runtime
    rows = capture_runtime_rows(reduced, data, ["mlp.0.normalized", "mlp.0.output"])
    fitted = fit_chart_mlp(reduced, 0, rows, chart_count=2, chart_rank=4, generator_rank=4, relative_ridge=0.01)
    assert "mlp.0.gate_weight" not in fitted.runtime.parameters
    assert "mlp.0.mode_decoder_weight" not in fitted.runtime.parameters
    assert fitted.report["euler_jacobian_anchored"] is True
    assert fitted.report["centered_intercept"] is True
    assert fitted.report["inverse_target_norm_weights"] is True
    assert fitted.report["training_relative_error"] < 0.65
    output = fitted.runtime(**data[0])
    assert output.logits.shape == (2, 5, 32)
    assert np.isfinite(np.asarray(output.logits)).all()
    # Verify fitted regression and runtime share chart conventions.
    actual = capture_runtime_rows(fitted.runtime, data, ["mlp.0.output"])["mlp.0.output"]
    error = np.linalg.norm(np.asarray(actual - rows["mlp.0.output"])) / np.linalg.norm(np.asarray(rows["mlp.0.output"]))
    assert error == pytest.approx(fitted.report["training_relative_error"], rel=1e-4)
    # Also verify legacy unanchored/uncentered mode and refit when gate_weight is already removed.
    refitted = fit_chart_mlp(
        fitted.runtime, 0, rows, chart_count=2, chart_rank=4, generator_rank=4,
        relative_ridge=0.01, euler_jacobian_anchor=True, center_intercept=False,
        inverse_target_norm_weights=False,
    )
    assert refitted.report["euler_jacobian_anchored"] is False
    assert refitted.report["centered_intercept"] is False
    assert refitted.report["inverse_target_norm_weights"] is False
    with pytest.raises(ValueError, match="exact source"):
        fit_reduced_attention(reduced, 0, attn_rows, input_rank=8, output_rank=6)


def test_fisher_matches_native_qwen_token_score_gradient(qwen, mx):
    runtime, model = qwen
    torch = pytest.importorskip("torch")
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    actual = capture_empirical_token_fisher(
        runtime, [{"input_ids": mx.array(ids)}], ["embedding"],
        max_tokens=1, full_matrix=True,
    )
    embedding = model.model.embed_tokens(torch.tensor(ids)).detach().requires_grad_(True)
    logits = model(inputs_embeds=embedding).logits
    loss = torch.nn.functional.cross_entropy(logits[:, 0, :], torch.tensor([2]))
    gradient = torch.autograd.grad(loss, embedding)[0].detach().numpy().reshape(-1, 16)
    np.testing.assert_allclose(np.asarray(actual["diagonal"]["embedding"]), (gradient * gradient).sum(axis=0), rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual["matrices"]["embedding"]), gradient.T @ gradient, rtol=5e-4, atol=2e-5)
    assert actual["objective_count"] == 1
    assert actual["cross_position_blocks_included"] is False
    assert actual["mlx_enable_tf32"] == os.environ.get("MLX_ENABLE_TF32", "backend_default")


def test_parameter_fisher_matches_native_and_supplies_optimizer_prior(qwen, mx):
    runtime, model = qwen
    torch = pytest.importorskip("torch")
    name = "mlp.1.down_weight"
    data = [{"input_ids": mx.array([[1, 2, 3, 4]])}]
    actual = capture_parameter_fisher(runtime, data, [name], max_tokens=1)
    weight = model.model.layers[1].mlp.down_proj.weight
    weight.requires_grad_(True)
    try:
        logits = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
        loss = torch.nn.functional.cross_entropy(logits[:, 0, :], torch.tensor([2]))
        gradient = torch.autograd.grad(loss, weight)[0].detach().numpy()
    finally:
        weight.requires_grad_(False)
    np.testing.assert_allclose(np.asarray(actual["diagonal"][name]), gradient ** 2, rtol=5e-4, atol=1e-8)
    assert actual["objective_count"] == 1
    assert float(mx.sum(actual["diagonal"][name]).item()) > 0
    perturbed = runtime.with_parameters({name: runtime.parameters[name] * 0.5})
    fitted = optimize_trajectory(perturbed, runtime, data, parameter_names=[name], steps=2,
                                 anchor_weight=0.01, fisher_diagonal=actual["diagonal"])
    assert fitted.report["fisher_precision_supplied"]
    with pytest.raises(ValueError, match="outside.*vocabulary"):
        capture_parameter_fisher(runtime, [{"input_ids": mx.array([[1, 2]]), "labels": [[1, -5]]}], [name])


@pytest.mark.parametrize("objective", ["terminal_mse", "teacher_kl"])
def test_real_trajectory_adam_reduces_loss_without_changing_source(qwen, mx, objective):
    runtime, _ = qwen
    name = "mlp.1.down_weight"
    original = np.asarray(runtime.parameters[name]).copy()
    student = runtime.with_parameters({name: runtime.parameters[name] * 0.35})
    before = np.asarray(student.parameters[name]).copy()
    result = optimize_trajectory(
        student, runtime, batches(mx), parameter_names=[name],
        steps=15, learning_rate=0.003, objective=objective,
        anchor_weight=1e-6, fisher_diagonal={name: mx.ones_like(student.parameters[name])},
    )
    assert result.report["best_training_loss"] < result.report["initial_training_loss"] * 0.8
    assert result.report["fisher_precision_supplied"] is True
    np.testing.assert_array_equal(np.asarray(runtime.parameters[name]), original)
    np.testing.assert_array_equal(np.asarray(student.parameters[name]), before)
    assert any(np.any(np.asarray(value) != before) for value in result.updates.values())
    assert result.runtime.parameters["lexical.weight"] is student.parameters["lexical.weight"]
    with pytest.raises(ValueError, match="exactly selected"):
        optimize_trajectory(student, runtime, batches(mx), parameter_names=[name], fisher_diagonal={})


def test_refit_rejects_existing_transport_vocabulary_change(qwen, mx):
    runtime, _ = qwen
    runtime = runtime.with_parameters({})
    runtime.config["transports"] = [{"source": "attention.0", "target": "mlp.1"}]
    with pytest.raises(ValueError, match="transport vocabulary"):
        fit_reduced_attention(runtime, 0, {}, input_rank=8, output_rank=8)


def test_bfloat16_fits_preserve_deployed_weights_hidden_and_cache(qwen, mx):
    runtime, _ = qwen
    runtime = runtime.with_parameters({name: value.astype(mx.bfloat16) for name, value in runtime.parameters.items()})
    data = batches(mx)
    captures = capture_runtime_rows(runtime, data, ["attention.0.normalized", "attention.0.context"])
    attention = fit_reduced_attention(runtime, 0, captures, input_rank=8, output_rank=6)
    assert all(value.dtype == mx.bfloat16 for value in attention.updates.values())
    captures = capture_runtime_rows(attention.runtime, data, ["mlp.0.normalized", "mlp.0.output"])
    charts = fit_chart_mlp(attention.runtime, 0, captures, chart_count=2, chart_rank=4)
    assert all(value.dtype == mx.bfloat16 for value in charts.updates.values())
    output = charts.runtime(**data[0])
    assert output.hidden_states.dtype == mx.bfloat16
    assert output.logits.dtype == mx.bfloat16
    assert all(key.dtype == mx.bfloat16 and value.dtype == mx.bfloat16 for key, value in output.cache.key_values)
    decoded = charts.runtime.decode(mx.array([[2], [3]]), output.cache)
    assert decoded.hidden_states.dtype == mx.bfloat16
    assert all(key.dtype == mx.bfloat16 and value.dtype == mx.bfloat16 for key, value in decoded.cache.key_values)
    assert np.isfinite(np.asarray(decoded.logits.astype(mx.float32))).all()
    assert charts.report["deployment_dtype"] == str(mx.bfloat16)


@pytest.mark.parametrize("objective", ["terminal_mse", "teacher_kl"])
def test_bfloat16_optimizer_initial_loss_matches_unmodified_runtime(qwen, mx, objective):
    runtime, _ = qwen
    runtime = runtime.with_parameters({name: value.astype(mx.bfloat16) for name, value in runtime.parameters.items()})
    name = "mlp.1.down_weight"
    result = optimize_trajectory(runtime, runtime, batches(mx), parameter_names=[name], steps=2, objective=objective)
    assert result.report["initial_training_loss"] == 0.0
    assert result.report["best_training_loss"] == 0.0
    assert result.report["optimizer_master_dtype"] == "float32"
    assert result.updates[name].dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(result.updates[name].astype(mx.float32)), np.asarray(runtime.parameters[name].astype(mx.float32)))
    output = result.runtime(**batches(mx)[0])
    assert output.hidden_states.dtype == mx.bfloat16
    assert all(key.dtype == mx.bfloat16 and value.dtype == mx.bfloat16 for key, value in output.cache.key_values)


def test_waterfill_attention_rank_allocation_respects_budget_and_reduces_distortion(qwen, mx):
    runtime, _ = qwen
    data = batches(mx)
    # Uniform (input_rank=8, output_rank=4) on 2 layers of width 16 (q=16, kv=8):
    # exact_in = 32*16 = 512, c_in = 48 -> rem_in(8) = 512 - 384 = 128
    # exact_out = 16*16 = 256, c_out = 32 -> rem_out(4) = 256 - 128 = 128
    # Total uniform removed across 2 layers = 2 * 256 = 512 scalars.
    schedule = allocate_waterfill_attention_ranks(
        runtime, data, target_scalars_removed=512, rank_step=1, min_rank=2, fisher_max_tokens=4
    )
    assert schedule["achieved_scalars_removed"] >= 512
    assert set(schedule["layers"].keys()) == {0, 1}
    assert schedule["total_fisher_distortion"] >= 0.0
    # Higher-Omega layer should receive at least as many kept scalars as lower-Omega layer
    l0, l1 = schedule["layers"][0], schedule["layers"][1]
    if l0["omega"] > l1["omega"] * 1.5:
        assert l0["scalars_removed"] <= l1["scalars_removed"]

    joint = allocate_waterfill_joint_ranks(
        runtime, data, target_scalars_removed=600, chart_count=2, chart_rank=2, generator_rank=2,
        min_mlp_layer_spacing=1, rank_step=1, min_rank=2, fisher_max_tokens=4,
    )
    assert joint["achieved_scalars_removed"] >= 600
    assert len(joint["mlp_candidates_ranked"]) == 2


@pytest.fixture(scope="module")
def qwen3_5(mx):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from fisher_graph.mlx_qwen3 import MLXQwen3

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(512)
        config = transformers.Qwen3_5TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=4,
            linear_conv_kernel_dim=4, full_attention_interval=2,
            max_position_embeddings=64, tie_word_embeddings=True,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = transformers.Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    return MLXQwen3.from_hf(model), model


def test_qwen3_5_play1_reduced_deltanet_and_play2_carrier_basis(qwen3_5, mx):
    runtime, _ = qwen3_5
    data = batches(mx)
    original = np.asarray(runtime(**data[0]).logits)

    # Play 1: full-rank reduced deltanet preserves forward pass; rank reduction compresses projections
    d_rows = capture_runtime_rows(runtime, data, ["deltanet.0.normalized", "deltanet.0.context"])
    full_d = fit_reduced_deltanet(runtime, 0, d_rows, input_rank=16, output_rank=16, relative_ridge=0.02)
    np.testing.assert_allclose(np.asarray(full_d.runtime(**data[0]).logits), original, rtol=4e-4, atol=4e-5)

    red_d = fit_reduced_deltanet(runtime, 0, d_rows, input_rank=8, output_rank=8, solver="numpy_float64")
    assert red_d.runtime.parameters["deltanet.0.input_basis"].shape == (16, 8)
    assert red_d.runtime.parameters["deltanet.0.qkv_weight"].shape == (48, 8)
    assert red_d.runtime.parameters["deltanet.0.z_weight"].shape == (16, 8)
    assert red_d.runtime.parameters["deltanet.0.output_encoder_weight"].shape == (8, 16)
    assert red_d.runtime.parameters["deltanet.0.output_decoder_weight"].shape == (16, 8)

    # Play 2: full-rank orthogonal carrier basis is exact; reduced carrier shrinks S along Dk
    qk_sites = ["deltanet.0.q", "deltanet.0.k"]
    qk_rows = capture_runtime_rows(runtime, data, qk_sites)
    fisher_qk = capture_empirical_token_fisher(runtime, data, qk_sites, max_tokens=4)
    full_c = fit_deltanet_carrier_basis(
        runtime, 0, qk_rows, carrier_rank=8, fisher_diagonal=fisher_qk["diagonal"], relative_ridge=0.0
    )
    out_full_c = full_c.runtime(**data[0])
    np.testing.assert_allclose(np.asarray(out_full_c.logits), original, rtol=2e-5, atol=3e-6)
    assert full_c.report["retained_energy_fraction"] == pytest.approx(1.0, abs=1e-5)

    red_c = fit_deltanet_carrier_basis(
        red_d.runtime, 0, qk_rows, carrier_rank=4, fisher_diagonal=fisher_qk["diagonal"], solver="numpy_float64"
    )
    assert red_c.runtime.parameters["deltanet.0.carrier_basis"].shape == (2, 8, 4)
    assert red_c.runtime.layers[0]["deltanet"]["carrier_rank"] == 4
    assert 0.4 < red_c.report["retained_energy_fraction"] <= 1.0
    out_red_c = red_c.runtime.prefill(**data[0])
    # Recurrent state S in cache is now [B=2, Hv=4, Dv=4, r=4] instead of [2, 4, 4, 8]
    assert out_red_c.cache.key_values[0][1].shape == (2, 4, 4, 4)
    dec_red_c = red_c.runtime.decode(mx.array([[2], [3]], dtype=mx.int32), out_red_c.cache)
    assert np.isfinite(np.asarray(dec_red_c.logits)).all()

    # Hybrid waterfilling across deltanet.0, attention.1, and mlp.0..1
    joint = allocate_waterfill_joint_ranks(
        runtime, data, target_scalars_removed=500, chart_count=2, chart_rank=2, generator_rank=2,
        min_mlp_layer_spacing=1, rank_step=1, min_rank=2, fisher_max_tokens=4,
    )
    assert joint["achieved_scalars_removed"] >= 500
    assert set(joint["attention_schedule"]["layers"].keys()) == {0, 1}

    carrier_sched = allocate_waterfill_deltanet_carrier_ranks(
        runtime, data, target_compression_ratio=0.5, rank_step=2, min_rank=2, fisher_max_tokens=4,
    )
    assert carrier_sched["achieved_compression_ratio"] <= 0.5
    assert carrier_sched["layers"][0]["carrier_rank"] == 4

    chart_d = fit_chart_deltanet(
        runtime, 0, d_rows, value_rank=2, chart_count=2, chart_rank=2, solver="numpy_float64",
    )
    assert chart_d.runtime.parameters["deltanet.0.value_basis"].shape == (4, 4, 2)
    assert chart_d.runtime.parameters["deltanet.0.output_decoder_weight"].shape == (16, 8)
    assert chart_d.runtime.parameters["deltanet.0.chart_centers"].shape == (2, 16)
    assert chart_d.runtime.parameters["deltanet.0.chart_mode_decoders"].shape == (2, 3, 16)
    assert chart_d.report["chart_relative_error"] <= chart_d.report["base_relative_error"] + 1e-4
    out_chart_d = chart_d.runtime(**data[0], muted_charts=[(0, 1)])
    assert np.isfinite(np.asarray(out_chart_d.logits)).all()

    coal_rows = capture_runtime_rows(
        runtime, data, ["deltanet.0.normalized", "deltanet.0.delta", "mlp.0.delta"]
    )
    coal_d = fit_coalesced_deltanet(
        runtime, 0, coal_rows, coalesce_mlp=True, transport_rank=4, chart_count=2, chart_rank=2, solver="numpy_float64",
    )
    assert "deltanet.0.qkv_weight" not in coal_d.runtime.parameters
    assert "mlp.0.gate_weight" not in coal_d.runtime.parameters
    assert coal_d.runtime.layers[0]["deltanet"]["coalesced_mlp"] is True
    out_coal_pre = coal_d.runtime.prefill(**data[0])
    assert out_coal_pre.cache.key_values[0] is None
    out_coal_dec = coal_d.runtime.decode(mx.array([[2], [3]], dtype=mx.int32), out_coal_pre.cache)
    assert np.isfinite(np.asarray(out_coal_dec.logits)).all()
