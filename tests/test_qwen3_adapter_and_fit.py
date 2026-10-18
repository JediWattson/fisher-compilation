"""Qwen adapter gradients, live semantics, and reduced-fit execution checks."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from fisher_graph.adapters.qwen3 import Qwen3CausalLMAdapter
from fisher_graph.qwen3_attention_generator import (
    Qwen3AttentionGenerator,
    build_qwen3_exact_attention_generator,
)
from fisher_graph.qwen3_model_fit import (
    fit_qwen3_pullback_ewc_reduced_attention_generator,
)


def _model(*, implementation="eager", bias=False):
    torch.manual_seed(29)
    config = transformers.Qwen3Config(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        attention_bias=bias,
    )
    config._attn_implementation = implementation
    model = transformers.Qwen3ForCausalLM(config).eval()
    return model, Qwen3CausalLMAdapter(model)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_projection_captures_retain_the_native_score_gradients(implementation):
    model, adapter = _model(implementation=implementation)
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    attention = model.model.layers[0].self_attn
    native_projections = {}
    handles = []
    for name in ("q", "k", "v"):
        def capture(_module, _args, output, name=name):
            output.retain_grad()
            native_projections[name] = output
        handles.append(getattr(attention, f"{name}_proj").register_forward_hook(capture))
    try:
        native = model(input_ids=ids, labels=ids, use_cache=False)
        native.loss.backward()
        native_gradients = {name: value.grad.detach().clone() for name, value in native_projections.items()}
    finally:
        for handle in handles:
            handle.remove()
    model.zero_grad(set_to_none=True)
    site_names = {"q": "query_projection", "k": "key_projection", "v": "value_projection"}
    sites = [f"layer.0.attention.{site}" for site in site_names.values()]
    run = adapter.forward({"input_ids": ids, "labels": ids}, capture_sites=sites, retain_gradients=True)
    torch.testing.assert_close(run.logits, native.logits, rtol=0, atol=0)
    run.raw_output.loss.backward()
    for name, site in site_names.items():
        captured = run.activations[f"layer.0.attention.{site}"]
        assert captured.grad is not None
        assert torch.count_nonzero(captured.grad) > 0
        torch.testing.assert_close(captured.grad, native_gradients[name].reshape_as(captured), rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["attention", "mlp", "qk", "terminal"])
def test_live_norm_epsilon_changes_fingerprints_and_specs(norm):
    model, adapter = _model()
    before_model = adapter.model_fingerprint()
    before_execution = adapter.execution_fingerprint()
    layer = model.model.layers[0]
    modules = {
        "attention": [layer.input_layernorm],
        "mlp": [layer.post_attention_layernorm],
        "qk": [layer.self_attn.q_norm, layer.self_attn.k_norm],
        "terminal": [model.model.norm],
    }[norm]
    for module in modules:
        module.variance_epsilon = 0.5
    assert adapter.execution_fingerprint() != before_execution
    assert adapter.model_fingerprint() != before_model
    semantics = adapter.layer("layer.0").transformer
    if norm == "attention":
        assert semantics.attention_input_norm.epsilon == 0.5
    elif norm == "mlp":
        assert semantics.feed_forward_input_norm.epsilon == 0.5
    elif norm == "qk":
        assert semantics.qk_norm.epsilon == 0.5

    # A compiler queried after the mutation must use the current epsilon.
    generator = build_qwen3_exact_attention_generator(adapter, "layer.0")
    ids = torch.tensor([[2, 3, 4, 5]])
    native = adapter.forward({"input_ids": ids}, capture_sites=["layer.0.input", "layer.0.attention.delta"])
    output = generator(native.activations["layer.0.input"], native.sequence)
    torch.testing.assert_close(output.residual_delta, native.activations["layer.0.attention.delta"], atol=2e-7, rtol=2e-6)


def test_unrepresentable_qk_norm_epsilon_drift_is_rejected():
    model, adapter = _model()
    model.model.layers[0].self_attn.q_norm.variance_epsilon = 0.5
    with pytest.raises(ValueError, match="Q/K norm epsilon values differ"):
        build_qwen3_exact_attention_generator(adapter, "layer.0")


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", ["left", "right"])
def test_segment_chain_matches_native_valid_queries_with_padding_and_positions(implementation, padding):
    _source, adapter = _model(implementation=implementation)
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    mask = torch.tensor([[0, 0, 1, 1, 1] if padding == "left" else [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    inputs = {"input_ids": ids, "attention_mask": mask, "position_ids": torch.tensor([[3, 4, 8, 10, 13]])}
    native = adapter.forward(inputs, capture_sites=[f"layer.{i}.mlp.normalized_input" for i in range(2)])
    sequence = native.sequence
    valid = sequence.query_valid_mask
    hidden = adapter.embed(inputs, sequence).hidden_states
    for i, segment in enumerate(adapter.segments):
        prefix = adapter.run_attention_prefix(segment, hidden, sequence)
        torch.testing.assert_close(prefix.normalized_mlp_input[valid], native.activations[f"layer.{i}.mlp.normalized_input"][valid], rtol=0, atol=0)
        hidden = adapter.run_segment(segment, hidden, sequence).hidden_states
    torch.testing.assert_close(adapter.project_logits(hidden, sequence)[valid], native.logits[valid], rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_full_rank_reduced_fit_preserves_qwen_operator_and_identity_delta(bias):
    model, adapter = _model(bias=bias)
    # Make wrong delta normalization conspicuous even in the small fixture.
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight.mul_(12)
    exact = build_qwen3_exact_attention_generator(adapter, "layer.0")
    ids = torch.randint(1, 71, (4, 16))
    sequence = adapter.prepare_sequence({"input_ids": ids})
    train = exact(torch.randn(4, 16, 32), sequence)
    fitted = fit_qwen3_pullback_ewc_reduced_attention_generator(
        exact,
        train.normalized_input.reshape(-1, 32),
        train.output_coordinates.values.reshape(-1, 32),
        op_outputs=train.operator_output.reshape(-1, 32),
        input_rank=32,
        output_rank=32,
    )
    assert isinstance(fitted, Qwen3AttentionGenerator)
    assert fitted.generator_kind == "reduced_coordinates"
    assert fitted.config.output_norm.kind == "identity"
    assert fitted.source_checkpoint_coefficients_retained is False
    test_sequence = adapter.prepare_sequence({"input_ids": ids[:2, :7]})
    heldout = torch.randn(2, 7, 32)
    expected, actual = exact(heldout, test_sequence), fitted(heldout, test_sequence)
    torch.testing.assert_close(actual.operator_output, expected.operator_output, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.residual_delta, expected.residual_delta, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.residual_delta, actual.operator_output, rtol=0, atol=0)


def test_reduced_fit_validates_weights_and_executes_compressed_coordinates():
    _model_source, adapter = _model()
    exact = build_qwen3_exact_attention_generator(adapter, "layer.0")
    sequence = adapter.prepare_sequence({"input_ids": torch.ones(2, 8, dtype=torch.long)})
    hidden = torch.randn(2, 8, 32)
    train = exact(hidden, sequence)
    x, c = train.normalized_input.reshape(-1, 32), train.output_coordinates.values.reshape(-1, 32)
    with pytest.raises(ValueError, match="row_weights"):
        fit_qwen3_pullback_ewc_reduced_attention_generator(exact, x, c, input_rank=16, output_rank=8, row_weights=torch.zeros(16))
    fitted = fit_qwen3_pullback_ewc_reduced_attention_generator(exact, x, c, input_rank=16, output_rank=8, row_weights=torch.linspace(0, 2, 16))
    # Qwen has no delta RMSNorm: passing the optional operator capture must
    # not introduce Gemma's inverse-output-RMS context weighting.
    with_output = fit_qwen3_pullback_ewc_reduced_attention_generator(
        exact, x, c, input_rank=16, output_rank=8,
        row_weights=torch.linspace(0, 2, 16),
        op_outputs=train.operator_output.reshape(-1, 32),
    )
    for name, parameter in fitted.named_parameters():
        torch.testing.assert_close(parameter, dict(with_output.named_parameters())[name], rtol=0, atol=0)
    assert fitted.input_coordinate_rank == 16
    assert fitted.output_coordinate_rank == 8
    output = fitted(hidden, sequence)
    assert bool(torch.isfinite(output.residual_delta).all())
    torch.testing.assert_close(output.residual_delta, output.operator_output, rtol=0, atol=0)
