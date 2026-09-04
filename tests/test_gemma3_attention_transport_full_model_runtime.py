from dataclasses import replace

import torch

from fisher_graph.adapters import (
    AttentionSpec,
    FeedForwardSpec,
    LayerSpec,
    NormalizationSpec,
    ResidualStageSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
    TransformerLayerSemantics,
)
from fisher_graph.gemma3_attention_generator import (
    compile_exact_attention_generator_from_structured_executor,
)
from fisher_graph.gemma3_attention_transport_full_model_cross_layer import (
    ROUTER_FEATURE_NAMES,
    FullModelCrossLayerAttentionTransport,
    SharedCrossLayerGeneratorBank,
    SharedTop2CrossLayerRouter,
)
from fisher_graph.gemma3_attention_transport_full_model_runtime import (
    FrozenCrossLayerTransportRuntime,
    Gemma3NativeQKSharedVOAttentionOperator,
    NativeHeadCompositeRuntime,
    native_head_composites_from_exact_attention,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


def _layer_spec() -> LayerSpec:
    width = 8
    head_dimension = 2
    norm = NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1.0e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    attention_stage = ResidualStageSpec(
        id="layer.0.attention",
        kind="attention",
        input_site="layer.0.input",
        normalized_input_site="layer.0.attention.normalized_input",
        operator_output_site="layer.0.attention.operator_output",
        delta_site="layer.0.attention.delta",
        output_site="layer.0.post_attention",
    )
    feed_forward_stage = ResidualStageSpec(
        id="layer.0.feed_forward",
        kind="feed_forward",
        input_site="layer.0.post_attention",
        normalized_input_site="layer.0.mlp.normalized_input",
        operator_output_site="layer.0.mlp.operator_output",
        delta_site="layer.0.mlp.delta",
        output_site="layer.0.output",
    )
    return LayerSpec(
        id="layer.0",
        ordinal=0,
        input_site="layer.0.input",
        output_site="layer.0.output",
        residual_width=width,
        kind="gemma3_decoder",
        attention=AttentionSpec(
            kind="sliding_causal",
            query_heads=4,
            key_value_heads=2,
            head_dimension=head_dimension,
            query_scale=0.5,
            qk_norm=True,
            window_size=3,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        transformer=TransformerLayerSemantics(
            residual_layout="sequential_attention_then_feed_forward_residual",
            attention_input_norm=norm,
            attention_output_norm=norm,
            qk_norm=replace(norm, width=head_dimension),
            attention_projection_bias=False,
            attention_dropout=0.0,
            attention_logit_softcap=7.0,
            feed_forward_input_norm=norm,
            feed_forward_output_norm=norm,
            feed_forward=FeedForwardSpec(
                kind="gated_multiplicative",
                intermediate_width=16,
                activation="gelu_pytorch_tanh",
                projection_bias=False,
            ),
            stages=(attention_stage, feed_forward_stage),
        ),
    )


def _exact_attention():
    torch.manual_seed(9_004)
    structured = StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(_layer_spec()),
        dtype=torch.float64,
    ).eval()
    return compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="mutation.1",
    )


def _sequence(batch: int = 2, length: int = 5) -> SequenceContext:
    mask = torch.ones(batch, length, dtype=torch.bool)
    positions = torch.arange(length, dtype=torch.long)[None].expand(batch, -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )


def _artifact() -> FullModelCrossLayerAttentionTransport:
    torch.manual_seed(77)
    generators = 3
    hidden = 8
    rank = 2
    heads = 4
    left = torch.stack(
        tuple(torch.linalg.qr(torch.randn(hidden, rank, dtype=torch.float64)).Q for _ in range(generators))
    )
    right = torch.stack(
        tuple(torch.linalg.qr(torch.randn(hidden, rank, dtype=torch.float64)).Q for _ in range(generators))
    )
    bank = SharedCrossLayerGeneratorBank(
        left_factors=left,
        base_singular_diagonal=torch.full(
            (generators, rank), 0.5, dtype=torch.float64
        ),
        depth_singular_slopes=torch.full(
            (generators, rank), 0.05, dtype=torch.float64
        ),
        site_singular_offsets=torch.zeros(
            6, generators, rank, dtype=torch.float64
        ),
        right_factors=right,
        site_head_coefficients=torch.randn(
            6, heads, generators, dtype=torch.float64
        ),
    )
    feature_count = len(ROUTER_FEATURE_NAMES)
    router = SharedTop2CrossLayerRouter(
        generator_bank_sha256=bank.artifact_sha256,
        selection_weight=torch.zeros(
            generators, feature_count, dtype=torch.float64
        ),
        selection_bias=torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64),
        amplitude_weight=torch.zeros(
            generators, feature_count, dtype=torch.float64
        ),
        amplitude_bias=torch.ones(generators, dtype=torch.float64),
    )
    return FullModelCrossLayerAttentionTransport(bank=bank, router=router)


def test_native_head_composite_wrapper_is_exact_vo_seam_control() -> None:
    exact = _exact_attention()
    composite = NativeHeadCompositeRuntime(
        "mutation.1",
        native_head_composites_from_exact_attention(exact),
    )
    wrapper = Gemma3NativeQKSharedVOAttentionOperator(exact, backend=composite)
    hidden = torch.randn(2, 5, 8, dtype=torch.float64)
    sequence = _sequence()
    incoming = torch.zeros(2, 5, 1, 1, dtype=torch.float64)

    teacher = exact.execute_with_incoming(
        hidden,
        sequence,
        torch.zeros(2, 5, 1, 8, dtype=torch.float64),
    )
    candidate = wrapper.execute_with_incoming(hidden, sequence, incoming)

    torch.testing.assert_close(
        candidate.operator_output,
        teacher.operator_output,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        candidate.residual_delta,
        teacher.residual_delta,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    manifest = wrapper.architecture_manifest()
    assert manifest["native_qk_retained"] is True
    assert manifest["native_output_rmsnorm_retained"] is True
    assert manifest["contains_native_value_or_output_weights"] is False
    assert wrapper.native_v_projection_call_count == 0
    assert wrapper.native_o_projection_call_count == 0
    assert not any(
        name.endswith("v_weight")
        or name.endswith("o_weight")
        or "output_decoder_weight" in name
        for name, _ in wrapper.named_parameters()
    )


def test_shared_runtime_applies_zero_and_fixed_permutation_controls() -> None:
    runtime = FrozenCrossLayerTransportRuntime(_artifact(), dtype=torch.float64)
    contexts = torch.randn(7, 4, 8, dtype=torch.float64, requires_grad=True)

    normal = runtime(
        invocation_id="mutation.13",
        head_contexts=contexts,
        control_mode="normal",
    )
    zero = runtime(
        invocation_id="mutation.13",
        head_contexts=contexts,
        control_mode="zero_held",
    )
    permuted = runtime(
        invocation_id="mutation.13",
        head_contexts=contexts,
        control_mode="permuted_held",
    )

    assert normal.routing_invocation_id == "mutation.13"
    assert permuted.routing_invocation_id == "mutation.15"
    assert torch.count_nonzero(normal.operator_outputs)
    assert not torch.count_nonzero(zero.operator_outputs)
    assert zero.operator_outputs.requires_grad
    assert not torch.equal(permuted.operator_outputs, normal.operator_outputs)
    assert normal.active_mask is not None
    assert torch.equal(
        normal.active_mask.sum(dim=-1),
        torch.full((7,), 2, dtype=torch.int64),
    )


def test_shared_runtime_wrapper_runs_from_live_candidate_state() -> None:
    exact = _exact_attention()
    runtime = FrozenCrossLayerTransportRuntime(_artifact(), dtype=torch.float64)
    wrapper = Gemma3NativeQKSharedVOAttentionOperator(exact, backend=runtime)
    sequence = _sequence(batch=1, length=4)
    incoming = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
    first = wrapper.execute_with_incoming(
        torch.randn(1, 4, 8, dtype=torch.float64),
        sequence,
        incoming,
    )
    second = wrapper.execute_with_incoming(
        torch.randn(1, 4, 8, dtype=torch.float64),
        sequence,
        incoming,
    )

    assert first.delta.shape == (1, 4, 8)
    assert not torch.equal(first.operator_output, second.operator_output)
    assert wrapper.execution_count == 2
    assert wrapper.native_q_projection_call_count == 2
    assert wrapper.native_k_projection_call_count == 2
    assert wrapper.shared_backend_call_count == 2
    assert wrapper.native_v_projection_call_count == 0
    assert wrapper.native_o_projection_call_count == 0
