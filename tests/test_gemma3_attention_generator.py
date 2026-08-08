from dataclasses import replace

import pytest
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
    ATTENTION_GENERATOR_KINDS,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator_from_structured_executor,
    compile_svd_reduced_attention_generator,
    transport_attention_coordinates,
)
from fisher_graph.gemma3_joint_cross_block_chart_executor import (
    JointGraphGeneratorMutation,
    build_joint_cross_block_event_schedule,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


def _layer_spec(
    *,
    attention_kind: str,
    window_size: int | None,
) -> LayerSpec:
    width = 8
    head_dimension = 2
    norm = NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    qk_norm = replace(norm, width=head_dimension)
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
    rope = RopeSpec(
        kind="rotary",
        theta=(10_000.0 if window_size is not None else 1_000_000.0),
        rotary_dimension=head_dimension,
        scaling_kind=(None if window_size is not None else "linear"),
        scaling_factor=(None if window_size is not None else 2.0),
    )
    return LayerSpec(
        id="layer.0",
        ordinal=0,
        input_site="layer.0.input",
        output_site="layer.0.output",
        residual_width=width,
        kind="gemma3_decoder",
        attention=AttentionSpec(
            kind=attention_kind,
            query_heads=4,
            key_value_heads=2,
            head_dimension=head_dimension,
            query_scale=0.5,
            qk_norm=True,
            window_size=window_size,
            rope=rope,
            cache_kind="none",
        ),
        transformer=TransformerLayerSemantics(
            residual_layout=(
                "sequential_attention_then_feed_forward_residual"
            ),
            attention_input_norm=norm,
            attention_output_norm=norm,
            qk_norm=qk_norm,
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


def _structured(
    *,
    attention_kind: str = "sliding_causal",
    window_size: int | None = 3,
) -> StructuredTransformerLayerExecutor:
    torch.manual_seed(81_911)
    return StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(
            _layer_spec(
                attention_kind=attention_kind,
                window_size=window_size,
            )
        ),
        dtype=torch.float64,
    ).eval()


def _sequence(mask: torch.Tensor) -> SequenceContext:
    positions = torch.arange(
        mask.shape[1],
        dtype=torch.long,
        device=mask.device,
    ).unsqueeze(0).expand(mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )


@pytest.mark.parametrize(
    ("attention_kind", "window_size"),
    (
        ("sliding_causal", 3),
        ("global_causal", None),
    ),
)
def test_exact_attention_generator_matches_source_free_structured_control(
    attention_kind: str,
    window_size: int | None,
) -> None:
    structured = _structured(
        attention_kind=attention_kind,
        window_size=window_size,
    )
    generator = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id=f"layer.0.{attention_kind}",
    )
    mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [False, True, True, True, True, False],
        ]
    )
    sequence = _sequence(mask)
    hidden = torch.randn(2, 6, 8, dtype=torch.float64)

    reference = structured.forward_components(hidden, sequence)
    candidate = generator(hidden, sequence)

    torch.testing.assert_close(
        candidate.normalized_input,
        reference.normalized_attention_input,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.operator_output[mask],
        reference.attention_operator_output[mask],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        candidate.residual_delta,
        reference.attention_delta,
        rtol=1e-12,
        atol=1e-12,
    )
    assert candidate.receipt.attention_kind == attention_kind
    assert candidate.receipt.source_projection_call_count == 0
    assert candidate.receipt.source_fallback_call_count == 0
    assert generator.architecture_manifest()["attention_contract"][
        "window_size"
    ] == window_size


def test_full_rank_svd_control_recovers_exact_attention_mutation() -> None:
    exact = compile_exact_attention_generator_from_structured_executor(
        _structured(),
        invocation_id="layer.0.attention",
    )
    factorized = compile_svd_reduced_attention_generator(
        exact,
        input_rank=8,
        output_rank=8,
    )
    sequence = _sequence(torch.ones(2, 5, dtype=torch.bool))
    hidden = torch.randn(2, 5, 8, dtype=torch.float64)

    expected = exact(hidden, sequence)
    actual = factorized(hidden, sequence)
    torch.testing.assert_close(
        actual.operator_output,
        expected.operator_output,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        actual.residual_delta,
        expected.residual_delta,
        rtol=1e-11,
        atol=1e-11,
    )


def test_reduced_coordinates_are_source_free_and_reusable() -> None:
    structured = _structured()
    exact = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="layer.0.attention",
    )
    reduced = compile_svd_reduced_attention_generator(
        exact,
        input_rank=3,
        output_rank=3,
    )
    sequence = _sequence(
        torch.tensor([[True, True, True, True, False]])
    )
    hidden = torch.randn(1, 5, 8, dtype=torch.float64)

    dense = reduced(hidden, sequence)
    packet_only = reduced.forward_from_coordinates(
        dense.input_coordinates,
        sequence,
    )
    torch.testing.assert_close(
        packet_only.residual_delta,
        dense.residual_delta,
        rtol=0.0,
        atol=0.0,
    )
    assert dense.input_coordinates.rank == 3
    assert dense.output_coordinates.rank == 3
    assert reduced.executor_local_source_free
    assert not reduced.owns_source_module
    assert not reduced.owns_source_model_weights
    assert not reduced.owns_source_fallback
    assert reduced.source_projection_call_count == 0
    assert all(
        not isinstance(module, StructuredTransformerLayerExecutor)
        for module in reduced.modules()
    )
    exact_pointers = {
        parameter.untyped_storage().data_ptr()
        for parameter in exact.parameters()
    }
    reduced_pointers = {
        parameter.untyped_storage().data_ptr()
        for parameter in reduced.parameters()
    }
    assert exact_pointers.isdisjoint(reduced_pointers)
    assert all(
        not bool(value.any())
        for value in (
            dense.input_coordinates.values[~sequence.query_valid_mask],
            dense.output_coordinates.values[~sequence.query_valid_mask],
            dense.residual_delta[~sequence.query_valid_mask],
        )
    )


def test_exact_control_preserves_checkpoint_coefficient_provenance() -> None:
    structured = _structured()
    structured._weight_origin.fill_(1)  # noqa: SLF001
    exact = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="mutation.1",
    )

    assert exact.owns_source_model_weights
    assert not exact.owns_source_module
    assert not exact.owns_source_fallback
    assert exact.executor_local_source_free
    manifest = exact.architecture_manifest()
    assert manifest["source_independent_runtime"]
    assert manifest["source_checkpoint_coefficients_retained"]


def test_output_packet_transports_into_a_cross_block_input_chart() -> None:
    structured = _structured()
    first_exact = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="layer.0.attention",
    )
    second_exact = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="layer.1.attention",
    )
    first = compile_svd_reduced_attention_generator(
        first_exact,
        input_rank=3,
        output_rank=3,
    )
    second = compile_svd_reduced_attention_generator(
        second_exact,
        input_rank=3,
        output_rank=3,
    )
    sequence = _sequence(torch.ones(1, 4, dtype=torch.bool))
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    first_result = first(hidden, sequence)

    transported = transport_attention_coordinates(
        first_result.output_coordinates,
        torch.eye(3, dtype=torch.float64),
        target_coordinate_system_sha256=(
            second.input_coordinate_system_sha256
        ),
        target_invocation_id=second.config.invocation_id,
    )
    second_result = second.forward_from_coordinates(transported, sequence)

    assert transported.semantic_role == "transported_input"
    assert second_result.normalized_input is None
    assert second_result.receipt.invocation_id == "layer.1.attention"
    assert second_result.receipt.source_projection_call_count == 0


def test_joint_graph_bridge_adds_incoming_coordinates_before_qkv() -> None:
    exact = compile_exact_attention_generator_from_structured_executor(
        _structured(),
        invocation_id="layer.0.attention",
    )
    reduced = compile_svd_reduced_attention_generator(
        exact,
        input_rank=3,
        output_rank=2,
    )
    sequence = _sequence(torch.ones(1, 4, dtype=torch.bool))
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    zeros = torch.zeros(1, 4, 1, 3, dtype=torch.float64)

    baseline = reduced(hidden, sequence)
    bridged = reduced.execute_with_incoming(hidden, sequence, zeros)
    torch.testing.assert_close(
        bridged.residual_delta,
        baseline.residual_delta,
        rtol=0.0,
        atol=0.0,
    )
    assert bridged.emitted_generator_latents.shape == (1, 4, 1, 2)
    assert reduced.chart_count == 1
    assert reduced.incoming_chart_count == 1
    assert reduced.emitted_chart_count == 1
    assert reduced.incoming_generator_rank == 3
    assert reduced.emitted_generator_rank == 2

    incoming = zeros.clone()
    incoming[:, 1:, :, 0] = 0.25
    changed = reduced.execute_with_incoming(hidden, sequence, incoming)
    assert not torch.allclose(
        changed.residual_delta[:, 1:],
        baseline.residual_delta[:, 1:],
    )

    exact_zeros = torch.zeros(1, 4, 1, 8, dtype=torch.float64)
    exact_baseline = exact(hidden, sequence)
    exact_bridged = exact.execute_with_incoming(
        hidden,
        sequence,
        exact_zeros,
    )
    torch.testing.assert_close(
        exact_bridged.residual_delta,
        exact_baseline.residual_delta,
        rtol=0.0,
        atol=0.0,
    )


def test_reduced_parameter_and_logical_mac_ledgers_close() -> None:
    exact = compile_exact_attention_generator_from_structured_executor(
        _structured(),
        invocation_id="layer.0.attention",
    )
    reduced = compile_svd_reduced_attention_generator(
        exact,
        input_rank=3,
        output_rank=3,
    )
    sequence = _sequence(
        torch.tensor([[True, True, True, True, False]])
    )

    exact_parameters = exact.parameter_accounting()
    reduced_parameters = reduced.parameter_accounting()
    assert exact_parameters.source_native_coefficient_count == 212
    assert exact_parameters.candidate_stored_coefficient_count == 212
    assert exact_parameters.removed_coefficient_count == 0
    assert reduced_parameters.candidate_stored_coefficient_count == 140
    assert reduced_parameters.removed_coefficient_count == 72
    assert reduced_parameters.compression_ratio == pytest.approx(72 / 212)

    exact_macs = exact.logical_accounting(sequence)
    reduced_macs = reduced.logical_accounting(sequence)
    packet_macs = reduced.logical_accounting(
        sequence,
        input_coordinates_supplied=True,
    )
    assert exact_macs.logical_causal_key_pairs == 9
    assert exact_macs.logical_total_macs == 912
    assert reduced_macs.logical_total_macs == 624
    assert packet_macs.logical_total_macs == 528
    assert packet_macs.input_chart_macs == 0


def test_generator_config_and_packet_contracts_fail_closed() -> None:
    structured = _structured()
    config = Gemma3AttentionGeneratorConfig.from_structured_config(
        "layer.0.attention",
        structured.config,
    )
    with pytest.raises(ValueError, match="global attention"):
        replace(
            config,
            attention=replace(
                config.attention,
                kind="global_causal",
            ),
        )
    assert ATTENTION_GENERATOR_KINDS == (
        "exact_full_rank",
        "reduced_coordinates",
    )

    exact = compile_exact_attention_generator_from_structured_executor(
        structured,
        invocation_id="layer.0.attention",
    )
    reduced = compile_svd_reduced_attention_generator(
        exact,
        input_rank=3,
        output_rank=3,
    )
    sequence = _sequence(torch.ones(1, 3, dtype=torch.bool))
    packet = reduced(torch.randn(1, 3, 8, dtype=torch.float64), sequence)
    with pytest.raises(ValueError, match="does not match"):
        exact.forward_from_coordinates(packet.input_coordinates, sequence)


@pytest.mark.parametrize(
    ("event_index", "attention_kind", "window_size"),
    (
        (1, "sliding_causal", 3),
        (11, "global_causal", None),
    ),
)
def test_joint_mutation_directly_delegates_complete_attention_once(
    event_index: int,
    attention_kind: str,
    window_size: int | None,
) -> None:
    event = build_joint_cross_block_event_schedule()[event_index]
    generator = compile_exact_attention_generator_from_structured_executor(
        _structured(
            attention_kind=attention_kind,
            window_size=window_size,
        ),
        invocation_id=event.event_id,
    )
    mutation = JointGraphGeneratorMutation(
        event_spec=event,
        compiled_operator=generator,
        source_checkpoint_coefficients_retained=False,
    )
    sequence = _sequence(torch.ones(1, 4, dtype=torch.bool))
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    incoming = torch.zeros(
        1,
        4,
        generator.incoming_chart_count,
        generator.incoming_generator_rank,
        dtype=torch.float64,
    )

    expected = generator(hidden, sequence)
    actual = mutation.execute_with_incoming(hidden, sequence, incoming)

    torch.testing.assert_close(
        actual.delta,
        expected.residual_delta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.emitted_generator_latents,
        expected.emitted_generator_latents,
        rtol=0.0,
        atol=0.0,
    )
    metadata = mutation.metadata()
    assert metadata["direct_compiled_operator_present"]
    assert metadata["sequence_operator_present"]
