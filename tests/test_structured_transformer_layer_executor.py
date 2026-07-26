import copy
from dataclasses import asdict, replace

import pytest
import torch

from fisher_graph.activations import ActivationTrace
from fisher_graph.adapters import (
    AttentionSpec,
    FeedForwardSpec,
    Gemma3CausalLMAdapter,
    LayerSpec,
    NormalizationSpec,
    ResidualStageSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
    TransformerLayerSemantics,
)
from fisher_graph.compiler.manifest import (
    BackendSpec,
    CompiledSegment,
    SegmentProvenance,
    SegmentValidation,
    SequenceSpec as ManifestSequenceSpec,
)
from fisher_graph.gemma3_trajectory_experiment import (
    _validate_layer_metadata,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


def _layer_spec(
    *,
    attention_kind: str = "sliding_causal",
    window_size: int | None = 3,
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
    qk_norm = NormalizationSpec(
        kind="rms_norm",
        width=head_dimension,
        epsilon=1e-6,
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
            kind=attention_kind,
            query_heads=4,
            key_value_heads=2,
            head_dimension=head_dimension,
            query_scale=0.5,
            qk_norm=True,
            window_size=window_size,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
            ),
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
            attention_logit_softcap=None,
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


def _sequence(
    mask: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
) -> SequenceContext:
    if positions is None:
        positions = torch.arange(
            mask.shape[1],
            dtype=torch.long,
        ).unsqueeze(0).expand(mask.shape[0], -1)
    elif positions.shape[0] == 1 and mask.shape[0] > 1:
        positions = positions.expand(mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=positions is not None,
            cache_positions_supplied=False,
        ),
    )


def _executor(
    *,
    causal_edges_enabled: bool = True,
    attention_kind: str = "sliding_causal",
    window_size: int | None = 3,
) -> StructuredTransformerLayerExecutor:
    torch.manual_seed(7301)
    return StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(
            _layer_spec(
                attention_kind=attention_kind,
                window_size=window_size,
            ),
            causal_edges_enabled=causal_edges_enabled,
        )
    ).eval()


def _compiled_segment(
    *,
    input_activation: str = "layer.0.input",
    output_activation: str = "layer.0.output",
) -> CompiledSegment:
    return CompiledSegment(
        id="compiled.layer.0",
        order=0,
        source_layers=("layer.0",),
        input_activation=input_activation,
        output_activation=output_activation,
        backend=BackendSpec(id="unit.structured", abi_version=1),
        sequence=ManifestSequenceSpec(
            policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            causal=True,
            attention_mask="optional",
            padding="either",
            position_ids="optional",
            cache="none",
        ),
        fast_resources=("executor",),
        instrumentation_resources=(),
        instrumentation_policy="none",
        fallback_policy="disabled",
        provenance=SegmentProvenance(
            source_model_state_sha256="0" * 64,
            source_model_config_sha256="1" * 64,
            dependency_resources=("executor",),
            compile_config_sha256=None,
        ),
        validation=SegmentValidation(
            status="passed",
            validator_id="unit.structured",
            validator_version=1,
            report_resource="validation.report",
        ),
    )


def test_structured_executor_exposes_gqa_stages_and_invalid_passthrough() -> None:
    executor = _executor()
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    sequence = _sequence(mask)
    hidden = torch.randn(2, 5, 8)
    trace = ActivationTrace(retain_grad=False)
    result = executor.forward_components(hidden, sequence, trace=trace)

    assert trace["structured_layer.attention.query"].shape == (
        2,
        4,
        5,
        2,
    )
    assert trace["structured_layer.attention.key"].shape == (
        2,
        2,
        5,
        2,
    )
    assert trace["structured_layer.attention.value"].shape == (
        2,
        2,
        5,
        2,
    )
    assert result.attention_delta.shape == hidden.shape
    assert result.feed_forward_delta.shape == hidden.shape
    assert torch.equal(result.output[~mask], hidden[~mask])
    torch.testing.assert_close(
        result.post_attention[mask],
        (hidden + result.attention_delta)[mask],
    )
    torch.testing.assert_close(
        result.output[mask],
        (result.post_attention + result.feed_forward_delta)[mask],
    )
    assert executor.learned_parameter_count == 612
    assert executor.fixed_runtime_coefficient_count == 0
    assert executor.executor_local_source_free
    assert not executor.owns_source_model_weights
    assert not executor.owns_source_fallback


def test_sliding_attention_excludes_future_and_outside_window() -> None:
    executor = _executor()
    mask = torch.ones(1, 6, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(1, 6, 8)
    baseline = executor(hidden, sequence)

    future = hidden.clone()
    future[:, 5] += 10_000.0
    torch.testing.assert_close(
        executor(future, sequence)[:, :5],
        baseline[:, :5],
        rtol=0.0,
        atol=0.0,
    )

    outside = hidden.clone()
    outside[:, 0] += 10_000.0
    torch.testing.assert_close(
        executor(outside, sequence)[:, 4],
        baseline[:, 4],
        rtol=0.0,
        atol=0.0,
    )

    inside = hidden.clone()
    inside[:, 2] += 10.0
    assert not torch.allclose(
        executor(inside, sequence)[:, 4],
        baseline[:, 4],
    )


def test_rope_is_uniform_shift_invariant_but_reads_relative_spacing() -> None:
    executor = _executor(
        attention_kind="global_causal",
        window_size=None,
    )
    mask = torch.ones(1, 5, dtype=torch.bool)
    hidden = torch.randn(1, 5, 8)
    contiguous = _sequence(
        mask,
        positions=torch.tensor([[0, 1, 2, 3, 4]]),
    )
    shifted = _sequence(
        mask,
        positions=torch.tensor([[17, 18, 19, 20, 21]]),
    )
    gapped = _sequence(
        mask,
        positions=torch.tensor([[0, 2, 4, 6, 8]]),
    )

    contiguous_output = executor(hidden, contiguous)
    shifted_output = executor(hidden, shifted)
    gapped_output = executor(hidden, gapped)

    torch.testing.assert_close(
        shifted_output,
        contiguous_output,
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.allclose(
        gapped_output,
        contiguous_output,
        rtol=1e-5,
        atol=1e-6,
    )


def test_packed_positions_without_mask_fail_closed() -> None:
    executor = _executor()
    mask = torch.ones(1, 6, dtype=torch.bool)
    sequence = _sequence(
        mask,
        positions=torch.tensor([[0, 1, 2, 0, 1, 2]]),
    )
    sequence.input_origin = SequenceInputOrigin(
        attention_mask_supplied=False,
        position_ids_supplied=True,
        cache_positions_supplied=False,
    )
    hidden = torch.randn(1, 6, 8)

    with pytest.raises(ValueError, match="explicit attention_mask"):
        executor(hidden, sequence)
    with pytest.raises(ValueError, match="explicit attention_mask"):
        executor.logical_accounting(sequence)


def test_attention_disabled_control_is_storage_matched_and_position_local() -> None:
    causal = _executor(causal_edges_enabled=True)
    control = _executor(causal_edges_enabled=False)
    assert causal.learned_parameter_count == control.learned_parameter_count
    assert control.causal_edge_control == (
        "attention_output_zeroed_storage_matched"
    )

    mask = torch.ones(1, 5, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(1, 5, 8)
    changed = hidden.clone()
    changed[:, 0] += 100.0
    baseline = control(hidden, sequence)
    candidate = control(changed, sequence)
    torch.testing.assert_close(
        candidate[:, 1:],
        baseline[:, 1:],
        rtol=0.0,
        atol=0.0,
    )


def test_accounting_uses_gqa_width_and_sliding_pairs() -> None:
    executor = _executor()
    mask = torch.tensor([[True, True, True, True, False]])
    accounting = executor.logical_accounting(_sequence(mask))

    assert accounting.valid_tokens == 4
    assert accounting.logical_causal_key_pairs == 9
    assert accounting.attention_projection_macs == 768
    assert accounting.attention_score_macs == 72
    assert accounting.attention_value_macs == 72
    assert accounting.feed_forward_macs == 1_536
    assert accounting.logical_total_macs == 2_448


def test_strict_artifact_roundtrip_and_tamper_rejection() -> None:
    executor = _executor()
    mask = torch.ones(2, 4, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(2, 4, 8)
    expected = executor(hidden, sequence)
    artifact = executor.artifact_state_dict()

    restored = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        artifact
    )
    torch.testing.assert_close(
        restored(hidden, sequence),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert restored.execution_fingerprint() == executor.execution_fingerprint()

    changed_control = copy.deepcopy(artifact)
    changed_control["config"]["causal_edges_enabled"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            changed_control
        )

    nonfinite = copy.deepcopy(artifact)
    first = next(
        value
        for value in nonfinite["model_state_dict"].values()
        if value.is_floating_point()
    )
    first.flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        StructuredTransformerLayerExecutor.from_artifact_state_dict(nonfinite)


def test_artifact_requires_eval_and_preserves_weight_origin() -> None:
    executor = _executor()
    eval_fingerprint = executor.execution_fingerprint()
    executor.train()
    assert executor.execution_fingerprint() != eval_fingerprint
    with pytest.raises(RuntimeError, match="eval mode"):
        executor.artifact_state_dict()

    executor.eval()
    executor.attention.train()
    assert not executor.training
    assert executor.execution_fingerprint() != eval_fingerprint
    with pytest.raises(RuntimeError, match="eval mode"):
        executor.artifact_state_dict()

    executor.eval()
    assert executor.execution_fingerprint() == eval_fingerprint
    contaminated = copy.deepcopy(executor.state_dict())
    contaminated["_weight_origin"].fill_(1)
    restored_state = _executor()
    restored_state.load_state_dict(contaminated, strict=True)
    assert restored_state.owns_source_model_weights
    assert not restored_state.executor_local_source_free
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        restored_state.artifact_state_dict()


def test_artifact_writer_rejects_nonfinite_executor_state() -> None:
    executor = _executor()
    with torch.no_grad():
        next(executor.parameters()).flatten()[0] = float("nan")

    with pytest.raises(ValueError, match="tensor"):
        executor.artifact_state_dict()


def test_compiled_run_uses_canonical_stage_sites_once() -> None:
    executor = _executor()
    mask = torch.ones(1, 4, dtype=torch.bool)
    sequence = _sequence(mask)
    hidden = torch.randn(1, 4, 8)
    expected_sites = (
        "layer.0.input",
        "layer.0.attention.normalized_input",
        "layer.0.attention.operator_output",
        "layer.0.attention.delta",
        "layer.0.post_attention",
        "layer.0.mlp.normalized_input",
        "layer.0.mlp.operator_output",
        "layer.0.mlp.delta",
        "layer.0.output",
    )
    trace = ActivationTrace(
        retain_grad=False,
        capture_sites=expected_sites,
    )
    result = executor.run(
        _compiled_segment(),
        hidden,
        sequence,
        trace=trace,
    )

    trace.assert_all_captures_seen()
    assert trace.names == expected_sites
    torch.testing.assert_close(
        result.hidden_states,
        trace["layer.0.output"],
    )
    assert result.raw_output["executor_local_source_free"]

    with pytest.raises(ValueError, match="activation binding"):
        executor.run(
            _compiled_segment(input_activation="other.input"),
            hidden,
            sequence,
        )


def test_structured_config_rejects_semantic_contradictions() -> None:
    base = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    contradictions = (
        (
            lambda: replace(
                base,
                attention=replace(
                    base.attention,
                    kind="global_causal",
                ),
            ),
            "global attention",
        ),
        (
            lambda: replace(
                base,
                attention=replace(
                    base.attention,
                    rope=replace(
                        base.attention.rope,
                        scaling_factor=2.0,
                    ),
                ),
            ),
            "default RoPE",
        ),
        (
            lambda: replace(
                base,
                transformer=replace(
                    base.transformer,
                    feed_forward_output_norm=replace(
                        base.transformer.feed_forward_output_norm,
                        width=6,
                    ),
                ),
            ),
            "normalization widths",
        ),
        (
            lambda: replace(
                base,
                transformer=replace(
                    base.transformer,
                    qk_norm=replace(
                        base.transformer.qk_norm,
                        width=4,
                    ),
                ),
            ),
            "qk normalization width",
        ),
    )
    for build, message in contradictions:
        with pytest.raises(ValueError, match=message):
            build()


def test_trajectory_metadata_accepts_dataclass_tuple_stages() -> None:
    layer = _layer_spec()
    metadata = asdict(layer)
    assert isinstance(metadata["transformer"]["stages"], tuple)

    assert _validate_layer_metadata(
        [metadata],
        start_layer=0,
        end_layer=0,
        widths=(8, 8),
    ) == ("layer.0",)

    invalid = (
        (("attention", "query_heads"), 3, "attention metadata"),
        (("attention", "window_size"), 2.5, "attention metadata"),
        (
            ("attention", "rope", "rotary_dimension"),
            1.5,
            "RoPE metadata",
        ),
    )
    for path, value, message in invalid:
        changed = copy.deepcopy(metadata)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            _validate_layer_metadata(
                [changed],
                start_layer=0,
                end_layer=0,
                widths=(8, 8),
            )


@pytest.mark.parametrize(
    ("layer_type", "sliding_window"),
    [
        ("sliding_attention", 3),
        ("full_attention", 3),
    ],
)
def test_optional_real_gemma_native_weight_transplant_parity(
    layer_type: str,
    sliding_window: int,
) -> None:
    try:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    except ImportError:
        pytest.skip("optional Transformers dependency is not installed")

    torch.manual_seed(8803)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=sliding_window,
        layer_types=[layer_type],
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Gemma3ForCausalLM(config).eval()
    adapter = Gemma3CausalLMAdapter(model)
    layer = adapter.layers[0]
    executor = StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    ).eval()
    executor.transplant_gemma3_layer_weights_(
        adapter.source_module(layer.id)
    )
    assert executor.owns_source_model_weights
    assert not executor.executor_local_source_free

    state_transferred = StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(layer)
    ).eval()
    state_transferred.load_state_dict(executor.state_dict(), strict=True)
    assert state_transferred.owns_source_model_weights
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        state_transferred.artifact_state_dict()

    mask = torch.tensor(
        [[True, True, True, True, True, False, False]]
    )
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    inputs = {
        "input_ids": torch.tensor([[1, 7, 3, 9, 2, 0, 0]]),
        "attention_mask": mask,
        "position_ids": positions,
    }
    with pytest.raises(ValueError, match="explicit attention_mask"):
        adapter.prepare_sequence(
            {
                "input_ids": inputs["input_ids"],
                "position_ids": positions,
            }
        )
    native_run = adapter.forward(
        inputs,
        capture_sites=(layer.input_site, layer.output_site),
    )
    sequence = native_run.sequence
    hidden = native_run.activations[layer.input_site]
    native = native_run.activations[layer.output_site]
    segmented = adapter.run_segment(
        adapter.segments[0],
        hidden,
        sequence,
    ).hidden_states
    structured = executor(hidden, sequence)

    torch.testing.assert_close(
        segmented[mask],
        native[mask],
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        structured[mask],
        native[mask],
        rtol=2e-5,
        atol=2e-6,
    )
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        executor.artifact_state_dict()
