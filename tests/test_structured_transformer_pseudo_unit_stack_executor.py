import copy

import pytest
import torch

from fisher_graph.activations import ActivationTrace
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
from fisher_graph.compiler.manifest import (
    BackendSpec,
    CompiledSegment,
    SegmentProvenance,
    SegmentValidation,
    SequenceSpec as ManifestSequenceSpec,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)
from fisher_graph.structured_transformer_pseudo_unit_stack_executor import (
    PseudoUnitCarryEdge,
    StructuredTransformerPseudoUnitStackExecutor,
)


def _layer_spec(
    ordinal: int,
    *,
    width: int = 4,
    intermediate_width: int = 4,
) -> LayerSpec:
    layer_id = f"layer.{ordinal}"
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
        width=2,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    return LayerSpec(
        id=layer_id,
        ordinal=ordinal,
        input_site=f"{layer_id}.input",
        output_site=f"{layer_id}.output",
        residual_width=width,
        kind="gemma3_decoder",
        attention=AttentionSpec(
            kind="global_causal",
            query_heads=width // 2,
            key_value_heads=1,
            head_dimension=2,
            query_scale=2**-0.5,
            qk_norm=True,
            window_size=None,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=2,
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
                intermediate_width=intermediate_width,
                activation="gelu_pytorch_tanh",
                projection_bias=False,
            ),
            stages=(
                ResidualStageSpec(
                    id=f"{layer_id}.attention",
                    kind="attention",
                    input_site=f"{layer_id}.input",
                    normalized_input_site=(
                        f"{layer_id}.attention.normalized_input"
                    ),
                    operator_output_site=(
                        f"{layer_id}.attention.operator_output"
                    ),
                    delta_site=f"{layer_id}.attention.delta",
                    output_site=f"{layer_id}.post_attention",
                ),
                ResidualStageSpec(
                    id=f"{layer_id}.feed_forward",
                    kind="feed_forward",
                    input_site=f"{layer_id}.post_attention",
                    normalized_input_site=(
                        f"{layer_id}.mlp.normalized_input"
                    ),
                    operator_output_site=(
                        f"{layer_id}.mlp.operator_output"
                    ),
                    delta_site=f"{layer_id}.mlp.delta",
                    output_site=f"{layer_id}.output",
                ),
            ),
        ),
    )


def _parents(
    *,
    count: int = 3,
    width: int = 4,
    intermediate_width: int = 4,
) -> tuple[StructuredTransformerLayerExecutor, ...]:
    torch.manual_seed(9201)
    return tuple(
        StructuredTransformerLayerExecutor(
            StructuredTransformerLayerExecutorConfig.from_layer_spec(
                _layer_spec(
                    ordinal,
                    width=width,
                    intermediate_width=intermediate_width,
                )
            )
        ).eval()
        for ordinal in range(count)
    )


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


def _edge() -> PseudoUnitCarryEdge:
    return PseudoUnitCarryEdge(
        edge_id="layer0_unit0_to_layer2_unit1",
        anchor_layer_id="layer.0",
        anchor_source_index=0,
        consumer_layer_id="layer.2",
        consumer_source_index=1,
    )


def _storage_pointers(
    module: torch.nn.Module,
) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel() > 0
    }


def _exact_duplicate_parents(
) -> tuple[StructuredTransformerLayerExecutor, ...]:
    parents = _parents()
    with torch.no_grad():
        for parent in parents:
            parent.attention.o_proj.weight.zero_()
            parent.feed_forward.down_proj.weight.zero_()
            for norm in (
                parent.attention_input_norm,
                parent.attention_output_norm,
                parent.feed_forward_input_norm,
                parent.feed_forward_output_norm,
            ):
                norm.weight.zero_()

        parents[0].feed_forward.gate_proj.weight[0].copy_(
            parents[2].feed_forward.gate_proj.weight[1]
        )
        parents[0].feed_forward.up_proj.weight[0].copy_(
            parents[2].feed_forward.up_proj.weight[1]
        )
        parents[2].feed_forward.down_proj.weight[:, 1].copy_(
            torch.tensor([0.3, -0.2, 0.1, 0.4])
        )
    return parents


def _sequential_parent_output(
    parents: tuple[StructuredTransformerLayerExecutor, ...],
    hidden_states: torch.Tensor,
    sequence: SequenceContext,
) -> torch.Tensor:
    current = hidden_states
    for parent in parents:
        current = parent(current, sequence)
    return current


def _compiled_segment(
    *,
    source_layers: tuple[str, ...] = ("layer.0", "layer.1", "layer.2"),
    input_activation: str = "layer.0.input",
    output_activation: str = "layer.2.output",
) -> CompiledSegment:
    return CompiledSegment(
        id="compiled.layer.0-2",
        order=0,
        source_layers=source_layers,
        input_activation=input_activation,
        output_activation=output_activation,
        backend=BackendSpec(id="unit.pseudo_stack", abi_version=1),
        sequence=ManifestSequenceSpec(
            policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            causal=True,
            attention_mask="optional",
            padding="either",
            position_ids="required",
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
            validator_id="unit.pseudo_stack",
            validator_version=1,
            report_resource="validation.report",
        ),
    )


def test_nonadjacent_duplicate_reuse_is_exact_and_removes_consumer_generator(
) -> None:
    parents = _exact_duplicate_parents()
    parent_fingerprints = tuple(
        parent.execution_fingerprint() for parent in parents
    )
    parent_storage = set().union(
        *(_storage_pointers(parent) for parent in parents)
    )

    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        parents,
        (_edge(),),
    )
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    sequence = _sequence(mask)
    hidden_states = torch.randn(2, 5, 4)

    expected = _sequential_parent_output(
        parents,
        hidden_states,
        sequence,
    )
    actual = executor(hidden_states, sequence)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual[~mask], hidden_states[~mask])

    consumer = executor.layers[2].feed_forward
    assert consumer.gate_proj.weight.shape == (3, 4)
    assert consumer.up_proj.weight.shape == (3, 4)
    assert consumer.down_proj.weight.shape == (4, 3)
    retained = torch.tensor([0, 2, 3])
    torch.testing.assert_close(
        consumer.gate_proj.weight,
        parents[2].feed_forward.gate_proj.weight.index_select(0, retained),
    )
    torch.testing.assert_close(
        consumer.up_proj.weight,
        parents[2].feed_forward.up_proj.weight.index_select(0, retained),
    )
    torch.testing.assert_close(
        consumer.down_proj.weight,
        parents[2].feed_forward.down_proj.weight.index_select(1, retained),
    )
    torch.testing.assert_close(
        executor.consumer_decoder_matrix[0],
        parents[2].feed_forward.down_proj.weight[:, 1],
    )

    assert executor.learned_parameter_count == (
        executor.source_parameter_count - 2 * executor.width
    )
    assert tuple(
        parent.execution_fingerprint() for parent in parents
    ) == parent_fingerprints
    assert not (
        parent_storage
        & _storage_pointers(executor)
    )
    assert executor.executor_local_source_free
    assert not executor.owns_source_model_weights
    assert not executor.owns_source_fallback


def test_carry_is_token_aligned_masked_and_capture_friendly() -> None:
    parents = _exact_duplicate_parents()
    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        parents,
        (_edge(),),
    )
    mask = torch.tensor([[True, True, False, False]])
    sequence = _sequence(mask)
    hidden_states = torch.randn(1, 4, 4)
    trace = ActivationTrace(retain_grad=False)

    result = executor.forward_components(
        hidden_states,
        sequence,
        trace=trace,
    )

    carry_site = executor.carry_site(executor.edges[0])
    injection_site = executor.consumer_injection_site(executor.edges[0])
    assert carry_site in executor.capture_sites
    assert injection_site in executor.capture_sites
    assert carry_site in trace
    assert injection_site in trace
    assert result.carried_features[0].shape == mask.shape
    assert torch.count_nonzero(result.carried_features[0][~mask]) == 0
    assert torch.count_nonzero(trace[carry_site][~mask]) == 0
    assert torch.count_nonzero(trace[injection_site][~mask]) == 0

    changed_padding = hidden_states.clone()
    changed_padding[~mask] += 100_000.0
    changed = executor.forward_components(changed_padding, sequence)
    torch.testing.assert_close(
        changed.output[mask],
        result.output[mask],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        changed.carried_features[0][mask],
        result.carried_features[0][mask],
        rtol=0.0,
        atol=0.0,
    )


def test_stack_and_carry_do_not_create_future_token_edges() -> None:
    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        _parents(),
        (_edge(),),
    )
    sequence = _sequence(torch.ones(1, 5, dtype=torch.bool))
    hidden_states = torch.randn(1, 5, 4)
    baseline = executor.forward_components(hidden_states, sequence)

    future_changed = hidden_states.clone()
    future_changed[:, 4] += 100_000.0
    changed = executor.forward_components(future_changed, sequence)

    torch.testing.assert_close(
        changed.output[:, :4],
        baseline.output[:, :4],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        changed.carried_features[0][:, :4],
        baseline.carried_features[0][:, :4],
        rtol=0.0,
        atol=0.0,
    )


def test_accounting_counts_only_removed_gate_and_up_rows() -> None:
    parents = _exact_duplicate_parents()
    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        parents,
        (_edge(),),
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    ledger = executor.logical_accounting(_sequence(mask))

    assert ledger.valid_tokens == 5
    assert ledger.removed_parameter_count == 2 * executor.width
    assert ledger.carried_decoder_parameter_count == executor.width
    assert ledger.removed_logical_macs == 5 * 2 * executor.width
    assert ledger.carried_decoder_macs == 5 * executor.width
    assert (
        ledger.source_parameter_count
        - ledger.candidate_parameter_count
        == 2 * executor.width
    )
    assert (
        ledger.source_logical_macs - ledger.candidate_logical_macs
        == 5 * 2 * executor.width
    )
    assert ledger.source_parameter_count == sum(
        parent.learned_parameter_count for parent in parents
    )
    assert ledger.source_logical_macs == sum(
        parent.logical_accounting(_sequence(mask)).logical_total_macs
        for parent in parents
    )
    metadata = ledger.metadata()
    assert metadata["latency_measured"] is False
    assert metadata["kernel_speedup_claimed"] is False


def test_source_free_artifact_roundtrip_is_strict_and_exact() -> None:
    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        _exact_duplicate_parents(),
        (_edge(),),
    )
    mask = torch.tensor([[True, True, True, False]])
    sequence = _sequence(mask)
    hidden_states = torch.randn(1, 4, 4)
    expected = executor(hidden_states, sequence)

    artifact = executor.artifact_state_dict()
    assert artifact["contains_source_model_weights"] is False
    assert artifact["contains_source_fallback"] is False
    restored = (
        StructuredTransformerPseudoUnitStackExecutor
        .from_artifact_state_dict(artifact)
    )

    assert restored.execution_fingerprint() == executor.execution_fingerprint()
    torch.testing.assert_close(
        restored(hidden_states, sequence),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert not restored.owns_source_model_weights
    assert not restored.owns_source_fallback

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="artifact fields"):
        (
            StructuredTransformerPseudoUnitStackExecutor
            .from_artifact_state_dict(extra)
        )

    tampered = copy.deepcopy(artifact)
    model_state = tampered["model_state_dict"]
    assert isinstance(model_state, dict)
    model_state["consumer_decoder_matrix"][0, 0] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        (
            StructuredTransformerPseudoUnitStackExecutor
            .from_artifact_state_dict(tampered)
        )


def test_compiled_segment_run_requires_the_complete_window() -> None:
    executor = StructuredTransformerPseudoUnitStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        _exact_duplicate_parents(),
        (_edge(),),
    )
    sequence = _sequence(torch.ones(1, 3, dtype=torch.bool))
    hidden_states = torch.randn(1, 3, 4)

    run = executor.run(
        _compiled_segment(),
        hidden_states,
        sequence,
    )
    torch.testing.assert_close(
        run.hidden_states,
        executor(hidden_states, sequence),
        rtol=0.0,
        atol=0.0,
    )
    assert run.sequence is sequence
    assert run.raw_output["source_layers"] == (
        "layer.0",
        "layer.1",
        "layer.2",
    )

    with pytest.raises(ValueError, match="complete pseudo-unit stack"):
        executor.run(
            _compiled_segment(
                source_layers=("layer.0", "layer.1"),
                output_activation="layer.1.output",
            ),
            hidden_states,
            sequence,
        )


@pytest.mark.parametrize(
    "edges",
    [
        (
            PseudoUnitCarryEdge(
                edge_id="backwards",
                anchor_layer_id="layer.2",
                anchor_source_index=0,
                consumer_layer_id="layer.0",
                consumer_source_index=0,
            ),
        ),
        (
            PseudoUnitCarryEdge(
                edge_id="same_layer",
                anchor_layer_id="layer.1",
                anchor_source_index=0,
                consumer_layer_id="layer.1",
                consumer_source_index=1,
            ),
        ),
        (
            PseudoUnitCarryEdge(
                edge_id="out_of_range",
                anchor_layer_id="layer.0",
                anchor_source_index=10,
                consumer_layer_id="layer.2",
                consumer_source_index=0,
            ),
        ),
        (
            PseudoUnitCarryEdge(
                edge_id="duplicate_endpoint_a",
                anchor_layer_id="layer.0",
                anchor_source_index=0,
                consumer_layer_id="layer.1",
                consumer_source_index=0,
            ),
            PseudoUnitCarryEdge(
                edge_id="duplicate_endpoint_b",
                anchor_layer_id="layer.0",
                anchor_source_index=0,
                consumer_layer_id="layer.2",
                consumer_source_index=1,
            ),
        ),
    ],
)
def test_invalid_edges_are_rejected(
    edges: tuple[PseudoUnitCarryEdge, ...],
) -> None:
    with pytest.raises(ValueError):
        StructuredTransformerPseudoUnitStackExecutor(
            ("layer.0", "layer.1", "layer.2"),
            _parents(),
            edges,
        )


def test_noncontiguous_or_incompatible_stacks_are_rejected() -> None:
    parents = _parents()
    with pytest.raises(ValueError, match="contiguous"):
        StructuredTransformerPseudoUnitStackExecutor(
            ("layer.0", "layer.2", "layer.3"),
            parents,
            (_edge(),),
        )

    incompatible = (
        parents[0],
        parents[1],
        _parents(count=3, width=6)[2],
    )
    with pytest.raises(ValueError, match="share dtype, device"):
        StructuredTransformerPseudoUnitStackExecutor(
            ("layer.0", "layer.1", "layer.2"),
            incompatible,
            (_edge(),),
        )
