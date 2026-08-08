from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
    compile_svd_reduced_attention_generator,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)


_WIDTH = 3
_VOCABULARY = 5
_CHART_COUNT = 2
_GENERATOR_RANK = 2


def _sequence(*, valid: torch.Tensor | None = None) -> SequenceContext:
    if valid is None:
        valid = torch.tensor([[True, True, False]])
    positions = torch.arange(valid.shape[1]).expand(valid.shape[0], -1)
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions.clone(),
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


@dataclass(frozen=True)
class _ToyChartExecution:
    output: torch.Tensor
    emitted_generator_latents: torch.Tensor


class _ToyChartGenerator(nn.Module):
    hidden_width = _WIDTH
    chart_count = _CHART_COUNT
    generator_rank = _GENERATOR_RANK

    def __init__(self, *, delta: float = 0.01) -> None:
        super().__init__()
        self.delta = nn.Parameter(
            torch.full((_WIDTH,), delta),
            requires_grad=False,
        )
        self.emission = nn.Parameter(
            torch.ones(_CHART_COUNT, _GENERATOR_RANK),
            requires_grad=False,
        )
        self.calls = 0

    def execute_with_incoming(
        self,
        value: torch.Tensor,
        incoming: torch.Tensor,
    ) -> _ToyChartExecution:
        self.calls += 1
        incoming_delta = incoming.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        emitted = self.emission.expand(*value.shape[:2], -1, -1)
        return _ToyChartExecution(
            output=self.delta.expand_as(value) + incoming_delta,
            emitted_generator_latents=emitted,
        )


class _ToySequenceOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        value: torch.Tensor,
        sequence: SequenceContext,
    ) -> torch.Tensor:
        self.calls += 1
        assert value.shape[:2] == (
            sequence.batch_size,
            sequence.query_length,
        )
        return value


@dataclass(frozen=True)
class _ToyDirectExecution:
    delta: torch.Tensor
    emitted_generator_latents: torch.Tensor
    receipt: _ToyAttentionReceipt


@dataclass(frozen=True)
class _ToyAttentionReceipt:
    invocation_id: str
    generator_kind: str
    attention_kind: str
    input_coordinate_rank: int
    output_coordinate_rank: int
    logical_causal_key_pairs: int
    source_projection_call_count: int
    source_fallback_call_count: int


class _ToyDirectAttentionOperator(nn.Module):
    """Exercises the real attention seam with asymmetric packet ranks."""

    hidden_width = _WIDTH
    chart_count = 1
    incoming_chart_count = 1
    emitted_chart_count = 2
    incoming_generator_rank = 2
    emitted_generator_rank = 3

    def __init__(self, *, source_projection_call_count: int = 0) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(_WIDTH), requires_grad=False)
        self.source_projection_call_count = source_projection_call_count
        self.calls = 0

    def execute_with_incoming(
        self,
        state: torch.Tensor,
        sequence: SequenceContext,
        incoming: torch.Tensor,
    ) -> _ToyDirectExecution:
        self.calls += 1
        assert incoming.shape == (
            sequence.batch_size,
            sequence.query_length,
            self.incoming_chart_count,
            self.incoming_generator_rank,
        )
        incoming_delta = incoming.sum(dim=(-2, -1)).unsqueeze(-1)
        emitted = torch.ones(
            *state.shape[:2],
            self.emitted_chart_count,
            self.emitted_generator_rank,
            dtype=state.dtype,
            device=state.device,
        )
        return _ToyDirectExecution(
            delta=self.delta.expand_as(state) + incoming_delta,
            emitted_generator_latents=emitted,
            receipt=_ToyAttentionReceipt(
                invocation_id="mutation.1",
                generator_kind="reduced_coordinates",
                attention_kind="sliding_causal",
                input_coordinate_rank=self.incoming_generator_rank,
                output_coordinate_rank=self.emitted_generator_rank,
                logical_causal_key_pairs=3,
                source_projection_call_count=(
                    self.source_projection_call_count
                ),
                source_fallback_call_count=0,
            ),
        )

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "gemma3_attention_generator",
            "invocation_id": "mutation.1",
            "source_independent_runtime": True,
            "source_fallback": False,
            "attention_contract": {"kind": "sliding_causal"},
        }


class _UntiedLexical(nn.Module):
    def __init__(self, *, dishonest_metadata: bool = False) -> None:
        super().__init__()
        self.ingress_weight = nn.Parameter(
            torch.arange(_VOCABULARY * _WIDTH, dtype=torch.float32).view(
                _VOCABULARY, _WIDTH
            ),
            requires_grad=False,
        )
        self.egress_weight = nn.Parameter(
            self.ingress_weight.detach().clone(),
            requires_grad=False,
        )
        self.dishonest_metadata = dishonest_metadata

    def ingress(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.ingress_weight)

    def egress(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self.egress_weight)

    def metadata(self) -> dict[str, object]:
        return {
            "ingress_and_egress_share_the_same_storage": (
                self.dishonest_metadata
            ),
            "second_head_copy_present": not self.dishonest_metadata,
            "physical_storage_count": 1 if self.dishonest_metadata else 2,
        }


def _lexical() -> SharedTiedLexicalFactor:
    weight = torch.arange(
        _VOCABULARY * _WIDTH,
        dtype=torch.float32,
    ).view(_VOCABULARY, _WIDTH) / 10.0
    return SharedTiedLexicalFactor(exact_weight=weight)


def _reduced_attention_generator() -> nn.Module:
    width = 4
    head_dimension = 2
    norm = NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    config = Gemma3AttentionGeneratorConfig(
        invocation_id="mutation.1",
        residual_width=width,
        attention=AttentionSpec(
            kind="sliding_causal",
            query_heads=2,
            key_value_heads=1,
            head_dimension=head_dimension,
            query_scale=0.5,
            qk_norm=True,
            window_size=512,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=norm,
        output_norm=norm,
        qk_norm=NormalizationSpec(
            kind="rms_norm",
            width=head_dimension,
            epsilon=1e-6,
            affine=True,
            scale_parameterization="unit_offset",
            compute_dtype="float32",
        ),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    torch.manual_seed(1729)
    exact = compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.zeros(width, dtype=torch.float64),
            q_weight=torch.randn(width, width, dtype=torch.float64),
            k_weight=torch.randn(
                head_dimension,
                width,
                dtype=torch.float64,
            ),
            v_weight=torch.randn(
                head_dimension,
                width,
                dtype=torch.float64,
            ),
            o_weight=torch.randn(width, width, dtype=torch.float64),
            output_norm_weight=torch.zeros(width, dtype=torch.float64),
            q_norm_weight=torch.zeros(head_dimension, dtype=torch.float64),
            k_norm_weight=torch.zeros(head_dimension, dtype=torch.float64),
        ),
    )
    return compile_svd_reduced_attention_generator(
        exact,
        input_rank=2,
        output_rank=3,
    )


def _mutations(
    *,
    retained_source_event_id: str | None = None,
    shared_generator: bool = False,
) -> tuple[joint.JointGraphGeneratorMutation, ...]:
    shared = _ToyChartGenerator() if shared_generator else None
    result: list[joint.JointGraphGeneratorMutation] = []
    for event in joint.build_joint_cross_block_event_schedule()[1:-2]:
        result.append(
            joint.JointGraphGeneratorMutation(
                event_spec=event,
                input_norm=nn.Identity(),
                chart_generator=(shared or _ToyChartGenerator()),
                output_norm=nn.Identity(),
                sequence_operator=(
                    _ToySequenceOperator()
                    if event.event_kind == "attention_mutation"
                    else None
                ),
                source_checkpoint_coefficients_retained=(
                    event.event_id == retained_source_event_id
                ),
            )
        )
    return tuple(result)


def _mutations_with_direct_attention(
) -> tuple[
    tuple[joint.JointGraphGeneratorMutation, ...],
    _ToyDirectAttentionOperator,
]:
    direct = _ToyDirectAttentionOperator()
    result = list(_mutations())
    first_attention = joint.build_joint_cross_block_event_schedule()[1]
    result[0] = joint.JointGraphGeneratorMutation(
        event_spec=first_attention,
        compiled_operator=direct,
    )
    return tuple(result), direct


def _edge(
    source: str = "mutation.2",
    target: str = "mutation.3",
    *,
    transport_id: str = "transport.cross-block.mlp-to-attention",
    source_chart: int = 0,
    target_chart: int = 0,
) -> joint.JointGraphCrossBlockTransportEdge:
    return joint.JointGraphCrossBlockTransportEdge(
        spec=joint.JointGraphTransportSpec(
            transport_id=transport_id,
            source_event_id=source,
            target_event_id=target,
            source_chart_index=source_chart,
            target_chart_index=target_chart,
        ),
        transport_weight=torch.eye(_GENERATOR_RANK),
    )


def _executor(
    *,
    mutations: tuple[joint.JointGraphGeneratorMutation, ...] | None = None,
    transports: tuple[joint.JointGraphCrossBlockTransportEdge, ...] | None = None,
    lexical: nn.Module | None = None,
    arm_id: str = joint.JOINT_CROSS_BLOCK_CHART_ARM_ID,
) -> joint.Gemma3JointCrossBlockChartExecutor:
    return joint.Gemma3JointCrossBlockChartExecutor(
        lexical=lexical or _lexical(),
        mutations=mutations or _mutations(),
        transports=transports if transports is not None else (_edge(),),
        terminal_norm=nn.Identity(),
        arm_id=arm_id,
    )


def test_schedule_is_one_tied_lexical_boundary_and_all_36_body_mutations() -> None:
    schedule = joint.build_joint_cross_block_event_schedule()

    assert len(schedule) == joint.PINNED_GEMMA3_EVENT_COUNT == 39
    assert [event.event_index for event in schedule] == list(range(39))
    assert schedule[0].event_kind == "lexical_ingress"
    assert schedule[-2].event_kind == "terminal_rmsnorm"
    assert schedule[-1].event_kind == "lexical_egress"
    body = schedule[1:-2]
    assert len(body) == joint.PINNED_GEMMA3_MUTATION_COUNT == 36
    assert [event.event_kind for event in body] == [
        kind
        for _block in range(18)
        for kind in ("attention_mutation", "mlp_mutation")
    ]
    assert [event.block_ordinal for event in body] == [
        block for block in range(18) for _kind in range(2)
    ]
    assert {
        event.block_ordinal
        for event in body
        if event.attention_kind == "global"
    } == {5, 11, 17}
    assert sum(event.attention_kind == "sliding_512" for event in body) == 15


@pytest.mark.parametrize(
    "mutation_transform",
    [
        lambda values: values[:-1],
        lambda values: tuple(reversed(values)),
        lambda values: (values[0], *values),
    ],
)
def test_executor_rejects_incomplete_reordered_or_duplicate_body_coverage(
    mutation_transform: object,
) -> None:
    mutations = _mutations()
    malformed = mutation_transform(mutations)  # type: ignore[operator]
    with pytest.raises(ValueError, match="ordered 36-event body"):
        _executor(mutations=malformed)


def test_candidate_requires_one_edge_that_is_both_cross_block_and_cross_family(
) -> None:
    with pytest.raises(ValueError, match="cross-block, cross-family"):
        _executor(transports=())
    with pytest.raises(ValueError, match="cross-block, cross-family"):
        _executor(transports=(_edge("mutation.1", "mutation.2"),))
    with pytest.raises(ValueError, match="cross-block, cross-family"):
        _executor(transports=(_edge("mutation.1", "mutation.3"),))

    executor = _executor(transports=(_edge("mutation.2", "mutation.3"),))
    manifest = executor.architecture_manifest()
    assert manifest["cross_block_cross_component_transport_count"] == 1


def test_transport_topology_rejects_backward_dangling_duplicate_and_bad_charts(
) -> None:
    with pytest.raises(ValueError, match="strictly forward"):
        _executor(transports=(_edge("mutation.4", "mutation.1"),))
    with pytest.raises(ValueError, match="non-mutation"):
        _executor(transports=(_edge("mutation.2", "terminal.rmsnorm"),))
    with pytest.raises(ValueError, match="endpoint charts"):
        _executor(transports=(_edge(source_chart=_CHART_COUNT),))
    with pytest.raises(ValueError, match="duplicate chart transport"):
        _executor(
            transports=(
                _edge(transport_id="transport.first"),
                _edge(transport_id="transport.second"),
            )
        )


def test_candidate_rejects_native_checkpoint_mutations_and_an_untied_head() -> None:
    retained = _mutations(retained_source_event_id="mutation.7")
    with pytest.raises(ValueError, match="cannot retain source checkpoint"):
        _executor(mutations=retained)
    with pytest.raises(ValueError, match="one tied physical factor"):
        _executor(lexical=_UntiedLexical())
    with pytest.raises(ValueError, match="one tied physical factor"):
        _executor(lexical=_UntiedLexical(dishonest_metadata=True))


def test_manifest_authenticates_every_component_and_keeps_claims_false() -> None:
    executor = _executor(mutations=_mutations(shared_generator=True))
    manifest = executor.architecture_manifest()

    assert manifest["candidate_arm"] is True
    assert manifest["exact_control_arm"] is False
    assert manifest["event_count"] == 39
    assert manifest["attention_mutation_count"] == 18
    assert manifest["mlp_mutation_count"] == 18
    assert manifest["global_attention_block_ordinals"] == (5, 11, 17)
    assert manifest["sliding_attention_mutation_count"] == 15
    assert manifest["attention_sequence_operators_present"] is True
    assert manifest["tied_lexical_ingress_and_egress_present"] is True
    assert manifest["lexical_factor_registered_once"] is True
    assert manifest["generator_family_physical_sharing_present"] is True
    assert manifest["unique_chart_generator_module_count"] == 1
    assert manifest["residual_carrier"] == {
        "mutation_count": 36,
        "initial_version": 0,
        "final_version": 36,
        "identity_preserved_by_ordered_delta_updates": True,
        "factory_kind": "DenseResidualCarrierFactory",
    }
    assert manifest["transport_edge_count"] == 1
    assert manifest["cross_block_transport_count"] == 1
    assert manifest["cross_component_family_transport_count"] == 1
    assert manifest["cross_block_cross_component_transport_count"] == 1
    assert manifest["live_teacher_reference_present"] is False
    assert manifest["native_fallback_present"] is False
    assert manifest["native_layer_objects_present"] is False
    assert manifest["source_checkpoint_coefficients_retained_by_mutations"] is False
    assert manifest["compression_success_claimed"] is False
    assert manifest["source_free_artifact_established"] is False


def test_resource_inventory_counts_physical_sharing_once_and_closes() -> None:
    executor = _executor(mutations=_mutations(shared_generator=True))
    resources = executor.resource_inventory()

    assert resources["lexical_shared_physical_scalars"] == 15
    assert resources["mutation_scalars"] == 7
    assert resources["transport_scalars"] == 4
    assert resources["terminal_norm_scalars"] == 0
    assert resources["candidate_deployed_scalars"] == 26
    assert resources["candidate_deployed_scalars"] == executor.deployed_scalar_count
    assert resources["inventory_closes"] is True
    assert resources["lexical_ingress_egress_factor_counted_once"] is True
    assert resources["serialized_artifact_written"] is False
    assert resources["source_unique_parameter_reduction_claimed"] is False
    assert resources["whole_model_mac_reduction_claimed"] is False
    assert resources["latency_reduction_claimed"] is False


def test_exact_control_executes_the_complete_ordered_residual_carrier() -> None:
    mutations = _mutations()
    executor = _executor(
        mutations=mutations,
        transports=(),
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    sequence = _sequence()

    execution = executor.forward_components(input_ids, sequence)

    initial = executor.lexical.ingress(input_ids)
    expected = initial.clone()
    valid = sequence.query_valid_mask.unsqueeze(-1)
    for mutation in mutations:
        delta = mutation.chart_generator.delta.expand_as(expected)
        expected = expected + delta.masked_fill(~valid, 0)
    assert torch.equal(execution.output, expected)
    assert torch.equal(execution.logits, executor.lexical.egress(expected))
    assert execution.event_ids == executor.ordered_event_ids
    assert execution.carrier_version == 36
    assert [receipt.version_before for receipt in execution.mutation_receipts] == list(
        range(36)
    )
    assert [receipt.version_after for receipt in execution.mutation_receipts] == list(
        range(1, 37)
    )
    assert tuple(
        receipt.mutation_id for receipt in execution.mutation_receipts
    ) == executor.expected_mutation_ids
    assert execution.transport_edge_count == 0


def test_candidate_traverses_attention_mlp_and_cross_block_transport() -> None:
    mutations = _mutations()
    executor = _executor(mutations=mutations)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    sequence = _sequence()

    execution = executor.forward_components(input_ids, sequence)

    attention = [
        mutation
        for mutation in mutations
        if mutation.event_spec.event_kind == "attention_mutation"
    ]
    assert all(mutation.sequence_operator.calls == 1 for mutation in attention)
    assert all(mutation.chart_generator.calls == 1 for mutation in mutations)
    assert execution.transport_edge_count == 1
    assert execution.peak_live_transport_packets == 1
    assert execution.peak_live_transport_packet_bytes > 0
    assert torch.equal(
        execution.output[0, 2],
        executor.lexical.ingress(input_ids)[0, 2],
    )


def test_direct_attention_path_supports_asymmetric_cross_block_packets() -> None:
    mutations, direct = _mutations_with_direct_attention()
    edge = joint.JointGraphCrossBlockTransportEdge(
        spec=joint.JointGraphTransportSpec(
            transport_id="transport.direct-attention-to-next-block-mlp",
            source_event_id="mutation.1",
            target_event_id="mutation.4",
            source_chart_index=1,
            target_chart_index=0,
        ),
        transport_weight=torch.ones(3, 2),
    )
    executor = _executor(mutations=mutations, transports=(edge,))

    execution = executor.forward_components(
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        _sequence(),
    )

    assert direct.calls == 1
    assert execution.transport_edge_count == 1
    first_receipt = execution.operator_receipts[0]
    assert first_receipt.event_id == "mutation.1"
    assert first_receipt.operator_receipt_present is True
    assert first_receipt.generator_kind == "reduced_coordinates"
    assert first_receipt.source_projection_call_count == 0
    assert first_receipt.source_fallback_call_count == 0
    metadata = mutations[0].metadata()
    assert metadata["direct_compiled_operator_present"] is True
    assert metadata["incoming_generator_rank"] == 2
    assert metadata["emitted_generator_rank"] == 3


def test_real_reduced_attention_runs_as_one_direct_asymmetric_mutation() -> None:
    generator = _reduced_attention_generator()
    event = joint.build_joint_cross_block_event_schedule()[1]
    mutation = joint.JointGraphGeneratorMutation(
        event_spec=event,
        compiled_operator=generator,
    )
    sequence = _sequence(valid=torch.ones(1, 4, dtype=torch.bool))
    state = torch.randn(1, 4, 4, dtype=torch.float64)
    incoming = torch.zeros(1, 4, 1, 2, dtype=torch.float64)

    execution = mutation.execute_with_incoming(state, sequence, incoming)

    assert execution.delta.shape == state.shape
    assert execution.emitted_generator_latents.shape == (1, 4, 1, 3)
    assert mutation.incoming_generator_rank == 2
    assert mutation.emitted_generator_rank == 3
    assert mutation.metadata()["direct_compiled_operator_present"] is True
    assert execution.runtime_receipt.operator_receipt_present is True
    assert execution.runtime_receipt.source_projection_call_count == 0
    assert execution.runtime_receipt.source_fallback_call_count == 0


def test_direct_attention_runtime_fails_closed_on_a_source_projection_call() -> None:
    event = joint.build_joint_cross_block_event_schedule()[1]
    mutation = joint.JointGraphGeneratorMutation(
        event_spec=event,
        compiled_operator=_ToyDirectAttentionOperator(
            source_projection_call_count=1
        ),
    )
    state = torch.randn(1, 3, _WIDTH)
    incoming = torch.zeros(1, 3, 1, 2)

    with pytest.raises(RuntimeError, match="runtime receipt failed"):
        mutation.execute_with_incoming(state, _sequence(), incoming)


def test_factorized_hidden_identity_uses_no_dense_identity_storage() -> None:
    generator = joint.FactorizedHiddenChartGenerator.identity_zero_local_control(
        hidden_width=_WIDTH,
        chart_count=_CHART_COUNT,
        chart_rank=2,
        generator_rank=_GENERATOR_RANK,
    )
    features = torch.randn(1, 3, _WIDTH)
    incoming = torch.zeros(1, 3, _CHART_COUNT, _GENERATOR_RANK)

    execution = generator.execute_with_incoming(features, incoming)

    assert torch.equal(execution.output, features)
    assert torch.count_nonzero(execution.emitted_generator_latents) == 0
    assert generator.dense_base_scalar_count == 0
    assert generator.metadata()["dense_identity_matrix_stored"] is False


def test_exact_token_local_control_rejects_nonzero_transport_messages() -> None:
    generator = joint.ExactTokenLocalChartGenerator(
        operator=nn.Identity(),
        hidden_width=_WIDTH,
        chart_count=_CHART_COUNT,
        generator_rank=_GENERATOR_RANK,
    )
    features = torch.randn(1, 3, _WIDTH)
    zero = torch.zeros(1, 3, _CHART_COUNT, _GENERATOR_RANK)

    execution = generator.execute_with_incoming(features, zero)
    assert torch.equal(execution.output, features)
    with pytest.raises(ValueError, match="rejects nonzero transport"):
        generator.execute_with_incoming(features, torch.ones_like(zero))
