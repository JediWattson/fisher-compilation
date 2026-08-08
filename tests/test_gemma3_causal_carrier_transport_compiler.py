from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from fisher_graph.adapters import (
    AttentionSpec,
    Gemma3CausalLMAdapter,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph import gemma3_causal_carrier_transport_compiler as transport
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
    compile_exact_attention_generator_from_structured_executor,
)
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph import gemma3_joint_cross_block_compiler as joint_compiler
from fisher_graph.gemma3_full_body_residual_carrier_experiment import (
    _compile_complete_body,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)


_WIDTH = 4
_VOCABULARY = 11


class _ToyRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width), requires_grad=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized * (1.0 + self.weight.float())).to(values.dtype)


def _norm(width: int) -> NormalizationSpec:
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )


def _attention(event: joint.JointGraphEventSpec) -> nn.Module:
    kind = (
        "global_causal"
        if event.attention_kind == "global"
        else "sliding_causal"
    )
    config = Gemma3AttentionGeneratorConfig(
        invocation_id=event.event_id,
        residual_width=_WIDTH,
        attention=AttentionSpec(
            kind=kind,
            query_heads=2,
            key_value_heads=1,
            head_dimension=2,
            query_scale=2**-0.5,
            qk_norm=True,
            window_size=None if kind == "global_causal" else 2,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=2,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=_norm(_WIDTH),
        output_norm=_norm(_WIDTH),
        qk_norm=_norm(2),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    generator = torch.Generator().manual_seed(91_000 + event.event_index)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(rows, columns, generator=generator) / 8.0

    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.zeros(_WIDTH),
            q_weight=matrix(4, _WIDTH),
            k_weight=matrix(2, _WIDTH),
            v_weight=matrix(2, _WIDTH),
            o_weight=matrix(_WIDTH, 4),
            output_norm_weight=torch.zeros(_WIDTH),
            q_norm_weight=torch.zeros(2),
            k_norm_weight=torch.zeros(2),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _toy_exact_executor() -> joint.Gemma3JointCrossBlockChartExecutor:
    schedule = joint.build_joint_cross_block_event_schedule()
    mutations: list[joint.JointGraphGeneratorMutation] = []
    for event in schedule[1:-2]:
        if event.event_kind == "attention_mutation":
            mutations.append(
                joint.JointGraphGeneratorMutation(
                    event_spec=event,
                    compiled_operator=_attention(event),
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            operator = nn.Linear(_WIDTH, _WIDTH, bias=False)
            with torch.no_grad():
                operator.weight.copy_(
                    torch.eye(_WIDTH) * (0.001 * event.event_index)
                )
            mutations.append(
                joint.JointGraphGeneratorMutation(
                    event_spec=event,
                    input_norm=nn.Identity(),
                    chart_generator=joint.ExactTokenLocalChartGenerator(
                        operator=operator,
                        hidden_width=_WIDTH,
                        chart_count=1,
                        generator_rank=_WIDTH,
                    ),
                    output_norm=nn.Identity(),
                    source_checkpoint_coefficients_retained=True,
                )
            )
    lexical_weight = torch.arange(
        _VOCABULARY * _WIDTH,
        dtype=torch.float32,
    ).view(_VOCABULARY, _WIDTH) / 100.0
    return joint.Gemma3JointCrossBlockChartExecutor(
        lexical=SharedTiedLexicalFactor(exact_weight=lexical_weight),
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=_ToyRMSNorm(_WIDTH),
        final_logit_softcap=None,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )


def _sequence() -> SequenceContext:
    valid = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
        ]
    )
    positions = torch.arange(3).expand(2, -1)
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _transport_executor(
) -> transport.Gemma3CausalCarrierTransportExecutor:
    exact = _toy_exact_executor()
    return transport.assemble_gemma3_exact_causal_carrier_transport_executor(
        exact_executor=exact,
        source_model_parameter_count=exact.deployed_scalar_count,
        source_model_fingerprint="a" * 64,
        source_independence_audit={"passed": True},
    )


def test_manifest_is_exact_39_event_carrier_transport_without_compression(
) -> None:
    executor = _transport_executor()
    manifest = executor.architecture_manifest()
    resources = executor.resource_inventory()

    assert manifest["event_count"] == 39
    assert manifest["ordered_event_ids"][0] == "lexical.ingress"
    assert manifest["ordered_event_ids"][-2:] == (
        "terminal.rmsnorm",
        "lexical.egress",
    )
    assert manifest["carrier_boundary_count"] == 36
    assert manifest["identity_carry_message_count"] == 36
    assert manifest["primary_generator_message_count"] == 36
    assert manifest["attention_routed_message_count"] == 18
    assert manifest["mlp_token_local_message_count"] == 18
    assert manifest["global_attention_block_ordinals"] == (5, 11, 17)
    assert manifest["cross_block_correction_message_count"] == 0
    assert manifest["residual_add_node_present"] is False
    assert all(
        boundary["incoming_messages"][0] == {
            "message_id": (
                f"message.{boundary['event']['event_id']}.carry"
            ),
            "target_event_id": boundary["event"]["event_id"],
            "message_kind": "identity_carry",
            "routing_kind": "same_token_identity",
            "transform_kind": "parameter_free_identity",
            "stored_scalar_count": 0,
            "coefficient": 1.0,
        }
        for boundary in manifest["carrier_boundaries"]
    )
    assert manifest["message_count"] == 75
    assert manifest["ordered_message_ids"][0] == "message.lexical.ingress"
    assert manifest["protocol_sha256"] == (
        transport.GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256
    )
    assert manifest["core_transport_manifest"]["accounting"] == {
        "event_count": 36,
        "edge_count": 73,
        "lexical_message_count": 1,
        "identity_carry_count": 36,
        "primary_generator_message_count": 36,
        "forward_correction_count": 0,
        "stored_scalar_count": (
            resources["core_graph_declared_stored_scalars"]
        ),
        "initial_carrier_version": 0,
        "final_carrier_version": 36,
        "commits_per_event": 1,
    }
    assert resources["exact_stored_scalar_parity"] is True
    assert resources["identity_carry_stored_scalars"] == 0
    assert resources["attention_routing_stored_scalars"] == 0
    assert resources["cross_block_correction_scalars"] == 0
    assert resources["compression_attempted"] is False


def test_execution_authenticates_carry_generator_and_dynamic_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _transport_executor()
    input_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    sequence = _sequence()
    baseline = executor.exact_executor.forward_components(input_ids, sequence)
    monkeypatch.setattr(
        executor.exact_executor,
        "forward_components",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport view delegated to exact full forward")
        ),
    )
    monkeypatch.setattr(
        executor.exact_executor,
        "forward_components_with_observer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport view delegated to observer forward")
        ),
    )
    observed_messages: list[tuple[str, str]] = []
    observed_routing: list[torch.Tensor] = []

    def observer(**payload: object) -> None:
        values = payload["values"]
        assert isinstance(values, torch.Tensor)
        routing = payload.get("routing_weights")
        if isinstance(routing, torch.Tensor):
            observed_routing.append(routing.clone())
            routing.zero_()
        values.zero_()
        observed_messages.append(
            (str(payload["message_id"]), str(payload["message_kind"]))
        )

    execution = executor.execute(
        input_ids,
        sequence,
        observer=observer,
    )
    receipt = execution.receipt()

    assert torch.equal(execution.output, baseline.output)
    assert torch.equal(execution.normalized_output, baseline.normalized_output)
    assert torch.equal(execution.logits, baseline.logits)
    assert execution.mutation_delta_sha256s == (
        baseline.mutation_delta_sha256s
    )
    assert execution.mutation_chart_input_sha256s == (
        baseline.mutation_chart_input_sha256s
    )
    assert receipt["event_count"] == 39
    assert receipt["message_count"] == 75
    assert len(receipt["ordered_message_ids"]) == 75
    assert receipt["initial_carrier_version"] == 0
    assert receipt["final_carrier_version"] == 36
    assert receipt["identity_carry_message_count"] == 36
    assert receipt["primary_generator_message_count"] == 36
    assert receipt["attention_routed_message_count"] == 18
    assert receipt["correction_message_count"] == 0
    assert receipt["source_projection_call_count"] == 0
    assert receipt["source_fallback_call_count"] == 0
    assert receipt["residual_add_node_present"] is False
    assert receipt["core_live_carrier_transport"]["claims"] == {
        "live_carrier_advanced_by_core": True,
        "one_aggregate_commit_per_event": True,
        "identity_carry_exact": True,
        "residual_carrier_closure": True,
        "post_hoc_reconstruction": False,
        "compression": False,
    }
    assert receipt["core_live_carrier_transport"]["accounting"] == {
        "callback_call_count": 37,
        "identity_carry_count": 36,
        "primary_generator_message_count": 36,
        "correction_message_count": 0,
        "aggregate_commit_count": 36,
        "stored_scalar_count": (
            executor.transport_manifest.stored_scalar_count
        ),
    }
    assert receipt["routing_receipt_replay_logical_macs"] > 0
    assert receipt["routing_receipt_peak_dynamic_bytes"] > 0
    assert receipt["core_carrier_snapshot_bytes"] > 0
    assert receipt["analysis_peak_dynamic_bytes_lower_bound"] > 0
    assert len(execution.mutation_receipts) == 36
    assert tuple(
        item["event_kind"]
        for item in receipt["mutation_receipts"][:2]
    ) == (
        "attention_generator_message",
        "token_local_mlp_generator_message",
    )
    assert all(
        item.input_carrier_version == index
        and item.output_carrier_version == index + 1
        for index, item in enumerate(execution.mutation_receipts)
    )
    assert all(
        left.aggregate_carrier_sha256 == right.identity_carry_sha256
        for left, right in zip(
            execution.mutation_receipts,
            execution.mutation_receipts[1:],
        )
    )
    attention = tuple(
        item.attention_routing
        for item in execution.mutation_receipts
        if item.attention_routing is not None
    )
    assert len(attention) == 18
    assert tuple(
        item.block_ordinal
        for item in attention
        if item.attention_kind == "global_causal"
    ) == (5, 11, 17)
    assert all(item.maximum_row_sum_error <= 1e-6 for item in attention)
    assert all(item.maximum_forbidden_probability == 0.0 for item in attention)
    assert all(item.source_projection_call_count == 0 for item in attention)
    assert len(observed_routing) == 18
    first_routing = observed_routing[0]
    invalid_rows = ~sequence.query_valid_mask.unsqueeze(1).unsqueeze(-1)
    assert torch.count_nonzero(first_routing.masked_select(invalid_rows)) == 0
    row_sums = first_routing.sum(dim=-1)
    valid_rows = sequence.query_valid_mask.unsqueeze(1).expand_as(row_sums)
    assert torch.allclose(
        row_sums[valid_rows],
        torch.ones_like(row_sums[valid_rows]),
        rtol=0.0,
        atol=1e-6,
    )
    assert first_routing[:, :, 2, 0].count_nonzero() == 0
    assert torch.triu(first_routing, diagonal=1).count_nonzero() == 0
    assert len(observed_messages) == 75
    assert observed_messages[0] == (
        "message.lexical.ingress",
        "lexical_ingress",
    )
    assert observed_messages[-2:] == [
        ("message.terminal.rmsnorm", "terminal_normalization"),
        ("message.lexical.egress", "lexical_egress"),
    ]


def test_literal_valid_query_parity_is_zero_tolerance_and_masks_padding(
) -> None:
    executor = _transport_executor()
    input_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    sequence = _sequence()
    execution = executor.execute(input_ids, sequence)
    native = execution.logits.clone()
    native[1, 0] += 100.0

    passing = transport.verify_literal_valid_query_logit_parity(
        native_logits=native,
        compiled_execution=execution,
        valid_query_mask=sequence.query_valid_mask,
    )
    assert passing.passed is True
    assert passing.compared_valid_query_rows == 5
    assert passing.maximum_absolute_valid_query_logit_error == 0.0

    native[0, 1, 0] = torch.nextafter(
        native[0, 1, 0],
        torch.tensor(float("inf")),
    )
    failing = transport.verify_literal_valid_query_logit_parity(
        native_logits=native,
        compiled_execution=execution,
        valid_query_mask=sequence.query_valid_mask,
    )
    assert failing.passed is False
    assert failing.maximum_absolute_valid_query_logit_error > 0.0


def _tiny_hf_transport_executor(
) -> tuple[
    Gemma3CausalLMAdapter,
    transport.Gemma3CausalCarrierTransportExecutor,
]:
    transformers = pytest.importorskip("transformers")
    layer_types = [
        "full_attention" if block in (5, 11, 17) else "sliding_attention"
        for block in range(18)
    ]
    config = transformers.Gemma3TextConfig(
        vocab_size=23,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=18,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=3,
        layer_types=layer_types,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(63_004)
    model = transformers.Gemma3ForCausalLM(config).eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    structured = tuple(
        block.executor
        for block in _compile_complete_body(
            adapter,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).blocks
    )
    schedule = joint.build_joint_cross_block_event_schedule()
    mutations: list[joint.JointGraphGeneratorMutation] = []
    for block, layer in enumerate(structured):
        attention_event = schedule[1 + 2 * block]
        mlp_event = schedule[2 + 2 * block]
        mutations.append(
            joint.JointGraphGeneratorMutation(
                event_spec=attention_event,
                compiled_operator=(
                    compile_exact_attention_generator_from_structured_executor(
                        layer,
                        invocation_id=attention_event.event_id,
                    )
                ),
                source_checkpoint_coefficients_retained=True,
            )
        )
        mutations.append(
            joint.JointGraphGeneratorMutation(
                event_spec=mlp_event,
                input_norm=layer.feed_forward_input_norm,
                chart_generator=joint.ExactTokenLocalChartGenerator(
                    operator=joint_compiler._ExactStructuredFeedForwardOperator(
                        layer.feed_forward,
                        prefix=f"{mlp_event.event_id}.mlp",
                    ),
                    hidden_width=8,
                    chart_count=1,
                    generator_rank=8,
                ),
                output_norm=layer.feed_forward_output_norm,
                source_checkpoint_coefficients_retained=True,
            )
        )
    lexical = joint_compiler._exact_lexical_factor(adapter)
    terminal, softcap = joint_compiler._terminal_norm_and_softcap(adapter)
    exact = joint.Gemma3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=terminal,
        final_logit_softcap=softcap,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )
    source_count = sum(parameter.numel() for parameter in model.parameters())
    assert exact.deployed_scalar_count == source_count
    executor = transport.assemble_gemma3_exact_causal_carrier_transport_executor(
        exact_executor=exact,
        source_model_parameter_count=source_count,
        source_model_fingerprint=adapter.model_fingerprint(),
        source_independence_audit={"passed": True},
    )
    return adapter, executor


def test_optional_tiny_hf_18_block_literal_valid_query_parity() -> None:
    adapter, executor = _tiny_hf_transport_executor()
    model_inputs = {
        "input_ids": torch.tensor(
            [
                [0, 2, 7, 3],
                [2, 8, 9, 4],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [False, True, True, True],
                [True, True, True, True],
            ]
        ),
    }

    receipt = transport.verify_gemma3_native_literal_valid_query_parity(
        adapter=adapter,
        executor=executor,
        model_inputs=model_inputs,
    )

    assert receipt.passed is True
    assert receipt.compared_valid_query_rows == 7
    assert receipt.maximum_absolute_valid_query_logit_error == 0.0
    assert executor.resource_inventory()["exact_stored_scalar_parity"] is True
    assert executor.architecture_manifest()["native_fallback_present"] is False


def test_compilation_metadata_never_claims_compression_or_execution() -> None:
    executor = _transport_executor()
    compilation = transport.Gemma3CausalCarrierTransportCompilation(
        executor=executor
    )
    metadata = compilation.metadata()

    assert metadata["experiment_executed"] is False
    assert metadata["metrics_present"] is False
    assert metadata["compression_attempted"] is False
    assert metadata["compression_success_claimed"] is False
    assert metadata["model_mutated"] is False
    assert metadata["resource_inventory"]["exact_stored_scalar_parity"] is True
    assert compilation.reference_executor is executor.exact_executor
    assert copy.deepcopy(metadata) == metadata
