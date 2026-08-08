from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import json

import pytest
import torch
from torch import Tensor

from fisher_graph.causal_carrier_transport_graph import (
    CARRIER_TRANSPORT_MANIFEST_SCHEMA,
    CARRIER_TRANSPORT_RECEIPT_SCHEMA,
    CarrierEventSpec,
    CarrierTransportExecutionReceipt,
    CarrierTransportManifest,
    ForwardCorrectionSpec,
    TransportContext,
    TransportMessage,
    build_causal_carrier_transport_manifest,
    execute_causal_carrier_transport_graph,
    validate_carrier_transport_execution_receipt,
    validate_carrier_transport_manifest,
)


def _events() -> tuple[CarrierEventSpec, ...]:
    return (
        CarrierEventSpec(
            event_index=1,
            event_id="mutation.1",
            event_kind="attention",
            generator_invocation_id="attention.call.0",
            generator_family_id="attention.shared",
            input_carrier_version=0,
            output_carrier_version=1,
        ),
        CarrierEventSpec(
            event_index=2,
            event_id="mutation.2",
            event_kind="mlp",
            generator_invocation_id="mlp.call.0",
            generator_family_id="mlp.shared",
            input_carrier_version=1,
            output_carrier_version=2,
        ),
        CarrierEventSpec(
            event_index=3,
            event_id="mutation.3",
            event_kind="attention",
            generator_invocation_id="attention.call.1",
            generator_family_id="attention.shared",
            input_carrier_version=2,
            output_carrier_version=3,
        ),
    )


def _manifest(*, with_correction: bool = False) -> CarrierTransportManifest:
    corrections = (
        ForwardCorrectionSpec(
            correction_id="skip.0.to.2",
            generator_invocation_id="correction.call.0",
            generator_family_id="correction.shared",
            source_event_id="mutation.1",
            target_event_id="mutation.3",
        ),
    ) if with_correction else ()
    return build_causal_carrier_transport_manifest(
        lexical_invocation_id="lexical.call.0",
        lexical_generator_family_id="lexical.shared",
        events=_events(),
        corrections=corrections,
        lexical_stored_scalar_count=16,
        primary_stored_scalar_counts={
            "mutation.1": 12,
            "mutation.2": 8,
            "mutation.3": 12,
        },
        correction_stored_scalar_counts=(
            {"skip.0.to.2": 4} if with_correction else {}
        ),
    )


_EMBEDDING = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, -0.5],
        [0.25, 0.75],
        [-0.5, 1.25],
    ],
    dtype=torch.float64,
)


def _attention_message(carrier: Tensor) -> tuple[Tensor, Tensor]:
    scores = carrier.mean(dim=-1)
    logits = scores.unsqueeze(-1) * scores.unsqueeze(-2)
    sequence = carrier.shape[-2]
    causal_mask = torch.triu(
        torch.ones(sequence, sequence, dtype=torch.bool),
        diagonal=1,
    )
    logits = logits.masked_fill(causal_mask, float("-inf"))
    routing = torch.softmax(logits, dim=-1).unsqueeze(1)
    routed = torch.einsum("bhqk,bkd->bqhd", routing, carrier).mean(dim=-2)
    return routed * 0.2, routing


def _producer(context: TransportContext) -> TransportMessage:
    kind = context.edge.message_kind
    if kind == "lexical_ingress":
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=_EMBEDDING[context.token_input],
        )
    assert context.carrier_input is not None
    if kind == "attention_transport":
        delta, routing = _attention_message(context.carrier_input)
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=delta,
            attention_routing_weights=routing,
        )
    if kind == "mlp_mutation":
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=0.1 * context.carrier_input.square(),
        )
    if kind == "forward_correction":
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=0.05 * context.source_value,
        )
    raise AssertionError(f"unexpected callback edge kind {kind}")


def _producers(manifest: CarrierTransportManifest):
    return {
        edge.edge_id: _producer
        for edge in manifest.edges
        if edge.message_kind != "identity_carry"
    }


def _manual(token_input: Tensor, *, correction: bool) -> tuple[Tensor, ...]:
    state0 = _EMBEDDING[token_input]
    attention0, _ = _attention_message(state0)
    state1 = state0 + attention0
    state2 = state1 + 0.1 * state1.square()
    attention1, _ = _attention_message(state2)
    state3 = state2 + attention1
    if correction:
        state3 = state3 + 0.05 * state1
    return state0, state1, state2, state3


def test_manifest_unifies_messages_and_keeps_family_distinct_from_call() -> None:
    manifest = _manifest()
    assert manifest.schema == CARRIER_TRANSPORT_MANIFEST_SCHEMA
    assert manifest.initial_carrier_version == 0
    assert manifest.final_carrier_version == 3
    assert manifest.correction_count == 0
    assert len(manifest.edges) == 1 + 2 * len(manifest.events)
    assert [edge.message_kind for edge in manifest.edges] == [
        "lexical_ingress",
        "identity_carry",
        "attention_transport",
        "identity_carry",
        "mlp_mutation",
        "identity_carry",
        "attention_transport",
    ]
    assert all(
        edge.fixed_weight == 1.0 and edge.stored_scalar_count == 0
        for edge in manifest.edges
        if edge.message_kind == "identity_carry"
    )
    attention_events = [
        event for event in manifest.events if event.event_kind == "attention"
    ]
    assert attention_events[0].generator_family_id == (
        attention_events[1].generator_family_id
    )
    assert attention_events[0].generator_invocation_id != (
        attention_events[1].generator_invocation_id
    )
    payload = manifest.to_dict()
    assert payload["semantics"]["generator_family_is_invocation_identity"] is False
    assert payload["accounting"]["commits_per_event"] == 1
    assert json.loads(json.dumps(payload)) == payload
    assert CarrierTransportManifest.from_dict(payload) == manifest
    assert validate_carrier_transport_manifest(payload) == manifest


def test_manifest_fingerprint_authenticates_all_topology_and_accounting() -> None:
    manifest = _manifest(with_correction=True)
    first = manifest.to_dict()
    second = _manifest(with_correction=True).to_dict()
    assert first == second
    assert first["manifest_sha256"] == manifest.fingerprint_sha256
    assert first["accounting"] == {
        "event_count": 3,
        "edge_count": 8,
        "lexical_message_count": 1,
        "identity_carry_count": 3,
        "primary_generator_message_count": 3,
        "forward_correction_count": 1,
        "stored_scalar_count": 40,
        "initial_carrier_version": 0,
        "final_carrier_version": 3,
        "commits_per_event": 1,
    }
    tampered = deepcopy(first)
    tampered["accounting"]["edge_count"] = 9
    with pytest.raises(ValueError, match="authentication failed"):
        CarrierTransportManifest.from_dict(tampered)
    extra = deepcopy(first)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields are not exact"):
        CarrierTransportManifest.from_dict(extra)


def test_builder_rejects_broken_carrier_closure_and_future_corrections() -> None:
    broken = replace(
        _events()[1],
        input_carrier_version=0,
        output_carrier_version=1,
    )
    with pytest.raises(ValueError, match="consume every carrier version"):
        build_causal_carrier_transport_manifest(
            lexical_invocation_id="lexical.call",
            lexical_generator_family_id="lexical.family",
            events=(_events()[0], broken, _events()[2]),
        )
    future = ForwardCorrectionSpec(
        correction_id="backward",
        generator_invocation_id="correction.call",
        generator_family_id="correction.family",
        source_event_id="mutation.3",
        target_event_id="mutation.1",
    )
    with pytest.raises(ValueError, match="strictly forward"):
        build_causal_carrier_transport_manifest(
            lexical_invocation_id="lexical.call",
            lexical_generator_family_id="lexical.family",
            events=_events(),
            corrections=(future,),
        )


def test_generator_family_cannot_be_conflated_with_temporal_invocation() -> None:
    with pytest.raises(ValueError, match="family identity must differ"):
        replace(
            _events()[0],
            generator_family_id=_events()[0].generator_invocation_id,
        )
    with pytest.raises(ValueError, match="family identity must differ"):
        build_causal_carrier_transport_manifest(
            lexical_invocation_id="lexical.same",
            lexical_generator_family_id="lexical.same",
            events=_events(),
        )


def test_live_executor_exactly_matches_native_residual_schedule() -> None:
    manifest = _manifest()
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=tokens,
        message_producers=_producers(manifest),
        execution_nonce="toy-exact-0",
    )
    expected = _manual(tokens, correction=False)
    assert len(execution.carrier_snapshots) == 4
    for actual, wanted in zip(execution.carrier_snapshots, expected, strict=True):
        assert torch.equal(actual, wanted)
    assert torch.equal(execution.final_carrier, expected[-1])
    receipt = execution.receipt
    assert receipt.schema == CARRIER_TRANSPORT_RECEIPT_SCHEMA
    assert receipt.callback_call_count == 4
    assert receipt.identity_carry_count == 3
    assert receipt.primary_generator_message_count == 3
    assert receipt.correction_message_count == 0
    assert receipt.aggregate_commit_count == 3
    assert receipt.stored_scalar_count == 36
    assert receipt.payload()["claims"] == {
        "live_carrier_advanced_by_core": True,
        "one_aggregate_commit_per_event": True,
        "identity_carry_exact": True,
        "residual_carrier_closure": True,
        "post_hoc_reconstruction": False,
        "compression": False,
    }
    assert all(boundary.aggregate_commit_count == 1 for boundary in receipt.boundaries)
    assert all(
        boundary.message_receipts[0].value_sha256
        == boundary.input_carrier_sha256
        for boundary in receipt.boundaries
    )
    attention_receipts = [
        item
        for boundary in receipt.boundaries
        for item in boundary.message_receipts
        if item.message_kind == "attention_transport"
    ]
    assert len(attention_receipts) == 2
    assert all(item.routing_sha256 is not None for item in attention_receipts)
    assert json.loads(json.dumps(receipt.to_dict())) == receipt.to_dict()
    assert receipt.to_dict()["receipt_sha256"] == receipt.fingerprint_sha256
    assert validate_carrier_transport_execution_receipt(
        receipt,
        manifest=manifest,
    ) is receipt
    loaded = CarrierTransportExecutionReceipt.from_dict(receipt.to_dict())
    assert loaded.to_dict() == receipt.to_dict()
    assert isinstance(loaded.final_carrier_metadata["shape"], tuple)
    with pytest.raises(TypeError):
        loaded.final_carrier_metadata["shape"] = (99,)
    assert validate_carrier_transport_execution_receipt(
        receipt.to_dict(),
        manifest=manifest,
    ).to_dict() == receipt.to_dict()


def test_optional_forward_correction_reads_prior_snapshot_before_one_commit() -> None:
    manifest = _manifest(with_correction=True)
    tokens = torch.tensor([[1, 3, 2]], dtype=torch.long)
    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=tokens,
        message_producers=_producers(manifest),
        execution_nonce="toy-correction-0",
    )
    expected = _manual(tokens, correction=True)
    assert torch.equal(execution.final_carrier, expected[-1])
    assert execution.receipt.callback_call_count == 5
    assert execution.receipt.correction_message_count == 1
    last = execution.receipt.boundaries[-1]
    assert [item.message_kind for item in last.message_receipts] == [
        "identity_carry",
        "attention_transport",
        "forward_correction",
    ]
    assert last.aggregate_commit_count == 1


def test_executor_requires_exact_producer_coverage_and_typed_messages() -> None:
    manifest = _manifest()
    tokens = torch.tensor([[1, 2, 3]])
    producers = _producers(manifest)
    producers.pop("edge.mutation.2.generator")
    with pytest.raises(ValueError, match="exactly cover"):
        execute_causal_carrier_transport_graph(
            manifest,
            token_input=tokens,
            message_producers=producers,
            execution_nonce="missing-callback",
        )
    producers = _producers(manifest)
    producers["extra.edge"] = _producer
    with pytest.raises(ValueError, match="exactly cover"):
        execute_causal_carrier_transport_graph(
            manifest,
            token_input=tokens,
            message_producers=producers,
            execution_nonce="extra-callback",
        )


def test_executor_rejects_callback_state_replacement_and_context_mutation() -> None:
    manifest = _manifest()
    tokens = torch.tensor([[1, 2, 3]])
    producers = _producers(manifest)

    def wrong_edge(context: TransportContext) -> TransportMessage:
        message = _producer(context)
        return TransportMessage(edge_id="edge.wrong", value=message.value)

    producers["edge.mutation.2.generator"] = wrong_edge
    with pytest.raises(ValueError, match="wrong edge"):
        execute_causal_carrier_transport_graph(
            manifest,
            token_input=tokens,
            message_producers=producers,
            execution_nonce="wrong-edge",
        )

    producers = _producers(manifest)

    def mutating(context: TransportContext) -> TransportMessage:
        assert context.carrier_input is not None
        context.carrier_input.add_(1.0)
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=torch.zeros_like(context.carrier_input),
        )

    producers["edge.mutation.2.generator"] = mutating
    with pytest.raises(RuntimeError, match="mutated its context"):
        execute_causal_carrier_transport_graph(
            manifest,
            token_input=tokens,
            message_producers=producers,
            execution_nonce="mutation-attempt",
        )


@pytest.mark.parametrize("defect", ["missing", "noncausal", "nonstochastic"])
def test_attention_routing_evidence_fails_closed(defect: str) -> None:
    manifest = _manifest()
    tokens = torch.tensor([[1, 2, 3]])
    producers = _producers(manifest)

    def invalid_attention(context: TransportContext) -> TransportMessage:
        assert context.carrier_input is not None
        value = torch.zeros_like(context.carrier_input)
        if defect == "missing":
            routing = None
        elif defect == "noncausal":
            routing = torch.tensor(
                [[[[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]],
                dtype=context.carrier_input.dtype,
            )
        else:
            routing = torch.zeros(
                1,
                1,
                3,
                3,
                dtype=context.carrier_input.dtype,
            )
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=value,
            attention_routing_weights=routing,
        )

    producers["edge.mutation.1.generator"] = invalid_attention
    expected = {
        "missing": "require routing",
        "noncausal": "must be causal",
        "nonstochastic": "at least one valid row",
    }[defect]
    with pytest.raises(ValueError, match=expected):
        execute_causal_carrier_transport_graph(
            manifest,
            token_input=tokens,
            message_producers=producers,
            execution_nonce=f"invalid-{defect}",
        )


def test_attention_routing_allows_zero_inert_left_padding_rows() -> None:
    manifest = _manifest()
    tokens = torch.tensor([[0, 1, 2]])
    producers = _producers(manifest)

    def padded_attention(context: TransportContext) -> TransportMessage:
        assert context.carrier_input is not None
        routing = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]]]],
            dtype=context.carrier_input.dtype,
        )
        value = torch.einsum(
            "bhqk,bkd->bqhd",
            routing,
            context.carrier_input,
        ).mean(dim=-2)
        return TransportMessage(
            edge_id=context.edge.edge_id,
            value=value,
            attention_routing_weights=routing,
        )

    producers["edge.mutation.1.generator"] = padded_attention
    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=tokens,
        message_producers=producers,
        execution_nonce="left-padding",
    )
    assert torch.isfinite(execution.final_carrier).all()


def test_live_executor_preserves_autograd_through_carry_and_messages() -> None:
    manifest = _manifest()
    tokens = torch.tensor([[1, 2, 3]])
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def differentiable(context: TransportContext) -> TransportMessage:
        message = _producer(context)
        if context.edge.message_kind == "lexical_ingress":
            return TransportMessage(
                edge_id=context.edge.edge_id,
                value=message.value * scale,
            )
        return message

    producers = {
        edge.edge_id: differentiable
        for edge in manifest.edges
        if edge.message_kind != "identity_carry"
    }
    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=tokens,
        message_producers=producers,
        execution_nonce="autograd",
    )
    execution.final_carrier.square().sum().backward()
    assert scale.grad is not None
    assert torch.isfinite(scale.grad)
    assert float(scale.grad) != 0.0


def test_execution_receipt_rejects_accounting_and_manifest_drift() -> None:
    manifest = _manifest()
    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=torch.tensor([[1, 2, 3]]),
        message_producers=_producers(manifest),
        execution_nonce="receipt-drift",
    )
    with pytest.raises(ValueError, match="does not close"):
        replace(
            execution.receipt,
            callback_call_count=execution.receipt.callback_call_count + 1,
        )
    other = build_causal_carrier_transport_manifest(
        lexical_invocation_id="lexical.call.other",
        lexical_generator_family_id="lexical.shared",
        events=_events(),
    )
    with pytest.raises(ValueError, match="different manifest"):
        validate_carrier_transport_execution_receipt(
            execution.receipt,
            manifest=other,
        )


def test_exact_hashing_supports_bfloat16_carriers() -> None:
    event = CarrierEventSpec(
        event_index=1,
        event_id="mutation.1",
        event_kind="mlp",
        generator_invocation_id="mlp.call",
        generator_family_id="mlp.family",
        input_carrier_version=0,
        output_carrier_version=1,
    )
    manifest = build_causal_carrier_transport_manifest(
        lexical_invocation_id="lexical.call",
        lexical_generator_family_id="lexical.family",
        events=(event,),
    )

    def bf16(context: TransportContext) -> TransportMessage:
        if context.edge.message_kind == "lexical_ingress":
            value = torch.ones(1, 2, 3, dtype=torch.bfloat16)
        else:
            assert context.carrier_input is not None
            value = torch.zeros_like(context.carrier_input)
        return TransportMessage(edge_id=context.edge.edge_id, value=value)

    execution = execute_causal_carrier_transport_graph(
        manifest,
        token_input=torch.tensor([[1, 2]]),
        message_producers={
            edge.edge_id: bf16
            for edge in manifest.edges
            if edge.message_kind != "identity_carry"
        },
        execution_nonce="bf16",
    )
    assert execution.final_carrier.dtype == torch.bfloat16
    assert len(execution.receipt.final_carrier_metadata["sha256"]) == 64


def test_receipt_constructor_is_fail_closed() -> None:
    manifest = _manifest()
    receipt = execute_causal_carrier_transport_graph(
        manifest,
        token_input=torch.tensor([[1, 2, 3]]),
        message_producers=_producers(manifest),
        execution_nonce="receipt-constructor",
    ).receipt
    values = {
        field: getattr(receipt, field)
        for field in CarrierTransportExecutionReceipt.__dataclass_fields__
    }
    values["stored_scalar_count"] += 1
    with pytest.raises(ValueError, match="scalar accounting"):
        CarrierTransportExecutionReceipt(**values)
    tampered = deepcopy(receipt.to_dict())
    tampered["claims"]["post_hoc_reconstruction"] = True
    with pytest.raises(ValueError, match="claims are not exact"):
        CarrierTransportExecutionReceipt.from_dict(tampered)
    tampered = deepcopy(receipt.to_dict())
    tampered["boundaries"][0]["message_receipts"][0]["value_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="identity carry"):
        CarrierTransportExecutionReceipt.from_dict(tampered)
    tampered = deepcopy(receipt.to_dict())
    tampered["final_carrier_metadata"]["shape"] = [1, -1, 2]
    with pytest.raises(ValueError, match="nonnegative integers"):
        CarrierTransportExecutionReceipt.from_dict(tampered)
