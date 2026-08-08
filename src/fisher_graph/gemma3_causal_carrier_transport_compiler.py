"""Exact Gemma 3 lowering into a causal carrier-transport graph.

This is deliberately a *no-compression* compiler.  It reuses the standalone
exact 39-event Gemma executor, then gives every residual update one uniform
message interpretation:

``next carrier = identity carry message + primary generator message``.

There is no residual-add node in the lowered graph.  The residual path is a
fixed, coefficient-free identity edge.  Attention remains an exact generator
message whose internal token routing is state-conditioned, causal, and
row-stochastic.  MLP messages are token-local.  No fitted cross-block
correction messages are present in this exact rung.

The compiler never loads a checkpoint or opens prompt data.  The caller owns
the already-loaded :class:`~fisher_graph.adapters.Gemma3CausalLMAdapter`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .causal_carrier_transport_graph import (
    CarrierEventSpec,
    CarrierTransportExecutionReceipt,
    CarrierTransportManifest,
    TransportMessage,
    build_causal_carrier_transport_manifest,
    execute_causal_carrier_transport_graph,
    validate_carrier_transport_execution_receipt,
)
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    Gemma3JointCrossBlockChartExecutor,
    JOINT_EXACT_CONTROL_ARM_ID,
    JointGraphEventSpec,
    JointGraphGeneratorMutation,
    JointGraphMutationExecution,
    JointGraphWholeModelExecution,
    PINNED_GEMMA3_EVENT_COUNT,
    PINNED_GEMMA3_MUTATION_COUNT,
    _tensor_value_sha256 as _joint_tensor_sha256,
    build_joint_cross_block_event_schedule,
)
from .gemma3_joint_cross_block_compiler import (
    compile_gemma3_joint_exact_control,
)
from .gemma3_causal_carrier_transport_protocol import (
    build_gemma3_causal_carrier_transport_protocol,
    validate_gemma3_causal_carrier_transport_protocol,
)
from .residual_carrier import (
    ResidualGraphExecutionContext,
    ResidualMutationReceipt,
)


GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA = (
    "fisher_graph.gemma3_causal_carrier_transport.compilation.v1"
)
GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA = (
    "fisher_graph.gemma3_causal_carrier_transport.execution.v1"
)
GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID = (
    "exact_causal_carrier_transport"
)

_PROTOCOL = build_gemma3_causal_carrier_transport_protocol()
validate_gemma3_causal_carrier_transport_protocol(_PROTOCOL)
GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256 = str(
    _PROTOCOL["protocol_sha256"]
)

_TENSOR_HASH_DOMAIN = b"fisher_graph:causal-carrier-transport-tensor:v1\0"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("only tensors can be authenticated")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_exact_schedule(
    executor: Gemma3JointCrossBlockChartExecutor,
) -> tuple[JointGraphEventSpec, ...]:
    if not isinstance(executor, Gemma3JointCrossBlockChartExecutor):
        raise TypeError("exact_executor must be a joint Gemma graph executor")
    schedule = build_joint_cross_block_event_schedule()
    manifest = executor.architecture_manifest()
    if (
        executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID
        or len(executor.transports) != 0
        or manifest.get("exact_control_arm") is not True
        or manifest.get("candidate_arm") is not False
        or manifest.get("event_count") != PINNED_GEMMA3_EVENT_COUNT
        or tuple(manifest.get("ordered_event_ids", ()))
        != tuple(event.event_id for event in schedule)
        or manifest.get("attention_mutation_count") != 18
        or manifest.get("mlp_mutation_count") != 18
        or manifest.get("global_attention_block_ordinals") != (5, 11, 17)
        or manifest.get("transport_edge_count") != 0
        or manifest.get("native_fallback_present") is not False
        or manifest.get("native_layer_objects_present") is not False
        or manifest.get("compression_success_claimed") is not False
    ):
        raise ValueError(
            "carrier-transport lowering requires the exact transport-free "
            "39-event Gemma control"
        )
    lexical_metadata = executor.lexical.metadata()
    terminal_weight = getattr(executor.terminal_norm, "weight", None)
    if (
        not isinstance(lexical_metadata, Mapping)
        or lexical_metadata.get("is_exact_copied_matrix") is not True
        or not isinstance(terminal_weight, Tensor)
        or terminal_weight.ndim != 1
        or terminal_weight.shape[0] != executor.hidden_width
    ):
        raise ValueError(
            "exact lowering requires the copied tied lexical matrix and "
            "terminal RMSNorm weight"
        )
    for mutation, event in zip(executor.mutations, schedule[1:-2], strict=True):
        if mutation.event_spec != event:
            raise ValueError("exact mutation schedule drifted")
        if mutation.source_checkpoint_coefficients_retained is not True:
            raise ValueError("exact mutation omitted cloned checkpoint coefficients")
        if event.event_kind == "attention_mutation":
            operator = mutation.compiled_operator
            operator_manifest = (
                operator.architecture_manifest()
                if isinstance(operator, Gemma3AttentionGenerator)
                else None
            )
            if (
                not isinstance(operator, Gemma3AttentionGenerator)
                or not isinstance(operator_manifest, Mapping)
                or operator_manifest.get("generator_kind")
                != "exact_full_rank"
                or operator_manifest.get("source_independent_runtime")
                is not True
                or operator_manifest.get("source_fallback") is not False
                or operator_manifest.get("invocation_id") != event.event_id
            ):
                raise ValueError(
                    "every attention event must own an exact source-free "
                    "attention generator"
                )
        elif (
            not isinstance(
                mutation.chart_generator,
                ExactTokenLocalChartGenerator,
            )
            or mutation.compiled_operator is not None
            or mutation.sequence_operator is not None
        ):
            raise ValueError(
                "every MLP event must own an exact source-free token-local "
                "generator"
            )
    return schedule


def _build_core_transport_manifest(
    executor: Gemma3JointCrossBlockChartExecutor,
    schedule: tuple[JointGraphEventSpec, ...],
) -> CarrierTransportManifest:
    events = []
    primary_counts: dict[str, int] = {}
    for mutation, event in zip(
        executor.mutations,
        schedule[1:-2],
        strict=True,
    ):
        assert event.block_ordinal is not None
        family = (
            f"gemma3.attention.block.{event.block_ordinal}"
            if event.event_kind == "attention_mutation"
            else f"gemma3.mlp.block.{event.block_ordinal}"
        )
        events.append(
            CarrierEventSpec(
                event_index=event.event_index,
                event_id=event.event_id,
                event_kind=(
                    "attention"
                    if event.event_kind == "attention_mutation"
                    else "mlp"
                ),
                generator_invocation_id=event.event_id,
                generator_family_id=family,
                input_carrier_version=event.event_index - 1,
                output_carrier_version=event.event_index,
            )
        )
        primary_counts[event.event_id] = mutation.stored_scalar_count
    lexical_count = int(
        executor.resource_inventory()["lexical_shared_physical_scalars"]
    )
    return build_causal_carrier_transport_manifest(
        lexical_invocation_id="lexical.ingress",
        lexical_generator_family_id="gemma3.lexical.tied",
        events=tuple(events),
        corrections=(),
        lexical_stored_scalar_count=lexical_count,
        primary_stored_scalar_counts=primary_counts,
    )


@dataclass(frozen=True, slots=True)
class Gemma3AttentionTransportReceipt:
    """Authenticated dynamic token routing for one attention message."""

    event_id: str
    block_ordinal: int
    attention_kind: str
    probability_shape: tuple[int, int, int, int]
    routed_context_shape: tuple[int, int, int, int]
    probability_sha256: str
    routed_context_sha256: str
    state_condition_sha256: str
    logical_causal_key_pairs: int
    valid_attention_rows: int
    maximum_row_sum_error: float
    maximum_forbidden_probability: float
    source_projection_call_count: int
    source_fallback_call_count: int
    routing_receipt_replay_logical_macs: int
    routing_receipt_peak_dynamic_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_id, str)
            or not self.event_id
            or type(self.block_ordinal) is not int
            or not 0 <= self.block_ordinal < 18
            or self.attention_kind not in {"global_causal", "sliding_causal"}
            or len(self.probability_shape) != 4
            or len(self.routed_context_shape) != 4
            or any(
                type(value) is not int or value <= 0
                for value in (
                    *self.probability_shape,
                    *self.routed_context_shape,
                )
            )
            or not all(
                _is_sha256(value)
                for value in (
                    self.probability_sha256,
                    self.routed_context_sha256,
                    self.state_condition_sha256,
                )
            )
            or type(self.logical_causal_key_pairs) is not int
            or self.logical_causal_key_pairs <= 0
            or type(self.valid_attention_rows) is not int
            or self.valid_attention_rows <= 0
            or not math.isfinite(self.maximum_row_sum_error)
            or self.maximum_row_sum_error < 0
            or not math.isfinite(self.maximum_forbidden_probability)
            or self.maximum_forbidden_probability < 0
            or self.source_projection_call_count != 0
            or self.source_fallback_call_count != 0
            or type(self.routing_receipt_replay_logical_macs) is not int
            or self.routing_receipt_replay_logical_macs <= 0
            or type(self.routing_receipt_peak_dynamic_bytes) is not int
            or self.routing_receipt_peak_dynamic_bytes <= 0
        ):
            raise ValueError("attention transport receipt is invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "block_ordinal": self.block_ordinal,
            "attention_kind": (
                "global_causal"
                if self.attention_kind == "global_causal"
                else "sliding_causal_512"
            ),
            "runtime_attention_kind": self.attention_kind,
            "message_kind": "state_conditioned_causal_attention",
            "probability_shape": self.probability_shape,
            "routed_context_shape": self.routed_context_shape,
            "probability_sha256": self.probability_sha256,
            "routed_context_sha256": self.routed_context_sha256,
            "state_condition_sha256": self.state_condition_sha256,
            "logical_causal_key_pairs": self.logical_causal_key_pairs,
            "valid_attention_rows": self.valid_attention_rows,
            "maximum_row_sum_error": self.maximum_row_sum_error,
            "maximum_forbidden_probability": (
                self.maximum_forbidden_probability
            ),
            "causal_mask_enforced": True,
            "valid_rows_sum_to_one": self.maximum_row_sum_error <= 1e-6,
            "forbidden_routes_are_zero": (
                self.maximum_forbidden_probability == 0.0
            ),
            "invalid_query_rows_are_zero_inert": True,
            "state_conditioned": True,
            "routing_receipt_replayed_after_primary_generator_call": True,
            "routing_receipt_replay_logical_macs": (
                self.routing_receipt_replay_logical_macs
            ),
            "routing_receipt_dynamic_probability_elements": math.prod(
                self.probability_shape
            ),
            "routing_receipt_dynamic_context_elements": math.prod(
                self.routed_context_shape
            ),
            "routing_receipt_peak_dynamic_bytes": (
                self.routing_receipt_peak_dynamic_bytes
            ),
            "routing_receipt_peak_dynamic_bytes_kind": (
                "probability_plus_routed_context_lower_bound"
            ),
            "source_projection_call_count": (
                self.source_projection_call_count
            ),
            "source_fallback_call_count": self.source_fallback_call_count,
        }


@dataclass(frozen=True, slots=True)
class Gemma3CarrierTransportMutationReceipt:
    """One exact carrier boundary with carry and primary messages."""

    event_index: int
    event_id: str
    event_kind: str
    block_ordinal: int
    input_carrier_version: int
    output_carrier_version: int
    identity_carry_message_id: str
    primary_generator_message_id: str
    identity_carry_sha256: str
    primary_generator_sha256: str
    aggregate_carrier_sha256: str
    transport_view_delta_sha256: str
    attention_routing: Gemma3AttentionTransportReceipt | None
    source_projection_call_count: int
    source_fallback_call_count: int

    def __post_init__(self) -> None:
        expected_kind = (
            "attention_mutation"
            if self.event_index % 2 == 1
            else "mlp_mutation"
        )
        if (
            type(self.event_index) is not int
            or not 1 <= self.event_index <= PINNED_GEMMA3_MUTATION_COUNT
            or self.event_kind != expected_kind
            or type(self.block_ordinal) is not int
            or self.block_ordinal != (self.event_index - 1) // 2
            or self.input_carrier_version != self.event_index - 1
            or self.output_carrier_version != self.event_index
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.event_id,
                    self.identity_carry_message_id,
                    self.primary_generator_message_id,
                )
            )
            or not all(
                _is_sha256(value)
                for value in (
                    self.identity_carry_sha256,
                    self.primary_generator_sha256,
                    self.aggregate_carrier_sha256,
                    self.transport_view_delta_sha256,
                )
            )
            or (self.event_kind == "attention_mutation")
            != isinstance(
                self.attention_routing,
                Gemma3AttentionTransportReceipt,
            )
            or self.source_projection_call_count != 0
            or self.source_fallback_call_count != 0
        ):
            raise ValueError("carrier transport mutation receipt is invalid")

    def metadata(self) -> dict[str, object]:
        protocol_event_kind = (
            "attention_generator_message"
            if self.event_kind == "attention_mutation"
            else "token_local_mlp_generator_message"
        )
        return {
            "event_index": self.event_index,
            "event_id": self.event_id,
            "event_kind": protocol_event_kind,
            "joint_executor_event_kind": self.event_kind,
            "block_ordinal": self.block_ordinal,
            "input_carrier_version": self.input_carrier_version,
            "output_carrier_version": self.output_carrier_version,
            "identity_carry": {
                "message_id": self.identity_carry_message_id,
                "coefficient": 1.0,
                "stored_scalar_count": 0,
                "tensor_sha256": self.identity_carry_sha256,
            },
            "primary_generator": {
                "message_id": self.primary_generator_message_id,
                "message_kind": (
                    "state_conditioned_causal_attention"
                    if self.event_kind == "attention_mutation"
                    else "state_conditioned_token_local_mlp"
                ),
                "tensor_sha256": self.primary_generator_sha256,
            },
            "aggregate_carrier_sha256": self.aggregate_carrier_sha256,
            "transport_view_delta_sha256": (
                self.transport_view_delta_sha256
            ),
            "aggregation": "sum_messages_into_carrier",
            "residual_add_node_present": False,
            "correction_message_count": 0,
            "attention_routing": (
                None
                if self.attention_routing is None
                else self.attention_routing.metadata()
            ),
            "source_projection_call_count": (
                self.source_projection_call_count
            ),
            "source_fallback_call_count": self.source_fallback_call_count,
        }


@dataclass(frozen=True, slots=True)
class Gemma3CarrierTransportBoundaryReceipt:
    """Explicit lexical ingress, terminal norm, and tied lexical egress."""

    lexical_ingress_sha256: str
    final_carrier_sha256: str
    terminal_rmsnorm_sha256: str
    tied_lexical_egress_logits_sha256: str
    lexical_ingress_egress_storage_tied: bool
    final_logit_softcap: float | None

    def __post_init__(self) -> None:
        if (
            not all(
                _is_sha256(value)
                for value in (
                    self.lexical_ingress_sha256,
                    self.final_carrier_sha256,
                    self.terminal_rmsnorm_sha256,
                    self.tied_lexical_egress_logits_sha256,
                )
            )
            or self.lexical_ingress_egress_storage_tied is not True
            or (
                self.final_logit_softcap is not None
                and (
                    not math.isfinite(self.final_logit_softcap)
                    or self.final_logit_softcap <= 0
                )
            )
        ):
            raise ValueError("carrier transport boundary receipt is invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "lexical_ingress": {
                "message_id": "message.lexical.ingress",
                "message_kind": "lexical_ingress",
                "tensor_sha256": self.lexical_ingress_sha256,
            },
            "final_carrier_sha256": self.final_carrier_sha256,
            "terminal_rmsnorm": {
                "message_id": "message.terminal.rmsnorm",
                "message_kind": "terminal_normalization",
                "present": True,
                "tensor_sha256": self.terminal_rmsnorm_sha256,
            },
            "tied_lexical_egress": {
                "message_id": "message.lexical.egress",
                "message_kind": "lexical_egress",
                "present": True,
                "shares_ingress_storage": True,
                "logits_sha256": self.tied_lexical_egress_logits_sha256,
            },
            "final_logit_softcap": self.final_logit_softcap,
        }


@dataclass(frozen=True, slots=True)
class Gemma3CausalCarrierTransportExecution:
    """Exact outputs plus strict carrier/message authentication receipts."""

    transport_view: JointGraphWholeModelExecution
    core_execution_receipt: CarrierTransportExecutionReceipt
    core_transport_manifest_sha256: str
    mutation_receipts: tuple[Gemma3CarrierTransportMutationReceipt, ...]
    boundary_receipt: Gemma3CarrierTransportBoundaryReceipt
    event_ids: tuple[str, ...]
    source_projection_call_count: int
    source_fallback_call_count: int
    core_carrier_snapshot_bytes: int
    bound_source_fingerprint_before_execution: str
    bound_source_fingerprint_after_execution: str

    def __post_init__(self) -> None:
        schedule = build_joint_cross_block_event_schedule()
        if (
            not isinstance(self.transport_view, JointGraphWholeModelExecution)
            or not isinstance(
                self.core_execution_receipt,
                CarrierTransportExecutionReceipt,
            )
            or not _is_sha256(self.core_transport_manifest_sha256)
            or self.core_execution_receipt.manifest_sha256
            != self.core_transport_manifest_sha256
            or len(self.core_execution_receipt.boundaries)
            != PINNED_GEMMA3_MUTATION_COUNT
            or tuple(
                item.event_id
                for item in self.core_execution_receipt.boundaries
            )
            != tuple(event.event_id for event in schedule[1:-2])
            or self.core_execution_receipt.correction_message_count != 0
            or self.core_execution_receipt.aggregate_commit_count
            != PINNED_GEMMA3_MUTATION_COUNT
            or len(self.mutation_receipts) != PINNED_GEMMA3_MUTATION_COUNT
            or tuple(item.event_id for item in self.mutation_receipts)
            != tuple(event.event_id for event in schedule[1:-2])
            or self.event_ids != tuple(event.event_id for event in schedule)
            or self.transport_view.event_ids != self.event_ids
            or self.transport_view.carrier_version
            != PINNED_GEMMA3_MUTATION_COUNT
            or self.transport_view.transport_edge_count != 0
            or not isinstance(
                self.boundary_receipt,
                Gemma3CarrierTransportBoundaryReceipt,
            )
            or self.source_projection_call_count != 0
            or self.source_fallback_call_count != 0
            or type(self.core_carrier_snapshot_bytes) is not int
            or self.core_carrier_snapshot_bytes <= 0
            or self.bound_source_fingerprint_before_execution
            != self.bound_source_fingerprint_after_execution
            or not _is_sha256(
                self.bound_source_fingerprint_before_execution
            )
        ):
            raise ValueError("causal carrier-transport execution is invalid")

    @property
    def output(self) -> Tensor:
        return self.transport_view.output

    @property
    def normalized_output(self) -> Tensor:
        return self.transport_view.normalized_output

    @property
    def logits(self) -> Tensor:
        return self.transport_view.logits

    @property
    def mutation_delta_sha256s(self) -> tuple[str, ...]:
        return self.transport_view.mutation_delta_sha256s

    @property
    def mutation_chart_input_sha256s(self) -> tuple[str, ...]:
        return self.transport_view.mutation_chart_input_sha256s

    @property
    def ordered_message_ids(self) -> tuple[str, ...]:
        result = ["message.lexical.ingress"]
        for item in self.mutation_receipts:
            result.extend(
                (
                    item.identity_carry_message_id,
                    item.primary_generator_message_id,
                )
            )
        result.extend(
            ("message.terminal.rmsnorm", "message.lexical.egress")
        )
        return tuple(result)

    def receipt(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA,
            "protocol_sha256": (
                GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256
            ),
            "arm_id": GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
            "event_ids": self.event_ids,
            "event_count": len(self.event_ids),
            "ordered_message_ids": self.ordered_message_ids,
            "message_count": len(self.ordered_message_ids),
            "carrier_boundary_count": len(self.mutation_receipts),
            "initial_carrier_version": 0,
            "final_carrier_version": (
                self.transport_view.carrier_version
            ),
            "identity_carry_message_count": len(self.mutation_receipts),
            "primary_generator_message_count": len(self.mutation_receipts),
            "attention_routed_message_count": sum(
                item.attention_routing is not None
                for item in self.mutation_receipts
            ),
            "mlp_token_local_message_count": sum(
                item.event_kind == "mlp_mutation"
                for item in self.mutation_receipts
            ),
            "correction_message_count": 0,
            "routing_receipt_replay_logical_macs": sum(
                0
                if item.attention_routing is None
                else item.attention_routing.routing_receipt_replay_logical_macs
                for item in self.mutation_receipts
            ),
            "routing_receipt_dynamic_probability_elements": sum(
                0
                if item.attention_routing is None
                else math.prod(item.attention_routing.probability_shape)
                for item in self.mutation_receipts
            ),
            "routing_receipt_dynamic_context_elements": sum(
                0
                if item.attention_routing is None
                else math.prod(item.attention_routing.routed_context_shape)
                for item in self.mutation_receipts
            ),
            "routing_receipt_peak_dynamic_bytes": max(
                (
                    0
                    if item.attention_routing is None
                    else item.attention_routing.routing_receipt_peak_dynamic_bytes
                )
                for item in self.mutation_receipts
            ),
            "core_carrier_snapshot_bytes": self.core_carrier_snapshot_bytes,
            "analysis_peak_dynamic_bytes_lower_bound": (
                self.core_carrier_snapshot_bytes
                + max(
                    (
                        0
                        if item.attention_routing is None
                        else item.attention_routing.routing_receipt_peak_dynamic_bytes
                    )
                    for item in self.mutation_receipts
                )
            ),
            "mutation_receipts": tuple(
                item.metadata() for item in self.mutation_receipts
            ),
            "boundaries": self.boundary_receipt.metadata(),
            "transport_view": {
                "mutation_chart_input_sha256s": (
                    self.transport_view.mutation_chart_input_sha256s
                ),
                "mutation_delta_sha256s": (
                    self.transport_view.mutation_delta_sha256s
                ),
                "final_carrier_sha256": _tensor_sha256(self.output),
                "normalized_output_sha256": _tensor_sha256(
                    self.normalized_output
                ),
                "logits_sha256": _tensor_sha256(self.logits),
                "carrier_version": self.transport_view.carrier_version,
                "transport_edge_count": (
                    self.transport_view.transport_edge_count
                ),
            },
            "core_live_carrier_transport": (
                self.core_execution_receipt.to_dict()
            ),
            "core_transport_manifest_sha256": (
                self.core_transport_manifest_sha256
            ),
            "source_projection_call_count": (
                self.source_projection_call_count
            ),
            "source_fallback_call_count": self.source_fallback_call_count,
            "bound_source_fingerprint_before_execution": (
                self.bound_source_fingerprint_before_execution
            ),
            "bound_source_fingerprint_after_execution": (
                self.bound_source_fingerprint_after_execution
            ),
            "live_source_fingerprint_read_during_execution": False,
            "residual_add_node_present": False,
            "compression_attempted": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "model_mutated": False,
        }


CarrierTransportObserver = Callable[..., object]


class Gemma3CausalCarrierTransportExecutor(nn.Module):
    """Exact Gemma runtime viewed as one causal message-passing graph."""

    def __init__(
        self,
        *,
        exact_executor: Gemma3JointCrossBlockChartExecutor,
        source_model_parameter_count: int,
        source_model_fingerprint: str,
        source_independence_audit: Mapping[str, object],
    ) -> None:
        super().__init__()
        self._schedule = _require_exact_schedule(exact_executor)
        if (
            type(source_model_parameter_count) is not int
            or source_model_parameter_count <= 0
        ):
            raise ValueError("source model parameter count is invalid")
        if not _is_sha256(source_model_fingerprint):
            raise ValueError("source model fingerprint is invalid")
        if (
            not isinstance(source_independence_audit, Mapping)
            or source_independence_audit.get("passed") is not True
        ):
            raise ValueError("source independence audit did not pass")
        self.exact_executor = exact_executor
        self._transport_manifest = _build_core_transport_manifest(
            exact_executor,
            self._schedule,
        )
        self.source_model_parameter_count = source_model_parameter_count
        self.source_model_fingerprint = source_model_fingerprint
        self.source_independence_audit = (
            None
            if source_independence_audit is None
            else copy.deepcopy(dict(source_independence_audit))
        )
        inventory = self.resource_inventory()
        if inventory["exact_stored_scalar_parity"] is not True:
            raise ValueError(
                "exact carrier-transport scalar inventory does not match source"
            )
        self.eval()

    def train(self, mode: bool = True) -> Gemma3CausalCarrierTransportExecutor:
        if type(mode) is not bool:
            raise TypeError("mode must be boolean")
        if mode:
            raise RuntimeError("compiled carrier-transport graph is inference-only")
        return super().train(False)

    @property
    def ordered_event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self._schedule)

    @property
    def expected_mutation_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self._schedule[1:-2])

    @property
    def hidden_width(self) -> int:
        return self.exact_executor.hidden_width

    @property
    def transport_manifest(self) -> CarrierTransportManifest:
        return self._transport_manifest

    def resource_inventory(self) -> dict[str, object]:
        exact = self.exact_executor.resource_inventory()
        deployed = int(exact["candidate_deployed_scalars"])
        source = self.source_model_parameter_count
        return {
            "source_model_parameter_count": source,
            "exact_deployed_scalar_count": deployed,
            "physical_resident_compiled_scalar_count": deployed,
            "exact_stored_scalar_parity": (
                None if source is None else deployed == source
            ),
            "lexical_shared_physical_scalars": exact[
                "lexical_shared_physical_scalars"
            ],
            "mutation_scalars": exact["mutation_scalars"],
            "terminal_norm_scalars": exact["terminal_norm_scalars"],
            "core_graph_declared_stored_scalars": (
                self.transport_manifest.stored_scalar_count
            ),
            "core_graph_excludes_terminal_boundary_scalars": True,
            "cross_block_correction_scalars": 0,
            "identity_carry_stored_scalars": 0,
            "attention_routing_stored_scalars": 0,
            "attention_routing_receipt_replay_compute_per_forward": True,
            "attention_routing_receipt_dynamic_memory_per_forward": True,
            "attention_routing_replay_credited_as_model_reduction": False,
            "compiler_wrapper_additional_stored_scalars": 0,
            "reference_and_transport_view_share_one_coefficient_set": True,
            "lexical_ingress_egress_factor_counted_once": True,
            "inventory_closes": exact["inventory_closes"] is True,
            "compression_attempted": False,
            "parameter_reduction_claimed": False,
            "logical_mac_reduction_claimed": False,
            "logical_carrier_additions_removed": 0,
            "latency_reduction_claimed": False,
        }

    def architecture_manifest(self) -> dict[str, object]:
        graph_contract = _PROTOCOL["graph_contract"]
        if not isinstance(graph_contract, Mapping):
            raise RuntimeError("canonical protocol graph contract disappeared")
        protocol_events = tuple(graph_contract["events"])
        protocol_messages = tuple(graph_contract["messages"])
        event_by_id = {
            str(item["event_id"]): item
            for item in protocol_events
            if isinstance(item, Mapping)
        }
        message_by_id = {
            str(item["message_id"]): item
            for item in protocol_messages
            if isinstance(item, Mapping)
        }
        boundaries: list[dict[str, object]] = []
        for event in self._schedule[1:-2]:
            assert event.block_ordinal is not None
            carry_id = f"message.{event.event_id}.carry"
            generator_id = f"message.{event.event_id}.generator"
            boundaries.append(
                {
                    "event": copy.deepcopy(event_by_id[event.event_id]),
                    "joint_executor_event": event.metadata(),
                    "input_carrier_version": event.event_index - 1,
                    "output_carrier_version": event.event_index,
                    "incoming_messages": (
                        copy.deepcopy(message_by_id[carry_id]),
                        copy.deepcopy(message_by_id[generator_id]),
                    ),
                    "aggregation": "sum_messages_into_carrier",
                    "residual_add_node_present": False,
                    "correction_message_count": 0,
                }
            )
        exact_manifest = self.exact_executor.architecture_manifest()
        return {
            "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
            "protocol_sha256": (
                GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256
            ),
            "canonical_protocol_authenticated": True,
            "core_transport_manifest_sha256": (
                self.transport_manifest.fingerprint_sha256
            ),
            "core_transport_manifest": self.transport_manifest.to_dict(),
            "kind": "gemma3_causal_carrier_transport_executor",
            "arm_id": GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID,
            "exact_no_compression_rung": True,
            "source_independent_runtime": True,
            "source_free_serialized_artifact_established": False,
            "event_count": len(self._schedule),
            "ordered_event_ids": self.ordered_event_ids,
            "events": copy.deepcopy(protocol_events),
            "ordered_message_ids": tuple(graph_contract["message_ids"]),
            "message_count": len(protocol_messages),
            "lexical_ingress": copy.deepcopy(
                message_by_id["message.lexical.ingress"]
            ),
            "carrier_boundaries": tuple(boundaries),
            "carrier_boundary_count": len(boundaries),
            "identity_carry_message_count": len(boundaries),
            "identity_carry_coefficient": 1.0,
            "identity_carry_stored_scalar_count": 0,
            "primary_generator_message_count": len(boundaries),
            "attention_routed_message_count": 18,
            "attention_routing_is_state_conditioned": True,
            "attention_routing_is_causal_row_stochastic": True,
            "global_attention_block_ordinals": (5, 11, 17),
            "sliding_attention_message_count": 15,
            "mlp_token_local_message_count": 18,
            "cross_block_correction_message_count": 0,
            "residual_add_node_present": False,
            "terminal_rmsnorm": {
                "present": True,
                "event_id": "terminal.rmsnorm",
                "message": copy.deepcopy(
                    message_by_id["message.terminal.rmsnorm"]
                ),
            },
            "tied_lexical_egress": {
                "present": True,
                "event_id": "lexical.egress",
                "message": copy.deepcopy(
                    message_by_id["message.lexical.egress"]
                ),
                "shares_ingress_physical_storage": True,
                "final_logit_softcap": self.exact_executor.final_logit_softcap,
            },
            "exact_reference_executor_manifest": exact_manifest,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_independence_audit": copy.deepcopy(
                self.source_independence_audit
            ),
            "source_projection_call_count_required": 0,
            "source_fallback_call_count_required": 0,
            "live_source_model_reference_present": False,
            "native_fallback_present": False,
            "resource_inventory": self.resource_inventory(),
            "compression_attempted": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "model_mutation_allowed": False,
        }

    @staticmethod
    def _emit_observation(
        observer: CarrierTransportObserver | None,
        *,
        message_id: str,
        event_id: str,
        message_kind: str,
        values: Tensor,
        generator_input: Tensor | None = None,
        routing_weights: Tensor | None = None,
        aggregate_output: Tensor | None = None,
    ) -> None:
        if observer is None:
            return
        payload: dict[str, object] = {
            "message_id": message_id,
            "event_id": event_id,
            "message_kind": message_kind,
            "values": values.detach().clone(),
        }
        for name, value in (
            ("generator_input", generator_input),
            ("routing_weights", routing_weights),
            ("aggregate_output", aggregate_output),
        ):
            if value is not None:
                payload[name] = value.detach().clone()
        observer(**payload)

    @staticmethod
    def _attention_routing(
        *,
        operator: Gemma3AttentionGenerator,
        normalized_input: Tensor,
        sequence: SequenceContext,
        event: JointGraphEventSpec,
        runtime_logical_pairs: int | None,
    ) -> tuple[Gemma3AttentionTransportReceipt, Tensor]:
        if operator.generator_kind != "exact_full_rank":
            raise RuntimeError("attention routing requires the exact generator")
        attention = operator.config.attention
        batch = sequence.batch_size
        length = sequence.query_length
        dimension = attention.head_dimension
        values = normalized_input.detach().masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        with torch.no_grad():
            query = F.linear(values, operator.q_weight, operator.q_bias).view(
                batch,
                length,
                attention.query_heads,
                dimension,
            ).transpose(1, 2)
            key = F.linear(values, operator.k_weight, operator.k_bias).view(
                batch,
                length,
                attention.key_value_heads,
                dimension,
            ).transpose(1, 2)
            value = F.linear(values, operator.v_weight, operator.v_bias).view(
                batch,
                length,
                attention.key_value_heads,
                dimension,
            ).transpose(1, 2)
            if attention.qk_norm:
                assert operator.q_norm_weight is not None
                assert operator.k_norm_weight is not None
                assert operator.config.qk_norm is not None
                query = operator._rms_norm(
                    query,
                    operator.q_norm_weight,
                    operator.config.qk_norm,
                )
                key = operator._rms_norm(
                    key,
                    operator.k_norm_weight,
                    operator.config.qk_norm,
                )
            cosine, sine = operator._rotary_cos_sin(
                sequence.logical_positions,
                dtype=query.dtype,
            )
            query = query * cosine.unsqueeze(1) + operator._rotate_half(
                query
            ) * sine.unsqueeze(1)
            key = key * cosine.unsqueeze(1) + operator._rotate_half(
                key
            ) * sine.unsqueeze(1)
            repeats = attention.query_heads // attention.key_value_heads
            repeated_key = operator._repeat_key_value(key, repeats)
            repeated_value = operator._repeat_key_value(value, repeats)
            scores = torch.matmul(
                query,
                repeated_key.transpose(-2, -1),
            ) * float(attention.query_scale)
            if operator.config.logit_softcap is not None:
                cap = float(operator.config.logit_softcap)
                scores = torch.tanh(scores / cap) * cap
            allowed = operator._allowed_pairs(sequence)
            scores = scores.masked_fill(
                ~allowed.unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
            probabilities = F.softmax(
                scores,
                dim=-1,
                dtype=torch.float32,
            ).to(dtype=query.dtype)
            effective = probabilities.masked_fill(
                ~allowed.unsqueeze(1),
                0,
            )
            routed_context = torch.matmul(effective, repeated_value)
        logical_pairs = int(allowed.sum().item())
        if runtime_logical_pairs != logical_pairs:
            raise RuntimeError("attention logical-pair receipt drifted")
        valid_rows = sequence.query_valid_mask.unsqueeze(1).expand(
            -1,
            attention.query_heads,
            -1,
        )
        row_sums = effective.sum(dim=-1)
        maximum_row_sum_error = float(
            (row_sums[valid_rows].float() - 1.0).abs().max().item()
        )
        forbidden_values = effective.masked_select(~allowed.unsqueeze(1))
        maximum_forbidden = (
            0.0
            if forbidden_values.numel() == 0
            else float(forbidden_values.abs().max().item())
        )
        if maximum_row_sum_error > 1e-6 or maximum_forbidden != 0.0:
            raise RuntimeError("attention routing is not causal row-stochastic")
        invalid_routing = effective.masked_select(
            (~sequence.query_valid_mask).unsqueeze(1).unsqueeze(-1)
        )
        if invalid_routing.numel() and bool(
            torch.count_nonzero(invalid_routing)
        ):
            raise RuntimeError("invalid query routing rows are not inert")
        valid_tokens = int(sequence.query_valid_mask.sum().item())
        qkv_macs = valid_tokens * operator.config.residual_width * (
            attention.query_heads * dimension
            + 2 * attention.key_value_heads * dimension
        )
        route_macs = (
            2 * logical_pairs * attention.query_heads * dimension
        )
        replay_macs = qkv_macs + route_macs
        replay_peak_bytes = (
            effective.numel() + routed_context.numel()
        ) * effective.element_size()
        assert event.block_ordinal is not None
        return (
            Gemma3AttentionTransportReceipt(
                event_id=event.event_id,
                block_ordinal=event.block_ordinal,
                attention_kind=attention.kind,
                probability_shape=tuple(effective.shape),
                routed_context_shape=tuple(routed_context.shape),
                probability_sha256=_tensor_sha256(effective),
                routed_context_sha256=_tensor_sha256(routed_context),
                state_condition_sha256=_tensor_sha256(normalized_input),
                logical_causal_key_pairs=logical_pairs,
                valid_attention_rows=int(valid_rows.sum().item()),
                maximum_row_sum_error=maximum_row_sum_error,
                maximum_forbidden_probability=maximum_forbidden,
                source_projection_call_count=0,
                source_fallback_call_count=0,
                routing_receipt_replay_logical_macs=replay_macs,
                routing_receipt_peak_dynamic_bytes=replay_peak_bytes,
            ),
            effective,
        )

    def forward_components(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        observer: CarrierTransportObserver | None = None,
    ) -> Gemma3CausalCarrierTransportExecution:
        """Execute once and authenticate all exact carrier messages.

        ``observer`` is an optional detached sidecar.  Its return value is
        ignored, so it cannot alter the exact model path.
        """

        if observer is not None and not callable(observer):
            raise TypeError("carrier-transport observer must be callable")
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.shape
            != (sequence.batch_size, sequence.query_length)
        ):
            raise ValueError(
                "input_ids must be integer [batch, sequence] matching context"
            )
        context = ResidualGraphExecutionContext.bind(sequence)
        generated_by_event: dict[
            str,
            tuple[
                JointGraphMutationExecution,
                Gemma3AttentionTransportReceipt | None,
                Tensor | None,
            ],
        ] = {}
        mutation_by_event = {
            mutation.event_spec.event_id: mutation
            for mutation in self.exact_executor.mutations
        }
        event_by_id = {
            event.event_id: event for event in self._schedule[1:-2]
        }
        producers: dict[str, Callable[..., TransportMessage]] = {}
        lexical_edge = self.transport_manifest.edges[0]

        def lexical_producer(core_context: object) -> TransportMessage:
            edge = getattr(core_context, "edge", None)
            token_input = getattr(core_context, "token_input", None)
            if edge != lexical_edge or token_input is not input_ids:
                raise RuntimeError("core lexical context drifted")
            value = self.exact_executor.lexical.ingress(input_ids)
            if (
                not isinstance(value, Tensor)
                or value.shape != (*input_ids.shape, self.hidden_width)
                or not value.is_floating_point()
            ):
                raise ValueError("lexical ingress returned an invalid carrier")
            return TransportMessage(edge_id=lexical_edge.edge_id, value=value)

        producers[lexical_edge.edge_id] = lexical_producer
        for edge in self.transport_manifest.edges:
            if edge.message_kind not in {"attention_transport", "mlp_mutation"}:
                continue
            assert edge.target_event_id is not None
            event = event_by_id[edge.target_event_id]
            mutation = mutation_by_event[event.event_id]

            def primary_producer(
                core_context: object,
                *,
                edge_id: str = edge.edge_id,
                event: JointGraphEventSpec = event,
                mutation: JointGraphGeneratorMutation = mutation,
            ) -> TransportMessage:
                carrier = getattr(core_context, "carrier_input", None)
                core_edge = getattr(core_context, "edge", None)
                core_event = getattr(core_context, "event", None)
                if (
                    not isinstance(carrier, Tensor)
                    or getattr(core_edge, "edge_id", None) != edge_id
                    or getattr(core_event, "event_id", None) != event.event_id
                ):
                    raise RuntimeError("core primary-generator context drifted")
                incoming = torch.zeros(
                    *carrier.shape[:2],
                    mutation.incoming_chart_count,
                    mutation.incoming_generator_rank,
                    dtype=carrier.dtype,
                    device=carrier.device,
                )
                generated = mutation.execute_with_incoming(
                    carrier,
                    sequence,
                    incoming,
                )
                operator_receipt = generated.runtime_receipt
                if event.event_kind == "attention_mutation":
                    operator = mutation.compiled_operator
                    if not isinstance(operator, Gemma3AttentionGenerator):
                        raise RuntimeError("exact attention operator disappeared")
                    source_projection_calls = (
                        operator_receipt.source_projection_call_count
                    )
                    source_fallback_calls = (
                        operator_receipt.source_fallback_call_count
                    )
                    if (
                        source_projection_calls != 0
                        or source_fallback_calls != 0
                    ):
                        raise RuntimeError("exact attention used a source call")
                    routing, routing_weights = self._attention_routing(
                        operator=operator,
                        normalized_input=generated.chart_input,
                        sequence=sequence,
                        event=event,
                        runtime_logical_pairs=(
                            operator_receipt.logical_causal_key_pairs
                        ),
                    )
                else:
                    routing = None
                    routing_weights = None
                invalid_delta = generated.delta.masked_select(
                    (~sequence.query_valid_mask).unsqueeze(-1)
                )
                if invalid_delta.numel() and bool(
                    torch.count_nonzero(invalid_delta)
                ):
                    raise RuntimeError(
                        "primary generator mutated an invalid query row"
                    )
                if event.event_id in generated_by_event:
                    raise RuntimeError("core invoked a generator more than once")
                generated_by_event[event.event_id] = (
                    generated,
                    routing,
                    routing_weights,
                )
                return TransportMessage(
                    edge_id=edge_id,
                    value=generated.delta,
                    attention_routing_weights=routing_weights,
                )

            producers[edge.edge_id] = primary_producer
        execution_nonce = (
            f"gemma3.transport.{_tensor_sha256(input_ids)[:16]}"
        )
        core_execution = execute_causal_carrier_transport_graph(
            self.transport_manifest,
            token_input=input_ids,
            message_producers=producers,
            execution_nonce=execution_nonce,
        )
        validate_carrier_transport_execution_receipt(
            core_execution.receipt,
            manifest=self.transport_manifest,
        )
        if len(generated_by_event) != PINNED_GEMMA3_MUTATION_COUNT:
            raise RuntimeError("core did not invoke every Gemma generator once")
        context.assert_unchanged()
        state = core_execution.final_carrier
        lexical_sha = _tensor_sha256(core_execution.carrier_snapshots[0])
        self._emit_observation(
            observer,
            message_id="message.lexical.ingress",
            event_id="lexical.ingress",
            message_kind="lexical_ingress",
            values=core_execution.carrier_snapshots[0],
        )
        receipts: list[Gemma3CarrierTransportMutationReceipt] = []
        carrier_receipts: list[ResidualMutationReceipt] = []
        operator_receipts = []
        chart_input_hashes: list[str] = []
        delta_hashes: list[str] = []
        for ordinal, event in enumerate(self._schedule[1:-2], start=1):
            generated, routing_value, routing_weights = generated_by_event[
                event.event_id
            ]
            routing = (
                routing_value
                if isinstance(routing_value, Gemma3AttentionTransportReceipt)
                else None
            )
            chart_input = generated.chart_input
            generator = generated.delta
            operator_receipt = generated.runtime_receipt
            carry = core_execution.carrier_snapshots[ordinal - 1]
            aggregate = core_execution.carrier_snapshots[ordinal]
            carry_id = f"message.{event.event_id}.carry"
            primary_id = f"message.{event.event_id}.generator"
            self._emit_observation(
                observer,
                message_id=carry_id,
                event_id=event.event_id,
                message_kind="identity_carry",
                values=carry,
            )
            self._emit_observation(
                observer,
                message_id=primary_id,
                event_id=event.event_id,
                message_kind=(
                    "state_conditioned_causal_attention"
                    if routing is not None
                    else "state_conditioned_token_local_mlp"
                ),
                values=generator,
                generator_input=chart_input,
                routing_weights=routing_weights,
                aggregate_output=aggregate,
            )
            assert event.block_ordinal is not None
            receipts.append(
                Gemma3CarrierTransportMutationReceipt(
                    event_index=event.event_index,
                    event_id=event.event_id,
                    event_kind=event.event_kind,
                    block_ordinal=event.block_ordinal,
                    input_carrier_version=ordinal - 1,
                    output_carrier_version=ordinal,
                    identity_carry_message_id=carry_id,
                    primary_generator_message_id=primary_id,
                    identity_carry_sha256=_tensor_sha256(carry),
                    primary_generator_sha256=_tensor_sha256(generator),
                    aggregate_carrier_sha256=_tensor_sha256(aggregate),
                    transport_view_delta_sha256=_joint_tensor_sha256(
                        generator
                    ),
                    attention_routing=routing,
                    source_projection_call_count=0,
                    source_fallback_call_count=0,
                )
            )
            carrier_receipts.append(
                ResidualMutationReceipt(
                    wire_id="gemma3_causal_carrier_transport",
                    mutation_id=event.event_id,
                    version_before=ordinal - 1,
                    version_after=ordinal,
                    valid_token_count=context.valid_token_count,
                    invalid_token_count=(
                        context.batch_size * context.query_length
                        - context.valid_token_count
                    ),
                    combiner_kind=(
                        "core_identity_plus_primary_generator_message"
                    ),
                )
            )
            operator_receipts.append(operator_receipt)
            chart_input_hashes.append(_joint_tensor_sha256(chart_input))
            delta_hashes.append(_joint_tensor_sha256(generator))
        normalized = self.exact_executor.terminal_norm(state)
        if not isinstance(normalized, Tensor) or normalized.shape != state.shape:
            raise ValueError("terminal RMSNorm must preserve the carrier")
        logits = self.exact_executor.lexical.egress(normalized)
        if (
            not isinstance(logits, Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
        ):
            raise ValueError("tied lexical egress returned invalid logits")
        if self.exact_executor.final_logit_softcap is not None:
            cap = self.exact_executor.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        self._emit_observation(
            observer,
            message_id="message.terminal.rmsnorm",
            event_id="terminal.rmsnorm",
            message_kind="terminal_normalization",
            values=normalized,
        )
        self._emit_observation(
            observer,
            message_id="message.lexical.egress",
            event_id="lexical.egress",
            message_kind="lexical_egress",
            values=logits,
        )
        transport_view = JointGraphWholeModelExecution(
            output=state,
            normalized_output=normalized,
            logits=logits,
            event_ids=self.ordered_event_ids,
            mutation_receipts=tuple(carrier_receipts),
            operator_receipts=tuple(operator_receipts),
            mutation_chart_input_sha256s=tuple(chart_input_hashes),
            mutation_delta_sha256s=tuple(delta_hashes),
            carrier_version=PINNED_GEMMA3_MUTATION_COUNT,
            transport_edge_count=0,
            peak_live_transport_packets=0,
            peak_live_transport_packet_bytes=0,
        )
        boundary = Gemma3CarrierTransportBoundaryReceipt(
            lexical_ingress_sha256=lexical_sha,
            final_carrier_sha256=_tensor_sha256(state),
            terminal_rmsnorm_sha256=_tensor_sha256(normalized),
            tied_lexical_egress_logits_sha256=_tensor_sha256(logits),
            lexical_ingress_egress_storage_tied=True,
            final_logit_softcap=self.exact_executor.final_logit_softcap,
        )
        snapshot_bytes = sum(
            value.numel() * value.element_size()
            for value in core_execution.carrier_snapshots
        )
        return Gemma3CausalCarrierTransportExecution(
            transport_view=transport_view,
            core_execution_receipt=core_execution.receipt,
            core_transport_manifest_sha256=(
                self.transport_manifest.fingerprint_sha256
            ),
            mutation_receipts=tuple(receipts),
            boundary_receipt=boundary,
            event_ids=self.ordered_event_ids,
            source_projection_call_count=0,
            source_fallback_call_count=0,
            core_carrier_snapshot_bytes=snapshot_bytes,
            bound_source_fingerprint_before_execution=(
                self.source_model_fingerprint
            ),
            bound_source_fingerprint_after_execution=(
                self.source_model_fingerprint
            ),
        )

    execute = forward_components

    def forward(self, input_ids: Tensor, sequence: SequenceContext) -> Tensor:
        return self.forward_components(input_ids, sequence).logits


@dataclass(frozen=True, slots=True)
class Gemma3CausalCarrierTransportCompilation:
    """Compiler product and source-independence/resource receipts."""

    executor: Gemma3CausalCarrierTransportExecutor

    def __post_init__(self) -> None:
        if not isinstance(
            self.executor,
            Gemma3CausalCarrierTransportExecutor,
        ):
            raise TypeError("compilation executor has the wrong type")
        if (
            self.executor.resource_inventory()["exact_stored_scalar_parity"]
            is not True
        ):
            raise ValueError("compiled exact resource inventory did not close")

    @property
    def reference_executor(self) -> Gemma3JointCrossBlockChartExecutor:
        """The one physical exact coefficient set shared by both views."""

        return self.executor.exact_executor

    def metadata(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA,
            "protocol_sha256": (
                GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256
            ),
            "compilation_kind": "exact_no_compression_carrier_transport",
            "source_model_fingerprint": (
                self.executor.source_model_fingerprint
            ),
            "architecture_manifest": self.executor.architecture_manifest(),
            "resource_inventory": self.executor.resource_inventory(),
            "source_independence_audit": copy.deepcopy(
                self.executor.source_independence_audit
            ),
            "model_loaded_by_compiler": False,
            "prompt_data_opened": False,
            "tokenization_performed": False,
            "experiment_executed": False,
            "metrics_present": False,
            "compression_attempted": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "model_mutated": False,
        }


@dataclass(frozen=True, slots=True)
class Gemma3LiteralValidQueryParityReceipt:
    """Literal (zero-tolerance) native-versus-compiled logit check."""

    compared_valid_query_rows: int
    native_logits_sha256: str
    compiled_logits_sha256: str
    literal_valid_query_logit_parity: bool
    maximum_absolute_valid_query_logit_error: float
    native_model_call_count: int = 1
    compiled_executor_call_count: int = 1

    @property
    def passed(self) -> bool:
        return self.literal_valid_query_logit_parity

    def metadata(self) -> dict[str, object]:
        return {
            "compared_valid_query_rows": self.compared_valid_query_rows,
            "native_logits_sha256": self.native_logits_sha256,
            "compiled_logits_sha256": self.compiled_logits_sha256,
            "literal_valid_query_logit_parity": (
                self.literal_valid_query_logit_parity
            ),
            "maximum_absolute_valid_query_logit_error": (
                self.maximum_absolute_valid_query_logit_error
            ),
            "native_model_call_count": self.native_model_call_count,
            "compiled_executor_call_count": (
                self.compiled_executor_call_count
            ),
            "passed": self.passed,
        }


def verify_literal_valid_query_logit_parity(
    *,
    native_logits: Tensor,
    compiled_execution: Gemma3CausalCarrierTransportExecution,
    valid_query_mask: Tensor,
) -> Gemma3LiteralValidQueryParityReceipt:
    """Compare valid-query logits with literal ``torch.equal`` semantics."""

    if not isinstance(compiled_execution, Gemma3CausalCarrierTransportExecution):
        raise TypeError("compiled_execution has the wrong type")
    compiled_logits = compiled_execution.logits
    if (
        not isinstance(native_logits, Tensor)
        or native_logits.shape != compiled_logits.shape
        or native_logits.dtype != compiled_logits.dtype
        or native_logits.device != compiled_logits.device
    ):
        raise ValueError("native and compiled logits must have identical ABI")
    if (
        not isinstance(valid_query_mask, Tensor)
        or valid_query_mask.dtype is not torch.bool
        or valid_query_mask.shape != compiled_logits.shape[:2]
        or valid_query_mask.device != compiled_logits.device
        or not bool(valid_query_mask.any())
    ):
        raise ValueError("valid_query_mask is invalid")
    native_valid = native_logits[valid_query_mask]
    compiled_valid = compiled_logits[valid_query_mask]
    literal = torch.equal(native_valid, compiled_valid)
    maximum = float(
        (native_valid.float() - compiled_valid.float()).abs().max().item()
    )
    return Gemma3LiteralValidQueryParityReceipt(
        compared_valid_query_rows=int(valid_query_mask.sum().item()),
        native_logits_sha256=_tensor_sha256(native_valid),
        compiled_logits_sha256=_tensor_sha256(compiled_valid),
        literal_valid_query_logit_parity=literal,
        maximum_absolute_valid_query_logit_error=maximum,
    )


def verify_gemma3_native_literal_valid_query_parity(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3CausalCarrierTransportExecutor,
    model_inputs: Mapping[str, Tensor],
) -> Gemma3LiteralValidQueryParityReceipt:
    """Execute native and compiled Gemma once and require literal parity."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    if not isinstance(executor, Gemma3CausalCarrierTransportExecutor):
        raise TypeError("executor has the wrong type")
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, Tensor):
        raise ValueError("literal parity requires input_ids")
    sequence = adapter.prepare_sequence(model_inputs)
    with torch.inference_mode():
        native = adapter.forward(model_inputs)
        compiled = executor.forward_components(input_ids, sequence)
    receipt = verify_literal_valid_query_logit_parity(
        native_logits=native.logits,
        compiled_execution=compiled,
        valid_query_mask=sequence.query_valid_mask,
    )
    if not receipt.passed:
        raise RuntimeError("exact carrier-transport lowering lost native parity")
    return receipt


def assemble_gemma3_exact_causal_carrier_transport_executor(
    *,
    exact_executor: Gemma3JointCrossBlockChartExecutor,
    source_model_parameter_count: int,
    source_model_fingerprint: str,
    source_independence_audit: Mapping[str, object],
) -> Gemma3CausalCarrierTransportExecutor:
    """Lower an already-compiled exact 39-event executor into messages."""

    return Gemma3CausalCarrierTransportExecutor(
        exact_executor=exact_executor,
        source_model_parameter_count=source_model_parameter_count,
        source_model_fingerprint=source_model_fingerprint,
        source_independence_audit=source_independence_audit,
    )


def compile_gemma3_exact_causal_carrier_transport(
    adapter: Gemma3CausalLMAdapter,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Gemma3CausalCarrierTransportCompilation:
    """Clone pinned Gemma exactly and lower it to the message abstraction."""

    base = compile_gemma3_joint_exact_control(
        adapter,
        dtype=dtype,
        device=device,
    )
    executor = assemble_gemma3_exact_causal_carrier_transport_executor(
        exact_executor=base.executor,
        source_model_parameter_count=base.source_model_parameter_count,
        source_model_fingerprint=base.source_model_fingerprint,
        source_independence_audit=base.source_independence_audit,
    )
    return Gemma3CausalCarrierTransportCompilation(executor=executor)


__all__ = [
    "CarrierTransportObserver",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_ARM_ID",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_EXECUTION_SCHEMA",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_PROTOCOL_SHA256",
    "GEMMA3_CAUSAL_CARRIER_TRANSPORT_SCHEMA",
    "Gemma3AttentionTransportReceipt",
    "Gemma3CarrierTransportBoundaryReceipt",
    "Gemma3CarrierTransportMutationReceipt",
    "Gemma3CausalCarrierTransportCompilation",
    "Gemma3CausalCarrierTransportExecution",
    "Gemma3CausalCarrierTransportExecutor",
    "Gemma3LiteralValidQueryParityReceipt",
    "assemble_gemma3_exact_causal_carrier_transport_executor",
    "compile_gemma3_exact_causal_carrier_transport",
    "verify_gemma3_native_literal_valid_query_parity",
    "verify_literal_valid_query_logit_parity",
]
