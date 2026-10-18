"""Joint whole-model chart runtime for Qwen 3 dense causal LMs.

This module mirrors :mod:`gemma3_joint_cross_block_chart_executor` for the
Qwen 3 architecture.  The Gemma runtime file is source-pinned inside sealed
artifact manifests, so the Qwen executor lives in this parallel module
instead of editing frozen lineage.  All structurally generic machinery —
event specs, mutations, transport edges, the message session, the residual
carrier, and the observed-execution wrappers — is imported unchanged.

What differs is only what the architecture differs on:

* 28 decoder blocks instead of 18, hence a 59-event schedule
  (1 ingress + 56 mutations + 1 terminal norm + 1 egress) instead of 39.
* Every attention event is ``global``: Qwen 3 dense has no sliding-window
  layers, no per-layer attention-type alternation, and no logit softcap.
* Receipt and gradient-tape site inventories are parameterized by the
  28-block/56-mutation layout.

The runtime consumes already-fitted modules.  It does not load Qwen,
prompts, or a teacher, and it does not prescribe how Fisher rows or charts
are fitted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn

from .adapters import SequenceContext
from .gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    Gemma3JointCrossBlockChartExecutor,
    JointGraphAttentionOperatorOutputSnapshot,
    JointGraphAttentionRouteFactorObserver,
    JointGraphBackboneExecution,
    JointGraphCrossBlockTransportEdge,
    JointGraphEventGradientTape,
    JointGraphEventObserver,
    JointGraphEventSpec,
    JointGraphGeneratorMutation,
    JointGraphMutationRuntimeReceipt,
    JointGraphObservedBackboneExecution,
    JointGraphObservedExecution,
    JointGraphWholeModelExecution,
    _JointGraphMessageSession,
    _module_physical_spans,
    _module_scalar_count,
    _require_id,
    _tensor_value_sha256,
)
from .residual_carrier import (
    ResidualCarrierFactory,
    ResidualCarrierSession,
    ResidualGraphExecutionContext,
    ResidualMutationReceipt,
)


QWEN3_JOINT_BLOCK_COUNT = 28
QWEN3_JOINT_MUTATION_COUNT = 56
QWEN3_JOINT_EVENT_COUNT = 59

QWEN3_JOINT_CHART_ARM_ID = "qwen3_joint_cross_block_chart_candidate"
QWEN3_JOINT_EXACT_CONTROL_ARM_ID = "qwen3_joint_exact_59_event_control"


def build_qwen3_event_schedule(
    layer_count: int = QWEN3_JOINT_BLOCK_COUNT,
) -> tuple[JointGraphEventSpec, ...]:
    """Build the deterministic 59-event schedule used by the Qwen 3 arm.

    The topology matches the Gemma schedule exactly — ingress, alternating
    attention/MLP mutations per block, terminal norm, egress — but every
    attention event is ``global`` because Qwen 3 dense has no sliding-window
    layers.
    """

    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("layer_count must be positive")
    events = [
        JointGraphEventSpec(
            event_index=0,
            event_id="lexical.ingress",
            event_kind="lexical_ingress",
            block_ordinal=None,
            attention_kind=None,
            token_dependency="vocabulary_boundary",
        )
    ]
    for block in range(layer_count):
        attention_index = len(events)
        events.append(
            JointGraphEventSpec(
                event_index=attention_index,
                event_id=f"mutation.{attention_index}",
                event_kind="attention_mutation",
                block_ordinal=block,
                attention_kind="global",
                token_dependency="causal_prefix_mixer",
            )
        )
        mlp_index = len(events)
        events.append(
            JointGraphEventSpec(
                event_index=mlp_index,
                event_id=f"mutation.{mlp_index}",
                event_kind="mlp_mutation",
                block_ordinal=block,
                attention_kind=None,
                token_dependency="token_local",
            )
        )
    terminal_index = len(events)
    events.extend(
        (
            JointGraphEventSpec(
                event_index=terminal_index,
                event_id="terminal.rmsnorm",
                event_kind="terminal_rmsnorm",
                block_ordinal=None,
                attention_kind=None,
                token_dependency="token_local",
            ),
            JointGraphEventSpec(
                event_index=terminal_index + 1,
                event_id="lexical.egress",
                event_kind="lexical_egress",
                block_ordinal=None,
                attention_kind=None,
                token_dependency="vocabulary_boundary",
            ),
        )
    )
    return tuple(events)


def _qwen3_expected_observation_site_ids(
    mutation_count: int = QWEN3_JOINT_MUTATION_COUNT,
) -> tuple[str, ...]:
    return (
        "lexical.ingress.output",
        *(
            site_id
            for mutation_index in range(1, mutation_count + 1)
            for site_id in (
                f"mutation.{mutation_index}.input",
                f"mutation.{mutation_index}.output",
            )
        ),
    )


def _qwen3_expected_auxiliary_gradient_site_ids(
    mutation_count: int = QWEN3_JOINT_MUTATION_COUNT,
) -> tuple[str, ...]:
    return (
        *(
            f"mutation.{mutation_index}.operator_output"
            for mutation_index in range(1, mutation_count + 1, 2)
        ),
        "terminal.rmsnorm.output",
    )


class Qwen3JointGraphEventGradientTape(JointGraphEventGradientTape):
    """One-use exact suffix-gradient tape sized for the Qwen 3 site inventory."""

    __slots__ = ()

    def __init__(
        self,
        observations: Sequence[tuple[str, Tensor]],
        auxiliary_observations: Sequence[tuple[str, Tensor]],
    ) -> None:
        materialized = tuple(observations)
        expected = _qwen3_expected_observation_site_ids()
        if tuple(site_id for site_id, _ in materialized) != expected:
            raise ValueError("event gradient tape site order drifted")
        tensors = tuple(value for _, value in materialized)
        if any(
            not isinstance(value, Tensor)
            or value.ndim != 3
            or not value.requires_grad
            for value in tensors
        ):
            raise ValueError("event gradient tape requires live rank-three tensors")
        if len({id(value) for value in tensors}) != len(tensors):
            raise ValueError("event gradient tape tensors must be distinct")
        auxiliary = tuple(auxiliary_observations)
        expected_auxiliary = _qwen3_expected_auxiliary_gradient_site_ids()
        if tuple(site_id for site_id, _ in auxiliary) != expected_auxiliary:
            raise ValueError("auxiliary gradient tape site order drifted")
        auxiliary_tensors = tuple(value for _, value in auxiliary)
        if any(
            not isinstance(value, Tensor)
            or value.ndim != 3
            or not value.requires_grad
            for value in auxiliary_tensors
        ):
            raise ValueError(
                "auxiliary gradient tape requires live rank-three tensors"
            )
        if len({id(value) for value in auxiliary_tensors}) != len(
            auxiliary_tensors
        ):
            raise ValueError("auxiliary gradient tape tensors must be distinct")
        self._site_ids = expected
        self._tensors = tensors
        self._auxiliary_site_ids = expected_auxiliary
        self._auxiliary_tensors = auxiliary_tensors
        self._consumed = False


@dataclass(frozen=True, slots=True)
class Qwen3JointGraphBackboneExecution(JointGraphBackboneExecution):
    """Inspectable 58-event Qwen execution stopped before lexical egress."""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output, Tensor)
            or self.output.ndim != 3
            or not isinstance(self.normalized_output, Tensor)
            or self.normalized_output.shape != self.output.shape
        ):
            raise ValueError("joint graph backbone tensors are invalid")
        if len(self.event_ids) != QWEN3_JOINT_EVENT_COUNT - 1:
            raise ValueError("joint graph backbone did not execute 58 events")
        if self.event_ids[-1] != "terminal.rmsnorm":
            raise ValueError("joint graph backbone event boundary drifted")
        expected_operator_sites = (
            _qwen3_expected_auxiliary_gradient_site_ids()[:-1]
        )
        if (
            not isinstance(self.attention_operator_output_snapshots, tuple)
            or len(self.attention_operator_output_snapshots)
            != QWEN3_JOINT_BLOCK_COUNT
            or any(
                not isinstance(
                    snapshot,
                    JointGraphAttentionOperatorOutputSnapshot,
                )
                for snapshot in self.attention_operator_output_snapshots
            )
            or tuple(
                snapshot.site_id
                for snapshot in self.attention_operator_output_snapshots
            )
            != expected_operator_sites
            or tuple(
                snapshot.block_ordinal
                for snapshot in self.attention_operator_output_snapshots
            )
            != tuple(range(QWEN3_JOINT_BLOCK_COUNT))
            or any(
                snapshot.values.shape != self.normalized_output.shape
                for snapshot in self.attention_operator_output_snapshots
            )
        ):
            raise ValueError(
                "attention operator output snapshot inventory drifted"
            )
        if self.carrier_version != QWEN3_JOINT_MUTATION_COUNT:
            raise ValueError("joint graph residual carrier did not close")
        if len(self.mutation_receipts) != self.carrier_version:
            raise ValueError("joint graph mutation receipts are incomplete")
        if len(self.operator_receipts) != self.carrier_version:
            raise ValueError("joint graph operator receipts are incomplete")
        if tuple(item.event_id for item in self.operator_receipts) != tuple(
            item.mutation_id for item in self.mutation_receipts
        ):
            raise ValueError("operator and carrier receipt order disagrees")
        for label, values in (
            ("chart input", self.mutation_chart_input_sha256s),
            ("delta", self.mutation_delta_sha256s),
        ):
            if (
                len(values) != self.carrier_version
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in values
                )
            ):
                raise ValueError(f"mutation {label} SHA-256 inventory is invalid")


@dataclass(frozen=True, slots=True)
class Qwen3JointGraphWholeModelExecution(JointGraphWholeModelExecution):
    """Inspectable result of one complete 59-event Qwen graph traversal."""

    def __post_init__(self) -> None:
        if len(self.event_ids) != QWEN3_JOINT_EVENT_COUNT:
            raise ValueError("joint graph did not execute exactly 59 events")
        if self.carrier_version != QWEN3_JOINT_MUTATION_COUNT:
            raise ValueError("joint graph residual carrier did not close")
        if len(self.mutation_receipts) != self.carrier_version:
            raise ValueError("joint graph mutation receipts are incomplete")
        if len(self.operator_receipts) != self.carrier_version:
            raise ValueError("joint graph operator receipts are incomplete")
        if tuple(item.event_id for item in self.operator_receipts) != tuple(
            item.mutation_id for item in self.mutation_receipts
        ):
            raise ValueError("operator and carrier receipt order disagrees")
        for label, values in (
            ("chart input", self.mutation_chart_input_sha256s),
            ("delta", self.mutation_delta_sha256s),
        ):
            if (
                len(values) != self.carrier_version
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in values
                )
            ):
                raise ValueError(f"mutation {label} SHA-256 inventory is invalid")


class Qwen3JointCrossBlockChartExecutor(Gemma3JointCrossBlockChartExecutor):
    """Standalone 59-event executor for a fitted joint Qwen 3 candidate.

    Inherits the Gemma executor's public surface so the incremental decode
    session and downstream tooling recognize it, but owns a Qwen schedule
    and Qwen count validations.  Construction and the execution internals
    are re-implemented here because the pinned base validates the Gemma
    39-event inventory at the module level.
    """

    def __init__(
        self,
        *,
        lexical: nn.Module,
        mutations: Sequence[JointGraphGeneratorMutation],
        transports: Sequence[JointGraphCrossBlockTransportEdge],
        terminal_norm: nn.Module,
        final_logit_softcap: float | None = None,
        arm_id: str = QWEN3_JOINT_CHART_ARM_ID,
        carrier_factory: ResidualCarrierFactory | None = None,
        layer_count: int = QWEN3_JOINT_BLOCK_COUNT,
    ) -> None:
        nn.Module.__init__(self)
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if not isinstance(lexical, nn.Module) or not all(
            callable(getattr(lexical, method, None))
            for method in ("ingress", "egress")
        ):
            raise TypeError("lexical must expose tied ingress() and egress()")
        if not isinstance(terminal_norm, nn.Module):
            raise TypeError("terminal_norm must be an nn.Module")
        _require_id(arm_id, label="arm_id")
        if arm_id not in {
            QWEN3_JOINT_CHART_ARM_ID,
            QWEN3_JOINT_EXACT_CONTROL_ARM_ID,
        }:
            raise ValueError("qwen3 joint executor arm_id is unsupported")
        if final_logit_softcap is not None and (
            not isinstance(final_logit_softcap, (int, float))
            or isinstance(final_logit_softcap, bool)
            or not math.isfinite(float(final_logit_softcap))
            or float(final_logit_softcap) <= 0
        ):
            raise ValueError("final_logit_softcap must be positive when set")
        if carrier_factory is not None and not isinstance(
            carrier_factory,
            ResidualCarrierFactory,
        ):
            raise TypeError("carrier_factory does not implement the protocol")

        schedule = build_qwen3_event_schedule(layer_count)
        mutation_specs = schedule[1:-2]
        declared_mutations = tuple(mutations)
        mutation_count = 2 * layer_count
        if (
            len(declared_mutations) != mutation_count
            or any(
                not isinstance(item, JointGraphGeneratorMutation)
                for item in declared_mutations
            )
            or tuple(item.event_spec for item in declared_mutations)
            != mutation_specs
        ):
            raise ValueError(
                f"mutations must exactly cover the ordered {mutation_count}-event body"
            )
        hidden_widths = {item.hidden_width for item in declared_mutations}
        if len(hidden_widths) != 1:
            raise ValueError("joint mutations must share one residual width")

        declared_edges = tuple(transports)
        if any(
            not isinstance(edge, JointGraphCrossBlockTransportEdge)
            for edge in declared_edges
        ):
            raise TypeError("transports must be joint graph edges")
        event_by_id = {event.event_id: event for event in mutation_specs}
        if len({edge.spec.transport_id for edge in declared_edges}) != len(
            declared_edges
        ):
            raise ValueError("transport ids must be unique")
        coordinates: set[tuple[str, int, str, int]] = set()
        cross_block_count = 0
        cross_family_count = 0
        cross_block_cross_family_count = 0
        for edge in declared_edges:
            source = event_by_id.get(edge.spec.source_event_id)
            target = event_by_id.get(edge.spec.target_event_id)
            if source is None or target is None:
                raise ValueError("transport references a non-mutation event")
            if source.event_index >= target.event_index:
                raise ValueError("transport edges must be strictly forward")
            source_mutation = declared_mutations[source.event_index - 1]
            target_mutation = declared_mutations[target.event_index - 1]
            if (
                edge.spec.source_chart_index
                >= source_mutation.emitted_chart_count
                or edge.spec.target_chart_index
                >= target_mutation.incoming_chart_count
                or edge.source_generator_rank
                != source_mutation.emitted_generator_rank
                or edge.target_generator_rank
                != target_mutation.incoming_generator_rank
            ):
                raise ValueError("transport does not match endpoint charts")
            coordinate = (
                edge.spec.source_event_id,
                edge.spec.source_chart_index,
                edge.spec.target_event_id,
                edge.spec.target_chart_index,
            )
            if coordinate in coordinates:
                raise ValueError("duplicate chart transport is forbidden")
            coordinates.add(coordinate)
            if source.block_ordinal != target.block_ordinal:
                cross_block_count += 1
            if source.component_family != target.component_family:
                cross_family_count += 1
            if (
                source.block_ordinal != target.block_ordinal
                and source.component_family != target.component_family
            ):
                cross_block_cross_family_count += 1

        if arm_id == QWEN3_JOINT_CHART_ARM_ID and (
            not declared_edges
            or cross_block_cross_family_count == 0
        ):
            raise ValueError(
                "candidate requires a cross-block, cross-family transport"
            )
        lexical_metadata = getattr(lexical, "metadata", None)
        tie_receipt = (
            lexical_metadata() if callable(lexical_metadata) else None
        )
        observed_lexical_storage_count = len(
            _module_physical_spans(lexical)
        )
        if (
            not isinstance(tie_receipt, Mapping)
            or tie_receipt.get(
                "ingress_and_egress_share_the_same_storage"
            )
            is not True
            or tie_receipt.get("second_head_copy_present") is not False
            or tie_receipt.get("physical_storage_count")
            != observed_lexical_storage_count
            or observed_lexical_storage_count != 1
        ):
            raise ValueError(
                "lexical module did not authenticate one tied physical factor"
            )
        if arm_id == QWEN3_JOINT_CHART_ARM_ID and any(
            item.source_checkpoint_coefficients_retained
            for item in declared_mutations
        ):
            raise ValueError(
                "joint candidate cannot retain source checkpoint mutations"
            )

        self.lexical = lexical
        self.mutations = nn.ModuleList(declared_mutations)
        self.transports = nn.ModuleList(declared_edges)
        self.terminal_norm = terminal_norm
        self.arm_id = arm_id
        self.final_logit_softcap = (
            None
            if final_logit_softcap is None
            else float(final_logit_softcap)
        )
        self._schedule = schedule
        self._carrier_factory = carrier_factory
        self._layer_count = layer_count
        self._mutation_count = mutation_count
        self._cross_block_transport_count = cross_block_count
        self._cross_family_transport_count = cross_family_count
        self._cross_block_cross_family_transport_count = (
            cross_block_cross_family_count
        )
        self.eval().requires_grad_(False)

    def _forward_components_impl(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        observer: JointGraphEventObserver | None,
        attention_route_factor_observer: (
            JointGraphAttentionRouteFactorObserver | None
        ),
        retain_gradient_tape: bool,
        materialize_logits: bool,
        shadow_source_site_id: str | None = None,
        shadow_target_event_id: str | None = None,
        shadow_perturbation: Tensor | None = None,
    ) -> JointGraphObservedExecution | JointGraphObservedBackboneExecution:
        if observer is not None and not callable(observer):
            raise TypeError("event observer must be callable")
        if (
            attention_route_factor_observer is not None
            and not callable(attention_route_factor_observer)
        ):
            raise TypeError(
                "attention route factor observer must be callable"
            )
        if (
            attention_route_factor_observer is not None
            and self.arm_id != QWEN3_JOINT_EXACT_CONTROL_ARM_ID
        ):
            raise ValueError(
                "attention route factors require the exact control arm"
            )
        if type(retain_gradient_tape) is not bool:
            raise TypeError("retain_gradient_tape must be bool")
        if type(materialize_logits) is not bool:
            raise TypeError("materialize_logits must be bool")
        if not materialize_logits and not retain_gradient_tape:
            raise ValueError("logits-free execution requires a gradient tape")
        if retain_gradient_tape and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "event score gradients cannot be captured in inference mode"
            )
        shadow_requested = any(
            value is not None
            for value in (
                shadow_source_site_id,
                shadow_target_event_id,
                shadow_perturbation,
            )
        )
        source_event: JointGraphEventSpec | None = None
        source_role: str | None = None
        target_event: JointGraphEventSpec | None = None
        if shadow_requested:
            if (
                self.arm_id != QWEN3_JOINT_EXACT_CONTROL_ARM_ID
                or not isinstance(shadow_source_site_id, str)
                or not isinstance(shadow_target_event_id, str)
                or not isinstance(shadow_perturbation, Tensor)
            ):
                raise ValueError(
                    "shadow intervention requires an exact control and complete endpoints"
                )
            event_by_id = {event.event_id: event for event in self._schedule}
            try:
                source_event_id, source_role = shadow_source_site_id.rsplit(
                    ".",
                    1,
                )
            except ValueError as error:
                raise ValueError("shadow source must be an event chart site") from error
            source_event = event_by_id.get(source_event_id)
            target_event = event_by_id.get(shadow_target_event_id)
            if (
                source_event is None
                or target_event is None
                or source_role not in {"input", "output"}
                or (
                    source_event.event_kind == "lexical_ingress"
                    and source_role != "output"
                )
                or (
                    source_event.event_kind
                    in {"attention_mutation", "mlp_mutation"}
                    and source_role not in {"input", "output"}
                )
                or source_event.event_kind
                not in {"lexical_ingress", "attention_mutation", "mlp_mutation"}
                or target_event.event_kind
                not in {"attention_mutation", "mlp_mutation"}
                or source_event.event_index >= target_event.event_index
            ):
                raise ValueError(
                    "shadow intervention requires a forward output-to-mutation edge"
                )
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "input_ids must be an integer [batch, sequence] tensor"
            )
        if input_ids.shape != (
            sequence.batch_size,
            sequence.query_length,
        ):
            raise ValueError("input_ids do not match sequence context")
        context = ResidualGraphExecutionContext.bind(sequence)
        initial = self.lexical.ingress(input_ids)
        if (
            not isinstance(initial, Tensor)
            or initial.shape != (*input_ids.shape, self.hidden_width)
        ):
            raise ValueError("lexical ingress returned an invalid carrier")
        if shadow_requested:
            assert isinstance(shadow_perturbation, Tensor)
            if (
                shadow_perturbation.shape != initial.shape
                or shadow_perturbation.dtype != initial.dtype
                or shadow_perturbation.device != initial.device
                or not bool(torch.isfinite(shadow_perturbation).all())
            ):
                raise ValueError(
                    "shadow perturbation must match the exact residual carrier"
                )
        shadow_applied = False
        shadow_target_input: Tensor | None = None
        shadow_target_flow: Tensor | None = None
        if source_event is not None and source_event.event_kind == "lexical_ingress":
            assert isinstance(shadow_perturbation, Tensor)
            initial = initial + shadow_perturbation.masked_fill(
                ~context.query_valid_mask.unsqueeze(-1),
                0,
            )
            shadow_applied = True
        if retain_gradient_tape:
            initial = initial.detach().requires_grad_(True)
        live_observations: list[tuple[str, Tensor]] = []
        live_auxiliary_observations: list[tuple[str, Tensor]] = []
        detached_attention_operator_outputs: list[
            JointGraphAttentionOperatorOutputSnapshot
        ] = []

        def observe(event_id: str, role: str, value: Tensor) -> None:
            site_id = f"{event_id}.{role}"
            if retain_gradient_tape:
                if not value.requires_grad:
                    raise RuntimeError(
                        f"observed site {site_id} is detached from the exact suffix"
                    )
                live_observations.append((site_id, value))
            if observer is not None:
                # The callback can mutate its private snapshot or return any
                # value without gaining a write path into exact execution.
                observer(
                    event_id=event_id,
                    role=role,
                    values=value.detach().clone(),
                )

        session = ResidualCarrierSession.begin(
            initial_state=initial,
            context=context,
            expected_mutation_ids=self.expected_mutation_ids,
            wire_id="qwen3_joint_cross_block_chart",
            factory=self._carrier_factory,
        )
        mutation_by_id = {
            mutation.event_spec.event_id: mutation
            for mutation in self.mutations
        }
        message_session = _JointGraphMessageSession(
            mutations=mutation_by_id,
            edges=tuple(self.transports),
            valid_positions=context.query_valid_mask,
            event_indexes={
                event.event_id: event.event_index
                for event in self._schedule[1:-2]
            },
            dtype=initial.dtype,
            device=initial.device,
        )
        try:
            observe("lexical.ingress", "output", initial)
            operator_receipts: list[JointGraphMutationRuntimeReceipt] = []
            mutation_chart_input_sha256s: list[str] = []
            mutation_delta_sha256s: list[str] = []
            for mutation in self.mutations:
                incoming = message_session.incoming_for(mutation)
                input_perturbation = (
                    shadow_perturbation
                    if (
                        source_event is not None
                        and source_role == "input"
                        and source_event.event_id
                        == mutation.event_spec.event_id
                    )
                    else None
                )
                execution = mutation.execute_with_incoming(
                    session.materialize(),
                    sequence,
                    incoming,
                    chart_input_perturbation=input_perturbation,
                    capture_attention_route_factors=(
                        attention_route_factor_observer is not None
                        and mutation.event_spec.event_kind
                        == "attention_mutation"
                    ),
                )
                applied_delta = execution.delta
                if (
                    source_event is not None
                    and source_role == "output"
                    and source_event.event_id == mutation.event_spec.event_id
                ):
                    assert isinstance(shadow_perturbation, Tensor)
                    applied_delta = applied_delta + shadow_perturbation.masked_fill(
                        ~context.query_valid_mask.unsqueeze(-1),
                        0,
                    )
                    shadow_applied = True
                elif input_perturbation is not None:
                    shadow_applied = True
                mutation_chart_input_sha256s.append(
                    _tensor_value_sha256(execution.chart_input)
                )
                mutation_delta_sha256s.append(
                    _tensor_value_sha256(applied_delta)
                )
                if (
                    target_event is not None
                    and target_event.event_id == mutation.event_spec.event_id
                ):
                    shadow_target_input = execution.chart_input
                    shadow_target_flow = execution.delta
                observe(
                    mutation.event_spec.event_id,
                    "input",
                    execution.chart_input,
                )
                observe(
                    mutation.event_spec.event_id,
                    "output",
                    applied_delta,
                )
                if mutation.event_spec.event_kind == "attention_mutation":
                    operator_output = execution.attention_operator_output
                    if retain_gradient_tape:
                        if not isinstance(operator_output, Tensor):
                            raise RuntimeError(
                                "attention mutation omitted its operator output"
                            )
                        if not operator_output.requires_grad:
                            raise RuntimeError(
                                "attention operator output is detached from the exact suffix"
                            )
                        live_auxiliary_observations.append(
                            (
                                f"{mutation.event_spec.event_id}.operator_output",
                                operator_output,
                            )
                        )
                    if not materialize_logits:
                        if not isinstance(operator_output, Tensor):
                            raise RuntimeError(
                                "attention mutation omitted its operator output"
                            )
                        detached_attention_operator_outputs.append(
                            JointGraphAttentionOperatorOutputSnapshot(
                                site_id=(
                                    f"{mutation.event_spec.event_id}.operator_output"
                                ),
                                event_id=mutation.event_spec.event_id,
                                block_ordinal=(
                                    mutation.event_spec.block_ordinal
                                ),
                                values=operator_output,
                            )
                        )
                    if attention_route_factor_observer is not None:
                        snapshot = execution.attention_route_factor_snapshot
                        if snapshot is None:
                            raise RuntimeError(
                                "attention mutation omitted requested route factors"
                            )
                        attention_route_factor_observer(
                            event_id=mutation.event_spec.event_id,
                            block_ordinal=mutation.event_spec.block_ordinal,
                            snapshot=snapshot,
                        )
                session.apply_mutation(
                    mutation.event_spec.event_id,
                    applied_delta,
                )
                message_session.publish(
                    mutation,
                    execution.emitted_generator_latents,
                )
                operator_receipts.append(execution.runtime_receipt)
            message_session.finish(self.expected_mutation_ids)
            receipts = session.receipts
            carrier_version = session.version
            output = session.materialize()
            final_carrier = session.finish()
        except BaseException:
            if session.active:
                session.abort()
            if not message_session.closed:
                message_session.abort()
            raise
        if (
            carrier_version != self._mutation_count
            or final_carrier.version != carrier_version
            or tuple(receipt.mutation_id for receipt in receipts)
            != self.expected_mutation_ids
        ):
            raise RuntimeError("joint graph residual carrier closure drifted")
        if shadow_requested and (
            not shadow_applied
            or shadow_target_input is None
            or shadow_target_flow is None
        ):
            raise RuntimeError("shadow intervention did not traverse its endpoints")
        normalized = self.terminal_norm(output)
        if not isinstance(normalized, Tensor) or normalized.shape != output.shape:
            raise ValueError("terminal normalization must preserve carrier shape")
        if retain_gradient_tape:
            if not normalized.requires_grad:
                raise RuntimeError(
                    "terminal normalized output is detached from the exact suffix"
                )
            live_auxiliary_observations.append(
                ("terminal.rmsnorm.output", normalized)
            )
        tape = (
            Qwen3JointGraphEventGradientTape(
                live_observations,
                live_auxiliary_observations,
            )
            if retain_gradient_tape
            else None
        )
        common_execution = {
            "output": output,
            "normalized_output": normalized,
            "mutation_receipts": receipts,
            "operator_receipts": tuple(operator_receipts),
            "mutation_chart_input_sha256s": tuple(
                mutation_chart_input_sha256s
            ),
            "mutation_delta_sha256s": tuple(mutation_delta_sha256s),
            "carrier_version": carrier_version,
            "transport_edge_count": len(self.transports),
            "peak_live_transport_packets": (
                message_session.peak_live_packets
            ),
            "peak_live_transport_packet_bytes": (
                message_session.peak_live_packet_bytes
            ),
        }
        if not materialize_logits:
            if tape is None:
                raise RuntimeError("logits-free execution omitted its tape")
            return JointGraphObservedBackboneExecution(
                execution=Qwen3JointGraphBackboneExecution(
                    attention_operator_output_snapshots=tuple(
                        detached_attention_operator_outputs
                    ),
                    event_ids=self.ordered_event_ids[:-1],
                    **common_execution,
                ),
                gradient_tape=tape,
            )
        logits = self.lexical.egress(normalized)
        if (
            not isinstance(logits, Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] <= 0
        ):
            raise ValueError("lexical egress returned invalid logits")
        if self.final_logit_softcap is not None:
            cap = self.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        execution = Qwen3JointGraphWholeModelExecution(
            logits=logits,
            event_ids=self.ordered_event_ids,
            **common_execution,
        )
        return JointGraphObservedExecution(
            execution=execution,
            gradient_tape=tape,
            shadow_source_site_id=(
                None
                if source_event is None or source_role is None
                else f"{source_event.event_id}.{source_role}"
            ),
            shadow_target_site_id=(
                None
                if target_event is None
                else f"{target_event.event_id}.output"
            ),
            shadow_target_input=shadow_target_input,
            shadow_target_flow=shadow_target_flow,
        )


__all__ = [
    "QWEN3_JOINT_BLOCK_COUNT",
    "QWEN3_JOINT_CHART_ARM_ID",
    "QWEN3_JOINT_EVENT_COUNT",
    "QWEN3_JOINT_EXACT_CONTROL_ARM_ID",
    "QWEN3_JOINT_MUTATION_COUNT",
    "Qwen3JointCrossBlockChartExecutor",
    "Qwen3JointGraphBackboneExecution",
    "Qwen3JointGraphEventGradientTape",
    "Qwen3JointGraphWholeModelExecution",
    "build_qwen3_event_schedule",
]
