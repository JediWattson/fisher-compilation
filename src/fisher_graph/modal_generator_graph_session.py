"""Incremental traversal of an authenticated modal-generator graph.

The batch graph executor accepts every boundary input at once.  A transformer
model produces those boundaries sequentially as it walks through its layers.
This module provides the corresponding online session: feed one declared
boundary when it becomes available, execute every ready generator at that
boundary, carry only modal states needed by future interactions, and release a
state after its final consumer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .modal_generator_graph import (
    ModalGeneratorGraphExecution,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
)


__all__ = [
    "ModalGeneratorBoundaryExecution",
    "ModalGeneratorGraphSession",
]


@dataclass(frozen=True, slots=True)
class ModalGeneratorBoundaryExecution:
    """Outputs that became complete after feeding one runtime boundary."""

    input_boundary: str
    executed_nodes: tuple[str, ...]
    ready_outputs: dict[str, Tensor]
    live_state_width: int
    peak_live_state_width: int


class ModalGeneratorGraphSession:
    """One non-reentrant, sequential execution of a modal-generator DAG."""

    def __init__(
        self,
        plan: ModalGeneratorGraphPlan,
        *,
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
    ) -> None:
        if not isinstance(plan, ModalGeneratorGraphPlan):
            raise TypeError("plan must be a ModalGeneratorGraphPlan")
        plan.validate_integrity()
        self.plan = ModalGeneratorGraphPlan.from_state_dict(plan.state_dict())
        if type(capture_modal_states) is not bool or type(
            capture_edge_messages
        ) is not bool:
            raise TypeError("capture flags must be bool")

        nodes_by_boundary: defaultdict[str, list[ModalGeneratorNode]] = (
            defaultdict(list)
        )
        incoming: defaultdict[str, list[ModalGeneratorInteraction]] = (
            defaultdict(list)
        )
        outgoing_counts: defaultdict[str, int] = defaultdict(int)
        output_producers: defaultdict[str, set[str]] = defaultdict(set)
        for node in self.plan.nodes:
            nodes_by_boundary[node.input_boundary].append(node)
            output_producers[node.output_boundary].add(node.name)
        for edge in self.plan.interactions:
            incoming[edge.target_node].append(edge)
            outgoing_counts[edge.source_node] += 1

        self._nodes_by_boundary = {
            boundary: tuple(nodes)
            for boundary, nodes in nodes_by_boundary.items()
        }
        self._incoming = {
            name: tuple(edges) for name, edges in incoming.items()
        }
        self._remaining_consumers = {
            node.name: outgoing_counts[node.name] for node in self.plan.nodes
        }
        self._output_producers = {
            boundary: frozenset(names)
            for boundary, names in output_producers.items()
        }
        self._capture_modal_states = capture_modal_states
        self._capture_edge_messages = capture_edge_messages
        self._captured_states: dict[str, Tensor] | None = (
            {} if capture_modal_states else None
        )
        self._captured_messages: dict[str, Tensor] | None = (
            {} if capture_edge_messages else None
        )
        self._states: dict[str, Tensor] = {}
        self._pending_outputs: dict[str, Tensor] = {}
        self._executed_nodes: set[str] = set()
        self._fed_boundaries: set[str] = set()
        self._traversal: list[str] = []
        self._peak_live_state_width = 0
        self._finished = False

    @staticmethod
    def _runtime_weight(weight: Tensor, like: Tensor) -> Tensor:
        result = weight.to(device=like.device, dtype=like.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(
                "modal graph weight is not finite in the runtime dtype"
            )
        return result

    @property
    def expected_input_boundaries(self) -> tuple[str, ...]:
        return tuple(self._nodes_by_boundary)

    @property
    def fed_input_boundaries(self) -> tuple[str, ...]:
        return tuple(
            boundary
            for boundary in self.expected_input_boundaries
            if boundary in self._fed_boundaries
        )

    @property
    def remaining_input_boundaries(self) -> tuple[str, ...]:
        return tuple(
            boundary
            for boundary in self.expected_input_boundaries
            if boundary not in self._fed_boundaries
        )

    @property
    def traversal_order(self) -> tuple[str, ...]:
        return tuple(self._traversal)

    @property
    def live_state_width(self) -> int:
        return sum(
            next(
                node.latent_width
                for node in self.plan.nodes
                if node.name == name
            )
            for name in self._states
        )

    @property
    def peak_live_state_width(self) -> int:
        return self._peak_live_state_width

    @property
    def ready_output_boundaries(self) -> tuple[str, ...]:
        return tuple(
            boundary
            for boundary, producers in self._output_producers.items()
            if boundary in self._pending_outputs
            and producers.issubset(self._executed_nodes)
        )

    def _validate_boundary_input(
        self,
        boundary: str,
        value: Tensor,
    ) -> Tensor:
        if not isinstance(boundary, str) or boundary not in self._nodes_by_boundary:
            raise ValueError("input boundary is not declared by the graph")
        if boundary in self._fed_boundaries:
            raise RuntimeError("an input boundary may be fed only once")
        width = self.plan.input_boundary_widths[boundary]
        if (
            not isinstance(value, Tensor)
            or value.ndim < 1
            or value.shape[-1] != width
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"boundary {boundary!r} must be a nonempty finite floating "
                f"Tensor with trailing width {width}"
            )
        return value

    def _validate_boundary_causal_readiness(self, boundary: str) -> None:
        available = set(self._states)
        for node in self._nodes_by_boundary[boundary]:
            for edge in self._incoming.get(node.name, ()):
                if edge.source_node not in available:
                    raise RuntimeError(
                        "input boundary was fed before a causal source node "
                        "executed"
                    )
            available.add(node.name)

    def _edge_message(
        self,
        edge: ModalGeneratorInteraction,
        latent: Tensor,
    ) -> Tensor:
        try:
            source_state = self._states[edge.source_node]
        except KeyError as error:
            if edge.source_node in self._executed_nodes:
                raise RuntimeError(
                    "a modal source state was released before its consumer"
                ) from error
            raise RuntimeError(
                "input boundary was fed before a causal source node executed"
            ) from error
        if (
            source_state.shape[:-1] != latent.shape[:-1]
            or source_state.device != latent.device
            or source_state.dtype != latent.dtype
        ):
            raise ValueError(
                f"interaction {edge.key!r} runtime batch, device, or dtype "
                "dimensions drifted"
            )
        message = source_state @ self._runtime_weight(
            edge.message_matrix,
            source_state,
        )
        if edge.message_bias is not None:
            message = message + self._runtime_weight(
                edge.message_bias,
                message,
            )
        if self._captured_messages is not None:
            self._captured_messages[edge.key] = message.detach().clone()
        return message

    def _release_consumed_sources(
        self,
        edges: tuple[ModalGeneratorInteraction, ...],
    ) -> None:
        for edge in edges:
            remaining = self._remaining_consumers[edge.source_node] - 1
            if remaining < 0:
                raise RuntimeError("modal source consumer accounting underflowed")
            self._remaining_consumers[edge.source_node] = remaining
            if remaining == 0:
                try:
                    del self._states[edge.source_node]
                except KeyError as error:
                    raise RuntimeError(
                        "modal source state expired before final consumption"
                    ) from error

    def feed(
        self,
        input_boundary: str,
        value: Tensor,
    ) -> ModalGeneratorBoundaryExecution:
        """Execute all graph nodes driven by one newly available boundary."""

        # The session exposes its cloned plan for inspection.  Revalidate on
        # every state transition so in-place tensor mutation cannot execute
        # beneath stale artifact hashes.
        self.plan.validate_integrity()
        if self._finished:
            raise RuntimeError("modal graph session is already finished")
        own_input = self._validate_boundary_input(input_boundary, value)
        # Fail before mutating any session state when this entire boundary is
        # not causally ready.
        self._validate_boundary_causal_readiness(input_boundary)
        executed: list[str] = []
        for node in self._nodes_by_boundary[input_boundary]:
            weights = node.weights
            latent = own_input @ self._runtime_weight(
                weights.input_factor,
                own_input,
            )
            if weights.state_factor is not None:
                latent = latent @ self._runtime_weight(
                    weights.state_factor,
                    latent,
                )
            if weights.latent_bias is not None:
                latent = latent + self._runtime_weight(
                    weights.latent_bias,
                    latent,
                )
            edges = self._incoming.get(node.name, ())
            for edge in edges:
                latent = latent + self._edge_message(edge, latent)
            if not bool(torch.isfinite(latent).all()):
                raise ValueError(
                    f"modal state for node {node.name!r} became non-finite"
                )
            self._states[node.name] = latent
            self._peak_live_state_width = max(
                self._peak_live_state_width,
                self.live_state_width,
            )
            if self._captured_states is not None:
                self._captured_states[node.name] = latent.detach().clone()

            contribution = latent @ self._runtime_weight(
                weights.output_factor,
                latent,
            )
            if weights.output_bias is not None:
                contribution = contribution + self._runtime_weight(
                    weights.output_bias,
                    contribution,
                )
            if not bool(torch.isfinite(contribution).all()):
                raise ValueError(
                    f"residual contribution for node {node.name!r} "
                    "became non-finite"
                )
            prior = self._pending_outputs.get(node.output_boundary)
            if prior is None:
                self._pending_outputs[node.output_boundary] = contribution
            else:
                if (
                    prior.shape != contribution.shape
                    or prior.dtype != contribution.dtype
                    or prior.device != contribution.device
                ):
                    raise ValueError(
                        f"output boundary {node.output_boundary!r} runtime "
                        "dimensions drifted"
                    )
                self._pending_outputs[node.output_boundary] = (
                    prior + contribution
                )
            self._executed_nodes.add(node.name)
            self._traversal.append(node.name)
            executed.append(node.name)
            self._release_consumed_sources(edges)
            if self._remaining_consumers[node.name] == 0:
                del self._states[node.name]

        self._fed_boundaries.add(input_boundary)
        ready = {
            boundary: self._pending_outputs[boundary]
            for boundary in self.ready_output_boundaries
        }
        return ModalGeneratorBoundaryExecution(
            input_boundary=input_boundary,
            executed_nodes=tuple(executed),
            ready_outputs=ready,
            live_state_width=self.live_state_width,
            peak_live_state_width=self.peak_live_state_width,
        )

    def pop_output(self, output_boundary: str) -> Tensor:
        """Release and return one complete output boundary."""

        if output_boundary not in self.ready_output_boundaries:
            raise RuntimeError("output boundary is not complete")
        return self._pending_outputs.pop(output_boundary)

    def finish(self) -> ModalGeneratorGraphExecution:
        """Verify complete traversal and return any unconsumed outputs/captures."""

        self.plan.validate_integrity()
        if self._finished:
            raise RuntimeError("modal graph session is already finished")
        missing = set(node.name for node in self.plan.nodes) - self._executed_nodes
        if missing:
            raise RuntimeError(
                "modal graph traversal is incomplete; missing nodes "
                f"{sorted(missing)}"
            )
        if self._states:
            raise RuntimeError("modal graph session leaked live modal states")
        self._finished = True
        return ModalGeneratorGraphExecution(
            outputs=dict(self._pending_outputs),
            traversal_order=tuple(self._traversal),
            modal_states=(
                None
                if self._captured_states is None
                else dict(self._captured_states)
            ),
            edge_messages=(
                None
                if self._captured_messages is None
                else dict(self._captured_messages)
            ),
        )
