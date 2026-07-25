"""Hard-routed execution primitives for conditional Fisher modes.

The classes in this module make two materially different execution paths
explicit:

* :class:`ConditionalModalProjectionOracleExecutor` runs a native source
  layer first and conditionally projects its output.  It is an oracle for
  measuring route quality, not a source-layer compute optimization.
* :class:`HardRoutedSpecialistBank` routes directly from the current block
  input and calls only specialists that own selected token rows.  It is the
  minimal execution contract a future standalone conditional graph can
  implement to realize branch-level savings.

Both paths use a :class:`~fisher_graph.conditional_routing.ConditionalModalRoutingPlan`.
Consequently, route decisions are pointwise functions of the incoming hidden
state.  The executor never observes the native output, Fisher need profile, or
future tensor positions when selecting a route.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .conditional_routing import ConditionalModalRoutingPlan
from .layers import LayerExecutor
from .modes import FisherModeBasis


def _validate_block_inputs(
    block_inputs: Tensor,
    plan: ConditionalModalRoutingPlan,
) -> None:
    if not isinstance(block_inputs, Tensor):
        raise TypeError("block_inputs must be a Tensor")
    if (
        block_inputs.ndim < 2
        or block_inputs.shape[-1] != plan.router.input_features
    ):
        raise ValueError(
            "block_inputs must have shape [..., router input features]"
        )
    if not block_inputs.is_floating_point():
        raise ValueError("block_inputs must be floating point")
    if not torch.isfinite(block_inputs).all():
        raise ValueError("block_inputs must be finite")


def _valid_token_mask(
    leading_shape: torch.Size,
    valid_mask: Tensor | None,
    *,
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones(leading_shape, dtype=torch.bool, device=device)
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a Tensor")
    if valid_mask.shape != leading_shape:
        raise ValueError("valid_mask must match the token leading dimensions")
    if valid_mask.device != device:
        raise ValueError("valid_mask must share the input device")
    if valid_mask.dtype is torch.bool:
        return valid_mask
    if valid_mask.is_floating_point() and not torch.isfinite(valid_mask).all():
        raise ValueError("valid_mask must be finite")
    if not bool(((valid_mask == 0) | (valid_mask == 1)).all()):
        raise ValueError("valid_mask must be boolean or binary")
    return valid_mask.to(dtype=torch.bool)


def _validate_route_ids(
    route_ids: Tensor,
    *,
    leading_shape: torch.Size,
    route_count: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(route_ids, Tensor):
        raise TypeError("route_ids must be a Tensor")
    if route_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("route_ids must use an integer dtype")
    if route_ids.shape != leading_shape:
        raise ValueError("route_ids must match the token leading dimensions")
    if route_ids.device != device:
        raise ValueError("route_ids must share the input device")
    routes = route_ids.to(dtype=torch.int64)
    if routes.numel() and (
        int(routes.min().item()) < 0
        or int(routes.max().item()) >= route_count
    ):
        raise ValueError("route_ids exceed the routing plan")
    return routes


@dataclass(frozen=True, slots=True)
class ConditionalProjectionAccounting:
    """Logical modal activity and concrete grouped-matmul accounting.

    ``modal_matmul_calls`` counts the actual dense matrix multiplications
    issued by the reference implementation.  Each nonempty route with positive
    rank performs one encode and one decode multiplication.  The activity
    ratios are logical mode accounting, not a latency or FLOP claim for the
    native source layer.
    """

    total_tokens: int
    valid_tokens: int
    invalid_tokens: int
    route_token_counts: tuple[int, ...]
    active_modes_per_route: tuple[int, ...]
    active_mode_applications: int
    dense_mode_applications: int
    executed_route_groups: tuple[int, ...]
    modal_matmul_calls: int

    @property
    def average_active_modes(self) -> float:
        if self.valid_tokens == 0:
            return 0.0
        return self.active_mode_applications / self.valid_tokens

    @property
    def active_mode_ratio(self) -> float:
        if self.dense_mode_applications == 0:
            return 0.0
        return self.active_mode_applications / self.dense_mode_applications

    @property
    def ideal_mode_activation_reduction_fraction(self) -> float:
        if self.dense_mode_applications == 0:
            return 0.0
        return 1.0 - self.active_mode_ratio


@dataclass(frozen=True, slots=True)
class ConditionalProjectionResult:
    """Projected activations, the input-derived routes, and their accounting."""

    output: Tensor
    route_ids: Tensor
    accounting: ConditionalProjectionAccounting


class HardRoutedFisherProjection(nn.Module):
    """Project native activations through route-specific Fisher mode subsets.

    Valid rows are gathered route by route.  For route ``r`` with selected
    basis columns ``V_r``, the implementation computes

    ``mean + ((native - mean) @ V_r) @ V_r.T``.

    It does not form a dense full-width coordinate tensor and zero inactive
    coordinates afterward.  A rank-zero route performs no matrix
    multiplication and maps valid rows to the Fisher center.  Invalid rows are
    copied bit-for-bit from ``native_outputs``.
    """

    def __init__(
        self,
        basis: FisherModeBasis,
        routing_plan: ConditionalModalRoutingPlan,
    ) -> None:
        super().__init__()
        if not isinstance(basis, FisherModeBasis):
            raise TypeError("basis must be a FisherModeBasis")
        if not isinstance(routing_plan, ConditionalModalRoutingPlan):
            raise TypeError(
                "routing_plan must be a ConditionalModalRoutingPlan"
            )
        if routing_plan.mode_table.modes != basis.width:
            raise ValueError(
                "routing plan modal width must match the Fisher basis"
            )
        if (
            not basis.mean.is_floating_point()
            or not basis.vectors.is_floating_point()
            or not torch.isfinite(basis.mean).all()
            or not torch.isfinite(basis.vectors).all()
        ):
            raise ValueError("basis mean and vectors must be finite and floating")
        self.basis = basis
        self.routing_plan = routing_plan

    @property
    def width(self) -> int:
        return self.basis.width

    @property
    def routes(self) -> int:
        return self.routing_plan.mode_table.routes

    def route(self, block_inputs: Tensor) -> Tensor:
        """Choose routes solely from the current incoming hidden states."""

        _validate_block_inputs(block_inputs, self.routing_plan)
        return self.routing_plan.route(block_inputs)

    def _project_with_route_ids(
        self,
        native_outputs: Tensor,
        route_ids: Tensor,
        *,
        valid_mask: Tensor | None,
    ) -> ConditionalProjectionResult:
        if not isinstance(native_outputs, Tensor):
            raise TypeError("native_outputs must be a Tensor")
        if (
            native_outputs.ndim < 2
            or native_outputs.shape[-1] != self.width
        ):
            raise ValueError(
                "native_outputs must have shape [..., Fisher basis width]"
            )
        if not native_outputs.is_floating_point():
            raise ValueError("native_outputs must be floating point")
        if not torch.isfinite(native_outputs).all():
            raise ValueError("native_outputs must be finite")
        routes = _validate_route_ids(
            route_ids,
            leading_shape=native_outputs.shape[:-1],
            route_count=self.routes,
            device=native_outputs.device,
        )
        valid = _valid_token_mask(
            native_outputs.shape[:-1],
            valid_mask,
            device=native_outputs.device,
        )

        flat_native = native_outputs.reshape(-1, self.width)
        flat_routes = routes.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_output = flat_native.clone()
        route_counts: list[int] = []
        executed_groups: list[int] = []
        modal_matmul_calls = 0
        active_mode_applications = 0
        mean = self.basis.mean.to(
            device=native_outputs.device,
            dtype=native_outputs.dtype,
        )
        basis_vectors = self.basis.vectors.to(
            device=native_outputs.device,
            dtype=native_outputs.dtype,
        )

        for route in range(self.routes):
            selected_indices = (
                flat_valid & (flat_routes == route)
            ).nonzero(as_tuple=False).flatten()
            selected_count = int(selected_indices.numel())
            route_counts.append(selected_count)
            rank = self.routing_plan.mode_table.route_budgets[route]
            active_mode_applications += selected_count * rank
            if selected_count == 0:
                continue
            executed_groups.append(route)
            if rank == 0:
                projected = mean.expand(selected_count, -1)
            else:
                mode_mask = self.routing_plan.mode_table.mode_masks[route]
                mode_indices = mode_mask.nonzero(
                    as_tuple=False
                ).flatten().to(device=native_outputs.device)
                route_vectors = basis_vectors.index_select(1, mode_indices)
                selected_native = flat_native.index_select(
                    0,
                    selected_indices,
                )
                coordinates = torch.matmul(
                    selected_native - mean,
                    route_vectors,
                )
                projected = torch.matmul(
                    coordinates,
                    route_vectors.transpose(0, 1),
                ) + mean
                modal_matmul_calls += 2
            flat_output = flat_output.index_copy(
                0,
                selected_indices,
                projected,
            )

        valid_tokens = int(flat_valid.sum().item())
        accounting = ConditionalProjectionAccounting(
            total_tokens=flat_native.shape[0],
            valid_tokens=valid_tokens,
            invalid_tokens=flat_native.shape[0] - valid_tokens,
            route_token_counts=tuple(route_counts),
            active_modes_per_route=(
                self.routing_plan.mode_table.route_budgets
            ),
            active_mode_applications=active_mode_applications,
            dense_mode_applications=valid_tokens * self.width,
            executed_route_groups=tuple(executed_groups),
            modal_matmul_calls=modal_matmul_calls,
        )
        return ConditionalProjectionResult(
            output=flat_output.reshape_as(native_outputs),
            route_ids=routes,
            accounting=accounting,
        )

    def project_with_accounting(
        self,
        native_outputs: Tensor,
        block_inputs: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> ConditionalProjectionResult:
        """Route from ``block_inputs`` and conditionally project native output."""

        _validate_block_inputs(block_inputs, self.routing_plan)
        if native_outputs.shape[:-1] != block_inputs.shape[:-1]:
            raise ValueError(
                "native_outputs and block_inputs must share leading dimensions"
            )
        route_ids = self.routing_plan.route(block_inputs)
        return self._project_with_route_ids(
            native_outputs,
            route_ids,
            valid_mask=valid_mask,
        )

    def forward(
        self,
        native_outputs: Tensor,
        block_inputs: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.project_with_accounting(
            native_outputs,
            block_inputs,
            valid_mask=valid_mask,
        ).output


@dataclass(frozen=True, slots=True)
class ConditionalOracleExecutionStatus:
    """Cumulative audit proving that the oracle still runs its source block."""

    executor_calls: int
    native_source_block_calls: int
    routed_valid_tokens: int
    route_token_counts: tuple[int, ...]
    route_group_executions: tuple[int, ...]
    active_mode_applications: int
    dense_mode_applications: int
    source_block_savings_claimed: bool = False

    @property
    def logical_active_mode_ratio(self) -> float:
        if self.dense_mode_applications == 0:
            return 0.0
        return self.active_mode_applications / self.dense_mode_applications


class ConditionalModalProjectionOracleExecutor(LayerExecutor):
    """Native-output bottleneck oracle with fail-honest execution accounting.

    The source layer is executed exactly once on every call.  Conditional
    routing and projection happen *after* that native execution.  Therefore
    this wrapper can establish representational fidelity and active-mode
    statistics, but cannot establish source-block parameter, FLOP, or latency
    savings.
    """

    def __init__(
        self,
        source: LayerExecutor,
        projector: HardRoutedFisherProjection,
    ) -> None:
        super().__init__()
        if not isinstance(source, LayerExecutor):
            raise TypeError("source must be a LayerExecutor")
        if not isinstance(projector, HardRoutedFisherProjection):
            raise TypeError(
                "projector must be a HardRoutedFisherProjection"
            )
        self.source = source
        self.projector = projector
        self._executor_calls = 0
        self._native_source_block_calls = 0
        self._routed_valid_tokens = 0
        self._route_token_counts = [0] * projector.routes
        self._route_group_executions = [0] * projector.routes
        self._active_mode_applications = 0
        self._dense_mode_applications = 0
        self._last_execution: ConditionalProjectionAccounting | None = None

    @property
    def last_execution(self) -> ConditionalProjectionAccounting | None:
        return self._last_execution

    def execution_status(self) -> ConditionalOracleExecutionStatus:
        return ConditionalOracleExecutionStatus(
            executor_calls=self._executor_calls,
            native_source_block_calls=self._native_source_block_calls,
            routed_valid_tokens=self._routed_valid_tokens,
            route_token_counts=tuple(self._route_token_counts),
            route_group_executions=tuple(self._route_group_executions),
            active_mode_applications=self._active_mode_applications,
            dense_mode_applications=self._dense_mode_applications,
        )

    def reset_execution_counters(self) -> None:
        self._executor_calls = 0
        self._native_source_block_calls = 0
        self._routed_valid_tokens = 0
        self._route_token_counts = [0] * self.projector.routes
        self._route_group_executions = [0] * self.projector.routes
        self._active_mode_applications = 0
        self._dense_mode_applications = 0
        self._last_execution = None

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        _validate_block_inputs(hidden_states, self.projector.routing_plan)
        valid = _valid_token_mask(
            hidden_states.shape[:-1],
            attention_mask,
            device=hidden_states.device,
        )
        incoming = record(
            trace,
            f"{prefix}.conditional.input",
            hidden_states,
        )
        route_ids = self.projector.route(incoming)
        route_ids = record(
            trace,
            f"{prefix}.conditional.route_ids",
            route_ids,
        )
        route_ids = _validate_route_ids(
            route_ids,
            leading_shape=hidden_states.shape[:-1],
            route_count=self.projector.routes,
            device=hidden_states.device,
        )

        native_output = self.source(
            incoming,
            attention_mask=attention_mask,
            trace=None,
            prefix=f"{prefix}.conditional.native_source",
        )
        self._native_source_block_calls += 1
        native_output = record(
            trace,
            f"{prefix}.conditional.native_source_output",
            native_output,
        )
        result = self.projector._project_with_route_ids(
            native_output,
            route_ids,
            valid_mask=valid,
        )
        active_masks = self.projector.routing_plan.mode_table.masks_for(
            result.route_ids
        ).to(device=hidden_states.device)
        record(
            trace,
            f"{prefix}.conditional.active_mode_mask",
            active_masks,
        )

        accounting = result.accounting
        self._executor_calls += 1
        self._routed_valid_tokens += accounting.valid_tokens
        self._active_mode_applications += accounting.active_mode_applications
        self._dense_mode_applications += accounting.dense_mode_applications
        for route, count in enumerate(accounting.route_token_counts):
            self._route_token_counts[route] += count
        for route in accounting.executed_route_groups:
            self._route_group_executions[route] += 1
        self._last_execution = accounting
        return record(
            trace,
            f"{prefix}.output",
            result.output,
        )


@dataclass(frozen=True, slots=True)
class HardRoutedSpecialistAccounting:
    """Per-call proof that only populated route branches were invoked."""

    total_tokens: int
    valid_tokens: int
    invalid_tokens: int
    route_token_counts: tuple[int, ...]
    specialist_calls: tuple[int, ...]
    executed_routes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HardRoutedSpecialistResult:
    output: Tensor
    route_ids: Tensor
    accounting: HardRoutedSpecialistAccounting


@dataclass(frozen=True, slots=True)
class HardRoutedSpecialistStatus:
    executor_calls: int
    valid_tokens: int
    route_token_counts: tuple[int, ...]
    specialist_calls: tuple[int, ...]


class HardRoutedSpecialistBank(nn.Module):
    """Pointwise hard routing that evaluates no unselected specialist branch.

    Each specialist receives a gathered two-dimensional matrix
    ``[selected tokens, input features]`` and must return a matrix of the same
    shape.  This deliberately small contract is suitable for standalone
    pointwise modal graphs.  Sequence-mixing specialists need a richer causal
    packed-sequence executor and are outside this reference bank.

    Invalid rows bypass every specialist.  They are copied from
    ``invalid_output`` when supplied, otherwise from ``block_inputs``.
    """

    def __init__(
        self,
        routing_plan: ConditionalModalRoutingPlan,
        specialists: Sequence[nn.Module],
    ) -> None:
        super().__init__()
        if not isinstance(routing_plan, ConditionalModalRoutingPlan):
            raise TypeError(
                "routing_plan must be a ConditionalModalRoutingPlan"
            )
        if (
            not isinstance(specialists, Sequence)
            or isinstance(specialists, (str, bytes))
            or len(specialists) != routing_plan.mode_table.routes
            or any(not isinstance(specialist, nn.Module) for specialist in specialists)
        ):
            raise ValueError(
                "specialists must contain one nn.Module per route"
            )
        self.routing_plan = routing_plan
        self.specialists = nn.ModuleList(specialists)
        self._executor_calls = 0
        self._valid_tokens = 0
        self._route_token_counts = [0] * len(specialists)
        self._specialist_calls = [0] * len(specialists)
        self._last_execution: HardRoutedSpecialistAccounting | None = None

    @property
    def last_execution(self) -> HardRoutedSpecialistAccounting | None:
        return self._last_execution

    def execution_status(self) -> HardRoutedSpecialistStatus:
        return HardRoutedSpecialistStatus(
            executor_calls=self._executor_calls,
            valid_tokens=self._valid_tokens,
            route_token_counts=tuple(self._route_token_counts),
            specialist_calls=tuple(self._specialist_calls),
        )

    def execute_with_accounting(
        self,
        block_inputs: Tensor,
        *,
        valid_mask: Tensor | None = None,
        invalid_output: Tensor | None = None,
    ) -> HardRoutedSpecialistResult:
        _validate_block_inputs(block_inputs, self.routing_plan)
        valid = _valid_token_mask(
            block_inputs.shape[:-1],
            valid_mask,
            device=block_inputs.device,
        )
        if invalid_output is None:
            base_output = block_inputs
        else:
            if (
                not isinstance(invalid_output, Tensor)
                or invalid_output.shape != block_inputs.shape
                or invalid_output.dtype != block_inputs.dtype
                or invalid_output.device != block_inputs.device
            ):
                raise ValueError(
                    "invalid_output must exactly match block_inputs"
                )
            base_output = invalid_output
        route_ids = self.routing_plan.route(block_inputs)
        routes = _validate_route_ids(
            route_ids,
            leading_shape=block_inputs.shape[:-1],
            route_count=len(self.specialists),
            device=block_inputs.device,
        )
        width = block_inputs.shape[-1]
        flat_inputs = block_inputs.reshape(-1, width)
        flat_routes = routes.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_output = base_output.reshape(-1, width).clone()
        route_counts: list[int] = []
        calls: list[int] = []
        executed: list[int] = []

        for route, specialist in enumerate(self.specialists):
            selected_indices = (
                flat_valid & (flat_routes == route)
            ).nonzero(as_tuple=False).flatten()
            selected_count = int(selected_indices.numel())
            route_counts.append(selected_count)
            if selected_count == 0:
                calls.append(0)
                continue
            selected = flat_inputs.index_select(0, selected_indices)
            specialist_output = specialist(selected)
            if (
                not isinstance(specialist_output, Tensor)
                or specialist_output.shape != selected.shape
                or specialist_output.dtype != selected.dtype
                or specialist_output.device != selected.device
            ):
                raise ValueError(
                    "specialists must preserve selected token shape, dtype, "
                    "and device"
                )
            flat_output = flat_output.index_copy(
                0,
                selected_indices,
                specialist_output,
            )
            calls.append(1)
            executed.append(route)

        accounting = HardRoutedSpecialistAccounting(
            total_tokens=math.prod(block_inputs.shape[:-1]),
            valid_tokens=int(flat_valid.sum().item()),
            invalid_tokens=int((~flat_valid).sum().item()),
            route_token_counts=tuple(route_counts),
            specialist_calls=tuple(calls),
            executed_routes=tuple(executed),
        )
        self._executor_calls += 1
        self._valid_tokens += accounting.valid_tokens
        for route, count in enumerate(accounting.route_token_counts):
            self._route_token_counts[route] += count
        for route, count in enumerate(accounting.specialist_calls):
            self._specialist_calls[route] += count
        self._last_execution = accounting
        return HardRoutedSpecialistResult(
            output=flat_output.reshape_as(block_inputs),
            route_ids=routes,
            accounting=accounting,
        )

    def forward(
        self,
        block_inputs: Tensor,
        *,
        valid_mask: Tensor | None = None,
        invalid_output: Tensor | None = None,
    ) -> Tensor:
        return self.execute_with_accounting(
            block_inputs,
            valid_mask=valid_mask,
            invalid_output=invalid_output,
        ).output


__all__ = [
    "ConditionalModalProjectionOracleExecutor",
    "ConditionalOracleExecutionStatus",
    "ConditionalProjectionAccounting",
    "ConditionalProjectionResult",
    "HardRoutedFisherProjection",
    "HardRoutedSpecialistAccounting",
    "HardRoutedSpecialistBank",
    "HardRoutedSpecialistResult",
    "HardRoutedSpecialistStatus",
]
