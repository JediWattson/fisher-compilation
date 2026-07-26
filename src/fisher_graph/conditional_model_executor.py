"""Source-independent conditional execution for residual block deltas.

This module is the executable counterpart to a
:class:`~fisher_graph.conditional_routing.ConditionalModalRoutingPlan`.
Unlike the conditional projection oracle, it owns no native source block and
never asks one for an output.  A shared
:class:`~fisher_graph.dynamic_executor.StatefulCausalModalGraph` produces one
causal hidden trunk.  A route can be chosen either from the current block-input
row or from that causal hidden state, which lets a full-span executor condition
on the prefix without adding a second causal router.  Valid query rows are then
hard-gathered by route, and a route computes only its selected output-head and
linear-codec decoder columns.

For a positive-rank route with modal indices ``J`` the residual delta is

``delta = mean + (hidden @ W[:, J] + bias[J]) @ decoder[:, J].T``.

The returned block output is exactly ``input + delta`` at those rows.  A
rank-zero route is a true residual bypass: it runs neither output-head nor
decoder matrix multiplication and returns the incoming row bit-for-bit.
Invalid query rows are also exact bypasses, while valid key rows can still
contribute to later causal states.

The MAC counters in this module cover the route-specific output head and
decoder only.  They intentionally do not claim that the shared causal trunk,
router, gathers, additions, or a Python reference implementation are free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import torch
from torch import Tensor

from .activations import ActivationTrace, record
from .adapters.base import (
    MaskPolicy,
    SegmentRun,
    SequenceContext,
    SequenceInputOrigin,
    SequenceSpec,
    module_state_fingerprint,
)
from .compiler.capabilities import (
    CapabilityValues,
    LengthDomain,
    SequenceCapabilitySet,
)
from .compiler.manifest import CompiledSegment
from .conditional_routing import ConditionalModalRoutingPlan
from .dynamic_executor import StatefulCausalModalGraph
from .layers import LayerExecutor
from .linear_codec import LinearActivationCodec


RouteSource = Literal["block_input", "causal_hidden"]


_FINGERPRINT_DOMAIN = b"fisher_graph.conditional_model_executor.v2\0"
_ARTIFACT_KIND = "fisher_graph.conditional_causal_modal_block_executor"
_FORMAT_VERSION = 2
_ROUTE_SOURCES = {"block_input", "causal_hidden"}
_GRAPH_CONFIG_FIELDS = {
    "input_modes",
    "output_modes",
    "state_channels",
    "routing_width",
    "activation",
    "window_size",
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "input_activation_name",
    "output_activation_name",
    "route_source",
    "output_codec_method",
    "output_delta_mean",
    "output_decoder",
    "graph_config",
    "graph_dtype",
    "graph_state_dict",
    "routing_plan",
    "execution_fingerprint",
}
_FLOAT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_OUTPUT_CODEC_METHODS = {
    "native_fisher",
    "variance_weighted_fisher",
    "generalized_fisher",
}


def _update_fingerprint(digest: object, value: object) -> None:
    """Hash a small nested artifact payload without relying on pickle."""

    if not hasattr(digest, "update"):
        raise TypeError("digest must support update")
    update = digest.update
    if isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        update(b"tensor\0")
        update(str(tensor.dtype).encode("ascii"))
        update(json.dumps(tuple(tensor.shape)).encode("ascii"))
        update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        update(b"mapping\0")
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("fingerprinted mapping keys must be strings")
            update(key.encode("utf-8"))
            update(b"\0")
            _update_fingerprint(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        update(b"sequence\0")
        update(str(len(value)).encode("ascii"))
        update(b"\0")
        for item in value:
            _update_fingerprint(digest, item)
        return
    update(b"scalar\0")
    update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _normalize_binary_mask(
    mask: Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError("attention_mask must be a Tensor")
    if mask.shape != shape:
        raise ValueError(
            "attention_mask must match hidden-state batch and length"
        )
    if mask.device != device:
        raise ValueError("attention_mask must share the hidden-state device")
    if mask.dtype is torch.bool:
        return mask
    if not (
        mask.is_floating_point()
        or mask.dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
    ):
        raise ValueError("attention_mask must be boolean or binary")
    if not torch.isfinite(mask).all() or not (
        (mask == 0) | (mask == 1)
    ).all():
        raise ValueError("attention_mask must contain only zero and one")
    return mask.to(dtype=torch.bool)


def _validate_route_ids(
    route_ids: Tensor,
    *,
    shape: tuple[int, int],
    route_count: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(route_ids, Tensor):
        raise TypeError("route_ids must be a Tensor")
    if route_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("route_ids must use an integer dtype")
    if route_ids.shape != shape:
        raise ValueError("route_ids must match the query batch and length")
    if route_ids.device != device:
        raise ValueError("route_ids must share the hidden-state device")
    routes = route_ids.to(dtype=torch.int64)
    if routes.numel() and (
        int(routes.min().item()) < 0
        or int(routes.max().item()) >= route_count
    ):
        raise ValueError("route_ids exceed the conditional routing plan")
    return routes


@dataclass(frozen=True, slots=True)
class ConditionalModelExecutionAccounting:
    """Concrete per-call accounting for the hard-routed modal tail."""

    total_tokens: int
    valid_tokens: int
    invalid_tokens: int
    route_token_counts: tuple[int, ...]
    active_head_columns_per_route: tuple[int, ...]
    head_column_indices_per_route: tuple[tuple[int, ...], ...]
    populated_routes: tuple[int, ...]
    executed_compute_routes: tuple[int, ...]
    route_group_calls: tuple[int, ...]
    output_head_matmul_calls: int
    decoder_matmul_calls: int
    output_head_column_applications: int
    output_head_macs: int
    decoder_macs: int
    dense_output_head_macs: int
    dense_decoder_macs: int

    @property
    def routed_tail_macs(self) -> int:
        return self.output_head_macs + self.decoder_macs

    @property
    def dense_tail_macs(self) -> int:
        return self.dense_output_head_macs + self.dense_decoder_macs

    @property
    def ideal_tail_mac_reduction_fraction(self) -> float:
        if self.dense_tail_macs == 0:
            return 0.0
        return 1.0 - self.routed_tail_macs / self.dense_tail_macs


@dataclass(frozen=True, slots=True)
class ConditionalModelExecutionResult:
    """Output, input-derived routes, and the tail work actually issued."""

    output: Tensor
    route_ids: Tensor
    accounting: ConditionalModelExecutionAccounting


@dataclass(frozen=True, slots=True)
class ConditionalModelExecutionStatus:
    """Cumulative work issued by calls that entered this executor.

    This object deliberately makes no claim about calls handled elsewhere by a
    mixed-runtime dispatcher.  In particular, it cannot observe a dispatcher
    choosing a native source fallback instead of invoking this executor.  The
    executor itself owns no source module or fallback callable; that structural
    fact is exposed by :attr:`executor_local_source_free`.
    """

    executor_calls: int
    valid_tokens: int
    route_token_counts: tuple[int, ...]
    route_group_executions: tuple[int, ...]
    head_column_indices_per_route: tuple[tuple[int, ...], ...]
    output_head_matmul_calls: int
    decoder_matmul_calls: int
    output_head_column_applications: int
    output_head_macs: int
    decoder_macs: int
    dense_output_head_macs: int
    dense_decoder_macs: int

    @property
    def executor_local_source_free(self) -> bool:
        """Return the executor's structural ownership fact, not a runtime audit."""

        return True

    @property
    def routed_tail_macs(self) -> int:
        return self.output_head_macs + self.decoder_macs

    @property
    def dense_tail_macs(self) -> int:
        return self.dense_output_head_macs + self.dense_decoder_macs

    @property
    def ideal_tail_mac_reduction_fraction(self) -> float:
        if self.dense_tail_macs == 0:
            return 0.0
        return 1.0 - self.routed_tail_macs / self.dense_tail_macs


class ConditionalCausalModalBlockExecutor(LayerExecutor):
    """Shared causal trunk with a hard-routed, source-free modal delta tail.

    ``output_codec`` must describe the residual *delta* at the compiled block
    boundary.  Its mean and decoder are snapshotted into runtime buffers; its
    encoder is a fitting-time object and is intentionally not retained by this
    executor.
    """

    def __init__(
        self,
        *,
        graph: StatefulCausalModalGraph,
        routing_plan: ConditionalModalRoutingPlan,
        output_codec: LinearActivationCodec,
        input_activation_name: str,
        route_source: RouteSource = "block_input",
    ) -> None:
        super().__init__()
        if not isinstance(output_codec, LinearActivationCodec):
            raise TypeError("output_codec must be a LinearActivationCodec")
        self._initialize_runtime(
            graph=graph,
            routing_plan=routing_plan,
            input_activation_name=input_activation_name,
            output_activation_name=output_codec.activation_name,
            route_source=route_source,
            output_codec_method=output_codec.method,
            output_delta_mean=output_codec.mean,
            output_decoder=output_codec.decoder,
        )

    def _initialize_runtime(
        self,
        *,
        graph: StatefulCausalModalGraph,
        routing_plan: ConditionalModalRoutingPlan,
        input_activation_name: str,
        output_activation_name: str,
        route_source: RouteSource,
        output_codec_method: str,
        output_delta_mean: Tensor,
        output_decoder: Tensor,
    ) -> None:
        """Install validated runtime-only state on an initialized module."""

        if not isinstance(graph, StatefulCausalModalGraph):
            raise TypeError("graph must be a StatefulCausalModalGraph")
        if not isinstance(routing_plan, ConditionalModalRoutingPlan):
            raise TypeError(
                "routing_plan must be a ConditionalModalRoutingPlan"
            )
        for label, value in (
            ("input_activation_name", input_activation_name),
            ("output_activation_name", output_activation_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if (
            not isinstance(route_source, str)
            or route_source not in _ROUTE_SOURCES
        ):
            raise ValueError(
                "route_source must be 'block_input' or 'causal_hidden'"
            )
        if output_codec_method not in _OUTPUT_CODEC_METHODS:
            raise ValueError("output_codec_method is unsupported")
        if (
            not isinstance(output_delta_mean, Tensor)
            or not output_delta_mean.is_floating_point()
            or output_delta_mean.ndim != 1
            or output_delta_mean.numel() == 0
            or not torch.isfinite(output_delta_mean).all()
        ):
            raise ValueError(
                "output_delta_mean must be a finite floating vector"
            )
        width = int(output_delta_mean.numel())
        if (
            not isinstance(output_decoder, Tensor)
            or not output_decoder.is_floating_point()
            or output_decoder.shape != (width, width)
            or not torch.isfinite(output_decoder).all()
        ):
            raise ValueError(
                "output_decoder must be a finite floating width matrix"
            )
        expected_router_width = (
            width
            if route_source == "block_input"
            else graph.routing_width
        )
        if routing_plan.router.input_features != expected_router_width:
            raise ValueError(
                "routing-plan input width must match the selected route "
                "source"
            )
        if routing_plan.mode_table.modes != width:
            raise ValueError(
                "routing-plan modal width must match the output codec"
            )
        if graph.input_modes != width:
            raise ValueError(
                "shared graph must consume the full incoming residual width"
            )
        if graph.output_modes != width:
            raise ValueError(
                "shared graph output head must expose every codec mode"
            )

        reference = graph.state_input_weight
        self.graph = graph
        # Both artifact classes clone their tensors.  Reconstructing the plan
        # also prevents caller-owned dataclass instances from being swapped
        # into the live executor through ordinary attribute aliasing.
        self.routing_plan = ConditionalModalRoutingPlan.from_state_dict(
            routing_plan.state_dict()
        )
        self.input_activation_name = input_activation_name
        self.output_activation_name = output_activation_name
        self.route_source: RouteSource = route_source
        self.output_codec_method = output_codec_method
        self.register_buffer(
            "output_delta_mean",
            output_delta_mean.detach()
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
            .clone(),
        )
        self.register_buffer(
            "output_decoder",
            output_decoder.detach()
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
            .clone(),
        )

        self._executor_calls = 0
        self._valid_tokens = 0
        self._route_token_counts = [0] * self.routes
        self._route_group_executions = [0] * self.routes
        self._output_head_matmul_calls = 0
        self._decoder_matmul_calls = 0
        self._output_head_column_applications = 0
        self._output_head_macs = 0
        self._decoder_macs = 0
        self._dense_output_head_macs = 0
        self._dense_decoder_macs = 0
        self._last_execution: (
            ConditionalModelExecutionAccounting | None
        ) = None

    @property
    def width(self) -> int:
        return int(self.output_delta_mean.numel())

    @property
    def routes(self) -> int:
        return self.routing_plan.mode_table.routes

    @property
    def sequence_spec(self) -> SequenceSpec:
        return SequenceSpec(
            length_policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            mask=MaskPolicy(
                causal=True,
                padding_side="sparse",
                representation="boolean_valid",
            ),
            position_kind="logical_relative_exponential_state",
            supports_prefill=True,
            supports_decode=False,
            cache_kind="none",
        )

    @property
    def capabilities(self) -> SequenceCapabilitySet:
        """Return the exact runtime requests this module can execute."""

        reference = self.graph.state_input_weight
        dtype = str(reference.dtype).removeprefix("torch.")
        return SequenceCapabilitySet(
            length=LengthDomain(1, None),
            executions=CapabilityValues.known("prefill"),
            qk_relations=CapabilityValues.known("equal"),
            position_relations=CapabilityValues.known(
                "equal",
                "query_suffix",
                "arbitrary",
            ),
            mask_origins=CapabilityValues.known("omitted", "provided"),
            mask_patterns=CapabilityValues.known(
                "all_valid",
                "right_padded",
                "left_padded",
                "mixed_padded",
                "sparse",
                "custom",
            ),
            mask_representations=CapabilityValues.known(
                "boolean_valid"
            ),
            visibility_families=CapabilityValues.known(
                (
                    "global_causal"
                    if self.graph.window_size is None
                    else "sliding_causal"
                )
            ),
            position_origins=CapabilityValues.known(
                "omitted",
                "provided",
            ),
            position_domains=CapabilityValues.known(
                "zero_contiguous",
                "offset_contiguous",
                "arbitrary",
            ),
            cache_kinds=CapabilityValues.known("none"),
            dtypes=CapabilityValues.known(dtype),
            devices=CapabilityValues.known(reference.device.type),
            layouts=CapabilityValues.known("contiguous", "strided"),
        )

    @property
    def last_execution(
        self,
    ) -> ConditionalModelExecutionAccounting | None:
        return self._last_execution

    def execution_status(self) -> ConditionalModelExecutionStatus:
        return ConditionalModelExecutionStatus(
            executor_calls=self._executor_calls,
            valid_tokens=self._valid_tokens,
            route_token_counts=tuple(self._route_token_counts),
            route_group_executions=tuple(
                self._route_group_executions
            ),
            head_column_indices_per_route=(
                self._head_column_indices_per_route()
            ),
            output_head_matmul_calls=self._output_head_matmul_calls,
            decoder_matmul_calls=self._decoder_matmul_calls,
            output_head_column_applications=(
                self._output_head_column_applications
            ),
            output_head_macs=self._output_head_macs,
            decoder_macs=self._decoder_macs,
            dense_output_head_macs=self._dense_output_head_macs,
            dense_decoder_macs=self._dense_decoder_macs,
        )

    def reset_execution_counters(self) -> None:
        self._executor_calls = 0
        self._valid_tokens = 0
        self._route_token_counts = [0] * self.routes
        self._route_group_executions = [0] * self.routes
        self._output_head_matmul_calls = 0
        self._decoder_matmul_calls = 0
        self._output_head_column_applications = 0
        self._output_head_macs = 0
        self._decoder_macs = 0
        self._dense_output_head_macs = 0
        self._dense_decoder_macs = 0
        self._last_execution = None

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.width,
        )
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.shape != expected
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                "hidden_states must be floating with shape "
                f"{expected}"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        if sequence.key_length != sequence.query_length:
            raise ValueError(
                "conditional modal prefill requires equal key/query lengths"
            )
        if sequence.phase != "prefill":
            raise ValueError(
                "conditional modal executor does not support cached decode"
            )
        if sequence.cache_state is not None:
            raise ValueError(
                "conditional modal executor does not accept cache state"
            )
        if sequence.cache_positions is not None:
            raise ValueError(
                "conditional modal executor does not accept cache positions"
            )

    def _default_sequence_context(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
    ) -> SequenceContext:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            valid = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            valid = _normalize_binary_mask(
                attention_mask,
                shape=(batch_size, sequence_length),
                device=hidden_states.device,
            )
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch_size, -1)
        return SequenceContext(
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=positions,
            key_logical_positions=positions,
            cache_positions=None,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=attention_mask is not None,
                position_ids_supplied=False,
                cache_positions_supplied=False,
            ),
            cache_state=None,
            adapter_payload=None,
        )

    def _update_counters(
        self,
        accounting: ConditionalModelExecutionAccounting,
    ) -> None:
        self._executor_calls += 1
        self._valid_tokens += accounting.valid_tokens
        for route, count in enumerate(accounting.route_token_counts):
            self._route_token_counts[route] += count
        for route, count in enumerate(accounting.route_group_calls):
            self._route_group_executions[route] += count
        self._output_head_matmul_calls += (
            accounting.output_head_matmul_calls
        )
        self._decoder_matmul_calls += accounting.decoder_matmul_calls
        self._output_head_column_applications += (
            accounting.output_head_column_applications
        )
        self._output_head_macs += accounting.output_head_macs
        self._decoder_macs += accounting.decoder_macs
        self._dense_output_head_macs += (
            accounting.dense_output_head_macs
        )
        self._dense_decoder_macs += accounting.dense_decoder_macs
        self._last_execution = accounting

    def _head_column_indices_per_route(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                int(index)
                for index in mask.nonzero(
                    as_tuple=False
                ).flatten().tolist()
            )
            for mask in self.routing_plan.mode_table.mode_masks
        )

    def _select_routes(
        self,
        features: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        route_ids = self.routing_plan.route(features)
        route_ids = record(
            trace,
            f"{prefix}.conditional.route_ids",
            route_ids,
        )
        return _validate_route_ids(
            route_ids,
            shape=(sequence.batch_size, sequence.query_length),
            route_count=self.routes,
            device=features.device,
        )

    def _execute_instrumented(
        self,
        hidden_states: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> ConditionalModelExecutionResult:
        self._validate_inputs(hidden_states, sequence)
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a nonempty string")

        routes: Tensor | None = None
        if self.route_source == "block_input":
            routes = self._select_routes(
                hidden_states,
                sequence=sequence,
                trace=trace,
                prefix=prefix,
            )

        compute_inputs = hidden_states.to(
            dtype=self.graph.state_input_weight.dtype
        )
        causal_state = self.graph.compute_causal_state(
            compute_inputs,
            query_valid_mask=sequence.query_valid_mask,
            key_valid_mask=sequence.key_valid_mask,
            logical_positions=sequence.logical_positions,
            key_logical_positions=sequence.key_logical_positions,
        )
        causal_state = record(
            trace,
            f"{prefix}.conditional.modal.causal_state",
            causal_state,
        )
        hidden = self.graph.compute_hidden(
            causal_state,
            query_valid_mask=sequence.query_valid_mask,
        )
        hidden = record(
            trace,
            f"{prefix}.conditional.modal.hidden",
            hidden,
        )
        if self.route_source == "causal_hidden":
            routes = self._select_routes(
                hidden,
                sequence=sequence,
                trace=trace,
                prefix=prefix,
            )
        assert routes is not None

        flat_inputs = hidden_states.reshape(-1, self.width)
        flat_compute_inputs = compute_inputs.reshape(-1, self.width)
        flat_hidden = hidden.reshape(-1, self.graph.routing_width)
        flat_routes = routes.reshape(-1)
        flat_valid = sequence.query_valid_mask.reshape(-1)
        flat_output = flat_inputs.clone()
        flat_delta = compute_inputs.new_zeros(
            flat_inputs.shape[0],
            self.width,
        )
        flat_computed = torch.zeros(
            flat_inputs.shape[0],
            dtype=torch.bool,
            device=hidden_states.device,
        )

        route_counts: list[int] = []
        group_calls: list[int] = []
        populated: list[int] = []
        executed: list[int] = []
        output_head_calls = 0
        decoder_calls = 0
        head_column_applications = 0
        output_head_macs = 0
        decoder_macs = 0

        for route in range(self.routes):
            selected_indices = (
                flat_valid & (flat_routes == route)
            ).nonzero(as_tuple=False).flatten()
            selected_count = int(selected_indices.numel())
            route_counts.append(selected_count)
            if selected_count == 0:
                group_calls.append(0)
                continue
            populated.append(route)
            rank = self.routing_plan.mode_table.route_budgets[route]
            if rank == 0:
                group_calls.append(0)
                continue

            mode_indices = (
                self.routing_plan.mode_table.mode_masks[route]
                .nonzero(as_tuple=False)
                .flatten()
                .to(device=hidden_states.device)
            )
            selected_hidden = flat_hidden.index_select(
                0,
                selected_indices,
            )
            head_weight = self.graph.output_weight.index_select(
                1,
                mode_indices,
            )
            head_bias = self.graph.output_bias.index_select(
                0,
                mode_indices,
            )
            selected_coordinates = (
                selected_hidden @ head_weight + head_bias
            )
            decoder = self.output_decoder.index_select(
                1,
                mode_indices,
            )
            selected_delta = (
                selected_coordinates @ decoder.transpose(0, 1)
                + self.output_delta_mean
            )
            selected_input = flat_compute_inputs.index_select(
                0,
                selected_indices,
            )
            selected_output = selected_input + selected_delta

            flat_delta = flat_delta.index_copy(
                0,
                selected_indices,
                selected_delta,
            )
            flat_output = flat_output.index_copy(
                0,
                selected_indices,
                selected_output.to(dtype=hidden_states.dtype),
            )
            flat_computed[selected_indices] = True

            group_calls.append(1)
            executed.append(route)
            output_head_calls += 1
            decoder_calls += 1
            head_column_applications += selected_count * rank
            output_head_macs += (
                selected_count * self.graph.routing_width * rank
            )
            decoder_macs += selected_count * rank * self.width

        delta = flat_delta.reshape_as(compute_inputs)
        traced_delta = record(
            trace,
            f"{prefix}.conditional.modal.delta",
            delta,
        )
        if traced_delta is not delta:
            recomputed = (
                compute_inputs + traced_delta
            ).to(dtype=hidden_states.dtype)
            output = torch.where(
                flat_computed.reshape(
                    sequence.batch_size,
                    sequence.query_length,
                ).unsqueeze(-1),
                recomputed,
                hidden_states,
            )
        else:
            output = flat_output.reshape_as(hidden_states)

        valid_tokens = int(flat_valid.sum().item())
        full_modes = self.graph.output_modes
        accounting = ConditionalModelExecutionAccounting(
            total_tokens=math.prod(hidden_states.shape[:-1]),
            valid_tokens=valid_tokens,
            invalid_tokens=flat_valid.numel() - valid_tokens,
            route_token_counts=tuple(route_counts),
            active_head_columns_per_route=(
                self.routing_plan.mode_table.route_budgets
            ),
            head_column_indices_per_route=(
                self._head_column_indices_per_route()
            ),
            populated_routes=tuple(populated),
            executed_compute_routes=tuple(executed),
            route_group_calls=tuple(group_calls),
            output_head_matmul_calls=output_head_calls,
            decoder_matmul_calls=decoder_calls,
            output_head_column_applications=head_column_applications,
            output_head_macs=output_head_macs,
            decoder_macs=decoder_macs,
            dense_output_head_macs=(
                valid_tokens * self.graph.routing_width * full_modes
            ),
            dense_decoder_macs=valid_tokens * full_modes * self.width,
        )
        self._update_counters(accounting)
        return ConditionalModelExecutionResult(
            output=output,
            route_ids=routes,
            accounting=accounting,
        )

    def execute_context_with_accounting(
        self,
        hidden_states: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> ConditionalModelExecutionResult:
        """Execute an explicit prefill context and return the work audit."""

        instrumented_input = record(
            trace,
            f"{prefix}.input",
            hidden_states,
        )
        result = self._execute_instrumented(
            instrumented_input,
            sequence=sequence,
            trace=trace,
            prefix=prefix,
        )
        output = record(trace, f"{prefix}.output", result.output)
        return ConditionalModelExecutionResult(
            output=output,
            route_ids=result.route_ids,
            accounting=result.accounting,
        )

    def forward_context(
        self,
        hidden_states: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        return self.execute_context_with_accounting(
            hidden_states,
            sequence=sequence,
            trace=trace,
            prefix=prefix,
        ).output

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
        sequence_context: SequenceContext | None = None,
    ) -> Tensor:
        """Implement the toy ``LayerExecutor`` prefill contract."""

        if sequence_context is None:
            sequence_context = self._default_sequence_context(
                hidden_states,
                attention_mask=attention_mask,
            )
        elif attention_mask is not None:
            normalized = _normalize_binary_mask(
                attention_mask,
                shape=(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                ),
                device=hidden_states.device,
            )
            if not torch.equal(
                normalized,
                sequence_context.key_valid_mask,
            ):
                raise ValueError(
                    "attention_mask conflicts with the sequence key mask"
                )
        return self.forward_context(
            hidden_states,
            sequence=sequence_context,
            trace=trace,
            prefix=prefix,
        )

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        """Implement the backend-neutral compiled-segment executor protocol."""

        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        if (
            segment.input_activation != self.input_activation_name
            or segment.output_activation != self.output_activation_name
        ):
            raise ValueError(
                "compiled segment boundaries do not match this "
                "conditional executor"
            )
        instrumented_input = record(
            trace,
            segment.input_activation,
            hidden_states,
        )
        result = self._execute_instrumented(
            instrumented_input,
            sequence=sequence,
            trace=trace,
            prefix=segment.id,
        )
        output = record(
            trace,
            segment.output_activation,
            result.output,
        )
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output={
                "route_ids": result.route_ids,
                "accounting": result.accounting,
            },
        )

    def execution_fingerprint(self) -> str:
        """Hash every tensor and non-tensor option affecting execution."""

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(module_state_fingerprint(self).encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "graph": {
                        "input_modes": self.graph.input_modes,
                        "output_modes": self.graph.output_modes,
                        "state_channels": self.graph.state_channels,
                        "routing_width": self.graph.routing_width,
                        "activation": self.graph.activation,
                        "window_size": self.graph.window_size,
                    },
                    "input_activation_name": self.input_activation_name,
                    "output_activation_name": self.output_activation_name,
                    "route_source": self.route_source,
                    "output_codec_method": self.output_codec_method,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        _update_fingerprint(digest, self.routing_plan.state_dict())
        return digest.hexdigest()

    def _graph_config(self) -> dict[str, object]:
        return {
            "input_modes": self.graph.input_modes,
            "output_modes": self.graph.output_modes,
            "state_channels": self.graph.state_channels,
            "routing_width": self.graph.routing_width,
            "activation": self.graph.activation,
            "window_size": self.graph.window_size,
        }

    def _validated_cpu_graph_state(self) -> dict[str, Tensor]:
        reference = self.graph.state_input_weight
        state: dict[str, Tensor] = {}
        for name, value in sorted(self.graph.state_dict().items()):
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"graph state {name!r} must be finite and match the "
                    "graph dtype and device"
                )
            state[name] = value.detach().to(device="cpu").clone()
        return state

    def artifact_state_dict(self) -> dict[str, object]:
        """Return a strict, source-free, weights-only-safe runtime artifact.

        The fitting-time codec encoder is intentionally absent because runtime
        execution needs only the residual-delta mean and decoder.  The routing
        plan, both compiled boundaries, graph configuration and graph weights
        are all included, so this payload is complete without a native source
        model or caller-supplied plan.
        """

        reference = self.graph.state_input_weight
        dtype_name = str(reference.dtype).removeprefix("torch.")
        if dtype_name not in _FLOAT_DTYPES:
            raise ValueError("unsupported graph dtype for artifact export")
        for label, value in (
            ("output_delta_mean", self.output_delta_mean),
            ("output_decoder", self.output_decoder),
        ):
            if (
                value.dtype != reference.dtype
                or value.device != reference.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{label} must be finite and match the graph runtime"
                )
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "input_activation_name": self.input_activation_name,
            "output_activation_name": self.output_activation_name,
            "route_source": self.route_source,
            "output_codec_method": self.output_codec_method,
            "output_delta_mean": self.output_delta_mean.detach()
            .to(device="cpu")
            .clone(),
            "output_decoder": self.output_decoder.detach()
            .to(device="cpu")
            .clone(),
            "graph_config": self._graph_config(),
            "graph_dtype": dtype_name,
            "graph_state_dict": self._validated_cpu_graph_state(),
            "routing_plan": self.routing_plan.state_dict(),
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> ConditionalCausalModalBlockExecutor:
        """Strictly restore a complete source-independent runtime artifact."""

        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError(
                "conditional model executor artifact fields are invalid"
            )
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported conditional model executor artifact"
            )
        fingerprint = state["execution_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("artifact execution fingerprint is invalid")

        raw_config = state["graph_config"]
        if (
            not isinstance(raw_config, Mapping)
            or set(raw_config) != _GRAPH_CONFIG_FIELDS
        ):
            raise ValueError("conditional graph config fields are invalid")
        graph = StatefulCausalModalGraph(
            input_modes=raw_config["input_modes"],  # type: ignore[arg-type]
            output_modes=raw_config["output_modes"],  # type: ignore[arg-type]
            state_channels=raw_config["state_channels"],  # type: ignore[arg-type]
            routing_width=raw_config["routing_width"],  # type: ignore[arg-type]
            activation=raw_config["activation"],  # type: ignore[arg-type]
            window_size=raw_config["window_size"],  # type: ignore[arg-type]
        )
        dtype_name = state["graph_dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _FLOAT_DTYPES:
            raise ValueError("conditional graph dtype is invalid")
        try:
            device = torch.device(map_location)
        except (TypeError, RuntimeError) as error:
            raise ValueError("map_location is not a valid torch device") from error
        dtype = _FLOAT_DTYPES[dtype_name]
        graph = graph.to(device=device, dtype=dtype)

        raw_graph_state = state["graph_state_dict"]
        expected_graph_state = graph.state_dict()
        if (
            not isinstance(raw_graph_state, Mapping)
            or set(raw_graph_state) != set(expected_graph_state)
        ):
            raise ValueError("conditional graph state fields are invalid")
        restored_graph_state: dict[str, Tensor] = {}
        for name, expected in expected_graph_state.items():
            value = raw_graph_state[name]
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or not value.is_floating_point()
                or value.dtype != dtype
                or value.shape != expected.shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"conditional graph state {name!r} is invalid"
                )
            restored_graph_state[name] = value.detach().to(
                device=device,
                dtype=dtype,
            )
        graph.load_state_dict(restored_graph_state, strict=True)

        raw_plan = state["routing_plan"]
        if not isinstance(raw_plan, Mapping):
            raise ValueError("conditional routing plan state is invalid")
        routing_plan = ConditionalModalRoutingPlan.from_state_dict(raw_plan)

        output_delta_mean = state["output_delta_mean"]
        output_decoder = state["output_decoder"]
        width = graph.output_modes
        if (
            not isinstance(output_delta_mean, Tensor)
            or output_delta_mean.device.type != "cpu"
            or not output_delta_mean.is_floating_point()
            or output_delta_mean.dtype != dtype
            or output_delta_mean.shape != (width,)
            or not torch.isfinite(output_delta_mean).all()
        ):
            raise ValueError("artifact output_delta_mean is invalid")
        if (
            not isinstance(output_decoder, Tensor)
            or output_decoder.device.type != "cpu"
            or not output_decoder.is_floating_point()
            or output_decoder.dtype != dtype
            or output_decoder.shape != (width, width)
            or not torch.isfinite(output_decoder).all()
        ):
            raise ValueError("artifact output_decoder is invalid")
        input_activation_name = state["input_activation_name"]
        output_activation_name = state["output_activation_name"]
        route_source = state["route_source"]
        output_codec_method = state["output_codec_method"]

        result = cls.__new__(cls)
        LayerExecutor.__init__(result)
        result._initialize_runtime(
            graph=graph,
            routing_plan=routing_plan,
            input_activation_name=input_activation_name,  # type: ignore[arg-type]
            output_activation_name=output_activation_name,  # type: ignore[arg-type]
            route_source=route_source,  # type: ignore[arg-type]
            output_codec_method=output_codec_method,  # type: ignore[arg-type]
            output_delta_mean=output_delta_mean,
            output_decoder=output_decoder,
        )
        if result.execution_fingerprint() != fingerprint:
            raise ValueError(
                "conditional model executor execution fingerprint mismatch"
            )
        result.eval()
        return result


__all__ = [
    "ConditionalCausalModalBlockExecutor",
    "ConditionalModelExecutionAccounting",
    "ConditionalModelExecutionResult",
    "ConditionalModelExecutionStatus",
    "RouteSource",
]
