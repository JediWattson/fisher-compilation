"""Router-free causal execution inside a fixed residual-delta span.

This module is the structurally pruned static counterpart to the conditional
modal executor.  A :class:`~fisher_graph.dynamic_executor.StatefulCausalModalGraph`
reads the full incoming residual width but emits only ``rank`` modal
coordinates.  A fixed ``[width, rank]`` decoder maps those coordinates back to
the residual stream:

``output = input + modal_delta @ decoder.T``.

Only query-valid rows receive the predicted delta.  Every other row is an
exact passthrough of the caller's input, while key-valid rows can still
contribute to later causal states.  The executor owns neither a source block,
a router, nor unused output-head/decoder columns.

The PyTorch reference graph masks dense sequence-shaped intermediates after
issuing their matrix operations and can materialize a dense query/key state
when query and key masks differ.  The class therefore exposes logical
structure and a source-free runtime boundary, but does not claim that this
reference kernel is physically query sparse or faster.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

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
from .dynamic_executor import StatefulCausalModalGraph
from .layers import LayerExecutor


_ARTIFACT_KIND = "fisher_graph.static_span_block_executor"
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = b"fisher_graph.static_span_executor.v1\0"
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
    "decoder",
    "graph_config",
    "graph_dtype",
    "graph_state_dict",
    "execution_fingerprint",
}
_FLOAT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


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


@dataclass(frozen=True, slots=True)
class StaticSpanExecution:
    """Inspectable tensors produced by one fixed-span execution."""

    output: Tensor
    causal_state: Tensor
    hidden: Tensor
    modal_delta: Tensor
    decoded_delta: Tensor


class StaticSpanBlockExecutor(LayerExecutor):
    """A source-independent residual executor with a structural static rank."""

    def __init__(
        self,
        *,
        graph: StatefulCausalModalGraph,
        decoder: Tensor,
        input_activation_name: str,
        output_activation_name: str,
    ) -> None:
        super().__init__()
        if not isinstance(graph, StatefulCausalModalGraph):
            raise TypeError("graph must be a StatefulCausalModalGraph")
        for label, value in (
            ("input_activation_name", input_activation_name),
            ("output_activation_name", output_activation_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if (
            not isinstance(decoder, Tensor)
            or not decoder.is_floating_point()
            or decoder.ndim != 2
            or not torch.isfinite(decoder).all()
        ):
            raise ValueError("decoder must be a finite floating matrix")
        width, retained_rank = decoder.shape
        if width == 0 or retained_rank == 0:
            raise ValueError("decoder dimensions must be positive")
        if graph.input_modes != width:
            raise ValueError(
                "graph input modes must equal the residual width"
            )
        if graph.output_modes != retained_rank:
            raise ValueError(
                "graph output modes must equal the decoder rank"
            )
        reference = graph.state_input_weight
        if decoder.dtype != reference.dtype or decoder.device != reference.device:
            raise ValueError(
                "decoder must match the graph dtype and device"
            )

        self.graph = graph
        self.input_activation_name = input_activation_name
        self.output_activation_name = output_activation_name
        self.register_buffer("decoder", decoder.detach().clone())

    @property
    def width(self) -> int:
        return int(self.decoder.shape[0])

    @property
    def retained_rank(self) -> int:
        return int(self.decoder.shape[1])

    @property
    def learned_parameter_count(self) -> int:
        return self.graph.learned_parameters

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return self.decoder.numel()

    @property
    def runtime_stored_coefficient_count(self) -> int:
        return (
            self.learned_parameter_count
            + self.fixed_runtime_coefficient_count
        )

    @property
    def executor_local_source_free(self) -> bool:
        """This module owns no native source module or fallback callable."""

        return True

    @property
    def supports_query_sparse_prefill(self) -> bool:
        """Query demand may be a sparse subset of the causal key prefix."""

        return True

    @property
    def reference_kernel_physically_query_sparse(self) -> bool:
        """The reference implementation masks dense intermediate kernels."""

        return False

    @property
    def reference_kernel_speed_claim(self) -> bool:
        return False

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
                "static-span prefill requires equal key/query tensor lengths"
            )
        if sequence.phase != "prefill":
            raise ValueError(
                "static-span executor does not support cached decode"
            )
        if sequence.cache_state is not None:
            raise ValueError(
                "static-span executor does not accept cache state"
            )
        if sequence.cache_positions is not None:
            raise ValueError(
                "static-span executor does not accept cache positions"
            )

    def _execute(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> StaticSpanExecution:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a nonempty string")
        compute = hidden_states.to(dtype=self.graph.state_input_weight.dtype)
        causal_state = self.graph.compute_causal_state(
            compute,
            query_valid_mask=sequence.query_valid_mask,
            key_valid_mask=sequence.key_valid_mask,
            logical_positions=sequence.logical_positions,
            key_logical_positions=sequence.key_logical_positions,
        )
        causal_state = record(
            trace,
            f"{prefix}.static_span.causal_state",
            causal_state,
        )
        hidden = self.graph.compute_hidden(
            causal_state,
            query_valid_mask=sequence.query_valid_mask,
        )
        hidden = record(
            trace,
            f"{prefix}.static_span.hidden",
            hidden,
        )
        modal_delta = self.graph.compute_output(
            hidden,
            query_valid_mask=sequence.query_valid_mask,
        )
        modal_delta = record(
            trace,
            f"{prefix}.static_span.modal_delta",
            modal_delta,
        )
        decoded_delta = modal_delta @ self.decoder.T
        decoded_delta = record(
            trace,
            f"{prefix}.static_span.decoded_delta",
            decoded_delta,
        )
        predicted = compute + decoded_delta
        output = torch.where(
            sequence.query_valid_mask.unsqueeze(-1),
            predicted.to(dtype=hidden_states.dtype),
            hidden_states,
        )
        return StaticSpanExecution(
            output=output,
            causal_state=causal_state,
            hidden=hidden,
            modal_delta=modal_delta,
            decoded_delta=decoded_delta,
        )

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> StaticSpanExecution:
        """Execute without boundary tracing and expose modal intermediates."""

        self._validate_inputs(hidden_states, sequence)
        return self._execute(
            hidden_states,
            sequence,
            trace=None,
            prefix="static_span",
        )

    def forward_context(
        self,
        hidden_states: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None = None,
        prefix: str = "static_span",
    ) -> Tensor:
        """Execute an explicit query/key-mask prefill context."""

        self._validate_inputs(hidden_states, sequence)
        instrumented_input = record(
            trace,
            self.input_activation_name,
            hidden_states,
        )
        result = self._execute(
            instrumented_input,
            sequence,
            trace=trace,
            prefix=prefix,
        )
        return record(
            trace,
            self.output_activation_name,
            result.output,
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

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
        sequence_context: SequenceContext | None = None,
    ) -> Tensor:
        """Execute through the layer ABI or an explicit sparse query context."""

        if sequence_context is None:
            sequence_context = self._default_sequence_context(
                hidden_states,
                attention_mask=attention_mask,
            )
        elif attention_mask is not None:
            normalized = _normalize_binary_mask(
                attention_mask,
                shape=hidden_states.shape[:2],
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
        """Implement the backend-neutral grouped compiled-segment protocol."""

        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        if (
            segment.input_activation != self.input_activation_name
            or segment.output_activation != self.output_activation_name
        ):
            raise ValueError(
                "compiled segment boundaries do not match this executor"
            )
        output = self.forward_context(
            hidden_states,
            sequence=sequence,
            trace=trace,
            prefix=segment.id,
        )
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output=output,
        )

    def execution_fingerprint(self) -> str:
        """Hash learned/fixed tensor state and non-tensor execution options."""

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(module_state_fingerprint(self).encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "graph": self._graph_config(),
                    "input_activation_name": self.input_activation_name,
                    "output_activation_name": self.output_activation_name,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
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
        """Return a strict source-free payload accepted by weights-only load."""

        reference = self.graph.state_input_weight
        dtype_name = str(reference.dtype).removeprefix("torch.")
        if dtype_name not in _FLOAT_DTYPES:
            raise ValueError("unsupported graph dtype for artifact export")
        if (
            self.decoder.dtype != reference.dtype
            or self.decoder.device != reference.device
            or not torch.isfinite(self.decoder).all()
        ):
            raise ValueError(
                "decoder must be finite and match the graph runtime"
            )
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "input_activation_name": self.input_activation_name,
            "output_activation_name": self.output_activation_name,
            "decoder": self.decoder.detach().to(device="cpu").clone(),
            "graph_config": self._graph_config(),
            "graph_dtype": dtype_name,
            "graph_state_dict": self._validated_cpu_graph_state(),
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> StaticSpanBlockExecutor:
        """Strictly restore a complete source-independent runtime artifact."""

        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError("static-span executor artifact fields are invalid")
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("unsupported static-span executor artifact")
        fingerprint = state["execution_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint
            )
        ):
            raise ValueError("artifact execution fingerprint is invalid")

        raw_config = state["graph_config"]
        if (
            not isinstance(raw_config, Mapping)
            or set(raw_config) != _GRAPH_CONFIG_FIELDS
        ):
            raise ValueError("static-span graph config fields are invalid")
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
            raise ValueError("static-span graph dtype is invalid")
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
            raise ValueError("static-span graph state fields are invalid")
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
                    f"static-span graph state {name!r} is invalid"
                )
            restored_graph_state[name] = value.detach().to(
                device=device,
                dtype=dtype,
            )
        graph.load_state_dict(restored_graph_state, strict=True)

        decoder = state["decoder"]
        if (
            not isinstance(decoder, Tensor)
            or decoder.device.type != "cpu"
            or not decoder.is_floating_point()
            or decoder.dtype != dtype
            or decoder.shape
            != (graph.input_modes, graph.output_modes)
            or not torch.isfinite(decoder).all()
        ):
            raise ValueError("artifact decoder is invalid")
        result = cls(
            graph=graph,
            decoder=decoder.to(device=device, dtype=dtype),
            input_activation_name=state[  # type: ignore[arg-type]
                "input_activation_name"
            ],
            output_activation_name=state[  # type: ignore[arg-type]
                "output_activation_name"
            ],
        )
        if result.execution_fingerprint() != fingerprint:
            raise ValueError(
                "static-span executor execution fingerprint mismatch"
            )
        result.eval()
        return result


# A descriptive alias for callers that prefer the residual contract in the
# type name.  Both names refer to the same artifact/runtime implementation.
StaticResidualSpanExecutor = StaticSpanBlockExecutor


__all__ = [
    "StaticResidualSpanExecutor",
    "StaticSpanBlockExecutor",
    "StaticSpanExecution",
]
