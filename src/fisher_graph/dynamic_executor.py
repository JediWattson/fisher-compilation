"""Variable-length causal execution in shared Fisher-mode coordinates.

The fixed reference executor learns a separate routing matrix for every
``(input_position, output_position)`` pair.  That is maximally inspectable but
ties both its parameters and its runtime contract to one sequence length.

This module provides a complementary prefill backend whose parameters are
independent of sequence length.  It compresses residual-stream values into
shared Fisher modes, accumulates a small bank of exponentially decayed causal
states, and decodes shared output modes.  Logical-position gaps determine the
decay, so padding and nonzero position offsets do not become learned table
indices.

This is a trainable backend, not a conversion of the existing fixed-length
artifact.  Matching a source layer requires fitting its shared parameters on
calibration examples drawn from the desired sequence-length distribution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
from .layers import LayerExecutor
from .modes import FisherModeBasis

DynamicActivation = Literal["gelu", "silu", "tanh", "identity"]


def _require_positive_integer(value: int, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite_floating_tensor(
    value: Tensor,
    *,
    name: str,
    ndim: int,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class DynamicModalExecutorConfig:
    """Shape and activation contract for a shared causal modal executor."""

    input_activation: str
    output_activation: str
    width: int
    input_modes: int
    output_modes: int
    state_channels: int
    routing_width: int
    activation: DynamicActivation = "gelu"
    window_size: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_activation, str)
            or not self.input_activation
        ):
            raise ValueError("input_activation must be a nonempty string")
        if (
            not isinstance(self.output_activation, str)
            or not self.output_activation
        ):
            raise ValueError("output_activation must be a nonempty string")
        for name, value in (
            ("width", self.width),
            ("input_modes", self.input_modes),
            ("output_modes", self.output_modes),
            ("state_channels", self.state_channels),
            ("routing_width", self.routing_width),
        ):
            _require_positive_integer(value, name=name)
        if self.input_modes > self.width:
            raise ValueError("input_modes cannot exceed residual width")
        if self.output_modes > self.width:
            raise ValueError("output_modes cannot exceed residual width")
        if self.activation not in ("gelu", "silu", "tanh", "identity"):
            raise ValueError(
                f"unsupported dynamic modal activation: {self.activation!r}"
            )
        if self.window_size is not None:
            _require_positive_integer(
                self.window_size,
                name="window_size",
            )


class SharedModalProjection(nn.Module):
    """A sequence-length-independent projection around one pooled mean."""

    def __init__(
        self,
        *,
        activation_name: str,
        mean: Tensor,
        vectors: Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(activation_name, str) or not activation_name:
            raise ValueError("activation_name must be a nonempty string")
        _require_finite_floating_tensor(mean, name="mean", ndim=1)
        _require_finite_floating_tensor(vectors, name="vectors", ndim=2)
        if vectors.shape[0] != mean.shape[0]:
            raise ValueError("modal vectors must match the residual width")
        if vectors.shape[1] == 0:
            raise ValueError("a projection must retain at least one mode")
        if vectors.dtype != mean.dtype or vectors.device != mean.device:
            raise ValueError("mean and vectors must share dtype and device")
        self.activation_name = activation_name
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("vectors", vectors.detach().clone())

    @classmethod
    def from_basis(
        cls,
        basis: FisherModeBasis,
        *,
        modes: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> SharedModalProjection:
        """Build a shared projection from the pooled part of a Fisher basis."""

        if not isinstance(basis, FisherModeBasis):
            raise TypeError("basis must be a FisherModeBasis")
        if type(modes) is not int or not 1 <= modes <= basis.width:
            raise ValueError(f"modes must be between 1 and {basis.width}")
        if not dtype.is_floating_point:
            raise ValueError("projection dtype must be floating point")
        return cls(
            activation_name=basis.activation_name,
            mean=basis.mean.to(dtype=dtype, device=device),
            vectors=basis.vectors[:, :modes].to(
                dtype=dtype,
                device=device,
            ),
        )

    @property
    def width(self) -> int:
        return self.mean.shape[0]

    @property
    def modes(self) -> int:
        return self.vectors.shape[1]

    def _validate_values(self, values: Tensor, *, width: int) -> None:
        if not isinstance(values, Tensor):
            raise TypeError("modal projection values must be a Tensor")
        if values.ndim != 3:
            raise ValueError(
                "modal projection values must have shape "
                "[batch, sequence, features]"
            )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("modal projection values cannot be empty")
        if values.shape[2] != width:
            raise ValueError(
                f"expected feature width {width}, got {values.shape[2]}"
            )
        if not values.is_floating_point():
            raise ValueError("modal projection values must be floating point")
        if values.dtype != self.mean.dtype or values.device != self.mean.device:
            raise ValueError(
                "modal projection values must match projection dtype and device"
            )

    def encode(self, activations: Tensor) -> Tensor:
        self._validate_values(activations, width=self.width)
        return (activations - self.mean) @ self.vectors

    def decode(self, coordinates: Tensor) -> Tensor:
        self._validate_values(coordinates, width=self.modes)
        return coordinates @ self.vectors.transpose(0, 1) + self.mean

    def forward(self, activations: Tensor) -> Tensor:
        return self.decode(self.encode(activations))


class StatefulCausalModalGraph(nn.Module):
    """A shared recurrent modal graph with relative-position decay.

    For state channel ``c`` and valid token ``t`` the recurrence is

    ``state[c, t] = exp(-rate[c] * gap[t]) * state[c, t-1] + x[t] @ W[c]``.

    ``gap`` is measured between logical positions of consecutive valid keys.
    Unused/padded keys do not update either the state or its last position.
    All decay rates are positive via a softplus parameterization.
    """

    def __init__(
        self,
        *,
        input_modes: int,
        output_modes: int,
        state_channels: int,
        routing_width: int,
        activation: DynamicActivation = "gelu",
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_modes", input_modes),
            ("output_modes", output_modes),
            ("state_channels", state_channels),
            ("routing_width", routing_width),
        ):
            _require_positive_integer(value, name=name)
        if activation not in ("gelu", "silu", "tanh", "identity"):
            raise ValueError(
                f"unsupported dynamic modal activation: {activation!r}"
            )
        if window_size is not None:
            _require_positive_integer(window_size, name="window_size")
        self.input_modes = input_modes
        self.output_modes = output_modes
        self.state_channels = state_channels
        self.routing_width = routing_width
        self.activation = activation
        self.window_size = window_size

        self.state_input_weight = nn.Parameter(
            torch.empty(state_channels, input_modes, routing_width)
        )
        self.raw_decay_rate = nn.Parameter(torch.empty(state_channels))
        self.hidden_weight = nn.Parameter(
            torch.empty(
                state_channels * routing_width,
                routing_width,
            )
        )
        self.hidden_bias = nn.Parameter(torch.empty(routing_width))
        self.output_weight = nn.Parameter(
            torch.empty(routing_width, output_modes)
        )
        self.output_bias = nn.Parameter(torch.empty(output_modes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.state_input_weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.input_modes),
        )
        rates = torch.logspace(
            -2.0,
            0.0,
            self.state_channels,
            dtype=self.raw_decay_rate.dtype,
            device=self.raw_decay_rate.device,
        )
        with torch.no_grad():
            self.raw_decay_rate.copy_(torch.log(torch.expm1(rates)))
        nn.init.xavier_uniform_(self.hidden_weight)
        nn.init.zeros_(self.hidden_bias)
        nn.init.xavier_uniform_(self.output_weight)
        nn.init.zeros_(self.output_bias)

    @property
    def decay_rate(self) -> Tensor:
        return F.softplus(self.raw_decay_rate)

    @property
    def learned_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_coordinates(self, coordinates: Tensor) -> None:
        if not isinstance(coordinates, Tensor):
            raise TypeError("modal coordinates must be a Tensor")
        if coordinates.ndim != 3:
            raise ValueError(
                "modal coordinates must have shape "
                "[batch, sequence, input_modes]"
            )
        if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
            raise ValueError("modal coordinates cannot be empty")
        if coordinates.shape[2] != self.input_modes:
            raise ValueError(
                f"expected {self.input_modes} input modes, "
                f"got {coordinates.shape[2]}"
            )
        if not coordinates.is_floating_point():
            raise ValueError("modal coordinates must be floating point")
        if (
            coordinates.dtype != self.state_input_weight.dtype
            or coordinates.device != self.state_input_weight.device
        ):
            raise ValueError(
                "modal coordinates must match graph dtype and device"
            )

    def _normalize_sequence_inputs(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None,
        key_valid_mask: Tensor | None,
        logical_positions: Tensor | None,
        key_logical_positions: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_coordinates(coordinates)
        batch_size, sequence_length, _ = coordinates.shape
        if query_valid_mask is None and key_valid_mask is None:
            key_valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=coordinates.device,
            )
            query_valid_mask = key_valid_mask
        elif query_valid_mask is None:
            query_valid_mask = key_valid_mask
        elif key_valid_mask is None:
            key_valid_mask = query_valid_mask
        assert query_valid_mask is not None
        assert key_valid_mask is not None
        for name, mask in (
            ("query_valid_mask", query_valid_mask),
            ("key_valid_mask", key_valid_mask),
        ):
            if not isinstance(mask, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if mask.dtype is not torch.bool:
                raise ValueError(f"{name} must be boolean")
            if mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"{name} must have shape "
                    f"{(batch_size, sequence_length)}"
                )
            if mask.device != coordinates.device:
                raise ValueError(
                    f"{name} must share the coordinate device"
                )
        if logical_positions is None:
            logical_positions = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=coordinates.device,
            ).unsqueeze(0).expand(batch_size, -1)
        if key_logical_positions is None:
            key_logical_positions = logical_positions
        if not isinstance(logical_positions, Tensor):
            raise TypeError("logical_positions must be a Tensor")
        if not isinstance(key_logical_positions, Tensor):
            raise TypeError("key_logical_positions must be a Tensor")
        if logical_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("logical_positions must use an integer dtype")
        if key_logical_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "key_logical_positions must use an integer dtype"
            )
        if logical_positions.shape != (batch_size, sequence_length):
            raise ValueError(
                "logical_positions must match the coordinate batch and length"
            )
        if key_logical_positions.shape != (batch_size, sequence_length):
            raise ValueError(
                "key_logical_positions must match the coordinate "
                "batch and length"
            )
        if logical_positions.device != coordinates.device:
            raise ValueError(
                "logical_positions must share the coordinate device"
            )
        if key_logical_positions.device != coordinates.device:
            raise ValueError(
                "key_logical_positions must share the coordinate device"
            )
        if (logical_positions[query_valid_mask] < 0).any():
            raise ValueError("valid query logical positions cannot be negative")
        if (key_logical_positions[key_valid_mask] < 0).any():
            raise ValueError("valid key logical positions cannot be negative")

        seen = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=coordinates.device,
        )
        last_position = torch.zeros(
            batch_size,
            dtype=logical_positions.dtype,
            device=coordinates.device,
        )
        for index in range(sequence_length):
            valid = key_valid_mask[:, index]
            position = key_logical_positions[:, index]
            if (valid & seen & (position <= last_position)).any():
                raise ValueError(
                    "valid key logical positions must be strictly increasing "
                    "(and therefore nondecreasing)"
                )
            last_position = torch.where(valid, position, last_position)
            seen = seen | valid
        return (
            query_valid_mask,
            key_valid_mask,
            logical_positions,
            key_logical_positions,
        )

    def compute_causal_state(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        """Return the causal state bank as ``[batch, sequence, channel, route]``."""

        (
            query_valid_mask,
            key_valid_mask,
            logical_positions,
            key_logical_positions,
        ) = self._normalize_sequence_inputs(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        batch_size, sequence_length, _ = coordinates.shape
        safe_coordinates = torch.where(
            key_valid_mask.unsqueeze(-1),
            coordinates,
            torch.zeros_like(coordinates),
        )
        contributions = torch.einsum(
            "bsi,cih->bsch",
            safe_coordinates,
            self.state_input_weight,
        )
        if (
            self.window_size is not None
            or not torch.equal(logical_positions, key_logical_positions)
            or not torch.equal(query_valid_mask, key_valid_mask)
        ):
            relative_gap = (
                logical_positions.unsqueeze(2)
                - key_logical_positions.unsqueeze(1)
            )
            allowed = (
                query_valid_mask.unsqueeze(2)
                & key_valid_mask.unsqueeze(1)
                & (relative_gap >= 0)
            )
            if self.window_size is not None:
                allowed = allowed & (relative_gap < self.window_size)
            safe_gap = relative_gap.clamp_min(0).to(
                dtype=coordinates.dtype
            )
            decay = torch.exp(
                -safe_gap.unsqueeze(-1) * self.decay_rate.view(1, 1, 1, -1)
            )
            weights = torch.where(
                allowed.unsqueeze(-1),
                decay,
                torch.zeros_like(decay),
            )
            return torch.einsum(
                "bqkc,bkch->bqch",
                weights,
                contributions,
            )

        state = coordinates.new_zeros(
            batch_size,
            self.state_channels,
            self.routing_width,
        )
        seen = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=coordinates.device,
        )
        last_position = torch.zeros(
            batch_size,
            dtype=key_logical_positions.dtype,
            device=coordinates.device,
        )
        states: list[Tensor] = []
        rates = self.decay_rate.view(1, self.state_channels, 1)
        for index in range(sequence_length):
            valid = key_valid_mask[:, index]
            position = key_logical_positions[:, index]
            gap = torch.where(
                seen,
                position - last_position,
                torch.zeros_like(position),
            ).to(dtype=coordinates.dtype)
            decay = torch.exp(-gap.view(batch_size, 1, 1) * rates)
            updated = state * decay + contributions[:, index]
            state = torch.where(
                valid.view(batch_size, 1, 1),
                updated,
                state,
            )
            states.append(state)
            last_position = torch.where(valid, position, last_position)
            seen = seen | valid
        return torch.stack(states, dim=1)

    def _activate(self, values: Tensor) -> Tensor:
        if self.activation == "gelu":
            return F.gelu(values)
        if self.activation == "silu":
            return F.silu(values)
        if self.activation == "tanh":
            return torch.tanh(values)
        return values

    def compute_hidden(
        self,
        causal_state: Tensor,
        *,
        query_valid_mask: Tensor,
    ) -> Tensor:
        expected_tail = (self.state_channels, self.routing_width)
        if causal_state.ndim != 4 or causal_state.shape[2:] != expected_tail:
            raise ValueError(
                "causal_state must have shape "
                "[batch, sequence, state_channels, routing_width]"
            )
        if (
            causal_state.dtype != self.hidden_weight.dtype
            or causal_state.device != self.hidden_weight.device
        ):
            raise ValueError("causal_state must match graph dtype and device")
        if (
            query_valid_mask.dtype is not torch.bool
            or query_valid_mask.shape != causal_state.shape[:2]
            or query_valid_mask.device != causal_state.device
        ):
            raise ValueError(
                "query_valid_mask must be a matching boolean matrix"
            )
        flattened = causal_state.flatten(start_dim=2)
        hidden = self._activate(
            flattened @ self.hidden_weight + self.hidden_bias
        )
        return torch.where(
            query_valid_mask.unsqueeze(-1),
            hidden,
            torch.zeros_like(hidden),
        )

    def compute_output(
        self,
        hidden: Tensor,
        *,
        query_valid_mask: Tensor,
    ) -> Tensor:
        if (
            hidden.ndim != 3
            or hidden.shape[2] != self.routing_width
        ):
            raise ValueError(
                "hidden must have shape "
                "[batch, sequence, routing_width]"
            )
        if (
            hidden.dtype != self.output_weight.dtype
            or hidden.device != self.output_weight.device
        ):
            raise ValueError("hidden must match graph dtype and device")
        if (
            query_valid_mask.dtype is not torch.bool
            or query_valid_mask.shape != hidden.shape[:2]
            or query_valid_mask.device != hidden.device
        ):
            raise ValueError(
                "query_valid_mask must be a matching boolean matrix"
            )
        output = hidden @ self.output_weight + self.output_bias
        return torch.where(
            query_valid_mask.unsqueeze(-1),
            output,
            torch.zeros_like(output),
        )

    def forward(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        (
            query_valid_mask,
            key_valid_mask,
            logical_positions,
            key_logical_positions,
        ) = self._normalize_sequence_inputs(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        state = self.compute_causal_state(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        hidden = self.compute_hidden(
            state,
            query_valid_mask=query_valid_mask,
        )
        return self.compute_output(
            hidden,
            query_valid_mask=query_valid_mask,
        )


class VariableLengthCausalModalExecutor(LayerExecutor):
    """A trace-aware, prefill-only dynamic causal modal layer executor."""

    def __init__(
        self,
        input_projection: SharedModalProjection,
        graph: StatefulCausalModalGraph,
        output_projection: SharedModalProjection,
    ) -> None:
        super().__init__()
        if not isinstance(input_projection, SharedModalProjection):
            raise TypeError(
                "input_projection must be a SharedModalProjection"
            )
        if not isinstance(graph, StatefulCausalModalGraph):
            raise TypeError("graph must be a StatefulCausalModalGraph")
        if not isinstance(output_projection, SharedModalProjection):
            raise TypeError(
                "output_projection must be a SharedModalProjection"
            )
        if input_projection.width != output_projection.width:
            raise ValueError("input and output residual widths must match")
        if input_projection.modes != graph.input_modes:
            raise ValueError(
                "input projection mode count does not match the graph"
            )
        if output_projection.modes != graph.output_modes:
            raise ValueError(
                "output projection mode count does not match the graph"
            )
        reference = input_projection.mean
        if (
            output_projection.mean.dtype != reference.dtype
            or output_projection.mean.device != reference.device
            or graph.state_input_weight.dtype != reference.dtype
            or graph.state_input_weight.device != reference.device
        ):
            raise ValueError(
                "projections and graph must share dtype and device"
            )
        self.input_projection = input_projection
        self.graph = graph
        self.output_projection = output_projection

    @classmethod
    def from_bases(
        cls,
        input_basis: FisherModeBasis,
        output_basis: FisherModeBasis,
        *,
        input_modes: int,
        output_modes: int,
        state_channels: int,
        routing_width: int,
        activation: DynamicActivation = "gelu",
        window_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> VariableLengthCausalModalExecutor:
        """Initialize a trainable dynamic graph around two Fisher bases."""

        input_projection = SharedModalProjection.from_basis(
            input_basis,
            modes=input_modes,
            dtype=dtype,
            device=device,
        )
        output_projection = SharedModalProjection.from_basis(
            output_basis,
            modes=output_modes,
            dtype=dtype,
            device=device,
        )
        if input_projection.width != output_projection.width:
            raise ValueError("input and output Fisher widths must match")
        graph = StatefulCausalModalGraph(
            input_modes=input_modes,
            output_modes=output_modes,
            state_channels=state_channels,
            routing_width=routing_width,
            activation=activation,
            window_size=window_size,
        ).to(dtype=dtype, device=device)
        return cls(input_projection, graph, output_projection)

    @property
    def config(self) -> DynamicModalExecutorConfig:
        return DynamicModalExecutorConfig(
            input_activation=self.input_projection.activation_name,
            output_activation=self.output_projection.activation_name,
            width=self.input_projection.width,
            input_modes=self.graph.input_modes,
            output_modes=self.graph.output_modes,
            state_channels=self.graph.state_channels,
            routing_width=self.graph.routing_width,
            activation=self.graph.activation,
            window_size=self.graph.window_size,
        )

    def execution_fingerprint(self) -> str:
        """Hash learned state and every live option affecting execution."""

        payload = {
            "executor": (
                "fisher_graph.variable_length_causal_modal_executor.v1"
            ),
            "config": asdict(self.config),
            "module_training": tuple(
                module.training for module in self.modules()
            ),
            "state_sha256": module_state_fingerprint(self),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

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

    @staticmethod
    def _normalize_attention_mask(
        attention_mask: Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a Tensor")
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must match hidden-state batch and length"
            )
        if attention_mask.device != device:
            raise ValueError(
                "attention_mask must share the hidden-state device"
            )
        if attention_mask.dtype is torch.bool:
            return attention_mask
        if not (
            attention_mask.is_floating_point()
            or attention_mask.dtype
            in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        ):
            raise ValueError("attention_mask must be boolean or binary")
        if not torch.isfinite(attention_mask).all() or not (
            (attention_mask == 0) | (attention_mask == 1)
        ).all():
            raise ValueError("attention_mask must contain only zero and one")
        return attention_mask.to(dtype=torch.bool)

    def _default_sequence_context(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
    ) -> SequenceContext:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            valid_mask = self._normalize_attention_mask(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=hidden_states.device,
            )
        logical_positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch_size, -1)
        return SequenceContext(
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=logical_positions,
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

    def _validate_context(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        batch_size, sequence_length, _ = hidden_states.shape
        if sequence.batch_size != batch_size:
            raise ValueError(
                "sequence context and hidden states must share a batch size"
            )
        if sequence.query_length != sequence_length:
            raise ValueError(
                "sequence query length must match hidden-state length"
            )
        if sequence.key_length != sequence_length:
            raise ValueError(
                "dynamic modal prefill requires key and query lengths "
                "to match hidden-state length"
            )
        if sequence.device != hidden_states.device:
            raise ValueError(
                "sequence context must share the hidden-state device"
            )
        if sequence.phase != "prefill":
            raise ValueError(
                "dynamic modal executor does not yet support cached decode"
            )
        if sequence.cache_state is not None:
            raise ValueError(
                "dynamic modal executor does not accept cache state"
            )
        if sequence.cache_positions is not None:
            raise ValueError(
                "dynamic modal executor does not accept cache positions"
            )

    def forward_context(
        self,
        hidden_states: Tensor,
        *,
        sequence: SequenceContext,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        """Execute a normalized prefill context with explicit positions/masks."""

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a Tensor")
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, width]"
            )
        if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
            raise ValueError("hidden_states cannot be empty")
        if hidden_states.shape[2] != self.input_projection.width:
            raise ValueError(
                f"expected residual width {self.input_projection.width}, "
                f"got {hidden_states.shape[2]}"
            )
        if not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be floating point")
        if (
            hidden_states.dtype != self.input_projection.mean.dtype
            or hidden_states.device != self.input_projection.mean.device
        ):
            raise ValueError(
                "hidden_states must match executor dtype and device"
            )
        self._validate_context(hidden_states, sequence)

        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        input_modes = record(
            trace,
            f"{prefix}.modal.input",
            self.input_projection.encode(hidden_states),
        )
        causal_state = self.graph.compute_causal_state(
            input_modes,
            query_valid_mask=sequence.query_valid_mask,
            key_valid_mask=sequence.key_valid_mask,
            logical_positions=sequence.logical_positions,
            key_logical_positions=sequence.key_logical_positions,
        )
        causal_state = record(
            trace,
            f"{prefix}.modal.causal_state",
            causal_state,
        )
        hidden = self.graph.compute_hidden(
            causal_state,
            query_valid_mask=sequence.query_valid_mask,
        )
        hidden = record(trace, f"{prefix}.modal.hidden", hidden)
        output_modes = self.graph.compute_output(
            hidden,
            query_valid_mask=sequence.query_valid_mask,
        )
        output_modes = record(
            trace,
            f"{prefix}.modal.output",
            output_modes,
        )
        output = self.output_projection.decode(output_modes)
        output = torch.where(
            sequence.query_valid_mask.unsqueeze(-1),
            output,
            torch.zeros_like(output),
        )
        return record(trace, f"{prefix}.output", output)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
        sequence_context: SequenceContext | None = None,
    ) -> Tensor:
        """Execute via ``LayerExecutor`` or an explicit ``SequenceContext``."""

        if sequence_context is None:
            sequence_context = self._default_sequence_context(
                hidden_states,
                attention_mask=attention_mask,
            )
        elif attention_mask is not None:
            normalized_mask = self._normalize_attention_mask(
                attention_mask,
                batch_size=hidden_states.shape[0],
                sequence_length=hidden_states.shape[1],
                device=hidden_states.device,
            )
            if not torch.equal(
                normalized_mask,
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
        """Execute directly through the mixed-runtime backend protocol."""

        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        config = self.config
        if (
            segment.input_activation != config.input_activation
            or segment.output_activation != config.output_activation
        ):
            raise ValueError(
                "compiled segment boundaries do not match this dynamic executor"
            )
        input_suffix = ".input"
        candidate_prefix = (
            segment.input_activation[: -len(input_suffix)]
            if segment.input_activation.endswith(input_suffix)
            else segment.id
        )
        prefix = (
            candidate_prefix
            if segment.output_activation == f"{candidate_prefix}.output"
            else segment.id
        )
        output = self.forward_context(
            hidden_states,
            sequence=sequence,
            trace=trace,
            prefix=prefix,
        )
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output=output,
        )


__all__ = [
    "DynamicActivation",
    "DynamicModalExecutorConfig",
    "SharedModalProjection",
    "StatefulCausalModalGraph",
    "VariableLengthCausalModalExecutor",
]
