"""Frozen fused runtimes for completed position-conditioned modal graphs.

The unfused modal executor is deliberately easy to inspect: it materializes
input coordinates, routing features, output coordinates, completion
coordinates, and the decoded residual stream.  This module preserves that
logical execution when a trace or intervention is requested, while providing
an algebraically folded inference path when instrumentation is disabled.
"""

from __future__ import annotations

import hashlib
import io
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ActivationIntervention, ActivationTrace, record
from .config import TransformerConfig
from .layers import LayerExecutor
from .modal_completion import (
    LocalModalCompletionGraph,
    ModalCompletionConfig,
    ModalCompletionKind,
    PositionConditionedCompletedModalGraphExecutor,
    PositionConditionedModalCompletion,
)
from .modal_executor import (
    CausalModalMLPGraph,
    ModalExecutorConfig,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
)
from .model import ToyTransformer, TransformerOutput


@dataclass(frozen=True, slots=True)
class FusedModalLayerConfig:
    """Portable topology for one fused completed modal layer."""

    input_activation: str
    output_activation: str
    sequence_length: int
    width: int
    input_modes: int
    routing_width: int
    output_modes: int
    completion_kind: ModalCompletionKind

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_activation, str)
            or not isinstance(self.output_activation, str)
            or not self.input_activation
            or not self.output_activation
        ):
            raise ValueError("fused modal activation names cannot be empty")
        dimensions = (
            self.sequence_length,
            self.width,
            self.input_modes,
            self.routing_width,
            self.output_modes,
        )
        if any(type(value) is not int for value in dimensions):
            raise ValueError("fused modal dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("fused modal dimensions must be positive")
        if self.input_modes > self.width or self.output_modes >= self.width:
            raise ValueError("fused modal mode counts do not match the width")
        if self.completion_kind not in (
            "shared_local_linear",
            "position_local_linear",
        ):
            raise ValueError("unsupported fused modal completion kind")

    @property
    def tail_modes(self) -> int:
        return self.width - self.output_modes


@dataclass(frozen=True, slots=True)
class FusedTwoLayerStackConfig:
    """Portable topology and selected fast path for a two-layer stack."""

    first: FusedModalLayerConfig
    second: FusedModalLayerConfig
    cross_layer_bypass: bool

    def __post_init__(self) -> None:
        if not isinstance(self.first, FusedModalLayerConfig) or not isinstance(
            self.second,
            FusedModalLayerConfig,
        ):
            raise ValueError("fused stack layers must use fused layer configs")
        if type(self.cross_layer_bypass) is not bool:
            raise ValueError("cross_layer_bypass must be boolean")
        if self.first.sequence_length != self.second.sequence_length:
            raise ValueError("fused stack sequence lengths differ")
        if self.first.width != self.second.width:
            raise ValueError("fused stack residual widths differ")
        if self.first.output_activation != self.second.input_activation:
            raise ValueError("fused stack activation boundary names differ")


@dataclass(frozen=True, slots=True)
class LazyInstrumentationStatus:
    """Thread-safe snapshot of one lazy runtime's dispatch and cache state."""

    residency: str
    loaded: bool
    last_dispatch: str | None
    fast_path_calls: int
    instrumented_path_calls: int
    load_attempts: int
    successful_loads: int
    cache_hits: int
    failed_loads: int
    evictions: int
    derived_kernel_verifications: int
    resident_fast_tensor_bytes: int
    resident_sidecar_tensor_bytes: int
    sidecar_file_bytes_read: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class _SidecarArtifact:
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str):
            raise ValueError("lazy sidecar filename must be a string")
        if not isinstance(self.sha256, str):
            raise ValueError("lazy sidecar sha256 must be a string")
        path = Path(self.filename)
        if (
            not self.filename
            or path.is_absolute()
            or path.name != self.filename
            or self.filename in (".", "..")
        ):
            raise ValueError(
                "lazy sidecar filenames must be nonempty basenames"
            )
        if (
            len(self.sha256) != 64
            or self.sha256.lower() != self.sha256
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("lazy sidecar hashes must be lowercase SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("lazy sidecar size_bytes must be a positive integer")


_LAYER_BUFFER_NAMES = (
    "input_mean",
    "input_vectors",
    "input_scale",
    "coordinate_kernel",
    "input_kernel",
    "hidden_bias",
    "output_weight",
    "output_bias",
    "output_scale",
    "completion_weight",
    "completion_bias",
    "output_mean",
    "output_vectors",
    "fused_output_weight",
    "fused_output_bias",
)

_LAZY_FAST_STATE_NAMES = (
    "bridge_kernel",
    "bridge_bias",
    "first_input_mean",
    "first_input_kernel",
    "first_hidden_bias",
    "second_fused_output_weight",
    "second_fused_output_bias",
)

_LAZY_SIDECAR_NAMES = (
    "layer_0_executor",
    "layer_0_output_completion",
    "layer_1_executor",
    "layer_1_output_completion",
)

_LAZY_PROVENANCE_HASH_NAMES = (
    "checkpoint_sha256",
    "fisher_sha256",
    "teacher_state_sha256",
)

_MODAL_EXECUTOR_CONFIG_KEYS = {
    "input_activation",
    "output_activation",
    "sequence_length",
    "input_modes",
    "output_modes",
    "routing_width",
}

_MODAL_COMPLETION_CONFIG_KEYS = {
    "activation_name",
    "sequence_length",
    "width",
    "kept_modes",
    "graph_kind",
}


def _clone_buffer(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("fused executor buffers must be tensors")
    if not value.is_floating_point():
        raise ValueError("fused executor buffers must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError("fused executor buffers must be finite")
    return value.detach().clone()


def _require_lazy_provenance(
    provenance: Mapping[str, object],
) -> None:
    for name in _LAZY_PROVENANCE_HASH_NAMES:
        value = provenance.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"lazy runtime provenance requires lowercase SHA-256 {name}"
            )


def _causal_future_mask(sequence_length: int, *, device: torch.device) -> Tensor:
    positions = torch.arange(sequence_length, device=device)
    return positions.view(-1, 1) < positions.view(1, -1)


def _require_causal_zeros(name: str, kernel: Tensor) -> None:
    """Reject a folded kernel with any future-position dependency."""

    if kernel.ndim < 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"{name} must begin with a square position grid")
    future = _causal_future_mask(kernel.shape[0], device=kernel.device)
    if torch.count_nonzero(kernel[future]).item() != 0:
        raise ValueError(f"{name} contains noncausal future-position weights")


def _consistency_tolerances(dtype: torch.dtype, width: int) -> tuple[float, float]:
    epsilon = torch.finfo(dtype).eps
    tolerance = max(1e-9, float(epsilon) * max(width, 1) * 4)
    return tolerance, tolerance


def _require_consistent(
    name: str,
    actual: Tensor,
    expected: Tensor,
    *,
    width: int,
) -> None:
    rtol, atol = _consistency_tolerances(actual.dtype, width)
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        difference = (actual - expected).abs().max().item()
        raise ValueError(
            f"{name} is inconsistent with the logical fused buffers "
            f"(maximum difference {difference:.6g})"
        )


def _fold_input_kernel(
    input_vectors: Tensor,
    coordinate_kernel: Tensor,
    input_scale: Tensor,
) -> Tensor:
    normalized_coordinate_kernel = (
        coordinate_kernel
        / input_scale.view(
            1,
            input_scale.shape[0],
            input_scale.shape[1],
            1,
        )
    )
    return torch.einsum(
        "wi,tsih->tswh",
        input_vectors,
        normalized_coordinate_kernel,
    )


def _fold_output_decoder(
    *,
    output_weight: Tensor,
    output_bias: Tensor,
    output_scale: Tensor,
    completion_weight: Tensor,
    completion_bias: Tensor,
    output_mean: Tensor,
    output_vectors: Tensor,
    output_modes: int,
) -> tuple[Tensor, Tensor]:
    kept_decoder = output_vectors[:, :output_modes].T
    tail_decoder = output_vectors[:, output_modes:].T
    decoder = kept_decoder.unsqueeze(0) + torch.einsum(
        "sot,tw->sow",
        completion_weight,
        tail_decoder,
    )
    kept_weight = output_weight * output_scale.unsqueeze(1)
    kept_bias = output_bias * output_scale
    fused_output_weight = torch.einsum(
        "sho,sow->shw",
        kept_weight,
        decoder,
    )
    fused_output_bias = (
        torch.einsum("so,sow->sw", kept_bias, decoder)
        + completion_bias @ tail_decoder
        + output_mean
    )
    return fused_output_weight, fused_output_bias


class FusedCompletedModalLayer(LayerExecutor):
    """A trace-aware completed modal layer with a folded inference path.

    The fast path executes

    ``(x - position_mean) @ input_kernel -> GELU -> fused_output``

    and intentionally keeps input centering explicit.  Folding the
    position-dependent mean into a bias would make intervention semantics
    obscure and can change numerical behavior when the input tap is edited.
    """

    def __init__(
        self,
        config: FusedModalLayerConfig,
        *,
        input_mean: Tensor,
        input_vectors: Tensor,
        input_scale: Tensor,
        coordinate_kernel: Tensor,
        input_kernel: Tensor,
        hidden_bias: Tensor,
        output_weight: Tensor,
        output_bias: Tensor,
        output_scale: Tensor,
        completion_weight: Tensor,
        completion_bias: Tensor,
        output_mean: Tensor,
        output_vectors: Tensor,
        fused_output_weight: Tensor,
        fused_output_bias: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        expected = {
            "input_mean": (config.sequence_length, config.width),
            "input_vectors": (config.width, config.input_modes),
            "input_scale": (config.sequence_length, config.input_modes),
            "coordinate_kernel": (
                config.sequence_length,
                config.sequence_length,
                config.input_modes,
                config.routing_width,
            ),
            "input_kernel": (
                config.sequence_length,
                config.sequence_length,
                config.width,
                config.routing_width,
            ),
            "hidden_bias": (config.sequence_length, config.routing_width),
            "output_weight": (
                config.sequence_length,
                config.routing_width,
                config.output_modes,
            ),
            "output_bias": (config.sequence_length, config.output_modes),
            "output_scale": (config.sequence_length, config.output_modes),
            "completion_weight": (
                config.sequence_length,
                config.output_modes,
                config.tail_modes,
            ),
            "completion_bias": (
                config.sequence_length,
                config.tail_modes,
            ),
            "output_mean": (config.sequence_length, config.width),
            "output_vectors": (config.width, config.width),
            "fused_output_weight": (
                config.sequence_length,
                config.routing_width,
                config.width,
            ),
            "fused_output_bias": (config.sequence_length, config.width),
        }
        supplied = {
            "input_mean": input_mean,
            "input_vectors": input_vectors,
            "input_scale": input_scale,
            "coordinate_kernel": coordinate_kernel,
            "input_kernel": input_kernel,
            "hidden_bias": hidden_bias,
            "output_weight": output_weight,
            "output_bias": output_bias,
            "output_scale": output_scale,
            "completion_weight": completion_weight,
            "completion_bias": completion_bias,
            "output_mean": output_mean,
            "output_vectors": output_vectors,
            "fused_output_weight": fused_output_weight,
            "fused_output_bias": fused_output_bias,
        }
        for name, value in supplied.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
        dtype = input_mean.dtype
        device = input_mean.device
        for name, value in supplied.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, "
                    f"expected {expected[name]}"
                )
            if value.dtype != dtype or value.device != device:
                raise ValueError(
                    "all fused layer buffers must share a dtype and device"
                )
            self.register_buffer(name, _clone_buffer(value))
        if (self.input_scale <= 0).any() or (self.output_scale <= 0).any():
            raise ValueError("fused modal scales must be positive")
        _require_causal_zeros("coordinate_kernel", self.coordinate_kernel)
        _require_causal_zeros("input_kernel", self.input_kernel)
        if self.config.completion_kind == "shared_local_linear" and not all(
            torch.equal(self.completion_weight[0], weight)
            for weight in self.completion_weight[1:]
        ):
            raise ValueError(
                "shared completion weights must be identical at every position"
            )
        identity = torch.eye(
            config.width,
            dtype=self.output_vectors.dtype,
            device=self.output_vectors.device,
        )
        rtol, atol = _consistency_tolerances(
            self.output_vectors.dtype,
            config.width,
        )
        if not torch.allclose(
            self.output_vectors.T @ self.output_vectors,
            identity,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError("fused output modal vectors must be orthonormal")
        expected_input_kernel = _fold_input_kernel(
            self.input_vectors,
            self.coordinate_kernel,
            self.input_scale,
        )
        _require_consistent(
            "input_kernel",
            self.input_kernel,
            expected_input_kernel,
            width=config.width,
        )
        expected_output_weight, expected_output_bias = _fold_output_decoder(
            output_weight=self.output_weight,
            output_bias=self.output_bias,
            output_scale=self.output_scale,
            completion_weight=self.completion_weight,
            completion_bias=self.completion_bias,
            output_mean=self.output_mean,
            output_vectors=self.output_vectors,
            output_modes=config.output_modes,
        )
        _require_consistent(
            "fused_output_weight",
            self.fused_output_weight,
            expected_output_weight,
            width=config.width,
        )
        _require_consistent(
            "fused_output_bias",
            self.fused_output_bias,
            expected_output_bias,
            width=config.width,
        )

    @classmethod
    def from_executor(
        cls,
        executor: PositionConditionedCompletedModalGraphExecutor,
    ) -> FusedCompletedModalLayer:
        """Fold a completed nonlinear modal executor without source hashes."""

        base = executor.base_executor
        graph = base.graph
        completion = executor.output_completion
        if not isinstance(graph, CausalModalMLPGraph):
            raise TypeError("fusion requires a CausalModalMLPGraph")
        config = FusedModalLayerConfig(
            input_activation=base.input_projection.activation_name,
            output_activation=base.output_projection.activation_name,
            sequence_length=graph.sequence_length,
            width=base.input_projection.width,
            input_modes=graph.input_modes,
            routing_width=graph.hidden_modes,
            output_modes=graph.output_modes,
            completion_kind=completion.graph.graph_kind,
        )
        if base.output_projection.width != config.width:
            raise ValueError("input and output residual widths differ")

        reference = base.input_projection.position_mean
        tensors = list(executor.state_dict().values())
        if any(
            value.dtype != reference.dtype or value.device != reference.device
            for value in tensors
        ):
            raise ValueError(
                "fusion requires one floating dtype and device throughout"
            )

        coordinate_kernel = torch.zeros(
            config.sequence_length,
            config.sequence_length,
            config.input_modes,
            config.routing_width,
            dtype=reference.dtype,
            device=reference.device,
        )
        hidden_bias = torch.empty(
            config.sequence_length,
            config.routing_width,
            dtype=reference.dtype,
            device=reference.device,
        )
        output_weight = torch.empty(
            config.sequence_length,
            config.routing_width,
            config.output_modes,
            dtype=reference.dtype,
            device=reference.device,
        )
        output_bias = torch.empty(
            config.sequence_length,
            config.output_modes,
            dtype=reference.dtype,
            device=reference.device,
        )
        for position in range(config.sequence_length):
            input_layer = graph.input_layers[position]
            coordinate_kernel[position, : position + 1] = (
                input_layer.weight.detach()
                .reshape(
                    config.routing_width,
                    position + 1,
                    config.input_modes,
                )
                .permute(1, 2, 0)
            )
            hidden_bias[position] = input_layer.bias.detach()
            output_layer = graph.output_layers[position]
            output_weight[position] = output_layer.weight.detach().T
            output_bias[position] = output_layer.bias.detach()

        input_scale = graph.input_scale.detach()
        input_kernel = _fold_input_kernel(
            base.input_projection.vectors.detach(),
            coordinate_kernel,
            input_scale,
        )

        completion_weight = completion.graph.weight.detach()
        if completion.graph.shared_weights:
            completion_weight = completion_weight.unsqueeze(0).expand(
                config.sequence_length,
                -1,
                -1,
            )
        completion_weight = completion_weight.clone()
        completion_bias = completion.graph.bias.detach()
        vectors = completion.full_projection.vectors.detach()
        output_mean = completion.full_projection.position_mean.detach()
        fused_output_weight, fused_output_bias = _fold_output_decoder(
            output_weight=output_weight,
            output_bias=output_bias,
            output_scale=graph.output_scale.detach(),
            completion_weight=completion_weight,
            completion_bias=completion_bias,
            output_mean=output_mean,
            output_vectors=vectors,
            output_modes=config.output_modes,
        )
        return cls(
            config,
            input_mean=base.input_projection.position_mean.detach(),
            input_vectors=base.input_projection.vectors.detach(),
            input_scale=input_scale,
            coordinate_kernel=coordinate_kernel,
            input_kernel=input_kernel,
            hidden_bias=hidden_bias,
            output_weight=output_weight,
            output_bias=output_bias,
            output_scale=graph.output_scale.detach(),
            completion_weight=completion_weight,
            completion_bias=completion_bias,
            output_mean=output_mean,
            output_vectors=vectors,
            fused_output_weight=fused_output_weight,
            fused_output_bias=fused_output_bias,
        )

    @property
    def sequence_length(self) -> int:
        return self.config.sequence_length

    @property
    def width(self) -> int:
        return self.config.width

    def _validate_input(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> None:
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (
            self.sequence_length,
            self.width,
        ):
            raise ValueError(
                "fused modal input must have shape "
                "[batch, fixed_sequence, width]"
            )
        if (
            hidden_states.dtype != self.input_mean.dtype
            or hidden_states.device != self.input_mean.device
        ):
            raise ValueError(
                "fused modal input must match the runtime dtype and device"
            )
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            if not attention_mask.to(torch.bool).all():
                raise ValueError(
                    "the fused fixed-position runtime does not support padding"
                )

    def compute_hidden_fast(self, hidden_states: Tensor) -> Tensor:
        """Execute explicit centering, the folded input kernel, and GELU."""

        centered = hidden_states - self.input_mean
        preactivation = torch.einsum(
            "bsw,tswh->bth",
            centered,
            self.input_kernel,
        ) + self.hidden_bias
        return F.gelu(preactivation)

    def decode_hidden_fast(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (
            self.sequence_length,
            self.config.routing_width,
        ):
            raise ValueError("fused hidden activation has the wrong shape")
        if (
            hidden.dtype != self.fused_output_weight.dtype
            or hidden.device != self.fused_output_weight.device
        ):
            raise ValueError(
                "fused hidden activation must match the runtime dtype "
                "and device"
            )
        return torch.einsum(
            "bsh,shw->bsw",
            hidden,
            self.fused_output_weight,
        ) + self.fused_output_bias

    def forward_fast(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(hidden_states, attention_mask)
        return self.decode_hidden_fast(
            self.compute_hidden_fast(hidden_states)
        )

    def _forward_traced(
        self,
        hidden_states: Tensor,
        *,
        trace: ActivationTrace,
        prefix: str,
    ) -> Tensor:
        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        input_modes = record(
            trace,
            f"{prefix}.modal.input",
            (hidden_states - self.input_mean) @ self.input_vectors,
        )
        normalized = input_modes / self.input_scale
        hidden = torch.stack(
            [
                F.gelu(
                    F.linear(
                        normalized[:, : position + 1].flatten(
                            start_dim=1
                        ),
                        self.coordinate_kernel[
                            position, : position + 1
                        ]
                        .permute(2, 0, 1)
                        .reshape(self.config.routing_width, -1)
                        .contiguous(),
                        self.hidden_bias[position],
                    )
                )
                for position in range(self.sequence_length)
            ],
            dim=1,
        )
        hidden = record(trace, f"{prefix}.modal.hidden", hidden)
        standardized_output = torch.stack(
            [
                F.linear(
                    hidden[:, position],
                    self.output_weight[position].T.contiguous(),
                    self.output_bias[position],
                )
                for position in range(self.sequence_length)
            ],
            dim=1,
        )
        output_modes = record(
            trace,
            f"{prefix}.modal.output",
            standardized_output * self.output_scale,
        )
        if self.config.completion_kind == "shared_local_linear":
            predicted_tail = (
                output_modes @ self.completion_weight[0]
                + self.completion_bias
            )
        else:
            predicted_tail = (
                torch.einsum(
                    "bso,sot->bst",
                    output_modes,
                    self.completion_weight,
                )
                + self.completion_bias
            )
        tail = record(
            trace,
            f"{prefix}.modal.output_completion.tail",
            predicted_tail,
        )
        coordinates = record(
            trace,
            f"{prefix}.modal.output_completion.coordinates",
            torch.cat((output_modes, tail), dim=-1),
        )
        output = (
            coordinates @ self.output_vectors.T + self.output_mean
        )
        return record(trace, f"{prefix}.output", output)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        self._validate_input(hidden_states, attention_mask)
        if trace is None:
            return self.forward_fast(
                hidden_states,
                attention_mask=attention_mask,
            )
        return self._forward_traced(
            hidden_states,
            trace=trace,
            prefix=prefix,
        )


def _boundary_is_exact(
    first: FusedCompletedModalLayer,
    second: FusedCompletedModalLayer,
) -> bool:
    """Return whether decode/re-encode can be bypassed exactly by coordinates."""

    if (
        first.config.output_activation != second.config.input_activation
        or first.sequence_length != second.sequence_length
        or first.width != second.width
        or second.config.input_modes > first.width
    ):
        return False
    expected_vectors = first.output_vectors[
        :, : second.config.input_modes
    ]
    return bool(
        torch.equal(first.output_mean, second.input_mean)
        and torch.equal(expected_vectors, second.input_vectors)
    )


def _build_bridge(
    first: FusedCompletedModalLayer,
    second: FusedCompletedModalLayer,
) -> tuple[Tensor, Tensor]:
    """Fold the exact shared modal boundary into a hidden-to-hidden bridge."""

    if not _boundary_is_exact(first, second):
        raise ValueError("cross-layer modal boundary is not exactly compatible")
    first_modes = first.config.output_modes
    second_modes = second.config.input_modes
    coordinate_map = torch.zeros(
        first.sequence_length,
        first_modes,
        first.width,
        dtype=first.input_mean.dtype,
        device=first.input_mean.device,
    )
    identity = torch.eye(
        first_modes,
        dtype=coordinate_map.dtype,
        device=coordinate_map.device,
    )
    coordinate_map[:, :, :first_modes] = identity
    coordinate_map[:, :, first_modes:] = first.completion_weight
    coordinate_bias = torch.zeros(
        first.sequence_length,
        first.width,
        dtype=coordinate_map.dtype,
        device=coordinate_map.device,
    )
    coordinate_bias[:, first_modes:] = first.completion_bias

    first_kept_weight = (
        first.output_weight * first.output_scale.unsqueeze(1)
    )
    first_kept_bias = first.output_bias * first.output_scale
    to_second_modes = coordinate_map[:, :, :second_modes]
    second_coordinates_weight = torch.einsum(
        "shk,skq->shq",
        first_kept_weight,
        to_second_modes,
    )
    second_coordinates_bias = (
        torch.einsum(
            "sk,skq->sq",
            first_kept_bias,
            to_second_modes,
        )
        + coordinate_bias[:, :second_modes]
    )
    normalized_second_kernel = (
        second.coordinate_kernel
        / second.input_scale.view(
            1,
            second.sequence_length,
            second.config.input_modes,
            1,
        )
    )
    bridge_kernel = torch.einsum(
        "shq,tsqr->tshr",
        second_coordinates_weight,
        normalized_second_kernel,
    )
    bridge_bias = second.hidden_bias + torch.einsum(
        "sq,tsqr->tr",
        second_coordinates_bias,
        normalized_second_kernel,
    )
    return bridge_kernel, bridge_bias


class FusedTwoLayerModalStack(nn.Module):
    """Two frozen fused layers with an optional exact modal bridge bypass."""

    def __init__(
        self,
        first: FusedCompletedModalLayer,
        second: FusedCompletedModalLayer,
        *,
        require_cross_layer_bypass: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(first, FusedCompletedModalLayer) or not isinstance(
            second,
            FusedCompletedModalLayer,
        ):
            raise TypeError("fused stack layers must be fused modal layers")
        if type(require_cross_layer_bypass) is not bool:
            raise ValueError("require_cross_layer_bypass must be boolean")
        if (
            first.input_mean.dtype != second.input_mean.dtype
            or first.input_mean.device != second.input_mean.device
        ):
            raise ValueError("fused stack layers must share dtype and device")
        compatible = _boundary_is_exact(first, second)
        if require_cross_layer_bypass and not compatible:
            raise ValueError(
                "the requested cross-layer bypass is not exactly compatible"
            )
        self.first = first
        self.second = second
        self.config = FusedTwoLayerStackConfig(
            first=first.config,
            second=second.config,
            cross_layer_bypass=compatible,
        )
        if compatible:
            bridge_kernel, bridge_bias = _build_bridge(first, second)
        else:
            bridge_kernel = torch.empty(
                0,
                dtype=first.input_mean.dtype,
                device=first.input_mean.device,
            )
            bridge_bias = torch.empty(
                0,
                dtype=first.input_mean.dtype,
                device=first.input_mean.device,
            )
        self.register_buffer("bridge_kernel", _clone_buffer(bridge_kernel))
        self.register_buffer("bridge_bias", _clone_buffer(bridge_bias))
        if compatible:
            _require_causal_zeros("bridge_kernel", self.bridge_kernel)

    @classmethod
    def from_executors(
        cls,
        first: PositionConditionedCompletedModalGraphExecutor,
        second: PositionConditionedCompletedModalGraphExecutor,
        *,
        require_cross_layer_bypass: bool = False,
    ) -> FusedTwoLayerModalStack:
        return cls(
            FusedCompletedModalLayer.from_executor(first),
            FusedCompletedModalLayer.from_executor(second),
            require_cross_layer_bypass=require_cross_layer_bypass,
        )

    @property
    def uses_cross_layer_bypass(self) -> bool:
        return self.config.cross_layer_bypass

    @property
    def sequence_length(self) -> int:
        return self.config.first.sequence_length

    @property
    def width(self) -> int:
        return self.config.first.width

    @property
    def reference_tensor(self) -> Tensor:
        return self.first.input_mean

    def forward_fast(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        self.first._validate_input(hidden_states, attention_mask)
        if not self.uses_cross_layer_bypass:
            return self.second.forward_fast(
                self.first.forward_fast(
                    hidden_states,
                    attention_mask=attention_mask,
                ),
                attention_mask=attention_mask,
            )
        first_hidden = self.first.compute_hidden_fast(hidden_states)
        second_hidden = F.gelu(
            torch.einsum(
                "bsh,tshr->btr",
                first_hidden,
                self.bridge_kernel,
            )
            + self.bridge_bias
        )
        return self.second.decode_hidden_fast(second_hidden)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefixes: Sequence[str] = ("layer.0", "layer.1"),
    ) -> Tensor:
        if len(prefixes) != 2 or not all(
            isinstance(prefix, str) and prefix for prefix in prefixes
        ):
            raise ValueError("fused stack requires exactly two prefixes")
        if trace is None:
            return self.forward_fast(
                hidden_states,
                attention_mask=attention_mask,
            )
        hidden_states = self.first(
            hidden_states,
            attention_mask=attention_mask,
            trace=trace,
            prefix=prefixes[0],
        )
        return self.second(
            hidden_states,
            attention_mask=attention_mask,
            trace=trace,
            prefix=prefixes[1],
        )


def _portable_metadata(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _portable_metadata(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _portable_metadata(item, path=f"{path}[]")
            for item in value
        ]
    raise TypeError(f"{path} contains a non-portable value: {type(value)}")


def save_fused_modal_stack(
    path: str | Path,
    *,
    stack: FusedTwoLayerModalStack,
    metadata: Mapping[str, object],
) -> None:
    """Save a weights-only, source-hash-agnostic fused stack artifact."""

    if not isinstance(stack, FusedTwoLayerModalStack):
        raise TypeError("stack must be a FusedTwoLayerModalStack")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    torch.save(
        {
            "format_version": 1,
            "artifact_kind": "fused_two_layer_modal_stack",
            "config": asdict(stack.config),
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in stack.state_dict().items()
            },
            "metadata": _portable_metadata(metadata),
        },
        Path(path),
    )


def _require_exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing " + ", ".join(sorted(repr(item) for item in missing))
            )
        if extra:
            details.append(
                "unexpected "
                + ", ".join(sorted(repr(item) for item in extra))
            )
        raise ValueError(f"{label} has invalid keys: {'; '.join(details)}")


_LAYER_CONFIG_KEYS = {
    "input_activation",
    "output_activation",
    "sequence_length",
    "width",
    "input_modes",
    "routing_width",
    "output_modes",
    "completion_kind",
}


def _layer_config_from_dict(value: Mapping[str, object]) -> FusedModalLayerConfig:
    if not isinstance(value, Mapping):
        raise ValueError("fused modal layer config must be an object")
    _require_exact_keys(
        value,
        _LAYER_CONFIG_KEYS,
        label="fused modal layer config",
    )
    return FusedModalLayerConfig(**dict(value))  # type: ignore[arg-type]


def load_fused_modal_stack(
    path: str | Path,
) -> tuple[FusedTwoLayerModalStack, FusedTwoLayerStackConfig, dict[str, object]]:
    """Load an artifact written by :func:`save_fused_modal_stack`."""

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("fused modal stack artifact must be an object")
    _require_exact_keys(
        state,
        {
            "format_version",
            "artifact_kind",
            "config",
            "state_dict",
            "metadata",
        },
        label="fused modal stack artifact",
    )
    if type(state.get("format_version")) is not int or state.get(
        "format_version"
    ) != 1:
        raise ValueError("unsupported fused modal stack artifact format")
    if state.get("artifact_kind") != "fused_two_layer_modal_stack":
        raise ValueError("unsupported fused modal stack artifact kind")
    raw_config = state.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("fused modal stack config must be an object")
    _require_exact_keys(
        raw_config,
        {"first", "second", "cross_layer_bypass"},
        label="fused modal stack config",
    )
    raw_first = raw_config["first"]
    raw_second = raw_config["second"]
    if not isinstance(raw_first, Mapping) or not isinstance(
        raw_second,
        Mapping,
    ):
        raise ValueError("fused modal stack layer configs must be objects")
    raw_bypass = raw_config["cross_layer_bypass"]
    if type(raw_bypass) is not bool:
        raise ValueError("cross_layer_bypass must be boolean")
    first_config = _layer_config_from_dict(raw_first)
    second_config = _layer_config_from_dict(raw_second)
    config = FusedTwoLayerStackConfig(
        first=first_config,
        second=second_config,
        cross_layer_bypass=raw_bypass,
    )
    saved = state.get("state_dict")
    if not isinstance(saved, Mapping):
        raise ValueError("fused modal stack state must be an object")
    expected_state_keys = {"bridge_kernel", "bridge_bias"} | {
        f"{prefix}.{name}"
        for prefix in ("first", "second")
        for name in _LAYER_BUFFER_NAMES
    }
    _require_exact_keys(
        saved,
        expected_state_keys,
        label="fused modal stack state",
    )

    def build_layer(
        prefix: str,
        layer_config: FusedModalLayerConfig,
    ) -> FusedCompletedModalLayer:
        tensors: dict[str, Tensor] = {}
        for name in _LAYER_BUFFER_NAMES:
            key = f"{prefix}.{name}"
            value = saved.get(key)
            if not isinstance(value, Tensor):
                raise ValueError(f"missing fused modal stack tensor: {key}")
            tensors[name] = value
        return FusedCompletedModalLayer(layer_config, **tensors)

    first = build_layer("first", first_config)
    second = build_layer("second", second_config)
    stack = FusedTwoLayerModalStack(
        first,
        second,
        require_cross_layer_bypass=config.cross_layer_bypass,
    )
    if stack.config != config:
        raise ValueError("fused modal stack state does not match its config")
    for name in ("bridge_kernel", "bridge_bias"):
        value = saved[name]
        expected_value = getattr(stack, name)
        if not isinstance(value, Tensor):
            raise ValueError(f"missing fused modal stack tensor: {name}")
        if (
            value.shape != expected_value.shape
            or value.dtype != expected_value.dtype
            or value.device != expected_value.device
        ):
            raise ValueError(
                f"{name} does not match the derived fused bridge contract"
            )
        _require_consistent(
            name,
            value,
            expected_value,
            width=config.first.width,
        )
    stack.load_state_dict(dict(saved), strict=True)
    if stack.uses_cross_layer_bypass:
        _require_causal_zeros("bridge_kernel", stack.bridge_kernel)
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("fused modal stack metadata must be an object")
    try:
        portable_metadata = _portable_metadata(metadata)
    except TypeError as error:
        raise ValueError("fused modal stack metadata is not portable") from error
    assert isinstance(portable_metadata, dict)
    return stack, config, portable_metadata


def _tensor_state_bytes(module: nn.Module) -> int:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _load_modal_executor_bytes(
    data: bytes,
) -> tuple[
    PositionConditionedModalGraphExecutor,
    ModalExecutorConfig,
    dict[str, object],
]:
    state = torch.load(
        io.BytesIO(data),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise ValueError("lazy modal executor sidecar must be an object")
    _require_exact_keys(
        state,
        {
            "format_version",
            "graph_kind",
            "config",
            "executor_state_dict",
            "metadata",
        },
        label="lazy modal executor sidecar",
    )
    if (
        type(state.get("format_version")) is not int
        or state.get("format_version") != 1
    ):
        raise ValueError("unsupported lazy modal executor sidecar format")
    if state.get("graph_kind") != "causal_position_modal_mlp":
        raise ValueError("unsupported lazy modal executor sidecar graph kind")
    raw_config = state.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("lazy modal executor config must be an object")
    _require_exact_keys(
        raw_config,
        _MODAL_EXECUTOR_CONFIG_KEYS,
        label="lazy modal executor config",
    )
    if not isinstance(raw_config["input_activation"], str) or not isinstance(
        raw_config["output_activation"],
        str,
    ):
        raise ValueError(
            "lazy modal executor activation names must be strings"
        )
    if any(
        type(raw_config[name]) is not int
        for name in (
            "sequence_length",
            "input_modes",
            "output_modes",
            "routing_width",
        )
    ):
        raise ValueError(
            "lazy modal executor dimensions must be integers"
        )
    config = ModalExecutorConfig(**dict(raw_config))  # type: ignore[arg-type]
    executor_state = state.get("executor_state_dict")
    if not isinstance(executor_state, Mapping):
        raise ValueError("lazy modal executor state must be an object")
    if any(
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        for value in executor_state.values()
    ):
        raise ValueError(
            "lazy modal executor state tensors must be finite floating point"
        )
    try:
        input_projection = PositionConditionedModalProjection(
            activation_name=config.input_activation,
            position_mean=executor_state[
                "input_projection.position_mean"
            ],
            vectors=executor_state["input_projection.vectors"],
        )
        graph = CausalModalMLPGraph(
            input_modes=config.input_modes,
            output_modes=config.output_modes,
            sequence_length=config.sequence_length,
            hidden_modes=config.routing_width,
            input_scale=executor_state["graph.input_scale"],
            output_scale=executor_state["graph.output_scale"],
        )
        parameter_dtype = executor_state[
            "graph.input_layers.0.weight"
        ].dtype
        graph = graph.to(dtype=parameter_dtype)
        output_projection = PositionConditionedModalProjection(
            activation_name=config.output_activation,
            position_mean=executor_state[
                "output_projection.position_mean"
            ],
            vectors=executor_state["output_projection.vectors"],
        )
    except KeyError as error:
        raise ValueError(
            f"lazy modal executor state is missing {error.args[0]!r}"
        ) from error
    executor = PositionConditionedModalGraphExecutor(
        input_projection,
        graph,
        output_projection,
    )
    try:
        executor.load_state_dict(dict(executor_state), strict=True)
    except RuntimeError as error:
        raise ValueError(
            "lazy modal executor state does not match its config"
        ) from error
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("lazy modal executor metadata must be an object")
    return executor, config, dict(metadata)


def _load_modal_completion_bytes(
    data: bytes,
) -> tuple[
    PositionConditionedModalCompletion,
    ModalCompletionConfig,
    dict[str, object],
]:
    state = torch.load(
        io.BytesIO(data),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise ValueError("lazy modal completion sidecar must be an object")
    _require_exact_keys(
        state,
        {
            "format_version",
            "artifact_kind",
            "config",
            "completion_state_dict",
            "metadata",
        },
        label="lazy modal completion sidecar",
    )
    if (
        type(state.get("format_version")) is not int
        or state.get("format_version") != 1
    ):
        raise ValueError("unsupported lazy modal completion sidecar format")
    if state.get("artifact_kind") != "position_conditioned_modal_completion":
        raise ValueError("unsupported lazy modal completion sidecar kind")
    raw_config = state.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("lazy modal completion config must be an object")
    _require_exact_keys(
        raw_config,
        _MODAL_COMPLETION_CONFIG_KEYS,
        label="lazy modal completion config",
    )
    if not isinstance(raw_config["activation_name"], str) or not isinstance(
        raw_config["graph_kind"],
        str,
    ):
        raise ValueError(
            "lazy modal completion names must be strings"
        )
    if any(
        type(raw_config[name]) is not int
        for name in ("sequence_length", "width", "kept_modes")
    ):
        raise ValueError(
            "lazy modal completion dimensions must be integers"
        )
    config = ModalCompletionConfig(**dict(raw_config))  # type: ignore[arg-type]
    completion_state = state.get("completion_state_dict")
    if not isinstance(completion_state, Mapping):
        raise ValueError("lazy modal completion state must be an object")
    if any(
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        for value in completion_state.values()
    ):
        raise ValueError(
            "lazy modal completion tensors must be finite floating point"
        )
    try:
        projection = PositionConditionedModalProjection(
            activation_name=config.activation_name,
            position_mean=completion_state[
                "full_projection.position_mean"
            ],
            vectors=completion_state["full_projection.vectors"],
        )
        graph = LocalModalCompletionGraph(
            completion_state["graph.weight"],
            completion_state["graph.bias"],
            shared_weights=(
                config.graph_kind == "shared_local_linear"
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"lazy modal completion state is missing {error.args[0]!r}"
        ) from error
    completion = PositionConditionedModalCompletion(projection, graph)
    try:
        completion.load_state_dict(dict(completion_state), strict=True)
    except RuntimeError as error:
        raise ValueError(
            "lazy modal completion state does not match its config"
        ) from error
    if completion.config != config:
        raise ValueError(
            "lazy modal completion state does not match its config"
        )
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("lazy modal completion metadata must be an object")
    return completion, config, dict(metadata)


class LazyFusedTwoLayerModalStack(nn.Module):
    """Fast fused stack with logical instrumentation loaded on first use."""

    def __init__(
        self,
        config: FusedTwoLayerStackConfig,
        *,
        first_input_mean: Tensor,
        first_input_kernel: Tensor,
        first_hidden_bias: Tensor,
        bridge_kernel: Tensor,
        bridge_bias: Tensor,
        second_fused_output_weight: Tensor,
        second_fused_output_bias: Tensor,
        sidecars: Mapping[str, _SidecarArtifact],
        sidecar_root: str | Path,
        provenance: Mapping[str, object],
    ) -> None:
        super().__init__()
        if not isinstance(config, FusedTwoLayerStackConfig):
            raise TypeError("config must be a FusedTwoLayerStackConfig")
        if not config.cross_layer_bypass:
            raise ValueError(
                "the compact lazy runtime requires an exact cross-layer bypass"
            )
        if not isinstance(sidecars, Mapping):
            raise TypeError("sidecars must be a mapping")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        if set(sidecars) != set(_LAZY_SIDECAR_NAMES):
            raise ValueError("lazy runtime sidecar names mismatch")
        if any(
            not isinstance(descriptor, _SidecarArtifact)
            for descriptor in sidecars.values()
        ):
            raise TypeError("lazy runtime sidecars must use sidecar descriptors")
        supplied = {
            "first_input_mean": first_input_mean,
            "first_input_kernel": first_input_kernel,
            "first_hidden_bias": first_hidden_bias,
            "bridge_kernel": bridge_kernel,
            "bridge_bias": bridge_bias,
            "second_fused_output_weight": second_fused_output_weight,
            "second_fused_output_bias": second_fused_output_bias,
        }
        expected = {
            "first_input_mean": (
                config.first.sequence_length,
                config.first.width,
            ),
            "first_input_kernel": (
                config.first.sequence_length,
                config.first.sequence_length,
                config.first.width,
                config.first.routing_width,
            ),
            "first_hidden_bias": (
                config.first.sequence_length,
                config.first.routing_width,
            ),
            "bridge_kernel": (
                config.first.sequence_length,
                config.first.sequence_length,
                config.first.routing_width,
                config.second.routing_width,
            ),
            "bridge_bias": (
                config.second.sequence_length,
                config.second.routing_width,
            ),
            "second_fused_output_weight": (
                config.second.sequence_length,
                config.second.routing_width,
                config.second.width,
            ),
            "second_fused_output_bias": (
                config.second.sequence_length,
                config.second.width,
            ),
        }
        reference = first_input_mean
        if not isinstance(reference, Tensor):
            raise TypeError("lazy runtime fast state must contain tensors")
        for name, value in supplied.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if tuple(value.shape) != expected[name]:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, "
                    f"expected {expected[name]}"
                )
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    "all lazy runtime fast tensors must share dtype and device"
                )
            self.register_buffer(name, _clone_buffer(value))
        _require_causal_zeros(
            "first_input_kernel",
            self.first_input_kernel,
        )
        _require_causal_zeros("bridge_kernel", self.bridge_kernel)
        self.config = config
        self.sidecar_root = Path(sidecar_root)
        self.sidecars = dict(sidecars)
        portable = _portable_metadata(provenance, path="provenance")
        assert isinstance(portable, dict)
        _require_lazy_provenance(portable)
        self.provenance = portable
        self._instrumentation_lock = threading.RLock()
        self._logical_layers: (
            tuple[
                PositionConditionedCompletedModalGraphExecutor,
                PositionConditionedCompletedModalGraphExecutor,
            ]
            | None
        ) = None
        self._residency = "unloaded"
        self._last_dispatch: str | None = None
        self._fast_path_calls = 0
        self._instrumented_path_calls = 0
        self._load_attempts = 0
        self._successful_loads = 0
        self._cache_hits = 0
        self._failed_loads = 0
        self._evictions = 0
        self._derived_kernel_verifications = 0
        self._resident_sidecar_tensor_bytes = 0
        self._sidecar_file_bytes_read = 0
        self._last_error: str | None = None

    @classmethod
    def from_monolithic(
        cls,
        stack: FusedTwoLayerModalStack,
        *,
        sidecars: Mapping[str, _SidecarArtifact],
        sidecar_root: str | Path,
        provenance: Mapping[str, object],
    ) -> LazyFusedTwoLayerModalStack:
        if not isinstance(stack, FusedTwoLayerModalStack):
            raise TypeError("stack must be a FusedTwoLayerModalStack")
        if not stack.uses_cross_layer_bypass:
            raise ValueError(
                "the compact lazy runtime requires an exact cross-layer bypass"
            )
        return cls(
            stack.config,
            first_input_mean=stack.first.input_mean,
            first_input_kernel=stack.first.input_kernel,
            first_hidden_bias=stack.first.hidden_bias,
            bridge_kernel=stack.bridge_kernel,
            bridge_bias=stack.bridge_bias,
            second_fused_output_weight=(
                stack.second.fused_output_weight
            ),
            second_fused_output_bias=stack.second.fused_output_bias,
            sidecars=sidecars,
            sidecar_root=sidecar_root,
            provenance=provenance,
        )

    @property
    def uses_cross_layer_bypass(self) -> bool:
        return True

    @property
    def sequence_length(self) -> int:
        return self.config.first.sequence_length

    @property
    def width(self) -> int:
        return self.config.first.width

    @property
    def reference_tensor(self) -> Tensor:
        return self.first_input_mean

    @property
    def instrumentation_loaded(self) -> bool:
        with self._instrumentation_lock:
            return self._logical_layers is not None

    @property
    def fast_state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.state_dict().values()
        )

    def instrumentation_status(self) -> LazyInstrumentationStatus:
        with self._instrumentation_lock:
            return LazyInstrumentationStatus(
                residency=self._residency,
                loaded=self._logical_layers is not None,
                last_dispatch=self._last_dispatch,
                fast_path_calls=self._fast_path_calls,
                instrumented_path_calls=self._instrumented_path_calls,
                load_attempts=self._load_attempts,
                successful_loads=self._successful_loads,
                cache_hits=self._cache_hits,
                failed_loads=self._failed_loads,
                evictions=self._evictions,
                derived_kernel_verifications=(
                    self._derived_kernel_verifications
                ),
                resident_fast_tensor_bytes=self.fast_state_bytes,
                resident_sidecar_tensor_bytes=(
                    self._resident_sidecar_tensor_bytes
                ),
                sidecar_file_bytes_read=self._sidecar_file_bytes_read,
                last_error=self._last_error,
            )

    def _validate_input(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("lazy fused input must be a Tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (
            self.sequence_length,
            self.width,
        ):
            raise ValueError(
                "lazy fused input must have shape "
                "[batch, fixed_sequence, width]"
            )
        if (
            hidden_states.dtype != self.first_input_mean.dtype
            or hidden_states.device != self.first_input_mean.device
        ):
            raise ValueError(
                "lazy fused input must match the runtime dtype and device"
            )
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor):
                raise TypeError("attention_mask must be a Tensor")
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            if not attention_mask.to(torch.bool).all():
                raise ValueError(
                    "the lazy fixed-position runtime does not support padding"
                )

    def forward_fast(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Module-wide dtype/device changes also take this lock. Keeping the
        # seven-buffer contraction inside it prevents a concurrent ``to()``
        # from exposing a mixture of old and new runtime state.
        with self._instrumentation_lock:
            self._validate_input(hidden_states, attention_mask)
            centered = hidden_states - self.first_input_mean
            first_hidden = F.gelu(
                torch.einsum(
                    "bsw,tswh->bth",
                    centered,
                    self.first_input_kernel,
                )
                + self.first_hidden_bias
            )
            second_hidden = F.gelu(
                torch.einsum(
                    "bsh,tshr->btr",
                    first_hidden,
                    self.bridge_kernel,
                )
                + self.bridge_bias
            )
            output = (
                torch.einsum(
                    "bsh,shw->bsw",
                    second_hidden,
                    self.second_fused_output_weight,
                )
                + self.second_fused_output_bias
            )
            self._fast_path_calls += 1
            self._last_dispatch = "fast_cross_layer"
            return output

    def _read_verified_sidecars(self) -> dict[str, bytes]:
        payloads: dict[str, bytes] = {}
        for name in _LAZY_SIDECAR_NAMES:
            descriptor = self.sidecars[name]
            path = self.sidecar_root / descriptor.filename
            if path.is_symlink():
                raise ValueError(
                    f"lazy instrumentation sidecar cannot be a symlink: {path}"
                )
            if not path.is_file():
                raise FileNotFoundError(
                    f"lazy instrumentation sidecar is missing: {path}"
                )
            data = path.read_bytes()
            self._sidecar_file_bytes_read += len(data)
            if len(data) != descriptor.size_bytes:
                raise ValueError(
                    f"lazy instrumentation sidecar size mismatch: {name}"
                )
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != descriptor.sha256:
                raise ValueError(
                    f"lazy instrumentation sidecar hash mismatch: {name}"
                )
            payloads[name] = data
        return payloads

    def _validate_provenance(
        self,
        *,
        layer_index: int,
        executor_metadata: Mapping[str, object],
        completion_metadata: Mapping[str, object],
        executor_hash: str,
    ) -> None:
        expected = {
            "checkpoint_sha256": self.provenance.get(
                "checkpoint_sha256"
            ),
            "fisher_sha256": self.provenance.get("fisher_sha256"),
            "teacher_state_sha256": self.provenance.get(
                "teacher_state_sha256"
            ),
            "layer_index": layer_index,
        }
        for label, metadata in (
            ("executor", executor_metadata),
            ("completion", completion_metadata),
        ):
            for name, value in expected.items():
                actual = metadata.get(name)
                if value is not None and (
                    type(actual) is not type(value) or actual != value
                ):
                    raise ValueError(
                        f"lazy layer {layer_index} {label} {name} mismatch"
                    )
        if completion_metadata.get("modal_executor_sha256") != executor_hash:
            raise ValueError(
                f"lazy layer {layer_index} completion executor hash mismatch"
            )
        if completion_metadata.get("boundary_role") != "output":
            raise ValueError(
                f"lazy layer {layer_index} completion boundary mismatch"
            )

    def _build_logical_layers(
        self,
        payloads: Mapping[str, bytes],
    ) -> tuple[
        PositionConditionedCompletedModalGraphExecutor,
        PositionConditionedCompletedModalGraphExecutor,
    ]:
        completed: list[
            PositionConditionedCompletedModalGraphExecutor
        ] = []
        for layer_index in (0, 1):
            executor_name = f"layer_{layer_index}_executor"
            completion_name = (
                f"layer_{layer_index}_output_completion"
            )
            executor, executor_config, executor_metadata = (
                _load_modal_executor_bytes(payloads[executor_name])
            )
            completion, completion_config, completion_metadata = (
                _load_modal_completion_bytes(payloads[completion_name])
            )
            expected_config = (
                self.config.first
                if layer_index == 0
                else self.config.second
            )
            if (
                executor_config.input_activation
                != expected_config.input_activation
                or executor_config.output_activation
                != expected_config.output_activation
                or executor_config.sequence_length
                != expected_config.sequence_length
                or executor_config.input_modes
                != expected_config.input_modes
                or executor_config.output_modes
                != expected_config.output_modes
                or executor_config.routing_width
                != expected_config.routing_width
            ):
                raise ValueError(
                    f"lazy layer {layer_index} executor config mismatch"
                )
            if (
                completion_config.activation_name
                != expected_config.output_activation
                or completion_config.sequence_length
                != expected_config.sequence_length
                or completion_config.width != expected_config.width
                or completion_config.kept_modes
                != expected_config.output_modes
                or completion_config.graph_kind
                != expected_config.completion_kind
            ):
                raise ValueError(
                    f"lazy layer {layer_index} completion config mismatch"
                )
            self._validate_provenance(
                layer_index=layer_index,
                executor_metadata=executor_metadata,
                completion_metadata=completion_metadata,
                executor_hash=self.sidecars[executor_name].sha256,
            )
            layer = PositionConditionedCompletedModalGraphExecutor(
                executor,
                completion,
            )
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            completed.append(layer)
        layers = (completed[0], completed[1])
        # Reconstruct the derivation in the sidecars' canonical saved dtype.
        # Promoting an f32 orthonormal basis to f64 does not make it accurate
        # to f64 epsilon, and re-folding after promotion can round differently
        # from promoting the already-folded fast tensors.
        witness = FusedTwoLayerModalStack.from_executors(
            layers[0],
            layers[1],
            require_cross_layer_bypass=True,
        )
        if witness.config != self.config:
            raise ValueError("lazy sidecars do not match the fused config")
        checks = {
            "first_input_mean": witness.first.input_mean,
            "first_input_kernel": witness.first.input_kernel,
            "first_hidden_bias": witness.first.hidden_bias,
            "bridge_kernel": witness.bridge_kernel,
            "bridge_bias": witness.bridge_bias,
            "second_fused_output_weight": (
                witness.second.fused_output_weight
            ),
            "second_fused_output_bias": witness.second.fused_output_bias,
        }
        for name, expected in checks.items():
            actual = getattr(self, name)
            _require_consistent(
                name,
                actual,
                expected.to(
                    device=actual.device,
                    dtype=actual.dtype,
                ),
                width=self.width,
            )
        for layer in layers:
            layer.to(
                device=self.reference_tensor.device,
                dtype=self.reference_tensor.dtype,
            )
        return layers

    def _get_instrumentation_layers(
        self,
    ) -> tuple[
        PositionConditionedCompletedModalGraphExecutor,
        PositionConditionedCompletedModalGraphExecutor,
    ]:
        with self._instrumentation_lock:
            if self._logical_layers is not None:
                self._cache_hits += 1
                return self._logical_layers
            self._load_attempts += 1
            self._residency = "loading"
            self._last_error = None
            try:
                payloads = self._read_verified_sidecars()
                layers = self._build_logical_layers(payloads)
            except Exception as error:
                self._logical_layers = None
                self._resident_sidecar_tensor_bytes = 0
                self._failed_loads += 1
                self._residency = "failed"
                self._last_error = f"{type(error).__name__}: {error}"
                raise
            self._logical_layers = layers
            self._resident_sidecar_tensor_bytes = sum(
                _tensor_state_bytes(layer) for layer in layers
            )
            self._successful_loads += 1
            self._derived_kernel_verifications += 1
            self._residency = "loaded"
            return layers

    def load_instrumentation(self) -> LazyInstrumentationStatus:
        """Materialize and authenticate the logical executor sidecars.

        The cached modules deliberately remain private so callers cannot
        mutate them after validation. Instrumented ``forward`` calls use the
        same cache and expose activations through :class:`ActivationTrace`.
        """

        with self._instrumentation_lock:
            self._get_instrumentation_layers()
            return self.instrumentation_status()

    def evict_instrumentation(self) -> bool:
        with self._instrumentation_lock:
            if self._logical_layers is None:
                if self._residency == "failed":
                    self._residency = "unloaded"
                    self._last_error = None
                return False
            self._logical_layers = None
            self._resident_sidecar_tensor_bytes = 0
            self._evictions += 1
            self._residency = "unloaded"
            self._last_error = None
            return True

    def _apply(self, fn, recurse: bool = True):
        with self._instrumentation_lock:
            self.evict_instrumentation()
            return super()._apply(fn, recurse=recurse)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefixes: Sequence[str] = ("layer.0", "layer.1"),
    ) -> Tensor:
        if len(prefixes) != 2 or not all(
            isinstance(prefix, str) and prefix for prefix in prefixes
        ):
            raise ValueError("lazy fused stack requires exactly two prefixes")
        if trace is None:
            return self.forward_fast(
                hidden_states,
                attention_mask=attention_mask,
            )
        # Keep the cache resident and its dtype/device stable for the complete
        # logical replay. Traced calls are intentionally the uncommon path.
        with self._instrumentation_lock:
            self._validate_input(hidden_states, attention_mask)
            first, second = self._get_instrumentation_layers()
            self._instrumented_path_calls += 1
            self._last_dispatch = "logical_sidecar"
            hidden_states = first(
                hidden_states,
                attention_mask=attention_mask,
                trace=trace,
                prefix=prefixes[0],
            )
            return second(
                hidden_states,
                attention_mask=attention_mask,
                trace=trace,
                prefix=prefixes[1],
            )


class PackedTriangularFusedTwoLayerModalStack(nn.Module):
    """Non-instrumentable causal-pair packing of a lazy fused fast path.

    The authenticated lazy runtime remains the source of truth and the default
    instrumentable executor. This derived runtime copies its five position-
    local fast tensors and stores only the lower-triangular position pairs from
    its two causal kernels. It never retains or opens source sidecars.

    Packing is algebraically exact for finite inputs. Its gather/einsum/
    ``index_add`` reduction order differs from the dense source einsums, so
    floating-point results are intentionally not claimed to be bit-exact.
    """

    def __init__(
        self,
        source: LazyFusedTwoLayerModalStack,
    ) -> None:
        super().__init__()
        if not isinstance(source, LazyFusedTwoLayerModalStack):
            raise TypeError(
                "source must be a LazyFusedTwoLayerModalStack"
            )
        if not source.uses_cross_layer_bypass:
            raise ValueError(
                "packed triangular fusion requires the exact cross-layer "
                "bypass"
            )

        # ``LazyFusedTwoLayerModalStack._apply`` takes this same lock. Holding
        # it while cloning prevents a concurrent dtype/device conversion from
        # exposing a mixed source state, without loading instrumentation.
        with source._instrumentation_lock:
            _require_causal_zeros(
                "first_input_kernel",
                source.first_input_kernel,
            )
            _require_causal_zeros(
                "bridge_kernel",
                source.bridge_kernel,
            )
            target_indices, source_indices = torch.tril_indices(
                source.sequence_length,
                source.sequence_length,
                device=source.reference_tensor.device,
            )
            packed_first = source.first_input_kernel[
                target_indices,
                source_indices,
            ]
            packed_bridge = source.bridge_kernel[
                target_indices,
                source_indices,
            ]
            retained = {
                "first_input_mean": _clone_buffer(
                    source.first_input_mean
                ),
                "first_hidden_bias": _clone_buffer(
                    source.first_hidden_bias
                ),
                "bridge_bias": _clone_buffer(source.bridge_bias),
                "second_fused_output_weight": (
                    _clone_buffer(source.second_fused_output_weight)
                ),
                "second_fused_output_bias": (
                    _clone_buffer(source.second_fused_output_bias)
                ),
            }
            self.config = source.config
            self._source_fast_state_bytes = source.fast_state_bytes
            portable_provenance = _portable_metadata(
                source.provenance,
                path="packed source provenance",
            )
            assert isinstance(portable_provenance, dict)
            self._source_provenance = portable_provenance

        expected_pairs = (
            self.sequence_length * (self.sequence_length + 1) // 2
        )
        expected_first = (
            expected_pairs,
            self.width,
            self.config.first.routing_width,
        )
        expected_bridge = (
            expected_pairs,
            self.config.first.routing_width,
            self.config.second.routing_width,
        )
        if tuple(packed_first.shape) != expected_first:
            raise ValueError(
                "packed first input kernel has an invalid shape"
            )
        if tuple(packed_bridge.shape) != expected_bridge:
            raise ValueError("packed bridge kernel has an invalid shape")

        for name, value in retained.items():
            self.register_buffer(name, _clone_buffer(value))
        self.register_buffer(
            "packed_first_input_kernel",
            _clone_buffer(packed_first),
        )
        self.register_buffer(
            "packed_bridge_kernel",
            _clone_buffer(packed_bridge),
        )
        self.register_buffer(
            "causal_target_indices",
            target_indices.detach().clone(),
        )
        self.register_buffer(
            "causal_source_indices",
            source_indices.detach().clone(),
        )

    @classmethod
    def from_lazy(
        cls,
        source: LazyFusedTwoLayerModalStack,
    ) -> PackedTriangularFusedTwoLayerModalStack:
        """Derive a sidecar-free packed executor from lazy fast tensors."""

        return cls(source)

    @property
    def uses_cross_layer_bypass(self) -> bool:
        return True

    @property
    def sequence_length(self) -> int:
        return self.config.first.sequence_length

    @property
    def width(self) -> int:
        return self.config.first.width

    @property
    def reference_tensor(self) -> Tensor:
        return self.first_input_mean

    @property
    def dense_position_pair_count(self) -> int:
        return self.sequence_length * self.sequence_length

    @property
    def causal_pair_count(self) -> int:
        return self.causal_target_indices.numel()

    @property
    def eliminated_position_pair_count(self) -> int:
        return self.dense_position_pair_count - self.causal_pair_count

    @property
    def source_fast_state_bytes(self) -> int:
        return self._source_fast_state_bytes

    @property
    def packed_float_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in self.state_dict().values()
            if value.is_floating_point()
        )

    @property
    def packed_index_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in self.state_dict().values()
            if not value.is_floating_point()
        )

    @property
    def packed_state_bytes(self) -> int:
        return self.packed_float_state_bytes + self.packed_index_state_bytes

    @property
    def packed_fast_state_bytes(self) -> int:
        """Alias used by runtime storage reports."""

        return self.packed_state_bytes

    @property
    def fast_state_bytes(self) -> int:
        return self.packed_state_bytes

    def runtime_provenance(self) -> dict[str, object]:
        """Return portable derivation, arithmetic, and storage facts."""

        first_dense = (
            self.dense_position_pair_count
            * self.width
            * self.config.first.routing_width
        )
        bridge_dense = (
            self.dense_position_pair_count
            * self.config.first.routing_width
            * self.config.second.routing_width
        )
        output_local = (
            self.sequence_length
            * self.config.second.routing_width
            * self.width
        )
        first_packed = (
            self.causal_pair_count
            * self.width
            * self.config.first.routing_width
        )
        bridge_packed = (
            self.causal_pair_count
            * self.config.first.routing_width
            * self.config.second.routing_width
        )
        return {
            "runtime_kind": (
                "packed_triangular_fused_two_layer_modal_stack"
            ),
            "derived_from_runtime_kind": (
                "lazy_fused_two_layer_modal_stack"
            ),
            "derivation": "lower_triangular_position_pair_pack",
            "algebraically_exact_for_finite_inputs": True,
            "bit_exact_to_dense_source": False,
            "supports_activation_trace": False,
            "sequence_length": self.sequence_length,
            "dense_position_pair_count": self.dense_position_pair_count,
            "causal_pair_count": self.causal_pair_count,
            "eliminated_position_pair_count": (
                self.eliminated_position_pair_count
            ),
            "source_fast_state_bytes": self.source_fast_state_bytes,
            "packed_float_state_bytes": self.packed_float_state_bytes,
            "packed_index_state_bytes": self.packed_index_state_bytes,
            "packed_state_bytes": self.packed_state_bytes,
            "dense_scalar_multiplies": (
                first_dense + bridge_dense + output_local
            ),
            "packed_scalar_multiplies": (
                first_packed + bridge_packed + output_local
            ),
            "packed_components": {
                "input_to_layer_0_hidden": first_packed,
                "layer_0_hidden_to_layer_1_hidden": bridge_packed,
                "layer_1_hidden_to_residual_output": output_local,
            },
            "source_provenance": dict(self._source_provenance),
        }

    def _validate_input(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("packed triangular fused input must be a Tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (
            self.sequence_length,
            self.width,
        ):
            raise ValueError(
                "packed triangular fused input must have shape "
                "[batch, fixed_sequence, width]"
            )
        if (
            hidden_states.dtype != self.reference_tensor.dtype
            or hidden_states.device != self.reference_tensor.device
        ):
            raise ValueError(
                "packed triangular fused input must match the runtime "
                "dtype and device"
            )
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor):
                raise TypeError("attention_mask must be a Tensor")
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            if not attention_mask.to(torch.bool).all():
                raise ValueError(
                    "the packed fixed-position runtime does not support "
                    "padding"
                )

    def _causal_stage(
        self,
        values: Tensor,
        packed_kernel: Tensor,
        bias: Tensor,
    ) -> Tensor:
        gathered = values.index_select(
            1,
            self.causal_source_indices,
        )
        pair_outputs = torch.einsum(
            "bpi,pio->bpo",
            gathered,
            packed_kernel,
        )
        output = values.new_zeros(
            values.shape[0],
            self.sequence_length,
            packed_kernel.shape[2],
        )
        return (
            output.index_add(
                1,
                self.causal_target_indices,
                pair_outputs,
            )
            + bias
        )

    def forward_fast(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(hidden_states, attention_mask)
        centered = hidden_states - self.first_input_mean
        first_hidden = F.gelu(
            self._causal_stage(
                centered,
                self.packed_first_input_kernel,
                self.first_hidden_bias,
            )
        )
        second_hidden = F.gelu(
            self._causal_stage(
                first_hidden,
                self.packed_bridge_kernel,
                self.bridge_bias,
            )
        )
        return (
            torch.einsum(
                "bsh,shw->bsw",
                second_hidden,
                self.second_fused_output_weight,
            )
            + self.second_fused_output_bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefixes: Sequence[str] = ("layer.0", "layer.1"),
    ) -> Tensor:
        if len(prefixes) != 2 or not all(
            isinstance(prefix, str) and prefix for prefix in prefixes
        ):
            raise ValueError(
                "packed triangular fused stack requires exactly two prefixes"
            )
        if trace is not None:
            raise ValueError(
                "packed triangular runtime does not support activation "
                "traces; use the authenticated lazy fused source runtime"
            )
        return self.forward_fast(
            hidden_states,
            attention_mask=attention_mask,
        )


def save_lazy_fused_modal_stack(
    path: str | Path,
    *,
    stack: FusedTwoLayerModalStack,
    sidecar_paths: Mapping[str, str | Path],
    metadata: Mapping[str, object],
) -> None:
    """Save the seven-tensor fast runtime and verified sidecar descriptors.

    Descriptors store portable basenames. By default the loader resolves them
    beside this artifact; callers that keep sidecars elsewhere must pass that
    directory as ``sidecar_root`` when loading.
    """

    if not isinstance(stack, FusedTwoLayerModalStack):
        raise TypeError("stack must be a FusedTwoLayerModalStack")
    if not stack.uses_cross_layer_bypass:
        raise ValueError("lazy fusion requires the exact cross-layer bypass")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if not isinstance(sidecar_paths, Mapping):
        raise TypeError("sidecar_paths must be a mapping")
    portable_metadata = _portable_metadata(metadata)
    assert isinstance(portable_metadata, dict)
    _require_lazy_provenance(portable_metadata)
    if set(sidecar_paths) != set(_LAZY_SIDECAR_NAMES):
        raise ValueError("lazy runtime sidecar paths mismatch")
    descriptors: dict[str, dict[str, object]] = {}
    for name in _LAZY_SIDECAR_NAMES:
        source = Path(sidecar_paths[name])
        if not source.is_file():
            raise FileNotFoundError(source)
        data = source.read_bytes()
        descriptor = _SidecarArtifact(
            filename=source.name,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )
        descriptors[name] = asdict(descriptor)
    fast_state = {
        "first_input_mean": stack.first.input_mean,
        "first_input_kernel": stack.first.input_kernel,
        "first_hidden_bias": stack.first.hidden_bias,
        "bridge_kernel": stack.bridge_kernel,
        "bridge_bias": stack.bridge_bias,
        "second_fused_output_weight": (
            stack.second.fused_output_weight
        ),
        "second_fused_output_bias": stack.second.fused_output_bias,
    }
    torch.save(
        {
            "format_version": 2,
            "artifact_kind": "lazy_fused_two_layer_modal_stack",
            "config": asdict(stack.config),
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in fast_state.items()
            },
            "sidecars": descriptors,
            "metadata": portable_metadata,
        },
        Path(path),
    )


def load_lazy_fused_modal_stack(
    path: str | Path,
    *,
    sidecar_root: str | Path | None = None,
) -> tuple[
    LazyFusedTwoLayerModalStack,
    FusedTwoLayerStackConfig,
    dict[str, object],
]:
    """Load a compact fast runtime without opening instrumentation sidecars."""

    artifact_path = Path(path)
    state = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise ValueError("lazy fused runtime artifact must be an object")
    _require_exact_keys(
        state,
        {
            "format_version",
            "artifact_kind",
            "config",
            "state_dict",
            "sidecars",
            "metadata",
        },
        label="lazy fused runtime artifact",
    )
    if (
        type(state.get("format_version")) is not int
        or state.get("format_version") != 2
    ):
        raise ValueError("unsupported lazy fused runtime artifact format")
    if state.get("artifact_kind") != "lazy_fused_two_layer_modal_stack":
        raise ValueError("unsupported lazy fused runtime artifact kind")
    raw_config = state.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("lazy fused runtime config must be an object")
    _require_exact_keys(
        raw_config,
        {"first", "second", "cross_layer_bypass"},
        label="lazy fused runtime config",
    )
    raw_first = raw_config["first"]
    raw_second = raw_config["second"]
    if not isinstance(raw_first, Mapping) or not isinstance(
        raw_second,
        Mapping,
    ):
        raise ValueError("lazy fused runtime layer configs must be objects")
    raw_bypass = raw_config["cross_layer_bypass"]
    if type(raw_bypass) is not bool:
        raise ValueError("cross_layer_bypass must be boolean")
    config = FusedTwoLayerStackConfig(
        first=_layer_config_from_dict(raw_first),
        second=_layer_config_from_dict(raw_second),
        cross_layer_bypass=raw_bypass,
    )
    saved = state.get("state_dict")
    if not isinstance(saved, Mapping):
        raise ValueError("lazy fused runtime state must be an object")
    _require_exact_keys(
        saved,
        set(_LAZY_FAST_STATE_NAMES),
        label="lazy fused runtime state",
    )
    if any(not isinstance(saved[name], Tensor) for name in saved):
        raise ValueError("lazy fused runtime state values must be tensors")
    raw_sidecars = state.get("sidecars")
    if not isinstance(raw_sidecars, Mapping):
        raise ValueError("lazy fused runtime sidecars must be an object")
    _require_exact_keys(
        raw_sidecars,
        set(_LAZY_SIDECAR_NAMES),
        label="lazy fused runtime sidecars",
    )
    sidecars: dict[str, _SidecarArtifact] = {}
    for name in _LAZY_SIDECAR_NAMES:
        raw_descriptor = raw_sidecars[name]
        if not isinstance(raw_descriptor, Mapping):
            raise ValueError(
                f"lazy fused runtime sidecar {name} must be an object"
            )
        _require_exact_keys(
            raw_descriptor,
            {"filename", "sha256", "size_bytes"},
            label=f"lazy fused runtime sidecar {name}",
        )
        sidecars[name] = _SidecarArtifact(
            filename=raw_descriptor["filename"],  # type: ignore[arg-type]
            sha256=raw_descriptor["sha256"],  # type: ignore[arg-type]
            size_bytes=raw_descriptor["size_bytes"],  # type: ignore[arg-type]
        )
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("lazy fused runtime metadata must be an object")
    try:
        portable_metadata = _portable_metadata(metadata)
    except TypeError as error:
        raise ValueError(
            "lazy fused runtime metadata is not portable"
        ) from error
    assert isinstance(portable_metadata, dict)
    runtime = LazyFusedTwoLayerModalStack(
        config,
        **{name: saved[name] for name in _LAZY_FAST_STATE_NAMES},
        sidecars=sidecars,
        sidecar_root=(
            artifact_path.parent if sidecar_root is None else sidecar_root
        ),
        provenance=portable_metadata,
    )
    return runtime, config, portable_metadata


FusedStackRuntime = (
    FusedTwoLayerModalStack
    | LazyFusedTwoLayerModalStack
    | PackedTriangularFusedTwoLayerModalStack
)


class FusedToyTransformer(nn.Module):
    """A transformer-compatible frozen shell around a fused modal stack.

    Embeddings, final normalization, and the language-model head are copied
    into buffers.  No :class:`TransformerBlock` or trainable parameter remains.
    """

    def __init__(
        self,
        config: TransformerConfig,
        stack: FusedStackRuntime,
        *,
        token_embedding_weight: Tensor,
        position_embedding_weight: Tensor,
        final_norm_weight: Tensor,
        final_norm_bias: Tensor,
        final_norm_eps: float,
        lm_head_weight: Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be a TransformerConfig")
        if not isinstance(
            stack,
            (
                FusedTwoLayerModalStack,
                LazyFusedTwoLayerModalStack,
                PackedTriangularFusedTwoLayerModalStack,
            ),
        ):
            raise TypeError("stack must be a supported fused modal stack")
        if any(True for _ in stack.parameters()):
            raise ValueError("the fused stack must not contain parameters")
        if config.n_layers != 2:
            raise ValueError("the fused toy runtime requires exactly two layers")
        if stack.width != config.d_model:
            raise ValueError("fused stack width does not match the model")
        if stack.sequence_length > config.max_sequence_length:
            raise ValueError(
                "fused sequence length exceeds the configured maximum"
            )
        expected = {
            "token_embedding_weight": (config.vocab_size, config.d_model),
            "position_embedding_weight": (
                config.max_sequence_length,
                config.d_model,
            ),
            "final_norm_weight": (config.d_model,),
            "final_norm_bias": (config.d_model,),
            "lm_head_weight": (config.vocab_size, config.d_model),
        }
        supplied = {
            "token_embedding_weight": token_embedding_weight,
            "position_embedding_weight": position_embedding_weight,
            "final_norm_weight": final_norm_weight,
            "final_norm_bias": final_norm_bias,
            "lm_head_weight": lm_head_weight,
        }
        reference = stack.reference_tensor
        for name, value in supplied.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if tuple(value.shape) != expected[name]:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, "
                    f"expected {expected[name]}"
            )
            value = value.to(device=reference.device, dtype=reference.dtype)
            self.register_buffer(name, _clone_buffer(value))
        if (
            not isinstance(final_norm_eps, (float, int))
            or isinstance(final_norm_eps, bool)
            or not math.isfinite(float(final_norm_eps))
            or final_norm_eps <= 0
        ):
            raise ValueError("final_norm_eps must be positive")
        self.config = config
        self.stack = stack
        self.final_norm_eps = float(final_norm_eps)

    @classmethod
    def from_teacher(
        cls,
        teacher: ToyTransformer,
        stack: FusedStackRuntime,
    ) -> FusedToyTransformer:
        if not isinstance(teacher, ToyTransformer):
            raise TypeError("teacher must be a ToyTransformer")
        runtime = cls(
            teacher.config,
            stack,
            token_embedding_weight=teacher.token_embedding.weight,
            position_embedding_weight=teacher.position_embedding.weight,
            final_norm_weight=teacher.final_norm.weight,
            final_norm_bias=teacher.final_norm.bias,
            final_norm_eps=teacher.final_norm.eps,
            lm_head_weight=teacher.lm_head.weight,
        )
        return runtime.train(teacher.training)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        capture_activations: bool = False,
        retain_activation_gradients: bool = True,
        activation_interventions: (
            Mapping[str, ActivationIntervention] | None
        ) = None,
    ) -> TransformerOutput:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a Tensor")
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer index dtype")
        if input_ids.device != self.token_embedding_weight.device:
            raise ValueError(
                "input_ids must match the fused runtime device"
            )
        batch_size, sequence_length = input_ids.shape
        if sequence_length != self.stack.sequence_length:
            raise ValueError(
                "the fused fixed-position runtime requires sequence length "
                f"{self.stack.sequence_length}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a Tensor")
        elif attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids shape"
            )
        else:
            attention_mask = attention_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
        if not attention_mask.all():
            raise ValueError(
                "the fused fixed-position runtime does not support padding"
            )

        needs_trace = capture_activations or bool(activation_interventions)
        trace = None
        if needs_trace:
            trace = ActivationTrace(
                retain_grad=(
                    retain_activation_gradients
                    if capture_activations
                    else False
                ),
                interventions=activation_interventions,
                store=capture_activations,
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        token_embeddings = F.embedding(
            input_ids,
            self.token_embedding_weight,
        )
        position_embeddings = F.embedding(
            positions,
            self.position_embedding_weight,
        )
        if capture_activations and retain_activation_gradients:
            token_embeddings.requires_grad_()
            position_embeddings.requires_grad_()
        token_embeddings = record(
            trace,
            "embedding.token",
            token_embeddings,
        )
        position_embeddings = record(
            trace,
            "embedding.position",
            position_embeddings,
        )
        hidden_states = record(
            trace,
            "embedding.output",
            F.dropout(
                token_embeddings + position_embeddings,
                p=self.config.dropout,
                training=self.training,
            ),
        )
        hidden_states = self.stack(
            hidden_states,
            attention_mask=attention_mask,
            trace=trace,
        )
        hidden_states = record(
            trace,
            "final_norm",
            F.layer_norm(
                hidden_states,
                (self.config.d_model,),
                self.final_norm_weight,
                self.final_norm_bias,
                self.final_norm_eps,
            ),
        )
        logits = record(
            trace,
            "logits",
            F.linear(hidden_states, self.lm_head_weight),
        )
        if trace is not None:
            trace.assert_all_interventions_applied()
        return TransformerOutput(
            logits=logits,
            activations=trace if capture_activations else None,
        )


__all__ = [
    "FusedCompletedModalLayer",
    "FusedModalLayerConfig",
    "FusedStackRuntime",
    "FusedToyTransformer",
    "FusedTwoLayerModalStack",
    "FusedTwoLayerStackConfig",
    "LazyFusedTwoLayerModalStack",
    "LazyInstrumentationStatus",
    "PackedTriangularFusedTwoLayerModalStack",
    "load_fused_modal_stack",
    "load_lazy_fused_modal_stack",
    "save_fused_modal_stack",
    "save_lazy_fused_modal_stack",
]
