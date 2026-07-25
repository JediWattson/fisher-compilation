"""Optional MLX and Metal lowering for the packed triangular runtime.

The PyTorch lazy runtime remains the authenticated source and instrumentation
oracle. This module performs a one-time copy into MLX-owned arrays and offers:

* an ordinary MLX reference graph for inspection and differentiation;
* the same graph compiled with :func:`mlx.core.compile`;
* a custom Metal causal-stage kernel that reads only packed causal pairs.

Importing this module does not import MLX. The optional dependency is loaded
only when an MLX runtime is constructed, so the base package remains portable.
"""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import Tensor

from .fused_executor import (
    FusedToyTransformer,
    PackedTriangularFusedTwoLayerModalStack,
)


MLXExecutionBackend = Literal["eager", "compiled", "metal"]

_STACK_STATE_NAMES = frozenset(
    {
        "first_input_mean",
        "packed_first_input_kernel",
        "first_hidden_bias",
        "packed_bridge_kernel",
        "bridge_bias",
        "second_fused_output_weight",
        "second_fused_output_bias",
    }
)
_SHELL_STATE_NAMES = frozenset(
    {
        "token_embedding_weight",
        "position_embedding_weight",
        "final_norm_weight",
        "final_norm_bias",
        "lm_head_weight",
    }
)
_STACK_PRIVATE_STATE_NAMES = frozenset(
    f"_{name}" for name in _STACK_STATE_NAMES
)
_SHELL_PRIVATE_STATE_NAMES = frozenset(
    f"_{name}" for name in _SHELL_STATE_NAMES
)
_STACK_IDENTITY_NAMES = frozenset(
    {
        "backend",
        "config",
        "sequence_length",
        "width",
        "first_routing_width",
        "second_routing_width",
        "causal_pair_count",
        "source_packed_state_bytes",
        "source_provenance",
        "source_float_state_sha256",
    }
)


class MLXBackendUnavailableError(RuntimeError):
    """Raised when the optional MLX or Metal runtime cannot be used."""


@dataclass(frozen=True, slots=True)
class MLXModalStackOutput:
    """Instrumented result from the ordinary MLX modal graph."""

    output: Any
    activations: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MLXTransformerOutput:
    """Backend-local equivalent of the PyTorch transformer output."""

    logits: Any
    activations: Mapping[str, Any] | None


class _FrozenMLXRuntime:
    _immutable_attributes: frozenset[str] = frozenset()
    _state_attribute_names: frozenset[str] = frozenset()

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_runtime_frozen", False)
            and name in self._immutable_attributes
        ):
            raise AttributeError(
                f"{name} is immutable after MLX runtime construction"
            )
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> object:
        if name in self._state_attribute_names:
            value = object.__getattribute__(self, f"_{name}")
            return value[:]
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )

    def _freeze_runtime(self) -> None:
        object.__setattr__(self, "_runtime_frozen", True)


_PACKED_CAUSAL_GELU_SOURCE = r"""
    uint o = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;

    uint out_index = (b * S + t) * O + o;
    ACC acc = ACC(bias[t * O + o]);

    for (uint s = 0; s <= t; ++s) {
        uint pair = t * (t + 1) / 2 + s;
        uint value_base = (b * S + s) * I;
        uint kernel_base = pair * I * O + o;
        for (uint i = 0; i < I; ++i) {
            ACC weight = ACC(packed[kernel_base + i * O]);
            acc += ACC(values[value_base + i]) * weight;
        }
    }

    ACC activated = ACC(0.5f) * acc *
        (ACC(1.0f) + fisher_erf(acc * ACC(0.7071067811865475f)));
    out[out_index] = activated;
"""

_PACKED_CAUSAL_GELU_HEADER = r"""
// Error-function approximation adapted from MLX's MIT-licensed
// mlx/backend/metal/kernels/erf.h.
// Copyright © 2023 Apple Inc.
// See NOTICE.md for the retained license.
// The approximation is based on:
// https://stackoverflow.com/a/35148199
float fisher_erf(float a) {
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f) {
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f);
        r = metal::fma(r, t, -6.34846687e-1f);
        r = metal::fma(r, t, -1.28717512e-1f);
        r = metal::fma(r, t, -t);
        r = 1.0f - metal::precise::exp(r);
        r = metal::copysign(r, a);
    } else {
        r = -5.96761703e-4f;
        r = metal::fma(r, s, 4.99119423e-3f);
        r = metal::fma(r, s, -2.67681349e-2f);
        r = metal::fma(r, s, 1.12819925e-1f);
        r = metal::fma(r, s, -3.76125336e-1f);
        r = metal::fma(r, s, 1.28379166e-1f);
        r = metal::fma(r, a, a);
    }
    return r;
}
"""

_MLX_EXECUTION_PROBE = """
import mlx.core as mx
import mlx.nn as nn

value = mx.array([0.0], dtype=mx.float32)
mx.eval(nn.gelu(value + 1.0))
mx.synchronize()
"""


def mlx_is_installed() -> bool:
    """Return whether the optional MLX Python package can be discovered."""

    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx.core") is not None
    )


def _torch_float_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not value.is_floating_point():
            continue
        host = value.detach().cpu().contiguous()
        array = host.numpy()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mlx_float_state_sha256(runtime: Any) -> str:
    core = runtime._core
    state = runtime._state_items()
    core.eval(list(state.values()))
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = np.array(value, copy=True)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_canonical_causal_indices(
    source: PackedTriangularFusedTwoLayerModalStack,
) -> None:
    expected_target, expected_source = torch.tril_indices(
        source.sequence_length,
        source.sequence_length,
        device=source.causal_target_indices.device,
    )
    if not torch.equal(
        source.causal_target_indices,
        expected_target,
    ) or not torch.equal(
        source.causal_source_indices,
        expected_source,
    ):
        raise ValueError(
            "packed source must use canonical target-major causal-pair order"
        )


@lru_cache(maxsize=1)
def _mlx_execution_probe() -> tuple[bool, str]:
    if not mlx_is_installed():
        return False, "the optional MLX package is not installed"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _MLX_EXECUTION_PROBE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"MLX execution probe failed: {error}"
    if completed.returncode == 0:
        return True, ""
    detail = completed.stderr.strip().splitlines()
    reason = next(
        (
            line
            for line in detail
            if "No Metal device available" in line
            or "[metal::load_device]" in line
        ),
        detail[-1] if detail else "no diagnostic was emitted",
    )
    return (
        False,
        "MLX execution probe exited with status "
        f"{completed.returncode}: {reason}",
    )


@lru_cache(maxsize=1)
def _require_mlx() -> tuple[Any, Any]:
    usable, reason = _mlx_execution_probe()
    if not usable:
        raise MLXBackendUnavailableError(reason)
    try:
        core = importlib.import_module("mlx.core")
        neural = importlib.import_module("mlx.nn")
    except ImportError as error:
        raise MLXBackendUnavailableError(
            "MLX backend requires the optional dependency; "
            "install fisher-graph[mlx]"
        ) from error
    return core, neural


def mlx_metal_is_usable() -> bool:
    """Safely probe MLX execution without risking the calling process."""

    return _mlx_execution_probe()[0]


def mlx_metal_kernel_source_sha256() -> str:
    """Return the hash of the compiled custom-Metal source contract."""

    return hashlib.sha256(
        (
            _PACKED_CAUSAL_GELU_HEADER
            + "\0"
            + _PACKED_CAUSAL_GELU_SOURCE
        ).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def _require_metal(core: Any) -> None:
    if not core.metal.is_available():
        raise MLXBackendUnavailableError(
            "MLX is installed but no Metal backend is available"
        )
    try:
        probe = core.array([0.0], dtype=core.float32)
        core.eval(probe + 1.0)
        core.synchronize()
    except RuntimeError as error:
        raise MLXBackendUnavailableError(
            "MLX is installed but the Metal device cannot execute work"
        ) from error


@lru_cache(maxsize=1)
def _packed_causal_gelu_kernel() -> Any:
    core, _ = _require_mlx()
    return core.fast.metal_kernel(
        name="fisher_packed_causal_gelu",
        input_names=["values", "packed", "bias"],
        output_names=["out"],
        source=_PACKED_CAUSAL_GELU_SOURCE,
        header=_PACKED_CAUSAL_GELU_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=False,
        compile_options={"math_mode": "safe"},
    )


def _copy_torch_float32(core: Any, value: Tensor, *, name: str) -> Any:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    host = value.detach().cpu().contiguous().numpy().copy()
    return core.array(host, dtype=core.float32)


def _validate_backend(backend: str) -> MLXExecutionBackend:
    if backend not in ("eager", "compiled", "metal"):
        raise ValueError("backend must be eager, compiled, or metal")
    return backend  # type: ignore[return-value]


class MLXPackedTriangularFusedTwoLayerModalStack(_FrozenMLXRuntime):
    """MLX lowering of a packed two-layer causal modal stack."""

    _state_attribute_names = _STACK_STATE_NAMES
    _immutable_attributes = (
        _STACK_STATE_NAMES
        | _STACK_PRIVATE_STATE_NAMES
        | _STACK_IDENTITY_NAMES
    )

    def __init__(
        self,
        source: PackedTriangularFusedTwoLayerModalStack,
        *,
        backend: MLXExecutionBackend = "metal",
    ) -> None:
        if not isinstance(
            source,
            PackedTriangularFusedTwoLayerModalStack,
        ):
            raise TypeError(
                "source must be a PackedTriangularFusedTwoLayerModalStack"
            )
        _require_canonical_causal_indices(source)
        backend = _validate_backend(backend)
        core, neural = _require_mlx()
        if backend == "metal":
            _require_metal(core)

        self._core = core
        self._neural = neural
        self.backend = backend
        self.config = source.config
        self.sequence_length = source.sequence_length
        self.width = source.width
        self.first_routing_width = source.config.first.routing_width
        self.second_routing_width = source.config.second.routing_width
        self.causal_pair_count = source.causal_pair_count
        self.source_packed_state_bytes = source.packed_state_bytes
        self.source_provenance = source.runtime_provenance()
        self.source_float_state_sha256 = _torch_float_state_sha256(source)

        state = {
            "first_input_mean": source.first_input_mean,
            "packed_first_input_kernel": (
                source.packed_first_input_kernel
            ),
            "first_hidden_bias": source.first_hidden_bias,
            "packed_bridge_kernel": source.packed_bridge_kernel,
            "bridge_bias": source.bridge_bias,
            "second_fused_output_weight": (
                source.second_fused_output_weight
            ),
            "second_fused_output_bias": source.second_fused_output_bias,
        }
        for name, value in state.items():
            setattr(
                self,
                f"_{name}",
                _copy_torch_float32(core, value, name=name),
            )
        core.eval(list(self._state_items().values()))
        if (
            _mlx_float_state_sha256(self)
            != self.source_float_state_sha256
        ):
            raise RuntimeError(
                "MLX conversion changed the packed floating-point state"
            )

        self._compiled_reference = core.compile(
            self._forward_reference_output
        )
        self._compiled_metal = core.compile(self._forward_metal_output)
        self._freeze_runtime()

    @classmethod
    def from_torch(
        cls,
        source: PackedTriangularFusedTwoLayerModalStack,
        *,
        backend: MLXExecutionBackend = "metal",
    ) -> MLXPackedTriangularFusedTwoLayerModalStack:
        """Copy one authenticated packed PyTorch runtime into MLX."""

        return cls(source, backend=backend)

    def _state_items(self) -> dict[str, Any]:
        return {
            "first_input_mean": self._first_input_mean,
            "packed_first_input_kernel": (
                self._packed_first_input_kernel
            ),
            "first_hidden_bias": self._first_hidden_bias,
            "packed_bridge_kernel": self._packed_bridge_kernel,
            "bridge_bias": self._bridge_bias,
            "second_fused_output_weight": (
                self._second_fused_output_weight
            ),
            "second_fused_output_bias": self._second_fused_output_bias,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return non-aliasing MLX array copies of the frozen state."""

        return {
            name: value[:]
            for name, value in self._state_items().items()
        }

    @property
    def state_bytes(self) -> int:
        return sum(
            int(value.size) * int(value.itemsize)
            for value in self._state_items().values()
        )

    def _validate_hidden_states(self, hidden_states: Any) -> None:
        if not isinstance(hidden_states, self._core.array):
            raise TypeError("hidden_states must be an mlx.core.array")
        if tuple(hidden_states.shape[1:]) != (
            self.sequence_length,
            self.width,
        ) or hidden_states.ndim != 3:
            raise ValueError(
                "MLX packed input must have shape "
                "[batch, fixed_sequence, width]"
            )
        if hidden_states.dtype != self._core.float32:
            raise ValueError("MLX packed input must use float32")
        if hidden_states.shape[0] <= 0:
            raise ValueError("MLX packed input batch must be nonempty")

    def _causal_stage_reference(
        self,
        values: Any,
        packed: Any,
        bias: Any,
    ) -> Any:
        outputs = []
        for target in range(self.sequence_length):
            accumulator = self._core.broadcast_to(
                bias[target],
                (values.shape[0], bias.shape[1]),
            )
            pair_offset = target * (target + 1) // 2
            for source in range(target + 1):
                accumulator = (
                    accumulator
                    + values[:, source, :] @ packed[pair_offset + source]
                )
            outputs.append(self._neural.gelu(accumulator))
        return self._core.stack(outputs, axis=1)

    def _causal_stage_metal(
        self,
        values: Any,
        packed: Any,
        bias: Any,
    ) -> Any:
        batch_size, sequence_length, input_width = values.shape
        pair_count, packed_input_width, output_width = packed.shape
        if sequence_length != self.sequence_length:
            raise ValueError("Metal causal stage sequence length mismatch")
        if packed_input_width != input_width:
            raise ValueError("Metal causal stage input width mismatch")
        if pair_count != self.causal_pair_count:
            raise ValueError("Metal causal stage pair count mismatch")
        if tuple(bias.shape) != (sequence_length, output_width):
            raise ValueError("Metal causal stage bias shape mismatch")
        uint32_max = int(np.iinfo(np.uint32).max)
        uint32_capacity = uint32_max + 1
        indexed_sizes = (
            batch_size * sequence_length * input_width,
            batch_size * sequence_length * output_width,
            pair_count * input_width * output_width,
            sequence_length * output_width,
        )
        if (
            sequence_length * (sequence_length - 1) > uint32_max
            or any(size > uint32_capacity for size in indexed_sizes)
        ):
            raise ValueError(
                "Metal causal stage exceeds its 32-bit index limit"
            )
        return _packed_causal_gelu_kernel()(
            inputs=[values, packed, bias],
            output_shapes=[
                (batch_size, sequence_length, output_width)
            ],
            output_dtypes=[values.dtype],
            grid=(output_width, sequence_length, batch_size),
            threadgroup=(min(32, output_width), 1, 1),
            template=[
                ("ACC", self._core.float32),
                ("S", sequence_length),
                ("I", input_width),
                ("O", output_width),
            ],
        )[0]

    def _forward_reference_with_activations(
        self,
        hidden_states: Any,
    ) -> tuple[Any, dict[str, Any]]:
        centered = hidden_states - self._first_input_mean
        first_hidden = self._causal_stage_reference(
            centered,
            self._packed_first_input_kernel,
            self._first_hidden_bias,
        )
        second_hidden = self._causal_stage_reference(
            first_hidden,
            self._packed_bridge_kernel,
            self._bridge_bias,
        )
        output = (
            self._core.einsum(
                "bsh,shw->bsw",
                second_hidden,
                self._second_fused_output_weight,
            )
            + self._second_fused_output_bias
        )
        return output, {
            "layer.0.modal.hidden": first_hidden,
            "layer.1.modal.hidden": second_hidden,
            "layer.1.output": output,
        }

    def _forward_reference_output(self, hidden_states: Any) -> Any:
        return self._forward_reference_with_activations(hidden_states)[0]

    def _forward_metal_output(self, hidden_states: Any) -> Any:
        centered = hidden_states - self._first_input_mean
        first_hidden = self._causal_stage_metal(
            centered,
            self._packed_first_input_kernel,
            self._first_hidden_bias,
        )
        second_hidden = self._causal_stage_metal(
            first_hidden,
            self._packed_bridge_kernel,
            self._bridge_bias,
        )
        return (
            self._core.einsum(
                "bsh,shw->bsw",
                second_hidden,
                self._second_fused_output_weight,
            )
            + self._second_fused_output_bias
        )

    def __call__(
        self,
        hidden_states: Any,
        *,
        backend: MLXExecutionBackend | None = None,
        capture_activations: bool = False,
    ) -> Any | MLXModalStackOutput:
        self._validate_hidden_states(hidden_states)
        selected = self.backend if backend is None else _validate_backend(
            backend
        )
        if capture_activations:
            output, activations = self._forward_reference_with_activations(
                hidden_states
            )
            return MLXModalStackOutput(output, activations)
        if selected == "eager":
            return self._forward_reference_output(hidden_states)
        if selected == "compiled":
            return self._compiled_reference(hidden_states)
        _require_metal(self._core)
        return self._compiled_metal(hidden_states)


class MLXFusedToyTransformer(_FrozenMLXRuntime):
    """Inference-only MLX shell around the packed modal stack."""

    _state_attribute_names = _SHELL_STATE_NAMES
    _immutable_attributes = (
        _SHELL_STATE_NAMES
        | _SHELL_PRIVATE_STATE_NAMES
        | frozenset(
            {"backend", "config", "stack", "final_norm_eps"}
        )
    )

    def __init__(
        self,
        source: FusedToyTransformer,
        *,
        stack: MLXPackedTriangularFusedTwoLayerModalStack | None = None,
        backend: MLXExecutionBackend = "metal",
    ) -> None:
        if not isinstance(source, FusedToyTransformer):
            raise TypeError("source must be a FusedToyTransformer")
        if source.training:
            raise ValueError("the MLX fused runtime requires an eval source")
        if not isinstance(
            source.stack,
            PackedTriangularFusedTwoLayerModalStack,
        ):
            raise TypeError(
                "the MLX fused runtime requires a packed triangular stack"
            )
        backend = _validate_backend(backend)
        if stack is None:
            stack = MLXPackedTriangularFusedTwoLayerModalStack.from_torch(
                source.stack,
                backend=backend,
            )
        elif not isinstance(
            stack,
            MLXPackedTriangularFusedTwoLayerModalStack,
        ):
            raise TypeError("stack must be an MLX packed triangular stack")
        elif (
            stack.sequence_length != source.stack.sequence_length
            or stack.width != source.stack.width
        ):
            raise ValueError("MLX stack does not match the source model")
        else:
            _require_canonical_causal_indices(source.stack)
            if (
                stack.config != source.stack.config
                or stack.source_float_state_sha256
                != _torch_float_state_sha256(source.stack)
                or _mlx_float_state_sha256(stack)
                != stack.source_float_state_sha256
            ):
                raise ValueError(
                    "MLX stack is not derived from the source model stack"
                )

        self._core = stack._core
        self.backend = backend
        self.config = source.config
        self.stack = stack
        self.final_norm_eps = source.final_norm_eps
        shell = {
            "token_embedding_weight": source.token_embedding_weight,
            "position_embedding_weight": source.position_embedding_weight,
            "final_norm_weight": source.final_norm_weight,
            "final_norm_bias": source.final_norm_bias,
            "lm_head_weight": source.lm_head_weight,
        }
        for name, value in shell.items():
            setattr(
                self,
                f"_{name}",
                _copy_torch_float32(self._core, value, name=name),
            )
        self._core.eval(list(self._shell_state_items().values()))
        self._compiled_reference = self._core.compile(
            self._forward_reference_logits
        )
        self._compiled_metal = self._core.compile(
            self._forward_metal_logits
        )
        self._freeze_runtime()

    @classmethod
    def from_torch(
        cls,
        source: FusedToyTransformer,
        *,
        stack: MLXPackedTriangularFusedTwoLayerModalStack | None = None,
        backend: MLXExecutionBackend = "metal",
    ) -> MLXFusedToyTransformer:
        return cls(source, stack=stack, backend=backend)

    def _shell_state_items(self) -> dict[str, Any]:
        return {
            "token_embedding_weight": self._token_embedding_weight,
            "position_embedding_weight": self._position_embedding_weight,
            "final_norm_weight": self._final_norm_weight,
            "final_norm_bias": self._final_norm_bias,
            "lm_head_weight": self._lm_head_weight,
        }

    def shell_state_dict(self) -> dict[str, Any]:
        """Return non-aliasing MLX array copies of the frozen shell."""

        return {
            name: value[:]
            for name, value in self._shell_state_items().items()
        }

    @property
    def state_bytes(self) -> int:
        shell_bytes = sum(
            int(value.size) * int(value.itemsize)
            for value in self._shell_state_items().values()
        )
        return self.stack.state_bytes + shell_bytes

    def _validate_inputs(
        self,
        input_ids: Any,
        attention_mask: Any | None,
    ) -> None:
        if not isinstance(input_ids, self._core.array):
            raise TypeError("input_ids must be an mlx.core.array")
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )
        if input_ids.shape[0] <= 0:
            raise ValueError("input_ids batch must be nonempty")
        if input_ids.dtype not in (
            self._core.int32,
            self._core.int64,
        ):
            raise ValueError("input_ids must use an integer index dtype")
        if input_ids.shape[1] != self.stack.sequence_length:
            raise ValueError(
                "the MLX fixed-position runtime requires sequence length "
                f"{self.stack.sequence_length}"
            )
        if not bool(
            self._core.all(
                (input_ids >= 0)
                & (input_ids < self.config.vocab_size)
            ).item()
        ):
            raise ValueError(
                "input_ids must be within the configured vocabulary"
            )
        if attention_mask is not None:
            if not isinstance(attention_mask, self._core.array):
                raise TypeError(
                    "attention_mask must be an mlx.core.array"
                )
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    "attention_mask must match input_ids shape"
                )
            if not bool(self._core.all(attention_mask).item()):
                raise ValueError(
                    "the MLX fixed-position runtime does not support padding"
                )

    def _embed(self, input_ids: Any) -> tuple[Any, Any, Any]:
        token_embeddings = self._core.take(
            self._token_embedding_weight,
            input_ids,
            axis=0,
        )
        position_embeddings = self._core.broadcast_to(
            self._position_embedding_weight[
                : self.stack.sequence_length
            ],
            token_embeddings.shape,
        )
        return (
            token_embeddings,
            position_embeddings,
            token_embeddings + position_embeddings,
        )

    def _finish(self, hidden_states: Any) -> tuple[Any, Any]:
        normalized = self._core.fast.layer_norm(
            hidden_states,
            self._final_norm_weight,
            self._final_norm_bias,
            self.final_norm_eps,
        )
        logits = normalized @ self._core.transpose(
            self._lm_head_weight
        )
        return normalized, logits

    def _forward_reference_logits(self, input_ids: Any) -> Any:
        _, _, hidden_states = self._embed(input_ids)
        hidden_states = self.stack._forward_reference_output(
            hidden_states
        )
        return self._finish(hidden_states)[1]

    def _forward_metal_logits(self, input_ids: Any) -> Any:
        _, _, hidden_states = self._embed(input_ids)
        hidden_states = self.stack._forward_metal_output(hidden_states)
        return self._finish(hidden_states)[1]

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        capture_activations: bool = False,
        activation_interventions: Mapping[str, object] | None = None,
        backend: MLXExecutionBackend | None = None,
    ) -> MLXTransformerOutput:
        self._validate_inputs(input_ids, attention_mask)
        if activation_interventions:
            raise ValueError(
                "the MLX backend does not yet support activation "
                "interventions"
            )
        selected = self.backend if backend is None else _validate_backend(
            backend
        )
        if capture_activations:
            token, position, hidden_states = self._embed(input_ids)
            modal = self.stack(
                hidden_states,
                backend="eager",
                capture_activations=True,
            )
            assert isinstance(modal, MLXModalStackOutput)
            normalized, logits = self._finish(modal.output)
            activations = {
                "embedding.token": token,
                "embedding.position": position,
                "embedding.output": hidden_states,
                **modal.activations,
                "final_norm": normalized,
                "logits": logits,
            }
            return MLXTransformerOutput(logits, activations)
        if selected == "eager":
            logits = self._forward_reference_logits(input_ids)
        elif selected == "compiled":
            logits = self._compiled_reference(input_ids)
        else:
            _require_metal(self._core)
            logits = self._compiled_metal(input_ids)
        return MLXTransformerOutput(logits, None)


def mlx_array_from_torch(value: Tensor) -> Any:
    """Copy a CPU-compatible PyTorch tensor into an MLX-owned array."""

    if not isinstance(value, Tensor):
        raise TypeError("value must be a torch.Tensor")
    core, _ = _require_mlx()
    host = value.detach().cpu().contiguous().numpy().copy()
    return core.array(host)


def mlx_array_to_numpy(value: Any) -> np.ndarray:
    """Synchronize one MLX array and return an independent NumPy copy."""

    core, _ = _require_mlx()
    if not isinstance(value, core.array):
        raise TypeError("value must be an mlx.core.array")
    core.eval(value)
    return np.array(value, copy=True)


def mlx_runtime_provenance(
    runtime: MLXPackedTriangularFusedTwoLayerModalStack,
) -> dict[str, object]:
    """Return portable facts for an in-memory derived MLX runtime."""

    if not isinstance(
        runtime,
        MLXPackedTriangularFusedTwoLayerModalStack,
    ):
        raise TypeError("runtime must be an MLX packed triangular stack")
    state_sha256 = _mlx_float_state_sha256(runtime)
    return {
        "runtime_kind": "mlx_packed_triangular_modal_stack",
        "source_runtime_kind": (
            "packed_triangular_fused_two_layer_modal_stack"
        ),
        "source_float_state_sha256": (
            runtime.source_float_state_sha256
        ),
        "state_sha256": state_sha256,
        "state_matches_source": (
            state_sha256 == runtime.source_float_state_sha256
        ),
        "source_provenance": dict(runtime.source_provenance),
        "backend": runtime.backend,
        "default_backend": runtime.backend,
        "available_execution_backends": [
            "eager",
            "compiled",
            "metal",
        ],
        "sequence_length": runtime.sequence_length,
        "causal_pair_count": runtime.causal_pair_count,
        "causal_pair_order": "target_major_lower_triangle",
        "state_bytes": runtime.state_bytes,
        "weights_updated": False,
        "serialized_artifact": False,
        "supports_activation_capture": True,
        "capture_backend": "ordinary_mlx_graph",
        "metal_kernel": "fisher_packed_causal_gelu",
        "metal_kernel_source_sha256": (
            mlx_metal_kernel_source_sha256()
        ),
        "finite_input_algebraic_equivalence": True,
        "bit_exact_to_pytorch": False,
    }


__all__ = [
    "MLXBackendUnavailableError",
    "MLXExecutionBackend",
    "MLXFusedToyTransformer",
    "MLXModalStackOutput",
    "MLXPackedTriangularFusedTwoLayerModalStack",
    "MLXTransformerOutput",
    "mlx_array_from_torch",
    "mlx_array_to_numpy",
    "mlx_is_installed",
    "mlx_metal_kernel_source_sha256",
    "mlx_metal_is_usable",
    "mlx_runtime_provenance",
]
