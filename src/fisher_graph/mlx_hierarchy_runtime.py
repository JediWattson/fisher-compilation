"""Optional MLX lowering for one prepared connectivity decomposition.

The authenticated PyTorch decomposition remains the analysis authority.  This
module validates it once, copies derived float32 benchmark state into
MLX-owned arrays once, and exposes three compiled executions over the same
canonical input tuple:

* ``source_dense`` evaluates the exact affine source transfer;
* ``candidate_dense`` evaluates the materialized ``P @ R`` candidate;
* ``candidate_factorized`` evaluates ``R`` followed by ``P`` with centering
  folded into the output bias.

The runtime is benchmark-only.  It carries no merge, pruning, routing,
mutation, or source-replacement authority.  MLX is discovered and imported
lazily so importing this module remains safe when the optional dependency or
an executable Metal device is unavailable.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from functools import lru_cache
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .modal_connectivity_modes import ModalConnectivityDecomposition


__all__ = [
    "MLXHierarchyBackendUnavailableError",
    "PreparedMLXHierarchyRuntime",
    "mlx_hierarchy_is_installed",
    "mlx_hierarchy_is_usable",
]


_FORMAT_VERSION = 1
_RUNTIME_KIND = "fisher_graph.prepared_mlx_hierarchy_runtime"
_RUNTIME_DOMAIN = b"fisher_graph.prepared_mlx_hierarchy_runtime.v1\0"
_STATE_DOMAIN = b"fisher_graph.prepared_mlx_hierarchy_state.v1\0"
_SYSTEMS = (
    "source_dense",
    "candidate_dense",
    "candidate_factorized",
)

_MLX_EXECUTION_PROBE = """
import mlx.core as mx

value = mx.array([0.0], dtype=mx.float32)
mx.eval(value + 1.0)
mx.synchronize()
"""


class MLXHierarchyBackendUnavailableError(RuntimeError):
    """Raised when the optional MLX hierarchy runtime cannot execute."""


def mlx_hierarchy_is_installed() -> bool:
    """Return whether the optional MLX package can be discovered."""

    try:
        return (
            importlib.util.find_spec("mlx") is not None
            and importlib.util.find_spec("mlx.core") is not None
        )
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _mlx_execution_probe() -> tuple[bool, str]:
    if not mlx_hierarchy_is_installed():
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


def mlx_hierarchy_is_usable() -> bool:
    """Safely probe whether MLX can execute and synchronize device work."""

    return _mlx_execution_probe()[0]


@lru_cache(maxsize=1)
def _require_mlx() -> Any:
    usable, reason = _mlx_execution_probe()
    if not usable:
        raise MLXHierarchyBackendUnavailableError(reason)
    try:
        return importlib.import_module("mlx.core")
    except ImportError as error:
        raise MLXHierarchyBackendUnavailableError(
            "MLX hierarchy runtime requires the optional MLX dependency"
        ) from error


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _host_float32(value: Tensor, *, label: str) -> np.ndarray:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()
    if not torch.isfinite(canonical).all():
        raise ValueError(f"{label} must contain only finite values")
    return canonical.numpy().copy()


def _state_sha256(state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(_STATE_DOMAIN)
    for name, value in sorted(state.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mlx_state_sha256(core: Any, state: dict[str, Any]) -> str:
    core.eval(list(state.values()))
    core.synchronize()
    host = {
        name: np.array(value, copy=True)
        for name, value in state.items()
    }
    return _state_sha256(host)


def _state_bytes(state: dict[str, np.ndarray], names: tuple[str, ...]) -> int:
    return sum(int(state[name].nbytes) for name in names)


class PreparedMLXHierarchyRuntime:
    """Frozen benchmark lowering of one connectivity decomposition."""

    def __init__(
        self,
        decomposition: ModalConnectivityDecomposition,
    ) -> None:
        if not isinstance(decomposition, ModalConnectivityDecomposition):
            raise TypeError(
                "decomposition must be ModalConnectivityDecomposition"
            )

        # Frozen dataclasses do not make contained tensors immutable.  Validate
        # immediately before copying, then retain no tensor aliases.
        decomposition.validate_integrity()
        transfer = decomposition.source_transfer
        factors = decomposition.factors

        input_ports = transfer.input_ports
        output_ports = transfer.output_ports
        self.input_port_names = tuple(port.name for port in input_ports)
        self.output_port_names = tuple(port.name for port in output_ports)
        self.input_widths = tuple(port.width for port in input_ports)
        self.output_widths = tuple(port.width for port in output_ports)
        input_index = {
            name: index for index, name in enumerate(self.input_port_names)
        }
        self.input_prefix_indices = tuple(
            tuple(input_index[name] for name in prefix)
            for prefix in transfer.input_prefixes
        )
        self.retained_ranks = tuple(
            factor.retained_rank for factor in factors
        )
        self.source_decomposition_sha256 = decomposition.artifact_sha256
        self.source_transfer_sha256 = transfer.artifact_sha256
        self.source_factor_sha256s = tuple(
            factor.artifact_sha256 for factor in factors
        )

        host_state: dict[str, np.ndarray] = {}
        source_names: list[str] = []
        candidate_dense_names: list[str] = []
        candidate_factorized_names: list[str] = []
        for index, (source_matrix, source_offset, factor) in enumerate(
            zip(
                transfer.transfer_matrices,
                transfer.affine_offsets,
                factors,
                strict=True,
            )
        ):
            suffix = f"{index:03d}"
            source_matrix_name = f"source.matrix.{suffix}"
            source_offset_name = f"source.offset.{suffix}"
            host_state[source_matrix_name] = _host_float32(
                source_matrix,
                label=source_matrix_name,
            )
            host_state[source_offset_name] = _host_float32(
                source_offset,
                label=source_offset_name,
            )
            source_names.extend((source_matrix_name, source_offset_name))

            candidate_matrix = (
                factor.prolongation @ factor.restriction
            ).contiguous()
            candidate_offset = (
                factor.output_mean
                - candidate_matrix @ factor.input_mean
            ).contiguous()
            candidate_matrix_name = f"candidate_dense.matrix.{suffix}"
            candidate_offset_name = f"candidate_dense.offset.{suffix}"
            host_state[candidate_matrix_name] = _host_float32(
                candidate_matrix,
                label=candidate_matrix_name,
            )
            host_state[candidate_offset_name] = _host_float32(
                candidate_offset,
                label=candidate_offset_name,
            )
            candidate_dense_names.extend(
                (candidate_matrix_name, candidate_offset_name)
            )

            factor_values = (
                ("restriction", factor.restriction),
                ("prolongation", factor.prolongation),
                ("output_bias", candidate_offset),
            )
            for value_name, value in factor_values:
                state_name = (
                    f"candidate_factorized.{value_name}.{suffix}"
                )
                host_state[state_name] = _host_float32(
                    value,
                    label=state_name,
                )
                candidate_factorized_names.append(state_name)

        system_state_names = {
            "source_dense": tuple(source_names),
            "candidate_dense": tuple(candidate_dense_names),
            "candidate_factorized": tuple(candidate_factorized_names),
        }
        state_sha256 = _state_sha256(host_state)

        core = _require_mlx()
        self._core = core
        mlx_state = {
            name: core.array(value, dtype=core.float32)
            for name, value in host_state.items()
        }
        if _mlx_state_sha256(core, mlx_state) != state_sha256:
            raise RuntimeError(
                "MLX conversion changed prepared hierarchy state"
            )

        def arrays(prefix: str) -> tuple[Any, ...]:
            return tuple(
                mlx_state[f"{prefix}.{index:03d}"]
                for index in range(len(factors))
            )

        self._source_matrices = arrays("source.matrix")
        self._source_offsets = arrays("source.offset")
        self._candidate_matrices = arrays("candidate_dense.matrix")
        self._candidate_offsets = arrays("candidate_dense.offset")
        self._restrictions = arrays("candidate_factorized.restriction")
        self._prolongations = arrays("candidate_factorized.prolongation")
        self._factorized_output_biases = arrays(
            "candidate_factorized.output_bias"
        )
        self._state = tuple(sorted(mlx_state.items()))
        self.state_sha256 = state_sha256
        self.state_bytes = sum(
            int(value.nbytes) for value in host_state.values()
        )
        self._system_state_bytes = tuple(
            (
                system,
                _state_bytes(host_state, system_state_names[system]),
            )
            for system in _SYSTEMS
        )
        self._system_stored_scalar_counts = tuple(
            (system, byte_count // 4)
            for system, byte_count in self._system_state_bytes
        )
        self.source_dense_macs_per_row = sum(
            int(matrix.size) for matrix in self._source_matrices
        )
        self.candidate_dense_macs_per_row = sum(
            int(matrix.size) for matrix in self._candidate_matrices
        )
        self.candidate_factorized_macs_per_row = (
            decomposition.candidate_macs_per_row
        )

        payload = {
            "artifact_kind": _RUNTIME_KIND,
            "format_version": _FORMAT_VERSION,
            "source_decomposition_sha256": (
                self.source_decomposition_sha256
            ),
            "source_transfer_sha256": self.source_transfer_sha256,
            "source_factor_sha256s": self.source_factor_sha256s,
            "input_ports": tuple(
                {
                    "name": port.name,
                    "width": port.width,
                    "artifact_sha256": port.artifact_sha256,
                }
                for port in input_ports
            ),
            "output_ports": tuple(
                {
                    "name": port.name,
                    "width": port.width,
                    "artifact_sha256": port.artifact_sha256,
                }
                for port in output_ports
            ),
            "input_prefix_indices": self.input_prefix_indices,
            "retained_ranks": self.retained_ranks,
            "state_sha256": self.state_sha256,
            "state_bytes": self.state_bytes,
            "system_state_bytes": self.system_state_bytes,
            "system_stored_scalar_counts": (
                self.system_stored_scalar_counts
            ),
            "source_dense_macs_per_row": self.source_dense_macs_per_row,
            "candidate_dense_macs_per_row": (
                self.candidate_dense_macs_per_row
            ),
            "candidate_factorized_macs_per_row": (
                self.candidate_factorized_macs_per_row
            ),
            "systems": _SYSTEMS,
            "dtype": "float32",
            "factorized_centering": "folded_output_bias",
            "factorized_hot_path_stores_input_mean": False,
            "factorized_hot_path_stores_output_mean": False,
            "analysis_only": True,
            "benchmark_only": True,
            "authorizes_replacement": False,
        }
        self.artifact_sha256 = _json_sha256(
            payload,
            domain=_RUNTIME_DOMAIN,
        )
        self._provenance_payload = payload
        self._compiled_source_dense = core.compile(
            self._source_dense_forward
        )
        self._compiled_candidate_dense = core.compile(
            self._candidate_dense_forward
        )
        self._compiled_candidate_factorized = core.compile(
            self._candidate_factorized_forward
        )
        self._runtime_frozen = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_runtime_frozen", False):
            raise AttributeError(
                "prepared MLX hierarchy runtime is immutable"
            )
        object.__setattr__(self, name, value)

    @classmethod
    def from_decomposition(
        cls,
        decomposition: ModalConnectivityDecomposition,
    ) -> PreparedMLXHierarchyRuntime:
        """Validate and lower one decomposition into MLX-owned state."""

        return cls(decomposition)

    @property
    def system_state_bytes(self) -> dict[str, int]:
        """Return per-system MLX state bytes without exposing mutable state."""

        return dict(self._system_state_bytes)

    @property
    def system_stored_scalar_counts(self) -> dict[str, int]:
        """Return per-system stored float32 scalar counts."""

        return dict(self._system_stored_scalar_counts)

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> None:
        if type(inputs) is not tuple:
            raise TypeError("prepared hierarchy inputs must be a tuple")
        if len(inputs) != len(self.input_port_names):
            raise ValueError(
                "prepared hierarchy inputs do not match canonical ports"
            )
        leading_shape: tuple[int, ...] | None = None
        for name, width, value in zip(
            self.input_port_names,
            self.input_widths,
            inputs,
            strict=True,
        ):
            if not isinstance(value, self._core.array):
                raise TypeError(
                    f"prepared hierarchy input {name} must be an MLX array"
                )
            if value.ndim < 1 or value.shape[-1] != width:
                raise ValueError(
                    f"prepared hierarchy input {name} has the wrong width"
                )
            if value.dtype != self._core.float32:
                raise ValueError(
                    f"prepared hierarchy input {name} must use float32"
                )
            current_leading = tuple(value.shape[:-1])
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError(
                    "prepared hierarchy inputs must share a leading shape"
                )

    def _joined_prefix(
        self,
        inputs: tuple[Any, ...],
        indices: tuple[int, ...],
    ) -> Any:
        values = tuple(inputs[index] for index in indices)
        if len(values) == 1:
            return values[0]
        return self._core.concatenate(values, axis=-1)

    def _source_dense_forward(self, *inputs: Any) -> tuple[Any, ...]:
        canonical = tuple(inputs)
        return tuple(
            self._joined_prefix(canonical, indices) @ matrix.T + offset
            for indices, matrix, offset in zip(
                self.input_prefix_indices,
                self._source_matrices,
                self._source_offsets,
                strict=True,
            )
        )

    def _candidate_dense_forward(
        self,
        *inputs: Any,
    ) -> tuple[Any, ...]:
        canonical = tuple(inputs)
        return tuple(
            self._joined_prefix(canonical, indices) @ matrix.T + offset
            for indices, matrix, offset in zip(
                self.input_prefix_indices,
                self._candidate_matrices,
                self._candidate_offsets,
                strict=True,
            )
        )

    def _candidate_factorized_forward(
        self,
        *inputs: Any,
    ) -> tuple[Any, ...]:
        canonical = tuple(inputs)
        leading_shape = tuple(canonical[0].shape[:-1])
        outputs: list[Any] = []
        for (
            indices,
            rank,
            output_width,
            restriction,
            prolongation,
            output_bias,
        ) in zip(
            self.input_prefix_indices,
            self.retained_ranks,
            self.output_widths,
            self._restrictions,
            self._prolongations,
            self._factorized_output_biases,
            strict=True,
        ):
            joined = self._joined_prefix(canonical, indices)
            if rank == 0:
                outputs.append(
                    self._core.broadcast_to(
                        output_bias,
                        (*leading_shape, output_width),
                    )
                )
                continue
            modes = joined @ restriction.T
            outputs.append(modes @ prolongation.T + output_bias)
        return tuple(outputs)

    def source_dense(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Execute the compiled exact dense source transfer."""

        self._validate_inputs(inputs)
        return self._normalize_outputs(
            self._compiled_source_dense(*inputs)
        )

    def candidate_dense(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Execute the compiled materialized ``P @ R`` candidate."""

        self._validate_inputs(inputs)
        return self._normalize_outputs(
            self._compiled_candidate_dense(*inputs)
        )

    def candidate_factorized(
        self,
        inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Execute the compiled factorized ``R`` then ``P`` candidate."""

        self._validate_inputs(inputs)
        return self._normalize_outputs(
            self._compiled_candidate_factorized(*inputs)
        )

    def _normalize_outputs(self, outputs: Any) -> tuple[Any, ...]:
        if (
            len(self.output_port_names) == 1
            and isinstance(outputs, self._core.array)
        ):
            return (outputs,)
        result = tuple(outputs)
        if len(result) != len(self.output_port_names):
            raise RuntimeError(
                "compiled MLX hierarchy output structure changed"
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return non-aliasing public copies of all benchmark state."""

        return {name: value[:] for name, value in self._state}

    def runtime_provenance(self) -> dict[str, object]:
        """Return portable benchmark scope, lineage, and resource facts."""

        return {
            **copy.deepcopy(self._provenance_payload),
            "artifact_sha256": self.artifact_sha256,
            "runtime_kind": _RUNTIME_KIND,
            "weights_updated": False,
            "serialized_artifact": False,
            "authorizes_merge": False,
            "authorizes_pruning": False,
            "authorizes_routing": False,
            "authorizes_mutation": False,
            "authorizes_replacement": False,
            "available_systems": _SYSTEMS,
        }
