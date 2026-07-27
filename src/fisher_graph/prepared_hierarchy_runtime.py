"""Validate-once Torch controls for one connectivity decomposition.

This module deliberately separates artifact verification from timed tensor
execution.  :class:`PreparedTorchHierarchyRuntime` validates its source
decomposition once, copies every runtime tensor to one Torch device and dtype,
and folds candidate centering into one output bias per factor.

The prepared object is an analysis and benchmarking bundle.  Its resident
dense source and dense-candidate controls are not an authorized fallback, and
its accounting makes no replacement, deployed-storage-reduction, or latency
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor

from .modal_connectivity_modes import ModalConnectivityDecomposition


__all__ = [
    "PreparedHierarchyAccounting",
    "PreparedTorchHierarchyRuntime",
]


TensorTuple: TypeAlias = tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class PreparedHierarchyAccounting:
    """Exact tensor-scalar and linear-MAC counts for the prepared controls.

    Stored-scalar counts include the folded output bias used by a path.  MAC
    counts cover only the dense linear maps for one leading-dimension row;
    concatenation, bias addition, allocation, and dispatch are intentionally
    outside that conventional count.
    """

    source_dense_stored_scalar_count: int
    source_dense_macs_per_row: int
    candidate_dense_stored_scalar_count: int
    candidate_dense_macs_per_row: int
    candidate_factorized_stored_scalar_count: int
    candidate_factorized_macs_per_row: int
    prepared_unique_stored_scalar_count: int
    bytes_per_scalar: int
    prepared_unique_storage_bytes: int
    analysis_only: bool = True
    source_dense_is_benchmark_control: bool = True
    authorizes_replacement: bool = False
    authorizes_fallback: bool = False
    deployed_storage_reduction_authorized: bool = False
    latency_reduction_claim: bool = False

    def __post_init__(self) -> None:
        count_fields = (
            "source_dense_stored_scalar_count",
            "source_dense_macs_per_row",
            "candidate_dense_stored_scalar_count",
            "candidate_dense_macs_per_row",
            "candidate_factorized_stored_scalar_count",
            "candidate_factorized_macs_per_row",
            "prepared_unique_stored_scalar_count",
            "bytes_per_scalar",
            "prepared_unique_storage_bytes",
        )
        for name in count_fields:
            value = getattr(self, name)
            minimum = 1 if name == "bytes_per_scalar" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.prepared_unique_storage_bytes != (
            self.prepared_unique_stored_scalar_count * self.bytes_per_scalar
        ):
            raise ValueError("prepared byte accounting is inconsistent")
        if (
            self.analysis_only is not True
            or self.source_dense_is_benchmark_control is not True
            or self.authorizes_replacement is not False
            or self.authorizes_fallback is not False
            or self.deployed_storage_reduction_authorized is not False
            or self.latency_reduction_claim is not False
        ):
            raise ValueError(
                "prepared hierarchy accounting is analysis-only and cannot "
                "authorize replacement, fallback, storage, or latency claims"
            )


def _copy_runtime_tensor(
    value: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Make one independent contiguous runtime copy during preparation."""

    return (
        value.detach()
        .to(device=device, dtype=dtype)
        .contiguous()
        .clone()
    )


@dataclass(frozen=True, slots=True, init=False)
class PreparedTorchHierarchyRuntime:
    """Device-resident source and factorized benchmark controls.

    Inputs are canonical tuples ordered exactly like
    ``decomposition.source_transfer.input_ports``.  Outputs are tuples ordered
    like its output ports.  The three execution methods perform no artifact
    validation, hashing, or device/dtype conversion.
    """

    source_transfer_sha256: str
    decomposition_sha256: str
    device: torch.device
    dtype: torch.dtype
    input_names: tuple[str, ...]
    input_widths: tuple[int, ...]
    output_names: tuple[str, ...]
    output_widths: tuple[int, ...]
    retained_ranks: tuple[int, ...]
    accounting: PreparedHierarchyAccounting
    _prefix_input_indices: tuple[tuple[int, ...], ...]
    _source_matrices: TensorTuple
    _source_biases: TensorTuple
    _candidate_dense_matrices: TensorTuple
    _candidate_restrictions: TensorTuple
    _candidate_prolongations: TensorTuple
    _candidate_biases: TensorTuple

    @classmethod
    def from_decomposition(
        cls,
        decomposition: ModalConnectivityDecomposition,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedTorchHierarchyRuntime:
        """Validate and prepare one decomposition for Torch execution."""

        return cls(
            decomposition,
            device=device,
            dtype=dtype,
        )

    def __init__(
        self,
        decomposition: ModalConnectivityDecomposition,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(decomposition, ModalConnectivityDecomposition):
            raise TypeError(
                "decomposition must be a ModalConnectivityDecomposition"
            )
        if dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError("dtype must be a supported floating Torch dtype")
        try:
            runtime_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a Torch device") from error

        # This is the sole artifact-validation boundary.  No reference to the
        # mutable decomposition is retained after its tensors are copied.
        decomposition.validate_integrity()
        transfer = decomposition.source_transfer

        source_matrices = tuple(
            _copy_runtime_tensor(
                matrix,
                device=runtime_device,
                dtype=dtype,
            )
            for matrix in transfer.transfer_matrices
        )
        source_biases = tuple(
            _copy_runtime_tensor(
                offset,
                device=runtime_device,
                dtype=dtype,
            )
            for offset in transfer.affine_offsets
        )

        restrictions: list[Tensor] = []
        prolongations: list[Tensor] = []
        dense_matrices: list[Tensor] = []
        candidate_biases: list[Tensor] = []
        for factor in decomposition.factors:
            restriction = _copy_runtime_tensor(
                factor.restriction,
                device=runtime_device,
                dtype=dtype,
            )
            prolongation = _copy_runtime_tensor(
                factor.prolongation,
                device=runtime_device,
                dtype=dtype,
            )
            input_mean = _copy_runtime_tensor(
                factor.input_mean,
                device=runtime_device,
                dtype=dtype,
            )
            output_mean = _copy_runtime_tensor(
                factor.output_mean,
                device=runtime_device,
                dtype=dtype,
            )
            dense_matrix = (prolongation @ restriction).contiguous()
            folded_bias = (
                output_mean - dense_matrix @ input_mean
            ).contiguous()
            restrictions.append(restriction)
            prolongations.append(prolongation)
            dense_matrices.append(dense_matrix)
            candidate_biases.append(folded_bias)

        input_names = tuple(port.name for port in transfer.input_ports)
        input_index = {
            name: index for index, name in enumerate(input_names)
        }
        prefix_input_indices = tuple(
            tuple(input_index[name] for name in prefix)
            for prefix in transfer.input_prefixes
        )
        restrictions_tuple = tuple(restrictions)
        prolongations_tuple = tuple(prolongations)
        dense_matrices_tuple = tuple(dense_matrices)
        candidate_biases_tuple = tuple(candidate_biases)

        source_stored = sum(
            matrix.numel() + bias.numel()
            for matrix, bias in zip(
                source_matrices,
                source_biases,
                strict=True,
            )
        )
        source_macs = sum(matrix.numel() for matrix in source_matrices)
        candidate_dense_stored = sum(
            matrix.numel() + bias.numel()
            for matrix, bias in zip(
                dense_matrices_tuple,
                candidate_biases_tuple,
                strict=True,
            )
        )
        candidate_dense_macs = sum(
            matrix.numel() for matrix in dense_matrices_tuple
        )
        candidate_factorized_stored = sum(
            restriction.numel() + prolongation.numel() + bias.numel()
            for restriction, prolongation, bias in zip(
                restrictions_tuple,
                prolongations_tuple,
                candidate_biases_tuple,
                strict=True,
            )
        )
        candidate_factorized_macs = sum(
            restriction.numel() + prolongation.numel()
            for restriction, prolongation in zip(
                restrictions_tuple,
                prolongations_tuple,
                strict=True,
            )
        )
        # Candidate biases are shared by the dense and factorized controls.
        prepared_unique_stored = (
            source_stored
            + candidate_factorized_stored
            + sum(matrix.numel() for matrix in dense_matrices_tuple)
        )
        bytes_per_scalar = torch.empty((), dtype=dtype).element_size()
        accounting = PreparedHierarchyAccounting(
            source_dense_stored_scalar_count=source_stored,
            source_dense_macs_per_row=source_macs,
            candidate_dense_stored_scalar_count=candidate_dense_stored,
            candidate_dense_macs_per_row=candidate_dense_macs,
            candidate_factorized_stored_scalar_count=(
                candidate_factorized_stored
            ),
            candidate_factorized_macs_per_row=candidate_factorized_macs,
            prepared_unique_stored_scalar_count=prepared_unique_stored,
            bytes_per_scalar=bytes_per_scalar,
            prepared_unique_storage_bytes=(
                prepared_unique_stored * bytes_per_scalar
            ),
        )

        object.__setattr__(
            self,
            "source_transfer_sha256",
            transfer.artifact_sha256,
        )
        object.__setattr__(
            self,
            "decomposition_sha256",
            decomposition.artifact_sha256,
        )
        object.__setattr__(self, "device", runtime_device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(
            self,
            "input_widths",
            tuple(port.width for port in transfer.input_ports),
        )
        object.__setattr__(
            self,
            "output_names",
            tuple(port.name for port in transfer.output_ports),
        )
        object.__setattr__(
            self,
            "output_widths",
            tuple(port.width for port in transfer.output_ports),
        )
        object.__setattr__(
            self,
            "retained_ranks",
            tuple(factor.retained_rank for factor in decomposition.factors),
        )
        object.__setattr__(self, "accounting", accounting)
        object.__setattr__(
            self,
            "_prefix_input_indices",
            prefix_input_indices,
        )
        object.__setattr__(self, "_source_matrices", source_matrices)
        object.__setattr__(self, "_source_biases", source_biases)
        object.__setattr__(
            self,
            "_candidate_dense_matrices",
            dense_matrices_tuple,
        )
        object.__setattr__(
            self,
            "_candidate_restrictions",
            restrictions_tuple,
        )
        object.__setattr__(
            self,
            "_candidate_prolongations",
            prolongations_tuple,
        )
        object.__setattr__(
            self,
            "_candidate_biases",
            candidate_biases_tuple,
        )

    def _canonical_inputs(self, inputs: TensorTuple) -> TensorTuple:
        if type(inputs) is not tuple:
            raise TypeError("prepared inputs must be a canonical Tensor tuple")
        if len(inputs) != len(self.input_widths):
            raise ValueError("prepared input count does not match the boundary")
        leading_shape: tuple[int, ...] | None = None
        for index, (value, width) in enumerate(
            zip(inputs, self.input_widths, strict=True)
        ):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"prepared input {index} must be floating Tensor")
            if value.shape[-1:] != (width,):
                raise ValueError(f"prepared input {index} has the wrong width")
            if value.device != self.device:
                raise ValueError(
                    f"prepared input {index} is on the wrong device"
                )
            if value.dtype != self.dtype:
                raise ValueError(
                    f"prepared input {index} has the wrong dtype"
                )
            current_leading = tuple(value.shape[:-1])
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError(
                    "prepared inputs must share the same leading shape"
                )
        return inputs

    def _joined_prefixes(self, inputs: TensorTuple) -> TensorTuple:
        return tuple(
            torch.cat(tuple(inputs[index] for index in indices), dim=-1)
            for indices in self._prefix_input_indices
        )

    def source_dense(self, inputs: TensorTuple) -> TensorTuple:
        """Execute the copied exact dense boundary controls."""

        canonical = self._canonical_inputs(inputs)
        prefixes = self._joined_prefixes(canonical)
        return tuple(
            joined @ matrix.T + bias
            for joined, matrix, bias in zip(
                prefixes,
                self._source_matrices,
                self._source_biases,
                strict=True,
            )
        )

    def candidate_dense(self, inputs: TensorTuple) -> TensorTuple:
        """Execute the materialized ``P @ R`` candidate control."""

        canonical = self._canonical_inputs(inputs)
        prefixes = self._joined_prefixes(canonical)
        return tuple(
            joined @ matrix.T + bias
            for joined, matrix, bias in zip(
                prefixes,
                self._candidate_dense_matrices,
                self._candidate_biases,
                strict=True,
            )
        )

    def candidate_factorized(self, inputs: TensorTuple) -> TensorTuple:
        """Execute retained ``R`` then ``P`` without materializing the product."""

        canonical = self._canonical_inputs(inputs)
        prefixes = self._joined_prefixes(canonical)
        return tuple(
            (joined @ restriction.T) @ prolongation.T + bias
            for joined, restriction, prolongation, bias in zip(
                prefixes,
                self._candidate_restrictions,
                self._candidate_prolongations,
                self._candidate_biases,
                strict=True,
            )
        )
