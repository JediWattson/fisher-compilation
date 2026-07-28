"""Lag-aware projected JVP estimation for causal modal edges.

This module estimates a stationary, signed modal convolution from exact
Jacobian-vector products.  For a differentiable sequence function ``f`` and
modal source probes ``z`` it evaluates

``J_f(x) [z @ source_decoder.T] @ target_encoder``

with :func:`torch.autograd.functional.jvp`, then fits

``target[t] = sum_lag source[t - lag] @ kernel[lag]``.

Logical positions, rather than tensor offsets, define the lag.  Consequently,
padding and gaps do not accidentally create causal edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import torch
from torch import Tensor


__all__ = [
    "CausalEdgeJVPFit",
    "apply_causal_lag_convolution",
    "estimate_causal_edge_jvp",
    "fit_causal_edge_jvp",
]


_ARTIFACT_KIND = "fisher_graph.causal_edge_jvp_fit"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher_graph.causal_edge_jvp_fit.v1\0"
_CAUSAL_DIRECTION = "source_at_or_before_target"
_LAG_DEFINITION = "target_logical_position_minus_source_logical_position"
_STATIONARITY = "one_shared_matrix_per_nonnegative_logical_lag"
_PROBE_DISTRIBUTION = "independent_rademacher_minus_one_plus_one"
_JVP_BACKEND = "torch.autograd.functional.jvp"


def _strict_keys(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _as_positions_and_mask(
    logical_positions: Tensor,
    valid_mask: Tensor,
    *,
    sequence_length: int,
) -> tuple[Tensor, Tensor, tuple[int, ...], tuple[bool, ...]]:
    if not isinstance(logical_positions, Tensor):
        raise TypeError("logical_positions must be a Tensor")
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a Tensor")
    positions = logical_positions.detach()
    mask = valid_mask.detach()
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if positions.ndim != 1 or positions.shape[0] != sequence_length:
        raise ValueError(
            "logical_positions must have shape [S] or [1, S]"
        )
    if mask.ndim != 1 or mask.shape[0] != sequence_length:
        raise ValueError("valid_mask must have shape [S] or [1, S]")
    if positions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("logical_positions must use an integer dtype")
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must use torch.bool")
    positions_cpu = positions.to(device="cpu", dtype=torch.int64).contiguous()
    mask_cpu = mask.to(device="cpu", dtype=torch.bool).contiguous()
    valid_values = positions_cpu[mask_cpu]
    if valid_values.numel() == 0:
        raise ValueError("valid_mask must select at least one position")
    if valid_values.numel() > 1 and not bool(
        torch.all(valid_values[1:] > valid_values[:-1])
    ):
        raise ValueError(
            "valid logical positions must be strictly increasing in sequence "
            "order"
        )
    position_tuple = tuple(int(value) for value in positions_cpu.tolist())
    mask_tuple = tuple(bool(value) for value in mask_cpu.tolist())
    return positions_cpu, mask_cpu, position_tuple, mask_tuple


def _as_finite_matrix(
    value: Tensor,
    *,
    label: str,
    first_width: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != 2:
        raise ValueError(f"{label} must be rank two")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if first_width is not None and value.shape[0] != first_width:
        raise ValueError(f"{label} first dimension is incompatible")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{label} dimensions must be positive")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value.detach()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CausalEdgeJVPFit:
    """Authenticated stationary causal convolution fitted from projected JVPs.

    ``kernel[lag]`` maps source modal row vectors to target modal row vectors.
    Lag zero is included, and lag is always measured in logical-position
    units rather than physical tensor offsets.
    """

    kernel: Tensor
    max_lag: int
    probe_count: int
    probe_seed: int
    ridge: float
    baseline_source_sha256: str
    source_decoder_sha256: str
    target_encoder_sha256: str
    fit_logical_positions: tuple[int, ...]
    fit_valid_mask: tuple[bool, ...]
    design_row_count: int
    design_column_count: int
    design_rank: int
    design_largest_singular_value: float
    design_smallest_retained_singular_value: float
    output_frobenius: float
    output_residual_frobenius: float
    relative_output_residual: float
    design_normal_equation_residual_frobenius: float
    causal_direction: str = _CAUSAL_DIRECTION
    lag_definition: str = _LAG_DEFINITION
    stationarity: str = _STATIONARITY
    probe_distribution: str = _PROBE_DISTRIBUTION
    jvp_backend: str = _JVP_BACKEND
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        kernel = _as_finite_matrix_kernel(self.kernel)
        object.__setattr__(self, "kernel", kernel)
        max_lag = _require_nonnegative_int(self.max_lag, label="max_lag")
        if kernel.shape[0] != max_lag + 1:
            raise ValueError("kernel lag dimension must equal max_lag + 1")
        probe_count = _require_positive_int(
            self.probe_count,
            label="probe_count",
        )
        _require_nonnegative_int(self.probe_seed, label="probe_seed")
        ridge = _require_finite_nonnegative(self.ridge, label="ridge")
        object.__setattr__(self, "ridge", ridge)
        _require_sha256(
            self.baseline_source_sha256,
            label="baseline_source_sha256",
        )
        _require_sha256(
            self.source_decoder_sha256,
            label="source_decoder_sha256",
        )
        _require_sha256(
            self.target_encoder_sha256,
            label="target_encoder_sha256",
        )
        if (
            type(self.fit_logical_positions) is not tuple
            or not self.fit_logical_positions
            or any(type(value) is not int for value in self.fit_logical_positions)
        ):
            raise ValueError(
                "fit_logical_positions must be a nonempty tuple of integers"
            )
        if (
            type(self.fit_valid_mask) is not tuple
            or len(self.fit_valid_mask) != len(self.fit_logical_positions)
            or any(type(value) is not bool for value in self.fit_valid_mask)
        ):
            raise ValueError(
                "fit_valid_mask must be a matching tuple of booleans"
            )
        valid_positions = tuple(
            position
            for position, is_valid in zip(
                self.fit_logical_positions,
                self.fit_valid_mask,
                strict=True,
            )
            if is_valid
        )
        if not valid_positions:
            raise ValueError("fit_valid_mask must contain a valid position")
        if any(
            right <= left
            for left, right in zip(
                valid_positions,
                valid_positions[1:],
            )
        ):
            raise ValueError(
                "fit valid logical positions must be strictly increasing"
            )
        rows = _require_positive_int(
            self.design_row_count,
            label="design_row_count",
        )
        expected_rows = probe_count * len(valid_positions)
        if rows != expected_rows:
            raise ValueError(
                "design_row_count does not match probes times valid positions"
            )
        columns = _require_positive_int(
            self.design_column_count,
            label="design_column_count",
        )
        if columns != (max_lag + 1) * self.source_rank:
            raise ValueError(
                "design_column_count does not match lagged source rank"
            )
        rank = _require_nonnegative_int(self.design_rank, label="design_rank")
        if rank > min(rows, columns):
            raise ValueError("design_rank exceeds the design dimensions")
        for field in (
            "design_largest_singular_value",
            "design_smallest_retained_singular_value",
            "output_frobenius",
            "output_residual_frobenius",
            "relative_output_residual",
            "design_normal_equation_residual_frobenius",
        ):
            object.__setattr__(
                self,
                field,
                _require_finite_nonnegative(getattr(self, field), label=field),
            )
        if (
            self.design_smallest_retained_singular_value
            > self.design_largest_singular_value
        ):
            raise ValueError(
                "smallest retained singular value exceeds the largest"
            )
        if self.causal_direction != _CAUSAL_DIRECTION:
            raise ValueError("causal_direction provenance is invalid")
        if self.lag_definition != _LAG_DEFINITION:
            raise ValueError("lag_definition provenance is invalid")
        if self.stationarity != _STATIONARITY:
            raise ValueError("stationarity provenance is invalid")
        if self.probe_distribution != _PROBE_DISTRIBUTION:
            raise ValueError("probe_distribution provenance is invalid")
        if self.jvp_backend != _JVP_BACKEND:
            raise ValueError("jvp_backend provenance is invalid")
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("causal edge JVP artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("causal edge JVP artifact hash mismatch")

    @property
    def source_rank(self) -> int:
        return int(self.kernel.shape[1])

    @property
    def target_rank(self) -> int:
        return int(self.kernel.shape[2])

    @property
    def valid_position_count(self) -> int:
        return sum(self.fit_valid_mask)

    @property
    def jvp_evaluation_count(self) -> int:
        return self.probe_count

    @property
    def lags(self) -> tuple[int, ...]:
        return tuple(range(self.max_lag + 1))

    @property
    def per_lag_matrices(self) -> dict[int, Tensor]:
        return {
            lag: self.kernel[lag].clone()
            for lag in range(self.max_lag + 1)
        }

    def lag_matrix(self, lag: int) -> Tensor:
        _require_nonnegative_int(lag, label="lag")
        if lag > self.max_lag:
            raise ValueError("lag exceeds max_lag")
        return self.kernel[lag].clone()

    def execute(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return apply_causal_lag_convolution(
            source_modes,
            kernel=self.kernel,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "kernel_sha256": _tensor_sha256(self.kernel),
            "kernel_shape": tuple(int(value) for value in self.kernel.shape),
            "max_lag": self.max_lag,
            "probe_count": self.probe_count,
            "probe_seed": self.probe_seed,
            "ridge": self.ridge,
            "baseline_source_sha256": self.baseline_source_sha256,
            "source_decoder_sha256": self.source_decoder_sha256,
            "target_encoder_sha256": self.target_encoder_sha256,
            "fit_logical_positions": self.fit_logical_positions,
            "fit_valid_mask": self.fit_valid_mask,
            "design_row_count": self.design_row_count,
            "design_column_count": self.design_column_count,
            "design_rank": self.design_rank,
            "design_largest_singular_value": (
                self.design_largest_singular_value
            ),
            "design_smallest_retained_singular_value": (
                self.design_smallest_retained_singular_value
            ),
            "output_frobenius": self.output_frobenius,
            "output_residual_frobenius": self.output_residual_frobenius,
            "relative_output_residual": self.relative_output_residual,
            "design_normal_equation_residual_frobenius": (
                self.design_normal_equation_residual_frobenius
            ),
            "causal_direction": self.causal_direction,
            "lag_definition": self.lag_definition,
            "stationarity": self.stationarity,
            "probe_distribution": self.probe_distribution,
            "jvp_backend": self.jvp_backend,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload())

    def validate_integrity(self) -> None:
        if not bool(torch.isfinite(self.kernel).all()):
            raise ValueError("kernel must remain finite")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("causal edge JVP artifact hash mismatch")

    def validate_binding(
        self,
        *,
        baseline_source: Tensor,
        source_decoder: Tensor,
        target_encoder: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """Authenticate the exact tensors and logical grid used for this fit.

        A kernel digest alone does not prove that a caller is using the same
        linearization point or modal bases.  This check intentionally accepts
        those original objects and recomputes this artifact's own hashes,
        avoiding unsafe comparisons with tensor hashes from another module.
        """

        self.validate_integrity()
        if not isinstance(baseline_source, Tensor):
            raise TypeError("baseline_source must be a Tensor")
        if baseline_source.ndim != 3 or baseline_source.shape[0] != 1:
            raise ValueError("baseline_source must have shape [1, S, d_in]")
        if (
            not baseline_source.is_floating_point()
            or not bool(torch.isfinite(baseline_source).all())
            or baseline_source.shape[1] <= 0
            or baseline_source.shape[2] <= 0
        ):
            raise ValueError("baseline_source must be finite floating data")
        sequence_length = int(baseline_source.shape[1])
        input_width = int(baseline_source.shape[2])
        _, _, position_tuple, mask_tuple = _as_positions_and_mask(
            logical_positions,
            valid_mask,
            sequence_length=sequence_length,
        )
        decoder = _as_finite_matrix(
            source_decoder,
            label="source_decoder",
            first_width=input_width,
        )
        encoder = _as_finite_matrix(
            target_encoder,
            label="target_encoder",
        )
        if decoder.shape[1] != self.source_rank:
            raise ValueError("source_decoder rank does not match the JVP fit")
        if encoder.shape[1] != self.target_rank:
            raise ValueError("target_encoder rank does not match the JVP fit")
        bindings = (
            (
                _tensor_sha256(baseline_source),
                self.baseline_source_sha256,
                "baseline_source",
            ),
            (
                _tensor_sha256(decoder),
                self.source_decoder_sha256,
                "source_decoder",
            ),
            (
                _tensor_sha256(encoder),
                self.target_encoder_sha256,
                "target_encoder",
            ),
        )
        for actual, expected, label in bindings:
            if actual != expected:
                raise ValueError(f"{label} does not match the JVP fit")
        if position_tuple != self.fit_logical_positions:
            raise ValueError("logical_positions do not match the JVP fit")
        if mask_tuple != self.fit_valid_mask:
            raise ValueError("valid_mask does not match the JVP fit")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "lags": self.lags,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "valid_position_count": self.valid_position_count,
            "jvp_evaluation_count": self.jvp_evaluation_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "kernel": self.kernel.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CausalEdgeJVPFit:
        expected = {
            "artifact_kind",
            "format_version",
            "kernel_sha256",
            "kernel_shape",
            "kernel",
            "max_lag",
            "probe_count",
            "probe_seed",
            "ridge",
            "baseline_source_sha256",
            "source_decoder_sha256",
            "target_encoder_sha256",
            "fit_logical_positions",
            "fit_valid_mask",
            "design_row_count",
            "design_column_count",
            "design_rank",
            "design_largest_singular_value",
            "design_smallest_retained_singular_value",
            "output_frobenius",
            "output_residual_frobenius",
            "relative_output_residual",
            "design_normal_equation_residual_frobenius",
            "causal_direction",
            "lag_definition",
            "stationarity",
            "probe_distribution",
            "jvp_backend",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="causal edge JVP fit")
        kernel = state["kernel"]
        if not isinstance(kernel, Tensor):
            raise TypeError("serialized kernel must be a Tensor")
        expected_shape = state["kernel_shape"]
        if (
            not isinstance(expected_shape, tuple)
            or tuple(kernel.shape) != expected_shape
        ):
            raise ValueError("serialized kernel shape drifted")
        if _tensor_sha256(kernel) != state["kernel_sha256"]:
            raise ValueError("serialized kernel hash mismatch")
        return cls(
            kernel=kernel,
            max_lag=state["max_lag"],  # type: ignore[arg-type]
            probe_count=state["probe_count"],  # type: ignore[arg-type]
            probe_seed=state["probe_seed"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            baseline_source_sha256=state[
                "baseline_source_sha256"
            ],  # type: ignore[arg-type]
            source_decoder_sha256=state[
                "source_decoder_sha256"
            ],  # type: ignore[arg-type]
            target_encoder_sha256=state[
                "target_encoder_sha256"
            ],  # type: ignore[arg-type]
            fit_logical_positions=state[
                "fit_logical_positions"
            ],  # type: ignore[arg-type]
            fit_valid_mask=state["fit_valid_mask"],  # type: ignore[arg-type]
            design_row_count=state[
                "design_row_count"
            ],  # type: ignore[arg-type]
            design_column_count=state[
                "design_column_count"
            ],  # type: ignore[arg-type]
            design_rank=state["design_rank"],  # type: ignore[arg-type]
            design_largest_singular_value=state[
                "design_largest_singular_value"
            ],  # type: ignore[arg-type]
            design_smallest_retained_singular_value=state[
                "design_smallest_retained_singular_value"
            ],  # type: ignore[arg-type]
            output_frobenius=state[
                "output_frobenius"
            ],  # type: ignore[arg-type]
            output_residual_frobenius=state[
                "output_residual_frobenius"
            ],  # type: ignore[arg-type]
            relative_output_residual=state[
                "relative_output_residual"
            ],  # type: ignore[arg-type]
            design_normal_equation_residual_frobenius=state[
                "design_normal_equation_residual_frobenius"
            ],  # type: ignore[arg-type]
            causal_direction=state["causal_direction"],  # type: ignore[arg-type]
            lag_definition=state["lag_definition"],  # type: ignore[arg-type]
            stationarity=state["stationarity"],  # type: ignore[arg-type]
            probe_distribution=state[
                "probe_distribution"
            ],  # type: ignore[arg-type]
            jvp_backend=state["jvp_backend"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def _as_finite_matrix_kernel(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("kernel must be a Tensor")
    if value.ndim != 3:
        raise ValueError("kernel must have shape [max_lag + 1, r_in, r_out]")
    if not value.is_floating_point():
        raise TypeError("kernel must use a floating dtype")
    if any(int(width) <= 0 for width in value.shape):
        raise ValueError("kernel dimensions must be positive")
    kernel = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(kernel).all()):
        raise ValueError("kernel must be finite")
    return kernel.clone()


def _lagged_design(
    source_modes: Tensor,
    *,
    positions: Tensor,
    mask: Tensor,
    max_lag: int,
) -> Tensor:
    """Return one row per valid target and one block per causal lag."""

    position_to_index = {
        int(positions[index]): int(index)
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()
    }
    zero = torch.zeros(
        source_modes.shape[1],
        dtype=source_modes.dtype,
        device=source_modes.device,
    )
    rows: list[Tensor] = []
    for target_index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        target_position = int(positions[target_index])
        blocks = []
        for lag in range(max_lag + 1):
            source_index = position_to_index.get(target_position - lag)
            blocks.append(
                zero if source_index is None else source_modes[source_index]
            )
        rows.append(torch.cat(blocks, dim=0))
    return torch.stack(rows, dim=0)


def apply_causal_lag_convolution(
    source_modes: Tensor,
    *,
    kernel: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Apply ``kernel`` using nonnegative logical-position lags.

    ``source_modes`` may have shape ``[S, r_in]`` or ``[B, S, r_in]``.  Invalid
    target rows are returned as zeros, and invalid source rows never
    contribute.
    """

    if not isinstance(source_modes, Tensor):
        raise TypeError("source_modes must be a Tensor")
    if source_modes.ndim not in (2, 3):
        raise ValueError("source_modes must have shape [S, r] or [B, S, r]")
    if not source_modes.is_floating_point():
        raise TypeError("source_modes must use a floating dtype")
    if not bool(torch.isfinite(source_modes).all()):
        raise ValueError("source_modes must be finite")
    squeeze = source_modes.ndim == 2
    batched = source_modes.unsqueeze(0) if squeeze else source_modes
    sequence_length = int(batched.shape[1])
    positions, mask, _, _ = _as_positions_and_mask(
        logical_positions,
        valid_mask,
        sequence_length=sequence_length,
    )
    kernel_checked = _as_finite_matrix_kernel(kernel).to(
        device=source_modes.device,
        dtype=source_modes.dtype,
    )
    if batched.shape[2] != kernel_checked.shape[1]:
        raise ValueError("source modal width does not match kernel source rank")
    result = torch.zeros(
        (batched.shape[0], sequence_length, kernel_checked.shape[2]),
        dtype=source_modes.dtype,
        device=source_modes.device,
    )
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    position_to_index = {
        int(positions[index]): int(index) for index in valid_indices
    }
    for target_index in valid_indices:
        target_position = int(positions[target_index])
        value = result[:, target_index, :]
        for lag in range(kernel_checked.shape[0]):
            source_index = position_to_index.get(target_position - lag)
            if source_index is not None:
                value = value + (
                    batched[:, source_index, :] @ kernel_checked[lag]
                )
        result[:, target_index, :] = value
    return result[0] if squeeze else result


def estimate_causal_edge_jvp(
    function: Callable[[Tensor], Tensor],
    *,
    baseline_source: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    source_decoder: Tensor,
    target_encoder: Tensor,
    max_lag: int,
    probe_count: int,
    probe_seed: int,
    ridge: float,
) -> CausalEdgeJVPFit:
    """Estimate a stationary signed causal modal edge from exact JVPs.

    The function must map ``[1, S, d_in]`` to ``[1, S, d_out]``.  One exact
    JVP is evaluated for each deterministic modal-sequence Rademacher probe.
    Regression is performed in CPU float64.  ``ridge`` means an unscaled
    ``ridge * I`` addition to the design Gram matrix.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(baseline_source, Tensor):
        raise TypeError("baseline_source must be a Tensor")
    if baseline_source.ndim != 3 or baseline_source.shape[0] != 1:
        raise ValueError("baseline_source must have shape [1, S, d_in]")
    if not baseline_source.is_floating_point():
        raise TypeError("baseline_source must use a floating dtype")
    if not bool(torch.isfinite(baseline_source).all()):
        raise ValueError("baseline_source must be finite")
    if baseline_source.shape[1] == 0 or baseline_source.shape[2] == 0:
        raise ValueError("baseline_source dimensions must be positive")
    sequence_length = int(baseline_source.shape[1])
    input_width = int(baseline_source.shape[2])
    positions, mask, position_tuple, mask_tuple = _as_positions_and_mask(
        logical_positions,
        valid_mask,
        sequence_length=sequence_length,
    )
    decoder = _as_finite_matrix(
        source_decoder,
        label="source_decoder",
        first_width=input_width,
    )
    encoder = _as_finite_matrix(
        target_encoder,
        label="target_encoder",
    )
    max_lag = _require_nonnegative_int(max_lag, label="max_lag")
    probe_count = _require_positive_int(probe_count, label="probe_count")
    probe_seed = _require_nonnegative_int(probe_seed, label="probe_seed")
    ridge = _require_finite_nonnegative(ridge, label="ridge")
    source_rank = int(decoder.shape[1])
    target_rank = int(encoder.shape[1])
    baseline = baseline_source.detach().clone()
    decoder_device = decoder.to(
        device=baseline.device,
        dtype=baseline.dtype,
    )
    encoder_device = encoder.to(
        device=baseline.device,
        dtype=baseline.dtype,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(probe_seed)
    design_parts: list[Tensor] = []
    output_parts: list[Tensor] = []
    valid_indices_device = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten().to(device=baseline.device)
    for _ in range(probe_count):
        random_bits = torch.randint(
            0,
            2,
            (sequence_length, source_rank),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )
        source_modes = (random_bits * 2 - 1).to(
            device=baseline.device,
            dtype=baseline.dtype,
        )
        source_modes = source_modes * mask.to(
            device=baseline.device,
            dtype=baseline.dtype,
        ).unsqueeze(1)
        perturbation = (source_modes @ decoder_device.transpose(0, 1)).unsqueeze(
            0
        )
        primal_output, output_jvp = torch.autograd.functional.jvp(
            function,
            (baseline,),
            (perturbation,),
            create_graph=False,
            strict=False,
        )
        for label, value in (
            ("function output", primal_output),
            ("function JVP output", output_jvp),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{label} must be a Tensor")
            if value.ndim != 3 or value.shape[:2] != (1, sequence_length):
                raise ValueError(
                    f"{label} must have shape [1, S, d_out]"
                )
            if value.shape[2] != encoder.shape[0]:
                raise ValueError(
                    f"{label} width does not match target_encoder"
                )
            if not value.is_floating_point():
                raise TypeError(f"{label} must use a floating dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{label} must be finite")
        projected = output_jvp[0] @ encoder_device
        design_parts.append(
            _lagged_design(
                source_modes,
                positions=positions,
                mask=mask,
                max_lag=max_lag,
            ).to(device="cpu", dtype=torch.float64)
        )
        output_parts.append(
            projected.index_select(0, valid_indices_device)
            .to(device="cpu", dtype=torch.float64)
        )
    design = torch.cat(design_parts, dim=0)
    outputs = torch.cat(output_parts, dim=0)
    singular_values = torch.linalg.svdvals(design)
    tolerance = (
        max(design.shape)
        * torch.finfo(design.dtype).eps
        * (
            float(singular_values[0])
            if singular_values.numel() > 0
            else 0.0
        )
    )
    retained = singular_values[singular_values > tolerance]
    design_rank = int(retained.numel())
    largest = (
        float(singular_values[0])
        if singular_values.numel() > 0
        else 0.0
    )
    smallest_retained = float(retained[-1]) if design_rank > 0 else 0.0
    gram = design.transpose(0, 1) @ design
    right_hand_side = design.transpose(0, 1) @ outputs
    if ridge > 0.0:
        coefficients = torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
            right_hand_side,
        )
    else:
        coefficients = torch.linalg.lstsq(design, outputs).solution
    prediction = design @ coefficients
    residual = prediction - outputs
    output_frobenius = float(torch.linalg.vector_norm(outputs))
    output_residual = float(torch.linalg.vector_norm(residual))
    denominator = max(output_frobenius, torch.finfo(torch.float64).eps)
    regularized_normal_residual = (
        (gram @ coefficients - right_hand_side)
        + ridge * coefficients
    )
    kernel = coefficients.reshape(max_lag + 1, source_rank, target_rank)
    return CausalEdgeJVPFit(
        kernel=kernel,
        max_lag=max_lag,
        probe_count=probe_count,
        probe_seed=probe_seed,
        ridge=ridge,
        baseline_source_sha256=_tensor_sha256(baseline_source),
        source_decoder_sha256=_tensor_sha256(decoder),
        target_encoder_sha256=_tensor_sha256(encoder),
        fit_logical_positions=position_tuple,
        fit_valid_mask=mask_tuple,
        design_row_count=int(design.shape[0]),
        design_column_count=int(design.shape[1]),
        design_rank=design_rank,
        design_largest_singular_value=largest,
        design_smallest_retained_singular_value=smallest_retained,
        output_frobenius=output_frobenius,
        output_residual_frobenius=output_residual,
        relative_output_residual=output_residual / denominator,
        design_normal_equation_residual_frobenius=float(
            torch.linalg.vector_norm(regularized_normal_residual)
        ),
    )


fit_causal_edge_jvp = estimate_causal_edge_jvp
