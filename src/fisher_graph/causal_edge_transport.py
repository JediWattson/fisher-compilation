"""Held-out validation and finite transport diagnostics for causal modal edges.

The primitives in this module deliberately separate four operations:

1. collect exact projected JVPs on a deterministic set of tangent directions;
2. pool one or more collections into a stationary causal lag-kernel fit;
3. evaluate that frozen kernel on direction-disjoint JVP collections; and
4. diagnose finite-displacement curvature with path-integrated exact JVPs.

The implementation is model-agnostic.  Logical positions, rather than tensor
offsets, define causal lags, and all regression and artifact tensors are kept
as CPU float64 values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .causal_edge_jvp import apply_causal_lag_convolution


__all__ = [
    "CausalEdgeJVPBatch",
    "CausalEdgeJVPHeldoutMetrics",
    "PathIntegratedJVPDiagnostic",
    "PooledCausalEdgeJVPFit",
    "collect_causal_edge_jvp_batch",
    "evaluate_pooled_causal_edge_jvp",
    "fit_pooled_causal_edge_jvp",
    "gauss_legendre_unit_interval",
    "integrate_path_jvp",
]


_BATCH_KIND = "fisher_graph.causal_edge_jvp_batch"
_FIT_KIND = "fisher_graph.pooled_causal_edge_jvp_fit"
_HELDOUT_KIND = "fisher_graph.causal_edge_jvp_heldout_metrics"
_PATH_KIND = "fisher_graph.path_integrated_jvp_diagnostic"
_FORMAT_VERSION = 1
_RNG_DOMAIN = b"fisher_graph.causal_edge_transport.rademacher.v1\0"
_DIRECTION_HASH_DOMAIN = b"fisher_graph.causal_edge_transport.direction.v1\0"
_BATCH_HASH_DOMAIN = b"fisher_graph.causal_edge_transport.batch.v1\0"
_FIT_HASH_DOMAIN = b"fisher_graph.causal_edge_transport.fit.v1\0"
_HELDOUT_HASH_DOMAIN = b"fisher_graph.causal_edge_transport.heldout.v1\0"
_PATH_HASH_DOMAIN = b"fisher_graph.causal_edge_transport.path.v1\0"
_PROBE_DISTRIBUTION = "independent_rademacher_minus_one_plus_one"
_JVP_BACKEND = "torch.autograd.functional.jvp"
_CAUSAL_DIRECTION = "source_at_or_before_target"
_LAG_DEFINITION = "target_logical_position_minus_source_logical_position"
_STATIONARITY = "one_shared_matrix_per_nonnegative_logical_lag"
_QUADRATURE_RULE = "gauss_legendre_on_unit_interval"


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


def _require_finite_unit(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < -1.0
        or float(value) > 1.0
    ):
        raise ValueError(f"{label} must be finite and between -1 and 1")
    return float(value)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_domain(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("direction_domain must be a nonempty string")
    encoded = value.encode("utf-8")
    if len(encoded) > 1024:
        raise ValueError("direction_domain must be at most 1024 UTF-8 bytes")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(
    payload: Mapping[str, object],
    *,
    domain: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _as_finite_float_tensor(
    value: Tensor,
    *,
    label: str,
    dimensions: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{label} must be rank {dimensions}")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if any(int(width) <= 0 for width in value.shape):
        raise ValueError(f"{label} dimensions must be positive")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _as_cpu_float64(
    value: Tensor,
    *,
    label: str,
    dimensions: int,
) -> Tensor:
    checked = _as_finite_float_tensor(
        value,
        label=label,
        dimensions=dimensions,
    )
    return (
        checked.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


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
        raise ValueError("logical_positions must have shape [S] or [1, S]")
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
    valid_positions = positions_cpu[mask_cpu]
    if valid_positions.numel() == 0:
        raise ValueError("valid_mask must select at least one position")
    if valid_positions.numel() > 1 and not bool(
        torch.all(valid_positions[1:] > valid_positions[:-1])
    ):
        raise ValueError(
            "valid logical positions must be strictly increasing in sequence "
            "order"
        )
    return (
        positions_cpu,
        mask_cpu,
        tuple(int(value) for value in positions_cpu.tolist()),
        tuple(bool(value) for value in mask_cpu.tolist()),
    )


def _tuples_to_positions_and_mask(
    positions: tuple[int, ...],
    mask: tuple[bool, ...],
) -> tuple[Tensor, Tensor]:
    return (
        torch.tensor(positions, dtype=torch.int64),
        torch.tensor(mask, dtype=torch.bool),
    )


def _validate_position_tuples(
    positions: object,
    mask: object,
    *,
    sequence_length: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    if (
        type(positions) is not tuple
        or len(positions) != sequence_length
        or any(type(value) is not int for value in positions)
    ):
        raise ValueError(
            "logical_positions must be a matching tuple of integers"
        )
    if (
        type(mask) is not tuple
        or len(mask) != sequence_length
        or any(type(value) is not bool for value in mask)
    ):
        raise ValueError("valid_mask must be a matching tuple of booleans")
    valid_positions = tuple(
        position
        for position, is_valid in zip(positions, mask, strict=True)
        if is_valid
    )
    if not valid_positions:
        raise ValueError("valid_mask must select at least one position")
    if any(
        right <= left
        for left, right in zip(valid_positions, valid_positions[1:])
    ):
        raise ValueError(
            "valid logical positions must be strictly increasing in sequence "
            "order"
        )
    return positions, mask


def _validate_function_output(
    value: object,
    *,
    label: str,
    sequence_length: int,
    output_width: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != 3 or value.shape[:2] != (1, sequence_length):
        raise ValueError(f"{label} must have shape [1, S, d_out]")
    if value.shape[2] != output_width:
        raise ValueError(f"{label} width does not match target_encoder")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _direction_seed(direction_domain: str, direction_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(_RNG_DOMAIN)
    encoded = direction_domain.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    encoded_seed = str(direction_seed).encode("ascii")
    digest.update(len(encoded_seed).to_bytes(8, "little"))
    digest.update(encoded_seed)
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _direction_sha256(
    source_modes: Tensor,
    *,
    baseline_source_sha256: str,
    source_decoder_sha256: str,
    logical_positions: tuple[int, ...],
    valid_mask: tuple[bool, ...],
) -> str:
    return _json_sha256(
        {
            "baseline_source_sha256": baseline_source_sha256,
            "source_decoder_sha256": source_decoder_sha256,
            "source_modes_sha256": _tensor_sha256(source_modes),
            "logical_positions": logical_positions,
            "valid_mask": valid_mask,
        },
        domain=_DIRECTION_HASH_DOMAIN,
    )


def _lagged_design(
    source_modes: Tensor,
    *,
    positions: Tensor,
    mask: Tensor,
    max_lag: int,
) -> Tensor:
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
        rows.append(
            torch.cat(
                [
                    (
                        zero
                        if position_to_index.get(target_position - lag) is None
                        else source_modes[
                            position_to_index[target_position - lag]
                        ]
                    )
                    for lag in range(max_lag + 1)
                ],
                dim=0,
            )
        )
    return torch.stack(rows, dim=0)


def _cosine(first: Tensor, second: Tensor) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_norm = float(torch.linalg.vector_norm(first_flat))
    second_norm = float(torch.linalg.vector_norm(second_flat))
    epsilon = torch.finfo(torch.float64).eps
    if first_norm <= epsilon and second_norm <= epsilon:
        return 1.0
    if first_norm <= epsilon or second_norm <= epsilon:
        return 0.0
    value = float(torch.dot(first_flat, second_flat)) / (
        first_norm * second_norm
    )
    return max(-1.0, min(1.0, value))


def _quantile(values: Sequence[float], quantile: float) -> float:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return float(torch.quantile(tensor, quantile))


@dataclass(frozen=True, slots=True)
class CausalEdgeJVPBatch:
    """Exact projected JVP observations for deterministic modal directions."""

    source_modes: Tensor
    target_jvps: Tensor
    logical_positions: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    direction_seed: int
    direction_domain: str
    direction_sha256s: tuple[str, ...]
    baseline_source_sha256: str
    source_decoder_sha256: str
    target_encoder_sha256: str
    probe_distribution: str = _PROBE_DISTRIBUTION
    jvp_backend: str = _JVP_BACKEND
    artifact_sha256: str = ""
    artifact_kind: str = _BATCH_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        source_modes = _as_cpu_float64(
            self.source_modes,
            label="source_modes",
            dimensions=3,
        )
        target_jvps = _as_cpu_float64(
            self.target_jvps,
            label="target_jvps",
            dimensions=3,
        )
        if source_modes.shape[:2] != target_jvps.shape[:2]:
            raise ValueError(
                "source_modes and target_jvps must share direction and "
                "sequence dimensions"
            )
        object.__setattr__(self, "source_modes", source_modes)
        object.__setattr__(self, "target_jvps", target_jvps)
        positions, mask = _validate_position_tuples(
            self.logical_positions,
            self.valid_mask,
            sequence_length=int(source_modes.shape[1]),
        )
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        valid_values = source_modes[:, mask_tensor, :]
        if not bool(torch.all(torch.abs(valid_values) == 1.0)):
            raise ValueError(
                "valid source mode entries must be Rademacher values"
            )
        if bool((~mask_tensor).any()) and not bool(
            torch.all(source_modes[:, ~mask_tensor, :] == 0.0)
        ):
            raise ValueError("invalid source mode rows must be zero")
        _require_nonnegative_int(self.direction_seed, label="direction_seed")
        object.__setattr__(
            self,
            "direction_domain",
            _require_domain(self.direction_domain),
        )
        for field in (
            "baseline_source_sha256",
            "source_decoder_sha256",
            "target_encoder_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        if (
            type(self.direction_sha256s) is not tuple
            or len(self.direction_sha256s) != self.direction_count
        ):
            raise ValueError(
                "direction_sha256s must match the direction dimension"
            )
        for index, digest in enumerate(self.direction_sha256s):
            _require_sha256(digest, label=f"direction_sha256s[{index}]")
            computed = _direction_sha256(
                source_modes[index],
                baseline_source_sha256=self.baseline_source_sha256,
                source_decoder_sha256=self.source_decoder_sha256,
                logical_positions=positions,
                valid_mask=mask,
            )
            if computed != digest:
                raise ValueError("direction hash does not match source_modes")
        if self.probe_distribution != _PROBE_DISTRIBUTION:
            raise ValueError("probe_distribution provenance is invalid")
        if self.jvp_backend != _JVP_BACKEND:
            raise ValueError("jvp_backend provenance is invalid")
        if (
            self.artifact_kind != _BATCH_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("causal edge JVP batch header is invalid")
        computed_artifact = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed_artifact)
        elif self.artifact_sha256 != computed_artifact:
            raise ValueError("causal edge JVP batch hash mismatch")

    @property
    def direction_count(self) -> int:
        return int(self.source_modes.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.source_modes.shape[1])

    @property
    def source_rank(self) -> int:
        return int(self.source_modes.shape[2])

    @property
    def target_rank(self) -> int:
        return int(self.target_jvps.shape[2])

    @property
    def valid_position_count(self) -> int:
        return sum(self.valid_mask)

    @property
    def jvp_evaluation_count(self) -> int:
        return self.direction_count

    @property
    def direction_hashes(self) -> tuple[str, ...]:
        return self.direction_sha256s

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_modes_sha256": _tensor_sha256(self.source_modes),
            "target_jvps_sha256": _tensor_sha256(self.target_jvps),
            "source_modes_shape": tuple(int(v) for v in self.source_modes.shape),
            "target_jvps_shape": tuple(int(v) for v in self.target_jvps.shape),
            "logical_positions": self.logical_positions,
            "valid_mask": self.valid_mask,
            "direction_seed": self.direction_seed,
            "direction_domain": self.direction_domain,
            "direction_sha256s": self.direction_sha256s,
            "baseline_source_sha256": self.baseline_source_sha256,
            "source_decoder_sha256": self.source_decoder_sha256,
            "target_encoder_sha256": self.target_encoder_sha256,
            "probe_distribution": self.probe_distribution,
            "jvp_backend": self.jvp_backend,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_BATCH_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        if (
            not bool(torch.isfinite(self.source_modes).all())
            or not bool(torch.isfinite(self.target_jvps).all())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise ValueError("causal edge JVP batch integrity check failed")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "direction_count": self.direction_count,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "valid_position_count": self.valid_position_count,
        }


@dataclass(frozen=True, slots=True)
class PooledCausalEdgeJVPFit:
    """Authenticated stationary lag kernel fitted across JVP batches."""

    kernel: Tensor
    max_lag: int
    ridge: float
    fit_batch_sha256s: tuple[str, ...]
    fit_direction_sha256s: tuple[str, ...]
    source_decoder_sha256: str
    target_encoder_sha256: str
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
    artifact_sha256: str = ""
    artifact_kind: str = _FIT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        kernel = _as_cpu_float64(
            self.kernel,
            label="kernel",
            dimensions=3,
        )
        object.__setattr__(self, "kernel", kernel)
        max_lag = _require_nonnegative_int(self.max_lag, label="max_lag")
        if kernel.shape[0] != max_lag + 1:
            raise ValueError("kernel lag dimension must equal max_lag + 1")
        object.__setattr__(
            self,
            "ridge",
            _require_finite_nonnegative(self.ridge, label="ridge"),
        )
        if (
            type(self.fit_batch_sha256s) is not tuple
            or not self.fit_batch_sha256s
        ):
            raise ValueError("fit_batch_sha256s must be a nonempty tuple")
        if (
            type(self.fit_direction_sha256s) is not tuple
            or not self.fit_direction_sha256s
        ):
            raise ValueError("fit_direction_sha256s must be a nonempty tuple")
        for field, values in (
            ("fit_batch_sha256s", self.fit_batch_sha256s),
            ("fit_direction_sha256s", self.fit_direction_sha256s),
        ):
            for index, digest in enumerate(values):
                _require_sha256(digest, label=f"{field}[{index}]")
        _require_sha256(
            self.source_decoder_sha256,
            label="source_decoder_sha256",
        )
        _require_sha256(
            self.target_encoder_sha256,
            label="target_encoder_sha256",
        )
        rows = _require_positive_int(
            self.design_row_count,
            label="design_row_count",
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
        if (
            self.artifact_kind != _FIT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("pooled causal edge JVP fit header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("pooled causal edge JVP fit hash mismatch")

    @property
    def source_rank(self) -> int:
        return int(self.kernel.shape[1])

    @property
    def target_rank(self) -> int:
        return int(self.kernel.shape[2])

    @property
    def lags(self) -> tuple[int, ...]:
        return tuple(range(self.max_lag + 1))

    @property
    def fit_direction_count(self) -> int:
        return len(self.fit_direction_sha256s)

    @property
    def jvp_evaluation_count(self) -> int:
        return self.fit_direction_count

    @property
    def fit_direction_hashes(self) -> tuple[str, ...]:
        return self.fit_direction_sha256s

    def execute(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        self.validate_integrity()
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
            "ridge": self.ridge,
            "fit_batch_sha256s": self.fit_batch_sha256s,
            "fit_direction_sha256s": self.fit_direction_sha256s,
            "source_decoder_sha256": self.source_decoder_sha256,
            "target_encoder_sha256": self.target_encoder_sha256,
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
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_FIT_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        if (
            not bool(torch.isfinite(self.kernel).all())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise ValueError("pooled causal edge JVP fit integrity check failed")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "lags": self.lags,
            "fit_direction_count": self.fit_direction_count,
            "unique_fit_direction_count": len(
                set(self.fit_direction_sha256s)
            ),
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
    ) -> PooledCausalEdgeJVPFit:
        expected = {
            "artifact_kind",
            "format_version",
            "kernel_sha256",
            "kernel_shape",
            "kernel",
            "max_lag",
            "ridge",
            "fit_batch_sha256s",
            "fit_direction_sha256s",
            "source_decoder_sha256",
            "target_encoder_sha256",
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
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="pooled causal edge JVP fit",
        )
        kernel = state["kernel"]
        if not isinstance(kernel, Tensor):
            raise TypeError("serialized kernel must be a Tensor")
        shape = state["kernel_shape"]
        if not isinstance(shape, tuple) or tuple(kernel.shape) != shape:
            raise ValueError("serialized kernel shape drifted")
        if _tensor_sha256(kernel) != state["kernel_sha256"]:
            raise ValueError("serialized kernel hash mismatch")
        return cls(
            kernel=kernel,
            max_lag=state["max_lag"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            fit_batch_sha256s=state[
                "fit_batch_sha256s"
            ],  # type: ignore[arg-type]
            fit_direction_sha256s=state[
                "fit_direction_sha256s"
            ],  # type: ignore[arg-type]
            source_decoder_sha256=state[
                "source_decoder_sha256"
            ],  # type: ignore[arg-type]
            target_encoder_sha256=state[
                "target_encoder_sha256"
            ],  # type: ignore[arg-type]
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
            causal_direction=state[
                "causal_direction"
            ],  # type: ignore[arg-type]
            lag_definition=state[
                "lag_definition"
            ],  # type: ignore[arg-type]
            stationarity=state["stationarity"],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CausalEdgeJVPHeldoutMetrics:
    """Fixed-kernel metrics on direction-disjoint exact JVP observations."""

    fit_artifact_sha256: str
    heldout_batch_sha256s: tuple[str, ...]
    heldout_direction_sha256s: tuple[str, ...]
    output_frobenius: float
    output_residual_frobenius: float
    relative_output_residual: float
    output_cosine: float
    per_direction_relative_residuals: tuple[float, ...]
    per_direction_cosines: tuple[float, ...]
    relative_residual_p50: float
    relative_residual_p90: float
    relative_residual_worst: float
    zero_target_direction_count: int
    artifact_sha256: str = ""
    artifact_kind: str = _HELDOUT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.fit_artifact_sha256,
            label="fit_artifact_sha256",
        )
        if (
            type(self.heldout_batch_sha256s) is not tuple
            or not self.heldout_batch_sha256s
        ):
            raise ValueError(
                "heldout_batch_sha256s must be a nonempty tuple"
            )
        if (
            type(self.heldout_direction_sha256s) is not tuple
            or not self.heldout_direction_sha256s
        ):
            raise ValueError(
                "heldout_direction_sha256s must be a nonempty tuple"
            )
        for field, values in (
            ("heldout_batch_sha256s", self.heldout_batch_sha256s),
            ("heldout_direction_sha256s", self.heldout_direction_sha256s),
        ):
            for index, digest in enumerate(values):
                _require_sha256(digest, label=f"{field}[{index}]")
        count = len(self.heldout_direction_sha256s)
        for field in (
            "per_direction_relative_residuals",
            "per_direction_cosines",
        ):
            values = getattr(self, field)
            if type(values) is not tuple or len(values) != count:
                raise ValueError(f"{field} must match heldout directions")
        for field in (
            "output_frobenius",
            "output_residual_frobenius",
            "relative_output_residual",
            "relative_residual_p50",
            "relative_residual_p90",
            "relative_residual_worst",
        ):
            object.__setattr__(
                self,
                field,
                _require_finite_nonnegative(getattr(self, field), label=field),
            )
        object.__setattr__(
            self,
            "output_cosine",
            _require_finite_unit(self.output_cosine, label="output_cosine"),
        )
        for index, value in enumerate(
            self.per_direction_relative_residuals
        ):
            _require_finite_nonnegative(
                value,
                label=f"per_direction_relative_residuals[{index}]",
            )
        for index, value in enumerate(self.per_direction_cosines):
            _require_finite_unit(
                value,
                label=f"per_direction_cosines[{index}]",
            )
        zeros = _require_nonnegative_int(
            self.zero_target_direction_count,
            label="zero_target_direction_count",
        )
        if zeros > count:
            raise ValueError(
                "zero_target_direction_count exceeds heldout directions"
            )
        if not (
            self.relative_residual_p50
            <= self.relative_residual_p90
            <= self.relative_residual_worst
        ):
            raise ValueError("relative residual quantiles are inconsistent")
        if (
            self.artifact_kind != _HELDOUT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("heldout JVP metrics header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("heldout JVP metrics hash mismatch")

    @property
    def direction_count(self) -> int:
        return len(self.heldout_direction_sha256s)

    @property
    def heldout_direction_count(self) -> int:
        return self.direction_count

    @property
    def direction_hashes(self) -> tuple[str, ...]:
        return self.heldout_direction_sha256s

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "fit_artifact_sha256": self.fit_artifact_sha256,
            "heldout_batch_sha256s": self.heldout_batch_sha256s,
            "heldout_direction_sha256s": self.heldout_direction_sha256s,
            "output_frobenius": self.output_frobenius,
            "output_residual_frobenius": self.output_residual_frobenius,
            "relative_output_residual": self.relative_output_residual,
            "output_cosine": self.output_cosine,
            "per_direction_relative_residuals": (
                self.per_direction_relative_residuals
            ),
            "per_direction_cosines": self.per_direction_cosines,
            "relative_residual_p50": self.relative_residual_p50,
            "relative_residual_p90": self.relative_residual_p90,
            "relative_residual_worst": self.relative_residual_worst,
            "zero_target_direction_count": self.zero_target_direction_count,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_HELDOUT_HASH_DOMAIN)

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "direction_count": self.direction_count,
            "direction_disjoint_from_fit": True,
            "fixed_kernel_evaluation": True,
        }


@dataclass(frozen=True, slots=True)
class PathIntegratedJVPDiagnostic:
    """Projected finite delta and its Gauss-Legendre JVP path integral."""

    integrated_target_delta: Tensor
    endpoint_target_delta: Tensor
    valid_mask: tuple[bool, ...]
    quadrature_order: int
    quadrature_nodes: tuple[float, ...]
    quadrature_weights: tuple[float, ...]
    endpoint_frobenius: float
    integration_residual_frobenius: float
    relative_integration_residual: float
    integrated_endpoint_cosine: float
    baseline_source_sha256: str
    source_displacement_sha256: str
    target_encoder_sha256: str
    quadrature_rule: str = _QUADRATURE_RULE
    jvp_backend: str = _JVP_BACKEND
    artifact_sha256: str = ""
    artifact_kind: str = _PATH_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        integrated = _as_cpu_float64(
            self.integrated_target_delta,
            label="integrated_target_delta",
            dimensions=3,
        )
        endpoint = _as_cpu_float64(
            self.endpoint_target_delta,
            label="endpoint_target_delta",
            dimensions=3,
        )
        if integrated.shape != endpoint.shape or integrated.shape[0] != 1:
            raise ValueError(
                "path target deltas must share shape [1, S, r_out]"
            )
        object.__setattr__(self, "integrated_target_delta", integrated)
        object.__setattr__(self, "endpoint_target_delta", endpoint)
        if (
            type(self.valid_mask) is not tuple
            or len(self.valid_mask) != integrated.shape[1]
            or any(type(value) is not bool for value in self.valid_mask)
            or not any(self.valid_mask)
        ):
            raise ValueError("valid_mask must select target positions")
        order = _require_positive_int(
            self.quadrature_order,
            label="quadrature_order",
        )
        if order > 4:
            raise ValueError("quadrature_order must be between 1 and 4")
        if (
            type(self.quadrature_nodes) is not tuple
            or type(self.quadrature_weights) is not tuple
            or len(self.quadrature_nodes) != order
            or len(self.quadrature_weights) != order
        ):
            raise ValueError(
                "quadrature nodes and weights must match quadrature_order"
            )
        for index, node in enumerate(self.quadrature_nodes):
            if (
                isinstance(node, bool)
                or not isinstance(node, (int, float))
                or not math.isfinite(float(node))
                or not 0.0 < float(node) < 1.0
            ):
                raise ValueError(
                    f"quadrature_nodes[{index}] must lie inside (0, 1)"
                )
        for index, weight in enumerate(self.quadrature_weights):
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
            ):
                raise ValueError(
                    f"quadrature_weights[{index}] must be positive"
                )
        if not math.isclose(
            sum(self.quadrature_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError("quadrature weights must sum to one")
        for field in (
            "endpoint_frobenius",
            "integration_residual_frobenius",
            "relative_integration_residual",
        ):
            object.__setattr__(
                self,
                field,
                _require_finite_nonnegative(getattr(self, field), label=field),
            )
        object.__setattr__(
            self,
            "integrated_endpoint_cosine",
            _require_finite_unit(
                self.integrated_endpoint_cosine,
                label="integrated_endpoint_cosine",
            ),
        )
        for field in (
            "baseline_source_sha256",
            "source_displacement_sha256",
            "target_encoder_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        if self.quadrature_rule != _QUADRATURE_RULE:
            raise ValueError("quadrature_rule provenance is invalid")
        if self.jvp_backend != _JVP_BACKEND:
            raise ValueError("jvp_backend provenance is invalid")
        if (
            self.artifact_kind != _PATH_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("path-integrated JVP diagnostic header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("path-integrated JVP diagnostic hash mismatch")

    @property
    def jvp_evaluation_count(self) -> int:
        return self.quadrature_order

    @property
    def function_evaluation_count(self) -> int:
        return self.quadrature_order + 2

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "integrated_target_delta_sha256": _tensor_sha256(
                self.integrated_target_delta
            ),
            "endpoint_target_delta_sha256": _tensor_sha256(
                self.endpoint_target_delta
            ),
            "target_delta_shape": tuple(
                int(value) for value in self.endpoint_target_delta.shape
            ),
            "valid_mask": self.valid_mask,
            "quadrature_order": self.quadrature_order,
            "quadrature_nodes": self.quadrature_nodes,
            "quadrature_weights": self.quadrature_weights,
            "endpoint_frobenius": self.endpoint_frobenius,
            "integration_residual_frobenius": (
                self.integration_residual_frobenius
            ),
            "relative_integration_residual": (
                self.relative_integration_residual
            ),
            "integrated_endpoint_cosine": self.integrated_endpoint_cosine,
            "baseline_source_sha256": self.baseline_source_sha256,
            "source_displacement_sha256": self.source_displacement_sha256,
            "target_encoder_sha256": self.target_encoder_sha256,
            "quadrature_rule": self.quadrature_rule,
            "jvp_backend": self.jvp_backend,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_PATH_HASH_DOMAIN)

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "jvp_evaluation_count": self.jvp_evaluation_count,
            "function_evaluation_count": self.function_evaluation_count,
            "oracle_diagnostic_only": True,
        }


def collect_causal_edge_jvp_batch(
    function: Callable[[Tensor], Tensor],
    *,
    baseline_source: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    source_decoder: Tensor,
    target_encoder: Tensor,
    direction_count: int,
    direction_seed: int,
    direction_domain: str,
) -> CausalEdgeJVPBatch:
    """Collect deterministic exact projected JVPs at one linearization point.

    The Rademacher generator seed is derived from both ``direction_seed`` and
    the explicit ``direction_domain``.  Direction hashes omit that generation
    label and instead authenticate the actual tangent query, so the same query
    cannot masquerade as held out merely by changing its domain string.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    baseline = _as_finite_float_tensor(
        baseline_source,
        label="baseline_source",
        dimensions=3,
    )
    if baseline.shape[0] != 1:
        raise ValueError("baseline_source must have shape [1, S, d_in]")
    sequence_length = int(baseline.shape[1])
    input_width = int(baseline.shape[2])
    positions, mask, position_tuple, mask_tuple = _as_positions_and_mask(
        logical_positions,
        valid_mask,
        sequence_length=sequence_length,
    )
    decoder = _as_finite_float_tensor(
        source_decoder,
        label="source_decoder",
        dimensions=2,
    )
    if decoder.shape[0] != input_width:
        raise ValueError("source_decoder first dimension is incompatible")
    encoder = _as_finite_float_tensor(
        target_encoder,
        label="target_encoder",
        dimensions=2,
    )
    direction_count = _require_positive_int(
        direction_count,
        label="direction_count",
    )
    direction_seed = _require_nonnegative_int(
        direction_seed,
        label="direction_seed",
    )
    direction_domain = _require_domain(direction_domain)
    source_rank = int(decoder.shape[1])
    baseline_bound = baseline.detach().clone()
    decoder_device = decoder.detach().to(
        device=baseline_bound.device,
        dtype=baseline_bound.dtype,
    )
    encoder_device = encoder.detach().to(
        device=baseline_bound.device,
        dtype=baseline_bound.dtype,
    )
    mask_device = mask.to(
        device=baseline_bound.device,
        dtype=baseline_bound.dtype,
    ).unsqueeze(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_direction_seed(direction_domain, direction_seed))
    source_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    direction_hashes: list[str] = []
    baseline_sha256 = _tensor_sha256(baseline_bound)
    decoder_sha256 = _tensor_sha256(decoder)
    encoder_sha256 = _tensor_sha256(encoder)
    for _ in range(direction_count):
        bits = torch.randint(
            0,
            2,
            (sequence_length, source_rank),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )
        source_modes = (bits * 2 - 1).to(
            device=baseline_bound.device,
            dtype=baseline_bound.dtype,
        )
        source_modes = source_modes * mask_device
        perturbation = (
            source_modes @ decoder_device.transpose(0, 1)
        ).unsqueeze(0)
        primal, output_jvp = torch.autograd.functional.jvp(
            function,
            (baseline_bound,),
            (perturbation,),
            create_graph=False,
            strict=False,
        )
        _validate_function_output(
            primal,
            label="function output",
            sequence_length=sequence_length,
            output_width=int(encoder.shape[0]),
        )
        output_jvp = _validate_function_output(
            output_jvp,
            label="function JVP output",
            sequence_length=sequence_length,
            output_width=int(encoder.shape[0]),
        )
        projected = output_jvp[0] @ encoder_device
        source_cpu = source_modes.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        source_rows.append(source_cpu)
        target_rows.append(
            projected.detach().to(device="cpu", dtype=torch.float64)
        )
        direction_hashes.append(
            _direction_sha256(
                source_cpu,
                baseline_source_sha256=baseline_sha256,
                source_decoder_sha256=decoder_sha256,
                logical_positions=position_tuple,
                valid_mask=mask_tuple,
            )
        )
    return CausalEdgeJVPBatch(
        source_modes=torch.stack(source_rows, dim=0),
        target_jvps=torch.stack(target_rows, dim=0),
        logical_positions=position_tuple,
        valid_mask=mask_tuple,
        direction_seed=direction_seed,
        direction_domain=direction_domain,
        direction_sha256s=tuple(direction_hashes),
        baseline_source_sha256=baseline_sha256,
        source_decoder_sha256=decoder_sha256,
        target_encoder_sha256=encoder_sha256,
    )


def _validate_batches(
    batches: Sequence[CausalEdgeJVPBatch],
    *,
    label: str,
) -> tuple[CausalEdgeJVPBatch, ...]:
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError(f"{label} must be a sequence of JVP batches")
    resolved = tuple(batches)
    if not resolved:
        raise ValueError(f"{label} must contain at least one JVP batch")
    for batch in resolved:
        if not isinstance(batch, CausalEdgeJVPBatch):
            raise TypeError(f"{label} must contain only JVP batches")
        batch.validate_integrity()
    first = resolved[0]
    for batch in resolved[1:]:
        if batch.source_rank != first.source_rank:
            raise ValueError("JVP batches must share one source rank")
        if batch.target_rank != first.target_rank:
            raise ValueError("JVP batches must share one target rank")
        if batch.source_decoder_sha256 != first.source_decoder_sha256:
            raise ValueError("JVP batches must share one source decoder")
        if batch.target_encoder_sha256 != first.target_encoder_sha256:
            raise ValueError("JVP batches must share one target encoder")
    return resolved


def _design_and_outputs(
    batches: Sequence[CausalEdgeJVPBatch],
    *,
    max_lag: int,
) -> tuple[Tensor, Tensor]:
    designs: list[Tensor] = []
    outputs: list[Tensor] = []
    for batch in batches:
        positions, mask = _tuples_to_positions_and_mask(
            batch.logical_positions,
            batch.valid_mask,
        )
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        for source_modes, target_jvp in zip(
            batch.source_modes,
            batch.target_jvps,
            strict=True,
        ):
            designs.append(
                _lagged_design(
                    source_modes,
                    positions=positions,
                    mask=mask,
                    max_lag=max_lag,
                )
            )
            outputs.append(target_jvp.index_select(0, valid_indices))
    return torch.cat(designs, dim=0), torch.cat(outputs, dim=0)


def fit_pooled_causal_edge_jvp(
    batches: Sequence[CausalEdgeJVPBatch],
    *,
    max_lag: int,
    ridge: float,
) -> PooledCausalEdgeJVPFit:
    """Fit one stationary causal lag kernel across exact-JVP batches."""

    resolved = _validate_batches(batches, label="batches")
    max_lag = _require_nonnegative_int(max_lag, label="max_lag")
    ridge = _require_finite_nonnegative(ridge, label="ridge")
    design, outputs = _design_and_outputs(resolved, max_lag=max_lag)
    singular_values = torch.linalg.svdvals(design)
    tolerance = (
        max(design.shape)
        * torch.finfo(design.dtype).eps
        * (float(singular_values[0]) if singular_values.numel() else 0.0)
    )
    retained = singular_values[singular_values > tolerance]
    design_rank = int(retained.numel())
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    smallest_retained = float(retained[-1]) if retained.numel() else 0.0
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
    normal_residual = (
        gram @ coefficients - right_hand_side + ridge * coefficients
    )
    first = resolved[0]
    kernel = coefficients.reshape(
        max_lag + 1,
        first.source_rank,
        first.target_rank,
    )
    return PooledCausalEdgeJVPFit(
        kernel=kernel,
        max_lag=max_lag,
        ridge=ridge,
        fit_batch_sha256s=tuple(
            batch.artifact_sha256 for batch in resolved
        ),
        fit_direction_sha256s=tuple(
            digest
            for batch in resolved
            for digest in batch.direction_sha256s
        ),
        source_decoder_sha256=first.source_decoder_sha256,
        target_encoder_sha256=first.target_encoder_sha256,
        design_row_count=int(design.shape[0]),
        design_column_count=int(design.shape[1]),
        design_rank=design_rank,
        design_largest_singular_value=largest,
        design_smallest_retained_singular_value=smallest_retained,
        output_frobenius=output_frobenius,
        output_residual_frobenius=output_residual,
        relative_output_residual=output_residual / denominator,
        design_normal_equation_residual_frobenius=float(
            torch.linalg.vector_norm(normal_residual)
        ),
    )


def evaluate_pooled_causal_edge_jvp(
    fit: PooledCausalEdgeJVPFit,
    batches: Sequence[CausalEdgeJVPBatch],
) -> CausalEdgeJVPHeldoutMetrics:
    """Evaluate a frozen pooled kernel on direction-disjoint JVP batches."""

    if not isinstance(fit, PooledCausalEdgeJVPFit):
        raise TypeError("fit must be a PooledCausalEdgeJVPFit")
    fit.validate_integrity()
    resolved = _validate_batches(batches, label="batches")
    heldout_hashes = tuple(
        digest
        for batch in resolved
        for digest in batch.direction_sha256s
    )
    overlap = set(fit.fit_direction_sha256s).intersection(heldout_hashes)
    if overlap:
        first_overlap = sorted(overlap)[0]
        raise ValueError(
            "fit and heldout JVP directions overlap: "
            f"{first_overlap}"
        )
    first = resolved[0]
    if first.source_rank != fit.source_rank:
        raise ValueError("heldout source rank does not match the fit")
    if first.target_rank != fit.target_rank:
        raise ValueError("heldout target rank does not match the fit")
    if first.source_decoder_sha256 != fit.source_decoder_sha256:
        raise ValueError("heldout source decoder does not match the fit")
    if first.target_encoder_sha256 != fit.target_encoder_sha256:
        raise ValueError("heldout target encoder does not match the fit")
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    per_relative: list[float] = []
    per_cosine: list[float] = []
    zero_targets = 0
    epsilon = torch.finfo(torch.float64).eps
    for batch in resolved:
        positions, mask = _tuples_to_positions_and_mask(
            batch.logical_positions,
            batch.valid_mask,
        )
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        for source_modes, target in zip(
            batch.source_modes,
            batch.target_jvps,
            strict=True,
        ):
            prediction = fit.execute(
                source_modes,
                logical_positions=positions,
                valid_mask=mask,
            ).index_select(0, valid_indices)
            selected_target = target.index_select(0, valid_indices)
            residual = prediction - selected_target
            target_norm = float(torch.linalg.vector_norm(selected_target))
            if target_norm <= epsilon:
                zero_targets += 1
            per_relative.append(
                float(torch.linalg.vector_norm(residual))
                / max(target_norm, epsilon)
            )
            per_cosine.append(_cosine(prediction, selected_target))
            predictions.append(prediction)
            targets.append(selected_target)
    prediction_all = torch.cat(predictions, dim=0)
    target_all = torch.cat(targets, dim=0)
    residual_all = prediction_all - target_all
    output_frobenius = float(torch.linalg.vector_norm(target_all))
    output_residual = float(torch.linalg.vector_norm(residual_all))
    return CausalEdgeJVPHeldoutMetrics(
        fit_artifact_sha256=fit.artifact_sha256,
        heldout_batch_sha256s=tuple(
            batch.artifact_sha256 for batch in resolved
        ),
        heldout_direction_sha256s=heldout_hashes,
        output_frobenius=output_frobenius,
        output_residual_frobenius=output_residual,
        relative_output_residual=output_residual
        / max(output_frobenius, epsilon),
        output_cosine=_cosine(prediction_all, target_all),
        per_direction_relative_residuals=tuple(per_relative),
        per_direction_cosines=tuple(per_cosine),
        relative_residual_p50=_quantile(per_relative, 0.5),
        relative_residual_p90=_quantile(per_relative, 0.9),
        relative_residual_worst=max(per_relative),
        zero_target_direction_count=zero_targets,
    )


def gauss_legendre_unit_interval(
    order: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the exact checked Gauss--Legendre rule on ``[0, 1]``.

    The constants are intentionally literal rather than generated at runtime.
    Besides avoiding a NumPy dependency, this makes path-evidence receipts
    stable across platforms.  Orders one through four retain the historical
    transport behavior; the private spelling below remains as a compatibility
    shim for research code that imported it before it became public.
    """

    order = _require_positive_int(order, label="quadrature_order")
    rules = {
        1: ((0.5,), (1.0,)),
        2: (
            (
                0.21132486540518713,
                0.7886751345948129,
            ),
            (0.5, 0.5),
        ),
        3: (
            (
                0.1127016653792583,
                0.5,
                0.8872983346207417,
            ),
            (
                0.2777777777777778,
                0.4444444444444444,
                0.2777777777777778,
            ),
        ),
        4: (
            (
                0.06943184420297371,
                0.33000947820757187,
                0.6699905217924281,
                0.9305681557970262,
            ),
            (
                0.17392742256872692,
                0.32607257743127305,
                0.32607257743127305,
                0.17392742256872692,
            ),
        ),
    }
    try:
        return rules[order]
    except KeyError as error:
        raise ValueError("quadrature_order must be between 1 and 4") from error


def _gauss_legendre_unit_interval(
    order: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compatibility wrapper for the former private quadrature helper."""

    return gauss_legendre_unit_interval(order)


def integrate_path_jvp(
    function: Callable[[Tensor], Tensor],
    *,
    baseline_source: Tensor,
    source_displacement: Tensor,
    target_encoder: Tensor,
    valid_mask: Tensor,
    quadrature_order: int,
) -> PathIntegratedJVPDiagnostic:
    """Compare a finite projected delta with JVP transport along its path.

    Every integrand evaluation is an exact autograd JVP.  The integral itself
    is a fixed Gauss-Legendre quadrature approximation of order one through
    four, making this an oracle diagnostic rather than a compiled executor.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    baseline = _as_finite_float_tensor(
        baseline_source,
        label="baseline_source",
        dimensions=3,
    )
    displacement = _as_finite_float_tensor(
        source_displacement,
        label="source_displacement",
        dimensions=3,
    )
    if baseline.shape != displacement.shape or baseline.shape[0] != 1:
        raise ValueError(
            "baseline_source and source_displacement must share shape "
            "[1, S, d_in]"
        )
    encoder = _as_finite_float_tensor(
        target_encoder,
        label="target_encoder",
        dimensions=2,
    )
    sequence_length = int(baseline.shape[1])
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a Tensor")
    mask = valid_mask.detach()
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if (
        mask.ndim != 1
        or mask.shape[0] != sequence_length
        or mask.dtype != torch.bool
    ):
        raise ValueError("valid_mask must have shape [S] or [1, S] and bool")
    mask_cpu = mask.to(device="cpu", dtype=torch.bool).contiguous()
    if not bool(mask_cpu.any()):
        raise ValueError("valid_mask must select at least one position")
    nodes, weights = _gauss_legendre_unit_interval(quadrature_order)
    baseline_bound = baseline.detach().clone()
    displacement_bound = displacement.detach().clone()
    encoder_device = encoder.detach().to(
        device=baseline_bound.device,
        dtype=baseline_bound.dtype,
    )
    integrated: Tensor | None = None
    for node, weight in zip(nodes, weights, strict=True):
        path_point = baseline_bound + node * displacement_bound
        primal, output_jvp = torch.autograd.functional.jvp(
            function,
            (path_point,),
            (displacement_bound,),
            create_graph=False,
            strict=False,
        )
        _validate_function_output(
            primal,
            label="function output",
            sequence_length=sequence_length,
            output_width=int(encoder.shape[0]),
        )
        output_jvp = _validate_function_output(
            output_jvp,
            label="function JVP output",
            sequence_length=sequence_length,
            output_width=int(encoder.shape[0]),
        )
        projected_jvp = output_jvp @ encoder_device
        term = weight * projected_jvp
        integrated = term if integrated is None else integrated + term
    assert integrated is not None
    start_output = _validate_function_output(
        function(baseline_bound),
        label="function output",
        sequence_length=sequence_length,
        output_width=int(encoder.shape[0]),
    )
    end_output = _validate_function_output(
        function(baseline_bound + displacement_bound),
        label="function output",
        sequence_length=sequence_length,
        output_width=int(encoder.shape[0]),
    )
    endpoint_delta = (end_output - start_output) @ encoder_device
    integrated_cpu = integrated.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    endpoint_cpu = endpoint_delta.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    valid_indices = torch.nonzero(mask_cpu, as_tuple=False).flatten()
    selected_integrated = integrated_cpu[0].index_select(0, valid_indices)
    selected_endpoint = endpoint_cpu[0].index_select(0, valid_indices)
    residual = selected_integrated - selected_endpoint
    endpoint_frobenius = float(torch.linalg.vector_norm(selected_endpoint))
    residual_frobenius = float(torch.linalg.vector_norm(residual))
    return PathIntegratedJVPDiagnostic(
        integrated_target_delta=integrated_cpu,
        endpoint_target_delta=endpoint_cpu,
        valid_mask=tuple(bool(value) for value in mask_cpu.tolist()),
        quadrature_order=quadrature_order,
        quadrature_nodes=nodes,
        quadrature_weights=weights,
        endpoint_frobenius=endpoint_frobenius,
        integration_residual_frobenius=residual_frobenius,
        relative_integration_residual=residual_frobenius
        / max(endpoint_frobenius, torch.finfo(torch.float64).eps),
        integrated_endpoint_cosine=_cosine(
            selected_integrated,
            selected_endpoint,
        ),
        baseline_source_sha256=_tensor_sha256(baseline_bound),
        source_displacement_sha256=_tensor_sha256(displacement_bound),
        target_encoder_sha256=_tensor_sha256(encoder),
    )
