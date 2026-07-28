"""Tangent-preserving finite-displacement correction for causal modal edges.

The stationary JVP edge is a first-order model.  This module adds a compact
second-order residual without changing that measured tangent.  For the
logical-lag source row

``z[t] = concat(m[t], m[t-1], ..., m[t-L+1])``

the correction is

``e[t] = ((z[t] @ A) * (z[t] @ C)) @ B``.

There are deliberately no biases or linear skip terms.  Consequently
``e(0) = 0`` and the Jacobian of ``e`` at zero is exactly zero.  The fixed
base JVP kernel therefore remains solely responsible for the local tangent,
while this artifact can fit finite-displacement curvature.

Artifacts are canonical CPU/float64 values with strict, domain-separated
hashes.  Training samples are represented in modal coordinates and only
their aggregate digest is retained by the fitted edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "TangentPreservingQuadraticEdge",
    "TangentPreservingQuadraticEdgeMetrics",
    "TangentPreservingQuadraticSample",
    "build_causal_lagged_modal_design",
    "evaluate_tangent_preserving_quadratic_edge",
    "fit_tangent_preserving_quadratic_edge",
]


_FORMAT_VERSION = 1
_ARTIFACT_KIND = "fisher_graph.tangent_preserving_quadratic_edge"
_ARTIFACT_DOMAIN = b"fisher_graph.tangent_preserving_quadratic_edge.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.tangent_preserving_quadratic.tensor.v1\0"
_SAMPLE_DOMAIN = b"fisher_graph.tangent_preserving_quadratic.sample.v1\0"
_SAMPLE_SET_DOMAIN = (
    b"fisher_graph.tangent_preserving_quadratic.sample_set.v1\0"
)
_OBJECTIVE = "finite_residual_mse_plus_mean_square_factor_ridge"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_POSITION_DTYPES = frozenset({torch.int32, torch.int64})


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _canonical_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in canonical.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_base_kernel(value: Tensor) -> Tensor:
    kernel = _canonical_float_tensor(
        value,
        label="base_kernel",
        ndim=3,
    )
    return kernel


def _canonical_positions_and_mask(
    logical_positions: Tensor,
    valid_mask: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not isinstance(logical_positions, Tensor):
        raise TypeError("logical_positions must be a Tensor")
    if logical_positions.dtype not in _POSITION_DTYPES:
        raise TypeError("logical_positions must use torch.int32 or torch.int64")
    if logical_positions.device != device:
        raise ValueError("logical_positions and source_modes must share device")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a boolean Tensor")
    if valid_mask.device != device:
        raise ValueError("valid_mask and source_modes must share device")
    allowed_shapes = {(sequence_length,), (batch_size, sequence_length)}
    if (
        tuple(logical_positions.shape) not in allowed_shapes
        or tuple(valid_mask.shape) not in allowed_shapes
    ):
        raise ValueError(
            "logical_positions and valid_mask must have shape [S] or [B, S]"
        )
    positions = (
        logical_positions.unsqueeze(0).expand(batch_size, sequence_length)
        if logical_positions.ndim == 1
        else logical_positions
    )
    mask = (
        valid_mask.unsqueeze(0).expand(batch_size, sequence_length)
        if valid_mask.ndim == 1
        else valid_mask
    )
    for batch in range(batch_size):
        selected = positions[batch][mask[batch]]
        if selected.numel() == 0:
            raise ValueError("every sequence must contain a valid position")
        if bool((selected < 0).any()):
            raise ValueError("valid logical positions must be nonnegative")
        if selected.numel() > 1 and not bool(
            torch.all(selected[1:] > selected[:-1])
        ):
            raise ValueError(
                "valid logical positions must be strictly increasing"
            )
    return positions, mask


def build_causal_lagged_modal_design(
    source_modes: Tensor,
    *,
    logical_positions: Tensor,
    valid_mask: Tensor,
    lag_count: int,
) -> Tensor:
    """Build exact logical-lag rows without using physical-offset fallback.

    ``source_modes`` may be ``[S, r]`` or ``[B, S, r]``.  The returned tensor
    has the matching leading shape and final width ``lag_count * r``.  Invalid
    targets are zero.  A lag block is zero when the corresponding logical
    source position is missing or invalid.
    """

    lag_count = _positive_int(lag_count, label="lag_count")
    if not isinstance(source_modes, Tensor):
        raise TypeError("source_modes must be a Tensor")
    if (
        source_modes.ndim not in (2, 3)
        or not source_modes.is_floating_point()
        or source_modes.dtype not in _RUNTIME_DTYPES
        or any(int(width) <= 0 for width in source_modes.shape)
        or not bool(torch.isfinite(source_modes).all())
    ):
        raise ValueError(
            "source_modes must be a finite nonempty [S, r] or [B, S, r] "
            "floating Tensor"
        )
    squeeze = source_modes.ndim == 2
    batched = source_modes.unsqueeze(0) if squeeze else source_modes
    batch_size, sequence_length, source_rank = (
        int(batched.shape[0]),
        int(batched.shape[1]),
        int(batched.shape[2]),
    )
    positions, mask = _canonical_positions_and_mask(
        logical_positions,
        valid_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=source_modes.device,
    )
    design = source_modes.new_zeros(
        (batch_size, sequence_length, lag_count * source_rank)
    )
    for batch in range(batch_size):
        valid_indices = torch.nonzero(
            mask[batch],
            as_tuple=False,
        ).flatten().tolist()
        by_position = {
            int(positions[batch, index].item()): int(index)
            for index in valid_indices
        }
        for target_index in valid_indices:
            target_position = int(
                positions[batch, target_index].item()
            )
            for lag in range(lag_count):
                source_index = by_position.get(target_position - lag)
                if source_index is None:
                    continue
                start = lag * source_rank
                design[
                    batch,
                    target_index,
                    start : start + source_rank,
                ] = batched[batch, source_index]
    return design[0] if squeeze else design


def _base_from_design(design: Tensor, kernel: Tensor) -> Tensor:
    source_rank = int(kernel.shape[1])
    lag_count = int(kernel.shape[0])
    if design.shape[-1] != lag_count * source_rank:
        raise ValueError("lagged design and base kernel shapes disagree")
    result = design.new_zeros(
        (*design.shape[:-1], int(kernel.shape[2]))
    )
    for lag in range(lag_count):
        block = design[
            ..., lag * source_rank : (lag + 1) * source_rank
        ]
        result = result + block @ kernel[lag]
    return result


@dataclass(frozen=True, slots=True, eq=False)
class TangentPreservingQuadraticSample:
    """One finite source displacement and target modal response."""

    source_modes: Tensor
    target_modes: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        source = _canonical_float_tensor(
            self.source_modes,
            label="source_modes",
            ndim=2,
        )
        target = _canonical_float_tensor(
            self.target_modes,
            label="target_modes",
            ndim=2,
        )
        if source.shape[0] != target.shape[0]:
            raise ValueError("source_modes and target_modes lengths differ")
        if not isinstance(self.logical_positions, Tensor):
            raise TypeError("logical_positions must be a Tensor")
        if self.logical_positions.dtype not in _POSITION_DTYPES:
            raise TypeError(
                "logical_positions must use torch.int32 or torch.int64"
            )
        positions = (
            self.logical_positions.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
            .clone()
        )
        if (
            positions.ndim != 1
            or positions.shape[0] != source.shape[0]
        ):
            raise ValueError("logical_positions must have shape [S]")
        if (
            not isinstance(self.valid_mask, Tensor)
            or self.valid_mask.dtype != torch.bool
        ):
            raise TypeError("valid_mask must be a boolean Tensor")
        mask = (
            self.valid_mask.detach()
            .to(device="cpu", dtype=torch.bool)
            .contiguous()
            .clone()
        )
        if mask.ndim != 1 or mask.shape[0] != source.shape[0]:
            raise ValueError("valid_mask must have shape [S]")
        selected = positions[mask]
        if selected.numel() == 0:
            raise ValueError("sample must contain a valid position")
        if bool((selected < 0).any()):
            raise ValueError("valid logical positions must be nonnegative")
        if selected.numel() > 1 and not bool(
            torch.all(selected[1:] > selected[:-1])
        ):
            raise ValueError(
                "valid logical positions must be strictly increasing"
            )
        object.__setattr__(self, "source_modes", source)
        object.__setattr__(self, "target_modes", target)
        object.__setattr__(self, "logical_positions", positions)
        object.__setattr__(self, "valid_mask", mask)
        computed = _json_sha256(
            {
                "source_modes_sha256": _tensor_sha256(source),
                "target_modes_sha256": _tensor_sha256(target),
                "logical_positions": tuple(int(value) for value in positions),
                "valid_mask": tuple(bool(value) for value in mask),
            },
            domain=_SAMPLE_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="sample artifact_sha256",
                )
                != computed
            ):
                raise ValueError("quadratic sample hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_rank(self) -> int:
        return int(self.source_modes.shape[1])

    @property
    def target_rank(self) -> int:
        return int(self.target_modes.shape[1])

    @property
    def valid_target_rows(self) -> int:
        return int(self.valid_mask.sum().item())


def _sample_set_sha256(
    samples: Sequence[TangentPreservingQuadraticSample],
) -> str:
    return _json_sha256(
        tuple(sample.artifact_sha256 for sample in samples),
        domain=_SAMPLE_SET_DOMAIN,
    )


def _cosine(left: Tensor, right: Tensor) -> float:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = (
        torch.linalg.vector_norm(left64)
        * torch.linalg.vector_norm(right64)
    )
    if float(denominator.item()) == 0.0:
        return 1.0 if bool(torch.equal(left64, right64)) else 0.0
    return float(torch.dot(left64, right64).item() / denominator.item())


@dataclass(frozen=True, slots=True)
class TangentPreservingQuadraticEdgeMetrics:
    """Finite-response metrics before and after the quadratic correction."""

    sequence_count: int
    valid_target_rows: int
    target_frobenius: float
    base_residual_frobenius: float
    corrected_residual_frobenius: float
    base_relative_error: float
    corrected_relative_error: float
    base_cosine: float
    corrected_cosine: float

    def __post_init__(self) -> None:
        _positive_int(self.sequence_count, label="sequence_count")
        _positive_int(self.valid_target_rows, label="valid_target_rows")
        for name in (
            "target_frobenius",
            "base_residual_frobenius",
            "corrected_residual_frobenius",
            "base_relative_error",
            "corrected_relative_error",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    label=name,
                    nonnegative=True,
                ),
            )
        for name in ("base_cosine", "corrected_cosine"):
            value = _finite_float(getattr(self, name), label=name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_count": self.sequence_count,
            "valid_target_rows": self.valid_target_rows,
            "target_frobenius": self.target_frobenius,
            "base_residual_frobenius": self.base_residual_frobenius,
            "corrected_residual_frobenius": (
                self.corrected_residual_frobenius
            ),
            "base_relative_error": self.base_relative_error,
            "corrected_relative_error": self.corrected_relative_error,
            "base_cosine": self.base_cosine,
            "corrected_cosine": self.corrected_cosine,
        }

    @classmethod
    def from_dict(
        cls,
        state: Mapping[str, object],
    ) -> TangentPreservingQuadraticEdgeMetrics:
        expected = {
            "sequence_count",
            "valid_target_rows",
            "target_frobenius",
            "base_residual_frobenius",
            "corrected_residual_frobenius",
            "base_relative_error",
            "corrected_relative_error",
            "base_cosine",
            "corrected_cosine",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("quadratic edge metric fields are invalid")
        return cls(**state)  # type: ignore[arg-type]


def _metrics_from_rows(
    *,
    targets: Tensor,
    base: Tensor,
    correction: Tensor,
    sequence_count: int,
) -> TangentPreservingQuadraticEdgeMetrics:
    corrected = base + correction
    target_norm = float(torch.linalg.vector_norm(targets).item())
    base_residual = float(
        torch.linalg.vector_norm(base - targets).item()
    )
    corrected_residual = float(
        torch.linalg.vector_norm(corrected - targets).item()
    )
    denominator = max(target_norm, torch.finfo(torch.float64).eps)
    return TangentPreservingQuadraticEdgeMetrics(
        sequence_count=sequence_count,
        valid_target_rows=int(targets.shape[0]),
        target_frobenius=target_norm,
        base_residual_frobenius=base_residual,
        corrected_residual_frobenius=corrected_residual,
        base_relative_error=base_residual / denominator,
        corrected_relative_error=corrected_residual / denominator,
        base_cosine=_cosine(base, targets),
        corrected_cosine=_cosine(corrected, targets),
    )


@dataclass(frozen=True, slots=True, eq=False)
class TangentPreservingQuadraticEdge:
    """Authenticated low-rank quadratic correction with a zero local tangent."""

    A: Tensor
    C: Tensor
    B: Tensor
    lag_count: int
    base_kernel_sha256: str
    training_samples_sha256: str
    heldout_samples_sha256: str | None
    training_sample_count: int
    heldout_sample_count: int
    steps: int
    learning_rate: float
    ridge: float
    seed: int
    minibatch_rows: int
    train_metrics: TangentPreservingQuadraticEdgeMetrics
    heldout_metrics: TangentPreservingQuadraticEdgeMetrics | None
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        A = _canonical_float_tensor(self.A, label="A", ndim=2)
        C = _canonical_float_tensor(self.C, label="C", ndim=2)
        B = _canonical_float_tensor(self.B, label="B", ndim=2)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "B", B)
        lag_count = _positive_int(self.lag_count, label="lag_count")
        if A.shape != C.shape or A.shape[1] != B.shape[0]:
            raise ValueError("quadratic factor shapes are incompatible")
        if A.shape[0] % lag_count:
            raise ValueError("quadratic input width is not divisible by lag_count")
        _require_sha256(
            self.base_kernel_sha256,
            label="base_kernel_sha256",
        )
        _require_sha256(
            self.training_samples_sha256,
            label="training_samples_sha256",
        )
        _positive_int(
            self.training_sample_count,
            label="training_sample_count",
        )
        _nonnegative_int(
            self.heldout_sample_count,
            label="heldout_sample_count",
        )
        _positive_int(self.steps, label="steps")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                self.learning_rate,
                label="learning_rate",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "ridge",
            _finite_float(self.ridge, label="ridge", nonnegative=True),
        )
        _nonnegative_int(self.seed, label="seed")
        _nonnegative_int(self.minibatch_rows, label="minibatch_rows")
        if not isinstance(
            self.train_metrics,
            TangentPreservingQuadraticEdgeMetrics,
        ):
            raise TypeError("train_metrics has an invalid type")
        if (
            self.train_metrics.sequence_count
            != self.training_sample_count
        ):
            raise ValueError("training metric sample count drifted")
        if self.heldout_sample_count:
            if (
                self.heldout_samples_sha256 is None
                or not isinstance(
                    self.heldout_metrics,
                    TangentPreservingQuadraticEdgeMetrics,
                )
                or self.heldout_metrics.sequence_count
                != self.heldout_sample_count
            ):
                raise ValueError("heldout metric provenance is incomplete")
            _require_sha256(
                self.heldout_samples_sha256,
                label="heldout_samples_sha256",
            )
        elif (
            self.heldout_samples_sha256 is not None
            or self.heldout_metrics is not None
        ):
            raise ValueError("empty heldout data cannot retain metrics")
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("quadratic edge artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("quadratic edge artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_rank(self) -> int:
        return int(self.A.shape[0]) // self.lag_count

    @property
    def target_rank(self) -> int:
        return int(self.B.shape[1])

    @property
    def hidden_width(self) -> int:
        return int(self.A.shape[1])

    @property
    def stored_scalar_count(self) -> int:
        return self.A.numel() + self.C.numel() + self.B.numel()

    @property
    def macs_per_target_row(self) -> int:
        """Linear MACs only; excludes ``hidden_width`` elementwise products."""

        return self.stored_scalar_count

    @property
    def elementwise_multiplies_per_target_row(self) -> int:
        return self.hidden_width

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "architecture": "B_times_elementwise_zA_zC_no_bias",
            "objective": _OBJECTIVE,
            "zero_value_at_zero": True,
            "zero_jacobian_at_zero": True,
            "lag_count": self.lag_count,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "hidden_width": self.hidden_width,
            "tensor_sha256s": {
                "A": _tensor_sha256(self.A),
                "C": _tensor_sha256(self.C),
                "B": _tensor_sha256(self.B),
            },
            "base_kernel_sha256": self.base_kernel_sha256,
            "training_samples_sha256": self.training_samples_sha256,
            "heldout_samples_sha256": self.heldout_samples_sha256,
            "training_sample_count": self.training_sample_count,
            "heldout_sample_count": self.heldout_sample_count,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "ridge": self.ridge,
            "seed": self.seed,
            "minibatch_rows": self.minibatch_rows,
            "train_metrics": self.train_metrics.to_dict(),
            "heldout_metrics": (
                None
                if self.heldout_metrics is None
                else self.heldout_metrics.to_dict()
            ),
            "stored_scalar_count": self.stored_scalar_count,
            "macs_per_target_row": self.macs_per_target_row,
            "elementwise_multiplies_per_target_row": (
                self.elementwise_multiplies_per_target_row
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        for name in ("A", "C", "B"):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"quadratic factor {name} drifted")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("quadratic edge artifact hash mismatch")

    def execute(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Return the quadratic correction only, excluding the base JVP edge."""

        self.validate_integrity()
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.ndim not in (2, 3)
            or source_modes.shape[-1] != self.source_rank
            or source_modes.dtype not in _RUNTIME_DTYPES
        ):
            raise ValueError("source_modes do not match the quadratic edge")
        design = build_causal_lagged_modal_design(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            lag_count=self.lag_count,
        )
        A = self.A.to(device=source_modes.device, dtype=source_modes.dtype)
        C = self.C.to(device=source_modes.device, dtype=source_modes.dtype)
        B = self.B.to(device=source_modes.device, dtype=source_modes.dtype)
        return ((design @ A) * (design @ C)) @ B

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "A": self.A.clone(),
            "C": self.C.clone(),
            "B": self.B.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> TangentPreservingQuadraticEdge:
        expected = {
            "artifact_kind",
            "format_version",
            "architecture",
            "objective",
            "zero_value_at_zero",
            "zero_jacobian_at_zero",
            "lag_count",
            "source_rank",
            "target_rank",
            "hidden_width",
            "tensor_sha256s",
            "base_kernel_sha256",
            "training_samples_sha256",
            "heldout_samples_sha256",
            "training_sample_count",
            "heldout_sample_count",
            "steps",
            "learning_rate",
            "ridge",
            "seed",
            "minibatch_rows",
            "train_metrics",
            "heldout_metrics",
            "stored_scalar_count",
            "macs_per_target_row",
            "elementwise_multiplies_per_target_row",
            "A",
            "C",
            "B",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("quadratic edge state fields are invalid")
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or state["format_version"] != _FORMAT_VERSION
            or state["architecture"] != "B_times_elementwise_zA_zC_no_bias"
            or state["objective"] != _OBJECTIVE
            or state["zero_value_at_zero"] is not True
            or state["zero_jacobian_at_zero"] is not True
        ):
            raise ValueError("quadratic edge state declaration is invalid")
        tensors = {name: state[name] for name in ("A", "C", "B")}
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("quadratic edge state tensors are invalid")
        tensor_hashes = state["tensor_sha256s"]
        if (
            not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != {"A", "C", "B"}
            or any(
                _tensor_sha256(tensors[name]) != tensor_hashes[name]
                for name in tensors
            )
        ):
            raise ValueError("quadratic edge tensor hash mismatch")
        train_metrics = state["train_metrics"]
        heldout_metrics = state["heldout_metrics"]
        result = cls(
            A=tensors["A"],  # type: ignore[arg-type]
            C=tensors["C"],  # type: ignore[arg-type]
            B=tensors["B"],  # type: ignore[arg-type]
            lag_count=state["lag_count"],  # type: ignore[arg-type]
            base_kernel_sha256=state[
                "base_kernel_sha256"
            ],  # type: ignore[arg-type]
            training_samples_sha256=state[
                "training_samples_sha256"
            ],  # type: ignore[arg-type]
            heldout_samples_sha256=state[
                "heldout_samples_sha256"
            ],  # type: ignore[arg-type]
            training_sample_count=state[
                "training_sample_count"
            ],  # type: ignore[arg-type]
            heldout_sample_count=state[
                "heldout_sample_count"
            ],  # type: ignore[arg-type]
            steps=state["steps"],  # type: ignore[arg-type]
            learning_rate=state["learning_rate"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            seed=state["seed"],  # type: ignore[arg-type]
            minibatch_rows=state["minibatch_rows"],  # type: ignore[arg-type]
            train_metrics=TangentPreservingQuadraticEdgeMetrics.from_dict(
                train_metrics  # type: ignore[arg-type]
            ),
            heldout_metrics=(
                None
                if heldout_metrics is None
                else TangentPreservingQuadraticEdgeMetrics.from_dict(
                    heldout_metrics  # type: ignore[arg-type]
                )
            ),
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            result.source_rank != state["source_rank"]
            or result.target_rank != state["target_rank"]
            or result.hidden_width != state["hidden_width"]
            or result.stored_scalar_count != state["stored_scalar_count"]
            or result.macs_per_target_row != state["macs_per_target_row"]
            or result.elementwise_multiplies_per_target_row
            != state["elementwise_multiplies_per_target_row"]
        ):
            raise ValueError("quadratic edge derived accounting drifted")
        return result


def _canonical_samples(
    samples: Sequence[TangentPreservingQuadraticSample],
    *,
    label: str,
) -> tuple[TangentPreservingQuadraticSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(samples)
    if not result or any(
        not isinstance(sample, TangentPreservingQuadraticSample)
        for sample in result
    ):
        raise ValueError(
            f"{label} must contain TangentPreservingQuadraticSample values"
        )
    return result


def _rows_for_samples(
    samples: Sequence[TangentPreservingQuadraticSample],
    *,
    kernel: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    designs: list[Tensor] = []
    bases: list[Tensor] = []
    targets: list[Tensor] = []
    for sample in samples:
        design = build_causal_lagged_modal_design(
            sample.source_modes,
            logical_positions=sample.logical_positions,
            valid_mask=sample.valid_mask,
            lag_count=int(kernel.shape[0]),
        )
        base = _base_from_design(design, kernel)
        selected = sample.valid_mask
        designs.append(design[selected])
        bases.append(base[selected])
        targets.append(sample.target_modes[selected])
    return (
        torch.cat(designs, dim=0).to(dtype=torch.float64),
        torch.cat(bases, dim=0).to(dtype=torch.float64),
        torch.cat(targets, dim=0).to(dtype=torch.float64),
    )


def _validate_sample_geometry(
    samples: Sequence[TangentPreservingQuadraticSample],
    *,
    source_rank: int,
    target_rank: int,
    label: str,
) -> None:
    for index, sample in enumerate(samples):
        if (
            sample.source_rank != source_rank
            or sample.target_rank != target_rank
        ):
            raise ValueError(
                f"{label}[{index}] modal ranks differ from the base kernel"
            )


def _correction_from_weights(
    design: Tensor,
    A: Tensor,
    C: Tensor,
    B: Tensor,
) -> Tensor:
    return ((design @ A) * (design @ C)) @ B


def fit_tangent_preserving_quadratic_edge(
    samples: Sequence[TangentPreservingQuadraticSample],
    *,
    base_kernel: Tensor,
    hidden_width: int,
    steps: int,
    learning_rate: float,
    ridge: float,
    seed: int,
    heldout_samples: Sequence[TangentPreservingQuadraticSample] = (),
    minibatch_rows: int | None = None,
) -> TangentPreservingQuadraticEdge:
    """Fit a deterministic finite residual while preserving the base tangent."""

    train = _canonical_samples(samples, label="samples")
    heldout = (
        ()
        if not heldout_samples
        else _canonical_samples(heldout_samples, label="heldout_samples")
    )
    kernel = _canonical_base_kernel(base_kernel)
    lag_count, source_rank, target_rank = (
        int(kernel.shape[0]),
        int(kernel.shape[1]),
        int(kernel.shape[2]),
    )
    hidden_width = _positive_int(hidden_width, label="hidden_width")
    steps = _positive_int(steps, label="steps")
    learning_rate = _finite_float(
        learning_rate,
        label="learning_rate",
        positive=True,
    )
    ridge = _finite_float(ridge, label="ridge", nonnegative=True)
    seed = _nonnegative_int(seed, label="seed")
    if minibatch_rows is None:
        minibatch = 0
    else:
        minibatch = _positive_int(minibatch_rows, label="minibatch_rows")
    _validate_sample_geometry(
        train,
        source_rank=source_rank,
        target_rank=target_rank,
        label="samples",
    )
    _validate_sample_geometry(
        heldout,
        source_rank=source_rank,
        target_rank=target_rank,
        label="heldout_samples",
    )
    train_hashes = {sample.artifact_sha256 for sample in train}
    heldout_hashes = {sample.artifact_sha256 for sample in heldout}
    if train_hashes & heldout_hashes:
        raise ValueError("training and heldout samples must be disjoint")

    design, base, targets = _rows_for_samples(train, kernel=kernel)
    residual_targets = targets - base
    input_width = lag_count * source_rank
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_scale = 1.0 / math.sqrt(input_width)
    hidden_scale = 1.0 / math.sqrt(hidden_width)
    A = (
        torch.randn(
            (input_width, hidden_width),
            generator=generator,
            dtype=torch.float64,
        )
        * input_scale
    ).requires_grad_(True)
    C = (
        torch.randn(
            (input_width, hidden_width),
            generator=generator,
            dtype=torch.float64,
        )
        * input_scale
    ).requires_grad_(True)
    B = (
        torch.randn(
            (hidden_width, target_rank),
            generator=generator,
            dtype=torch.float64,
        )
        * hidden_scale
        * 1e-2
    ).requires_grad_(True)
    optimizer = torch.optim.Adam((A, C, B), lr=learning_rate)
    row_count = int(design.shape[0])
    for _ in range(steps):
        if minibatch and minibatch < row_count:
            indices = torch.randint(
                0,
                row_count,
                (minibatch,),
                generator=generator,
                dtype=torch.int64,
            )
            fit_design = design.index_select(0, indices)
            fit_targets = residual_targets.index_select(0, indices)
        else:
            fit_design = design
            fit_targets = residual_targets
        optimizer.zero_grad(set_to_none=True)
        prediction = _correction_from_weights(fit_design, A, C, B)
        loss = (prediction - fit_targets).square().mean()
        if ridge:
            loss = loss + ridge * (
                A.square().mean()
                + C.square().mean()
                + B.square().mean()
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("quadratic edge fit became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (A, C, B),
            max_norm=10.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
    factors = tuple(
        value.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for value in (A, C, B)
    )
    train_correction = _correction_from_weights(design, *factors)
    train_metrics = _metrics_from_rows(
        targets=targets,
        base=base,
        correction=train_correction,
        sequence_count=len(train),
    )
    heldout_metrics = None
    if heldout:
        heldout_design, heldout_base, heldout_targets = _rows_for_samples(
            heldout,
            kernel=kernel,
        )
        heldout_metrics = _metrics_from_rows(
            targets=heldout_targets,
            base=heldout_base,
            correction=_correction_from_weights(
                heldout_design,
                *factors,
            ),
            sequence_count=len(heldout),
        )
    return TangentPreservingQuadraticEdge(
        A=factors[0],
        C=factors[1],
        B=factors[2],
        lag_count=lag_count,
        base_kernel_sha256=_tensor_sha256(kernel),
        training_samples_sha256=_sample_set_sha256(train),
        heldout_samples_sha256=(
            None if not heldout else _sample_set_sha256(heldout)
        ),
        training_sample_count=len(train),
        heldout_sample_count=len(heldout),
        steps=steps,
        learning_rate=learning_rate,
        ridge=ridge,
        seed=seed,
        minibatch_rows=minibatch,
        train_metrics=train_metrics,
        heldout_metrics=heldout_metrics,
    )


def evaluate_tangent_preserving_quadratic_edge(
    edge: TangentPreservingQuadraticEdge,
    samples: Sequence[TangentPreservingQuadraticSample],
    *,
    base_kernel: Tensor,
) -> TangentPreservingQuadraticEdgeMetrics:
    """Evaluate one frozen correction against its authenticated base kernel."""

    if not isinstance(edge, TangentPreservingQuadraticEdge):
        raise TypeError("edge must be a TangentPreservingQuadraticEdge")
    edge.validate_integrity()
    materialized = _canonical_samples(samples, label="samples")
    kernel = _canonical_base_kernel(base_kernel)
    if _tensor_sha256(kernel) != edge.base_kernel_sha256:
        raise ValueError("base_kernel does not match the quadratic edge")
    if tuple(kernel.shape) != (
        edge.lag_count,
        edge.source_rank,
        edge.target_rank,
    ):
        raise ValueError("base_kernel geometry does not match the edge")
    _validate_sample_geometry(
        materialized,
        source_rank=edge.source_rank,
        target_rank=edge.target_rank,
        label="samples",
    )
    design, base, targets = _rows_for_samples(
        materialized,
        kernel=kernel,
    )
    correction = _correction_from_weights(
        design,
        edge.A,
        edge.C,
        edge.B,
    )
    return _metrics_from_rows(
        targets=targets,
        base=base,
        correction=correction,
        sequence_count=len(materialized),
    )
