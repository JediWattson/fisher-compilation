"""Bounded tangent-preserving correction for finite modal displacements.

The frozen all-on graph produces target-modal predictions ``p``.  This module
adds a small radial residual driven by causal retained-latent energy ``E``:

``g = E / (kappa + E)``

``p* = p + features(p, g) @ W``

where ``features`` is either ``g * p`` or ``[g * p, g**2 * p]``.  ``W`` may
be stored densely or as canonical-sign reduced-rank factors ``A @ B``.
There is deliberately no bias.

Under a global displacement scale ``epsilon``, retained latents and graph
predictions are first order while energy is second order.  The correction is
therefore third order: it is exactly zero at zero displacement and preserves
the graph's tangent/JVP at the reference point.

Plans are strict, domain-separated artifacts.  Coefficient tensors are
canonical contiguous CPU float64 values and are bound to both the frozen
source-graph artifact and an internally derived exact fit-panel binding.  The
runtime is explicitly all-on-only; routed masks, route fractions, and selective
validity masks are rejected.

This is a generic post-map correction primitive, not a projection-capacity or
missing-carrier remedy.  In particular, fitting one of these plans does not
authorize the current 64-mode Gemma candidate: that candidate's projection and
carrier gates fail.  The primitive is intentionally not integrated with Gemma
and may only be evaluated after a candidate independently passes those gates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

import torch
from torch import Tensor, nn


RadialCorrectionRepresentation = Literal["factorized", "dense"]

__all__ = [
    "CausalRetainedLatentEnergyAccounting",
    "PreparedRadialFiniteDisplacementCorrection",
    "RadialCorrectionExecutionAccounting",
    "RadialCorrectionRepresentation",
    "RadialFiniteDisplacementCorrectionPlan",
    "causal_retained_latent_energy",
    "causal_retained_latent_energy_accounting",
    "derive_radial_finite_displacement_fit_binding_sha256",
    "family_balanced_row_weights",
    "fit_radial_finite_displacement_correction",
    "load_radial_finite_displacement_correction_plan",
]


_ARTIFACT_KIND = (
    "fisher_graph.radial_finite_displacement_correction_plan"
)
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction_plan.v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction.tensor.v1\0"
)
_RUNTIME_TENSOR_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction.runtime.v1\0"
)
_RUNTIME_HEADER_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction."
    b"prepared_header.v1\0"
)
_WEIGHT_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction.weights.v1\0"
)
_FIT_BINDING_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction.fit_binding.v1\0"
)
_BOOLEAN_TENSOR_DOMAIN = (
    b"fisher_graph.radial_finite_displacement_correction.boolean_tensor.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPRESENTATIONS = frozenset({"factorized", "dense"})
_RUNTIME_DTYPES = frozenset({torch.float32, torch.float64})
_POSITION_DTYPES = frozenset({torch.int32, torch.int64})
_ARM_BINDING = "all_on"
_ARCHITECTURE = (
    "p_plus_bounded_radial_features_linear_map_without_bias"
)
_GATE_SEMANTICS = "g_equals_E_div_kappa_plus_E"
_FEATURE_SEMANTICS = {
    1: "concatenate_g_times_p",
    2: "concatenate_g_times_p_and_g_squared_times_p",
}
_FIT_OBJECTIVE = (
    "family_balanced_weighted_rms_scaled_ridge_then_"
    "canonical_sign_truncated_svd"
)
_FIT_BINDING_SEMANTICS = (
    "sha256_exact_canonical_fit_tensors_ordered_ids_mask_graph_geometry_and_"
    "hyperparameters"
)


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


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
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


def _strict_state_tensor(value: object, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or value.ndim != 2
        or any(int(width) <= 0 for width in value.shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite contiguous CPU float64 matrix"
        )
    return value


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "shape": tuple(int(width) for width in canonical.shape),
                "dtype": "float64",
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_RUNTIME_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "shape": tuple(int(width) for width in canonical.shape),
                "dtype": str(canonical.dtype),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _optional_tensor_sha256(value: Tensor | None) -> str | None:
    return None if value is None else _tensor_sha256(value)


def _optional_tensor_shape(
    value: Tensor | None,
) -> tuple[int, ...] | None:
    return (
        None
        if value is None
        else tuple(int(width) for width in value.shape)
    )


def _boolean_tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.device.type != "cpu"
        or not value.is_contiguous()
    ):
        raise ValueError(
            "boolean tensor binding requires a contiguous CPU bool Tensor"
        )
    digest = hashlib.sha256()
    digest.update(_BOOLEAN_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "shape": tuple(int(width) for width in value.shape),
                "dtype": "bool",
            }
        )
    )
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_string_ids(
    values: Sequence[str],
    *,
    label: str,
    expected_length: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(values)
    if (
        len(result) != expected_length
        or any(type(value) is not str or not value for value in result)
    ):
        raise ValueError(
            f"{label} must contain one nonempty string per row"
        )
    return result


def _canonical_fit_valid_mask(
    valid_mask: Tensor | None,
    *,
    row_count: int,
) -> Tensor:
    if valid_mask is None:
        return torch.ones(row_count, dtype=torch.bool)
    if (
        not isinstance(valid_mask, Tensor)
        or valid_mask.dtype != torch.bool
        or valid_mask.shape != (row_count,)
    ):
        raise ValueError("valid_mask must be boolean [rows]")
    result = valid_mask.detach().to(device="cpu").contiguous().clone()
    if not bool(result.any()):
        raise ValueError("fit requires at least one eligible row")
    return result


def _fit_binding_from_canonical(
    prediction: Tensor,
    retained_latent_energy: Tensor,
    target: Tensor,
    *,
    family_ids: tuple[str, ...],
    example_ids: tuple[str, ...],
    valid_mask: Tensor,
    source_graph_artifact_sha256: str,
    source_rank: int,
    lag_count: int,
    feature_order: int,
    kappa: float,
    ridge: float,
    representation: RadialCorrectionRepresentation,
    reduced_rank: int | None,
    arm_binding: str,
) -> str:
    return _json_sha256(
        {
            "binding_kind": (
                "exact_canonical_radial_finite_displacement_fit_inputs"
            ),
            "architecture": _ARCHITECTURE,
            "fit_objective": _FIT_OBJECTIVE,
            "fit_binding_semantics": _FIT_BINDING_SEMANTICS,
            "arm_binding": arm_binding,
            "source_graph_artifact_sha256": source_graph_artifact_sha256,
            "source_rank": source_rank,
            "lag_count": lag_count,
            "feature_order": feature_order,
            "kappa": kappa,
            "ridge": ridge,
            "representation": representation,
            "reduced_rank": reduced_rank,
            "row_count": int(prediction.shape[0]),
            "target_modes": int(prediction.shape[1]),
            "tensor_sha256s": {
                "prediction": _tensor_sha256(prediction),
                "retained_latent_energy": _tensor_sha256(
                    retained_latent_energy.reshape(-1, 1)
                ),
                "target": _tensor_sha256(target),
                "valid_mask": _boolean_tensor_sha256(valid_mask),
            },
            "family_ids": list(family_ids),
            "example_ids": list(example_ids),
        },
        domain=_FIT_BINDING_DOMAIN,
    )


def derive_radial_finite_displacement_fit_binding_sha256(
    prediction: Tensor,
    retained_latent_energy: Tensor,
    target: Tensor,
    *,
    family_ids: Sequence[str],
    example_ids: Sequence[str],
    source_graph_artifact_sha256: str,
    source_rank: int,
    lag_count: int,
    feature_order: int,
    kappa: float,
    ridge: float,
    representation: RadialCorrectionRepresentation,
    reduced_rank: int | None = None,
    valid_mask: Tensor | None = None,
    arm_binding: str = _ARM_BINDING,
) -> str:
    """Derive the exact canonical fit-panel and hyperparameter binding.

    This is the only supported source of a fit binding.  The digest covers the
    complete pre-selection tensors, ordered family/example identities, exact
    validity mask, source graph identity, geometry, and all fit choices.
    """

    source_graph = _require_sha256(
        source_graph_artifact_sha256,
        label="source_graph_artifact_sha256",
    )
    source_rank = _positive_int(source_rank, label="source_rank")
    lag_count = _positive_int(lag_count, label="lag_count")
    if type(feature_order) is not int or feature_order not in (1, 2):
        raise ValueError("feature_order must be 1 or 2")
    kappa = _finite_float(kappa, label="kappa", positive=True)
    ridge = _finite_float(ridge, label="ridge", nonnegative=True)
    if arm_binding != _ARM_BINDING:
        raise ValueError("radial correction fitting rejects routed arm semantics")
    if representation not in _REPRESENTATIONS:
        raise ValueError("representation is invalid")

    predictions = _canonical_float_tensor(
        prediction,
        label="prediction",
        ndim=2,
    )
    targets = _canonical_float_tensor(
        target,
        label="target",
        ndim=2,
    )
    if not isinstance(retained_latent_energy, Tensor):
        raise TypeError("retained_latent_energy must be a Tensor")
    if retained_latent_energy.ndim != 1:
        raise ValueError("retained_latent_energy must have shape [rows]")
    energy = _canonical_float_tensor(
        retained_latent_energy.reshape(-1, 1),
        label="retained_latent_energy",
        ndim=2,
    ).reshape(-1)
    if (
        predictions.shape != targets.shape
        or energy.shape[0] != predictions.shape[0]
        or bool((energy < 0.0).any())
    ):
        raise ValueError("fit prediction, energy, and target shapes differ")
    row_count = int(predictions.shape[0])
    families = _canonical_string_ids(
        family_ids,
        label="family_ids",
        expected_length=row_count,
    )
    examples = _canonical_string_ids(
        example_ids,
        label="example_ids",
        expected_length=row_count,
    )
    selected = _canonical_fit_valid_mask(valid_mask, row_count=row_count)

    target_modes = int(predictions.shape[1])
    input_width = feature_order * target_modes
    if representation == "factorized":
        if (
            type(reduced_rank) is not int
            or not 1 <= reduced_rank < min(input_width, target_modes)
        ):
            raise ValueError(
                "factorized fit requires a genuinely reduced positive rank"
            )
    elif reduced_rank is not None:
        raise ValueError("dense fit cannot declare reduced_rank")

    return _fit_binding_from_canonical(
        predictions,
        energy,
        targets,
        family_ids=families,
        example_ids=examples,
        valid_mask=selected,
        source_graph_artifact_sha256=source_graph,
        source_rank=source_rank,
        lag_count=lag_count,
        feature_order=feature_order,
        kappa=kappa,
        ridge=ridge,
        representation=representation,
        reduced_rank=reduced_rank,
        arm_binding=arm_binding,
    )


def _canonical_positions_and_mask(
    positions: Tensor,
    mask: Tensor,
    *,
    label: str,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(positions, Tensor)
        or positions.dtype not in _POSITION_DTYPES
    ):
        raise TypeError(f"{label}_positions must use int32 or int64")
    if positions.device != device:
        raise ValueError(f"{label}_positions are on the wrong device")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError(f"{label}_mask must be a boolean Tensor")
    if mask.device != device:
        raise ValueError(f"{label}_mask is on the wrong device")
    allowed = {(sequence_length,), (batch_size, sequence_length)}
    if tuple(positions.shape) not in allowed or tuple(mask.shape) not in allowed:
        raise ValueError(
            f"{label}_positions and {label}_mask must have shape "
            "[S] or [B, S]"
        )
    expanded_positions = (
        positions.unsqueeze(0).expand(batch_size, sequence_length)
        if positions.ndim == 1
        else positions
    )
    expanded_mask = (
        mask.unsqueeze(0).expand(batch_size, sequence_length)
        if mask.ndim == 1
        else mask
    )
    for batch in range(batch_size):
        selected = expanded_positions[batch][expanded_mask[batch]]
        if selected.numel() > 0:
            if bool((selected < 0).any()):
                raise ValueError(
                    f"valid {label} logical positions must be nonnegative"
                )
            if selected.numel() > 1 and not bool(
                torch.all(selected[1:] > selected[:-1])
            ):
                raise ValueError(
                    f"valid {label} logical positions must be strictly "
                    "increasing"
                )
    return expanded_positions, expanded_mask


@dataclass(frozen=True, slots=True)
class CausalRetainedLatentEnergyAccounting:
    """Formula-level arithmetic counts for the causal energy helper.

    The counts include latent norms, logical lag tests, admitted-source
    accumulation, and normalization. Overflow-safety scaling and saturation
    guards, tensor allocation, masking/index enumeration, and memory traffic
    are intentionally outside their scope.
    """

    batch_size: int
    source_sequence_length: int
    target_sequence_length: int
    source_rank: int
    valid_source_rows: int
    valid_target_rows: int
    targets_with_support: int
    examined_source_target_pairs: int
    admitted_causal_pairs: int

    def __post_init__(self) -> None:
        for field in (
            "batch_size",
            "source_sequence_length",
            "target_sequence_length",
            "source_rank",
        ):
            _positive_int(getattr(self, field), label=field)
        for field in (
            "valid_source_rows",
            "valid_target_rows",
            "targets_with_support",
            "examined_source_target_pairs",
            "admitted_causal_pairs",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if (
            self.valid_source_rows
            > self.batch_size * self.source_sequence_length
            or self.valid_target_rows
            > self.batch_size * self.target_sequence_length
            or self.targets_with_support > self.valid_target_rows
            or self.targets_with_support > self.admitted_causal_pairs
            or self.admitted_causal_pairs
            > self.examined_source_target_pairs
            or self.examined_source_target_pairs
            > self.valid_source_rows * self.valid_target_rows
            or self.admitted_causal_pairs
            > self.targets_with_support * self.source_sequence_length
        ):
            raise ValueError("causal energy accounting is inconsistent")
        if (
            (self.valid_source_rows == 0 or self.valid_target_rows == 0)
            and (
                self.targets_with_support != 0
                or self.examined_source_target_pairs != 0
                or self.admitted_causal_pairs != 0
            )
        ):
            raise ValueError("causal energy accounting is inconsistent")

    @property
    def latent_square_multiplies(self) -> int:
        return self.valid_source_rows * self.source_rank

    @property
    def latent_norm_reduction_additions(self) -> int:
        return self.valid_source_rows * (self.source_rank - 1)

    @property
    def causal_energy_accumulation_additions(self) -> int:
        return self.admitted_causal_pairs - self.targets_with_support

    @property
    def energy_normalization_divisions(self) -> int:
        return self.targets_with_support

    @property
    def logical_lag_subtractions(self) -> int:
        return self.examined_source_target_pairs

    @property
    def logical_lag_range_comparisons(self) -> int:
        return 2 * self.examined_source_target_pairs

    def metadata(self) -> dict[str, object]:
        fields = (
            "batch_size",
            "source_sequence_length",
            "target_sequence_length",
            "source_rank",
            "valid_source_rows",
            "valid_target_rows",
            "targets_with_support",
            "examined_source_target_pairs",
            "admitted_causal_pairs",
            "latent_square_multiplies",
            "latent_norm_reduction_additions",
            "causal_energy_accumulation_additions",
            "energy_normalization_divisions",
            "logical_lag_subtractions",
            "logical_lag_range_comparisons",
        )
        return {
            field: int(getattr(self, field))
            for field in fields
        } | {
            "overflow_safety_guards_included": False,
            "tensor_allocation_included": False,
            "mask_index_enumeration_included": False,
            "memory_traffic_included": False,
        }


def _overflow_safe_mean_square(value: Tensor) -> Tensor:
    """Return a finite float64 mean square, saturating only if unrepresentable."""

    canonical = value.to(dtype=torch.float64)
    scale = canonical.abs().amax()
    if bool(scale == 0.0):
        return canonical.sum() * 0.0
    normalized_mean_square = (canonical / scale).square().mean()
    log_value = (
        2.0 * torch.log(scale) + torch.log(normalized_mean_square)
    )
    maximum = torch.finfo(torch.float64).max
    if bool(log_value >= math.log(maximum)):
        return canonical.new_tensor(maximum)
    result = scale.square() * normalized_mean_square
    if not bool(torch.isfinite(result)):
        raise RuntimeError("stable retained-latent mean square became nonfinite")
    return result


def _overflow_safe_nonnegative_mean(value: Tensor) -> Tensor:
    """Return a finite mean for a nonnegative finite float64 vector."""

    if value.numel() == 0:
        raise ValueError("stable nonnegative mean requires at least one value")
    scale = value.amax()
    if bool(scale == 0.0):
        return value.sum() * 0.0
    result = (value / scale).mean() * scale
    if not bool(torch.isfinite(result)):
        raise RuntimeError("stable retained-latent energy became nonfinite")
    return result


def _causal_energy_and_accounting(
    retained_latent: Tensor,
    *,
    source_positions: Tensor,
    source_mask: Tensor,
    target_positions: Tensor,
    target_mask: Tensor,
    lag_count: int,
) -> tuple[Tensor, CausalRetainedLatentEnergyAccounting]:
    lag_count = _positive_int(lag_count, label="lag_count")
    if (
        not isinstance(retained_latent, Tensor)
        or retained_latent.ndim not in (2, 3)
        or retained_latent.dtype not in _RUNTIME_DTYPES
        or any(int(width) <= 0 for width in retained_latent.shape)
        or not bool(torch.isfinite(retained_latent).all())
    ):
        raise ValueError(
            "retained_latent must be a finite nonempty float32/float64 "
            "[source_sequence, rank] or [batch, source_sequence, rank] Tensor"
        )
    squeeze = retained_latent.ndim == 2
    batched = (
        retained_latent.unsqueeze(0)
        if squeeze
        else retained_latent
    )
    batch_size = int(batched.shape[0])
    source_length = int(batched.shape[1])
    source_rank = int(batched.shape[2])
    if not isinstance(target_positions, Tensor):
        raise TypeError("target_positions must be a Tensor")
    if target_positions.ndim not in (1, 2):
        raise ValueError("target_positions must have shape [T] or [B, T]")
    target_length = int(target_positions.shape[-1])
    source_grid, sources = _canonical_positions_and_mask(
        source_positions,
        source_mask,
        label="source",
        batch_size=batch_size,
        sequence_length=source_length,
        device=retained_latent.device,
    )
    target_grid, targets = _canonical_positions_and_mask(
        target_positions,
        target_mask,
        label="target",
        batch_size=batch_size,
        sequence_length=target_length,
        device=retained_latent.device,
    )

    source_mean_squares = torch.zeros(
        (batch_size, source_length),
        dtype=torch.float64,
        device=retained_latent.device,
    )
    source_indices_by_batch: list[list[int]] = []
    for batch in range(batch_size):
        indices = torch.nonzero(
            sources[batch],
            as_tuple=False,
        ).flatten().tolist()
        source_indices_by_batch.append(indices)
        if indices:
            index = torch.tensor(
                indices,
                dtype=torch.int64,
                device=retained_latent.device,
            )
            mean_squares = torch.stack(
                tuple(
                    _overflow_safe_mean_square(batched[batch, row])
                    for row in indices
                )
            )
            source_mean_squares[batch] = source_mean_squares[
                batch
            ].index_copy(0, index, mean_squares)

    energy = torch.zeros(
        (batch_size, target_length),
        dtype=torch.float64,
        device=retained_latent.device,
    )
    valid_targets = 0
    supported_targets = 0
    examined_pairs = 0
    admitted_pairs = 0
    for batch in range(batch_size):
        source_indices = source_indices_by_batch[batch]
        target_indices = torch.nonzero(
            targets[batch],
            as_tuple=False,
        ).flatten().tolist()
        valid_targets += len(target_indices)
        for target_index in target_indices:
            target_position = int(
                target_grid[batch, target_index].item()
            )
            admitted: list[int] = []
            for source_index in source_indices:
                examined_pairs += 1
                lag = target_position - int(
                    source_grid[batch, source_index].item()
                )
                if 0 <= lag < lag_count:
                    admitted.append(source_index)
            if not admitted:
                continue
            supported_targets += 1
            admitted_pairs += len(admitted)
            index = torch.tensor(
                admitted,
                dtype=torch.int64,
                device=retained_latent.device,
            )
            energy[batch, target_index] = _overflow_safe_nonnegative_mean(
                source_mean_squares[batch].index_select(0, index)
            )

    accounting = CausalRetainedLatentEnergyAccounting(
        batch_size=batch_size,
        source_sequence_length=source_length,
        target_sequence_length=target_length,
        source_rank=source_rank,
        valid_source_rows=int(sources.sum().item()),
        valid_target_rows=valid_targets,
        targets_with_support=supported_targets,
        examined_source_target_pairs=examined_pairs,
        admitted_causal_pairs=admitted_pairs,
    )
    return (energy[0] if squeeze else energy), accounting


def causal_retained_latent_energy(
    retained_latent: Tensor,
    *,
    source_positions: Tensor,
    source_mask: Tensor,
    target_positions: Tensor,
    target_mask: Tensor,
    lag_count: int,
) -> Tensor:
    """Return causal mean retained-latent energy at explicit target positions.

    For each valid target with ``n`` admitted sources, the value is

    ``sum(||z_s||^2) / (source_rank * n)``

    over source rows satisfying ``0 <= target_position - source_position <
    lag_count``.  Invalid targets and valid targets without causal support are
    exactly zero. Computation uses scaled float64 means and saturates only when
    the mathematical energy exceeds the largest representable float64 value.
    """

    energy, _ = _causal_energy_and_accounting(
        retained_latent,
        source_positions=source_positions,
        source_mask=source_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        lag_count=lag_count,
    )
    return energy


def causal_retained_latent_energy_accounting(
    retained_latent: Tensor,
    *,
    source_positions: Tensor,
    source_mask: Tensor,
    target_positions: Tensor,
    target_mask: Tensor,
    lag_count: int,
) -> CausalRetainedLatentEnergyAccounting:
    """Return scoped formula counts for the causal energy computation."""

    _, accounting = _causal_energy_and_accounting(
        retained_latent,
        source_positions=source_positions,
        source_mask=source_mask,
        target_positions=target_positions,
        target_mask=target_mask,
        lag_count=lag_count,
    )
    return accounting


@dataclass(frozen=True, slots=True)
class RadialCorrectionExecutionAccounting:
    """Exact deployed correction arithmetic from precomputed energy.

    Integrity hashing, tensor allocation, row indexing, and memory traffic are
    outside the arithmetic counts.  Retained-latent energy has its own exact
    scoped accounting helper above.
    """

    eligible_target_rows: int
    target_modes: int
    feature_order: int
    representation: RadialCorrectionRepresentation
    reduced_rank: int | None

    def __post_init__(self) -> None:
        if (
            type(self.eligible_target_rows) is not int
            or self.eligible_target_rows < 0
        ):
            raise ValueError("eligible_target_rows must be nonnegative")
        _positive_int(self.target_modes, label="target_modes")
        if self.feature_order not in (1, 2):
            raise ValueError("feature_order must be 1 or 2")
        if self.representation not in _REPRESENTATIONS:
            raise ValueError("representation is invalid")
        if self.representation == "factorized":
            _positive_int(self.reduced_rank, label="reduced_rank")
            if self.reduced_rank >= min(
                self.input_width,
                self.target_modes,
            ):
                raise ValueError(
                    "factorized representation must be reduced rank"
                )
        elif self.reduced_rank is not None:
            raise ValueError("dense representation cannot have reduced_rank")

    @property
    def input_width(self) -> int:
        return self.feature_order * self.target_modes

    @property
    def stored_coefficient_count(self) -> int:
        if self.representation == "dense":
            return 1 + self.input_width * self.target_modes
        assert self.reduced_rank is not None
        return (
            1
            + self.input_width * self.reduced_rank
            + self.reduced_rank * self.target_modes
        )

    @property
    def linear_macs_per_target_row(self) -> int:
        if self.representation == "dense":
            return self.input_width * self.target_modes
        assert self.reduced_rank is not None
        return (
            self.input_width * self.reduced_rank
            + self.reduced_rank * self.target_modes
        )

    @property
    def linear_macs(self) -> int:
        return self.eligible_target_rows * self.linear_macs_per_target_row

    @property
    def gate_denominator_additions(self) -> int:
        return self.eligible_target_rows

    @property
    def gate_divisions(self) -> int:
        return 2 * self.eligible_target_rows

    @property
    def gate_branch_comparisons(self) -> int:
        return self.eligible_target_rows

    @property
    def gate_power_multiplies(self) -> int:
        return self.eligible_target_rows * (self.feature_order - 1)

    @property
    def feature_multiplies(self) -> int:
        return (
            self.eligible_target_rows
            * self.input_width
        )

    @property
    def output_additions(self) -> int:
        return self.eligible_target_rows * self.target_modes

    def metadata(self) -> dict[str, object]:
        fields = (
            "eligible_target_rows",
            "target_modes",
            "feature_order",
            "input_width",
            "representation",
            "reduced_rank",
            "stored_coefficient_count",
            "linear_macs_per_target_row",
            "linear_macs",
            "gate_denominator_additions",
            "gate_divisions",
            "gate_branch_comparisons",
            "gate_power_multiplies",
            "feature_multiplies",
            "output_additions",
        )
        return {field: getattr(self, field) for field in fields} | {
            "integrity_hashing_included": False,
            "retained_latent_energy_included": False,
            "tensor_allocation_included": False,
            "row_indexing_included": False,
            "memory_traffic_included": False,
        }


def _validate_canonical_factor_signs(right: Tensor) -> None:
    for component in range(int(right.shape[0])):
        row = right[component]
        pivot = int(row.abs().argmax().item())
        if float(row[pivot].item()) <= 0.0:
            raise ValueError(
                "factorized right rows must have nonzero canonical signs"
            )


@dataclass(frozen=True, slots=True, eq=False)
class RadialFiniteDisplacementCorrectionPlan:
    """Authenticated all-on-only radial finite-displacement correction."""

    source_graph_artifact_sha256: str
    fit_binding_sha256: str
    fit_weight_sha256: str
    source_rank: int
    target_modes: int
    lag_count: int
    feature_order: int
    kappa: float
    ridge: float
    representation: RadialCorrectionRepresentation
    left: Tensor | None
    right: Tensor | None
    dense: Tensor | None
    fit_row_count: int
    fit_family_count: int
    fit_example_count: int
    arm_binding: str = _ARM_BINDING
    routing_supported: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_graph_artifact_sha256,
            label="source_graph_artifact_sha256",
        )
        _require_sha256(
            self.fit_binding_sha256,
            label="fit_binding_sha256",
        )
        _require_sha256(
            self.fit_weight_sha256,
            label="fit_weight_sha256",
        )
        _positive_int(self.source_rank, label="source_rank")
        _positive_int(self.target_modes, label="target_modes")
        _positive_int(self.lag_count, label="lag_count")
        if type(self.feature_order) is not int or self.feature_order not in (1, 2):
            raise ValueError("feature_order must be 1 or 2")
        object.__setattr__(
            self,
            "kappa",
            _finite_float(self.kappa, label="kappa", positive=True),
        )
        object.__setattr__(
            self,
            "ridge",
            _finite_float(self.ridge, label="ridge", nonnegative=True),
        )
        for field in (
            "fit_row_count",
            "fit_family_count",
            "fit_example_count",
        ):
            _positive_int(getattr(self, field), label=field)
        if not (
            self.fit_family_count
            <= self.fit_example_count
            <= self.fit_row_count
        ):
            raise ValueError("fit family/example/row counts are inconsistent")
        if (
            self.arm_binding != _ARM_BINDING
            or type(self.routing_supported) is not bool
            or self.routing_supported
        ):
            raise ValueError(
                "radial correction must be bound to all_on with routing "
                "disabled"
            )
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("radial correction artifact header is invalid")
        if self.representation not in _REPRESENTATIONS:
            raise ValueError("representation is invalid")

        if self.representation == "factorized":
            if self.left is None or self.right is None or self.dense is not None:
                raise ValueError(
                    "factorized representation requires only left and right"
                )
            left = _canonical_float_tensor(
                self.left,
                label="left",
                ndim=2,
            )
            right = _canonical_float_tensor(
                self.right,
                label="right",
                ndim=2,
            )
            if (
                left.shape[0] != self.input_width
                or right.shape[1] != self.target_modes
                or left.shape[1] != right.shape[0]
                or int(left.shape[1])
                >= min(self.input_width, self.target_modes)
            ):
                raise ValueError(
                    "factorized correction shapes are incompatible or "
                    "not reduced rank"
                )
            _validate_canonical_factor_signs(right)
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)
            object.__setattr__(self, "dense", None)
        else:
            if self.left is not None or self.right is not None or self.dense is None:
                raise ValueError(
                    "dense representation requires only the dense matrix"
                )
            dense = _canonical_float_tensor(
                self.dense,
                label="dense",
                ndim=2,
            )
            if dense.shape != (self.input_width, self.target_modes):
                raise ValueError("dense correction shape is incompatible")
            object.__setattr__(self, "left", None)
            object.__setattr__(self, "right", None)
            object.__setattr__(self, "dense", dense)

        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("radial correction artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def input_width(self) -> int:
        return self.feature_order * self.target_modes

    @property
    def reduced_rank(self) -> int | None:
        return (
            None
            if self.representation == "dense"
            else int(self.left.shape[1])  # type: ignore[union-attr]
        )

    @property
    def stored_coefficient_count(self) -> int:
        return self.execution_accounting(
            eligible_target_rows=0
        ).stored_coefficient_count

    @property
    def linear_macs_per_target_row(self) -> int:
        return self.execution_accounting(
            eligible_target_rows=0
        ).linear_macs_per_target_row

    @property
    def gate_denominator_additions_per_target_row(self) -> int:
        return 1

    @property
    def gate_divisions_per_target_row(self) -> int:
        return 2

    @property
    def gate_branch_comparisons_per_target_row(self) -> int:
        return 1

    @property
    def gate_power_multiplies_per_target_row(self) -> int:
        return self.feature_order - 1

    @property
    def feature_multiplies_per_target_row(self) -> int:
        return self.input_width

    @property
    def output_additions_per_target_row(self) -> int:
        return self.target_modes

    def execution_accounting(
        self,
        *,
        eligible_target_rows: int,
    ) -> RadialCorrectionExecutionAccounting:
        return RadialCorrectionExecutionAccounting(
            eligible_target_rows=eligible_target_rows,
            target_modes=self.target_modes,
            feature_order=self.feature_order,
            representation=self.representation,
            reduced_rank=self.reduced_rank,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "architecture": _ARCHITECTURE,
            "energy_gate": _GATE_SEMANTICS,
            "feature_semantics": _FEATURE_SEMANTICS[self.feature_order],
            "fit_objective": _FIT_OBJECTIVE,
            "fit_binding_semantics": _FIT_BINDING_SEMANTICS,
            "no_bias": True,
            "zero_response_at_zero_displacement": True,
            "zero_correction_jacobian_at_zero": True,
            "post_map_only": True,
            "projection_capacity_evidence": False,
            "carrier_reconstruction_evidence": False,
            "candidate_authorization": False,
            "arm_binding": self.arm_binding,
            "routing_supported": self.routing_supported,
            "source_graph_artifact_sha256": (
                self.source_graph_artifact_sha256
            ),
            "fit_binding_sha256": self.fit_binding_sha256,
            "fit_weight_sha256": self.fit_weight_sha256,
            "source_rank": self.source_rank,
            "target_modes": self.target_modes,
            "lag_count": self.lag_count,
            "feature_order": self.feature_order,
            "input_width": self.input_width,
            "kappa": self.kappa,
            "ridge": self.ridge,
            "representation": self.representation,
            "reduced_rank": self.reduced_rank,
            "fit_row_count": self.fit_row_count,
            "fit_family_count": self.fit_family_count,
            "fit_example_count": self.fit_example_count,
            "tensor_sha256s": {
                "left": _optional_tensor_sha256(self.left),
                "right": _optional_tensor_sha256(self.right),
                "dense": _optional_tensor_sha256(self.dense),
            },
            "tensor_shapes": {
                "left": _optional_tensor_shape(self.left),
                "right": _optional_tensor_shape(self.right),
                "dense": _optional_tensor_shape(self.dense),
            },
            "stored_coefficient_count": self.stored_coefficient_count,
            "linear_macs_per_target_row": (
                self.linear_macs_per_target_row
            ),
            "elementwise_per_target_row": {
                "gate_denominator_additions": (
                    self.gate_denominator_additions_per_target_row
                ),
                "gate_divisions": self.gate_divisions_per_target_row,
                "gate_branch_comparisons": (
                    self.gate_branch_comparisons_per_target_row
                ),
                "gate_power_multiplies": (
                    self.gate_power_multiplies_per_target_row
                ),
                "feature_multiplies": (
                    self.feature_multiplies_per_target_row
                ),
                "output_additions": (
                    self.output_additions_per_target_row
                ),
            },
            "accounting_scope": {
                "retained_latent_energy": (
                    "reported_by_separate_causal_energy_helper"
                ),
                "integrity_hashing_included": False,
                "tensor_allocation_included": False,
                "row_indexing_included": False,
                "memory_traffic_included": False,
            },
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_ARTIFACT_DOMAIN,
        )

    def validate_integrity(self) -> None:
        for label, value in (
            ("left", self.left),
            ("right", self.right),
            ("dense", self.dense),
        ):
            if value is None:
                continue
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float64
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"radial correction tensor {label} drifted"
                )
        if self.right is not None:
            _validate_canonical_factor_signs(self.right)
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("radial correction artifact hash mismatch")

    def validate_fit_binding(
        self,
        prediction: Tensor,
        retained_latent_energy: Tensor,
        target: Tensor,
        *,
        family_ids: Sequence[str],
        example_ids: Sequence[str],
        valid_mask: Tensor | None = None,
    ) -> None:
        """Recompute and validate the exact panel that produced this plan."""

        self.validate_integrity()
        actual = derive_radial_finite_displacement_fit_binding_sha256(
            prediction,
            retained_latent_energy,
            target,
            family_ids=family_ids,
            example_ids=example_ids,
            source_graph_artifact_sha256=(
                self.source_graph_artifact_sha256
            ),
            source_rank=self.source_rank,
            lag_count=self.lag_count,
            feature_order=self.feature_order,
            kappa=self.kappa,
            ridge=self.ridge,
            representation=self.representation,
            reduced_rank=self.reduced_rank,
            valid_mask=valid_mask,
            arm_binding=self.arm_binding,
        )
        if actual != self.fit_binding_sha256:
            raise ValueError(
                "radial correction fit data differs from its exact binding"
            )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "left": None if self.left is None else self.left.clone(),
            "right": None if self.right is None else self.right.clone(),
            "dense": None if self.dense is None else self.dense.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "RadialFiniteDisplacementCorrectionPlan":
        payload_keys = {
            "artifact_kind",
            "format_version",
            "architecture",
            "energy_gate",
            "feature_semantics",
            "fit_objective",
            "fit_binding_semantics",
            "no_bias",
            "zero_response_at_zero_displacement",
            "zero_correction_jacobian_at_zero",
            "post_map_only",
            "projection_capacity_evidence",
            "carrier_reconstruction_evidence",
            "candidate_authorization",
            "arm_binding",
            "routing_supported",
            "source_graph_artifact_sha256",
            "fit_binding_sha256",
            "fit_weight_sha256",
            "source_rank",
            "target_modes",
            "lag_count",
            "feature_order",
            "input_width",
            "kappa",
            "ridge",
            "representation",
            "reduced_rank",
            "fit_row_count",
            "fit_family_count",
            "fit_example_count",
            "tensor_sha256s",
            "tensor_shapes",
            "stored_coefficient_count",
            "linear_macs_per_target_row",
            "elementwise_per_target_row",
            "accounting_scope",
        }
        expected = payload_keys | {
            "left",
            "right",
            "dense",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("radial correction state fields are invalid")
        tensors: dict[str, Tensor | None] = {}
        for label in ("left", "right", "dense"):
            raw = state[label]
            tensors[label] = (
                None
                if raw is None
                else _strict_state_tensor(raw, label=label)
            )
        declared_hashes = state["tensor_sha256s"]
        declared_shapes = state["tensor_shapes"]
        if (
            not isinstance(declared_hashes, Mapping)
            or set(declared_hashes) != {"left", "right", "dense"}
            or not isinstance(declared_shapes, Mapping)
            or set(declared_shapes) != {"left", "right", "dense"}
        ):
            raise ValueError("radial correction tensor declarations are invalid")
        for label, value in tensors.items():
            if (
                declared_hashes[label]
                != _optional_tensor_sha256(value)
                or declared_shapes[label]
                != _optional_tensor_shape(value)
            ):
                raise ValueError(
                    f"radial correction {label} tensor declaration drifted"
                )
        result = cls(
            source_graph_artifact_sha256=state[
                "source_graph_artifact_sha256"
            ],  # type: ignore[arg-type]
            fit_binding_sha256=state[
                "fit_binding_sha256"
            ],  # type: ignore[arg-type]
            fit_weight_sha256=state[
                "fit_weight_sha256"
            ],  # type: ignore[arg-type]
            source_rank=state["source_rank"],  # type: ignore[arg-type]
            target_modes=state["target_modes"],  # type: ignore[arg-type]
            lag_count=state["lag_count"],  # type: ignore[arg-type]
            feature_order=state["feature_order"],  # type: ignore[arg-type]
            kappa=state["kappa"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            representation=state[
                "representation"
            ],  # type: ignore[arg-type]
            left=tensors["left"],
            right=tensors["right"],
            dense=tensors["dense"],
            fit_row_count=state[
                "fit_row_count"
            ],  # type: ignore[arg-type]
            fit_family_count=state[
                "fit_family_count"
            ],  # type: ignore[arg-type]
            fit_example_count=state[
                "fit_example_count"
            ],  # type: ignore[arg-type]
            arm_binding=state["arm_binding"],  # type: ignore[arg-type]
            routing_supported=state[
                "routing_supported"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        raw_payload = {key: state[key] for key in payload_keys}
        computed_payload = result._hash_payload()
        try:
            canonical_payload_drifted = (
                _canonical_json_bytes(raw_payload)
                != _canonical_json_bytes(computed_payload)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "radial correction derived declarations drifted"
            ) from error
        if raw_payload != computed_payload or canonical_payload_drifted:
            raise ValueError("radial correction derived declarations drifted")
        return result

    def prepare(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "PreparedRadialFiniteDisplacementCorrection":
        self.validate_integrity()
        return PreparedRadialFiniteDisplacementCorrection(
            self,
            device=device,
            dtype=dtype,
        )


class PreparedRadialFiniteDisplacementCorrection(nn.Module):
    """Validate-on-use runtime for one frozen radial correction plan."""

    def __init__(
        self,
        plan: RadialFiniteDisplacementCorrectionPlan,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not isinstance(plan, RadialFiniteDisplacementCorrectionPlan):
            raise TypeError(
                "plan must be a RadialFiniteDisplacementCorrectionPlan"
            )
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("runtime dtype must be float32 or float64")
        plan.validate_integrity()
        runtime_device = torch.device(device)
        self.plan_artifact_sha256 = plan.artifact_sha256
        self.source_graph_artifact_sha256 = (
            plan.source_graph_artifact_sha256
        )
        self.fit_binding_sha256 = plan.fit_binding_sha256
        self.fit_weight_sha256 = plan.fit_weight_sha256
        self.arm_binding = plan.arm_binding
        self.routing_supported = plan.routing_supported
        self.source_rank = plan.source_rank
        self.target_modes = plan.target_modes
        self.lag_count = plan.lag_count
        self.feature_order = plan.feature_order
        self.representation = plan.representation
        self.reduced_rank = plan.reduced_rank
        self.stored_coefficient_count = plan.stored_coefficient_count
        self.linear_macs_per_target_row = (
            plan.linear_macs_per_target_row
        )
        self.register_buffer(
            "kappa",
            torch.tensor(
                plan.kappa,
                device=runtime_device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "left",
            (
                None
                if plan.left is None
                else plan.left.to(
                    device=runtime_device,
                    dtype=dtype,
                ).contiguous().clone()
            ),
        )
        self.register_buffer(
            "right",
            (
                None
                if plan.right is None
                else plan.right.to(
                    device=runtime_device,
                    dtype=dtype,
                ).contiguous().clone()
            ),
        )
        self.register_buffer(
            "dense",
            (
                None
                if plan.dense is None
                else plan.dense.to(
                    device=runtime_device,
                    dtype=dtype,
                ).contiguous().clone()
            ),
        )
        self._expected_buffer_sha256s = {
            label: (
                None
                if value is None
                else _runtime_tensor_sha256(value)
            )
            for label, value in (
                ("kappa", self.kappa),
                ("left", self.left),
                ("right", self.right),
                ("dense", self.dense),
            )
        }
        self._expected_header_sha256 = _json_sha256(
            self._runtime_header_payload(),
            domain=_RUNTIME_HEADER_DOMAIN,
        )
        self.validate_integrity()

    @property
    def device(self) -> torch.device:
        return self.kappa.device

    @property
    def dtype(self) -> torch.dtype:
        return self.kappa.dtype

    def _runtime_header_payload(self) -> dict[str, object]:
        return {
            "plan_artifact_sha256": self.plan_artifact_sha256,
            "source_graph_artifact_sha256": (
                self.source_graph_artifact_sha256
            ),
            "fit_binding_sha256": self.fit_binding_sha256,
            "fit_weight_sha256": self.fit_weight_sha256,
            "arm_binding": self.arm_binding,
            "routing_supported": self.routing_supported,
            "source_rank": self.source_rank,
            "target_modes": self.target_modes,
            "lag_count": self.lag_count,
            "feature_order": self.feature_order,
            "representation": self.representation,
            "reduced_rank": self.reduced_rank,
            "stored_coefficient_count": self.stored_coefficient_count,
            "linear_macs_per_target_row": (
                self.linear_macs_per_target_row
            ),
            "runtime_device": str(self.device),
            "runtime_dtype": str(self.dtype),
            "post_map_only": True,
            "candidate_authorization": False,
        }

    def validate_integrity(self) -> None:
        try:
            header_sha256 = _json_sha256(
                self._runtime_header_payload(),
                domain=_RUNTIME_HEADER_DOMAIN,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "prepared radial correction header drifted"
            ) from error
        if (
            header_sha256 != self._expected_header_sha256
            or self.arm_binding != _ARM_BINDING
            or self.routing_supported
            or self.kappa.ndim != 0
            or not bool(torch.isfinite(self.kappa))
            or float(self.kappa.item()) <= 0.0
        ):
            raise ValueError("prepared radial correction header drifted")
        for label in ("kappa", "left", "right", "dense"):
            value = getattr(self, label)
            expected = self._expected_buffer_sha256s[label]
            if value is None:
                if expected is not None:
                    raise ValueError(
                        f"prepared radial correction {label} drifted"
                    )
                continue
            if (
                value.device != self.device
                or value.dtype != self.dtype
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
                or _runtime_tensor_sha256(value) != expected
            ):
                raise ValueError(
                    f"prepared radial correction {label} drifted"
                )

    def _validate_source_graph_binding(
        self,
        source_graph_artifact_sha256: str,
    ) -> None:
        actual = _require_sha256(
            source_graph_artifact_sha256,
            label="source_graph_artifact_sha256",
        )
        if actual != self.source_graph_artifact_sha256:
            raise ValueError(
                "executing source graph artifact does not match the radial "
                "correction binding"
            )

    def execution_accounting(
        self,
        *,
        eligible_target_rows: int,
    ) -> RadialCorrectionExecutionAccounting:
        self.validate_integrity()
        return RadialCorrectionExecutionAccounting(
            eligible_target_rows=eligible_target_rows,
            target_modes=self.target_modes,
            feature_order=self.feature_order,
            representation=self.representation,
            reduced_rank=self.reduced_rank,
        )

    def _reject_routing(
        self,
        *,
        arm: str,
        route_mask: Tensor | None,
        route_fraction: float | None,
    ) -> None:
        if (
            arm != _ARM_BINDING
            or route_mask is not None
            or route_fraction is not None
        ):
            raise ValueError(
                "radial correction is bound to all_on and rejects routing "
                "semantics"
            )

    def _validate_inputs(
        self,
        prediction: Tensor,
        energy: Tensor,
        valid_target_mask: Tensor,
    ) -> None:
        if (
            not isinstance(prediction, Tensor)
            or prediction.ndim not in (2, 3)
            or prediction.shape[-1] != self.target_modes
            or prediction.dtype != self.dtype
            or prediction.device != self.device
            or not bool(torch.isfinite(prediction).all())
        ):
            raise ValueError(
                "prediction must be finite runtime [T, modes] or "
                "[B, T, modes] data"
            )
        prefix = prediction.shape[:-1]
        if (
            not isinstance(energy, Tensor)
            or energy.shape != prefix
            or energy.dtype not in _RUNTIME_DTYPES
            or energy.device != self.device
            or not bool(torch.isfinite(energy).all())
            or bool((energy < 0.0).any())
        ):
            raise ValueError(
                "energy must be finite nonnegative runtime data matching "
                "prediction rows"
            )
        if (
            not isinstance(valid_target_mask, Tensor)
            or valid_target_mask.dtype != torch.bool
            or valid_target_mask.device != self.device
            or valid_target_mask.shape != prefix
        ):
            raise ValueError(
                "valid_target_mask must be matching boolean runtime data"
            )
        if not bool(valid_target_mask.all()):
            raise ValueError(
                "all_on radial correction requires every supplied runtime row; "
                "selective or padding masks require a provenance-bound graph "
                "integration"
            )

    def _selected_correction(
        self,
        prediction: Tensor,
        energy: Tensor,
        valid_target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        flat_prediction = prediction.reshape(-1, self.target_modes)
        flat_energy = energy.reshape(-1)
        flat_mask = valid_target_mask.reshape(-1)
        indices = torch.nonzero(
            flat_mask,
            as_tuple=False,
        ).flatten()
        if indices.numel() == 0:
            return indices, flat_prediction.new_empty(
                (0, self.target_modes)
            )
        selected_prediction = flat_prediction.index_select(0, indices)
        selected_energy = flat_energy.index_select(0, indices)
        gate = _overflow_safe_radial_gate(
            selected_energy,
            self.kappa,
        ).to(dtype=selected_prediction.dtype)
        features = [gate.unsqueeze(-1) * selected_prediction]
        if self.feature_order == 2:
            features.append(gate.square().unsqueeze(-1) * selected_prediction)
        design = torch.cat(features, dim=-1)
        if self.representation == "factorized":
            assert self.left is not None and self.right is not None
            selected_correction = (design @ self.left) @ self.right
        else:
            assert self.dense is not None
            selected_correction = design @ self.dense
        return indices, selected_correction

    def _correction(
        self,
        prediction: Tensor,
        energy: Tensor,
        valid_target_mask: Tensor,
    ) -> Tensor:
        indices, selected_correction = self._selected_correction(
            prediction,
            energy,
            valid_target_mask,
        )
        flat_correction = torch.zeros_like(
            prediction.reshape(-1, self.target_modes)
        )
        flat_correction = flat_correction.index_copy(
            0,
            indices,
            selected_correction,
        )
        return flat_correction.reshape_as(prediction)

    def correction(
        self,
        prediction: Tensor,
        energy: Tensor,
        *,
        valid_target_mask: Tensor,
        source_graph_artifact_sha256: str,
        arm: str = _ARM_BINDING,
        route_mask: Tensor | None = None,
        route_fraction: float | None = None,
    ) -> Tensor:
        """Return the correction for an all-valid, all-on runtime row set."""

        self._reject_routing(
            arm=arm,
            route_mask=route_mask,
            route_fraction=route_fraction,
        )
        self.validate_integrity()
        self._validate_source_graph_binding(
            source_graph_artifact_sha256
        )
        self._validate_inputs(prediction, energy, valid_target_mask)
        return self._correction(
            prediction,
            energy,
            valid_target_mask,
        )

    def forward(
        self,
        prediction: Tensor,
        energy: Tensor,
        *,
        valid_target_mask: Tensor,
        source_graph_artifact_sha256: str,
        arm: str = _ARM_BINDING,
        route_mask: Tensor | None = None,
        route_fraction: float | None = None,
    ) -> Tensor:
        """Return ``p*`` for an all-valid, all-on runtime row set.

        This generic primitive has no authenticated execution grid, so selective
        and padding masks are rejected rather than treated as a routing surface.
        """

        self._reject_routing(
            arm=arm,
            route_mask=route_mask,
            route_fraction=route_fraction,
        )
        self.validate_integrity()
        self._validate_source_graph_binding(
            source_graph_artifact_sha256
        )
        self._validate_inputs(prediction, energy, valid_target_mask)
        indices, selected_correction = self._selected_correction(
            prediction,
            energy,
            valid_target_mask,
        )
        flat_prediction = prediction.reshape(-1, self.target_modes)
        flat_output = flat_prediction.clone()
        selected_output = (
            flat_prediction.index_select(0, indices)
            + selected_correction
        )
        return flat_output.index_copy(
            0,
            indices,
            selected_output,
        ).reshape_as(prediction)


def family_balanced_row_weights(
    family_ids: Sequence[str],
    example_ids: Sequence[str],
) -> Tensor:
    """Return row weights with equal mass per family and example.

    Every family receives total weight ``1 / number_of_families``.  Within a
    family, every example receives equal mass, and every eligible row within
    an example receives equal mass.
    """

    if isinstance(family_ids, (str, bytes)) or not isinstance(
        family_ids,
        Sequence,
    ):
        raise TypeError("family_ids must be a sequence")
    row_count = len(family_ids)
    families = _canonical_string_ids(
        family_ids,
        label="family_ids",
        expected_length=row_count,
    )
    examples = _canonical_string_ids(
        example_ids,
        label="example_ids",
        expected_length=row_count,
    )
    if row_count == 0:
        raise ValueError("family-balanced weighting requires at least one row")
    example_family: dict[str, str] = {}
    for family, example in zip(families, examples, strict=True):
        previous = example_family.setdefault(example, family)
        if previous != family:
            raise ValueError("one example cannot belong to multiple families")
    unique_families = tuple(sorted(set(families)))
    examples_by_family = {
        family: tuple(
            sorted(
                example
                for example, owner in example_family.items()
                if owner == family
            )
        )
        for family in unique_families
    }
    row_counts_by_example = {
        example: examples.count(example)
        for example in example_family
    }
    family_count = len(unique_families)
    weights = torch.empty(row_count, dtype=torch.float64)
    for row, (family, example) in enumerate(
        zip(families, examples, strict=True)
    ):
        weights[row] = 1.0 / (
            family_count
            * len(examples_by_family[family])
            * row_counts_by_example[example]
        )
    if not torch.allclose(
        weights.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=0.0,
        atol=2e-15,
    ):
        raise RuntimeError("family-balanced row weights failed normalization")
    return weights.contiguous()


def _weight_sha256(weights: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(_WEIGHT_DOMAIN)
    digest.update(_tensor_sha256(weights.reshape(-1, 1)).encode("ascii"))
    return digest.hexdigest()


def _overflow_safe_radial_gate(
    energy: Tensor,
    kappa: float | Tensor,
) -> Tensor:
    """Compute ``E / (kappa + E)`` without overflowing the denominator."""

    canonical_energy = energy.to(dtype=torch.float64)
    canonical_kappa = torch.as_tensor(
        kappa,
        dtype=torch.float64,
        device=energy.device,
    )
    if (
        not bool(torch.isfinite(canonical_energy).all())
        or bool((canonical_energy < 0.0).any())
        or canonical_kappa.ndim != 0
        or not bool(torch.isfinite(canonical_kappa))
        or float(canonical_kappa.item()) <= 0.0
    ):
        raise ValueError("radial gate requires finite nonnegative energy and kappa")
    lower = canonical_energy <= canonical_kappa
    gate = torch.empty_like(canonical_energy)
    if bool(lower.any()):
        ratio = canonical_energy[lower] / canonical_kappa
        gate[lower] = ratio / (1.0 + ratio)
    upper = ~lower
    if bool(upper.any()):
        reciprocal_ratio = canonical_kappa / canonical_energy[upper]
        gate[upper] = 1.0 / (1.0 + reciprocal_ratio)
    if not bool(torch.isfinite(gate).all()):
        raise RuntimeError("stable radial gate became nonfinite")
    return gate


def _radial_features(
    prediction: Tensor,
    energy: Tensor,
    *,
    feature_order: int,
    kappa: float | Tensor,
) -> Tensor:
    gate = _overflow_safe_radial_gate(energy, kappa)
    values = [gate.unsqueeze(-1) * prediction]
    if feature_order == 2:
        values.append(gate.square().unsqueeze(-1) * prediction)
    return torch.cat(values, dim=-1)


def _overflow_safe_weighted_rms(
    design: Tensor,
    weights: Tensor,
) -> Tensor:
    """Return per-column weighted RMS without squaring full-scale values."""

    column_scale = design.abs().amax(dim=0)
    divisor = torch.where(
        column_scale > 0.0,
        column_scale,
        torch.ones_like(column_scale),
    )
    normalized = design / divisor
    normalized_second_moment = (
        weights.unsqueeze(-1) * normalized.square()
    ).sum(dim=0)
    result = column_scale * torch.sqrt(normalized_second_moment)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("stable weighted RMS became nonfinite")
    return result


def _canonical_sign_truncated_svd(
    dense: Tensor,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    U, singular, Vh = torch.linalg.svd(
        dense,
        full_matrices=False,
    )
    left = (U[:, :rank] * singular[:rank]).contiguous()
    right = Vh[:rank].contiguous()
    for component in range(rank):
        pivot = int(right[component].abs().argmax().item())
        if float(right[component, pivot].item()) < 0.0:
            left[:, component].neg_()
            right[component].neg_()
    _validate_canonical_factor_signs(right)
    return left, right


def fit_radial_finite_displacement_correction(
    prediction: Tensor,
    retained_latent_energy: Tensor,
    target: Tensor,
    *,
    family_ids: Sequence[str],
    example_ids: Sequence[str],
    source_graph_artifact_sha256: str,
    source_rank: int,
    lag_count: int,
    feature_order: int,
    kappa: float,
    ridge: float,
    representation: RadialCorrectionRepresentation,
    reduced_rank: int | None = None,
    valid_mask: Tensor | None = None,
    arm_binding: str = _ARM_BINDING,
) -> RadialFiniteDisplacementCorrectionPlan:
    """Fit family-balanced weighted ridge and optionally truncate its SVD.

    Feature columns are scaled by their weighted RMS without centering.  The
    scale is folded back into the fitted coefficient map, so the runtime stores
    no normalization vectors and cannot acquire an affine reference term.  The
    fit binding is derived internally from the exact canonical tensors, ordered
    identities, validity mask, source graph, geometry, and hyperparameters.
    """

    if arm_binding != _ARM_BINDING:
        raise ValueError(
            "radial correction fitting rejects routed arm semantics"
        )
    source_graph_artifact_sha256 = _require_sha256(
        source_graph_artifact_sha256,
        label="source_graph_artifact_sha256",
    )
    source_rank = _positive_int(source_rank, label="source_rank")
    lag_count = _positive_int(lag_count, label="lag_count")
    if type(feature_order) is not int or feature_order not in (1, 2):
        raise ValueError("feature_order must be 1 or 2")
    kappa = _finite_float(kappa, label="kappa", positive=True)
    ridge = _finite_float(ridge, label="ridge", nonnegative=True)
    if representation not in _REPRESENTATIONS:
        raise ValueError("representation is invalid")

    predictions = _canonical_float_tensor(
        prediction,
        label="prediction",
        ndim=2,
    )
    targets = _canonical_float_tensor(
        target,
        label="target",
        ndim=2,
    )
    if not isinstance(retained_latent_energy, Tensor):
        raise TypeError("retained_latent_energy must be a Tensor")
    if retained_latent_energy.ndim != 1:
        raise ValueError("retained_latent_energy must have shape [rows]")
    energy = _canonical_float_tensor(
        retained_latent_energy.reshape(-1, 1),
        label="retained_latent_energy",
        ndim=2,
    ).reshape(-1)
    if (
        predictions.shape != targets.shape
        or energy.shape[0] != predictions.shape[0]
        or bool((energy < 0.0).any())
    ):
        raise ValueError("fit prediction, energy, and target shapes differ")
    row_count = int(predictions.shape[0])
    target_modes = int(predictions.shape[1])
    input_width = feature_order * target_modes
    if representation == "factorized":
        if (
            type(reduced_rank) is not int
            or not 1 <= reduced_rank < min(input_width, target_modes)
        ):
            raise ValueError(
                "factorized fit requires a genuinely reduced positive rank"
            )
    elif reduced_rank is not None:
        raise ValueError("dense fit cannot declare reduced_rank")
    families = _canonical_string_ids(
        family_ids,
        label="family_ids",
        expected_length=row_count,
    )
    examples = _canonical_string_ids(
        example_ids,
        label="example_ids",
        expected_length=row_count,
    )
    selected = _canonical_fit_valid_mask(
        valid_mask,
        row_count=row_count,
    )
    fit_binding_sha256 = _fit_binding_from_canonical(
        predictions,
        energy,
        targets,
        family_ids=families,
        example_ids=examples,
        valid_mask=selected,
        source_graph_artifact_sha256=source_graph_artifact_sha256,
        source_rank=source_rank,
        lag_count=lag_count,
        feature_order=feature_order,
        kappa=kappa,
        ridge=ridge,
        representation=representation,
        reduced_rank=reduced_rank,
        arm_binding=arm_binding,
    )
    selected_indices = torch.nonzero(
        selected,
        as_tuple=False,
    ).flatten()
    if selected_indices.numel() == 0:
        raise ValueError("fit requires at least one eligible row")
    selected_predictions = predictions.index_select(0, selected_indices)
    selected_targets = targets.index_select(0, selected_indices)
    selected_energy = energy.index_select(0, selected_indices)
    selected_rows = selected_indices.tolist()
    selected_families = tuple(families[index] for index in selected_rows)
    selected_examples = tuple(examples[index] for index in selected_rows)
    weights = family_balanced_row_weights(
        selected_families,
        selected_examples,
    )
    design = _radial_features(
        selected_predictions,
        selected_energy,
        feature_order=feature_order,
        kappa=kappa,
    )
    residual = selected_targets - selected_predictions

    weighted_rms = _overflow_safe_weighted_rms(design, weights)
    scale_floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(
        weighted_rms > scale_floor,
        weighted_rms,
        torch.ones_like(weighted_rms),
    )
    standardized = design / scales
    root_weights = torch.sqrt(weights).unsqueeze(-1)
    weighted_design = standardized * root_weights
    weighted_residual = residual * root_weights
    if ridge > 0.0:
        gram = weighted_design.T @ weighted_design
        cross = weighted_design.T @ weighted_residual
        fitted_standardized = torch.linalg.solve(
            gram
            + ridge
            * torch.eye(
                gram.shape[0],
                dtype=torch.float64,
            ),
            cross,
        )
    else:
        fitted_standardized = torch.linalg.lstsq(
            weighted_design,
            weighted_residual,
        ).solution
    fitted_dense = (
        fitted_standardized / scales.unsqueeze(-1)
    ).contiguous()
    if not bool(torch.isfinite(fitted_dense).all()):
        raise RuntimeError("radial correction fit became nonfinite")

    if representation == "factorized":
        assert reduced_rank is not None
        left, right = _canonical_sign_truncated_svd(
            fitted_dense,
            rank=reduced_rank,
        )
        dense = None
    else:
        left = None
        right = None
        dense = fitted_dense

    return RadialFiniteDisplacementCorrectionPlan(
        source_graph_artifact_sha256=source_graph_artifact_sha256,
        fit_binding_sha256=fit_binding_sha256,
        fit_weight_sha256=_weight_sha256(weights),
        source_rank=source_rank,
        target_modes=target_modes,
        lag_count=lag_count,
        feature_order=feature_order,
        kappa=kappa,
        ridge=ridge,
        representation=representation,
        left=left,
        right=right,
        dense=dense,
        fit_row_count=int(selected_indices.numel()),
        fit_family_count=len(set(selected_families)),
        fit_example_count=len(set(selected_examples)),
    )


def load_radial_finite_displacement_correction_plan(
    path: Path | str,
) -> RadialFiniteDisplacementCorrectionPlan:
    """Load one strict plan state from a local Torch artifact."""

    raw = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, Mapping):
        raise TypeError("radial correction artifact must contain a mapping")
    return RadialFiniteDisplacementCorrectionPlan.from_state_dict(raw)
