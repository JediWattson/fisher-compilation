"""Pure compiler utilities for the V20q token-VJP local-field rung.

The serving graph is deliberately absent from this module.  V20q uses the
already compiled V20p local signed-field provider and changes only how its two
continuous coefficients are chosen.  The helpers here authenticate the two
ephemeral objects needed by that compiler:

* a full teacher-logit grid reconstructed from capability-selected rows; and
* post-cast central H4 secants measured through the executable provider.

Raw teacher rows, logits, H4 tensors, secants, and gradients are never part of
the scalar receipts returned by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping

import torch
from torch import Tensor


__all__ = [
    "SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP",
    "SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP",
    "SelectedTeacherGridReceipt",
    "SoftPolarityPostCastH4SecantReceipt",
    "build_selected_teacher_grid",
    "build_soft_polarity_post_cast_h4_secants",
    "materialize_complete_h4_post_cast",
    "soft_polarity_post_cast_h4_secant_stability",
    "validate_selected_teacher_grid_replay",
]


SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP = 2.0**-6
SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP = 2.0**-7

_SHA = re.compile(r"^[0-9a-f]{64}$")
_GRID_DOMAIN = b"fisher-graph:selected-teacher-grid:v20q\0"
_SECANT_DOMAIN = b"fisher-graph:soft-polarity-post-cast-h4-secants:v20q\0"
_TENSOR_DOMAIN = b"fisher-graph:v20q-compiler-tensor\0"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(domain: bytes, payload: object) -> str:
    return hashlib.sha256(domain + _canonical(payload)).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("V20q compiler tensor must be a strided tensor")
    array = value.detach().to(device="cpu").contiguous()
    raw = array.view(torch.uint8).numpy().tobytes()
    payload = {
        "dtype": str(array.dtype),
        "shape": tuple(int(width) for width in array.shape),
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return _hash(_TENSOR_DOMAIN, payload)


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_indices(indices: Tensor, *, batch: int, sequence: int) -> Tensor:
    if (
        not isinstance(indices, Tensor)
        or indices.ndim != 2
        or indices.shape[1] != 2
        or indices.dtype != torch.int64
        or indices.requires_grad
        or indices.shape[0] <= 0
    ):
        raise ValueError("selected teacher indices must be nonempty int64 [N,2]")
    result = indices.detach().to(device="cpu").contiguous().clone()
    if (
        bool((result[:, 0] < 0).any())
        or bool((result[:, 0] >= batch).any())
        or bool((result[:, 1] < 0).any())
        or bool((result[:, 1] >= sequence).any())
    ):
        raise ValueError("selected teacher indices escape the target grid")
    flattened = result[:, 0] * sequence + result[:, 1]
    if result.shape[0] > 1 and not bool((flattened[1:] > flattened[:-1]).all()):
        raise ValueError("selected teacher indices must be canonical and unique")
    return result


@dataclass(frozen=True, slots=True)
class SelectedTeacherGridReceipt:
    """Scalar/hash receipt for one ephemeral selected-row teacher grid."""

    selected_rows_sha256: str
    supervised_indices_sha256: str
    teacher_grid_sha256: str
    teacher_grid_shape: tuple[int, int, int]
    teacher_grid_dtype: str
    selected_row_count: int
    zero_filled_row_count: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "selected_rows_sha256",
            "supervised_indices_sha256",
            "teacher_grid_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha(getattr(self, name), label=name)
            )
        if (
            not isinstance(self.teacher_grid_shape, tuple)
            or len(self.teacher_grid_shape) != 3
            or any(type(width) is not int or width <= 0 for width in self.teacher_grid_shape)
            or not isinstance(self.teacher_grid_dtype, str)
            or not self.teacher_grid_dtype.startswith("torch.")
            or type(self.selected_row_count) is not int
            or self.selected_row_count <= 0
            or type(self.zero_filled_row_count) is not int
            or self.zero_filled_row_count < 0
            or self.selected_row_count + self.zero_filled_row_count
            != self.teacher_grid_shape[0] * self.teacher_grid_shape[1]
        ):
            raise ValueError("selected teacher grid receipt geometry differs")
        computed = _hash(_GRID_DOMAIN, self.metadata(include_artifact=False))
        if self.artifact_sha256:
            if _require_sha(self.artifact_sha256, label="teacher grid artifact") != computed:
                raise ValueError("selected teacher grid receipt hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "fisher_graph.selected_teacher_grid.v20q",
            "selected_rows_sha256": self.selected_rows_sha256,
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "teacher_grid_sha256": self.teacher_grid_sha256,
            "teacher_grid_shape": self.teacher_grid_shape,
            "teacher_grid_dtype": self.teacher_grid_dtype,
            "selected_row_count": self.selected_row_count,
            "zero_filled_row_count": self.zero_filled_row_count,
            "selected_rows_replay_exact": True,
            "unselected_rows_are_canonical_zero_fill": True,
            "raw_teacher_or_grid_tensors_serialized": False,
            "compiler_only_not_runtime_input": True,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _hash(_GRID_DOMAIN, self.metadata(include_artifact=False)) != self.artifact_sha256:
            raise RuntimeError("selected teacher grid receipt drifted")


def build_selected_teacher_grid(
    selected_rows: Tensor,
    supervised_indices: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[Tensor, SelectedTeacherGridReceipt]:
    """Scatter authenticated ``[N,V]`` rows into a canonical zero-filled grid."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("teacher grid batch size must be a positive integer")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("teacher grid sequence length must be a positive integer")
    if (
        not isinstance(selected_rows, Tensor)
        or selected_rows.ndim != 2
        or selected_rows.shape[0] <= 0
        or selected_rows.shape[1] <= 1
        or not selected_rows.is_floating_point()
        or selected_rows.requires_grad
        or selected_rows.layout != torch.strided
        or not bool(torch.isfinite(selected_rows).all())
    ):
        raise ValueError("selected teacher rows must be finite floating [N,V]")
    rows = selected_rows.detach().contiguous().clone()
    indices = _canonical_indices(
        supervised_indices, batch=batch_size, sequence=sequence_length
    )
    if rows.shape[0] != indices.shape[0]:
        raise ValueError("selected teacher rows and indices differ")
    grid = torch.zeros(
        (batch_size, sequence_length, int(rows.shape[1])),
        dtype=rows.dtype,
        device=rows.device,
    )
    live_indices = indices.to(device=rows.device)
    grid[live_indices[:, 0], live_indices[:, 1]] = rows
    grid = grid.contiguous()
    if not torch.equal(grid[live_indices[:, 0], live_indices[:, 1]], rows):
        raise RuntimeError("selected teacher grid scatter was not exact")
    receipt = SelectedTeacherGridReceipt(
        selected_rows_sha256=_tensor_sha256(rows),
        supervised_indices_sha256=_tensor_sha256(indices),
        teacher_grid_sha256=_tensor_sha256(grid),
        teacher_grid_shape=tuple(int(width) for width in grid.shape),
        teacher_grid_dtype=str(grid.dtype),
        selected_row_count=int(rows.shape[0]),
        zero_filled_row_count=batch_size * sequence_length - int(rows.shape[0]),
    )
    validate_selected_teacher_grid_replay(grid, rows, indices, receipt)
    return grid, receipt


def validate_selected_teacher_grid_replay(
    grid: Tensor,
    selected_rows: Tensor,
    supervised_indices: Tensor,
    receipt: SelectedTeacherGridReceipt,
) -> None:
    if not isinstance(receipt, SelectedTeacherGridReceipt):
        raise TypeError("teacher grid replay needs a typed receipt")
    receipt.validate_integrity()
    if (
        not isinstance(grid, Tensor)
        or tuple(grid.shape) != receipt.teacher_grid_shape
        or str(grid.dtype) != receipt.teacher_grid_dtype
        or grid.requires_grad
        or not grid.is_contiguous()
        or _tensor_sha256(grid) != receipt.teacher_grid_sha256
    ):
        raise RuntimeError("selected teacher grid replay differs")
    indices = _canonical_indices(
        supervised_indices,
        batch=receipt.teacher_grid_shape[0],
        sequence=receipt.teacher_grid_shape[1],
    )
    rows = selected_rows.detach().contiguous()
    if (
        rows.shape[0] != indices.shape[0]
        or _tensor_sha256(rows) != receipt.selected_rows_sha256
        or _tensor_sha256(indices) != receipt.supervised_indices_sha256
    ):
        raise RuntimeError("selected teacher replay source differs")
    live = indices.to(device=grid.device)
    if not torch.equal(
        grid[live[:, 0], live[:, 1]], rows.to(device=grid.device)
    ):
        raise RuntimeError("selected teacher rows do not replay from the grid")
    occupied = torch.zeros(grid.shape[:2], dtype=torch.bool, device=grid.device)
    occupied[live[:, 0], live[:, 1]] = True
    if bool((grid[~occupied] != 0).any()):
        raise RuntimeError("unselected teacher grid rows are not canonical zero")


def _finite_h4(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 3
        or not value.is_floating_point()
        or value.requires_grad
        or value.layout != torch.strided
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating [B,S,W] tensor")
    return value.detach().contiguous()


def materialize_complete_h4_post_cast(
    reference_h4: Tensor,
    correction: Tensor,
    support_mask: Tensor,
) -> Tensor:
    """Replay the bridge's exact float64-add then live-dtype-cast H4 write."""

    reference = _finite_h4(reference_h4, label="reference H4")
    delta = _finite_h4(correction, label="complete-H4 correction")
    if delta.shape != reference.shape:
        raise ValueError("complete-H4 correction geometry differs")
    if (
        not isinstance(support_mask, Tensor)
        or support_mask.shape != reference.shape[:2]
        or support_mask.dtype != torch.bool
        or support_mask.requires_grad
    ):
        raise ValueError("complete-H4 support mask geometry differs")
    support = support_mask.detach().to(device=reference.device).contiguous()
    if bool((delta[~support] != 0.0).any()):
        raise ValueError("complete-H4 correction escapes authenticated support")
    result = reference.clone()
    if bool(support.any()):
        result[support] = (
            reference[support].to(device=delta.device, dtype=torch.float64)
            + delta[support].to(dtype=torch.float64)
        ).to(device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("post-cast complete H4 became nonfinite")
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class SoftPolarityPostCastH4SecantReceipt:
    """Hash-only receipt binding four providers, endpoints, and two secants."""

    reference_provider_sha256: str
    reference_h4_sha256: str
    center_h4_sha256: str
    bias_minus_provider_sha256: str
    bias_plus_provider_sha256: str
    slope_minus_provider_sha256: str
    slope_plus_provider_sha256: str
    bias_minus_h4_sha256: str
    bias_plus_h4_sha256: str
    slope_minus_h4_sha256: str
    slope_plus_h4_sha256: str
    bias_secant_sha256: str
    slope_secant_sha256: str
    support_mask_sha256: str
    half_step: float
    tangent_shape: tuple[int, int, int, int]
    nonzero_bias_entries: int
    nonzero_slope_entries: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "reference_provider_sha256",
            "reference_h4_sha256",
            "center_h4_sha256",
            "bias_minus_provider_sha256",
            "bias_plus_provider_sha256",
            "slope_minus_provider_sha256",
            "slope_plus_provider_sha256",
            "bias_minus_h4_sha256",
            "bias_plus_h4_sha256",
            "slope_minus_h4_sha256",
            "slope_plus_h4_sha256",
            "bias_secant_sha256",
            "slope_secant_sha256",
            "support_mask_sha256",
        ):
            object.__setattr__(self, name, _require_sha(getattr(self, name), label=name))
        step = float(self.half_step)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or step not in (
                SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
                SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
            )
            or not isinstance(self.tangent_shape, tuple)
            or len(self.tangent_shape) != 4
            or self.tangent_shape[0] != 2
            or any(type(width) is not int or width <= 0 for width in self.tangent_shape)
            or type(self.nonzero_bias_entries) is not int
            or self.nonzero_bias_entries <= 0
            or type(self.nonzero_slope_entries) is not int
            or self.nonzero_slope_entries <= 0
        ):
            raise ValueError("post-cast H4 secant receipt geometry differs")
        object.__setattr__(self, "half_step", step)
        computed = _hash(_SECANT_DOMAIN, self.metadata(include_artifact=False))
        if self.artifact_sha256:
            if _require_sha(self.artifact_sha256, label="H4 secant artifact") != computed:
                raise ValueError("post-cast H4 secant receipt hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "fisher_graph.soft_polarity_post_cast_h4_secants.v20q",
            "reference_provider_sha256": self.reference_provider_sha256,
            "reference_h4_sha256": self.reference_h4_sha256,
            "center_h4_sha256": self.center_h4_sha256,
            "bias_minus_provider_sha256": self.bias_minus_provider_sha256,
            "bias_plus_provider_sha256": self.bias_plus_provider_sha256,
            "slope_minus_provider_sha256": self.slope_minus_provider_sha256,
            "slope_plus_provider_sha256": self.slope_plus_provider_sha256,
            "bias_minus_h4_sha256": self.bias_minus_h4_sha256,
            "bias_plus_h4_sha256": self.bias_plus_h4_sha256,
            "slope_minus_h4_sha256": self.slope_minus_h4_sha256,
            "slope_plus_h4_sha256": self.slope_plus_h4_sha256,
            "perturbation_bindings": (
                (
                    "bias_minus",
                    self.bias_minus_provider_sha256,
                    self.bias_minus_h4_sha256,
                ),
                (
                    "bias_plus",
                    self.bias_plus_provider_sha256,
                    self.bias_plus_h4_sha256,
                ),
                (
                    "slope_minus",
                    self.slope_minus_provider_sha256,
                    self.slope_minus_h4_sha256,
                ),
                (
                    "slope_plus",
                    self.slope_plus_provider_sha256,
                    self.slope_plus_h4_sha256,
                ),
            ),
            "bias_secant_sha256": self.bias_secant_sha256,
            "slope_secant_sha256": self.slope_secant_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "parameter_order": ("bias", "slope"),
            "half_step": self.half_step,
            "half_step_hex": self.half_step.hex(),
            "tangent_shape": self.tangent_shape,
            "nonzero_bias_entries": self.nonzero_bias_entries,
            "nonzero_slope_entries": self.nonzero_slope_entries,
            "method": "central_finite_secant_through_exact_post_cast_h4_write",
            "not_claimed_as": "analytic_Jacobian_at_abs_or_clamp_kink",
            "raw_h4_correction_or_secant_tensors_serialized": False,
            "compiler_only_not_runtime_input": True,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _hash(_SECANT_DOMAIN, self.metadata(include_artifact=False)) != self.artifact_sha256:
            raise RuntimeError("post-cast H4 secant receipt drifted")


def build_soft_polarity_post_cast_h4_secants(
    *,
    reference_h4: Tensor,
    center_correction: Tensor,
    bias_minus_correction: Tensor,
    bias_plus_correction: Tensor,
    slope_minus_correction: Tensor,
    slope_plus_correction: Tensor,
    support_mask: Tensor,
    half_step: float,
    reference_provider_sha256: str,
    bias_minus_provider_sha256: str,
    bias_plus_provider_sha256: str,
    slope_minus_provider_sha256: str,
    slope_plus_provider_sha256: str,
) -> tuple[Tensor, Tensor, SoftPolarityPostCastH4SecantReceipt]:
    """Return center H4 and ``[bias,slope,B,S,W]`` post-cast secants."""

    step = float(half_step)
    if step not in (
        SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
        SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    ):
        raise ValueError("V20q H4 secant half-step is outside the frozen pair")
    reference = _finite_h4(reference_h4, label="reference H4")
    center = materialize_complete_h4_post_cast(
        reference, center_correction, support_mask
    )
    b_minus = materialize_complete_h4_post_cast(
        reference, bias_minus_correction, support_mask
    )
    b_plus = materialize_complete_h4_post_cast(
        reference, bias_plus_correction, support_mask
    )
    a_minus = materialize_complete_h4_post_cast(
        reference, slope_minus_correction, support_mask
    )
    a_plus = materialize_complete_h4_post_cast(
        reference, slope_plus_correction, support_mask
    )
    bias = (b_plus.to(torch.float64) - b_minus.to(torch.float64)) / (2.0 * step)
    slope = (a_plus.to(torch.float64) - a_minus.to(torch.float64)) / (2.0 * step)
    tangents = torch.stack((bias, slope), dim=0).contiguous()
    support = support_mask.detach().to(device=tangents.device)
    if (
        not bool(torch.isfinite(tangents).all())
        or bool((tangents[:, ~support] != 0.0).any())
    ):
        raise RuntimeError("post-cast H4 secants are invalid or escape support")
    nonzero_bias = int((tangents[0] != 0.0).sum())
    nonzero_slope = int((tangents[1] != 0.0).sum())
    if nonzero_bias <= 0 or nonzero_slope <= 0:
        raise RuntimeError("post-cast H4 secant direction quantized to zero")
    receipt = SoftPolarityPostCastH4SecantReceipt(
        reference_provider_sha256=reference_provider_sha256,
        reference_h4_sha256=_tensor_sha256(reference),
        center_h4_sha256=_tensor_sha256(center),
        bias_minus_provider_sha256=bias_minus_provider_sha256,
        bias_plus_provider_sha256=bias_plus_provider_sha256,
        slope_minus_provider_sha256=slope_minus_provider_sha256,
        slope_plus_provider_sha256=slope_plus_provider_sha256,
        bias_minus_h4_sha256=_tensor_sha256(b_minus),
        bias_plus_h4_sha256=_tensor_sha256(b_plus),
        slope_minus_h4_sha256=_tensor_sha256(a_minus),
        slope_plus_h4_sha256=_tensor_sha256(a_plus),
        bias_secant_sha256=_tensor_sha256(tangents[0]),
        slope_secant_sha256=_tensor_sha256(tangents[1]),
        support_mask_sha256=_tensor_sha256(support_mask.detach().to(device="cpu").contiguous()),
        half_step=step,
        tangent_shape=tuple(int(width) for width in tangents.shape),
        nonzero_bias_entries=nonzero_bias,
        nonzero_slope_entries=nonzero_slope,
    )
    return center, tangents, receipt


def soft_polarity_post_cast_h4_secant_stability(
    primary_tangents: Tensor,
    audit_tangents: Tensor,
) -> dict[str, object]:
    """Authenticate the frozen primary/audit secant agreement gate."""

    primary = _finite_h4_parameter_tangents(
        primary_tangents, label="primary post-cast H4 secants"
    )
    audit = _finite_h4_parameter_tangents(
        audit_tangents, label="audit post-cast H4 secants"
    )
    if primary.shape != audit.shape:
        raise ValueError("primary and audit H4 secant geometry differs")
    cosines: list[float] = []
    ratios: list[float] = []
    for parameter in range(2):
        left = primary[parameter].reshape(-1)
        right = audit[parameter].reshape(-1)
        left_norm = float(torch.linalg.vector_norm(left))
        right_norm = float(torch.linalg.vector_norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            raise ValueError("post-cast H4 secant stability needs nonzero directions")
        cosine = float(torch.dot(left, right) / (left_norm * right_norm))
        ratio = right_norm / left_norm
        if not math.isfinite(cosine) or not math.isfinite(ratio):
            raise RuntimeError("post-cast H4 secant stability became nonfinite")
        cosines.append(cosine)
        ratios.append(ratio)
    passed = all(value >= 0.99 for value in cosines) and all(
        0.80 <= value <= 1.25 for value in ratios
    )
    return {
        "schema": "fisher_graph.soft_polarity_post_cast_h4_secant_stability.v20q",
        "parameter_order": ("bias", "slope"),
        "primary_secants_sha256": _tensor_sha256(primary),
        "audit_secants_sha256": _tensor_sha256(audit),
        "cosine_by_parameter": tuple(cosines),
        "audit_to_primary_norm_ratio_by_parameter": tuple(ratios),
        "minimum_cosine": 0.99,
        "minimum_norm_ratio": 0.80,
        "maximum_norm_ratio": 1.25,
        "passed": passed,
        "raw_secant_tensors_serialized": False,
    }


def _finite_h4_parameter_tangents(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 4
        or value.shape[0] != 2
        or not value.is_floating_point()
        or value.requires_grad
        or value.layout != torch.strided
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite floating [2,B,S,W]")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()
