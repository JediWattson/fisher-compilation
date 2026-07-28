"""Prompt-blind manifold lift for frozen Gemma L3 synthetic probes.

The synthetic protocol declares standardized coordinates in the frozen
top-64 L3 Fisher basis.  Those coordinates cannot be treated as exact
RMSNorm outputs: the stored mean is an aggregate that is off the rowwise
RMSNorm manifold, and Gemma's unit-offset gain may contain an exact null.

This module therefore implements the frozen option-B construction.  A
requested modal seed is decoded only to select a direction.  After division
by nonnull unit-offset gain, that direction is normalized to unit RMS in
pre-normalization hidden space.  The protocol radial multiplier is applied
to nonnull coordinates and the protocol null feature is then installed as
an absolute hidden coordinate.  The live RMSNorm module produces the actual
L3 normalized input, from which realized Fisher coordinates are recomputed.

No model, prompt, tokenizer, or artifact is loaded here.  The caller supplies
an already-authenticated basis package, live frozen RMSNorm module, and
authenticated materialized batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
import sys

import torch
from torch import Tensor, nn

from .adapters import module_state_fingerprint
from .gemma3_l3_l4_basis_package import Gemma3L3L4BasisPackage
from .gemma3_l3_l4_synthetic_materialization import (
    MaterializedSyntheticReferenceBatch,
)
from .gemma3_l3_l4_synthetic_reference_protocol import (
    DEFAULT_PROTOCOL_SHA256,
    SyntheticReferenceProbe,
    default_synthetic_reference_protocol,
)


__all__ = [
    "MANIFOLD_LIFT_FORMULA_VERSION",
    "Gemma3L3L4ManifoldLift",
    "ManifoldLiftDiagnostics",
    "lift_synthetic_reference_batch_to_gemma3_manifold",
]


MANIFOLD_LIFT_FORMULA_VERSION = 1
_MODAL_RANK = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l3-l4-manifold-tensor:v1\0"
_DIAGNOSTICS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-manifold-diagnostics:v1\0"
)
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-manifold-lift:v1\0"
_FORMULA_NAME = "option_b_unit_rms_direction_then_radial_and_null"
_FLOAT_TENSOR_FIELDS = (
    "requested_standardized_modes",
    "hidden_states",
    "actual_x3",
    "absolute_realized_standardized_modes",
    "neutral_delta_realized_standardized_modes",
    "row_rms",
    "normalized_null_features",
    "discarded_requested_x3_null",
    "neutral_hidden_state",
    "neutral_actual_x3",
)
_ALL_TENSOR_FIELDS = (*_FLOAT_TENSOR_FIELDS, "active_mask")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_float_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(size) <= 0 for size in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_bool_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{label} must be a bool Tensor")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.ndim != ndim or any(int(size) <= 0 for size in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bool:
        dtype = "bool"
    else:
        tensor = tensor.to(dtype=torch.float64)
        dtype = "float64"
    header = {
        "dtype": dtype,
        "shape": [int(size) for size in tensor.shape],
        "byte_order": "little",
    }
    array = tensor.numpy()
    if sys.byteorder == "big" and tensor.dtype == torch.float64:
        array = array.byteswap()
    digest = hashlib.sha256(
        _TENSOR_DOMAIN + _canonical_json_bytes(header) + b"\0"
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be finite")
    return float(value)


def _nearest_percentile(values: Tensor, fraction: float) -> float:
    flattened = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    if flattened.numel() == 0:
        return 0.0
    ordered = torch.sort(flattened).values
    index = max(
        0,
        min(
            ordered.numel() - 1,
            math.ceil(fraction * ordered.numel()) - 1,
        ),
    )
    return float(ordered[index])


def _maximum_or_zero(values: Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.abs().max())


@dataclass(frozen=True, slots=True)
class ManifoldLiftDiagnostics:
    """Hash-bound measurements that distinguish requests from realizations."""

    active_row_count: int
    inactive_row_count: int
    neutral_pre_projection_q: float
    active_pre_projection_q_minimum: float
    active_pre_projection_q_median: float
    active_pre_projection_q_maximum: float
    discarded_null_maximum_abs: float
    discarded_null_l2: float
    active_hidden_row_rms_minimum: float
    active_hidden_row_rms_maximum: float
    active_actual_pre_gain_q_minimum: float
    active_actual_pre_gain_q_maximum: float
    live_vs_analytic_maximum_abs: float
    live_vs_analytic_relative_l2: float
    inactive_hidden_maximum_abs_difference: float
    inactive_actual_x3_maximum_abs_difference: float
    inactive_delta_mode_maximum_abs: float
    requested_vs_realized_cosine_minimum: float
    requested_vs_realized_cosine_median: float
    requested_vs_realized_cosine_maximum: float
    requested_vs_realized_relative_error_median: float
    requested_vs_realized_relative_error_p90: float
    requested_vs_realized_relative_error_maximum: float
    absolute_realized_mode_l2_maximum: float
    delta_realized_mode_l2_maximum: float

    def __post_init__(self) -> None:
        for name in ("active_row_count", "inactive_row_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.active_row_count <= 0:
            raise ValueError("a manifold lift must contain an active row")
        for field in fields(self):
            if field.name in {"active_row_count", "inactive_row_count"}:
                continue
            value = _finite_float(getattr(self, field.name), label=field.name)
            object.__setattr__(self, field.name, value)
        for name in (
            "requested_vs_realized_cosine_minimum",
            "requested_vs_realized_cosine_median",
            "requested_vs_realized_cosine_maximum",
        ):
            if not -1.000000000001 <= getattr(self, name) <= 1.000000000001:
                raise ValueError("realized cosine lies outside [-1, 1]")
        nonnegative = (
            field.name
            for field in fields(self)
            if field.name
            not in {
                "active_row_count",
                "inactive_row_count",
                "requested_vs_realized_cosine_minimum",
                "requested_vs_realized_cosine_median",
                "requested_vs_realized_cosine_maximum",
            }
        )
        if any(getattr(self, name) < 0.0 for name in nonnegative):
            raise ValueError("nonnegative manifold diagnostic is negative")

    def state_dict(self) -> dict[str, float | int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ManifoldLift:
    """Authenticated option-B lift and its live realized L3 coordinates."""

    formula_version: int
    basis_payload_sha256: str
    source_model_sha256: str
    norm_module_sha256: str
    protocol_sha256: str
    materialized_batch_artifact_sha256: str
    materialized_batch_tensor_sha256: str
    probe_ids: tuple[str, ...]
    probe_artifact_sha256s: tuple[str, ...]
    probe_tensor_sha256s: tuple[str, ...]
    sequence_length: int
    residual_width: int
    epsilon: float
    null_gain_indices: tuple[int, ...]
    requested_standardized_modes: Tensor
    hidden_states: Tensor
    actual_x3: Tensor
    absolute_realized_standardized_modes: Tensor
    neutral_delta_realized_standardized_modes: Tensor
    row_rms: Tensor
    normalized_null_features: Tensor
    active_mask: Tensor
    discarded_requested_x3_null: Tensor
    neutral_hidden_state: Tensor
    neutral_actual_x3: Tensor
    diagnostics: ManifoldLiftDiagnostics
    diagnostics_sha256: str = ""
    tensor_sha256s: tuple[tuple[str, str], ...] = ()
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.formula_version != MANIFOLD_LIFT_FORMULA_VERSION:
            raise ValueError("manifold lift formula version drifted")
        for name in (
            "basis_payload_sha256",
            "source_model_sha256",
            "norm_module_sha256",
            "protocol_sha256",
            "materialized_batch_artifact_sha256",
            "materialized_batch_tensor_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.protocol_sha256 != DEFAULT_PROTOCOL_SHA256:
            raise ValueError("manifold lift protocol hash is not frozen")
        if (
            type(self.probe_ids) is not tuple
            or not self.probe_ids
            or len(set(self.probe_ids)) != len(self.probe_ids)
            or type(self.probe_artifact_sha256s) is not tuple
            or type(self.probe_tensor_sha256s) is not tuple
            or len(self.probe_ids) != len(self.probe_artifact_sha256s)
            or len(self.probe_ids) != len(self.probe_tensor_sha256s)
        ):
            raise ValueError("manifold lift probe identity is invalid")
        for digest in (
            *self.probe_artifact_sha256s,
            *self.probe_tensor_sha256s,
        ):
            _require_sha256(digest, label="manifold lift probe hash")
        if (
            type(self.sequence_length) is not int
            or self.sequence_length <= 0
            or type(self.residual_width) is not int
            or self.residual_width < _MODAL_RANK
        ):
            raise ValueError("manifold lift geometry is invalid")
        epsilon = _finite_float(self.epsilon, label="epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)
        if (
            type(self.null_gain_indices) is not tuple
            or len(self.null_gain_indices) != 1
            or tuple(sorted(set(self.null_gain_indices)))
            != self.null_gain_indices
            or not 0 <= self.null_gain_indices[0] < self.residual_width
        ):
            raise ValueError("manifold lift requires one exact gain null")
        if not isinstance(self.diagnostics, ManifoldLiftDiagnostics):
            raise TypeError("diagnostics must be ManifoldLiftDiagnostics")

        for name in _FLOAT_TENSOR_FIELDS:
            ndim = (
                1
                if name in {"neutral_hidden_state", "neutral_actual_x3"}
                else 3
            )
            if name == "row_rms":
                ndim = 2
            object.__setattr__(
                self,
                name,
                _canonical_float_tensor(
                    getattr(self, name),
                    label=name,
                    ndim=ndim,
                ),
            )
        object.__setattr__(
            self,
            "active_mask",
            _canonical_bool_tensor(
                self.active_mask,
                label="active_mask",
                ndim=2,
            ),
        )
        self._validate_geometry_and_relations()

        computed_diagnostics = _digest(
            self.diagnostics.state_dict(),
            domain=_DIAGNOSTICS_DOMAIN,
        )
        if self.diagnostics_sha256:
            if self.diagnostics_sha256 != computed_diagnostics:
                raise ValueError("manifold lift diagnostics hash mismatch")
        else:
            object.__setattr__(
                self,
                "diagnostics_sha256",
                computed_diagnostics,
            )

        computed_tensors = tuple(
            (name, _tensor_sha256(getattr(self, name)))
            for name in _ALL_TENSOR_FIELDS
        )
        if self.tensor_sha256s:
            if self.tensor_sha256s != computed_tensors:
                raise ValueError("manifold lift tensor hashes mismatch")
        else:
            object.__setattr__(self, "tensor_sha256s", computed_tensors)
        computed_artifact = _digest(self._payload(), domain=_ARTIFACT_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed_artifact:
                raise ValueError("manifold lift artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    @property
    def batch_size(self) -> int:
        return len(self.probe_ids)

    @property
    def requested_seeds(self) -> Tensor:
        """Alias emphasizing that declared modes are construction seeds."""

        return self.requested_standardized_modes

    def _validate_geometry_and_relations(self) -> None:
        batch = len(self.probe_ids)
        sequence = self.sequence_length
        width = self.residual_width
        null_count = len(self.null_gain_indices)
        expected = {
            "requested_standardized_modes": (
                batch,
                sequence,
                _MODAL_RANK,
            ),
            "hidden_states": (batch, sequence, width),
            "actual_x3": (batch, sequence, width),
            "absolute_realized_standardized_modes": (
                batch,
                sequence,
                _MODAL_RANK,
            ),
            "neutral_delta_realized_standardized_modes": (
                batch,
                sequence,
                _MODAL_RANK,
            ),
            "row_rms": (batch, sequence),
            "normalized_null_features": (
                batch,
                sequence,
                null_count,
            ),
            "active_mask": (batch, sequence),
            "discarded_requested_x3_null": (
                batch,
                sequence,
                null_count,
            ),
            "neutral_hidden_state": (width,),
            "neutral_actual_x3": (width,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} geometry is invalid")
        active = (
            torch.linalg.vector_norm(
                self.requested_standardized_modes,
                dim=-1,
            )
            > 0.0
        )
        if not torch.equal(active, self.active_mask):
            raise ValueError("active mask differs from requested seeds")
        inactive = ~active
        if bool(inactive.any()):
            neutral = self.neutral_hidden_state.view(1, 1, -1).expand(
                batch,
                sequence,
                -1,
            )
            if not torch.equal(
                self.hidden_states[inactive],
                neutral[inactive],
            ):
                raise ValueError("inactive rows differ from neutral hidden")
        computed_rms = self.hidden_states.square().mean(dim=-1).sqrt()
        if not torch.allclose(
            computed_rms,
            self.row_rms,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("row RMS does not match hidden states")
        null = list(self.null_gain_indices)
        denominator = (self.row_rms.square() + self.epsilon).sqrt()
        expected_null = self.hidden_states[..., null] / denominator[..., None]
        if not torch.allclose(
            expected_null,
            self.normalized_null_features,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("normalized null features are inconsistent")
        if bool((self.actual_x3[..., null] != 0.0).any()):
            raise ValueError("actual x3 is nonzero on an exact gain null")
        if (
            self.diagnostics.active_row_count != int(active.sum())
            or self.diagnostics.inactive_row_count != int(inactive.sum())
        ):
            raise ValueError("diagnostic row counts differ from active mask")

    def _payload(self) -> dict[str, object]:
        return {
            "formula": _FORMULA_NAME,
            "formula_version": self.formula_version,
            "basis_payload_sha256": self.basis_payload_sha256,
            "source_model_sha256": self.source_model_sha256,
            "norm_module_sha256": self.norm_module_sha256,
            "protocol_sha256": self.protocol_sha256,
            "materialized_batch_artifact_sha256": (
                self.materialized_batch_artifact_sha256
            ),
            "materialized_batch_tensor_sha256": (
                self.materialized_batch_tensor_sha256
            ),
            "probe_ids": list(self.probe_ids),
            "probe_artifact_sha256s": list(
                self.probe_artifact_sha256s
            ),
            "probe_tensor_sha256s": list(self.probe_tensor_sha256s),
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "residual_width": self.residual_width,
            "modal_rank": _MODAL_RANK,
            "epsilon": self.epsilon,
            "null_gain_indices": list(self.null_gain_indices),
            "inactive_row_semantics": "exact_neutral_h0",
            "null_coordinate_semantics": "absolute_pre_norm_hidden_value",
            "normalized_null_semantics": (
                "hidden_null_over_sqrt_row_mean_square_plus_epsilon"
            ),
            "diagnostics_sha256": self.diagnostics_sha256,
            "tensor_sha256s": dict(self.tensor_sha256s),
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "diagnostics": self.diagnostics.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        self._validate_geometry_and_relations()
        expected_tensors = tuple(
            (name, _tensor_sha256(getattr(self, name)))
            for name in _ALL_TENSOR_FIELDS
        )
        if expected_tensors != self.tensor_sha256s:
            raise ValueError("manifold lift tensor was mutated")
        expected_diagnostics = _digest(
            self.diagnostics.state_dict(),
            domain=_DIAGNOSTICS_DOMAIN,
        )
        if expected_diagnostics != self.diagnostics_sha256:
            raise ValueError("manifold lift diagnostics were mutated")
        if _digest(self._payload(), domain=_ARTIFACT_DOMAIN) != (
            self.artifact_sha256
        ):
            raise ValueError("manifold lift metadata was mutated")


def _authenticate_basis(
    basis: Gemma3L3L4BasisPackage,
) -> Gemma3L3L4BasisPackage:
    if not isinstance(basis, Gemma3L3L4BasisPackage):
        raise TypeError("basis must be a Gemma3L3L4BasisPackage")
    return Gemma3L3L4BasisPackage(
        basis_payload_sha256=basis.basis_payload_sha256,
        **basis.binding(),
        **basis.tensors(),
    )


def _authenticate_batch_and_probes(
    batch: MaterializedSyntheticReferenceBatch,
    probes: Sequence[SyntheticReferenceProbe],
) -> tuple[SyntheticReferenceProbe, ...]:
    if not isinstance(batch, MaterializedSyntheticReferenceBatch):
        raise TypeError("batch must be a MaterializedSyntheticReferenceBatch")
    batch.validate_integrity()
    requested = tuple(probes)
    if len(requested) != len(batch.probe_ids):
        raise ValueError("probe specifications do not match batch size")
    frozen_protocol = default_synthetic_reference_protocol()
    if (
        batch.protocol_sha256 != frozen_protocol.protocol_sha256
        or batch.protocol_sha256 != DEFAULT_PROTOCOL_SHA256
    ):
        raise ValueError("materialized batch protocol authentication failed")
    members = {
        probe.artifact_sha256: probe for probe in frozen_protocol.probes
    }
    authenticated = []
    for index, probe in enumerate(requested):
        if not isinstance(probe, SyntheticReferenceProbe):
            raise TypeError("probe specifications must be synthetic probes")
        member = members.get(probe.artifact_sha256)
        if member is None or member != probe:
            raise ValueError("probe is not a frozen protocol member")
        if (
            probe.probe_id != batch.probe_ids[index]
            or probe.artifact_sha256
            != batch.probe_artifact_sha256s[index]
            or probe.sequence_length != batch.sequence_length
        ):
            raise ValueError("probe specifications are not in exact batch order")
        authenticated.append(member)
    return tuple(authenticated)


def _validate_live_unit_offset_norm(
    module: nn.Module,
    *,
    epsilon: float,
    width: int,
) -> tuple[Tensor, tuple[int, ...], str]:
    if not isinstance(module, nn.Module):
        raise TypeError("rmsnorm must be a torch.nn.Module")
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("RMSNorm must be frozen and in evaluation mode")
    weight = getattr(module, "weight", None)
    if (
        not isinstance(weight, Tensor)
        or not weight.is_floating_point()
        or weight.ndim != 1
        or weight.numel() != width
    ):
        raise TypeError("RMSNorm must expose one floating width-sized weight")
    for name in ("variance_epsilon", "eps"):
        declared = getattr(module, name, None)
        if declared is None:
            continue
        declared_float = _finite_float(
            declared,
            label=f"RMSNorm {name}",
        )
        if declared_float != epsilon:
            raise ValueError("RMSNorm epsilon differs from requested epsilon")
    gain = 1.0 + weight.detach().to(device="cpu", dtype=torch.float64)
    tolerance = torch.finfo(torch.float64).eps * 32.0
    null = tuple(
        int(index)
        for index in torch.nonzero(
            gain.abs() <= tolerance,
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )
    if len(null) != 1:
        raise ValueError("RMSNorm must expose exactly one unit-offset gain null")
    return gain, null, module_state_fingerprint(module)


def _live_tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-3
    if dtype == torch.float32:
        return 5e-6
    return 1e-10


def _diagnostics(
    *,
    requested: Tensor,
    active: Tensor,
    pre_projection_q: Tensor,
    neutral_q: float,
    discarded_null: Tensor,
    hidden: Tensor,
    actual_x3: Tensor,
    neutral_hidden: Tensor,
    neutral_x3: Tensor,
    absolute_modes: Tensor,
    delta_modes: Tensor,
    row_rms: Tensor,
    gain: Tensor,
    null_indices: tuple[int, ...],
    analytic_x3: Tensor,
) -> ManifoldLiftDiagnostics:
    inactive = ~active
    active_q = pre_projection_q[active]
    actual_pre_gain = torch.zeros_like(actual_x3)
    nonnull = torch.ones(gain.numel(), dtype=torch.bool)
    nonnull[list(null_indices)] = False
    actual_pre_gain[..., nonnull] = (
        actual_x3[..., nonnull] / gain[nonnull]
    )
    actual_q = actual_pre_gain.square().mean(dim=-1)[active]
    requested_active = requested[active]
    delta_active = delta_modes[active]
    requested_norm = torch.linalg.vector_norm(requested_active, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta_active, dim=-1)
    denominator = (requested_norm * delta_norm).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    cosine = (requested_active * delta_active).sum(dim=-1) / denominator
    relative = (
        torch.linalg.vector_norm(
            delta_active - requested_active,
            dim=-1,
        )
        / requested_norm
    )
    live_difference = actual_x3 - analytic_x3
    live_denominator = max(
        float(torch.linalg.vector_norm(actual_x3)),
        torch.finfo(torch.float64).eps,
    )
    neutral_hidden_grid = neutral_hidden.view(1, 1, -1).expand_as(hidden)
    neutral_x3_grid = neutral_x3.view(1, 1, -1).expand_as(actual_x3)
    return ManifoldLiftDiagnostics(
        active_row_count=int(active.sum()),
        inactive_row_count=int(inactive.sum()),
        neutral_pre_projection_q=neutral_q,
        active_pre_projection_q_minimum=float(active_q.min()),
        active_pre_projection_q_median=_nearest_percentile(active_q, 0.5),
        active_pre_projection_q_maximum=float(active_q.max()),
        discarded_null_maximum_abs=_maximum_or_zero(
            discarded_null[active]
        ),
        discarded_null_l2=float(
            torch.linalg.vector_norm(discarded_null[active])
        ),
        active_hidden_row_rms_minimum=float(row_rms[active].min()),
        active_hidden_row_rms_maximum=float(row_rms[active].max()),
        active_actual_pre_gain_q_minimum=float(actual_q.min()),
        active_actual_pre_gain_q_maximum=float(actual_q.max()),
        live_vs_analytic_maximum_abs=_maximum_or_zero(live_difference),
        live_vs_analytic_relative_l2=(
            float(torch.linalg.vector_norm(live_difference))
            / live_denominator
        ),
        inactive_hidden_maximum_abs_difference=_maximum_or_zero(
            hidden[inactive] - neutral_hidden_grid[inactive]
        ),
        inactive_actual_x3_maximum_abs_difference=_maximum_or_zero(
            actual_x3[inactive] - neutral_x3_grid[inactive]
        ),
        inactive_delta_mode_maximum_abs=_maximum_or_zero(
            delta_modes[inactive]
        ),
        requested_vs_realized_cosine_minimum=float(cosine.min()),
        requested_vs_realized_cosine_median=_nearest_percentile(cosine, 0.5),
        requested_vs_realized_cosine_maximum=float(cosine.max()),
        requested_vs_realized_relative_error_median=_nearest_percentile(
            relative,
            0.5,
        ),
        requested_vs_realized_relative_error_p90=_nearest_percentile(
            relative,
            0.9,
        ),
        requested_vs_realized_relative_error_maximum=float(relative.max()),
        absolute_realized_mode_l2_maximum=float(
            torch.linalg.vector_norm(absolute_modes, dim=-1).max()
        ),
        delta_realized_mode_l2_maximum=float(delta_norm.max()),
    )


def lift_synthetic_reference_batch_to_gemma3_manifold(
    basis: Gemma3L3L4BasisPackage,
    rmsnorm: nn.Module,
    *,
    epsilon: float,
    batch: MaterializedSyntheticReferenceBatch,
    probes: Sequence[SyntheticReferenceProbe],
) -> Gemma3L3L4ManifoldLift:
    """Lift one authenticated equal-length batch into valid hidden states.

    Requested modal coordinates select directions only.  Consumers must use
    ``absolute_realized_standardized_modes`` or
    ``neutral_delta_realized_standardized_modes`` as their measured inputs.
    """

    epsilon_float = _finite_float(epsilon, label="epsilon")
    if epsilon_float <= 0.0:
        raise ValueError("epsilon must be positive")
    authenticated_basis = _authenticate_basis(basis)
    if authenticated_basis.residual_width < _MODAL_RANK:
        raise ValueError("basis width is smaller than the frozen modal rank")
    exact_probes = _authenticate_batch_and_probes(batch, probes)
    gain, null_indices, norm_sha256 = _validate_live_unit_offset_norm(
        rmsnorm,
        epsilon=epsilon_float,
        width=authenticated_basis.residual_width,
    )
    null = list(null_indices)
    nonnull = torch.ones(authenticated_basis.residual_width, dtype=torch.bool)
    nonnull[null] = False
    if bool((authenticated_basis.x3_mean[null] != 0.0).any()):
        raise ValueError("basis x3 mean is nonzero on exact gain null")

    requested = batch.values.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    requested = requested.contiguous().clone()
    active = torch.linalg.vector_norm(requested, dim=-1) > 0.0
    if not bool(active.any()):
        raise ValueError("materialized batch contains no active modal row")
    sigma = authenticated_basis.source_mode_standard_deviations(
        _MODAL_RANK
    )
    decoder = authenticated_basis.P3[:, :_MODAL_RANK]
    decoded = (requested * sigma.view(1, 1, -1)) @ decoder.T
    requested_x3 = authenticated_basis.x3_mean.view(1, 1, -1) + decoded
    discarded_null = requested_x3[..., null].clone()

    pre_norm = torch.zeros_like(requested_x3)
    pre_norm[..., nonnull] = requested_x3[..., nonnull] / gain[nonnull]
    pre_projection_q = pre_norm.square().mean(dim=-1)
    if (
        not bool(torch.isfinite(pre_projection_q).all())
        or bool((pre_projection_q <= torch.finfo(torch.float64).tiny).any())
    ):
        raise ValueError("decoded Fisher direction has degenerate RMS")
    directions = pre_norm / pre_projection_q.sqrt().unsqueeze(-1)

    neutral_pre_norm = torch.zeros_like(authenticated_basis.x3_mean)
    neutral_pre_norm[nonnull] = (
        authenticated_basis.x3_mean[nonnull] / gain[nonnull]
    )
    neutral_q = float(neutral_pre_norm.square().mean())
    if not math.isfinite(neutral_q) or neutral_q <= 0.0:
        raise ValueError("neutral Fisher mean direction has degenerate RMS")
    neutral_ideal = neutral_pre_norm / math.sqrt(neutral_q)
    neutral_ideal[null] = 0.0

    radial = torch.tensor(
        [probe.radial_scale for probe in exact_probes],
        dtype=torch.float64,
    ).view(-1, 1, 1)
    null_feature = torch.tensor(
        [probe.null_coordinate for probe in exact_probes],
        dtype=torch.float64,
    ).view(-1, 1, 1)
    active_ideal = directions * radial
    active_ideal[..., null] = null_feature
    neutral_grid = neutral_ideal.view(1, 1, -1).expand_as(active_ideal)
    hidden_ideal = torch.where(
        active.unsqueeze(-1),
        active_ideal,
        neutral_grid,
    )

    weight = getattr(rmsnorm, "weight")
    assert isinstance(weight, Tensor)
    runtime_hidden = hidden_ideal.to(
        device=weight.device,
        dtype=weight.dtype,
    )
    runtime_neutral = neutral_ideal.to(
        device=weight.device,
        dtype=weight.dtype,
    ).view(1, 1, -1)
    fingerprint_before = norm_sha256
    with torch.no_grad():
        live_output = rmsnorm(runtime_hidden)
        live_neutral = rmsnorm(runtime_neutral)
    if (
        not isinstance(live_output, Tensor)
        or not live_output.is_floating_point()
        or live_output.shape != runtime_hidden.shape
        or live_output.device != runtime_hidden.device
        or live_output.dtype != runtime_hidden.dtype
        or not isinstance(live_neutral, Tensor)
        or not live_neutral.is_floating_point()
        or live_neutral.shape != runtime_neutral.shape
        or live_neutral.device != runtime_neutral.device
        or live_neutral.dtype != runtime_neutral.dtype
    ):
        raise TypeError("RMSNorm returned an invalid live output")
    if module_state_fingerprint(rmsnorm) != fingerprint_before:
        raise RuntimeError("RMSNorm state changed during manifold lift")

    hidden = runtime_hidden.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    actual_x3 = live_output.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    neutral_hidden = runtime_neutral[0, 0].detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    neutral_x3 = live_neutral[0, 0].detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(actual_x3).all()) or not bool(
        torch.isfinite(neutral_x3).all()
    ):
        raise ValueError("RMSNorm produced nonfinite live outputs")

    denominator = (
        hidden.square().mean(dim=-1, keepdim=True) + epsilon_float
    ).sqrt()
    analytic_x3 = gain.view(1, 1, -1) * hidden / denominator
    live_difference = actual_x3 - analytic_x3
    tolerance = _live_tolerance(weight.dtype)
    live_scale = max(float(actual_x3.abs().max()), 1.0)
    if _maximum_or_zero(live_difference) > tolerance * live_scale:
        raise ValueError("live module is not the declared unit-offset RMSNorm")
    if bool((actual_x3[..., null] != 0.0).any()):
        raise ValueError("live RMSNorm is nonzero on its exact gain null")

    encoder = authenticated_basis.R3[:_MODAL_RANK]
    absolute_modes = (
        (actual_x3 - authenticated_basis.x3_mean.view(1, 1, -1))
        @ encoder.T
    ) / sigma.view(1, 1, -1)
    delta_modes = (
        (actual_x3 - neutral_x3.view(1, 1, -1)) @ encoder.T
    ) / sigma.view(1, 1, -1)
    row_rms = hidden.square().mean(dim=-1).sqrt()
    normalized_null = hidden[..., null] / denominator

    diagnostics = _diagnostics(
        requested=requested,
        active=active,
        pre_projection_q=pre_projection_q,
        neutral_q=neutral_q,
        discarded_null=discarded_null,
        hidden=hidden,
        actual_x3=actual_x3,
        neutral_hidden=neutral_hidden,
        neutral_x3=neutral_x3,
        absolute_modes=absolute_modes,
        delta_modes=delta_modes,
        row_rms=row_rms,
        gain=gain,
        null_indices=null_indices,
        analytic_x3=analytic_x3,
    )
    return Gemma3L3L4ManifoldLift(
        formula_version=MANIFOLD_LIFT_FORMULA_VERSION,
        basis_payload_sha256=authenticated_basis.basis_payload_sha256,
        source_model_sha256=authenticated_basis.source_model_sha256,
        norm_module_sha256=norm_sha256,
        protocol_sha256=batch.protocol_sha256,
        materialized_batch_artifact_sha256=batch.artifact_sha256,
        materialized_batch_tensor_sha256=batch.tensor_sha256,
        probe_ids=batch.probe_ids,
        probe_artifact_sha256s=batch.probe_artifact_sha256s,
        probe_tensor_sha256s=batch.probe_tensor_sha256s,
        sequence_length=batch.sequence_length,
        residual_width=authenticated_basis.residual_width,
        epsilon=epsilon_float,
        null_gain_indices=null_indices,
        requested_standardized_modes=requested,
        hidden_states=hidden,
        actual_x3=actual_x3,
        absolute_realized_standardized_modes=absolute_modes,
        neutral_delta_realized_standardized_modes=delta_modes,
        row_rms=row_rms,
        normalized_null_features=normalized_null,
        active_mask=active,
        discarded_requested_x3_null=discarded_null,
        neutral_hidden_state=neutral_hidden,
        neutral_actual_x3=neutral_x3,
        diagnostics=diagnostics,
    )
