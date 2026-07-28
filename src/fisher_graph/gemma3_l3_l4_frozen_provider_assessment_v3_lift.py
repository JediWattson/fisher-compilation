"""Authenticated option-B manifold lift for the fresh V3 assessment panel.

V3 probes are standardized coordinates in the frozen top-64 L3 Fisher basis,
not valid post-RMSNorm activations.  This module applies the exact construction
used by the V2 assessment:

1. decode the requested modes only to obtain a nonnull hidden direction;
2. normalize that direction to unit RMS before normalization;
3. apply the probe's radial scale and absolute gain-null coordinate;
4. evaluate the supplied frozen, live unit-offset RMSNorm; and
5. re-encode the realized activation into Fisher coordinates.

The result keeps the requested seeds separate from the realized modes and
hash-binds every returned tensor, the diagnostics, and the complete artifact.
No model, prompt, tokenizer, or checkpoint is loaded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sys

import torch
from torch import Tensor, nn

from .adapters import module_state_fingerprint
from .gemma3_l3_l4_basis_package import Gemma3L3L4BasisPackage
from .gemma3_l3_l4_frozen_provider_assessment_v3_materialization import (
    MaterializedV3Batch,
)
from .gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    DEFAULT_V3_PANEL_SPEC_SHA256,
    DEFAULT_V3_PROTOCOL_SHA256,
    V3ProbeSpec,
    default_v3_assessment_protocol,
)
from .gemma3_l3_l4_manifold_lift import (
    ManifoldLiftDiagnostics,
    _authenticate_basis,
    _diagnostics,
    _live_tolerance,
    _maximum_or_zero,
    _validate_live_unit_offset_norm,
)


__all__ = [
    "FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION",
    "FrozenProviderAssessmentV3Lift",
    "lift_frozen_provider_assessment_v3_batch",
]


FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION = 1
_MODAL_RANK = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMULA_NAME = "option_b_unit_rms_direction_then_radial_and_null"
_TENSOR_DOMAIN = b"fisher-graph:l3-l4-frozen-provider-v3-lift-tensor:v1\0"
_DIAGNOSTICS_DOMAIN = (
    b"fisher-graph:l3-l4-frozen-provider-v3-lift-diagnostics:v1\0"
)
_ARTIFACT_DOMAIN = (
    b"fisher-graph:l3-l4-frozen-provider-v3-manifold-lift:v1\0"
)
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


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be finite")
    return float(value)


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


@dataclass(frozen=True, slots=True)
class FrozenProviderAssessmentV3Lift:
    """Hash-bound live realization of one authenticated V3 batch."""

    formula_version: int
    basis_payload_sha256: str
    source_model_sha256: str
    norm_module_sha256: str
    protocol_sha256: str
    panel_spec_sha256: str
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
        if (
            self.formula_version
            != FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION
        ):
            raise ValueError("V3 manifold lift formula version drifted")
        for name in (
            "basis_payload_sha256",
            "source_model_sha256",
            "norm_module_sha256",
            "protocol_sha256",
            "panel_spec_sha256",
            "materialized_batch_artifact_sha256",
            "materialized_batch_tensor_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.protocol_sha256 != DEFAULT_V3_PROTOCOL_SHA256
            or self.panel_spec_sha256 != DEFAULT_V3_PANEL_SPEC_SHA256
        ):
            raise ValueError("V3 manifold lift is not bound to the frozen panel")
        if (
            type(self.probe_ids) is not tuple
            or not self.probe_ids
            or len(set(self.probe_ids)) != len(self.probe_ids)
            or type(self.probe_artifact_sha256s) is not tuple
            or type(self.probe_tensor_sha256s) is not tuple
            or len(self.probe_ids) != len(self.probe_artifact_sha256s)
            or len(self.probe_ids) != len(self.probe_tensor_sha256s)
        ):
            raise ValueError("V3 manifold lift probe identity is invalid")
        for value in (
            *self.probe_artifact_sha256s,
            *self.probe_tensor_sha256s,
        ):
            _require_sha256(value, label="V3 manifold lift probe hash")
        if (
            type(self.sequence_length) is not int
            or self.sequence_length <= 0
            or type(self.residual_width) is not int
            or self.residual_width < _MODAL_RANK
        ):
            raise ValueError("V3 manifold lift geometry is invalid")
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
            raise ValueError("V3 manifold lift requires one exact gain null")
        if not isinstance(self.diagnostics, ManifoldLiftDiagnostics):
            raise TypeError("diagnostics must be ManifoldLiftDiagnostics")

        for name in _FLOAT_TENSOR_FIELDS:
            ndim = 3
            if name in {
                "neutral_hidden_state",
                "neutral_actual_x3",
            }:
                ndim = 1
            elif name == "row_rms":
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
        if (
            self.diagnostics_sha256
            and self.diagnostics_sha256 != computed_diagnostics
        ):
            raise ValueError("V3 manifold lift diagnostics hash mismatch")
        object.__setattr__(
            self,
            "diagnostics_sha256",
            computed_diagnostics,
        )
        computed_tensors = tuple(
            (name, _tensor_sha256(getattr(self, name)))
            for name in _ALL_TENSOR_FIELDS
        )
        if self.tensor_sha256s and self.tensor_sha256s != computed_tensors:
            raise ValueError("V3 manifold lift tensor hashes mismatch")
        object.__setattr__(self, "tensor_sha256s", computed_tensors)
        computed_artifact = _digest(self._payload(), domain=_ARTIFACT_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed_artifact:
            raise ValueError("V3 manifold lift artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed_artifact)

    @property
    def batch_size(self) -> int:
        return len(self.probe_ids)

    @property
    def requested_seeds(self) -> Tensor:
        """Alias emphasizing that declared coordinates are direction seeds."""

        return self.requested_standardized_modes

    def _validate_geometry_and_relations(self) -> None:
        batch = self.batch_size
        sequence = self.sequence_length
        width = self.residual_width
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
            "normalized_null_features": (batch, sequence, 1),
            "active_mask": (batch, sequence),
            "discarded_requested_x3_null": (batch, sequence, 1),
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
            raise ValueError("active mask differs from requested V3 seeds")
        inactive = ~active
        if bool(inactive.any()):
            neutral_hidden = self.neutral_hidden_state.view(1, 1, -1)
            neutral_hidden = neutral_hidden.expand_as(self.hidden_states)
            if not torch.equal(
                self.hidden_states[inactive],
                neutral_hidden[inactive],
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
            raise ValueError("actual x3 is nonzero on the exact gain null")
        if (
            self.diagnostics.active_row_count != int(active.sum())
            or self.diagnostics.inactive_row_count != int(inactive.sum())
        ):
            raise ValueError("diagnostic row counts differ from active mask")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.l3_l4_frozen_provider_v3_lift.v1",
            "formula": _FORMULA_NAME,
            "formula_version": self.formula_version,
            "basis_payload_sha256": self.basis_payload_sha256,
            "source_model_sha256": self.source_model_sha256,
            "norm_module_sha256": self.norm_module_sha256,
            "protocol_sha256": self.protocol_sha256,
            "panel_spec_sha256": self.panel_spec_sha256,
            "materialized_batch_artifact_sha256": (
                self.materialized_batch_artifact_sha256
            ),
            "materialized_batch_tensor_sha256": (
                self.materialized_batch_tensor_sha256
            ),
            "probe_ids": list(self.probe_ids),
            "probe_artifact_sha256s": list(self.probe_artifact_sha256s),
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
            raise ValueError("V3 manifold lift tensor was mutated")
        expected_diagnostics = _digest(
            self.diagnostics.state_dict(),
            domain=_DIAGNOSTICS_DOMAIN,
        )
        if expected_diagnostics != self.diagnostics_sha256:
            raise ValueError("V3 manifold lift diagnostics were mutated")
        if _digest(self._payload(), domain=_ARTIFACT_DOMAIN) != (
            self.artifact_sha256
        ):
            raise ValueError("V3 manifold lift metadata was mutated")


def _authenticate_batch_and_probes(
    batch: MaterializedV3Batch,
    probes: Sequence[V3ProbeSpec],
) -> tuple[V3ProbeSpec, ...]:
    if not isinstance(batch, MaterializedV3Batch):
        raise TypeError("batch must be a MaterializedV3Batch")
    batch.validate_integrity()
    requested = tuple(probes)
    if len(requested) != batch.batch_size:
        raise ValueError("V3 probe specifications do not match batch size")
    protocol = default_v3_assessment_protocol()
    if (
        batch.protocol_sha256 != protocol.protocol_sha256
        or batch.protocol_sha256 != DEFAULT_V3_PROTOCOL_SHA256
        or batch.panel_spec_sha256 != protocol.panel_spec_sha256
        or batch.panel_spec_sha256 != DEFAULT_V3_PANEL_SPEC_SHA256
    ):
        raise ValueError("materialized V3 batch authentication failed")
    members = {probe.artifact_sha256: probe for probe in protocol.probes}
    authenticated: list[V3ProbeSpec] = []
    for index, probe in enumerate(requested):
        if not isinstance(probe, V3ProbeSpec):
            raise TypeError("probe specifications must be V3 probes")
        member = members.get(probe.artifact_sha256)
        if member is None or member != probe:
            raise ValueError("probe is not a frozen V3 protocol member")
        if (
            probe.probe_id != batch.probe_ids[index]
            or probe.artifact_sha256
            != batch.probe_artifact_sha256s[index]
            or probe.sequence_length != batch.sequence_length
        ):
            raise ValueError("V3 probes are not in exact batch order")
        authenticated.append(member)
    radial = torch.tensor(
        [probe.radial_scale for probe in authenticated],
        dtype=torch.float64,
    )
    null = torch.tensor(
        [probe.null_coordinate for probe in authenticated],
        dtype=torch.float64,
    )
    if not torch.equal(radial, batch.radial_scales) or not torch.equal(
        null,
        batch.null_coordinates,
    ):
        raise ValueError("V3 lift scalars differ from probe declarations")
    return tuple(authenticated)


def lift_frozen_provider_assessment_v3_batch(
    basis: Gemma3L3L4BasisPackage,
    rmsnorm: nn.Module,
    *,
    epsilon: float,
    batch: MaterializedV3Batch,
    probes: Sequence[V3ProbeSpec],
) -> FrozenProviderAssessmentV3Lift:
    """Lift one authenticated equal-length V3 batch through live RMSNorm."""

    epsilon_float = _finite_float(epsilon, label="epsilon")
    if epsilon_float <= 0.0:
        raise ValueError("epsilon must be positive")
    authenticated_basis = _authenticate_basis(basis)
    if authenticated_basis.residual_width < _MODAL_RANK:
        raise ValueError("basis width is smaller than the frozen modal rank")
    _authenticate_batch_and_probes(batch, probes)
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

    requested = (
        batch.values.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    active = torch.linalg.vector_norm(requested, dim=-1) > 0.0
    if not bool(active.any()):
        raise ValueError("materialized V3 batch contains no active modal row")
    sigma = authenticated_basis.source_mode_standard_deviations(_MODAL_RANK)
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
        raise ValueError("decoded V3 Fisher direction has degenerate RMS")
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

    radial = batch.radial_scales.view(-1, 1, 1)
    null_feature = batch.null_coordinates.view(-1, 1, 1)
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
    if module_state_fingerprint(rmsnorm) != norm_sha256:
        raise RuntimeError("RMSNorm state changed during V3 manifold lift")

    hidden = (
        runtime_hidden.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    actual_x3 = (
        live_output.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    neutral_hidden = (
        runtime_neutral[0, 0]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    neutral_x3 = (
        live_neutral[0, 0]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
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
    return FrozenProviderAssessmentV3Lift(
        formula_version=(
            FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION
        ),
        basis_payload_sha256=authenticated_basis.basis_payload_sha256,
        source_model_sha256=authenticated_basis.source_model_sha256,
        norm_module_sha256=norm_sha256,
        protocol_sha256=batch.protocol_sha256,
        panel_spec_sha256=batch.panel_spec_sha256,
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
