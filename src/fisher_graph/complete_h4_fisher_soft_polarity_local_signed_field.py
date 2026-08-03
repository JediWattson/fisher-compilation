"""Local signed-field continuation of the V20m simplex response.

V20p replaces V20o's one global signed scalar by a bounded scalar field over
the already available two-dimensional Fisher coordinates.  For one selected
feature ``psi`` in ``{c1, c2, c1*c2, z}``::

    phi = [1, c1, c2, c1*c2]
    z = d @ phi
    s(c) = clamp(b + a*psi(c), -1, 1)
    A = (1-u*z**2)*tanh(r*z)
    B = v*z**2
    q_field = 1 + s*A + abs(s)*(B-1)
    gain = asinh(9*c2)/asinh(9) * q_field

The executable evaluates the algebraically identical, endpoint-stable form
``(1-abs(s)) + s*A + abs(s)*B``.  That reassociation makes the three constant
fields bit-identical to the V20m mirror, fixed-positive envelope, and V20m
source provider respectively.  Clamp and absolute value are fused pointwise
operations; there is no activation-dependent routing or discrete expert
selection.

This module is analysis-only.  Its receipts do not authorize serving,
compression, fidelity, or speed claims.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import _require_sha256, _sha256, _tensor_sha256
from .complete_h4_fisher_conditional_pedal import (
    fisher_xy_pointwise_bounded_direction,
)
from .complete_h4_fisher_continuous_transfer import (
    fisher_continuous_factor_direction,
    fisher_continuous_pedal_logit,
)
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    _finite_runtime_tensor,
    fisher_finite_joint_direction_features,
)
from .complete_h4_fisher_soft_polarity import fisher_soft_polarity_envelope
from .complete_h4_fisher_soft_polarity_signed_stack import (
    _canonical_evidence_tree,
    _direction,
    _require_canonical_equal,
    _require_exact_keys,
    _response_scalar_sha256,
    _strict_evidence_integer,
    _validate_evidence_sha_fields,
)
from .complete_h4_fisher_soft_polarity_simplex_response import (
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION,
    AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_response,
    fisher_soft_polarity_simplex_response_constant_tensor_sha256s,
    fisher_soft_polarity_simplex_response_direction_sha256,
    fisher_soft_polarity_simplex_response_projection,
    fisher_soft_polarity_simplex_response_provider_artifact_sha256,
    normalize_fisher_soft_polarity_simplex_response_direction,
    validate_fisher_soft_polarity_simplex_response_provider_evidence,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_ID_COUNT",
    "FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_LINEAGE_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256",
    "FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_RUNTIME_FITTED_SCALAR_COUNT",
    "AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider",
    "AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider",
    "FisherSoftPolarityLocalSignedFieldProviderEvidence",
    "FisherSoftPolarityLocalSignedFieldRuntimeProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field",
    "fisher_soft_polarity_local_signed_field_box_certificate",
    "fisher_soft_polarity_local_signed_field_calibrator",
    "fisher_soft_polarity_local_signed_field_constant_tensor_sha256s",
    "fisher_soft_polarity_local_signed_field_direction_sha256",
    "fisher_soft_polarity_local_signed_field_feature",
    "fisher_soft_polarity_local_signed_field_gain",
    "fisher_soft_polarity_local_signed_field_modal_terms",
    "fisher_soft_polarity_local_signed_field_projection",
    "fisher_soft_polarity_local_signed_field_provider_artifact_sha256",
    "fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256",
    "fisher_soft_polarity_local_signed_field_signed_scalar",
    "fisher_soft_polarity_local_signed_field_value",
    "normalize_fisher_soft_polarity_local_signed_field_direction",
    "validate_fisher_soft_polarity_local_signed_field_provider_evidence",
    "validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence",
]


_DIRECTION_COUNT = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT
_SOURCE_RESPONSE_SCALAR_COUNT = 3
_LOCAL_FIELD_SCALAR_COUNT = 2
_FEATURE_ID_COUNT = 1
_FIT_LINEAGE_SCALAR_COUNT = (
    _DIRECTION_COUNT + _SOURCE_RESPONSE_SCALAR_COUNT + _LOCAL_FIELD_SCALAR_COUNT
)
_RUNTIME_FITTED_SCALAR_COUNT = (
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT
    + _LOCAL_FIELD_SCALAR_COUNT
)
_RADIUS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX
_SHRINK_MASS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX
_TRUST_FRACTION = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION
_FEATURE_NAMES = ("c1", "c2", "c1_times_c2", "source_z")
_FEATURE_IDS = {name: index for index, name in enumerate(_FEATURE_NAMES)}
_FEATURE_ALIASES = {"c1*c2": 2, "z": 3}
_PROJECTION_TOLERANCE = 1.0e-12

FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_ID_COUNT = _FEATURE_ID_COUNT
FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_LINEAGE_SCALAR_COUNT = (
    _FIT_LINEAGE_SCALAR_COUNT
)
FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_RUNTIME_FITTED_SCALAR_COUNT = (
    _RUNTIME_FITTED_SCALAR_COUNT
)

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-local-signed-"
    b"field:provider:v20p\0"
)
_RUNTIME_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-local-signed-"
    b"field:runtime-provider:v20p\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-local-signed-"
    b"field:protocol:v20p\0"
)
_FORMULA = (
    "phi=[1,c1,c2,c1*c2];z=d_dot_phi;"
    "psi=select_one_of_[c1,c2,c1*c2,z];"
    "s(c)=clamp(b+a*psi,-1,1);"
    "A=(1-u*z_squared)*tanh(r*z);B=v*z_squared;"
    "q_field=1+s*A+abs(s)*(B-1);"
    "g_field=e(c)*q_field;e(c)=asinh(9*c2)/asinh(9)"
)
_RUNTIME_FORMULA = (
    "q_field=(1-abs(s))+s*A+abs(s)*B_endpoint_stable_reassociation;"
    "gain=e(c)*q_field"
)
_PROTOCOL = {
    "schema": "fisher_graph.complete_h4_soft_polarity_local_signed_field.v20p",
    "formula": _FORMULA,
    "runtime_formula": _RUNTIME_FORMULA,
    "feature_set": list(_FEATURE_NAMES),
    "local_field": "clamp(b+a*psi,-1,1)",
    "runtime_activation_dependent_router": False,
    "pointwise_ops": ["clamp", "absolute_value"],
    "minus_one_constant_identity": "exact_v20m_negated_direction_mirror",
    "zero_constant_identity": "exact_fixed_positive_envelope",
    "plus_one_constant_identity": "exact_v20m_source_direction",
    "boundedness": "pointwise_convex_plus_one_and_signed_v20m_response",
    "serving_authorized": False,
    "compression_claim_authorized": False,
    "fidelity_claim_authorized": False,
    "speed_claim_authorized": False,
}
FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256 = _sha256(
    _PROTOCOL_DOMAIN, _PROTOCOL
)


def _strict_scalar(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    return result


def _bounded_scalar(
    value: object, *, label: str, lower: float, upper: float
) -> float:
    result = _strict_scalar(value, label=label)
    if result < lower or result > upper:
        raise ValueError(f"{label} must be inside [{lower},{upper}]")
    return result


def _source_response(
    radius: object, shrink_mass: object, polarity_bias: object
) -> tuple[float, float, float]:
    rate = _bounded_scalar(
        radius, label="local-signed-field radius", lower=0.0, upper=_RADIUS_MAX
    )
    mass = _bounded_scalar(
        shrink_mass,
        label="local-signed-field shrink mass",
        lower=0.0,
        upper=_SHRINK_MASS_MAX,
    )
    bias = _bounded_scalar(
        polarity_bias,
        label="local-signed-field polarity bias",
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )
    if abs(bias) > mass:
        raise ValueError(
            "local-signed-field response must satisfy "
            "abs(polarity_bias) <= shrink_mass"
        )
    return rate, mass, bias


def _feature_id(value: object) -> int:
    if isinstance(value, str):
        if value in _FEATURE_IDS:
            return _FEATURE_IDS[value]
        if value in _FEATURE_ALIASES:
            return _FEATURE_ALIASES[value]
        if value not in _FEATURE_IDS:
            raise ValueError(
                "local-signed-field feature must be one of "
                + ", ".join(_FEATURE_NAMES)
            )
        raise AssertionError("unreachable local-signed-field feature")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("local-signed-field feature id must be an integer or name")
    selected = int(value)
    if selected < 0 or selected >= len(_FEATURE_NAMES):
        raise ValueError("local-signed-field feature id is outside the frozen set")
    return selected


def _feature_name(value: object) -> str:
    return _FEATURE_NAMES[_feature_id(value)]


def fisher_soft_polarity_local_signed_field_constant_tensor_sha256s() -> dict[str, str]:
    return fisher_soft_polarity_simplex_response_constant_tensor_sha256s()


def normalize_fisher_soft_polarity_local_signed_field_direction(
    direction: Tensor,
) -> Tensor:
    return normalize_fisher_soft_polarity_simplex_response_direction(direction)


def fisher_soft_polarity_local_signed_field_direction_sha256(
    direction: Tensor,
) -> str:
    return fisher_soft_polarity_simplex_response_direction_sha256(direction)


def fisher_soft_polarity_local_signed_field_projection(
    coordinates: Tensor, direction: Tensor
) -> Tensor:
    return fisher_soft_polarity_simplex_response_projection(coordinates, direction)


def fisher_soft_polarity_local_signed_field_feature(
    coordinates: Tensor,
    projection: Tensor,
    feature_id: object,
) -> Tensor:
    selected = _feature_id(feature_id)
    bounded = _finite_runtime_tensor(
        coordinates, label="local-signed-field coordinates", ndim=2
    )
    z = _finite_runtime_tensor(
        projection, label="local-signed-field projection", ndim=1
    ).to(device=bounded.device, dtype=bounded.dtype)
    if bounded.shape != (z.shape[0], 2) or bool((bounded.abs() > 1.0).any()):
        raise ValueError(
            "local-signed-field coordinates must match projection inside [-1,1]"
        )
    if bool((z.abs() > 1.0 + _PROJECTION_TOLERANCE).any()):
        raise ValueError("local-signed-field projection must remain inside [-1,1]")
    if selected == 0:
        result = bounded[:, 0]
    elif selected == 1:
        result = bounded[:, 1]
    elif selected == 2:
        result = bounded[:, 0] * bounded[:, 1]
    else:
        result = z
    if not bool(torch.isfinite(result).all()) or bool(
        (result.abs() > 1.0 + _PROJECTION_TOLERANCE).any()
    ):
        raise RuntimeError("local-signed-field feature violated its box bound")
    return result.contiguous()


def fisher_soft_polarity_local_signed_field_signed_scalar(
    feature: Tensor,
    field_bias: object,
    field_slope: object,
) -> Tensor:
    psi = _finite_runtime_tensor(
        feature, label="local-signed-field selected feature", ndim=1
    )
    if bool((psi.abs() > 1.0 + _PROJECTION_TOLERANCE).any()):
        raise ValueError("local-signed-field feature must remain inside [-1,1]")
    bias = _strict_scalar(field_bias, label="local-signed-field bias")
    slope = _strict_scalar(field_slope, label="local-signed-field slope")
    result = torch.clamp(bias + slope * psi, min=-1.0, max=1.0)
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("local-signed-field scalar violated its clamp")
    return result.contiguous()


def fisher_soft_polarity_local_signed_field_calibrator(
    projection: Tensor,
    feature: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
) -> Tensor:
    z = _finite_runtime_tensor(
        projection, label="local-signed-field projection", ndim=1
    )
    psi = _finite_runtime_tensor(
        feature, label="local-signed-field feature", ndim=1
    ).to(device=z.device, dtype=z.dtype)
    if z.shape != psi.shape:
        raise ValueError("local-signed-field projection/feature geometry differs")
    if bool((z.abs() > 1.0 + _PROJECTION_TOLERANCE).any()):
        raise ValueError("local-signed-field projection must remain inside [-1,1]")
    rate, mass, bias = _source_response(radius, shrink_mass, polarity_bias)
    signed = fisher_soft_polarity_local_signed_field_signed_scalar(
        psi, field_bias, field_slope
    )
    z_squared = z.square()
    odd = (1.0 - mass * z_squared) * torch.tanh(rate * z)
    even = bias * z_squared
    magnitude = signed.abs()
    # Exact-endpoint reassociation of 1+s*A+abs(s)*(B-1).  The symbolic
    # formula is unchanged, while s in {-1,0,+1} avoids cancellation ULPs.
    result = (1.0 - magnitude) + signed * odd + magnitude * even
    if not bool(torch.isfinite(result).all()) or bool(
        (result.abs() > 1.0 + _PROJECTION_TOLERANCE).any()
    ):
        raise RuntimeError("local-signed-field calibrator violated its bound")
    return result.contiguous()


def fisher_soft_polarity_local_signed_field_value(
    coordinates: Tensor,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
    feature_id: object,
) -> Tensor:
    projection = fisher_soft_polarity_local_signed_field_projection(
        coordinates, direction
    )
    feature = fisher_soft_polarity_local_signed_field_feature(
        coordinates, projection, feature_id
    )
    return fisher_soft_polarity_local_signed_field_calibrator(
        projection,
        feature,
        radius,
        shrink_mass,
        polarity_bias,
        field_bias,
        field_slope,
    )


def fisher_soft_polarity_local_signed_field_gain(
    coordinates: Tensor,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
    feature_id: object,
) -> Tensor:
    envelope = fisher_soft_polarity_envelope(coordinates)
    value = fisher_soft_polarity_local_signed_field_value(
        coordinates,
        direction,
        radius,
        shrink_mass,
        polarity_bias,
        field_bias,
        field_slope,
        feature_id,
    )
    result = envelope * value
    if not bool(torch.isfinite(result).all()) or bool(
        (result.abs() > 1.0 + _PROJECTION_TOLERANCE).any()
    ):
        raise RuntimeError("local-signed-field gain violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_local_signed_field_modal_terms(
    parent_modal: Tensor,
    coordinates: Tensor,
    base_direction_left: Tensor,
    base_direction_right: Tensor,
    proposal_direction_left: Tensor,
    proposal_direction_right: Tensor,
    base_pedal_weight: Tensor,
    base_pedal_bias: Tensor,
    proposal_pedal_weight: Tensor,
    proposal_pedal_bias: Tensor,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
    feature_id: object,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    parent = _finite_runtime_tensor(
        parent_modal, label="local-signed-field parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="local-signed-field coordinates", ndim=2
    ).to(parent.device)
    if (
        bounded_coordinates.shape != (parent.shape[0], 2)
        or bool((bounded_coordinates.abs() > 1.0).any())
    ):
        raise ValueError(
            "local-signed-field coordinates must match parent rows inside [-1,1]"
        )
    if trust_fraction != _TRUST_FRACTION:
        raise ValueError("local-signed-field trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_local_signed_field_gain(
        bounded_coordinates,
        direction,
        radius,
        shrink_mass,
        polarity_bias,
        field_bias,
        field_slope,
        feature_id,
    )
    features = fisher_finite_joint_direction_features(parent, bounded_coordinates)
    direction_value = fisher_continuous_factor_direction(
        features,
        base_direction_left,
        base_direction_right,
        proposal_direction_left,
        proposal_direction_right,
        gain,
    )
    bounded = fisher_xy_pointwise_bounded_direction(
        parent, direction_value, trust_fraction=trust_fraction
    )
    logit = fisher_continuous_pedal_logit(
        bounded_coordinates,
        base_pedal_weight,
        base_pedal_bias,
        proposal_pedal_weight,
        proposal_pedal_bias,
        gain,
    )
    pedal = torch.sigmoid(logit)
    delta = pedal.unsqueeze(1) * bounded
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("local-signed-field modal delta became nonfinite")
    return gain, direction_value, bounded, logit, pedal, delta.contiguous()


def _box_certificate_payload(
    *,
    direction_sha256: str,
    radius: float,
    shrink_mass: float,
    polarity_bias: float,
    field_bias: float,
    field_slope: float,
    feature_id: int,
) -> dict[str, object]:
    constant_anchor = field_slope == 0.0 and field_bias in (-1.0, 0.0, 1.0)
    return {
        "schema": "fisher_graph.fisher_soft_polarity_local_signed_field_box_certificate.v20p",
        "protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "formula": _FORMULA,
        "runtime_formula": _RUNTIME_FORMULA,
        "direction_sha256": direction_sha256,
        "radius": radius,
        "shrink_mass": shrink_mass,
        "polarity_bias": polarity_bias,
        "field_bias": field_bias,
        "field_slope": field_slope,
        "feature_id": feature_id,
        "feature_name": _FEATURE_NAMES[feature_id],
        "coordinate_box": [-1.0, 1.0],
        "projection_max_abs": 1.0,
        "feature_max_abs": 1.0,
        "signed_field_closed_interval": [-1.0, 1.0],
        "runtime_activation_dependent_router": False,
        "pointwise_clamp_and_abs_only": True,
        "constant_anchor": constant_anchor,
        "minus_one_exact_v20m_mirror": constant_anchor and field_bias == -1.0,
        "zero_exact_fixed_plus": constant_anchor and field_bias == 0.0,
        "plus_one_exact_v20m_source": constant_anchor and field_bias == 1.0,
        "response_max_abs": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "at_each_coordinate_abs_s_is_in_zero_one_and_q_field_is_the_"
            "convex_combination_of_plus_one_and_q_of_sign_s_times_z_"
            "whose_v20m_simplex_response_is_bounded_by_one"
        ),
        "serving_status": "analysis_only_no_serving_compression_fidelity_or_speed_claim",
    }


_BOX_CERTIFICATE_KEYS = frozenset(
    _box_certificate_payload(
        direction_sha256="0" * 64,
        radius=0.0,
        shrink_mass=0.0,
        polarity_bias=0.0,
        field_bias=0.0,
        field_slope=0.0,
        feature_id=0,
    )
)


def fisher_soft_polarity_local_signed_field_box_certificate(
    direction: Tensor,
    *,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
    feature_id: object,
) -> dict[str, object]:
    # The V20m direction is already normalized.  Do not move it by another ULP.
    selected = _direction(
        direction,
        detach=True,
        normalize=False,
        label="local-signed-field certificate direction",
    )
    rate, mass, polarity = _source_response(radius, shrink_mass, polarity_bias)
    bias = _strict_scalar(field_bias, label="local-signed-field bias")
    slope = _strict_scalar(field_slope, label="local-signed-field slope")
    selected_feature = _feature_id(feature_id)
    return _box_certificate_payload(
        direction_sha256=fisher_soft_polarity_local_signed_field_direction_sha256(
            selected
        ),
        radius=rate,
        shrink_mass=mass,
        polarity_bias=polarity,
        field_bias=bias,
        field_slope=slope,
        feature_id=selected_feature,
    )


_RUNTIME_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "site",
        "write_scope",
        "protocol_sha256",
        "bridge_binding_sha256",
        "parent_provider_artifact_sha256",
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "start_provider_artifact_sha256",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "direction_sha256",
        "radius",
        "shrink_mass",
        "polarity_bias",
        "field_bias",
        "field_bias_hex",
        "field_bias_sha256",
        "field_slope",
        "field_slope_hex",
        "field_slope_sha256",
        "feature_id",
        "feature_name",
        "feature_id_uint8_count",
        "source_simplex_response_provider_artifact_sha256",
        "source_simplex_response_provider_payload",
        "runtime_fitted_float_scalar_count",
        "runtime_formula",
        "runtime_inputs",
        "runtime_forbidden_inputs",
        "routing_control_flow",
        "boundedness_certificate",
        "serving_status",
    }
)
_RUNTIME_METADATA_EXTRA_KEYS = frozenset(
    {
        "artifact_sha256",
        "box_certificate",
        "source_simplex_response_provider_metadata",
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_runtime_parameter_bytes_uint8",
        "runtime_parameter_bytes_uint8",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "local_signed_field_projection_dot_macs_per_token",
        "local_signed_field_feature_macs_per_token_upper_bound",
        "local_signed_field_affine_macs_per_token",
        "local_signed_field_elementwise_scalar_arithmetic_per_token",
        "local_signed_field_nonlinear_scalar_ops_per_token",
        "runtime_state_float_scalars_per_sequence",
        "logical_macs_accounting_scope",
        "pointwise_trust_certificate_scope",
    }
)


def _validate_runtime_payload(value: object) -> dict[str, object]:
    payload = _require_exact_keys(
        value, _RUNTIME_PAYLOAD_KEYS, label="local-signed-field runtime payload"
    )
    _canonical_evidence_tree(payload, label="local-signed-field runtime payload")
    _validate_evidence_sha_fields(
        payload, label="local-signed-field runtime payload"
    )
    frozen = {
        "schema": (
            "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
            "local_signed_field_runtime_provider.v20p"
        ),
        "site": _H4_SITE,
        "write_scope": "complete_h4_causal_support",
        "protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "feature_id_uint8_count": _FEATURE_ID_COUNT,
        "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
        "runtime_formula": _RUNTIME_FORMULA,
        "routing_control_flow": "none_pointwise_clamp_and_abs_validation_guards_only",
        "boundedness_certificate": "pointwise_convex_plus_one_and_signed_v20m_response",
        "serving_status": "analysis_only_no_serving_compression_fidelity_or_speed_claim",
    }
    for key, expected in frozen.items():
        _require_canonical_equal(
            payload[key], expected, label=f"local-signed-field runtime {key}"
        )
    rate, mass, polarity = _source_response(
        payload["radius"], payload["shrink_mass"], payload["polarity_bias"]
    )
    # Builder aliases and real-number conveniences end at serialization.  A
    # persisted receipt has one canonical JSON/Python representation so an
    # attacker cannot rehash an alias or integer-valued coefficient into a
    # second artifact for the same executable field.
    if type(payload["field_bias"]) is not float:
        raise ValueError("local-signed-field runtime bias must be a canonical float")
    if type(payload["field_slope"]) is not float:
        raise ValueError("local-signed-field runtime slope must be a canonical float")
    if type(payload["feature_id"]) is not int:
        raise ValueError(
            "local-signed-field runtime feature id must be a canonical integer"
        )
    field_bias = _strict_scalar(
        payload["field_bias"], label="local-signed-field runtime bias"
    )
    field_slope = _strict_scalar(
        payload["field_slope"], label="local-signed-field runtime slope"
    )
    feature = int(payload["feature_id"])
    if feature < 0 or feature >= len(_FEATURE_NAMES):
        raise ValueError("local-signed-field runtime feature id is outside [0,3]")
    if payload["feature_name"] != _FEATURE_NAMES[feature]:
        raise ValueError("local-signed-field runtime feature name differs")
    for prefix, scalar in (("field_bias", field_bias), ("field_slope", field_slope)):
        if payload[f"{prefix}_hex"] != scalar.hex():
            raise ValueError(f"local-signed-field runtime {prefix} hex differs")
        if payload[f"{prefix}_sha256"] != _response_scalar_sha256(scalar):
            raise ValueError(f"local-signed-field runtime {prefix} hash differs")
    nested_payload = payload["source_simplex_response_provider_payload"]
    nested_artifact = fisher_soft_polarity_simplex_response_provider_artifact_sha256(
        nested_payload
    )
    if payload["source_simplex_response_provider_artifact_sha256"] != nested_artifact:
        raise ValueError("local-signed-field source V20m artifact differs")
    nested = _canonical_evidence_tree(
        nested_payload, label="local-signed-field source V20m payload"
    )
    assert isinstance(nested, dict)
    bindings = {
        "site": nested["site"],
        "write_scope": nested["write_scope"],
        "bridge_binding_sha256": nested["bridge_binding_sha256"],
        "parent_provider_artifact_sha256": nested[
            "parent_provider_artifact_sha256"
        ],
        "base_provider_artifact_sha256": nested["base_provider_artifact_sha256"],
        "proposal_provider_artifact_sha256": nested[
            "proposal_provider_artifact_sha256"
        ],
        "start_provider_artifact_sha256": nested[
            "start_provider_artifact_sha256"
        ],
        "transfer_protocol_sha256": nested["transfer_protocol_sha256"],
        "transfer_evidence_sha256": nested["transfer_evidence_sha256"],
        "direction_sha256": nested["direction_sha256"],
        "radius": nested["radius"],
        "shrink_mass": nested["shrink_mass"],
        "polarity_bias": nested["polarity_bias"],
        "runtime_inputs": nested["runtime_inputs"],
        "runtime_forbidden_inputs": nested["runtime_forbidden_inputs"],
    }
    if (rate, mass, polarity) != (
        float(nested["radius"]),
        float(nested["shrink_mass"]),
        float(nested["polarity_bias"]),
    ):
        raise ValueError("local-signed-field response differs from source V20m")
    for key, expected in bindings.items():
        _require_canonical_equal(
            payload[key], expected, label=f"local-signed-field runtime binding {key}"
        )
    canonical = _canonical_evidence_tree(
        payload, label="local-signed-field runtime payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_RUNTIME_PROVIDER_DOMAIN, _validate_runtime_payload(payload))


@dataclass(frozen=True, slots=True)
class FisherSoftPolarityLocalSignedFieldRuntimeProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence(
    payload: object, metadata: object
) -> FisherSoftPolarityLocalSignedFieldRuntimeProviderEvidence:
    canonical_payload = _validate_runtime_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _RUNTIME_PAYLOAD_KEYS | _RUNTIME_METADATA_EXTRA_KEYS,
        label="local-signed-field runtime metadata",
    )
    _canonical_evidence_tree(selected, label="local-signed-field runtime metadata")
    _validate_evidence_sha_fields(
        selected, label="local-signed-field runtime metadata"
    )
    for key in _RUNTIME_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"local-signed-field runtime metadata {key}",
        )
    artifact = fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact:
        raise ValueError("local-signed-field runtime artifact hash differs")
    nested = validate_fisher_soft_polarity_simplex_response_provider_evidence(
        canonical_payload["source_simplex_response_provider_payload"],
        selected["source_simplex_response_provider_metadata"],
    )
    if nested.artifact_sha256 != canonical_payload[
        "source_simplex_response_provider_artifact_sha256"
    ]:
        raise ValueError("local-signed-field nested V20m evidence differs")
    for key in ("rank", "conditional_rank", "runtime_state_float_scalars_per_sequence"):
        if _strict_evidence_integer(selected, key) != _strict_evidence_integer(
            nested.metadata, key
        ):
            raise ValueError(f"local-signed-field delegated {key} differs")
    feature_id = _feature_id(canonical_payload["feature_id"])
    feature_macs = 1 if feature_id == 2 else 0
    expected_integers = {
        "incremental_prepared_float_scalar_count": int(
            nested.metadata["incremental_prepared_float_scalar_count"]
        )
        + _LOCAL_FIELD_SCALAR_COUNT,
        "prepared_float_scalar_count": int(nested.metadata["prepared_float_scalar_count"])
        + _LOCAL_FIELD_SCALAR_COUNT,
        "incremental_runtime_parameter_bytes_float64": int(
            nested.metadata["incremental_runtime_parameter_bytes_float64"]
        )
        + 16,
        "runtime_parameter_bytes_float64": int(
            nested.metadata["runtime_parameter_bytes_float64"]
        )
        + 16,
        "incremental_runtime_parameter_bytes_uint8": 1,
        "runtime_parameter_bytes_uint8": 1,
        "incremental_logical_macs_per_token_upper_bound": int(
            nested.metadata["incremental_logical_macs_per_token_upper_bound"]
        )
        + feature_macs
        + 1,
        "logical_macs_per_token_upper_bound": int(
            nested.metadata["logical_macs_per_token_upper_bound"]
        )
        + feature_macs
        + 1,
        "local_signed_field_projection_dot_macs_per_token": _DIRECTION_COUNT,
        "local_signed_field_feature_macs_per_token_upper_bound": feature_macs,
        "local_signed_field_affine_macs_per_token": 1,
        "local_signed_field_elementwise_scalar_arithmetic_per_token": 14,
        "local_signed_field_nonlinear_scalar_ops_per_token": 4,
    }
    for key, expected in expected_integers.items():
        if _strict_evidence_integer(selected, key) != expected:
            raise ValueError(f"local-signed-field runtime accounting {key} differs")
    for key in ("logical_macs_accounting_scope", "pointwise_trust_certificate_scope"):
        _require_canonical_equal(
            selected[key], nested.metadata[key], label=f"local-signed-field {key}"
        )
    certificate = _require_exact_keys(
        selected["box_certificate"],
        _BOX_CERTIFICATE_KEYS,
        label="local-signed-field runtime box certificate",
    )
    expected_certificate = _box_certificate_payload(
        direction_sha256=str(canonical_payload["direction_sha256"]),
        radius=float(canonical_payload["radius"]),
        shrink_mass=float(canonical_payload["shrink_mass"]),
        polarity_bias=float(canonical_payload["polarity_bias"]),
        field_bias=float(canonical_payload["field_bias"]),
        field_slope=float(canonical_payload["field_slope"]),
        feature_id=feature_id,
    )
    for key, expected in expected_certificate.items():
        _require_canonical_equal(
            certificate[key], expected, label=f"local-signed-field certificate {key}"
        )
    canonical_metadata = _canonical_evidence_tree(
        selected, label="local-signed-field runtime metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolarityLocalSignedFieldRuntimeProviderEvidence(
        canonical_payload, canonical_metadata, artifact
    )


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider(
    Gemma3L3L4CorrectionProvider
):
    """The V20p executable: one fused local field over a V20m lineage."""

    source_simplex_provider: AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
    field_bias: float
    field_slope: float
    feature_id: int
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_simplex_provider,
            AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
        ):
            raise TypeError("local-signed-field runtime needs one source V20m provider")
        object.__setattr__(
            self,
            "field_bias",
            _strict_scalar(self.field_bias, label="local-signed-field bias"),
        )
        object.__setattr__(
            self,
            "field_slope",
            _strict_scalar(self.field_slope, label="local-signed-field slope"),
        )
        object.__setattr__(self, "feature_id", _feature_id(self.feature_id))
        self.source_simplex_provider.validate_integrity()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="local-signed-field runtime provider"
            ) != computed:
                raise ValueError("local-signed-field runtime artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def base_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.source_simplex_provider.base_provider

    @property
    def proposal_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.source_simplex_provider.proposal_provider

    @property
    def parent_provider(self):
        return self.source_simplex_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.source_simplex_provider.bridge_binding_sha256

    @property
    def transfer_protocol_sha256(self) -> str:
        return self.source_simplex_provider.transfer_protocol_sha256

    @property
    def transfer_evidence_sha256(self) -> str:
        return self.source_simplex_provider.transfer_evidence_sha256

    @property
    def direction(self) -> Tensor:
        return self.source_simplex_provider.direction

    @property
    def radius(self) -> float:
        return self.source_simplex_provider.radius

    @property
    def shrink_mass(self) -> float:
        return self.source_simplex_provider.shrink_mass

    @property
    def polarity_bias(self) -> float:
        return self.source_simplex_provider.polarity_bias

    @property
    def feature_name(self) -> str:
        return _FEATURE_NAMES[self.feature_id]

    @property
    def trust_fraction(self) -> float:
        return self.source_simplex_provider.trust_fraction

    @property
    def rank(self) -> int:
        return self.source_simplex_provider.rank

    @property
    def conditional_rank(self) -> int:
        return self.source_simplex_provider.conditional_rank

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return (
            self.source_simplex_provider.incremental_prepared_float_scalar_count
            + _LOCAL_FIELD_SCALAR_COUNT
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.source_simplex_provider.prepared_float_scalar_count
            + _LOCAL_FIELD_SCALAR_COUNT
        )

    @property
    def local_feature_macs_per_token(self) -> int:
        return 1 if self.feature_id == 2 else 0

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.source_simplex_provider.incremental_logical_macs_per_token_upper_bound
            + self.local_feature_macs_per_token
            + 1
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.source_simplex_provider.logical_macs_per_token_upper_bound
            + self.local_feature_macs_per_token
            + 1
        )

    def _payload(self) -> dict[str, object]:
        nested = self.source_simplex_provider.artifact_payload()
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "local_signed_field_runtime_provider.v20p"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": self.proposal_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.base_provider.start_provider_artifact_sha256,
            "transfer_protocol_sha256": self.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.transfer_evidence_sha256,
            "direction_sha256": fisher_soft_polarity_local_signed_field_direction_sha256(
                self.direction
            ),
            "radius": self.radius,
            "shrink_mass": self.shrink_mass,
            "polarity_bias": self.polarity_bias,
            "field_bias": self.field_bias,
            "field_bias_hex": self.field_bias.hex(),
            "field_bias_sha256": _response_scalar_sha256(self.field_bias),
            "field_slope": self.field_slope,
            "field_slope_hex": self.field_slope.hex(),
            "field_slope_sha256": _response_scalar_sha256(self.field_slope),
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "feature_id_uint8_count": _FEATURE_ID_COUNT,
            "source_simplex_response_provider_artifact_sha256": (
                self.source_simplex_provider.artifact_sha256
            ),
            "source_simplex_response_provider_payload": nested,
            "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
            "runtime_formula": _RUNTIME_FORMULA,
            "runtime_inputs": nested["runtime_inputs"],
            "runtime_forbidden_inputs": nested["runtime_forbidden_inputs"],
            "routing_control_flow": (
                "none_pointwise_clamp_and_abs_validation_guards_only"
            ),
            "boundedness_certificate": (
                "pointwise_convex_plus_one_and_signed_v20m_response"
            ),
            "serving_status": (
                "analysis_only_no_serving_compression_fidelity_or_speed_claim"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_RUNTIME_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        self.source_simplex_provider.validate_integrity()
        _strict_scalar(self.field_bias, label="local-signed-field stored bias")
        _strict_scalar(self.field_slope, label="local-signed-field stored slope")
        _feature_id(self.feature_id)
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("local-signed-field runtime payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        nested = self.source_simplex_provider.metadata()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "box_certificate": _box_certificate_payload(
                direction_sha256=str(self._payload()["direction_sha256"]),
                radius=self.radius,
                shrink_mass=self.shrink_mass,
                polarity_bias=self.polarity_bias,
                field_bias=self.field_bias,
                field_slope=self.field_slope,
                feature_id=self.feature_id,
            ),
            "source_simplex_response_provider_metadata": nested,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": self.incremental_prepared_float_scalar_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": (
                self.incremental_prepared_float_scalar_count * 8
            ),
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "incremental_runtime_parameter_bytes_uint8": 1,
            "runtime_parameter_bytes_uint8": 1,
            "incremental_logical_macs_per_token_upper_bound": (
                self.incremental_logical_macs_per_token_upper_bound
            ),
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "local_signed_field_projection_dot_macs_per_token": _DIRECTION_COUNT,
            "local_signed_field_feature_macs_per_token_upper_bound": (
                self.local_feature_macs_per_token
            ),
            "local_signed_field_affine_macs_per_token": 1,
            "local_signed_field_elementwise_scalar_arithmetic_per_token": 14,
            "local_signed_field_nonlinear_scalar_ops_per_token": 4,
            "runtime_state_float_scalars_per_sequence": nested[
                "runtime_state_float_scalars_per_sequence"
            ],
            "logical_macs_accounting_scope": nested["logical_macs_accounting_scope"],
            "pointwise_trust_certificate_scope": nested[
                "pointwise_trust_certificate_scope"
            ],
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.source_simplex_provider.bounded_coordinates(parent_modal)

    def response_gain(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(coordinates, Tensor)
            or coordinates.ndim < 2
            or coordinates.shape[-1] != 2
            or not coordinates.is_floating_point()
            or not bool(torch.isfinite(coordinates).all())
            or bool((coordinates.abs() > 1.0).any())
        ):
            raise ValueError("local-signed-field coordinates differ")
        leading = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_local_signed_field_gain(
            flat,
            self.direction.to(device=flat.device, dtype=flat.dtype),
            self.radius,
            self.shrink_mass,
            self.polarity_bias,
            self.field_bias,
            self.field_slope,
            self.feature_id,
        )
        return gain.reshape(leading).contiguous()

    def terms_from_parent(
        self, parent_modal: Tensor, coordinates: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("local-signed-field parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("local-signed-field parent/coordinate geometry differs")
        original_shape = parent.shape
        values = fisher_soft_polarity_local_signed_field_modal_terms(
            parent.reshape(-1, self.rank),
            bounded.reshape(-1, 2),
            self.base_provider.direction_left.to(parent.device),
            self.base_provider.direction_right.to(parent.device),
            self.proposal_provider.direction_left.to(parent.device),
            self.proposal_provider.direction_right.to(parent.device),
            self.base_provider.pedal_weight.to(parent.device),
            self.base_provider.pedal_bias.to(parent.device),
            self.proposal_provider.pedal_weight.to(parent.device),
            self.proposal_provider.pedal_bias.to(parent.device),
            self.direction.to(parent.device),
            self.radius,
            self.shrink_mass,
            self.polarity_bias,
            self.field_bias,
            self.field_slope,
            self.feature_id,
            trust_fraction=self.trust_fraction,
        )
        leading = original_shape[:-1]
        return (
            values[0].reshape(leading).contiguous(),
            values[1].reshape(original_shape).contiguous(),
            values[2].reshape(original_shape).contiguous(),
            values[3].reshape(leading).contiguous(),
            values[4].reshape(leading).contiguous(),
            values[5].reshape(original_shape).contiguous(),
        )

    def unbounded_direction(
        self, parent_modal: Tensor, coordinates: Tensor | None = None
    ) -> Tensor:
        return self.terms_from_parent(parent_modal, coordinates)[1]

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        gain = self.response_gain(coordinates)
        leading = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2).to(dtype=torch.float64)
        result = fisher_continuous_pedal_logit(
            flat,
            self.base_provider.pedal_weight.to(flat.device),
            self.base_provider.pedal_bias.to(flat.device),
            self.proposal_provider.pedal_weight.to(flat.device),
            self.proposal_provider.pedal_bias.to(flat.device),
            gain.reshape(-1),
        )
        return result.reshape(leading).contiguous()

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return torch.sigmoid(self.pedal_logits(coordinates)).contiguous()

    def _modal_terms(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        parent = self.parent_provider.modal_correction(prefix, realized_state)
        coordinates = self.bounded_coordinates(parent)
        gain, direction, bounded, logit, pedal, delta = self.terms_from_parent(
            parent, coordinates
        )
        return parent, coordinates, gain, direction, bounded, logit, pedal, delta

    def modal_correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        prefix_sha = prefix.artifact_sha256
        realized_sha = _tensor_sha256(realized_state)
        parent, _coordinates, _gain, _direction, _bounded, _logit, _pedal, delta = (
            self._modal_terms(prefix, realized_state)
        )
        modal = parent + delta
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
        if (
            prefix.artifact_sha256 != prefix_sha
            or _tensor_sha256(realized_state) != realized_sha
        ):
            raise RuntimeError("local-signed-field provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("local-signed-field modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("local-signed-field modal correction escaped support")
        self.validate_integrity()
        prefix.validate_integrity()
        return modal.contiguous()

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        return self.parent_provider.decode_modal(
            prefix, self.modal_correction(prefix, realized_state), like=realized_state
        )


_PROVIDER_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "site",
        "write_scope",
        "protocol_sha256",
        "compiled_runtime_provider_artifact_sha256",
        "compiled_runtime_provider_payload",
        "fit_lineage_float_scalar_count",
        "runtime_fitted_float_scalar_count",
        "feature_id_uint8_count",
        "gain_formula",
        "runtime_inputs",
        "runtime_forbidden_inputs",
        "routing_control_flow",
        "boundedness_certificate",
        "serving_status",
    }
)
_PROVIDER_METADATA_EXTRA_KEYS = frozenset(
    {
        "artifact_sha256",
        "box_certificate",
        "compiled_runtime_provider_metadata",
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_runtime_parameter_bytes_uint8",
        "runtime_parameter_bytes_uint8",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_scalars_per_sequence",
        "logical_macs_accounting_scope",
        "pointwise_trust_certificate_scope",
    }
)


def _validate_provider_payload(value: object) -> dict[str, object]:
    payload = _require_exact_keys(
        value, _PROVIDER_PAYLOAD_KEYS, label="local-signed-field provider payload"
    )
    _canonical_evidence_tree(payload, label="local-signed-field provider payload")
    _validate_evidence_sha_fields(payload, label="local-signed-field provider payload")
    frozen = {
        "schema": (
            "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
            "local_signed_field_provider.v20p"
        ),
        "site": _H4_SITE,
        "write_scope": "complete_h4_causal_support",
        "protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
        "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
        "feature_id_uint8_count": _FEATURE_ID_COUNT,
        "gain_formula": _FORMULA,
        "routing_control_flow": "none_pointwise_clamp_and_abs_validation_guards_only",
        "boundedness_certificate": "pointwise_convex_plus_one_and_signed_v20m_response",
        "serving_status": "analysis_only_no_serving_compression_fidelity_or_speed_claim",
    }
    for key, expected in frozen.items():
        _require_canonical_equal(
            payload[key], expected, label=f"local-signed-field provider {key}"
        )
    runtime_payload = payload["compiled_runtime_provider_payload"]
    runtime_artifact = (
        fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256(
            runtime_payload
        )
    )
    if payload["compiled_runtime_provider_artifact_sha256"] != runtime_artifact:
        raise ValueError("local-signed-field compiled runtime artifact differs")
    runtime = _canonical_evidence_tree(
        runtime_payload, label="local-signed-field compiled runtime payload"
    )
    assert isinstance(runtime, dict)
    for key in ("site", "write_scope", "runtime_inputs", "runtime_forbidden_inputs"):
        _require_canonical_equal(
            payload[key], runtime[key], label=f"local-signed-field runtime binding {key}"
        )
    canonical = _canonical_evidence_tree(
        payload, label="local-signed-field provider payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_local_signed_field_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_PROVIDER_DOMAIN, _validate_provider_payload(payload))


@dataclass(frozen=True, slots=True)
class FisherSoftPolarityLocalSignedFieldProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def validate_fisher_soft_polarity_local_signed_field_provider_evidence(
    payload: object, metadata: object
) -> FisherSoftPolarityLocalSignedFieldProviderEvidence:
    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="local-signed-field provider metadata",
    )
    _canonical_evidence_tree(selected, label="local-signed-field provider metadata")
    _validate_evidence_sha_fields(
        selected, label="local-signed-field provider metadata"
    )
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"local-signed-field provider metadata {key}",
        )
    artifact = fisher_soft_polarity_local_signed_field_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact:
        raise ValueError("local-signed-field provider artifact hash differs")
    runtime = validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence(
        canonical_payload["compiled_runtime_provider_payload"],
        selected["compiled_runtime_provider_metadata"],
    )
    if runtime.artifact_sha256 != canonical_payload[
        "compiled_runtime_provider_artifact_sha256"
    ]:
        raise ValueError("local-signed-field nested runtime evidence differs")
    for key in (
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_runtime_parameter_bytes_uint8",
        "runtime_parameter_bytes_uint8",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_scalars_per_sequence",
    ):
        if _strict_evidence_integer(selected, key) != _strict_evidence_integer(
            runtime.metadata, key
        ):
            raise ValueError(f"local-signed-field delegated accounting {key} differs")
    for key in ("logical_macs_accounting_scope", "pointwise_trust_certificate_scope"):
        _require_canonical_equal(
            selected[key], runtime.metadata[key], label=f"local-signed-field {key}"
        )
    certificate = _require_exact_keys(
        selected["box_certificate"],
        _BOX_CERTIFICATE_KEYS,
        label="local-signed-field provider box certificate",
    )
    for key, expected in runtime.metadata["box_certificate"].items():
        _require_canonical_equal(
            certificate[key], expected, label=f"local-signed-field certificate {key}"
        )
    canonical_metadata = _canonical_evidence_tree(
        selected, label="local-signed-field provider metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolarityLocalSignedFieldProviderEvidence(
        canonical_payload, canonical_metadata, artifact
    )


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider(
    Gemma3L3L4CorrectionProvider
):
    """Authenticated fit-lineage wrapper around the V20p executable."""

    compiled_provider: AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.compiled_provider,
            AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider,
        ):
            raise TypeError("local-signed-field wrapper needs one compiled runtime")
        self.compiled_provider.validate_integrity()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="local-signed-field provider"
            ) != computed:
                raise ValueError("local-signed-field provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def runtime_provider(
        self,
    ) -> AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider:
        return self.compiled_provider

    @property
    def base_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.compiled_provider.base_provider

    @property
    def proposal_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.compiled_provider.proposal_provider

    @property
    def parent_provider(self):
        return self.compiled_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.compiled_provider.bridge_binding_sha256

    @property
    def direction(self) -> Tensor:
        return self.compiled_provider.direction

    @property
    def radius(self) -> float:
        return self.compiled_provider.radius

    @property
    def shrink_mass(self) -> float:
        return self.compiled_provider.shrink_mass

    @property
    def polarity_bias(self) -> float:
        return self.compiled_provider.polarity_bias

    @property
    def field_bias(self) -> float:
        return self.compiled_provider.field_bias

    @property
    def field_slope(self) -> float:
        return self.compiled_provider.field_slope

    @property
    def feature_id(self) -> int:
        return self.compiled_provider.feature_id

    @property
    def feature_name(self) -> str:
        return self.compiled_provider.feature_name

    @property
    def rank(self) -> int:
        return self.compiled_provider.rank

    @property
    def conditional_rank(self) -> int:
        return self.compiled_provider.conditional_rank

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return self.compiled_provider.incremental_prepared_float_scalar_count

    @property
    def prepared_float_scalar_count(self) -> int:
        return self.compiled_provider.prepared_float_scalar_count

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        return self.compiled_provider.incremental_logical_macs_per_token_upper_bound

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return self.compiled_provider.logical_macs_per_token_upper_bound

    def _payload(self) -> dict[str, object]:
        runtime = self.compiled_provider.artifact_payload()
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "local_signed_field_provider.v20p"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
            "compiled_runtime_provider_artifact_sha256": (
                self.compiled_provider.artifact_sha256
            ),
            "compiled_runtime_provider_payload": runtime,
            "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
            "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
            "feature_id_uint8_count": _FEATURE_ID_COUNT,
            "gain_formula": _FORMULA,
            "runtime_inputs": runtime["runtime_inputs"],
            "runtime_forbidden_inputs": runtime["runtime_forbidden_inputs"],
            "routing_control_flow": (
                "none_pointwise_clamp_and_abs_validation_guards_only"
            ),
            "boundedness_certificate": (
                "pointwise_convex_plus_one_and_signed_v20m_response"
            ),
            "serving_status": (
                "analysis_only_no_serving_compression_fidelity_or_speed_claim"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        self.compiled_provider.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("local-signed-field provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        runtime = self.compiled_provider.metadata()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "box_certificate": runtime["box_certificate"],
            "compiled_runtime_provider_metadata": runtime,
            "rank": runtime["rank"],
            "conditional_rank": runtime["conditional_rank"],
            "incremental_prepared_float_scalar_count": runtime[
                "incremental_prepared_float_scalar_count"
            ],
            "prepared_float_scalar_count": runtime["prepared_float_scalar_count"],
            "incremental_runtime_parameter_bytes_float64": runtime[
                "incremental_runtime_parameter_bytes_float64"
            ],
            "runtime_parameter_bytes_float64": runtime[
                "runtime_parameter_bytes_float64"
            ],
            "incremental_runtime_parameter_bytes_uint8": runtime[
                "incremental_runtime_parameter_bytes_uint8"
            ],
            "runtime_parameter_bytes_uint8": runtime["runtime_parameter_bytes_uint8"],
            "incremental_logical_macs_per_token_upper_bound": runtime[
                "incremental_logical_macs_per_token_upper_bound"
            ],
            "logical_macs_per_token_upper_bound": runtime[
                "logical_macs_per_token_upper_bound"
            ],
            "runtime_state_float_scalars_per_sequence": runtime[
                "runtime_state_float_scalars_per_sequence"
            ],
            "logical_macs_accounting_scope": runtime["logical_macs_accounting_scope"],
            "pointwise_trust_certificate_scope": runtime[
                "pointwise_trust_certificate_scope"
            ],
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.compiled_provider.bounded_coordinates(parent_modal)

    def response_gain(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.response_gain(coordinates)

    def terms_from_parent(self, parent_modal: Tensor, coordinates: Tensor | None = None):
        return self.compiled_provider.terms_from_parent(parent_modal, coordinates)

    def unbounded_direction(
        self, parent_modal: Tensor, coordinates: Tensor | None = None
    ) -> Tensor:
        return self.compiled_provider.unbounded_direction(parent_modal, coordinates)

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_logits(coordinates)

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_values(coordinates)

    def modal_correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        return self.compiled_provider.modal_correction(prefix, realized_state)

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        return self.compiled_provider.correction(prefix, realized_state)


def build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    field_bias: object,
    field_slope: object,
    feature_id: object,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider:
    # V20m already emitted a box-normalized float64 direction.  Preserve every
    # bit so constant +/-1 fields are identical to the source/mirror controls.
    source_direction = _direction(
        direction,
        detach=True,
        normalize=False,
        label="local-signed-field source direction",
    )
    rate, mass, polarity = _source_response(radius, shrink_mass, polarity_bias)
    bias = _strict_scalar(field_bias, label="local-signed-field bias")
    slope = _strict_scalar(field_slope, label="local-signed-field slope")
    selected_feature = _feature_id(feature_id)
    simplex = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base_provider,
        proposal_provider,
        direction=source_direction,
        radius=rate,
        shrink_mass=mass,
        polarity_bias=polarity,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
    runtime = AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldRuntimeProvider(
        source_simplex_provider=simplex,
        field_bias=bias,
        field_slope=slope,
        feature_id=selected_feature,
    )
    return AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider(
        compiled_provider=runtime
    )
