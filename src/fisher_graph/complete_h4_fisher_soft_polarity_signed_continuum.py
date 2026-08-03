"""Branch-free signed continuum between V20m mirror, fixed-plus, and V20m.

For the already selected V20m response ``(r, u, v)`` and normalized
bilinear direction ``d``, V20o introduces one fit-time scalar ``s``::

    e(c) = asinh(9*c2) / asinh(9)
    z    = d @ [1, c1, c2, c1*c2]
    q(z) = (1-u*z**2)*tanh(r*z) + v*z**2
    g_s  = e(c) * ((1-abs(s)) + abs(s)*q(sign(s)*z))

where ``s`` is in ``[-1, 1]`` and ``sign(0)`` is frozen to ``+1``.  Sign and
absolute value are compiled when the provider is built: the runtime stores a
possibly negated direction and a nonnegative mixture scalar.  Inference then
executes one fused arithmetic path with no activation-dependent router.

The anchors are exact by construction: ``s=-1`` is the V20m provider with the
negated direction, ``s=0`` is the fixed positive envelope, and ``s=+1`` is the
original V20m provider.  The response remains bounded because it is a convex
combination of ``1`` and the bounded V20m simplex response.  This module is an
analysis-only provider and grants no serving, compression, or speed claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from numbers import Real

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
    fisher_soft_polarity_simplex_response_calibrator,
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
    "FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_FIT_LINEAGE_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256",
    "FISHER_SOFT_POLARITY_SIMPLEX_SIGNED_CONTINUUM_PROTOCOL_SHA256",
    "FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_RUNTIME_FITTED_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_SIGNED_SCALAR_MAX",
    "AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider",
    "AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider",
    "FisherSoftPolaritySignedContinuumProviderEvidence",
    "FisherSoftPolaritySignedContinuumRuntimeProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum",
    "fisher_soft_polarity_signed_continuum_box_certificate",
    "fisher_soft_polarity_signed_continuum_calibrator",
    "fisher_soft_polarity_signed_continuum_constant_tensor_sha256s",
    "fisher_soft_polarity_signed_continuum_direction_sha256",
    "fisher_soft_polarity_signed_continuum_gain",
    "fisher_soft_polarity_signed_continuum_materialized_parameters",
    "fisher_soft_polarity_signed_continuum_modal_terms",
    "fisher_soft_polarity_signed_continuum_projection",
    "fisher_soft_polarity_signed_continuum_provider_artifact_sha256",
    "fisher_soft_polarity_signed_continuum_runtime_provider_artifact_sha256",
    "fisher_soft_polarity_signed_continuum_value",
    "normalize_fisher_soft_polarity_signed_continuum_direction",
    "validate_fisher_soft_polarity_signed_continuum_provider_evidence",
    "validate_fisher_soft_polarity_signed_continuum_runtime_provider_evidence",
]


_DIRECTION_COUNT = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT
_SOURCE_RESPONSE_SCALAR_COUNT = 3
_FIT_LINEAGE_SCALAR_COUNT = _DIRECTION_COUNT + _SOURCE_RESPONSE_SCALAR_COUNT + 1
_RUNTIME_FITTED_SCALAR_COUNT = (
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT + 1
)
_RADIUS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX
_SHRINK_MASS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX
_TRUST_FRACTION = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION
_SIGNED_SCALAR_MAX = 1.0

FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_FIT_LINEAGE_SCALAR_COUNT = (
    _FIT_LINEAGE_SCALAR_COUNT
)
FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_RUNTIME_FITTED_SCALAR_COUNT = (
    _RUNTIME_FITTED_SCALAR_COUNT
)
FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_SIGNED_SCALAR_MAX = _SIGNED_SCALAR_MAX

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-"
    b"continuum:provider:v20o\0"
)
_RUNTIME_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-"
    b"continuum:runtime-provider:v20o\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-"
    b"continuum:protocol:v20o\0"
)
_DIRECTION_BINDING_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-"
    b"continuum:direction-binding:v20o\0"
)
_FORMULA = (
    "e(c)=asinh(9*c2)/asinh(9);z=d_dot_[1,c1,c2,c1*c2];"
    "q(z)=(1-u*z_squared)*tanh(r*z)+v*z_squared;"
    "g_s(c)=e(c)*((1-abs(s))+abs(s)*q(sign_nonnegative_zero(s)*z))"
)
_RUNTIME_FORMULA = (
    "e(c)*((1-compiled_mix)+compiled_mix*q(compiled_direction_dot_features))"
)
_PROTOCOL = {
    "schema": "fisher_graph.complete_h4_soft_polarity_signed_continuum.v20o",
    "formula": _FORMULA,
    "signed_scalar_constraint": "minus_one_le_s_le_plus_one",
    "zero_sign": "plus_one",
    "compile_time_materialization": (
        "compiled_direction_equals_sign_nonnegative_zero_s_times_source_"
        "direction_and_compiled_mix_equals_abs_s"
    ),
    "runtime_formula": _RUNTIME_FORMULA,
    "runtime_activation_dependent_branch": False,
    "minus_one_identity": "exact_v20m_negated_direction_mirror",
    "zero_identity": "exact_fixed_positive_envelope",
    "plus_one_identity": "exact_v20m_source_direction",
    "boundedness": "convex_combination_of_plus_one_and_v20m_simplex_response",
    "serving_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
}
FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256 = _sha256(
    _PROTOCOL_DOMAIN, _PROTOCOL
)
# Compatibility spelling used by the first V20o runner scaffold.  Both names
# bind the same frozen protocol; new code should prefer the shorter spelling.
FISHER_SOFT_POLARITY_SIMPLEX_SIGNED_CONTINUUM_PROTOCOL_SHA256 = (
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
)


def _strict_scalar(
    value: object, *, label: str, lower: float, upper: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if result < lower or result > upper:
        raise ValueError(f"{label} must be inside [{lower},{upper}]")
    return result


def _source_response(
    radius: object, shrink_mass: object, polarity_bias: object
) -> tuple[float, float, float]:
    rate = _strict_scalar(
        radius, label="signed-continuum radius", lower=0.0, upper=_RADIUS_MAX
    )
    mass = _strict_scalar(
        shrink_mass,
        label="signed-continuum shrink mass",
        lower=0.0,
        upper=_SHRINK_MASS_MAX,
    )
    bias = _strict_scalar(
        polarity_bias,
        label="signed-continuum polarity bias",
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )
    if abs(bias) > mass:
        raise ValueError(
            "signed-continuum response must satisfy abs(polarity_bias) <= "
            "shrink_mass"
        )
    return rate, mass, bias


def fisher_soft_polarity_signed_continuum_materialized_parameters(
    signed_scalar: object,
) -> tuple[int, float]:
    """Return the compile-time direction sign and nonnegative runtime mix."""

    signed = _strict_scalar(
        signed_scalar,
        label="signed-continuum scalar",
        lower=-_SIGNED_SCALAR_MAX,
        upper=_SIGNED_SCALAR_MAX,
    )
    return (1 if signed >= 0.0 else -1), abs(signed)


def fisher_soft_polarity_signed_continuum_constant_tensor_sha256s() -> dict[str, str]:
    return fisher_soft_polarity_simplex_response_constant_tensor_sha256s()


def normalize_fisher_soft_polarity_signed_continuum_direction(
    direction: Tensor,
) -> Tensor:
    return normalize_fisher_soft_polarity_simplex_response_direction(direction)


def fisher_soft_polarity_signed_continuum_direction_sha256(
    direction: Tensor,
) -> str:
    return fisher_soft_polarity_simplex_response_direction_sha256(direction)


def fisher_soft_polarity_signed_continuum_projection(
    coordinates: Tensor, compiled_direction: Tensor
) -> Tensor:
    return fisher_soft_polarity_simplex_response_projection(
        coordinates, compiled_direction
    )


def fisher_soft_polarity_signed_continuum_calibrator(
    projection: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    compiled_mix: object,
) -> Tensor:
    """Evaluate the one-path convex response with already compiled scalars."""

    rate, mass, bias = _source_response(radius, shrink_mass, polarity_bias)
    mix = _strict_scalar(
        compiled_mix,
        label="signed-continuum compiled mix",
        lower=0.0,
        upper=1.0,
    )
    response = fisher_soft_polarity_simplex_response_calibrator(
        projection, rate, mass, bias
    )
    result = (1.0 - mix) + mix * response
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("signed-continuum calibrator violated its bound")
    return result.contiguous()


def fisher_soft_polarity_signed_continuum_value(
    coordinates: Tensor,
    compiled_direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    compiled_mix: object,
) -> Tensor:
    projection = fisher_soft_polarity_signed_continuum_projection(
        coordinates, compiled_direction
    )
    return fisher_soft_polarity_signed_continuum_calibrator(
        projection, radius, shrink_mass, polarity_bias, compiled_mix
    )


def fisher_soft_polarity_signed_continuum_gain(
    coordinates: Tensor,
    compiled_direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    compiled_mix: object,
) -> Tensor:
    envelope = fisher_soft_polarity_envelope(coordinates)
    value = fisher_soft_polarity_signed_continuum_value(
        coordinates,
        compiled_direction,
        radius,
        shrink_mass,
        polarity_bias,
        compiled_mix,
    )
    result = envelope * value
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("signed-continuum gain violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_signed_continuum_modal_terms(
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
    compiled_direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    compiled_mix: object,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    parent = _finite_runtime_tensor(
        parent_modal, label="signed-continuum parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="signed-continuum coordinates", ndim=2
    ).to(parent.device)
    if (
        bounded_coordinates.shape != (parent.shape[0], 2)
        or bool((bounded_coordinates.abs() > 1.0).any())
    ):
        raise ValueError(
            "signed-continuum coordinates must match parent rows inside [-1,1]"
        )
    if trust_fraction != _TRUST_FRACTION:
        raise ValueError("signed-continuum trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_signed_continuum_gain(
        bounded_coordinates,
        compiled_direction,
        radius,
        shrink_mass,
        polarity_bias,
        compiled_mix,
    )
    features = fisher_finite_joint_direction_features(parent, bounded_coordinates)
    direction = fisher_continuous_factor_direction(
        features,
        base_direction_left,
        base_direction_right,
        proposal_direction_left,
        proposal_direction_right,
        gain,
    )
    bounded = fisher_xy_pointwise_bounded_direction(
        parent, direction, trust_fraction=trust_fraction
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
        raise RuntimeError("signed-continuum modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


def _direction_binding_sha256(
    source_direction_sha256: str, compiled_sign: int, compiled_direction_sha256: str
) -> str:
    return _sha256(
        _DIRECTION_BINDING_DOMAIN,
        {
            "source_direction_sha256": _require_sha256(
                source_direction_sha256, label="signed-continuum source direction"
            ),
            "compiled_sign": compiled_sign,
            "compiled_direction_sha256": _require_sha256(
                compiled_direction_sha256,
                label="signed-continuum compiled direction",
            ),
        },
    )


def _box_certificate_payload(
    *,
    source_direction_sha256: str,
    compiled_direction_sha256: str,
    compiled_sign: int,
    signed_scalar: float,
    compiled_mix: float,
    radius: float,
    shrink_mass: float,
    polarity_bias: float,
) -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.fisher_soft_polarity_signed_continuum_box_"
            "certificate.v20o"
        ),
        "protocol_sha256": FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
        "formula": _FORMULA,
        "source_direction_sha256": source_direction_sha256,
        "compiled_direction_sha256": compiled_direction_sha256,
        "compiled_sign": compiled_sign,
        "signed_scalar": signed_scalar,
        "compiled_mix": compiled_mix,
        "radius": radius,
        "shrink_mass": shrink_mass,
        "polarity_bias": polarity_bias,
        "signed_scalar_in_closed_interval": (-1.0, 1.0),
        "compiled_mix_in_closed_unit_interval": True,
        "source_simplex_constraints_satisfied": True,
        "runtime_activation_dependent_branch": False,
        "minus_one_exact_v20m_mirror": signed_scalar == -1.0,
        "zero_exact_fixed_plus": signed_scalar == 0.0,
        "plus_one_exact_v20m_reflected": signed_scalar == 1.0,
        "simplex_response_max_abs": 1.0,
        "constant_response_max_abs": 1.0,
        "continuum_response_max_abs": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "compiled_mix_in_zero_one_makes_the_runtime_response_a_convex_"
            "combination_of_plus_one_and_the_bounded_v20m_simplex_response_"
            "then_the_bounded_envelope_preserves_absolute_gain_at_most_one"
        ),
    }


_BOX_CERTIFICATE_KEYS = frozenset(
    _box_certificate_payload(
        source_direction_sha256="0" * 64,
        compiled_direction_sha256="0" * 64,
        compiled_sign=1,
        signed_scalar=0.0,
        compiled_mix=0.0,
        radius=0.0,
        shrink_mass=0.0,
        polarity_bias=0.0,
    )
)


def fisher_soft_polarity_signed_continuum_box_certificate(
    direction: Tensor,
    *,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    signed_scalar: object,
) -> dict[str, object]:
    # Certificate the exact frozen V20m values.  Re-normalizing here would make
    # the certificate disagree with the bit-identical runtime/provider payload
    # for directions whose normalization is not float64-idempotent.
    selected = _direction(
        direction,
        detach=True,
        normalize=False,
        label="signed-continuum certificate direction",
    )
    source_sha = fisher_soft_polarity_signed_continuum_direction_sha256(selected)
    rate, mass, bias = _source_response(radius, shrink_mass, polarity_bias)
    signed = _strict_scalar(
        signed_scalar,
        label="signed-continuum scalar",
        lower=-1.0,
        upper=1.0,
    )
    compiled_sign, mix = fisher_soft_polarity_signed_continuum_materialized_parameters(
        signed
    )
    compiled = selected if compiled_sign == 1 else -selected
    compiled_sha = fisher_soft_polarity_signed_continuum_direction_sha256(compiled)
    return _box_certificate_payload(
        source_direction_sha256=source_sha,
        compiled_direction_sha256=compiled_sha,
        compiled_sign=compiled_sign,
        signed_scalar=signed,
        compiled_mix=mix,
        radius=rate,
        shrink_mass=mass,
        polarity_bias=bias,
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
        "compiled_direction_sha256",
        "radius",
        "shrink_mass",
        "polarity_bias",
        "compiled_mix",
        "compiled_mix_hex",
        "compiled_mix_sha256",
        "compiled_simplex_response_provider_artifact_sha256",
        "compiled_simplex_response_provider_payload",
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
        "compiled_simplex_response_provider_metadata",
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "signed_continuum_projection_dot_macs_per_token",
        "signed_continuum_calibrator_scalar_arithmetic_per_token",
        "signed_continuum_elementwise_scalar_arithmetic_per_token",
        "signed_continuum_nonlinear_scalar_ops_per_token",
        "runtime_state_float_scalars_per_sequence",
        "logical_macs_accounting_scope",
        "pointwise_trust_certificate_scope",
    }
)


def _validate_runtime_payload(value: object) -> dict[str, object]:
    payload = _require_exact_keys(
        value, _RUNTIME_PAYLOAD_KEYS, label="signed-continuum runtime payload"
    )
    _canonical_evidence_tree(payload, label="signed-continuum runtime payload")
    _validate_evidence_sha_fields(payload, label="signed-continuum runtime payload")
    frozen = {
        "schema": (
            "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
            "signed_continuum_runtime_provider.v20o"
        ),
        "site": _H4_SITE,
        "write_scope": "complete_h4_causal_support",
        "protocol_sha256": FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
        "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
        "runtime_formula": _RUNTIME_FORMULA,
        "routing_control_flow": "none_validation_guards_only",
        "boundedness_certificate": "convex_plus_one_and_v20m_simplex_response",
        "serving_status": "analysis_only_no_serving_compression_or_speed_claim",
    }
    for key, expected in frozen.items():
        _require_canonical_equal(
            payload[key], expected, label=f"signed-continuum runtime {key}"
        )
    mix = payload["compiled_mix"]
    if type(mix) is not float:
        raise ValueError("signed-continuum runtime mix must be a float")
    mix = _strict_scalar(
        mix, label="signed-continuum runtime mix", lower=0.0, upper=1.0
    )
    if payload["compiled_mix_hex"] != mix.hex():
        raise ValueError("signed-continuum runtime mix hex differs")
    if payload["compiled_mix_sha256"] != _response_scalar_sha256(mix):
        raise ValueError("signed-continuum runtime mix hash differs")
    rate, mass, bias = _source_response(
        payload["radius"], payload["shrink_mass"], payload["polarity_bias"]
    )
    nested_payload = payload["compiled_simplex_response_provider_payload"]
    nested_artifact = fisher_soft_polarity_simplex_response_provider_artifact_sha256(
        nested_payload
    )
    if payload["compiled_simplex_response_provider_artifact_sha256"] != nested_artifact:
        raise ValueError("signed-continuum nested V20m artifact differs")
    nested = _canonical_evidence_tree(
        nested_payload, label="signed-continuum nested V20m payload"
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
        "compiled_direction_sha256": nested["direction_sha256"],
        "radius": nested["radius"],
        "shrink_mass": nested["shrink_mass"],
        "polarity_bias": nested["polarity_bias"],
        "runtime_inputs": nested["runtime_inputs"],
        "runtime_forbidden_inputs": nested["runtime_forbidden_inputs"],
    }
    if (rate, mass, bias) != (
        float(nested["radius"]),
        float(nested["shrink_mass"]),
        float(nested["polarity_bias"]),
    ):
        raise ValueError("signed-continuum runtime response differs from V20m")
    for key, expected in bindings.items():
        _require_canonical_equal(
            payload[key], expected, label=f"signed-continuum runtime binding {key}"
        )
    canonical = _canonical_evidence_tree(
        payload, label="signed-continuum runtime payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_signed_continuum_runtime_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_RUNTIME_PROVIDER_DOMAIN, _validate_runtime_payload(payload))


@dataclass(frozen=True, slots=True)
class FisherSoftPolaritySignedContinuumRuntimeProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def validate_fisher_soft_polarity_signed_continuum_runtime_provider_evidence(
    payload: object, metadata: object
) -> FisherSoftPolaritySignedContinuumRuntimeProviderEvidence:
    canonical_payload = _validate_runtime_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _RUNTIME_PAYLOAD_KEYS | _RUNTIME_METADATA_EXTRA_KEYS,
        label="signed-continuum runtime metadata",
    )
    _canonical_evidence_tree(selected, label="signed-continuum runtime metadata")
    _validate_evidence_sha_fields(
        selected, label="signed-continuum runtime metadata"
    )
    for key in _RUNTIME_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"signed-continuum runtime metadata {key}",
        )
    artifact = fisher_soft_polarity_signed_continuum_runtime_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact:
        raise ValueError("signed-continuum runtime artifact hash differs")
    nested = validate_fisher_soft_polarity_simplex_response_provider_evidence(
        canonical_payload["compiled_simplex_response_provider_payload"],
        selected["compiled_simplex_response_provider_metadata"],
    )
    if nested.artifact_sha256 != canonical_payload[
        "compiled_simplex_response_provider_artifact_sha256"
    ]:
        raise ValueError("signed-continuum nested runtime evidence differs")
    for key in (
        "rank",
        "conditional_rank",
        "runtime_state_float_scalars_per_sequence",
    ):
        if _strict_evidence_integer(selected, key) != _strict_evidence_integer(
            nested.metadata, key
        ):
            raise ValueError(f"signed-continuum runtime delegated {key} differs")
    expected_integers = {
        "incremental_prepared_float_scalar_count": (
            int(nested.metadata["incremental_prepared_float_scalar_count"]) + 1
        ),
        "prepared_float_scalar_count": int(nested.metadata["prepared_float_scalar_count"])
        + 1,
        "incremental_runtime_parameter_bytes_float64": int(
            nested.metadata["incremental_runtime_parameter_bytes_float64"]
        )
        + 8,
        "runtime_parameter_bytes_float64": int(
            nested.metadata["runtime_parameter_bytes_float64"]
        )
        + 8,
        "incremental_logical_macs_per_token_upper_bound": int(
            nested.metadata["incremental_logical_macs_per_token_upper_bound"]
        ),
        "logical_macs_per_token_upper_bound": int(
            nested.metadata["logical_macs_per_token_upper_bound"]
        ),
        "signed_continuum_projection_dot_macs_per_token": _DIRECTION_COUNT,
        "signed_continuum_calibrator_scalar_arithmetic_per_token": 10,
        "signed_continuum_elementwise_scalar_arithmetic_per_token": 13,
        "signed_continuum_nonlinear_scalar_ops_per_token": 2,
    }
    for key, expected in expected_integers.items():
        if _strict_evidence_integer(selected, key) != expected:
            raise ValueError(f"signed-continuum runtime accounting {key} differs")
    _require_canonical_equal(
        selected["logical_macs_accounting_scope"],
        nested.metadata["logical_macs_accounting_scope"],
        label="signed-continuum runtime MAC scope",
    )
    _require_canonical_equal(
        selected["pointwise_trust_certificate_scope"],
        nested.metadata["pointwise_trust_certificate_scope"],
        label="signed-continuum runtime trust scope",
    )
    certificate = _require_exact_keys(
        selected["box_certificate"],
        _BOX_CERTIFICATE_KEYS,
        label="signed-continuum runtime box certificate",
    )
    expected_certificate = _box_certificate_payload(
        source_direction_sha256=str(canonical_payload["compiled_direction_sha256"]),
        compiled_direction_sha256=str(canonical_payload["compiled_direction_sha256"]),
        compiled_sign=1,
        signed_scalar=float(canonical_payload["compiled_mix"]),
        compiled_mix=float(canonical_payload["compiled_mix"]),
        radius=float(canonical_payload["radius"]),
        shrink_mass=float(canonical_payload["shrink_mass"]),
        polarity_bias=float(canonical_payload["polarity_bias"]),
    )
    for key, expected in expected_certificate.items():
        _require_canonical_equal(
            certificate[key],
            expected,
            label=f"signed-continuum runtime certificate {key}",
        )
    canonical_metadata = _canonical_evidence_tree(
        selected, label="signed-continuum runtime metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolaritySignedContinuumRuntimeProviderEvidence(
        canonical_payload, canonical_metadata, artifact
    )


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider(
    Gemma3L3L4CorrectionProvider
):
    """The only V20o object installed for inference execution."""

    compiled_simplex_provider: AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
    compiled_mix: float
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.compiled_simplex_provider,
            AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
        ):
            raise TypeError("signed-continuum runtime needs one compiled V20m provider")
        object.__setattr__(
            self,
            "compiled_mix",
            _strict_scalar(
                self.compiled_mix,
                label="signed-continuum compiled mix",
                lower=0.0,
                upper=1.0,
            ),
        )
        self.compiled_simplex_provider.validate_integrity()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="signed-continuum runtime provider"
            ) != computed:
                raise ValueError("signed-continuum runtime artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def base_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.compiled_simplex_provider.base_provider

    @property
    def proposal_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.compiled_simplex_provider.proposal_provider

    @property
    def parent_provider(self):
        return self.compiled_simplex_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.compiled_simplex_provider.bridge_binding_sha256

    @property
    def transfer_protocol_sha256(self) -> str:
        return self.compiled_simplex_provider.transfer_protocol_sha256

    @property
    def transfer_evidence_sha256(self) -> str:
        return self.compiled_simplex_provider.transfer_evidence_sha256

    @property
    def direction(self) -> Tensor:
        return self.compiled_simplex_provider.direction

    @property
    def radius(self) -> float:
        return self.compiled_simplex_provider.radius

    @property
    def shrink_mass(self) -> float:
        return self.compiled_simplex_provider.shrink_mass

    @property
    def polarity_bias(self) -> float:
        return self.compiled_simplex_provider.polarity_bias

    @property
    def trust_fraction(self) -> float:
        return self.compiled_simplex_provider.trust_fraction

    @property
    def rank(self) -> int:
        return self.compiled_simplex_provider.rank

    @property
    def conditional_rank(self) -> int:
        return self.compiled_simplex_provider.conditional_rank

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return self.compiled_simplex_provider.incremental_prepared_float_scalar_count + 1

    @property
    def prepared_float_scalar_count(self) -> int:
        return self.compiled_simplex_provider.prepared_float_scalar_count + 1

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        return self.compiled_simplex_provider.incremental_logical_macs_per_token_upper_bound

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return self.compiled_simplex_provider.logical_macs_per_token_upper_bound

    def _payload(self) -> dict[str, object]:
        nested = self.compiled_simplex_provider.artifact_payload()
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "signed_continuum_runtime_provider.v20o"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "protocol_sha256": FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": self.proposal_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.base_provider.start_provider_artifact_sha256,
            "transfer_protocol_sha256": self.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.transfer_evidence_sha256,
            "compiled_direction_sha256": (
                fisher_soft_polarity_signed_continuum_direction_sha256(self.direction)
            ),
            "radius": self.radius,
            "shrink_mass": self.shrink_mass,
            "polarity_bias": self.polarity_bias,
            "compiled_mix": self.compiled_mix,
            "compiled_mix_hex": self.compiled_mix.hex(),
            "compiled_mix_sha256": _response_scalar_sha256(self.compiled_mix),
            "compiled_simplex_response_provider_artifact_sha256": (
                self.compiled_simplex_provider.artifact_sha256
            ),
            "compiled_simplex_response_provider_payload": nested,
            "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
            "runtime_formula": _RUNTIME_FORMULA,
            "runtime_inputs": nested["runtime_inputs"],
            "runtime_forbidden_inputs": nested["runtime_forbidden_inputs"],
            "routing_control_flow": "none_validation_guards_only",
            "boundedness_certificate": "convex_plus_one_and_v20m_simplex_response",
            "serving_status": "analysis_only_no_serving_compression_or_speed_claim",
        }

    def _computed_sha256(self) -> str:
        return _sha256(_RUNTIME_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        self.compiled_simplex_provider.validate_integrity()
        _strict_scalar(
            self.compiled_mix,
            label="signed-continuum stored compiled mix",
            lower=0.0,
            upper=1.0,
        )
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("signed-continuum runtime payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        nested = self.compiled_simplex_provider.metadata()
        # Source sign/s are intentionally absent from the executable metadata.
        certificate = _box_certificate_payload(
            source_direction_sha256=self._payload()["compiled_direction_sha256"],
            compiled_direction_sha256=self._payload()["compiled_direction_sha256"],
            compiled_sign=1,
            signed_scalar=self.compiled_mix,
            compiled_mix=self.compiled_mix,
            radius=self.radius,
            shrink_mass=self.shrink_mass,
            polarity_bias=self.polarity_bias,
        )
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "box_certificate": certificate,
            "compiled_simplex_response_provider_metadata": nested,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": self.incremental_prepared_float_scalar_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": self.incremental_prepared_float_scalar_count * 8,
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "incremental_logical_macs_per_token_upper_bound": self.incremental_logical_macs_per_token_upper_bound,
            "logical_macs_per_token_upper_bound": self.logical_macs_per_token_upper_bound,
            "signed_continuum_projection_dot_macs_per_token": _DIRECTION_COUNT,
            "signed_continuum_calibrator_scalar_arithmetic_per_token": 10,
            "signed_continuum_elementwise_scalar_arithmetic_per_token": 13,
            "signed_continuum_nonlinear_scalar_ops_per_token": 2,
            "runtime_state_float_scalars_per_sequence": nested[
                "runtime_state_float_scalars_per_sequence"
            ],
            "logical_macs_accounting_scope": nested["logical_macs_accounting_scope"],
            "pointwise_trust_certificate_scope": nested[
                "pointwise_trust_certificate_scope"
            ],
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.compiled_simplex_provider.bounded_coordinates(parent_modal)

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
            raise ValueError("signed-continuum coordinates differ")
        leading = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_signed_continuum_gain(
            flat,
            self.direction.to(device=flat.device, dtype=flat.dtype),
            self.radius,
            self.shrink_mass,
            self.polarity_bias,
            self.compiled_mix,
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
            raise ValueError("signed-continuum parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("signed-continuum parent/coordinate geometry differs")
        original_shape = parent.shape
        values = fisher_soft_polarity_signed_continuum_modal_terms(
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
            self.compiled_mix,
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
        if prefix.artifact_sha256 != prefix_sha or _tensor_sha256(realized_state) != realized_sha:
            raise RuntimeError("signed-continuum provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("signed-continuum modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("signed-continuum modal correction escaped support")
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
        "bridge_binding_sha256",
        "parent_provider_artifact_sha256",
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "start_provider_artifact_sha256",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "source_direction_sha256",
        "compiled_direction_sha256",
        "compiled_direction_sign",
        "direction_binding_sha256",
        "source_radius",
        "source_shrink_mass",
        "source_polarity_bias",
        "signed_scalar",
        "signed_scalar_hex",
        "signed_scalar_sha256",
        "compiled_mix",
        "compiled_mix_hex",
        "compiled_mix_sha256",
        "compiled_runtime_provider_artifact_sha256",
        "compiled_runtime_provider_payload",
        "fit_lineage_float_scalar_count",
        "runtime_fitted_float_scalar_count",
        "source_signed_scalar_runtime_float_scalar_count",
        "sign_and_absolute_value_materialized_at_construction",
        "minus_one_identity",
        "zero_identity",
        "plus_one_identity",
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
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_scalars_per_sequence",
        "logical_macs_accounting_scope",
        "pointwise_trust_certificate_scope",
    }
)


def _validate_provider_payload(value: object) -> dict[str, object]:
    payload = _require_exact_keys(
        value, _PROVIDER_PAYLOAD_KEYS, label="signed-continuum provider payload"
    )
    _canonical_evidence_tree(payload, label="signed-continuum provider payload")
    _validate_evidence_sha_fields(payload, label="signed-continuum provider payload")
    frozen = {
        "schema": (
            "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
            "signed_continuum_provider.v20o"
        ),
        "site": _H4_SITE,
        "write_scope": "complete_h4_causal_support",
        "protocol_sha256": FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
        "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
        "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
        "source_signed_scalar_runtime_float_scalar_count": 0,
        "sign_and_absolute_value_materialized_at_construction": True,
        "minus_one_identity": "exact_v20m_negated_direction_mirror",
        "zero_identity": "exact_fixed_positive_envelope",
        "plus_one_identity": "exact_v20m_source_direction",
        "gain_formula": _FORMULA,
        "routing_control_flow": "none_validation_guards_only",
        "boundedness_certificate": "convex_plus_one_and_v20m_simplex_response",
        "serving_status": "analysis_only_no_serving_compression_or_speed_claim",
    }
    for key, expected in frozen.items():
        _require_canonical_equal(
            payload[key], expected, label=f"signed-continuum payload {key}"
        )
    rate, mass, bias = _source_response(
        payload["source_radius"],
        payload["source_shrink_mass"],
        payload["source_polarity_bias"],
    )
    signed = payload["signed_scalar"]
    if type(signed) is not float:
        raise ValueError("signed-continuum payload signed scalar must be a float")
    signed = _strict_scalar(
        signed, label="signed-continuum payload signed scalar", lower=-1.0, upper=1.0
    )
    sign, mix = fisher_soft_polarity_signed_continuum_materialized_parameters(signed)
    if payload["compiled_direction_sign"] != sign:
        raise ValueError("signed-continuum compiled direction sign differs")
    if payload["compiled_mix"] != mix:
        raise ValueError("signed-continuum compiled mix differs")
    for prefix, scalar in (("signed_scalar", signed), ("compiled_mix", mix)):
        if payload[f"{prefix}_hex"] != scalar.hex():
            raise ValueError(f"signed-continuum {prefix} hex differs")
        if payload[f"{prefix}_sha256"] != _response_scalar_sha256(scalar):
            raise ValueError(f"signed-continuum {prefix} hash differs")
    expected_binding = _direction_binding_sha256(
        str(payload["source_direction_sha256"]),
        sign,
        str(payload["compiled_direction_sha256"]),
    )
    if payload["direction_binding_sha256"] != expected_binding:
        raise ValueError("signed-continuum direction binding differs")
    runtime_payload = payload["compiled_runtime_provider_payload"]
    runtime_artifact = fisher_soft_polarity_signed_continuum_runtime_provider_artifact_sha256(
        runtime_payload
    )
    if payload["compiled_runtime_provider_artifact_sha256"] != runtime_artifact:
        raise ValueError("signed-continuum runtime artifact differs")
    runtime = _canonical_evidence_tree(
        runtime_payload, label="signed-continuum compiled runtime payload"
    )
    assert isinstance(runtime, dict)
    bindings = {
        "site": runtime["site"],
        "write_scope": runtime["write_scope"],
        "bridge_binding_sha256": runtime["bridge_binding_sha256"],
        "parent_provider_artifact_sha256": runtime[
            "parent_provider_artifact_sha256"
        ],
        "base_provider_artifact_sha256": runtime["base_provider_artifact_sha256"],
        "proposal_provider_artifact_sha256": runtime[
            "proposal_provider_artifact_sha256"
        ],
        "start_provider_artifact_sha256": runtime[
            "start_provider_artifact_sha256"
        ],
        "transfer_protocol_sha256": runtime["transfer_protocol_sha256"],
        "transfer_evidence_sha256": runtime["transfer_evidence_sha256"],
        "compiled_direction_sha256": runtime["compiled_direction_sha256"],
        "compiled_mix": runtime["compiled_mix"],
        "runtime_inputs": runtime["runtime_inputs"],
        "runtime_forbidden_inputs": runtime["runtime_forbidden_inputs"],
    }
    if (rate, mass, bias) != (
        float(runtime["radius"]),
        float(runtime["shrink_mass"]),
        float(runtime["polarity_bias"]),
    ):
        raise ValueError("signed-continuum source response differs from runtime")
    for key, expected in bindings.items():
        _require_canonical_equal(
            payload[key], expected, label=f"signed-continuum runtime binding {key}"
        )
    canonical = _canonical_evidence_tree(
        payload, label="signed-continuum provider payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_signed_continuum_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_PROVIDER_DOMAIN, _validate_provider_payload(payload))


@dataclass(frozen=True, slots=True)
class FisherSoftPolaritySignedContinuumProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def validate_fisher_soft_polarity_signed_continuum_provider_evidence(
    payload: object, metadata: object
) -> FisherSoftPolaritySignedContinuumProviderEvidence:
    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="signed-continuum provider metadata",
    )
    _canonical_evidence_tree(selected, label="signed-continuum provider metadata")
    _validate_evidence_sha_fields(selected, label="signed-continuum provider metadata")
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"signed-continuum metadata {key}",
        )
    artifact = fisher_soft_polarity_signed_continuum_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact:
        raise ValueError("signed-continuum provider artifact hash differs")
    runtime = validate_fisher_soft_polarity_signed_continuum_runtime_provider_evidence(
        canonical_payload["compiled_runtime_provider_payload"],
        selected["compiled_runtime_provider_metadata"],
    )
    if runtime.artifact_sha256 != canonical_payload[
        "compiled_runtime_provider_artifact_sha256"
    ]:
        raise ValueError("signed-continuum nested runtime evidence differs")
    for key in (
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_scalars_per_sequence",
    ):
        if _strict_evidence_integer(selected, key) != _strict_evidence_integer(
            runtime.metadata, key
        ):
            raise ValueError(f"signed-continuum delegated accounting {key} differs")
    for key in ("logical_macs_accounting_scope", "pointwise_trust_certificate_scope"):
        _require_canonical_equal(
            selected[key], runtime.metadata[key], label=f"signed-continuum {key}"
        )
    certificate = _require_exact_keys(
        selected["box_certificate"],
        _BOX_CERTIFICATE_KEYS,
        label="signed-continuum box certificate",
    )
    expected_certificate = _box_certificate_payload(
        source_direction_sha256=str(canonical_payload["source_direction_sha256"]),
        compiled_direction_sha256=str(
            canonical_payload["compiled_direction_sha256"]
        ),
        compiled_sign=int(canonical_payload["compiled_direction_sign"]),
        signed_scalar=float(canonical_payload["signed_scalar"]),
        compiled_mix=float(canonical_payload["compiled_mix"]),
        radius=float(canonical_payload["source_radius"]),
        shrink_mass=float(canonical_payload["source_shrink_mass"]),
        polarity_bias=float(canonical_payload["source_polarity_bias"]),
    )
    for key, expected in expected_certificate.items():
        _require_canonical_equal(
            certificate[key], expected, label=f"signed-continuum certificate {key}"
        )
    canonical_metadata = _canonical_evidence_tree(
        selected, label="signed-continuum provider metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolaritySignedContinuumProviderEvidence(
        canonical_payload, canonical_metadata, artifact
    )


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider(
    Gemma3L3L4CorrectionProvider
):
    """Authenticated fit-lineage wrapper around one compiled runtime provider."""

    compiled_provider: AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider
    source_direction_sha256: str
    source_radius: float
    source_shrink_mass: float
    source_polarity_bias: float
    signed_scalar: float
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.compiled_provider,
            AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider,
        ):
            raise TypeError("signed-continuum wrapper needs one compiled runtime")
        object.__setattr__(
            self,
            "source_direction_sha256",
            _require_sha256(
                self.source_direction_sha256,
                label="signed-continuum source direction",
            ),
        )
        source = _source_response(
            self.source_radius, self.source_shrink_mass, self.source_polarity_bias
        )
        for name, value in zip(
            ("source_radius", "source_shrink_mass", "source_polarity_bias"),
            source,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "signed_scalar",
            _strict_scalar(
                self.signed_scalar,
                label="signed-continuum scalar",
                lower=-1.0,
                upper=1.0,
            ),
        )
        self._validate_compiled_binding()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="signed-continuum provider"
            ) != computed:
                raise ValueError("signed-continuum provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def runtime_provider(
        self,
    ) -> AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider:
        return self.compiled_provider

    @property
    def base_provider(self):
        return self.compiled_provider.base_provider

    @property
    def proposal_provider(self):
        return self.compiled_provider.proposal_provider

    @property
    def parent_provider(self):
        return self.compiled_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.compiled_provider.bridge_binding_sha256

    @property
    def direction(self) -> Tensor:
        sign, _mix = fisher_soft_polarity_signed_continuum_materialized_parameters(
            self.signed_scalar
        )
        compiled = self.compiled_provider.direction
        return compiled if sign == 1 else -compiled

    @property
    def compiled_direction_sign(self) -> int:
        return fisher_soft_polarity_signed_continuum_materialized_parameters(
            self.signed_scalar
        )[0]

    @property
    def compiled_mix(self) -> float:
        return self.compiled_provider.compiled_mix

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

    def _validate_compiled_binding(self) -> None:
        self.compiled_provider.validate_integrity()
        sign, mix = fisher_soft_polarity_signed_continuum_materialized_parameters(
            self.signed_scalar
        )
        if self.compiled_provider.compiled_mix != mix:
            raise RuntimeError("signed-continuum compiled mixture drifted")
        if (
            self.source_radius,
            self.source_shrink_mass,
            self.source_polarity_bias,
        ) != (
            self.compiled_provider.radius,
            self.compiled_provider.shrink_mass,
            self.compiled_provider.polarity_bias,
        ):
            raise RuntimeError("signed-continuum compiled response drifted")
        reconstructed = self.compiled_provider.direction
        if sign == -1:
            reconstructed = -reconstructed
        if (
            fisher_soft_polarity_signed_continuum_direction_sha256(reconstructed)
            != self.source_direction_sha256
        ):
            raise RuntimeError("signed-continuum compiled direction drifted")

    def _payload(self) -> dict[str, object]:
        runtime = self.compiled_provider.artifact_payload()
        signed = self.signed_scalar
        mix = self.compiled_mix
        compiled_sha = str(runtime["compiled_direction_sha256"])
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "signed_continuum_provider.v20o"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "protocol_sha256": FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": self.proposal_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.base_provider.start_provider_artifact_sha256,
            "transfer_protocol_sha256": self.compiled_provider.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.compiled_provider.transfer_evidence_sha256,
            "source_direction_sha256": self.source_direction_sha256,
            "compiled_direction_sha256": compiled_sha,
            "compiled_direction_sign": self.compiled_direction_sign,
            "direction_binding_sha256": _direction_binding_sha256(
                self.source_direction_sha256,
                self.compiled_direction_sign,
                compiled_sha,
            ),
            "source_radius": self.source_radius,
            "source_shrink_mass": self.source_shrink_mass,
            "source_polarity_bias": self.source_polarity_bias,
            "signed_scalar": signed,
            "signed_scalar_hex": signed.hex(),
            "signed_scalar_sha256": _response_scalar_sha256(signed),
            "compiled_mix": mix,
            "compiled_mix_hex": mix.hex(),
            "compiled_mix_sha256": _response_scalar_sha256(mix),
            "compiled_runtime_provider_artifact_sha256": self.compiled_provider.artifact_sha256,
            "compiled_runtime_provider_payload": runtime,
            "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
            "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
            "source_signed_scalar_runtime_float_scalar_count": 0,
            "sign_and_absolute_value_materialized_at_construction": True,
            "minus_one_identity": "exact_v20m_negated_direction_mirror",
            "zero_identity": "exact_fixed_positive_envelope",
            "plus_one_identity": "exact_v20m_source_direction",
            "gain_formula": _FORMULA,
            "runtime_inputs": runtime["runtime_inputs"],
            "runtime_forbidden_inputs": runtime["runtime_forbidden_inputs"],
            "routing_control_flow": "none_validation_guards_only",
            "boundedness_certificate": "convex_plus_one_and_v20m_simplex_response",
            "serving_status": "analysis_only_no_serving_compression_or_speed_claim",
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        self._validate_compiled_binding()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("signed-continuum provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        runtime = self.compiled_provider.metadata()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "box_certificate": _box_certificate_payload(
                source_direction_sha256=self.source_direction_sha256,
                compiled_direction_sha256=str(
                    self._payload()["compiled_direction_sha256"]
                ),
                compiled_sign=self.compiled_direction_sign,
                signed_scalar=self.signed_scalar,
                compiled_mix=self.compiled_mix,
                radius=self.source_radius,
                shrink_mass=self.source_shrink_mass,
                polarity_bias=self.source_polarity_bias,
            ),
            "compiled_runtime_provider_metadata": runtime,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": self.incremental_prepared_float_scalar_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": runtime[
                "incremental_runtime_parameter_bytes_float64"
            ],
            "runtime_parameter_bytes_float64": runtime[
                "runtime_parameter_bytes_float64"
            ],
            "incremental_logical_macs_per_token_upper_bound": self.incremental_logical_macs_per_token_upper_bound,
            "logical_macs_per_token_upper_bound": self.logical_macs_per_token_upper_bound,
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

    def unbounded_direction(self, parent_modal: Tensor, coordinates: Tensor | None = None):
        return self.compiled_provider.unbounded_direction(parent_modal, coordinates)

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_logits(coordinates)

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_values(coordinates)

    def modal_correction(self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor) -> Tensor:
        return self.compiled_provider.modal_correction(prefix, realized_state)

    def correction(self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor) -> Tensor:
        return self.compiled_provider.correction(prefix, realized_state)


def build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    signed_scalar: object,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider:
    # The V20m direction is already the frozen, box-normalized fit output.  Keep
    # its exact float64 values here: normalizing it a second time can move a
    # component by one ULP and breaks the promised bit-identical +/-1 anchors.
    source_direction = _direction(
        direction,
        detach=True,
        normalize=False,
        label="signed-continuum source direction",
    )
    source_direction_sha = fisher_soft_polarity_signed_continuum_direction_sha256(
        source_direction
    )
    rate, mass, bias = _source_response(radius, shrink_mass, polarity_bias)
    signed = _strict_scalar(
        signed_scalar,
        label="signed-continuum scalar",
        lower=-1.0,
        upper=1.0,
    )
    sign, mix = fisher_soft_polarity_signed_continuum_materialized_parameters(signed)
    compiled_direction = source_direction if sign == 1 else -source_direction
    simplex = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base_provider,
        proposal_provider,
        direction=compiled_direction,
        radius=rate,
        shrink_mass=mass,
        polarity_bias=bias,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
    runtime = AutonomousCompleteH4FisherSoftPolaritySignedContinuumRuntimeProvider(
        compiled_simplex_provider=simplex,
        compiled_mix=mix,
    )
    return AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider(
        compiled_provider=runtime,
        source_direction_sha256=source_direction_sha,
        source_radius=rate,
        source_shrink_mass=mass,
        source_polarity_bias=bias,
        signed_scalar=signed,
    )
