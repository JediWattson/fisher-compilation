"""Continuous shrinkage of the bounded V20m simplex response.

V20n introduces one fit-time scalar ``lambda`` and no new runtime branch::

    m(z)        = tanh(radius * z)
    q_lambda(z) = (1 - lambda * shrink_mass * z**2) * m(z)
                  + lambda * polarity_bias * z**2

The source coefficients obey ``radius >= 0``, ``0 <= shrink_mass <= 1/2``,
``abs(polarity_bias) <= shrink_mass``, and ``0 <= lambda <= 1``.  The builder
materializes ``effective_shrink_mass = lambda * shrink_mass`` and
``effective_polarity_bias = lambda * polarity_bias`` into the existing V20m
simplex provider.  The builder returns an evidence wrapper whose
``runtime_provider`` is that materialized provider.  Only ``runtime_provider``
is installed for inference; the wrapper retains the source coefficients and
``lambda`` strictly as authenticated fit lineage.  The installed runtime thus
stores and executes exactly the same three response coefficients as V20m.

``lambda=0`` is bit-identical to the matched-linear V20m response and
``lambda=1`` is bit-identical to the source V20m response.  For an interior
``lambda`` and positive source shrink mass, the effective parameterization is
strictly between those endpoints.  All boundedness and pointwise-trust proofs
are inherited from V20m because ``abs(lambda*v) <= lambda*u <= 1/2``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from numbers import Real

from torch import Tensor

from .complete_h4_autonomous_residual import _require_sha256, _sha256
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
)
from .complete_h4_fisher_soft_polarity_signed_stack import (
    _canonical_evidence_tree,
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
    fisher_soft_polarity_simplex_response_box_certificate,
    fisher_soft_polarity_simplex_response_calibrator,
    fisher_soft_polarity_simplex_response_constant_tensor_sha256s,
    fisher_soft_polarity_simplex_response_direction_sha256,
    fisher_soft_polarity_simplex_response_gain,
    fisher_soft_polarity_simplex_response_modal_terms,
    fisher_soft_polarity_simplex_response_projection,
    fisher_soft_polarity_simplex_response_provider_artifact_sha256,
    fisher_soft_polarity_simplex_response_value,
    normalize_fisher_soft_polarity_simplex_response_direction,
    validate_fisher_soft_polarity_simplex_response_provider_evidence,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_LINEAGE_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_LAMBDA_MAX",
    "FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256",
    "FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_RUNTIME_FITTED_SCALAR_COUNT",
    "AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider",
    "FisherSoftPolaritySimplexShrinkageProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage",
    "fisher_soft_polarity_simplex_shrinkage_box_certificate",
    "fisher_soft_polarity_simplex_shrinkage_calibrator",
    "fisher_soft_polarity_simplex_shrinkage_constant_tensor_sha256s",
    "fisher_soft_polarity_simplex_shrinkage_direction_sha256",
    "fisher_soft_polarity_simplex_shrinkage_effective_parameters",
    "fisher_soft_polarity_simplex_shrinkage_gain",
    "fisher_soft_polarity_simplex_shrinkage_modal_terms",
    "fisher_soft_polarity_simplex_shrinkage_projection",
    "fisher_soft_polarity_simplex_shrinkage_provider_artifact_sha256",
    "fisher_soft_polarity_simplex_shrinkage_value",
    "normalize_fisher_soft_polarity_simplex_shrinkage_direction",
    "validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence",
]


_DIRECTION_COUNT = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT
_RUNTIME_FITTED_SCALAR_COUNT = (
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT
)
_FIT_LINEAGE_SCALAR_COUNT = _RUNTIME_FITTED_SCALAR_COUNT + 1
_RADIUS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX
_SHRINK_MASS_MAX = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX
_TRUST_FRACTION = FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION
_LAMBDA_MAX = 1.0

FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_LINEAGE_SCALAR_COUNT = (
    _FIT_LINEAGE_SCALAR_COUNT
)
FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_LAMBDA_MAX = _LAMBDA_MAX
FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_RUNTIME_FITTED_SCALAR_COUNT = (
    _RUNTIME_FITTED_SCALAR_COUNT
)

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-simplex-"
    b"shrinkage:provider:v20n\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-simplex-"
    b"shrinkage:protocol:v20n\0"
)
_FORMULA = (
    "q_lambda(z)=(1-lambda*source_u*z_squared)*tanh(source_r*z)"
    "+lambda*source_v*z_squared"
)
_PROTOCOL = {
    "schema": "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage.v20n",
    "formula": _FORMULA,
    "source_constraints": (
        "source_r_nonnegative_zero_le_source_u_le_one_half_"
        "abs_source_v_le_source_u_zero_le_lambda_le_one"
    ),
    "materialization": (
        "effective_r_equals_source_r_effective_u_equals_lambda_source_u_"
        "effective_v_equals_lambda_source_v"
    ),
    "lambda_zero_identity": "exact_matched_linear_v20m_response",
    "lambda_one_identity": "exact_source_v20m_simplex_response",
    "interior_identity": (
        "effective_parameters_strictly_between_endpoints_when_source_u_positive"
    ),
    "runtime_parameterization": "effective_r_u_v_only",
    "inference_executor": "materialized_compiled_provider_not_lineage_wrapper",
    "source_coefficients_and_lambda_runtime_scalars": 0,
    "activation_dependent_branch": False,
    "boundedness": "inherited_v20m_simplex_convex_certificate",
    "serving_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
}
FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256 = _sha256(
    _PROTOCOL_DOMAIN, _PROTOCOL
)


def _strict_scalar(
    value: object,
    *,
    label: str,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    if selected == 0.0 and math.copysign(1.0, selected) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if selected < lower or selected > upper:
        raise ValueError(f"{label} must be inside [{lower},{upper}]")
    return selected


def _source_parameters(
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> tuple[float, float, float, float]:
    rate = _strict_scalar(
        radius, label="simplex-shrinkage source radius", lower=0.0, upper=_RADIUS_MAX
    )
    mass = _strict_scalar(
        shrink_mass,
        label="simplex-shrinkage source shrink mass",
        lower=0.0,
        upper=_SHRINK_MASS_MAX,
    )
    bias = _strict_scalar(
        polarity_bias,
        label="simplex-shrinkage source polarity bias",
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )
    shrinkage = _strict_scalar(
        lambda_,
        label="simplex-shrinkage lambda",
        lower=0.0,
        upper=_LAMBDA_MAX,
    )
    if abs(bias) > mass:
        raise ValueError(
            "simplex-shrinkage source must satisfy abs(polarity_bias) <= "
            "shrink_mass"
        )
    return rate, mass, bias, shrinkage


def _materialized_product(lambda_: float, value: float) -> float:
    # Canonicalize the lambda-zero endpoint to positive zero even when the
    # source bias is negative.  This is required for bit-exact matched-linear
    # payloads and deterministic float hex receipts.
    if lambda_ == 0.0 or value == 0.0:
        return 0.0
    result = float(lambda_ * value)
    if not math.isfinite(result):
        raise RuntimeError("simplex-shrinkage effective coefficient became nonfinite")
    return result


def fisher_soft_polarity_simplex_shrinkage_effective_parameters(
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> tuple[float, float, float]:
    """Return the three coefficients materialized into the V20m runtime."""

    rate, mass, bias, shrinkage = _source_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    effective_mass = _materialized_product(shrinkage, mass)
    effective_bias = _materialized_product(shrinkage, bias)
    if abs(effective_bias) > effective_mass:
        raise RuntimeError("simplex-shrinkage effective simplex constraint failed")
    return rate, effective_mass, effective_bias


def fisher_soft_polarity_simplex_shrinkage_constant_tensor_sha256s() -> dict[str, str]:
    """V20n introduces no tensor beyond the inherited V20m constants."""

    return fisher_soft_polarity_simplex_response_constant_tensor_sha256s()


def normalize_fisher_soft_polarity_simplex_shrinkage_direction(
    direction: Tensor,
) -> Tensor:
    return normalize_fisher_soft_polarity_simplex_response_direction(direction)


def fisher_soft_polarity_simplex_shrinkage_direction_sha256(
    direction: Tensor,
) -> str:
    return fisher_soft_polarity_simplex_response_direction_sha256(direction)


def fisher_soft_polarity_simplex_shrinkage_projection(
    coordinates: Tensor,
    direction: Tensor,
) -> Tensor:
    return fisher_soft_polarity_simplex_response_projection(coordinates, direction)


def fisher_soft_polarity_simplex_shrinkage_calibrator(
    projection: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> Tensor:
    effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    return fisher_soft_polarity_simplex_response_calibrator(projection, *effective)


def fisher_soft_polarity_simplex_shrinkage_value(
    coordinates: Tensor,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> Tensor:
    effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    return fisher_soft_polarity_simplex_response_value(
        coordinates, direction, *effective
    )


def fisher_soft_polarity_simplex_shrinkage_gain(
    coordinates: Tensor,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> Tensor:
    effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    return fisher_soft_polarity_simplex_response_gain(
        coordinates, direction, *effective
    )


def fisher_soft_polarity_simplex_shrinkage_modal_terms(
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
    simplex_direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    return fisher_soft_polarity_simplex_response_modal_terms(
        parent_modal,
        coordinates,
        base_direction_left,
        base_direction_right,
        proposal_direction_left,
        proposal_direction_right,
        base_pedal_weight,
        base_pedal_bias,
        proposal_pedal_weight,
        proposal_pedal_bias,
        simplex_direction,
        *effective,
        trust_fraction=trust_fraction,
    )


def _shrinkage_certificate_payload(
    *,
    source_r: float,
    source_u: float,
    source_v: float,
    shrinkage: float,
    effective_r: float,
    effective_u: float,
    effective_v: float,
    inherited: object,
) -> dict[str, object]:
    inherited_certificate = _canonical_evidence_tree(
        inherited, label="simplex-shrinkage inherited V20m certificate"
    )
    if not isinstance(inherited_certificate, dict):
        raise ValueError("simplex-shrinkage inherited certificate must be a mapping")
    interior = 0.0 < shrinkage < 1.0
    return {
        "schema": (
            "fisher_graph.fisher_soft_polarity_simplex_shrinkage_box_"
            "certificate.v20n"
        ),
        "protocol_sha256": FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256,
        "formula": _FORMULA,
        "source_radius": source_r,
        "source_shrink_mass": source_u,
        "source_polarity_bias": source_v,
        "shrinkage_lambda": shrinkage,
        "effective_radius": effective_r,
        "effective_shrink_mass": effective_u,
        "effective_polarity_bias": effective_v,
        "source_radius_hex": source_r.hex(),
        "source_shrink_mass_hex": source_u.hex(),
        "source_polarity_bias_hex": source_v.hex(),
        "shrinkage_lambda_hex": shrinkage.hex(),
        "effective_radius_hex": effective_r.hex(),
        "effective_shrink_mass_hex": effective_u.hex(),
        "effective_polarity_bias_hex": effective_v.hex(),
        "source_radius_sha256": _response_scalar_sha256(source_r),
        "source_shrink_mass_sha256": _response_scalar_sha256(source_u),
        "source_polarity_bias_sha256": _response_scalar_sha256(source_v),
        "shrinkage_lambda_sha256": _response_scalar_sha256(shrinkage),
        "effective_radius_sha256": _response_scalar_sha256(effective_r),
        "effective_shrink_mass_sha256": _response_scalar_sha256(effective_u),
        "effective_polarity_bias_sha256": _response_scalar_sha256(effective_v),
        "lambda_in_closed_unit_interval": True,
        "source_simplex_constraints_satisfied": True,
        "effective_simplex_constraints_satisfied": True,
        "lambda_zero_exact_matched_linear": shrinkage == 0.0,
        "lambda_one_exact_source_simplex": shrinkage == 1.0,
        "interior_lambda": interior,
        "interior_effective_parameters_distinct_when_source_u_positive": (
            not interior
            or source_u == 0.0
            or (0.0 < effective_u < source_u)
        ),
        "source_coefficients_and_lambda_receipt_only": True,
        "runtime_materializes_effective_r_u_v_only": True,
        "shrinkage_lambda_runtime_float_scalar_count": 0,
        "activation_dependent_branch": False,
        "boundedness_proof": (
            "zero_le_lambda_u_le_one_half_and_abs_lambda_v_le_lambda_u_"
            "therefore_the_inherited_v20m_three_vertex_simplex_certificate_"
            "applies_without_change"
        ),
        "inherited_v20m_box_certificate": inherited_certificate,
    }


_BOX_CERTIFICATE_KEYS = frozenset(
    _shrinkage_certificate_payload(
        source_r=0.0,
        source_u=0.0,
        source_v=0.0,
        shrinkage=0.0,
        effective_r=0.0,
        effective_u=0.0,
        effective_v=0.0,
        inherited={},
    )
)


def fisher_soft_polarity_simplex_shrinkage_box_certificate(
    direction: Tensor,
    *,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
) -> dict[str, object]:
    source_r, source_u, source_v, shrinkage = _source_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    effective_r, effective_u, effective_v = (
        fisher_soft_polarity_simplex_shrinkage_effective_parameters(
            source_r, source_u, source_v, shrinkage
        )
    )
    inherited = fisher_soft_polarity_simplex_response_box_certificate(
        direction,
        radius=effective_r,
        shrink_mass=effective_u,
        polarity_bias=effective_v,
    )
    return _shrinkage_certificate_payload(
        source_r=source_r,
        source_u=source_u,
        source_v=source_v,
        shrinkage=shrinkage,
        effective_r=effective_r,
        effective_u=effective_u,
        effective_v=effective_v,
        inherited=inherited,
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
        "direction_sha256",
        "source_radius",
        "source_shrink_mass",
        "source_polarity_bias",
        "shrinkage_lambda",
        "effective_radius",
        "effective_shrink_mass",
        "effective_polarity_bias",
        "source_radius_hex",
        "source_shrink_mass_hex",
        "source_polarity_bias_hex",
        "shrinkage_lambda_hex",
        "effective_radius_hex",
        "effective_shrink_mass_hex",
        "effective_polarity_bias_hex",
        "source_radius_sha256",
        "source_shrink_mass_sha256",
        "source_polarity_bias_sha256",
        "shrinkage_lambda_sha256",
        "effective_radius_sha256",
        "effective_shrink_mass_sha256",
        "effective_polarity_bias_sha256",
        "compiled_simplex_response_provider_artifact_sha256",
        "compiled_simplex_response_provider_payload",
        "fit_lineage_float_scalar_count",
        "runtime_fitted_float_scalar_count",
        "source_coefficients_and_lambda_runtime_float_scalar_count",
        "lambda_materialized_into_effective_parameters",
        "lambda_zero_identity",
        "lambda_one_identity",
        "interior_parameter_identity",
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
        "compiled_simplex_response_provider_metadata",
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "shrinkage_projection_dot_macs_per_token",
        "shrinkage_calibrator_scalar_arithmetic_per_token",
        "shrinkage_elementwise_scalar_arithmetic_per_token",
        "shrinkage_nonlinear_scalar_ops_per_token",
        "shrinkage_runtime_extra_scalar_count",
        "runtime_state_float_scalars_per_sequence",
        "logical_macs_accounting_scope",
        "pointwise_trust_certificate_scope",
    }
)


def _payload_float(
    payload: dict[str, object],
    key: str,
    *,
    lower: float,
    upper: float,
) -> float:
    value = payload[key]
    if type(value) is not float:
        raise ValueError(f"simplex-shrinkage payload {key} must be a float")
    return _strict_scalar(value, label=f"simplex-shrinkage payload {key}", lower=lower, upper=upper)


def _validate_provider_payload(value: object) -> dict[str, object]:
    payload = _require_exact_keys(
        value,
        _PROVIDER_PAYLOAD_KEYS,
        label="simplex-shrinkage provider payload",
    )
    _canonical_evidence_tree(payload, label="simplex-shrinkage provider payload")
    _validate_evidence_sha_fields(payload, label="simplex-shrinkage provider payload")
    frozen = {
        "schema": (
            "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
            "simplex_shrinkage_provider.v20n"
        ),
        "site": _H4_SITE,
        "write_scope": "complete_h4_causal_support",
        "protocol_sha256": FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256,
        "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
        "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
        "source_coefficients_and_lambda_runtime_float_scalar_count": 0,
        "lambda_materialized_into_effective_parameters": True,
        "lambda_zero_identity": "exact_matched_linear_v20m_response",
        "lambda_one_identity": "exact_source_v20m_simplex_response",
        "interior_parameter_identity": (
            "strictly_between_endpoints_when_source_shrink_mass_positive"
        ),
        "gain_formula": _FORMULA,
        "routing_control_flow": "none_validation_guards_only",
        "boundedness_certificate": "inherited_v20m_three_vertex_simplex",
        "serving_status": "analysis_only_no_serving_compression_or_speed_claim",
    }
    for key, expected in frozen.items():
        _require_canonical_equal(
            payload[key], expected, label=f"simplex-shrinkage payload {key}"
        )

    source_r = _payload_float(payload, "source_radius", lower=0.0, upper=_RADIUS_MAX)
    source_u = _payload_float(
        payload, "source_shrink_mass", lower=0.0, upper=_SHRINK_MASS_MAX
    )
    source_v = _payload_float(
        payload,
        "source_polarity_bias",
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )
    shrinkage = _payload_float(
        payload, "shrinkage_lambda", lower=0.0, upper=1.0
    )
    expected_effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        source_r, source_u, source_v, shrinkage
    )
    effective = (
        _payload_float(payload, "effective_radius", lower=0.0, upper=_RADIUS_MAX),
        _payload_float(
            payload, "effective_shrink_mass", lower=0.0, upper=_SHRINK_MASS_MAX
        ),
        _payload_float(
            payload,
            "effective_polarity_bias",
            lower=-_SHRINK_MASS_MAX,
            upper=_SHRINK_MASS_MAX,
        ),
    )
    if effective != expected_effective:
        raise ValueError("simplex-shrinkage effective parameters differ")
    scalar_bindings = {
        "source_radius": source_r,
        "source_shrink_mass": source_u,
        "source_polarity_bias": source_v,
        "shrinkage_lambda": shrinkage,
        "effective_radius": effective[0],
        "effective_shrink_mass": effective[1],
        "effective_polarity_bias": effective[2],
    }
    for name, scalar in scalar_bindings.items():
        if payload[f"{name}_hex"] != scalar.hex():
            raise ValueError(f"simplex-shrinkage {name} hex differs")
        if payload[f"{name}_sha256"] != _response_scalar_sha256(scalar):
            raise ValueError(f"simplex-shrinkage {name} hash differs")

    compiled_payload = payload["compiled_simplex_response_provider_payload"]
    compiled_artifact = fisher_soft_polarity_simplex_response_provider_artifact_sha256(
        compiled_payload
    )
    if payload["compiled_simplex_response_provider_artifact_sha256"] != compiled_artifact:
        raise ValueError("simplex-shrinkage compiled provider artifact differs")
    compiled = _canonical_evidence_tree(
        compiled_payload, label="simplex-shrinkage compiled provider payload"
    )
    assert isinstance(compiled, dict)
    bindings = {
        "site": compiled["site"],
        "write_scope": compiled["write_scope"],
        "bridge_binding_sha256": compiled["bridge_binding_sha256"],
        "parent_provider_artifact_sha256": compiled[
            "parent_provider_artifact_sha256"
        ],
        "base_provider_artifact_sha256": compiled[
            "base_provider_artifact_sha256"
        ],
        "proposal_provider_artifact_sha256": compiled[
            "proposal_provider_artifact_sha256"
        ],
        "start_provider_artifact_sha256": compiled[
            "start_provider_artifact_sha256"
        ],
        "transfer_protocol_sha256": compiled["transfer_protocol_sha256"],
        "transfer_evidence_sha256": compiled["transfer_evidence_sha256"],
        "direction_sha256": compiled["direction_sha256"],
        "effective_radius": compiled["radius"],
        "effective_shrink_mass": compiled["shrink_mass"],
        "effective_polarity_bias": compiled["polarity_bias"],
        "runtime_inputs": compiled["runtime_inputs"],
        "runtime_forbidden_inputs": compiled["runtime_forbidden_inputs"],
    }
    for key, expected in bindings.items():
        _require_canonical_equal(
            payload[key], expected, label=f"simplex-shrinkage compiled binding {key}"
        )
    canonical = _canonical_evidence_tree(
        payload, label="simplex-shrinkage provider payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_simplex_shrinkage_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_PROVIDER_DOMAIN, _validate_provider_payload(payload))


@dataclass(frozen=True, slots=True)
class FisherSoftPolaritySimplexShrinkageProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence(
    payload: object,
    metadata: object,
) -> FisherSoftPolaritySimplexShrinkageProviderEvidence:
    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="simplex-shrinkage provider metadata",
    )
    _canonical_evidence_tree(selected, label="simplex-shrinkage provider metadata")
    _validate_evidence_sha_fields(
        selected, label="simplex-shrinkage provider metadata"
    )
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"simplex-shrinkage metadata payload field {key}",
        )
    artifact = fisher_soft_polarity_simplex_shrinkage_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact:
        raise ValueError("simplex-shrinkage provider artifact hash differs")

    compiled_evidence = validate_fisher_soft_polarity_simplex_response_provider_evidence(
        canonical_payload["compiled_simplex_response_provider_payload"],
        selected["compiled_simplex_response_provider_metadata"],
    )
    if (
        canonical_payload["compiled_simplex_response_provider_artifact_sha256"]
        != compiled_evidence.artifact_sha256
    ):
        raise ValueError("simplex-shrinkage nested V20m evidence differs")

    expected_certificate = _shrinkage_certificate_payload(
        source_r=float(canonical_payload["source_radius"]),
        source_u=float(canonical_payload["source_shrink_mass"]),
        source_v=float(canonical_payload["source_polarity_bias"]),
        shrinkage=float(canonical_payload["shrinkage_lambda"]),
        effective_r=float(canonical_payload["effective_radius"]),
        effective_u=float(canonical_payload["effective_shrink_mass"]),
        effective_v=float(canonical_payload["effective_polarity_bias"]),
        inherited=compiled_evidence.metadata["box_certificate"],
    )
    supplied_certificate = _require_exact_keys(
        selected["box_certificate"],
        _BOX_CERTIFICATE_KEYS,
        label="simplex-shrinkage box certificate",
    )
    for key, expected in expected_certificate.items():
        _require_canonical_equal(
            supplied_certificate[key],
            expected,
            label=f"simplex-shrinkage box certificate {key}",
        )
    _validate_evidence_sha_fields(
        supplied_certificate, label="simplex-shrinkage box certificate"
    )

    delegated_integer_fields = (
        "rank",
        "conditional_rank",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_scalars_per_sequence",
    )
    for key in delegated_integer_fields:
        current = _strict_evidence_integer(selected, key)
        inherited = _strict_evidence_integer(compiled_evidence.metadata, key)
        if current != inherited:
            raise ValueError(f"simplex-shrinkage delegated accounting {key} differs")
    frozen_metadata = {
        "shrinkage_projection_dot_macs_per_token": compiled_evidence.metadata[
            "simplex_response_projection_dot_macs_per_token"
        ],
        "shrinkage_calibrator_scalar_arithmetic_per_token": compiled_evidence.metadata[
            "simplex_response_calibrator_scalar_arithmetic_per_token"
        ],
        "shrinkage_elementwise_scalar_arithmetic_per_token": compiled_evidence.metadata[
            "simplex_response_elementwise_scalar_arithmetic_per_token"
        ],
        "shrinkage_nonlinear_scalar_ops_per_token": compiled_evidence.metadata[
            "simplex_response_nonlinear_scalar_ops_per_token"
        ],
        "shrinkage_runtime_extra_scalar_count": 0,
        "logical_macs_accounting_scope": compiled_evidence.metadata[
            "logical_macs_accounting_scope"
        ],
        "pointwise_trust_certificate_scope": compiled_evidence.metadata[
            "pointwise_trust_certificate_scope"
        ],
    }
    for key, expected in frozen_metadata.items():
        _require_canonical_equal(
            selected[key], expected, label=f"simplex-shrinkage metadata {key}"
        )
    for key in (
        "shrinkage_projection_dot_macs_per_token",
        "shrinkage_calibrator_scalar_arithmetic_per_token",
        "shrinkage_elementwise_scalar_arithmetic_per_token",
        "shrinkage_nonlinear_scalar_ops_per_token",
        "shrinkage_runtime_extra_scalar_count",
    ):
        _strict_evidence_integer(selected, key)
    canonical_metadata = _canonical_evidence_tree(
        selected, label="simplex-shrinkage provider metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolaritySimplexShrinkageProviderEvidence(
        payload=canonical_payload,
        metadata=canonical_metadata,
        artifact_sha256=artifact,
    )


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider(
    Gemma3L3L4CorrectionProvider
):
    """Fit-lineage evidence around one materialized V20m runtime provider."""

    compiled_provider: AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
    source_radius: float
    source_shrink_mass: float
    source_polarity_bias: float
    shrinkage_lambda: float
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.compiled_provider,
            AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
        ):
            raise TypeError("simplex-shrinkage needs a materialized V20m provider")
        source = _source_parameters(
            self.source_radius,
            self.source_shrink_mass,
            self.source_polarity_bias,
            self.shrinkage_lambda,
        )
        for name, value in zip(
            (
                "source_radius",
                "source_shrink_mass",
                "source_polarity_bias",
                "shrinkage_lambda",
            ),
            source,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        self._validate_compiled_binding()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="simplex-shrinkage provider"
            ) != computed:
                raise ValueError("simplex-shrinkage provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def base_provider(self) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
        return self.compiled_provider.base_provider

    @property
    def runtime_provider(
        self,
    ) -> AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider:
        """Return the only provider authorized for inference execution."""

        return self.compiled_provider

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
    def transfer_protocol_sha256(self) -> str:
        return self.compiled_provider.transfer_protocol_sha256

    @property
    def transfer_evidence_sha256(self) -> str:
        return self.compiled_provider.transfer_evidence_sha256

    @property
    def direction(self) -> Tensor:
        return self.compiled_provider.direction

    @property
    def radius(self) -> float:
        return self.source_radius

    @property
    def shrink_mass(self) -> float:
        return self.source_shrink_mass

    @property
    def polarity_bias(self) -> float:
        return self.source_polarity_bias

    @property
    def lambda_(self) -> float:
        return self.shrinkage_lambda

    @property
    def effective_radius(self) -> float:
        return self.compiled_provider.radius

    @property
    def effective_shrink_mass(self) -> float:
        return self.compiled_provider.shrink_mass

    @property
    def effective_polarity_bias(self) -> float:
        return self.compiled_provider.polarity_bias

    @property
    def trust_fraction(self) -> float:
        return self.compiled_provider.trust_fraction

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
        effective = fisher_soft_polarity_simplex_shrinkage_effective_parameters(
            self.source_radius,
            self.source_shrink_mass,
            self.source_polarity_bias,
            self.shrinkage_lambda,
        )
        observed = (
            float(self.compiled_provider.radius),
            float(self.compiled_provider.shrink_mass),
            float(self.compiled_provider.polarity_bias),
        )
        if observed != effective:
            raise RuntimeError("simplex-shrinkage materialized V20m coefficients drifted")

    def _payload(self) -> dict[str, object]:
        compiled_payload = self.compiled_provider.artifact_payload()
        source = (
            self.source_radius,
            self.source_shrink_mass,
            self.source_polarity_bias,
            self.shrinkage_lambda,
        )
        effective = (
            self.effective_radius,
            self.effective_shrink_mass,
            self.effective_polarity_bias,
        )
        scalar_bindings = {
            "source_radius": source[0],
            "source_shrink_mass": source[1],
            "source_polarity_bias": source[2],
            "shrinkage_lambda": source[3],
            "effective_radius": effective[0],
            "effective_shrink_mass": effective[1],
            "effective_polarity_bias": effective[2],
        }
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "simplex_shrinkage_provider.v20n"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "protocol_sha256": FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": self.proposal_provider.artifact_sha256,
            "start_provider_artifact_sha256": self.base_provider.start_provider_artifact_sha256,
            "transfer_protocol_sha256": self.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.transfer_evidence_sha256,
            "direction_sha256": fisher_soft_polarity_simplex_shrinkage_direction_sha256(
                self.direction
            ),
            **scalar_bindings,
            **{f"{key}_hex": value.hex() for key, value in scalar_bindings.items()},
            **{
                f"{key}_sha256": _response_scalar_sha256(value)
                for key, value in scalar_bindings.items()
            },
            "compiled_simplex_response_provider_artifact_sha256": (
                self.compiled_provider.artifact_sha256
            ),
            "compiled_simplex_response_provider_payload": compiled_payload,
            "fit_lineage_float_scalar_count": _FIT_LINEAGE_SCALAR_COUNT,
            "runtime_fitted_float_scalar_count": _RUNTIME_FITTED_SCALAR_COUNT,
            "source_coefficients_and_lambda_runtime_float_scalar_count": 0,
            "lambda_materialized_into_effective_parameters": True,
            "lambda_zero_identity": "exact_matched_linear_v20m_response",
            "lambda_one_identity": "exact_source_v20m_simplex_response",
            "interior_parameter_identity": (
                "strictly_between_endpoints_when_source_shrink_mass_positive"
            ),
            "gain_formula": _FORMULA,
            "runtime_inputs": compiled_payload["runtime_inputs"],
            "runtime_forbidden_inputs": compiled_payload["runtime_forbidden_inputs"],
            "routing_control_flow": "none_validation_guards_only",
            "boundedness_certificate": "inherited_v20m_three_vertex_simplex",
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
            raise RuntimeError("simplex-shrinkage provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        compiled = self.compiled_provider.metadata()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "box_certificate": fisher_soft_polarity_simplex_shrinkage_box_certificate(
                self.direction,
                radius=self.source_radius,
                shrink_mass=self.source_shrink_mass,
                polarity_bias=self.source_polarity_bias,
                lambda_=self.shrinkage_lambda,
            ),
            "compiled_simplex_response_provider_metadata": compiled,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": compiled[
                "incremental_runtime_parameter_bytes_float64"
            ],
            "runtime_parameter_bytes_float64": compiled[
                "runtime_parameter_bytes_float64"
            ],
            "incremental_logical_macs_per_token_upper_bound": (
                self.incremental_logical_macs_per_token_upper_bound
            ),
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "shrinkage_projection_dot_macs_per_token": compiled[
                "simplex_response_projection_dot_macs_per_token"
            ],
            "shrinkage_calibrator_scalar_arithmetic_per_token": compiled[
                "simplex_response_calibrator_scalar_arithmetic_per_token"
            ],
            "shrinkage_elementwise_scalar_arithmetic_per_token": compiled[
                "simplex_response_elementwise_scalar_arithmetic_per_token"
            ],
            "shrinkage_nonlinear_scalar_ops_per_token": compiled[
                "simplex_response_nonlinear_scalar_ops_per_token"
            ],
            "shrinkage_runtime_extra_scalar_count": 0,
            "runtime_state_float_scalars_per_sequence": compiled[
                "runtime_state_float_scalars_per_sequence"
            ],
            "logical_macs_accounting_scope": compiled[
                "logical_macs_accounting_scope"
            ],
            "pointwise_trust_certificate_scope": compiled[
                "pointwise_trust_certificate_scope"
            ],
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.compiled_provider.bounded_coordinates(parent_modal)

    def response_gain(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        return self.compiled_provider.response_gain(coordinates)

    def terms_from_parent(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.validate_integrity()
        return self.compiled_provider.terms_from_parent(parent_modal, coordinates)

    def unbounded_direction(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> Tensor:
        return self.compiled_provider.unbounded_direction(parent_modal, coordinates)

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_logits(coordinates)

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return self.compiled_provider.pedal_values(coordinates)

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        return self.compiled_provider.modal_correction(prefix, realized_state)

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        return self.compiled_provider.correction(prefix, realized_state)


def build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    radius: object,
    shrink_mass: object,
    polarity_bias: object,
    lambda_: object,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider:
    source_r, source_u, source_v, shrinkage = _source_parameters(
        radius, shrink_mass, polarity_bias, lambda_
    )
    effective_r, effective_u, effective_v = (
        fisher_soft_polarity_simplex_shrinkage_effective_parameters(
            source_r, source_u, source_v, shrinkage
        )
    )
    compiled = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base_provider,
        proposal_provider,
        direction=direction,
        radius=effective_r,
        shrink_mass=effective_u,
        polarity_bias=effective_v,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
    return AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider(
        compiled_provider=compiled,
        source_radius=source_r,
        source_shrink_mass=source_u,
        source_polarity_bias=source_v,
        shrinkage_lambda=shrinkage,
    )
