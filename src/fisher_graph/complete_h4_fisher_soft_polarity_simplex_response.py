"""Bounded simplex-response calibration for complete-H4 Fisher transfer.

For a box-normalized bilinear projection ``z`` this V20m provider uses

``m(z) = tanh(radius*z)``

``q(z) = (1-shrink_mass*z**2)*m(z) + polarity_bias*z**2``

with ``radius >= 0``, ``0 <= shrink_mass <= 1/2``, and
``abs(polarity_bias) <= shrink_mass``.  Equivalently,

``q = w0*m + w_plus*(+1) + w_minus*(-1)``

where ``w0=1-u*z**2``, ``w_plus=(u+v)*z**2/2``, and
``w_minus=(u-v)*z**2/2``.  These weights are nonnegative and sum to one,
which proves ``abs(q) <= 1``.  The endpoint gain is the inherited bounded
envelope ``asinh(9*c2)/asinh(9)`` times ``q``.

The formula is continuous and has no activation-dependent branch.  It obeys
``q(0)=0`` and is bit-identical to the linear response when ``u=v=0``.
Oddness is certified only when ``v=0``; no general monotonicity is claimed.
The provider is analysis-only and makes no serving, compression, or speed
claim.
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
    _validate_endpoint_pair,
    fisher_continuous_factor_direction,
    fisher_continuous_pedal_logit,
)
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    _finite_runtime_tensor,
    fisher_finite_joint_direction_features,
)
from .complete_h4_fisher_soft_polarity import (
    fisher_soft_polarity_constant_tensor_sha256s as _soft_constant_hashes,
    fisher_soft_polarity_envelope as _soft_envelope,
    fisher_soft_polarity_features as _soft_features,
)
from .complete_h4_fisher_soft_polarity_signed_stack import (
    _BOX_CORNER_FEATURES,
    _DIRECTION_COUNT,
    _DIRECTION_INPUT_MAX_ABS,
    _NORMALIZATION_TOLERANCE,
    _RADIUS_MAX,
    _TRUST_FRACTION,
    _canonical_evidence_tree,
    _direction,
    _normalized_direction,
    _radius_scalar,
    _require_canonical_equal,
    _require_exact_keys,
    _response_scalar_sha256,
    _strict_evidence_integer,
    _validate_evidence_sha_fields,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX",
    "FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX",
    "FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION",
    "AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider",
    "FisherSoftPolaritySimplexResponseProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_simplex_response",
    "fisher_soft_polarity_simplex_response_box_certificate",
    "fisher_soft_polarity_simplex_response_calibrator",
    "fisher_soft_polarity_simplex_response_constant_tensor_sha256s",
    "fisher_soft_polarity_simplex_response_direction_sha256",
    "fisher_soft_polarity_simplex_response_gain",
    "fisher_soft_polarity_simplex_response_modal_terms",
    "fisher_soft_polarity_simplex_response_projection",
    "fisher_soft_polarity_simplex_response_provider_artifact_sha256",
    "fisher_soft_polarity_simplex_response_value",
    "normalize_fisher_soft_polarity_simplex_response_direction",
    "validate_fisher_soft_polarity_simplex_response_provider_evidence",
]


_RESPONSE_SCALAR_COUNT = 3
_FITTED_SCALAR_COUNT = _DIRECTION_COUNT + _RESPONSE_SCALAR_COUNT
_SHRINK_MASS_MAX = 0.5

FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT = _FITTED_SCALAR_COUNT
FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX = _RADIUS_MAX
FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX = _SHRINK_MASS_MAX
FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_TRUST_FRACTION = _TRUST_FRACTION

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-simplex-response:"
    b"provider:v1\0"
)
_CONSTANT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-simplex-response:"
    b"constant:v1\0"
)
_DIRECTION_COUNT_TENSOR = torch.tensor((_DIRECTION_COUNT,), dtype=torch.int64)
_RADIUS_MAX_TENSOR = torch.tensor((_RADIUS_MAX,), dtype=torch.float64)
_SHRINK_MASS_MAX_TENSOR = torch.tensor((_SHRINK_MASS_MAX,), dtype=torch.float64)
_CONSTANT_TENSOR_SHA256S = {
    "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
    "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
    "radius_max": _tensor_sha256(_RADIUS_MAX_TENSOR),
    "shrink_mass_max": _tensor_sha256(_SHRINK_MASS_MAX_TENSOR),
}
_GAIN_FORMULA = (
    "asinh_9c2_over_asinh_9_times_one_minus_shrink_mass_z_squared_"
    "times_tanh_radius_z_plus_polarity_bias_z_squared_for_box_normalized_"
    "bilinear_z"
)
_CONSTANT_BUNDLE_SHA256 = _sha256(
    _CONSTANT_DOMAIN,
    {
        "formula": _GAIN_FORMULA,
        "direction_count": _DIRECTION_COUNT,
        "response_scalar_count": _RESPONSE_SCALAR_COUNT,
        "fitted_scalar_count": _FITTED_SCALAR_COUNT,
        "direction_input_max_abs": _DIRECTION_INPUT_MAX_ABS,
        "radius_max": _RADIUS_MAX,
        "shrink_mass_max": _SHRINK_MASS_MAX,
        "trust_fraction": _TRUST_FRACTION,
        "normalization_tolerance": _NORMALIZATION_TOLERANCE,
        "constant_tensor_sha256s": _CONSTANT_TENSOR_SHA256S,
        "inherited_soft_polarity_constant_tensor_sha256s": (
            _soft_constant_hashes()
        ),
    },
)


def _validate_constant_tensors() -> None:
    observed = {
        "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
        "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
        "radius_max": _tensor_sha256(_RADIUS_MAX_TENSOR),
        "shrink_mass_max": _tensor_sha256(_SHRINK_MASS_MAX_TENSOR),
    }
    if observed != _CONSTANT_TENSOR_SHA256S:
        raise RuntimeError("simplex-response frozen formula tensors drifted")


def fisher_soft_polarity_simplex_response_constant_tensor_sha256s() -> dict[str, str]:
    return dict(_CONSTANT_TENSOR_SHA256S)


def normalize_fisher_soft_polarity_simplex_response_direction(
    direction: Tensor,
) -> Tensor:
    _validate_constant_tensors()
    return _direction(
        direction,
        detach=False,
        normalize=True,
        label="simplex-response direction",
    )


def fisher_soft_polarity_simplex_response_direction_sha256(direction: Tensor) -> str:
    _validate_constant_tensors()
    selected = _normalized_direction(
        direction,
        detach=True,
        label="simplex-response hashed direction",
    )
    return _tensor_sha256(selected)


def _bounded_scalar(
    value: object,
    *,
    detach: bool,
    label: str,
    lower: float,
    upper: float,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(f"{label} must be a finite floating scalar")
        selected = (
            value.detach().to(dtype=torch.float64).clone()
            if detach
            else _finite_runtime_tensor(value, label=label, ndim=0)
        )
        if not bool(torch.isfinite(selected)):
            raise ValueError(f"{label} must be finite")
    elif isinstance(value, Real) and not isinstance(value, bool):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{label} must be finite")
        selected = torch.tensor(scalar, dtype=torch.float64)
    else:
        raise ValueError(f"{label} must be a finite floating scalar")
    scalar = float(selected.detach())
    if scalar == 0.0 and math.copysign(1.0, scalar) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if scalar < lower or scalar > upper:
        raise ValueError(f"{label} must be inside [{lower},{upper}]")
    return selected.contiguous()


def _shrink_mass_scalar(value: object, *, detach: bool, label: str) -> Tensor:
    return _bounded_scalar(
        value,
        detach=detach,
        label=label,
        lower=0.0,
        upper=_SHRINK_MASS_MAX,
    )


def _polarity_bias_scalar(value: object, *, detach: bool, label: str) -> Tensor:
    return _bounded_scalar(
        value,
        detach=detach,
        label=label,
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )


def _response_pair(
    shrink_mass: object,
    polarity_bias: object,
    *,
    detach: bool,
    label: str,
) -> tuple[Tensor, Tensor]:
    mass = _shrink_mass_scalar(
        shrink_mass, detach=detach, label=f"{label} shrink mass"
    )
    bias = _polarity_bias_scalar(
        polarity_bias, detach=detach, label=f"{label} polarity bias"
    )
    if abs(float(bias.detach())) > float(mass.detach()):
        raise ValueError(f"{label} must satisfy abs(polarity_bias) <= shrink_mass")
    return mass, bias


def fisher_soft_polarity_simplex_response_projection(
    coordinates: Tensor,
    direction: Tensor,
) -> Tensor:
    features = _soft_features(coordinates)
    normalized = _normalized_direction(
        direction,
        detach=False,
        label="simplex-response direction",
    ).to(device=features.device, dtype=features.dtype)
    result = features @ normalized
    if (
        not bool(torch.isfinite(result).all())
        or float(result.detach().abs().max())
        > 1.0 + _NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError("simplex-response projection violated its box bound")
    return result.contiguous()


def fisher_soft_polarity_simplex_response_calibrator(
    projection: Tensor,
    radius: float | Tensor,
    shrink_mass: float | Tensor,
    polarity_bias: float | Tensor,
) -> Tensor:
    z = _finite_runtime_tensor(projection, label="simplex projection", ndim=1)
    if float(z.detach().abs().max()) > 1.0 + _NORMALIZATION_TOLERANCE:
        raise ValueError("simplex-response projection must remain inside [-1,1]")
    rate = _radius_scalar(
        radius, detach=False, label="simplex-response radius"
    ).to(device=z.device, dtype=z.dtype)
    mass, bias = _response_pair(
        shrink_mass,
        polarity_bias,
        detach=False,
        label="simplex-response",
    )
    mass = mass.to(device=z.device, dtype=z.dtype)
    bias = bias.to(device=z.device, dtype=z.dtype)
    z_squared = z.square()
    base = torch.tanh(rate * z)
    result = (1.0 - mass * z_squared) * base + bias * z_squared
    if (
        not bool(torch.isfinite(result).all())
        or bool((result.abs() > 1.0 + _NORMALIZATION_TOLERANCE).any())
    ):
        raise RuntimeError("simplex-response calibrator violated its bound")
    return result.contiguous()


def fisher_soft_polarity_simplex_response_value(
    coordinates: Tensor,
    direction: Tensor,
    radius: float | Tensor,
    shrink_mass: float | Tensor,
    polarity_bias: float | Tensor,
) -> Tensor:
    projection = fisher_soft_polarity_simplex_response_projection(
        coordinates, direction
    )
    return fisher_soft_polarity_simplex_response_calibrator(
        projection, radius, shrink_mass, polarity_bias
    )


def fisher_soft_polarity_simplex_response_gain(
    coordinates: Tensor,
    direction: Tensor,
    radius: float | Tensor,
    shrink_mass: float | Tensor,
    polarity_bias: float | Tensor,
) -> Tensor:
    envelope = _soft_envelope(coordinates)
    value = fisher_soft_polarity_simplex_response_value(
        coordinates, direction, radius, shrink_mass, polarity_bias
    )
    result = envelope * value
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("simplex-response gain violated its analytic bound")
    return result.contiguous()


_PROVIDER_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "site",
        "write_scope",
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
        "radius_sha256",
        "shrink_mass_sha256",
        "polarity_bias_sha256",
        "direction_float64_scalar_count",
        "response_float64_scalar_count",
        "fitted_float64_scalar_count",
        "radius_max",
        "shrink_mass_max",
        "constant_tensor_sha256s",
        "constant_bundle_sha256",
        "inherited_soft_polarity_constant_tensor_sha256s",
        "trust_fraction",
        "runtime_inputs",
        "runtime_forbidden_inputs",
        "gain_formula",
        "direction_normalization",
        "calibrator_certificate",
        "routing_control_flow",
        "global_gain_certificate",
        "factor_semantics",
        "bounded_direction_semantics",
        "experimental_serving_status",
        "prepared_payload_deduplication_semantics",
    }
)
_PROVIDER_METADATA_EXTRA_KEYS = frozenset(
    {
        "artifact_sha256",
        "rank",
        "conditional_rank",
        "box_certificate",
        "incremental_prepared_float_scalar_count",
        "prepared_float_scalar_count",
        "simplex_response_fitted_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "simplex_response_projection_dot_macs_per_token",
        "simplex_response_calibrator_scalar_arithmetic_per_token",
        "simplex_response_elementwise_scalar_arithmetic_per_token",
        "simplex_response_nonlinear_scalar_ops_per_token",
        "simplex_response_elementwise_scope",
        "simplex_response_nonlinear_scope",
        "logical_macs_accounting_scope",
        "runtime_state_float_scalars_per_sequence",
        "pointwise_trust_certificate_scope",
    }
)
_BOX_CERTIFICATE_KEYS = frozenset(
    {
        "schema",
        "coordinate_box",
        "direction_count",
        "direction_normalization",
        "direction_box_corner_logits",
        "projection_max_abs",
        "radius",
        "shrink_mass",
        "polarity_bias",
        "radius_nonnegative",
        "shrink_mass_in_closed_interval",
        "polarity_bias_within_shrink_mass",
        "base_response_max_abs_upper_bound",
        "simplex_weight_formulas",
        "simplex_weights_nonnegative",
        "simplex_weights_sum_to_one",
        "base_weight_min_lower_bound",
        "calibrator_center_value",
        "calibrator_odd_when_polarity_bias_zero",
        "calibrator_oddness_claim_when_polarity_bias_nonzero",
        "calibrator_monotonicity_claim",
        "calibrator_max_abs_upper_bound",
        "envelope_max_abs",
        "gain_max_abs",
        "pointwise_trust_fraction",
        "proof",
        "numerical_totality_proof",
        "direction_sha256",
        "radius_sha256",
        "shrink_mass_sha256",
        "polarity_bias_sha256",
        "constant_bundle_sha256",
    }
)

_FROZEN_PROVIDER_PAYLOAD = {
    "schema": (
        "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
        "simplex_response_provider.v1"
    ),
    "site": _H4_SITE,
    "write_scope": "complete_h4_causal_support",
    "direction_float64_scalar_count": _DIRECTION_COUNT,
    "response_float64_scalar_count": _RESPONSE_SCALAR_COUNT,
    "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
    "radius_max": _RADIUS_MAX,
    "shrink_mass_max": _SHRINK_MASS_MAX,
    "constant_tensor_sha256s": _CONSTANT_TENSOR_SHA256S,
    "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    "inherited_soft_polarity_constant_tensor_sha256s": _soft_constant_hashes(),
    "trust_fraction": _TRUST_FRACTION,
    "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
    "runtime_forbidden_inputs": (
        "native_h4",
        "targets",
        "logits",
        "gradients",
        "family_ids",
        "prompt_text",
        "token_ids",
        "fit_examples",
        "optimizer_state",
    ),
    "gain_formula": _GAIN_FORMULA,
    "direction_normalization": "max_absolute_bilinear_box_corner_logit",
    "calibrator_certificate": (
        "bounded_three_vertex_simplex_odd_only_at_zero_polarity_bias_no_"
        "monotonicity_claim"
    ),
    "routing_control_flow": "none_validation_guards_only",
    "global_gain_certificate": "absolute_gain_at_most_one",
    "factor_semantics": (
        "FL0R0_plus_g_FdLR0_plus_FL0dR_plus_g_squared_FdLdR"
    ),
    "bounded_direction_semantics": (
        "pointwise_q_norm_at_most_0.25_parent_modal_norm"
    ),
    "experimental_serving_status": (
        "analysis_only_retains_two_endpoints_and_executes_extra_endpoint_terms"
    ),
    "prepared_payload_deduplication_semantics": (
        "common_parent_artifact_once_both_complete_endpoint_increments_retained"
    ),
}


def fisher_soft_polarity_simplex_response_box_certificate(
    direction: Tensor,
    *,
    radius: float,
    shrink_mass: float,
    polarity_bias: float,
) -> dict[str, object]:
    _validate_constant_tensors()
    normalized = _normalized_direction(
        direction,
        detach=True,
        label="simplex-response certificate direction",
    )
    rate = _radius_scalar(
        radius, detach=True, label="simplex-response certificate radius"
    )
    mass, bias = _response_pair(
        shrink_mass,
        polarity_bias,
        detach=True,
        label="simplex-response certificate",
    )
    rate_value = float(rate)
    mass_value = float(mass)
    bias_value = float(bias)
    corners = _BOX_CORNER_FEATURES @ normalized
    return {
        "schema": (
            "fisher_graph.fisher_soft_polarity_simplex_response_box_"
            "certificate.v1"
        ),
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "direction_box_corner_logits": tuple(float(item) for item in corners),
        "projection_max_abs": 1.0,
        "radius": rate_value,
        "shrink_mass": mass_value,
        "polarity_bias": bias_value,
        "radius_nonnegative": True,
        "shrink_mass_in_closed_interval": (0.0, 0.5),
        "polarity_bias_within_shrink_mass": True,
        "base_response_max_abs_upper_bound": math.tanh(rate_value),
        "simplex_weight_formulas": (
            "w0=1-u*z_squared;w_plus=(u+v)*z_squared/2;"
            "w_minus=(u-v)*z_squared/2"
        ),
        "simplex_weights_nonnegative": True,
        "simplex_weights_sum_to_one": True,
        "base_weight_min_lower_bound": 1.0 - mass_value,
        "calibrator_center_value": 0.0,
        "calibrator_odd_when_polarity_bias_zero": True,
        "calibrator_oddness_claim_when_polarity_bias_nonzero": "none",
        "calibrator_monotonicity_claim": "none",
        "calibrator_max_abs_upper_bound": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_projection_"
            "at_most_one_and_constraints_make_w0_w_plus_w_minus_nonnegative_"
            "and_sum_to_one_so_q_is_a_convex_combination_of_tanh_radius_z_"
            "plus_one_and_minus_one_then_envelope_times_q_has_absolute_value_"
            "at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_radius_at_most_float64_max_"
            "over_eight_shrink_mass_at_most_one_half_and_bias_abs_at_most_"
            "shrink_mass_keep_all_post_tanh_products_and_sums_bounded"
        ),
        "direction_sha256": fisher_soft_polarity_simplex_response_direction_sha256(
            normalized
        ),
        "radius_sha256": _response_scalar_sha256(rate_value),
        "shrink_mass_sha256": _response_scalar_sha256(mass_value),
        "polarity_bias_sha256": _response_scalar_sha256(bias_value),
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }


@dataclass(frozen=True, slots=True)
class FisherSoftPolaritySimplexResponseProviderEvidence:
    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def _strict_float(
    value: object,
    *,
    label: str,
    lower: float,
    upper: float,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if value < lower or value > upper:
        raise ValueError(f"{label} must be inside [{lower},{upper}]")
    return value


def _validate_provider_payload(payload: object) -> dict[str, object]:
    _validate_constant_tensors()
    selected = _require_exact_keys(
        payload, _PROVIDER_PAYLOAD_KEYS, label="simplex-response provider payload"
    )
    _canonical_evidence_tree(selected, label="simplex-response provider payload")
    _validate_evidence_sha_fields(
        selected, label="simplex-response provider payload"
    )
    for key, expected in _FROZEN_PROVIDER_PAYLOAD.items():
        _require_canonical_equal(
            selected[key],
            expected,
            label=f"simplex-response provider payload {key}",
        )
    rate = _strict_float(
        selected["radius"], label="payload radius", lower=0.0, upper=_RADIUS_MAX
    )
    mass = _strict_float(
        selected["shrink_mass"],
        label="payload shrink_mass",
        lower=0.0,
        upper=_SHRINK_MASS_MAX,
    )
    bias = _strict_float(
        selected["polarity_bias"],
        label="payload polarity_bias",
        lower=-_SHRINK_MASS_MAX,
        upper=_SHRINK_MASS_MAX,
    )
    if abs(bias) > mass:
        raise ValueError("payload must satisfy abs(polarity_bias) <= shrink_mass")
    for key, value in (
        ("radius_sha256", rate),
        ("shrink_mass_sha256", mass),
        ("polarity_bias_sha256", bias),
    ):
        if selected[key] != _response_scalar_sha256(value):
            raise ValueError(f"simplex-response provider {key} differs")
    canonical = _canonical_evidence_tree(
        selected, label="simplex-response provider payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_simplex_response_provider_artifact_sha256(
    payload: object,
) -> str:
    return _sha256(_PROVIDER_DOMAIN, _validate_provider_payload(payload))


def _validate_box_certificate(
    certificate: object,
    *,
    payload: dict[str, object],
) -> None:
    selected = _require_exact_keys(
        certificate,
        _BOX_CERTIFICATE_KEYS,
        label="simplex-response provider box certificate",
    )
    _canonical_evidence_tree(
        selected, label="simplex-response provider box certificate"
    )
    _validate_evidence_sha_fields(
        selected, label="simplex-response provider box certificate"
    )
    rate = payload["radius"]
    mass = payload["shrink_mass"]
    bias = payload["polarity_bias"]
    assert type(rate) is float and type(mass) is float and type(bias) is float
    expected = {
        "schema": (
            "fisher_graph.fisher_soft_polarity_simplex_response_box_"
            "certificate.v1"
        ),
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "projection_max_abs": 1.0,
        "radius": rate,
        "shrink_mass": mass,
        "polarity_bias": bias,
        "radius_nonnegative": True,
        "shrink_mass_in_closed_interval": (0.0, 0.5),
        "polarity_bias_within_shrink_mass": True,
        "base_response_max_abs_upper_bound": math.tanh(rate),
        "simplex_weight_formulas": (
            "w0=1-u*z_squared;w_plus=(u+v)*z_squared/2;"
            "w_minus=(u-v)*z_squared/2"
        ),
        "simplex_weights_nonnegative": True,
        "simplex_weights_sum_to_one": True,
        "base_weight_min_lower_bound": 1.0 - mass,
        "calibrator_center_value": 0.0,
        "calibrator_odd_when_polarity_bias_zero": True,
        "calibrator_oddness_claim_when_polarity_bias_nonzero": "none",
        "calibrator_monotonicity_claim": "none",
        "calibrator_max_abs_upper_bound": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_projection_"
            "at_most_one_and_constraints_make_w0_w_plus_w_minus_nonnegative_"
            "and_sum_to_one_so_q_is_a_convex_combination_of_tanh_radius_z_"
            "plus_one_and_minus_one_then_envelope_times_q_has_absolute_value_"
            "at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_radius_at_most_float64_max_"
            "over_eight_shrink_mass_at_most_one_half_and_bias_abs_at_most_"
            "shrink_mass_keep_all_post_tanh_products_and_sums_bounded"
        ),
        "direction_sha256": payload["direction_sha256"],
        "radius_sha256": payload["radius_sha256"],
        "shrink_mass_sha256": payload["shrink_mass_sha256"],
        "polarity_bias_sha256": payload["polarity_bias_sha256"],
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }
    for key, expected_value in expected.items():
        _require_canonical_equal(
            selected[key],
            expected_value,
            label=f"simplex-response provider box certificate {key}",
        )
    corners = selected["direction_box_corner_logits"]
    if type(corners) not in (tuple, list) or len(corners) != 4:
        raise ValueError("simplex-response corner logits must contain four floats")
    if any(type(item) is not float or not math.isfinite(item) for item in corners):
        raise ValueError("simplex-response corner logits must be finite floats")
    if abs(max(abs(item) for item in corners) - 1.0) > _NORMALIZATION_TOLERANCE:
        raise ValueError("simplex-response corner normalization differs")


def validate_fisher_soft_polarity_simplex_response_provider_evidence(
    payload: object,
    metadata: object,
) -> FisherSoftPolaritySimplexResponseProviderEvidence:
    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="simplex-response provider metadata",
    )
    _canonical_evidence_tree(selected, label="simplex-response provider metadata")
    _validate_evidence_sha_fields(
        selected, label="simplex-response provider metadata"
    )
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"simplex-response provider metadata payload field {key}",
        )
    artifact_sha256 = fisher_soft_polarity_simplex_response_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact_sha256:
        raise ValueError("simplex-response provider artifact hash differs")
    _validate_box_certificate(
        selected["box_certificate"], payload=canonical_payload
    )

    rank = _strict_evidence_integer(selected, "rank", minimum=1)
    conditional_rank = _strict_evidence_integer(
        selected, "conditional_rank", minimum=1
    )
    incremental_prepared = _strict_evidence_integer(
        selected,
        "incremental_prepared_float_scalar_count",
        minimum=_FITTED_SCALAR_COUNT,
    )
    prepared = _strict_evidence_integer(
        selected, "prepared_float_scalar_count", minimum=incremental_prepared
    )
    incremental_bytes = _strict_evidence_integer(
        selected, "incremental_runtime_parameter_bytes_float64"
    )
    total_bytes = _strict_evidence_integer(
        selected, "runtime_parameter_bytes_float64"
    )
    incremental_macs = _strict_evidence_integer(
        selected, "incremental_logical_macs_per_token_upper_bound"
    )
    logical_macs = _strict_evidence_integer(
        selected,
        "logical_macs_per_token_upper_bound",
        minimum=incremental_macs,
    )
    endpoint_increment = 2 * rank + 4 * rank * conditional_rank + 8
    expected_prepared = 2 * endpoint_increment + _FITTED_SCALAR_COUNT
    expected_macs = 2 * rank + _DIRECTION_COUNT + 10 * rank * conditional_rank + 6
    if incremental_prepared != expected_prepared:
        raise ValueError("simplex-response incremental prepared formula differs")
    if incremental_bytes != incremental_prepared * 8:
        raise ValueError("simplex-response incremental parameter bytes differ")
    if total_bytes != prepared * 8:
        raise ValueError("simplex-response total parameter bytes differ")
    if incremental_macs != expected_macs:
        raise ValueError("simplex-response incremental MAC formula differs")
    if logical_macs < incremental_macs:
        raise ValueError("simplex-response total MAC count differs")

    frozen_metadata = {
        "simplex_response_fitted_float_scalar_count": _FITTED_SCALAR_COUNT,
        "simplex_response_projection_dot_macs_per_token": _DIRECTION_COUNT,
        "simplex_response_calibrator_scalar_arithmetic_per_token": 7,
        "simplex_response_elementwise_scalar_arithmetic_per_token": 10,
        "simplex_response_nonlinear_scalar_ops_per_token": 2,
        "simplex_response_elementwise_scope": (
            "c1c2_product_kappa_scale_envelope_asinh_normalization_radius_"
            "scale_z_square_shrink_scale_one_minus_weight_base_scale_polarity_"
            "bias_scale_simplex_sum_and_envelope_product"
        ),
        "simplex_response_nonlinear_scope": "one_asinh_and_one_tanh",
        "logical_macs_accounting_scope": (
            "experimental_dense_upper_bound_includes_both_endpoint_factor_"
            "paths_four_term_projection_and_two_pedal_logits_elementwise_"
            "simplex_response_and_nonlinear_operations_reported_separately"
        ),
        "runtime_state_float_scalars_per_sequence": 0,
        "pointwise_trust_certificate_scope": (
            "emitted_modal_amplitude_not_full_nonlinear_jacobian_or_lipschitz"
        ),
    }
    for key, expected in frozen_metadata.items():
        _require_canonical_equal(
            selected[key],
            expected,
            label=f"simplex-response provider metadata {key}",
        )
    for key in (
        "simplex_response_fitted_float_scalar_count",
        "simplex_response_projection_dot_macs_per_token",
        "simplex_response_calibrator_scalar_arithmetic_per_token",
        "simplex_response_elementwise_scalar_arithmetic_per_token",
        "simplex_response_nonlinear_scalar_ops_per_token",
        "runtime_state_float_scalars_per_sequence",
    ):
        _strict_evidence_integer(selected, key)
    canonical_metadata = _canonical_evidence_tree(
        selected, label="simplex-response provider metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolaritySimplexResponseProviderEvidence(
        payload=canonical_payload,
        metadata=canonical_metadata,
        artifact_sha256=artifact_sha256,
    )


def fisher_soft_polarity_simplex_response_modal_terms(
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
    radius: float | Tensor,
    shrink_mass: float | Tensor,
    polarity_bias: float | Tensor,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    parent = _finite_runtime_tensor(
        parent_modal, label="simplex-response parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="simplex-response coordinates", ndim=2
    ).to(parent.device)
    if (
        bounded_coordinates.shape != (parent.shape[0], 2)
        or bool((bounded_coordinates.abs() > 1.0).any())
    ):
        raise ValueError(
            "simplex-response coordinates must match parent rows inside [-1,1]"
        )
    if trust_fraction != _TRUST_FRACTION:
        raise ValueError("simplex-response trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_simplex_response_gain(
        bounded_coordinates,
        simplex_direction,
        radius,
        shrink_mass,
        polarity_bias,
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
        raise RuntimeError("simplex-response modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider(
    Gemma3L3L4CorrectionProvider
):
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    direction: Tensor
    radius: float
    shrink_mass: float
    polarity_bias: float
    transfer_protocol_sha256: str
    transfer_evidence_sha256: str
    trust_fraction: float = _TRUST_FRACTION
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_constant_tensors()
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        object.__setattr__(
            self,
            "direction",
            _direction(
                self.direction,
                detach=True,
                normalize=False,
                label="simplex-response direction",
            ),
        )
        object.__setattr__(
            self,
            "radius",
            float(
                _radius_scalar(
                    self.radius, detach=True, label="simplex-response radius"
                )
            ),
        )
        mass, bias = _response_pair(
            self.shrink_mass,
            self.polarity_bias,
            detach=True,
            label="simplex-response",
        )
        object.__setattr__(self, "shrink_mass", float(mass))
        object.__setattr__(self, "polarity_bias", float(bias))
        for name in ("transfer_protocol_sha256", "transfer_evidence_sha256"):
            _require_sha256(getattr(self, name), label=name)
        if self.trust_fraction != _TRUST_FRACTION:
            raise ValueError("simplex-response trust fraction is frozen at 0.25")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="simplex-response provider"
            ) != computed:
                raise ValueError("simplex-response provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def parent_provider(self):
        return self.base_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.base_provider.bridge_binding_sha256

    @property
    def rank(self) -> int:
        return self.base_provider.rank

    @property
    def conditional_rank(self) -> int:
        return self.base_provider.conditional_rank

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        return int(
            self.base_provider.incremental_prepared_float_scalar_count
            + self.proposal_provider.incremental_prepared_float_scalar_count
            + _FITTED_SCALAR_COUNT
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_provider.prepared_float_scalar_count
            + self.incremental_prepared_float_scalar_count
        )

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        return int(
            2 * self.rank
            + _DIRECTION_COUNT
            + 10 * self.rank * self.conditional_rank
            + 6
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_provider.logical_macs_per_token_upper_bound
            + self.incremental_logical_macs_per_token_upper_bound
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
                "simplex_response_provider.v1"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": (
                self.proposal_provider.artifact_sha256
            ),
            "start_provider_artifact_sha256": (
                self.base_provider.start_provider_artifact_sha256
            ),
            "transfer_protocol_sha256": self.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.transfer_evidence_sha256,
            "direction_sha256": fisher_soft_polarity_simplex_response_direction_sha256(
                self.direction
            ),
            "radius": self.radius,
            "shrink_mass": self.shrink_mass,
            "polarity_bias": self.polarity_bias,
            "radius_sha256": _response_scalar_sha256(self.radius),
            "shrink_mass_sha256": _response_scalar_sha256(self.shrink_mass),
            "polarity_bias_sha256": _response_scalar_sha256(self.polarity_bias),
            "direction_float64_scalar_count": _DIRECTION_COUNT,
            "response_float64_scalar_count": _RESPONSE_SCALAR_COUNT,
            "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
            "radius_max": _RADIUS_MAX,
            "shrink_mass_max": _SHRINK_MASS_MAX,
            "constant_tensor_sha256s": dict(_CONSTANT_TENSOR_SHA256S),
            "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
            "inherited_soft_polarity_constant_tensor_sha256s": (
                _soft_constant_hashes()
            ),
            "trust_fraction": self.trust_fraction,
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": (
                "native_h4",
                "targets",
                "logits",
                "gradients",
                "family_ids",
                "prompt_text",
                "token_ids",
                "fit_examples",
                "optimizer_state",
            ),
            "gain_formula": _GAIN_FORMULA,
            "direction_normalization": "max_absolute_bilinear_box_corner_logit",
            "calibrator_certificate": (
                "bounded_three_vertex_simplex_odd_only_at_zero_polarity_bias_"
                "no_monotonicity_claim"
            ),
            "routing_control_flow": "none_validation_guards_only",
            "global_gain_certificate": "absolute_gain_at_most_one",
            "factor_semantics": (
                "FL0R0_plus_g_FdLR0_plus_FL0dR_plus_g_squared_FdLdR"
            ),
            "bounded_direction_semantics": (
                "pointwise_q_norm_at_most_0.25_parent_modal_norm"
            ),
            "experimental_serving_status": (
                "analysis_only_retains_two_endpoints_and_executes_extra_endpoint_terms"
            ),
            "prepared_payload_deduplication_semantics": (
                "common_parent_artifact_once_both_complete_endpoint_increments_retained"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        _validate_constant_tensors()
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        _direction(
            self.direction,
            detach=False,
            normalize=False,
            label="simplex-response stored direction",
        )
        _radius_scalar(
            self.radius, detach=False, label="simplex-response stored radius"
        )
        _response_pair(
            self.shrink_mass,
            self.polarity_bias,
            detach=False,
            label="simplex-response stored",
        )
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("simplex-response provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "box_certificate": fisher_soft_polarity_simplex_response_box_certificate(
                self.direction,
                radius=self.radius,
                shrink_mass=self.shrink_mass,
                polarity_bias=self.polarity_bias,
            ),
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "simplex_response_fitted_float_scalar_count": _FITTED_SCALAR_COUNT,
            "incremental_runtime_parameter_bytes_float64": (
                self.incremental_prepared_float_scalar_count * 8
            ),
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "incremental_logical_macs_per_token_upper_bound": (
                self.incremental_logical_macs_per_token_upper_bound
            ),
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "simplex_response_projection_dot_macs_per_token": _DIRECTION_COUNT,
            "simplex_response_calibrator_scalar_arithmetic_per_token": 7,
            "simplex_response_elementwise_scalar_arithmetic_per_token": 10,
            "simplex_response_nonlinear_scalar_ops_per_token": 2,
            "simplex_response_elementwise_scope": (
                "c1c2_product_kappa_scale_envelope_asinh_normalization_radius_"
                "scale_z_square_shrink_scale_one_minus_weight_base_scale_"
                "polarity_bias_scale_simplex_sum_and_envelope_product"
            ),
            "simplex_response_nonlinear_scope": "one_asinh_and_one_tanh",
            "logical_macs_accounting_scope": (
                "experimental_dense_upper_bound_includes_both_endpoint_factor_"
                "paths_four_term_projection_and_two_pedal_logits_elementwise_"
                "simplex_response_and_nonlinear_operations_reported_separately"
            ),
            "runtime_state_float_scalars_per_sequence": 0,
            "pointwise_trust_certificate_scope": (
                "emitted_modal_amplitude_not_full_nonlinear_jacobian_or_lipschitz"
            ),
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.base_provider.bounded_coordinates(parent_modal)

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
            raise ValueError("simplex-response coordinates differ")
        leading = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_simplex_response_gain(
            flat,
            self.direction.to(device=flat.device, dtype=flat.dtype),
            self.radius,
            self.shrink_mass,
            self.polarity_bias,
        )
        return gain.reshape(leading).contiguous()

    def terms_from_parent(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("simplex-response parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("simplex-response parent/coordinate geometry differs")
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, 2)
        gain, direction, bounded_direction, logit, pedal, delta = (
            fisher_soft_polarity_simplex_response_modal_terms(
                flat_parent,
                flat_coordinates,
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
                trust_fraction=self.trust_fraction,
            )
        )
        leading = original_shape[:-1]
        return (
            gain.reshape(leading).contiguous(),
            direction.reshape(original_shape).contiguous(),
            bounded_direction.reshape(original_shape).contiguous(),
            logit.reshape(leading).contiguous(),
            pedal.reshape(leading).contiguous(),
            delta.reshape(original_shape).contiguous(),
        )

    def unbounded_direction(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> Tensor:
        return self.terms_from_parent(parent_modal, coordinates)[1]

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        gain = self.response_gain(coordinates)
        leading = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2).to(dtype=torch.float64)
        logit = fisher_continuous_pedal_logit(
            flat,
            self.base_provider.pedal_weight.to(flat.device),
            self.base_provider.pedal_bias.to(flat.device),
            self.proposal_provider.pedal_weight.to(flat.device),
            self.proposal_provider.pedal_bias.to(flat.device),
            gain.reshape(-1),
        )
        return logit.reshape(leading).contiguous()

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return torch.sigmoid(self.pedal_logits(coordinates)).contiguous()

    def _modal_terms(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        parent = self.parent_provider.modal_correction(prefix, realized_state)
        coordinates = self.bounded_coordinates(parent)
        gain, direction, bounded, logit, pedal, delta = self.terms_from_parent(
            parent, coordinates
        )
        return parent, coordinates, gain, direction, bounded, logit, pedal, delta

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
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
            raise RuntimeError("simplex-response provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("simplex-response modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("simplex-response modal correction escaped support")
        self.validate_integrity()
        prefix.validate_integrity()
        return modal.contiguous()

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        return self.parent_provider.decode_modal(
            prefix,
            self.modal_correction(prefix, realized_state),
            like=realized_state,
        )


def build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    radius: float,
    shrink_mass: float,
    polarity_bias: float,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider:
    return AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider(
        base_provider=base_provider,
        proposal_provider=proposal_provider,
        direction=direction,
        radius=radius,
        shrink_mass=shrink_mass,
        polarity_bias=polarity_bias,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
