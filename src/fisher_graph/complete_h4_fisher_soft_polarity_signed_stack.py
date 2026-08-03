"""Bounded signed-stack calibration for complete-H4 soft-polarity transfer.

The provider retains the authenticated complete-H4 endpoint pair, factor and
pedal interpolation, and pointwise trust certificate.  A box-normalized
bilinear Fisher projection ``z`` is mapped continuously by

``m(z) = tanh(radius*z)``

``w(z) = abs(signed_mix)*z**2``

``q(z) = (1-w(z))*m(z) + signed_mix*z**2``

and the endpoint gain is ``asinh(9*c2)/asinh(9) * q(z)``.  Since
``abs(signed_mix) <= 0.5`` and ``abs(z) <= 1``, ``w`` is in ``[0, 0.5]``.
For nonzero ``signed_mix``, the second term is ``w`` times the fixed target
``sign(signed_mix)``; therefore ``q`` is a convex combination of two values in
``[-1, 1]``.  At zero mix the formula is exactly the bounded linear response.

The runtime formula has no activation-dependent branch.  Nonzero signed mix
intentionally breaks oddness and is not certified monotone: it stacks both
projection signs toward one fixed polarity.  This is an analysis provider and
makes no serving, compression, or speed claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from numbers import Real

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    _float_tensor,
    _require_sha256,
    _sha256,
    _tensor_sha256,
)
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
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
    _finite_runtime_tensor,
    fisher_finite_joint_direction_features,
)
from .complete_h4_fisher_soft_polarity import (
    fisher_soft_polarity_constant_tensor_sha256s as _soft_constant_hashes,
    fisher_soft_polarity_envelope as _soft_envelope,
    fisher_soft_polarity_features as _soft_features,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_SOFT_POLARITY_SIGNED_STACK_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_SIGNED_STACK_FITTED_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_SIGNED_STACK_RADIUS_MAX",
    "FISHER_SOFT_POLARITY_SIGNED_STACK_SIGNED_MIX_MAX_ABS",
    "FISHER_SOFT_POLARITY_SIGNED_STACK_TRUST_FRACTION",
    "AutonomousCompleteH4FisherSoftPolaritySignedStackProvider",
    "FisherSoftPolaritySignedStackProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_signed_stack",
    "fisher_soft_polarity_signed_stack_box_certificate",
    "fisher_soft_polarity_signed_stack_calibrator",
    "fisher_soft_polarity_signed_stack_constant_tensor_sha256s",
    "fisher_soft_polarity_signed_stack_direction_sha256",
    "fisher_soft_polarity_signed_stack_gain",
    "fisher_soft_polarity_signed_stack_modal_terms",
    "fisher_soft_polarity_signed_stack_projection",
    "fisher_soft_polarity_signed_stack_provider_artifact_sha256",
    "fisher_soft_polarity_signed_stack_value",
    "normalize_fisher_soft_polarity_signed_stack_direction",
    "validate_fisher_soft_polarity_signed_stack_provider_evidence",
]


_DIRECTION_COUNT = 4
_RESPONSE_SCALAR_COUNT = 2
_FITTED_SCALAR_COUNT = _DIRECTION_COUNT + _RESPONSE_SCALAR_COUNT
_TRUST_FRACTION = FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION
_DIRECTION_INPUT_MAX_ABS = float(torch.finfo(torch.float64).max / 8.0)
_RADIUS_MAX = float(torch.finfo(torch.float64).max / 8.0)
_SIGNED_MIX_MAX_ABS = 0.5
_NORMALIZATION_TOLERANCE = float(128.0 * torch.finfo(torch.float64).eps)

FISHER_SOFT_POLARITY_SIGNED_STACK_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_SIGNED_STACK_FITTED_SCALAR_COUNT = _FITTED_SCALAR_COUNT
FISHER_SOFT_POLARITY_SIGNED_STACK_RADIUS_MAX = _RADIUS_MAX
FISHER_SOFT_POLARITY_SIGNED_STACK_SIGNED_MIX_MAX_ABS = _SIGNED_MIX_MAX_ABS
FISHER_SOFT_POLARITY_SIGNED_STACK_TRUST_FRACTION = _TRUST_FRACTION

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-stack:"
    b"provider:v1\0"
)
_CONSTANT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-signed-stack:"
    b"constant:v1\0"
)
_BOX_CORNER_FEATURES = torch.tensor(
    (
        (1.0, -1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0, 1.0),
    ),
    dtype=torch.float64,
)
_DIRECTION_COUNT_TENSOR = torch.tensor((_DIRECTION_COUNT,), dtype=torch.int64)
_RADIUS_MAX_TENSOR = torch.tensor((_RADIUS_MAX,), dtype=torch.float64)
_SIGNED_MIX_MAX_ABS_TENSOR = torch.tensor(
    (_SIGNED_MIX_MAX_ABS,), dtype=torch.float64
)
_CONSTANT_TENSOR_SHA256S = {
    "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
    "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
    "radius_max": _tensor_sha256(_RADIUS_MAX_TENSOR),
    "signed_mix_max_abs": _tensor_sha256(_SIGNED_MIX_MAX_ABS_TENSOR),
}
_GAIN_FORMULA = (
    "asinh_9c2_over_asinh_9_times_one_minus_abs_signed_mix_z_squared_"
    "times_tanh_radius_z_plus_signed_mix_z_squared_for_box_normalized_"
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
        "signed_mix_max_abs": _SIGNED_MIX_MAX_ABS,
        "trust_fraction": _TRUST_FRACTION,
        "normalization_tolerance": _NORMALIZATION_TOLERANCE,
        "constant_tensor_sha256s": _CONSTANT_TENSOR_SHA256S,
        "inherited_soft_polarity_constant_tensor_sha256s": (
            _soft_constant_hashes()
        ),
    },
)

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
        "signed_mix",
        "radius_sha256",
        "signed_mix_sha256",
        "direction_float64_scalar_count",
        "response_float64_scalar_count",
        "fitted_float64_scalar_count",
        "radius_max",
        "signed_mix_max_abs",
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
        "signed_stack_polarity_fitted_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "signed_stack_projection_dot_macs_per_token",
        "signed_stack_calibrator_scalar_arithmetic_per_token",
        "signed_stack_elementwise_scalar_arithmetic_per_token",
        "signed_stack_nonlinear_scalar_ops_per_token",
        "signed_stack_elementwise_scope",
        "signed_stack_nonlinear_scope",
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
        "signed_mix",
        "radius_nonnegative",
        "signed_mix_in_closed_interval",
        "signed_mix_max_abs",
        "base_response_max_abs_upper_bound",
        "stack_weight_formula",
        "stack_weight_nonnegative",
        "stack_weight_max_upper_bound",
        "stack_target_semantics",
        "calibrator_center_value",
        "calibrator_odd_when_signed_mix_zero",
        "calibrator_oddness_claim_when_signed_mix_nonzero",
        "calibrator_monotonicity_claim_when_signed_mix_nonzero",
        "calibrator_max_abs_upper_bound",
        "envelope_max_abs",
        "gain_max_abs",
        "pointwise_trust_fraction",
        "proof",
        "numerical_totality_proof",
        "direction_sha256",
        "radius_sha256",
        "signed_mix_sha256",
        "constant_bundle_sha256",
    }
)

_FROZEN_PROVIDER_PAYLOAD = {
    "schema": (
        "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
        "signed_stack_provider.v1"
    ),
    "site": _H4_SITE,
    "write_scope": "complete_h4_causal_support",
    "direction_float64_scalar_count": _DIRECTION_COUNT,
    "response_float64_scalar_count": _RESPONSE_SCALAR_COUNT,
    "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
    "radius_max": _RADIUS_MAX,
    "signed_mix_max_abs": _SIGNED_MIX_MAX_ABS,
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
        "bounded_convex_signed_stack_no_nonzero_mix_oddness_or_monotonicity_claim"
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


@dataclass(frozen=True, slots=True)
class FisherSoftPolaritySignedStackProviderEvidence:
    """Canonical model-free evidence for one signed-stack provider."""

    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def fisher_soft_polarity_signed_stack_constant_tensor_sha256s() -> dict[str, str]:
    """Return hashes for every signed-stack-specific frozen tensor."""

    return dict(_CONSTANT_TENSOR_SHA256S)


def _validate_constant_tensors() -> None:
    observed = {
        "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
        "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
        "radius_max": _tensor_sha256(_RADIUS_MAX_TENSOR),
        "signed_mix_max_abs": _tensor_sha256(_SIGNED_MIX_MAX_ABS_TENSOR),
    }
    if observed != _CONSTANT_TENSOR_SHA256S:
        raise RuntimeError("signed-stack-polarity frozen formula tensors drifted")


def _direction(value: object, *, detach: bool, normalize: bool, label: str) -> Tensor:
    selected = (
        _float_tensor(value, label=label, ndim=1)
        if detach
        else _finite_runtime_tensor(value, label=label, ndim=1)
    )
    if selected.shape != (_DIRECTION_COUNT,):
        raise ValueError(f"{label} must contain exactly four float64 values")
    if bool((selected.abs() > _DIRECTION_INPUT_MAX_ABS).any()):
        raise ValueError(f"{label} exceeds the certified numerical magnitude limit")
    corners = _BOX_CORNER_FEATURES.to(
        device=selected.device, dtype=selected.dtype
    ) @ selected
    scale = corners.abs().max()
    if not bool(torch.isfinite(corners).all()) or float(scale.detach()) <= 0.0:
        raise ValueError(f"{label} must have a finite nonzero box projection")
    result = selected / scale if normalize else selected
    normalized_corners = _BOX_CORNER_FEATURES.to(
        device=result.device, dtype=result.dtype
    ) @ result
    normalized_scale = float(normalized_corners.detach().abs().max())
    if normalize:
        if abs(normalized_scale - 1.0) > _NORMALIZATION_TOLERANCE:
            raise RuntimeError(f"{label} normalization failed its corner certificate")
    elif abs(normalized_scale - 1.0) > _NORMALIZATION_TOLERANCE:
        raise ValueError(f"{label} must be box-corner normalized")
    return result.contiguous()


def normalize_fisher_soft_polarity_signed_stack_direction(
    direction: Tensor,
) -> Tensor:
    """Normalize a direction by its maximum absolute box-corner projection."""

    _validate_constant_tensors()
    return _direction(
        direction,
        detach=False,
        normalize=True,
        label="signed-stack-polarity direction",
    )


def _normalized_direction(value: object, *, detach: bool, label: str) -> Tensor:
    return _direction(value, detach=detach, normalize=False, label=label)


def fisher_soft_polarity_signed_stack_direction_sha256(direction: Tensor) -> str:
    """Return the exact hash for one normalized float64 direction."""

    _validate_constant_tensors()
    selected = _normalized_direction(
        direction,
        detach=True,
        label="signed-stack-polarity hashed direction",
    )
    return _tensor_sha256(selected)


def _radius_scalar(value: object, *, detach: bool, label: str) -> Tensor:
    if isinstance(value, Tensor):
        selected = (
            _float_tensor(value, label=label, ndim=0)
            if detach
            else _finite_runtime_tensor(value, label=label, ndim=0)
        )
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
    if scalar < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    if scalar > _RADIUS_MAX:
        raise ValueError(f"{label} exceeds the certified numerical magnitude limit")
    return selected.contiguous()


def _signed_mix_scalar(value: object, *, detach: bool, label: str) -> Tensor:
    if isinstance(value, Tensor):
        selected = (
            _float_tensor(value, label=label, ndim=0)
            if detach
            else _finite_runtime_tensor(value, label=label, ndim=0)
        )
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
    if abs(scalar) > _SIGNED_MIX_MAX_ABS:
        raise ValueError(f"{label} must be inside [-0.5,0.5]")
    return selected.contiguous()


def _response_scalar_sha256(value: float) -> str:
    return _tensor_sha256(torch.tensor((value,), dtype=torch.float64))


def fisher_soft_polarity_signed_stack_projection(
    coordinates: Tensor, direction: Tensor
) -> Tensor:
    """Return the certified box-normalized bilinear projection ``z``."""

    features = _soft_features(coordinates)
    normalized = _normalized_direction(
        direction,
        detach=False,
        label="signed-stack-polarity direction",
    ).to(device=features.device, dtype=features.dtype)
    result = features @ normalized
    if (
        not bool(torch.isfinite(result).all())
        or float(result.detach().abs().max())
        > 1.0 + _NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError("signed-stack-polarity projection violated its box bound")
    return result.contiguous()


def fisher_soft_polarity_signed_stack_calibrator(
    projection: Tensor,
    radius: float | Tensor,
    signed_mix: float | Tensor,
) -> Tensor:
    """Return the continuous bounded signed stack ``(1-w)m + signed_mix*z²``."""

    z = _finite_runtime_tensor(
        projection, label="signed-stack-polarity projection", ndim=1
    )
    if float(z.detach().abs().max()) > 1.0 + _NORMALIZATION_TOLERANCE:
        raise ValueError("signed-stack-polarity projection must remain inside [-1,1]")
    rate = _radius_scalar(
        radius,
        detach=False,
        label="signed-stack-polarity response radius",
    ).to(device=z.device, dtype=z.dtype)
    mix = _signed_mix_scalar(
        signed_mix,
        detach=False,
        label="signed-stack-polarity signed mix",
    ).to(device=z.device, dtype=z.dtype)
    z_squared = z.square()
    base_response = torch.tanh(rate * z)
    weight = torch.abs(mix) * z_squared
    result = (1.0 - weight) * base_response + mix * z_squared
    if (
        not bool(torch.isfinite(result).all())
        or bool((result.abs() > 1.0 + _NORMALIZATION_TOLERANCE).any())
    ):
        raise RuntimeError("signed-stack-polarity calibrator violated its bound")
    return result.contiguous()


def fisher_soft_polarity_signed_stack_value(
    coordinates: Tensor,
    direction: Tensor,
    radius: float | Tensor,
    signed_mix: float | Tensor,
) -> Tensor:
    """Return the signed-stack value for bounded coordinates."""

    projection = fisher_soft_polarity_signed_stack_projection(coordinates, direction)
    return fisher_soft_polarity_signed_stack_calibrator(
        projection, radius, signed_mix
    )


def fisher_soft_polarity_signed_stack_gain(
    coordinates: Tensor,
    direction: Tensor,
    radius: float | Tensor,
    signed_mix: float | Tensor,
) -> Tensor:
    """Return the analytically bounded signed-stack endpoint gain."""

    envelope = _soft_envelope(coordinates)
    value = fisher_soft_polarity_signed_stack_value(
        coordinates, direction, radius, signed_mix
    )
    result = envelope * value
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("signed-stack-polarity gain violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_signed_stack_box_certificate(
    direction: Tensor,
    *,
    radius: float,
    signed_mix: float,
) -> dict[str, object]:
    """Return the analytic box, convex-stack, and trust prerequisites."""

    _validate_constant_tensors()
    normalized = _normalized_direction(
        direction,
        detach=True,
        label="signed-stack-polarity certificate direction",
    )
    rate = _radius_scalar(
        radius,
        detach=True,
        label="signed-stack-polarity certificate response radius",
    )
    mix = _signed_mix_scalar(
        signed_mix,
        detach=True,
        label="signed-stack-polarity certificate signed mix",
    )
    corners = _BOX_CORNER_FEATURES @ normalized
    rate_value = float(rate)
    mix_value = float(mix)
    return {
        "schema": (
            "fisher_graph.fisher_soft_polarity_signed_stack_box_certificate.v1"
        ),
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "direction_box_corner_logits": tuple(float(item) for item in corners),
        "projection_max_abs": 1.0,
        "radius": rate_value,
        "signed_mix": mix_value,
        "radius_nonnegative": True,
        "signed_mix_in_closed_interval": (-0.5, 0.5),
        "signed_mix_max_abs": _SIGNED_MIX_MAX_ABS,
        "base_response_max_abs_upper_bound": math.tanh(rate_value),
        "stack_weight_formula": "abs_signed_mix_times_z_squared",
        "stack_weight_nonnegative": True,
        "stack_weight_max_upper_bound": abs(mix_value),
        "stack_target_semantics": (
            "signed_mix_z_squared_equals_weight_times_fixed_sign_target_for_"
            "nonzero_signed_mix_and_zero_when_signed_mix_is_zero"
        ),
        "calibrator_center_value": 0.0,
        "calibrator_odd_when_signed_mix_zero": True,
        "calibrator_oddness_claim_when_signed_mix_nonzero": "none",
        "calibrator_monotonicity_claim_when_signed_mix_nonzero": "none",
        "calibrator_max_abs_upper_bound": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_projection_"
            "at_most_one_weight_is_abs_signed_mix_times_z_squared_in_zero_to_"
            "one_half_and_q_is_a_convex_combination_of_tanh_radius_z_and_the_"
            "fixed_signed_polarity_target_so_abs_q_at_most_one_then_envelope_"
            "times_q_has_absolute_value_at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_radius_at_most_float64_max_"
            "over_eight_signed_mix_abs_at_most_one_half_and_all_post_tanh_"
            "products_and_sums_remain_bounded"
        ),
        "direction_sha256": fisher_soft_polarity_signed_stack_direction_sha256(
            normalized
        ),
        "radius_sha256": _response_scalar_sha256(rate_value),
        "signed_mix_sha256": _response_scalar_sha256(mix_value),
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }


def _canonical_evidence_tree(value: object, *, label: str) -> object:
    if isinstance(value, Tensor):
        raise ValueError(f"{label} must not contain raw tensors")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite floats")
        return value
    if type(value) in (tuple, list):
        return [
            _canonical_evidence_tree(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError(f"{label} keys must be strings")
        return {
            key: _canonical_evidence_tree(value[key], label=f"{label}.{key}")
            for key in sorted(value)
        }
    raise ValueError(f"{label} contains a non-JSON scalar")


def _canonical_evidence_bytes(value: object, *, label: str) -> bytes:
    canonical = _canonical_evidence_tree(value, label=label)
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a plain dictionary")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} keys must be strings")
    if frozenset(value) != expected:
        raise ValueError(f"{label} key set differs")
    return value


def _require_canonical_equal(
    observed: object,
    expected: object,
    *,
    label: str,
) -> None:
    if _canonical_evidence_bytes(observed, label=label) != (
        _canonical_evidence_bytes(expected, label=f"expected {label}")
    ):
        raise ValueError(f"{label} differs from its frozen value")


def _validate_evidence_sha_fields(value: object, *, label: str) -> None:
    if type(value) is not dict:
        return
    for key, item in value.items():
        field_label = f"{label}.{key}"
        if key.endswith("_sha256"):
            _require_sha256(item, label=field_label)
        elif key.endswith("_sha256s"):
            if type(item) is not dict or not item:
                raise ValueError(f"{field_label} must be a nonempty hash dictionary")
            if any(type(name) is not str or not name for name in item):
                raise ValueError(f"{field_label} keys must be nonempty strings")
            for name, digest in item.items():
                _require_sha256(digest, label=f"{field_label}.{name}")
        if type(item) is dict:
            _validate_evidence_sha_fields(item, label=field_label)


def _strict_evidence_radius(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    if value > _RADIUS_MAX:
        raise ValueError(f"{label} exceeds the certified numerical magnitude limit")
    return value


def _strict_evidence_signed_mix(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if abs(value) > _SIGNED_MIX_MAX_ABS:
        raise ValueError(f"{label} must be inside [-0.5,0.5]")
    return value


def _validate_provider_payload(payload: object) -> dict[str, object]:
    _validate_constant_tensors()
    selected = _require_exact_keys(
        payload,
        _PROVIDER_PAYLOAD_KEYS,
        label="signed-stack provider payload",
    )
    _canonical_evidence_tree(selected, label="signed-stack provider payload")
    _validate_evidence_sha_fields(selected, label="signed-stack provider payload")
    for key, expected in _FROZEN_PROVIDER_PAYLOAD.items():
        _require_canonical_equal(
            selected[key],
            expected,
            label=f"signed-stack provider payload {key}",
        )
    rate = _strict_evidence_radius(
        selected["radius"], label="signed-stack provider payload radius"
    )
    mix = _strict_evidence_signed_mix(
        selected["signed_mix"],
        label="signed-stack provider payload signed_mix",
    )
    if selected["radius_sha256"] != _response_scalar_sha256(rate):
        raise ValueError("signed-stack provider radius hash differs")
    if selected["signed_mix_sha256"] != _response_scalar_sha256(mix):
        raise ValueError("signed-stack provider signed_mix hash differs")
    canonical = _canonical_evidence_tree(
        selected, label="signed-stack provider payload"
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_signed_stack_provider_artifact_sha256(
    payload: object,
) -> str:
    """Validate and domain-hash one scalar/hash-only provider payload."""

    canonical = _validate_provider_payload(payload)
    return _sha256(_PROVIDER_DOMAIN, canonical)


def _validate_box_certificate(
    certificate: object,
    *,
    payload: dict[str, object],
) -> dict[str, object]:
    selected = _require_exact_keys(
        certificate,
        _BOX_CERTIFICATE_KEYS,
        label="signed-stack provider box certificate",
    )
    _canonical_evidence_tree(selected, label="signed-stack provider box certificate")
    _validate_evidence_sha_fields(
        selected, label="signed-stack provider box certificate"
    )
    rate = payload["radius"]
    mix = payload["signed_mix"]
    assert type(rate) is float and type(mix) is float
    expected = {
        "schema": (
            "fisher_graph.fisher_soft_polarity_signed_stack_box_certificate.v1"
        ),
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "projection_max_abs": 1.0,
        "radius": rate,
        "signed_mix": mix,
        "radius_nonnegative": True,
        "signed_mix_in_closed_interval": (-0.5, 0.5),
        "signed_mix_max_abs": _SIGNED_MIX_MAX_ABS,
        "base_response_max_abs_upper_bound": math.tanh(rate),
        "stack_weight_formula": "abs_signed_mix_times_z_squared",
        "stack_weight_nonnegative": True,
        "stack_weight_max_upper_bound": abs(mix),
        "stack_target_semantics": (
            "signed_mix_z_squared_equals_weight_times_fixed_sign_target_for_"
            "nonzero_signed_mix_and_zero_when_signed_mix_is_zero"
        ),
        "calibrator_center_value": 0.0,
        "calibrator_odd_when_signed_mix_zero": True,
        "calibrator_oddness_claim_when_signed_mix_nonzero": "none",
        "calibrator_monotonicity_claim_when_signed_mix_nonzero": "none",
        "calibrator_max_abs_upper_bound": 1.0,
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_projection_"
            "at_most_one_weight_is_abs_signed_mix_times_z_squared_in_zero_to_"
            "one_half_and_q_is_a_convex_combination_of_tanh_radius_z_and_the_"
            "fixed_signed_polarity_target_so_abs_q_at_most_one_then_envelope_"
            "times_q_has_absolute_value_at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_radius_at_most_float64_max_"
            "over_eight_signed_mix_abs_at_most_one_half_and_all_post_tanh_"
            "products_and_sums_remain_bounded"
        ),
        "direction_sha256": payload["direction_sha256"],
        "radius_sha256": payload["radius_sha256"],
        "signed_mix_sha256": payload["signed_mix_sha256"],
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }
    for key, expected_value in expected.items():
        _require_canonical_equal(
            selected[key],
            expected_value,
            label=f"signed-stack provider box certificate {key}",
        )
    corners = selected["direction_box_corner_logits"]
    if type(corners) not in (tuple, list) or len(corners) != 4:
        raise ValueError(
            "signed-stack provider box certificate corner logits must contain four floats"
        )
    if any(type(item) is not float or not math.isfinite(item) for item in corners):
        raise ValueError(
            "signed-stack provider box certificate corner logits must contain finite floats"
        )
    maximum = max(abs(item) for item in corners)
    if abs(maximum - 1.0) > _NORMALIZATION_TOLERANCE:
        raise ValueError(
            "signed-stack provider box certificate corner normalization differs"
        )
    canonical = _canonical_evidence_tree(
        selected, label="signed-stack provider box certificate"
    )
    assert isinstance(canonical, dict)
    return canonical


def _strict_evidence_integer(
    metadata: dict[str, object],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = metadata[key]
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"signed-stack provider metadata {key} must be an integer >= {minimum}"
        )
    return value


def validate_fisher_soft_polarity_signed_stack_provider_evidence(
    payload: object,
    metadata: object,
) -> FisherSoftPolaritySignedStackProviderEvidence:
    """Validate a JSON-replayable payload and its full provider metadata."""

    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="signed-stack provider metadata",
    )
    _canonical_evidence_tree(selected, label="signed-stack provider metadata")
    _validate_evidence_sha_fields(selected, label="signed-stack provider metadata")
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"signed-stack provider metadata payload field {key}",
        )
    artifact_sha256 = fisher_soft_polarity_signed_stack_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact_sha256:
        raise ValueError("signed-stack provider metadata artifact hash differs")
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
        selected,
        "prepared_float_scalar_count",
        minimum=incremental_prepared,
    )
    incremental_parameter_bytes = _strict_evidence_integer(
        selected, "incremental_runtime_parameter_bytes_float64"
    )
    parameter_bytes = _strict_evidence_integer(
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
    expected_endpoint_incremental_prepared = (
        2 * rank + 4 * rank * conditional_rank + 8
    )
    expected_incremental_prepared = (
        2 * expected_endpoint_incremental_prepared + _FITTED_SCALAR_COUNT
    )
    expected_incremental_macs = (
        2 * rank + _DIRECTION_COUNT + 10 * rank * conditional_rank + 6
    )
    if incremental_prepared != expected_incremental_prepared:
        raise ValueError(
            "signed-stack provider incremental prepared-scalar formula differs"
        )
    if incremental_parameter_bytes != incremental_prepared * 8:
        raise ValueError("signed-stack provider incremental parameter bytes differ")
    if parameter_bytes != prepared * 8:
        raise ValueError("signed-stack provider total parameter bytes differ")
    if incremental_macs != expected_incremental_macs:
        raise ValueError("signed-stack provider incremental logical MAC formula differs")
    if logical_macs < incremental_macs:
        raise ValueError("signed-stack provider total logical MAC count differs")

    frozen_metadata = {
        "signed_stack_polarity_fitted_float_scalar_count": _FITTED_SCALAR_COUNT,
        "signed_stack_projection_dot_macs_per_token": _DIRECTION_COUNT,
        "signed_stack_calibrator_scalar_arithmetic_per_token": 8,
        "signed_stack_elementwise_scalar_arithmetic_per_token": 11,
        "signed_stack_nonlinear_scalar_ops_per_token": 2,
        "signed_stack_elementwise_scope": (
            "c1c2_product_kappa_scale_envelope_asinh_normalization_radius_"
            "scale_z_square_abs_mix_weight_one_minus_weight_base_scale_signed_"
            "mix_scale_stack_sum_and_envelope_product"
        ),
        "signed_stack_nonlinear_scope": "one_asinh_and_one_tanh",
        "logical_macs_accounting_scope": (
            "experimental_dense_upper_bound_includes_both_endpoint_factor_paths_"
            "four_term_projection_and_two_pedal_logits_elementwise_signed_stack_"
            "and_nonlinear_operations_reported_separately"
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
            label=f"signed-stack provider metadata {key}",
        )
    for key in (
        "signed_stack_polarity_fitted_float_scalar_count",
        "signed_stack_projection_dot_macs_per_token",
        "signed_stack_calibrator_scalar_arithmetic_per_token",
        "signed_stack_elementwise_scalar_arithmetic_per_token",
        "signed_stack_nonlinear_scalar_ops_per_token",
        "runtime_state_float_scalars_per_sequence",
    ):
        _strict_evidence_integer(selected, key)

    canonical_metadata = _canonical_evidence_tree(
        selected, label="signed-stack provider metadata"
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolaritySignedStackProviderEvidence(
        payload=canonical_payload,
        metadata=canonical_metadata,
        artifact_sha256=artifact_sha256,
    )


def fisher_soft_polarity_signed_stack_modal_terms(
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
    signed_stack_direction: Tensor,
    radius: float | Tensor,
    signed_mix: float | Tensor,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return differentiable ``(gain,direction,bounded,logit,pedal,delta)``."""

    parent = _finite_runtime_tensor(
        parent_modal, label="signed-stack-polarity parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="signed-stack-polarity coordinates", ndim=2
    ).to(parent.device)
    if (
        bounded_coordinates.shape != (parent.shape[0], 2)
        or bool((bounded_coordinates.abs() > 1.0).any())
    ):
        raise ValueError(
            "signed-stack-polarity coordinates must match parent rows inside [-1,1]"
        )
    if trust_fraction != _TRUST_FRACTION:
        raise ValueError("signed-stack-polarity trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_signed_stack_gain(
        bounded_coordinates,
        signed_stack_direction,
        radius,
        signed_mix,
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
        raise RuntimeError("signed-stack-polarity modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolaritySignedStackProvider(
    Gemma3L3L4CorrectionProvider
):
    """Immutable six-scalar signed-stack calibrator over a V19 endpoint pair."""

    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    direction: Tensor
    radius: float
    signed_mix: float
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
                label="signed-stack-polarity direction",
            ),
        )
        object.__setattr__(
            self,
            "radius",
            float(
                _radius_scalar(
                    self.radius,
                    detach=True,
                    label="signed-stack-polarity response radius",
                )
            ),
        )
        object.__setattr__(
            self,
            "signed_mix",
            float(
                _signed_mix_scalar(
                    self.signed_mix,
                    detach=True,
                    label="signed-stack-polarity signed mix",
                )
            ),
        )
        for name in ("transfer_protocol_sha256", "transfer_evidence_sha256"):
            _require_sha256(getattr(self, name), label=name)
        if self.trust_fraction != _TRUST_FRACTION:
            raise ValueError("signed-stack-polarity trust fraction is frozen at 0.25")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="signed-stack-polarity provider"
            ) != computed:
                raise ValueError("signed-stack-polarity provider artifact hash mismatch")
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
        # Dense endpoint paths, projection, and pedal logits are MAC-counted.
        # Signed-stack scalar/nonlinear work is reported separately.
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
                "signed_stack_provider.v1"
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
            "direction_sha256": fisher_soft_polarity_signed_stack_direction_sha256(
                self.direction
            ),
            "radius": self.radius,
            "signed_mix": self.signed_mix,
            "radius_sha256": _response_scalar_sha256(self.radius),
            "signed_mix_sha256": _response_scalar_sha256(self.signed_mix),
            "direction_float64_scalar_count": _DIRECTION_COUNT,
            "response_float64_scalar_count": _RESPONSE_SCALAR_COUNT,
            "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
            "radius_max": _RADIUS_MAX,
            "signed_mix_max_abs": _SIGNED_MIX_MAX_ABS,
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
                "bounded_convex_signed_stack_no_nonzero_mix_oddness_or_"
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

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def artifact_payload(self) -> dict[str, object]:
        """Return an independent scalar/hash-only copy of the hashed payload."""

        self.validate_integrity()
        return copy.deepcopy(self._payload())

    def validate_integrity(self) -> None:
        _validate_constant_tensors()
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        _direction(
            self.direction,
            detach=False,
            normalize=False,
            label="signed-stack-polarity stored direction",
        )
        _radius_scalar(
            self.radius,
            detach=False,
            label="signed-stack-polarity stored response radius",
        )
        _signed_mix_scalar(
            self.signed_mix,
            detach=False,
            label="signed-stack-polarity stored signed mix",
        )
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("signed-stack-polarity provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "box_certificate": fisher_soft_polarity_signed_stack_box_certificate(
                self.direction,
                radius=self.radius,
                signed_mix=self.signed_mix,
            ),
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "signed_stack_polarity_fitted_float_scalar_count": (
                _FITTED_SCALAR_COUNT
            ),
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
            "signed_stack_projection_dot_macs_per_token": _DIRECTION_COUNT,
            "signed_stack_calibrator_scalar_arithmetic_per_token": 8,
            "signed_stack_elementwise_scalar_arithmetic_per_token": 11,
            "signed_stack_nonlinear_scalar_ops_per_token": 2,
            "signed_stack_elementwise_scope": (
                "c1c2_product_kappa_scale_envelope_asinh_normalization_radius_"
                "scale_z_square_abs_mix_weight_one_minus_weight_base_scale_"
                "signed_mix_scale_stack_sum_and_envelope_product"
            ),
            "signed_stack_nonlinear_scope": "one_asinh_and_one_tanh",
            "logical_macs_accounting_scope": (
                "experimental_dense_upper_bound_includes_both_endpoint_factor_"
                "paths_four_term_projection_and_two_pedal_logits_elementwise_"
                "signed_stack_and_nonlinear_operations_reported_separately"
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
            raise ValueError("signed-stack-polarity response coordinates differ")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_signed_stack_gain(
            flat,
            self.direction.to(device=flat.device, dtype=flat.dtype),
            self.radius,
            self.signed_mix,
        )
        return gain.reshape(original_shape).contiguous()

    def terms_from_parent(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Expose pure runtime terms for replay and trust auditing."""

        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("signed-stack-polarity parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("signed-stack-polarity parent/coordinate geometry differs")
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, 2)
        gain, direction, bounded_direction, logit, pedal, delta = (
            fisher_soft_polarity_signed_stack_modal_terms(
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
                self.signed_mix,
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
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2).to(dtype=torch.float64)
        logit = fisher_continuous_pedal_logit(
            flat,
            self.base_provider.pedal_weight.to(flat.device),
            self.base_provider.pedal_bias.to(flat.device),
            self.proposal_provider.pedal_weight.to(flat.device),
            self.proposal_provider.pedal_bias.to(flat.device),
            gain.reshape(-1),
        )
        return logit.reshape(original_shape).contiguous()

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
            raise RuntimeError("signed-stack-polarity provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("signed-stack-polarity modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("signed-stack-polarity modal correction escaped support")
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


def build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    radius: float,
    signed_mix: float,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolaritySignedStackProvider:
    """Build and hash-bind one normalized signed-stack provider."""

    return AutonomousCompleteH4FisherSoftPolaritySignedStackProvider(
        base_provider=base_provider,
        proposal_provider=proposal_provider,
        direction=direction,
        radius=radius,
        signed_mix=signed_mix,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
