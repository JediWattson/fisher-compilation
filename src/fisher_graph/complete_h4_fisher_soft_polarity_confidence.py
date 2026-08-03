"""Bounded confidence calibration for complete-H4 soft-polarity transfer.

This provider keeps the authenticated complete-H4 endpoint pair, factor
interpolation, pedal interpolation, and pointwise trust certificate of
``complete_h4_fisher_soft_polarity``.  It replaces the unconstrained
four-vector polarity logit with a normalized direction and a monotone odd
confidence calibrator.  For bounded Fisher coordinates ``c=(c1,c2)``:

``phi(c) = [1, c1, c2, c1*c2]``

``z(c) = d @ phi(c)``, with ``max_box_corner(abs(z)) == 1``

``q(z) = tanh(a*z + b*z**3)``, with ``a >= 0`` and ``b >= 0``

``gain(c) = asinh(9*c2)/asinh(9) * q(z(c))``.

The bilinear projection reaches its extrema at the four box corners, so the
normalization certifies ``abs(z) <= 1`` throughout the closed coordinate box.
The calibrator is odd and monotone nondecreasing because its pre-tanh
derivative is ``a + 3*b*z**2 >= 0``.  Together with the normalized envelope,
this analytically certifies ``abs(gain) <= 1``.  The module is an analysis
provider and does not claim an inference optimization.
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
    "FISHER_SOFT_POLARITY_CONFIDENCE_CALIBRATOR_MAX",
    "FISHER_SOFT_POLARITY_CONFIDENCE_DIRECTION_COUNT",
    "FISHER_SOFT_POLARITY_CONFIDENCE_FITTED_SCALAR_COUNT",
    "FISHER_SOFT_POLARITY_CONFIDENCE_TRUST_FRACTION",
    "AutonomousCompleteH4FisherSoftPolarityConfidenceProvider",
    "FisherSoftPolarityConfidenceProviderEvidence",
    "build_autonomous_complete_h4_fisher_soft_polarity_confidence",
    "fisher_soft_polarity_confidence_box_certificate",
    "fisher_soft_polarity_confidence_calibrator",
    "fisher_soft_polarity_confidence_constant_tensor_sha256s",
    "fisher_soft_polarity_confidence_direction_sha256",
    "fisher_soft_polarity_confidence_gain",
    "fisher_soft_polarity_confidence_modal_terms",
    "fisher_soft_polarity_confidence_projection",
    "fisher_soft_polarity_confidence_provider_artifact_sha256",
    "fisher_soft_polarity_confidence_value",
    "normalize_fisher_soft_polarity_confidence_direction",
    "validate_fisher_soft_polarity_confidence_provider_evidence",
]


_DIRECTION_COUNT = 4
_CALIBRATOR_SCALAR_COUNT = 2
_FITTED_SCALAR_COUNT = _DIRECTION_COUNT + _CALIBRATOR_SCALAR_COUNT
_TRUST_FRACTION = FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION
_DIRECTION_INPUT_MAX_ABS = float(torch.finfo(torch.float64).max / 8.0)
_CALIBRATOR_MAX = float(torch.finfo(torch.float64).max / 8.0)
_NORMALIZATION_TOLERANCE = float(128.0 * torch.finfo(torch.float64).eps)

FISHER_SOFT_POLARITY_CONFIDENCE_DIRECTION_COUNT = _DIRECTION_COUNT
FISHER_SOFT_POLARITY_CONFIDENCE_FITTED_SCALAR_COUNT = _FITTED_SCALAR_COUNT
FISHER_SOFT_POLARITY_CONFIDENCE_CALIBRATOR_MAX = _CALIBRATOR_MAX
FISHER_SOFT_POLARITY_CONFIDENCE_TRUST_FRACTION = _TRUST_FRACTION

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-confidence:"
    b"provider:v1\0"
)
_CONSTANT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity-confidence:"
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
_CALIBRATOR_MAX_TENSOR = torch.tensor((_CALIBRATOR_MAX,), dtype=torch.float64)
_CONSTANT_TENSOR_SHA256S = {
    "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
    "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
    "calibrator_max": _tensor_sha256(_CALIBRATOR_MAX_TENSOR),
}
_CONSTANT_BUNDLE_SHA256 = _sha256(
    _CONSTANT_DOMAIN,
    {
        "formula": (
            "asinh_9c2_over_asinh_9_times_"
            "tanh_a_z_plus_b_z_cubed_for_box_normalized_bilinear_z"
        ),
        "direction_count": _DIRECTION_COUNT,
        "calibrator_scalar_count": _CALIBRATOR_SCALAR_COUNT,
        "fitted_scalar_count": _FITTED_SCALAR_COUNT,
        "direction_input_max_abs": _DIRECTION_INPUT_MAX_ABS,
        "calibrator_max": _CALIBRATOR_MAX,
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
        "linear_coefficient",
        "cubic_coefficient",
        "linear_coefficient_sha256",
        "cubic_coefficient_sha256",
        "direction_float64_scalar_count",
        "calibrator_float64_scalar_count",
        "fitted_float64_scalar_count",
        "calibrator_max",
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
        "confidence_polarity_fitted_float_scalar_count",
        "incremental_runtime_parameter_bytes_float64",
        "runtime_parameter_bytes_float64",
        "incremental_logical_macs_per_token_upper_bound",
        "logical_macs_per_token_upper_bound",
        "confidence_projection_dot_macs_per_token",
        "confidence_calibrator_scalar_arithmetic_per_token",
        "confidence_elementwise_scalar_arithmetic_per_token",
        "confidence_nonlinear_scalar_ops_per_token",
        "confidence_elementwise_scope",
        "confidence_nonlinear_scope",
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
        "linear_coefficient",
        "cubic_coefficient",
        "linear_coefficient_nonnegative",
        "cubic_coefficient_nonnegative",
        "calibrator_odd",
        "calibrator_monotone_nondecreasing",
        "calibrator_strictly_monotone",
        "calibrator_derivative",
        "calibrator_pre_tanh_derivative_min",
        "calibrator_argument_max_abs_upper_bound",
        "calibrator_max_abs_upper_bound",
        "envelope_max_abs",
        "gain_max_abs",
        "pointwise_trust_fraction",
        "proof",
        "numerical_totality_proof",
        "direction_sha256",
        "linear_coefficient_sha256",
        "cubic_coefficient_sha256",
        "constant_bundle_sha256",
    }
)

_FROZEN_PROVIDER_PAYLOAD = {
    "schema": (
        "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_"
        "confidence_provider.v1"
    ),
    "site": _H4_SITE,
    "write_scope": "complete_h4_causal_support",
    "direction_float64_scalar_count": _DIRECTION_COUNT,
    "calibrator_float64_scalar_count": _CALIBRATOR_SCALAR_COUNT,
    "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
    "calibrator_max": _CALIBRATOR_MAX,
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
    "gain_formula": (
        "asinh_9c2_over_asinh_9_times_tanh_a_z_plus_b_z_cubed_"
        "for_box_normalized_bilinear_z"
    ),
    "direction_normalization": "max_absolute_bilinear_box_corner_logit",
    "calibrator_certificate": "odd_monotone_nondecreasing",
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
class FisherSoftPolarityConfidenceProviderEvidence:
    """Canonical, model-free replay evidence for one confidence provider."""

    payload: dict[str, object]
    metadata: dict[str, object]
    artifact_sha256: str


def fisher_soft_polarity_confidence_constant_tensor_sha256s() -> dict[str, str]:
    """Return hashes for every confidence-specific frozen tensor."""

    return dict(_CONSTANT_TENSOR_SHA256S)


def fisher_soft_polarity_confidence_direction_sha256(direction: Tensor) -> str:
    """Return the provider's exact hash for one normalized float64 direction."""

    _validate_constant_tensors()
    selected = _normalized_direction(
        direction,
        detach=True,
        label="confidence-polarity hashed direction",
    )
    return _tensor_sha256(selected)


def _validate_constant_tensors() -> None:
    observed = {
        "box_corner_features": _tensor_sha256(_BOX_CORNER_FEATURES),
        "direction_count": _tensor_sha256(_DIRECTION_COUNT_TENSOR),
        "calibrator_max": _tensor_sha256(_CALIBRATOR_MAX_TENSOR),
    }
    if observed != _CONSTANT_TENSOR_SHA256S:
        raise RuntimeError("confidence-polarity frozen formula tensors drifted")


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
    if (
        not bool(torch.isfinite(corners).all())
        or float(scale.detach()) <= 0.0
    ):
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


def normalize_fisher_soft_polarity_confidence_direction(direction: Tensor) -> Tensor:
    """Normalize a finite direction by its maximum absolute box-corner logit.

    The operation remains differentiable for a unique active corner.  Provider
    construction separately detaches and copies the normalized result.
    """

    _validate_constant_tensors()
    return _direction(
        direction,
        detach=False,
        normalize=True,
        label="confidence-polarity direction",
    )


def _normalized_direction(value: object, *, detach: bool, label: str) -> Tensor:
    return _direction(value, detach=detach, normalize=False, label=label)


def _calibrator_scalar(value: object, *, detach: bool, label: str) -> Tensor:
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
    if scalar < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    if scalar > _CALIBRATOR_MAX:
        raise ValueError(f"{label} exceeds the certified numerical magnitude limit")
    return selected.contiguous()


def _calibrator_scalar_sha256(value: float) -> str:
    return _tensor_sha256(torch.tensor((value,), dtype=torch.float64))


def fisher_soft_polarity_confidence_projection(
    coordinates: Tensor, direction: Tensor
) -> Tensor:
    """Return the certified box-normalized bilinear projection ``z``."""

    features = _soft_features(coordinates)
    normalized = _normalized_direction(
        direction,
        detach=False,
        label="confidence-polarity direction",
    ).to(device=features.device, dtype=features.dtype)
    result = features @ normalized
    if (
        not bool(torch.isfinite(result).all())
        or float(result.detach().abs().max())
        > 1.0 + _NORMALIZATION_TOLERANCE
    ):
        raise RuntimeError("confidence-polarity projection violated its box bound")
    return result.contiguous()


def fisher_soft_polarity_confidence_calibrator(
    projection: Tensor,
    linear_coefficient: float | Tensor,
    cubic_coefficient: float | Tensor,
) -> Tensor:
    """Return the monotone odd calibrator ``tanh(a*z + b*z**3)``."""

    z = _finite_runtime_tensor(
        projection, label="confidence-polarity projection", ndim=1
    )
    if float(z.detach().abs().max()) > 1.0 + _NORMALIZATION_TOLERANCE:
        raise ValueError("confidence-polarity projection must remain inside [-1,1]")
    linear = _calibrator_scalar(
        linear_coefficient,
        detach=False,
        label="confidence-polarity linear coefficient",
    ).to(device=z.device, dtype=z.dtype)
    cubic = _calibrator_scalar(
        cubic_coefficient,
        detach=False,
        label="confidence-polarity cubic coefficient",
    ).to(device=z.device, dtype=z.dtype)
    argument = linear * z + cubic * z.square() * z
    result = torch.tanh(argument)
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("confidence-polarity calibrator violated its bound")
    return result.contiguous()


def fisher_soft_polarity_confidence_value(
    coordinates: Tensor,
    direction: Tensor,
    linear_coefficient: float | Tensor,
    cubic_coefficient: float | Tensor,
) -> Tensor:
    """Return the confidence-calibrated polarity value for bounded coordinates."""

    projection = fisher_soft_polarity_confidence_projection(coordinates, direction)
    return fisher_soft_polarity_confidence_calibrator(
        projection,
        linear_coefficient,
        cubic_coefficient,
    )


def fisher_soft_polarity_confidence_gain(
    coordinates: Tensor,
    direction: Tensor,
    linear_coefficient: float | Tensor,
    cubic_coefficient: float | Tensor,
) -> Tensor:
    """Return the analytically bounded confidence-calibrated endpoint gain."""

    envelope = _soft_envelope(coordinates)
    value = fisher_soft_polarity_confidence_value(
        coordinates,
        direction,
        linear_coefficient,
        cubic_coefficient,
    )
    result = envelope * value
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("confidence-polarity gain violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_confidence_box_certificate(
    direction: Tensor,
    *,
    linear_coefficient: float,
    cubic_coefficient: float,
) -> dict[str, object]:
    """Return the analytic box, oddness, monotonicity, and trust prerequisites."""

    _validate_constant_tensors()
    normalized = _normalized_direction(
        direction,
        detach=True,
        label="confidence-polarity certificate direction",
    )
    linear = _calibrator_scalar(
        linear_coefficient,
        detach=True,
        label="confidence-polarity certificate linear coefficient",
    )
    cubic = _calibrator_scalar(
        cubic_coefficient,
        detach=True,
        label="confidence-polarity certificate cubic coefficient",
    )
    corners = _BOX_CORNER_FEATURES @ normalized
    linear_value = float(linear)
    cubic_value = float(cubic)
    return {
        "schema": "fisher_graph.fisher_soft_polarity_confidence_box_certificate.v1",
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "direction_box_corner_logits": tuple(float(item) for item in corners),
        "projection_max_abs": 1.0,
        "linear_coefficient": linear_value,
        "cubic_coefficient": cubic_value,
        "linear_coefficient_nonnegative": True,
        "cubic_coefficient_nonnegative": True,
        "calibrator_odd": True,
        "calibrator_monotone_nondecreasing": True,
        "calibrator_strictly_monotone": bool(
            linear_value > 0.0 or cubic_value > 0.0
        ),
        "calibrator_derivative": "sech_squared_times_a_plus_3b_z_squared",
        "calibrator_pre_tanh_derivative_min": linear_value,
        "calibrator_argument_max_abs_upper_bound": linear_value + cubic_value,
        "calibrator_max_abs_upper_bound": math.tanh(
            linear_value + cubic_value
        ),
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_asinh_at_most_"
            "one_times_abs_monotone_odd_tanh_calibrator_at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_and_nonnegative_a_b_each_at_"
            "most_float64_max_over_eight_leave_accumulation_headroom"
        ),
        "direction_sha256": fisher_soft_polarity_confidence_direction_sha256(
            normalized
        ),
        "linear_coefficient_sha256": _calibrator_scalar_sha256(linear_value),
        "cubic_coefficient_sha256": _calibrator_scalar_sha256(cubic_value),
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


def _strict_evidence_calibrator(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must not be signed negative zero")
    if value > _CALIBRATOR_MAX:
        raise ValueError(f"{label} exceeds the certified numerical magnitude limit")
    return value


def _validate_provider_payload(payload: object) -> dict[str, object]:
    _validate_constant_tensors()
    selected = _require_exact_keys(
        payload,
        _PROVIDER_PAYLOAD_KEYS,
        label="confidence provider payload",
    )
    _canonical_evidence_tree(selected, label="confidence provider payload")
    _validate_evidence_sha_fields(selected, label="confidence provider payload")
    for key, expected in _FROZEN_PROVIDER_PAYLOAD.items():
        _require_canonical_equal(
            selected[key],
            expected,
            label=f"confidence provider payload {key}",
        )

    linear = _strict_evidence_calibrator(
        selected["linear_coefficient"],
        label="confidence provider payload linear_coefficient",
    )
    cubic = _strict_evidence_calibrator(
        selected["cubic_coefficient"],
        label="confidence provider payload cubic_coefficient",
    )
    if selected["linear_coefficient_sha256"] != _calibrator_scalar_sha256(linear):
        raise ValueError("confidence provider linear coefficient hash differs")
    if selected["cubic_coefficient_sha256"] != _calibrator_scalar_sha256(cubic):
        raise ValueError("confidence provider cubic coefficient hash differs")
    canonical = _canonical_evidence_tree(
        selected,
        label="confidence provider payload",
    )
    assert isinstance(canonical, dict)
    return canonical


def fisher_soft_polarity_confidence_provider_artifact_sha256(
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
        label="confidence provider box certificate",
    )
    _canonical_evidence_tree(selected, label="confidence provider box certificate")
    _validate_evidence_sha_fields(
        selected,
        label="confidence provider box certificate",
    )
    linear = payload["linear_coefficient"]
    cubic = payload["cubic_coefficient"]
    assert type(linear) is float and type(cubic) is float
    expected = {
        "schema": "fisher_graph.fisher_soft_polarity_confidence_box_certificate.v1",
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "direction_count": _DIRECTION_COUNT,
        "direction_normalization": "max_absolute_bilinear_box_corner_logit",
        "projection_max_abs": 1.0,
        "linear_coefficient": linear,
        "cubic_coefficient": cubic,
        "linear_coefficient_nonnegative": True,
        "cubic_coefficient_nonnegative": True,
        "calibrator_odd": True,
        "calibrator_monotone_nondecreasing": True,
        "calibrator_strictly_monotone": bool(linear > 0.0 or cubic > 0.0),
        "calibrator_derivative": "sech_squared_times_a_plus_3b_z_squared",
        "calibrator_pre_tanh_derivative_min": linear,
        "calibrator_argument_max_abs_upper_bound": linear + cubic,
        "calibrator_max_abs_upper_bound": math.tanh(linear + cubic),
        "envelope_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "pointwise_trust_fraction": _TRUST_FRACTION,
        "proof": (
            "bilinear_box_extrema_are_corners_then_abs_normalized_asinh_at_most_"
            "one_times_abs_monotone_odd_tanh_calibrator_at_most_one"
        ),
        "numerical_totality_proof": (
            "normalized_projection_abs_at_most_one_and_nonnegative_a_b_each_at_"
            "most_float64_max_over_eight_leave_accumulation_headroom"
        ),
        "direction_sha256": payload["direction_sha256"],
        "linear_coefficient_sha256": payload["linear_coefficient_sha256"],
        "cubic_coefficient_sha256": payload["cubic_coefficient_sha256"],
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }
    for key, expected_value in expected.items():
        _require_canonical_equal(
            selected[key],
            expected_value,
            label=f"confidence provider box certificate {key}",
        )

    corners = selected["direction_box_corner_logits"]
    if type(corners) not in (tuple, list) or len(corners) != 4:
        raise ValueError(
            "confidence provider box certificate corner logits must contain four floats"
        )
    if any(type(item) is not float or not math.isfinite(item) for item in corners):
        raise ValueError(
            "confidence provider box certificate corner logits must contain finite floats"
        )
    maximum = max(abs(item) for item in corners)
    if (
        maximum > 1.0 + _NORMALIZATION_TOLERANCE
        or abs(maximum - 1.0) > _NORMALIZATION_TOLERANCE
    ):
        raise ValueError(
            "confidence provider box certificate corner normalization differs"
        )
    canonical = _canonical_evidence_tree(
        selected,
        label="confidence provider box certificate",
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
            f"confidence provider metadata {key} must be an integer >= {minimum}"
        )
    return value


def validate_fisher_soft_polarity_confidence_provider_evidence(
    payload: object,
    metadata: object,
) -> FisherSoftPolarityConfidenceProviderEvidence:
    """Validate a JSON-replayable provider payload and its full metadata.

    The check is model-free: evidence contains scalar, string, list, and hash
    values only.  Endpoint lineage remains explicit in ``payload`` so a caller
    can bind it to its independently authenticated endpoint receipts.
    """

    canonical_payload = _validate_provider_payload(payload)
    selected = _require_exact_keys(
        metadata,
        _PROVIDER_PAYLOAD_KEYS | _PROVIDER_METADATA_EXTRA_KEYS,
        label="confidence provider metadata",
    )
    _canonical_evidence_tree(selected, label="confidence provider metadata")
    _validate_evidence_sha_fields(selected, label="confidence provider metadata")
    for key in _PROVIDER_PAYLOAD_KEYS:
        _require_canonical_equal(
            selected[key],
            canonical_payload[key],
            label=f"confidence provider metadata payload field {key}",
        )

    artifact_sha256 = fisher_soft_polarity_confidence_provider_artifact_sha256(
        canonical_payload
    )
    if selected["artifact_sha256"] != artifact_sha256:
        raise ValueError("confidence provider metadata artifact hash differs")
    _validate_box_certificate(
        selected["box_certificate"],
        payload=canonical_payload,
    )

    rank = _strict_evidence_integer(selected, "rank", minimum=1)
    conditional_rank = _strict_evidence_integer(
        selected,
        "conditional_rank",
        minimum=1,
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
        selected,
        "incremental_runtime_parameter_bytes_float64",
    )
    parameter_bytes = _strict_evidence_integer(
        selected,
        "runtime_parameter_bytes_float64",
    )
    incremental_macs = _strict_evidence_integer(
        selected,
        "incremental_logical_macs_per_token_upper_bound",
    )
    logical_macs = _strict_evidence_integer(
        selected,
        "logical_macs_per_token_upper_bound",
        minimum=incremental_macs,
    )
    expected_incremental_macs = (
        2 * rank
        + _DIRECTION_COUNT
        + 10 * rank * conditional_rank
        + 6
    )
    expected_endpoint_incremental_prepared = (
        2 * rank + 4 * rank * conditional_rank + 8
    )
    expected_incremental_prepared = (
        2 * expected_endpoint_incremental_prepared + _FITTED_SCALAR_COUNT
    )
    if incremental_prepared != expected_incremental_prepared:
        raise ValueError(
            "confidence provider incremental prepared-scalar formula differs"
        )
    if incremental_parameter_bytes != incremental_prepared * 8:
        raise ValueError("confidence provider incremental parameter bytes differ")
    if parameter_bytes != prepared * 8:
        raise ValueError("confidence provider total parameter bytes differ")
    if incremental_macs != expected_incremental_macs:
        raise ValueError("confidence provider incremental logical MAC formula differs")
    if logical_macs < incremental_macs:
        raise ValueError("confidence provider total logical MAC count differs")

    frozen_metadata = {
        "confidence_polarity_fitted_float_scalar_count": _FITTED_SCALAR_COUNT,
        "confidence_projection_dot_macs_per_token": _DIRECTION_COUNT,
        "confidence_calibrator_scalar_arithmetic_per_token": 5,
        "confidence_elementwise_scalar_arithmetic_per_token": 9,
        "confidence_nonlinear_scalar_ops_per_token": 2,
        "confidence_elementwise_scope": (
            "c1c2_product_kappa_scale_asinh_normalization_z_square_z_cube_"
            "a_scale_b_scale_calibrator_sum_and_envelope_product"
        ),
        "confidence_nonlinear_scope": "one_asinh_and_one_tanh",
        "logical_macs_accounting_scope": (
            "experimental_dense_upper_bound_includes_both_endpoint_factor_paths_"
            "four_term_projection_and_two_pedal_logits_elementwise_polynomial_"
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
            label=f"confidence provider metadata {key}",
        )
    for key in (
        "confidence_polarity_fitted_float_scalar_count",
        "confidence_projection_dot_macs_per_token",
        "confidence_calibrator_scalar_arithmetic_per_token",
        "confidence_elementwise_scalar_arithmetic_per_token",
        "confidence_nonlinear_scalar_ops_per_token",
        "runtime_state_float_scalars_per_sequence",
    ):
        _strict_evidence_integer(selected, key)

    canonical_metadata = _canonical_evidence_tree(
        selected,
        label="confidence provider metadata",
    )
    assert isinstance(canonical_metadata, dict)
    return FisherSoftPolarityConfidenceProviderEvidence(
        payload=canonical_payload,
        metadata=canonical_metadata,
        artifact_sha256=artifact_sha256,
    )


def fisher_soft_polarity_confidence_modal_terms(
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
    confidence_direction: Tensor,
    linear_coefficient: float | Tensor,
    cubic_coefficient: float | Tensor,
    *,
    trust_fraction: float = _TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return differentiable ``(gain,direction,bounded,logit,pedal,delta)``."""

    parent = _finite_runtime_tensor(
        parent_modal, label="confidence-polarity parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="confidence-polarity coordinates", ndim=2
    ).to(parent.device)
    if (
        bounded_coordinates.shape != (parent.shape[0], 2)
        or bool((bounded_coordinates.abs() > 1.0).any())
    ):
        raise ValueError(
            "confidence-polarity coordinates must match parent rows inside [-1,1]"
        )
    if trust_fraction != _TRUST_FRACTION:
        raise ValueError("confidence-polarity trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_confidence_gain(
        bounded_coordinates,
        confidence_direction,
        linear_coefficient,
        cubic_coefficient,
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
        parent,
        direction,
        trust_fraction=trust_fraction,
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
        raise RuntimeError("confidence-polarity modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolarityConfidenceProvider(
    Gemma3L3L4CorrectionProvider
):
    """Immutable six-scalar confidence calibrator over a V19 endpoint pair."""

    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    direction: Tensor
    linear_coefficient: float
    cubic_coefficient: float
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
                label="confidence-polarity direction",
            ),
        )
        object.__setattr__(
            self,
            "linear_coefficient",
            float(
                _calibrator_scalar(
                    self.linear_coefficient,
                    detach=True,
                    label="confidence-polarity linear coefficient",
                )
            ),
        )
        object.__setattr__(
            self,
            "cubic_coefficient",
            float(
                _calibrator_scalar(
                    self.cubic_coefficient,
                    detach=True,
                    label="confidence-polarity cubic coefficient",
                )
            ),
        )
        for name in ("transfer_protocol_sha256", "transfer_evidence_sha256"):
            _require_sha256(getattr(self, name), label=name)
        if self.trust_fraction != _TRUST_FRACTION:
            raise ValueError("confidence-polarity trust fraction is frozen at 0.25")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="confidence-polarity provider"
            ) != computed:
                raise ValueError("confidence-polarity provider artifact hash mismatch")
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
        # Endpoint factor paths, the four-term projection, and two pedal
        # logits are counted as dense MACs.  Scalar polynomial/nonlinear work
        # is reported separately in metadata, matching the V20f convention.
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
                "confidence_provider.v1"
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
            "direction_sha256": fisher_soft_polarity_confidence_direction_sha256(
                self.direction
            ),
            "linear_coefficient": self.linear_coefficient,
            "cubic_coefficient": self.cubic_coefficient,
            "linear_coefficient_sha256": _calibrator_scalar_sha256(
                self.linear_coefficient
            ),
            "cubic_coefficient_sha256": _calibrator_scalar_sha256(
                self.cubic_coefficient
            ),
            "direction_float64_scalar_count": _DIRECTION_COUNT,
            "calibrator_float64_scalar_count": _CALIBRATOR_SCALAR_COUNT,
            "fitted_float64_scalar_count": _FITTED_SCALAR_COUNT,
            "calibrator_max": _CALIBRATOR_MAX,
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
            "gain_formula": (
                "asinh_9c2_over_asinh_9_times_tanh_a_z_plus_b_z_cubed_"
                "for_box_normalized_bilinear_z"
            ),
            "direction_normalization": "max_absolute_bilinear_box_corner_logit",
            "calibrator_certificate": "odd_monotone_nondecreasing",
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
            label="confidence-polarity stored direction",
        )
        _calibrator_scalar(
            self.linear_coefficient,
            detach=False,
            label="confidence-polarity stored linear coefficient",
        )
        _calibrator_scalar(
            self.cubic_coefficient,
            detach=False,
            label="confidence-polarity stored cubic coefficient",
        )
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("confidence-polarity provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "box_certificate": fisher_soft_polarity_confidence_box_certificate(
                self.direction,
                linear_coefficient=self.linear_coefficient,
                cubic_coefficient=self.cubic_coefficient,
            ),
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "confidence_polarity_fitted_float_scalar_count": _FITTED_SCALAR_COUNT,
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
            "confidence_projection_dot_macs_per_token": _DIRECTION_COUNT,
            "confidence_calibrator_scalar_arithmetic_per_token": 5,
            "confidence_elementwise_scalar_arithmetic_per_token": 9,
            "confidence_nonlinear_scalar_ops_per_token": 2,
            "confidence_elementwise_scope": (
                "c1c2_product_kappa_scale_asinh_normalization_z_square_z_cube_"
                "a_scale_b_scale_calibrator_sum_and_envelope_product"
            ),
            "confidence_nonlinear_scope": "one_asinh_and_one_tanh",
            "logical_macs_accounting_scope": (
                "experimental_dense_upper_bound_includes_both_endpoint_factor_paths_"
                "four_term_projection_and_two_pedal_logits_elementwise_polynomial_"
                "and_nonlinear_operations_reported_separately"
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
            raise ValueError("confidence-polarity response coordinates differ")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_confidence_gain(
            flat,
            self.direction.to(device=flat.device, dtype=flat.dtype),
            self.linear_coefficient,
            self.cubic_coefficient,
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
            raise ValueError("confidence-polarity parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("confidence-polarity parent/coordinate geometry differs")
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, 2)
        gain, direction, bounded_direction, logit, pedal, delta = (
            fisher_soft_polarity_confidence_modal_terms(
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
                self.linear_coefficient,
                self.cubic_coefficient,
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
            raise RuntimeError("confidence-polarity provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("confidence-polarity modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("confidence-polarity modal correction escaped support")
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


def build_autonomous_complete_h4_fisher_soft_polarity_confidence(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    direction: Tensor,
    linear_coefficient: float,
    cubic_coefficient: float,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolarityConfidenceProvider:
    """Build and hash-bind one normalized confidence-calibrated provider."""

    return AutonomousCompleteH4FisherSoftPolarityConfidenceProvider(
        base_provider=base_provider,
        proposal_provider=proposal_provider,
        direction=direction,
        linear_coefficient=linear_coefficient,
        cubic_coefficient=cubic_coefficient,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )
