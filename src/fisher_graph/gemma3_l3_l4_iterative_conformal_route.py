"""Affine conformal routing over the frozen Gemma lag-B parent modes.

Iteration four keeps the causal top-two balance state from iteration two, but
uses that state both as a gate and as a continuous controller for one
complex-linear (conformal) ``2 x 2`` transform.  With coefficient order

``(shared_real, shared_imag, contrast_real, contrast_imag)``

the runtime correction is

``delta_t = (g_t * m_t) @ C(a0 + g_t * a1, b0 + g_t * b1)``,

where ``C(a, b) = [[a, -b], [b, a]]`` and ``m_t`` contains the selected
top-two parent modes.  The corresponding prompt-level linearization has the
four shared/contrast features ``g*m`` and ``g^2*m``.

Incremental callers must supply parent-modal chunks produced by an upstream
lag-aware executor.  This provider carries only the two balance accumulators;
it never recomputes a lagged parent from an isolated chunk.  The current
parent-modal API cannot authenticate chunk ordering because the repository
does not yet expose a cached parent producer/cursor, so ordered, nonduplicated
chunks remain an explicit caller obligation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from .gemma3_l3_l4_iterative_residual_campaign import (
    GemmaIterativeResidualCampaignRecipe,
)
from .gemma3_l3_l4_iterative_state_router import (
    ROUTE_OPERATOR_NORM_BOUND,
    ROUTE_RIDGE,
    _balance_feature,
    _source_only_parent,
    top2_lag_b_output_modes,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    _tensor_sha256,
)


__all__ = [
    "CONFORMAL_COEFFICIENT_COUNT",
    "CONFORMAL_LINEAR_MACS_PER_TOKEN",
    "CONFORMAL_NONLINEAR_SCALAR_OPS_PER_TOKEN",
    "CONFORMAL_OPERATOR_NORM_BOUND",
    "CONFORMAL_RIDGE",
    "GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE",
    "GemmaCausalTop2ConformalRouteH4Provider",
    "GemmaCausalTop2ConformalRouteState",
    "GemmaIterativeConformalRouteFitRecord",
    "GemmaIterativeConformalRouteFoldFit",
    "build_gemma_iterative_conformal_route_fit_record",
    "fit_gemma_iterative_conformal_route_fold",
    "fit_gemma_iterative_conformal_route_fold_provider",
    "fit_gemma_iterative_conformal_route_full_provider",
    "gemma_causal_top2_conformal_route_provider_artifact_sha256",
]


CONFORMAL_COEFFICIENT_COUNT = 4
CONFORMAL_LINEAR_MACS_PER_TOKEN = 8
CONFORMAL_NONLINEAR_SCALAR_OPS_PER_TOKEN = 5
CONFORMAL_OPERATOR_NORM_BOUND = ROUTE_OPERATOR_NORM_BOUND
CONFORMAL_RIDGE = ROUTE_RIDGE
_PREPARED_FLOAT_SCALAR_COUNT = 6
_DERIVED_CONSTANT_FLOAT_COUNT = 2
_RUNTIME_STATE_FLOAT_COUNT = 2
_LINEAR_ACCUMULATOR_OPS_PER_TOKEN = 4
_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN = 1
_COLUMN_SUPPORT_EPSILON = 1.0e-12
_DESIGN_RANK_TOLERANCE = 1.0e-12
_H4_SITE = "layer.4.output"
_ROUTE_STATE_SEMANTICS = (
    "top2_parent_lag_b_modal_cumulative_balance_v1"
)
_CONFORMAL_ROUTE_SEMANTICS = (
    "delta=(g*selected_top2)@C(a0+g*a1,b0+g*b1)"
)
_COEFFICIENT_ORDER = (
    "shared_real",
    "shared_imag",
    "contrast_real",
    "contrast_imag",
)
_ENDPOINT_ORDER = ("g=-1", "g=+1")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_RECORD_DOMAIN = b"fisher-graph:gemma-conformal-route-fit-record:v1\0"
_FOLD_FIT_DOMAIN = b"fisher-graph:gemma-conformal-route-fold-fit:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma-top2-conformal-provider:v1\0"
_RESOURCE_DOMAIN = b"fisher-graph:gemma-conformal-route-resources:v1\0"


GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE = (
    GemmaIterativeResidualCampaignRecipe(
        recipe_id=(
            "causal_top2_lag_b_modal_balance_affine_conformal_route"
        ),
        fit_record_jacobian_field="jacobian_by_conformal_coefficient",
        fold_coefficient_field="coefficients_by_conformal_coefficient",
        coefficient_count=CONFORMAL_COEFFICIENT_COUNT,
        learned_parameter_attribute=(
            "marginal_learned_float_scalar_count"
        ),
        learned_parameter_fallback_attribute=None,
        expected_learned_parameter_count=CONFORMAL_COEFFICIENT_COUNT,
        logical_macs_attribute=(
            "marginal_logical_macs_per_token_upper_bound"
        ),
        logical_macs_fallback_attribute=None,
        expected_logical_macs_per_token_upper_bound=(
            CONFORMAL_LINEAR_MACS_PER_TOKEN
        ),
        logical_macs_must_equal_residual_width=False,
        extra_resource_expectations=(
            (
                "prepared_float_scalar_count",
                "marginal_prepared_float_scalar_count",
                _PREPARED_FLOAT_SCALAR_COUNT,
            ),
            (
                "derived_constant_float_count",
                "marginal_derived_prepared_float_scalar_count",
                _DERIVED_CONSTANT_FLOAT_COUNT,
            ),
            (
                "runtime_state_float_count_per_sequence",
                "runtime_state_float_scalars_per_sequence",
                _RUNTIME_STATE_FLOAT_COUNT,
            ),
            (
                "nonlinear_scalar_ops_per_token_upper_bound",
                "nonlinear_scalar_ops_per_token_upper_bound",
                CONFORMAL_NONLINEAR_SCALAR_OPS_PER_TOKEN,
            ),
            (
                "linear_accumulator_scalar_ops_per_token_upper_bound",
                "linear_accumulator_scalar_ops_per_token_upper_bound",
                _LINEAR_ACCUMULATOR_OPS_PER_TOKEN,
            ),
            (
                "zero_denominator_comparisons_per_token_upper_bound",
                "zero_denominator_comparisons_per_token_upper_bound",
                _ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN,
            ),
            (
                "parent_decoder_invocations_per_token",
                "parent_decoder_invocations_per_token",
                1,
            ),
        ),
        audit_recipe_fields=(
            (
                "execution_mode",
                "fit_only_two_phase_family_blocked_iterative_conformal_route",
            ),
            ("conformal_matrix_shape", (2, 2)),
            ("conformal_coefficient_order", _COEFFICIENT_ORDER),
            ("route_state_semantics", _ROUTE_STATE_SEMANTICS),
            ("conformal_route_semantics", _CONFORMAL_ROUTE_SEMANTICS),
            (
                "endpoint_operator_norm_bound",
                CONFORMAL_OPERATOR_NORM_BOUND,
            ),
        ),
        provider_audit_fields=(
            (
                "routed_parent_decoder_mode_indices",
                "top_mode_indices",
            ),
        ),
        parent_tensor_audit_fields=(
            ("parent_h4_decoder_sha256", "decoder"),
            ("parent_h4_lag_kernel_sha256", "lag_kernel"),
        ),
        fold_projection_field="trust_projection_applied",
        fold_projection_count_audit_field="fold_trust_projection_count",
        projection_interpretation_audit_field=(
            "trust_projection_interpretation"
        ),
        projection_interpretation=(
            "global_radial_endpoint_operator_norm_projection_is_"
            "linearization_extrapolation"
        ),
        resource_envelope_error=(
            "fixed conformal state route exceeds its resource envelope"
        ),
        linearization_error=(
            "OOF conformal-route linearization requires four finite "
            "coefficients"
        ),
    )
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float2(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _float4(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != CONFORMAL_COEFFICIENT_COUNT
    ):
        raise ValueError(f"{label} must contain exactly four scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _indices2(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or value[0] == value[1]
    ):
        raise ValueError(f"{label} must contain two distinct nonnegative indices")
    return (value[0], value[1])


def _endpoint_operator_norms(
    coefficients: Sequence[float],
) -> tuple[float, float]:
    """Return exact conformal operator norms at ``g=-1`` and ``g=+1``."""

    a0, b0, a1, b1 = _float4(
        coefficients,
        label="conformal coefficients",
    )
    return (
        math.hypot(a0 - a1, b0 - b1),
        math.hypot(a0 + a1, b0 + b1),
    )


def _project_conformal_coefficients(
    coefficients: Tensor,
    *,
    supported: tuple[int, ...],
) -> tuple[
    Tensor,
    tuple[float, float],
    tuple[float, float],
    float,
    bool,
]:
    """Globally radial-project all four coefficients by endpoint norm."""

    if (
        not isinstance(coefficients, Tensor)
        or coefficients.shape != (CONFORMAL_COEFFICIENT_COUNT,)
        or not coefficients.is_floating_point()
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise ValueError("conformal projection requires four finite coefficients")
    if (
        type(supported) is not tuple
        or supported != tuple(sorted(set(supported)))
        or any(
            type(index) is not int
            or not 0 <= index < CONFORMAL_COEFFICIENT_COUNT
            for index in supported
        )
    ):
        raise ValueError("supported conformal coefficients must be canonical")
    unsupported = tuple(
        index
        for index in range(CONFORMAL_COEFFICIENT_COUNT)
        if index not in supported
    )
    if any(float(coefficients[index]) != 0.0 for index in unsupported):
        raise ValueError("unsupported conformal coefficients must be zero")

    pre = _endpoint_operator_norms(
        tuple(float(value) for value in coefficients)
    )
    maximum = max(pre)
    applied = maximum > CONFORMAL_OPERATOR_NORM_BOUND
    scale = (
        CONFORMAL_OPERATOR_NORM_BOUND
        * (1.0 - 1.0e-12)
        / maximum
        if applied
        else 1.0
    )
    projected = (coefficients * scale).contiguous()
    for index in unsupported:
        projected[index] = 0.0
    post = _endpoint_operator_norms(
        tuple(float(value) for value in projected)
    )
    if max(post) > CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12:
        raise RuntimeError("conformal endpoint projection failed")
    return projected, pre, post, scale, applied


def _resource_payload() -> dict[str, object]:
    return {
        "semantics": _CONFORMAL_ROUTE_SEMANTICS,
        "coefficient_order": _COEFFICIENT_ORDER,
        "learned_float_scalar_count": CONFORMAL_COEFFICIENT_COUNT,
        "derived_prepared_float_scalar_count": (
            _DERIVED_CONSTANT_FLOAT_COUNT
        ),
        "prepared_float_scalar_count": _PREPARED_FLOAT_SCALAR_COUNT,
        "logical_linear_macs_per_token_upper_bound": (
            CONFORMAL_LINEAR_MACS_PER_TOKEN
        ),
        "nonlinear_scalar_ops_per_token_upper_bound": (
            CONFORMAL_NONLINEAR_SCALAR_OPS_PER_TOKEN
        ),
        "linear_accumulator_scalar_ops_per_token_upper_bound": (
            _LINEAR_ACCUMULATOR_OPS_PER_TOKEN
        ),
        "zero_denominator_comparisons_per_token_upper_bound": (
            _ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN
        ),
        "runtime_state_float_scalars_per_sequence": (
            _RUNTIME_STATE_FLOAT_COUNT
        ),
        "parent_modal_values_reused": True,
        "parent_lag_cache_owned_upstream": True,
        "parent_decoder_invocations_per_token": 1,
    }


def gemma_causal_top2_conformal_route_provider_artifact_sha256(
    *,
    parent_artifact_sha256: str,
    parent_h4_artifact_sha256: str,
    bridge_binding_sha256: str,
    decoder_sha256: str,
    lag_kernel_sha256: str,
    fold_receipt_sha256: str,
    top_mode_indices: Sequence[int],
    top_mode_norms: Sequence[float],
    coefficients_by_conformal_coefficient: Sequence[float],
) -> str:
    """Replay a provider identity without loading parent tensors."""

    indices = _indices2(top_mode_indices, label="top_mode_indices")
    norms = _float2(top_mode_norms, label="top_mode_norms")
    if any(value <= 0.0 for value in norms):
        raise ValueError("top-mode norms must be positive")
    coefficients = _float4(
        coefficients_by_conformal_coefficient,
        label="coefficients_by_conformal_coefficient",
    )
    endpoint_norms = _endpoint_operator_norms(coefficients)
    if max(endpoint_norms) > CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12:
        raise ValueError("conformal coefficients exceed the endpoint bound")
    resources = _resource_payload()
    payload = {
        "semantics": _CONFORMAL_ROUTE_SEMANTICS,
        "route_state_semantics": _ROUTE_STATE_SEMANTICS,
        "site": _H4_SITE,
        "parent_artifact_sha256": _require_sha256(
            parent_artifact_sha256,
            label="parent artifact",
        ),
        "parent_h4_artifact_sha256": _require_sha256(
            parent_h4_artifact_sha256,
            label="parent H4 artifact",
        ),
        "bridge_binding_sha256": _require_sha256(
            bridge_binding_sha256,
            label="bridge binding",
        ),
        "decoder_sha256": _require_sha256(
            decoder_sha256,
            label="decoder",
        ),
        "lag_kernel_sha256": _require_sha256(
            lag_kernel_sha256,
            label="lag kernel",
        ),
        "fold_receipt_sha256": _require_sha256(
            fold_receipt_sha256,
            label="fold receipt",
        ),
        "top_mode_indices": indices,
        "top_mode_norms": norms,
        "coefficients_by_conformal_coefficient": coefficients,
        "coefficient_order": _COEFFICIENT_ORDER,
        "endpoint_order": _ENDPOINT_ORDER,
        "endpoint_operator_norms": endpoint_norms,
        "operator_norm_bound": CONFORMAL_OPERATOR_NORM_BOUND,
        "resources": resources,
        "resource_receipt_sha256": _sha256(
            _RESOURCE_DOMAIN,
            resources,
        ),
    }
    return _sha256(_PROVIDER_DOMAIN, payload)


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2ConformalRouteState:
    """The exact two-float causal balance carry from iteration two."""

    numerator: Tensor
    denominator: Tensor
    provider_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.provider_artifact_sha256,
            label="conformal-state provider",
        )
        if (
            not isinstance(self.numerator, Tensor)
            or not isinstance(self.denominator, Tensor)
            or self.numerator.ndim != 1
            or self.denominator.shape != self.numerator.shape
            or not self.numerator.is_floating_point()
            or self.denominator.dtype != self.numerator.dtype
            or self.denominator.device != self.numerator.device
            or not bool(torch.isfinite(self.numerator).all())
            or not bool(torch.isfinite(self.denominator).all())
            or bool((self.denominator < 0).any())
        ):
            raise ValueError("conformal-route runtime state is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.numerator.shape[0])

    def validate_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class GemmaIterativeConformalRouteFitRecord:
    """One prompt's tensor-free four-coordinate NLL linearization."""

    example_id: str
    family_id: str
    model_inputs_sha256: str
    parent_execution_sha256: str
    parent_observation_sha256: str
    parent_h4_artifact_sha256: str
    prefix_sha256: str
    gradient_sha256: str
    parent_modal_sha256: str
    balance_feature_sha256: str
    shared_gated_feature_sha256: str
    contrast_gated_feature_sha256: str
    supervised_tokens: int
    parent_signed_delta_nll_per_token: float
    jacobian_by_conformal_coefficient: tuple[
        float,
        float,
        float,
        float,
    ]
    active_row_count: int
    top_mode_indices: tuple[int, int]
    top_mode_norms: tuple[float, float]
    balance_feature_std: float
    top2_modal_energy_fraction: float
    fit_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="fit record example_id")
        _identifier(self.family_id, label="fit record family_id")
        for name in (
            "model_inputs_sha256",
            "parent_execution_sha256",
            "parent_observation_sha256",
            "parent_h4_artifact_sha256",
            "prefix_sha256",
            "gradient_sha256",
            "parent_modal_sha256",
            "balance_feature_sha256",
            "shared_gated_feature_sha256",
            "contrast_gated_feature_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"fit record {name}")
        if type(self.supervised_tokens) is not int or self.supervised_tokens <= 0:
            raise ValueError("fit record supervised_tokens must be positive")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fit record active_row_count must be positive")
        object.__setattr__(
            self,
            "parent_signed_delta_nll_per_token",
            _finite(
                self.parent_signed_delta_nll_per_token,
                label="parent signed delta NLL/token",
            ),
        )
        object.__setattr__(
            self,
            "jacobian_by_conformal_coefficient",
            _float4(
                self.jacobian_by_conformal_coefficient,
                label="jacobian_by_conformal_coefficient",
            ),
        )
        indices = _indices2(
            self.top_mode_indices,
            label="top_mode_indices",
        )
        norms = _float2(self.top_mode_norms, label="top_mode_norms")
        if any(value <= 0.0 for value in norms):
            raise ValueError("fit-record top-mode norms must be positive")
        object.__setattr__(self, "top_mode_indices", indices)
        object.__setattr__(self, "top_mode_norms", norms)
        balance_std = _finite(
            self.balance_feature_std,
            label="balance_feature_std",
        )
        energy = _finite(
            self.top2_modal_energy_fraction,
            label="top2_modal_energy_fraction",
        )
        if balance_std < 0.0 or not 0.0 <= energy <= 1.0 + 1.0e-12:
            raise ValueError("fit-record conformal feature receipt is invalid")
        object.__setattr__(self, "balance_feature_std", balance_std)
        object.__setattr__(
            self,
            "top2_modal_energy_fraction",
            min(1.0, energy),
        )
        object.__setattr__(
            self,
            "fit_record_sha256",
            _sha256(_FIT_RECORD_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "model_inputs_sha256": self.model_inputs_sha256,
            "parent_execution_sha256": self.parent_execution_sha256,
            "parent_observation_sha256": self.parent_observation_sha256,
            "parent_h4_artifact_sha256": self.parent_h4_artifact_sha256,
            "prefix_sha256": self.prefix_sha256,
            "gradient_sha256": self.gradient_sha256,
            "parent_modal_sha256": self.parent_modal_sha256,
            "balance_feature_sha256": self.balance_feature_sha256,
            "shared_gated_feature_sha256": (
                self.shared_gated_feature_sha256
            ),
            "contrast_gated_feature_sha256": (
                self.contrast_gated_feature_sha256
            ),
            "supervised_tokens": self.supervised_tokens,
            "parent_signed_delta_nll_per_token": (
                self.parent_signed_delta_nll_per_token
            ),
            "jacobian_by_conformal_coefficient": (
                self.jacobian_by_conformal_coefficient
            ),
            "active_row_count": self.active_row_count,
            "top_mode_indices": self.top_mode_indices,
            "top_mode_norms": self.top_mode_norms,
            "balance_feature_std": self.balance_feature_std,
            "top2_modal_energy_fraction": (
                self.top2_modal_energy_fraction
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fit_record_sha256": self.fit_record_sha256,
        }


def _record(value: object) -> GemmaIterativeConformalRouteFitRecord:
    if isinstance(value, GemmaIterativeConformalRouteFitRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("fit records must be mappings or strict records")
    expected = {
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "parent_execution_sha256",
        "parent_observation_sha256",
        "parent_h4_artifact_sha256",
        "prefix_sha256",
        "gradient_sha256",
        "parent_modal_sha256",
        "balance_feature_sha256",
        "shared_gated_feature_sha256",
        "contrast_gated_feature_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_conformal_coefficient",
        "active_row_count",
        "top_mode_indices",
        "top_mode_norms",
        "balance_feature_std",
        "top2_modal_energy_fraction",
        "fit_record_sha256",
    }
    if set(value) != expected:
        raise ValueError("serialized conformal fit-record fields differ")
    result = GemmaIterativeConformalRouteFitRecord(
        example_id=value["example_id"],  # type: ignore[arg-type]
        family_id=value["family_id"],  # type: ignore[arg-type]
        model_inputs_sha256=value[  # type: ignore[arg-type]
            "model_inputs_sha256"
        ],
        parent_execution_sha256=value[  # type: ignore[arg-type]
            "parent_execution_sha256"
        ],
        parent_observation_sha256=value[  # type: ignore[arg-type]
            "parent_observation_sha256"
        ],
        parent_h4_artifact_sha256=value[  # type: ignore[arg-type]
            "parent_h4_artifact_sha256"
        ],
        prefix_sha256=value["prefix_sha256"],  # type: ignore[arg-type]
        gradient_sha256=value["gradient_sha256"],  # type: ignore[arg-type]
        parent_modal_sha256=value["parent_modal_sha256"],  # type: ignore[arg-type]
        balance_feature_sha256=value[  # type: ignore[arg-type]
            "balance_feature_sha256"
        ],
        shared_gated_feature_sha256=value[  # type: ignore[arg-type]
            "shared_gated_feature_sha256"
        ],
        contrast_gated_feature_sha256=value[  # type: ignore[arg-type]
            "contrast_gated_feature_sha256"
        ],
        supervised_tokens=value["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=value[  # type: ignore[arg-type]
            "parent_signed_delta_nll_per_token"
        ],
        jacobian_by_conformal_coefficient=value[  # type: ignore[arg-type]
            "jacobian_by_conformal_coefficient"
        ],
        active_row_count=value["active_row_count"],  # type: ignore[arg-type]
        top_mode_indices=value["top_mode_indices"],  # type: ignore[arg-type]
        top_mode_norms=value["top_mode_norms"],  # type: ignore[arg-type]
        balance_feature_std=value["balance_feature_std"],  # type: ignore[arg-type]
        top2_modal_energy_fraction=value[  # type: ignore[arg-type]
            "top2_modal_energy_fraction"
        ],
    )
    if result.fit_record_sha256 != value["fit_record_sha256"]:
        raise ValueError("conformal fit-record hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeConformalRouteFoldFit:
    """Replayable family-balanced fit with a global endpoint projection."""

    held_family_id: str
    train_example_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    train_fit_record_sha256s: tuple[str, ...]
    coefficients_by_conformal_coefficient: tuple[
        float,
        float,
        float,
        float,
    ]
    unsupported_conformal_coefficient_indices: tuple[int, ...]
    active_row_count: int
    weighted_column_norm_by_conformal_coefficient: tuple[
        float,
        float,
        float,
        float,
    ]
    weighted_design_rank: int
    normal_condition_number: float
    pre_projection_endpoint_operator_norms: tuple[float, float]
    post_projection_endpoint_operator_norms: tuple[float, float]
    trust_projection_scale: float
    linearized_rmse_before: float
    linearized_rmse_after: float
    trust_projection_applied: bool
    ridge: float = CONFORMAL_RIDGE
    operator_norm_bound: float = CONFORMAL_OPERATOR_NORM_BOUND
    fold_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.held_family_id != "__full_fit__":
            _identifier(self.held_family_id, label="held_family_id")
        for name in (
            "train_example_ids",
            "train_family_ids",
            "train_fit_record_sha256s",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be canonical and unique")
        for value in self.train_fit_record_sha256s:
            _require_sha256(value, label="training fit-record receipt")
        coefficients = _float4(
            self.coefficients_by_conformal_coefficient,
            label="coefficients_by_conformal_coefficient",
        )
        object.__setattr__(
            self,
            "coefficients_by_conformal_coefficient",
            coefficients,
        )
        if (
            type(self.unsupported_conformal_coefficient_indices) is not tuple
            or self.unsupported_conformal_coefficient_indices
            != tuple(
                sorted(
                    set(self.unsupported_conformal_coefficient_indices)
                )
            )
            or any(
                type(index) is not int
                or not 0 <= index < CONFORMAL_COEFFICIENT_COUNT
                for index in self.unsupported_conformal_coefficient_indices
            )
        ):
            raise ValueError(
                "unsupported conformal coefficients must be canonical"
            )
        if any(
            coefficients[index] != 0.0
            for index in self.unsupported_conformal_coefficient_indices
        ):
            raise ValueError("unsupported conformal coefficients must be zero")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fold active_row_count must be positive")
        column_norms = _float4(
            self.weighted_column_norm_by_conformal_coefficient,
            label="weighted_column_norm_by_conformal_coefficient",
        )
        if any(value < 0.0 for value in column_norms):
            raise ValueError("weighted column norms must be nonnegative")
        object.__setattr__(
            self,
            "weighted_column_norm_by_conformal_coefficient",
            column_norms,
        )
        if (
            type(self.weighted_design_rank) is not int
            or not 0 <= self.weighted_design_rank
            <= CONFORMAL_COEFFICIENT_COUNT
        ):
            raise ValueError("weighted conformal design rank is invalid")
        for name in (
            "normal_condition_number",
            "trust_projection_scale",
            "linearized_rmse_before",
            "linearized_rmse_after",
            "ridge",
            "operator_norm_bound",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        pre = _float2(
            self.pre_projection_endpoint_operator_norms,
            label="pre_projection_endpoint_operator_norms",
        )
        post = _float2(
            self.post_projection_endpoint_operator_norms,
            label="post_projection_endpoint_operator_norms",
        )
        if any(value < 0.0 for value in (*pre, *post)):
            raise ValueError("endpoint operator norms must be nonnegative")
        object.__setattr__(
            self,
            "pre_projection_endpoint_operator_norms",
            pre,
        )
        object.__setattr__(
            self,
            "post_projection_endpoint_operator_norms",
            post,
        )
        if (
            self.ridge != CONFORMAL_RIDGE
            or self.operator_norm_bound != CONFORMAL_OPERATOR_NORM_BOUND
        ):
            raise ValueError("the frozen conformal route cannot be retuned")
        observed_post = _endpoint_operator_norms(coefficients)
        if any(
            abs(left - right) > 1.0e-10
            for left, right in zip(post, observed_post, strict=True)
        ):
            raise ValueError("reported post-projection endpoint norms differ")
        if max(post) > self.operator_norm_bound + 1.0e-12:
            raise ValueError("fitted conformal route exceeds its trust bound")
        if type(self.trust_projection_applied) is not bool:
            raise TypeError("trust_projection_applied must be boolean")
        expected_applied = max(pre) > self.operator_norm_bound
        if self.trust_projection_applied != expected_applied:
            raise ValueError("trust projection receipt is inconsistent")
        expected_scale = (
            self.operator_norm_bound
            * (1.0 - 1.0e-12)
            / max(pre)
            if expected_applied
            else 1.0
        )
        if abs(self.trust_projection_scale - expected_scale) > 1.0e-12:
            raise ValueError("trust projection scale receipt differs")
        if any(
            abs(after - before * self.trust_projection_scale) > 1.0e-10
            for before, after in zip(pre, post, strict=True)
        ):
            raise ValueError("endpoint projection norm receipts differ")
        object.__setattr__(
            self,
            "fold_receipt_sha256",
            _sha256(_FOLD_FIT_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "held_family_id": self.held_family_id,
            "train_example_ids": self.train_example_ids,
            "train_family_ids": self.train_family_ids,
            "train_fit_record_sha256s": self.train_fit_record_sha256s,
            "coefficients_by_conformal_coefficient": (
                self.coefficients_by_conformal_coefficient
            ),
            "unsupported_conformal_coefficient_indices": (
                self.unsupported_conformal_coefficient_indices
            ),
            "active_row_count": self.active_row_count,
            "weighted_column_norm_by_conformal_coefficient": (
                self.weighted_column_norm_by_conformal_coefficient
            ),
            "weighted_design_rank": self.weighted_design_rank,
            "normal_condition_number": self.normal_condition_number,
            "pre_projection_endpoint_operator_norms": (
                self.pre_projection_endpoint_operator_norms
            ),
            "post_projection_endpoint_operator_norms": (
                self.post_projection_endpoint_operator_norms
            ),
            "trust_projection_scale": self.trust_projection_scale,
            "linearized_rmse_before": self.linearized_rmse_before,
            "linearized_rmse_after": self.linearized_rmse_after,
            "trust_projection_applied": self.trust_projection_applied,
            "ridge": self.ridge,
            "operator_norm_bound": self.operator_norm_bound,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fold_receipt_sha256": self.fold_receipt_sha256,
        }


def fit_gemma_iterative_conformal_route_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeConformalRouteFoldFit:
    """Fit four prompt derivatives with equal family and prompt mass."""

    selected = tuple(
        sorted(
            (_record(value) for value in records),
            key=lambda row: row.example_id,
        )
    )
    if not selected or len({row.example_id for row in selected}) != len(selected):
        raise ValueError("fit records must be nonempty and unique")
    family_counts = Counter(row.family_id for row in selected)
    if held_family_id != "__full_fit__" and held_family_id in family_counts:
        raise ValueError("the held family leaked into its training records")
    if (
        len({row.parent_h4_artifact_sha256 for row in selected}) != 1
        or len({row.top_mode_indices for row in selected}) != 1
        or len({row.top_mode_norms for row in selected}) != 1
    ):
        raise ValueError("fit records belong to different conformal features")

    design = torch.tensor(
        [row.jacobian_by_conformal_coefficient for row in selected],
        dtype=torch.float64,
    )
    target = -torch.tensor(
        [row.parent_signed_delta_nll_per_token for row in selected],
        dtype=torch.float64,
    )
    family_mass = 1.0 / len(family_counts)
    weights = torch.tensor(
        [
            family_mass / family_counts[row.family_id]
            for row in selected
        ],
        dtype=torch.float64,
    )
    weighted_design = weights.sqrt().unsqueeze(1) * design
    weighted_design_rank = int(
        (
            torch.linalg.svdvals(weighted_design)
            > _DESIGN_RANK_TOLERANCE
        ).sum()
    )
    column_norms = torch.sqrt(
        (weights[:, None] * design.square()).sum(dim=0)
    )
    supported = tuple(
        index
        for index in range(CONFORMAL_COEFFICIENT_COUNT)
        if float(column_norms[index]) > _COLUMN_SUPPORT_EPSILON
    )
    coefficients = torch.zeros(
        CONFORMAL_COEFFICIENT_COUNT,
        dtype=torch.float64,
    )
    condition = 0.0
    if supported:
        indices = torch.tensor(supported, dtype=torch.int64)
        x = design.index_select(1, indices)
        supported_singular_values = torch.linalg.svdvals(
            weights.sqrt().unsqueeze(1) * x
        )
        numerically_supported_singular_values = (
            supported_singular_values[
                supported_singular_values > _DESIGN_RANK_TOLERANCE
            ]
        )
        condition = (
            float(
                (
                    numerically_supported_singular_values.max()
                    / numerically_supported_singular_values.min()
                ).square()
            )
            if numerically_supported_singular_values.numel()
            else 0.0
        )
        normal = x.T @ (weights[:, None] * x)
        solved = torch.linalg.solve(
            normal
            + CONFORMAL_RIDGE
            * torch.eye(len(supported), dtype=torch.float64),
            x.T @ (weights * target),
        )
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("conformal-route ridge fit became nonfinite")
        coefficients[indices] = solved

    (
        coefficients,
        pre_endpoint_norms,
        post_endpoint_norms,
        projection_scale,
        projection_applied,
    ) = _project_conformal_coefficients(
        coefficients,
        supported=supported,
    )
    before = float(torch.sqrt((weights * target.square()).sum()))
    after = float(
        torch.sqrt(
            (
                weights
                * (design @ coefficients - target).square()
            ).sum()
        )
    )
    return GemmaIterativeConformalRouteFoldFit(
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_conformal_coefficient=tuple(
            float(value) for value in coefficients
        ),  # type: ignore[arg-type]
        unsupported_conformal_coefficient_indices=tuple(
            index
            for index in range(CONFORMAL_COEFFICIENT_COUNT)
            if index not in supported
        ),
        active_row_count=sum(row.active_row_count for row in selected),
        weighted_column_norm_by_conformal_coefficient=tuple(
            float(value) for value in column_norms
        ),  # type: ignore[arg-type]
        weighted_design_rank=weighted_design_rank,
        normal_condition_number=condition,
        pre_projection_endpoint_operator_norms=pre_endpoint_norms,
        post_projection_endpoint_operator_norms=post_endpoint_norms,
        trust_projection_scale=projection_scale,
        linearized_rmse_before=before,
        linearized_rmse_after=after,
        trust_projection_applied=projection_applied,
    )


def _conformal_jacobian(
    feature: Tensor,
    gradient_modes: Tensor,
    active: Tensor,
) -> tuple[Tensor, Tensor]:
    selected_feature = feature[active]
    selected_gradient = gradient_modes[active]
    real = (
        selected_feature[:, 0] * selected_gradient[:, 0]
        + selected_feature[:, 1] * selected_gradient[:, 1]
    ).sum()
    imag = (
        selected_feature[:, 1] * selected_gradient[:, 0]
        - selected_feature[:, 0] * selected_gradient[:, 1]
    ).sum()
    return real, imag


def build_gemma_iterative_conformal_route_fit_record(
    *,
    example: object,
    parent_execution: object,
    gradient: Tensor,
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeConformalRouteFitRecord:
    """Reduce one exact parent H4 NLL-VJP to four conformal derivatives."""

    parent = _source_only_parent(parent_h4)
    if not isinstance(
        parent_observation,
        GemmaH4DampingFiniteNLLObservation,
    ):
        raise TypeError("parent_observation has the wrong type")
    validate = getattr(parent_execution, "validate_integrity", None)
    if not callable(validate):
        raise TypeError("parent execution lacks integrity validation")
    validate()
    prefix = getattr(parent_execution, "prefix", None)
    candidate_h4 = getattr(parent_execution, "candidate_h4", None)
    if not isinstance(prefix, Gemma3L3L4OnePassPrefix):
        raise TypeError("parent execution omitted its authenticated prefix")
    prefix.validate_integrity()
    if (
        not isinstance(candidate_h4, Tensor)
        or not isinstance(gradient, Tensor)
        or gradient.shape != candidate_h4.shape
        or gradient.shape != prefix.clamped_y3.shape
        or not gradient.is_floating_point()
        or not candidate_h4.is_floating_point()
    ):
        raise ValueError("conformal behavior-gradient geometry differs")
    active = prefix.target_affected_mask
    active_gradient = active.to(gradient.device)
    if (
        not bool(active.any())
        or not bool(torch.isfinite(gradient[active_gradient]).all())
    ):
        raise ValueError("conformal behavior gradient is invalid")

    example_id = _identifier(
        getattr(example, "example_id", None),
        label="example_id",
    )
    family_id = _identifier(
        getattr(example, "family_id", None),
        label="family_id",
    )
    model_inputs_sha256 = _require_sha256(
        getattr(example, "model_inputs_sha256", None),
        label="example model inputs",
    )
    if (
        parent_observation.example_id != example_id
        or parent_observation.family_id != family_id
        or getattr(parent_execution, "model_inputs_sha256", None)
        != model_inputs_sha256
        or getattr(
            parent_execution,
            "h4_head_sha256",
            parent.artifact_sha256,
        )
        != parent.artifact_sha256
        or prefix.bridge_binding_sha256 != parent.bridge_binding_sha256
    ):
        raise ValueError("conformal fit-record identities differ")

    parent_modal = parent.modal_correction(prefix, candidate_h4)
    top_indices, top_norms = top2_lag_b_output_modes(parent)
    zeros = torch.zeros(
        parent_modal.shape[0],
        device=parent_modal.device,
        dtype=parent_modal.dtype,
    )
    balance, _numerator, _denominator = _balance_feature(
        prefix=prefix,
        parent_modal=parent_modal,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        initial_numerator=zeros,
        initial_denominator=zeros,
    )
    selected_indices = torch.tensor(
        top_indices,
        device=parent_modal.device,
        dtype=torch.int64,
    )
    selected_modal = parent_modal.index_select(2, selected_indices)
    shared_feature = balance.unsqueeze(-1) * selected_modal
    contrast_feature = balance.unsqueeze(-1) * shared_feature
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(top_indices, dtype=torch.int64),
    ).to(device=gradient.device, dtype=torch.float64)
    gradient_modes = gradient.to(torch.float64) @ decoder.T
    feature_active = active.to(shared_feature.device)
    shared_real, shared_imag = _conformal_jacobian(
        shared_feature.to(
            device=gradient_modes.device,
            dtype=torch.float64,
        ),
        gradient_modes,
        active_gradient,
    )
    contrast_real, contrast_imag = _conformal_jacobian(
        contrast_feature.to(
            device=gradient_modes.device,
            dtype=torch.float64,
        ),
        gradient_modes,
        active_gradient,
    )
    jacobian = (
        torch.stack(
            (
                shared_real,
                shared_imag,
                contrast_real,
                contrast_imag,
            )
        )
        / parent_observation.supervised_tokens
    )
    active_balance = balance[
        active.to(balance.device)
    ].to(torch.float64)
    balance_std = (
        0.0
        if active_balance.numel() <= 1
        else float(active_balance.std(unbiased=False))
    )
    parent_active = parent_modal[
        active.to(parent_modal.device)
    ].to(torch.float64)
    top_active = selected_modal[
        active.to(selected_modal.device)
    ].to(torch.float64)
    total_energy = float(parent_active.square().sum())
    top_energy = float(top_active.square().sum())
    signed = (
        parent_observation.candidate_summed_nll
        - parent_observation.source_summed_nll
    ) / parent_observation.supervised_tokens
    if (
        not bool(torch.isfinite(shared_feature[feature_active]).all())
        or not bool(torch.isfinite(contrast_feature[feature_active]).all())
        or not bool(torch.isfinite(jacobian).all())
    ):
        raise ValueError("conformal fit features became nonfinite")
    return GemmaIterativeConformalRouteFitRecord(
        example_id=example_id,
        family_id=family_id,
        model_inputs_sha256=model_inputs_sha256,
        parent_execution_sha256=_require_sha256(
            getattr(parent_execution, "artifact_sha256", None),
            label="parent execution",
        ),
        parent_observation_sha256=parent_observation.observation_sha256,
        parent_h4_artifact_sha256=parent.artifact_sha256,
        prefix_sha256=prefix.artifact_sha256,
        gradient_sha256=_tensor_sha256(gradient),
        parent_modal_sha256=_tensor_sha256(parent_modal),
        balance_feature_sha256=_tensor_sha256(balance),
        shared_gated_feature_sha256=_tensor_sha256(shared_feature),
        contrast_gated_feature_sha256=_tensor_sha256(contrast_feature),
        supervised_tokens=parent_observation.supervised_tokens,
        parent_signed_delta_nll_per_token=signed,
        jacobian_by_conformal_coefficient=tuple(
            float(value) for value in jacobian
        ),  # type: ignore[arg-type]
        active_row_count=int(active.sum()),
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        balance_feature_std=balance_std,
        top2_modal_energy_fraction=(
            0.0 if total_energy == 0.0 else top_energy / total_energy
        ),
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2ConformalRouteH4Provider(
    Gemma3L3L4CorrectionProvider
):
    """Authenticated affine conformal route over source-only lag-B modes."""

    parent_h4: GemmaCausalResidualHead
    parent_artifact_sha256: str
    fold_fit: GemmaIterativeConformalRouteFoldFit
    site: str = field(init=False, default=_H4_SITE)
    artifact_sha256: str = field(init=False)
    _parent_h4_sha256: str = field(init=False, repr=False)
    _bridge_binding_sha256: str = field(init=False, repr=False)
    _decoder_sha256: str = field(init=False, repr=False)
    _lag_kernel_sha256: str = field(init=False, repr=False)
    _top_mode_indices: tuple[int, int] = field(init=False, repr=False)
    _top_mode_norms: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parent = _source_only_parent(self.parent_h4)
        if not isinstance(
            self.fold_fit,
            GemmaIterativeConformalRouteFoldFit,
        ):
            raise TypeError("fold_fit must be a strict conformal-route fit")
        parent_h4_sha256 = _require_sha256(
            parent.artifact_sha256,
            label="parent H4 artifact",
        )
        parent_artifact_sha256 = _require_sha256(
            self.parent_artifact_sha256,
            label="parent artifact",
        )
        bridge_binding = _require_sha256(
            parent.bridge_binding_sha256,
            label="bridge binding",
        )
        decoder_sha256 = _tensor_sha256(parent.decoder)
        lag_kernel_sha256 = _tensor_sha256(parent.lag_kernel)
        top_indices, top_norms = top2_lag_b_output_modes(parent)
        object.__setattr__(self, "_parent_h4_sha256", parent_h4_sha256)
        object.__setattr__(self, "_bridge_binding_sha256", bridge_binding)
        object.__setattr__(self, "_decoder_sha256", decoder_sha256)
        object.__setattr__(self, "_lag_kernel_sha256", lag_kernel_sha256)
        object.__setattr__(self, "_top_mode_indices", top_indices)
        object.__setattr__(self, "_top_mode_norms", top_norms)
        object.__setattr__(
            self,
            "artifact_sha256",
            gemma_causal_top2_conformal_route_provider_artifact_sha256(
                parent_artifact_sha256=parent_artifact_sha256,
                parent_h4_artifact_sha256=parent_h4_sha256,
                bridge_binding_sha256=bridge_binding,
                decoder_sha256=decoder_sha256,
                lag_kernel_sha256=lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=top_indices,
                top_mode_norms=top_norms,
                coefficients_by_conformal_coefficient=(
                    self.coefficients_by_conformal_coefficient
                ),
            ),
        )
        self.validate_integrity()

    @property
    def bridge_binding_sha256(self) -> str:
        return self._bridge_binding_sha256

    @property
    def coefficients_by_conformal_coefficient(
        self,
    ) -> tuple[float, float, float, float]:
        return self.fold_fit.coefficients_by_conformal_coefficient

    @property
    def conformal_coefficient_order(self) -> tuple[str, str, str, str]:
        return _COEFFICIENT_ORDER

    @property
    def top_mode_indices(self) -> tuple[int, int]:
        return self._top_mode_indices

    @property
    def top_mode_norms(self) -> tuple[float, float]:
        return self._top_mode_norms

    @property
    def decoder_sha256(self) -> str:
        return self._decoder_sha256

    @property
    def lag_kernel_sha256(self) -> str:
        return self._lag_kernel_sha256

    @property
    def endpoint_operator_norms(self) -> tuple[float, float]:
        return _endpoint_operator_norms(
            self.coefficients_by_conformal_coefficient
        )

    @property
    def route_state_semantics(self) -> str:
        return _ROUTE_STATE_SEMANTICS

    @property
    def conformal_route_semantics(self) -> str:
        return _CONFORMAL_ROUTE_SEMANTICS

    @property
    def width(self) -> int:
        return self.parent_h4.width

    @property
    def marginal_learned_float_scalar_count(self) -> int:
        return CONFORMAL_COEFFICIENT_COUNT

    @property
    def marginal_derived_prepared_float_scalar_count(self) -> int:
        return _DERIVED_CONSTANT_FLOAT_COUNT

    @property
    def marginal_prepared_float_scalar_count(self) -> int:
        return _PREPARED_FLOAT_SCALAR_COUNT

    @property
    def marginal_logical_macs_per_token_upper_bound(self) -> int:
        return CONFORMAL_LINEAR_MACS_PER_TOKEN

    @property
    def nonlinear_scalar_ops_per_token_upper_bound(self) -> int:
        return CONFORMAL_NONLINEAR_SCALAR_OPS_PER_TOKEN

    @property
    def linear_accumulator_scalar_ops_per_token_upper_bound(self) -> int:
        return _LINEAR_ACCUMULATOR_OPS_PER_TOKEN

    @property
    def zero_denominator_comparisons_per_token_upper_bound(self) -> int:
        return _ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN

    @property
    def runtime_state_float_scalars_per_sequence(self) -> int:
        return _RUNTIME_STATE_FLOAT_COUNT

    @property
    def parent_decoder_invocations_per_token(self) -> int:
        return 1

    @property
    def resource_receipt(self) -> Mapping[str, object]:
        return _resource_payload()

    @property
    def resource_receipt_sha256(self) -> str:
        return _sha256(_RESOURCE_DOMAIN, self.resource_receipt)

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_h4.prepared_float_scalar_count
            + self.marginal_prepared_float_scalar_count
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_h4.logical_macs_per_token_upper_bound
            + self.marginal_logical_macs_per_token_upper_bound
        )

    def validate_integrity(self) -> None:
        parent = _source_only_parent(self.parent_h4)
        top_indices, top_norms = top2_lag_b_output_modes(parent)
        if (
            parent.artifact_sha256 != self._parent_h4_sha256
            or parent.bridge_binding_sha256 != self._bridge_binding_sha256
            or _tensor_sha256(parent.decoder) != self._decoder_sha256
            or _tensor_sha256(parent.lag_kernel)
            != self._lag_kernel_sha256
            or top_indices != self._top_mode_indices
            or top_norms != self._top_mode_norms
            or gemma_causal_top2_conformal_route_provider_artifact_sha256(
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=self._parent_h4_sha256,
                bridge_binding_sha256=self._bridge_binding_sha256,
                decoder_sha256=self._decoder_sha256,
                lag_kernel_sha256=self._lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=self._top_mode_indices,
                top_mode_norms=self._top_mode_norms,
                coefficients_by_conformal_coefficient=(
                    self.coefficients_by_conformal_coefficient
                ),
            )
            != self.artifact_sha256
            or self.resource_receipt_sha256
            != _sha256(_RESOURCE_DOMAIN, _resource_payload())
        ):
            raise RuntimeError("causal top-two conformal provider drifted")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> GemmaCausalTop2ConformalRouteState:
        self.validate_integrity()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("conformal-route batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("conformal-route state dtype must be floating point")
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        return GemmaCausalTop2ConformalRouteState(
            numerator=zeros,
            denominator=zeros.clone(),
            provider_artifact_sha256=self.artifact_sha256,
        )

    def route_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2ConformalRouteState,
    ) -> tuple[Tensor, GemmaCausalTop2ConformalRouteState]:
        """Route one upstream lag-aware parent-modal chunk and advance carry."""

        self.validate_integrity()
        prefix.validate_integrity()
        if not isinstance(state, GemmaCausalTop2ConformalRouteState):
            raise TypeError("state must be a conformal-route runtime state")
        state.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.shape
            != (*prefix.logical_positions.shape, self.parent_h4.rank)
            or not parent_modal.is_floating_point()
            or state.provider_artifact_sha256 != self.artifact_sha256
            or state.batch_size != parent_modal.shape[0]
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
        ):
            raise ValueError("state, parent modal chunk, and prefix differ")
        active = prefix.target_affected_mask.to(parent_modal.device)
        if (
            bool(active.any())
            and not bool(torch.isfinite(parent_modal[active]).all())
        ):
            raise ValueError("parent modal chunk is nonfinite")
        inactive = ~active
        if (
            bool(inactive.any())
            and not bool((parent_modal[inactive] == 0).all())
        ):
            raise ValueError("parent modal chunk is off support")
        balance, numerator, denominator = _balance_feature(
            prefix=prefix,
            parent_modal=parent_modal,
            top_mode_indices=self.top_mode_indices,
            top_mode_norms=self.top_mode_norms,
            initial_numerator=state.numerator,
            initial_denominator=state.denominator,
        )
        if (
            bool(active.any())
            and bool((balance[active].abs() > 1.0 + 1.0e-6).any())
        ):
            raise RuntimeError("causal balance escaped its normalized range")
        selected_indices = torch.tensor(
            self.top_mode_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        )
        selected = parent_modal.index_select(2, selected_indices)
        a0, b0, a1, b1 = (
            torch.tensor(
                self.coefficients_by_conformal_coefficient,
                device=parent_modal.device,
                dtype=parent_modal.dtype,
            ).unbind()
        )
        a = a0 + balance * a1
        b = b0 + balance * b1
        gated = balance.unsqueeze(-1) * selected
        delta_top = torch.stack(
            (
                gated[..., 0] * a + gated[..., 1] * b,
                -gated[..., 0] * b + gated[..., 1] * a,
            ),
            dim=-1,
        )
        routed = parent_modal.clone()
        if any(
            value != 0.0
            for value in self.coefficients_by_conformal_coefficient
        ):
            routed.index_copy_(
                2,
                selected_indices,
                selected + delta_top,
            )
        next_state = GemmaCausalTop2ConformalRouteState(
            numerator=numerator.detach().contiguous(),
            denominator=denominator.detach().contiguous(),
            provider_artifact_sha256=self.artifact_sha256,
        )
        if (
            bool(active.any())
            and not bool(torch.isfinite(routed[active]).all())
        ):
            raise ValueError("conformal route became nonfinite")
        if (
            bool(inactive.any())
            and not bool((routed[inactive] == 0).all())
        ):
            raise RuntimeError("conformal route is off support")
        self.validate_integrity()
        prefix.validate_integrity()
        return routed, next_state

    def correction_from_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2ConformalRouteState,
    ) -> tuple[Tensor, GemmaCausalTop2ConformalRouteState]:
        """Decode an upstream lag-aware modal chunk without resetting lag."""

        self.validate_integrity()
        prefix_sha256 = prefix.artifact_sha256
        routed, next_state = self.route_parent_modal_with_state(
            prefix,
            parent_modal,
            state,
        )
        result = self.parent_h4.decode_modal(
            prefix,
            routed,
            like=prefix.clamped_y3,
        )
        self.parent_h4.validate_integrity()
        prefix.validate_integrity()
        if prefix.artifact_sha256 != prefix_sha256:
            raise RuntimeError("conformal route mutated the prefix")
        self.validate_integrity()
        return result, next_state

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Use zero balance carry for the full-sequence provider bridge ABI."""

        self.validate_integrity()
        parent_modal = self.parent_h4.modal_correction(
            prefix,
            realized_state,
        )
        state = self.initial_state(
            prefix.logical_positions.shape[0],
            device=parent_modal.device,
            dtype=parent_modal.dtype,
        )
        result, _next_state = self.correction_from_parent_modal_with_state(
            prefix,
            parent_modal,
            state,
        )
        return result


def fit_gemma_iterative_conformal_route_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2ConformalRouteH4Provider:
    """Fit one OOF conformal-route provider."""

    fit = fit_gemma_iterative_conformal_route_fold(
        records,
        held_family_id=held_family,
    )
    return GemmaCausalTop2ConformalRouteH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_conformal_route_full_provider(
    *,
    records: Sequence[object],
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2ConformalRouteH4Provider:
    """Fit a full-data provider only after external retention."""

    return fit_gemma_iterative_conformal_route_fold_provider(
        records=records,
        held_family="__full_fit__",
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )
