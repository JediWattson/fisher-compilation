"""Two-regime causal modal experts over the frozen Gemma lag-B parent.

Iteration three preserves the top-two causal balance state introduced by the
single-router rung, but stops asking one matrix to serve both signs of that
state.  A negative balance executes the negative ``2 x 2`` expert and a zero
or positive balance executes the nonnegative expert:

``delta_top_t = (g_t * parent_modal_top2_t) @ theta_regime(t)``.

Only the selected expert executes for a row.  Both experts are independently
bounded in operator norm and zero coefficients reproduce the parent exactly.
The explicit parent-modal routing API makes lag-cache ownership an upstream
boundary rather than silently resetting the lag-B convolution across chunks.
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
    "EXPERT_EDGE_COUNT",
    "EXPERT_LINEAR_MACS_PER_TOKEN",
    "EXPERT_NONLINEAR_SCALAR_OPS_PER_TOKEN",
    "GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE",
    "GemmaCausalTop2StateExpertsH4Provider",
    "GemmaCausalTop2StateExpertsState",
    "GemmaIterativeStateExpertsFitRecord",
    "GemmaIterativeStateExpertsFoldFit",
    "build_gemma_iterative_state_experts_fit_record",
    "fit_gemma_iterative_state_experts_fold",
    "fit_gemma_iterative_state_experts_fold_provider",
    "fit_gemma_iterative_state_experts_full_provider",
    "gemma_causal_top2_state_experts_provider_artifact_sha256",
]


EXPERT_EDGE_COUNT = 8
EXPERT_LINEAR_MACS_PER_TOKEN = 6
EXPERT_NONLINEAR_SCALAR_OPS_PER_TOKEN = 6
_EXPERT_PREPARED_FLOAT_SCALAR_COUNT = 10
_EXPERT_DERIVED_CONSTANT_FLOAT_COUNT = 2
_EXPERT_RUNTIME_STATE_FLOAT_COUNT = 2
_EXPERT_LINEAR_ACCUMULATOR_OPS_PER_TOKEN = 4
_EXPERT_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN = 1
_EXPERT_REGIME_COMPARISONS_PER_TOKEN = 1
_COLUMN_SUPPORT_EPSILON = 1.0e-12
_DESIGN_RANK_TOLERANCE = 1.0e-12
_H4_SITE = "layer.4.output"
_ROUTE_STATE_SEMANTICS = (
    "top2_parent_lag_b_modal_cumulative_balance_v1"
)
_EXPERT_DISPATCH_SEMANTICS = (
    "negative_if_balance_lt_0_else_nonnegative"
)
_EXPERT_REGIME_ORDER = ("negative", "nonnegative")
_EXPERT_ROUTE_EDGE_ORDER = (
    "negative_0_to_0",
    "negative_0_to_1",
    "negative_1_to_0",
    "negative_1_to_1",
    "nonnegative_0_to_0",
    "nonnegative_0_to_1",
    "nonnegative_1_to_0",
    "nonnegative_1_to_1",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_RECORD_DOMAIN = b"fisher-graph:gemma-state-experts-fit-record:v1\0"
_FOLD_FIT_DOMAIN = b"fisher-graph:gemma-state-experts-fold-fit:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma-top2-state-experts-provider:v1\0"
_RESOURCE_DOMAIN = b"fisher-graph:gemma-top2-state-experts-resources:v1\0"


GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE = (
    GemmaIterativeResidualCampaignRecipe(
        recipe_id="causal_top2_state_experts",
        fit_record_jacobian_field="jacobian_by_expert_route_edge",
        fold_coefficient_field="coefficients_by_expert_route_edge",
        coefficient_count=EXPERT_EDGE_COUNT,
        learned_parameter_attribute=(
            "marginal_learned_float_scalar_count"
        ),
        learned_parameter_fallback_attribute=None,
        expected_learned_parameter_count=EXPERT_EDGE_COUNT,
        logical_macs_attribute=(
            "marginal_logical_macs_per_token_upper_bound"
        ),
        logical_macs_fallback_attribute=None,
        expected_logical_macs_per_token_upper_bound=(
            EXPERT_LINEAR_MACS_PER_TOKEN
        ),
        logical_macs_must_equal_residual_width=False,
        extra_resource_expectations=(
            (
                "derived_constant_float_count",
                "marginal_derived_prepared_float_scalar_count",
                _EXPERT_DERIVED_CONSTANT_FLOAT_COUNT,
            ),
            (
                "runtime_state_float_count_per_sequence",
                "runtime_state_float_scalars_per_sequence",
                _EXPERT_RUNTIME_STATE_FLOAT_COUNT,
            ),
            (
                "nonlinear_scalar_ops_per_token_upper_bound",
                "nonlinear_scalar_ops_per_token_upper_bound",
                EXPERT_NONLINEAR_SCALAR_OPS_PER_TOKEN,
            ),
        ),
        audit_recipe_fields=(
            (
                "execution_mode",
                "fit_only_two_phase_family_blocked_iterative_state_experts",
            ),
            ("expert_route_matrix_shape", (2, 2, 2)),
            ("expert_regime_order", _EXPERT_REGIME_ORDER),
            ("expert_route_edge_order", _EXPERT_ROUTE_EDGE_ORDER),
            ("route_state_semantics", _ROUTE_STATE_SEMANTICS),
            (
                "expert_dispatch_semantics",
                _EXPERT_DISPATCH_SEMANTICS,
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
            "independent_expert_operator_norm_projection_is_"
            "linearization_extrapolation"
        ),
        resource_envelope_error=(
            "fixed top-two state experts exceed their resource envelope"
        ),
        linearization_error=(
            "OOF state-expert linearization requires eight finite route edges"
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


def _float8(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float, float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != EXPERT_EDGE_COUNT:
        raise ValueError(f"{label} must contain exactly eight scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _int2(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{label} must contain two nonnegative integers")
    return (value[0], value[1])


def _indices2(value: object, *, label: str) -> tuple[int, int]:
    result = _int2(value, label=label)
    if result[0] == result[1]:
        raise ValueError(f"{label} indices must be distinct")
    return result


def _bool2(value: object, *, label: str) -> tuple[bool, bool]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not bool for item in value)
    ):
        raise ValueError(f"{label} must contain exactly two booleans")
    return (value[0], value[1])


def _operator_norms_by_expert(
    coefficients: Sequence[float],
) -> tuple[float, float]:
    values = torch.tensor(
        _float8(coefficients, label="expert coefficients"),
        dtype=torch.float64,
    ).reshape(2, 2, 2)
    return (
        float(torch.linalg.svdvals(values[0]).max()),
        float(torch.linalg.svdvals(values[1]).max()),
    )


def _resource_payload() -> dict[str, object]:
    return {
        "semantics": "top2_causal_balance_two_regime_experts_v1",
        "learned_float_scalar_count": EXPERT_EDGE_COUNT,
        "derived_prepared_float_scalar_count": (
            _EXPERT_DERIVED_CONSTANT_FLOAT_COUNT
        ),
        "prepared_float_scalar_count": (
            _EXPERT_PREPARED_FLOAT_SCALAR_COUNT
        ),
        "logical_linear_macs_per_token_upper_bound": (
            EXPERT_LINEAR_MACS_PER_TOKEN
        ),
        "nonlinear_scalar_ops_per_token_upper_bound": (
            EXPERT_NONLINEAR_SCALAR_OPS_PER_TOKEN
        ),
        "linear_accumulator_scalar_ops_per_token_upper_bound": (
            _EXPERT_LINEAR_ACCUMULATOR_OPS_PER_TOKEN
        ),
        "zero_denominator_comparisons_per_token_upper_bound": (
            _EXPERT_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN
        ),
        "expert_regime_comparisons_per_token_upper_bound": (
            _EXPERT_REGIME_COMPARISONS_PER_TOKEN
        ),
        "runtime_state_float_scalars_per_sequence": (
            _EXPERT_RUNTIME_STATE_FLOAT_COUNT
        ),
        "parent_modal_values_reused": True,
        "experts_evaluated_per_active_row": 1,
        "parent_decoder_invocations_per_token": 1,
    }


def gemma_causal_top2_state_experts_provider_artifact_sha256(
    *,
    parent_artifact_sha256: str,
    parent_h4_artifact_sha256: str,
    bridge_binding_sha256: str,
    decoder_sha256: str,
    lag_kernel_sha256: str,
    fold_receipt_sha256: str,
    top_mode_indices: Sequence[int],
    top_mode_norms: Sequence[float],
    coefficients_by_expert_route_edge: Sequence[float],
) -> str:
    """Replay one provider identity without loading the parent tensors."""

    indices = _indices2(top_mode_indices, label="top_mode_indices")
    norms = _float2(top_mode_norms, label="top_mode_norms")
    if any(value <= 0.0 for value in norms):
        raise ValueError("top-mode norms must be positive")
    coefficients = _float8(
        coefficients_by_expert_route_edge,
        label="coefficients_by_expert_route_edge",
    )
    if any(
        value > ROUTE_OPERATOR_NORM_BOUND + 1.0e-12
        for value in _operator_norms_by_expert(coefficients)
    ):
        raise ValueError("an expert exceeds the operator-norm bound")
    resources = _resource_payload()
    return _sha256(
        _PROVIDER_DOMAIN,
        {
            "semantics": "top2_causal_balance_two_regime_experts_v1",
            "site": _H4_SITE,
            "route_state_semantics": _ROUTE_STATE_SEMANTICS,
            "expert_dispatch_semantics": _EXPERT_DISPATCH_SEMANTICS,
            "expert_regime_order": _EXPERT_REGIME_ORDER,
            "expert_route_edge_order": _EXPERT_ROUTE_EDGE_ORDER,
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
            "coefficients_by_expert_route_edge": coefficients,
            "operator_norm_bound": ROUTE_OPERATOR_NORM_BOUND,
            "resources": resources,
            "resource_receipt_sha256": _sha256(
                _RESOURCE_DOMAIN,
                resources,
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2StateExpertsState:
    """Chunkable causal balance carry owned by one experts provider."""

    numerator: Tensor
    denominator: Tensor
    provider_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.provider_artifact_sha256,
            label="state-experts provider",
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
            raise ValueError("state-experts runtime state is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.numerator.shape[0])

    def validate_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class GemmaIterativeStateExpertsFitRecord:
    """One prompt's tensor-free two-regime behavioral linearization."""

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
    negative_gated_modal_sha256: str
    nonnegative_gated_modal_sha256: str
    supervised_tokens: int
    parent_signed_delta_nll_per_token: float
    jacobian_by_expert_route_edge: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    active_row_count: int
    active_row_count_by_expert: tuple[int, int]
    active_expert_mask: tuple[bool, bool]
    jacobian_support_by_expert: tuple[bool, bool]
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
            "negative_gated_modal_sha256",
            "nonnegative_gated_modal_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"fit record {name}")
        if type(self.supervised_tokens) is not int or self.supervised_tokens <= 0:
            raise ValueError("fit record supervised_tokens must be positive")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fit record active_row_count must be positive")
        counts = _int2(
            self.active_row_count_by_expert,
            label="active_row_count_by_expert",
        )
        active_mask = _bool2(
            self.active_expert_mask,
            label="active_expert_mask",
        )
        support_mask = _bool2(
            self.jacobian_support_by_expert,
            label="jacobian_support_by_expert",
        )
        if (
            sum(counts) != self.active_row_count
            or active_mask != (counts[0] > 0, counts[1] > 0)
            or any(support and not active for support, active in zip(
                support_mask,
                active_mask,
                strict=True,
            ))
        ):
            raise ValueError("fit-record expert activity receipt differs")
        object.__setattr__(self, "active_row_count_by_expert", counts)
        object.__setattr__(self, "active_expert_mask", active_mask)
        object.__setattr__(
            self,
            "jacobian_support_by_expert",
            support_mask,
        )
        object.__setattr__(
            self,
            "parent_signed_delta_nll_per_token",
            _finite(
                self.parent_signed_delta_nll_per_token,
                label="parent signed delta NLL/token",
            ),
        )
        jacobian = _float8(
            self.jacobian_by_expert_route_edge,
            label="jacobian_by_expert_route_edge",
        )
        observed_support = tuple(
            any(
                abs(value) > _COLUMN_SUPPORT_EPSILON
                for value in jacobian[offset : offset + 4]
            )
            for offset in (0, 4)
        )
        if observed_support != support_mask:
            raise ValueError("fit-record Jacobian support receipt differs")
        object.__setattr__(
            self,
            "jacobian_by_expert_route_edge",
            jacobian,
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
            raise ValueError("fit-record route feature receipt is invalid")
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
            "negative_gated_modal_sha256": (
                self.negative_gated_modal_sha256
            ),
            "nonnegative_gated_modal_sha256": (
                self.nonnegative_gated_modal_sha256
            ),
            "supervised_tokens": self.supervised_tokens,
            "parent_signed_delta_nll_per_token": (
                self.parent_signed_delta_nll_per_token
            ),
            "jacobian_by_expert_route_edge": (
                self.jacobian_by_expert_route_edge
            ),
            "active_row_count": self.active_row_count,
            "active_row_count_by_expert": (
                self.active_row_count_by_expert
            ),
            "active_expert_mask": self.active_expert_mask,
            "jacobian_support_by_expert": (
                self.jacobian_support_by_expert
            ),
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


def _record(value: object) -> GemmaIterativeStateExpertsFitRecord:
    if isinstance(value, GemmaIterativeStateExpertsFitRecord):
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
        "negative_gated_modal_sha256",
        "nonnegative_gated_modal_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_expert_route_edge",
        "active_row_count",
        "active_row_count_by_expert",
        "active_expert_mask",
        "jacobian_support_by_expert",
        "top_mode_indices",
        "top_mode_norms",
        "balance_feature_std",
        "top2_modal_energy_fraction",
        "fit_record_sha256",
    }
    if set(value) != expected:
        raise ValueError("serialized state-experts fit-record fields differ")
    result = GemmaIterativeStateExpertsFitRecord(
        example_id=value["example_id"],  # type: ignore[arg-type]
        family_id=value["family_id"],  # type: ignore[arg-type]
        model_inputs_sha256=value["model_inputs_sha256"],  # type: ignore[arg-type]
        parent_execution_sha256=value["parent_execution_sha256"],  # type: ignore[arg-type]
        parent_observation_sha256=value["parent_observation_sha256"],  # type: ignore[arg-type]
        parent_h4_artifact_sha256=value["parent_h4_artifact_sha256"],  # type: ignore[arg-type]
        prefix_sha256=value["prefix_sha256"],  # type: ignore[arg-type]
        gradient_sha256=value["gradient_sha256"],  # type: ignore[arg-type]
        parent_modal_sha256=value["parent_modal_sha256"],  # type: ignore[arg-type]
        balance_feature_sha256=value["balance_feature_sha256"],  # type: ignore[arg-type]
        negative_gated_modal_sha256=value["negative_gated_modal_sha256"],  # type: ignore[arg-type]
        nonnegative_gated_modal_sha256=value["nonnegative_gated_modal_sha256"],  # type: ignore[arg-type]
        supervised_tokens=value["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=value[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_expert_route_edge=value[
            "jacobian_by_expert_route_edge"
        ],  # type: ignore[arg-type]
        active_row_count=value["active_row_count"],  # type: ignore[arg-type]
        active_row_count_by_expert=value[
            "active_row_count_by_expert"
        ],  # type: ignore[arg-type]
        active_expert_mask=value["active_expert_mask"],  # type: ignore[arg-type]
        jacobian_support_by_expert=value[
            "jacobian_support_by_expert"
        ],  # type: ignore[arg-type]
        top_mode_indices=value["top_mode_indices"],  # type: ignore[arg-type]
        top_mode_norms=value["top_mode_norms"],  # type: ignore[arg-type]
        balance_feature_std=value["balance_feature_std"],  # type: ignore[arg-type]
        top2_modal_energy_fraction=value[
            "top2_modal_energy_fraction"
        ],  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != value["fit_record_sha256"]:
        raise ValueError("state-experts fit-record hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeStateExpertsFoldFit:
    """Replayable family-balanced fit for two independently bounded experts."""

    held_family_id: str
    train_example_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    train_fit_record_sha256s: tuple[str, ...]
    coefficients_by_expert_route_edge: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    unsupported_expert_route_edge_indices: tuple[int, ...]
    active_row_count: int
    active_row_count_by_expert: tuple[int, int]
    supported_route_edge_count_by_expert: tuple[int, int]
    weighted_column_norm_by_expert_route_edge: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    weighted_design_rank: int
    weighted_design_rank_by_expert: tuple[int, int]
    normal_condition_number: float
    pre_projection_operator_norm_by_expert: tuple[float, float]
    post_projection_operator_norm_by_expert: tuple[float, float]
    trust_projection_applied_by_expert: tuple[bool, bool]
    trust_projection_applied: bool
    linearized_rmse_before: float
    linearized_rmse_after: float
    ridge: float = ROUTE_RIDGE
    operator_norm_bound: float = ROUTE_OPERATOR_NORM_BOUND
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
        coefficients = _float8(
            self.coefficients_by_expert_route_edge,
            label="coefficients_by_expert_route_edge",
        )
        object.__setattr__(
            self,
            "coefficients_by_expert_route_edge",
            coefficients,
        )
        unsupported = self.unsupported_expert_route_edge_indices
        if (
            type(unsupported) is not tuple
            or unsupported != tuple(sorted(set(unsupported)))
            or any(
                type(value) is not int
                or not 0 <= value < EXPERT_EDGE_COUNT
                for value in unsupported
            )
        ):
            raise ValueError("unsupported expert route edges are invalid")
        if any(coefficients[index] != 0.0 for index in unsupported):
            raise ValueError("unsupported expert route edge became active")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fold active_row_count must be positive")
        active_counts = _int2(
            self.active_row_count_by_expert,
            label="active_row_count_by_expert",
        )
        support_counts = _int2(
            self.supported_route_edge_count_by_expert,
            label="supported_route_edge_count_by_expert",
        )
        if (
            sum(active_counts) != self.active_row_count
            or any(value > 4 for value in support_counts)
            or support_counts
            != (
                sum(index not in unsupported for index in range(0, 4)),
                sum(index not in unsupported for index in range(4, 8)),
            )
        ):
            raise ValueError("fold expert activity or support receipt differs")
        object.__setattr__(
            self,
            "active_row_count_by_expert",
            active_counts,
        )
        object.__setattr__(
            self,
            "supported_route_edge_count_by_expert",
            support_counts,
        )
        column_norms = _float8(
            self.weighted_column_norm_by_expert_route_edge,
            label="weighted_column_norm_by_expert_route_edge",
        )
        object.__setattr__(
            self,
            "weighted_column_norm_by_expert_route_edge",
            column_norms,
        )
        if (
            type(self.weighted_design_rank) is not int
            or not 0 <= self.weighted_design_rank <= EXPERT_EDGE_COUNT
        ):
            raise ValueError("weighted design rank is invalid")
        rank_by_expert = _int2(
            self.weighted_design_rank_by_expert,
            label="weighted_design_rank_by_expert",
        )
        if any(value > 4 for value in rank_by_expert):
            raise ValueError("per-expert weighted design rank is invalid")
        object.__setattr__(
            self,
            "weighted_design_rank_by_expert",
            rank_by_expert,
        )
        pre_norms = _float2(
            self.pre_projection_operator_norm_by_expert,
            label="pre_projection_operator_norm_by_expert",
        )
        post_norms = _float2(
            self.post_projection_operator_norm_by_expert,
            label="post_projection_operator_norm_by_expert",
        )
        observed_post = _operator_norms_by_expert(coefficients)
        if any(
            abs(reported - observed) > 1.0e-10
            for reported, observed in zip(
                post_norms,
                observed_post,
                strict=True,
            )
        ):
            raise ValueError("post-projection expert norm receipt differs")
        if any(
            value > self.operator_norm_bound + 1.0e-12
            for value in observed_post
        ):
            raise ValueError("an expert exceeds the operator-norm bound")
        projection_by_expert = _bool2(
            self.trust_projection_applied_by_expert,
            label="trust_projection_applied_by_expert",
        )
        expected_projection = tuple(
            value > self.operator_norm_bound for value in pre_norms
        )
        if projection_by_expert != expected_projection:
            raise ValueError("per-expert trust projection receipt differs")
        if (
            type(self.trust_projection_applied) is not bool
            or self.trust_projection_applied != any(projection_by_expert)
        ):
            raise ValueError("overall trust projection receipt differs")
        object.__setattr__(
            self,
            "pre_projection_operator_norm_by_expert",
            pre_norms,
        )
        object.__setattr__(
            self,
            "post_projection_operator_norm_by_expert",
            post_norms,
        )
        object.__setattr__(
            self,
            "trust_projection_applied_by_expert",
            projection_by_expert,
        )
        for name in (
            "normal_condition_number",
            "linearized_rmse_before",
            "linearized_rmse_after",
            "ridge",
            "operator_norm_bound",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if (
            self.ridge != ROUTE_RIDGE
            or self.operator_norm_bound != ROUTE_OPERATOR_NORM_BOUND
        ):
            raise ValueError("the frozen state-experts recipe cannot be retuned")
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
            "coefficients_by_expert_route_edge": (
                self.coefficients_by_expert_route_edge
            ),
            "unsupported_expert_route_edge_indices": (
                self.unsupported_expert_route_edge_indices
            ),
            "active_row_count": self.active_row_count,
            "active_row_count_by_expert": (
                self.active_row_count_by_expert
            ),
            "supported_route_edge_count_by_expert": (
                self.supported_route_edge_count_by_expert
            ),
            "weighted_column_norm_by_expert_route_edge": (
                self.weighted_column_norm_by_expert_route_edge
            ),
            "weighted_design_rank": self.weighted_design_rank,
            "weighted_design_rank_by_expert": (
                self.weighted_design_rank_by_expert
            ),
            "normal_condition_number": self.normal_condition_number,
            "pre_projection_operator_norm_by_expert": (
                self.pre_projection_operator_norm_by_expert
            ),
            "post_projection_operator_norm_by_expert": (
                self.post_projection_operator_norm_by_expert
            ),
            "trust_projection_applied_by_expert": (
                self.trust_projection_applied_by_expert
            ),
            "trust_projection_applied": self.trust_projection_applied,
            "linearized_rmse_before": self.linearized_rmse_before,
            "linearized_rmse_after": self.linearized_rmse_after,
            "ridge": self.ridge,
            "operator_norm_bound": self.operator_norm_bound,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fold_receipt_sha256": self.fold_receipt_sha256,
        }


def _project_expert(
    theta: Tensor,
    *,
    supported_mask: Tensor,
) -> tuple[Tensor, float, float, bool]:
    if theta.shape != (2, 2) or supported_mask.shape != theta.shape:
        raise ValueError("expert projection geometry differs")
    theta = theta.clone()
    theta[~supported_mask] = 0.0
    pre_norm = float(torch.linalg.svdvals(theta).max())
    projected = theta
    applied = pre_norm > ROUTE_OPERATOR_NORM_BOUND
    if applied:
        u, singular, vh = torch.linalg.svd(theta, full_matrices=False)
        ceiling = ROUTE_OPERATOR_NORM_BOUND * (1.0 - 1.0e-12)
        projected = (u * singular.clamp(max=ceiling).unsqueeze(0)) @ vh
        projected[~supported_mask] = 0.0
        observed = float(torch.linalg.svdvals(projected).max())
        if observed > ceiling:
            projected = projected * (ceiling / observed)
            projected[~supported_mask] = 0.0
    projected[~supported_mask] = 0.0
    post_norm = float(torch.linalg.svdvals(projected).max())
    if (
        post_norm > ROUTE_OPERATOR_NORM_BOUND
        or bool((projected[~supported_mask] != 0).any())
    ):
        raise RuntimeError("independent expert trust projection failed")
    return projected.contiguous(), pre_norm, post_norm, applied


def fit_gemma_iterative_state_experts_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeStateExpertsFoldFit:
    """Fit one family-balanced eight-edge ridge and bound each expert."""

    selected = tuple(
        sorted((_record(value) for value in records), key=lambda row: row.example_id)
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
        raise ValueError("fit records belong to different expert features")

    design = torch.tensor(
        [row.jacobian_by_expert_route_edge for row in selected],
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

    def numerical_rank(value: Tensor) -> int:
        return int(
            (torch.linalg.svdvals(value) > _DESIGN_RANK_TOLERANCE).sum()
        )

    weighted_rank = numerical_rank(weighted_design)
    rank_by_expert = (
        numerical_rank(weighted_design[:, :4]),
        numerical_rank(weighted_design[:, 4:]),
    )
    column_norms = torch.sqrt(
        (weights[:, None] * design.square()).sum(dim=0)
    )
    supported = tuple(
        index
        for index in range(EXPERT_EDGE_COUNT)
        if float(column_norms[index]) > _COLUMN_SUPPORT_EPSILON
    )
    coefficients = torch.zeros(EXPERT_EDGE_COUNT, dtype=torch.float64)
    condition = 0.0
    if supported:
        indices = torch.tensor(supported, dtype=torch.int64)
        x = design.index_select(1, indices)
        normal = x.T @ (weights[:, None] * x)
        eigenvalues = torch.linalg.eigvalsh(normal)
        positive = eigenvalues[eigenvalues > _COLUMN_SUPPORT_EPSILON]
        condition = (
            float(eigenvalues.max() / positive.min())
            if positive.numel()
            else 0.0
        )
        solved = torch.linalg.solve(
            normal
            + ROUTE_RIDGE
            * torch.eye(len(supported), dtype=torch.float64),
            x.T @ (weights * target),
        )
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("state-experts ridge fit became nonfinite")
        coefficients[indices] = solved

    coefficient_tensor = coefficients.reshape(2, 2, 2)
    projected_experts: list[Tensor] = []
    pre_norms: list[float] = []
    post_norms: list[float] = []
    projections: list[bool] = []
    for expert in range(2):
        supported_mask = torch.tensor(
            [
                (expert * 4 + index) in supported
                for index in range(4)
            ],
            dtype=torch.bool,
        ).reshape(2, 2)
        projected, pre, post, applied = _project_expert(
            coefficient_tensor[expert],
            supported_mask=supported_mask,
        )
        projected_experts.append(projected)
        pre_norms.append(pre)
        post_norms.append(post)
        projections.append(applied)
    coefficients = torch.stack(projected_experts).reshape(-1).contiguous()
    before = float(torch.sqrt((weights * target.square()).sum()))
    after = float(
        torch.sqrt(
            (
                weights
                * (design @ coefficients - target).square()
            ).sum()
        )
    )
    active_counts = (
        sum(row.active_row_count_by_expert[0] for row in selected),
        sum(row.active_row_count_by_expert[1] for row in selected),
    )
    return GemmaIterativeStateExpertsFoldFit(
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_expert_route_edge=tuple(
            float(value) for value in coefficients
        ),  # type: ignore[arg-type]
        unsupported_expert_route_edge_indices=tuple(
            index for index in range(EXPERT_EDGE_COUNT) if index not in supported
        ),
        active_row_count=sum(active_counts),
        active_row_count_by_expert=active_counts,
        supported_route_edge_count_by_expert=(
            sum(index in supported for index in range(0, 4)),
            sum(index in supported for index in range(4, 8)),
        ),
        weighted_column_norm_by_expert_route_edge=tuple(
            float(value) for value in column_norms
        ),  # type: ignore[arg-type]
        weighted_design_rank=weighted_rank,
        weighted_design_rank_by_expert=rank_by_expert,
        normal_condition_number=condition,
        pre_projection_operator_norm_by_expert=tuple(pre_norms),  # type: ignore[arg-type]
        post_projection_operator_norm_by_expert=tuple(post_norms),  # type: ignore[arg-type]
        trust_projection_applied_by_expert=tuple(projections),  # type: ignore[arg-type]
        trust_projection_applied=any(projections),
        linearized_rmse_before=before,
        linearized_rmse_after=after,
    )


def build_gemma_iterative_state_experts_fit_record(
    *,
    example: object,
    parent_execution: object,
    gradient: Tensor,
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeStateExpertsFitRecord:
    """Reduce one exact parent H4 NLL-VJP to eight expert derivatives."""

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
        raise ValueError("state-experts behavior-gradient geometry differs")
    active = prefix.target_affected_mask
    if (
        not bool(active.any())
        or not bool(torch.isfinite(gradient[active.to(gradient.device)]).all())
    ):
        raise ValueError("state-experts behavior gradient is invalid")

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
        or getattr(parent_execution, "h4_head_sha256", parent.artifact_sha256)
        != parent.artifact_sha256
        or prefix.bridge_binding_sha256 != parent.bridge_binding_sha256
    ):
        raise ValueError("state-experts fit-record identities differ")

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
    selected = parent_modal.index_select(
        2,
        torch.tensor(
            top_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        ),
    )
    gated = balance.unsqueeze(-1) * selected
    active_on_modal = active.to(parent_modal.device)
    negative_mask = active_on_modal & (balance < 0)
    nonnegative_mask = active_on_modal & ~negative_mask
    negative_gated = torch.zeros_like(gated)
    nonnegative_gated = torch.zeros_like(gated)
    if bool(negative_mask.any()):
        negative_gated[negative_mask] = gated[negative_mask]
    if bool(nonnegative_mask.any()):
        nonnegative_gated[nonnegative_mask] = gated[nonnegative_mask]
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(top_indices, dtype=torch.int64),
    ).to(device=gradient.device, dtype=torch.float64)
    gradient_modes = gradient.to(torch.float64) @ decoder.T
    negative_on_gradient = negative_mask.to(gradient_modes.device)
    nonnegative_on_gradient = nonnegative_mask.to(gradient_modes.device)

    def jacobian_for(gated_modal: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return torch.zeros((2, 2), dtype=torch.float64)
        return torch.einsum(
            "na,nb->ab",
            gated_modal.to(
                device=gradient_modes.device,
                dtype=torch.float64,
            )[mask],
            gradient_modes[mask],
        ) / parent_observation.supervised_tokens

    negative_jacobian = jacobian_for(
        negative_gated,
        negative_on_gradient,
    )
    nonnegative_jacobian = jacobian_for(
        nonnegative_gated,
        nonnegative_on_gradient,
    )
    jacobian = torch.cat(
        (
            negative_jacobian.reshape(-1),
            nonnegative_jacobian.reshape(-1),
        )
    )
    active_counts = (
        int(negative_mask.sum()),
        int(nonnegative_mask.sum()),
    )
    support = (
        bool((negative_jacobian.abs() > _COLUMN_SUPPORT_EPSILON).any()),
        bool((nonnegative_jacobian.abs() > _COLUMN_SUPPORT_EPSILON).any()),
    )
    active_balance = balance[active_on_modal].to(torch.float64)
    balance_std = (
        0.0
        if active_balance.numel() <= 1
        else float(active_balance.std(unbiased=False))
    )
    parent_active = parent_modal[active_on_modal].to(torch.float64)
    top_active = selected[active_on_modal].to(torch.float64)
    total_energy = float(parent_active.square().sum())
    top_energy = float(top_active.square().sum())
    signed = (
        parent_observation.candidate_summed_nll
        - parent_observation.source_summed_nll
    ) / parent_observation.supervised_tokens
    return GemmaIterativeStateExpertsFitRecord(
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
        negative_gated_modal_sha256=_tensor_sha256(negative_gated),
        nonnegative_gated_modal_sha256=_tensor_sha256(
            nonnegative_gated
        ),
        supervised_tokens=parent_observation.supervised_tokens,
        parent_signed_delta_nll_per_token=signed,
        jacobian_by_expert_route_edge=tuple(
            float(value) for value in jacobian
        ),  # type: ignore[arg-type]
        active_row_count=sum(active_counts),
        active_row_count_by_expert=active_counts,
        active_expert_mask=(
            active_counts[0] > 0,
            active_counts[1] > 0,
        ),
        jacobian_support_by_expert=support,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        balance_feature_std=balance_std,
        top2_modal_energy_fraction=(
            0.0 if total_energy == 0.0 else top_energy / total_energy
        ),
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2StateExpertsH4Provider(
    Gemma3L3L4CorrectionProvider
):
    """Authenticated sign-dispatched experts over source-only lag-B modes."""

    parent_h4: GemmaCausalResidualHead
    parent_artifact_sha256: str
    fold_fit: GemmaIterativeStateExpertsFoldFit
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
        if not isinstance(self.fold_fit, GemmaIterativeStateExpertsFoldFit):
            raise TypeError("fold_fit must be a strict state-experts fit")
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
            gemma_causal_top2_state_experts_provider_artifact_sha256(
                parent_artifact_sha256=parent_artifact_sha256,
                parent_h4_artifact_sha256=parent_h4_sha256,
                bridge_binding_sha256=bridge_binding,
                decoder_sha256=decoder_sha256,
                lag_kernel_sha256=lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=top_indices,
                top_mode_norms=top_norms,
                coefficients_by_expert_route_edge=(
                    self.coefficients_by_expert_route_edge
                ),
            ),
        )
        self.validate_integrity()

    @property
    def bridge_binding_sha256(self) -> str:
        return self._bridge_binding_sha256

    @property
    def coefficients_by_expert_route_edge(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        return self.fold_fit.coefficients_by_expert_route_edge

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
    def width(self) -> int:
        return self.parent_h4.width

    @property
    def marginal_learned_float_scalar_count(self) -> int:
        return EXPERT_EDGE_COUNT

    @property
    def marginal_derived_prepared_float_scalar_count(self) -> int:
        return _EXPERT_DERIVED_CONSTANT_FLOAT_COUNT

    @property
    def marginal_prepared_float_scalar_count(self) -> int:
        return _EXPERT_PREPARED_FLOAT_SCALAR_COUNT

    @property
    def marginal_logical_macs_per_token_upper_bound(self) -> int:
        return EXPERT_LINEAR_MACS_PER_TOKEN

    @property
    def nonlinear_scalar_ops_per_token_upper_bound(self) -> int:
        return EXPERT_NONLINEAR_SCALAR_OPS_PER_TOKEN

    @property
    def linear_accumulator_scalar_ops_per_token_upper_bound(self) -> int:
        return _EXPERT_LINEAR_ACCUMULATOR_OPS_PER_TOKEN

    @property
    def zero_denominator_comparisons_per_token_upper_bound(self) -> int:
        return _EXPERT_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN

    @property
    def expert_regime_comparisons_per_token_upper_bound(self) -> int:
        return _EXPERT_REGIME_COMPARISONS_PER_TOKEN

    @property
    def runtime_state_float_scalars_per_sequence(self) -> int:
        return _EXPERT_RUNTIME_STATE_FLOAT_COUNT

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
            or gemma_causal_top2_state_experts_provider_artifact_sha256(
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=self._parent_h4_sha256,
                bridge_binding_sha256=self._bridge_binding_sha256,
                decoder_sha256=self._decoder_sha256,
                lag_kernel_sha256=self._lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=self._top_mode_indices,
                top_mode_norms=self._top_mode_norms,
                coefficients_by_expert_route_edge=(
                    self.coefficients_by_expert_route_edge
                ),
            )
            != self.artifact_sha256
            or self.resource_receipt_sha256
            != _sha256(_RESOURCE_DOMAIN, _resource_payload())
        ):
            raise RuntimeError("causal top-two state-experts provider drifted")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> GemmaCausalTop2StateExpertsState:
        self.validate_integrity()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("state-experts batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("state-experts state dtype must be floating point")
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        return GemmaCausalTop2StateExpertsState(
            numerator=zeros,
            denominator=zeros.clone(),
            provider_artifact_sha256=self.artifact_sha256,
        )

    def route_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2StateExpertsState,
    ) -> tuple[Tensor, GemmaCausalTop2StateExpertsState]:
        """Route an upstream lag-aware parent-modal chunk and advance carry."""

        self.validate_integrity()
        prefix.validate_integrity()
        if not isinstance(state, GemmaCausalTop2StateExpertsState):
            raise TypeError("state must be a state-experts runtime state")
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
        selected_indices = torch.tensor(
            self.top_mode_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        )
        selected = parent_modal.index_select(2, selected_indices)
        gated = balance.unsqueeze(-1) * selected
        coefficients = torch.tensor(
            self.coefficients_by_expert_route_edge,
            device=parent_modal.device,
            dtype=parent_modal.dtype,
        ).reshape(2, 2, 2)
        negative = active & (balance < 0)
        nonnegative = active & ~negative
        delta_top = torch.zeros_like(selected)
        if bool(negative.any()):
            delta_top[negative] = (
                gated[negative] @ coefficients[0]
            )
        if bool(nonnegative.any()):
            delta_top[nonnegative] = (
                gated[nonnegative] @ coefficients[1]
            )
        routed = parent_modal.clone()
        if any(
            value != 0.0
            for value in self.coefficients_by_expert_route_edge
        ):
            routed.index_copy_(
                2,
                selected_indices,
                selected + delta_top,
            )
        next_state = GemmaCausalTop2StateExpertsState(
            numerator=numerator.detach().contiguous(),
            denominator=denominator.detach().contiguous(),
            provider_artifact_sha256=self.artifact_sha256,
        )
        if (
            bool(active.any())
            and not bool(torch.isfinite(routed[active]).all())
        ):
            raise ValueError("state-expert route became nonfinite")
        if (
            bool(inactive.any())
            and not bool((routed[inactive] == 0).all())
        ):
            raise RuntimeError("state-expert route is off support")
        self.validate_integrity()
        prefix.validate_integrity()
        return routed, next_state

    def correction_from_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2StateExpertsState,
    ) -> tuple[Tensor, GemmaCausalTop2StateExpertsState]:
        """Decode one upstream lag-aware parent-modal chunk after routing."""

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
            raise RuntimeError("state experts mutated the prefix")
        self.validate_integrity()
        return result, next_state

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Use zero balance carry for the correction-provider bridge ABI."""

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


def fit_gemma_iterative_state_experts_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2StateExpertsH4Provider:
    """Fit one OOF two-expert provider."""

    fit = fit_gemma_iterative_state_experts_fold(
        records,
        held_family_id=held_family,
    )
    return GemmaCausalTop2StateExpertsH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_state_experts_full_provider(
    *,
    records: Sequence[object],
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2StateExpertsH4Provider:
    """Fit a full-data provider only after external retention."""

    return fit_gemma_iterative_state_experts_fold_provider(
        records=records,
        held_family="__full_fit__",
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )
