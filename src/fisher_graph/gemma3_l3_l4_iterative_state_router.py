"""Causal top-two modal balance routing over the frozen Gemma lag-B head.

The accepted X4 plus source-only lag-B parent remains frozen.  This module
adds one deliberately small state-conditioned child at the H4 boundary:

* choose the two lag-B output modes with greatest lag-kernel Frobenius norm;
* form a causal, scale-normalized running balance from those modal values;
* mix the two gated values through one learned ``2 x 2`` matrix; and
* decode the modified parent modal correction exactly once.

The route owns four learned scalars.  Its running numerator and nonnegative
mass denominator are explicit generation state.  No prompt IDs, token IDs,
family labels, final sequence length, or future rows enter the runtime rule.
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
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    _tensor_sha256,
)


__all__ = [
    "BALANCE_RUNTIME_STATE_FLOATS_PER_SEQUENCE",
    "ROUTE_EDGE_COUNT",
    "ROUTE_LINEAR_MACS_PER_TOKEN",
    "ROUTE_NONLINEAR_SCALAR_OPS_PER_TOKEN",
    "ROUTE_OPERATOR_NORM_BOUND",
    "ROUTE_RIDGE",
    "GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE",
    "GemmaCausalTop2BalanceH4Provider",
    "GemmaCausalTop2BalanceState",
    "GemmaIterativeStateRouterFitRecord",
    "GemmaIterativeStateRouterFoldFit",
    "build_gemma_iterative_state_router_fit_record",
    "fit_gemma_iterative_state_router_fold",
    "fit_gemma_iterative_state_router_fold_provider",
    "fit_gemma_iterative_state_router_full_provider",
    "gemma_causal_top2_balance_provider_artifact_sha256",
    "top2_lag_b_output_modes",
]


ROUTE_RIDGE = 1.0e-6
ROUTE_OPERATOR_NORM_BOUND = 0.25
ROUTE_EDGE_COUNT = 4
ROUTE_LINEAR_MACS_PER_TOKEN = 6
ROUTE_NONLINEAR_SCALAR_OPS_PER_TOKEN = 5
BALANCE_RUNTIME_STATE_FLOATS_PER_SEQUENCE = 2
_ROUTE_LINEAR_ACCUMULATOR_OPS_PER_TOKEN = 4
_ROUTE_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN = 1
_COLUMN_SUPPORT_EPSILON = 1.0e-12
_DESIGN_RANK_TOLERANCE = 1.0e-12
_H4_SITE = "layer.4.output"
_ROUTE_SEMANTICS = "top2_parent_lag_b_modal_cumulative_balance_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_RECORD_DOMAIN = b"fisher-graph:gemma-state-router-fit-record:v1\0"
_FOLD_FIT_DOMAIN = b"fisher-graph:gemma-state-router-fold-fit:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma-top2-balance-provider:v1\0"
_RESOURCE_DOMAIN = b"fisher-graph:gemma-top2-balance-resources:v1\0"


GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE = (
    GemmaIterativeResidualCampaignRecipe(
        recipe_id="causal_top2_balance_router",
        fit_record_jacobian_field="jacobian_by_route_edge",
        fold_coefficient_field="coefficients_by_route_edge",
        coefficient_count=ROUTE_EDGE_COUNT,
        learned_parameter_attribute=(
            "marginal_learned_float_scalar_count"
        ),
        learned_parameter_fallback_attribute=None,
        expected_learned_parameter_count=ROUTE_EDGE_COUNT,
        logical_macs_attribute=(
            "marginal_logical_macs_per_token_upper_bound"
        ),
        logical_macs_fallback_attribute=None,
        expected_logical_macs_per_token_upper_bound=(
            ROUTE_LINEAR_MACS_PER_TOKEN
        ),
        logical_macs_must_equal_residual_width=False,
        extra_resource_expectations=(
            (
                "derived_constant_float_count",
                "marginal_derived_prepared_float_scalar_count",
                2,
            ),
            (
                "runtime_state_float_count_per_sequence",
                "runtime_state_float_scalars_per_sequence",
                BALANCE_RUNTIME_STATE_FLOATS_PER_SEQUENCE,
            ),
            (
                "nonlinear_scalar_ops_per_token_upper_bound",
                "nonlinear_scalar_ops_per_token_upper_bound",
                ROUTE_NONLINEAR_SCALAR_OPS_PER_TOKEN,
            ),
        ),
        audit_recipe_fields=(
            (
                "execution_mode",
                "fit_only_two_phase_family_blocked_iterative_state_router",
            ),
            ("route_matrix_shape", (2, 2)),
            (
                "route_edge_order",
                ("0_to_0", "0_to_1", "1_to_0", "1_to_1"),
            ),
            ("route_state_semantics", _ROUTE_SEMANTICS),
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
            "operator_norm_projection_is_linearization_extrapolation"
        ),
        resource_envelope_error=(
            "fixed top-two state router exceeds its resource envelope"
        ),
        linearization_error=(
            "OOF state-router linearization requires four finite route edges"
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
    result = tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _float4(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four scalars")
    result = tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _indices2(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or value[0] == value[1]
    ):
        raise ValueError(f"{label} must contain two distinct nonnegative indices")
    return (value[0], value[1])


def _theta_operator_norm(value: Sequence[float]) -> float:
    theta = torch.tensor(
        _float4(value, label="route coefficients"),
        dtype=torch.float64,
    ).reshape(2, 2)
    return float(torch.linalg.svdvals(theta).max())


def _project_route_coefficients(
    coefficients: Tensor,
    *,
    supported: tuple[int, ...],
) -> tuple[Tensor, float, float, bool]:
    """Apply the frozen trust bound without inventing unsupported edges."""

    if coefficients.shape != (ROUTE_EDGE_COUNT,):
        raise ValueError("route projection requires four coefficients")
    unsupported = tuple(
        index for index in range(ROUTE_EDGE_COUNT) if index not in supported
    )
    if any(float(coefficients[index]) != 0.0 for index in unsupported):
        raise ValueError("unsupported route coefficients must be zero")

    theta = coefficients.reshape(2, 2)
    u, singular, vh = torch.linalg.svd(theta, full_matrices=False)
    pre_operator_norm = float(singular.max())
    projection_applied = pre_operator_norm > ROUTE_OPERATOR_NORM_BOUND
    if projection_applied:
        projection_ceiling = ROUTE_OPERATOR_NORM_BOUND * (1.0 - 1.0e-12)
        clipped_singular = singular.clamp(max=projection_ceiling)
        projected = (u * clipped_singular.unsqueeze(0)) @ vh
    else:
        projected = theta.clone()

    flat = projected.reshape(-1)
    for index in unsupported:
        flat[index] = 0.0

    post_operator_norm = float(torch.linalg.svdvals(projected).max())
    if post_operator_norm > ROUTE_OPERATOR_NORM_BOUND:
        projected = projected * (
            ROUTE_OPERATOR_NORM_BOUND
            * (1.0 - 1.0e-12)
            / post_operator_norm
        )
        flat = projected.reshape(-1)
        for index in unsupported:
            flat[index] = 0.0
        post_operator_norm = float(torch.linalg.svdvals(projected).max())
    if post_operator_norm > ROUTE_OPERATOR_NORM_BOUND + 1.0e-12:
        raise RuntimeError("state-router operator-norm projection failed")
    return (
        projected.reshape(-1).contiguous(),
        pre_operator_norm,
        post_operator_norm,
        projection_applied,
    )


def _source_only_parent(value: object) -> GemmaCausalResidualHead:
    if not isinstance(value, GemmaCausalResidualHead):
        raise TypeError(
            "state router requires a concrete GemmaCausalResidualHead parent"
        )
    value.validate_integrity()
    if (
        value.site != _H4_SITE
        or value.conditioning != "l3_source_modes"
        or value.state_encoder is not None
        or value.state_kernel.shape != (0, 0)
        or value.rank < 2
    ):
        raise ValueError(
            "state router requires a source-only rank-at-least-two H4 parent"
        )
    return value


def top2_lag_b_output_modes(
    parent_h4: GemmaCausalResidualHead,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return deterministic top modes and their positive lag-kernel norms."""

    parent = _source_only_parent(parent_h4)
    norms = torch.linalg.vector_norm(
        parent.lag_kernel.detach().to(device="cpu", dtype=torch.float64),
        dim=(0, 1),
    )
    if (
        norms.ndim != 1
        or norms.numel() != parent.rank
        or not bool(torch.isfinite(norms).all())
    ):
        raise ValueError("lag-B output-mode norms are invalid")
    ordered = tuple(
        sorted(
            range(parent.rank),
            key=lambda index: (-float(norms[index]), index),
        )
    )
    selected = (ordered[0], ordered[1])
    selected_norms = (
        float(norms[selected[0]]),
        float(norms[selected[1]]),
    )
    if any(value <= 0.0 for value in selected_norms):
        raise ValueError("lag-B top-two output modes must have positive norm")
    return selected, selected_norms


def _resource_payload() -> dict[str, object]:
    return {
        "semantics": _ROUTE_SEMANTICS,
        "learned_float_scalar_count": ROUTE_EDGE_COUNT,
        "derived_prepared_float_scalar_count": 2,
        "prepared_float_scalar_count": ROUTE_EDGE_COUNT + 2,
        "logical_linear_macs_per_token_upper_bound": (
            ROUTE_LINEAR_MACS_PER_TOKEN
        ),
        "nonlinear_scalar_ops_per_token_upper_bound": (
            ROUTE_NONLINEAR_SCALAR_OPS_PER_TOKEN
        ),
        "linear_accumulator_scalar_ops_per_token_upper_bound": (
            _ROUTE_LINEAR_ACCUMULATOR_OPS_PER_TOKEN
        ),
        "zero_denominator_comparisons_per_token_upper_bound": (
            _ROUTE_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN
        ),
        "runtime_state_float_scalars_per_sequence": (
            BALANCE_RUNTIME_STATE_FLOATS_PER_SEQUENCE
        ),
        "parent_modal_values_reused": True,
        "parent_decoder_invocations_per_token": 1,
    }


def gemma_causal_top2_balance_provider_artifact_sha256(
    *,
    parent_artifact_sha256: str,
    parent_h4_artifact_sha256: str,
    bridge_binding_sha256: str,
    decoder_sha256: str,
    lag_kernel_sha256: str,
    fold_receipt_sha256: str,
    top_mode_indices: Sequence[int],
    top_mode_norms: Sequence[float],
    coefficients_by_route_edge: Sequence[float],
) -> str:
    """Replay a provider identity without loading its parent tensors."""

    indices = _indices2(top_mode_indices, label="top_mode_indices")
    norms = _float2(top_mode_norms, label="top_mode_norms")
    if any(value <= 0.0 for value in norms):
        raise ValueError("top-mode norms must be positive")
    coefficients = _float4(
        coefficients_by_route_edge,
        label="coefficients_by_route_edge",
    )
    if (
        _theta_operator_norm(coefficients)
        > ROUTE_OPERATOR_NORM_BOUND + 1.0e-12
    ):
        raise ValueError("route coefficients exceed the operator-norm bound")
    payload = {
        "semantics": _ROUTE_SEMANTICS,
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
        "coefficients_by_route_edge": coefficients,
        "route_edge_order": (
            "row_major_read_mode_then_write_mode"
        ),
        "operator_norm_bound": ROUTE_OPERATOR_NORM_BOUND,
        "resources": _resource_payload(),
        "resource_receipt_sha256": _sha256(
            _RESOURCE_DOMAIN,
            _resource_payload(),
        ),
    }
    return _sha256(_PROVIDER_DOMAIN, payload)


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2BalanceState:
    """Two-scalar chunkable running state for one provider realization."""

    numerator: Tensor
    denominator: Tensor
    provider_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.provider_artifact_sha256,
            label="router-state provider",
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
            raise ValueError("causal balance runtime state is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.numerator.shape[0])

    def validate_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class GemmaIterativeStateRouterFitRecord:
    """One prompt's tensor-free parent-point route linearization."""

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
    gated_modal_sha256: str
    supervised_tokens: int
    parent_signed_delta_nll_per_token: float
    jacobian_by_route_edge: tuple[float, float, float, float]
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
            "gated_modal_sha256",
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
            "jacobian_by_route_edge",
            _float4(
                self.jacobian_by_route_edge,
                label="jacobian_by_route_edge",
            ),
        )
        indices = _indices2(
            self.top_mode_indices,
            label="top_mode_indices",
        )
        norms = _float2(self.top_mode_norms, label="top_mode_norms")
        if any(value <= 0.0 for value in norms):
            raise ValueError("fit record top-mode norms must be positive")
        object.__setattr__(self, "top_mode_indices", indices)
        object.__setattr__(self, "top_mode_norms", norms)
        balance_std = _finite(
            self.balance_feature_std,
            label="balance_feature_std",
        )
        energy_fraction = _finite(
            self.top2_modal_energy_fraction,
            label="top2_modal_energy_fraction",
        )
        if balance_std < 0.0:
            raise ValueError("balance feature std must be nonnegative")
        if not 0.0 <= energy_fraction <= 1.0 + 1.0e-12:
            raise ValueError("top-two modal energy fraction must be in [0, 1]")
        object.__setattr__(self, "balance_feature_std", balance_std)
        object.__setattr__(
            self,
            "top2_modal_energy_fraction",
            min(1.0, energy_fraction),
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
            "gated_modal_sha256": self.gated_modal_sha256,
            "supervised_tokens": self.supervised_tokens,
            "parent_signed_delta_nll_per_token": (
                self.parent_signed_delta_nll_per_token
            ),
            "jacobian_by_route_edge": self.jacobian_by_route_edge,
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


def _record(value: object) -> GemmaIterativeStateRouterFitRecord:
    if isinstance(value, GemmaIterativeStateRouterFitRecord):
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
        "gated_modal_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_route_edge",
        "active_row_count",
        "top_mode_indices",
        "top_mode_norms",
        "balance_feature_std",
        "top2_modal_energy_fraction",
        "fit_record_sha256",
    }
    if set(value) != expected:
        raise ValueError("serialized state-router fit-record fields differ")
    result = GemmaIterativeStateRouterFitRecord(
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
        gated_modal_sha256=value["gated_modal_sha256"],  # type: ignore[arg-type]
        supervised_tokens=value["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=value[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_route_edge=value["jacobian_by_route_edge"],  # type: ignore[arg-type]
        active_row_count=value["active_row_count"],  # type: ignore[arg-type]
        top_mode_indices=value["top_mode_indices"],  # type: ignore[arg-type]
        top_mode_norms=value["top_mode_norms"],  # type: ignore[arg-type]
        balance_feature_std=value["balance_feature_std"],  # type: ignore[arg-type]
        top2_modal_energy_fraction=value[
            "top2_modal_energy_fraction"
        ],  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != value["fit_record_sha256"]:
        raise ValueError("state-router fit-record hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeStateRouterFoldFit:
    """Replayable result of one family-balanced four-edge ridge fit."""

    held_family_id: str
    train_example_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    train_fit_record_sha256s: tuple[str, ...]
    coefficients_by_route_edge: tuple[float, float, float, float]
    unsupported_route_edge_indices: tuple[int, ...]
    active_row_count: int
    weighted_column_norm_by_route_edge: tuple[float, float, float, float]
    weighted_design_rank: int
    normal_condition_number: float
    pre_projection_operator_norm: float
    post_projection_operator_norm: float
    linearized_rmse_before: float
    linearized_rmse_after: float
    trust_projection_applied: bool
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
        coefficients = _float4(
            self.coefficients_by_route_edge,
            label="coefficients_by_route_edge",
        )
        object.__setattr__(self, "coefficients_by_route_edge", coefficients)
        if (
            _theta_operator_norm(coefficients)
            > self.operator_norm_bound + 1.0e-12
        ):
            raise ValueError("fitted route exceeds the operator-norm bound")
        if (
            type(self.unsupported_route_edge_indices) is not tuple
            or self.unsupported_route_edge_indices
            != tuple(sorted(set(self.unsupported_route_edge_indices)))
            or any(
                type(value) is not int or not 0 <= value < ROUTE_EDGE_COUNT
                for value in self.unsupported_route_edge_indices
            )
        ):
            raise ValueError("unsupported route edges must be canonical")
        if any(
            coefficients[index] != 0.0
            for index in self.unsupported_route_edge_indices
        ):
            raise ValueError("unsupported route coefficients must be zero")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fold active_row_count must be positive")
        object.__setattr__(
            self,
            "weighted_column_norm_by_route_edge",
            _float4(
                self.weighted_column_norm_by_route_edge,
                label="weighted_column_norm_by_route_edge",
            ),
        )
        if (
            type(self.weighted_design_rank) is not int
            or not 0 <= self.weighted_design_rank <= ROUTE_EDGE_COUNT
        ):
            raise ValueError("weighted design rank is invalid")
        for name in (
            "normal_condition_number",
            "pre_projection_operator_norm",
            "post_projection_operator_norm",
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
            raise ValueError("the frozen state-router recipe cannot be retuned")
        if (
            abs(
                self.post_projection_operator_norm
                - _theta_operator_norm(coefficients)
            )
            > 1.0e-10
        ):
            raise ValueError("reported post-projection operator norm differs")
        if type(self.trust_projection_applied) is not bool:
            raise TypeError("trust_projection_applied must be boolean")
        if (
            self.trust_projection_applied
            != (
                self.pre_projection_operator_norm
                > self.operator_norm_bound
            )
        ):
            raise ValueError("trust projection receipt is inconsistent")
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
            "coefficients_by_route_edge": self.coefficients_by_route_edge,
            "unsupported_route_edge_indices": (
                self.unsupported_route_edge_indices
            ),
            "active_row_count": self.active_row_count,
            "weighted_column_norm_by_route_edge": (
                self.weighted_column_norm_by_route_edge
            ),
            "weighted_design_rank": self.weighted_design_rank,
            "normal_condition_number": self.normal_condition_number,
            "pre_projection_operator_norm": (
                self.pre_projection_operator_norm
            ),
            "post_projection_operator_norm": (
                self.post_projection_operator_norm
            ),
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


def fit_gemma_iterative_state_router_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeStateRouterFoldFit:
    """Fit the frozen four-edge recipe with equal family and prompt mass."""

    selected = tuple(
        sorted((_record(value) for value in records), key=lambda row: row.example_id)
    )
    if not selected or len({row.example_id for row in selected}) != len(selected):
        raise ValueError("fit records must be nonempty and unique")
    family_counts = Counter(row.family_id for row in selected)
    if held_family_id != "__full_fit__" and held_family_id in family_counts:
        raise ValueError("the held family leaked into its training records")
    parent_ids = {row.parent_h4_artifact_sha256 for row in selected}
    mode_indices = {row.top_mode_indices for row in selected}
    mode_norms = {row.top_mode_norms for row in selected}
    if (
        len(parent_ids) != 1
        or len(mode_indices) != 1
        or len(mode_norms) != 1
    ):
        raise ValueError("fit records belong to different route features")

    design = torch.tensor(
        [row.jacobian_by_route_edge for row in selected],
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
    singular_values = torch.linalg.svdvals(weighted_design)
    weighted_design_rank = int(
        (singular_values > _DESIGN_RANK_TOLERANCE).sum()
    )
    column_squares = (weights[:, None] * design.square()).sum(dim=0)
    column_norms = torch.sqrt(column_squares)
    supported = tuple(
        index
        for index in range(ROUTE_EDGE_COUNT)
        if float(column_norms[index]) > _COLUMN_SUPPORT_EPSILON
    )
    coefficients = torch.zeros(ROUTE_EDGE_COUNT, dtype=torch.float64)
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
        regularized = normal + ROUTE_RIDGE * torch.eye(
            len(supported),
            dtype=torch.float64,
        )
        right = x.T @ (weights * target)
        solved = torch.linalg.solve(regularized, right)
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("state-router ridge fit became nonfinite")
        coefficients[indices] = solved

    (
        coefficients,
        pre_operator_norm,
        post_operator_norm,
        projection_applied,
    ) = _project_route_coefficients(coefficients, supported=supported)

    prediction_before = torch.zeros_like(target)
    prediction_after = design @ coefficients
    before = float(
        torch.sqrt((weights * (prediction_before - target).square()).sum())
    )
    after = float(
        torch.sqrt((weights * (prediction_after - target).square()).sum())
    )
    return GemmaIterativeStateRouterFoldFit(
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_route_edge=tuple(
            float(value) for value in coefficients
        ),  # type: ignore[arg-type]
        unsupported_route_edge_indices=tuple(
            index for index in range(ROUTE_EDGE_COUNT) if index not in supported
        ),
        active_row_count=sum(row.active_row_count for row in selected),
        weighted_column_norm_by_route_edge=tuple(
            float(value) for value in column_norms
        ),  # type: ignore[arg-type]
        weighted_design_rank=weighted_design_rank,
        normal_condition_number=condition,
        pre_projection_operator_norm=pre_operator_norm,
        post_projection_operator_norm=post_operator_norm,
        linearized_rmse_before=before,
        linearized_rmse_after=after,
        trust_projection_applied=projection_applied,
    )


def _balance_feature(
    *,
    prefix: Gemma3L3L4OnePassPrefix,
    parent_modal: Tensor,
    top_mode_indices: tuple[int, int],
    top_mode_norms: tuple[float, float],
    initial_numerator: Tensor,
    initial_denominator: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        not isinstance(parent_modal, Tensor)
        or parent_modal.ndim != 3
        or parent_modal.shape[:2] != prefix.logical_positions.shape
        or not parent_modal.is_floating_point()
        or max(top_mode_indices) >= parent_modal.shape[-1]
        or initial_numerator.shape != (parent_modal.shape[0],)
        or initial_denominator.shape != initial_numerator.shape
    ):
        raise ValueError("causal balance feature geometry differs")
    modal = parent_modal.index_select(
        2,
        torch.tensor(
            top_mode_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        ),
    )
    norms = torch.tensor(
        top_mode_norms,
        device=modal.device,
        dtype=modal.dtype,
    )
    normalized = modal / norms
    balance = torch.zeros(
        parent_modal.shape[:2],
        device=modal.device,
        dtype=modal.dtype,
    )
    numerators = initial_numerator.to(
        device=modal.device,
        dtype=modal.dtype,
    ).clone()
    denominators = initial_denominator.to(
        device=modal.device,
        dtype=modal.dtype,
    ).clone()
    active = prefix.target_affected_mask.to(modal.device)
    for batch in range(modal.shape[0]):
        numerator = numerators[batch]
        denominator = denominators[batch]
        for row in range(modal.shape[1]):
            if not bool(active[batch, row]):
                continue
            values = normalized[batch, row]
            numerator = numerator + values[0] - values[1]
            denominator = denominator + values[0].abs() + values[1].abs()
            balance[batch, row] = (
                torch.zeros((), device=modal.device, dtype=modal.dtype)
                if bool(denominator == 0)
                else numerator / denominator
            )
        numerators[batch] = numerator
        denominators[batch] = denominator
    if (
        not bool(torch.isfinite(balance[active]).all())
        or not bool(torch.isfinite(numerators).all())
        or not bool(torch.isfinite(denominators).all())
        or bool((denominators < 0).any())
    ):
        raise ValueError("causal balance feature became invalid")
    return balance, numerators, denominators


def build_gemma_iterative_state_router_fit_record(
    *,
    example: object,
    parent_execution: object,
    gradient: Tensor,
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeStateRouterFitRecord:
    """Reduce one exact parent H4 NLL-VJP to four route derivatives."""

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
        raise ValueError("state-router behavior-gradient geometry differs")
    active = prefix.target_affected_mask
    if (
        not bool(active.any())
        or not bool(torch.isfinite(gradient[active.to(gradient.device)]).all())
    ):
        raise ValueError("state-router behavior gradient is invalid")

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
        raise ValueError("state-router fit-record identities differ")

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
    selected_modal = parent_modal.index_select(
        2,
        torch.tensor(
            top_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        ),
    )
    decoder = parent.decoder.index_select(
        0,
        torch.tensor(top_indices, dtype=torch.int64),
    ).to(device=gradient.device, dtype=torch.float64)
    gradient_modes = gradient.to(torch.float64) @ decoder.T
    gated_modal = (
        balance.unsqueeze(-1) * selected_modal
    ).to(device=gradient_modes.device, dtype=torch.float64)
    active_gradient = active.to(gradient_modes.device)
    jacobian = torch.einsum(
        "na,nb->ab",
        gated_modal[active_gradient],
        gradient_modes[active_gradient],
    ) / parent_observation.supervised_tokens
    active_balance = balance[active.to(balance.device)].to(torch.float64)
    balance_std = (
        0.0
        if active_balance.numel() <= 1
        else float(active_balance.std(unbiased=False))
    )
    parent_active = parent_modal[active.to(parent_modal.device)].to(
        torch.float64
    )
    top_active = selected_modal[active.to(selected_modal.device)].to(
        torch.float64
    )
    total_energy = float(parent_active.square().sum())
    top_energy = float(top_active.square().sum())
    energy_fraction = (
        0.0 if total_energy == 0.0 else top_energy / total_energy
    )
    signed = (
        parent_observation.candidate_summed_nll
        - parent_observation.source_summed_nll
    ) / parent_observation.supervised_tokens
    return GemmaIterativeStateRouterFitRecord(
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
        gated_modal_sha256=_tensor_sha256(gated_modal),
        supervised_tokens=parent_observation.supervised_tokens,
        parent_signed_delta_nll_per_token=signed,
        jacobian_by_route_edge=tuple(
            float(value) for value in jacobian.reshape(-1)
        ),  # type: ignore[arg-type]
        active_row_count=int(active.sum()),
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        balance_feature_std=balance_std,
        top2_modal_energy_fraction=energy_fraction,
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2BalanceH4Provider(Gemma3L3L4CorrectionProvider):
    """Authenticated four-scalar causal router over a source-only lag-B head."""

    parent_h4: GemmaCausalResidualHead
    parent_artifact_sha256: str
    fold_fit: GemmaIterativeStateRouterFoldFit
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
            label="parent H4 bridge binding",
        )
        if not isinstance(self.fold_fit, GemmaIterativeStateRouterFoldFit):
            raise TypeError("fold_fit must be a strict state-router fold fit")
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
            gemma_causal_top2_balance_provider_artifact_sha256(
                parent_artifact_sha256=parent_artifact_sha256,
                parent_h4_artifact_sha256=parent_h4_sha256,
                bridge_binding_sha256=bridge_binding,
                decoder_sha256=decoder_sha256,
                lag_kernel_sha256=lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=top_indices,
                top_mode_norms=top_norms,
                coefficients_by_route_edge=(
                    self.coefficients_by_route_edge
                ),
            ),
        )
        self.validate_integrity()

    @property
    def bridge_binding_sha256(self) -> str:
        return self._bridge_binding_sha256

    @property
    def coefficients_by_route_edge(
        self,
    ) -> tuple[float, float, float, float]:
        return self.fold_fit.coefficients_by_route_edge

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
    def route_state_semantics(self) -> str:
        return _ROUTE_SEMANTICS

    @property
    def width(self) -> int:
        return self.parent_h4.width

    @property
    def marginal_learned_float_scalar_count(self) -> int:
        return ROUTE_EDGE_COUNT

    @property
    def marginal_prepared_float_scalar_count(self) -> int:
        return ROUTE_EDGE_COUNT + 2

    @property
    def marginal_derived_prepared_float_scalar_count(self) -> int:
        return 2

    @property
    def marginal_logical_macs_per_token_upper_bound(self) -> int:
        return ROUTE_LINEAR_MACS_PER_TOKEN

    @property
    def nonlinear_scalar_ops_per_token_upper_bound(self) -> int:
        return ROUTE_NONLINEAR_SCALAR_OPS_PER_TOKEN

    @property
    def linear_accumulator_scalar_ops_per_token_upper_bound(self) -> int:
        return _ROUTE_LINEAR_ACCUMULATOR_OPS_PER_TOKEN

    @property
    def zero_denominator_comparisons_per_token_upper_bound(self) -> int:
        return _ROUTE_ZERO_DENOMINATOR_COMPARISONS_PER_TOKEN

    @property
    def runtime_state_float_scalars_per_sequence(self) -> int:
        return BALANCE_RUNTIME_STATE_FLOATS_PER_SEQUENCE

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
            or gemma_causal_top2_balance_provider_artifact_sha256(
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=self._parent_h4_sha256,
                bridge_binding_sha256=self._bridge_binding_sha256,
                decoder_sha256=self._decoder_sha256,
                lag_kernel_sha256=self._lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=self._top_mode_indices,
                top_mode_norms=self._top_mode_norms,
                coefficients_by_route_edge=(
                    self.coefficients_by_route_edge
                ),
            )
            != self.artifact_sha256
            or self.resource_receipt_sha256
            != _sha256(_RESOURCE_DOMAIN, _resource_payload())
        ):
            raise RuntimeError("causal top-two balance provider drifted")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> GemmaCausalTop2BalanceState:
        self.validate_integrity()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("router batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("router state dtype must be floating point")
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        return GemmaCausalTop2BalanceState(
            numerator=zeros,
            denominator=zeros.clone(),
            provider_artifact_sha256=self.artifact_sha256,
        )

    def route_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2BalanceState,
    ) -> tuple[Tensor, GemmaCausalTop2BalanceState]:
        """Route an already-computed parent modal tensor and advance carry."""

        self.validate_integrity()
        prefix.validate_integrity()
        if not isinstance(state, GemmaCausalTop2BalanceState):
            raise TypeError("state must be a causal top-two balance state")
        state.validate_integrity()
        if (
            state.provider_artifact_sha256 != self.artifact_sha256
            or state.batch_size != parent_modal.shape[0]
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or parent_modal.shape
            != (*prefix.logical_positions.shape, self.parent_h4.rank)
            or not parent_modal.is_floating_point()
        ):
            raise ValueError("router state, modal tensor, and prefix differ")
        active = prefix.target_affected_mask.to(parent_modal.device)
        if (
            bool(active.any())
            and not bool(torch.isfinite(parent_modal[active]).all())
        ):
            raise ValueError("parent modal correction is nonfinite")
        inactive = ~active
        if (
            bool(inactive.any())
            and not bool((parent_modal[inactive] == 0).all())
        ):
            raise ValueError("parent modal correction is off support")
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
        coefficients = torch.tensor(
            self.coefficients_by_route_edge,
            device=parent_modal.device,
            dtype=parent_modal.dtype,
        ).reshape(2, 2)
        routed = parent_modal.clone()
        if any(value != 0.0 for value in self.coefficients_by_route_edge):
            delta_top = (balance.unsqueeze(-1) * selected) @ coefficients
            routed_top = selected + delta_top
            routed.index_copy_(2, selected_indices, routed_top)
        next_state = GemmaCausalTop2BalanceState(
            numerator=numerator.detach().contiguous(),
            denominator=denominator.detach().contiguous(),
            provider_artifact_sha256=self.artifact_sha256,
        )
        if (
            bool(active.any())
            and not bool(torch.isfinite(routed[active]).all())
        ):
            raise ValueError("routed modal correction became nonfinite")
        if (
            bool(inactive.any())
            and not bool((routed[inactive] == 0).all())
        ):
            raise RuntimeError("routed modal correction is off support")
        self.validate_integrity()
        prefix.validate_integrity()
        return routed, next_state

    def correction_from_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2BalanceState,
    ) -> tuple[Tensor, GemmaCausalTop2BalanceState]:
        """Decode one upstream parent-modal chunk and advance router carry.

        Incremental callers must obtain ``parent_modal`` from the parent
        executor with its own lag history/cache intact. This provider owns
        only the two scalar balance accumulators.
        """

        self.validate_integrity()
        prefix_sha256 = prefix.artifact_sha256
        routed, next_state = self.route_modal_with_state(
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
            raise RuntimeError("state router mutated its authenticated prefix")
        self.validate_integrity()
        return result, next_state

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Use zero router carry for the correction-provider bridge ABI."""

        state = self.initial_state(
            prefix.logical_positions.shape[0],
            device=prefix.source_modes.device,
            dtype=prefix.source_modes.dtype,
        )
        parent_modal = self.parent_h4.modal_correction(
            prefix,
            realized_state,
        )
        correction, _next_state = self.correction_from_parent_modal_with_state(
            prefix,
            parent_modal,
            state,
        )
        return correction


def fit_gemma_iterative_state_router_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2BalanceH4Provider:
    """Fit one OOF realization of the frozen top-two route recipe."""

    fit = fit_gemma_iterative_state_router_fold(
        records,
        held_family_id=held_family,
    )
    return GemmaCausalTop2BalanceH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_state_router_full_provider(
    *,
    records: Sequence[object],
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2BalanceH4Provider:
    """Fit a full-data provider only after the external retention decision."""

    return fit_gemma_iterative_state_router_fold_provider(
        records=records,
        held_family="__full_fit__",
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )
