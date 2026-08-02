"""Six-coordinate causal occupancy routing over frozen Gemma lag-B modes.

Iteration five retains Iteration four's cumulative top-two balance ``g`` and
adds one centered negative-balance occupancy controller ``o``:

``delta = (g * selected_top2) @ C(C0 + g * Cg + o * Co)``.

The occupancy is either cumulative or exponentially weighted.  Both consume
the current active token before routing it, both are bounded in ``[-1, 1]``,
and both carry four floats per sequence: two for balance and two for
occupancy.  A single parent NLL-VJP produces both variants' six-coordinate
linearizations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from .gemma3_l3_l4_iterative_conformal_route import (
    CONFORMAL_OPERATOR_NORM_BOUND,
    _conformal_jacobian,
)
from .gemma3_l3_l4_iterative_residual_campaign import (
    GemmaIterativeResidualCampaignRecipe,
)
from .gemma3_l3_l4_iterative_state_router import (
    _balance_feature,
    _source_only_parent,
    top2_lag_b_output_modes,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    _tensor_sha256,
)


__all__ = [
    "CENTERED_CUMULATIVE_OCCUPANCY",
    "CENTERED_EW_OCCUPANCY",
    "GEMMA_ITERATIVE_CUMULATIVE_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE",
    "GEMMA_ITERATIVE_EW_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE",
    "OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT",
    "OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND",
    "OCCUPANCY_EW_DECAY",
    "OCCUPANCY_EW_HALF_LIFE",
    "OCCUPANCY_FIT_COORDINATE_DIRECT",
    "OCCUPANCY_FIT_COORDINATE_RESIDUALIZED",
    "OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE",
    "OCCUPANCY_RESIDUAL_SVD_ABSOLUTE_TOLERANCE",
    "OCCUPANCY_STANDARDIZED_RIDGE",
    "GemmaCausalTop2OccupancyConformalRouteH4Provider",
    "GemmaCausalTop2OccupancyConformalRouteState",
    "GemmaIterativeOccupancyConformalRouteFitRecord",
    "GemmaIterativeOccupancyConformalRouteFoldFit",
    "build_gemma_iterative_occupancy_conformal_route_fit_record",
    "fit_gemma_iterative_cumulative_occupancy_route_fold",
    "fit_gemma_iterative_cumulative_occupancy_route_fold_provider",
    "fit_gemma_iterative_cumulative_occupancy_route_full_provider",
    "fit_gemma_iterative_ew_occupancy_route_fold",
    "fit_gemma_iterative_ew_occupancy_route_fold_provider",
    "fit_gemma_iterative_ew_occupancy_route_full_provider",
    "fit_gemma_iterative_occupancy_conformal_route_fold",
    "fit_gemma_iterative_occupancy_conformal_route_fold_provider",
    "fit_gemma_iterative_occupancy_conformal_route_full_provider",
    "gemma_causal_top2_occupancy_conformal_route_provider_artifact_sha256",
    "project_occupancy_route_coefficients",
]


OccupancyKind = Literal["centered_cumulative", "centered_ew"]
CENTERED_CUMULATIVE_OCCUPANCY: OccupancyKind = "centered_cumulative"
CENTERED_EW_OCCUPANCY: OccupancyKind = "centered_ew"
_OCCUPANCY_KINDS = (
    CENTERED_CUMULATIVE_OCCUPANCY,
    CENTERED_EW_OCCUPANCY,
)
OCCUPANCY_EW_HALF_LIFE = 16
OCCUPANCY_EW_DECAY = 2.0 ** (-1.0 / OCCUPANCY_EW_HALF_LIFE)
OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT = 6
OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND = (
    CONFORMAL_OPERATOR_NORM_BOUND
)
OCCUPANCY_STANDARDIZED_RIDGE = 1.0e-6
OCCUPANCY_FIT_COORDINATE_DIRECT = "direct_standardized_ridge_v1"
OCCUPANCY_FIT_COORDINATE_RESIDUALIZED = (
    "training_fold_weighted_occupancy_residual_v1"
)
OCCUPANCY_RESIDUAL_SVD_ABSOLUTE_TOLERANCE = 1.0e-12
OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE = 1.0e-10
_COLUMN_SUPPORT_EPSILON = 1.0e-12
_RANK_TOLERANCE = 1.0e-12
_H4_SITE = "layer.4.output"
_COEFFICIENT_ORDER = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "occupancy_contrast_real",
    "occupancy_contrast_imag",
)
_CORNER_ORDER = (
    "g=-1,o=-1",
    "g=-1,o=+1",
    "g=+1,o=-1",
    "g=+1,o=+1",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_DOMAIN = b"fisher-graph:gemma-occupancy-route-fit-record:v1\0"
_FOLD_DOMAIN = b"fisher-graph:gemma-occupancy-route-fold:v2\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma-occupancy-route-provider:v1\0"
_RESOURCE_DOMAIN = b"fisher-graph:gemma-occupancy-route-resource:v1\0"


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
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _kind(value: object) -> OccupancyKind:
    if value not in _OCCUPANCY_KINDS:
        raise ValueError("occupancy kind must be cumulative or EW")
    return value  # type: ignore[return-value]


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _indices2(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
        or value[0] == value[1]
    ):
        raise ValueError(f"{label} must contain two distinct indices")
    return (value[0], value[1])


def _carry_is_bounded(numerator: Tensor, denominator: Tensor) -> bool:
    """Reject tiny-denominator states whose ratio escapes ``[-1, 1]``."""

    zero = denominator == 0
    if bool((zero & (numerator != 0)).any()):
        return False
    nonzero = ~zero
    return not bool(
        (
            numerator[nonzero].abs() / denominator[nonzero]
            > 1.0 + 1.0e-6
        ).any()
    )


def _corner_operator_norms(
    coefficients: Sequence[float],
) -> tuple[float, float, float, float]:
    a0, b0, ag, bg, ao, bo = _float_tuple(
        coefficients,
        count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        label="occupancy conformal coefficients",
    )
    result = tuple(
        math.hypot(
            a0 + g * ag + occupancy * ao,
            b0 + g * bg + occupancy * bo,
        )
        for g, occupancy in (
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        )
    )
    if any(not math.isfinite(value) for value in result):
        raise ValueError("occupancy conformal corner norm is nonfinite")
    return result  # type: ignore[return-value]


def project_occupancy_route_coefficients(
    coefficients: Tensor,
    *,
    supported: tuple[int, ...],
) -> tuple[
    Tensor,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    float,
    bool,
]:
    if (
        not isinstance(coefficients, Tensor)
        or coefficients.shape != (
            OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        )
        or not coefficients.is_floating_point()
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise ValueError("occupancy projection requires six finite values")
    if (
        type(supported) is not tuple
        or supported != tuple(sorted(set(supported)))
        or any(
            type(index) is not int
            or not 0 <= index < OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            for index in supported
        )
    ):
        raise ValueError("supported occupancy coordinates are invalid")
    unsupported = tuple(
        index
        for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
        if index not in supported
    )
    if any(float(coefficients[index]) != 0.0 for index in unsupported):
        raise ValueError("unsupported occupancy coordinates must be zero")
    pre = _corner_operator_norms(tuple(float(x) for x in coefficients))
    maximum = max(pre)
    applied = maximum > OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    scale = (
        OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
        * (1.0 - 1.0e-12)
        / maximum
        if applied
        else 1.0
    )
    projected = (coefficients * scale).contiguous()
    for index in unsupported:
        projected[index] = 0.0
    post = _corner_operator_norms(tuple(float(x) for x in projected))
    if max(post) > OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12:
        raise RuntimeError("occupancy corner projection failed")
    return projected, pre, post, scale, applied


# Retain the private spelling for the existing internal callers while exposing
# the trust projection as a stable public boundary for analysis modules.
_project_coefficients = project_occupancy_route_coefficients


def _condition_number(
    design: Tensor,
    weights: Tensor,
) -> tuple[int, float]:
    singular = torch.linalg.svdvals(weights.sqrt().unsqueeze(1) * design)
    supported = singular[singular > _RANK_TOLERANCE]
    rank = int(supported.numel())
    condition = (
        float((supported.max() / supported.min()).square())
        if rank
        else 0.0
    )
    return rank, condition


def _occupancy_feature(
    *,
    balance: Tensor,
    active: Tensor,
    initial_numerator: Tensor,
    initial_denominator: Tensor,
    occupancy_kind: OccupancyKind,
    collect_sign_counts: bool = True,
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    """Compute current-token centered occupancy without touching padding."""

    kind = _kind(occupancy_kind)
    if type(collect_sign_counts) is not bool:
        raise TypeError("collect_sign_counts must be boolean")
    if (
        balance.ndim != 2
        or active.shape != balance.shape
        or initial_numerator.shape != (balance.shape[0],)
        or initial_denominator.shape != initial_numerator.shape
        or initial_numerator.dtype != balance.dtype
        or initial_denominator.dtype != balance.dtype
        or initial_numerator.device != balance.device
        or initial_denominator.device != balance.device
        or not bool(torch.isfinite(initial_numerator).all())
        or not bool(torch.isfinite(initial_denominator).all())
        or bool((initial_denominator < 0).any())
        or not _carry_is_bounded(
            initial_numerator, initial_denominator
        )
    ):
        raise ValueError("occupancy state and feature geometry differ")
    result = torch.zeros_like(balance)
    numerators = initial_numerator.clone()
    denominators = initial_denominator.clone()
    negative_count = 0
    nonnegative_count = 0
    decay = (
        1.0
        if kind == CENTERED_CUMULATIVE_OCCUPANCY
        else OCCUPANCY_EW_DECAY
    )
    decay_tensor = torch.tensor(
        decay,
        device=balance.device,
        dtype=balance.dtype,
    )
    for batch in range(balance.shape[0]):
        numerator = numerators[batch]
        denominator = denominators[batch]
        for row in range(balance.shape[1]):
            if not bool(active[batch, row]):
                continue
            is_negative = bool(balance[batch, row] < 0)
            signed = torch.tensor(
                1.0 if is_negative else -1.0,
                device=balance.device,
                dtype=balance.dtype,
            )
            if kind == CENTERED_CUMULATIVE_OCCUPANCY:
                numerator = numerator + signed
                denominator = denominator + 1.0
            else:
                numerator = decay_tensor * numerator + signed
                denominator = decay_tensor * denominator + 1.0
            result[batch, row] = numerator / denominator
            if collect_sign_counts:
                negative_count += int(is_negative)
                nonnegative_count += int(not is_negative)
        numerators[batch] = numerator
        denominators[batch] = denominator
    if (
        not bool(torch.isfinite(result[active]).all())
        or not bool(torch.isfinite(numerators).all())
        or not bool(torch.isfinite(denominators).all())
        or bool((denominators < 0).any())
        or not _carry_is_bounded(numerators, denominators)
        or bool(active.any())
        and bool((result[active].abs() > 1.0 + 1.0e-6).any())
    ):
        raise ValueError("centered occupancy escaped its causal range")
    return (
        result,
        numerators,
        denominators,
        negative_count,
        nonnegative_count,
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2OccupancyConformalRouteState:
    balance_numerator: Tensor
    balance_denominator: Tensor
    occupancy_numerator: Tensor
    occupancy_denominator: Tensor
    occupancy_kind: OccupancyKind
    provider_artifact_sha256: str

    def __post_init__(self) -> None:
        _kind(self.occupancy_kind)
        _require_sha256(
            self.provider_artifact_sha256,
            label="occupancy state provider",
        )
        values = (
            self.balance_numerator,
            self.balance_denominator,
            self.occupancy_numerator,
            self.occupancy_denominator,
        )
        first = values[0]
        if (
            not isinstance(first, Tensor)
            or first.ndim != 1
            or not first.is_floating_point()
            or any(
                not isinstance(value, Tensor)
                or value.shape != first.shape
                or value.dtype != first.dtype
                or value.device != first.device
                or not bool(torch.isfinite(value).all())
                for value in values
            )
            or bool((self.balance_denominator < 0).any())
            or bool((self.occupancy_denominator < 0).any())
            or not _carry_is_bounded(
                self.balance_numerator, self.balance_denominator
            )
            or not _carry_is_bounded(
                self.occupancy_numerator, self.occupancy_denominator
            )
        ):
            raise ValueError("occupancy route state is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.balance_numerator.shape[0])

    def validate_integrity(self) -> None:
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class GemmaIterativeOccupancyConformalRouteFitRecord:
    """One prompt's shared-VJP linearizations for both occupancy variants."""

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
    cumulative_occupancy_feature_sha256: str
    ew_occupancy_feature_sha256: str
    shared_feature_sha256: str
    balance_contrast_feature_sha256: str
    cumulative_occupancy_contrast_feature_sha256: str
    ew_occupancy_contrast_feature_sha256: str
    supervised_tokens: int
    parent_signed_delta_nll_per_token: float
    jacobian_by_cumulative_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    jacobian_by_ew_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    active_row_count: int
    negative_balance_row_count: int
    nonnegative_balance_row_count: int
    top_mode_indices: tuple[int, int]
    top_mode_norms: tuple[float, float]
    balance_feature_std: float
    cumulative_occupancy_feature_std: float
    ew_occupancy_feature_std: float
    top2_modal_energy_fraction: float
    fit_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="fit example")
        _identifier(self.family_id, label="fit family")
        for name in (
            "model_inputs_sha256",
            "parent_execution_sha256",
            "parent_observation_sha256",
            "parent_h4_artifact_sha256",
            "prefix_sha256",
            "gradient_sha256",
            "parent_modal_sha256",
            "balance_feature_sha256",
            "cumulative_occupancy_feature_sha256",
            "ew_occupancy_feature_sha256",
            "shared_feature_sha256",
            "balance_contrast_feature_sha256",
            "cumulative_occupancy_contrast_feature_sha256",
            "ew_occupancy_contrast_feature_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"fit {name}")
        if (
            type(self.supervised_tokens) is not int
            or self.supervised_tokens <= 0
            or type(self.active_row_count) is not int
            or self.active_row_count <= 0
            or type(self.negative_balance_row_count) is not int
            or type(self.nonnegative_balance_row_count) is not int
            or self.negative_balance_row_count < 0
            or self.nonnegative_balance_row_count < 0
            or self.negative_balance_row_count
            + self.nonnegative_balance_row_count
            != self.active_row_count
        ):
            raise ValueError("occupancy fit row counts are invalid")
        for name in (
            "jacobian_by_cumulative_occupancy_conformal_coefficient",
            "jacobian_by_ew_occupancy_conformal_coefficient",
        ):
            parsed = _float_tuple(
                getattr(self, name),
                count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
                label=name,
            )
            object.__setattr__(self, name, parsed)
        if (
            self.jacobian_by_cumulative_occupancy_conformal_coefficient[:4]
            != self.jacobian_by_ew_occupancy_conformal_coefficient[:4]
        ):
            raise ValueError("shared VJP coordinates differ across arms")
        object.__setattr__(
            self,
            "parent_signed_delta_nll_per_token",
            _finite(
                self.parent_signed_delta_nll_per_token,
                label="parent signed delta NLL",
            ),
        )
        indices = _indices2(self.top_mode_indices, label="top modes")
        norms = _float_tuple(
            self.top_mode_norms,
            count=2,
            label="top mode norms",
        )
        if any(value <= 0.0 for value in norms):
            raise ValueError("top mode norms must be positive")
        object.__setattr__(self, "top_mode_indices", indices)
        object.__setattr__(self, "top_mode_norms", norms)
        for name in (
            "balance_feature_std",
            "cumulative_occupancy_feature_std",
            "ew_occupancy_feature_std",
            "top2_modal_energy_fraction",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.top2_modal_energy_fraction > 1.0 + 1.0e-12:
            raise ValueError("top-two energy fraction exceeds one")
        object.__setattr__(
            self,
            "fit_record_sha256",
            _sha256(_FIT_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fit_record_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fit_record_sha256": self.fit_record_sha256}

    def validate_integrity(self) -> None:
        if _sha256(_FIT_DOMAIN, self._payload()) != self.fit_record_sha256:
            raise RuntimeError("occupancy fit record drifted")


def _record(
    value: object,
) -> GemmaIterativeOccupancyConformalRouteFitRecord:
    if isinstance(value, GemmaIterativeOccupancyConformalRouteFitRecord):
        value.validate_integrity()
        return value
    if not isinstance(value, Mapping):
        raise TypeError("occupancy fit records must be mappings or records")
    expected = set(
        GemmaIterativeOccupancyConformalRouteFitRecord.__dataclass_fields__
    )
    if set(value) != expected:
        raise ValueError("serialized occupancy fit-record fields differ")
    payload = dict(value)
    receipt = payload.pop("fit_record_sha256")
    result = GemmaIterativeOccupancyConformalRouteFitRecord(
        **payload,  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != receipt:
        raise ValueError("occupancy fit-record hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeOccupancyConformalRouteFoldFit:
    """Replayable standardized-ridge fit with a four-corner trust bound."""

    occupancy_kind: OccupancyKind
    held_family_id: str
    train_example_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    train_fit_record_sha256s: tuple[str, ...]
    coefficients_by_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    unsupported_occupancy_conformal_coefficient_indices: tuple[int, ...]
    active_row_count: int
    weighted_column_scale_by_occupancy_conformal_coefficient: tuple[
        float, float, float, float, float, float
    ]
    raw_weighted_design_rank: int
    standardized_weighted_design_rank: int
    raw_normal_condition_number: float
    standardized_normal_condition_number: float
    pre_projection_corner_operator_norms: tuple[
        float, float, float, float
    ]
    post_projection_corner_operator_norms: tuple[
        float, float, float, float
    ]
    trust_projection_scale: float
    linearized_rmse_before: float
    linearized_rmse_after: float
    trust_projection_applied: bool
    ridge: float = OCCUPANCY_STANDARDIZED_RIDGE
    operator_norm_bound: float = OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    fit_coordinate_system: str = OCCUPANCY_FIT_COORDINATE_DIRECT
    occupancy_projection_on_base_by_base_and_occupancy_coordinate: tuple[
        float, float, float, float, float, float, float, float
    ] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    residualization_base_weighted_design_rank: int = 0
    pre_residualization_weighted_occupancy_column_scales: tuple[
        float, float
    ] = (0.0, 0.0)
    occupancy_residual_energy_fraction_by_coordinate: tuple[
        float, float
    ] = (0.0, 0.0)
    maximum_absolute_weighted_base_residual_correlation: float = 0.0
    fold_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occupancy_kind", _kind(self.occupancy_kind))
        if self.held_family_id != "__full_fit__":
            _identifier(self.held_family_id, label="held family")
        for name in (
            "train_example_ids",
            "train_family_ids",
            "train_fit_record_sha256s",
        ):
            raw_values = getattr(self, name)
            if not isinstance(raw_values, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            values = tuple(raw_values)
            if (
                not values
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be canonical and unique")
            object.__setattr__(self, name, values)
        for value in self.train_fit_record_sha256s:
            _require_sha256(value, label="training fit-record receipt")
        coefficients = _float_tuple(
            self.coefficients_by_occupancy_conformal_coefficient,
            count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
            label="occupancy conformal coefficients",
        )
        object.__setattr__(
            self,
            "coefficients_by_occupancy_conformal_coefficient",
            coefficients,
        )
        raw_unsupported = (
            self.unsupported_occupancy_conformal_coefficient_indices
        )
        if not isinstance(raw_unsupported, (tuple, list)):
            raise ValueError(
                "unsupported occupancy coordinates must be a sequence"
            )
        unsupported = tuple(raw_unsupported)
        if (
            unsupported != tuple(sorted(set(unsupported)))
            or any(
                type(index) is not int
                or not 0 <= index
                < OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
                for index in unsupported
            )
        ):
            raise ValueError(
                "unsupported occupancy coordinates must be canonical"
            )
        object.__setattr__(
            self,
            "unsupported_occupancy_conformal_coefficient_indices",
            unsupported,
        )
        if any(coefficients[index] != 0.0 for index in unsupported):
            raise ValueError("unsupported occupancy coordinates must be zero")
        if type(self.active_row_count) is not int or self.active_row_count <= 0:
            raise ValueError("fold active row count must be positive")
        scales = _float_tuple(
            self.weighted_column_scale_by_occupancy_conformal_coefficient,
            count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
            label="weighted occupancy column scales",
        )
        if any(value < 0.0 for value in scales):
            raise ValueError("weighted occupancy scales must be nonnegative")
        object.__setattr__(
            self,
            "weighted_column_scale_by_occupancy_conformal_coefficient",
            scales,
        )
        for name in (
            "raw_weighted_design_rank",
            "standardized_weighted_design_rank",
        ):
            value = getattr(self, name)
            if (
                type(value) is not int
                or not 0 <= value
                <= OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
            ):
                raise ValueError(f"{name} is invalid")
        for name in (
            "raw_normal_condition_number",
            "standardized_normal_condition_number",
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
        pre = _float_tuple(
            self.pre_projection_corner_operator_norms,
            count=4,
            label="pre-projection corner norms",
        )
        post = _float_tuple(
            self.post_projection_corner_operator_norms,
            count=4,
            label="post-projection corner norms",
        )
        if any(value < 0.0 for value in (*pre, *post)):
            raise ValueError("corner norms must be nonnegative")
        object.__setattr__(
            self, "pre_projection_corner_operator_norms", pre
        )
        object.__setattr__(
            self, "post_projection_corner_operator_norms", post
        )
        if (
            self.ridge != OCCUPANCY_STANDARDIZED_RIDGE
            or self.operator_norm_bound
            != OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
        ):
            raise ValueError("the frozen occupancy route cannot be retuned")
        if self.fit_coordinate_system not in (
            OCCUPANCY_FIT_COORDINATE_DIRECT,
            OCCUPANCY_FIT_COORDINATE_RESIDUALIZED,
        ):
            raise ValueError("occupancy fit coordinate system is invalid")
        projection = _float_tuple(
            self.occupancy_projection_on_base_by_base_and_occupancy_coordinate,
            count=8,
            label="occupancy projection on base",
        )
        pre_occupancy_scales = _float_tuple(
            self.pre_residualization_weighted_occupancy_column_scales,
            count=2,
            label="pre-residualization occupancy scales",
        )
        residual_energy = _float_tuple(
            self.occupancy_residual_energy_fraction_by_coordinate,
            count=2,
            label="occupancy residual energy fractions",
        )
        maximum_correlation = _finite(
            self.maximum_absolute_weighted_base_residual_correlation,
            label="maximum weighted base-residual correlation",
        )
        if (
            type(self.residualization_base_weighted_design_rank) is not int
            or not 0
            <= self.residualization_base_weighted_design_rank
            <= 4
            or any(value < 0.0 for value in pre_occupancy_scales)
            or any(
                value < 0.0 or value > 1.0 + 1.0e-10
                for value in residual_energy
            )
            or maximum_correlation < 0.0
        ):
            raise ValueError("occupancy residualization receipt is invalid")
        if self.fit_coordinate_system == OCCUPANCY_FIT_COORDINATE_DIRECT:
            if (
                any(value != 0.0 for value in projection)
                or self.residualization_base_weighted_design_rank != 0
                or any(value != 0.0 for value in pre_occupancy_scales)
                or any(value != 0.0 for value in residual_energy)
                or maximum_correlation != 0.0
            ):
                raise ValueError(
                    "direct occupancy fits cannot carry residual metadata"
                )
        elif (
            maximum_correlation
            > OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE
        ):
            raise ValueError(
                "occupancy residual is not weighted-orthogonal to base"
            )
        object.__setattr__(
            self,
            "occupancy_projection_on_base_by_base_and_occupancy_coordinate",
            projection,
        )
        object.__setattr__(
            self,
            "pre_residualization_weighted_occupancy_column_scales",
            pre_occupancy_scales,
        )
        object.__setattr__(
            self,
            "occupancy_residual_energy_fraction_by_coordinate",
            residual_energy,
        )
        object.__setattr__(
            self,
            "maximum_absolute_weighted_base_residual_correlation",
            maximum_correlation,
        )
        observed = _corner_operator_norms(coefficients)
        if any(
            abs(left - right) > 1.0e-10
            for left, right in zip(post, observed, strict=True)
        ):
            raise ValueError("reported post-projection corner norms differ")
        if max(post) > self.operator_norm_bound + 1.0e-12:
            raise ValueError("occupancy route exceeds its trust bound")
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
            raise ValueError("corner projection norm receipts differ")
        object.__setattr__(
            self,
            "fold_receipt_sha256",
            _sha256(_FOLD_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fold_receipt_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fold_receipt_sha256": self.fold_receipt_sha256,
        }

    def validate_integrity(self) -> None:
        if _sha256(_FOLD_DOMAIN, self._payload()) != self.fold_receipt_sha256:
            raise RuntimeError("occupancy fold fit drifted")


def fit_gemma_iterative_occupancy_conformal_route_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
    occupancy_kind: str,
) -> GemmaIterativeOccupancyConformalRouteFoldFit:
    """Fit one arm with family-balanced, uncentered standardized ridge."""

    kind = _kind(occupancy_kind)
    selected = tuple(
        sorted((_record(value) for value in records), key=lambda x: x.example_id)
    )
    if not selected or len({row.example_id for row in selected}) != len(
        selected
    ):
        raise ValueError("fit records must be nonempty and unique")
    family_counts = Counter(row.family_id for row in selected)
    if held_family_id != "__full_fit__" and held_family_id in family_counts:
        raise ValueError("the held family leaked into its training records")
    if (
        len({row.parent_h4_artifact_sha256 for row in selected}) != 1
        or len({row.top_mode_indices for row in selected}) != 1
        or len({row.top_mode_norms for row in selected}) != 1
    ):
        raise ValueError("fit records belong to different occupancy features")
    jacobian_field = (
        "jacobian_by_cumulative_occupancy_conformal_coefficient"
        if kind == CENTERED_CUMULATIVE_OCCUPANCY
        else "jacobian_by_ew_occupancy_conformal_coefficient"
    )
    design = torch.tensor(
        [getattr(row, jacobian_field) for row in selected],
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
    scales = torch.sqrt((weights[:, None] * design.square()).sum(dim=0))
    supported = tuple(
        index
        for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
        if float(scales[index]) > _COLUMN_SUPPORT_EPSILON
    )
    raw_rank, raw_condition = _condition_number(design, weights)
    coefficients = torch.zeros(
        OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        dtype=torch.float64,
    )
    standardized_rank = 0
    standardized_condition = 0.0
    if supported:
        indices = torch.tensor(supported, dtype=torch.int64)
        standardized = design.index_select(1, indices) / scales.index_select(
            0, indices
        )
        standardized_rank, standardized_condition = _condition_number(
            standardized, weights
        )
        normal = standardized.T @ (weights[:, None] * standardized)
        beta = torch.linalg.solve(
            normal
            + OCCUPANCY_STANDARDIZED_RIDGE
            * torch.eye(len(supported), dtype=torch.float64),
            standardized.T @ (weights * target),
        )
        solved = beta / scales.index_select(0, indices)
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("occupancy standardized ridge became nonfinite")
        coefficients[indices] = solved
    (
        coefficients,
        pre,
        post,
        projection_scale,
        projection_applied,
    ) = _project_coefficients(coefficients, supported=supported)
    before = float(torch.sqrt((weights * target.square()).sum()))
    after = float(
        torch.sqrt(
            (weights * (design @ coefficients - target).square()).sum()
        )
    )
    return GemmaIterativeOccupancyConformalRouteFoldFit(
        occupancy_kind=kind,
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_occupancy_conformal_coefficient=tuple(
            float(value) for value in coefficients
        ),  # type: ignore[arg-type]
        unsupported_occupancy_conformal_coefficient_indices=tuple(
            index
            for index in range(OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT)
            if index not in supported
        ),
        active_row_count=sum(row.active_row_count for row in selected),
        weighted_column_scale_by_occupancy_conformal_coefficient=tuple(
            float(value) for value in scales
        ),  # type: ignore[arg-type]
        raw_weighted_design_rank=raw_rank,
        standardized_weighted_design_rank=standardized_rank,
        raw_normal_condition_number=raw_condition,
        standardized_normal_condition_number=standardized_condition,
        pre_projection_corner_operator_norms=pre,
        post_projection_corner_operator_norms=post,
        trust_projection_scale=projection_scale,
        linearized_rmse_before=before,
        linearized_rmse_after=after,
        trust_projection_applied=projection_applied,
    )


def build_gemma_iterative_occupancy_conformal_route_fit_record(
    *,
    example: object,
    parent_execution: object,
    gradient: Tensor,
    parent_h4: GemmaCausalResidualHead,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeOccupancyConformalRouteFitRecord:
    """Reduce one parent NLL-VJP to both six-coordinate arm records."""

    parent = _source_only_parent(parent_h4)
    if not isinstance(
        parent_observation, GemmaH4DampingFiniteNLLObservation
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
        raise ValueError("occupancy behavior-gradient geometry differs")
    active = prefix.target_affected_mask
    active_gradient = active.to(gradient.device)
    if (
        not bool(active.any())
        or not bool(torch.isfinite(gradient[active_gradient]).all())
    ):
        raise ValueError("occupancy behavior gradient is invalid")
    example_id = _identifier(
        getattr(example, "example_id", None), label="example_id"
    )
    family_id = _identifier(
        getattr(example, "family_id", None), label="family_id"
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
            parent_execution, "h4_head_sha256", parent.artifact_sha256
        )
        != parent.artifact_sha256
        or prefix.bridge_binding_sha256 != parent.bridge_binding_sha256
    ):
        raise ValueError("occupancy fit-record identities differ")

    parent_modal = parent.modal_correction(prefix, candidate_h4)
    top_indices, top_norms = top2_lag_b_output_modes(parent)
    zeros = torch.zeros(
        parent_modal.shape[0],
        device=parent_modal.device,
        dtype=parent_modal.dtype,
    )
    balance, _balance_numerator, _balance_denominator = _balance_feature(
        prefix=prefix,
        parent_modal=parent_modal,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        initial_numerator=zeros,
        initial_denominator=zeros,
    )
    feature_active = active.to(parent_modal.device)
    (
        cumulative_occupancy,
        _cumulative_numerator,
        _cumulative_denominator,
        negative_count,
        nonnegative_count,
    ) = _occupancy_feature(
        balance=balance,
        active=feature_active,
        initial_numerator=zeros,
        initial_denominator=zeros,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
    )
    (
        ew_occupancy,
        _ew_numerator,
        _ew_denominator,
        ew_negative_count,
        ew_nonnegative_count,
    ) = _occupancy_feature(
        balance=balance,
        active=feature_active,
        initial_numerator=zeros,
        initial_denominator=zeros,
        occupancy_kind=CENTERED_EW_OCCUPANCY,
    )
    if (negative_count, nonnegative_count) != (
        ew_negative_count,
        ew_nonnegative_count,
    ):
        raise RuntimeError("occupancy arms saw different causal signs")
    selected_indices = torch.tensor(
        top_indices, device=parent_modal.device, dtype=torch.int64
    )
    selected_modal = parent_modal.index_select(2, selected_indices)
    shared_feature = balance.unsqueeze(-1) * selected_modal
    balance_contrast = balance.unsqueeze(-1) * shared_feature
    cumulative_contrast = (
        cumulative_occupancy.unsqueeze(-1) * shared_feature
    )
    ew_contrast = ew_occupancy.unsqueeze(-1) * shared_feature
    decoder = parent.decoder.index_select(
        0, torch.tensor(top_indices, dtype=torch.int64)
    ).to(device=gradient.device, dtype=torch.float64)
    gradient_modes = gradient.to(torch.float64) @ decoder.T

    def derivative(feature: Tensor) -> tuple[Tensor, Tensor]:
        return _conformal_jacobian(
            feature.to(device=gradient.device, dtype=torch.float64),
            gradient_modes,
            active_gradient,
        )

    shared_real, shared_imag = derivative(shared_feature)
    balance_real, balance_imag = derivative(balance_contrast)
    cumulative_real, cumulative_imag = derivative(cumulative_contrast)
    ew_real, ew_imag = derivative(ew_contrast)
    denominator = parent_observation.supervised_tokens
    shared_coordinates = tuple(
        float(value / denominator)
        for value in (
            shared_real,
            shared_imag,
            balance_real,
            balance_imag,
        )
    )
    cumulative_jacobian = (
        *shared_coordinates,
        float(cumulative_real / denominator),
        float(cumulative_imag / denominator),
    )
    ew_jacobian = (
        *shared_coordinates,
        float(ew_real / denominator),
        float(ew_imag / denominator),
    )

    def feature_std(value: Tensor) -> float:
        selected = value[feature_active].to(torch.float64)
        return (
            0.0
            if selected.numel() <= 1
            else float(selected.std(unbiased=False))
        )

    parent_active = parent_modal[feature_active].to(torch.float64)
    top_active = selected_modal[feature_active].to(torch.float64)
    total_energy = float(parent_active.square().sum())
    top_energy = float(top_active.square().sum())
    signed = (
        parent_observation.candidate_summed_nll
        - parent_observation.source_summed_nll
    ) / denominator
    for name, tensor in (
        ("shared", shared_feature),
        ("balance contrast", balance_contrast),
        ("cumulative contrast", cumulative_contrast),
        ("EW contrast", ew_contrast),
    ):
        if not bool(torch.isfinite(tensor[feature_active]).all()):
            raise ValueError(f"{name} occupancy feature became nonfinite")
    return GemmaIterativeOccupancyConformalRouteFitRecord(
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
        cumulative_occupancy_feature_sha256=_tensor_sha256(
            cumulative_occupancy
        ),
        ew_occupancy_feature_sha256=_tensor_sha256(ew_occupancy),
        shared_feature_sha256=_tensor_sha256(shared_feature),
        balance_contrast_feature_sha256=_tensor_sha256(balance_contrast),
        cumulative_occupancy_contrast_feature_sha256=_tensor_sha256(
            cumulative_contrast
        ),
        ew_occupancy_contrast_feature_sha256=_tensor_sha256(ew_contrast),
        supervised_tokens=denominator,
        parent_signed_delta_nll_per_token=signed,
        jacobian_by_cumulative_occupancy_conformal_coefficient=(
            cumulative_jacobian
        ),
        jacobian_by_ew_occupancy_conformal_coefficient=ew_jacobian,
        active_row_count=int(active.sum()),
        negative_balance_row_count=negative_count,
        nonnegative_balance_row_count=nonnegative_count,
        top_mode_indices=top_indices,
        top_mode_norms=top_norms,
        balance_feature_std=feature_std(balance),
        cumulative_occupancy_feature_std=feature_std(
            cumulative_occupancy
        ),
        ew_occupancy_feature_std=feature_std(ew_occupancy),
        top2_modal_energy_fraction=(
            0.0 if total_energy == 0.0 else top_energy / total_energy
        ),
    )


_ROUTE_STATE_SEMANTICS = (
    "top2_parent_lag_b_modal_cumulative_balance_plus_centered_occupancy_v1"
)
_ROUTE_SEMANTICS = "delta=(g*selected_top2)@C(C0+g*Cg+o*Co)"
_RUNTIME_STATE_FLOAT_COUNT = 4
_LINEAR_MACS_PER_TOKEN = 10
_NONLINEAR_SCALAR_OPS_PER_TOKEN = 6


def _campaign_recipe(
    occupancy_kind: OccupancyKind,
) -> GemmaIterativeResidualCampaignRecipe:
    kind = _kind(occupancy_kind)
    is_ew = kind == CENTERED_EW_OCCUPANCY
    return GemmaIterativeResidualCampaignRecipe(
        recipe_id=f"causal_top2_{kind}_occupancy_conformal_route",
        fit_record_jacobian_field=(
            "jacobian_by_cumulative_occupancy_conformal_coefficient"
            if not is_ew
            else "jacobian_by_ew_occupancy_conformal_coefficient"
        ),
        fold_coefficient_field=(
            "coefficients_by_occupancy_conformal_coefficient"
        ),
        coefficient_count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        learned_parameter_attribute="marginal_learned_float_scalar_count",
        learned_parameter_fallback_attribute=None,
        expected_learned_parameter_count=(
            OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
        ),
        logical_macs_attribute=(
            "marginal_logical_macs_per_token_upper_bound"
        ),
        logical_macs_fallback_attribute=None,
        expected_logical_macs_per_token_upper_bound=(
            _LINEAR_MACS_PER_TOKEN
        ),
        logical_macs_must_equal_residual_width=False,
        extra_resource_expectations=(
            (
                "prepared_float_scalar_count",
                "marginal_prepared_float_scalar_count",
                9 if is_ew else 8,
            ),
            (
                "derived_constant_float_count",
                "marginal_derived_prepared_float_scalar_count",
                2,
            ),
            (
                "fixed_decay_float_scalar_count",
                "marginal_fixed_decay_float_scalar_count",
                int(is_ew),
            ),
            (
                "runtime_state_float_count_per_sequence",
                "runtime_state_float_scalars_per_sequence",
                _RUNTIME_STATE_FLOAT_COUNT,
            ),
            (
                "nonlinear_scalar_ops_per_token_upper_bound",
                "nonlinear_scalar_ops_per_token_upper_bound",
                _NONLINEAR_SCALAR_OPS_PER_TOKEN,
            ),
            (
                "linear_accumulator_scalar_ops_per_token_upper_bound",
                "linear_accumulator_scalar_ops_per_token_upper_bound",
                8 if is_ew else 6,
            ),
            (
                "explicit_scalar_multiplications_per_token_upper_bound",
                "explicit_scalar_multiplications_per_token_upper_bound",
                2 if is_ew else 0,
            ),
            (
                "zero_denominator_comparisons_per_token_upper_bound",
                "zero_denominator_comparisons_per_token_upper_bound",
                1,
            ),
            (
                "negative_balance_comparisons_per_token_upper_bound",
                "negative_balance_comparisons_per_token_upper_bound",
                1,
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
                "shared_vjp_family_blocked_occupancy_selection",
            ),
            ("occupancy_kind", kind),
            ("occupancy_conformal_matrix_shape", (2, 2)),
            (
                "occupancy_conformal_coefficient_order",
                _COEFFICIENT_ORDER,
            ),
            ("route_state_semantics", _ROUTE_STATE_SEMANTICS),
            ("occupancy_route_semantics", _ROUTE_SEMANTICS),
            (
                "corner_operator_norm_bound",
                OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
            ),
        ),
        provider_audit_fields=(
            ("routed_parent_decoder_mode_indices", "top_mode_indices"),
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
            "global_radial_four_corner_operator_norm_projection_is_"
            "linearization_extrapolation"
        ),
        resource_envelope_error=(
            "fixed occupancy conformal route exceeds its resource envelope"
        ),
        linearization_error=(
            "OOF occupancy route requires six finite coordinates"
        ),
    )


GEMMA_ITERATIVE_CUMULATIVE_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE = (
    _campaign_recipe(CENTERED_CUMULATIVE_OCCUPANCY)
)
GEMMA_ITERATIVE_EW_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE = _campaign_recipe(
    CENTERED_EW_OCCUPANCY
)


def _resource_payload(occupancy_kind: OccupancyKind) -> dict[str, object]:
    kind = _kind(occupancy_kind)
    is_ew = kind == CENTERED_EW_OCCUPANCY
    return {
        "semantics": _ROUTE_SEMANTICS,
        "route_state_semantics": _ROUTE_STATE_SEMANTICS,
        "occupancy_kind": kind,
        "occupancy_ew_half_life": (
            OCCUPANCY_EW_HALF_LIFE if is_ew else None
        ),
        "occupancy_ew_decay": OCCUPANCY_EW_DECAY if is_ew else None,
        "coefficient_order": _COEFFICIENT_ORDER,
        "learned_float_scalar_count": (
            OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT
        ),
        "derived_prepared_float_scalar_count": 2,
        "fixed_decay_float_scalar_count": int(is_ew),
        "prepared_float_scalar_count": 9 if is_ew else 8,
        "logical_linear_macs_per_token_upper_bound": (
            _LINEAR_MACS_PER_TOKEN
        ),
        "nonlinear_scalar_ops_per_token_upper_bound": (
            _NONLINEAR_SCALAR_OPS_PER_TOKEN
        ),
        "linear_accumulator_scalar_ops_per_token_upper_bound": (
            8 if is_ew else 6
        ),
        "explicit_scalar_multiplications_per_token_upper_bound": (
            2 if is_ew else 0
        ),
        "zero_denominator_comparisons_per_token_upper_bound": 1,
        "negative_balance_comparisons_per_token_upper_bound": 1,
        "runtime_state_float_scalars_per_sequence": (
            _RUNTIME_STATE_FLOAT_COUNT
        ),
        "parent_modal_values_reused": True,
        "parent_lag_cache_owned_upstream": True,
        "parent_decoder_invocations_per_token": 1,
    }


def gemma_causal_top2_occupancy_conformal_route_provider_artifact_sha256(
    *,
    occupancy_kind: str,
    parent_artifact_sha256: str,
    parent_h4_artifact_sha256: str,
    bridge_binding_sha256: str,
    decoder_sha256: str,
    lag_kernel_sha256: str,
    fold_receipt_sha256: str,
    top_mode_indices: Sequence[int],
    top_mode_norms: Sequence[float],
    coefficients_by_occupancy_conformal_coefficient: Sequence[float],
) -> str:
    """Replay a provider identity without loading its parent tensors."""

    kind = _kind(occupancy_kind)
    indices = _indices2(top_mode_indices, label="top mode indices")
    norms = _float_tuple(top_mode_norms, count=2, label="top mode norms")
    if any(value <= 0.0 for value in norms):
        raise ValueError("top mode norms must be positive")
    coefficients = _float_tuple(
        coefficients_by_occupancy_conformal_coefficient,
        count=OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT,
        label="occupancy conformal coefficients",
    )
    corners = _corner_operator_norms(coefficients)
    if max(corners) > OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12:
        raise ValueError("occupancy coefficients exceed the corner bound")
    resources = _resource_payload(kind)
    payload = {
        "semantics": _ROUTE_SEMANTICS,
        "route_state_semantics": _ROUTE_STATE_SEMANTICS,
        "site": _H4_SITE,
        "occupancy_kind": kind,
        "occupancy_ew_half_life": (
            OCCUPANCY_EW_HALF_LIFE
            if kind == CENTERED_EW_OCCUPANCY
            else None
        ),
        "occupancy_ew_decay": (
            OCCUPANCY_EW_DECAY
            if kind == CENTERED_EW_OCCUPANCY
            else None
        ),
        "parent_artifact_sha256": _require_sha256(
            parent_artifact_sha256, label="parent artifact"
        ),
        "parent_h4_artifact_sha256": _require_sha256(
            parent_h4_artifact_sha256, label="parent H4 artifact"
        ),
        "bridge_binding_sha256": _require_sha256(
            bridge_binding_sha256, label="bridge binding"
        ),
        "decoder_sha256": _require_sha256(
            decoder_sha256, label="decoder"
        ),
        "lag_kernel_sha256": _require_sha256(
            lag_kernel_sha256, label="lag kernel"
        ),
        "fold_receipt_sha256": _require_sha256(
            fold_receipt_sha256, label="fold receipt"
        ),
        "top_mode_indices": indices,
        "top_mode_norms": norms,
        "coefficients_by_occupancy_conformal_coefficient": coefficients,
        "coefficient_order": _COEFFICIENT_ORDER,
        "corner_order": _CORNER_ORDER,
        "corner_operator_norms": corners,
        "operator_norm_bound": OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
        "resources": resources,
        "resource_receipt_sha256": _sha256(_RESOURCE_DOMAIN, resources),
    }
    return _sha256(_PROVIDER_DOMAIN, payload)


@dataclass(frozen=True, slots=True)
class GemmaCausalTop2OccupancyConformalRouteH4Provider(
    Gemma3L3L4CorrectionProvider
):
    """Authenticated six-scalar route with four causal state floats."""

    parent_h4: GemmaCausalResidualHead
    parent_artifact_sha256: str
    fold_fit: GemmaIterativeOccupancyConformalRouteFoldFit
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
            self.fold_fit, GemmaIterativeOccupancyConformalRouteFoldFit
        ):
            raise TypeError("fold_fit must be a strict occupancy-route fit")
        self.fold_fit.validate_integrity()
        parent_h4_sha256 = _require_sha256(
            parent.artifact_sha256, label="parent H4 artifact"
        )
        parent_artifact_sha256 = _require_sha256(
            self.parent_artifact_sha256, label="parent artifact"
        )
        bridge_binding = _require_sha256(
            parent.bridge_binding_sha256, label="bridge binding"
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
            gemma_causal_top2_occupancy_conformal_route_provider_artifact_sha256(
                occupancy_kind=self.occupancy_kind,
                parent_artifact_sha256=parent_artifact_sha256,
                parent_h4_artifact_sha256=parent_h4_sha256,
                bridge_binding_sha256=bridge_binding,
                decoder_sha256=decoder_sha256,
                lag_kernel_sha256=lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=top_indices,
                top_mode_norms=top_norms,
                coefficients_by_occupancy_conformal_coefficient=(
                    self.coefficients_by_occupancy_conformal_coefficient
                ),
            ),
        )
        self.validate_integrity()

    @property
    def occupancy_kind(self) -> OccupancyKind:
        return self.fold_fit.occupancy_kind

    @property
    def bridge_binding_sha256(self) -> str:
        return self._bridge_binding_sha256

    @property
    def coefficients_by_occupancy_conformal_coefficient(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        return self.fold_fit.coefficients_by_occupancy_conformal_coefficient

    @property
    def occupancy_conformal_coefficient_order(
        self,
    ) -> tuple[str, str, str, str, str, str]:
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
    def corner_operator_norms(self) -> tuple[float, float, float, float]:
        return _corner_operator_norms(
            self.coefficients_by_occupancy_conformal_coefficient
        )

    @property
    def route_state_semantics(self) -> str:
        return _ROUTE_STATE_SEMANTICS

    @property
    def occupancy_route_semantics(self) -> str:
        return _ROUTE_SEMANTICS

    @property
    def width(self) -> int:
        return self.parent_h4.width

    @property
    def marginal_learned_float_scalar_count(self) -> int:
        return OCCUPANCY_CONFORMAL_COEFFICIENT_COUNT

    @property
    def marginal_derived_prepared_float_scalar_count(self) -> int:
        return 2

    @property
    def marginal_fixed_decay_float_scalar_count(self) -> int:
        return int(self.occupancy_kind == CENTERED_EW_OCCUPANCY)

    @property
    def marginal_prepared_float_scalar_count(self) -> int:
        return 9 if self.occupancy_kind == CENTERED_EW_OCCUPANCY else 8

    @property
    def marginal_logical_macs_per_token_upper_bound(self) -> int:
        return _LINEAR_MACS_PER_TOKEN

    @property
    def nonlinear_scalar_ops_per_token_upper_bound(self) -> int:
        return _NONLINEAR_SCALAR_OPS_PER_TOKEN

    @property
    def linear_accumulator_scalar_ops_per_token_upper_bound(self) -> int:
        return 8 if self.occupancy_kind == CENTERED_EW_OCCUPANCY else 6

    @property
    def explicit_scalar_multiplications_per_token_upper_bound(self) -> int:
        return 2 if self.occupancy_kind == CENTERED_EW_OCCUPANCY else 0

    @property
    def zero_denominator_comparisons_per_token_upper_bound(self) -> int:
        return 1

    @property
    def negative_balance_comparisons_per_token_upper_bound(self) -> int:
        return 1

    @property
    def runtime_state_float_scalars_per_sequence(self) -> int:
        return _RUNTIME_STATE_FLOAT_COUNT

    @property
    def parent_decoder_invocations_per_token(self) -> int:
        return 1

    @property
    def resource_receipt(self) -> Mapping[str, object]:
        return _resource_payload(self.occupancy_kind)

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
        self.fold_fit.validate_integrity()
        top_indices, top_norms = top2_lag_b_output_modes(parent)
        expected = (
            gemma_causal_top2_occupancy_conformal_route_provider_artifact_sha256(
                occupancy_kind=self.occupancy_kind,
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=self._parent_h4_sha256,
                bridge_binding_sha256=self._bridge_binding_sha256,
                decoder_sha256=self._decoder_sha256,
                lag_kernel_sha256=self._lag_kernel_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                top_mode_indices=self._top_mode_indices,
                top_mode_norms=self._top_mode_norms,
                coefficients_by_occupancy_conformal_coefficient=(
                    self.coefficients_by_occupancy_conformal_coefficient
                ),
            )
        )
        if (
            parent.artifact_sha256 != self._parent_h4_sha256
            or parent.bridge_binding_sha256 != self._bridge_binding_sha256
            or _tensor_sha256(parent.decoder) != self._decoder_sha256
            or _tensor_sha256(parent.lag_kernel) != self._lag_kernel_sha256
            or top_indices != self._top_mode_indices
            or top_norms != self._top_mode_norms
            or expected != self.artifact_sha256
            or self.resource_receipt_sha256
            != _sha256(_RESOURCE_DOMAIN, _resource_payload(self.occupancy_kind))
        ):
            raise RuntimeError("causal top-two occupancy provider drifted")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> GemmaCausalTop2OccupancyConformalRouteState:
        self.validate_integrity()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("occupancy-route batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("occupancy-route state must be floating point")
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        return GemmaCausalTop2OccupancyConformalRouteState(
            balance_numerator=zeros,
            balance_denominator=zeros.clone(),
            occupancy_numerator=zeros.clone(),
            occupancy_denominator=zeros.clone(),
            occupancy_kind=self.occupancy_kind,
            provider_artifact_sha256=self.artifact_sha256,
        )

    def route_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2OccupancyConformalRouteState,
    ) -> tuple[Tensor, GemmaCausalTop2OccupancyConformalRouteState]:
        """Route one upstream lag-aware chunk and advance both carries."""

        self.validate_integrity()
        prefix.validate_integrity()
        if not isinstance(
            state, GemmaCausalTop2OccupancyConformalRouteState
        ):
            raise TypeError("state must be an occupancy-route state")
        state.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.shape
            != (*prefix.logical_positions.shape, self.parent_h4.rank)
            or not parent_modal.is_floating_point()
            or state.provider_artifact_sha256 != self.artifact_sha256
            or state.occupancy_kind != self.occupancy_kind
            or state.batch_size != parent_modal.shape[0]
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or any(
                value.device != parent_modal.device
                or value.dtype != parent_modal.dtype
                for value in (
                    state.balance_numerator,
                    state.balance_denominator,
                    state.occupancy_numerator,
                    state.occupancy_denominator,
                )
            )
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
        balance, balance_numerator, balance_denominator = _balance_feature(
            prefix=prefix,
            parent_modal=parent_modal,
            top_mode_indices=self.top_mode_indices,
            top_mode_norms=self.top_mode_norms,
            initial_numerator=state.balance_numerator,
            initial_denominator=state.balance_denominator,
        )
        if (
            bool(active.any())
            and bool((balance[active].abs() > 1.0 + 1.0e-6).any())
        ):
            raise RuntimeError("causal balance escaped its normalized range")
        (
            occupancy,
            occupancy_numerator,
            occupancy_denominator,
            _negative_count,
            _nonnegative_count,
        ) = _occupancy_feature(
            balance=balance,
            active=active,
            initial_numerator=state.occupancy_numerator,
            initial_denominator=state.occupancy_denominator,
            occupancy_kind=self.occupancy_kind,
            collect_sign_counts=False,
        )
        selected_indices = torch.tensor(
            self.top_mode_indices,
            device=parent_modal.device,
            dtype=torch.int64,
        )
        selected = parent_modal.index_select(2, selected_indices)
        a0, b0, ag, bg, ao, bo = torch.tensor(
            self.coefficients_by_occupancy_conformal_coefficient,
            device=parent_modal.device,
            dtype=parent_modal.dtype,
        ).unbind()
        a = a0 + balance * ag + occupancy * ao
        b = b0 + balance * bg + occupancy * bo
        gated = balance.unsqueeze(-1) * selected
        delta = torch.stack(
            (
                gated[..., 0] * a + gated[..., 1] * b,
                -gated[..., 0] * b + gated[..., 1] * a,
            ),
            dim=-1,
        )
        routed = parent_modal.clone()
        if any(
            value != 0.0
            for value in self.coefficients_by_occupancy_conformal_coefficient
        ):
            routed.index_copy_(2, selected_indices, selected + delta)
        next_state = GemmaCausalTop2OccupancyConformalRouteState(
            balance_numerator=balance_numerator.detach().contiguous(),
            balance_denominator=balance_denominator.detach().contiguous(),
            occupancy_numerator=occupancy_numerator.detach().contiguous(),
            occupancy_denominator=occupancy_denominator.detach().contiguous(),
            occupancy_kind=self.occupancy_kind,
            provider_artifact_sha256=self.artifact_sha256,
        )
        if (
            bool(active.any())
            and not bool(torch.isfinite(routed[active]).all())
        ):
            raise ValueError("occupancy route became nonfinite")
        if bool(inactive.any()) and not bool((routed[inactive] == 0).all()):
            raise RuntimeError("occupancy route is off support")
        self.validate_integrity()
        prefix.validate_integrity()
        return routed, next_state

    def correction_from_parent_modal_with_state(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        parent_modal: Tensor,
        state: GemmaCausalTop2OccupancyConformalRouteState,
    ) -> tuple[Tensor, GemmaCausalTop2OccupancyConformalRouteState]:
        """Decode an upstream lag-aware modal chunk without resetting lag."""

        prefix_sha256 = prefix.artifact_sha256
        routed, next_state = self.route_parent_modal_with_state(
            prefix, parent_modal, state
        )
        result = self.parent_h4.decode_modal(
            prefix, routed, like=prefix.clamped_y3
        )
        self.parent_h4.validate_integrity()
        prefix.validate_integrity()
        if prefix.artifact_sha256 != prefix_sha256:
            raise RuntimeError("occupancy route mutated the prefix")
        self.validate_integrity()
        return result, next_state

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Use zero carries for the full-sequence correction-provider ABI."""

        self.validate_integrity()
        parent_modal = self.parent_h4.modal_correction(
            prefix, realized_state
        )
        state = self.initial_state(
            prefix.logical_positions.shape[0],
            device=parent_modal.device,
            dtype=parent_modal.dtype,
        )
        result, _next_state = self.correction_from_parent_modal_with_state(
            prefix, parent_modal, state
        )
        return result


def fit_gemma_iterative_occupancy_conformal_route_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    occupancy_kind: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    fit = fit_gemma_iterative_occupancy_conformal_route_fold(
        records,
        held_family_id=held_family,
        occupancy_kind=occupancy_kind,
    )
    return GemmaCausalTop2OccupancyConformalRouteH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_occupancy_conformal_route_full_provider(
    *,
    records: Sequence[object],
    occupancy_kind: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_occupancy_conformal_route_fold_provider(
        records=records,
        held_family="__full_fit__",
        occupancy_kind=occupancy_kind,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def fit_gemma_iterative_cumulative_occupancy_route_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeOccupancyConformalRouteFoldFit:
    return fit_gemma_iterative_occupancy_conformal_route_fold(
        records,
        held_family_id=held_family_id,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
    )


def fit_gemma_iterative_ew_occupancy_route_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeOccupancyConformalRouteFoldFit:
    return fit_gemma_iterative_occupancy_conformal_route_fold(
        records,
        held_family_id=held_family_id,
        occupancy_kind=CENTERED_EW_OCCUPANCY,
    )


def fit_gemma_iterative_cumulative_occupancy_route_full_provider(
    *,
    records: Sequence[object],
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_occupancy_conformal_route_full_provider(
        records=records,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def fit_gemma_iterative_cumulative_occupancy_route_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_occupancy_conformal_route_fold_provider(
        records=records,
        held_family=held_family,
        occupancy_kind=CENTERED_CUMULATIVE_OCCUPANCY,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def fit_gemma_iterative_ew_occupancy_route_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_occupancy_conformal_route_fold_provider(
        records=records,
        held_family=held_family,
        occupancy_kind=CENTERED_EW_OCCUPANCY,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def fit_gemma_iterative_ew_occupancy_route_full_provider(
    *,
    records: Sequence[object],
    parent_h4: GemmaCausalResidualHead,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return fit_gemma_iterative_occupancy_conformal_route_full_provider(
        records=records,
        occupancy_kind=CENTERED_EW_OCCUPANCY,
        parent_h4=parent_h4,
        parent_artifact_sha256=parent_artifact_sha256,
    )
