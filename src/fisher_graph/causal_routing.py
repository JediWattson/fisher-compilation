"""Causal exponential-state features for conditional route prediction.

The pointwise router in :mod:`fisher_graph.conditional_routing` can consume a
transformer boundary that already summarizes its causal prefix.  That
assumption does not hold at an embedding boundary.  This module supplies a
small, model-agnostic alternative: fixed exponential state channels summarize
raw boundary rows at exact logical-position gaps, and a fitted
``PointwiseCausalRouter`` classifies the flattened state.

For query position ``q``, channel ``c``, and raw feature ``d`` the state is

``state[q, c, d] = sum_{k <= q} exp(-rate[c] * (q - k)) * input[k, d]``.

Only valid keys participate and only valid queries are classified.  Rates are
fixed nonnegative scalars, not learned parameters.  The implementation uses
the same definition for contiguous, gapped, right-padded, and sparse layouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .conditional_routing import (
    PointwiseCausalRouter,
    RouterClassificationMetrics,
    fit_pointwise_causal_router,
)


_ARTIFACT_KIND = "fisher_graph.causal_exponential_state_router"
_FORMAT_VERSION = 1


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_decay_rates(decay_rates: Tensor) -> Tensor:
    if (
        not isinstance(decay_rates, Tensor)
        or not decay_rates.is_floating_point()
        or decay_rates.ndim != 1
        or decay_rates.numel() == 0
    ):
        raise ValueError(
            "decay_rates must be a nonempty floating vector"
        )
    rates = decay_rates.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("decay_rates must be finite and nonnegative")
    return rates.clone()


def _validate_monotonic_positions(
    positions: Tensor,
    valid_mask: Tensor,
    *,
    label: str,
) -> None:
    for row in range(positions.shape[0]):
        selected = positions[row][valid_mask[row]]
        if selected.numel() > 1 and (selected[1:] <= selected[:-1]).any():
            raise ValueError(
                f"valid {label} must be strictly increasing in tensor order"
            )


@dataclass(frozen=True, slots=True)
class _CausalInputs:
    values: Tensor
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    logical_positions: Tensor
    key_logical_positions: Tensor
    allowed_pairs: Tensor


def _normalize_causal_inputs(
    boundary_inputs: Tensor,
    *,
    query_valid_mask: Tensor | None,
    key_valid_mask: Tensor | None,
    logical_positions: Tensor | None,
    key_logical_positions: Tensor | None,
) -> _CausalInputs:
    if (
        not isinstance(boundary_inputs, Tensor)
        or boundary_inputs.ndim != 3
        or boundary_inputs.shape[0] == 0
        or boundary_inputs.shape[1] == 0
        or boundary_inputs.shape[2] == 0
        or not boundary_inputs.is_floating_point()
        or not torch.isfinite(boundary_inputs).all()
    ):
        raise ValueError(
            "boundary_inputs must be a finite floating Tensor with shape "
            "[batch, key positions, input features]"
        )
    batch, key_count, _ = boundary_inputs.shape
    device = boundary_inputs.device

    if key_valid_mask is None:
        keys_valid = torch.ones(
            batch,
            key_count,
            dtype=torch.bool,
            device=device,
        )
    else:
        if (
            not isinstance(key_valid_mask, Tensor)
            or key_valid_mask.dtype is not torch.bool
            or key_valid_mask.shape != (batch, key_count)
            or key_valid_mask.device != device
        ):
            raise ValueError(
                "key_valid_mask must be a matching boolean Tensor"
            )
        keys_valid = key_valid_mask

    query_shapes: list[tuple[int, int]] = []
    if query_valid_mask is not None:
        if (
            not isinstance(query_valid_mask, Tensor)
            or query_valid_mask.dtype is not torch.bool
            or query_valid_mask.ndim != 2
            or query_valid_mask.shape[0] != batch
            or query_valid_mask.device != device
        ):
            raise ValueError(
                "query_valid_mask must be a batch-aligned boolean Tensor"
            )
        query_shapes.append(tuple(query_valid_mask.shape))
    if logical_positions is not None:
        if (
            not isinstance(logical_positions, Tensor)
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or logical_positions.ndim != 2
            or logical_positions.shape[0] != batch
            or logical_positions.device != device
        ):
            raise ValueError(
                "logical_positions must be a batch-aligned integer Tensor"
            )
        query_shapes.append(tuple(logical_positions.shape))
    if query_shapes and any(shape != query_shapes[0] for shape in query_shapes):
        raise ValueError(
            "query_valid_mask and logical_positions must share shape"
        )
    query_count = query_shapes[0][1] if query_shapes else key_count
    if query_count == 0:
        raise ValueError("query inputs must contain at least one position")

    if query_valid_mask is None:
        if query_count == key_count:
            queries_valid = keys_valid
        else:
            queries_valid = torch.ones(
                batch,
                query_count,
                dtype=torch.bool,
                device=device,
            )
    else:
        queries_valid = query_valid_mask

    if key_logical_positions is None:
        key_positions = torch.arange(
            key_count,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0).expand(batch, -1)
    else:
        if (
            not isinstance(key_logical_positions, Tensor)
            or key_logical_positions.dtype not in (
                torch.int32,
                torch.int64,
            )
            or key_logical_positions.shape != (batch, key_count)
            or key_logical_positions.device != device
        ):
            raise ValueError(
                "key_logical_positions must be a matching integer Tensor"
            )
        key_positions = key_logical_positions.to(dtype=torch.int64)

    if logical_positions is None:
        query_positions = torch.arange(
            query_count,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0).expand(batch, -1)
    else:
        query_positions = logical_positions.to(dtype=torch.int64)

    if (query_positions[queries_valid] < 0).any():
        raise ValueError("valid logical_positions cannot be negative")
    if (key_positions[keys_valid] < 0).any():
        raise ValueError("valid key_logical_positions cannot be negative")
    _validate_monotonic_positions(
        query_positions,
        queries_valid,
        label="logical_positions",
    )
    _validate_monotonic_positions(
        key_positions,
        keys_valid,
        label="key_logical_positions",
    )

    gaps = (
        query_positions.unsqueeze(2)
        - key_positions.unsqueeze(1)
    )
    allowed = (
        queries_valid.unsqueeze(2)
        & keys_valid.unsqueeze(1)
        & (gaps >= 0)
    )
    return _CausalInputs(
        values=boundary_inputs,
        query_valid_mask=queries_valid,
        key_valid_mask=keys_valid,
        logical_positions=query_positions,
        key_logical_positions=key_positions,
        allowed_pairs=allowed,
    )


def causal_exponential_state_features(
    boundary_inputs: Tensor,
    decay_rates: Tensor,
    *,
    query_valid_mask: Tensor | None = None,
    key_valid_mask: Tensor | None = None,
    logical_positions: Tensor | None = None,
    key_logical_positions: Tensor | None = None,
) -> Tensor:
    """Return flattened causal exponential states for every query row.

    The result has shape
    ``[batch, query positions, state channels * input features]``.  Invalid
    query rows are exactly zero.  Half and bfloat16 inputs compute in float32;
    other floating dtypes are preserved.
    """

    rates = _normalized_decay_rates(decay_rates)
    normalized = _normalize_causal_inputs(
        boundary_inputs,
        query_valid_mask=query_valid_mask,
        key_valid_mask=key_valid_mask,
        logical_positions=logical_positions,
        key_logical_positions=key_logical_positions,
    )
    compute_dtype = (
        torch.float32
        if boundary_inputs.dtype in (torch.float16, torch.bfloat16)
        else boundary_inputs.dtype
    )
    values = normalized.values.to(dtype=compute_dtype)
    query_positions = normalized.logical_positions
    key_positions = normalized.key_logical_positions
    gaps = (
        query_positions.unsqueeze(2)
        - key_positions.unsqueeze(1)
    ).clamp_min(0).to(dtype=compute_dtype)
    live_rates = rates.to(
        device=boundary_inputs.device,
        dtype=compute_dtype,
    )
    weights = torch.exp(
        -gaps.unsqueeze(-1) * live_rates.view(1, 1, 1, -1)
    )
    weights = torch.where(
        normalized.allowed_pairs.unsqueeze(-1),
        weights,
        torch.zeros_like(weights),
    )
    states = torch.einsum("bqkc,bkd->bqcd", weights, values)
    states = torch.where(
        normalized.query_valid_mask.unsqueeze(-1).unsqueeze(-1),
        states,
        torch.zeros_like(states),
    )
    return states.flatten(start_dim=2)


@dataclass(frozen=True, slots=True)
class CausalRouterAccounting:
    """Logical nonzero MACs and stored scalar state for one router call.

    ``state_macs`` counts one multiply-accumulate for every visible
    key/query/channel/raw-feature tuple.  ``classifier_macs`` counts the dense
    affine classifier on valid query rows.  Gap construction, exponentials,
    masking, feature standardization, bias additions, and route ``argmax`` are
    deliberately not MACs; backend latency must account for those operations
    separately.
    """

    sequences: int
    query_positions: int
    key_positions: int
    valid_queries: int
    valid_keys: int
    causal_pairs: int
    input_features: int
    state_channels: int
    routes: int
    state_macs: int
    classifier_macs: int
    fixed_state_parameters: int
    normalization_parameters: int
    classifier_parameters: int

    def __post_init__(self) -> None:
        for label in (
            "sequences",
            "query_positions",
            "key_positions",
            "input_features",
            "state_channels",
            "routes",
        ):
            _positive_int(getattr(self, label), label=label)
        for label in (
            "valid_queries",
            "valid_keys",
            "causal_pairs",
            "state_macs",
            "classifier_macs",
            "fixed_state_parameters",
            "normalization_parameters",
            "classifier_parameters",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")

    @property
    def total_macs(self) -> int:
        return self.state_macs + self.classifier_macs

    @property
    def total_stored_parameters(self) -> int:
        return (
            self.fixed_state_parameters
            + self.normalization_parameters
            + self.classifier_parameters
        )


@dataclass(frozen=True, slots=True)
class CausalExponentialStateRouter:
    """A serialized fixed-state causal feature map plus ridge classifier."""

    input_features: int
    decay_rates: Tensor
    pointwise_router: PointwiseCausalRouter

    def __post_init__(self) -> None:
        features = _positive_int(
            self.input_features,
            label="input_features",
        )
        rates = _normalized_decay_rates(self.decay_rates)
        if not isinstance(self.pointwise_router, PointwiseCausalRouter):
            raise TypeError(
                "pointwise_router must be a PointwiseCausalRouter"
            )
        if self.pointwise_router.input_features != features * rates.numel():
            raise ValueError(
                "pointwise router feature width must equal input features "
                "times state channels"
            )
        object.__setattr__(self, "decay_rates", rates)
        object.__setattr__(
            self,
            "pointwise_router",
            PointwiseCausalRouter.from_state_dict(
                self.pointwise_router.state_dict()
            ),
        )

    @property
    def state_channels(self) -> int:
        return self.decay_rates.numel()

    @property
    def state_features(self) -> int:
        return self.input_features * self.state_channels

    @property
    def routes(self) -> int:
        return self.pointwise_router.routes

    def features(
        self,
        boundary_inputs: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        if (
            not isinstance(boundary_inputs, Tensor)
            or boundary_inputs.ndim != 3
            or boundary_inputs.shape[-1] != self.input_features
        ):
            raise ValueError(
                "boundary_inputs must match the router input width"
            )
        return causal_exponential_state_features(
            boundary_inputs,
            self.decay_rates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )

    def logits(
        self,
        boundary_inputs: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        normalized = _normalize_causal_inputs(
            boundary_inputs,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        features = self.features(
            boundary_inputs,
            query_valid_mask=normalized.query_valid_mask,
            key_valid_mask=normalized.key_valid_mask,
            logical_positions=normalized.logical_positions,
            key_logical_positions=normalized.key_logical_positions,
        )
        flat_features = features.reshape(-1, self.state_features)
        flat_valid = normalized.query_valid_mask.reshape(-1)
        selected = flat_valid.nonzero(as_tuple=False).flatten()
        flat_logits = features.new_zeros(
            flat_valid.numel(),
            self.routes,
        )
        if selected.numel():
            selected_logits = self.pointwise_router.logits(
                flat_features.index_select(0, selected)
            )
            flat_logits.index_copy_(0, selected, selected_logits)
        return flat_logits.reshape(
            normalized.query_valid_mask.shape[0],
            normalized.query_valid_mask.shape[1],
            self.routes,
        )

    def predict(
        self,
        boundary_inputs: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        return self.logits(
            boundary_inputs,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        ).argmax(dim=-1)

    def analytic_accounting(
        self,
        boundary_inputs: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> CausalRouterAccounting:
        normalized = _normalize_causal_inputs(
            boundary_inputs,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        valid_queries = int(normalized.query_valid_mask.sum().item())
        valid_keys = int(normalized.key_valid_mask.sum().item())
        causal_pairs = int(normalized.allowed_pairs.sum().item())
        classifier_parameters = (
            self.pointwise_router.weight.numel()
            + self.pointwise_router.bias.numel()
        )
        return CausalRouterAccounting(
            sequences=boundary_inputs.shape[0],
            query_positions=normalized.query_valid_mask.shape[1],
            key_positions=boundary_inputs.shape[1],
            valid_queries=valid_queries,
            valid_keys=valid_keys,
            causal_pairs=causal_pairs,
            input_features=self.input_features,
            state_channels=self.state_channels,
            routes=self.routes,
            state_macs=(
                causal_pairs
                * self.state_channels
                * self.input_features
            ),
            classifier_macs=(
                valid_queries * self.state_features * self.routes
            ),
            fixed_state_parameters=self.state_channels,
            normalization_parameters=(
                self.pointwise_router.feature_mean.numel()
                + self.pointwise_router.feature_scale.numel()
            ),
            classifier_parameters=classifier_parameters,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "input_features": self.input_features,
            "decay_rates": self.decay_rates.detach().clone(),
            "pointwise_router": self.pointwise_router.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CausalExponentialStateRouter:
        expected = {
            "artifact_kind",
            "format_version",
            "input_features",
            "decay_rates",
            "pointwise_router",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "causal exponential-state router fields are invalid"
            )
        if state["artifact_kind"] != _ARTIFACT_KIND:
            raise ValueError(
                "unsupported causal exponential-state router kind"
            )
        if (
            type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported causal exponential-state router version"
            )
        pointwise_state = state["pointwise_router"]
        if not isinstance(pointwise_state, Mapping):
            raise TypeError("pointwise_router must be a mapping")
        return cls(
            input_features=state["input_features"],  # type: ignore[arg-type]
            decay_rates=state["decay_rates"],  # type: ignore[arg-type]
            pointwise_router=PointwiseCausalRouter.from_state_dict(
                pointwise_state
            ),
        )


def fit_causal_exponential_state_router(
    boundary_inputs: Tensor,
    route_labels: Tensor,
    *,
    decay_rates: Tensor,
    route_count: int,
    query_valid_mask: Tensor | None = None,
    key_valid_mask: Tensor | None = None,
    logical_positions: Tensor | None = None,
    key_logical_positions: Tensor | None = None,
    sample_weights: Tensor | None = None,
    ridge: float = 1e-3,
) -> tuple[CausalExponentialStateRouter, RouterClassificationMetrics]:
    """Fit a ridge route classifier over fixed causal exponential states."""

    rates = _normalized_decay_rates(decay_rates)
    normalized = _normalize_causal_inputs(
        boundary_inputs,
        query_valid_mask=query_valid_mask,
        key_valid_mask=key_valid_mask,
        logical_positions=logical_positions,
        key_logical_positions=key_logical_positions,
    )
    features = causal_exponential_state_features(
        boundary_inputs,
        rates,
        query_valid_mask=normalized.query_valid_mask,
        key_valid_mask=normalized.key_valid_mask,
        logical_positions=normalized.logical_positions,
        key_logical_positions=normalized.key_logical_positions,
    )
    pointwise, metrics = fit_pointwise_causal_router(
        features,
        route_labels,
        route_count=route_count,
        valid_mask=normalized.query_valid_mask,
        sample_weights=sample_weights,
        ridge=ridge,
    )
    return (
        CausalExponentialStateRouter(
            input_features=boundary_inputs.shape[-1],
            decay_rates=rates,
            pointwise_router=pointwise,
        ),
        metrics,
    )


__all__ = [
    "CausalExponentialStateRouter",
    "CausalRouterAccounting",
    "causal_exponential_state_features",
    "fit_causal_exponential_state_router",
]
