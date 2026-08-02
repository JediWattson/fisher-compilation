"""Reusable causal innovation-v2 features for fixed generator bases.

Version 1 fixed both the temporal half-life and the unit-temperature
softsign.  This module leaves that published API untouched and factors the
next experiment into three independently testable operations:

* a causal, padding-safe exponentially weighted innovation recurrence with a
  caller-selected half-life or decay;
* robust train-only calibration and application of per-channel softsign
  temperatures; and
* one fixed-basis tangent reduction shared by an arbitrary bank of bounded
  innovation variants.

The separation is intentional.  A caller can fit temperatures on an already
open development split, freeze them, and then apply the exact same causal
recurrence on a later split.  The tangent-bank representation also avoids
duplicating the static generator tangents for every temporal or temperature
variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "DEFAULT_CAUSAL_INNOVATION_HALF_LIFE",
    "CausalModalInnovationV2State",
    "CausalModalInnovationV2Trace",
    "FixedGeneratorActivationTangentBank",
    "RobustChannelTemperatureCalibration",
    "causal_modal_innovation_v2",
    "ew_decay_from_half_life",
    "fit_robust_channel_temperatures",
    "fixed_generator_innovation_activation_tangent_bank",
    "resolve_ew_decay",
    "temperature_softsign",
    "temperature_softsign_bank",
]


DEFAULT_CAUSAL_INNOVATION_HALF_LIFE = 16.0


def _float64_array(
    value: ArrayLike,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    result = np.asarray(value)
    if (
        result.dtype == np.bool_
        or not np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.complexfloating)
    ):
        raise TypeError(f"{label} must contain real numeric values")
    result = np.asarray(result, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {result.shape}")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _readonly_float64(
    value: ArrayLike,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    result = _float64_array(value, label=label, shape=shape).copy()
    result.setflags(write=False)
    return result


def _readonly_bool(
    value: ArrayLike,
    *,
    label: str,
    shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    result = np.asarray(value)
    if result.dtype != np.bool_:
        raise TypeError(f"{label} must be boolean")
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {result.shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _real_scalar(
    value: object,
    *,
    label: str,
    allow_infinity: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if np.isnan(result) or (not allow_infinity and not np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def ew_decay_from_half_life(half_life: float) -> float:
    """Return the EW decay whose mass halves after ``half_life`` updates.

    Positive infinity is accepted as the explicit cumulative-mean control and
    maps to a decay of one.  Very small positive half-lives may underflow to a
    decay of zero, which remains a valid one-step-memory recurrence.
    """

    value = _real_scalar(
        half_life,
        label="half_life",
        allow_infinity=True,
    )
    if value <= 0.0:
        raise ValueError("half_life must be strictly positive")
    if np.isposinf(value):
        return 1.0
    return float(2.0 ** (-1.0 / value))


def resolve_ew_decay(
    *,
    half_life: float | None = None,
    decay: float | None = None,
) -> float:
    """Resolve exactly one temporal parameter into a validated EW decay.

    Supplying neither parameter uses
    :data:`DEFAULT_CAUSAL_INNOVATION_HALF_LIFE`.  Supplying both is rejected
    even when numerically equivalent, so an experiment receipt has one
    unambiguous source of truth.  Direct decays may range from zero through
    one; one is the no-forgetting cumulative-mean control.
    """

    if half_life is not None and decay is not None:
        raise ValueError("provide half_life or decay, not both")
    if half_life is not None:
        return ew_decay_from_half_life(half_life)
    if decay is None:
        return ew_decay_from_half_life(
            DEFAULT_CAUSAL_INNOVATION_HALF_LIFE
        )
    value = _real_scalar(decay, label="decay")
    if value < 0.0 or value > 1.0:
        raise ValueError("decay must lie in [0, 1]")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class CausalModalInnovationV2State:
    """Immutable chunk carry for a channel-generic innovation recurrence."""

    weighted_sum: NDArray[np.float64]
    mass: NDArray[np.float64]
    decay: float

    def __post_init__(self) -> None:
        weighted_sum = _readonly_float64(
            self.weighted_sum,
            label="weighted_sum",
        )
        if weighted_sum.ndim != 2 or weighted_sum.shape[1] == 0:
            raise ValueError(
                "weighted_sum must have shape [batch, channels] "
                "with at least one channel"
            )
        mass = _readonly_float64(
            self.mass,
            label="mass",
            shape=(weighted_sum.shape[0],),
        )
        if bool((mass < 0.0).any()):
            raise ValueError("mass must be nonnegative")
        if bool((weighted_sum[mass == 0.0] != 0.0).any()):
            raise ValueError("zero-mass state must have a zero weighted_sum")
        decay = resolve_ew_decay(decay=self.decay)
        object.__setattr__(self, "weighted_sum", weighted_sum)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "decay", decay)

    @property
    def batch_size(self) -> int:
        return int(self.mass.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.weighted_sum.shape[1])

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        channel_count: int,
        *,
        decay: float,
    ) -> "CausalModalInnovationV2State":
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if type(channel_count) is not int or channel_count <= 0:
            raise ValueError("channel_count must be a positive integer")
        resolved_decay = resolve_ew_decay(decay=decay)
        return cls(
            weighted_sum=np.zeros(
                (batch_size, channel_count),
                dtype=np.float64,
            ),
            mass=np.zeros((batch_size,), dtype=np.float64),
            decay=resolved_decay,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CausalModalInnovationV2Trace:
    """Causal raw and temperature-bounded innovation rows plus final carry."""

    normalized_modal_rows: NDArray[np.float64]
    prior_rows: NDArray[np.float64]
    prior_mass_rows: NDArray[np.float64]
    raw_innovation_rows: NDArray[np.float64]
    bounded_innovation_rows: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    normalization_scales: NDArray[np.float64]
    temperatures: NDArray[np.float64]
    decay: float
    final_state: CausalModalInnovationV2State

    def __post_init__(self) -> None:
        normalized = _readonly_float64(
            self.normalized_modal_rows,
            label="normalized_modal_rows",
        )
        if normalized.ndim != 3 or normalized.shape[2] == 0:
            raise ValueError(
                "normalized_modal_rows must have shape "
                "[batch, time, channels]"
            )
        row_shape = normalized.shape
        prefix_shape = row_shape[:2]
        prior = _readonly_float64(
            self.prior_rows,
            label="prior_rows",
            shape=row_shape,
        )
        prior_mass = _readonly_float64(
            self.prior_mass_rows,
            label="prior_mass_rows",
            shape=prefix_shape,
        )
        if bool((prior_mass < 0.0).any()):
            raise ValueError("prior_mass_rows must be nonnegative")
        raw = _readonly_float64(
            self.raw_innovation_rows,
            label="raw_innovation_rows",
            shape=row_shape,
        )
        bounded = _readonly_float64(
            self.bounded_innovation_rows,
            label="bounded_innovation_rows",
            shape=row_shape,
        )
        if bool((np.abs(bounded) > 1.0).any()):
            raise ValueError("bounded_innovation_rows must lie in [-1, 1]")
        active = _readonly_bool(
            self.active_mask,
            label="active_mask",
            shape=prefix_shape,
        )
        scales = _readonly_float64(
            self.normalization_scales,
            label="normalization_scales",
            shape=(row_shape[2],),
        )
        if bool((scales <= 0.0).any()):
            raise ValueError("normalization_scales must be strictly positive")
        temperatures = _readonly_float64(
            self.temperatures,
            label="temperatures",
            shape=(row_shape[2],),
        )
        if bool((temperatures <= 0.0).any()):
            raise ValueError("temperatures must be strictly positive")
        inactive = ~active
        for label, rows in (
            ("normalized_modal_rows", normalized),
            ("prior_rows", prior),
            ("raw_innovation_rows", raw),
            ("bounded_innovation_rows", bounded),
        ):
            if bool((rows[inactive] != 0.0).any()):
                raise ValueError(f"{label} must be zero on padding rows")
        if bool((prior_mass[inactive] != 0.0).any()):
            raise ValueError("prior_mass_rows must be zero on padding rows")
        decay = resolve_ew_decay(decay=self.decay)
        if self.final_state.batch_size != row_shape[0]:
            raise ValueError("final_state batch does not match emitted rows")
        if self.final_state.channel_count != row_shape[2]:
            raise ValueError("final_state channels do not match emitted rows")
        if self.final_state.decay != decay:
            raise ValueError("final_state decay does not match trace decay")
        object.__setattr__(self, "normalized_modal_rows", normalized)
        object.__setattr__(self, "prior_rows", prior)
        object.__setattr__(self, "prior_mass_rows", prior_mass)
        object.__setattr__(self, "raw_innovation_rows", raw)
        object.__setattr__(self, "bounded_innovation_rows", bounded)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "normalization_scales", scales)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "decay", decay)


def temperature_softsign(
    raw_innovation_rows: ArrayLike,
    frozen_positive_temperatures: ArrayLike,
    *,
    active_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Apply ``raw / (temperature + abs(raw))`` per final-axis channel."""

    raw = _float64_array(
        raw_innovation_rows,
        label="raw_innovation_rows",
    )
    if raw.ndim == 0 or raw.shape[-1] == 0:
        raise ValueError(
            "raw_innovation_rows must have a nonempty channel axis"
        )
    temperatures = _float64_array(
        frozen_positive_temperatures,
        label="frozen_positive_temperatures",
        shape=(raw.shape[-1],),
    )
    if bool((temperatures <= 0.0).any()):
        raise ValueError(
            "frozen_positive_temperatures must be strictly positive"
        )
    prefix_shape = raw.shape[:-1]
    if active_mask is None:
        mask = np.ones(prefix_shape, dtype=np.bool_)
    else:
        mask = _readonly_bool(
            active_mask,
            label="active_mask",
            shape=prefix_shape,
        )
    bounded = np.zeros_like(raw, dtype=np.float64)
    bounded[mask] = raw[mask] / (
        temperatures + np.abs(raw[mask])
    )
    if not bool(np.isfinite(bounded).all()):
        raise ValueError("temperature softsign rows are not finite")
    return bounded


def temperature_softsign_bank(
    raw_innovation_rows: ArrayLike,
    frozen_positive_temperature_bank: ArrayLike,
    *,
    active_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Vectorize temperature softsign over a ``[variants, channels]`` bank.

    The return shape is ``[variants, *raw_prefix, channels]``.  This is useful
    for temperature sweeps that share one temporal recurrence.
    """

    raw = _float64_array(
        raw_innovation_rows,
        label="raw_innovation_rows",
    )
    if raw.ndim == 0 or raw.shape[-1] == 0:
        raise ValueError(
            "raw_innovation_rows must have a nonempty channel axis"
        )
    temperature_bank = _float64_array(
        frozen_positive_temperature_bank,
        label="frozen_positive_temperature_bank",
    )
    if (
        temperature_bank.ndim != 2
        or temperature_bank.shape[0] == 0
        or temperature_bank.shape[1] != raw.shape[-1]
    ):
        raise ValueError(
            "frozen_positive_temperature_bank must have shape "
            "[variants, channels]"
        )
    if bool((temperature_bank <= 0.0).any()):
        raise ValueError(
            "frozen_positive_temperature_bank must be strictly positive"
        )
    prefix_shape = raw.shape[:-1]
    if active_mask is None:
        mask = np.ones(prefix_shape, dtype=np.bool_)
    else:
        mask = _readonly_bool(
            active_mask,
            label="active_mask",
            shape=prefix_shape,
        )
    broadcast_temperatures = temperature_bank.reshape(
        (temperature_bank.shape[0],)
        + (1,) * len(prefix_shape)
        + (raw.shape[-1],)
    )
    raw_bank = raw[np.newaxis, ...]
    bounded = raw_bank / (
        broadcast_temperatures + np.abs(raw_bank)
    )
    bounded = np.where(mask[np.newaxis, ..., np.newaxis], bounded, 0.0)
    if not bool(np.isfinite(bounded).all()):
        raise ValueError("temperature softsign bank is not finite")
    return np.asarray(bounded, dtype=np.float64)


def causal_modal_innovation_v2(
    parent_modal_rows: ArrayLike,
    frozen_positive_scales: ArrayLike,
    frozen_positive_temperatures: ArrayLike,
    *,
    active_mask: ArrayLike | None = None,
    initial_state: CausalModalInnovationV2State | None = None,
    half_life: float | None = None,
    decay: float | None = None,
) -> CausalModalInnovationV2Trace:
    """Compute a parameterized, prior-before-update causal innovation trace.

    ``half_life`` and ``decay`` are mutually exclusive.  Padding rows emit
    exact zeros and leave the carry unchanged.  The state records its resolved
    decay, preventing an accidental timescale change between chunks.
    """

    modal = _float64_array(parent_modal_rows, label="parent_modal_rows")
    if modal.ndim != 3 or modal.shape[2] == 0:
        raise ValueError(
            "parent_modal_rows must have shape [batch, time, channels]"
        )
    batch_size, time_size, channel_count = modal.shape
    scales = _float64_array(
        frozen_positive_scales,
        label="frozen_positive_scales",
        shape=(channel_count,),
    )
    if bool((scales <= 0.0).any()):
        raise ValueError("frozen_positive_scales must be strictly positive")
    temperatures = _float64_array(
        frozen_positive_temperatures,
        label="frozen_positive_temperatures",
        shape=(channel_count,),
    )
    if bool((temperatures <= 0.0).any()):
        raise ValueError(
            "frozen_positive_temperatures must be strictly positive"
        )
    if active_mask is None:
        mask = np.ones((batch_size, time_size), dtype=np.bool_)
    else:
        mask = _readonly_bool(
            active_mask,
            label="active_mask",
            shape=(batch_size, time_size),
        )
    resolved_decay = resolve_ew_decay(
        half_life=half_life,
        decay=decay,
    )
    if initial_state is None:
        state = CausalModalInnovationV2State.zeros(
            batch_size,
            channel_count,
            decay=resolved_decay,
        )
    elif not isinstance(initial_state, CausalModalInnovationV2State):
        raise TypeError(
            "initial_state must be a CausalModalInnovationV2State or None"
        )
    else:
        state = initial_state
    if state.batch_size != batch_size:
        raise ValueError("initial_state batch does not match modal rows")
    if state.channel_count != channel_count:
        raise ValueError("initial_state channels do not match modal rows")
    if state.decay != resolved_decay:
        raise ValueError(
            "initial_state decay does not match the requested temporal decay"
        )

    weighted_sum = state.weighted_sum.copy()
    mass = state.mass.copy()
    normalized = np.zeros_like(modal, dtype=np.float64)
    prior = np.zeros_like(modal, dtype=np.float64)
    prior_mass = np.zeros((batch_size, time_size), dtype=np.float64)
    raw = np.zeros_like(modal, dtype=np.float64)

    for time_index in range(time_size):
        active = mask[:, time_index]
        if not bool(active.any()):
            continue
        current = modal[active, time_index, :] / scales
        active_mass = mass[active]
        current_prior = np.zeros_like(current, dtype=np.float64)
        has_prior = active_mass > 0.0
        current_prior[has_prior] = (
            weighted_sum[active][has_prior]
            / active_mass[has_prior, np.newaxis]
        )
        current_raw = current - current_prior

        normalized[active, time_index, :] = current
        prior[active, time_index, :] = current_prior
        prior_mass[active, time_index] = active_mass
        raw[active, time_index, :] = current_raw

        weighted_sum[active] = (
            resolved_decay * weighted_sum[active] + current
        )
        mass[active] = resolved_decay * active_mass + 1.0

    bounded = temperature_softsign(
        raw,
        temperatures,
        active_mask=mask,
    )
    return CausalModalInnovationV2Trace(
        normalized_modal_rows=normalized,
        prior_rows=prior,
        prior_mass_rows=prior_mass,
        raw_innovation_rows=raw,
        bounded_innovation_rows=bounded,
        active_mask=mask,
        normalization_scales=scales,
        temperatures=temperatures,
        decay=resolved_decay,
        final_state=CausalModalInnovationV2State(
            weighted_sum=weighted_sum,
            mass=mass,
            decay=resolved_decay,
        ),
    )


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total_weight = float(np.sum(ordered_weights))
    positions = (
        np.cumsum(ordered_weights) - 0.5 * ordered_weights
    ) / total_weight
    return float(
        np.interp(
            quantile,
            positions,
            ordered_values,
            left=ordered_values[0],
            right=ordered_values[-1],
        )
    )


@dataclass(frozen=True, slots=True, eq=False)
class RobustChannelTemperatureCalibration:
    """Frozen per-channel absolute-quantile softsign calibration."""

    temperatures: NDArray[np.float64]
    raw_absolute_quantiles: NDArray[np.float64]
    floor_applied: NDArray[np.bool_]
    active_count: int
    effective_weight: float
    absolute_quantile: float
    minimum_temperature: float

    def __post_init__(self) -> None:
        temperatures = _readonly_float64(
            self.temperatures,
            label="temperatures",
        )
        if temperatures.ndim != 1 or temperatures.shape[0] == 0:
            raise ValueError("temperatures must have shape [channels]")
        if bool((temperatures <= 0.0).any()):
            raise ValueError("temperatures must be strictly positive")
        raw_quantiles = _readonly_float64(
            self.raw_absolute_quantiles,
            label="raw_absolute_quantiles",
            shape=temperatures.shape,
        )
        if bool((raw_quantiles < 0.0).any()):
            raise ValueError("raw_absolute_quantiles must be nonnegative")
        floor_applied = _readonly_bool(
            self.floor_applied,
            label="floor_applied",
            shape=temperatures.shape,
        )
        if type(self.active_count) is not int or self.active_count <= 0:
            raise ValueError("active_count must be a positive integer")
        effective_weight = _real_scalar(
            self.effective_weight,
            label="effective_weight",
        )
        if effective_weight <= 0.0:
            raise ValueError("effective_weight must be strictly positive")
        quantile = _real_scalar(
            self.absolute_quantile,
            label="absolute_quantile",
        )
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("absolute_quantile must lie in [0, 1]")
        minimum = _real_scalar(
            self.minimum_temperature,
            label="minimum_temperature",
        )
        if minimum <= 0.0:
            raise ValueError("minimum_temperature must be strictly positive")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(
            self,
            "raw_absolute_quantiles",
            raw_quantiles,
        )
        object.__setattr__(self, "floor_applied", floor_applied)
        object.__setattr__(self, "effective_weight", effective_weight)
        object.__setattr__(self, "absolute_quantile", quantile)
        object.__setattr__(self, "minimum_temperature", minimum)

    def transform(
        self,
        raw_innovation_rows: ArrayLike,
        *,
        active_mask: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Apply this frozen calibration without reading evaluation rows."""

        return temperature_softsign(
            raw_innovation_rows,
            self.temperatures,
            active_mask=active_mask,
        )

    def temperature_bank(
        self,
        positive_multipliers: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return temperatures scaled by explicit positive variant factors."""

        multipliers = _float64_array(
            positive_multipliers,
            label="positive_multipliers",
        )
        if multipliers.ndim != 1 or multipliers.shape[0] == 0:
            raise ValueError(
                "positive_multipliers must have shape [variants]"
            )
        if bool((multipliers <= 0.0).any()):
            raise ValueError(
                "positive_multipliers must be strictly positive"
            )
        return np.asarray(
            multipliers[:, np.newaxis] * self.temperatures[np.newaxis, :],
            dtype=np.float64,
        )


def fit_robust_channel_temperatures(
    train_raw_innovation_rows: ArrayLike,
    *,
    active_mask: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    absolute_quantile: float = 0.5,
    minimum_temperature: float = 1.0e-12,
) -> RobustChannelTemperatureCalibration:
    """Fit robust per-channel temperatures on explicitly supplied train rows.

    The temperature is the selected weighted quantile of ``abs(raw)`` in each
    channel, floored by ``minimum_temperature``.  At the default median, a
    nondegenerate channel's median absolute bounded feature is approximately
    one half.  ``sample_weight`` has the row-prefix shape, allowing callers to
    choose token-, prompt-, or family-balanced calibration without embedding
    an experiment protocol in this utility.
    """

    raw = _float64_array(
        train_raw_innovation_rows,
        label="train_raw_innovation_rows",
    )
    if raw.ndim == 0 or raw.shape[-1] == 0:
        raise ValueError(
            "train_raw_innovation_rows must have a nonempty channel axis"
        )
    prefix_shape = raw.shape[:-1]
    if active_mask is None:
        mask = np.ones(prefix_shape, dtype=np.bool_)
    else:
        mask = _readonly_bool(
            active_mask,
            label="active_mask",
            shape=prefix_shape,
        )
    if sample_weight is None:
        weights = np.ones(prefix_shape, dtype=np.float64)
    else:
        weights = _float64_array(
            sample_weight,
            label="sample_weight",
            shape=prefix_shape,
        )
        if bool((weights < 0.0).any()):
            raise ValueError("sample_weight must be nonnegative")
    selected = mask & (weights > 0.0)
    active_count = int(np.count_nonzero(selected))
    if active_count == 0:
        raise ValueError(
            "calibration requires at least one positive-weight active row"
        )
    selected_weights = weights[selected]
    effective_weight = float(np.sum(selected_weights))
    quantile = _real_scalar(
        absolute_quantile,
        label="absolute_quantile",
    )
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("absolute_quantile must lie in [0, 1]")
    minimum = _real_scalar(
        minimum_temperature,
        label="minimum_temperature",
    )
    if minimum <= 0.0:
        raise ValueError("minimum_temperature must be strictly positive")

    selected_absolute = np.abs(raw[selected])
    raw_quantiles = np.asarray(
        [
            _weighted_quantile(
                selected_absolute[:, channel_index],
                selected_weights,
                quantile,
            )
            for channel_index in range(raw.shape[-1])
        ],
        dtype=np.float64,
    )
    floor_applied = raw_quantiles < minimum
    temperatures = np.maximum(raw_quantiles, minimum)
    return RobustChannelTemperatureCalibration(
        temperatures=temperatures,
        raw_absolute_quantiles=raw_quantiles,
        floor_applied=floor_applied,
        active_count=active_count,
        effective_weight=effective_weight,
        absolute_quantile=quantile,
        minimum_temperature=minimum,
    )


@dataclass(frozen=True, slots=True, eq=False)
class FixedGeneratorActivationTangentBank:
    """Static and variant-conditioned tangents without shared duplication.

    ``shared_activation_tangents`` has shape
    ``[batch, time, channels, hidden]``.  The conditioned bank has shape
    ``[variants, batch, time, channels, hidden]``.  Use :meth:`variant` only
    when a downstream API needs the conventional
    ``[batch, time, 2 * channels, hidden]`` materialization.
    """

    shared_activation_tangents: NDArray[np.float64]
    conditioned_activation_tangents: NDArray[np.float64]

    def __post_init__(self) -> None:
        shared = _readonly_float64(
            self.shared_activation_tangents,
            label="shared_activation_tangents",
        )
        if shared.ndim != 4 or shared.shape[2] == 0:
            raise ValueError(
                "shared_activation_tangents must have shape "
                "[batch, time, channels, hidden]"
            )
        conditioned = _readonly_float64(
            self.conditioned_activation_tangents,
            label="conditioned_activation_tangents",
        )
        expected_suffix = shared.shape
        if (
            conditioned.ndim != 5
            or conditioned.shape[0] == 0
            or conditioned.shape[1:] != expected_suffix
        ):
            raise ValueError(
                "conditioned_activation_tangents must have shape "
                "[variants, batch, time, channels, hidden]"
            )
        object.__setattr__(self, "shared_activation_tangents", shared)
        object.__setattr__(
            self,
            "conditioned_activation_tangents",
            conditioned,
        )

    @property
    def variant_count(self) -> int:
        return int(self.conditioned_activation_tangents.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.shared_activation_tangents.shape[2])

    def variant(self, variant_index: int) -> NDArray[np.float64]:
        """Materialize one shared-then-conditioned tangent bank."""

        if (
            type(variant_index) is not int
            or variant_index < 0
            or variant_index >= self.variant_count
        ):
            raise IndexError("variant_index is out of range")
        return np.concatenate(
            (
                self.shared_activation_tangents,
                self.conditioned_activation_tangents[variant_index],
            ),
            axis=2,
        )

    def materialize(self) -> NDArray[np.float64]:
        """Materialize all variants as ``[V, B, T, 2C, H]``."""

        shared_bank = np.broadcast_to(
            self.shared_activation_tangents[np.newaxis, ...],
            (
                self.variant_count,
                *self.shared_activation_tangents.shape,
            ),
        )
        return np.concatenate(
            (shared_bank, self.conditioned_activation_tangents),
            axis=3,
        )


def fixed_generator_innovation_activation_tangent_bank(
    source_coordinate_activation_tangents: ArrayLike,
    fixed_generator_basis: ArrayLike,
    bounded_innovation_bank: ArrayLike,
) -> FixedGeneratorActivationTangentBank:
    """Reduce a fixed basis once, then condition it for every feature variant.

    Args:
        source_coordinate_activation_tangents: Shape
            ``[batch, time, source_coordinates, hidden]``.
        fixed_generator_basis: Shape ``[source_coordinates, channels]``.
        bounded_innovation_bank: Shape
            ``[variants, batch, time, channels]``.

    The returned bank stores one copy of the shared fixed-U reduction rather
    than repeating it for every variant.  As in v1, innovation multiplication
    happens at the activation-position tangent before any loss-gradient
    contraction.
    """

    tangents = _float64_array(
        source_coordinate_activation_tangents,
        label="source_coordinate_activation_tangents",
    )
    if (
        tangents.ndim != 4
        or tangents.shape[2] == 0
        or tangents.shape[3] == 0
    ):
        raise ValueError(
            "source_coordinate_activation_tangents must have shape "
            "[batch, time, source_coordinates, hidden]"
        )
    basis = _float64_array(
        fixed_generator_basis,
        label="fixed_generator_basis",
    )
    if (
        basis.ndim != 2
        or basis.shape[0] != tangents.shape[2]
        or basis.shape[1] == 0
    ):
        raise ValueError(
            "fixed_generator_basis must have shape "
            "[source_coordinates, channels]"
        )
    innovation = _float64_array(
        bounded_innovation_bank,
        label="bounded_innovation_bank",
    )
    expected_suffix = (
        tangents.shape[0],
        tangents.shape[1],
        basis.shape[1],
    )
    if (
        innovation.ndim != 4
        or innovation.shape[0] == 0
        or innovation.shape[1:] != expected_suffix
    ):
        raise ValueError(
            "bounded_innovation_bank must have shape "
            "[variants, batch, time, channels]"
        )
    if bool((np.abs(innovation) > 1.0).any()):
        raise ValueError("bounded_innovation_bank must lie in [-1, 1]")

    shared = np.einsum(
        "btkh,kc->btch",
        tangents,
        basis,
        optimize=False,
    )
    conditioned = (
        innovation[..., np.newaxis] * shared[np.newaxis, ...]
    )
    if not bool(np.isfinite(conditioned).all()):
        raise ValueError("fixed generator tangent bank is not finite")
    return FixedGeneratorActivationTangentBank(
        shared_activation_tangents=shared,
        conditioned_activation_tangents=conditioned,
    )
