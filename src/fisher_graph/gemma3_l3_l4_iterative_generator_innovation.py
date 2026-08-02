"""Causal modal innovation features for the fixed Gemma generator basis.

The innovation controller is deliberately smaller than the modal route it may
eventually control.  For each active token it:

1. divides the frozen parent's top-two modal rows by two frozen positive
   scales;
2. compares the normalized row with an exponentially weighted summary of
   *earlier* active rows;
3. bounds that signed innovation with componentwise softsign; and
4. updates the exponentially weighted summary only after emitting the feature.

Padding rows emit zeros and do not change the carry.  The explicit carry makes
whole-sequence and chunked execution use the same recurrence.

The activation-tangent helper in this module performs the other important
ordering constraint.  It first reduces a six-coordinate activation tangent
bank through the fixed ``6 x 2`` generator basis, then multiplies the two
reduced activation tangents by their token-local innovation features.  A
caller must contract the resulting four activation tangents with token-loss
gradients afterwards; multiplying an already aggregated loss tangent would be
a different approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "GENERATOR_INNOVATION_EW_DECAY",
    "GENERATOR_INNOVATION_EW_HALF_LIFE",
    "GENERATOR_INNOVATION_STATE_FLOATS_PER_SEQUENCE",
    "GENERATOR_INNOVATION_TANGENT_ORDER",
    "CausalModalInnovationState",
    "CausalModalInnovationTrace",
    "causal_modal_innovation",
    "fixed_generator_innovation_activation_tangents",
]


GENERATOR_INNOVATION_EW_HALF_LIFE = 16
GENERATOR_INNOVATION_EW_DECAY = 2.0 ** (
    -1.0 / GENERATOR_INNOVATION_EW_HALF_LIFE
)
GENERATOR_INNOVATION_STATE_FLOATS_PER_SEQUENCE = 3
GENERATOR_INNOVATION_TANGENT_ORDER = (
    "generator_real_shared",
    "generator_imag_shared",
    "generator_real_innovation",
    "generator_imag_innovation",
)

_CHANNEL_COUNT = 2
_SOURCE_COORDINATE_COUNT = 6


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


@dataclass(frozen=True, slots=True, eq=False)
class CausalModalInnovationState:
    """Chunk-resumable exponentially weighted state for a batch.

    ``weighted_sum`` has shape ``[batch, 2]`` and ``mass`` has shape
    ``[batch]``.  Both arrays are copied into immutable float64 storage so a
    caller cannot silently mutate a carry after handing it to another chunk.
    """

    weighted_sum: NDArray[np.float64]
    mass: NDArray[np.float64]

    def __post_init__(self) -> None:
        weighted_sum = _readonly_float64(
            self.weighted_sum,
            label="weighted_sum",
        )
        if weighted_sum.ndim != 2 or weighted_sum.shape[1] != _CHANNEL_COUNT:
            raise ValueError("weighted_sum must have shape [batch, 2]")
        mass = _readonly_float64(
            self.mass,
            label="mass",
            shape=(weighted_sum.shape[0],),
        )
        if bool((mass < 0.0).any()):
            raise ValueError("mass must be nonnegative")
        zero_mass = mass == 0.0
        if bool((weighted_sum[zero_mass] != 0.0).any()):
            raise ValueError("zero-mass state must have a zero weighted_sum")
        object.__setattr__(self, "weighted_sum", weighted_sum)
        object.__setattr__(self, "mass", mass)

    @property
    def batch_size(self) -> int:
        return int(self.mass.shape[0])

    @classmethod
    def zeros(cls, batch_size: int) -> "CausalModalInnovationState":
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        return cls(
            weighted_sum=np.zeros(
                (batch_size, _CHANNEL_COUNT),
                dtype=np.float64,
            ),
            mass=np.zeros((batch_size,), dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CausalModalInnovationTrace:
    """Emitted causal feature rows and the carry after the final row."""

    normalized_modal_rows: NDArray[np.float64]
    prior_rows: NDArray[np.float64]
    raw_innovation_rows: NDArray[np.float64]
    bounded_innovation_rows: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    final_state: CausalModalInnovationState

    def __post_init__(self) -> None:
        normalized = _readonly_float64(
            self.normalized_modal_rows,
            label="normalized_modal_rows",
        )
        if normalized.ndim != 3 or normalized.shape[2] != _CHANNEL_COUNT:
            raise ValueError(
                "normalized_modal_rows must have shape [batch, time, 2]"
            )
        row_shape = normalized.shape
        prior = _readonly_float64(
            self.prior_rows,
            label="prior_rows",
            shape=row_shape,
        )
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
            shape=row_shape[:2],
        )
        inactive = ~active
        for label, rows in (
            ("normalized_modal_rows", normalized),
            ("prior_rows", prior),
            ("raw_innovation_rows", raw),
            ("bounded_innovation_rows", bounded),
        ):
            if bool((rows[inactive] != 0.0).any()):
                raise ValueError(f"{label} must be zero on padding rows")
        if self.final_state.batch_size != row_shape[0]:
            raise ValueError("final_state batch does not match emitted rows")
        object.__setattr__(self, "normalized_modal_rows", normalized)
        object.__setattr__(self, "prior_rows", prior)
        object.__setattr__(self, "raw_innovation_rows", raw)
        object.__setattr__(self, "bounded_innovation_rows", bounded)
        object.__setattr__(self, "active_mask", active)


def causal_modal_innovation(
    parent_top2_modal_rows: ArrayLike,
    frozen_positive_scales: ArrayLike,
    *,
    active_mask: ArrayLike | None = None,
    initial_state: CausalModalInnovationState | None = None,
) -> CausalModalInnovationTrace:
    """Compute the fixed two-channel prior-before-update innovation feature.

    Args:
        parent_top2_modal_rows: Frozen-parent modal values with shape
            ``[batch, time, 2]``.
        frozen_positive_scales: Two positive scales fixed before the evaluated
            sequence is opened.  They are never estimated from these rows.
        active_mask: Optional boolean ``[batch, time]`` mask.  False rows emit
            zeros and leave the carry exactly unchanged.
        initial_state: Optional carry produced by an earlier chunk with the
            same batch size.

    Returns:
        The normalized rows, earlier-row priors, raw and bounded innovations,
        and the carry to pass to the next chunk.

    The recurrence for an active normalized row ``x`` is::

        prior = weighted_sum / mass if mass > 0 else 0
        innovation = x - prior
        h = innovation / (1 + abs(innovation))
        weighted_sum = decay * weighted_sum + x
        mass = decay * mass + 1

    The prior therefore contains only earlier active positions.  The current
    position enters the state after its feature has been emitted.
    """

    modal = _float64_array(
        parent_top2_modal_rows,
        label="parent_top2_modal_rows",
    )
    if modal.ndim != 3 or modal.shape[2] != _CHANNEL_COUNT:
        raise ValueError(
            "parent_top2_modal_rows must have shape [batch, time, 2]"
        )
    batch_size, time_size, _ = modal.shape
    scales = _float64_array(
        frozen_positive_scales,
        label="frozen_positive_scales",
        shape=(_CHANNEL_COUNT,),
    )
    if bool((scales <= 0.0).any()):
        raise ValueError("frozen_positive_scales must be strictly positive")

    if active_mask is None:
        mask = np.ones((batch_size, time_size), dtype=np.bool_)
    else:
        mask_value = np.asarray(active_mask)
        if mask_value.dtype != np.bool_:
            raise TypeError("active_mask must be boolean")
        if mask_value.shape != (batch_size, time_size):
            raise ValueError(
                "active_mask must have shape "
                f"{(batch_size, time_size)}, got {mask_value.shape}"
            )
        mask = mask_value.copy()

    if initial_state is None:
        state = CausalModalInnovationState.zeros(batch_size)
    elif not isinstance(initial_state, CausalModalInnovationState):
        raise TypeError(
            "initial_state must be a CausalModalInnovationState or None"
        )
    else:
        state = initial_state
    if state.batch_size != batch_size:
        raise ValueError(
            "initial_state batch does not match parent_top2_modal_rows"
        )

    weighted_sum = state.weighted_sum.copy()
    mass = state.mass.copy()
    normalized = np.zeros_like(modal, dtype=np.float64)
    prior = np.zeros_like(modal, dtype=np.float64)
    raw = np.zeros_like(modal, dtype=np.float64)
    bounded = np.zeros_like(modal, dtype=np.float64)

    for time_index in range(time_size):
        for batch_index in range(batch_size):
            if not bool(mask[batch_index, time_index]):
                continue
            current = modal[batch_index, time_index] / scales
            if mass[batch_index] == 0.0:
                current_prior = np.zeros((_CHANNEL_COUNT,), dtype=np.float64)
            else:
                current_prior = (
                    weighted_sum[batch_index] / mass[batch_index]
                )
            current_raw = current - current_prior
            current_bounded = current_raw / (1.0 + np.abs(current_raw))

            normalized[batch_index, time_index] = current
            prior[batch_index, time_index] = current_prior
            raw[batch_index, time_index] = current_raw
            bounded[batch_index, time_index] = current_bounded

            weighted_sum[batch_index] = (
                GENERATOR_INNOVATION_EW_DECAY
                * weighted_sum[batch_index]
                + current
            )
            mass[batch_index] = (
                GENERATOR_INNOVATION_EW_DECAY * mass[batch_index] + 1.0
            )

    return CausalModalInnovationTrace(
        normalized_modal_rows=normalized,
        prior_rows=prior,
        raw_innovation_rows=raw,
        bounded_innovation_rows=bounded,
        active_mask=mask,
        final_state=CausalModalInnovationState(
            weighted_sum=weighted_sum,
            mass=mass,
        ),
    )


def fixed_generator_innovation_activation_tangents(
    six_coordinate_activation_tangents: ArrayLike,
    fixed_generator_basis: ArrayLike,
    bounded_innovation_rows: ArrayLike,
) -> NDArray[np.float64]:
    """Reduce six activation tangents into four fixed-generator tangents.

    Shapes are ``[batch, time, 6, hidden]`` for the input tangent bank,
    ``[6, 2]`` for the fixed generator basis, and ``[batch, time, 2]`` for
    the bounded causal innovation rows.  The returned shape is
    ``[batch, time, 4, hidden]`` in
    :data:`GENERATOR_INNOVATION_TANGENT_ORDER`.

    Innovation multiplies the generator-reduced *activation-position*
    tangents here.  Token-loss gradient contraction must happen after this
    function returns.
    """

    tangents = _float64_array(
        six_coordinate_activation_tangents,
        label="six_coordinate_activation_tangents",
    )
    if tangents.ndim != 4 or tangents.shape[2] != _SOURCE_COORDINATE_COUNT:
        raise ValueError(
            "six_coordinate_activation_tangents must have shape "
            "[batch, time, 6, hidden]"
        )
    basis = _float64_array(
        fixed_generator_basis,
        label="fixed_generator_basis",
        shape=(_SOURCE_COORDINATE_COUNT, _CHANNEL_COUNT),
    )
    innovation = _float64_array(
        bounded_innovation_rows,
        label="bounded_innovation_rows",
        shape=(tangents.shape[0], tangents.shape[1], _CHANNEL_COUNT),
    )
    if bool((np.abs(innovation) > 1.0).any()):
        raise ValueError("bounded_innovation_rows must lie in [-1, 1]")

    shared = np.einsum(
        "btkh,kc->btch",
        tangents,
        basis,
        optimize=False,
    )
    conditioned = shared * innovation[..., :, np.newaxis]
    result = np.stack(
        (
            shared[:, :, 0, :],
            shared[:, :, 1, :],
            conditioned[:, :, 0, :],
            conditioned[:, :, 1, :],
        ),
        axis=2,
    )
    if not bool(np.isfinite(result).all()):
        raise ValueError("fixed generator activation tangents are not finite")
    return np.asarray(result, dtype=np.float64)
