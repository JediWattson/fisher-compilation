from __future__ import annotations

import numpy as np
import pytest

from fisher_graph.gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
    causal_modal_innovation,
    fixed_generator_innovation_activation_tangents,
)


def _modal_rows() -> np.ndarray:
    return np.asarray(
        (
            (
                (2.0, -1.0),
                (4.0, 1.0),
                (100.0, 100.0),
                (3.0, 2.0),
                (-2.0, 4.0),
            ),
        ),
        dtype=np.float64,
    )


def test_causal_innovation_is_prefix_causal_padding_safe_and_chunk_exact() -> None:
    rows = _modal_rows()
    mask = np.asarray(((True, True, False, True, True),), dtype=np.bool_)
    scales = np.asarray((2.0, 1.0), dtype=np.float64)
    full = causal_modal_innovation(rows, scales, active_mask=mask)

    assert np.array_equal(
        full.normalized_modal_rows[0, 0],
        np.asarray((1.0, -1.0)),
    )
    assert np.array_equal(full.prior_rows[0, 0], np.zeros(2))
    assert np.array_equal(
        full.bounded_innovation_rows[0, 0],
        np.asarray((0.5, -0.5)),
    )
    assert np.array_equal(
        full.bounded_innovation_rows[0, 2],
        np.zeros(2),
    )

    changed_future = rows.copy()
    changed_future[0, 4] = (1.0e8, -1.0e8)
    changed = causal_modal_innovation(
        changed_future,
        scales,
        active_mask=mask,
    )
    assert np.array_equal(
        changed.bounded_innovation_rows[:, :4],
        full.bounded_innovation_rows[:, :4],
    )

    first = causal_modal_innovation(
        rows[:, :3],
        scales,
        active_mask=mask[:, :3],
    )
    second = causal_modal_innovation(
        rows[:, 3:],
        scales,
        active_mask=mask[:, 3:],
        initial_state=first.final_state,
    )
    assert np.array_equal(
        np.concatenate(
            (
                first.bounded_innovation_rows,
                second.bounded_innovation_rows,
            ),
            axis=1,
        ),
        full.bounded_innovation_rows,
    )
    assert np.array_equal(
        second.final_state.weighted_sum,
        full.final_state.weighted_sum,
    )
    assert np.array_equal(second.final_state.mass, full.final_state.mass)


def test_activation_tangent_reduction_multiplies_innovation_before_contract() -> None:
    tangents = np.zeros((1, 2, 6, 3), dtype=np.float64)
    for coordinate in range(6):
        tangents[:, :, coordinate, :] = coordinate + 1
    basis = np.zeros((6, 2), dtype=np.float64)
    basis[(0, 2, 4), 0] = (1.0, 2.0, -1.0)
    basis[(1, 3, 5), 1] = (0.5, -1.0, 2.0)
    innovation = np.asarray(
        (((0.25, -0.5), (-0.75, 0.125)),),
        dtype=np.float64,
    )

    result = fixed_generator_innovation_activation_tangents(
        tangents,
        basis,
        innovation,
    )
    shared_real = np.einsum("btkh,k->bth", tangents, basis[:, 0])
    shared_imag = np.einsum("btkh,k->bth", tangents, basis[:, 1])

    assert GENERATOR_INNOVATION_TANGENT_ORDER == (
        "generator_real_shared",
        "generator_imag_shared",
        "generator_real_innovation",
        "generator_imag_innovation",
    )
    np.testing.assert_allclose(result[:, :, 0], shared_real)
    np.testing.assert_allclose(result[:, :, 1], shared_imag)
    np.testing.assert_allclose(
        result[:, :, 2],
        shared_real * innovation[:, :, 0, None],
    )
    np.testing.assert_allclose(
        result[:, :, 3],
        shared_imag * innovation[:, :, 1, None],
    )


def test_innovation_rejects_nonpositive_scales_and_nonboolean_mask() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        causal_modal_innovation(_modal_rows(), (1.0, 0.0))
    with pytest.raises(TypeError, match="active_mask must be boolean"):
        causal_modal_innovation(
            _modal_rows(),
            (1.0, 1.0),
            active_mask=np.ones((1, 5), dtype=np.int64),
        )
