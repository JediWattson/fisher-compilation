from __future__ import annotations

import numpy as np
import pytest

from fisher_graph.gemma3_l3_l4_iterative_generator_innovation import (
    causal_modal_innovation,
)
from fisher_graph.gemma3_l3_l4_iterative_generator_innovation_v2 import (
    CausalModalInnovationV2State,
    causal_modal_innovation_v2,
    ew_decay_from_half_life,
    fit_robust_channel_temperatures,
    fixed_generator_innovation_activation_tangent_bank,
    resolve_ew_decay,
    temperature_softsign,
    temperature_softsign_bank,
)


def _rows_and_mask() -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(
        (
            (
                (2.0, -1.0),
                (4.0, 1.0),
                (1.0e6, -1.0e6),
                (3.0, 2.0),
                (-2.0, 4.0),
                (1.0, -3.0),
            ),
            (
                (-1.0, 2.0),
                (7.0, 7.0),
                (2.0, 3.0),
                (0.0, -2.0),
                (5.0, 1.0),
                (8.0, 8.0),
            ),
        ),
        dtype=np.float64,
    )
    mask = np.asarray(
        (
            (True, True, False, True, True, True),
            (True, False, True, True, True, False),
        ),
        dtype=np.bool_,
    )
    return rows, mask


def test_half_life_and_direct_decay_are_explicit_and_equivalent() -> None:
    assert ew_decay_from_half_life(1.0) == 0.5
    assert ew_decay_from_half_life(np.inf) == 1.0
    assert resolve_ew_decay() == ew_decay_from_half_life(16.0)
    assert resolve_ew_decay(decay=0.25) == 0.25

    rows, mask = _rows_and_mask()
    by_half_life = causal_modal_innovation_v2(
        rows,
        (2.0, 1.0),
        (0.75, 2.0),
        active_mask=mask,
        half_life=2.0,
    )
    by_decay = causal_modal_innovation_v2(
        rows,
        (2.0, 1.0),
        (0.75, 2.0),
        active_mask=mask,
        decay=ew_decay_from_half_life(2.0),
    )
    assert np.array_equal(
        by_half_life.raw_innovation_rows,
        by_decay.raw_innovation_rows,
    )
    assert np.array_equal(
        by_half_life.bounded_innovation_rows,
        by_decay.bounded_innovation_rows,
    )

    with pytest.raises(ValueError, match="not both"):
        resolve_ew_decay(half_life=2.0, decay=0.5)
    with pytest.raises(ValueError, match="strictly positive"):
        ew_decay_from_half_life(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_ew_decay(decay=1.01)


def test_l16_unit_temperature_is_the_exact_v1_feature_control() -> None:
    rows, mask = _rows_and_mask()
    scales = np.asarray((2.0, 1.0), dtype=np.float64)
    v1 = causal_modal_innovation(rows, scales, active_mask=mask)
    v2 = causal_modal_innovation_v2(
        rows,
        scales,
        (1.0, 1.0),
        active_mask=mask,
        half_life=16.0,
    )

    for field in (
        "normalized_modal_rows",
        "prior_rows",
        "raw_innovation_rows",
        "bounded_innovation_rows",
        "active_mask",
    ):
        assert np.array_equal(getattr(v2, field), getattr(v1, field))
    assert np.array_equal(
        v2.final_state.weighted_sum,
        v1.final_state.weighted_sum,
    )
    assert np.array_equal(v2.final_state.mass, v1.final_state.mass)


def test_v2_is_prefix_causal_padding_safe_and_chunk_exact() -> None:
    rows, mask = _rows_and_mask()
    scales = np.asarray((2.0, 1.0), dtype=np.float64)
    temperatures = np.asarray((0.75, 2.0), dtype=np.float64)
    full = causal_modal_innovation_v2(
        rows,
        scales,
        temperatures,
        active_mask=mask,
        half_life=3.0,
    )

    assert np.array_equal(full.prior_rows[:, 0], np.zeros((2, 2)))
    assert np.array_equal(
        full.raw_innovation_rows[0, 0],
        np.asarray((1.0, -1.0)),
    )
    np.testing.assert_allclose(
        full.bounded_innovation_rows[0, 0],
        np.asarray(
            (
                1.0 / (0.75 + 1.0),
                -1.0 / (2.0 + 1.0),
            )
        ),
    )
    assert np.array_equal(
        full.bounded_innovation_rows[~mask],
        np.zeros((int(np.count_nonzero(~mask)), 2)),
    )
    assert np.array_equal(full.prior_mass_rows[~mask], np.zeros(3))

    changed_future = rows.copy()
    changed_future[:, 5] = (1.0e12, -1.0e12)
    changed = causal_modal_innovation_v2(
        changed_future,
        scales,
        temperatures,
        active_mask=mask,
        half_life=3.0,
    )
    assert np.array_equal(
        changed.bounded_innovation_rows[:, :5],
        full.bounded_innovation_rows[:, :5],
    )

    first = causal_modal_innovation_v2(
        rows[:, :3],
        scales,
        temperatures,
        active_mask=mask[:, :3],
        half_life=3.0,
    )
    second = causal_modal_innovation_v2(
        rows[:, 3:],
        scales,
        temperatures,
        active_mask=mask[:, 3:],
        initial_state=first.final_state,
        half_life=3.0,
    )
    for field in (
        "normalized_modal_rows",
        "prior_rows",
        "prior_mass_rows",
        "raw_innovation_rows",
        "bounded_innovation_rows",
        "active_mask",
    ):
        chunked = np.concatenate(
            (getattr(first, field), getattr(second, field)),
            axis=1,
        )
        assert np.array_equal(chunked, getattr(full, field))
    assert np.array_equal(
        second.final_state.weighted_sum,
        full.final_state.weighted_sum,
    )
    assert np.array_equal(second.final_state.mass, full.final_state.mass)

    wrong_state = CausalModalInnovationV2State.zeros(
        2,
        2,
        decay=0.5,
    )
    with pytest.raises(ValueError, match="temporal decay"):
        causal_modal_innovation_v2(
            rows,
            scales,
            temperatures,
            active_mask=mask,
            initial_state=wrong_state,
            decay=0.25,
        )


def test_robust_train_calibration_sets_median_half_response_and_floors() -> None:
    raw = np.asarray(
        (
            (
                (1.0, 2.0, 0.0),
                (-2.0, -4.0, 0.0),
                (100.0, 200.0, 0.0),
                (1.0e9, 1.0e9, 1.0e9),
            ),
        ),
        dtype=np.float64,
    )
    mask = np.asarray(((True, True, True, False),), dtype=np.bool_)
    calibration = fit_robust_channel_temperatures(
        raw,
        active_mask=mask,
        minimum_temperature=0.25,
    )

    np.testing.assert_allclose(
        calibration.raw_absolute_quantiles,
        np.asarray((2.0, 4.0, 0.0)),
    )
    np.testing.assert_allclose(
        calibration.temperatures,
        np.asarray((2.0, 4.0, 0.25)),
    )
    assert np.array_equal(
        calibration.floor_applied,
        np.asarray((False, False, True)),
    )
    assert calibration.active_count == 3
    assert calibration.effective_weight == 3.0

    bounded = calibration.transform(raw, active_mask=mask)
    assert np.median(np.abs(bounded[mask, 0])) == 0.5
    assert np.median(np.abs(bounded[mask, 1])) == 0.5
    assert np.array_equal(bounded[~mask], np.zeros((1, 3)))
    np.testing.assert_allclose(
        calibration.temperature_bank((0.5, 1.0, 2.0)),
        np.asarray(
            (
                (1.0, 2.0, 0.125),
                (2.0, 4.0, 0.25),
                (4.0, 8.0, 0.5),
            )
        ),
    )


def test_weighted_calibration_and_softsign_bank_are_vectorized() -> None:
    raw = np.asarray(
        (
            ((1.0, -2.0), (3.0, -4.0), (9.0, 10.0)),
        ),
        dtype=np.float64,
    )
    mask = np.asarray(((True, True, False),), dtype=np.bool_)
    weights = np.asarray(((1.0, 3.0, 100.0),), dtype=np.float64)
    calibration = fit_robust_channel_temperatures(
        raw,
        active_mask=mask,
        sample_weight=weights,
        absolute_quantile=0.5,
    )
    assert calibration.active_count == 2
    assert calibration.effective_weight == 4.0
    assert np.all(calibration.temperatures > 0.0)

    temperature_bank = calibration.temperature_bank((0.5, 1.0, 2.0))
    bank = temperature_softsign_bank(
        raw,
        temperature_bank,
        active_mask=mask,
    )
    assert bank.shape == (3, 1, 3, 2)
    for variant_index in range(3):
        expected = temperature_softsign(
            raw,
            temperature_bank[variant_index],
            active_mask=mask,
        )
        np.testing.assert_allclose(bank[variant_index], expected)
    assert np.array_equal(bank[:, ~mask], np.zeros((3, 1, 2)))


def test_fixed_u_tangent_bank_reduces_once_and_materializes_variants() -> None:
    tangents = np.arange(1, 1 + 1 * 2 * 3 * 2, dtype=np.float64).reshape(
        (1, 2, 3, 2)
    )
    basis = np.asarray(
        (
            (1.0, 0.0),
            (0.5, -1.0),
            (-0.25, 2.0),
        ),
        dtype=np.float64,
    )
    innovation = np.asarray(
        (
            (((0.25, -0.5), (0.75, 0.125)),),
            (((-0.5, 0.25), (0.0, -0.75)),),
            (((0.1, 0.2), (0.3, 0.4)),),
        ),
        dtype=np.float64,
    )
    bank = fixed_generator_innovation_activation_tangent_bank(
        tangents,
        basis,
        innovation,
    )
    expected_shared = np.einsum(
        "btkh,kc->btch",
        tangents,
        basis,
        optimize=False,
    )
    expected_conditioned = (
        innovation[..., np.newaxis] * expected_shared[np.newaxis, ...]
    )

    assert bank.variant_count == 3
    assert bank.channel_count == 2
    assert bank.shared_activation_tangents.shape == (1, 2, 2, 2)
    assert bank.conditioned_activation_tangents.shape == (3, 1, 2, 2, 2)
    np.testing.assert_allclose(
        bank.shared_activation_tangents,
        expected_shared,
    )
    np.testing.assert_allclose(
        bank.conditioned_activation_tangents,
        expected_conditioned,
    )
    np.testing.assert_allclose(
        bank.variant(1),
        np.concatenate(
            (expected_shared, expected_conditioned[1]),
            axis=2,
        ),
    )
    materialized = bank.materialize()
    assert materialized.shape == (3, 1, 2, 4, 2)
    for variant_index in range(3):
        np.testing.assert_allclose(
            materialized[variant_index],
            bank.variant(variant_index),
        )

    with pytest.raises(ValueError, match="bounded_innovation_bank"):
        fixed_generator_innovation_activation_tangent_bank(
            tangents,
            basis,
            innovation[:, :, :, :1],
        )
