from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_l3_l4_soft_polarity_token_vjp_compiler import (
    SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
    build_selected_teacher_grid,
    build_soft_polarity_post_cast_h4_secants,
    materialize_complete_h4_post_cast,
    soft_polarity_post_cast_h4_secant_stability,
    validate_selected_teacher_grid_replay,
)


def _sha(byte: str = "a") -> str:
    return byte * 64


def test_selected_teacher_rows_scatter_and_replay_without_serializing_tensors() -> None:
    rows = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16
    )
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    grid, receipt = build_selected_teacher_grid(
        rows, indices, batch_size=2, sequence_length=4
    )

    assert grid.shape == (2, 4, 3)
    assert grid.dtype == torch.bfloat16
    assert torch.equal(grid[0, 1], rows[0])
    assert torch.equal(grid[1, 2], rows[1])
    assert int((grid != 0).sum()) == 6
    assert receipt.selected_row_count == 2
    assert receipt.zero_filled_row_count == 6
    assert receipt.metadata()["raw_teacher_or_grid_tensors_serialized"] is False
    assert not any(isinstance(value, torch.Tensor) for value in receipt.metadata().values())
    validate_selected_teacher_grid_replay(grid, rows, indices, receipt)


def test_selected_teacher_grid_rejects_noncanonical_or_tampered_replay() -> None:
    rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    indices = torch.tensor([[0, 0], [0, 2]], dtype=torch.int64)
    grid, receipt = build_selected_teacher_grid(
        rows, indices, batch_size=1, sequence_length=3
    )
    with pytest.raises(ValueError, match="canonical"):
        build_selected_teacher_grid(
            rows,
            torch.tensor([[0, 2], [0, 0]], dtype=torch.int64),
            batch_size=1,
            sequence_length=3,
        )
    tampered = grid.clone()
    tampered[0, 1, 0] = 1.0
    with pytest.raises(RuntimeError, match="replay differs"):
        validate_selected_teacher_grid_replay(tampered, rows, indices, receipt)
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(receipt, selected_row_count=1, zero_filled_row_count=2)


def test_complete_h4_post_cast_matches_float64_add_then_live_cast() -> None:
    reference = torch.tensor(
        [[[1.0, -2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32
    )
    correction = torch.zeros_like(reference, dtype=torch.float64)
    correction[0, :2] = torch.tensor(
        [[2.0**-25, -(2.0**-24)], [0.25, -0.5]], dtype=torch.float64
    )
    support = torch.tensor([[True, True, False]])
    result = materialize_complete_h4_post_cast(reference, correction, support)
    expected = reference.clone()
    expected[support] = (
        reference[support].to(torch.float64) + correction[support]
    ).to(torch.float32)
    assert torch.equal(result, expected)
    assert torch.equal(result[~support], reference[~support])


def test_post_cast_secants_use_frozen_steps_and_stay_on_support() -> None:
    reference = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32
    )
    support = torch.tensor([[True, True, False]])
    zero = torch.zeros_like(reference, dtype=torch.float64)
    step = SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP
    bias_direction = torch.tensor(
        [[[1.0, -2.0], [0.5, 3.0], [0.0, 0.0]]], dtype=torch.float64
    )
    slope_direction = torch.tensor(
        [[[2.0, 1.0], [-1.0, 0.25], [0.0, 0.0]]], dtype=torch.float64
    )

    center, tangents, receipt = build_soft_polarity_post_cast_h4_secants(
        reference_h4=reference,
        center_correction=zero,
        bias_minus_correction=-step * bias_direction,
        bias_plus_correction=step * bias_direction,
        slope_minus_correction=-step * slope_direction,
        slope_plus_correction=step * slope_direction,
        support_mask=support,
        half_step=step,
        reference_provider_sha256=_sha(),
        bias_minus_provider_sha256=_sha("b"),
        bias_plus_provider_sha256=_sha("c"),
        slope_minus_provider_sha256=_sha("d"),
        slope_plus_provider_sha256=_sha("e"),
    )

    assert torch.equal(center, reference)
    assert tangents.shape == (2, 1, 3, 2)
    assert torch.allclose(tangents[0], bias_direction, atol=8.0e-6, rtol=0.0)
    assert torch.allclose(tangents[1], slope_direction, atol=8.0e-6, rtol=0.0)
    assert bool((tangents[:, ~support] == 0.0).all())
    assert receipt.half_step == step
    assert receipt.metadata()["not_claimed_as"] == (
        "analytic_Jacobian_at_abs_or_clamp_kink"
    )
    assert receipt.metadata()["raw_h4_correction_or_secant_tensors_serialized"] is False
    assert receipt.metadata()["perturbation_bindings"] == (
        (
            "bias_minus",
            _sha("b"),
            receipt.bias_minus_h4_sha256,
        ),
        (
            "bias_plus",
            _sha("c"),
            receipt.bias_plus_h4_sha256,
        ),
        (
            "slope_minus",
            _sha("d"),
            receipt.slope_minus_h4_sha256,
        ),
        (
            "slope_plus",
            _sha("e"),
            receipt.slope_plus_h4_sha256,
        ),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            receipt,
            bias_minus_provider_sha256=receipt.bias_plus_provider_sha256,
            bias_plus_provider_sha256=receipt.bias_minus_provider_sha256,
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            receipt,
            slope_minus_h4_sha256=receipt.slope_plus_h4_sha256,
            slope_plus_h4_sha256=receipt.slope_minus_h4_sha256,
        )
    object.__setattr__(receipt, "bias_minus_h4_sha256", "f" * 64)
    with pytest.raises(RuntimeError, match="receipt drifted"):
        receipt.validate_integrity()


def test_post_cast_secants_reject_zero_quantization_escape_and_unknown_step() -> None:
    reference = torch.zeros((1, 2, 2), dtype=torch.float32)
    support = torch.tensor([[True, False]])
    zero = torch.zeros_like(reference, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="quantized to zero"):
        build_soft_polarity_post_cast_h4_secants(
            reference_h4=reference,
            center_correction=zero,
            bias_minus_correction=zero,
            bias_plus_correction=zero,
            slope_minus_correction=zero,
            slope_plus_correction=zero,
            support_mask=support,
            half_step=SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
            reference_provider_sha256=_sha("b"),
            bias_minus_provider_sha256=_sha("c"),
            bias_plus_provider_sha256=_sha("d"),
            slope_minus_provider_sha256=_sha("e"),
            slope_plus_provider_sha256=_sha("f"),
        )
    escaped = zero.clone()
    escaped[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="escapes"):
        materialize_complete_h4_post_cast(reference, escaped, support)
    with pytest.raises(ValueError, match="outside the frozen pair"):
        build_soft_polarity_post_cast_h4_secants(
            reference_h4=reference,
            center_correction=zero,
            bias_minus_correction=zero,
            bias_plus_correction=zero,
            slope_minus_correction=zero,
            slope_plus_correction=zero,
            support_mask=support,
            half_step=0.01,
            reference_provider_sha256=_sha("c"),
            bias_minus_provider_sha256=_sha("d"),
            bias_plus_provider_sha256=_sha("e"),
            slope_minus_provider_sha256=_sha("f"),
            slope_plus_provider_sha256=_sha("1"),
        )


def test_post_cast_secant_stability_applies_frozen_cosine_and_norm_gates() -> None:
    primary = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[2.0, -1.0], [0.5, 3.0]]],
        ],
        dtype=torch.float64,
    )
    passing = soft_polarity_post_cast_h4_secant_stability(
        primary, primary * 1.1
    )
    assert passing["passed"] is True
    assert passing["cosine_by_parameter"] == pytest.approx((1.0, 1.0))
    assert passing["audit_to_primary_norm_ratio_by_parameter"] == pytest.approx(
        (1.1, 1.1)
    )

    rotated = primary.clone()
    rotated[1] *= -1.0
    failing = soft_polarity_post_cast_h4_secant_stability(primary, rotated)
    assert failing["passed"] is False
    assert failing["cosine_by_parameter"][1] == pytest.approx(-1.0)
