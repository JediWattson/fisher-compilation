from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.conditional_quadratic_edge import (
    TangentPreservingQuadraticEdge,
    TangentPreservingQuadraticSample,
    build_causal_lagged_modal_design,
    evaluate_tangent_preserving_quadratic_edge,
    fit_tangent_preserving_quadratic_edge,
)


FLOAT64 = torch.float64


def _base_kernel() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.5, -0.2], [0.1, 0.3]],
            [[-0.1, 0.4], [0.2, -0.3]],
        ],
        dtype=FLOAT64,
    )


def _true_factors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.1, 0.3, -0.4],
                [-0.3, 0.2, 0.25],
                [0.2, 0.1, -0.15],
            ],
            dtype=FLOAT64,
        ),
        torch.tensor(
            [
                [-0.1, 0.3, 0.2],
                [0.4, -0.2, 0.1],
                [0.2, 0.2, -0.3],
                [-0.25, 0.15, 0.4],
            ],
            dtype=FLOAT64,
        ),
        torch.tensor(
            [[0.7, -0.2], [-0.3, 0.5], [0.2, 0.4]],
            dtype=FLOAT64,
        ),
    )


def _sample(seed: int, *, gapped: bool = False) -> TangentPreservingQuadraticSample:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(8, 2, generator=generator, dtype=FLOAT64)
    positions = torch.tensor(
        [0, 1, 2, 4, 5, 6, 8, 9] if gapped else list(range(8)),
        dtype=torch.int64,
    )
    mask = torch.ones(8, dtype=torch.bool)
    design = build_causal_lagged_modal_design(
        source,
        logical_positions=positions,
        valid_mask=mask,
        lag_count=2,
    )
    kernel = _base_kernel()
    base = sum(
        design[:, lag * 2 : (lag + 1) * 2] @ kernel[lag]
        for lag in range(2)
    )
    A, C, B = _true_factors()
    target = base + ((design @ A) * (design @ C)) @ B
    return TangentPreservingQuadraticSample(
        source_modes=source,
        target_modes=target,
        logical_positions=positions,
        valid_mask=mask,
    )


def _fit(
    *,
    steps: int = 500,
    minibatch_rows: int | None = None,
) -> TangentPreservingQuadraticEdge:
    return fit_tangent_preserving_quadratic_edge(
        tuple(_sample(index, gapped=index % 2 == 0) for index in range(8)),
        base_kernel=_base_kernel(),
        hidden_width=6,
        steps=steps,
        learning_rate=2e-2,
        ridge=1e-8,
        seed=3,
        heldout_samples=tuple(
            _sample(100 + index, gapped=index % 2 == 0)
            for index in range(4)
        ),
        minibatch_rows=minibatch_rows,
    )


def test_lagged_design_uses_logical_positions_and_masks() -> None:
    source = torch.tensor(
        [[1.0], [2.0], [4.0], [8.0], [1000.0]],
        dtype=FLOAT64,
    )
    positions = torch.tensor([0, 1, 3, 4, 99], dtype=torch.int64)
    mask = torch.tensor([True, True, True, True, False])

    design = build_causal_lagged_modal_design(
        source,
        logical_positions=positions,
        valid_mask=mask,
        lag_count=3,
    )

    torch.testing.assert_close(
        design,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [4.0, 0.0, 2.0],
                [8.0, 4.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=FLOAT64,
        ),
    )
    batched = build_causal_lagged_modal_design(
        torch.stack((source, source * 2.0)),
        logical_positions=positions,
        valid_mask=mask,
        lag_count=3,
    )
    torch.testing.assert_close(batched[0], design)
    torch.testing.assert_close(batched[1], design * 2.0)


def test_correction_has_zero_value_and_zero_jacobian_at_origin() -> None:
    edge = _fit(steps=20)
    positions = torch.arange(7, dtype=torch.int64)
    mask = torch.ones(7, dtype=torch.bool)
    zero = torch.zeros(7, edge.source_rank, dtype=FLOAT64)
    direction = torch.randn_like(zero)

    primal, tangent = torch.autograd.functional.jvp(
        lambda value: edge.execute(
            value,
            logical_positions=positions,
            valid_mask=mask,
        ),
        (zero,),
        (direction,),
    )

    assert torch.count_nonzero(primal) == 0
    assert torch.count_nonzero(tangent) == 0


def test_deterministic_fit_recovers_finite_train_and_heldout_response() -> None:
    first = _fit()
    second = _fit()

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.source_rank == 2
    assert first.target_rank == 2
    assert first.lag_count == 2
    assert first.hidden_width == 6
    assert first.stored_scalar_count == 60
    assert first.macs_per_target_row == 60
    assert first.elementwise_multiplies_per_target_row == 6
    assert first.train_metrics.base_relative_error > 0.2
    assert first.train_metrics.corrected_relative_error < 1e-6
    assert first.heldout_metrics is not None
    assert first.heldout_metrics.base_relative_error > 0.2
    assert first.heldout_metrics.corrected_relative_error < 1e-6
    assert (
        first.heldout_metrics.corrected_cosine
        > first.heldout_metrics.base_cosine
    )

    sample = _sample(999, gapped=True)
    metrics = evaluate_tangent_preserving_quadratic_edge(
        first,
        (sample,),
        base_kernel=_base_kernel(),
    )
    assert metrics.corrected_relative_error < 1e-6
    assert metrics.corrected_relative_error < metrics.base_relative_error


def test_minibatch_fit_is_deterministic_and_records_its_protocol() -> None:
    first = _fit(steps=80, minibatch_rows=12)
    second = _fit(steps=80, minibatch_rows=12)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.minibatch_rows == 12
    torch.testing.assert_close(first.A, second.A)
    torch.testing.assert_close(first.C, second.C)
    torch.testing.assert_close(first.B, second.B)


def test_state_roundtrip_is_strict_and_detects_tensor_mutation() -> None:
    edge = _fit(steps=20)
    restored = TangentPreservingQuadraticEdge.from_state_dict(
        edge.state_dict()
    )

    assert restored.artifact_sha256 == edge.artifact_sha256
    torch.testing.assert_close(restored.A, edge.A)
    torch.testing.assert_close(restored.C, edge.C)
    torch.testing.assert_close(restored.B, edge.B)

    unknown = copy.deepcopy(edge.state_dict())
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="state fields"):
        TangentPreservingQuadraticEdge.from_state_dict(unknown)

    tampered = copy.deepcopy(edge.state_dict())
    tampered["A"][0, 0] += 1.0
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        TangentPreservingQuadraticEdge.from_state_dict(tampered)

    edge.A[0, 0] += 1.0
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        edge.validate_integrity()


def test_evaluation_rejects_a_different_base_kernel() -> None:
    edge = _fit(steps=20)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_tangent_preserving_quadratic_edge(
            edge,
            (_sample(501),),
            base_kernel=_base_kernel() + 1.0,
        )


def test_fit_rejects_overlapping_train_and_heldout_samples() -> None:
    sample = _sample(1)
    with pytest.raises(ValueError, match="must be disjoint"):
        fit_tangent_preserving_quadratic_edge(
            (sample,),
            base_kernel=_base_kernel(),
            hidden_width=2,
            steps=2,
            learning_rate=1e-2,
            ridge=0.0,
            seed=0,
            heldout_samples=(sample,),
        )


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        (
            torch.tensor([0.0, 1.0, 2.0]),
            torch.ones(3, dtype=torch.bool),
            "int32 or torch.int64",
        ),
        (
            torch.tensor([0, 2, 1]),
            torch.ones(3, dtype=torch.bool),
            "strictly increasing",
        ),
        (
            torch.tensor([0, -1, 2]),
            torch.ones(3, dtype=torch.bool),
            "nonnegative",
        ),
        (
            torch.tensor([0, 1, 2]),
            torch.zeros(3, dtype=torch.bool),
            "valid position",
        ),
    ],
)
def test_lagged_design_fails_closed_on_invalid_grids(
    positions: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_causal_lagged_modal_design(
            torch.ones(3, 1),
            logical_positions=positions,
            valid_mask=mask,
            lag_count=2,
        )
