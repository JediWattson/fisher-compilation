from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.causal_edge_transport import (
    PooledCausalEdgeJVPFit,
    collect_causal_edge_jvp_batch,
    evaluate_pooled_causal_edge_jvp,
    fit_pooled_causal_edge_jvp,
    integrate_path_jvp,
)


def _logical_causal_function(
    kernel: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
):
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    position_to_index = {
        int(positions[index]): int(index) for index in valid_indices
    }

    def function(source: torch.Tensor) -> torch.Tensor:
        rows = []
        for target_index in range(source.shape[1]):
            zero = source[:, target_index, :] @ torch.zeros_like(kernel[0])
            if not bool(mask[target_index]):
                rows.append(zero)
                continue
            target_position = int(positions[target_index])
            value = zero
            for lag in range(kernel.shape[0]):
                source_index = position_to_index.get(target_position - lag)
                if source_index is not None:
                    value = value + source[:, source_index, :] @ kernel[lag]
            rows.append(value)
        return torch.stack(rows, dim=1)

    return function


def _collect(
    function,
    *,
    baseline: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    count: int,
    seed: int,
    domain: str,
    decoder: torch.Tensor | None = None,
    encoder: torch.Tensor | None = None,
):
    width = baseline.shape[2]
    return collect_causal_edge_jvp_batch(
        function,
        baseline_source=baseline,
        logical_positions=positions,
        valid_mask=mask,
        source_decoder=(
            torch.eye(width, dtype=baseline.dtype)
            if decoder is None
            else decoder
        ),
        target_encoder=(
            torch.eye(width, dtype=baseline.dtype)
            if encoder is None
            else encoder
        ),
        direction_count=count,
        direction_seed=seed,
        direction_domain=domain,
    )


def test_pooled_fit_recovers_exact_kernel_and_generalizes_to_new_directions() -> None:
    kernel = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 2.0]],
            [[-0.75, 0.3], [1.25, -1.0]],
            [[0.2, 0.4], [-0.6, 0.8]],
        ],
        dtype=torch.float64,
    )
    positions_a = torch.arange(8, dtype=torch.int64)
    mask_a = torch.ones(8, dtype=torch.bool)
    positions_b = torch.tensor(
        [0, 1, 3, 4, 5, 7, 8, 99],
        dtype=torch.int64,
    )
    mask_b = torch.tensor(
        [True, True, True, True, True, True, True, False]
    )
    baseline_a = torch.randn(1, 8, 2, dtype=torch.float64)
    baseline_b = torch.randn(1, 8, 2, dtype=torch.float64)
    fit_a = _collect(
        _logical_causal_function(kernel, positions_a, mask_a),
        baseline=baseline_a,
        positions=positions_a,
        mask=mask_a,
        count=12,
        seed=101,
        domain="fit/prompt-a",
    )
    fit_b = _collect(
        _logical_causal_function(kernel, positions_b, mask_b),
        baseline=baseline_b,
        positions=positions_b,
        mask=mask_b,
        count=12,
        seed=202,
        domain="fit/prompt-b",
    )

    fit = fit_pooled_causal_edge_jvp(
        (fit_a, fit_b),
        max_lag=2,
        ridge=0.0,
    )

    torch.testing.assert_close(fit.kernel, kernel, atol=1e-11, rtol=1e-11)
    assert fit.source_rank == 2
    assert fit.target_rank == 2
    assert fit.lags == (0, 1, 2)
    assert fit.fit_direction_count == 24
    assert fit.design_rank == 6
    assert fit.relative_output_residual < 1e-12

    heldout = _collect(
        _logical_causal_function(kernel, positions_a, mask_a),
        baseline=baseline_a,
        positions=positions_a,
        mask=mask_a,
        count=8,
        seed=303,
        domain="heldout/prompt-a",
    )
    metrics = evaluate_pooled_causal_edge_jvp(fit, (heldout,))

    assert not set(fit.fit_direction_hashes).intersection(
        metrics.direction_hashes
    )
    assert metrics.direction_count == 8
    assert metrics.relative_output_residual < 1e-12
    assert metrics.relative_residual_worst < 1e-12
    assert metrics.output_cosine > 1.0 - 1e-12
    assert metrics.metadata()["fixed_kernel_evaluation"] is True


def test_domain_separation_is_deterministic_and_overlap_is_rejected() -> None:
    baseline = torch.randn(1, 7, 3, dtype=torch.float64)
    positions = torch.arange(7)
    mask = torch.ones(7, dtype=torch.bool)
    arguments = {
        "baseline": baseline,
        "positions": positions,
        "mask": mask,
        "count": 6,
        "seed": 55,
    }
    first = _collect(
        lambda value: value,
        domain="fit",
        **arguments,
    )
    repeated = _collect(
        lambda value: value,
        domain="fit",
        **arguments,
    )
    heldout = _collect(
        lambda value: value,
        domain="heldout",
        **arguments,
    )

    assert first.artifact_sha256 == repeated.artifact_sha256
    assert first.direction_hashes == repeated.direction_hashes
    torch.testing.assert_close(first.source_modes, repeated.source_modes)
    assert not set(first.direction_hashes).intersection(
        heldout.direction_hashes
    )
    assert not torch.equal(first.source_modes, heldout.source_modes)

    fit = fit_pooled_causal_edge_jvp((first,), max_lag=0, ridge=0.0)
    with pytest.raises(ValueError, match="directions overlap"):
        evaluate_pooled_causal_edge_jvp(fit, (repeated,))
    metrics = evaluate_pooled_causal_edge_jvp(fit, (heldout,))
    assert metrics.relative_output_residual < 1e-12


def test_direction_identity_includes_the_linearization_point() -> None:
    positions = torch.arange(6)
    mask = torch.ones(6, dtype=torch.bool)
    first = _collect(
        lambda value: value,
        baseline=torch.zeros(1, 6, 2, dtype=torch.float64),
        positions=positions,
        mask=mask,
        count=4,
        seed=7,
        domain="shared-generator",
    )
    second = _collect(
        lambda value: value,
        baseline=torch.ones(1, 6, 2, dtype=torch.float64),
        positions=positions,
        mask=mask,
        count=4,
        seed=7,
        domain="shared-generator",
    )

    torch.testing.assert_close(first.source_modes, second.source_modes)
    assert not set(first.direction_hashes).intersection(
        second.direction_hashes
    )
    fit = fit_pooled_causal_edge_jvp((first,), max_lag=0, ridge=0.0)
    metrics = evaluate_pooled_causal_edge_jvp(fit, (second,))
    assert metrics.relative_output_residual < 1e-12


def test_fixed_kernel_heldout_metrics_expose_missing_lag() -> None:
    kernel = torch.tensor(
        [
            [[1.0]],
            [[0.0]],
            [[2.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.arange(10)
    mask = torch.ones(10, dtype=torch.bool)
    function = _logical_causal_function(kernel, positions, mask)
    fit_batch = _collect(
        function,
        baseline=torch.zeros(1, 10, 1, dtype=torch.float64),
        positions=positions,
        mask=mask,
        count=12,
        seed=11,
        domain="fit",
    )
    heldout_batch = _collect(
        function,
        baseline=torch.zeros(1, 10, 1, dtype=torch.float64),
        positions=positions,
        mask=mask,
        count=8,
        seed=12,
        domain="heldout",
    )

    fit = fit_pooled_causal_edge_jvp(
        (fit_batch,),
        max_lag=1,
        ridge=0.0,
    )
    metrics = evaluate_pooled_causal_edge_jvp(fit, (heldout_batch,))

    assert fit.relative_output_residual > 0.5
    assert metrics.relative_output_residual > 0.5
    assert metrics.relative_residual_p90 > 0.5


def test_fit_strictly_round_trips_and_detects_tensor_tampering() -> None:
    positions = torch.arange(6)
    mask = torch.ones(6, dtype=torch.bool)
    batch = _collect(
        lambda value: 1.5 * value,
        baseline=torch.randn(1, 6, 2, dtype=torch.float64),
        positions=positions,
        mask=mask,
        count=8,
        seed=77,
        domain="fit",
    )
    fit = fit_pooled_causal_edge_jvp(
        (batch,),
        max_lag=0,
        ridge=1e-8,
    )

    restored = PooledCausalEdgeJVPFit.from_state_dict(fit.state_dict())
    assert restored.artifact_sha256 == fit.artifact_sha256
    torch.testing.assert_close(restored.kernel, fit.kernel)

    unknown = copy.deepcopy(fit.state_dict())
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        PooledCausalEdgeJVPFit.from_state_dict(unknown)

    tampered_state = copy.deepcopy(fit.state_dict())
    tampered_state["kernel"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="kernel hash mismatch"):
        PooledCausalEdgeJVPFit.from_state_dict(tampered_state)

    fit.kernel[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="integrity check failed"):
        fit.validate_integrity()


def test_heldout_evaluation_rejects_modal_basis_drift() -> None:
    baseline = torch.randn(1, 6, 2, dtype=torch.float64)
    positions = torch.arange(6)
    mask = torch.ones(6, dtype=torch.bool)
    fit_batch = _collect(
        lambda value: value,
        baseline=baseline,
        positions=positions,
        mask=mask,
        count=6,
        seed=1,
        domain="fit",
    )
    fit = fit_pooled_causal_edge_jvp(
        (fit_batch,),
        max_lag=0,
        ridge=0.0,
    )
    drifted = _collect(
        lambda value: value,
        baseline=baseline,
        positions=positions,
        mask=mask,
        count=6,
        seed=2,
        domain="heldout",
        decoder=2.0 * torch.eye(2, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="source decoder"):
        evaluate_pooled_causal_edge_jvp(fit, (drifted,))


def test_path_integrated_jvp_closes_a_finite_nonlinear_displacement() -> None:
    baseline = torch.tensor(
        [[[0.2], [-0.4], [0.7], [1.1]]],
        dtype=torch.float64,
    )
    displacement = torch.tensor(
        [[[0.7], [0.3], [-0.2], [0.5]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([True, True, True, False])

    midpoint = integrate_path_jvp(
        lambda value: value.pow(5),
        baseline_source=baseline,
        source_displacement=displacement,
        target_encoder=torch.ones(1, 1, dtype=torch.float64),
        valid_mask=mask,
        quadrature_order=1,
    )
    order_three = integrate_path_jvp(
        lambda value: value.pow(5),
        baseline_source=baseline,
        source_displacement=displacement,
        target_encoder=torch.ones(1, 1, dtype=torch.float64),
        valid_mask=mask,
        quadrature_order=3,
    )
    expected_delta = (baseline + displacement).pow(5) - baseline.pow(5)

    torch.testing.assert_close(
        order_three.endpoint_target_delta,
        expected_delta,
        atol=1e-13,
        rtol=1e-13,
    )
    torch.testing.assert_close(
        order_three.integrated_target_delta,
        expected_delta,
        atol=1e-12,
        rtol=1e-12,
    )
    assert order_three.relative_integration_residual < 1e-12
    assert order_three.integrated_endpoint_cosine > 1.0 - 1e-12
    assert midpoint.relative_integration_residual > 1e-3
    assert order_three.jvp_evaluation_count == 3
    assert order_three.metadata()["oracle_diagnostic_only"] is True


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_all_supported_path_orders_are_exact_for_a_linear_map(
    order: int,
) -> None:
    matrix = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0]],
        dtype=torch.float64,
    )
    baseline = torch.randn(1, 5, 2, dtype=torch.float64)
    displacement = torch.randn(1, 5, 2, dtype=torch.float64)
    diagnostic = integrate_path_jvp(
        lambda value: value @ matrix,
        baseline_source=baseline,
        source_displacement=displacement,
        target_encoder=torch.eye(2, dtype=torch.float64),
        valid_mask=torch.ones(5, dtype=torch.bool),
        quadrature_order=order,
    )

    torch.testing.assert_close(
        diagnostic.integrated_target_delta,
        displacement @ matrix,
        atol=1e-12,
        rtol=1e-12,
    )
    assert diagnostic.relative_integration_residual < 1e-12


def test_path_integral_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        integrate_path_jvp(
            lambda value: value,
            baseline_source=torch.zeros(1, 2, 1, dtype=torch.float64),
            source_displacement=torch.ones(1, 2, 1, dtype=torch.float64),
            target_encoder=torch.ones(1, 1, dtype=torch.float64),
            valid_mask=torch.ones(2, dtype=torch.bool),
            quadrature_order=5,
        )
