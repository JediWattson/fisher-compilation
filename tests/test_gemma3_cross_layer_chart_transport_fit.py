from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_cross_layer_chart_transport_fit import (
    factor_chart_edge_bank,
    fit_fisher_weighted_affine_base,
    fit_fisher_weighted_dense_chart_edge_bank,
    fit_greedy_adjacent_transport_edges,
)


def _orthonormal_decoders(
    *,
    charts: int,
    rank: int,
    modes: int,
) -> torch.Tensor:
    values = []
    for chart in range(charts):
        generator = torch.Generator().manual_seed(100 + chart)
        matrix = torch.randn(modes, rank, generator=generator, dtype=torch.float64)
        basis, _ = torch.linalg.qr(matrix, mode="reduced")
        values.append(basis.T)
    return torch.stack(values)


def test_weighted_affine_dual_ridge_recovers_known_map_as_float32() -> None:
    generator = torch.Generator().manual_seed(2)
    inputs = torch.randn(48, 3, generator=generator, dtype=torch.float64)
    input_to_modes = torch.tensor(
        [[1.0, -0.5], [0.25, 0.75], [-1.5, 0.1]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.3, -0.2], dtype=torch.float64)
    states = inputs @ input_to_modes + bias
    weights = torch.linspace(0.2, 2.0, inputs.shape[0], dtype=torch.float64)

    affine_weight, affine_bias, metrics = fit_fisher_weighted_affine_base(
        inputs=inputs,
        modal_states=states,
        row_fisher_weights=weights,
        relative_ridge=1.0e-10,
    )

    assert affine_weight.dtype == torch.float32
    assert affine_bias.dtype == torch.float32
    assert torch.allclose(
        affine_weight.double(),
        input_to_modes.T,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert torch.allclose(
        affine_bias.double(),
        bias,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert metrics["weighted_relative_rmse"] < 1.0e-7
    assert metrics["fit_algorithm"] == (
        "fisher_weighted_centered_dual_ridge_float64"
    )


def test_weighted_affine_fit_uses_supplied_row_weights() -> None:
    inputs = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    states = torch.tensor([[0.0], [10.0], [0.0]], dtype=torch.float64)
    uniform_weight, uniform_bias, _ = fit_fisher_weighted_affine_base(
        inputs=inputs,
        modal_states=states,
        row_fisher_weights=torch.ones(3, dtype=torch.float64),
        relative_ridge=1.0e-3,
    )
    center_weight, center_bias, _ = fit_fisher_weighted_affine_base(
        inputs=inputs,
        modal_states=states,
        row_fisher_weights=torch.tensor([1.0, 100.0, 1.0]),
        relative_ridge=1.0e-3,
    )
    assert torch.allclose(uniform_weight, center_weight)
    assert float(center_bias) > float(uniform_bias)


def test_joint_weighted_dense_chart_ridge_recovers_edge_bank() -> None:
    generator = torch.Generator().manual_seed(5)
    rows = 96
    charts = 2
    rank = 2
    modes = 4
    logits = torch.randn(rows, charts, generator=generator, dtype=torch.float64)
    memberships = logits.softmax(dim=1)
    coordinates = torch.randn(
        rows,
        charts,
        rank,
        generator=generator,
        dtype=torch.float64,
    )
    expected = torch.randn(
        charts,
        rank + 1,
        modes,
        generator=generator,
        dtype=torch.float64,
    )
    augmented = torch.cat(
        (torch.ones(rows, charts, 1, dtype=torch.float64), coordinates),
        dim=2,
    )
    residual = torch.einsum(
        "nc,ncs,csk->nk",
        memberships,
        augmented,
        expected,
    )

    observed, metrics = fit_fisher_weighted_dense_chart_edge_bank(
        memberships=memberships,
        coordinates=coordinates,
        base_modal_residual=residual,
        row_fisher_weights=torch.linspace(0.5, 1.5, rows),
        relative_ridge=1.0e-10,
    )

    assert observed.dtype == torch.float32
    assert observed.shape == expected.shape
    assert torch.allclose(
        observed.double(),
        expected,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    assert metrics["weighted_explained_fraction"] == pytest.approx(
        1.0,
        abs=1.0e-12,
    )
    assert sum(metrics["hard_chart_occupancy"]) == rows


def test_joint_chart_ridge_is_one_coupled_fit_not_chartwise_fit() -> None:
    memberships = torch.tensor(
        [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]],
        dtype=torch.float64,
    )
    coordinates = torch.tensor(
        [[[0.0], [1.0]], [[1.0], [0.0]], [[2.0], [-1.0]], [[3.0], [-2.0]]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [[[1.0], [0.5]], [[-0.25], [0.75]]],
        dtype=torch.float64,
    )
    augmented = torch.cat(
        (torch.ones(4, 2, 1, dtype=torch.float64), coordinates),
        dim=2,
    )
    residual = torch.einsum(
        "nc,ncs,csk->nk",
        memberships,
        augmented,
        expected,
    )
    observed, _ = fit_fisher_weighted_dense_chart_edge_bank(
        memberships=memberships,
        coordinates=coordinates,
        base_modal_residual=residual,
        row_fisher_weights=torch.ones(4),
        relative_ridge=1.0e-10,
    )
    prediction = torch.einsum(
        "nc,ncs,csk->nk",
        memberships.float(),
        augmented.float(),
        observed,
    )
    assert torch.allclose(prediction, residual.float(), atol=2.0e-5)


def test_weighted_affine_and_chart_fits_reject_bad_fisher_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_fisher_weighted_affine_base(
            inputs=torch.ones(3, 2),
            modal_states=torch.ones(3, 4),
            row_fisher_weights=torch.tensor([1.0, 0.0, 1.0]),
            relative_ridge=1.0e-3,
        )
    with pytest.raises(ValueError, match="sum to one"):
        fit_fisher_weighted_dense_chart_edge_bank(
            memberships=torch.full((3, 2), 0.25),
            coordinates=torch.ones(3, 2, 1),
            base_modal_residual=torch.ones(3, 4),
            row_fisher_weights=torch.ones(3),
            relative_ridge=1.0e-3,
        )


def test_factor_chart_edge_bank_exactly_recovers_low_rank_coefficients() -> None:
    generator = torch.Generator().manual_seed(3)
    local = torch.randn(3, 5, 2, generator=generator, dtype=torch.float64)
    decoder = _orthonormal_decoders(charts=3, rank=2, modes=7)
    dense = torch.einsum("csg,cgk->csk", local, decoder)

    observed_local, observed_decoder, metrics = factor_chart_edge_bank(
        dense,
        2,
    )
    reconstructed = torch.einsum(
        "csg,cgk->csk",
        observed_local,
        observed_decoder,
    )

    assert torch.allclose(reconstructed, dense, rtol=1.0e-10, atol=1.0e-10)
    assert torch.allclose(
        observed_decoder @ observed_decoder.transpose(-1, -2),
        torch.eye(2, dtype=torch.float64).expand(3, -1, -1),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert metrics["captured_energy_fraction"] == pytest.approx(1.0)
    for row in observed_decoder.reshape(-1, observed_decoder.shape[-1]):
        pivot = int(torch.argmax(row.abs()).item())
        assert float(row[pivot]) >= 0.0


def test_factor_chart_edge_bank_is_deterministic_and_reports_truncation() -> None:
    generator = torch.Generator().manual_seed(7)
    dense = torch.randn(2, 4, 6, generator=generator, dtype=torch.float32)
    first = factor_chart_edge_bank(dense, 2)
    second = factor_chart_edge_bank(dense, 2)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[0].dtype == torch.float32
    assert first[1].dtype == torch.float32
    assert 0.0 < first[2]["captured_energy_fraction"] < 1.0
    assert first[2]["relative_frobenius_error"] > 0.0


@pytest.mark.parametrize(
    ("value", "rank", "message"),
    (
        (torch.ones(2, 3), 1, "chart_edge_weight"),
        (torch.ones(1, 2, 3), 0, "generator_rank"),
        (torch.ones(1, 2, 3), 3, "generator_rank"),
        (torch.tensor([[[float("nan")]]]), 1, "chart_edge_weight"),
    ),
)
def test_factor_chart_edge_bank_rejects_invalid_inputs(
    value: torch.Tensor,
    rank: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factor_chart_edge_bank(value, rank)


def test_greedy_transport_recovers_one_known_edge() -> None:
    generator = torch.Generator().manual_seed(11)
    rows = 80
    rank = 2
    modes = 5
    source = torch.randn(rows, 2, rank, generator=generator, dtype=torch.float64)
    source[:, 1] = 0.0
    memberships = torch.zeros(rows, 2, dtype=torch.float64)
    memberships[:, 0] = 1.0
    decoders = _orthonormal_decoders(charts=2, rank=rank, modes=modes)
    expected_weight = torch.tensor(
        [[1.25, -0.5], [0.2, 0.75]],
        dtype=torch.float64,
    )
    residual = source[:, 0] @ expected_weight @ decoders[0]

    edges, metrics = fit_greedy_adjacent_transport_edges(
        source_emitted_latents=source,
        target_memberships=memberships,
        target_mode_decoders=decoders,
        target_modal_residual=residual,
        row_fisher_weights=torch.ones(rows, dtype=torch.float64),
        maximum_incoming_charts_per_target=1,
        relative_ridge=1.0e-10,
    )

    assert len(edges) == 1
    assert edges[0].coordinate == (0, 0)
    assert torch.allclose(
        edges[0].transport_weight,
        expected_weight,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert metrics["transport_partial_explained_fraction"] == pytest.approx(
        1.0,
        abs=1.0e-12,
    )
    assert metrics["incoming_edge_count_by_target"] == [1, 0]


def test_greedy_transport_selects_two_complementary_sources() -> None:
    generator = torch.Generator().manual_seed(29)
    rows = 120
    rank = 2
    modes = 6
    source = torch.randn(rows, 3, rank, generator=generator, dtype=torch.float64)
    source[:, 2] = 0.0
    memberships = torch.ones(rows, 1, dtype=torch.float64)
    decoders = _orthonormal_decoders(charts=1, rank=rank, modes=modes)
    left = torch.tensor([[1.0, 0.1], [-0.3, 0.8]], dtype=torch.float64)
    right = torch.tensor([[0.2, -0.7], [0.6, 0.4]], dtype=torch.float64)
    residual = (
        source[:, 0] @ left @ decoders[0]
        + source[:, 1] @ right @ decoders[0]
    )

    edges, metrics = fit_greedy_adjacent_transport_edges(
        source_emitted_latents=source,
        target_memberships=memberships,
        target_mode_decoders=decoders,
        target_modal_residual=residual,
        row_fisher_weights=torch.linspace(0.5, 1.5, rows),
        maximum_incoming_charts_per_target=2,
        relative_ridge=1.0e-9,
    )

    assert {edge.coordinate for edge in edges} == {(0, 0), (1, 0)}
    prediction = sum(
        (
            source[:, edge.source_chart_index]
            @ edge.transport_weight
            @ decoders[edge.target_chart_index]
            for edge in edges
        ),
        torch.zeros_like(residual),
    )
    assert torch.allclose(prediction, residual, rtol=1.0e-7, atol=1.0e-7)
    assert metrics["selected_edge_count"] == 2
    assert metrics["refinement_sweeps_completed"] >= 1


def test_zero_transport_residual_selects_no_edges() -> None:
    source = torch.ones(6, 2, 1, dtype=torch.float64)
    memberships = torch.full((6, 2), 0.5, dtype=torch.float64)
    decoders = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    edges, metrics = fit_greedy_adjacent_transport_edges(
        source_emitted_latents=source,
        target_memberships=memberships,
        target_mode_decoders=decoders,
        target_modal_residual=torch.zeros(6, 2, dtype=torch.float64),
        row_fisher_weights=torch.ones(6, dtype=torch.float64),
        maximum_incoming_charts_per_target=1,
        relative_ridge=1.0e-3,
    )
    assert edges == ()
    assert metrics["selected_edge_count"] == 0
    assert metrics["transport_partial_explained_fraction"] == 0.0


def test_transport_fit_rejects_nonorthonormal_decoders_or_bad_membership() -> None:
    source = torch.ones(5, 2, 1)
    residual = torch.ones(5, 2)
    weights = torch.ones(5)
    with pytest.raises(ValueError, match="orthonormal"):
        fit_greedy_adjacent_transport_edges(
            source_emitted_latents=source,
            target_memberships=torch.ones(5, 1),
            target_mode_decoders=torch.ones(1, 1, 2),
            target_modal_residual=residual,
            row_fisher_weights=weights,
            maximum_incoming_charts_per_target=1,
            relative_ridge=1.0e-3,
        )
    with pytest.raises(ValueError, match="sum to one"):
        fit_greedy_adjacent_transport_edges(
            source_emitted_latents=source,
            target_memberships=torch.full((5, 1), 0.5),
            target_mode_decoders=torch.tensor([[[1.0, 0.0]]]),
            target_modal_residual=residual,
            row_fisher_weights=weights,
            maximum_incoming_charts_per_target=1,
            relative_ridge=1.0e-3,
        )
