from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_joint_cross_block_transport_fit import (
    fit_greedy_joint_cross_block_transports,
)


def _problem() -> dict[str, object]:
    generator = torch.Generator().manual_seed(1729)
    rows = 96
    source_rank = 5
    target_rank = 3
    modes = 11
    source = torch.randn(rows, 1, source_rank, generator=generator)
    memberships = torch.ones(rows, 1)
    decoder, _ = torch.linalg.qr(
        torch.randn(modes, target_rank, generator=generator),
        mode="reduced",
    )
    decoder = decoder.T.contiguous()
    planted = torch.randn(source_rank, target_rank, generator=generator)
    residual = source[:, 0] @ planted @ decoder
    return {
        "source_packets": {"mutation.1": source},
        "target_memberships": memberships,
        "target_mode_decoders": decoder[None],
        "target_modal_residual": residual,
        "row_fisher_weights": torch.linspace(0.5, 2.0, rows),
        "maximum_incoming_sources_per_target_chart": 1,
        "relative_ridge": 1.0e-10,
    }


def test_rectangular_transport_recovers_a_planted_cross_event_map() -> None:
    fitted, metrics = fit_greedy_joint_cross_block_transports(**_problem())

    assert len(fitted) == 1
    edge = fitted[0]
    assert edge.source_event_id == "mutation.1"
    assert edge.transport_weight.shape == (5, 3)
    assert metrics["selected_edge_count"] == 1
    assert metrics["transport_partial_explained_fraction"] > 0.999999


def test_greedy_fit_selects_the_real_source_over_an_unrelated_packet() -> None:
    problem = _problem()
    real = problem["source_packets"]["mutation.1"]
    assert isinstance(real, torch.Tensor)
    problem["source_packets"] = {
        "mutation.1": real,
        "mutation.3": torch.randn_like(real),
    }

    fitted, _ = fit_greedy_joint_cross_block_transports(**problem)

    assert fitted[0].source_event_id == "mutation.1"


def test_zero_residual_selects_no_transport() -> None:
    problem = _problem()
    target = problem["target_modal_residual"]
    assert isinstance(target, torch.Tensor)
    problem["target_modal_residual"] = torch.zeros_like(target)

    fitted, metrics = fit_greedy_joint_cross_block_transports(**problem)

    assert fitted == ()
    assert metrics["selected_edge_count"] == 0
    assert metrics["transport_partial_explained_fraction"] == 0.0


@pytest.mark.parametrize("failure", ("weights", "simplex", "decoder"))
def test_fit_rejects_invalid_fisher_and_chart_contracts(failure: str) -> None:
    problem = _problem()
    if failure == "weights":
        weights = problem["row_fisher_weights"]
        assert isinstance(weights, torch.Tensor)
        weights[0] = 0
    elif failure == "simplex":
        memberships = problem["target_memberships"]
        assert isinstance(memberships, torch.Tensor)
        memberships[0, 0] = 0.5
    else:
        decoder = problem["target_mode_decoders"]
        assert isinstance(decoder, torch.Tensor)
        decoder[0, 0] *= 2

    with pytest.raises(ValueError):
        fit_greedy_joint_cross_block_transports(**problem)
