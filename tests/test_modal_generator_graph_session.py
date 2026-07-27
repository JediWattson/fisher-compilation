from __future__ import annotations

import pytest
import torch

from fisher_graph.modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
)
from fisher_graph.modal_generator_graph_session import (
    ModalGeneratorGraphSession,
)


def _weights(seed: int, input_width: int, latent: int, output: int):
    generator = torch.Generator().manual_seed(seed)
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=f"{seed:064x}",
        source_model_sha256="a" * 64,
        parameter_cluster_plan_sha256="b" * 64,
        input_factor=torch.randn(
            input_width,
            latent,
            generator=generator,
            dtype=torch.float64,
        ),
        output_factor=torch.randn(
            latent,
            output,
            generator=generator,
            dtype=torch.float64,
        ),
        latent_bias=torch.randn(
            latent,
            generator=generator,
            dtype=torch.float64,
        ),
        output_bias=torch.randn(
            output,
            generator=generator,
            dtype=torch.float64,
        ),
    )


def _factorized_weights(
    seed: int,
    input_width: int,
    private_width: int,
    latent_width: int,
    output_width: int,
) -> LinearModalGeneratorNodeWeights:
    generator = torch.Generator().manual_seed(seed)
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=f"{seed:064x}",
        source_model_sha256="a" * 64,
        parameter_cluster_plan_sha256="b" * 64,
        input_factor=torch.randn(
            input_width,
            private_width,
            generator=generator,
            dtype=torch.float64,
        ),
        state_factor=torch.randn(
            private_width,
            latent_width,
            generator=generator,
            dtype=torch.float64,
        ),
        output_factor=torch.randn(
            latent_width,
            output_width,
            generator=generator,
            dtype=torch.float64,
        ),
        latent_bias=torch.randn(
            latent_width,
            generator=generator,
            dtype=torch.float64,
        ),
        output_bias=None,
    )


def _plan() -> ModalGeneratorGraphPlan:
    nodes = (
        ModalGeneratorNode(
            name="root",
            causal_order=0,
            input_boundary="layer.0.input",
            output_boundary="layer.0.output",
            weights=_weights(1, 4, 2, 4),
        ),
        ModalGeneratorNode(
            name="branch-a",
            causal_order=1,
            input_boundary="layer.1.input",
            output_boundary="layer.1.output",
            weights=_weights(2, 4, 3, 4),
        ),
        ModalGeneratorNode(
            name="branch-b",
            causal_order=2,
            input_boundary="layer.1.input",
            output_boundary="layer.1.output",
            weights=_weights(3, 4, 2, 4),
        ),
        ModalGeneratorNode(
            name="fanin",
            causal_order=3,
            input_boundary="layer.2.input",
            output_boundary="layer.2.output",
            weights=_weights(4, 4, 2, 4),
        ),
    )
    edges = (
        ModalGeneratorInteraction(
            source_node="branch-a",
            target_node="fanin",
            message_matrix=torch.tensor(
                ((0.2, -0.1), (0.4, 0.3), (-0.2, 0.5)),
                dtype=torch.float64,
            ),
            message_bias=torch.zeros(2, dtype=torch.float64),
        ),
        ModalGeneratorInteraction(
            source_node="branch-b",
            target_node="fanin",
            message_matrix=torch.tensor(
                ((0.1, 0.2), (-0.3, 0.4)),
                dtype=torch.float64,
            ),
            message_bias=torch.tensor((0.05, -0.02), dtype=torch.float64),
        ),
        ModalGeneratorInteraction(
            source_node="root",
            target_node="branch-a",
            message_matrix=torch.tensor(
                ((0.1, 0.0, 0.3), (-0.2, 0.4, 0.1)),
                dtype=torch.float64,
            ),
            message_bias=torch.zeros(3, dtype=torch.float64),
        ),
        ModalGeneratorInteraction(
            source_node="root",
            target_node="branch-b",
            message_matrix=torch.tensor(
                ((0.2, -0.4), (0.1, 0.3)),
                dtype=torch.float64,
            ),
            message_bias=torch.zeros(2, dtype=torch.float64),
        ),
    )
    return ModalGeneratorGraphPlan(
        model_fingerprint="a" * 64,
        parameter_cluster_plan_sha256="b" * 64,
        nodes=nodes,
        interactions=edges,
    )


def test_incremental_session_matches_all_at_once_fanout_fanin() -> None:
    plan = _plan()
    inputs = {
        "layer.0.input": torch.randn(2, 5, 4),
        "layer.1.input": torch.randn(2, 5, 4),
        "layer.2.input": torch.randn(2, 5, 4),
    }
    expected = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
    )
    session = ModalGeneratorGraphSession(
        plan,
        capture_modal_states=True,
        capture_edge_messages=True,
    )
    first = session.feed("layer.0.input", inputs["layer.0.input"])
    second = session.feed("layer.1.input", inputs["layer.1.input"])
    third = session.feed("layer.2.input", inputs["layer.2.input"])
    actual = session.finish()

    assert first.executed_nodes == ("root",)
    assert second.executed_nodes == ("branch-a", "branch-b")
    assert third.executed_nodes == ("fanin",)
    assert first.live_state_width == 2
    assert second.live_state_width == 5
    assert third.live_state_width == 0
    assert session.peak_live_state_width == 7
    assert actual.traversal_order == expected.traversal_order
    assert actual.modal_states is not None
    assert actual.edge_messages is not None
    assert expected.modal_states is not None
    assert expected.edge_messages is not None
    for boundary, value in expected.outputs.items():
        torch.testing.assert_close(actual.outputs[boundary], value)
    for name, value in expected.modal_states.items():
        torch.testing.assert_close(actual.modal_states[name], value)
    for name, value in expected.edge_messages.items():
        torch.testing.assert_close(actual.edge_messages[name], value)


def test_incremental_session_matches_factorized_coordinate_node() -> None:
    node = ModalGeneratorNode(
        name="coordinate-node",
        causal_order=0,
        input_boundary="layer.0.input",
        output_boundary="layer.0.output",
        weights=_factorized_weights(9, 4, 2, 3, 4),
    )
    plan = ModalGeneratorGraphPlan(
        model_fingerprint="a" * 64,
        parameter_cluster_plan_sha256="b" * 64,
        nodes=(node,),
        interactions=(),
    )
    inputs = {"layer.0.input": torch.randn(2, 5, 4)}
    expected = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_modal_states=True,
    )
    session = ModalGeneratorGraphSession(
        plan,
        capture_modal_states=True,
    )
    session.feed("layer.0.input", inputs["layer.0.input"])
    actual = session.finish()

    torch.testing.assert_close(
        actual.outputs["layer.0.output"],
        expected.outputs["layer.0.output"],
    )
    assert actual.modal_states is not None
    assert expected.modal_states is not None
    torch.testing.assert_close(
        actual.modal_states["coordinate-node"],
        expected.modal_states["coordinate-node"],
    )


def test_pop_output_releases_completed_boundary_without_affecting_state() -> None:
    plan = _plan()
    session = ModalGeneratorGraphSession(plan)
    first = session.feed("layer.0.input", torch.randn(1, 3, 4))

    assert tuple(first.ready_outputs) == ("layer.0.output",)
    output = session.pop_output("layer.0.output")
    assert output.shape == (1, 3, 4)
    assert session.ready_output_boundaries == ()
    with pytest.raises(RuntimeError, match="not complete"):
        session.pop_output("layer.0.output")

    session.feed("layer.1.input", torch.randn(1, 3, 4))
    session.pop_output("layer.1.output")
    session.feed("layer.2.input", torch.randn(1, 3, 4))
    session.pop_output("layer.2.output")
    execution = session.finish()
    assert execution.outputs == {}


def test_session_rejects_future_duplicate_invalid_and_incomplete_inputs() -> None:
    plan = _plan()
    session = ModalGeneratorGraphSession(plan)
    with pytest.raises(RuntimeError, match="before a causal source"):
        session.feed("layer.1.input", torch.randn(1, 2, 4))
    with pytest.raises(ValueError, match="not declared"):
        session.feed("foreign", torch.randn(1, 2, 4))
    with pytest.raises(ValueError, match="trailing width"):
        session.feed("layer.0.input", torch.randn(1, 2, 3))

    session.feed("layer.0.input", torch.randn(1, 2, 4))
    with pytest.raises(RuntimeError, match="only once"):
        session.feed("layer.0.input", torch.randn(1, 2, 4))
    with pytest.raises(RuntimeError, match="incomplete"):
        session.finish()


def test_session_revalidates_its_private_plan_before_each_feed() -> None:
    session = ModalGeneratorGraphSession(_plan())
    session.plan.nodes[0].weights.input_factor.fill_(1000.0)

    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        session.feed("layer.0.input", torch.ones(1, 2, 4))
