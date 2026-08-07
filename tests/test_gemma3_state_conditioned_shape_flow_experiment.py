from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fisher_graph.gemma3_state_conditioned_shape_flow_experiment import (
    _canonical_finite_floats,
    _scaled_state_conditioned_graph,
    calibrate_gemma3_state_conditioned_shape_flow_gain,
)
from fisher_graph.modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
    StateConditionedModalGeneratorInteraction,
)


DTYPE = torch.float64
MODEL_SHA256 = "a" * 64
CLUSTER_SHA256 = "b" * 64


def _node(name: str, causal_order: int) -> ModalGeneratorNode:
    return ModalGeneratorNode(
        name=name,
        causal_order=causal_order,
        input_boundary=f"{name}.input",
        output_boundary=f"{name}.output",
        weights=LinearModalGeneratorNodeWeights(
            generator_artifact_sha256=f"{causal_order + 1:064x}",
            source_model_sha256=MODEL_SHA256,
            parameter_cluster_plan_sha256=CLUSTER_SHA256,
            input_factor=torch.eye(2, dtype=DTYPE),
            output_factor=torch.eye(2, dtype=DTYPE),
            output_bias=torch.zeros(2, dtype=DTYPE),
        ),
    )


def _conditional_edge(
    target_node: str,
    *,
    quadratic: bool,
    gate_weight: tuple[float, float],
    gate_bias: float,
) -> StateConditionedModalGeneratorInteraction:
    return StateConditionedModalGeneratorInteraction(
        source_node="root",
        target_node=target_node,
        routing_group="frozen-route",
        message_matrix=torch.tensor(
            ((2.0, -1.0), (3.0, 4.0)),
            dtype=DTYPE,
        ),
        message_bias=torch.tensor((1.0, -2.0), dtype=DTYPE),
        gate_weight=torch.tensor(gate_weight, dtype=DTYPE),
        gate_bias=torch.tensor((gate_bias,), dtype=DTYPE),
        quadratic_left=(
            torch.tensor(((1.0, 0.0), (0.0, 2.0)), dtype=DTYPE)
            if quadratic
            else None
        ),
        quadratic_right=(
            torch.tensor(((0.0, 2.0), (1.0, 0.0)), dtype=DTYPE)
            if quadratic
            else None
        ),
        quadratic_output=(
            torch.tensor(((2.0, -1.0), (1.0, 3.0)), dtype=DTYPE)
            if quadratic
            else None
        ),
        temperature=0.75,
        top_k=1,
    )


def _conditional_graph() -> ModalGeneratorGraphPlan:
    return ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        nodes=(
            _node("root", 0),
            _node("affine", 1),
            _node("quadratic", 2),
        ),
        interactions=(
            _conditional_edge(
                "affine",
                quadratic=False,
                gate_weight=(1.0, -0.5),
                gate_bias=-0.25,
            ),
            _conditional_edge(
                "quadratic",
                quadratic=True,
                gate_weight=(-0.5, 1.0),
                gate_bias=0.5,
            ),
        ),
    )


@pytest.mark.parametrize("gain", (2.0, -1.0, 0.0))
def test_scaled_graph_multiplies_proposals_and_preserves_routes(
    gain: float,
) -> None:
    graph = _conditional_graph()
    scaled = _scaled_state_conditioned_graph(graph, gain)
    source = torch.tensor(
        ((2.0, 1.0), (-1.0, 3.0), (0.0, -2.0)),
        dtype=DTYPE,
    )

    original_logits = []
    scaled_logits = []
    for original_edge, scaled_edge in zip(
        graph.interactions,
        scaled.interactions,
        strict=True,
    ):
        assert isinstance(
            original_edge,
            StateConditionedModalGeneratorInteraction,
        )
        assert isinstance(
            scaled_edge,
            StateConditionedModalGeneratorInteraction,
        )
        torch.testing.assert_close(
            scaled_edge.proposed_message(source),
            original_edge.proposed_message(source) * gain,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.equal(
            scaled_edge.message_matrix,
            original_edge.message_matrix * gain,
        )
        assert torch.equal(
            scaled_edge.message_bias,
            original_edge.message_bias * gain,
        )
        if original_edge.quadratic_output is None:
            assert scaled_edge.quadratic_output is None
        else:
            assert scaled_edge.quadratic_output is not None
            assert torch.equal(
                scaled_edge.quadratic_output,
                original_edge.quadratic_output * gain,
            )
        assert torch.equal(scaled_edge.gate_weight, original_edge.gate_weight)
        assert torch.equal(scaled_edge.gate_bias, original_edge.gate_bias)
        assert scaled_edge.temperature == original_edge.temperature
        assert scaled_edge.top_k == original_edge.top_k
        original_logits.append(original_edge.routing_logit(source))
        scaled_logits.append(scaled_edge.routing_logit(source))

    original_route_logits = torch.stack(tuple(original_logits), dim=-1)
    scaled_route_logits = torch.stack(tuple(scaled_logits), dim=-1)
    assert torch.equal(scaled_route_logits, original_route_logits)
    assert torch.equal(
        scaled_route_logits.argmax(dim=-1),
        original_route_logits.argmax(dim=-1),
    )
    assert scaled.model_fingerprint == graph.model_fingerprint
    assert (
        scaled.parameter_cluster_plan_sha256
        == graph.parameter_cluster_plan_sha256
    )
    scaled.validate_integrity()


@pytest.mark.parametrize("gain", (math.nan, math.inf, -math.inf))
def test_scaled_graph_rejects_nonfinite_gain(gain: float) -> None:
    with pytest.raises(ValueError, match="gain must be finite"):
        _scaled_state_conditioned_graph(_conditional_graph(), gain)


def test_scaled_graph_rejects_empty_legacy_mixed_and_invalid_graphs() -> None:
    conditional = _conditional_graph()
    empty = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        nodes=conditional.nodes,
        interactions=(),
    )
    legacy_edge = ModalGeneratorInteraction(
        source_node="root",
        target_node="affine",
        message_matrix=torch.eye(2, dtype=DTYPE),
        message_bias=torch.zeros(2, dtype=DTYPE),
    )
    legacy = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        nodes=conditional.nodes,
        interactions=(legacy_edge,),
    )
    mixed_nodes = (*conditional.nodes, _node("legacy", 3))
    mixed_edge = ModalGeneratorInteraction(
        source_node="root",
        target_node="legacy",
        message_matrix=torch.eye(2, dtype=DTYPE),
        message_bias=torch.zeros(2, dtype=DTYPE),
    )
    mixed = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        nodes=mixed_nodes,
        interactions=tuple(
            sorted(
                (*conditional.interactions, mixed_edge),
                key=lambda edge: (edge.source_node, edge.target_node),
            )
        ),
    )

    for graph in (empty, legacy, mixed):
        with pytest.raises(ValueError, match="all-state-conditioned graph"):
            _scaled_state_conditioned_graph(graph, 0.5)

    invalid = _conditional_graph()
    invalid.interactions[0].message_matrix[0, 0] += 1.0
    with pytest.raises(ValueError, match="message_matrix hash mismatch"):
        _scaled_state_conditioned_graph(invalid, 0.5)


def test_canonical_finite_gain_grid_accepts_signed_zero_and_fractions() -> None:
    assert _canonical_finite_floats(
        (-1, -0.25, 0, 0.125, 1),
        label="gains",
    ) == (-1.0, -0.25, 0.0, 0.125, 1.0)


def test_gain_calibration_requires_zero_control_before_loading_candidate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="include the zero control"):
        calibrate_gemma3_state_conditioned_shape_flow_gain(
            candidate_path=tmp_path / "missing-parent.pt",
            output=tmp_path / "gain-v2.pt",
            gains=(-1.0, 1.0),
        )


@pytest.mark.parametrize(
    "values",
    (
        (),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, math.nan),
        (0.0, math.inf),
        (-math.inf, 0.0),
    ),
)
def test_canonical_finite_gain_grid_rejects_invalid_values(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="finite, unique, and increasing"):
        _canonical_finite_floats(values, label="gains")


@pytest.mark.parametrize("values", ("-1,0,1", b"-1,0,1", None))
def test_canonical_finite_gain_grid_requires_a_sequence(values: object) -> None:
    with pytest.raises(TypeError, match="gains must be a sequence"):
        _canonical_finite_floats(values, label="gains")  # type: ignore[arg-type]
