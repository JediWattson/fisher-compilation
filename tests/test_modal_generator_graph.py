from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from fisher_graph.modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    fit_modal_generator_rate_curve,
)


def _weights(
    seed: str,
    input_factor: list[list[float]],
    output_factor: list[list[float]],
    output_bias: list[float],
) -> LinearModalGeneratorNodeWeights:
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=seed * 64,
        source_model_sha256="e" * 64,
        parameter_cluster_plan_sha256="f" * 64,
        input_factor=torch.tensor(input_factor, dtype=torch.float64),
        output_factor=torch.tensor(output_factor, dtype=torch.float64),
        output_bias=torch.tensor(output_bias, dtype=torch.float64),
    )


def _node(
    name: str,
    order: int,
    input_boundary: str,
    output_boundary: str,
    weights: LinearModalGeneratorNodeWeights,
) -> ModalGeneratorNode:
    return ModalGeneratorNode(
        name=name,
        causal_order=order,
        input_boundary=input_boundary,
        output_boundary=output_boundary,
        weights=weights,
    )


def _edge(
    source: str,
    target: str,
    matrix: list[list[float]],
    bias: list[float],
) -> ModalGeneratorInteraction:
    return ModalGeneratorInteraction(
        source_node=source,
        target_node=target,
        message_matrix=torch.tensor(matrix, dtype=torch.float64),
        message_bias=torch.tensor(bias, dtype=torch.float64),
    )


def _plan(
    nodes: tuple[ModalGeneratorNode, ...],
    edges: tuple[ModalGeneratorInteraction, ...],
) -> ModalGeneratorGraphPlan:
    return ModalGeneratorGraphPlan(
        model_fingerprint="e" * 64,
        parameter_cluster_plan_sha256="f" * 64,
        nodes=nodes,
        interactions=tuple(
            sorted(edges, key=lambda edge: (edge.source_node, edge.target_node))
        ),
    )


def _chain_plan() -> ModalGeneratorGraphPlan:
    first = _node(
        "first",
        0,
        "layer.0.input",
        "layer.0.residual",
        _weights(
            "a",
            [[1.0, 2.0], [-1.0, 0.5]],
            [[2.0, 0.0], [0.0, -1.0]],
            [0.25, -0.5],
        ),
    )
    second = _node(
        "second",
        1,
        "layer.1.input",
        "layer.1.residual",
        _weights(
            "b",
            [[1.0], [3.0]],
            [[-2.0, 4.0]],
            [1.0, -1.0],
        ),
    )
    interaction = _edge(
        "first",
        "second",
        [[0.5], [-2.0]],
        [0.75],
    )
    return _plan((first, second), (interaction,))


def test_chain_traversal_matches_manual_linear_algebra() -> None:
    plan = _chain_plan()
    executor = ModalGeneratorGraphExecutor(plan)
    x0 = torch.tensor(
        [[1.0, 2.0], [-2.0, 0.5]],
        dtype=torch.float64,
    )
    x1 = torch.tensor(
        [[0.5, -1.0], [2.0, 3.0]],
        dtype=torch.float64,
    )

    execution = executor(
        {
            "layer.0.input": x0,
            "layer.1.input": x1,
        },
        capture_modal_states=True,
        capture_edge_messages=True,
    )

    first = plan.nodes[0]
    second = plan.nodes[1]
    edge = plan.interactions[0]
    z0 = x0 @ first.weights.input_factor
    message = z0 @ edge.message_matrix + edge.message_bias
    z1 = x1 @ second.weights.input_factor + message
    expected0 = (
        z0 @ first.weights.output_factor + first.weights.output_bias
    )
    expected1 = (
        z1 @ second.weights.output_factor + second.weights.output_bias
    )

    assert execution.traversal_order == ("first", "second")
    assert torch.equal(execution.modal_states["first"], z0)
    assert torch.equal(execution.modal_states["second"], z1)
    assert torch.equal(execution.edge_messages["first->second"], message)
    assert torch.equal(
        execution.outputs["layer.0.residual"],
        expected0,
    )
    assert torch.equal(
        execution.outputs["layer.1.residual"],
        expected1,
    )


def test_fanout_and_fanin_use_final_generated_states() -> None:
    scalar = lambda seed: _weights(seed, [[1.0]], [[1.0]], [0.0])
    nodes = (
        _node("root", 0, "x.root", "y.root", scalar("a")),
        _node("left", 1, "x.left", "y.branch", scalar("b")),
        _node("right", 2, "x.right", "y.branch", scalar("c")),
        _node("sink", 3, "x.sink", "y.sink", scalar("d")),
    )
    edges = (
        _edge("root", "left", [[2.0]], [0.0]),
        _edge("root", "right", [[-1.0]], [1.0]),
        _edge("left", "sink", [[3.0]], [0.5]),
        _edge("right", "sink", [[4.0]], [-0.5]),
    )
    plan = _plan(nodes, edges)

    execution = ModalGeneratorGraphExecutor(plan)(
        {
            "x.root": torch.tensor([[2.0]], dtype=torch.float64),
            "x.left": torch.tensor([[1.0]], dtype=torch.float64),
            "x.right": torch.tensor([[3.0]], dtype=torch.float64),
            "x.sink": torch.tensor([[5.0]], dtype=torch.float64),
        },
        capture_modal_states=True,
        capture_edge_messages=True,
    )

    # root=2; left=1+2*2=5; right=3+(-2+1)=2;
    # sink=5+(5*3+.5)+(2*4-.5)=28.
    assert execution.modal_states["root"].item() == 2.0
    assert execution.modal_states["left"].item() == 5.0
    assert execution.modal_states["right"].item() == 2.0
    assert execution.modal_states["sink"].item() == 28.0
    # Two nodes target the same residual boundary, so contributions sum.
    assert execution.outputs["y.branch"].item() == 7.0
    assert set(execution.edge_messages) == {
        "root->left",
        "root->right",
        "left->sink",
        "right->sink",
    }


def test_instrumentation_is_absent_until_requested() -> None:
    executor = ModalGeneratorGraphExecutor(_chain_plan())
    values = {
        "layer.0.input": torch.ones(1, 2, dtype=torch.float64),
        "layer.1.input": torch.ones(1, 2, dtype=torch.float64),
    }

    plain = executor(values)
    states_only = executor(values, capture_modal_states=True)
    messages_only = executor(values, capture_edge_messages=True)

    assert plain.modal_states is None
    assert plain.edge_messages is None
    assert states_only.modal_states is not None
    assert states_only.edge_messages is None
    assert messages_only.modal_states is None
    assert messages_only.edge_messages is not None
    assert torch.equal(plain.outputs["layer.1.residual"], states_only.outputs[
        "layer.1.residual"
    ])


def test_adapter_copies_fitted_factors_and_never_calls_source() -> None:
    class Factors:
        input_factor = torch.tensor([[2.0]], dtype=torch.float64)
        output_factor = torch.tensor([[3.0]], dtype=torch.float64)
        bias = torch.tensor([4.0], dtype=torch.float64)

    class SourcePlan:
        artifact_sha256 = "a" * 64
        factors = Factors()
        binding = type(
            "Binding",
            (),
            {
                "source_model_sha256": "e" * 64,
                "cluster_plan_sha256": "f" * 64,
            },
        )()

        def validate_integrity(self) -> None:
            return None

        def apply(self, value: torch.Tensor) -> torch.Tensor:
            raise AssertionError("source plan must not be called")

    source = SourcePlan()
    copied = LinearModalGeneratorNodeWeights.from_modal_generator_plan(source)
    graph = _plan(
        (_node("only", 0, "input", "output", copied),),
        (),
    )
    # Poison the source after compilation.  Runtime owns copied factors.
    source.factors.input_factor.fill_(999.0)

    output = ModalGeneratorGraphExecutor(graph)(
        {"input": torch.tensor([[5.0]], dtype=torch.float64)}
    ).outputs["output"]
    assert output.item() == 34.0


def test_adapter_authenticates_real_fitted_generator_roundtrip() -> None:
    binding = ModalGeneratorBinding.create(
        generator_id="cluster.0",
        input_kind="native_layer_input",
        input_site="layer.0.input",
        output_site="layer.0.cluster.0.residual",
        source_model_sha256="e" * 64,
        input_catalog_sha256="1" * 64,
        output_catalog_sha256="2" * 64,
        cluster_plan_sha256="f" * 64,
        fit_split_sha256="3" * 64,
        eval_split_sha256="4" * 64,
    )
    X = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    Y = torch.tensor(
        [[2.0], [-1.0], [1.0]],
        dtype=torch.float64,
    )
    X_eval = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        dtype=torch.float64,
    )
    Y_eval = torch.tensor(
        [[4.0], [-2.0], [2.0]],
        dtype=torch.float64,
    )
    curve = fit_modal_generator_rate_curve(
        X,
        Y,
        torch.ones(3, dtype=torch.float64),
        X_eval,
        Y_eval,
        (1,),
        binding=binding,
        fit_intercept=False,
    )
    source_plan = curve.points[0].plan

    copied = LinearModalGeneratorNodeWeights.from_modal_generator_plan(
        source_plan
    )
    assert copied.generator_artifact_sha256 == source_plan.artifact_sha256
    assert copied.source_model_sha256 == binding.source_model_sha256
    assert (
        copied.parameter_cluster_plan_sha256
        == binding.cluster_plan_sha256
    )
    assert torch.equal(
        copied.input_factor,
        source_plan.factors.input_factor,
    )

    source_plan.factors.input_factor[0, 0] += 1.0
    with pytest.raises(ValueError, match="does not match tensor"):
        LinearModalGeneratorNodeWeights.from_modal_generator_plan(source_plan)


def test_state_roundtrip_and_hash_poisoning_are_strict() -> None:
    plan = _chain_plan()
    restored = ModalGeneratorGraphPlan.from_state_dict(plan.to_state_dict())

    assert restored.metadata() == plan.metadata()
    assert torch.equal(
        restored.nodes[0].weights.input_factor,
        plan.nodes[0].weights.input_factor,
    )

    poisoned_tensor = deepcopy(plan.to_state_dict())
    poisoned_tensor["nodes"][0]["weights"]["input_factor"][0, 0] += 1.0
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        ModalGeneratorGraphPlan.from_state_dict(poisoned_tensor)

    poisoned_edge = deepcopy(plan.to_state_dict())
    poisoned_edge["interactions"][0]["message_bias"][0] += 1.0
    with pytest.raises(ValueError, match="message_bias hash mismatch"):
        ModalGeneratorGraphPlan.from_state_dict(poisoned_edge)

    with pytest.raises(ValueError, match="source model"):
        replace(plan, model_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="graph artifact hash mismatch"):
        replace(plan, artifact_sha256="0" * 64)


def test_executor_isolated_from_post_construction_artifact_mutation() -> None:
    plan = _chain_plan()
    executor = ModalGeneratorGraphExecutor(plan)
    plan.nodes[0].weights.input_factor.fill_(1000.0)
    values = {
        "layer.0.input": torch.ones(1, 2, dtype=torch.float64),
        "layer.1.input": torch.ones(1, 2, dtype=torch.float64),
    }

    # The private runtime copy remains authenticated and unchanged.
    output = executor(values).outputs["layer.0.residual"]
    assert torch.equal(
        output,
        torch.tensor([[0.25, -3.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        ModalGeneratorGraphExecutor(plan)


def test_executor_revalidates_its_private_plan_before_execution() -> None:
    executor = ModalGeneratorGraphExecutor(_chain_plan())
    executor.plan.nodes[0].weights.input_factor.fill_(1000.0)
    values = {
        "layer.0.input": torch.ones(1, 2, dtype=torch.float64),
        "layer.1.input": torch.ones(1, 2, dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        executor(values)


def test_exact_parameter_mac_and_addition_accounting() -> None:
    first = _node(
        "first",
        0,
        "x.first",
        "shared.out",
        _weights(
            "a",
            [[1.0, 0.0], [0.0, 1.0]],  # 2 * 2
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # 2 * 3
            [0.0, 0.0, 0.0],  # 3
        ),
    )
    second = _node(
        "second",
        1,
        "x.second",
        "shared.out",
        _weights(
            "b",
            [[1.0], [0.0], [0.0]],  # 3 * 1
            [[1.0, 0.0, 0.0]],  # 1 * 3
            [0.0, 0.0, 0.0],  # 3
        ),
    )
    plan = _plan(
        (first, second),
        (_edge("first", "second", [[1.0], [0.0]], [0.0]),),
    )
    accounting = plan.accounting

    # Nodes: (4+6+3) + (3+3+3) = 22 params, 10+6 = 16 MACs.
    # Edge: 2*1 matrix + one bias = 3 params, 2 MACs.
    assert accounting.node_parameter_count == 22
    assert accounting.interaction_parameter_count == 3
    assert accounting.parameter_count == plan.parameter_count == 25
    assert accounting.node_macs_per_token == 16
    assert accounting.interaction_macs_per_token == 2
    assert accounting.macs_per_token == plan.macs_per_token == 18
    # Six node output biases plus one edge bias.
    assert accounting.bias_additions_per_token == 7
    # One edge message enters a target; two node outputs share one width-3 sum.
    assert accounting.message_accumulation_additions_per_token == 1
    assert accounting.output_accumulation_additions_per_token == 3
    assert accounting.elementwise_additions_per_token == 11


def test_rejects_noncausal_cycles_backedges_and_dimension_drift() -> None:
    one = _weights("a", [[1.0]], [[1.0]], [0.0])
    two_latent = _weights(
        "b",
        [[1.0, 0.0]],
        [[1.0], [0.0]],
        [0.0],
    )
    early = _node("early", 0, "x.early", "y.early", one)
    late = _node("late", 1, "x.late", "y.late", one)

    with pytest.raises(ValueError, match="strictly forward"):
        _plan(
            (early, late),
            (_edge("late", "early", [[1.0]], [0.0]),),
        )
    with pytest.raises(ValueError, match="strictly forward"):
        _plan(
            (early, late),
            (
                _edge("early", "late", [[1.0]], [0.0]),
                _edge("late", "early", [[1.0]], [0.0]),
            ),
        )
    with pytest.raises(ValueError, match="source dimension"):
        _plan(
            (
                _node("early", 0, "x.early", "y.early", two_latent),
                late,
            ),
            (_edge("early", "late", [[1.0]], [0.0]),),
        )
    with pytest.raises(ValueError, match="target dimension"):
        _plan(
            (
                early,
                _node("late", 1, "x.late", "y.late", two_latent),
            ),
            (_edge("early", "late", [[1.0]], [0.0]),),
        )


def test_rejects_boundary_catalog_and_runtime_shape_drift() -> None:
    scalar = _weights("a", [[1.0]], [[1.0]], [0.0])
    input_two = _weights(
        "b",
        [[1.0], [0.0]],
        [[1.0]],
        [0.0],
    )
    output_two = _weights(
        "c",
        [[1.0]],
        [[1.0, 0.0]],
        [0.0, 0.0],
    )

    with pytest.raises(ValueError, match="input boundary"):
        _plan(
            (
                _node("a", 0, "shared.input", "y.a", scalar),
                _node("b", 1, "shared.input", "y.b", input_two),
            ),
            (),
        )
    with pytest.raises(ValueError, match="output boundary"):
        _plan(
            (
                _node("a", 0, "x.a", "shared.output", scalar),
                _node("b", 1, "x.b", "shared.output", output_two),
            ),
            (),
        )

    plan = _plan(
        (
            _node("a", 0, "x.a", "y.a", scalar),
            _node("b", 1, "x.b", "y.b", scalar),
        ),
        (_edge("a", "b", [[1.0]], [0.0]),),
    )
    executor = ModalGeneratorGraphExecutor(plan)
    with pytest.raises(ValueError, match="external input boundaries mismatch"):
        executor({"x.a": torch.ones(1, 1, dtype=torch.float64)})
    with pytest.raises(ValueError, match="runtime batch"):
        executor(
            {
                "x.a": torch.ones(2, 1, dtype=torch.float64),
                "x.b": torch.ones(3, 1, dtype=torch.float64),
            }
        )
