from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fisher_graph.modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
    StateConditionedModalGeneratorInteraction,
)
from fisher_graph.modal_generator_graph_session import ModalGeneratorGraphSession
from fisher_graph.state_conditioned_modal_fitting import (
    fit_state_conditioned_modal_interactions,
    teacher_flow_routing,
)


DTYPE = torch.float64
MODEL_SHA256 = "e" * 64
CLUSTER_SHA256 = "f" * 64


def _identity_weights(seed: int, width: int) -> LinearModalGeneratorNodeWeights:
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=f"{seed:064x}",
        source_model_sha256=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        input_factor=torch.eye(width, dtype=DTYPE),
        output_factor=torch.eye(width, dtype=DTYPE),
        output_bias=torch.zeros(width, dtype=DTYPE),
    )


def _identity_node(name: str, order: int, width: int) -> ModalGeneratorNode:
    return ModalGeneratorNode(
        name=name,
        causal_order=order,
        input_boundary=f"x.{name}",
        output_boundary=f"y.{name}",
        weights=_identity_weights(order + 1, width),
    )


def _plan(
    nodes: tuple[ModalGeneratorNode, ...],
    interactions: tuple[
        ModalGeneratorInteraction | StateConditionedModalGeneratorInteraction,
        ...,
    ],
) -> ModalGeneratorGraphPlan:
    return ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        nodes=nodes,
        interactions=tuple(
            sorted(
                interactions,
                key=lambda edge: (edge.source_node, edge.target_node),
            )
        ),
    )


def _conditional_edge(
    target: str,
    *,
    gate_weight: float,
    gate_bias: float = 0.0,
    message_bias: float = 1.0,
    routing_group: str = "choice",
    temperature: float = 1.0,
    top_k: int = 1,
) -> StateConditionedModalGeneratorInteraction:
    return StateConditionedModalGeneratorInteraction(
        source_node="root",
        target_node=target,
        routing_group=routing_group,
        message_matrix=torch.zeros(1, 1, dtype=DTYPE),
        message_bias=torch.tensor([message_bias], dtype=DTYPE),
        gate_weight=torch.tensor([gate_weight], dtype=DTYPE),
        gate_bias=torch.tensor([gate_bias], dtype=DTYPE),
        temperature=temperature,
        top_k=top_k,
    )


def _three_route_plan(*, top_k: int) -> ModalGeneratorGraphPlan:
    nodes = (
        _identity_node("root", 0, 1),
        _identity_node("alpha", 1, 1),
        _identity_node("beta", 2, 1),
        _identity_node("gamma", 3, 1),
    )
    interactions = (
        _conditional_edge("alpha", gate_weight=1.0, top_k=top_k),
        _conditional_edge("beta", gate_weight=-1.0, top_k=top_k),
        _conditional_edge("gamma", gate_weight=0.0, top_k=top_k),
    )
    return _plan(nodes, interactions)


def _route_inputs(values: torch.Tensor) -> dict[str, torch.Tensor]:
    zeros = torch.zeros((*values.shape[:-1], 1), dtype=values.dtype)
    return {
        "x.root": values,
        "x.alpha": zeros.clone(),
        "x.beta": zeros.clone(),
        "x.gamma": zeros.clone(),
    }


def _legacy_weights(
    seed: str,
    input_factor: list[list[float]],
    output_factor: list[list[float]],
    output_bias: list[float],
) -> LinearModalGeneratorNodeWeights:
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=seed * 64,
        source_model_sha256=MODEL_SHA256,
        parameter_cluster_plan_sha256=CLUSTER_SHA256,
        input_factor=torch.tensor(input_factor, dtype=DTYPE),
        output_factor=torch.tensor(output_factor, dtype=DTYPE),
        output_bias=torch.tensor(output_bias, dtype=DTYPE),
    )


def _legacy_chain_plan() -> ModalGeneratorGraphPlan:
    first = ModalGeneratorNode(
        name="first",
        causal_order=0,
        input_boundary="layer.0.input",
        output_boundary="layer.0.residual",
        weights=_legacy_weights(
            "a",
            [[1.0, 2.0], [-1.0, 0.5]],
            [[2.0, 0.0], [0.0, -1.0]],
            [0.25, -0.5],
        ),
    )
    second = ModalGeneratorNode(
        name="second",
        causal_order=1,
        input_boundary="layer.1.input",
        output_boundary="layer.1.residual",
        weights=_legacy_weights(
            "b",
            [[1.0], [3.0]],
            [[-2.0, 4.0]],
            [1.0, -1.0],
        ),
    )
    interaction = ModalGeneratorInteraction(
        source_node="first",
        target_node="second",
        message_matrix=torch.tensor([[0.5], [-2.0]], dtype=DTYPE),
        message_bias=torch.tensor([0.75], dtype=DTYPE),
    )
    return _plan((first, second), (interaction,))


def _assert_tensor_maps_close(
    actual: dict[str, torch.Tensor] | None,
    expected: dict[str, torch.Tensor] | None,
) -> None:
    assert actual is not None
    assert expected is not None
    assert set(actual) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0.0, atol=0.0)


def test_manual_factorized_quadratic_proposal_and_local_tangent() -> None:
    edge = StateConditionedModalGeneratorInteraction(
        source_node="root",
        target_node="target",
        routing_group="polynomial",
        message_matrix=torch.tensor([[2.0], [-1.0]], dtype=DTYPE),
        message_bias=torch.tensor([0.5], dtype=DTYPE),
        gate_weight=torch.tensor([0.25, -0.5], dtype=DTYPE),
        gate_bias=torch.tensor([0.75], dtype=DTYPE),
        quadratic_left=torch.tensor([[1.0], [0.0]], dtype=DTYPE),
        quadratic_right=torch.tensor([[0.0], [1.0]], dtype=DTYPE),
        quadratic_output=torch.tensor([[3.0]], dtype=DTYPE),
    )
    source = torch.tensor([[2.0, 4.0], [-1.0, 3.0]], dtype=DTYPE)

    torch.testing.assert_close(
        edge.proposed_message(source),
        torch.tensor([[24.5], [-13.5]], dtype=DTYPE),
    )
    torch.testing.assert_close(
        edge.routing_logit(source),
        source @ torch.tensor([0.25, -0.5], dtype=DTYPE) + 0.75,
    )

    zero = torch.zeros(3, 2, dtype=DTYPE)
    direction = torch.randn_like(zero)
    quadratic_value, quadratic_tangent = torch.autograd.functional.jvp(
        lambda value: edge.proposed_message(value)
        - (
            value @ edge.message_matrix
            + edge.message_bias
        ),
        (zero,),
        (direction,),
    )
    assert torch.count_nonzero(quadratic_value) == 0
    assert torch.count_nonzero(quadratic_tangent) == 0
    assert edge.parameter_count == 11
    assert edge.routing_macs_per_token == 2
    assert edge.message_macs_per_selected_token == 7
    assert edge.bias_additions_per_token == 3


def test_per_token_top_k_is_stable_renormalized_and_counts_evaluated_rows() -> None:
    values = torch.tensor([[[-1.0], [0.0], [1.0]]], dtype=DTYPE)
    plan = _three_route_plan(top_k=2)
    execution = ModalGeneratorGraphExecutor(plan).execute(
        _route_inputs(values),
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )
    assert execution.routing_weights is not None
    assert execution.evaluated_edge_rows is not None
    assert execution.edge_messages is not None

    high = torch.exp(torch.tensor(1.0, dtype=DTYPE))
    preferred = high / (high + 1.0)
    secondary = 1.0 / (high + 1.0)
    expected = {
        "root->alpha": torch.tensor(
            [[0.0, 0.5, preferred]], dtype=DTYPE
        ),
        "root->beta": torch.tensor(
            [[preferred, 0.5, 0.0]], dtype=DTYPE
        ),
        "root->gamma": torch.tensor(
            [[secondary, 0.0, secondary]], dtype=DTYPE
        ),
    }
    for key, weights in expected.items():
        torch.testing.assert_close(execution.routing_weights[key], weights)
        torch.testing.assert_close(execution.edge_messages[key].squeeze(-1), weights)
        assert execution.evaluated_edge_rows[key] == 2
    total = sum(execution.routing_weights.values())
    torch.testing.assert_close(total, torch.ones_like(total))
    # All logits tie at the middle token. Stable target ordering retains alpha
    # and beta and drops gamma.
    assert execution.routing_weights["root->gamma"][0, 1].item() == 0.0
    assert sum(execution.evaluated_edge_rows.values()) == values.numel() * 2
    assert plan.conditional_routing_macs_per_token == 3
    assert plan.conditional_dense_message_macs_per_token == 3
    assert plan.conditional_selected_message_macs_per_token_upper_bound == 2
    assert plan.conditional_dense_elementwise_multiplications_per_token == 3
    assert (
        plan.conditional_selected_elementwise_multiplications_per_token_upper_bound
        == 2
    )

    hard = ModalGeneratorGraphExecutor(_three_route_plan(top_k=1)).execute(
        _route_inputs(values),
        capture_routing=True,
    )
    assert hard.routing_weights is not None
    assert hard.evaluated_edge_rows == {
        "root->alpha": 2,
        "root->beta": 1,
        "root->gamma": 0,
    }
    torch.testing.assert_close(
        hard.routing_weights["root->alpha"],
        torch.tensor([[0.0, 1.0, 1.0]], dtype=DTYPE),
    )
    torch.testing.assert_close(
        hard.routing_weights["root->beta"],
        torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
    )

    # Evaluation follows the explicit top-k mask, not a post-softmax `> 0`
    # test: a selected route remains evaluated even if its probability
    # underflows to an exact zero.
    underflow_plan = _plan(
        _three_route_plan(top_k=2).nodes,
        (
            _conditional_edge(
                "alpha",
                gate_weight=0.0,
                gate_bias=0.0,
                top_k=2,
            ),
            _conditional_edge(
                "beta",
                gate_weight=0.0,
                gate_bias=-1_000.0,
                top_k=2,
            ),
            _conditional_edge(
                "gamma",
                gate_weight=0.0,
                gate_bias=-2_000.0,
                top_k=2,
            ),
        ),
    )
    underflow = ModalGeneratorGraphExecutor(underflow_plan).execute(
        _route_inputs(torch.zeros(1, 2, 1, dtype=DTYPE)),
        capture_routing=True,
    )
    assert underflow.routing_weights is not None
    assert torch.count_nonzero(underflow.routing_weights["root->beta"]) == 0
    assert underflow.evaluated_edge_rows == {
        "root->alpha": 2,
        "root->beta": 2,
        "root->gamma": 0,
    }


def test_legacy_affine_edge_and_graph_hashes_remain_pinned() -> None:
    plan = _legacy_chain_plan()
    edge = plan.interactions[0]
    assert isinstance(edge, ModalGeneratorInteraction)
    assert edge.artifact_sha256 == (
        "134192641e0da85f1c33da3957e6330ed6ae670b31c9280a27de0dc079eb7f23"
    )
    assert plan.artifact_sha256 == (
        "18239492cec81018f22da97702bf3a6a9e428c6d02486d21ef0a17c394a90dfa"
    )

    restored = ModalGeneratorGraphPlan.from_state_dict(plan.state_dict())
    assert restored.artifact_sha256 == plan.artifact_sha256
    assert isinstance(restored.interactions[0], ModalGeneratorInteraction)
    assert restored.interactions[0].artifact_sha256 == edge.artifact_sha256


def test_static_and_state_conditioned_edges_execute_without_interference() -> None:
    nodes = (
        _identity_node("root", 0, 1),
        _identity_node("alpha", 1, 1),
        _identity_node("always", 2, 1),
        _identity_node("beta", 3, 1),
    )
    static = ModalGeneratorInteraction(
        source_node="root",
        target_node="always",
        message_matrix=torch.tensor([[2.0]], dtype=DTYPE),
        message_bias=torch.tensor([0.5], dtype=DTYPE),
    )
    conditional = (
        _conditional_edge(
            "alpha",
            gate_weight=1.0,
            message_bias=3.0,
            top_k=1,
        ),
        _conditional_edge(
            "beta",
            gate_weight=-1.0,
            message_bias=5.0,
            top_k=1,
        ),
    )
    plan = _plan(nodes, (static, *conditional))
    values = torch.tensor([[-1.0], [1.0]], dtype=DTYPE)
    zeros = torch.zeros_like(values)
    execution = ModalGeneratorGraphExecutor(plan).execute(
        {
            "x.root": values,
            "x.alpha": zeros,
            "x.always": zeros,
            "x.beta": zeros,
        },
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )

    assert execution.modal_states is not None
    assert execution.edge_messages is not None
    assert execution.routing_weights is not None
    torch.testing.assert_close(
        execution.edge_messages["root->always"],
        2.0 * values + 0.5,
    )
    torch.testing.assert_close(
        execution.modal_states["always"],
        2.0 * values + 0.5,
    )
    torch.testing.assert_close(
        execution.modal_states["alpha"],
        torch.tensor([[0.0], [3.0]], dtype=DTYPE),
    )
    torch.testing.assert_close(
        execution.modal_states["beta"],
        torch.tensor([[5.0], [0.0]], dtype=DTYPE),
    )
    assert set(execution.routing_weights) == {"root->alpha", "root->beta"}


def test_routing_reads_final_source_state_and_cannot_read_future_rows() -> None:
    nodes = (
        _identity_node("upstream", 0, 1),
        _identity_node("root", 1, 1),
        _identity_node("alpha", 2, 1),
        _identity_node("beta", 3, 1),
    )
    plan = _plan(
        nodes,
        (
            ModalGeneratorInteraction(
                source_node="upstream",
                target_node="root",
                message_matrix=torch.ones(1, 1, dtype=DTYPE),
                message_bias=torch.zeros(1, dtype=DTYPE),
            ),
            _conditional_edge("alpha", gate_weight=1.0, top_k=1),
            _conditional_edge("beta", gate_weight=-1.0, top_k=1),
        ),
    )
    prefix = torch.tensor([[-1.0], [1.0]], dtype=DTYPE)

    def run(upstream: torch.Tensor):
        zeros = torch.zeros_like(upstream)
        return ModalGeneratorGraphExecutor(plan).execute(
            {
                "x.upstream": upstream,
                "x.root": zeros,
                "x.alpha": zeros,
                "x.beta": zeros,
            },
            capture_modal_states=True,
            capture_routing=True,
        )

    short = run(prefix)
    extended = run(torch.cat((prefix, torch.tensor([[100.0]], dtype=DTYPE))))
    assert short.modal_states is not None
    assert short.routing_weights is not None
    assert extended.modal_states is not None
    assert extended.routing_weights is not None
    torch.testing.assert_close(short.modal_states["root"], prefix)
    torch.testing.assert_close(
        short.routing_weights["root->alpha"],
        torch.tensor([0.0, 1.0], dtype=DTYPE),
    )
    for key, value in short.outputs.items():
        torch.testing.assert_close(value, extended.outputs[key][:-1])
    for key, value in short.routing_weights.items():
        torch.testing.assert_close(value, extended.routing_weights[key][:-1])


def test_conditional_graph_roundtrip_tamper_and_group_validation() -> None:
    plan = _three_route_plan(top_k=2)
    values = torch.tensor([[[-1.0], [0.25], [2.0]]], dtype=DTYPE)
    expected = ModalGeneratorGraphExecutor(plan).execute(
        _route_inputs(values),
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )
    restored = ModalGeneratorGraphPlan.from_state_dict(plan.state_dict())
    actual = ModalGeneratorGraphExecutor(restored).execute(
        _route_inputs(values),
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )
    assert restored.artifact_sha256 == plan.artifact_sha256
    assert all(
        isinstance(edge, StateConditionedModalGeneratorInteraction)
        for edge in restored.interactions
    )
    _assert_tensor_maps_close(actual.outputs, expected.outputs)
    _assert_tensor_maps_close(actual.modal_states, expected.modal_states)
    _assert_tensor_maps_close(actual.edge_messages, expected.edge_messages)
    _assert_tensor_maps_close(actual.routing_weights, expected.routing_weights)
    assert actual.evaluated_edge_rows == expected.evaluated_edge_rows

    serialized_edge_keys = set(restored.interactions[0].state_dict())
    assert serialized_edge_keys.isdisjoint(
        {
            "teacher_flow",
            "candidate_displacements",
            "target_messages",
            "route_labels",
            "callback",
        }
    )

    tampered = deepcopy(plan.state_dict())
    tampered["interactions"][0]["gate_weight"][0] += 1.0
    with pytest.raises(ValueError, match="gate_weight hash mismatch"):
        ModalGeneratorGraphPlan.from_state_dict(tampered)

    partial_quadratic = dict(
        source_node="root",
        target_node="alpha",
        routing_group="choice",
        message_matrix=torch.zeros(1, 1, dtype=DTYPE),
        message_bias=torch.zeros(1, dtype=DTYPE),
        gate_weight=torch.zeros(1, dtype=DTYPE),
        gate_bias=torch.zeros(1, dtype=DTYPE),
        quadratic_left=torch.ones(1, 1, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="all present or all absent"):
        StateConditionedModalGeneratorInteraction(**partial_quadratic)

    nodes = plan.nodes[:3]
    with pytest.raises(ValueError, match="at least two"):
        _plan(nodes, (_conditional_edge("alpha", gate_weight=1.0),))
    with pytest.raises(ValueError, match="inconsistent configuration"):
        _plan(
            nodes,
            (
                _conditional_edge("alpha", gate_weight=1.0, temperature=1.0),
                _conditional_edge("beta", gate_weight=-1.0, temperature=2.0),
            ),
        )
    with pytest.raises(ValueError, match="top_k exceeds"):
        _plan(
            nodes,
            (
                _conditional_edge("alpha", gate_weight=1.0, top_k=3),
                _conditional_edge("beta", gate_weight=-1.0, top_k=3),
            ),
        )

    # Names may contain `:`. Distinct (source, group) pairs that have the same
    # human-readable concatenation must remain separate routing groups.
    collision_nodes = (
        _identity_node("a", 0, 1),
        _identity_node("a:b", 1, 1),
        _identity_node("p", 2, 1),
        _identity_node("q", 3, 1),
        _identity_node("r", 4, 1),
        _identity_node("s", 5, 1),
    )

    def collision_edge(
        source: str,
        target: str,
        group: str,
        bias: float,
    ) -> StateConditionedModalGeneratorInteraction:
        return StateConditionedModalGeneratorInteraction(
            source_node=source,
            target_node=target,
            routing_group=group,
            message_matrix=torch.zeros(1, 1, dtype=DTYPE),
            message_bias=torch.tensor((bias,), dtype=DTYPE),
            gate_weight=torch.zeros(1, dtype=DTYPE),
            gate_bias=torch.tensor((bias,), dtype=DTYPE),
        )

    collision_plan = _plan(
        collision_nodes,
        (
            collision_edge("a", "p", "b:c", 1.0),
            collision_edge("a", "q", "b:c", 0.0),
            collision_edge("a:b", "r", "c", 1.0),
            collision_edge("a:b", "s", "c", 0.0),
        ),
    )
    collision_execution = ModalGeneratorGraphExecutor(collision_plan).execute(
        {
            node.input_boundary: torch.ones(1, 1, dtype=DTYPE)
            for node in collision_nodes
        },
        capture_routing=True,
    )
    assert collision_execution.routing_weights is not None
    assert len(collision_execution.routing_weights) == 4


def test_incremental_session_matches_all_at_once_conditional_graph() -> None:
    plan = _three_route_plan(top_k=2)
    generator = torch.Generator().manual_seed(91)
    values = torch.randn(2, 4, 1, generator=generator, dtype=DTYPE)
    inputs = _route_inputs(values)
    expected = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )

    session = ModalGeneratorGraphSession(
        plan,
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )
    for node in plan.nodes:
        session.feed(node.input_boundary, inputs[node.input_boundary])
    actual = session.finish()

    assert actual.traversal_order == expected.traversal_order
    _assert_tensor_maps_close(actual.outputs, expected.outputs)
    _assert_tensor_maps_close(actual.modal_states, expected.modal_states)
    _assert_tensor_maps_close(actual.edge_messages, expected.edge_messages)
    _assert_tensor_maps_close(actual.routing_weights, expected.routing_weights)
    assert actual.evaluated_edge_rows == expected.evaluated_edge_rows


def test_teacher_flow_routing_uses_flow_error_and_canonical_ties() -> None:
    teacher = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
        dtype=DTYPE,
    )
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        ],
        dtype=DTYPE,
    )
    routing = teacher_flow_routing(
        teacher,
        candidates,
        temperature=0.5,
    )

    expected_scores = torch.tensor(
        [
            [1.0, -2.0, -5.0],
            [-2.0, 1.0, -2.0],
            [-5.0, -2.0, 1.0],
            [1.0, 1.0, -5.0],
        ],
        dtype=DTYPE,
    )
    torch.testing.assert_close(
        routing.responsibilities,
        torch.softmax(expected_scores / 0.5, dim=-1),
    )
    assert routing.route_labels.tolist() == [0, 1, 2, 0]
    assert routing.routes == 3
    assert routing.observations == 4
    assert routing.mean_best_alignment == pytest.approx(1.0)
    assert routing.minimum_best_alignment == pytest.approx(1.0)
    assert routing.mean_selected_relative_residual == pytest.approx(0.0)
    assert routing.maximum_selected_relative_residual == pytest.approx(0.0)

    magnitude = teacher_flow_routing(
        torch.tensor(((1.0, 0.0),), dtype=DTYPE),
        torch.tensor((((100.0, 0.0), (1.0, 0.0)),), dtype=DTYPE),
    )
    assert magnitude.route_labels.tolist() == [1]

    near_hard = teacher_flow_routing(
        torch.tensor(((1.0, 0.0),), dtype=DTYPE),
        torch.tensor((((1.0, 0.0), (0.0, 1.0)),), dtype=DTYPE),
        temperature=1e-323,
    )
    assert torch.isfinite(near_hard.responsibilities).all()
    torch.testing.assert_close(
        near_hard.responsibilities,
        torch.tensor(((1.0, 0.0),), dtype=DTYPE),
    )


def test_synthetic_fitter_recovers_source_only_router_and_affine_messages() -> None:
    source = torch.tensor(
        [
            [-4.0, -1.0],
            [-3.0, 1.0],
            [-2.0, -2.0],
            [-1.0, 2.0],
            [1.0, -2.0],
            [2.0, 1.0],
            [3.0, -1.0],
            [4.0, 2.0],
        ],
        dtype=DTYPE,
    )
    labels = (source[:, 0] > 0).to(dtype=torch.int64)
    left = (
        source @ torch.tensor([[2.0], [-0.5]], dtype=DTYPE)
        + 0.75
    )
    right = (
        source @ torch.tensor([[-1.5], [3.0]], dtype=DTYPE)
        - 0.25
    )

    fit = fit_state_conditioned_modal_interactions(
        source,
        {"right": right, "left": left},
        labels,
        source_node="source",
        routing_group="teacher-distilled",
        top_k=1,
        router_ridge=1e-8,
        message_ridge=0.0,
        quadratic_rank=0,
    )
    repeated = fit_state_conditioned_modal_interactions(
        source,
        {"left": left, "right": right},
        labels,
        source_node="source",
        routing_group="teacher-distilled",
        top_k=1,
        router_ridge=1e-8,
        message_ridge=0.0,
        quadratic_rank=0,
    )

    assert tuple(edge.target_node for edge in fit.interactions) == (
        "left",
        "right",
    )
    assert fit.route_counts == (4, 4)
    assert fit.router_metrics.accuracy == pytest.approx(1.0)
    assert fit.router_fold_audit_observations == source.shape[0]
    assert (
        fit.router_fold_float32_max_abs_error
        <= fit.router_fold_float32_tolerance
    )
    assert tuple(metric.observations for metric in fit.edge_metrics) == (4, 4)
    assert all(metric.affine_mse < 1e-24 for metric in fit.edge_metrics)
    assert tuple(edge.artifact_sha256 for edge in fit.interactions) == tuple(
        edge.artifact_sha256 for edge in repeated.interactions
    )

    logits = torch.stack(
        tuple(edge.routing_logit(source) for edge in fit.interactions),
        dim=-1,
    )
    assert logits.argmax(dim=-1).tolist() == labels.tolist()
    for route, edge in enumerate(fit.interactions):
        selected = labels == route
        expected = left[selected] if route == 0 else right[selected]
        torch.testing.assert_close(
            edge.proposed_message(source[selected]),
            expected,
            rtol=1e-11,
            atol=1e-11,
        )

    plan = _plan(
        (
            _identity_node("source", 0, 2),
            _identity_node("left", 1, 1),
            _identity_node("right", 2, 1),
        ),
        fit.interactions,
    )
    zeros = torch.zeros(source.shape[0], 1, dtype=DTYPE)
    execution = ModalGeneratorGraphExecutor(plan).execute(
        {"x.source": source, "x.left": zeros, "x.right": zeros},
        capture_modal_states=True,
        capture_routing=True,
    )
    assert execution.modal_states is not None
    torch.testing.assert_close(
        execution.modal_states["left"],
        torch.where(labels[:, None] == 0, left, zeros),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        execution.modal_states["right"],
        torch.where(labels[:, None] == 1, right, zeros),
        rtol=1e-11,
        atol=1e-11,
    )

    with pytest.raises(ValueError, match="requires top_k=1"):
        fit_state_conditioned_modal_interactions(
            source,
            {"left": left, "right": right},
            labels,
            source_node="source",
            routing_group="teacher-distilled",
            top_k=2,
        )

    unstable_generator = torch.Generator().manual_seed(0)
    offsets = torch.randn(
        64,
        2,
        generator=unstable_generator,
        dtype=DTYPE,
    ) * torch.tensor((10.0, 20.0), dtype=DTYPE)
    offset_source = offsets + torch.tensor((10_000.0, 23_000.0), dtype=DTYPE)
    offset_labels = (
        offsets[:, 0] + 0.3 * offsets[:, 1] > 0
    ).to(dtype=torch.int64)
    zero_targets = torch.zeros(64, 1, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="not float32 route-stable"):
        fit_state_conditioned_modal_interactions(
            offset_source,
            {"left": zero_targets, "right": zero_targets},
            offset_labels,
            source_node="source",
            routing_group="unstable-fold",
        )

    held_fit = torch.tensor(
        (
            (12_070_332.301619021, 27_647_677.019239385),
            (12_070_333.301619021, 27_647_669.019239385),
        ),
        dtype=DTYPE,
    )
    held_labels = torch.tensor((1, 0), dtype=torch.int64)
    held_probe = torch.tensor(
        ((12_070_333.0, 27_647_676.0),),
        dtype=DTYPE,
    )
    held_targets = torch.zeros(2, 1, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="not float32 route-stable"):
        fit_state_conditioned_modal_interactions(
            held_fit,
            {"left": held_targets, "right": held_targets},
            held_labels,
            source_node="source",
            routing_group="held-fold",
            router_validation_states=held_probe,
            router_ridge=1e-8,
        )
