from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fisher_graph.modal_generator_graph import ModalGeneratorInteraction
from fisher_graph.modal_interaction_fitting import (
    ModalInteractionBinding,
    ModalInteractionRateCurve,
    ModalInteractionSelection,
    fit_modal_interaction_rate_curve,
    select_modal_interactions_greedily,
)


DTYPE = torch.float64
MODEL_HASH = "a" * 64
CLUSTER_HASH = "b" * 64
FIT_HASH = "c" * 64
EVAL_HASH = "d" * 64


def _binding(
    source: str = "source",
    target: str = "target",
    source_order: int = 0,
    target_order: int = 1,
    *,
    eval_split_role: str = "open_development",
) -> ModalInteractionBinding:
    return ModalInteractionBinding(
        source_node=source,
        target_node=target,
        source_causal_order=source_order,
        target_causal_order=target_order,
        source_model_sha256=MODEL_HASH,
        parameter_cluster_plan_sha256=CLUSTER_HASH,
        source_generator_sha256="e" * 64,
        target_generator_sha256="f" * 64,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        eval_split_role=eval_split_role,
    )


def _selection(
    fit_states: dict[str, torch.Tensor],
    eval_states: dict[str, torch.Tensor],
    fit_targets: dict[str, torch.Tensor],
    eval_targets: dict[str, torch.Tensor],
    *,
    orders: dict[str, int],
    edges: tuple[tuple[str, str], ...],
    ridges: tuple[float, ...] = (0.0,),
    fit_intercept: bool = False,
    threshold: float = 0.0,
    max_incoming: int = 1,
    eval_split_role: str = "open_development",
) -> ModalInteractionSelection:
    return select_modal_interactions_greedily(
        fit_states,
        eval_states,
        fit_targets,
        eval_targets,
        node_causal_orders=orders,
        generator_artifact_sha256s={
            name: f"{index + 1:064x}"
            for index, name in enumerate(sorted(orders))
        },
        source_model_sha256=MODEL_HASH,
        parameter_cluster_plan_sha256=CLUSTER_HASH,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        candidate_edges=edges,
        ridges=ridges,
        fit_intercept=fit_intercept,
        minimum_heldout_improvement=threshold,
        max_incoming_edges=max_incoming,
        eval_split_role=eval_split_role,
    )


def test_exact_affine_interaction_recovery_and_zero_edge_baseline() -> None:
    X_fit = torch.tensor(
        [
            [-2.0, 1.0],
            [-1.0, -2.0],
            [0.0, 3.0],
            [1.0, -1.0],
            [2.0, 2.0],
            [3.0, -3.0],
        ],
        dtype=DTYPE,
    )
    X_eval = torch.tensor(
        [
            [-1.5, 0.5],
            [0.5, -2.5],
            [1.5, 3.5],
            [4.0, -1.5],
        ],
        dtype=DTYPE,
    )
    matrix = torch.tensor(
        [[2.0, -1.0, 0.5], [-0.25, 3.0, 1.5]],
        dtype=DTYPE,
    )
    bias = torch.tensor([0.75, -1.25, 2.0], dtype=DTYPE)
    Y_fit = X_fit @ matrix + bias
    Y_eval = X_eval @ matrix + bias
    fit_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 1.0, 5.0], dtype=DTYPE)
    eval_weights = torch.tensor([5.0, 1.0, 3.0, 2.0], dtype=DTYPE)

    curve = fit_modal_interaction_rate_curve(
        X_fit,
        Y_fit,
        X_eval,
        Y_eval,
        binding=_binding(),
        ridges=(0.0, 0.01),
        fisher_weights_fit=fit_weights,
        fisher_weights_eval=eval_weights,
    )
    exact = curve.candidate_for_ridge(0.0)

    assert curve.zero_fit_metrics.nrmse == pytest.approx(1.0)
    assert curve.zero_eval_metrics.weighted_nrmse == pytest.approx(1.0)
    assert exact.fit_metrics.nrmse < 1e-12
    assert exact.eval_metrics.weighted_nrmse < 1e-12
    assert exact.eval_metrics.cosine_similarity == pytest.approx(1.0)
    assert exact.eval_metrics.weighted_cosine_similarity == pytest.approx(1.0)
    assert torch.allclose(exact.factors.message_matrix, matrix, atol=1e-11)
    assert torch.allclose(exact.factors.message_bias, bias, atol=1e-11)

    edge = exact.to_graph_interaction()
    assert isinstance(edge, ModalGeneratorInteraction)
    assert edge.source_node == "source"
    assert edge.target_node == "target"
    assert torch.equal(edge.message_matrix, exact.factors.message_matrix)
    assert torch.equal(edge.message_bias, exact.factors.message_bias)


def test_no_signal_edge_is_rejected() -> None:
    source = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        dtype=DTYPE,
    )
    # Zero mean and exactly orthogonal to source.
    target = torch.tensor(
        [[1.0], [-2.0], [1.0], [1.0], [-2.0], [1.0]],
        dtype=DTYPE,
    )
    zeros = torch.zeros_like(source)
    selection = _selection(
        {"source": source, "target": zeros},
        {"source": source, "target": zeros},
        {"target": target},
        {"target": target},
        orders={"source": 0, "target": 1},
        edges=(("source", "target"),),
    )

    assert selection.steps == ()
    assert selection.interactions == ()
    assert selection.parameter_count == 0
    assert selection.macs_per_token == 0


def test_calibration_overfit_that_reverses_heldout_is_rejected() -> None:
    source_fit = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]],
        dtype=DTYPE,
    )
    source_eval = torch.tensor(
        [[-3.0], [-0.5], [0.5], [3.0]],
        dtype=DTYPE,
    )
    zeros_fit = torch.zeros_like(source_fit)
    zeros_eval = torch.zeros_like(source_eval)
    selection = _selection(
        {"source": source_fit, "target": zeros_fit},
        {"source": source_eval, "target": zeros_eval},
        {"target": 4.0 * source_fit},
        {"target": -4.0 * source_eval},
        orders={"source": 0, "target": 1},
        edges=(("source", "target"),),
    )

    # The edge is exact on fit, but doubles error relative to deletion on eval.
    assert selection.steps == ()


def test_greedy_selection_supports_fanout_and_incremental_fanin() -> None:
    a_fit = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0], [0.0]],
        dtype=DTYPE,
    )
    b_fit = torch.tensor(
        [[0.0], [0.0], [-2.0], [0.0], [0.0], [1.0]],
        dtype=DTYPE,
    )
    a_eval = torch.tensor(
        [[-3.0], [0.0], [1.5], [0.0], [2.5], [-1.0]],
        dtype=DTYPE,
    )
    b_eval = torch.tensor(
        [[0.0], [-2.0], [0.0], [1.0], [0.0], [0.0]],
        dtype=DTYPE,
    )
    zeros_fit = torch.zeros_like(a_fit)
    zeros_eval = torch.zeros_like(a_eval)
    selection = _selection(
        {
            "a": a_fit,
            "b": b_fit,
            "branch": zeros_fit,
            "sink": zeros_fit,
        },
        {
            "a": a_eval,
            "b": b_eval,
            "branch": zeros_eval,
            "sink": zeros_eval,
        },
        {
            "branch": 1.5 * a_fit,
            "sink": 2.0 * a_fit - 3.0 * b_fit,
        },
        {
            "branch": 1.5 * a_eval,
            "sink": 2.0 * a_eval - 3.0 * b_eval,
        },
        orders={"a": 0, "b": 1, "branch": 2, "sink": 3},
        edges=(
            ("a", "branch"),
            ("a", "sink"),
            ("b", "sink"),
        ),
        max_incoming=2,
        threshold=1e-6,
    )

    selected = {
        (
            step.candidate.binding.source_node,
            step.candidate.binding.target_node,
        )
        for step in selection.steps
    }
    assert selected == {
        ("a", "branch"),
        ("a", "sink"),
        ("b", "sink"),
    }
    assert len(selection.interactions) == 3
    assert all(step.heldout_improvement > 0 for step in selection.steps)
    # The final accepted message leaves each selected target essentially exact.
    final_by_target = {}
    for step in selection.steps:
        final_by_target[step.candidate.binding.target_node] = (
            step.cumulative_eval_metrics_after
        )
    assert final_by_target["branch"].nrmse < 1e-12
    assert final_by_target["sink"].nrmse < 1e-12


def test_multihop_edges_fit_the_updated_runtime_source_state() -> None:
    a_fit = torch.tensor(
        [[-3.0], [-1.0], [1.0], [3.0]],
        dtype=DTYPE,
    )
    a_eval = torch.tensor(
        [[-2.0], [-0.5], [0.5], [2.0]],
        dtype=DTYPE,
    )
    zero_fit = torch.zeros_like(a_fit)
    zero_eval = torch.zeros_like(a_eval)
    selection = _selection(
        {"a": a_fit, "b": zero_fit, "c": zero_fit},
        {"a": a_eval, "b": zero_eval, "c": zero_eval},
        {"b": 2.0 * a_fit, "c": 6.0 * a_fit},
        {"b": 2.0 * a_eval, "c": 6.0 * a_eval},
        orders={"a": 0, "b": 1, "c": 2},
        edges=(("a", "b"), ("b", "c")),
        threshold=1e-6,
    )

    assert tuple(
        (
            step.candidate.binding.source_node,
            step.candidate.binding.target_node,
        )
        for step in selection.steps
    ) == (("a", "b"), ("b", "c"))
    by_pair = {
        (edge.source_node, edge.target_node): edge
        for edge in selection.interactions
    }
    b_runtime = (
        zero_eval
        + a_eval @ by_pair[("a", "b")].message_matrix
        + by_pair[("a", "b")].message_bias
    )
    c_runtime = (
        zero_eval
        + b_runtime @ by_pair[("b", "c")].message_matrix
        + by_pair[("b", "c")].message_bias
    )
    torch.testing.assert_close(b_runtime, 2.0 * a_eval, atol=1e-12, rtol=0)
    torch.testing.assert_close(c_runtime, 6.0 * a_eval, atol=1e-12, rtol=0)


def test_tiny_exact_interaction_remains_scale_invariant() -> None:
    source_fit = (
        torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=DTYPE) * 1e-15
    )
    source_eval = (
        torch.tensor([[-3.0], [-0.5], [0.5], [3.0]], dtype=DTYPE) * 1e-15
    )
    zero_fit = torch.zeros_like(source_fit)
    zero_eval = torch.zeros_like(source_eval)
    selection = _selection(
        {"source": source_fit, "target": zero_fit},
        {"source": source_eval, "target": zero_eval},
        {"target": 2.0 * source_fit},
        {"target": 2.0 * source_eval},
        orders={"source": 0, "target": 1},
        edges=(("source", "target"),),
        threshold=0.5,
    )

    assert len(selection.steps) == 1
    step = selection.steps[0]
    assert step.cumulative_eval_metrics_before.weighted_nrmse == pytest.approx(
        1.0
    )
    assert step.cumulative_eval_metrics_after.weighted_nrmse < 1e-12


def test_ties_are_deterministic_and_do_not_depend_on_candidate_order() -> None:
    source = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]],
        dtype=DTYPE,
    )
    zeros = torch.zeros_like(source)
    arguments = (
        {"a": source, "b": source, "target": zeros},
        {"a": source, "b": source, "target": zeros},
        {"target": 2.0 * source},
        {"target": 2.0 * source},
    )
    forward = _selection(
        *arguments,
        orders={"a": 0, "b": 0, "target": 1},
        edges=(("a", "target"), ("b", "target")),
    )
    reverse = _selection(
        *arguments,
        orders={"a": 0, "b": 0, "target": 1},
        edges=(("b", "target"), ("a", "target")),
    )

    assert len(forward.steps) == len(reverse.steps) == 1
    assert forward.steps[0].candidate.binding.source_node == "a"
    assert reverse.steps[0].candidate.binding.source_node == "a"
    assert forward.artifact_sha256 == reverse.artifact_sha256


def test_backedges_and_closed_split_selection_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly forward"):
        _binding(
            source="late",
            target="early",
            source_order=2,
            target_order=1,
        )

    values = torch.tensor([[-1.0], [1.0]], dtype=DTYPE)
    zeros = torch.zeros_like(values)
    with pytest.raises(ValueError, match="strictly forward"):
        _selection(
            {"early": values, "late": zeros},
            {"early": values, "late": zeros},
            {"early": values},
            {"early": values},
            orders={"early": 0, "late": 1},
            edges=(("late", "early"),),
        )
    with pytest.raises(ValueError, match="closed guard/test"):
        _selection(
            {"early": values, "late": zeros},
            {"early": values, "late": zeros},
            {"late": values},
            {"late": values},
            orders={"early": 0, "late": 1},
            edges=(("early", "late"),),
            eval_split_role="closed_test",
        )


def test_exact_edge_resource_accounting_matches_graph_interaction() -> None:
    X = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]],
        dtype=DTYPE,
    )
    Y = X @ torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]],
        dtype=DTYPE,
    )
    candidate = fit_modal_interaction_rate_curve(
        X,
        Y,
        X,
        Y,
        binding=_binding(),
        ridges=0.0,
        fit_intercept=False,
    ).candidates[0]
    edge = candidate.to_graph_interaction()

    # Matrix: 2*3, bias: 3.  Graph MAC policy counts matrix multiplies only.
    assert candidate.parameter_count == edge.parameter_count == 9
    assert candidate.macs_per_token == edge.macs_per_token == 6
    assert (
        candidate.bias_additions_per_token
        == edge.bias_additions_per_token
        == 3
    )


def test_rate_curve_and_selection_roundtrip_detect_hash_poisoning() -> None:
    X = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]],
        dtype=DTYPE,
    )
    Y = 3.0 * X + 0.5
    curve = fit_modal_interaction_rate_curve(
        X,
        Y,
        X,
        Y,
        binding=_binding(),
        ridges=(0.0, 0.1),
    )
    restored_curve = ModalInteractionRateCurve.from_state_dict(
        curve.state_dict()
    )
    assert restored_curve.artifact_sha256 == curve.artifact_sha256
    assert torch.equal(
        restored_curve.candidates[0].factors.message_matrix,
        curve.candidates[0].factors.message_matrix,
    )

    poisoned_curve = deepcopy(curve.state_dict())
    poisoned_curve["candidates"][0]["factors"]["message_matrix"][0, 0] += 1.0
    with pytest.raises(ValueError, match="message_matrix hash mismatch"):
        ModalInteractionRateCurve.from_state_dict(poisoned_curve)

    zeros = torch.zeros_like(X)
    selection = _selection(
        {"source": X, "target": zeros},
        {"source": X, "target": zeros},
        {"target": Y},
        {"target": Y},
        orders={"source": 0, "target": 1},
        edges=(("source", "target"),),
        fit_intercept=True,
    )
    restored_selection = ModalInteractionSelection.from_state_dict(
        selection.state_dict()
    )
    assert restored_selection.artifact_sha256 == selection.artifact_sha256
    assert len(restored_selection.interactions) == 1

    poisoned_metrics = deepcopy(selection.state_dict())
    poisoned_metrics["steps"][0]["candidate"]["eval_metrics"]["nrmse"] += 0.1
    with pytest.raises(ValueError, match="candidate hash mismatch"):
        ModalInteractionSelection.from_state_dict(poisoned_metrics)


def test_selection_rejects_nested_binding_rerouting_after_fit() -> None:
    source = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]],
        dtype=DTYPE,
    )
    zeros = torch.zeros_like(source)
    selection = _selection(
        {"a": source, "b": source, "target": zeros},
        {"a": source, "b": source, "target": zeros},
        {"target": 2.0 * source},
        {"target": 2.0 * source},
        orders={"a": 0, "b": 0, "target": 1},
        edges=(("a", "target"), ("b", "target")),
    )
    binding = selection.steps[0].candidate.binding
    rerouted_source = "b" if binding.source_node == "a" else "a"
    object.__setattr__(binding, "source_node", rerouted_source)

    with pytest.raises(ValueError, match="binding hash mismatch"):
        _ = selection.interactions


def test_artifacts_retain_coefficients_but_no_raw_rows_or_prompt_data() -> None:
    X = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0]],
        dtype=DTYPE,
    )
    zeros = torch.zeros_like(X)
    selection = _selection(
        {"source": X, "target": zeros},
        {"source": X, "target": zeros},
        {"target": 2.0 * X},
        {"target": 2.0 * X},
        orders={"source": 0, "target": 1},
        edges=(("source", "target"),),
    )
    state = selection.state_dict()
    serialized_keys = repr(state)

    assert state["contains_source_model_weights"] is False
    assert state["contains_prompt_text"] is False
    assert state["contains_raw_latent_rows"] is False
    assert state["contains_target_residual_rows"] is False
    assert state["contains_generator_weights"] is False
    assert state["contains_interaction_weights"] is True
    assert state["executable"] is True
    assert state["tuned_on_closed_split"] is False
    assert "node_states_fit" not in serialized_keys
    assert "target_residuals_fit" not in serialized_keys
    assert "source_fit" not in serialized_keys
    assert "source_eval" not in serialized_keys
    assert "prompt_text': True" not in serialized_keys
