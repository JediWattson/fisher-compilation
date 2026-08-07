from __future__ import annotations

import json
import math

import pytest
import torch

from fisher_graph.modal_generator_graph import (
    StateConditionedModalGeneratorInteraction,
)
from fisher_graph.state_conditioned_modal_evaluation import (
    evaluate_state_conditioned_modal_flow,
)


DTYPE = torch.float64


def _edge(
    target: str,
    *,
    message: tuple[float, float],
    gate: tuple[float, float],
    top_k: int = 1,
) -> StateConditionedModalGeneratorInteraction:
    return StateConditionedModalGeneratorInteraction(
        source_node="source",
        target_node=target,
        routing_group="same-layer-choice",
        message_matrix=torch.tensor(message, dtype=DTYPE).reshape(2, 1),
        message_bias=torch.zeros(1, dtype=DTYPE),
        gate_weight=torch.tensor(gate, dtype=DTYPE),
        gate_bias=torch.zeros(1, dtype=DTYPE),
        top_k=top_k,
    )


def _fixture() -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    tuple[StateConditionedModalGeneratorInteraction, ...],
]:
    source = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=DTYPE,
    )
    # Deliberately reverse every caller-controlled catalog. Evaluation must
    # still use canonical target order (alpha, beta).
    corrections = {
        "beta": torch.tensor([[1.0], [1.0], [2.0], [2.0]], dtype=DTYPE),
        "alpha": torch.tensor([[2.0], [2.0], [1.0], [1.0]], dtype=DTYPE),
    }
    decoders = {
        "beta": torch.tensor([[0.0, 1.0]], dtype=DTYPE),
        "alpha": torch.tensor([[1.0, 0.0]], dtype=DTYPE),
    }
    interactions = (
        _edge("beta", message=(1.0, 2.0), gate=(0.0, 1.0)),
        _edge("alpha", message=(2.0, 1.0), gate=(1.0, 0.0)),
    )
    return source, corrections, decoders, interactions


def test_canonical_shape_flow_evaluation_and_controls() -> None:
    source, corrections, decoders, interactions = _fixture()
    source_before = source.clone()
    corrections_before = {
        name: value.clone() for name, value in corrections.items()
    }
    decoders_before = {name: value.clone() for name, value in decoders.items()}
    edge_hashes = tuple(edge.artifact_sha256 for edge in interactions)
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=DTYPE)

    evaluation = evaluate_state_conditioned_modal_flow(
        source,
        corrections,
        decoders,
        interactions,
        family_ids=("family-a", "family-a", "family-b", "family-b"),
        row_weights=weights,
    )

    assert evaluation.target_nodes == ("alpha", "beta")
    assert evaluation.interaction_artifact_sha256s == (
        interactions[1].artifact_sha256,
        interactions[0].artifact_sha256,
    )
    assert evaluation.observations == 4
    assert evaluation.residual_width == 2
    assert evaluation.routes.accuracy == 1.0
    assert evaluation.routes.weighted_accuracy == 1.0
    assert evaluation.routes.macro_recall == 1.0
    assert evaluation.routes.oracle_route_counts == (2, 2)
    assert evaluation.routes.predicted_route_counts == (2, 2)
    assert evaluation.routes.oracle_route_weight_mass == (4.0, 6.0)
    assert evaluation.routes.predicted_route_weight_mass == (4.0, 6.0)
    assert evaluation.routes.majority_route_ordinal == 0
    assert evaluation.routes.majority_route_target == "alpha"
    assert evaluation.routes.majority_baseline_accuracy == 0.5
    assert evaluation.routes.weighted_majority_baseline_accuracy == 0.6
    assert evaluation.routes.oracle_max_share == 0.5
    assert evaluation.routes.predicted_max_share == 0.5
    assert evaluation.routes.oracle_normalized_entropy == pytest.approx(1.0)
    assert evaluation.routes.confusion_matrix == ((2, 0), (0, 2))

    routed = evaluation.routed_graph
    assert routed.teacher_squared_l2 == pytest.approx(20.0)
    assert routed.residual_squared_l2 == pytest.approx(4.0)
    assert routed.nrmse == pytest.approx(math.sqrt(0.2))
    assert routed.weighted_teacher_squared_l2 == pytest.approx(50.0)
    assert routed.weighted_residual_squared_l2 == pytest.approx(10.0)
    assert routed.weighted_nrmse == pytest.approx(math.sqrt(0.2))
    assert routed.sse_improvement_over_edgeless == pytest.approx(0.8)
    assert routed.weighted_sse_improvement_over_edgeless == pytest.approx(0.8)
    assert routed.aggregate_cosine == pytest.approx(2.0 / math.sqrt(5.0))
    assert routed.weighted_aggregate_cosine == pytest.approx(
        2.0 / math.sqrt(5.0)
    )
    assert routed.p90_relative_error == pytest.approx(1.0 / math.sqrt(5.0))
    assert routed.p10_cosine == pytest.approx(2.0 / math.sqrt(5.0))

    dense = evaluation.dense_all_target
    assert dense.nrmse == 0.0
    assert dense.weighted_nrmse == 0.0
    assert dense.sse_improvement_over_edgeless == 1.0
    assert dense.aggregate_cosine == pytest.approx(1.0)

    constant = evaluation.constant_oracle_majority
    assert constant.nrmse == pytest.approx(math.sqrt(0.5))
    assert constant.weighted_nrmse == pytest.approx(math.sqrt(28.0 / 50.0))
    assert constant.sse_improvement_over_edgeless == pytest.approx(0.5)
    assert tuple(family.family_id for family in evaluation.families) == (
        "family-a",
        "family-b",
    )
    assert evaluation.families[0].routes.accuracy == 1.0
    assert evaluation.families[1].routes.accuracy == 1.0
    assert evaluation.families[0].dense_all_target.nrmse == 0.0
    assert evaluation.families[1].dense_all_target.nrmse == 0.0
    # The global oracle counts tie and therefore select alpha, while family-b
    # contains only beta routes.  Each family control must use its own reported
    # majority instead of silently reusing the global constant route.
    assert evaluation.families[0].routes.majority_route_target == "alpha"
    assert evaluation.families[1].routes.majority_route_target == "beta"
    assert evaluation.families[0].constant_oracle_majority.nrmse == pytest.approx(
        math.sqrt(0.2)
    )
    assert evaluation.families[1].constant_oracle_majority.nrmse == pytest.approx(
        math.sqrt(0.2)
    )
    family_conditions = evaluation.families[1].metadata()["conditions"]
    assert isinstance(family_conditions, dict)
    constant_metadata = family_conditions["constant_oracle_majority"]
    assert isinstance(constant_metadata, dict)
    assert constant_metadata["route_target"] == "beta"
    assert constant_metadata["route_source"] == (
        "family_assessment_oracle_majority_diagnostic_only"
    )

    # The receipt contains no tensors and rejects NaN/Infinity JSON output.
    encoded = json.dumps(
        evaluation.metadata(),
        sort_keys=True,
        allow_nan=False,
    )
    assert '"assessment_read_only": true' in encoded
    assert '"weighted_nrmse"' in encoded
    assert '"assessment_oracle_majority_diagnostic_only"' in encoded

    torch.testing.assert_close(source, source_before)
    for name in corrections:
        torch.testing.assert_close(corrections[name], corrections_before[name])
        torch.testing.assert_close(decoders[name], decoders_before[name])
    assert tuple(edge.artifact_sha256 for edge in interactions) == edge_hashes


def test_router_failure_is_separate_from_dense_proposal_capacity() -> None:
    source, corrections, decoders, _ = _fixture()
    reversed_router = (
        _edge("alpha", message=(2.0, 1.0), gate=(0.0, 1.0)),
        _edge("beta", message=(1.0, 2.0), gate=(1.0, 0.0)),
    )

    evaluation = evaluate_state_conditioned_modal_flow(
        source,
        corrections,
        decoders,
        reversed_router,
    )

    assert evaluation.routes.accuracy == 0.0
    assert evaluation.routes.macro_recall == 0.0
    assert evaluation.routes.confusion_matrix == ((0, 2), (2, 0))
    assert evaluation.routed_graph.nrmse == pytest.approx(math.sqrt(0.8))
    assert evaluation.routed_graph.sse_improvement_over_edgeless == pytest.approx(
        0.2
    )
    # Both message proposals remain exact when executed densely, isolating the
    # failure to route selection rather than proposal capacity.
    assert evaluation.dense_all_target.nrmse == 0.0


def test_evaluator_rejects_incompatible_or_unscorable_inputs() -> None:
    source, corrections, decoders, interactions = _fixture()

    with pytest.raises(ValueError, match="top-1 routing group"):
        evaluate_state_conditioned_modal_flow(
            source,
            corrections,
            decoders,
            tuple(
                _edge(
                    edge.target_node,
                    message=tuple(
                        float(value)
                        for value in edge.message_matrix[:, 0].tolist()
                    ),
                    gate=tuple(
                        float(value) for value in edge.gate_weight.tolist()
                    ),
                    top_k=2,
                )
                for edge in interactions
            ),
        )

    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_state_conditioned_modal_flow(
            source,
            {"alpha": corrections["alpha"]},
            decoders,
            interactions,
        )

    with pytest.raises(ValueError, match="one residual boundary"):
        evaluate_state_conditioned_modal_flow(
            source,
            corrections,
            {
                "alpha": decoders["alpha"],
                "beta": torch.ones(1, 3, dtype=DTYPE),
            },
            interactions,
        )

    with pytest.raises(ValueError, match="positive total mass"):
        evaluate_state_conditioned_modal_flow(
            source,
            corrections,
            decoders,
            interactions,
            row_weights=torch.zeros(4, dtype=DTYPE),
        )

    with pytest.raises(ValueError, match="no aggregate signal"):
        evaluate_state_conditioned_modal_flow(
            source,
            {
                name: torch.zeros_like(value)
                for name, value in corrections.items()
            },
            decoders,
            interactions,
        )


def test_family_with_zero_weight_mass_fails_closed() -> None:
    source, corrections, decoders, interactions = _fixture()
    with pytest.raises(ValueError, match="positive row-weight mass"):
        evaluate_state_conditioned_modal_flow(
            source,
            corrections,
            decoders,
            interactions,
            family_ids=("zero", "zero", "positive", "positive"),
            row_weights=torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=DTYPE),
        )
