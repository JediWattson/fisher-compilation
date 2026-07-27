from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockDiscoveryResult,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    ModeKey,
    build_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)
from fisher_graph.structured_mlp_cross_block_plan import (
    StructuredMLPCrossBlockPlan,
    UnresolvedCrossBlockCarryProposal,
    plan_structured_mlp_cross_block_carries,
)


def _provenance() -> CrossBlockDiscoveryProvenance:
    return CrossBlockDiscoveryProvenance(
        model_fingerprint="a" * 64,
        calibration_split_sha256="b" * 64,
        objective_sha256="c" * 64,
        score_reduction="sum",
        normalizer="valid_activation_positions",
    )


def _discovery_result(
    *,
    layer_count: int,
    width: int,
    paired_coordinates: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        ...,
    ],
    amplitudes: dict[tuple[int, int], float] | None = None,
) -> CrossBlockDiscoveryResult:
    specs = tuple(
        CrossBlockLayerSpec(
            layer_id=f"layer.{ordinal}",
            layer_ordinal=ordinal,
            activation_site=f"layer.{ordinal}.mlp.down_input",
            width=width,
        )
        for ordinal in range(layer_count)
    )
    coordinate_count = layer_count * width
    observations = max(64, coordinate_count)
    paired_row: dict[tuple[int, int], int] = {}
    next_row = 0
    for first, second in paired_coordinates:
        paired_row[first] = next_row
        paired_row[second] = next_row
        next_row += 1
    for layer_ordinal in range(layer_count):
        for mode_index in range(width):
            coordinate = (layer_ordinal, mode_index)
            if coordinate not in paired_row:
                paired_row[coordinate] = next_row
                next_row += 1
    assert next_row <= observations

    activation_by_site: dict[str, torch.Tensor] = {}
    gradient_by_site: dict[str, torch.Tensor] = {}
    for spec in specs:
        activation = torch.zeros(
            observations,
            width,
            dtype=torch.float64,
        )
        for mode_index in range(width):
            coordinate = (spec.layer_ordinal, mode_index)
            amplitude = (
                float(mode_index + 1)
                if amplitudes is None
                else amplitudes.get(coordinate, float(mode_index + 1))
            )
            activation[paired_row[coordinate], mode_index] = amplitude
        activation_by_site[spec.activation_site] = activation
        gradient_by_site[spec.activation_site] = torch.ones_like(
            activation
        )

    rows = (
        ActivationScoreGradientRows(
            activations=activation_by_site,
            score_gradients=gradient_by_site,
            logical_positions=torch.arange(
                observations,
                dtype=torch.int64,
            ),
            loss=0.0,
            example_id="proposal-plan-fit",
        ),
    )
    sketch = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=_provenance(),
        config=CrossBlockSketchConfig(
            sketch_size=4096,
            sketch_seed=417,
            per_layer_pool_size=width,
            neighbors_per_mode=max(8, width),
            proxy_min_signed_correlation=0.9,
        ),
    )
    return replay_cross_block_discovery_shortlist(
        rows,
        sketch=sketch,
    )


def _overlapping_result() -> CrossBlockDiscoveryResult:
    pairs = (
        ((0, 2), (2, 0)),
        ((1, 1), (4, 2)),
        ((6, 0), (15, 1)),
    )
    amplitudes = {
        (0, 0): 4.0,
        (0, 1): 3.0,
        (0, 2): 5.0,
        (2, 0): 2.0,
        (2, 1): 4.0,
        (2, 2): 3.0,
        (1, 0): 2.0,
        (1, 1): 3.0,
        (1, 2): 1.0,
        (4, 0): 5.0,
        (4, 1): 4.0,
        (4, 2): 3.0,
        (6, 0): 2.0,
        (6, 1): 1.0,
        (6, 2): 0.5,
        (15, 0): 4.0,
        (15, 1): 2.0,
        (15, 2): 3.0,
    }
    result = _discovery_result(
        layer_count=16,
        width=3,
        paired_coordinates=pairs,
        amplitudes=amplitudes,
    )
    assert {
        (
            first.layer_ordinal,
            first.mode_index,
            second.layer_ordinal,
            second.mode_index,
        )
        for first, second in result.selected_pairs
    } == {
        (0, 2, 2, 0),
        (1, 1, 4, 2),
        (6, 0, 15, 1),
    }
    return result


def test_native_indices_and_fisher_ranks_stay_distinct() -> None:
    result = _overlapping_result()
    first = plan_structured_mlp_cross_block_carries(result)
    second = plan_structured_mlp_cross_block_carries(result)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.metadata() == second.metadata()
    assert first.source_discovery_artifact_sha256 == result.artifact_sha256
    assert len(first.proposals) == 3
    for proposal in first.proposals:
        assert proposal.anchor_source_index == proposal.anchor.mode_index
        assert proposal.consumer_source_index == proposal.consumer.mode_index
        assert proposal.consumer_decoder_scale is None
        assert proposal.intervention_required
        assert proposal.discovery_only
        assert not proposal.authorizes_static_merge
        assert not proposal.authorizes_intervention
        assert not proposal.authorizes_compilation
        assert not proposal.authorizes_execution
        assert not proposal.authorizes_guard
        assert not proposal.authorizes_b

    nonadjacent_rank = next(
        proposal
        for proposal in first.proposals
        if proposal.inclusive_interval == (0, 2)
    )
    assert (
        nonadjacent_rank.anchor.mode_index,
        nonadjacent_rank.anchor.fisher_rank,
    ) == (2, 0)
    assert (
        nonadjacent_rank.consumer.mode_index,
        nonadjacent_rank.consumer.fisher_rank,
    ) == (0, 2)


def test_overlapping_intervals_merge_but_disjoint_windows_do_not() -> None:
    plan = plan_structured_mlp_cross_block_carries(
        _overlapping_result()
    )

    assert tuple(
        window.inclusive_interval for window in plan.windows
    ) == ((0, 4), (6, 15))
    assert plan.windows[0].layer_ids == tuple(
        f"layer.{ordinal}" for ordinal in range(5)
    )
    assert plan.windows[1].layer_ids == tuple(
        f"layer.{ordinal}" for ordinal in range(6, 16)
    )
    assert len(plan.windows[0].proposal_ids) == 2
    assert len(plan.windows[1].proposal_ids) == 1
    assert all(
        window.intervention_required
        and window.discovery_only
        and not window.authorizes_intervention
        and not window.authorizes_compilation
        and not window.authorizes_execution
        and not window.authorizes_guard
        and not window.authorizes_b
        for window in plan.windows
    )

    assert plan.intervention_required
    assert plan.discovery_only
    assert not plan.consumer_decoder_scales_resolved
    assert not plan.authorizes_static_merge
    assert not plan.authorizes_intervention
    assert not plan.authorizes_compilation
    assert not plan.authorizes_execution
    assert not plan.authorizes_guard
    assert not plan.authorizes_b
    assert not plan.contains_source_model_weights
    assert not plan.contains_corpus_rows


def test_zero_selected_edges_produces_a_valid_empty_plan() -> None:
    result = _discovery_result(
        layer_count=2,
        width=1,
        paired_coordinates=(),
    )
    assert result.selected_pairs == ()

    plan = plan_structured_mlp_cross_block_carries(result)

    assert plan.proposals == ()
    assert plan.windows == ()
    assert plan.metadata()["proposal_count"] == 0
    assert plan.metadata()["window_count"] == 0
    restored = StructuredMLPCrossBlockPlan.from_state_dict(
        plan.state_dict()
    )
    assert restored.metadata() == plan.metadata()


def test_strict_roundtrip_and_tamper_rejection() -> None:
    plan = plan_structured_mlp_cross_block_carries(
        _overlapping_result()
    )
    restored = StructuredMLPCrossBlockPlan.from_state_dict(
        plan.state_dict()
    )
    assert restored.metadata() == plan.metadata()

    unexpected = copy.deepcopy(plan.state_dict())
    unexpected["executor"] = "forbidden"
    with pytest.raises(ValueError, match="state fields"):
        StructuredMLPCrossBlockPlan.from_state_dict(unexpected)

    resolved_scale = copy.deepcopy(plan.state_dict())
    resolved_scale["proposals"][0]["consumer_decoder_scale"] = 1.0
    with pytest.raises(ValueError, match="decoder scale"):
        StructuredMLPCrossBlockPlan.from_state_dict(resolved_scale)

    authorized = copy.deepcopy(plan.state_dict())
    authorized["authorizes_guard"] = True
    with pytest.raises(ValueError, match="safety"):
        StructuredMLPCrossBlockPlan.from_state_dict(authorized)

    changed_window = copy.deepcopy(plan.state_dict())
    changed_window["windows"][0]["end_layer_ordinal"] = 5
    with pytest.raises(ValueError, match="window"):
        StructuredMLPCrossBlockPlan.from_state_dict(changed_window)

    changed_hash = copy.deepcopy(plan.state_dict())
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        StructuredMLPCrossBlockPlan.from_state_dict(changed_hash)


def test_endpoint_disjointness_uses_native_coordinate_not_fisher_rank() -> None:
    plan = plan_structured_mlp_cross_block_carries(
        _overlapping_result()
    )
    first = plan.proposals[0]
    second = plan.proposals[1]
    disguised_same_anchor = ModeKey(
        layer_ordinal=first.anchor.layer_ordinal,
        layer_id=first.anchor.layer_id,
        activation_site=first.anchor.activation_site,
        mode_index=first.anchor.mode_index,
        fisher_rank=first.anchor.fisher_rank + 99,
    )
    collision = UnresolvedCrossBlockCarryProposal.from_mode_keys(
        disguised_same_anchor,
        second.consumer,
    )
    proposals = tuple(
        sorted(
            (first, collision),
            key=lambda proposal: (
                proposal.anchor,
                proposal.consumer,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="endpoint-disjoint by native coordinate",
    ):
        StructuredMLPCrossBlockPlan(
            source_discovery_artifact_sha256=(
                plan.source_discovery_artifact_sha256
            ),
            source_sketch_artifact_sha256=(
                plan.source_sketch_artifact_sha256
            ),
            source_model_fingerprint=plan.source_model_fingerprint,
            source_layer_specs=plan.source_layer_specs,
            proposals=proposals,
            windows=(),
            artifact_sha256="0" * 64,
        )


def test_planner_rejects_non_result_values() -> None:
    with pytest.raises(TypeError, match="CrossBlockDiscoveryResult"):
        plan_structured_mlp_cross_block_carries({})  # type: ignore[arg-type]
