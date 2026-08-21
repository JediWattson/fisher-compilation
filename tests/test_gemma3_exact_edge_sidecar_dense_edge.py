from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_joint_cross_block_chart_executor import (
    build_joint_cross_block_event_schedule,
)
from fisher_graph.gemma3_unified_event_chart_sidecar import (
    FittedUnifiedEventDenseEdge,
    UnifiedEventChartPacket,
    UnifiedEventLocalBaseline,
    evaluate_unified_event_edge,
    fit_forward_unified_event_dense_edge,
)


_SOURCE_FINGERPRINT = "a" * 64


def _sha(number: int) -> str:
    return f"{number:064x}"


def _event(event_id: str):
    return next(
        event
        for event in build_joint_cross_block_event_schedule()
        if event.event_id == event_id
    )


def _dense_problem(
    *,
    rows: int,
    execution_fingerprint: str,
    row_prefix: str,
    seed: int,
    source_event_id: str = "mutation.1",
    target_event_id: str = "mutation.6",
) -> tuple[UnifiedEventChartPacket, UnifiedEventLocalBaseline, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source_coordinates = torch.randn(rows, 1, 3, generator=generator)
    coefficient = torch.tensor(
        [
            [0.8, -0.4, 0.2, 0.7],
            [-0.3, 0.9, 0.5, -0.1],
            [0.6, 0.1, -0.8, 0.4],
        ]
    )
    local = 0.02 * torch.randn(rows, 4, generator=generator)
    target_flow = local + source_coordinates[:, 0] @ coefficient
    row_keys = tuple((row_prefix, index) for index in range(rows))
    source_event = _event(source_event_id)
    target_event = _event(target_event_id)
    source = UnifiedEventChartPacket(
        event_id=source_event.event_id,
        event_index=source_event.event_index,
        event_kind=source_event.event_kind,
        component_family=source_event.component_family,
        block_ordinal=source_event.block_ordinal,
        site_id=f"{source_event.event_id}.output",
        site_role="output",
        row_keys=row_keys,
        coordinates=source_coordinates,
        source_fingerprint=_SOURCE_FINGERPRINT,
        execution_fingerprint=execution_fingerprint,
    )
    target = UnifiedEventLocalBaseline(
        event_id=target_event.event_id,
        event_index=target_event.event_index,
        event_kind=target_event.event_kind,
        component_family=target_event.component_family,
        block_ordinal=target_event.block_ordinal,
        row_keys=row_keys,
        target_flow=target_flow,
        local_predicted_flow=local,
        target_memberships=torch.ones(rows, 1),
        row_fisher_weights=torch.linspace(0.5, 1.5, rows),
        source_fingerprint=_SOURCE_FINGERPRINT,
        execution_fingerprint=execution_fingerprint,
    )
    return source, target, coefficient


def test_dense_edge_is_one_unfactorized_map_and_closes_heldout_controls() -> None:
    fit_source, fit_target, expected_coefficient = _dense_problem(
        rows=96,
        execution_fingerprint=_sha(1),
        row_prefix="A-fit",
        seed=11,
    )
    edge = fit_forward_unified_event_dense_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=fit_source,
        target=fit_target,
        relative_ridge=1.0e-9,
    )

    assert isinstance(edge, FittedUnifiedEventDenseEdge)
    assert edge.coefficient.shape == (3, 4)
    assert edge.stored_scalar_count == 12
    assert not hasattr(edge, "left_factor")
    assert not hasattr(edge, "right_factor")
    assert torch.allclose(
        edge.coefficient,
        expected_coefficient,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    metadata = edge.metadata()
    assert metadata["fit_algorithm"] == (
        "fisher_weighted_dense_rectangular_ridge_v1"
    )
    assert metadata["edge_low_rank_factorization_used"] is False
    assert metadata["coefficient_shape"] == [3, 4]
    assert metadata["stored_scalar_count"] == 12
    assert metadata["edge_capacity_rank"] == 3
    assert metadata["realized_numerical_rank"] == 3
    assert metadata["numerical_rank_tolerance"] > 0.0
    assert metadata["numerical_rank_tolerance_rule"] == (
        "max(matrix_shape)*stored_dtype_epsilon*largest_singular_value"
    )
    assert metadata["strictly_forward"] is True
    assert metadata["cross_block"] is True
    assert metadata["cross_family"] is True
    assert metadata["event_or_layer_features_in_numerical_design"] is False

    coefficient_before = edge.coefficient.clone()
    held_source, held_target, _ = _dense_problem(
        rows=48,
        execution_fingerprint=_sha(2),
        row_prefix="B-predictive",
        seed=17,
    )
    evaluation = evaluate_unified_event_edge(
        edge,
        source=held_source,
        target=held_target,
    )
    assert evaluation.fit_rows_disjoint is True
    assert evaluation.fit_execution_disjoint is True
    assert evaluation.real_beats_zero is True
    assert evaluation.real_beats_shuffle is True
    assert evaluation.real_edge_gain_fraction > 0.999999
    assert evaluation.shuffled_source_gain_fraction < 0.25
    assert evaluation.metadata()["heldout_refit_performed"] is False
    assert torch.equal(edge.coefficient, coefficient_before)


def test_dense_fit_numerics_are_invariant_to_eligible_source_provenance() -> None:
    first_source, target, _ = _dense_problem(
        rows=72,
        execution_fingerprint=_sha(3),
        row_prefix="provenance-invariance",
        seed=23,
    )
    alternate_event = _event("mutation.3")
    alternate_source = replace(
        first_source,
        event_id=alternate_event.event_id,
        event_index=alternate_event.event_index,
        event_kind=alternate_event.event_kind,
        component_family=alternate_event.component_family,
        block_ordinal=alternate_event.block_ordinal,
        site_id=f"{alternate_event.event_id}.output",
    )

    first = fit_forward_unified_event_dense_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=first_source,
        target=target,
        relative_ridge=1.0e-7,
    )
    alternate = fit_forward_unified_event_dense_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=alternate_source,
        target=target,
        relative_ridge=1.0e-7,
    )

    assert first.spec.source_event_id != alternate.spec.source_event_id
    assert torch.equal(first.coefficient, alternate.coefficient)
    assert first.absolute_ridge == alternate.absolute_ridge
    assert first.fit_baseline_weighted_sse == (
        alternate.fit_baseline_weighted_sse
    )
    assert first.fit_edge_weighted_sse == alternate.fit_edge_weighted_sse


def test_dense_edge_fails_closed_on_topology_rows_and_execution_firewall() -> None:
    source, target, _ = _dense_problem(
        rows=32,
        execution_fingerprint=_sha(4),
        row_prefix="A-firewall",
        seed=29,
    )
    same_family_event = _event("mutation.4")
    same_family_source = replace(
        source,
        event_id=same_family_event.event_id,
        event_index=same_family_event.event_index,
        event_kind=same_family_event.event_kind,
        component_family=same_family_event.component_family,
        block_ordinal=same_family_event.block_ordinal,
        site_id=f"{same_family_event.event_id}.output",
    )
    with pytest.raises(ValueError, match="cross-family"):
        fit_forward_unified_event_dense_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=same_family_source,
            target=target,
            relative_ridge=1.0e-6,
        )
    later_event = _event("mutation.7")
    later_source = replace(
        source,
        event_id=later_event.event_id,
        event_index=later_event.event_index,
        event_kind=later_event.event_kind,
        component_family=later_event.component_family,
        block_ordinal=later_event.block_ordinal,
        site_id=f"{later_event.event_id}.output",
    )
    with pytest.raises(ValueError, match="strictly forward"):
        fit_forward_unified_event_dense_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=later_source,
            target=target,
            relative_ridge=1.0e-6,
            require_cross_block=False,
            require_cross_family=False,
        )
    with pytest.raises(ValueError, match="rows are not aligned"):
        fit_forward_unified_event_dense_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=source,
            target=replace(target, row_keys=tuple(reversed(target.row_keys))),
            relative_ridge=1.0e-6,
        )
    with pytest.raises(ValueError, match="one exact execution"):
        fit_forward_unified_event_dense_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=source,
            target=replace(target, execution_fingerprint=_sha(5)),
            relative_ridge=1.0e-6,
        )

    edge = fit_forward_unified_event_dense_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=source,
        target=target,
        relative_ridge=1.0e-6,
    )
    with pytest.raises(ValueError, match="reused fit"):
        evaluate_unified_event_edge(edge, source=source, target=target)
