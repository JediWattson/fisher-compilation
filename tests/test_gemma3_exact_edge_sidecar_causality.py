from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_exact_edge_sidecar_causality import (
    evaluate_exact_dense_edge_causality,
)
from fisher_graph.gemma3_unified_event_chart_sidecar import (
    FittedUnifiedEventDenseEdge,
    FittedUnifiedEventLocalBaseline,
    UnifiedEventEdgeSpec,
    _row_identity_sha256,
)
from test_gemma3_exact_edge_sidecar_observer import _exact_executor, _sequence


def _fitted_edge() -> FittedUnifiedEventDenseEdge:
    rows = (("fit.0", 0), ("fit.1", 0))
    return FittedUnifiedEventDenseEdge(
        spec=UnifiedEventEdgeSpec(
            source_event_id="mutation.2",
            source_event_index=2,
            source_event_kind="mlp_mutation",
            source_component_family="mlp",
            source_block_ordinal=0,
            source_site_id="mutation.2.output",
            source_site_role="output",
            source_chart_index=0,
            target_event_id="mutation.3",
            target_event_index=3,
            target_event_kind="attention_mutation",
            target_component_family="attention",
            target_block_ordinal=1,
            target_chart_index=0,
        ),
        coefficient=0.01 * torch.eye(4),
        relative_ridge=1.0e-4,
        absolute_ridge=1.0e-4,
        fit_baseline_weighted_sse=1.0,
        fit_edge_weighted_sse=0.5,
        fit_row_keys=rows,
        fit_row_identity_sha256=_row_identity_sha256(rows),
        source_fingerprint="a" * 64,
        fit_execution_fingerprint="b" * 64,
    )


def _local_baseline() -> FittedUnifiedEventLocalBaseline:
    rows = (("fit.0", 0), ("fit.1", 0))
    return FittedUnifiedEventLocalBaseline(
        event_id="mutation.3",
        event_index=3,
        event_kind="attention_mutation",
        component_family="attention",
        block_ordinal=1,
        input_site_id="mutation.3.input",
        output_site_id="mutation.3.output",
        weight=torch.zeros(4, 4),
        bias=torch.zeros(4),
        relative_ridge=1.0e-4,
        absolute_ridge=1.0e-4,
        fit_zero_weighted_sse=1.0,
        fit_weighted_sse=0.5,
        fit_row_keys=rows,
        fit_row_identity_sha256=_row_identity_sha256(rows),
        source_fingerprint="a" * 64,
        fit_execution_fingerprint="b" * 64,
    )


def test_dense_edge_causality_matches_exact_jvp_and_finite_shadows() -> None:
    result = evaluate_exact_dense_edge_causality(
        executor=_exact_executor(),
        input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        sequence=_sequence(),
        edge=_fitted_edge(),
        local_baseline=_local_baseline(),
        direction_count=2,
        epsilon_ladder=(0.001, 0.003),
    )

    assert result.source_site_id == "mutation.2.output"
    assert result.target_site_id == "mutation.3.output"
    assert result.direction_count == 2
    assert result.baseline_vs_zero_shadow_max_absolute_error == 0.0
    assert result.minimum_jvp_cosine > 0.99
    assert result.maximum_jvp_relative_error < 0.05
    assert result.minimum_finite_cosine > 0.99
    assert result.maximum_finite_relative_error < 0.05
    assert result.absolute_jvp_thresholds_passed is True
    assert result.absolute_finite_thresholds_passed is True
    assert result.metadata()["baseline_model_output_replaced"] is False


def test_dense_edge_causality_rejects_mixed_fit_lineage() -> None:
    with pytest.raises(ValueError, match="fit lineage disagrees"):
        evaluate_exact_dense_edge_causality(
            executor=_exact_executor(),
            input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            sequence=_sequence(),
            edge=_fitted_edge(),
            local_baseline=replace(
                _local_baseline(),
                fit_execution_fingerprint="c" * 64,
            ),
            direction_count=1,
            epsilon_ladder=(0.001,),
        )
