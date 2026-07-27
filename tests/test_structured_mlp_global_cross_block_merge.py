from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    build_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)
from fisher_graph.structured_mlp_global_cross_block_merge import (
    GlobalCrossBlockMergePlan,
    plan_global_cross_block_merges,
)


def _discovery():
    specs = tuple(
        CrossBlockLayerSpec(
            layer_id=f"layer.{ordinal}",
            layer_ordinal=ordinal,
            activation_site=f"layer.{ordinal}.mlp.down_input",
            width=1,
        )
        for ordinal in range(4)
    )
    rows = tuple(
        ActivationScoreGradientRows(
            activations={
                spec.activation_site: torch.tensor(
                    [[value], [0.0]],
                    dtype=torch.float64,
                )
                for spec in specs
            },
            score_gradients={
                spec.activation_site: torch.ones(
                    2,
                    1,
                    dtype=torch.float64,
                )
                for spec in specs
            },
            logical_positions=torch.tensor([0, 1]),
            loss=0.0,
            example_id=f"sequence-{index}",
        )
        for index, value in enumerate((1.0, -1.0, 2.0, -2.0))
    )
    sketch = build_cross_block_discovery_sketch(
        rows,
        layer_specs=specs,
        provenance=CrossBlockDiscoveryProvenance(
            model_fingerprint="a" * 64,
            calibration_split_sha256="b" * 64,
            objective_sha256="c" * 64,
            score_reduction="sum",
            normalizer="valid_activation_positions",
        ),
        config=CrossBlockSketchConfig(
            sketch_size=32,
            sketch_seed=7,
            per_layer_pool_size=1,
            neighbors_per_mode=8,
            proxy_min_signed_correlation=-1.0,
        ),
    )
    return replay_cross_block_discovery_shortlist(rows, sketch=sketch)


def test_global_plan_removes_endpoint_quota_and_allows_native_fanout() -> None:
    discovery = _discovery()
    plan = plan_global_cross_block_merges(discovery)

    assert len(discovery.evidence) == 6
    assert len(discovery.selected_hypotheses) == 2
    assert plan.qualified_hypothesis_count == 6
    assert plan.merge_count == 3
    assert plan.maximum_accepted_merges is None
    assert plan.anchor_fanout_unbounded
    assert plan.maximum_anchor_fanout_observed == 3
    assert tuple(
        merge.anchor.layer_ordinal for merge in plan.merges
    ) == (0, 0, 0)
    assert tuple(
        merge.consumer.layer_ordinal for merge in plan.merges
    ) == (1, 2, 3)
    assert all(
        merge.activation_scale == pytest.approx(1.0)
        and merge.activation_residual_nrmse == pytest.approx(0.0)
        for merge in plan.merges
    )
    restored = GlobalCrossBlockMergePlan.from_state_dict(plan.state_dict())
    assert restored.metadata() == plan.metadata()


def test_global_plan_rejects_removed_consumer_as_a_runtime_root() -> None:
    plan = plan_global_cross_block_merges(_discovery())
    poisoned = (
        plan.merges[0],
        replace(
            plan.merges[1],
            anchor=plan.merges[0].consumer,
        ),
    )
    with pytest.raises(ValueError, match="removed consumer"):
        replace(
            plan,
            merges=poisoned,
            qualified_hypothesis_count=2,
        )
