from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    ModeKey,
    build_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)
from fisher_graph.structured_mlp_cross_block_fanout import (
    GlobalCrossBlockFanoutPlan,
    build_fanout_plan_from_global_merges,
    create_cross_block_fanout_group,
    create_global_cross_block_fanout_plan,
    fit_grouped_fanout_decoder,
)
from fisher_graph.structured_mlp_global_cross_block_merge import (
    plan_global_cross_block_merges,
)


def _spec(ordinal: int, width: int) -> CrossBlockLayerSpec:
    return CrossBlockLayerSpec(
        layer_id=f"layer.{ordinal}",
        layer_ordinal=ordinal,
        activation_site=f"layer.{ordinal}.mlp.down_input",
        width=width,
    )


def _mode(spec: CrossBlockLayerSpec, index: int) -> ModeKey:
    return spec.mode_key(index, fisher_rank=index)


def _one_root_two_consumer_merge_plan():
    specs = (_spec(0, 1), _spec(1, 2))
    rows = tuple(
        ActivationScoreGradientRows(
            activations={
                specs[0].activation_site: torch.tensor(
                    [[value], [0.0]],
                    dtype=torch.float64,
                ),
                specs[1].activation_site: torch.tensor(
                    [[value, value], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
            },
            score_gradients={
                specs[0].activation_site: torch.ones(
                    2,
                    1,
                    dtype=torch.float64,
                ),
                specs[1].activation_site: torch.ones(
                    2,
                    2,
                    dtype=torch.float64,
                ),
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
            per_layer_pool_size=2,
            neighbors_per_mode=8,
            proxy_min_signed_correlation=-1.0,
        ),
    )
    discovery = replay_cross_block_discovery_shortlist(rows, sketch=sketch)
    return plan_global_cross_block_merges(discovery)


def test_weighted_ridge_exactly_recovers_multi_anchor_total_output() -> None:
    anchors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
            [0.5, -0.25],
        ],
        dtype=torch.float64,
    )
    expected_decoder = torch.tensor(
        [
            [2.0, -1.0],
            [0.5, 3.0],
            [-2.0, 0.25],
        ],
        dtype=torch.float64,
    )
    removed_output = anchors @ expected_decoder.T
    fit = fit_grouped_fanout_decoder(
        anchors,
        removed_output,
        torch.eye(3, dtype=torch.float64),
        row_weights=torch.tensor(
            [1.0, 2.0, 0.5, 3.0, 1.5, 0.75],
            dtype=torch.float64,
        ),
        fold_ids=torch.tensor([1, 0, 1, 0, 1, 0]),
        ridge=0.0,
    )

    assert torch.allclose(
        fit.fused_decoder,
        expected_decoder,
        rtol=1e-12,
        atol=1e-12,
    )
    assert fit.fit_nrmse == pytest.approx(0.0, abs=1e-14)
    assert fit.deletion_nrmse == 1.0
    assert fit.recovery_fraction_vs_deletion == pytest.approx(1.0)
    assert tuple(metric.fold_id for metric in fit.fold_metrics) == (0, 1)
    assert all(
        metric.fit_nrmse == pytest.approx(0.0, abs=1e-14)
        and metric.deletion_nrmse == 1.0
        and metric.recovery_fraction_vs_deletion == pytest.approx(1.0)
        for metric in fit.fold_metrics
    )
    assert torch.allclose(fit.predict(anchors), removed_output)


def test_builder_fuses_one_root_many_consumers_into_one_decoder() -> None:
    merge_plan = _one_root_two_consumer_merge_plan()
    assert merge_plan.merge_count == 2
    assert len({merge.anchor for merge in merge_plan.merges}) == 1
    down = torch.tensor(
        [[2.0, -3.0], [0.5, 4.0]],
        dtype=torch.float64,
    )

    plan = build_fanout_plan_from_global_merges(
        merge_plan,
        native_down_weights={1: down},
    )

    assert plan.group_count == 1
    group = plan.groups[0]
    assert group.anchor_count == 1
    assert group.consumer_count == 2
    expected = sum(
        merge.activation_scale
        * down[:, merge.consumer.mode_index]
        for merge in merge_plan.merges
    )
    assert torch.allclose(group.fused_decoder[:, 0], expected)
    assert group.native_removed_parameter_count == 12
    assert group.fused_decoder_parameter_count == 2
    assert group.net_parameter_savings == 10
    assert group.net_mac_savings_per_token == 10
    assert plan.maximum_anchor_fanout_observed == 2
    assert (
        plan.source_merge_plan_artifact_sha256
        == merge_plan.artifact_sha256
    )


def test_plan_roundtrip_authenticates_decoder_and_metadata() -> None:
    plan = build_fanout_plan_from_global_merges(
        _one_root_two_consumer_merge_plan(),
        native_down_weights={
            "layer.1": torch.tensor(
                [[1.0, 2.0], [-1.0, 0.5]],
                dtype=torch.float64,
            )
        },
    )
    restored = GlobalCrossBlockFanoutPlan.from_state_dict(plan.state_dict())

    assert restored.metadata() == plan.metadata()
    assert torch.equal(
        restored.groups[0].fused_decoder,
        plan.groups[0].fused_decoder,
    )

    poisoned = plan.state_dict()
    poisoned["groups"][0]["fused_decoder"][0, 0] += 1.0
    with pytest.raises(ValueError, match="decoder hash mismatch"):
        GlobalCrossBlockFanoutPlan.from_state_dict(poisoned)

    poisoned_header = plan.state_dict()
    poisoned_header["source_model_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="plan artifact hash mismatch"):
        GlobalCrossBlockFanoutPlan.from_state_dict(poisoned_header)


def test_constraints_and_exact_group_accounting() -> None:
    specs = (_spec(0, 2), _spec(1, 2), _spec(2, 2))
    supplied_decoder = torch.tensor(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [40.0, 4.0],
        ]
    )
    group = create_cross_block_fanout_group(
        anchors=(_mode(specs[1], 1), _mode(specs[0], 0)),
        consumers=(_mode(specs[2], 1), _mode(specs[2], 0)),
        fused_decoder=supplied_decoder,
    )
    plan = create_global_cross_block_fanout_plan(
        source_discovery_artifact_sha256="1" * 64,
        source_model_fingerprint="2" * 64,
        layer_specs=specs,
        groups=(group,),
    )

    assert group.anchors == tuple(sorted(group.anchors))
    assert group.consumers == tuple(sorted(group.consumers))
    assert torch.equal(
        group.fused_decoder,
        supplied_decoder.to(torch.float64)[:, (1, 0)],
    )
    assert group.native_removed_parameter_count == 3 * 4 * 2
    assert group.fused_decoder_parameter_count == 4 * 2
    assert group.net_parameter_savings == 4 * (3 * 2 - 2)
    assert plan.net_parameter_savings == group.net_parameter_savings
    assert plan.net_mac_savings_per_token == group.net_parameter_savings

    with pytest.raises(ValueError, match="positive net compression"):
        create_cross_block_fanout_group(
            anchors=(
                _mode(specs[0], 0),
                _mode(specs[0], 1),
                _mode(specs[1], 0),
            ),
            consumers=(_mode(specs[2], 0),),
            fused_decoder=torch.ones(4, 3),
        )

    with pytest.raises(ValueError, match="strictly forward"):
        create_cross_block_fanout_group(
            anchors=(_mode(specs[2], 0),),
            consumers=(_mode(specs[2], 1),),
            fused_decoder=torch.ones(4, 1),
        )

    first = create_cross_block_fanout_group(
        anchors=(_mode(specs[0], 0),),
        consumers=(_mode(specs[1], 0),),
        fused_decoder=torch.ones(4, 1),
    )
    second = create_cross_block_fanout_group(
        anchors=(_mode(specs[1], 0),),
        consumers=(_mode(specs[2], 0),),
        fused_decoder=torch.ones(4, 1),
    )
    with pytest.raises(ValueError, match="removed consumer"):
        create_global_cross_block_fanout_plan(
            source_discovery_artifact_sha256="1" * 64,
            source_model_fingerprint="2" * 64,
            layer_specs=specs,
            groups=(first, second),
        )

    with pytest.raises(ValueError, match="policy fields"):
        replace(plan, removed_consumers_may_anchor=True)
