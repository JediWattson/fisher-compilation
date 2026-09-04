from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisher_graph.fisher_compact_message_completion_capture import (
    write_capture_shard,
)
from fisher_graph.fisher_compact_message_completion_protocol import (
    build_shard_plan,
    verify_capture_shard_files,
)
from fisher_graph.fisher_compact_message_completion_synthetic import (
    SYNTHETIC_EVIDENCE_SCOPE,
    SyntheticCampaignConfig,
    build_synthetic_campaign,
    exact_synthetic_mlp,
    synthetic_rmsnorm,
)


def test_synthetic_fixture_rejects_nonfrozen_chart_and_head_counts() -> None:
    with pytest.raises(ValueError, match="four charts"):
        SyntheticCampaignConfig(chart_count=2)
    with pytest.raises(ValueError, match="four heads"):
        SyntheticCampaignConfig(head_count=2)


def test_synthetic_campaign_has_exact_disjoint_unit_splits() -> None:
    campaign = build_synthetic_campaign()
    assert SYNTHETIC_EVIDENCE_SCOPE == "synthetic_qualification_only"
    assert len(campaign.fit.analysis_unit_commitments) == 48
    assert len(campaign.tune.analysis_unit_commitments) == 24
    assert len(campaign.development_confirm.analysis_unit_commitments) == 24
    assert campaign.fit_split.left_name == "fit_monitor"
    assert campaign.fit_split.right_name == "fit_train"
    assert len(campaign.fit_split.left_units) == 10
    assert len(campaign.fit_split.right_units) == 38
    assert len(campaign.o1_split.left_units) == 12
    assert set(campaign.tune.analysis_unit_commitments).isdisjoint(
        campaign.development_confirm.analysis_unit_commitments
    )
    assert set(campaign.fit.analysis_unit_commitments).isdisjoint(
        campaign.tune.analysis_unit_commitments
    )


def test_synthetic_capture_has_frozen_shapes_types_and_exact_seams() -> None:
    campaign = build_synthetic_campaign()
    partition = campaign.fit
    tensors = partition.tensors
    row_count = 48 * 4
    assert partition.row_index.row_count == row_count
    assert tensors["message"].shape == (row_count, 4, 64)
    assert tensors["q_A_head"].shape == (row_count, 4, 64)
    assert tensors["h"].shape == (row_count, 64)
    assert tensors["target_token_id"].shape == (row_count,)
    assert tensors["message"].dtype == np.float32
    assert tensors["target_token_id"].dtype == np.int64
    assert all(not value.flags.writeable for value in tensors.values())

    q_a = tensors["q_A_head"].astype(np.float64).sum(axis=1)
    assert np.allclose(q_a, tensors["q_A"], atol=2e-6, rtol=2e-6)
    v_a = synthetic_rmsnorm(
        tensors["q_A"], campaign.source.post_attention_norm_weight
    )
    assert np.allclose(v_a, tensors["v_A"], atol=2e-6, rtol=2e-6)
    assert np.allclose(
        tensors["h"].astype(np.float64) + tensors["v_A"],
        tensors["h_A"],
        atol=2e-6,
        rtol=2e-6,
    )
    q_m, v_m = exact_synthetic_mlp(tensors["h_A"], campaign.source)
    assert np.allclose(q_m, tensors["q_M"], atol=2e-6, rtol=2e-6)
    assert np.allclose(v_m, tensors["v_M"], atol=2e-6, rtol=2e-6)
    assert np.allclose(
        tensors["h_A"].astype(np.float64) + tensors["v_M"],
        tensors["h_plus"],
        atol=2e-6,
        rtol=2e-6,
    )


def test_synthetic_generation_and_row_alignment_are_deterministic() -> None:
    first = build_synthetic_campaign()
    second = build_synthetic_campaign()
    for role in ("fit", "tune", "development_confirm"):
        first_partition = first.partition(role)
        second_partition = second.partition(role)
        assert first_partition.row_index.canonical_bytes == (
            second_partition.row_index.canonical_bytes
        )
        assert first_partition.metadata == second_partition.metadata
        assert set(first_partition.tensors) == set(second_partition.tensors)
        for name in first_partition.tensors:
            assert np.array_equal(
                first_partition.tensors[name], second_partition.tensors[name]
            )


def test_synthetic_partition_writes_real_reopenable_shards(tmp_path: Path) -> None:
    partition = build_synthetic_campaign().tune
    plan = build_shard_plan(partition.row_index, 31)
    for entry in plan.entries:
        write_capture_shard(
            tmp_path,
            entry,
            partition.tensors_for_shard(entry),
        )
    verification = verify_capture_shard_files(plan, tmp_path)
    assert verification.row_count == partition.row_index.row_count
    assert verification.captured_row_set_sha256 == plan.row_set_sha256
    assert verification.captured_ordered_row_sequence_sha256 == (
        plan.ordered_row_sequence_sha256
    )
