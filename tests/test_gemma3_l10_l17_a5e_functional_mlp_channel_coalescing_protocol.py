from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from fisher_graph.gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_protocol import (
    A5E_ARM_IDS,
    A5E_MERGE_RATE_LADDER,
    A5eFunctionalMlpChannelCoalescingProtocol,
    ChannelMerge,
    GatedMlpProjectionShapes,
    PhysicalChannelCoalescingContract,
    build_a5e_functional_mlp_channel_coalescing_protocol,
    validate_compacted_weight_mocks,
    validate_matched_naive_deletion,
)


@dataclass(frozen=True)
class _TensorMock:
    shape: tuple[int, int]


def _shapes() -> GatedMlpProjectionShapes:
    return GatedMlpProjectionShapes.from_weight_mocks(
        gate_weight=_TensorMock((100, 8)),
        up_weight=_TensorMock((100, 8)),
        down_weight=_TensorMock((8, 100)),
    )


def _five_percent_contract() -> PhysicalChannelCoalescingContract:
    return PhysicalChannelCoalescingContract(
        original_shapes=_shapes(),
        merges=(
            ChannelMerge(1, 20),
            ChannelMerge(3, 20),
            ChannelMerge(5, 21),
            ChannelMerge(7, 22),
            ChannelMerge(9, 23),
        ),
    )


def test_protocol_freezes_controls_rates_and_source_safe_scope() -> None:
    protocol = build_a5e_functional_mlp_channel_coalescing_protocol(
        hidden_width=8,
        intermediate_width=100,
    )
    state = protocol.state_dict()

    assert state["artifact_role"] == "a5e_test_scaffold_only"
    assert state["actual_model_experiment_implemented"] is False
    assert state["model_weights_loaded"] is False
    assert state["tensor_values_stored"] is False
    assert state["target_layer_ordinals"] == [10, 17]
    assert tuple(arm["arm_id"] for arm in state["arms"]) == A5E_ARM_IDS
    assert protocol.merge_rates == A5E_MERGE_RATE_LADDER


def test_native_residual_control_is_intact_and_gets_no_compression_credit() -> None:
    arms = build_a5e_functional_mlp_channel_coalescing_protocol(
        hidden_width=8,
        intermediate_width=100,
    ).state_dict()["arms"]
    native, diagnostic, deletion, coalescing = arms

    assert native["native_mlp_intact"] is True
    assert native["compression_credit_allowed"] is False
    assert diagnostic["native_mlp_intact"] is True
    assert diagnostic["native_mlp_identity_preserved"] is True
    assert diagnostic["native_mlp_call_preserved"] is True
    assert diagnostic["residual_diagnostic"] is True
    assert diagnostic["residual_candidate_policy"] == (
        "reuse_identical_frozen_fit_only_residual_candidate"
    )
    assert diagnostic["native_specific_residual_refit_allowed"] is False
    assert diagnostic["residual_application_boundary"] == "layer.L.mlp.delta"
    assert diagnostic["post_feedforward_rmsnorm_attached"] is True
    assert diagnostic["physically_compacted"] is False
    assert diagnostic["compression_credit_allowed"] is False
    assert deletion["functional_survivor_refit"] is False
    assert coalescing["functional_survivor_refit"] is True


def test_rate_ladder_is_matched_at_one_two_five_and_ten_percent() -> None:
    trials = build_a5e_functional_mlp_channel_coalescing_protocol(
        hidden_width=8,
        intermediate_width=100,
    ).trials

    assert [trial.removed_channel_count for trial in trials] == [1, 2, 5, 10]
    assert [trial.compacted_shapes.intermediate_width for trial in trials] == [
        99,
        98,
        95,
        90,
    ]
    assert [trial.native_parameter_savings for trial in trials] == [
        24,
        48,
        120,
        240,
    ]
    assert all(
        trial.state_dict()["matched_compressed_arms"]
        == list(A5E_ARM_IDS[-2:])
        for trial in trials
    )


def test_functional_contract_physically_compacts_all_three_projections() -> None:
    contract = _five_percent_contract()
    state = contract.state_dict()

    assert contract.donor_channels == (1, 3, 5, 7, 9)
    assert contract.compacted_shapes == GatedMlpProjectionShapes(
        gate_proj=(95, 8),
        up_proj=(95, 8),
        down_proj=(8, 95),
    )
    assert state["physical_compaction_axes"] == {
        "gate_proj": "compact_matching_rows",
        "up_proj": "compact_matching_rows",
        "down_proj": "compact_matching_columns",
    }
    assert state["functional_survivor_refit_required"] is True
    assert state["direct_weight_averaging_is_exact"] is False
    assert state["removed_parameter_count"] == 120
    assert state["removed_matrix_macs_per_token"] == 120


def test_future_materializer_must_emit_compacted_gate_up_and_down_shapes() -> None:
    contract = _five_percent_contract()

    assert validate_compacted_weight_mocks(
        contract=contract,
        gate_weight=_TensorMock((95, 8)),
        up_weight=_TensorMock((95, 8)),
        down_weight=_TensorMock((8, 95)),
    ) is None
    with pytest.raises(ValueError, match="do not match the plan"):
        validate_compacted_weight_mocks(
            contract=contract,
            gate_weight=_TensorMock((95, 8)),
            up_weight=_TensorMock((95, 8)),
            down_weight=_TensorMock((8, 100)),
        )


def test_naive_deletion_uses_identical_donors_and_physical_rate() -> None:
    contract = _five_percent_contract()

    assert validate_matched_naive_deletion(
        contract=contract,
        naive_deleted_channels=(1, 3, 5, 7, 9),
        expected_removed_channel_count=5,
    ) is None
    with pytest.raises(ValueError, match="guided arm's donors"):
        validate_matched_naive_deletion(
            contract=contract,
            naive_deleted_channels=(0, 2, 4, 6, 8),
            expected_removed_channel_count=5,
        )


@pytest.mark.parametrize(
    "merges, message",
    [
        ((ChannelMerge(1, 20), ChannelMerge(1, 21)), "removed only once"),
        ((ChannelMerge(1, 2), ChannelMerge(2, 3)), "also be a survivor"),
        ((ChannelMerge(1, 100),), "outside the MLP width"),
    ],
)
def test_merge_contract_rejects_duplicate_cyclic_or_invalid_topology(
    merges: tuple[ChannelMerge, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PhysicalChannelCoalescingContract(
            original_shapes=_shapes(),
            merges=merges,
        )


def test_protocol_forbids_held_selection_and_orders_fit_freeze_score() -> None:
    state = build_a5e_functional_mlp_channel_coalescing_protocol(
        hidden_width=8,
        intermediate_width=100,
    ).state_dict()

    assert state["held_data_may_select_pairs"] is False
    assert state["held_data_may_refit_survivors"] is False
    assert state["residual_control_contract"] == {
        "candidate_reused_without_refit": True,
        "candidate_fit_role": "fit_only_then_frozen",
        "application_boundary": "layer.L.mlp.delta",
        "post_feedforward_rmsnorm_attached": True,
        "native_mlp_identity_and_call_preserved": True,
        "compression_credit_allowed": False,
    }
    assert state["scientific_order"] == [
        "fit_only_compute_grouped_fisher_and_channel_jacobians",
        "fit_only_rank_and_assign_donors_to_survivors",
        "fit_only_refit_functional_survivor_triplets",
        "freeze_topology_weights_and_diagnostic_artifacts",
        "held_only_score_each_frozen_arm_once",
    ]
    assert state["nonlinear_merge_contract"][
        "direct_weight_averaging_is_exact"
    ] is False


def test_protocol_rejects_scientific_scope_or_rate_drift() -> None:
    with pytest.raises(ValueError, match="L10/L17"):
        A5eFunctionalMlpChannelCoalescingProtocol(
            original_shapes=_shapes(),
            target_layer_ordinals=(17,),
        )
    with pytest.raises(ValueError, match="1/2/5/10"):
        A5eFunctionalMlpChannelCoalescingProtocol(
            original_shapes=_shapes(),
            merge_rates=(0.01, 0.05, 0.10),
        )
    with pytest.raises(ValueError, match="too small"):
        build_a5e_functional_mlp_channel_coalescing_protocol(
            hidden_width=8,
            intermediate_width=99,
        )


def test_protocol_module_has_no_tensor_or_model_runtime_dependency() -> None:
    path = Path(
        "src/fisher_graph/"
        "gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_protocol.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    assert roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "math",
    }
