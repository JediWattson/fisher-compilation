from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

import pytest

from fisher_graph.gemma3_l3_l4_function_preserving_expert_count_control_protocol import (
    DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256,
    EXPERT_COUNT_CONTROL_EXECUTION_DEVICE,
    EXPERT_COUNT_CONTROL_EXECUTION_DTYPE,
    EXPERT_COUNT_CONTROL_EXPERT_RANK,
    EXPERT_COUNT_CONTROL_OUTER_RANK,
    EXPERT_COUNT_CONTROL_PRIMARY_SEED,
    EXPERT_COUNT_CONTROL_REPLICATION_SEED,
    EXPERT_COUNT_CONTROL_SOURCE_EXPERTS,
    EXPERT_COUNT_CONTROL_TARGET_EXPERTS,
    DormantChildExpertCountLiftSpec,
    ExpertCountExecutorSpec,
    ExpertCountPreflightBindings,
    FunctionPreservingExpertCountControlProtocol,
    PairedExpertCountTrainingSpec,
    SourceExpertRankResultBindings,
    default_function_preserving_expert_count_control_protocol,
    fit_replay_sequence_sha256,
)


def test_protocol_literal_trust_anchor_and_strict_round_trip() -> None:
    protocol = default_function_preserving_expert_count_control_protocol()

    assert protocol.protocol_sha256 == (
        DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256
    )
    assert protocol.protocol_sha256 == (
        "3d4cfbc2e69434e5cfb5845ad59ae3087457b175faa02afefa7edad5935acc27"
    )
    assert (
        FunctionPreservingExpertCountControlProtocol.from_state_dict(
            protocol.state_dict()
        )
        == protocol
    )


def test_protocol_module_imports_only_standard_library() -> None:
    path = Path(
        "src/fisher_graph/"
        "gemma3_l3_l4_function_preserving_expert_count_control_protocol.py"
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
        "hashlib",
        "json",
        "re",
    }


def test_fixed_geometry_changes_only_routed_expert_count() -> None:
    protocol = default_function_preserving_expert_count_control_protocol()
    state = protocol.state_dict()

    assert EXPERT_COUNT_CONTROL_OUTER_RANK == 64
    assert EXPERT_COUNT_CONTROL_EXPERT_RANK == 64
    assert EXPERT_COUNT_CONTROL_SOURCE_EXPERTS == 2
    assert EXPERT_COUNT_CONTROL_TARGET_EXPERTS == 4
    assert protocol.e2_executor == ExpertCountExecutorSpec(expert_count=2)
    assert protocol.e4_executor == ExpertCountExecutorSpec(expert_count=4)
    for executor in (protocol.e2_executor, protocol.e4_executor):
        assert executor.input_modes == 66
        assert executor.output_modes == 64
        assert executor.expert_rank == 64
        assert executor.router_width == 16
        assert executor.same_position_skip is False
        assert executor.max_positive_lag is None
        assert executor.router_activation == "tanh"
        assert executor.source_normalized_routing is True
    assert state["controlled_change"] == (
        "gradient_open_dormant_child_expert_count_2_to_4_at_fixed_outer64_"
        "expert_rank64_router_width16_and_exact_initial_observable_and_jvp"
    )
    assert state["expert_count_is_only_controlled_change"] is True
    assert state["router_output_cardinality_change_unavoidable"] is True


def test_training_freezes_d3_objective_data_and_schedule() -> None:
    training = PairedExpertCountTrainingSpec()
    state = training.state_dict()

    assert training.primary_seed == (
        EXPERT_COUNT_CONTROL_PRIMARY_SEED
    ) == 20_260_728_402
    assert training.replication_seed == (
        EXPERT_COUNT_CONTROL_REPLICATION_SEED
    ) == 20_260_729_402
    assert training.steps == 600
    assert training.learning_rate == 1e-3
    assert (
        training.pointwise_weight,
        training.relative_delta_weight,
        training.direction_weight,
        training.midpoint_jvp_weight,
        training.intended_null_weight,
    ) == (1.0, 2.0, 2.0, 1.0, 1.0)
    assert training.fit_data_binding_sha256 == (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    assert training.standardized_gauge_sha256 == (
        "e1bd1659b762476aee4622d3473fa12d6efcca70f7e72bd4b4b45ee86a8413b7"
    )
    assert training.unit_rms_gauge_sha256 == (
        "4a553347335815c56643fdde56c32247e32153ed20e8c713626e35a9a072c312"
    )
    assert state["initialization_schedule"] == (
        "regenerate_d3_outer64_expert_rank64_split_cold_start_per_seed"
    )
    assert state["source_final_provider_initialization_allowed"] is False
    assert state["optimizer"] == "fresh_adam_per_arm"


def test_dormant_child_lift_is_algebraically_function_preserving() -> None:
    lift = DormantChildExpertCountLiftSpec()
    state = lift.state_dict()

    assert lift.source_expert_count == 2
    assert lift.target_expert_count == 4
    assert lift.parent_to_child_rule == (
        "parent_e_maps_to_active_child_e_and_dormant_child_e_plus_2"
    )
    assert lift.child_index_groups == (
        "active_children_0_1_dormant_children_2_3"
    )
    assert lift.router_logit_rule == (
        "copy_parent_router_column_and_bias_to_child_e_and_child_e_plus_2"
    )
    assert lift.router_bias_log2_adjustment == 0.0
    assert lift.active_child_u_scale == 1.0
    assert lift.active_child_v_scale == 2.0
    assert lift.dormant_child_u_scale == 1.0
    assert lift.dormant_child_v_initialization == "exact_zero"
    assert (lift.base_rank, lift.extra_rank, lift.concatenated_rank) == (
        16,
        48,
        64,
    )
    parent_probability = 0.37
    parent_product = 1.9
    active = parent_probability / 2.0 * 2.0 * parent_product
    dormant = parent_probability / 2.0 * 0.0 * parent_product
    assert active + dormant == pytest.approx(
        parent_probability * parent_product
    )
    assert state["observable_initialization"] == "exact_e2_r64_function"
    assert "base16_and_extra48" in state["split_rank_preservation"]


def test_source_expert_rank_receipt_plan_result_and_metrics_are_exact() -> None:
    source = SourceExpertRankResultBindings()
    state = source.state_dict()

    assert source.expert_rank_protocol_sha256 == (
        "94b24068fa583c627faa7d06838c6cd80065f6180c3047ee2923ed95b587014c"
    )
    assert source.expert_rank_code_bundle_sha256 == (
        "bcdd356aea62fedbdffd57bca39e1287f6da1374bb7477a2b28de374c9afebc3"
    )
    assert source.expert_rank_logical_artifact_sha256 == (
        "9759407bf2f2c0a1deb1d29aba7fdbf453bdda8a727aa1e672452a00299a48a9"
    )
    assert source.expert_rank_tensor_file_sha256 == (
        "2139696efebcee68dd379f8226e04cb5edce57c10571f7db5b697700854a2a61"
    )
    assert source.expert_rank_report_sha256 == (
        "f2a8cb19f5aeebf9e5a1b46ac880655611db45138ab9943216ed9d2acea78c7c"
    )
    assert source.expert_rank_outcome == "primary_both_fail"
    assert source.expert_rank_primary_comparison_status == "both_fail"
    assert source.expert_rank_primary_treatment_valid is True
    assert source.expert_rank_replication_executed is False
    assert source.expert_rank_expert_count_control_authorized is True
    assert source.expert_rank_compressed_rank_ladder_authorized is False
    assert source.expert_rank_fresh_c3_authorized is False
    assert source.expert_rank_two_seed_support is False
    assert source.expert_rank_primary_e2r64_plan_sha256 == (
        "57268369d21a66e464f7155a5f9b99868b1240e5f1ca0e3d59b2d40f5d5373de"
    )
    assert source.expert_rank_primary_e2r64_result_sha256 == (
        "2d7492248d279df96b6c6f60049fbc1ed4bd2907f37cd160700ad8d1532a3395"
    )
    assert source.expert_rank_primary_e2r64_initial_metrics_sha256 == (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )
    assert source.expert_rank_primary_e2r64_final_metrics_sha256 == (
        "92e16d27f160144fc24cab76dbcfee58e5c88df143fbb1d7916dc8092a4ac882"
    )
    assert state["source_final_parameter_use"] == "validation_only"
    assert state["source_final_parameter_initialization_allowed"] is False


def test_two_step_fit_only_preflight_is_frozen_for_both_seeds() -> None:
    protocol = default_function_preserving_expert_count_control_protocol()
    preflight = protocol.preflight
    primary = preflight.for_role("primary")
    replication = preflight.for_role("replication")

    assert preflight.bindings_finalized is True
    assert preflight.preflight_steps == 2
    assert preflight.expected_jvp_pair_count == 32
    assert primary["seed"] == 20_260_728_402
    assert replication["seed"] == 20_260_729_402
    assert primary["initialization_audit_sha256"] == (
        "305af575c3fa3579e1573971b949b412e0f8f4626ec6477c794ef146641bc21b"
    )
    assert replication["initialization_audit_sha256"] == (
        "698819152030ff0b33c3a6dab146c9851178691e828edb5976d89d865c374275"
    )
    assert primary["treatment_gradient_audit_sha256"] == (
        "9b9ad88620d86f1062ab28dec15a213e60dda73fefdba0a93f48a12868008c45"
    )
    assert replication["treatment_gradient_audit_sha256"] == (
        "9b1816fc91a33b2e182e49a7df8caf459dcb7c2d560607946944780233360547"
    )
    assert primary["two_step_postfit_parity_sha256"] == (
        "490c1c7aee61c9340aeef85ef9bc61f1a4b40b765e4b7d263a570da108883c1f"
    )
    assert replication["two_step_postfit_parity_sha256"] == (
        "9134c050e07de59ff0a182801c4e6135ddb4976ce1700a0a4a5f92bd1bc97e6e"
    )
    for binding in (primary, replication):
        initial = binding["initial_equivalence"]
        gradient = binding["treatment_gradient"]
        assert len(gradient) == 24
        assert initial["causal_masks_exact"] is True
        assert initial["outside_edge_routes_zero"] is True
        assert all(initial["lift_parameter_flags"].values())
        for name in (
            "step1_dormant_base_input_gradient_norm",
            "step1_dormant_base_input_delta_norm",
            "step1_active_extra_input_gradient_norm",
            "step1_active_extra_input_delta_norm",
            "step1_dormant_extra_input_gradient_norm",
            "step1_dormant_extra_input_delta_norm",
        ):
            assert gradient[name] == 0.0
        assert all(
            value > 1e-12
            for name, value in gradient.items()
            if name
            not in {
                "step1_dormant_base_input_gradient_norm",
                "step1_dormant_base_input_delta_norm",
                "step1_active_extra_input_gradient_norm",
                "step1_active_extra_input_delta_norm",
                "step1_dormant_extra_input_gradient_norm",
                "step1_dormant_extra_input_delta_norm",
            }
        )
    assert protocol.state_dict()["outcome_run_allowed"] is True
    assert "must_replay" in preflight.state_dict()["freeze_policy"]
    with pytest.raises(ValueError, match="pair role"):
        preflight.for_role("other")


def test_exact_capacity_oracle_accounting_is_frozen() -> None:
    state = default_function_preserving_expert_count_control_protocol().state_dict()
    accounting = state["exact_accounting"]

    assert accounting == {
        "e2_executor_parameters": 23106,
        "e4_executor_parameters": 39780,
        "e2_total_stored_scalars": 31492,
        "e4_total_stored_scalars": 48166,
        "e2_canonical_core_macs_length128": 4499008,
        "e4_canonical_core_macs_length128": 7929024,
        "e2_canonical_total_macs_length128": 5555776,
        "e4_canonical_total_macs_length128": 8985792,
        "e2_fit_panel_core_macs": 506306160,
        "e4_fit_panel_core_macs": 897039840,
        "e2_fit_panel_total_macs": 610001520,
        "e4_fit_panel_total_macs": 1000735200,
    }
    assert state["capacity_oracle_not_compression_or_speed_evidence"] is True


def test_decision_ladder_is_narrow_and_preregistered() -> None:
    state = default_function_preserving_expert_count_control_protocol().state_dict()

    assert state["primary_decision"] == (
        "e2_fail_e4_pass_authorizes_paired_replication_only"
    )
    assert state["replication_decision"] == (
        "e2_fail_e4_pass_supports_additional_routed_expert_partitions"
    )
    assert state["both_fail_authority"] == (
        "e4_insufficient_and_e8_full_count_oracle_only"
    )
    assert state["two_seed_expert_count_support_next_rung"] == (
        "separately_preregister_e3_threshold_or_descending_count_ladder_only"
    )
    assert state["c3_allowed"] is False
    assert state["compression_claim_allowed"] is False
    assert state["wall_clock_speed_claim_allowed"] is False


def test_nested_artifact_hashes_and_strict_keys_reject_mutation() -> None:
    protocol = default_function_preserving_expert_count_control_protocol()
    state = protocol.state_dict()

    assert len(protocol.e2_executor.artifact_sha256) == 64
    assert len(protocol.e4_executor.artifact_sha256) == 64
    assert len(protocol.training.artifact_sha256) == 64
    assert len(protocol.lift.artifact_sha256) == 64
    assert len(protocol.preflight.artifact_sha256) == 64
    assert len(protocol.source.artifact_sha256) == 64

    mutated = copy.deepcopy(state)
    mutated["extra"] = True
    with pytest.raises(ValueError, match="keys differ"):
        FunctionPreservingExpertCountControlProtocol.from_state_dict(mutated)

    mutated = copy.deepcopy(state)
    mutated["e4_executor"]["expert_count"] = 8
    with pytest.raises(ValueError):
        FunctionPreservingExpertCountControlProtocol.from_state_dict(mutated)

    mutated = copy.deepcopy(state)
    mutated["source"]["expert_rank_outcome"] = "primary_treatment_pass"
    with pytest.raises(ValueError):
        FunctionPreservingExpertCountControlProtocol.from_state_dict(mutated)


def test_frozen_dataclasses_reject_constructor_drift() -> None:
    with pytest.raises(ValueError):
        ExpertCountExecutorSpec(expert_count=8)
    with pytest.raises(ValueError):
        ExpertCountExecutorSpec(expert_rank=16)
    with pytest.raises(ValueError):
        DormantChildExpertCountLiftSpec(active_child_v_scale=1.0)
    with pytest.raises(ValueError):
        PairedExpertCountTrainingSpec(steps=601)
    with pytest.raises(ValueError):
        SourceExpertRankResultBindings(
            expert_rank_primary_treatment_valid=False
        )

    protocol = default_function_preserving_expert_count_control_protocol()
    with pytest.raises(ValueError):
        replace(protocol, execution_device="mps")


def test_fit_replay_sequence_digest_is_order_sensitive_and_strict() -> None:
    a = "a" * 64
    b = "b" * 64

    assert fit_replay_sequence_sha256("fit", (a, b)) != (
        fit_replay_sequence_sha256("fit", (b, a))
    )
    with pytest.raises(ValueError):
        fit_replay_sequence_sha256("", (a,))
    with pytest.raises(ValueError):
        fit_replay_sequence_sha256("fit", (a.upper(),))
    with pytest.raises(ValueError):
        fit_replay_sequence_sha256("fit", [])
