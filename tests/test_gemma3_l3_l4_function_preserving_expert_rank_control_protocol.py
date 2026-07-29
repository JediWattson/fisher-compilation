from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

import pytest

from fisher_graph.gemma3_l3_l4_function_preserving_expert_rank_control_protocol import (
    DEFAULT_FUNCTION_PRESERVING_EXPERT_RANK_CONTROL_PROTOCOL_SHA256,
    EXPERT_RANK_CONTROL_EXECUTION_DEVICE,
    EXPERT_RANK_CONTROL_EXECUTION_DTYPE,
    EXPERT_RANK_CONTROL_OUTER_RANK,
    EXPERT_RANK_CONTROL_PRIMARY_SEED,
    EXPERT_RANK_CONTROL_REPLICATION_SEED,
    EXPERT_RANK_CONTROL_SOURCE_RANK,
    EXPERT_RANK_CONTROL_TARGET_RANK,
    ExpertRankExecutorSpec,
    ExpertRankPreflightBindings,
    FunctionPreservingExpertRankControlProtocol,
    NestedExpertRankLiftSpec,
    PairedExpertRankTrainingSpec,
    SourceWidthResultBindings,
    default_function_preserving_expert_rank_control_protocol,
)


def test_protocol_literal_trust_anchor_and_strict_round_trip() -> None:
    protocol = default_function_preserving_expert_rank_control_protocol()

    assert protocol.protocol_sha256 == (
        DEFAULT_FUNCTION_PRESERVING_EXPERT_RANK_CONTROL_PROTOCOL_SHA256
    )
    assert protocol.protocol_sha256 == (
        "94b24068fa583c627faa7d06838c6cd80065f6180c3047ee2923ed95b587014c"
    )
    assert (
        FunctionPreservingExpertRankControlProtocol.from_state_dict(
            protocol.state_dict()
        )
        == protocol
    )


def test_protocol_module_imports_only_standard_library() -> None:
    path = Path(
        "src/fisher_graph/"
        "gemma3_l3_l4_function_preserving_expert_rank_control_protocol.py"
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


def test_fixed_outer64_geometry_changes_only_expert_rank() -> None:
    protocol = default_function_preserving_expert_rank_control_protocol()
    state = protocol.state_dict()

    assert EXPERT_RANK_CONTROL_OUTER_RANK == 64
    assert EXPERT_RANK_CONTROL_SOURCE_RANK == 16
    assert EXPERT_RANK_CONTROL_TARGET_RANK == 64
    assert protocol.expert16_executor == ExpertRankExecutorSpec(
        expert_rank=16
    )
    assert protocol.expert64_executor == ExpertRankExecutorSpec(
        expert_rank=64
    )
    for executor in (
        protocol.expert16_executor,
        protocol.expert64_executor,
    ):
        assert executor.input_modes == 66
        assert executor.output_modes == 64
        assert executor.expert_count == 2
        assert executor.router_width == 16
        assert executor.same_position_skip is False
        assert executor.max_positive_lag is None
        assert executor.router_activation == "tanh"
        assert executor.source_normalized_routing is True
    assert state["controlled_change"] == (
        "gradient_open_nested_expert_rank_16_to_64_at_fixed_outer64_"
        "and_exact_initial_observable_and_jvp"
    )
    assert "outer_rank" in state["unchanged_contract"]


def test_training_freezes_exact_d3_objective_data_gauge_and_gates() -> None:
    training = PairedExpertRankTrainingSpec()
    state = training.state_dict()

    assert training.primary_seed == (
        EXPERT_RANK_CONTROL_PRIMARY_SEED
    ) == 20_260_728_402
    assert training.replication_seed == (
        EXPERT_RANK_CONTROL_REPLICATION_SEED
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
    assert training.ordinary_gates_sha256 == (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    assert training.contrast_gates_sha256 == (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )
    assert state["fit_materialization_roles"] == ["pilot", "fit"]
    assert state["canonical_scoring_semantics"] == (
        "raw_canonical_fisher_metric_for_all_fit_scoring"
    )


def test_lift_is_balanced_signed_identity_behind_zero_v() -> None:
    lift = NestedExpertRankLiftSpec()
    state = lift.state_dict()

    assert lift.source_rank == 16
    assert lift.target_rank == 64
    assert lift.added_rank == lift.added_coordinate_count == 48
    assert lift.coordinate_indexing == "zero_based"
    assert lift.added_coordinate_rule == (
        "for_j_0_through_47_u_row_17_plus_j_column_16_plus_j"
    )
    assert lift.expert0_added_u_value == 1.0
    assert lift.expert1_added_u_value == -1.0
    assert lift.added_v_initialization == "exact_zero"
    assert lift.equivalence_absolute_tolerance == 1e-12
    assert lift.equivalence_relative_tolerance == 1e-12
    assert lift.gradient_norm_floor == 1e-12
    assert state["u_tensor_semantics"] == (
        "expert_input_weight_expert_input_mode_expert_rank"
    )
    assert state["v_tensor_semantics"] == (
        "expert_output_weight_expert_rank_output_mode"
    )
    assert state["gradient_open_rule"] == (
        "step1_added_v_gradient_above_floor_added_u_gradient_exact_"
        "zero_then_step2_added_u_gradient_and_u_v_parameter_deltas_"
        "above_floor"
    )


def test_width_result_receipt_outcome_plan_and_metrics_are_exact() -> None:
    source = SourceWidthResultBindings()
    state = source.state_dict()

    assert source.width_protocol_sha256 == (
        "c3ad81c84d41108839b5fcab13e3b5d47d99a55ae9a9223c3f116edb6b457597"
    )
    assert source.width_code_bundle_sha256 == (
        "5c314fff7959f659257911ca0190605ea4ef41c556bd18a27108acb48d2545a4"
    )
    assert source.width_logical_artifact_sha256 == (
        "9e07c7208b3b690a8024bd809a0d80c2842145cfa73e655bb737e5497913ce47"
    )
    assert source.width_tensor_file_sha256 == (
        "5a3c8de7bd6731a78904a14c488648f6641d6b3cbe96167438f633b65f9104c5"
    )
    assert source.width_report_sha256 == (
        "6aacad6f05e3b43bbeba62b6ce7ae35897af6af60d53f9dfa96eec951ad6965f"
    )
    assert source.width_outcome == "primary_both_fail"
    assert source.width_primary_comparison_status == "both_fail"
    assert source.width_primary_treatment_valid is True
    assert source.width_expert_core_control_authorized is True
    assert source.width_primary_rank64_plan_sha256 == (
        "b5de47f2fb89e0198a38d851649e566eaa65a884e2be0eb5201528078d58383b"
    )
    assert source.width_primary_rank64_result_sha256 == (
        "7e3b675cdd031d5c09b8cbbd0b72506cf2ac79254cb3c57f21e56147c203f59a"
    )
    assert source.width_primary_rank64_initial_metrics_sha256 == (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )
    assert source.width_primary_rank64_final_metrics_sha256 == (
        "a2d1581a38f0f5c4cc1b66c363642e93fce5ba205b88d55670bb351d0e35d1e1"
    )
    assert state["source_final_parameter_use"] == "validation_only"
    assert state["source_final_parameter_initialization_allowed"] is False


def test_real_two_step_preflight_is_frozen_for_both_seed_roles() -> None:
    preflight = ExpertRankPreflightBindings()
    primary = preflight.for_role("primary")
    replication = preflight.for_role("replication")

    assert primary["seed"] == 20_260_728_402
    assert replication["seed"] == 20_260_729_402
    assert primary["initialization_audit_sha256"] == (
        "1e0581479ad9b78339c616fa60ef882381698426647bb7e15878d15e8668cae4"
    )
    assert replication["initialization_audit_sha256"] == (
        "b738c7c98120817a6dc796ad6ed7e7d3473c551234d7881b6bcd504cf8fe9206"
    )
    assert primary["control_initial_executor_sha256"] == (
        "916a22f45e0f6a34213261e361eb060a0eed094cce56bd2a795283379f59434d"
    )
    assert primary["treatment_initial_executor_sha256"] == (
        "eb0493e6f7c20cae8c6a8caa3bc80d30709d04e9a7c4da6380f4c97f973a6e63"
    )
    primary_gradient = primary["treatment_gradient"]
    replication_gradient = replication["treatment_gradient"]
    assert isinstance(primary_gradient, dict)
    assert isinstance(replication_gradient, dict)
    assert primary_gradient["step1_extra_input_gradient_norm"] == 0.0
    assert primary_gradient["step1_extra_output_gradient_norm"] > 1e-12
    assert primary_gradient["step2_extra_input_gradient_norm"] > 1e-12
    assert primary_gradient["step2_extra_output_gradient_norm"] > 1e-12
    assert replication_gradient["step1_extra_input_gradient_norm"] == 0.0
    assert replication_gradient["step1_extra_output_gradient_norm"] > 1e-12
    assert replication_gradient["step2_extra_input_gradient_norm"] > 1e-12
    assert replication_gradient["step2_extra_output_gradient_norm"] > 1e-12
    assert primary["two_step_postfit_parity_sha256"] == (
        "8d97d6af566afdd8d3910fa9d1b98c4ae35c2576e81503c1de35c84d842bceb9"
    )
    assert replication["two_step_postfit_parity_sha256"] == (
        "3cd6fe3db3ad3c3443fe0c3867b04f7ebcc3ea87de2e7352010cda22be8d50ec"
    )
    assert primary["control_two_step"] == {
        "metrics_sha256": (
            "c33c956512f103f38789afd6b5118121adeb44a531f55ba52475238a7aef41a0"
        ),
        "executor_sha256": (
            "693e989bbc44b69fbb1c2a4bdc3795e6b0f55efc147d338534ddd59161d56d2b"
        ),
    }
    assert replication["control_two_step"] == {
        "metrics_sha256": (
            "cec05fc379a63d38d19a1d12979a6264b665594d92a4c97eb29b68d96390da5e"
        ),
        "executor_sha256": (
            "84df9102b6aa7a5c80a5321ad57eca8731fe56610a798f754cd1004722f5d468"
        ),
    }
    for role in (primary, replication):
        parity = role["two_step_postfit_parity"]
        assert isinstance(parity, dict)
        assert parity["maximum_output_absolute_error"] <= 1e-12
        assert parity["maximum_jvp_absolute_error"] <= 1e-12
        assert parity["weighted_total_absolute_error"] == 0.0
    with pytest.raises(ValueError, match="pair role"):
        preflight.for_role("other")


def test_cold_initialization_never_warm_starts_from_source_final() -> None:
    protocol = default_function_preserving_expert_rank_control_protocol()
    training = protocol.training.state_dict()
    state = protocol.state_dict()

    assert training["initialization_schedule"] == (
        "regenerate_width_control_initial_outer64_lift_per_seed"
    )
    assert training["source_final_provider_initialization_allowed"] is False
    assert training["primary_schedule"] == (
        "expert16_width_primary_replay_then_expert64_nested_lift"
    )
    assert state["source_final_parameters_used_for_initialization"] is False


def test_authority_is_fit_only_and_decision_contingent() -> None:
    state = (
        default_function_preserving_expert_rank_control_protocol().state_dict()
    )

    assert state["scientific_scope"] == (
        "fit_only_paired_function_preserving_expert_rank_control"
    )
    assert state["primary_decision"] == (
        "expert16_fail_expert64_pass_authorizes_paired_replication_only"
    )
    assert state["replication_decision"] == (
        "expert16_fail_expert64_pass_supports_inner_expert_rank_attribution"
    )
    assert state["both_fail_authority"] == (
        "expert_rank_alone_insufficient_under_matched_fit_budget"
    )
    assert state["valid_primary_both_fail_next_rung"] == (
        "separately_preregister_fixed_outer64_matched_expert_count_"
        "control_only"
    )
    assert state["two_seed_expert_rank_support_next_rung"] == (
        "separately_preregister_descending_expert_rank_ladder_only"
    )
    assert state["fresh_c3_authority"] == (
        "not_authorized_by_this_control_only_after_successful_"
        "descending_expert_rank_ladder"
    )
    assert state["result_authentication"] == (
        "mandatory_external_logical_tensor_report_sha256_triple"
    )
    assert (
        state["self_hashes_authoritative_without_external_receipt"] is False
    )
    for name in (
        "c2_selection_allowed",
        "c2_provider_artifact_loading_allowed",
        "c3_allowed",
        "compression_claim_allowed",
        "held_out_generalization_claim_allowed",
        "natural_prompt_fidelity_claim_allowed",
        "full_model_replacement_claim_allowed",
        "wall_clock_speed_claim_allowed",
    ):
        assert state[name] is False
    assert state["expert_count_change_mixed_into_primary"] is False
    assert state["outer_width_change_mixed_into_primary"] is False
    assert state["longer_optimization_mixed_into_primary"] is False


def test_nested_declarations_round_trip_strictly() -> None:
    protocol = default_function_preserving_expert_rank_control_protocol()

    assert ExpertRankExecutorSpec.from_state_dict(
        protocol.expert16_executor.state_dict()
    ) == protocol.expert16_executor
    assert ExpertRankExecutorSpec.from_state_dict(
        protocol.expert64_executor.state_dict()
    ) == protocol.expert64_executor
    assert PairedExpertRankTrainingSpec.from_state_dict(
        protocol.training.state_dict()
    ) == protocol.training
    assert NestedExpertRankLiftSpec.from_state_dict(
        protocol.lift.state_dict()
    ) == protocol.lift
    assert ExpertRankPreflightBindings.from_state_dict(
        protocol.preflight.state_dict()
    ) == protocol.preflight
    assert SourceWidthResultBindings.from_state_dict(
        protocol.source.state_dict()
    ) == protocol.source


@pytest.mark.parametrize(
    ("factory", "change"),
    [
        (ExpertRankExecutorSpec, {"router_width": 32}),
        (PairedExpertRankTrainingSpec, {"steps": 601}),
        (NestedExpertRankLiftSpec, {"expert1_added_u_value": 1.0}),
        (
            ExpertRankPreflightBindings,
            {"primary_step1_u_gradient_norm": 1e-6},
        ),
        (
            SourceWidthResultBindings,
            {"width_outcome": "primary_rank16_fail_rank64_pass"},
        ),
    ],
)
def test_nested_declarations_reject_scientific_drift(
    factory: object,
    change: dict[str, object],
) -> None:
    value = factory()  # type: ignore[operator]

    with pytest.raises(ValueError):
        replace(value, **change)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("artifact_sha256",), "0" * 64),
        (("outer_rank",), 16),
        (("execution_device",), "mps"),
        (("expert16_executor", "expert_rank"), 64),
        (("expert64_executor", "router_width"), 32),
        (("training", "learning_rate"), 2e-3),
        (("lift", "added_coordinate_rule"), "dense_random"),
        (("preflight", "primary_gradient_audit_sha256"), "2" * 64),
        (("source", "width_outcome"), "primary_rank16_fail_rank64_pass"),
        (("source", "width_primary_rank64_result_sha256"), "1" * 64),
        (("compression_claim_allowed",), True),
        (("source_final_parameters_used_for_initialization",), True),
    ],
)
def test_protocol_rejects_semantic_or_hash_tampering(
    path: tuple[object, ...],
    value: object,
) -> None:
    state = copy.deepcopy(
        default_function_preserving_expert_rank_control_protocol().state_dict()
    )
    target: object = state
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        FunctionPreservingExpertRankControlProtocol.from_state_dict(state)


def test_protocol_rejects_extra_or_missing_keys() -> None:
    state = (
        default_function_preserving_expert_rank_control_protocol().state_dict()
    )
    state["unexpected"] = True
    with pytest.raises(ValueError, match="keys differ"):
        FunctionPreservingExpertRankControlProtocol.from_state_dict(state)

    state = (
        default_function_preserving_expert_rank_control_protocol().state_dict()
    )
    del state["primary_decision"]
    with pytest.raises(ValueError, match="keys differ"):
        FunctionPreservingExpertRankControlProtocol.from_state_dict(state)
