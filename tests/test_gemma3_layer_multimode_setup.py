from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fisher_graph import gemma3_layer_multimode_setup as setup


def test_setup_freezes_exact_local_flow_math_and_replacement_boundary() -> None:
    state = setup.build_setup()
    boundary = state["replacement_boundary"]
    assert boundary["layer_count"] == 18
    assert boundary["only_token_local_MLP_residual_maps_are_replaced"]
    assert boundary["exact_attention_retained"]
    assert boundary["exact_lexical_ingress_and_egress_retained"]
    assert not boundary["all_layer_RMSNorms_retained"]
    assert boundary["only_MLP_input_RMSNorm_is_retained"]
    assert boundary["native_output_RMSNorm_is_folded_into_v_T_and_removed"]
    assert boundary[
        "candidate_output_RMSNorm_or_double_normalization_is_forbidden"
    ]
    assert boundary["compiled_layer_output"] == "h_prime=h+v_hat(x)"
    assert boundary["x_h_and_v_T_are_three_distinct_runtime_sites"]
    assert boundary["source_free_runtime_reads_x_and_h_but_not_v_T"]

    abi = state["local_flow_mathematical_ABI"]
    atlas = abi["per_layer_atlas"]
    modes = abi["within_chart_modes"]
    assert atlas["tangent_coordinate"] == "z_i=B_i^T*(x-mu_i)"
    assert atlas["normal_residual"] == (
        "rho_i_squared=||(I-B_i*B_i^T)*(x-mu_i)||_2^2"
    )
    assert atlas["chart_partition"] == (
        "q_i=softmax_i(-rho_i_squared/tau_c)"
    )
    assert atlas["C1_boundary"] == {
        "q_is_exactly": [1.0],
        "tau_c_is_fixed_to": 1.0,
        "serialized_log_tau_c_is_fixed_to": 0.0,
        "one_temperature_scalar_is_still_serialized_and_counted": True,
    }
    assert atlas["C2_temperature_fit"] == (
        "one_positive_tau_c_fitted_on_A_fit_rows_only"
    )
    assert modes["affine_router"] == "a_ik(z_i)=w_ik^T*z_i+b_ik"
    assert modes["mode_partition"] == "pi_ik=softmax_k(a_ik(z_i))"
    assert modes["affine_tangent_velocity"] == "t_ik(z_i)=A_ik*z_i+c_ik"
    assert modes["local_generator"] == "G_ik=B_i*t_ik(z_i)"
    assert abi["compiled_velocity"] == (
        "v_hat=sum_i(q_i*sum_k(pi_ik*G_ik))"
    )
    assert abi["compiled_fused_velocity"] == {
        "within_chart_tangent_velocity": "u_i=sum_k(pi_ik*t_ik(z_i))",
        "single_decode_per_chart": "v_hat=sum_i(q_i*B_i*u_i)",
        "per_mode_H640_decode_or_mixture_is_forbidden": True,
        "mathematically_equal_to_unfused_generator_sum": True,
    }
    assert abi["fit_and_selection_reductions"] == "CPU_float64_fixed_order"
    assert abi["deployed_coefficient_dtype"] == "float32"


def test_setup_freezes_complete_18_candidate_grid_per_layer() -> None:
    contract = setup.build_setup()["candidate_grid_and_accounting"]
    assert contract["chart_count_candidates_in_declared_order"] == [2, 1]
    assert contract["modes_per_chart_candidates_in_declared_order"] == [4, 2, 1]
    assert contract["tangent_rank_candidates_in_declared_order"] == [64, 32, 16]
    assert contract["candidate_count_per_layer"] == 18
    assert contract["complete_C_times_M_times_R_grid_required_per_layer"]
    table = contract["candidate_table"]
    assert len(table) == 18
    assert len(
        {
            (row["chart_count"], row["modes_per_chart"], row["tangent_rank"])
            for row in table
        }
    ) == 18
    assert table[0]["candidate_ordinal"] == 0
    assert (
        table[0]["chart_count"],
        table[0]["modes_per_chart"],
        table[0]["tangent_rank"],
    ) == (2, 4, 64)
    assert (
        table[-1]["chart_count"],
        table[-1]["modes_per_chart"],
        table[-1]["tangent_rank"],
    ) == (1, 1, 16)
    assert contract["MAC_formula"] == "2*C*H*R+C*M*R+C*M*R*R"
    assert contract[
        "normal_residual_matrix_reconstruction_is_eliminated_using_"
        "orthonormal_Pythagorean_identity"
    ]
    assert contract["within_chart_tangent_mixture_precedes_single_basis_decode"]


def test_parameter_accounting_is_exact_and_type_sensitive() -> None:
    smallest = setup.replacement_parameter_accounting(1, 1, 16)
    assert smallest == {
        "retained_rmsnorm_parameter_count": 640,
        "source_mlp_rmsnorm_parameter_count": 1_280,
        "source_mlp_matrix_parameter_count": 3_932_160,
        "source_total_parameter_count": 3_933_440,
        "chart_center_parameter_count": 640,
        "chart_basis_parameter_count": 10_240,
        "fixed_atlas_parameter_count": 10_880,
        "chart_temperature_parameter_count": 1,
        "router_parameter_count": 17,
        "local_generator_parameter_count": 272,
        "candidate_total_parameter_count": 11_810,
        "float32_coefficient_payload_bytes": 47_240,
        "net_parameter_savings": 3_921_630,
        "conservative_parameter_savings_vs_source_MLP_matrices_only": 3_920_350,
        "source_matrix_macs_per_token": 3_932_160,
        "chart_projection_macs_per_token": 10_240,
        "fused_chart_velocity_decode_macs_per_token": 10_240,
        "router_macs_per_token": 16,
        "local_generator_macs_per_token": 256,
        "candidate_matrix_and_mixture_macs_per_token": 20_752,
        "net_macs_per_token": 3_911_408,
    }
    largest = setup.replacement_parameter_accounting(2, 4, 64)
    assert largest["candidate_total_parameter_count"] == 117_641
    assert largest["float32_coefficient_payload_bytes"] == 470_564
    assert setup.NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER == 3_932_160
    with pytest.raises(ValueError, match="chart_count"):
        setup.replacement_parameter_accounting(True, 1, 16)
    with pytest.raises(ValueError, match="mode_count"):
        setup.replacement_parameter_accounting(1, 3, 16)
    with pytest.raises(ValueError, match="tangent_rank"):
        setup.replacement_parameter_accounting(1, 1, 8)


def test_A_selection_freezes_fidelity_distinctness_and_minimum_size_rule() -> None:
    selection = setup.build_setup()["A_development_selection"]
    assert selection["only_old_v2_A_may_be_used"]
    assert selection["v2_B_C_D_and_reserve_are_forbidden"]
    assert selection["per_family_fit_target_token_lengths"] == [96, 512, 514, 640]
    assert selection["per_family_validation_target_token_lengths"] == [
        511,
        513,
        576,
        768,
    ]
    assert selection["base_validation_gates"] == {
        "all_values_finite": True,
        "velocity_NRMSE_at_most": 0.25,
        "aggregate_velocity_cosine_at_least": 0.95,
        "every_family_velocity_NRMSE_at_most": 0.35,
    }
    gates = selection["multimode_only_gates"]
    assert gates[
        "validation_SSE_improvement_over_same_layer_C_R_M1_at_least"
    ] == pytest.approx(0.05)
    assert selection["multichart_only_gates"] == {
        "applies_exactly_when_C_greater_than_1": True,
        "each_chart_soft_mass_at_least": 0.05,
        "chart_mass_is_observed_not_balanced_or_hardened": True,
    }
    assert gates["each_chart_conditional_mode_mass_at_least"] == pytest.approx(
        0.05
    )
    assert gates[
        "each_chart_state_mode_mutual_information_nats_at_least"
    ] == pytest.approx(0.05)
    assert gates[
        "minimum_within_each_chart_pairwise_velocity_separation_ratio_at_least"
    ] == 0.05
    assert gates["minimum_split_matched_prediction_cosine_at_least"] == 0.8
    assert gates[
        "pooled_split_hard_route_adjusted_rand_index_at_least"
    ] == 0.6
    stats = selection["multimode_statistic_definitions"]
    assert stats["chart_conditional_state_mode_MI_nats"].startswith("H(U_i)")
    assert "RMS(G_im-G_ik)" in stats["within_chart_velocity_separation"]
    assert stats["M1_boundary_values"]["multimode_thresholds_bypassed"]
    capture = selection["production_capture_and_partition"]
    assert capture["complete_role_capture_required"] == {
        "batch_count": 45,
        "example_count": 48,
        "row_count_per_mode": 24732,
    }
    assert capture["fit_row_count"] == 10548
    assert capture["validation_row_count"] == 14184
    assert capture["fit_validation_masks_disjoint_and_exhaustive"]
    assert capture["source_capture_scope"] == (
        "production_full_A_RoleCaptureReceipt_spools"
    )
    materialization = capture["materialization_inventory"]
    assert materialization["receipt_type"] == (
        "LayerMLPFlowMaterializationInventoryReceipt"
    )
    assert materialization["live_opener_created_spool_input_count"] == 18
    assert materialization[
        "all_spool_inputs_must_be_live_and_opener_created"
    ]
    assert materialization["A_partition_is_rebuilt_from_live_scalar_axes"]
    assert materialization["parser_only_receipt_cannot_authorize_compile"]
    assert materialization[
        "selected_fit_source_row_receipts_must_match_inventory"
    ]
    assert capture[
        "same_partition_receipt_and_derived_fit_validation_digests_"
        "required_for_all_324_candidates"
    ]
    assert capture[
        "production_compile_requires_original_live_fitter_authority_for_"
        "every_selected_candidate"
    ]
    assert capture[
        "production_M_greater_than_1_requires_original_live_split_evaluator_"
        "authority"
    ]
    assert capture[
        "capture_materialization_atlas_fit_split_and_inventory_direct_"
        "construction_copy_deepcopy_replace_or_parser_replay_cannot_"
        "recover_live_authority"
    ]
    fit = selection["deterministic_fit_protocol"]
    assert fit["device_and_dtype"] == "CPU_float64_only"
    assert fit["public_atlas_and_candidate_subset_hash_role"] == (
        "A_full_fit_atlas_and_optimizer"
    )
    assert fit["supplied_atlas_requires_original_live_atlas_fitter_authority"]
    assert fit["one_original_live_atlas_is_reusable_across_M_1_2_4"]
    assert fit["C2_lloyd_pass_count"] == 12
    assert fit["optimizer_steps"] == 256
    assert fit["minibatch_size"] == 512
    assert fit["optimizer_id"] == (
        "torch_adam_cpu_float64_fixed_minibatch_separate_tau_clip.v2"
    )
    assert fit["fit_policy_sha256"] == (
        "be4908d50da4a0924484b192baac82e453e39ed316854e749e9a867252eae636"
    )
    assert fit["gradient_clip_grouping_id"] == (
        "non_tau_global_l2_and_chart_temperature_scalar_independent.v1"
    )
    assert fit["non_tau_parameters_share_one_global_L2_clip"]
    assert fit["C2_chart_temperature_uses_an_independent_scalar_clip"]
    assert fit["C1_chart_temperature_is_frozen_and_absent_from_optimizer"]
    assert fit["mode_regularizers_use_q_detached_and_cannot_fit_tau"]
    assert fit["only_teacher_velocity_fidelity_may_fit_C2_tau"]
    assert fit["regularizer_gradient_magnitude_cannot_rescale_tau_update"]
    split_closure = selection["multimode_statistic_definitions"][
        "split_refit_scalar_closure"
    ]
    assert split_closure["base_fit_row_count"] == 10548
    assert split_closure["partition_row_counts"] == [5274, 5274]
    assert split_closure["intersection_row_count"] == 0
    assert split_closure["union_equals_base_fit"]
    assert split_closure[
        "strict_production_replay_regenerates_checkerboard_masks_from_shared_"
        "A_partition_manifest"
    ]
    assert selection[
        "all_M_candidates_at_fixed_layer_C_R_share_one_exact_state_atlas"
    ]
    assert selection[
        "production_selector_requires_all_324_original_live_adapter_"
        "candidate_evaluations"
    ]
    assert selection[
        "production_compile_requires_original_selector_result_over_all_"
        "324_live_candidates"
    ]
    assert selection[
        "coordinated_306_failure_relabels_around_18_live_winners_are_rejected"
    ]
    assert selection[
        "candidate_evaluation_and_selection_direct_construction_copy_deepcopy_"
        "replace_or_parser_replay_cannot_recover_live_authority"
    ]
    assert selection["selection_tie_key"] == [
        "serialized_parameter_count",
        "validation_velocity_NRMSE",
        "chart_count_numeric_ascending",
        "mode_count_numeric_ascending",
        "tangent_rank_numeric_ascending",
    ]
    assert selection["one_candidate_must_select_for_each_of_18_layers_before_B3"]

    state = setup.build_setup()
    b3 = state["B3_teacher_forced_layerwise"]
    c3 = state["C3_full_stack_source_free"]
    assert b3["all_18_layers_must_pass_before_C3"]
    assert c3["strict_typed_passing_B3_receipt_required"]
    assert c3["B3_C3_source_panel_protocol_and_selection_must_match"]
    assert (
        b3["heldout_evaluation_policy_sha256"]
        == c3["heldout_evaluation_policy_sha256"]
        == "4a6f6379571f348fc7a0a477c83ffdf52ae119903cb6e8859a12c4f66e4e9266"
    )


def test_fresh_roles_are_strictly_staged_and_old_v2_roles_are_not_reused() -> None:
    firewall = setup.build_setup()["fresh_panel_firewall"]
    assert firewall["ordered_roles"] == [
        setup.FIT_ROLE,
        setup.DISCOVERY_ROLE,
        setup.CONFIRMATION_ROLE,
        setup.FINAL_ROLE,
        setup.RESERVE_ROLE,
    ]
    assert firewall[
        "B3_C3_D3_and_reserve_must_be_new_and_frozen_before_preflight"
    ]
    assert firewall[
        "B3_C3_D3_and_reserve_must_be_disjoint_from_every_v1_v2_corpus"
    ]
    assert firewall[
        "B3_C3_D3_and_reserve_must_be_pairwise_semantic_lineage_disjoint"
    ]
    assert firewall["existing_v2_D_is_not_repurposed"]
    assert firewall["existing_v2_reserve_is_not_repurposed_or_opened"]
    assert firewall["reserve_remains_semantically_unopened"]
    assert firewall["later_role_early_access_invalidates_the_run"]


def test_B3_C3_and_D3_gates_are_exact_and_nonadaptive() -> None:
    state = setup.build_setup()
    B = state["B3_teacher_forced_layerwise"]
    assert B["evaluation_is_layerwise_on_native_teacher_forced_inputs"]
    assert B["fidelity_accumulation"].startswith("unweighted_complete_H640")
    assert B["every_layer_velocity_NRMSE_at_most"] == 0.30
    assert B["every_layer_aggregate_velocity_cosine_at_least"] == 0.90
    assert B["every_layer_every_family_velocity_NRMSE_at_most"] == 0.40
    assert B["selected_M_greater_than_1_additional_gates"] == {
        "each_chart_conditional_mode_mass_at_least": 0.03,
        "each_chart_state_mode_mutual_information_nats_at_least": 0.03,
    }
    assert B["B3_may_not_refit_reselect_or_change_thresholds"]

    C = state["C3_full_stack_source_free"]
    assert C["all_18_selected_MLP_replacements_execute_together"]
    assert C["no_native_MLP_teacher_inputs_or_intermediate_fallback"]
    assert C["native_and_candidate_scored_positions_must_be_identical"]
    assert C["pooled_point_gates"] == {
        "delta_NLL_nats_per_token_at_most": 0.02,
        "native_to_candidate_KL_per_token_at_most": 0.02,
        "native_top1_agreement_at_least": 0.95,
    }
    assert C["every_family_point_gates"] == {
        "delta_NLL_nats_per_token_at_most": 0.04,
        "native_to_candidate_KL_per_token_at_most": 0.04,
        "native_top1_agreement_at_least": 0.90,
    }
    assert C["hard_long_strata_repeat_pooled_point_gates"]

    D = state["D3_final_validation"]
    assert D["confidence_unit"] == "whole_example"
    assert D["confidence_level"] == 0.95
    assert D["bootstrap_is_approximate_not_distribution_free"]
    assert D["bootstrap_replicate_count"] == 10000
    assert D["bootstrap_stratification"] == (
        "all_12_nonempty_family_id_times_hard_long_cells"
    )
    assert D["bootstrap_policy_sha256"] == (
        "fdfc8be6463e959f02cc17ca5b17168405d3bca987d57e9a9ed3d621063cd026"
    )
    assert D["one_sided_pooled_bounds"] == {
        "delta_NLL_nats_per_token_upper_bound_at_most": 0.02,
        "native_to_candidate_KL_per_token_upper_bound_at_most": 0.02,
        "native_top1_agreement_lower_bound_at_least": 0.95,
    }
    assert D["native_MLP_source_fallback_alias_or_hidden_call_forbidden"]
    assert D["exact_formula_parameter_count_must_be_lower"]
    assert D["actual_serialized_full_model_bytes_must_be_lower"]


def test_v2_negative_result_is_bound_but_non_authorizing() -> None:
    provenance = setup.build_setup()["methodology_provenance"]
    assert provenance["predecessor_v2_setup_sha256"] == (
        setup.PREDECESSOR_V2_SETUP_SHA256
    )
    assert provenance["predecessor_v2_report_sha256"] == (
        setup.PREDECESSOR_V2_REPORT_SHA256
    )
    assert provenance["predecessor_v2_report_file_sha256"] == (
        setup.PREDECESSOR_V2_REPORT_FILE_SHA256
    )
    assert provenance["predecessor_v2_selected_chart_count"] == 2
    assert provenance["predecessor_v2_B_raw_mutual_pair_count"] == 5
    assert provenance["predecessor_v2_B_frozen_candidate_count"] == 0
    assert provenance["predecessor_v2_C_confirmed_edge_count"] == 0
    assert provenance["predecessor_is_completed_negative_evidence"]
    assert provenance["predecessor_cannot_authorize_v3_selection_or_claims"]


def test_setup_binds_production_execution_fingerprint_checks() -> None:
    compiler = setup.build_setup()["production_compiler_and_bundle_loader"]
    assert compiler["production_compile_recomputes_source_model_fingerprint"]
    assert compiler[
        "production_compile_recomputes_source_execution_fingerprint"
    ]
    assert compiler[
        "production_compile_requires_entire_source_model_in_eval_mode"
    ]
    assert compiler[
        "bundle_loader_recomputes_compiled_execution_fingerprint"
    ]


def test_prospective_check_passes_but_preflight_and_run_fail_closed(tmp_path) -> None:
    output = tmp_path / "setup.json"
    expected = setup.write_setup(output)
    assert setup.check_setup(output, verify_predecessor=False) == expected
    readiness = expected["implementation_readiness"]
    assert readiness["default_setup_artifact_published"] is False
    assert readiness["panel_manifest_frozen"] is False
    assert readiness["implementation_qualification_frozen"] is False
    assert readiness["preflight_authorized"] is False
    assert readiness["run_authorized"] is False
    with pytest.raises(setup.ExecutionNotAuthorizedError, match="not authorized"):
        setup.preflight_setup(output, verify_predecessor=False)
    with pytest.raises(setup.ExecutionNotAuthorizedError, match="not authorized"):
        setup.authorize_run(output, verify_predecessor=False)


def test_setup_validation_rejects_drift_and_bool_integer_substitution() -> None:
    drifted = copy.deepcopy(setup.build_setup())
    drifted["candidate_grid_and_accounting"][
        "chart_count_candidates_in_declared_order"
    ] = [1, 2]
    drifted["setup_sha256"] = setup.setup_sha256(drifted)
    with pytest.raises(ValueError, match="differs from source"):
        setup.validate_setup(drifted)

    bool_version = copy.deepcopy(setup.build_setup())
    bool_version["format_version"] = True
    bool_version["setup_sha256"] = setup.setup_sha256(bool_version)
    with pytest.raises(ValueError, match="differs from source"):
        setup.validate_setup(bool_version)


def test_setup_roundtrip_is_exact_and_no_clobber(tmp_path) -> None:
    output = tmp_path / "setup.json"
    expected = setup.write_setup(output)
    assert output.read_bytes() == setup.render_setup_bytes()
    assert setup.load_setup(output) == expected
    with pytest.raises(FileExistsError):
        setup.write_setup(output)

    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(ValueError, match="formatting drifted"):
        setup.check_setup(reformatted, verify_predecessor=False)


def test_strict_loader_rejects_duplicate_keys_and_symlinks(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON is invalid"):
        setup.load_setup(duplicate)

    target = tmp_path / "target.json"
    target.write_bytes(setup.render_setup_bytes())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="not a regular file"):
        setup.load_setup(link)


def test_setup_dataclass_and_cli_require_explicit_action() -> None:
    wrapped = setup.Gemma3LayerMultimodeSetup.build()
    assert wrapped.to_dict() == setup.build_setup()
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 2
