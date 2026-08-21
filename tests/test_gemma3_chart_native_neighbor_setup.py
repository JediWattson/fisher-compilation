from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_chart_native_neighbor_setup import (
    BOOTSTRAP_CATALOG_DOMAIN,
    BOOTSTRAP_DRAW_DOMAIN,
    BOOTSTRAP_DRAW_PROTOCOL,
    BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION,
    BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY,
    BOOTSTRAP_RESAMPLE_COUNT,
    CARRIER_SITE_COUNT,
    CHART_COUNT,
    CHART_RANK,
    COMMON_COMPARISON_ABI,
    DEFAULT_OUTPUT,
    EVENT_COUNT,
    GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA,
    Gemma3ChartNativeNeighborSetup,
    HIDDEN_WIDTH,
    LOGICAL_MODE_COUNT,
    MAX_DIRECTIONAL_NORMALIZED_RESIDUAL,
    MUTUAL_K,
    PINNED_RUNTIME_QUALIFICATION_FILE_SHA256,
    PINNED_RUNTIME_QUALIFICATION_PATH,
    PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    ROW_SHUFFLE_CATALOG_DOMAIN,
    ROW_SHUFFLE_DOMAIN,
    ROW_SHUFFLE_SEED,
    build_setup,
    canonical_json_bytes,
    check_setup,
    main,
    setup_sha256,
    validate_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "src/fisher_graph/gemma3_chart_native_neighbor_setup.py"
_SHIM = _ROOT / "src/fisher_graph_chart_native_neighbor_setup.py"
_ARTIFACT = _ROOT / DEFAULT_OUTPUT


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_setup_is_standard_library_only_import_inert_and_empty() -> None:
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "runpy",
        "stat",
        "tempfile",
        "typing",
    }
    assert _import_roots(_MODULE) <= allowed
    assert _import_roots(_SHIM) <= allowed
    setup = build_setup()
    assert setup["schema"] == GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA
    assert not any(setup["setup_receipt"].values())
    assert setup["claim_boundary"][
        "setup_artifact_contains_no_measurement"
    ] is True
    assert json.loads(canonical_json_bytes(setup)) == setup


def test_exact_39_event_schedule_and_73_site_capture_are_closed() -> None:
    setup = build_setup()
    lineage = setup["exact_execution_lineage"]
    events = lineage["ordered_events"]
    assert len(events) == lineage["ordered_event_count"] == EVENT_COUNT == 39
    assert [event["event_index"] for event in events] == list(range(39))
    assert events[0]["event_id"] == "lexical.ingress"
    assert events[-2]["event_id"] == "terminal.rmsnorm"
    assert events[-1]["event_id"] == "lexical.egress"
    attention = [e for e in events if e["event_kind"] == "attention_mutation"]
    mlp = [e for e in events if e["event_kind"] == "mlp_mutation"]
    assert len(attention) == len(mlp) == 18
    assert [e["block_ordinal"] for e in attention if e["attention_kind"] == "global"] == [
        5,
        11,
        17,
    ]

    site_lineage = setup["carrier_observation_lineage"]
    sites = site_lineage["ordered_sites"]
    assert len(sites) == site_lineage["ordered_site_count"] == (
        CARRIER_SITE_COUNT
    ) == 73
    assert sites[0]["site_id"] == "lexical.ingress.output"
    assert sites[1]["site_id"] == "mutation.1.input"
    assert sites[2]["site_id"] == "mutation.1.output"
    assert sites[-1]["site_id"] == "mutation.36.output"
    assert all("terminal" not in site["site_id"] for site in sites)
    assert all("egress" not in site["site_id"] for site in sites)
    assert site_lineage["terminal_and_egress_are_not_silently_added_to_the_73_sites"]


def test_38_logical_modes_have_exact_context_and_contribution_sites() -> None:
    ontology = build_setup()["logical_mode_ontology"]
    modes = ontology["ordered_modes"]
    assert len(modes) == ontology["logical_mode_count"] == (
        LOGICAL_MODE_COUNT
    ) == 38
    assert [mode["mode_index"] for mode in modes] == list(range(38))
    assert len({mode["native_ABI"] for mode in modes}) == 4
    assert {mode["comparison_ABI"] for mode in modes} == {
        COMMON_COMPARISON_ABI
    } == {"carrier_chart_energy.h640.v1"}
    assert modes[0]["context_site_id"] == "lexical.ingress.output"
    assert modes[0]["contribution_site_id"] == "lexical.ingress.output"
    body = modes[1:-1]
    assert all(mode["context_site_id"].endswith(".input") for mode in body)
    assert all(mode["contribution_site_id"].endswith(".output") for mode in body)
    egress = modes[-1]
    assert egress["mode_id"] == "mode.lexical.egress"
    assert egress["context_site_id"] == "terminal.rmsnorm.output"
    assert egress["value_site_id"] == "terminal.rmsnorm.output"
    assert egress["score_gradient_site_id"] == (
        "lexical.egress.readout_pullback"
    )
    assert ontology["tied_lexical_weight_equality_is_not_chart_equivalence"]


def test_prompt_free_runtime_qualification_is_authenticated_before_A() -> None:
    frozen = build_setup()["frozen_runtime_qualification"]
    assert frozen == {
        "authenticated_present": True,
        "path": str(PINNED_RUNTIME_QUALIFICATION_PATH),
        "file_sha256": PINNED_RUNTIME_QUALIFICATION_FILE_SHA256,
        "qualification_receipt_sha256": (
            PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256
        ),
        "panel_corpus_bytes_authenticated_opaque": True,
        "prompt_text_decoded_or_semantically_opened": False,
        "panel_semantic_access_performed": False,
        "tokenizer_and_model_loaded_only_after_static_preflight": True,
        "qualifies_compiled_runtime_before_A_access": True,
    }
    assert _file_sha256(_ROOT / PINNED_RUNTIME_QUALIFICATION_PATH) == (
        PINNED_RUNTIME_QUALIFICATION_FILE_SHA256
    )


def test_shared_chart_atlas_is_one_common_full_rank_axis_system() -> None:
    setup = build_setup()
    abi = setup["common_comparison_ABI"]
    atlas = setup["shared_global_chart_atlas"]
    assert abi["abi_id"] == COMMON_COMPARISON_ABI
    assert abi["hidden_width"] == atlas["hidden_width"] == HIDDEN_WIDTH == 640
    assert abi["row_key"] == [
        "example_id",
        "prediction_query_position",
        "target_label_position",
    ]
    assert abi["row_semantics"] == (
        "aligned_example_prediction_query_and_target_label_positions"
    )
    assert abi["row_axis_contract"]["exact_relation"] == "t=q+1"
    assert abi["row_axis_contract"][
        "ordered_role_family_example_q_t_catalog_sha256_required"
    ]
    assert atlas["one_atlas_shared_by_all_38_modes"]
    assert atlas["mode_specific_chart_axes_forbidden"]
    assert atlas["chart_count"] == CHART_COUNT == 8
    assert atlas["tangent_rank"] == CHART_RANK == 16
    assert atlas["membership_contract"] == "finite_nonnegative_row_simplex"
    metric = atlas["local_SPD_metric"]
    assert metric["metric_is_full_rank_SPD"]
    assert metric["semantics"] == (
        "shared_membership_spd_tangent_inverse_scale_plus_isotropic_orthogonal"
    )
    assert metric["alpha_c_finite_strictly_positive"]
    assert metric["isotropic_orthogonal_residual_may_not_be_zeroed"]
    assert atlas["shared_coordinate_signature_required_for_every_mode"]
    assert atlas["balanced_row_weight_formula"].startswith("w_i_f_e_r=1/")
    center = atlas["center_fit"]
    assert center["initialization"] == "deterministic_weighted_farthest_point"
    assert center["lloyd_iteration_count"] == 12
    assert center["argmax_tie_break"] == "earliest_row_in_canonical_fit_pool"
    assert center["minimum_hard_rows_per_chart"] == CHART_RANK + 1 == 17
    assert center["minimum_balanced_hard_occupancy_weight"] == 0.005
    assert atlas["membership_formula"].startswith("q_r_c=softmax_c")
    tangent = atlas["tangent_fit"]
    assert tangent["input"] == "centered_context_x_r_minus_center_c"
    assert tangent["solver"] == (
        "symmetric_eigh_float64_descending_eigenvalue"
    )
    assert tangent["generalized_contributions_do_not_fit_router_or_metric"]
    streaming = atlas["production_streaming_execution"]
    assert streaming["required_for_live_h640_run"]
    assert streaming[
        "in_memory_fit_shared_chart_atlas_is_fixture_reference_only"
    ]
    assert streaming["fixed_row_chunk_size"] == 256
    assert streaming["fit_pool_row_count"] == 38 * 24_732
    assert streaming["full_context_pool_torch_cat_forbidden"]
    assert streaming["broadcast_N_by_8_by_640_distance_tensor_forbidden"]
    assert streaming["raw_spool_bytes_per_production_mode"] == 189_941_760
    assert streaming["raw_spool_bytes_per_production_role"] == 7_217_786_880
    assert streaming["estimated_peak_A_payload_plus_atomic_temp_bytes"] == (
        10_829_054_592
    )
    assert streaming["maximum_live_resident_set_bytes"] == 8 * 1024**3


def test_generalized_contribution_and_energy_keep_sign_and_full_width() -> None:
    setup = build_setup()
    contribution = setup["common_comparison_ABI"]["generalized_contribution"]
    energy = setup["Fisher_chart_energy"]
    assert contribution["output_shape"] == "h640"
    assert contribution["semantics"] == (
        "value_hadamard_summed_next_token_nll_score_gradient"
    )
    assert contribution["signed_values_retained"]
    assert contribution[
        "absolute_value_or_squared_value_before_inner_product_forbidden"
    ]
    assert "elementwise_mul" in contribution["formula"]
    assert energy["self_energy_formula"].startswith("E_i_r=sum_c q_i_r_c")
    assert energy["semantics"] == "membership_weighted_spd_self_cross_mismatch"
    assert energy["weighted_activation_energy_is_diagnostic_only"]
    assert energy["primary_self_energy_field"] == "weighted_fisher_energy"
    assert energy["full_width_energy_closure"].endswith(
        "equals weighted_fisher_energy"
    )
    assert "sqrt(q_i_r_c q_j_r_c)" in energy["cross_inner_product_formula"]
    assert energy["mismatch_formula"] == "Delta_ij_r=E_i_r+E_j_r-2C_ij_r"
    assert energy["sign_and_cross_terms_retained"]
    assert energy["coordinate_only_energy_without_orthogonal_closure_forbidden"]
    assert energy["raw_activation_norm_without_score_gradient_forbidden"]
    assert energy[
        "individual_authenticated_row_Fisher_energy_may_be_exactly_zero"
    ]
    assert energy["minimum_row_Fisher_energy_is_finite_and_nonnegative"]
    assert energy["negative_or_nonfinite_row_Fisher_energy_fails_closed"]
    assert energy["zero_energy_rows_are_never_dropped"]
    assert energy[
        "per_family_directed_target_energy_at_or_below_epsilon_fails_closed"
    ]


def test_bidirectional_transport_distance_uses_target_energy_and_equal_families() -> None:
    transport = build_setup()["directed_chart_transport"]
    assert transport["fit_all_ordered_distinct_mode_pairs"]
    assert transport["ordered_pair_count"] == 38 * 37
    assert transport["source_feature_width"] == 8 * 17
    assert "exact_target" in transport["directional_normalization"]
    assert transport["within_family_directional_distance"].startswith(
        "sum_family_row_residual_energy"
    )
    assert transport["pooled_directional_distance"] == (
        "equal_family_mean_of_within_family_directional_distances"
    )
    assert transport[
        "global_sum_residual_over_sum_target_energy_is_diagnostic_only"
    ]
    assert transport["symmetric_neighbor_distance"] == (
        "0.5 times (distance_a_to_b plus distance_b_to_a)"
    )
    assert transport["both_directed_receipts_required"]
    wls = transport["canonical_WLS"]
    assert wls["X"].endswith("shape_N_by_136")
    assert wls["Y"].endswith("shape_N_by_640")
    assert "max(target_row_weighted_fisher_energy,1e-12)" in (
        wls["unnormalized_W_diagonal"]
    )
    assert wls["zero_Fisher_rows_retained_with_floor"]
    assert wls["absolute_ridge"] == "1e-3*max(ridge_scale,1e-12)"
    assert wls["solve"] == (
        "Beta=solve(G+absolute_ridge*I_136,X^T*W*Y)"
    )
    assert "effective_normalized_W_sha256" in (
        transport["per_pair_fit_receipts_bind"]
    )
    assert transport["required_directed_evaluation_fields"][-8:] == [
        "weighted_predicted_fisher_energy",
        "weighted_target_fisher_energy",
        "weighted_predicted_target_cross_inner_product",
        "weighted_transport_residual_energy",
        "weighted_zero_prediction_residual_energy",
        "weighted_row_shuffled_transport_residual_energy",
        "real_gain_over_zero",
        "row_shuffle_gain_over_zero",
    ]
    assert transport["weighted_transport_residual_energy_formula"].startswith(
        "weighted_predicted_fisher_energy"
    )
    assert transport["weighted_zero_prediction_residual_energy_formula"] == (
        "weighted_target_fisher_energy"
    )
    assert transport["zero_prediction_control_required_per_family_and_direction"]
    assert transport[
        "within_example_source_row_shuffle_control_required_per_direction"
    ]
    shuffle = transport["row_shuffle_control"]
    assert shuffle["real_A_fitted_transport_reused_without_refit"]
    assert shuffle["separately_fitted_shuffled_transport_forbidden"]
    assert shuffle["source_feature_rows_permuted_within_each_family_example"]
    assert shuffle["cross_example_source_rows_forbidden"]
    assert shuffle["diagnostic_514_rows_cannot_enter_primary_control"]
    assert shuffle["target_rows_unchanged"]
    assert shuffle["seed"] == ROW_SHUFFLE_SEED == 6_607
    assert shuffle["domain_hex"] == ROW_SHUFFLE_DOMAIN.hex()
    assert shuffle["catalog_digest_domain_hex"] == (
        ROW_SHUFFLE_CATALOG_DOMAIN.hex()
    )
    assert shuffle["derangement_guaranteed_within_each_family_example"]
    assert shuffle[
        "one_role_family_example_permutation_catalog_reused_for_all_directed_pairs_"
        "point_bootstrap_and_LOFO_replicates"
    ]
    assert shuffle["subset_stable_under_whole_example_selection"]
    assert shuffle["permutation_applied_before_example_resampling"]
    production = transport["production_streaming_execution"]
    assert production["prediction_kernel_id"] == (
        "torch_mm_cpu_float64_streamed_fixed_rows_v1"
    )
    assert production["compact_store_field_dtypes"] == {
        "source_features_N_by_136": "float64_little_endian",
        "memberships_N_by_8": "float64_little_endian",
        "signed_generalized_contributions_N_by_640": (
            "float32_little_endian"
        ),
        "weighted_activation_energy_N": "float64_little_endian",
        "weighted_chart_tangent_energy_N": "float64_little_endian",
        "weighted_chart_orthogonal_energy_N": "float64_little_endian",
        "weighted_fisher_energy_N": "float64_little_endian",
    }
    assert production["compact_bytes_per_production_mode"] == 92_596_608
    assert production["compact_bytes_per_production_role"] == 3_518_671_104
    assert production["prediction_row_chunk_size"] == 256
    assert production["one_python_matrix_vector_call_per_row_forbidden"]
    assert production["all_1406_coefficient_unique_float64_bytes"] == (
        979_025_920
    )


def test_attention_requires_explicit_causal_cross_token_routes() -> None:
    setup = build_setup()
    modes = setup["logical_mode_ontology"]["ordered_modes"]
    routed = [mode for mode in modes if mode["requires_cross_token_route_receipt"]]
    assert len(routed) == 18
    assert {mode["mode_kind"] for mode in routed} == {"attention"}
    route = setup["attention_cross_token_route_chart"]
    assert route["required_for_every_attention_mode"]
    assert route["route_row_key"] == [
        "example_id",
        "context_fingerprint_sha256",
        "prediction_query_position",
        "query_head",
        "source_token_position",
    ]
    assert route["causal_constraint"] == (
        "source_token_position_le_prediction_query_position"
    )
    assert route["no_top_k_or_energy_threshold_route_truncation"]
    assert route["route_sum_must_close_to_pre_postnorm_operator_contribution"]
    assert route["route_sum_does_not_equal_common_postnorm_delta_contribution"]
    factorized = route["production_factorized_aggregation"]
    assert factorized["kernel_id"] == (
        "factorized_attention_route_energy.p_u_g.float64_chunked.v1"
    )
    assert factorized["materialized_h640_route_coordinate_scalar_count"] == 0
    assert factorized["prediction_query_chunk_size"] == 32
    assert factorized["source_key_chunk_size"] == 128
    assert factorized[
        "stored_native_probability_bytes_preserved_without_renormalization"
    ]
    assert (
        factorized["probability_simplex_audit_accumulation_dtype"]
        == "torch.float64"
    )
    assert factorized[
        "float32_probability_sum_then_float64_cast_forbidden"
    ]
    assert factorized["probability_simplex_fixed_tolerance_forbidden"]
    assert factorized["source_softmax_dtype"] == "torch.float32"
    assert factorized["source_softmax_device"] == "cpu"
    assert factorized["probability_simplex_forward_error_formula_id"] == (
        "float32_softmax_simplex."
        "gamma32_support_plus_2_gamma64_audit_plus_ftz.v1"
    )
    assert factorized["probability_simplex_support_derivation_id"] == (
        "query_key_valid_masks_logical_causal_optional_strict_window.v1"
    )
    assert factorized["probability_simplex_support_minimum"] == 1
    assert factorized["probability_simplex_support_maximum"] == 768
    assert factorized["probability_simplex_float32_unit_roundoff_power"] == -24
    assert factorized["probability_simplex_float64_unit_roundoff_power"] == -53
    assert factorized["probability_simplex_float32_FTZ_safety_power"] == -126
    assert factorized["probability_simplex_bound_formula"] == (
        "gamma32(k+2)+gamma64(k-1)*(1+gamma32(k+2))+k*2^-126"
    )
    assert factorized["finite_probability_in_closed_unit_interval_required"]
    assert factorized[
        "masked_and_invalid_route_probabilities_exact_zero_required"
    ]
    assert factorized[
        "native_dense_support_equals_authenticated_logical_support_"
        "at_capture_required"
    ]
    assert factorized[
        "per_snapshot_and_per_example_simplex_audit_receipt_required"
    ]
    assert factorized["frozen_kernel_receipt_sha256"] == (
        "228433c4960c61660e9e0cb6e3f08d1a2a0fbf54f6bd1f9f6175d8beac23020e"
    )
    assert factorized["literal_full_route_coordinate_stream_hash_forbidden"]
    assert factorized[
        "operator_intrinsic_and_common_postnorm_energy_zero_support_"
        "equivalence_not_required"
    ]
    bridge = route["operator_output_to_common_postnorm_delta_bridge"]
    assert bridge["required"]
    assert bridge["reason"] == "post_attention_RMSNorm_is_nonlinear"
    assert bridge["authenticated_exact_native_transform_receipt_required"]
    assert bridge["bridge_is_provenance_not_neighbor_distance"]
    assert route["same_token_only_route_receipt_is_invalid"]
    assert route["supplemental_to_common_axis_neighbor_distance"]
    assert route["route_similarity_is_diagnostic_only_in_v1"]


def test_lexical_egress_uses_exact_softcap_readout_pullback_not_logits() -> None:
    readout = build_setup()["lexical_egress_readout_chart"]
    assert readout["required"]
    assert readout["separate_from_ordered_73_carrier_sites"]
    assert readout["context_and_value_site_id"] == "terminal.rmsnorm.output"
    assert readout["pullback_site_id"] == "lexical.egress.readout_pullback"
    assert readout["native_tied_embedding_projection_authenticated"]
    assert readout["exact_model_config_logit_softcap_authenticated"]
    assert readout["softcap_derivative_included_in_pullback"]
    assert readout["raw_vocabulary_logits_as_chart_coordinates_forbidden"]
    assert readout["vocabulary_width_never_enters_common_comparison_ABI"]
    lexical = readout["chunked_full_vocabulary_NLL_VJP"]
    assert lexical["kernel_receipt_sha256"] == (
        "e79c953bd99483d87e8c09406388deecf2474a9dc20c468c0fd68c38d9deffdb"
    )
    assert lexical["float64_reduction_required"]
    assert lexical["dense_reference_dtype"] == "torch.float64"
    assert lexical["float32_dense_reference_forbidden"]
    assert lexical["first_A_dense_parity_selected_prediction_rows"] == 2


def test_graph_is_mutual_k2_over_all_cross_kind_pairs_with_strict_gates() -> None:
    setup = build_setup()
    graph = setup["neighbor_graph_protocol"]
    gates = setup["fresh_confirmation_gates"]
    assert graph["node_count"] == 38
    assert graph["unordered_pair_count"] == 38 * 37 // 2
    assert graph["complete_cross_kind_ranking_universe"]
    assert graph["k"] == MUTUAL_K == 2
    assert graph["mutual_neighbor_required"]
    assert graph["maximum_each_directional_normalized_residual"] == (
        MAX_DIRECTIONAL_NORMALIZED_RESIDUAL
    ) == 1.0
    assert graph["rank_boundary_tie_policy"] == (
        "fail_closed_undefined_node_neighbors"
    )
    assert graph["mode_kind_or_event_index_tie_break_forbidden"]
    assert gates["within_family_bootstrap"]["resample_count"] == (
        BOOTSTRAP_RESAMPLE_COUNT
    ) == 10_000
    assert gates["within_family_bootstrap"]["minimum_mutual_frequency"] == (
        BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY
    ) == 0.80
    assert gates["leave_one_family_out"]["minimum_mutual_count"] == 5
    assert gates["long_context_stratum_repeats_point_topology_and_error_gates"]
    assert gates["partial_gate_pass_never_confirms_an_edge"]


def test_fresh_roles_prevent_fit_confirmation_and_response_leakage() -> None:
    firewall = build_setup()["fresh_data_firewall"]
    assert firewall["ordered_roles"] == [
        "A_chart_transport_fit",
        "B_neighbor_discovery",
        "C_neighbor_confirmation",
        "D_response_validation",
        "reserve_unopened",
    ]
    assert firewall["all_five_role_files_are_authenticated_and_present"]
    assert firewall["one_authoritative_frozen_fresh_manifest_and_receipt"]
    assert len(firewall["role_file_sha256s"]) == 5
    assert firewall["A_only_may_fit_shared_atlas_or_transport_tensors"]
    assert firewall["B_C_D_may_not_refit_or_select_atlas_transport_or_thresholds"]
    assert firewall["C_may_not_change_k_candidate_catalog_or_gate_thresholds"]
    assert firewall["D_may_not_add_remove_or_rerank_chart_neighbor_edges"]
    assert firewall["reserve_opaque_byte_hash_access_allowed"]
    assert firewall[
        "reserve_JSON_decode_tokenizer_or_model_access_invalidates_run"
    ]
    assert firewall["semantic_access_sequence"][0] == "parse_and_tokenize_A_only"
    assert firewall["semantic_access_sequence"][-1].startswith(
        "optionally_parse_and_tokenize_D"
    )
    eligible = firewall["B_C_primary_eligibility"]
    assert eligible["eligible_examples_per_family"] == 7
    assert eligible["eligible_token_counts"] == [
        96,
        511,
        512,
        513,
        576,
        640,
        768,
    ]
    assert firewall["transition_514_reporting"]["diagnostic_only"]
    assert firewall["hard_long_confirmation_veto"][
        "hard_long_subset_may_not_nominate_new_edges"
    ]


def test_bootstrap_draw_catalog_is_exact_and_shared() -> None:
    bootstrap = build_setup()["fresh_confirmation_gates"][
        "within_family_bootstrap"
    ]
    assert bootstrap["draw_protocol"] == BOOTSTRAP_DRAW_PROTOCOL == (
        "sha256_counter_modulo_example_count_v1"
    )
    assert bootstrap["draw_protocol_domain_hex"] == BOOTSTRAP_DRAW_DOMAIN.hex()
    assert bootstrap["draw_catalog_digest_domain_hex"] == (
        BOOTSTRAP_CATALOG_DOMAIN.hex()
    )
    assert bootstrap["maximum_degenerate_fraction"] == (
        BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
    ) == 0.01
    assert bootstrap["draws_per_family"] == 7
    assert bootstrap[
        "one_draw_catalog_reused_for_every_pair_direction_control_and_"
        "topology_calculation"
    ]


def test_response_ablation_is_validation_only_and_never_defines_graph() -> None:
    response = build_setup()["response_ablation_validation_only"]
    assert response["opened_after_chart_graph_freeze"]
    assert response["whole_model_native_location_interventions_required"]
    assert response["response_measurements_may_nominate_edges"] is False
    assert response["response_measurements_may_remove_chart_edges"] is False
    assert response["response_measurements_may_rerank_chart_edges"] is False
    assert response["response_only_neighbor_claim_forbidden"]
    assert response["compression_followup_requires_downstream_response_support"]


def test_failed_old_edge_catalogs_are_not_inherited_as_neighbor_truth() -> None:
    lineage = build_setup()["frozen_predecessor_lineage"]
    assert lineage["old_adjacent_same_token_edge_catalog_inherited"] is False
    assert lineage["old_exact_edge_catalog_inherited"] is False
    assert lineage["old_response_neighbor_catalog_inherited"] is False
    assert lineage["predecessors_supply_lineage_and_negative_evidence_only"]
    assert [row["outcome"] for row in lineage["spent_development_result_receipts"]] == [
        "heldout_edge_gain_gate_failed",
        "heldout_predictive_or_causal_edge_gate_failed",
    ]
    assert not any(
        row["inherited_as_neighbor_truth"]
        for row in lineage["spent_development_result_receipts"]
    )


def test_setup_claims_no_result_and_truthfully_pins_worker_runtime() -> None:
    setup = build_setup()
    status = setup["implementation_status"]
    boundary = setup["claim_boundary"]
    assert status["worker_entrypoint_implemented_and_source_pinned"]
    assert status["concrete_runtime_API_implemented_and_source_pinned"]
    assert status[
        "conservative_first_party_functional_closure_source_pinned"
    ]
    assert status["setup_module_is_reviewed_non_self_hashed_trust_anchor"]
    assert status["package_eager_import_occurs_before_static_preflight"]
    assert status[
        "package_eager_import_is_outside_preexecution_authentication_boundary"
    ]
    assert not status["whole_process_or_every_executable_tamper_proof_claimed"]
    assert boundary["fresh_panels_exist"]
    assert boundary["fresh_panels_are_frozen_and_byte_authenticated"]
    scientific = {
        key: value
        for key, value in boundary.items()
        if key
        not in {
            "setup_artifact_contains_no_measurement",
            "fresh_panels_exist",
            "fresh_panels_are_frozen_and_byte_authenticated",
        }
    }
    assert not any(scientific.values())
    never = setup["claim_taxonomy"]["never_authorized_by_this_protocol"]
    assert all(never.values())


def test_strict_validator_rejects_semantic_and_lineage_tampering() -> None:
    setup = build_setup()
    assert validate_setup(setup) == setup
    assert Gemma3ChartNativeNeighborSetup(setup).to_dict() == setup

    attacks: list[dict[str, object]] = []
    changed_abi = deepcopy(setup)
    changed_abi["logical_mode_ontology"]["ordered_modes"][1][
        "comparison_ABI"
    ] = "native_attention_only"
    attacks.append(changed_abi)
    changed_site = deepcopy(setup)
    changed_site["logical_mode_ontology"]["ordered_modes"][1][
        "context_site_id"
    ] = "mutation.1.output"
    attacks.append(changed_site)
    changed_route = deepcopy(setup)
    changed_route["attention_cross_token_route_chart"][
        "same_token_only_route_receipt_is_invalid"
    ] = False
    attacks.append(changed_route)
    changed_readout = deepcopy(setup)
    changed_readout["lexical_egress_readout_chart"][
        "softcap_derivative_included_in_pullback"
    ] = False
    attacks.append(changed_readout)
    changed_claim = deepcopy(setup)
    changed_claim["claim_boundary"]["neighbor_edge_claimed"] = True
    attacks.append(changed_claim)
    for attack in attacks:
        attack["setup_sha256"] = setup_sha256(attack)
        with pytest.raises(ValueError):
            validate_setup(attack)


def test_canonical_artifact_and_public_hash_pins_close() -> None:
    payload = check_setup(_ARTIFACT)
    assert payload == build_setup()
    assert payload["setup_sha256"] == PINNED_SETUP_SHA256
    assert _file_sha256(_ARTIFACT) == PINNED_SETUP_FILE_SHA256


def test_cli_check_is_read_only_and_refuses_overwrite(tmp_path: Path) -> None:
    assert main(["--check", "--output", str(_ARTIFACT)]) == 0
    destination = tmp_path / "setup.json"
    assert main(["--output", str(destination)]) == 0
    assert validate_setup(json.loads(destination.read_text())) == build_setup()
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--output", str(destination)])
    assert destination.read_bytes() == before

    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(build_setup()), encoding="utf-8")
    with pytest.raises(ValueError, match="file bytes drifted"):
        check_setup(reformatted)
