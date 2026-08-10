"""Freeze the prospective Gemma-3 local-flow multimode v3 protocol.

The protocol replaces only the eighteen token-local MLP residual maps.  A
normal-residual chart partition routes each MLP input state, a second soft
partition routes among affine tangent velocities inside each chart, and the
decoded local velocities are added to the untouched residual carrier.

This module is standard-library-only.  It may build, strictly load, and check
the prospective contract, but it cannot import a tensor library or run a
model.  Preflight and run remain fail-closed until new held-out panels and the
complete implementation qualification have been frozen into this source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence


SCHEMA = "fisher_graph.gemma3_layer_multimode.setup.v1"
FORMAT_VERSION = 1
SETUP_ID = "gemma3-layer-local-flow-multimode-v3"

DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-layer-multimode-setup-v1.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/layer-multimode-experiment-v1.json"
)
DEFAULT_BUNDLE_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/layer-multimode-experiment-v1.bundle"
)
FUTURE_PANEL_MANIFEST = Path(
    "artifacts/research/gemma3-layer-multimode-fresh-panels-v1.json"
)
FUTURE_IMPLEMENTATION_QUALIFICATION = Path(
    "artifacts/research/"
    "gemma3-layer-multimode-implementation-qualification-v1.json"
)

# Deliberately absent during prospective implementation.  A reviewed freeze
# replaces these with lowercase SHA-256 digests before any live preflight.
PINNED_PANEL_MANIFEST_FILE_SHA256: str | None = None
PINNED_IMPLEMENTATION_QUALIFICATION_FILE_SHA256: str | None = None

PREDECESSOR_V2_SETUP = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v2.json"
)
PREDECESSOR_V2_SETUP_SHA256 = (
    "f786a320c9a646f7141d1832913c11149297b09d3d01eca033acfe1fd9cb7f50"
)
PREDECESSOR_V2_SETUP_FILE_SHA256 = (
    "2e6341417956ea3e2dd021657c951f87221078777be9d185c173d651afae4231"
)
PREDECESSOR_V2_REPORT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v2.json"
)
PREDECESSOR_V2_REPORT_SHA256 = (
    "bc6bbe8913ee89ba1b9d47b264f1e3686ecd830e4f7928e0075e964f76f87a2b"
)
PREDECESSOR_V2_REPORT_FILE_SHA256 = (
    "d8a3d1f528a889d4aa941e688825ca1c608da6b28a2c25ce0b70efdc406c3ee6"
)

LAYER_COUNT = 18
HIDDEN_WIDTH = 640
MLP_INTERMEDIATE_WIDTH = 2_048
CHART_COUNT_CANDIDATES = (2, 1)
MODE_COUNT_CANDIDATES = (4, 2, 1)
TANGENT_RANK_CANDIDATES = (64, 32, 16)
CANDIDATE_COUNT_PER_LAYER = (
    len(CHART_COUNT_CANDIDATES)
    * len(MODE_COUNT_CANDIDATES)
    * len(TANGENT_RANK_CANDIDATES)
)
NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER = (
    MLP_INTERMEDIATE_WIDTH * 3 * HIDDEN_WIDTH
)

FIT_ROLE = "A_exposed_development"
DISCOVERY_ROLE = "B3_teacher_forced_layerwise"
CONFIRMATION_ROLE = "C3_full_stack_source_free"
FINAL_ROLE = "D3_final_heldout_validation"
RESERVE_ROLE = "reserve_v3_unopened"
ROLE_ORDER = (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE, FINAL_ROLE, RESERVE_ROLE)
PRODUCTION_FIT_POLICY_SHA256 = (
    "be4908d50da4a0924484b192baac82e453e39ed316854e749e9a867252eae636"
)
PRODUCTION_FULL_A_CAPTURE_SCOPE = (
    "production_full_A_RoleCaptureReceipt_spools"
)
FIT_OPTIMIZER_ID = (
    "torch_adam_cpu_float64_fixed_minibatch_separate_tau_clip.v2"
)
GRADIENT_CLIP_GROUPING_ID = (
    "non_tau_global_l2_and_chart_temperature_scalar_independent.v1"
)

_SETUP_DOMAIN = b"fisher_graph:gemma3-layer-local-flow-multimode-setup:v1\0"
_MAX_SETUP_BYTES = 8 * 1024 * 1024
_MAX_PREDECESSOR_BYTES = 512 * 1024 * 1024


class ExecutionNotAuthorizedError(RuntimeError):
    """Raised before runtime access when prospective prerequisites are absent."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def setup_sha256(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("setup_sha256", None)
    return hashlib.sha256(
        _SETUP_DOMAIN + canonical_json_bytes(unsigned)
    ).hexdigest()


def replacement_parameter_accounting(
    chart_count: int,
    mode_count: int,
    tangent_rank: int,
) -> dict[str, int]:
    """Return the exact frozen per-layer deployable parameter accounting."""

    if type(chart_count) is not int or chart_count not in CHART_COUNT_CANDIDATES:
        raise ValueError("chart_count is outside the frozen candidate grid")
    if type(mode_count) is not int or mode_count not in MODE_COUNT_CANDIDATES:
        raise ValueError("mode_count is outside the frozen candidate grid")
    if (
        type(tangent_rank) is not int
        or tangent_rank not in TANGENT_RANK_CANDIDATES
    ):
        raise ValueError("tangent_rank is outside the frozen candidate grid")
    # The teacher velocity is captured after the native output RMSNorm.  The
    # compiled map predicts that velocity directly, so deployment keeps only
    # the input RMSNorm that produces x.  The source boundary being replaced
    # still contains both native RMSNorm vectors.
    retained_norms = HIDDEN_WIDTH
    source_norms = 2 * HIDDEN_WIDTH
    source_mlp_matrices = NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER
    source_total = source_norms + source_mlp_matrices
    centers = chart_count * HIDDEN_WIDTH
    bases = chart_count * HIDDEN_WIDTH * tangent_rank
    fixed_atlas = centers + bases
    chart_temperature = 1
    routers = chart_count * mode_count * (tangent_rank + 1)
    generators = chart_count * mode_count * (
        tangent_rank * tangent_rank + tangent_rank
    )
    total = (
        retained_norms
        + centers
        + bases
        + chart_temperature
        + routers
        + generators
    )
    source_macs = source_mlp_matrices
    chart_projection_macs = chart_count * HIDDEN_WIDTH * tangent_rank
    fused_chart_velocity_decode_macs = chart_projection_macs
    router_macs = chart_count * mode_count * tangent_rank
    local_generator_macs = chart_count * mode_count * tangent_rank * tangent_rank
    candidate_macs = (
        chart_projection_macs
        + fused_chart_velocity_decode_macs
        + router_macs
        + local_generator_macs
    )
    return {
        "retained_rmsnorm_parameter_count": retained_norms,
        "source_mlp_rmsnorm_parameter_count": source_norms,
        "source_mlp_matrix_parameter_count": source_mlp_matrices,
        "source_total_parameter_count": source_total,
        "chart_center_parameter_count": centers,
        "chart_basis_parameter_count": bases,
        "fixed_atlas_parameter_count": fixed_atlas,
        "chart_temperature_parameter_count": chart_temperature,
        "router_parameter_count": routers,
        "local_generator_parameter_count": generators,
        "candidate_total_parameter_count": total,
        "float32_coefficient_payload_bytes": total * 4,
        "net_parameter_savings": source_total - total,
        "conservative_parameter_savings_vs_source_MLP_matrices_only": (
            source_mlp_matrices - total
        ),
        "source_matrix_macs_per_token": source_macs,
        "chart_projection_macs_per_token": chart_projection_macs,
        "fused_chart_velocity_decode_macs_per_token": (
            fused_chart_velocity_decode_macs
        ),
        "router_macs_per_token": router_macs,
        "local_generator_macs_per_token": local_generator_macs,
        "candidate_matrix_and_mixture_macs_per_token": candidate_macs,
        "net_macs_per_token": source_macs - candidate_macs,
    }


def _candidate_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for chart_count in CHART_COUNT_CANDIDATES:
        for mode_count in MODE_COUNT_CANDIDATES:
            for tangent_rank in TANGENT_RANK_CANDIDATES:
                rows.append(
                    {
                        "candidate_ordinal": ordinal,
                        "chart_count": chart_count,
                        "modes_per_chart": mode_count,
                        "tangent_rank": tangent_rank,
                        "accounting": replacement_parameter_accounting(
                            chart_count, mode_count, tangent_rank
                        ),
                    }
                )
                ordinal += 1
    return rows


def _build_state() -> dict[str, object]:
    panel_ready = PINNED_PANEL_MANIFEST_FILE_SHA256 is not None
    implementation_ready = (
        PINNED_IMPLEMENTATION_QUALIFICATION_FILE_SHA256 is not None
    )
    return {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "methodology_provenance": {
            "protocol_generation": "v3_local_flow_multimode",
            "predecessor_v2_setup_path": str(PREDECESSOR_V2_SETUP),
            "predecessor_v2_setup_sha256": PREDECESSOR_V2_SETUP_SHA256,
            "predecessor_v2_setup_file_sha256": (
                PREDECESSOR_V2_SETUP_FILE_SHA256
            ),
            "predecessor_v2_report_path": str(PREDECESSOR_V2_REPORT),
            "predecessor_v2_report_sha256": PREDECESSOR_V2_REPORT_SHA256,
            "predecessor_v2_report_file_sha256": (
                PREDECESSOR_V2_REPORT_FILE_SHA256
            ),
            "predecessor_v2_selected_chart_count": 2,
            "predecessor_v2_selected_tangent_rank": 16,
            "predecessor_v2_selected_feature_width": 34,
            "predecessor_v2_B_raw_mutual_pair_count": 5,
            "predecessor_v2_B_frozen_candidate_count": 0,
            "predecessor_v2_C_confirmed_edge_count": 0,
            "predecessor_v2_compression_ready": False,
            "predecessor_is_completed_negative_evidence": True,
            "predecessor_cannot_authorize_v3_selection_or_claims": True,
            "v3_changes_the_hypothesis_from_neighbor_edges_to_local_flow_"
            "approximation": True,
            "whole_protocol_preregistered_from_scratch_claimed": False,
        },
        "replacement_boundary": {
            "layer_count": LAYER_COUNT,
            "only_token_local_MLP_residual_maps_are_replaced": True,
            "exact_attention_retained": True,
            "exact_lexical_ingress_and_egress_retained": True,
            "all_layer_RMSNorms_retained": False,
            "only_MLP_input_RMSNorm_is_retained": True,
            "native_output_RMSNorm_is_folded_into_v_T_and_removed": True,
            "candidate_output_RMSNorm_or_double_normalization_is_forbidden": (
                True
            ),
            "native_MLP_input_state_x": "MLP_input_RMSNorm_output_h640",
            "native_teacher_velocity_v_T": (
                "post_output_norm_native_MLP_residual_delta_h640"
            ),
            "residual_carrier_h": "pre_MLP_residual_state_h640",
            "compiled_layer_output": "h_prime=h+v_hat(x)",
            "x_h_and_v_T_are_three_distinct_runtime_sites": True,
            "parent_input_output_or_carrier_conflation_is_forbidden": True,
            "teacher_velocity_is_fit_or_teacher_forced_evaluation_only": True,
            "source_free_runtime_reads_x_and_h_but_not_v_T": True,
        },
        "local_flow_mathematical_ABI": {
            "per_layer_atlas": {
                "centers": "mu_i_in_R^640_for_i_1_through_C",
                "orthonormal_bases": "B_i_in_R^(640_by_R)",
                "tangent_coordinate": "z_i=B_i^T*(x-mu_i)",
                "normal_residual": (
                    "rho_i_squared=||(I-B_i*B_i^T)*(x-mu_i)||_2^2"
                ),
                "equivalent_normal_residual": (
                    "||(x-mu_i)-B_i*z_i||_2^2"
                ),
                "compiled_normal_residual_identity": (
                    "rho_i_squared=||x-mu_i||_2^2-||z_i||_2^2_under_"
                    "orthonormal_B_i"
                ),
                "chart_partition": (
                    "q_i=softmax_i(-rho_i_squared/tau_c)"
                ),
                "tau_c": (
                    "one_positive_scalar_shared_by_all_charts_in_the_layer_"
                    "under_the_C_specific_policy_below"
                ),
                "tau_c_parameterization": "tau_c=exp(log_tau_c)",
                "C1_boundary": {
                    "q_is_exactly": [1.0],
                    "tau_c_is_fixed_to": 1.0,
                    "serialized_log_tau_c_is_fixed_to": 0.0,
                    "one_temperature_scalar_is_still_serialized_and_counted": (
                        True
                    ),
                },
                "C2_temperature_fit": (
                    "one_positive_tau_c_fitted_on_A_fit_rows_only"
                ),
                "q_rows_must_be_finite_nonnegative_and_sum_to_one": True,
            },
            "within_chart_modes": {
                "affine_router": "a_ik(z_i)=w_ik^T*z_i+b_ik",
                "mode_partition": "pi_ik=softmax_k(a_ik(z_i))",
                "mode_temperature": "fixed_nonlearned_1.0",
                "affine_tangent_velocity": "t_ik(z_i)=A_ik*z_i+c_ik",
                "local_generator": "G_ik=B_i*t_ik(z_i)",
                "pi_rows_must_be_finite_nonnegative_and_sum_to_one": True,
            },
            "compiled_velocity": "v_hat=sum_i(q_i*sum_k(pi_ik*G_ik))",
            "compiled_fused_velocity": {
                "within_chart_tangent_velocity": (
                    "u_i=sum_k(pi_ik*t_ik(z_i))"
                ),
                "single_decode_per_chart": "v_hat=sum_i(q_i*B_i*u_i)",
                "per_mode_H640_decode_or_mixture_is_forbidden": True,
                "mathematically_equal_to_unfused_generator_sum": True,
            },
            "compiled_output": "h_prime=h+v_hat",
            "no_hard_routing_or_top_k_at_runtime": True,
            "fit_and_selection_reductions": "CPU_float64_fixed_order",
            "deployed_coefficient_dtype": "float32",
            "float32_deployment_parity_receipt_required": True,
            "model_output_contains_no_teacher_velocity_or_native_MLP_fallback": (
                True
            ),
        },
        "candidate_grid_and_accounting": {
            "chart_count_candidates_in_declared_order": list(
                CHART_COUNT_CANDIDATES
            ),
            "modes_per_chart_candidates_in_declared_order": list(
                MODE_COUNT_CANDIDATES
            ),
            "tangent_rank_candidates_in_declared_order": list(
                TANGENT_RANK_CANDIDATES
            ),
            "complete_C_times_M_times_R_grid_required_per_layer": True,
            "candidate_count_per_layer": CANDIDATE_COUNT_PER_LAYER,
            "all_18_candidates_fit_before_layer_selection": True,
            "parameter_formula": (
                "H+C*H+C*H*R+1+C*M*(R+1)+C*M*(R*R+R)"
            ),
            "native_replaced_boundary_parameter_formula": (
                "2H+2048*3*H"
            ),
            "parameter_formula_terms": {
                "retained_input_RMSNorm": "H",
                "centers": "C*H",
                "bases": "C*H*R",
                "chart_temperature": "1",
                "routers": "C*M*(R+1)",
                "generators": "C*M*(R*R+R)",
            },
            "hidden_width_H": HIDDEN_WIDTH,
            "native_MLP_intermediate_width": MLP_INTERMEDIATE_WIDTH,
            "native_MLP_matrix_parameter_formula": "2048*3*640",
            "native_MLP_matrix_parameter_count_per_layer": (
                NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER
            ),
            "candidate_includes_one_retained_input_RMSNorm_vector": True,
            "source_comparison_includes_two_RMSNorm_vectors_and_three_MLP_"
            "matrices": True,
            "native_output_RMSNorm_is_not_deployed_or_double_applied": True,
            "candidate_table": _candidate_table(),
            "selection_receipt_serialized_parameter_count_must_equal_"
            "candidate_total_parameter_count": True,
            "MAC_formula": "2*C*H*R+C*M*R+C*M*R*R",
            "MAC_formula_terms_in_order": [
                "chart_projection",
                "once_per_chart_fused_velocity_basis_decode",
                "affine_router",
                "affine_local_tangent_generator",
            ],
            "normal_residual_matrix_reconstruction_is_eliminated_using_"
            "orthonormal_Pythagorean_identity": True,
            "within_chart_tangent_mixture_precedes_single_basis_decode": True,
            "MAC_convention_excludes": [
                "RMSNorms",
                "nonlinearities",
                "softmax",
                "vector_adds",
                "scalar_normal_distance_reductions",
                "scalar_soft_mixture_weights",
            ],
            "actual_serialized_artifact_bytes_required_not_estimated": True,
            "coefficient_payload_bytes_alone_do_not_satisfy_serialized_byte_"
            "claim": True,
            "per_layer_and_all_layer_accounting_receipts_required": True,
        },
        "A_development_selection": {
            "role": FIT_ROLE,
            "only_old_v2_A_may_be_used": True,
            "old_v2_A_is_disclosed_development_not_fresh_confirmation": True,
            "v2_B_C_D_and_reserve_are_forbidden": True,
            "per_family_fit_target_token_lengths": [96, 512, 514, 640],
            "per_family_validation_target_token_lengths": [511, 513, 576, 768],
            "fit_and_validation_examples_are_disjoint": True,
            "fit_weighting": (
                "equal_family_then_equal_example_then_equal_scored_row_"
                "within_example"
            ),
            "fit_is_independent_per_physical_layer": True,
            "all_grid_candidates_use_identical_A_fit_and_validation_rows": True,
            "all_M_candidates_at_fixed_layer_C_R_share_one_exact_state_atlas": (
                True
            ),
            "selector_rejects_per_layer_row_subset_or_fixed_C_R_atlas_drift": (
                True
            ),
            "production_selector_requires_all_324_original_live_adapter_"
            "candidate_evaluations": True,
            "production_compile_requires_original_selector_result_over_all_"
            "324_live_candidates": True,
            "coordinated_306_failure_relabels_around_18_live_winners_are_"
            "rejected": True,
            "candidate_evaluation_and_selection_direct_construction_copy_"
            "deepcopy_replace_or_parser_replay_cannot_recover_live_"
            "authority": True,
            "production_capture_and_partition": {
                "capture_source": (
                    "one_new_replay_of_the_complete_predecessor_A_role_using_"
                    "the_frozen_38_mode_runtime"
                ),
                "complete_role_capture_required": {
                    "batch_count": 45,
                    "example_count": 48,
                    "row_count_per_mode": 24732,
                },
                "selected_spools": (
                    "the_18_even_MLP_modes_in_physical_layer_order"
                ),
                "spool_chart_state": "exact_MLP_input_RMSNorm_output_x",
                "spool_teacher_velocity": (
                    "exact_post_output_RMSNorm_native_MLP_residual_delta_v_T"
                ),
                "spools_do_not_capture_residual_carrier_h": True,
                "unchanged_decoder_layer_still_owns_h_plus_v_residual_add": (
                    True
                ),
                "full_A_bridge_must_replay_role_session_panel_spool_row_and_"
                "tensor_lineage": True,
                "shared_partition_receipt_required": True,
                "source_capture_scope": PRODUCTION_FULL_A_CAPTURE_SCOPE,
                "materialization_inventory": {
                    "receipt_type": (
                        "LayerMLPFlowMaterializationInventoryReceipt"
                    ),
                    "live_opener_created_spool_input_count": 18,
                    "all_spool_inputs_must_be_live_and_opener_created": True,
                    "per_layer_semantic_replay": (
                        "exact_float64_semantic_replay_then_float32_bit_"
                        "equality_then_LayerMLPFlowRows_receipt"
                    ),
                    "A_partition_is_rebuilt_from_live_scalar_axes": True,
                    "source_fingerprint_is_derived_from_typed_predecessor_"
                    "runtime_qualification_and_session": True,
                    "parser_only_receipt_cannot_authorize_compile": True,
                    "selected_fit_source_row_receipts_must_match_inventory": (
                        True
                    ),
                },
                "fit_row_count": 10548,
                "validation_row_count": 14184,
                "total_row_count": 24732,
                "fit_validation_masks_disjoint_and_exhaustive": True,
                "partition_masks_regenerated_from_the_frozen_scalar_manifest_"
                "without_tensor_reads": True,
                "same_partition_receipt_and_derived_fit_validation_digests_"
                "required_for_all_324_candidates": True,
                "layer_specific_x_v_T_tensor_and_fit_receipts_remain_distinct": (
                    True
                ),
                "production_compile_requires_original_live_fitter_authority_"
                "for_every_selected_candidate": True,
                "cached_prefit_atlas_requires_original_live_atlas_fitter_"
                "authority": True,
                "production_M_greater_than_1_requires_original_live_split_"
                "evaluator_authority": True,
                "capture_materialization_atlas_fit_split_and_inventory_direct_"
                "construction_copy_deepcopy_replace_or_parser_replay_cannot_"
                "recover_live_authority": True,
            },
            "deterministic_fit_protocol": {
                "device_and_dtype": "CPU_float64_only",
                "atlas_cache_key": "physical_layer_chart_count_tangent_rank",
                "one_atlas_is_shared_across_M_for_fixed_layer_C_R": True,
                "public_atlas_and_candidate_subset_hash_role": (
                    "A_full_fit_atlas_and_optimizer"
                ),
                "supplied_atlas_requires_original_live_atlas_fitter_"
                "authority": True,
                "one_original_live_atlas_is_reusable_across_M_1_2_4": True,
                "C1_center": "family_example_row_weighted_state_mean",
                "C2_center_initialization": (
                    "center0_weighted_mean_then_center1_first_argmax_of_"
                    "w_times_squared_distance_to_center0"
                ),
                "C2_lloyd_assignment": "nearest_center_squared_Euclidean",
                "C2_lloyd_update": "weighted_mean",
                "C2_lloyd_pass_count": 12,
                "C2_final_chart_order": "lexicographic_center_digest",
                "per_chart_basis": (
                    "leading_R_eigenvectors_of_weighted_centered_state_"
                    "covariance_with_canonical_signs"
                ),
                "pre_optimizer_support_gates": [
                    "every_hard_chart_count_at_least_R_plus_1",
                    "every_weighted_chart_covariance_numerical_rank_at_least_R",
                ],
                "C1_tau": "serialized_1.0_frozen",
                "C2_tau_initialization": (
                    "weighted_median_positive_absolute_difference_of_the_two_"
                    "normal_squared_distances_with_float64_scale_floor"
                ),
                "C2_tau_parameterization": "exp_log_tau",
                "C2_tau_bounds": (
                    "initial_tau_div_16_through_initial_tau_times_16"
                ),
                "generator_initialization": (
                    "chart_weighted_ridge_affine_tangent_velocity_fit"
                ),
                "ridge_lambda": (
                    "1e-6_times_trace_of_coordinate_Gram_divided_by_R_plus_1"
                ),
                "M_greater_than_1_initialization": (
                    "deterministic_seeded_generator_perturbations_and_small_"
                    "affine_router_perturbations"
                ),
                "M1_router": "forced_exact_singleton_softmax_boundary",
                "optimizer": "torch_Adam",
                "optimizer_id": FIT_OPTIMIZER_ID,
                "fit_policy_sha256": PRODUCTION_FIT_POLICY_SHA256,
                "optimizer_steps": 256,
                "minibatch_size": 512,
                "learning_rate": 0.003,
                "adam_betas": [0.9, 0.99],
                "adam_epsilon": 1e-8,
                "gradient_clip_norm": 5.0,
                "gradient_clip_grouping_id": GRADIENT_CLIP_GROUPING_ID,
                "non_tau_parameters_share_one_global_L2_clip": True,
                "C2_chart_temperature_uses_an_independent_scalar_clip": True,
                "C1_chart_temperature_is_frozen_and_absent_from_optimizer": (
                    True
                ),
                "seed_base": 2026081000,
                "mode_temperature": 1.0,
                "loss_coefficients": {
                    "teacher_velocity_fidelity": 1.0,
                    "conditional_mode_load_balance": 0.02,
                    "conditional_mode_sparsity": 0.001,
                    "within_chart_mode_velocity_distinctness": 0.01,
                },
                "mode_regularizers_use_q_detached_and_cannot_fit_tau": True,
                "only_teacher_velocity_fidelity_may_fit_C2_tau": True,
                "regularizer_gradient_magnitude_cannot_rescale_tau_update": (
                    True
                ),
                "optimizer_and_atlas_read_A_fit_rows_only": True,
                "A_validation_is_opened_only_after_candidate_parameters_are_"
                "frozen": True,
            },
            "base_validation_gates": {
                "all_values_finite": True,
                "velocity_NRMSE_at_most": 0.25,
                "aggregate_velocity_cosine_at_least": 0.95,
                "every_family_velocity_NRMSE_at_most": 0.35,
            },
            "multichart_only_gates": {
                "applies_exactly_when_C_greater_than_1": True,
                "each_chart_soft_mass_at_least": 0.05,
                "chart_mass_is_observed_not_balanced_or_hardened": True,
            },
            "multimode_only_gates": {
                "applies_exactly_when_M_greater_than_1": True,
                "validation_SSE_improvement_over_same_layer_C_R_M1_at_least": (
                    0.05
                ),
                "each_chart_conditional_mode_mass_at_least": 0.05,
                "each_chart_state_mode_mutual_information_nats_at_least": (
                    0.05
                ),
                "minimum_within_each_chart_pairwise_velocity_separation_"
                "ratio_at_least": 0.05,
                "minimum_split_matched_prediction_cosine_at_least": 0.80,
                "pooled_split_hard_route_adjusted_rand_index_at_least": 0.60,
            },
            "multimode_statistic_definitions": {
                "chart_soft_mass": (
                    "Q_i=sum_n(w_n*q_ni),finite,nonnegative,sum_i(Q_i)=1;"
                    "Q_is_never_balanced_or_hardened"
                ),
                "conditional_mode_mass": (
                    "U_ik=sum_n(w_n*q_ni*pi_nik)/Q_i,finite_simplex_per_"
                    "chart,sum_k(U_ik)=1"
                ),
                "chart_conditional_state_mode_MI_nats": (
                    "H(U_i)-sum_n(w_n*q_ni/Q_i)*H(pi_ni),natural_log,"
                    "0log0=0"
                ),
                "within_chart_velocity_separation": (
                    "for_every_chart_i_minimum_m_less_k_RMS(G_im-G_ik)_"
                    "over_RMS(v_T)"
                ),
                "split_refits": (
                    "two_fixed_disjoint_A_fit_halves_refit_without_using_A_"
                    "validation_targets"
                ),
                "split_refit_scalar_closure": {
                    "base_fit_row_count": 10548,
                    "partition_row_counts": [5274, 5274],
                    "base_fit_example_count": 24,
                    "partition_example_counts": [12, 12],
                    "intersection_row_count": 0,
                    "union_equals_base_fit": True,
                    "base_and_union_mask_digests_must_match": True,
                    "all_split_mask_catalog_fit_and_model_digest_pairs_must_"
                    "be_distinct": True,
                    "strict_production_replay_regenerates_checkerboard_masks_"
                    "from_shared_A_partition_manifest": True,
                },
                "split_matched_prediction_cosine": (
                    "aggregate_cosine_between_the_two_complete_predicted_"
                    "velocity_fields_on_untouched_A_validation_rows"
                ),
                "split_hard_route_ARI": (
                    "Hubert_Arabie_ARI_of_pooled_A_validation_hard_joint_"
                    "chart_times_M_plus_mode_routes;label_permutation_"
                    "invariance_requires_no_manual_matching"
                ),
                "M1_boundary_values": {
                    "conditional_mass_per_chart": [1.0],
                    "mutual_information_nats_per_chart": 0.0,
                    "separation_ratio_per_chart": 0.0,
                    "split_matched_prediction_cosine": 1.0,
                    "split_hard_route_ARI": 1.0,
                    "multimode_thresholds_bypassed": True,
                },
            },
            "selection_rule_per_layer": (
                "among_all_passing_candidates_minimize_the_frozen_tie_key"
            ),
            "selection_tie_key": [
                "serialized_parameter_count",
                "validation_velocity_NRMSE",
                "chart_count_numeric_ascending",
                "mode_count_numeric_ascending",
                "tangent_rank_numeric_ascending",
            ],
            "failed_candidates_cannot_be_repaired_retuned_or_restarted": True,
            "one_candidate_must_select_for_each_of_18_layers_before_B3": True,
            "any_layer_without_a_passing_candidate_is_a_terminal_A_negative": (
                True
            ),
            "selected_artifacts_and_all_fit_coefficients_frozen_before_B3": True,
        },
        "fresh_panel_firewall": {
            "ordered_roles": list(ROLE_ORDER),
            "B3_C3_D3_and_reserve_must_be_new_and_frozen_before_preflight": (
                True
            ),
            "B3_C3_D3_and_reserve_must_be_disjoint_from_every_v1_v2_corpus": (
                True
            ),
            "B3_C3_D3_and_reserve_must_be_pairwise_semantic_lineage_"
            "disjoint": True,
            "existing_v2_D_is_not_repurposed": True,
            "existing_v2_reserve_is_not_repurposed_or_opened": True,
            "future_manifest_path": str(FUTURE_PANEL_MANIFEST),
            "future_manifest_file_sha256": PINNED_PANEL_MANIFEST_FILE_SHA256,
            "panel_manifest_frozen_in_current_source": panel_ready,
            "all_role_raw_hashes_verified_before_semantic_access": True,
            "B3_opens_only_after_all_18_A_selections_are_frozen": True,
            "C3_opens_only_after_B3_terminal_receipt_is_frozen": True,
            "D3_opens_only_after_C3_terminal_receipt_is_frozen": True,
            "reserve_remains_semantically_unopened": True,
            "later_role_early_access_invalidates_the_run": True,
        },
        "B3_teacher_forced_layerwise": {
            "role": DISCOVERY_ROLE,
            "heldout_evaluation_policy_sha256": (
                "4a6f6379571f348fc7a0a477c83ffdf52ae119903cb6e8859a12c4f66e4e9266"
            ),
            "evaluation_is_layerwise_on_native_teacher_forced_inputs": True,
            "each_layer_uses_its_A_frozen_candidate_without_refit": True,
            "fidelity_accumulation": (
                "unweighted_complete_H640_rows_with_additive_SSE_teacher_"
                "energy_prediction_energy_and_prediction_teacher_dot"
            ),
            "routing_accumulation": (
                "unit_weight_per_evaluated_row_with_additive_q_q_times_pi_"
                "and_q_weighted_router_entropy"
            ),
            "all_values_finite": True,
            "every_layer_velocity_NRMSE_at_most": 0.30,
            "every_layer_aggregate_velocity_cosine_at_least": 0.90,
            "every_layer_every_family_velocity_NRMSE_at_most": 0.40,
            "selected_M_greater_than_1_additional_gates": {
                "each_chart_conditional_mode_mass_at_least": 0.03,
                "each_chart_state_mode_mutual_information_nats_at_least": (
                    0.03
                ),
            },
            "B3_may_not_refit_reselect_or_change_thresholds": True,
            "all_18_layers_must_pass_before_C3": True,
            "failure_is_a_valid_heldout_negative_result": True,
        },
        "C3_full_stack_source_free": {
            "role": CONFIRMATION_ROLE,
            "heldout_evaluation_policy_sha256": (
                "4a6f6379571f348fc7a0a477c83ffdf52ae119903cb6e8859a12c4f66e4e9266"
            ),
            "all_18_selected_MLP_replacements_execute_together": True,
            "strict_typed_passing_B3_receipt_required": True,
            "B3_C3_source_panel_protocol_and_selection_must_match": True,
            "no_native_MLP_teacher_inputs_or_intermediate_fallback": True,
            "candidate_states_are_recursively_source_free": True,
            "evidence_unit": "one_authenticated_scalar_record_per_example",
            "native_and_candidate_scored_positions_must_be_identical": True,
            "delta_NLL_definition": (
                "sum_over_positions(candidate_NLL_minus_native_NLL)"
            ),
            "KL_definition": (
                "sum_over_positions_and_full_vocabulary_of_p_native_times_"
                "log_p_native_over_p_candidate"
            ),
            "top1_definition": (
                "same_argmax_with_lowest_vocabulary_index_tie_break"
            ),
            "metric_denominator": "sum_scored_next_token_positions",
            "all_values_finite": True,
            "pooled_point_gates": {
                "delta_NLL_nats_per_token_at_most": 0.02,
                "native_to_candidate_KL_per_token_at_most": 0.02,
                "native_top1_agreement_at_least": 0.95,
            },
            "every_family_point_gates": {
                "delta_NLL_nats_per_token_at_most": 0.04,
                "native_to_candidate_KL_per_token_at_most": 0.04,
                "native_top1_agreement_at_least": 0.90,
            },
            "hard_long_strata_repeat_pooled_point_gates": True,
            "hard_long_failure_is_a_veto": True,
            "C3_may_not_refit_reselect_scale_or_repair": True,
            "all_gates_are_a_strict_conjunction": True,
            "failure_is_a_valid_full_stack_negative_result": True,
        },
        "D3_final_validation": {
            "role": FINAL_ROLE,
            "heldout_evaluation_policy_sha256": (
                "4a6f6379571f348fc7a0a477c83ffdf52ae119903cb6e8859a12c4f66e4e9266"
            ),
            "bootstrap_policy_sha256": (
                "fdfc8be6463e959f02cc17ca5b17168405d3bca987d57e9a9ed3d621063cd026"
            ),
            "full_stack_source_free_candidate_is_unchanged_from_C3": True,
            "confidence_unit": "whole_example",
            "confidence_level": 0.95,
            "bootstrap_is_approximate_not_distribution_free": True,
            "bootstrap_noncoverage_reason": "delta_NLL_and_KL_are_unbounded",
            "bootstrap_replicate_count": 10000,
            "bootstrap_seed_sha256": (
                "afa41f02e5b3e192ba9d09b6a3dee90ae0fe8bb1e07d4f4ee0ba282f4abeb296"
            ),
            "bootstrap_sampler": "SHA256_counter_uint64_rejection_v1",
            "bootstrap_record_order": "ascending_example_identity_sha256",
            "bootstrap_stratification": (
                "all_12_nonempty_family_id_times_hard_long_cells"
            ),
            "bootstrap_draw_count_per_cell": "observed_cell_example_count",
            "bootstrap_standard_error": (
                "sample_SD_across_whole_example_bootstrap_replicates"
            ),
            "bootstrap_studentization": "fixed_coordinate_standard_error",
            "bootstrap_max_T": (
                "one_shared_critical_across_pooled_every_family_hard_long_"
                "times_delta_NLL_KL_top1_error"
            ),
            "bootstrap_quantile_rank": (
                "ceil((10000+1)*0.95)_clamped_to_10000_one_based"
            ),
            "zero_SE_rule": (
                "allowed_only_when_every_centered_replicate_is_exactly_zero"
            ),
            "one_sided_pooled_bounds": {
                "delta_NLL_nats_per_token_upper_bound_at_most": 0.02,
                "native_to_candidate_KL_per_token_upper_bound_at_most": 0.02,
                "native_top1_agreement_lower_bound_at_least": 0.95,
            },
            "every_family_point_gates_equal_C3": True,
            "hard_long_point_gates_equal_C3": True,
            "all_values_finite": True,
            "native_MLP_source_fallback_alias_or_hidden_call_forbidden": True,
            "profiler_and_module_trace_must_prove_no_native_MLP_call": True,
            "exact_formula_parameter_count_must_be_lower": True,
            "actual_serialized_full_model_bytes_must_be_lower": True,
            "native_and_compiled_byte_counts_must_use_the_same_frozen_"
            "serialization_protocol": True,
            "D3_may_not_refit_reselect_scale_or_repair": True,
            "all_gates_are_a_strict_conjunction": True,
            "failure_is_a_valid_final_negative_result": True,
        },
        "claim_taxonomy": {
            "allowed_after_A_only": [
                "development_selected_local_flow_candidates",
                "no_fresh_generalization_or_compression_claim",
            ],
            "allowed_after_B3_only": [
                "teacher_forced_layerwise_heldout_fidelity",
                "no_source_free_full_stack_compression_claim",
            ],
            "allowed_only_after_C3_and_D3_all_gates": [
                "all_18_token_local_MLP_maps_replaced_source_free",
                "heldout_behavior_preserved_under_frozen_thresholds",
                "exact_reported_parameter_reduction",
                "actual_reported_serialized_byte_reduction",
            ],
            "never_authorized_without_separate_frozen_evidence": [
                "attention_or_lexical_compression",
                "latency_improvement",
                "MAC_reduction_without_exact_accounting_and_execution_proof",
                "architecture_independent_generator_ontology",
            ],
            "latency_claim_requires_separate_frozen_backend_benchmark": True,
        },
        "implementation_readiness": {
            "setup_is_prospective": True,
            "default_setup_artifact_published": False,
            "check_authenticates_without_authorizing_execution": True,
            "panel_manifest_path": str(FUTURE_PANEL_MANIFEST),
            "panel_manifest_file_sha256": PINNED_PANEL_MANIFEST_FILE_SHA256,
            "panel_manifest_frozen": panel_ready,
            "implementation_qualification_path": str(
                FUTURE_IMPLEMENTATION_QUALIFICATION
            ),
            "implementation_qualification_file_sha256": (
                PINNED_IMPLEMENTATION_QUALIFICATION_FILE_SHA256
            ),
            "implementation_qualification_frozen": implementation_ready,
            "qualification_must_bind_unchanged_setup_sha256": True,
            "qualification_must_bind_worker_and_full_dependency_source_closure": (
                True
            ),
            "qualification_must_bind_model_free_adversarial_tests": True,
            "preflight_authorized": panel_ready and implementation_ready,
            "run_authorized": panel_ready and implementation_ready,
            "model_tensor_or_panel_semantic_access_before_successful_preflight_"
            "forbidden": True,
        },
        "production_compiler_and_bundle_loader": {
            "production_compile_recomputes_source_model_fingerprint": True,
            "production_compile_recomputes_source_execution_fingerprint": True,
            "production_compile_requires_entire_source_model_in_eval_mode": True,
            "bundle_loader_recomputes_compiled_execution_fingerprint": True,
        },
        "output_contract": {
            "report": str(DEFAULT_REPORT_OUTPUT),
            "coefficient_bundle": str(DEFAULT_BUNDLE_OUTPUT),
            "outputs_are_distinct_from_v1_and_v2": True,
            "report_is_published_last_as_commit_marker": True,
            "large_numeric_payloads_are_content_addressed_out_of_line": True,
            "prompt_text_token_ids_teacher_velocities_and_raw_states_forbidden_"
            "from_report": True,
        },
        "setup_receipt": {
            "tensor_library_imported": False,
            "model_weights_loaded": False,
            "A_component_states_or_velocities_captured": False,
            "B3_semantically_opened": False,
            "C3_semantically_opened": False,
            "D3_semantically_opened": False,
            "reserve_semantically_opened": False,
            "candidate_fitted": False,
            "layer_candidate_selected": False,
            "compiled_model_written": False,
            "scientific_v3_result_present": False,
        },
    }


def build_setup() -> dict[str, object]:
    state = _build_state()
    state["setup_sha256"] = setup_sha256(state)
    return state


def validate_setup(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Gemma-3 layer-multimode setup must be a mapping")
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Gemma-3 layer-multimode setup is not canonical") from error
    if canonical_json_bytes(observed) != canonical_json_bytes(build_setup()):
        raise ValueError("Gemma-3 layer-multimode setup differs from source")
    return observed


def render_setup_bytes(value: Mapping[str, object] | None = None) -> bytes:
    observed = build_setup() if value is None else validate_setup(value)
    return (
        json.dumps(observed, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _regular_stat(path: Path, *, byte_cap: int) -> os.stat_result:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"required artifact is not a regular file: {path}")
    if before.st_size > byte_cap:
        raise ValueError(f"required artifact exceeds byte cap: {path}")
    return before


def _read_regular(path: Path, *, byte_cap: int = _MAX_SETUP_BYTES) -> bytes:
    before = _regular_stat(path, byte_cap=byte_cap)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"required artifact changed during open: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"required artifact was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"required artifact changed during read: {path}")
    return b"".join(chunks)


def _sha256_regular(path: Path, *, byte_cap: int) -> str:
    before = _regular_stat(path, byte_cap=byte_cap)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"required artifact changed during open: {path}")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"required artifact was truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"required artifact changed during read: {path}")
    return digest.hexdigest()


def load_setup(source: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    raw = _read_regular(Path(source))
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Gemma-3 layer-multimode setup JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("Gemma-3 layer-multimode setup must be a JSON object")
    return validate_setup(value)


def write_setup(destination: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = build_setup()
    payload = render_setup_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.lstat(temporary)
        os.link(temporary, path)
        published = True
        published_stat = path.lstat()
        if (
            stat.S_ISLNK(published_stat.st_mode)
            or not stat.S_ISREG(published_stat.st_mode)
            or (published_stat.st_dev, published_stat.st_ino)
            != (temporary_stat.st_dev, temporary_stat.st_ino)
        ):
            raise RuntimeError("layer-multimode setup publication identity drifted")
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _read_regular(path) != payload or load_setup(path) != value:
            raise RuntimeError("layer-multimode setup strict reload drifted")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            else:
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        raise
    return value


def _repository_root(repository_root: Path | str | None) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} path must be a string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path is unsafe")
    return path


def check_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
    verify_predecessor: bool = True,
) -> dict[str, object]:
    """Authenticate this prospective setup without authorizing execution."""

    path = Path(source)
    value = load_setup(path)
    if _read_regular(path) != render_setup_bytes(value):
        raise ValueError("Gemma-3 layer-multimode setup formatting drifted")
    if not verify_predecessor:
        return value
    root = _repository_root(repository_root)
    provenance = value["methodology_provenance"]
    predecessor_setup = root / _safe_relative(
        provenance["predecessor_v2_setup_path"],  # type: ignore[index]
        label="predecessor v2 setup",
    )
    predecessor_raw = _read_regular(predecessor_setup)
    if hashlib.sha256(predecessor_raw).hexdigest() != provenance[
        "predecessor_v2_setup_file_sha256"  # type: ignore[index]
    ]:
        raise ValueError("predecessor v2 setup file binding drifted")
    predecessor_value = _strict_json_loads(predecessor_raw)
    if (
        not isinstance(predecessor_value, Mapping)
        or predecessor_value.get("setup_sha256")
        != provenance["predecessor_v2_setup_sha256"]  # type: ignore[index]
    ):
        raise ValueError("predecessor v2 setup semantic binding drifted")
    predecessor_report = root / _safe_relative(
        provenance["predecessor_v2_report_path"],  # type: ignore[index]
        label="predecessor v2 report",
    )
    if _sha256_regular(
        predecessor_report, byte_cap=_MAX_PREDECESSOR_BYTES
    ) != provenance["predecessor_v2_report_file_sha256"]:  # type: ignore[index]
        raise ValueError("predecessor v2 report file binding drifted")
    return value


def preflight_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
    verify_predecessor: bool = True,
) -> dict[str, object]:
    """Fail before panel semantics, tensors, or model access until frozen."""

    value = check_setup(
        source,
        repository_root=repository_root,
        verify_predecessor=verify_predecessor,
    )
    readiness = value["implementation_readiness"]
    missing: list[str] = []
    if (
        readiness["panel_manifest_frozen"] is not True  # type: ignore[index]
        or readiness["panel_manifest_file_sha256"] is None  # type: ignore[index]
    ):
        missing.append("frozen fresh-panel manifest")
    if (
        readiness["implementation_qualification_frozen"]  # type: ignore[index]
        is not True
        or readiness["implementation_qualification_file_sha256"]  # type: ignore[index]
        is None
    ):
        missing.append("frozen implementation qualification")
    if (
        readiness["preflight_authorized"] is not True  # type: ignore[index]
        or readiness["run_authorized"] is not True  # type: ignore[index]
    ):
        missing.append("reviewed execution authorization")
    if missing:
        raise ExecutionNotAuthorizedError(
            "layer-multimode v3 execution is not authorized: " + ", ".join(missing)
        )
    raise ExecutionNotAuthorizedError(
        "readiness pins changed without adding their strict artifact validators"
    )


def authorize_run(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
    verify_predecessor: bool = True,
) -> dict[str, object]:
    """Expose only the static authorization barrier; this module has no runner."""

    return preflight_setup(
        source,
        repository_root=repository_root,
        verify_predecessor=verify_predecessor,
    )


@dataclass(frozen=True, slots=True)
class Gemma3LayerMultimodeSetup:
    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3LayerMultimodeSetup":
        return cls(build_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self.state))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.check:
        check_setup(arguments.output)
    elif arguments.preflight:
        preflight_setup(arguments.output)
    elif arguments.run:
        authorize_run(arguments.output)
    else:
        write_setup(arguments.output)
    return 0


__all__ = [
    "CANDIDATE_COUNT_PER_LAYER",
    "CHART_COUNT_CANDIDATES",
    "DEFAULT_BUNDLE_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "ExecutionNotAuthorizedError",
    "FORMAT_VERSION",
    "FUTURE_IMPLEMENTATION_QUALIFICATION",
    "FUTURE_PANEL_MANIFEST",
    "Gemma3LayerMultimodeSetup",
    "HIDDEN_WIDTH",
    "LAYER_COUNT",
    "MLP_INTERMEDIATE_WIDTH",
    "MODE_COUNT_CANDIDATES",
    "NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER",
    "PREDECESSOR_V2_REPORT",
    "PREDECESSOR_V2_REPORT_FILE_SHA256",
    "PREDECESSOR_V2_REPORT_SHA256",
    "SCHEMA",
    "SETUP_ID",
    "TANGENT_RANK_CANDIDATES",
    "authorize_run",
    "build_setup",
    "canonical_json_bytes",
    "check_setup",
    "load_setup",
    "main",
    "preflight_setup",
    "render_setup_bytes",
    "replacement_parameter_accounting",
    "setup_sha256",
    "validate_setup",
    "write_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
