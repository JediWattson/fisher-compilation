"""Freeze the A-informed, prospective-to-B/C chart-neighbor v2 protocol.

The v2 protocol replaces a fixed K=8 atlas with a deterministic descending
chart-capacity rule on fit-role A.  The rule is frozen before B or C is opened
and may use only intrinsic context geometry.  This setup module is
standard-library-only and contains no measurement or selected chart count.
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


SCHEMA = "fisher_graph.gemma3_chart_native_neighbor.setup.v2"
FORMAT_VERSION = 2
SETUP_ID = "gemma3-chart-native-neighbor-v2"
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v2.json"
)
PINNED_SETUP_SHA256 = (
    "f786a320c9a646f7141d1832913c11149297b09d3d01eca033acfe1fd9cb7f50"
)
PINNED_SETUP_FILE_SHA256 = (
    "2e6341417956ea3e2dd021657c951f87221078777be9d185c173d651afae4231"
)
PINNED_SELECTION_SOURCE_SHA256 = (
    "e414a95485eb5055292d1ae69c60e85295eb9c03f5f8d1bac56e1755023206a1"
)
PINNED_SELECTION_KERNEL_RECEIPT_SHA256 = (
    "133311b1c982ebc42cf98219faa8eed37c8b36b4348385c02f78edceebded9f6"
)
PINNED_A_DIAGNOSTIC_SETUP_SHA256 = (
    "bbb87b3314ce5dd317afe9d524e116ed6d53cf546f2f21cdd48bab9404608f32"
)
PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256 = (
    "34166b8f16f4f11ce058360e743fc29d3056a96fe942f9c0b01c36519d9e9b96"
)
_SETUP_DOMAIN = b"fisher_graph:gemma3-chart-native-neighbor-setup:v2\0"

V1_SETUP_PATH = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v1.json"
)
V1_SETUP_SHA256 = (
    "5ebc02aa5800add5d16f9b3f506224029adcf1fa257ac80a7d46155ce0a1e484"
)
V1_SETUP_FILE_SHA256 = (
    "d597c9210f0fddae71a514b1a1738612eda22e99978f94d109ef61b489fee042"
)
A_DIAGNOSTIC_SETUP_PATH = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-a-k8-occupancy-diagnostic-setup-v1.json"
)
A_DIAGNOSTIC_RESULT_PATH = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-a-k8-occupancy-diagnostic-v1.json"
)

FIT_ROLE = "A_chart_transport_fit"
DISCOVERY_ROLE = "B_neighbor_discovery"
CONFIRMATION_ROLE = "C_neighbor_confirmation"
RESPONSE_ROLE = "D_response_validation"
RESERVE_ROLE = "reserve_unopened"
ROLE_ORDER = (
    FIT_ROLE,
    DISCOVERY_ROLE,
    CONFIRMATION_ROLE,
    RESPONSE_ROLE,
    RESERVE_ROLE,
)

HIDDEN_WIDTH = 640
TANGENT_RANK = 16
CHART_COUNT_CANDIDATES = (8, 7, 6, 5, 4, 3, 2, 1)
MINIMUM_HARD_ROWS = TANGENT_RANK + 1
MINIMUM_BALANCED_HARD_OCCUPANCY = 0.005
MODE_COUNT = 38
ORDERED_PAIR_COUNT = MODE_COUNT * (MODE_COUNT - 1)
MUTUAL_NEIGHBOR_K = 2
BOOTSTRAP_SEED = 7_901
BOOTSTRAP_RESAMPLE_COUNT = 10_000
BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY = 0.80
BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION = 0.01
LOFO_MINIMUM_MUTUAL_COUNT = 5

DEFAULT_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v2.json"
)
DEFAULT_SVG_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v2.svg"
)
DEFAULT_ATLAS_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v2.atlas.bundle"
)
FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-v2-implementation-qualification-v1.json"
)

# These codes are geometry-dependent, anticipated failures.  Only these codes
# may reject a candidate and continue to the next smaller K.  Authentication,
# source, resource, I/O, backend, or unexpected exceptions are never caught by
# the selector and terminate the run.
INTRINSIC_REJECT_CODES = (
    "CENTER_INITIALIZATION_DEGENERATE",
    "LLOYD_EMPTY_CHART",
    "HARD_COUNT_BELOW_RANK_PLUS_ONE",
    "BALANCED_HARD_MASS_BELOW_MINIMUM",
    "RADIUS_CONTRACT_FAILED",
    "SOFT_OCCUPANCY_CONTRACT_FAILED",
    "COVARIANCE_CONTRACT_FAILED",
    "EIGENGAP_ORIENTATION_UNDEFINED",
    "SPD_CONTRACT_FAILED",
)
ADMISSIBLE_STATUS = "ADMISSIBLE"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def setup_sha256(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("setup_sha256", None)
    return _digest(_SETUP_DOMAIN, unsigned)


def _resource_rows() -> list[dict[str, int]]:
    # Exact production formulas, evaluated without importing torch.
    rows_per_mode = 24_732
    result: list[dict[str, int]] = []
    for chart_count in CHART_COUNT_CANDIDATES:
        feature_width = chart_count * (TANGENT_RANK + 1)
        applied_per_mode = rows_per_mode * (
            feature_width * 8 + chart_count * 8 + HIDDEN_WIDTH * 4 + 4 * 8
        )
        coefficient_bytes = (
            ORDERED_PAIR_COUNT * feature_width * HIDDEN_WIDTH * 8
        )
        result.append(
            {
                "chart_count": chart_count,
                "source_feature_width": feature_width,
                "applied_store_payload_bytes_per_mode": applied_per_mode,
                "applied_store_payload_bytes_per_role": (
                    applied_per_mode * MODE_COUNT
                ),
                "all_ordered_coefficient_bytes": coefficient_bytes,
            }
        )
    return result


def _build_state() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "methodology_provenance": {
            "predecessor_v1_setup_path": str(V1_SETUP_PATH),
            "predecessor_v1_setup_sha256": V1_SETUP_SHA256,
            "predecessor_v1_setup_file_sha256": V1_SETUP_FILE_SHA256,
            "v1_fixed_chart_count": 8,
            "v1_terminal_role": FIT_ROLE,
            "v1_terminal_gate": "minimum_balanced_hard_occupancy_0_005",
            "v1_all_charts_had_at_least_17_hard_rows": True,
            "v1_at_least_one_chart_had_balanced_mass_below_0_005": True,
            "v1_exact_occupancy_vectors_not_retained": True,
            "v1_atlas_frozen": False,
            "v1_transports_fitted": False,
            "v1_graph_defined": False,
            "v1_B_C_D_reserve_semantically_unopened": True,
            "v1_dispatch_record_jsonl_SHA256": (
                "2862739f786553412c003a305dd505e54afe720090ae3a7f5523224a592d93fa"
            ),
            "v1_terminal_record_jsonl_SHA256": (
                "3eab4751d8f72183c6f4420289405856f212ddf07388954e5a2c25e9594116bc"
            ),
            "v1_terminal_timestamp_UTC": "2026-08-10T11:30:39.349Z",
            "external_journal_path_is_not_an_executable_dependency": True,
            "v2_is_post_A_informed": True,
            "v2_is_prospective_to_B_and_C": True,
            "A_is_fit_development_not_fresh_confirmation": True,
            "whole_protocol_preregistered_from_scratch_claimed": False,
            "A_only_K8_diagnostic_setup_path": str(
                A_DIAGNOSTIC_SETUP_PATH
            ),
            "A_only_K8_diagnostic_setup_sha256": (
                PINNED_A_DIAGNOSTIC_SETUP_SHA256
            ),
            "A_only_K8_diagnostic_setup_file_sha256": (
                PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256
            ),
            "A_only_K8_diagnostic_result_path": str(
                A_DIAGNOSTIC_RESULT_PATH
            ),
            "A_only_K8_diagnostic_is_nonselecting": True,
            "A_only_K8_diagnostic_result_is_required_prior_evidence_for_"
            "workflow": True,
            "A_only_K8_diagnostic_result_must_strict_parse_before_v2": True,
            "A_only_K8_diagnostic_required_outcome": (
                "BALANCED_HARD_MASS_BELOW_MINIMUM"
            ),
            "A_only_K8_diagnostic_required_terminal_phase": (
                "final_hard_mass_gate"
            ),
            "A_only_K8_diagnostic_result_cannot_select_or_tune_K": True,
            "v2_independently_reproduces_K8_before_descending": True,
            "diagnostic_and_v2_K8_semantic_agreement_required": True,
            "diagnostic_K8_disagreement_is_terminal_before_B": True,
            "diagnostic_result_SHA_is_bound_by_live_v2_receipts_not_setup": (
                True
            ),
        },
        "role_firewall": {
            "ordered_roles": list(ROLE_ORDER),
            "A_may_fit_and_select_intrinsic_chart_geometry": True,
            "B_candidate_nomination_only": True,
            "C_confirmation_only": True,
            "D_response_validation_non_authorizing": True,
            "reserve_unopened": True,
            "selected_atlas_and_A_transports_frozen_before_B_open": True,
            "B_C_D_reserve_may_not_select_or_refit_chart_count": True,
            "one_selected_K_reaches_B_and_C": True,
            "running_B_or_C_for_multiple_K_values_invalidates_confirmation": (
                True
            ),
        },
        "chart_capacity_selection": {
            "chart_count_symbol": "K_chart",
            "candidate_order": list(CHART_COUNT_CANDIDATES),
            "candidate_range_is_exhaustive_integer_descent_8_through_1": True,
            "selection_rule": (
                "first_descending_candidate_completing_the_entire_frozen_"
                "intrinsic_context_only_atlas_contract"
            ),
            "selected_K_is_live_A_receipt_owned_not_setup_owned": True,
            "same_A_raw_spools_used_read_only_for_every_attempt": True,
            "stop_after_first_admissible_candidate": True,
            "lower_candidates_unattempted_after_selection": True,
            "no_candidate_admissible_aborts_before_B": True,
            "context_values_and_balanced_row_weights_are_the_only_selection_"
            "inputs": True,
            "generalized_contributions_forbidden_for_selection": True,
            "comparison_values_forbidden_for_selection": True,
            "transport_loss_or_fit_quality_forbidden_for_selection": True,
            "neighbor_distance_edge_count_or_topology_forbidden_for_selection": (
                True
            ),
            "B_C_bootstrap_LOFO_or_response_results_forbidden_for_selection": (
                True
            ),
            "alternate_seed_forbidden": True,
            "threshold_tuning_forbidden": True,
            "rank_tuning_forbidden": True,
            "candidate_attempts_stop_at_first_failed_intrinsic_gate": True,
            "early_N_less_than_K_times_rank_plus_one_abort_forbidden": True,
            "low_row_candidate_infeasibility_is_classified_by_center_"
            "initialization_Lloyd_or_final_hard_count_gate": True,
            "smaller_K_descent_continues_after_typed_low_row_reject": True,
            "malformed_or_zero_row_input_remains_terminal": True,
            "intrinsic_reject_codes_in_gate_order": list(
                INTRINSIC_REJECT_CODES
            ),
            "only_typed_K_dependent_intrinsic_rejects_descend": True,
            "full_pass_status": ADMISSIBLE_STATUS,
            "selection_receipt_outcomes": [
                "A_ATLAS_SELECTED",
                "A_ATLAS_NO_ADMISSIBLE_K",
            ],
            "run_terminal_statuses": [
                "A_ATLAS_SELECTED",
                "A_ATLAS_NO_ADMISSIBLE_K",
                "A_ATLAS_SELECTION_ENGINEERING_ABORT",
                "A_ATLAS_FREEZE_INTEGRITY_ABORT",
                "A_POST_SELECTION_TRANSPORT_ABORT",
            ],
            "unexpected_exception_is_terminal": True,
            "authentication_source_resource_IO_and_backend_failures_are_"
            "terminal": True,
            "selected_atlas_serialization_application_or_transport_failure_"
            "is_terminal_and_cannot_resume_K_search": True,
            "selected_atlas_hash_or_integrity_failure_is_terminal": True,
        },
        "unchanged_intrinsic_atlas_contract": {
            "hidden_width": HIDDEN_WIDTH,
            "tangent_rank": TANGENT_RANK,
            "lloyd_iteration_count": 12,
            "membership_temperature": 1.0,
            "radius_floor_fraction": 1.0e-8,
            "variance_floor_fraction": 1.0e-8,
            "minimum_hard_rows_per_chart": MINIMUM_HARD_ROWS,
            "minimum_balanced_hard_occupancy_weight": (
                MINIMUM_BALANCED_HARD_OCCUPANCY
            ),
            "occupancy_equality_at_0_005_passes": True,
            "eigenvalue_tie_relative_tolerance": 1.0e-12,
            "every_selected_eigenvalue_to_following_gap_required": True,
            "canonical_basis_sign_rule_unchanged": True,
            "full_rank_SPD_tangent_plus_isotropic_orthogonal_metric_required": (
                True
            ),
            "transport_relative_ridge": 1.0e-3,
            "ridge_scale_floor": 1.0e-12,
            "transport_fisher_energy_floor": 1.0e-12,
            "fit_pool_weighting": (
                "equal_mode_then_equal_family_then_equal_example_then_"
                "equal_row_within_example"
            ),
            "fit_pool_rows_per_mode": 24_732,
            "fit_pool_total_rows": 939_816,
            "fit_pool_balanced_mass_total": 1.0,
            "deterministic_weighted_farthest_initialization": True,
            "fixed_float64_CPU_reduction_order": True,
        },
        "K1_boundary": {
            "K1_is_mathematically_admissible_with_rank16": True,
            "reason": (
                "rank16_is_within_each_H640_chart_not_across_chart_count"
            ),
            "hard_row_requirement": 17,
            "membership_is_identically_one": True,
            "source_feature_width": 17,
            "allowed_result_wording": (
                "single_global_rank16_chart_transport_geometry"
            ),
            "multi_chart_local_chart_mixture_or_routing_claim_forbidden": (
                True
            ),
            "mode_neighbor_graph_remains_defined_because_modes_not_charts_"
            "are_graph_nodes": True,
        },
        "dynamic_transport_contract": {
            "source_feature_formula": "F=K_chart*(tangent_rank+1)=17*K_chart",
            "source_feature": "concat_c(q_source_c*[1,z_source_c])",
            "ridge_scale_formula": "trace(G)/F",
            "solve_formula": "Beta=solve(G+ridge*I_F,X^T*W*Y)",
            "all_F_coefficients_penalized": True,
            "separate_intercept": False,
            "membership_shape": "rows_by_selected_K",
            "tangent_coordinate_shape": "rows_by_selected_K_by_16",
            "ordered_pair_count": ORDERED_PAIR_COUNT,
            "resource_table": _resource_rows(),
            "selection_streaming_resource_config": {
                "row_chunk_size": 256,
                "atlas_chunk_size": 256,
                "max_resident_bytes": 8 * 1024**3,
                "max_artifact_bytes": 4 * 1024**3,
                "exact_payload_bound_by_traced_selection_kernel_receipt": (
                    True
                ),
            },
            "K8_is_preregistered_maximum_storage_envelope": True,
            "selected_K_exact_resource_receipt_required": True,
            "raw_A_spool_bytes_unchanged_by_K": True,
        },
        "selection_receipt": {
            "candidate_order_and_gate_order_required": True,
            "A_panel_row_context_weight_and_spool_catalog_hashes_required": (
                True
            ),
            "source_kernel_resource_config_and_runtime_hashes_required": True,
            "exact_traced_selection_kernel_receipt_required": (
                PINNED_SELECTION_KERNEL_RECEIPT_SHA256
            ),
            "attempted_prefix_and_unattempted_suffix_required": True,
            "per_attempt_typed_status_and_first_failed_gate_required": True,
            "seeds_and_all_12_Lloyd_pass_receipts_required": True,
            "per_chart_final_hard_counts_and_balanced_masses_required": True,
            "per_chart_radius_soft_occupancy_eigen_gap_scale_and_SPD_"
            "diagnostics_required_when_reached": True,
            "post_occupancy_gate_sufficient_statistics": {
                "encoding": (
                    "lowercase_big_endian_binary64_bits_for_observations_"
                    "bounds_and_signed_margins_including_nonfinite_classes"
                ),
                "gate_evidence_required_fields": [
                    "schema",
                    "gate_id",
                    "predicate_id",
                    "status",
                    "chart_ordinal",
                    "component_ordinal",
                    "global_row_ordinal",
                    "observed_binary64",
                    "bound_binary64",
                    "signed_margin_binary64",
                    "receipt_sha256",
                ],
                "canonical_predicate_order": [
                    "finite_and_positive_radius_squared",
                    "finite_membership",
                    "nonnegative_membership",
                    "membership_simplex_absolute_residual",
                    "finite_and_positive_soft_occupancy",
                    "finite_symmetric_covariance",
                    "strict_relative_eigengap",
                    "basis_orthonormality",
                    "finite_and_positive_tangent_variance",
                    "finite_and_positive_orthogonal_alpha",
                ],
                "PASS_evidence_required_for_every_reached_predicate": True,
                "reject_retains_exact_prior_PASS_prefix_plus_terminal_"
                "evidence": True,
                "unreached_gate_evidence_forbidden": True,
                "radius_contract": {
                    "predicate_id": "finite_and_positive_radius_squared",
                    "cardinality_when_reached": "K_chart",
                    "order": "chart_ordinal_0_through_K_chart_minus_1",
                    "observed_binary64": ["published_radius_squared"],
                    "bound_binary64": "0000000000000000",
                    "signed_margin": "published_radius_squared_minus_zero",
                    "nonfinite_classification_required": True,
                    "radius_numerator_unfloored_value_and_floor_not_exposed_"
                    "or_claimed": True,
                },
                "soft_occupancy_contract": {
                    "early_scan_terminal_predicates": [
                        {
                            "predicate_id": "finite_membership",
                            "status": "NONFINITE",
                        },
                        {
                            "predicate_id": "nonnegative_membership",
                            "status": "NEGATIVE",
                        },
                        {
                            "predicate_id": (
                                "membership_simplex_absolute_residual"
                            ),
                            "status": "RESIDUAL_EXCEEDS_BOUND",
                        },
                    ],
                    "finite_membership_PASS_rows_not_emitted": True,
                    "complete_scan_canonical_inventory": [
                        "K_chart_per_chart_minimum_nonnegative_membership",
                        "one_global_worst_membership_simplex_residual",
                        "K_chart_per_chart_positive_soft_occupancy_mass",
                    ],
                    "minimum_membership_is_per_chart_not_global": True,
                    "minimum_membership_row_fields": [
                        "chart_ordinal",
                        "global_row_ordinal",
                        "minimum_finite_membership",
                        "zero_bound",
                        "membership_minus_zero_margin",
                    ],
                    "minimum_membership_bound_binary64": (
                        "0000000000000000"
                    ),
                    "simplex_row_fields": [
                        "global_row_ordinal",
                        "row_membership_sum",
                        "absolute_residual_from_one",
                        "effective_bound",
                        "bound_minus_residual_margin",
                    ],
                    "simplex_rtol_binary64": "3d719799812dea11",
                    "simplex_atol_binary64": "3d719799812dea11",
                    "simplex_effective_bound_binary64": "3d819799812dea11",
                    "soft_occupancy_row_fields": [
                        "chart_ordinal",
                        "soft_occupancy_mass",
                        "zero_bound",
                        "mass_minus_zero_margin",
                    ],
                    "soft_occupancy_bound_binary64": "0000000000000000",
                },
                "covariance_contract": {
                    "predicate_id": "finite_symmetric_covariance",
                    "cardinality_when_reached": "K_chart",
                    "order": "chart_ordinal_0_through_K_chart_minus_1",
                    "covariance_bytes_remain_original_and_unsymmetrized": True,
                    "decisive_pair_rule": (
                        "minimum_signed_margin_pair_on_PASS_else_first_"
                        "failing_or_nonfinite_pair"
                    ),
                    "observed_binary64": [
                        "C_ij",
                        "C_ji",
                        "absolute_pair_residual",
                        "scale_max_1_abs_Cij_abs_Cji",
                    ],
                    "component_ordinal": "i_times_hidden_width_plus_j",
                    "relative_tolerance_binary64": "3d719799812dea11",
                    "bound_formula": (
                        "1e_minus_12_times_max_1_abs_Cij_abs_Cji"
                    ),
                    "signed_margin": "bound_minus_absolute_pair_residual",
                    "nonfinite_pair_classification_required": True,
                    "maximum_covariance_residual_not_exposed_or_claimed": True,
                },
                "eigengap_orientation_contract": {
                    "predicate_id": "strict_relative_eigengap",
                    "selected_atlas_cardinality": "K_chart_times_16",
                    "order": "chart_major_then_eigen_ordinal_0_through_15",
                    "observed_binary64": [
                        "selected_eigenvalue_lambda_j",
                        "following_eigenvalue_lambda_j_plus_1",
                        "absolute_gap",
                        "scale_max_1_abs_selected_abs_following",
                    ],
                    "component_ordinal": "selected_eigen_ordinal_j",
                    "relative_tolerance_binary64": "3d719799812dea11",
                    "bound_formula": (
                        "1e_minus_12_times_max_1_abs_selected_abs_following"
                    ),
                    "signed_margin": "absolute_gap_minus_bound",
                    "PASS_requires_strictly_positive_margin": True,
                    "terminal_failure_inventory": (
                        "all_prior_charts_times_16_PASS_then_current_chart_"
                        "ordinals_before_j_PASS_then_first_FAIL_j"
                    ),
                    "partial_spectrum_hash_policy": "option_A",
                    "spectrum_hashes_only_for_complete_16_row_charts": True,
                    "failure_hash_cardinality": (
                        "failing_chart_ordinal_plus_int_terminal_j_equals_15"
                    ),
                    "every_published_spectrum_hash_reconstructed_from_"
                    "chained_rows": True,
                    "partial_unreached_spectrum_preimage_forbidden": True,
                },
                "SPD_contract": {
                    "canonical_inventory_after_all_eigengaps_PASS": [
                        "K_chart_basis_orthonormality_decisive_components",
                        "K_chart_minimum_positive_tangent_variances",
                        "K_chart_positive_orthogonal_alpha_scalars",
                    ],
                    "basis_orthonormality": {
                        "observed_binary64": [
                            "basis_gram_ij",
                            "identity_ij",
                            "absolute_residual",
                        ],
                        "component_ordinal": "i_times_16_plus_j",
                        "decisive_component_rule": (
                            "minimum_signed_margin_if_finite_else_first_"
                            "nonfinite_component"
                        ),
                        "bound_formula": (
                            "1e_minus_9_plus_1e_minus_9_times_abs_identity_ij"
                        ),
                        "signed_margin": "bound_minus_absolute_residual",
                    },
                    "basis_gram_rtol_binary64": "3e112e0be826d695",
                    "basis_gram_atol_binary64": "3e112e0be826d695",
                    "tangent_variance": {
                        "observed_binary64": [
                            "minimum_published_tangent_variance"
                        ],
                        "component_ordinal": "minimum_component_0_through_15",
                        "bound_binary64": "0000000000000000",
                        "signed_margin": "published_variance_minus_zero",
                        "variance_floor_value_not_exposed_or_claimed": True,
                    },
                    "orthogonal_alpha": {
                        "observed_binary64": ["orthogonal_alpha"],
                        "bound_binary64": "0000000000000000",
                        "signed_margin": "orthogonal_alpha_minus_zero",
                    },
                    "nonfinite_classification_required": True,
                },
                "hash_only_rejected_candidate_diagnostics_forbidden": True,
            },
            "chart_by_mode_and_chart_by_family_occupancy_catalog_hashes_"
            "required": True,
            "selected_K_feature_width_and_atlas_sha256_required": True,
            "selection_receipt_sha256_enters_A_stage_and_all_later_stage_"
            "chains": True,
            "prompt_text_token_ids_and_raw_tensors_forbidden": True,
        },
        "B_C_neighbor_confirmation": {
            "chart_count_frozen_from_A_before_B": True,
            "mutual_neighbor_k_symbol": "k_neighbor",
            "mutual_neighbor_k": MUTUAL_NEIGHBOR_K,
            "K_chart_and_k_neighbor_are_distinct": True,
            "B_nominates_only_point_gated_mutual_k2_candidates": True,
            "C_tests_only_frozen_B_candidates": True,
            "within_family_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
                "sampling_unit": "whole_example_with_all_target_token_rows",
                "primary_examples_per_family": 7,
                "minimum_mutual_frequency": (
                    BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY
                ),
                "maximum_degenerate_fraction": (
                    BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
                ),
                "unchanged_from_v1": True,
            },
            "leave_one_family_out": {
                "family_count": 6,
                "replicate_count": 6,
                "minimum_mutual_count": LOFO_MINIMUM_MUTUAL_COUNT,
                "mutual_k2_recomputed_per_replicate": True,
                "unchanged_from_v1": True,
            },
            "token_count_514_excluded_from_primary_topology_bootstrap_LOFO_"
            "and_hard_gates": True,
            "hard_long_576_640_768_confirmation_veto_unchanged": True,
            "A_bootstrap_or_A_LOFO_does_not_authorize_K_selection": True,
            "optional_A_LOFO_is_diagnostic_only_and_cannot_change_K": True,
            "no_cross_K_B_C_multiplicity_because_only_one_K_reaches_B_C": (
                True
            ),
        },
        "output_contract": {
            "report": str(DEFAULT_REPORT_OUTPUT),
            "svg": str(DEFAULT_SVG_OUTPUT),
            "atlas": str(DEFAULT_ATLAS_OUTPUT),
            "all_names_are_distinct_from_v1": True,
            "report_is_published_last_as_commit_marker": True,
            "A_selection_receipt_embedded": True,
            "selected_K_rank_feature_width_and_claim_tier_reported": True,
            "rejected_attempt_prefix_and_sufficient_statistics_reported": (
                True
            ),
            "prior_K8_diagnostic_result_receipt_and_SHA_bound": True,
            "independent_v2_K8_agreement_receipt_bound": True,
        },
        "claim_taxonomy": {
            "allowed_after_C_all_gates": [
                "chart_native_mode_neighbor_edge_conditional_on_A_selected_K",
                "cross_native_ABI_chart_affinity_conditional_on_A_selected_K",
            ],
            "K_at_least_2_additional_allowed_wording": [
                "multi_chart_shared_geometry_measured"
            ],
            "K_equal_1_allowed_wording": [
                "single_global_rank16_chart_transport_geometry"
            ],
            "requires_separate_A_resampling_evidence": [
                "stable_atlas_geometry"
            ],
            "requires_separate_D_support": [
                "downstream_response_supported_chart_neighbor"
            ],
            "never_authorized": [
                "weights_alone_determine_neighbors",
                "raw_weight_similarity_neighbor",
                "causal_interchangeability",
                "cluster_or_generator_family",
                "parameter_sharing",
                "merge_or_deletion",
                "compression",
                "deployment_or_latency_improvement",
            ],
            "K8_failure_interpretation": (
                "K8_failed_this_A_balanced_solver_gate_not_K8_is_"
                "universally_impossible"
            ),
        },
        "implementation_status": {
            "v2_setup_frozen": True,
            "A_only_K8_diagnostic_protocol_defined": True,
            "descending_streaming_selector_API_required": (
                "select_and_fit_shared_chart_atlas_streaming"
            ),
            "full_v2_experiment_worker_implemented": False,
            "model_run_performed": False,
            "A_selected_K_known": False,
            "B_opened": False,
            "C_opened": False,
        },
        "implementation_sources": {
            "setup_module_is_reviewed_non_self_hashed_trust_anchor": True,
            "streaming_selector": {
                "path": (
                    "src/fisher_graph/"
                    "streaming_shared_chart_atlas_selection_v1.py"
                ),
                "file_sha256": PINNED_SELECTION_SOURCE_SHA256,
            },
            "traced_selection_kernel": {
                "function": (
                    "streaming_shared_chart_atlas_selection_kernel_receipt_v1"
                ),
                "receipt_sha256": (
                    PINNED_SELECTION_KERNEL_RECEIPT_SHA256
                ),
                "exact_receipt_required_in_every_attempt_and_selection": True,
            },
            "full_v2_worker_source_pin_deferred_until_worker_exists": True,
            "future_implementation_qualification": {
                "path": str(FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT),
                "must_bind_this_unchanged_setup_sha256": True,
                "must_bind_full_worker_and_dependency_source_closure": True,
                "must_not_modify_or_regenerate_this_scientific_setup": True,
                "diagnostic_values_forbidden_from_qualification_identity": (
                    True
                ),
            },
        },
        "setup_receipt": {
            "prompt_data_opened": False,
            "tensor_library_imported": False,
            "model_weights_loaded": False,
            "activations_captured": False,
            "chart_count_selected": False,
            "atlas_fitted": False,
            "transports_fitted": False,
            "neighbor_graph_built": False,
            "scientific_result_present": False,
        },
    }


def build_setup() -> dict[str, object]:
    state = _build_state()
    state["setup_sha256"] = setup_sha256(state)
    return state


def validate_setup(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("chart-native neighbor v2 setup must be a mapping")
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("chart-native neighbor v2 setup is not canonical") from error
    if canonical_json_bytes(observed) != canonical_json_bytes(build_setup()):
        raise ValueError("chart-native neighbor v2 setup differs from source")
    return observed


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


def _read_regular(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("chart-native neighbor v2 setup must be regular")
    if before.st_size > 8 * 1024 * 1024:
        raise ValueError("chart-native neighbor v2 setup exceeds byte cap")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("chart-native neighbor v2 setup changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("chart-native neighbor v2 setup was truncated")
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
        raise ValueError("chart-native neighbor v2 setup changed during read")
    return b"".join(chunks)


def load_setup(source: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    raw = _read_regular(Path(source))
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("chart-native neighbor v2 setup JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("chart-native neighbor v2 setup must be a JSON object")
    return validate_setup(value)


def write_setup(destination: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = build_setup()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
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
            raise RuntimeError("chart-native v2 setup publication identity drifted")
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if load_setup(path) != value:
            raise RuntimeError("chart-native neighbor v2 strict reload drifted")
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


def check_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    path = Path(source)
    value = load_setup(path)
    if value["setup_sha256"] != PINNED_SETUP_SHA256:
        raise ValueError("chart-native neighbor v2 semantic pin drifted")
    if hashlib.sha256(_read_regular(path)).hexdigest() != PINNED_SETUP_FILE_SHA256:
        raise ValueError("chart-native neighbor v2 file pin drifted")
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    source_receipt = value["implementation_sources"]["streaming_selector"]
    relative = Path(source_receipt["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("v2 streaming selector source path is unsafe")
    selector_sha = hashlib.sha256(_read_regular(root / relative)).hexdigest()
    if selector_sha != source_receipt["file_sha256"]:
        raise ValueError("v2 streaming selector source hash drifted")
    diagnostic_setup_path = root / A_DIAGNOSTIC_SETUP_PATH
    diagnostic_raw = _read_regular(diagnostic_setup_path)
    diagnostic_value = _strict_json_loads(diagnostic_raw)
    if (
        not isinstance(diagnostic_value, Mapping)
        or diagnostic_value.get("setup_sha256")
        != PINNED_A_DIAGNOSTIC_SETUP_SHA256
        or hashlib.sha256(diagnostic_raw).hexdigest()
        != PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256
    ):
        raise ValueError("v2 A-only diagnostic setup binding drifted")
    return value


@dataclass(frozen=True, slots=True)
class Gemma3ChartNativeNeighborV2Setup:
    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3ChartNativeNeighborV2Setup":
        return cls(build_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self.state))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.check:
        check_setup(arguments.output)
    else:
        write_setup(arguments.output)
    return 0


__all__ = [
    "A_DIAGNOSTIC_RESULT_PATH",
    "A_DIAGNOSTIC_SETUP_PATH",
    "CHART_COUNT_CANDIDATES",
    "DEFAULT_ATLAS_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "DEFAULT_SVG_OUTPUT",
    "Gemma3ChartNativeNeighborV2Setup",
    "FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT",
    "HIDDEN_WIDTH",
    "INTRINSIC_REJECT_CODES",
    "MINIMUM_BALANCED_HARD_OCCUPANCY",
    "MINIMUM_HARD_ROWS",
    "MUTUAL_NEIGHBOR_K",
    "PINNED_SETUP_FILE_SHA256",
    "PINNED_SETUP_SHA256",
    "PINNED_SELECTION_SOURCE_SHA256",
    "PINNED_SELECTION_KERNEL_RECEIPT_SHA256",
    "PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256",
    "PINNED_A_DIAGNOSTIC_SETUP_SHA256",
    "SCHEMA",
    "SETUP_ID",
    "TANGENT_RANK",
    "build_setup",
    "canonical_json_bytes",
    "check_setup",
    "load_setup",
    "main",
    "setup_sha256",
    "validate_setup",
    "write_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
